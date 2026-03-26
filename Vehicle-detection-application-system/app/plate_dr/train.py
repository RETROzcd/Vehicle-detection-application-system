import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.face_datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    print_mutation, set_logging
from utils.google_utils import attempt_download
from utils.loss import compute_loss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first

logger = logging.getLogger(__name__)
begin_save=1
try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")
    #安装Weights & Biases来进行实验日志记录


def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f'Hyperparameters {hyp}')#超参数
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    #目录设置
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  #创建目录
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    #保存运行设置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    #配置
    plots = not opt.evolve  #创建图表
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  #数据字典
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  #数据检查
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  #类别数
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  #类别名称
    #    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  #检查



    #模型
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  #如果本地不存在，则下载
        ckpt = torch.load(weights, map_location=device)  #加载模型
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  #强制使用自动锚框
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  #创建模型
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  #排除键
        state_dict = ckpt['model'].float().state_dict()  #转换为FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  #加载
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  #创建模型

    # Freeze
    freeze = []  #要冻结的参数名称
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    #优化器
    nbs = 64  #常规批量大小
    accumulate = max(round(nbs / total_batch_size), 1)
    #累计梯度的部署，根据总批量大小来确定
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs
    #权重衰减随累积梯度二点调整

    pg0, pg1, pg2 = [], [], []  #优化器参数组
    for k, v in model.named_modules():#遍历模型中的所有模块
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  #偏置项
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  #不使用权重衰减的BatchNorm2d的权重
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  #使用权重衰减的权重


    #根据设置选择使用Adam优化器或者SGD优化器
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  #调整beta1为动量值
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    #添加参数组到优化器中
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  #添加使用权重衰减的参数组
    optimizer.add_param_group({'params': pg2})  #添加偏置项参数组
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))#日志记录参数组信息
    del pg0, pg1, pg2#删除临时变量


    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    #学习率调度器设置
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  #cos退火学习率调度器函数
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)#使用lambdaLR设置学习率调度器
    # plot_lr_scheduler(optimizer, scheduler, epochs)#绘制学习率调度器



    #日志
    if wandb and wandb.run is None:
        opt.hyp = hyp  #添加超参数到wandb配置中
        wandb_run = wandb.init(config=opt, resume="allow",#初始化wandb运行
                               project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                               name=save_dir.stem,
                               id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)
    loggers = {'wandb': wandb}  #日志记录字典


    # Resume
    start_epoch, best_fitness = 0, 0.0#初始化起始轮数和最佳性能指标
    if pretrained:#如果使用预训练模型，则加载相关参数和结果
        #加载优化器状态
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])#加载优化器状态
            best_fitness = 0#重置最佳性能指标

        #加载训练
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  #写入文件结果results.txt

        #加载轮数信息
        #Epochs
        start_epoch = ckpt['epoch'] + 1#更新起始轮数
        if opt.resume:#无法继续训练
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)#判断是否能够回复训练
        if epochs < start_epoch:#额外轮
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  #更新总轮，用于微调

        del ckpt, state_dict#删除临时变量


    #图像尺寸设置部分
    gs = int(max(model.stride))  #网格小大（最大步长）
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  #验证图像是否尺寸为gs的倍数

    #数据并行模式设置
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)#使用DataParallel进行GPU训练

    #同步批归一化设置
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)#转化为SyncBatchNorm
        logger.info('Using SyncBatchNorm()')

    #EMA设置
    ema = ModelEMA(model) if rank in [-1, 0] else None

    #分布式数据并行模式设置
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    #训练数据并行模式设置
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights)#创建训练数据加载器
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  #最大标签类别
    nb = len(dataloader)  #批次数
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)
    #判断确保标签类别未超过预设的类别数


    #进程0的处理部分
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  #设置EMA更新次数
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,  #创建测试数据加载器
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers, pad=0.5)[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  #类别
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  #频率
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, save_dir, loggers)#绘制标签直方图
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)#将类别添加到TensorBoard

            #锚点设置
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)#检查锚点

    #模型参数设置
    hyp['cls'] *= nc / 80.  #将基于COCO调整的分类损失权重缩放到当前数据集
    model.nc = nc  #将类别数附加到模型上
    model.hyp = hyp  #将超参数附加到模型上
    model.gr = 1.0  #iou 损失比率(obj_loss=1.0或iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  #将类别权重附加到模型
    model.names = names#类别名称




    #开始训练
    t0 = time.time()#记录开始时间
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  #记录迭代次数，最少为1000次
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  #限制热身迭代次数不超过总训练论述的一半
    maps = np.zeros(nc)  #每个类的mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)#混合精度训练的梯度缩放器
    logger.info('Image sizes %g train, %g test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, save_dir, epochs))
    for epoch in range(start_epoch, epochs):
        #迭代次数
        model.train()


        #更新图像权重（可选）
        if opt.image_weights:
            #生成索引
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            #在分布式数据并行模式下广播
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders


        mloss = torch.zeros(5, device=device)#平均损失
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)#设置sampler的epoch
        pbar = enumerate(dataloader)#迭代器
        logger.info(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'landmark', 'total', 'targets', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)#进度条
        optimizer.zero_grad()#梯度清零




        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  #当前以及处理的批次数
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  #图像归一化

            #热身
            if ni <= nw:
                xi = [0, nw]  #x插值点
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())#累计梯度的步数
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])#学习率插值
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])#动量插值

            #多尺度的训练
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  #随机尺度
                sf = sz / max(imgs.shape[2:])  #尺度因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  #新尺寸
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            #向前传播
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  #前
                loss, loss_items = compute_loss(pred, targets.to(device), model)#计算损失
                if rank != -1:
                    loss *= opt.world_size  #在DDP模式下梯度在多个设备间平均

            #反向传播
            scaler.scale(loss).backward()

            #优化
            if ni % accumulate == 0:
                scaler.step(optimizer)#更新模型参数
                scaler.update()
                optimizer.zero_grad()#梯度清零
                if ema:
                    ema.update(model)#更新EMA

            #输出
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)#更新平均的损失
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)#现存占用情况
                s = ('%10s' * 2 + '%10.4g' * 7) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])#日志记录字符串
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  #图像保存的路径
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()#异步绘图
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                elif plots and ni == 3 and wandb:
                    wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('train*.jpg')]})#记录到wandb
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------



        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]#为TensorBoard记录学习
        scheduler.step()#更新学习率

        #DDP进程0或者单个GPU
        if rank in [-1, 0] and epoch > begin_save:
            # mAP
            if ema:#更新EMA模型属性
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            #是否为最后一个epoch
            if not opt.notest or final_epoch:
                #计算mAP
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 plots=False,
                                                 log_imgs=opt.log_imgs if wandb else 0)

            #写入文件结果
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))
                #上传结果文件到GCS（Google Cloud Storage

            #日志记录
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  #训练损失
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  #验证损失
                    'x/lr0', 'x/lr1', 'x/lr2']  #学习率
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)#TensorBoard记录
                if wandb:
                    wandb.log({tag: x})# W&B

            #更新性能最佳的mAP
            fi = fitness(np.array(results).reshape(1, -1))#计算综合指标
            if fi > best_fitness:
                best_fitness = fi

            #保存模型
            save = (not opt.nosave) or (final_epoch and not opt.evolve)#是否保存模型
            if save:
                with open(results_file, 'r') as f:  #创建检查点
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            'wandb_id': wandb_run.id if wandb else None}

                #保存最后一轮和最佳的模型
                torch.save(ckpt, last)#最后一轮模型
                if best_fitness == fi:
                    ckpt_best = {
                            'epoch': epoch,
                            'best_fitness': best_fitness,
                            # 'training_results': f.read(),
                            'model': ema.ema,
                            # 'optimizer': None if final_epoch else optimizer.state_dict(),
                            # 'wandb_id': wandb_run.id if wandb else None
                            }
                    torch.save(ckpt_best, best)#最佳模型
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training





    if rank in [-1, 0]:
        #去除优化器
        final = best if best.exists() else last  #最终模型
        for f in [last, best]:
            if f.exists():
                strip_optimizer(f)#去除优化器
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')#上传模型

        #绘图
        if plots:
            plot_results(save_dir=save_dir)#保存结果图
            if wandb:
                files = ['results.png', 'precision_recall_curve.png', 'confusion_matrix.png']
                wandb.log({"Results": [wandb.Image(str(save_dir / f), caption=f) for f in files
                                       if (save_dir / f).exists()]})#记录结果图到W&B
                if opt.log_artifacts:
                    wandb.log_artifact(artifact_or_path=str(final), type='model', name=save_dir.stem)#记录模型到W&B

        #测试最佳模型
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  #如果使用COCO数据集
            for conf, iou, save_json in ([0.25, 0.45, False], [0.001, 0.65, True]):  #测试speed, mAP
                results, _, _ = test.test(opt.data,
                                          batch_size=total_batch_size,
                                          imgsz=imgsz_test,
                                          conf_thres=conf,
                                          iou_thres=iou,
                                          model=attempt_load(final, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=save_json,
                                          plots=False)#测试模型

    else:
        dist.destroy_process_group()#销毁进程组

    wandb.run.finish() if wandb and wandb.run else None#结束W&B的运行
    torch.cuda.empty_cache()#清空显存
    return results#返回测试结果


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/plate_detect.pt', help='initial weights path')#初始权重路径
    parser.add_argument('--cfg', type=str, default='models/yolov5n-0.5.yaml', help='model.yaml path')#模型配置文件
    parser.add_argument('--data', type=str, default='data/plateAndCar.yaml', help='data.yaml path')#数据集配置文件路径
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')#超参数配置文件路径
    parser.add_argument('--epochs', type=int, default=120)#训练轮数
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')#总的批大小
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')#图像测试
    parser.add_argument('--rect', action='store_true', help='rectangular training')#使用矩形训练
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')#恢复最近的训练
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')#仅保存最终检查点
    parser.add_argument('--notest', action='store_true', help='only test final epoch')#仅测试最终epoch
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')#禁用自动anchor检查
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')#演化超参数
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')#gsutil存储桶
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')#缓存图像来加快训练速度
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')#使用加权图像选择进行训练
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')#cuda设备
    parser.add_argument('--multi-scale', action='store_true', default=True, help='vary img-size +/- 50%%')#变化图像尺寸
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')#将多类数据训练为单类
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')#使用torch.optim.Adam()优化器
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')#使用SyncBatchNorm，仅在DDP模式下
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')#DDP超参数
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')#用于W&B日志记录的图像数量，最大为100
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')#记录artifacts，即最终训练好的模型
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')#数据加载器的最大工作数
    parser.add_argument('--project', default='runs/train', help='save to project/name')#保存到项目/名称
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    #设置DDP变量
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    # if opt.global_rank in [-1, 0]:
    #     check_git_status()

    #回复训练
    if opt.resume:  #回复中断的运行
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  #指定或最近的路径
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  #替换
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'#必须指定--cfg或--weights
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  #扩展为2个尺寸
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  #增加运行编号

    #DDP模式
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  #分布式后端
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters超参数
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  #加载超参数
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    #训练
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  #初始化记录器
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
            #启动Tensorboard，在http://localhost:6006/上查看
            tb_writer = SummaryWriter(opt.save_dir)#Tensorboard
        train(hyp, opt, device, tb_writer, wandb)

    #进化超参数（可选）
    else:
        #超参数进化元数据（变异尺度0-1，下限，上限
        meta = {'lr0': (1, 1e-5, 1e-1),#初始化学习率(SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),#最终OneCycleLR学习率(lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  #SGD动量/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  #优化器权重衰减
                'warmup_epochs': (1, 0.0, 5.0),  #预热epochs
                'warmup_momentum': (1, 0.0, 0.95),  #预热初始动量
                'warmup_bias_lr': (1, 0.0, 0.2),  #预热初始偏执学习率
                'box': (1, 0.02, 0.2),  #box损失增益
                'cls': (1, 0.2, 4.0),  #cls损失增益
                'cls_pw': (1, 0.5, 2.0),  #cls BCELoss正权重
                'obj': (1, 0.2, 4.0),  # obj损失增益（与像素一起缩放）
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss 正权重
                'iou_t': (0, 0.1, 0.7),  #IoU训练阈值
                'anchor_t': (1, 2.0, 8.0),  #锚点-多阈值
                'anchors': (2, 2.0, 10.0),   #每个输出网格的锚点数（0表示忽略）
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma（efficientDet默认gamma=1.5）
                'hsv_h': (1, 0.0, 0.1),  #图像HSV-Hue增强(fraction)
                'hsv_s': (1, 0.0, 0.9),  #图像HSV-Saturation增强(fraction)
                'hsv_v': (1, 0.0, 0.9),  #图像HSV-Value增强(fraction)
                'degrees': (1, 0.0, 45.0),  #图像旋转(+/- deg)
                'translate': (1, 0.0, 0.9),  #图像平移(+/- fraction)
                'scale': (1, 0.0, 0.9),  #图像缩放(+/- gain)
                'shear': (1, 0.0, 10.0),  #图像剪切(+/- deg)
                'perspective': (0, 0.0, 0.001),  #图像透视(+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  #上下反转(probability)
                'fliplr': (0, 0.0, 1.0),  #左右翻转(probability)
                'mosaic': (1, 0.0, 1.0),  #图像混合(probability)
                'mixup': (1, 0.0, 1.0)}  # 图像混合（概率）

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        #在--evolve模式下不支持DDP模式
        opt.notest, opt.nosave = True, True  # only test/save final epoch#仅测试/保存最终结果
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  #可进化指数
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  #将最佳结果保存在此处
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  #如果存在，下载进化文件

        for _ in range(300):  #进化的世代数
            if Path('evolve.txt').exists():  #如果存在evolve.txt，选择最佳超参数并进行变异
                # Select parent(s)
                parent = 'single'  #父代选择方法'single'或'weights'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  #要考虑的先前结果数量
                x = x[np.argsort(-fitness(x))][:n]  #前n个变异
                w = fitness(x) - fitness(x).min()  #权重
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  #随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  #加权选择
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  #加权组合

                #变异
                mp, s = 0.8, 0.2  #变异概率，标准差
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  #增益0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  #变异直到发生改变
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  #变异

            #约束到限制范围内
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  #下限
                hyp[k] = min(hyp[k], v[2])  #上限
                hyp[k] = round(hyp[k], 5)  #保留有效数字

            #训练变异后的超参数
            results = train(hyp.copy(), opt, device, wandb=wandb)

            #写入变异结果
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        #绘制进化结果
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              #最佳结果保存
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
            #使用这些超参数训练新模型的命令 python train.py --hyp {yaml_file}'



"""
python train.py
"""
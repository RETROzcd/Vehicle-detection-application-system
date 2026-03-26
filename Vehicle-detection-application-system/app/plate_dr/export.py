import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
import onnx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--onnx2pb', action='store_true', default=False, help='export onnx to pb')
    parser.add_argument('--onnx_infer', action='store_true', default=True, help='onnx infer test')
    #=======================TensorRT=================================
    parser.add_argument('--onnx2trt', action='store_true', default=False, help='export onnx to tensorrt')
    parser.add_argument('--fp16_trt', action='store_true', default=False, help='fp16 infer')
    #================================================================
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    delattr(model.model[-1], 'anchor_grid')
    model.model[-1].anchor_grid=[torch.zeros(1)] * 3 # nl=3 number of detection layers
    model.model[-1].export_cat = True
    model.eval()
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
        if isinstance(m, models.common.ShuffleV2Block):#shufflenet block nn.SiLU
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
            for i in range(len(m.branch2)):
                if isinstance(m.branch2[i], nn.SiLU):
                    m.branch2[i] = SiLU()
        if isinstance(m, models.common.BlazeBlock):#shufflenet block nn.SiLU
            if isinstance(m.relu, nn.SiLU):
                m.relu = SiLU()
        if isinstance(m, models.common.DoubleBlazeBlock):#shufflenet block nn.SiLU
            if isinstance(m.relu, nn.SiLU):
                m.relu = SiLU()
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
            # for i in range(len(m.branch2)):
            #     if isinstance(m.branch2[i], nn.SiLU):
            #         m.branch2[i] = SiLU()
    y = model(img)  # dry run

    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    model.fuse()  # only for ONNX
    input_names=['input']
    output_names=['output']
    #tensorrt 7
    # grid = model.model[-1].anchor_grid
    # model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
    #tensorrt 7

    torch.onnx.export(model, img, f, verbose=False, opset_version=12, 
        input_names=input_names,
        output_names=output_names,
        dynamic_axes = {'input': {0: 'batch'},
                        'output': {0: 'batch'}
                        } if opt.dynamic else None)
                        
    # model.model[-1].anchor_grid = grid

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print('ONNX export success, saved as %s' % f)
    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


    # onnx infer
    if opt.onnx_infer:
        import onnxruntime
        import numpy as np
        providers =  ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(f, providers=providers)
        im = img.cpu().numpy().astype(np.float32) # torch to numpy
        y_onnx = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]
        print("pred's shape is ",y_onnx.shape)
        print("max(|torch_pred - onnx_pred|） =",abs(y.cpu().numpy()-y_onnx).max())


    # TensorRT export
    if opt.onnx2trt:
        from torch2trt.trt_model import ONNX_to_TRT
        print('\nStarting TensorRT...')
        ONNX_to_TRT(onnx_model_path=f,trt_engine_path=f.replace('.onnx', '.trt'),fp16_mode=opt.fp16_trt)

    # PB export
    if opt.onnx2pb:
        print('download the newest onnx_tf by https://github.com/onnx/onnx-tensorflow/tree/master/onnx_tf')
        from onnx_tf.backend import prepare
        import tensorflow as tf

        outpb = f.replace('.onnx', '.pb')  # filename
        # strict=True maybe leads to KeyError: 'pyfunc_0', check: https://github.com/onnx/onnx-tensorflow/issues/167
        tf_rep = prepare(onnx_model, strict=False)  # prepare tf representation
        tf_rep.export_graph(outpb)  # export the model

        out_onnx = tf_rep.run(img) # onnx output

        # check pb
        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with open(outpb, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                input_x = sess.graph.get_tensor_by_name(input_names[0]+':0')  # input
                outputs = []
                for i in output_names:
                    outputs.append(sess.graph.get_tensor_by_name(i+':0'))
                out_pb = sess.run(outputs, feed_dict={input_x: img})

        print(f'out_pytorch {y}')
        print(f'out_onnx {out_onnx}')
        print(f'out_pb {out_pb}')


'''
这段代码是用于将YOLOv5的PyTorch模型导出为ONNX格式的脚本。下面是代码的主要功能和流程：
导入必要的库和模块，包括PyTorch、ONNX以及YOLOv5相关的模块。
定义了命令行参数，包括模型权重路径、图像大小、批处理大小等。
加载YOLOv5的PyTorch模型，并将其转换为ONNX格式。
在模型转换为ONNX格式之前，对模型进行了一些修改，以确保ONNX导出的成功性，例如修改了激活函数类型等。
执行ONNX导出，并保存导出的ONNX模型文件。
对导出的ONNX模型进行了检查，并打印导出成功的消息。
如果指定了进行ONNX推理测试，则通过ONNX Runtime加载导出的ONNX模型，并对输入数据进行推理测试。
如果指定了将ONNX模型转换为TensorRT格式，则使用TensorRT将ONNX模型转换为TensorRT引擎。
如果指定了将ONNX模型转换为PB格式，则使用TensorFlow将ONNX模型转换为PB格式。
该脚本支持将YOLOv5模型导出为ONNX、TensorRT和PB格式，同时支持进行推理测试。
'''
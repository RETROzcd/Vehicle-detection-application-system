from __future__ import annotations

import argparse
import os
import time
from typing import List

import cv2

from plate_pipeline.core import detect_recognition_plate, draw_result, load_models
from plate_recognition.plate_rec import allFilePath, cv_imread


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect_model", nargs="+", type=str, default="weights/plate_detect.pt", help="model.pt path(s)")
    parser.add_argument("--rec_model", type=str, default="weights/plate_rec_color.pth", help="model.pt path(s)")
    parser.add_argument("--is_color", type=bool, default=True, help="plate color")
    parser.add_argument("--image_path", type=str, default="imgss", help="source")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--output", type=str, default="result", help="output directory")
    parser.add_argument("--video", type=str, default="", help="video path or 'vid' for camera")
    return parser


def run_images(
    detect_weights: str,
    rec_weights: str,
    is_color: bool,
    image_path: str,
    img_size: int,
    output_dir: str,
) -> None:
    models = load_models(detect_weights=detect_weights, rec_weights=rec_weights, is_color=is_color)
    os.makedirs(output_dir, exist_ok=True)

    time_all = 0.0
    time_begin = time.time()
    count = 0

    if not os.path.isfile(image_path):
        file_list: List[str] = []
        allFilePath(image_path, file_list)
        for img_path in file_list:
            print(count, img_path, end=" ")
            t0 = time.time()
            img = cv_imread(img_path)
            if img is None:
                continue
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            dict_list = detect_recognition_plate(
                models.detect_model, img, models.device, models.plate_rec_model, img_size=img_size, is_color=is_color
            )
            ori_img = draw_result(img, dict_list)
            img_name = os.path.basename(img_path)
            save_img_path = os.path.join(output_dir, img_name)
            dt = time.time() - t0
            if count:
                time_all += dt
            cv2.imwrite(save_img_path, ori_img)
            count += 1
        if len(file_list) > 1:
            print(f"sumTime time is {time.time() - time_begin} s, average pic time is {time_all / (len(file_list) - 1)}")
        return

    print(count, image_path, end=" ")
    img = cv_imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    dict_list = detect_recognition_plate(
        models.detect_model, img, models.device, models.plate_rec_model, img_size=img_size, is_color=is_color
    )
    ori_img = draw_result(img, dict_list)
    img_name = os.path.basename(image_path)
    save_img_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_img_path, ori_img)


def run_video(
    detect_weights: str,
    rec_weights: str,
    is_color: bool,
    video_source: str,
    img_size: int,
    output_path: str = "result.mp4",
) -> None:
    models = load_models(detect_weights=detect_weights, rec_weights=rec_weights, is_color=is_color)
    cap = cv2.VideoCapture(0 if video_source == "vid" else video_source)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        t1 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f"第{frame_count} 帧", end=" ")
        dict_list = detect_recognition_plate(
            models.detect_model, frame, models.device, models.plate_rec_model, img_size=img_size, is_color=is_color
        )
        frame_show = draw_result(frame, dict_list)
        fps_val = 1.0 / max(1e-6, (time.time() - t1))
        cv2.putText(frame_show, f"FPS: {fps_val:.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame_show)

        if video_source == "vid":
            cv2.imshow("Detection Result", frame_show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = build_parser()
    opt = parser.parse_args()

    detect_weights = opt.detect_model[0] if isinstance(opt.detect_model, list) else opt.detect_model
    if not opt.video:
        run_images(detect_weights, opt.rec_model, opt.is_color, opt.image_path, opt.img_size, opt.output)
        return
    run_video(detect_weights, opt.rec_model, opt.is_color, opt.video, opt.img_size)


if __name__ == "__main__":
    main()


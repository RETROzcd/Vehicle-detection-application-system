from __future__ import annotations

"""
Compatibility shim for older code that did `import dp` and expected:
- dp.dpfunc() to run detection
- dp.resultplate_array to be populated

New code should call `plate_pipeline.core.detect_recognition_plate` directly.
"""

import argparse
from typing import List

import cv2

from plate_pipeline.core import detect_recognition_plate, load_models
from plate_recognition.plate_rec import allFilePath, cv_imread

# Mimic the historical global variable
resultplate_array: List[str] = []


def dpfunc(argv: List[str] | None = None) -> List[str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect_model", nargs="+", type=str, default="weights/plate_detect.pt", help="model.pt path(s)")
    parser.add_argument("--rec_model", type=str, default="weights/plate_rec_color.pth", help="model.pt path(s)")
    parser.add_argument("--is_color", type=bool, default=True, help="plate color")
    parser.add_argument("--image_path", type=str, default="imgss", help="source")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--video", type=str, default="", help="video path or 'vid' for camera")
    opt = parser.parse_args(args=argv)

    detect_weights = opt.detect_model[0] if isinstance(opt.detect_model, list) else opt.detect_model
    models = load_models(detect_weights=detect_weights, rec_weights=opt.rec_model, is_color=opt.is_color)

    plates: List[str] = []

    if not opt.video:
        if not cv2.haveImageReader(opt.image_path) and not opt.image_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            file_list: List[str] = []
            allFilePath(opt.image_path, file_list)
            for img_path in file_list:
                img = cv_imread(img_path)
                if img is None:
                    continue
                if img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                dict_list = detect_recognition_plate(
                    models.detect_model,
                    img,
                    models.device,
                    models.plate_rec_model,
                    img_size=opt.img_size,
                    is_color=opt.is_color,
                )
                plates.extend([d.get("plate_no", "") for d in dict_list if d.get("plate_no")])
        else:
            img = cv_imread(opt.image_path)
            if img is not None and img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img is not None:
                dict_list = detect_recognition_plate(
                    models.detect_model,
                    img,
                    models.device,
                    models.plate_rec_model,
                    img_size=opt.img_size,
                    is_color=opt.is_color,
                )
                plates = [d.get("plate_no", "") for d in dict_list if d.get("plate_no")]

    global resultplate_array
    resultplate_array = plates
    return plates


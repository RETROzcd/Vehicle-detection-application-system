from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from plate_recognition.double_plate_split_merge import get_split_merge
from plate_recognition.plate_rec import get_plate_result, init_model
from utils.cv_puttext import cv2ImgAddText
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords


COLOURS: List[Tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]


@dataclass(frozen=True)
class Models:
    detect_model: Any
    plate_rec_model: Any
    device: torch.device


def load_detect_model(weights: str, device: torch.device):
    return attempt_load(weights, map_location=device)


def load_models(
    detect_weights: str = "weights/plate_detect.pt",
    rec_weights: str = "weights/plate_rec_color.pth",
    is_color: bool = True,
    device: Optional[torch.device] = None,
) -> Models:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detect_model = load_detect_model(detect_weights, device)
    plate_rec_model = init_model(device, rec_weights, is_color=is_color)
    return Models(detect_model=detect_model, plate_rec_model=plate_rec_model, device=device)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = pts.astype("float32")
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (maxWidth, maxHeight))


def scale_coords_landmarks(
    img1_shape: Sequence[int],
    coords: torch.Tensor,
    img0_shape: Sequence[int],
    ratio_pad=None,
) -> torch.Tensor:
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2, 4, 6]] -= pad[0]
    coords[:, [1, 3, 5, 7]] -= pad[1]
    coords[:, :8] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])
    return coords


def get_plate_rec_landmark(
    img: np.ndarray,
    xyxy: Sequence[float],
    conf: float,
    landmarks: Sequence[float],
    class_num: float,
    device: torch.device,
    plate_rec_model,
    is_color: bool = False,
) -> Dict[str, Any]:
    result_dict: Dict[str, Any] = {}
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    landmarks_np = np.zeros((4, 2))
    rect = [x1, y1, x2, y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])

    class_label = int(class_num)
    roi_img = four_point_transform(img, landmarks_np)
    if class_label:
        roi_img = get_split_merge(roi_img)

    if not is_color:
        plate_number, rec_prob = get_plate_result(roi_img, device, plate_rec_model, is_color=is_color)
        plate_color, color_conf = "", None
    else:
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model, is_color=is_color)

    result_dict["rect"] = rect
    result_dict["detect_conf"] = conf
    result_dict["landmarks"] = landmarks_np.tolist()
    result_dict["plate_no"] = plate_number
    result_dict["rec_conf"] = rec_prob
    result_dict["roi_height"] = roi_img.shape[0]
    result_dict["plate_color"] = plate_color
    if color_conf is not None:
        result_dict["color_conf"] = color_conf
    result_dict["plate_type"] = class_label
    return result_dict


def detect_recognition_plate(
    detect_model,
    orgimg: np.ndarray,
    device: torch.device,
    plate_rec_model,
    img_size: int = 640,
    is_color: bool = True,
    conf_thres: float = 0.3,
    iou_thres: float = 0.5,
) -> List[Dict[str, Any]]:
    dict_list: List[Dict[str, Any]] = []
    img0 = copy.deepcopy(orgimg)
    if orgimg is None:
        raise ValueError("Image is None")
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=detect_model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()

    img_t = torch.from_numpy(img).to(device)
    img_t = img_t.float() / 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    pred = detect_model(img_t)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], orgimg.shape).round()
            det[:, 5:13] = scale_coords_landmarks(img_t.shape[2:], det[:, 5:13], orgimg.shape).round()
            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = float(det[j, 4].cpu().numpy())
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = float(det[j, 13].cpu().numpy())
                result_dict = get_plate_rec_landmark(
                    orgimg,
                    xyxy,
                    conf,
                    landmarks,
                    class_num,
                    device,
                    plate_rec_model,
                    is_color=is_color,
                )
                dict_list.append(result_dict)
    return dict_list


def draw_result(orgimg: np.ndarray, dict_list: List[Dict[str, Any]]) -> np.ndarray:
    result_str = ""
    for result in dict_list:
        rect_area = result["rect"]
        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = max(0, int(y - padding_h))
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        landmarks = result["landmarks"]
        result_p = result["plate_no"]
        plate_color = result.get("plate_color", "")
        if result.get("plate_type", 0) == 0:
            result_p += f" {plate_color}"
        else:
            result_p += f" {plate_color}双层"

        result_str += result_p + " "
        for i in range(4):
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, COLOURS[i], -1)
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)

        label_size = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if rect_area[0] + label_size[0][0] > orgimg.shape[1]:
            rect_area[0] = int(orgimg.shape[1] - label_size[0][0])
        orgimg = cv2.rectangle(
            orgimg,
            (rect_area[0], int(rect_area[1] - round(1.6 * label_size[0][1]))),
            (int(rect_area[0] + round(1.2 * label_size[0][0])), rect_area[1] + label_size[1]),
            (255, 255, 255),
            cv2.FILLED,
        )
        orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], int(rect_area[1] - round(1.6 * label_size[0][1])), (0, 0, 0), 21)

    if result_str:
        print(result_str)
    return orgimg


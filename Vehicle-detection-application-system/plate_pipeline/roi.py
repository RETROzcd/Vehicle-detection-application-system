from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def apply_roi_mask(frame: np.ndarray, roi_points: np.ndarray) -> np.ndarray:
    """Apply polygon ROI mask and return masked frame."""
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi_points], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


def parse_roi_points(points: Iterable[Tuple[int, int]]) -> np.ndarray:
    return np.array(list(points), dtype=np.int32)


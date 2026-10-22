"""读取左右图。"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_gray(path: str | Path) -> np.ndarray:
    """读成 8 位灰度图。彩色图按 OpenCV 的 BGR 顺序转灰度；16 位图拒绝（位深换算需另行约定）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"图像不存在：{path}")
    # Windows 的 imread 对中文路径支持不完整，用 Python 读字节后解码。
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if not encoded.size:
        raise ValueError(f"图像文件为空：{path}")
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError(f"图像无法解码：{path}") from exc
    if image is None:
        raise ValueError(f"图像无法读取：{path}")
    if image.dtype != np.uint8:
        raise ValueError(f"只支持 8 位图，{path} 是 {image.dtype}")
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, code)
    return image


def load_gray_pair(left_path: str | Path, right_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    left, right = load_gray(left_path), load_gray(right_path)
    if left.shape != right.shape:
        raise ValueError(f"左右图尺寸不一致：{left.shape} vs {right.shape}")
    return left, right

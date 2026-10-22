"""OpenCV SGBM 封装：计算视差，并把定点输出转成像素单位。

OpenCV 的 StereoSGBM.compute 输出 int16，数值 = 视差 × 16；
无效点的值为 (min_disparity - 1) × 16。这里统一转成 float32，
无效点填 NaN，另给一张布尔掩膜。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import SGBMConfig

SGBM_SCALE = 16.0

_MODES = {
    "SGBM": cv2.STEREO_SGBM_MODE_SGBM,
    "HH": cv2.STEREO_SGBM_MODE_HH,
    "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
}


def create_matcher(config: SGBMConfig) -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(
        minDisparity=config.min_disparity,
        numDisparities=config.num_disparities,
        blockSize=config.block_size,
        P1=config.p1,
        P2=config.p2,
        disp12MaxDiff=config.disp12_max_diff,
        uniquenessRatio=config.uniqueness_ratio,
        speckleWindowSize=config.speckle_window_size,
        speckleRange=config.speckle_range,
        preFilterCap=config.pre_filter_cap,
        mode=_MODES[config.mode],
    )


def decode_disparity(raw: np.ndarray, min_disparity: int) -> Tuple[np.ndarray, np.ndarray]:
    """SGBM 的 int16 定点输出 → (视差 float32，无效为 NaN；有效掩膜)。"""
    if raw.dtype != np.int16:
        raise TypeError(f"SGBM 输出应为 int16 定点格式，收到 {raw.dtype}")
    disparity = raw.astype(np.float32) / SGBM_SCALE
    valid = disparity >= min_disparity  # 无效编码解码后是 min_disparity - 1
    disparity[~valid] = np.nan
    return disparity, valid


def compute_disparity(
    left: np.ndarray, right: np.ndarray, config: SGBMConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """左右 8 位灰度图 → (disparity_px, disparity_valid_mask)。视差以左图为准。"""
    if left.shape != right.shape:
        raise ValueError(f"左右图尺寸不一致：{left.shape} vs {right.shape}")
    if left.ndim != 2:
        raise ValueError(f"只接受单通道灰度图，收到形状 {left.shape}")
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError(f"只接受 8 位图，收到 {left.dtype} / {right.dtype}")
    raw = create_matcher(config).compute(left, right)
    return decode_disparity(raw, config.min_disparity)

"""深度估计入口：左右图（+ 标定）→ DepthResult。

算法只接收图像、参数和标定，不接收标准答案。
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from oven_stereo.contracts import DepthResult, RectifiedStereo

from .config import SGBMConfig
from .depth import disparity_to_depth
from .sgbm import compute_disparity

METHOD = "opencv_sgbm"


def estimate_depth(
    left: np.ndarray,
    right: np.ndarray,
    config: SGBMConfig,
    calib: Optional[RectifiedStereo] = None,
    sample_id: str = "",
) -> DepthResult:
    """计算视差；有标定时再换算距离，没有标定时状态为 disparity_only。"""
    if calib is not None:
        calib.validate(image_size=(left.shape[1], left.shape[0]))

    t0 = time.perf_counter()
    disparity, valid = compute_disparity(left, right, config)
    t1 = time.perf_counter()

    result = DepthResult(
        sample_id=sample_id,
        method=METHOD,
        config_id=config.config_id,
        disparity_px=disparity,
        disparity_valid_mask=valid,
        extra={"left_invalid_columns": config.left_invalid_columns},
    )
    if calib is None:
        result.status = "disparity_only"
        result.reason = "未提供标定，只输出视差"
    else:
        result.depth_m, result.depth_valid_mask = disparity_to_depth(disparity, valid, calib)
        result.calibration_id = calib.calibration_id
    t2 = time.perf_counter()

    result.timings_ms = {
        "sgbm": (t1 - t0) * 1e3,
        "depth": (t2 - t1) * 1e3,
        "total": (t2 - t0) * 1e3,
    }
    return result

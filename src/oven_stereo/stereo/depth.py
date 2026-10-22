"""视差 → 沿镜头方向的距离 Z（米），以左校正相机为准。"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from oven_stereo.contracts import RectifiedStereo


def disparity_to_depth(
    disparity_px: np.ndarray, valid_mask: np.ndarray, calib: RectifiedStereo
) -> Tuple[np.ndarray, np.ndarray]:
    """Z = f * B / (d + doffs)。分母不大于 0 的点无效。

    返回 (depth_m float32，无效为 NaN；depth_valid_mask)。
    """
    d = disparity_px.astype(np.float64)
    denom = d + calib.doffs_px
    valid = valid_mask & np.isfinite(denom) & (denom > 0)
    depth = np.full(d.shape, np.nan, dtype=np.float32)
    depth[valid] = calib.f_px * calib.baseline_m / denom[valid]
    return depth, valid

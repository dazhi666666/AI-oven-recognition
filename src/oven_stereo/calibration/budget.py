"""基于厂商水平视角的选型估算，不代替实际标定。"""

import math


def estimate_disparity_budget(
    *, image_width_px: int, horizontal_fov_deg: float, baseline_m: float,
    distance_m: float, relative_depth_error: float,
) -> dict:
    """保留水平视场、doffs=0；同时给出一阶和严格双向误差预算。"""
    if isinstance(image_width_px, bool) or not isinstance(image_width_px, int) or image_width_px <= 0:
        raise ValueError("image_width_px 必须是正整数")
    params = (horizontal_fov_deg, baseline_m, distance_m, relative_depth_error)
    if not all(math.isfinite(x) for x in params):
        raise ValueError("选型参数必须有限")
    if not 0 < horizontal_fov_deg < 180 or baseline_m <= 0 or distance_m <= 0:
        raise ValueError("视角须在 (0,180) 度，基线和距离须大于 0")
    if not 0 < relative_depth_error < 1:
        raise ValueError("relative_depth_error 必须在 (0,1) 之间")
    f_px = image_width_px / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
    disparity = f_px * baseline_m / distance_m
    return {
        "f_px": f_px,
        "disparity_px": disparity,
        "disparity_error_linear_px": disparity * relative_depth_error,
        "disparity_error_symmetric_px": disparity * relative_depth_error / (1 + relative_depth_error),
    }

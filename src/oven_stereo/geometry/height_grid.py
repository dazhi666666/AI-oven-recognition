"""只测已知平面物理网格，不负责从深度生成高度或补洞。"""

import numpy as np

from oven_stereo.contracts import GeometryResult


def measure_height_grid(
    height_m: np.ndarray,
    cell_area_m2: float | np.ndarray,
    target_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sample_id: str,
    config_id: str,
) -> GeometryResult:
    """按 h*A 积分；任一目标格无效时拒绝整件测量，不输出局部体积。"""
    heights = np.asarray(height_m)
    if heights.ndim != 2 or 0 in heights.shape or heights.dtype.kind not in "fiu":
        raise ValueError("height_m 必须是非空二维实数数组，单位米")
    target = np.asarray(target_mask)
    valid = np.asarray(valid_mask)
    for name, mask in (("target_mask", target), ("valid_mask", valid)):
        if mask.shape != heights.shape or mask.dtype != np.bool_:
            raise ValueError(f"{name} 必须是与高度形状一致的 bool 数组")
    areas = np.asarray(cell_area_m2)
    if areas.dtype.kind not in "fiu" or (areas.ndim != 0 and areas.shape != heights.shape):
        raise ValueError("cell_area_m2 必须是实数标量或与高度形状一致的数组")
    areas = np.broadcast_to(areas, heights.shape).astype(np.float64, copy=True)
    if not np.all(np.isfinite(areas) & (areas > 0)):
        raise ValueError("cell_area_m2 每格面积必须有限且大于 0，单位平方米")
    if any(not isinstance(value, str) or not value.strip() for value in (sample_id, config_id)):
        raise ValueError("sample_id 和 config_id 必须是非空字符串")

    heights = heights.astype(np.float64, copy=True)
    target = target.copy()
    valid = valid & np.isfinite(heights) & (heights >= 0)
    heights[~valid] = np.nan
    target_count = int(np.count_nonzero(target))
    valid_count = int(np.count_nonzero(target & valid))
    coverage = valid_count / target_count if target_count else None
    thickness = volume = None
    if not target_count:
        status, reason = "empty", "给定目标掩膜为空，按已确认空托盘处理"
        thickness, volume = 0.0, 0.0
    elif valid_count != target_count:
        status, reason = "rejected", "目标区域存在无效、非有限或负高度格；不补洞、不输出局部测量"
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            thickness = float(np.max(heights[target]) * 1000.0)
            volume = float(np.sum(heights[target] * areas[target], dtype=np.float64) * 1_000_000.0)
        if not np.isfinite(thickness) or not np.isfinite(volume):
            status, reason = "rejected", "测量数值溢出"
            thickness = volume = None
        else:
            status, reason = "ok", ""
    return GeometryResult(
        sample_id=sample_id, config_id=config_id, height_m=heights, cell_area_m2=areas,
        target_mask=target, valid_mask=valid, thickness_mm=thickness, volume_ml=volume,
        coverage=coverage, status=status, reason=reason,
    )

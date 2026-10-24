"""把左相机深度图转换为托盘法向高度和物理面积网格。"""

from __future__ import annotations

import math

import numpy as np

from oven_stereo.contracts import GeometryResult, PinholeIntrinsics, TrayPlane

from .height_grid import measure_height_grid


def _require_bool_mask(name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError(f"{name} 必须是与深度形状一致的 bool 数组")
    return mask


def backproject_depth(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    intrinsics: PinholeIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """把沿左相机 z 轴的深度反投影为相机坐标点；无效点为 NaN。"""
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or 0 in depth.shape or depth.dtype.kind not in "fiu":
        raise ValueError("depth_m 必须是非空二维实数数组，单位米")
    shape = depth.shape
    valid = _require_bool_mask("valid_mask", valid_mask, shape).copy()
    intrinsics.validate(image_size=(shape[1], shape[0]))

    depth = depth.astype(np.float64, copy=True)
    valid &= np.isfinite(depth) & (depth > 0)
    rows, cols = np.indices(shape, dtype=np.float64)
    points = np.empty((*shape, 3), dtype=np.float64)
    points[..., 2] = depth
    points[..., 0] = (cols - intrinsics.cx_px) * depth / intrinsics.fx_px
    points[..., 1] = (rows - intrinsics.cy_px) * depth / intrinsics.fy_px
    points[~valid] = np.nan
    return points, valid


def fit_tray_plane(
    points_m: np.ndarray,
    tray_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_rms_m: float | None = None,
) -> TrayPlane:
    """最小二乘拟合托盘平面，并把法向统一朝向相机原点。"""
    points = np.asarray(points_m)
    if points.ndim != 3 or points.shape[-1] != 3 or 0 in points.shape[:2]:
        raise ValueError("points_m 必须是 [高,宽,3] 的非空数组")
    shape = points.shape[:2]
    tray = _require_bool_mask("tray_mask", tray_mask, shape)
    valid = _require_bool_mask("valid_mask", valid_mask, shape)
    if max_rms_m is not None and (
        not math.isfinite(max_rms_m) or max_rms_m < 0
    ):
        raise ValueError("max_rms_m 必须是非负有限值或 None")

    selected = points[tray & valid & np.isfinite(points).all(axis=2)].astype(np.float64)
    if len(selected) < 3:
        raise ValueError("可靠托盘点少于 3 个，无法拟合平面")
    center = np.mean(selected, axis=0, dtype=np.float64)
    centered = selected - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] <= max(singular_values[0], 1.0) * 1e-12:
        raise ValueError("托盘点近似共线，无法确定平面")

    normal = vh[-1].astype(np.float64)
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, center))
    if offset < 0:
        normal = -normal
        offset = -offset
    if offset <= np.finfo(np.float64).eps:
        raise ValueError("托盘平面穿过相机原点，无法定义食物侧和像素面积")

    residuals = selected @ normal + offset
    rms = float(np.sqrt(np.mean(np.square(residuals), dtype=np.float64)))
    if max_rms_m is not None and rms > max_rms_m:
        raise ValueError(f"托盘平面拟合 RMS {rms:.6g} m 超过阈值 {max_rms_m:.6g} m")
    return TrayPlane(tuple(float(value) for value in normal), offset, rms, int(len(selected)))


def support_plane_cell_areas(
    shape: tuple[int, int],
    intrinsics: PinholeIntrinsics,
    plane: TrayPlane,
) -> np.ndarray:
    """把每个像素四角投到托盘平面，返回对应四边形面积（m²）。"""
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape 必须是正的 (高, 宽)")
    height, width = int(shape[0]), int(shape[1])
    intrinsics.validate(image_size=(width, height))
    normal = np.asarray(plane.normal_xyz, dtype=np.float64)
    rows, cols = np.indices((height, width), dtype=np.float64)
    ray_x = {
        offset: (cols + offset - intrinsics.cx_px) / intrinsics.fx_px
        for offset in (-0.5, 0.5)
    }
    ray_y = {
        offset: (rows + offset - intrinsics.cy_px) / intrinsics.fy_px
        for offset in (-0.5, 0.5)
    }
    ray_z = np.ones((height, width), dtype=np.float64)

    def intersect(col_offset: float, row_offset: float) -> tuple[np.ndarray, np.ndarray]:
        rays = np.stack(
            (ray_x[col_offset], ray_y[row_offset], ray_z),
            axis=2,
        )
        denominator = rays @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = -plane.offset_m / denominator
            intersections = rays * scale[..., None]
        valid = np.isfinite(intersections).all(axis=2) & (scale > 0) & (np.abs(denominator) > 1e-12)
        return intersections, valid

    top_left, valid_tl = intersect(-0.5, -0.5)
    top_right, valid_tr = intersect(0.5, -0.5)
    bottom_right, valid_br = intersect(0.5, 0.5)
    bottom_left, valid_bl = intersect(-0.5, 0.5)
    first = np.cross(top_right - top_left, bottom_right - top_left)
    second = np.cross(bottom_right - top_left, bottom_left - top_left)
    areas = 0.5 * (np.linalg.norm(first, axis=2) + np.linalg.norm(second, axis=2))
    valid = valid_tl & valid_tr & valid_br & valid_bl & np.isfinite(areas) & (areas > 0)
    areas[~valid] = np.nan
    return areas


def measure_depth_geometry(
    depth_m: np.ndarray,
    depth_valid_mask: np.ndarray,
    target_mask: np.ndarray,
    tray_mask: np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    sample_id: str,
    config_id: str,
    plane_max_rms_m: float | None = None,
    negative_height_tolerance_m: float = 1e-9,
) -> GeometryResult:
    """拟合托盘并沿法向测量；目标区域来自外部，不在此处自动分割。"""
    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("depth_m 必须是二维数组")
    shape = depth.shape
    target = _require_bool_mask("target_mask", target_mask, shape)
    tray = _require_bool_mask("tray_mask", tray_mask, shape)
    if np.any(target & tray):
        raise ValueError("target_mask 与 tray_mask 不能重叠")
    if not math.isfinite(negative_height_tolerance_m) or negative_height_tolerance_m < 0:
        raise ValueError("negative_height_tolerance_m 必须是非负有限值")

    points, point_valid = backproject_depth(depth, depth_valid_mask, intrinsics)
    plane = fit_tray_plane(
        points, tray, point_valid, max_rms_m=plane_max_rms_m,
    )
    normal = np.asarray(plane.normal_xyz, dtype=np.float64)
    heights = points @ normal + plane.offset_m
    tiny_negative = point_valid & (heights < 0) & (heights >= -negative_height_tolerance_m)
    heights[tiny_negative] = 0.0
    areas = support_plane_cell_areas(shape, intrinsics, plane)
    valid = point_valid & np.isfinite(areas) & (heights >= 0)
    heights[~valid] = np.nan
    if not np.all(np.isfinite(areas) & (areas > 0)):
        raise ValueError("部分像素射线不能与托盘正向相交，无法生成完整面积网格")

    result = measure_height_grid(
        heights,
        areas,
        target,
        valid,
        sample_id=sample_id,
        config_id=config_id,
    )
    result.method = "depth_plane_height_integral"
    result.roi_source = "reference"
    result.tray_plane = plane
    return result

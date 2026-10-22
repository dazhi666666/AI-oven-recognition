"""OpenCV SGBM 最小封装、**定点缩放与无效编码的实测核验**、视差解码。

对应任务书 C-01 步骤 2、3 与"最容易越界"第 1、5 条：

* 只封装现成 SGBM，不重写 SGM、不训练网络。
* 常见说法"定点视差需除 16"**必须由实际 API 输出核验**：
  本模块按 dtype 给出先验缩放，再用已知解析视差的人工图对实测比对；
  若先验与实测矛盾则**报错**，绝不静默地"顺手再除一次 16"。
* 无效编码：SGBM 以 ``minDisparity - 1`` 的定点形式标记无效
  （本机实测：``minDisparity=0`` 时原始值 ``-16``，解码后 ``-1.0``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .config import SGBMConfig
from .testpatterns import ARTIFICIAL_PROVENANCE, unit_check_pattern

__all__ = [
    "StereoUnitError",
    "DisparityUnitReport",
    "build_matcher",
    "compute_raw_disparity",
    "expected_scale_from_dtype",
    "invalid_raw_sentinel",
    "decode_disparity",
    "verify_disparity_units",
    "clear_unit_verification_cache",
]

_MODE_ENUM = {
    "SGBM": cv2.STEREO_SGBM_MODE_SGBM,
    "HH": cv2.STEREO_SGBM_MODE_HH,
    "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
}

# 实测核验的接受阈值（像素）：中位绝对误差
UNIT_CHECK_MEDIAN_ABS_ERR_MAX_PX = 0.5
# 正确缩放应显著优于错误假设
UNIT_CHECK_RELATIVE_MARGIN = 4.0

_UNIT_CACHE: Dict[Tuple[str, str], "DisparityUnitReport"] = {}


class StereoUnitError(RuntimeError):
    """视差单位/缩放无法核验，或 dtype 先验与实测矛盾。"""

    def __init__(self, message: str, code: str = "E_UNIT_SCALE_UNVERIFIED") -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------- 匹配
def _matcher_factory():
    """选择 SGBM 构造入口。

    实测（本机 OpenCV 5.0.0 / macOS）：``cv2.StereoSGBM(...)`` 类构造器一旦
    ``compute`` 就 **段错误**（EXIT=-11，与参数组合无关），而
    ``cv2.StereoSGBM_create(...)`` 在相同参数下工作正常。因此**优先**使用工厂函数，
    仅在工厂函数缺失（旧版本被移除）时才回退到类构造器。
    该选择在 :func:`..tests.stereo.test_sgbm_wrapper` 中被固定为可回归的断言。
    """
    factory = getattr(cv2, "StereoSGBM_create", None)
    if factory is not None:
        return factory
    return cv2.StereoSGBM  # pragma: no cover - 仅当新版本移除工厂函数时


def build_matcher(config: SGBMConfig):
    """按固定配置构造 OpenCV SGBM 匹配器（配置已在本地校验过）。"""
    config.validate()
    ctor = _matcher_factory()
    return ctor(
        minDisparity=int(config.min_disparity),
        numDisparities=int(config.num_disparities),
        blockSize=int(config.block_size),
        P1=int(config.p1),
        P2=int(config.p2),
        disp12MaxDiff=int(config.disp12_max_diff),
        uniquenessRatio=int(config.uniqueness_ratio),
        speckleWindowSize=int(config.speckle_window_size),
        speckleRange=int(config.speckle_range),
        preFilterCap=int(config.pre_filter_cap),
        mode=_MODE_ENUM[config.mode],
    )


def compute_raw_disparity(
    left_gray: np.ndarray, right_gray: np.ndarray, config: SGBMConfig
) -> np.ndarray:
    """原样返回 OpenCV 的视差输出（不做任何缩放/清洗）。"""
    left = np.ascontiguousarray(left_gray)
    right = np.ascontiguousarray(right_gray)
    if left.shape != right.shape:
        raise ValueError(f"左右图尺寸必须一致，收到 {left.shape} vs {right.shape}")
    if left.ndim != 2:
        raise ValueError(f"compute_raw_disparity 只接受单通道灰度图，收到 {left.ndim} 维")
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError(
            f"当前仅支持 8 位灰度输入，收到 {left.dtype}/{right.dtype}"
        )
    matcher = build_matcher(config)
    return matcher.compute(left, right)


# ----------------------------------------------------------------- 单位解码
def expected_scale_from_dtype(dtype: np.dtype) -> float:
    """dtype 先验缩放：整型为定点（×16），浮点为已解码像素值。"""
    dt = np.dtype(dtype)
    if dt in (np.dtype(np.int16), np.dtype(np.uint16), np.dtype(np.int32)):
        return 16.0
    if dt in (np.dtype(np.float32), np.dtype(np.float64)):
        return 1.0
    raise StereoUnitError(
        f"未支持的 OpenCV 视差输出类型 {dt}：无法确定定点缩放，拒绝猜测",
        code="E_UNSUPPORTED_RAW_DTYPE",
    )


def invalid_raw_sentinel(config: SGBMConfig, scale: float, dtype: np.dtype) -> Any:
    """无效编码的原始值：``(minDisparity - 1) * scale``（无符号类型按位宽回绕）。

    numpy 2 对负值写入无符号 dtype 会直接抛 OverflowError，因此这里显式做模运算，
    而不是依赖 numpy 的隐式转换。
    """
    dt = np.dtype(dtype)
    value = int(round((config.min_disparity - 1) * scale))
    if dt.kind == "u":
        return value % (1 << (dt.itemsize * 8))
    return value


def decode_disparity(
    raw: np.ndarray, config: SGBMConfig, scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """把 OpenCV 原始输出解码为 ``(disparity_px, valid_mask, info)``。

    * ``disparity_px``：``float32``，**无效位置为 NaN**（不填 0、不插值）。
    * ``valid_mask``：显式布尔掩膜，与 NaN 位置严格一致。
    * ``info``：缩放依据、无效编码值与计数，供运行元数据留存。
    """
    raw = np.asarray(raw)
    scale_given = scale is not None
    if scale is None:
        scale = expected_scale_from_dtype(raw.dtype)
    sentinel_raw = invalid_raw_sentinel(config, scale, raw.dtype)

    decoded = raw.astype(np.float64) / float(scale)
    invalid = np.zeros(decoded.shape, dtype=bool)
    if raw.dtype.kind in "iu":
        invalid |= raw == raw.dtype.type(sentinel_raw)
    else:
        invalid |= ~np.isfinite(raw)
    invalid |= ~np.isfinite(decoded)
    # 低于搜索下界的值无物理意义（含定点无效值解码后的 minDisparity-1）
    invalid |= decoded < float(config.min_disparity)

    valid = ~invalid
    disparity_px = np.where(valid, decoded, np.nan).astype(np.float32)

    info = {
        "raw_dtype": str(raw.dtype),
        "scale_applied": float(scale),
        "scale_source": "explicit" if scale_given else "dtype_prior",
        "invalid_raw_value": float(sentinel_raw),
        "min_disparity": int(config.min_disparity),
        "decoded_invalid_value": float((config.min_disparity - 1)),
        "valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
    }
    return disparity_px, valid, info


# --------------------------------------------------------------- 实测核验
@dataclass(frozen=True)
class DisparityUnitReport:
    """定点缩放实测核验报告。"""

    raw_dtype: str
    raw_shape: Tuple[int, ...]
    dtype_prior_scale: float
    verified_scale: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    pattern_provenance: str = ARTIFICIAL_PROVENANCE
    verified: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_dtype": self.raw_dtype,
            "raw_shape": list(self.raw_shape),
            "dtype_prior_scale": self.dtype_prior_scale,
            "verified_scale": self.verified_scale,
            "verified": self.verified,
            "pattern_provenance": self.pattern_provenance,
            "evidence": self.evidence,
            "note": (
                "缩放由本机 OpenCV 实际输出实测核验（已知解析视差的人工斜平面图对）；"
                "非真实场景精度声明"
            ),
        }


def _median_abs_err_px(
    decoded: np.ndarray, valid: np.ndarray, d_gt: np.ndarray, valid_gt: np.ndarray,
    config: SGBMConfig,
) -> Tuple[float, int]:
    in_range = (d_gt >= config.min_disparity) & (
        d_gt < config.min_disparity + config.num_disparities
    )
    sel = valid & valid_gt & in_range
    if not sel.any():
        return float("inf"), 0
    err = np.abs(decoded[sel].astype(np.float64) - d_gt[sel].astype(np.float64))
    return float(np.median(err)), int(sel.sum())


def verify_disparity_units(
    config: SGBMConfig, force: bool = False
) -> DisparityUnitReport:
    """实测核验 OpenCV 输出的定点缩放（结果按 配置ID+cv2版本 缓存）。

    Raises
    ------
    StereoUnitError
        dtype 先验与实测证据矛盾，或实测误差过大（说明缩放/编码假设不成立）。
    """
    key = (config.config_id, cv2.__version__)
    if not force and key in _UNIT_CACHE:
        return _UNIT_CACHE[key]

    pattern = unit_check_pattern()
    raw = compute_raw_disparity(pattern["left"], pattern["right"], config)
    prior = expected_scale_from_dtype(raw.dtype)
    candidates = [16.0, 1.0] if prior == 16.0 else [1.0, 16.0]

    evidence: Dict[str, Any] = {"candidates": {}}
    results: Dict[float, Tuple[float, int]] = {}
    for scale in candidates:
        decoded, valid, info = decode_disparity(raw, config, scale=scale)
        med, n = _median_abs_err_px(
            decoded, valid, pattern["d_gt_px"], pattern["valid_gt"], config
        )
        results[scale] = (med, n)
        evidence["candidates"][str(scale)] = {
            "median_abs_err_px": med if np.isfinite(med) else None,
            "compared_pixels": n,
            "decoded_median_px": (
                float(np.nanmedian(decoded)) if np.isfinite(decoded).any() else None
            ),
            "valid_ratio": info["valid_ratio"],
        }

    chosen = min(candidates, key=lambda s: results[s][0])
    chosen_err = results[chosen][0]
    other = [s for s in candidates if s != chosen][0]
    other_err = results[other][0]
    evidence["chosen"] = chosen
    evidence["chosen_median_abs_err_px"] = (
        chosen_err if np.isfinite(chosen_err) else None
    )
    evidence["rejected_scale"] = other
    evidence["rejected_median_abs_err_px"] = other_err if np.isfinite(other_err) else None
    evidence["pattern"] = pattern["meta"]

    if not np.isfinite(chosen_err):
        raise StereoUnitError(
            "解码核验失败：在人工斜平面图对上没有任何有效比较像素，"
            "无法确认缩放。请检查匹配参数与图像是否可读。",
            code="E_UNIT_SCALE_UNVERIFIED",
        )
    if chosen_err > UNIT_CHECK_MEDIAN_ABS_ERR_MAX_PX:
        raise StereoUnitError(
            f"解码核验失败：最佳候选缩放 {chosen} 的中位绝对误差 "
            f"{chosen_err:.3f} px > 阈值 {UNIT_CHECK_MEDIAN_ABS_ERR_MAX_PX} px，"
            "说明定点缩放/无效编码假设不成立，不能静默继续",
            code="E_UNIT_SCALE_UNVERIFIED",
        )
    contradictory = (
        chosen != prior
        and np.isfinite(other_err)
        and other_err <= UNIT_CHECK_RELATIVE_MARGIN * max(chosen_err, 1e-9)
    )
    if chosen != prior and not contradictory:
        # 实测明确支持另一种缩放（例如未来版本把输出改成 float）
        evidence["dtype_prior_overridden"] = True
    if contradictory:
        raise StereoUnitError(
            f"dtype 先验缩放 {prior} 与实测结果矛盾（实测最佳 {chosen}，"
            f"但另一个候选误差 {other_err:.3f} px 不显著更差），拒绝静默采用。",
            code="E_UNIT_SCALE_AMBIGUOUS",
        )

    report = DisparityUnitReport(
        raw_dtype=str(raw.dtype),
        raw_shape=tuple(raw.shape),
        dtype_prior_scale=float(prior),
        verified_scale=float(chosen),
        evidence=evidence,
        verified=True,
    )
    _UNIT_CACHE[key] = report
    return report


def clear_unit_verification_cache() -> None:
    """清空实测核验缓存（测试用）。"""
    _UNIT_CACHE.clear()

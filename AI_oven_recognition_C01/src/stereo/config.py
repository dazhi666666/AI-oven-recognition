"""固定 SGBM 配置、参数校验与视差搜索范围依据。

设计原则（对应任务书 C-01 步骤 2、7 与"最容易越界"第 2、6 条）：

* 每次运行必须使用**一套明确记录**的配置，配置可序列化、可哈希，写入运行元数据。
* 参数在进入 OpenCV 之前先做本地校验。实测 cv2 5.0.0 的 ``StereoSGBM_create``
  **不会**拒绝 ``numDisparities=0`` / 偶数 ``blockSize`` 等非法值，因此校验责任在本模块。
* 搜索范围不做"自动搜索"，只提供**几何推导**工具：由 fx、基线、目标深度区间
  反算所需视差区间，再检查配置是否覆盖；覆盖不足只报警告并记录，不偷偷改配置。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

__all__ = [
    "SGBMConfig",
    "StereoConfigError",
    "C01_DEFAULT",
    "C01_OVEN_RANGE",
    "MODES",
    "required_disparity_range",
    "check_search_range_covers",
]

MODES = ("SGBM", "HH", "SGBM_3WAY")


class StereoConfigError(ValueError):
    """非法匹配参数。错误码固定为 ``E_INVALID_SEARCH_PARAMS``。"""

    code = "E_INVALID_SEARCH_PARAMS"


@dataclass(frozen=True)
class SGBMConfig:
    """一套固定、可复跑的半全局匹配配置。

    ``min_disparity`` / ``num_disparities`` 共同定义搜索区间
    ``[min_disparity, min_disparity + num_disparities)``（单位：像素，左校正相机系）。
    """

    name: str = "c01_default"
    version: str = "1.0.0"
    min_disparity: int = 0
    num_disparities: int = 64
    block_size: int = 5
    p1: int = 200
    p2: int = 800
    disp12_max_diff: int = 1
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    pre_filter_cap: int = 63
    mode: str = "SGBM"
    rationale: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ 校验
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "SGBMConfig":
        if self.mode not in MODES:
            raise StereoConfigError(
                f"mode 必须是 {MODES} 之一，收到 {self.mode!r}"
            )
        if not isinstance(self.num_disparities, int) or self.num_disparities <= 0:
            raise StereoConfigError(
                f"num_disparities 必须为正整数，收到 {self.num_disparities!r}"
            )
        if self.num_disparities % 16 != 0:
            raise StereoConfigError(
                "num_disparities 必须为 16 的整数倍（SGBM 要求），"
                f"收到 {self.num_disparities}"
            )
        if not isinstance(self.min_disparity, int) or self.min_disparity < 0:
            raise StereoConfigError(
                f"min_disparity 必须为非负整数，收到 {self.min_disparity!r}"
            )
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise StereoConfigError(
                f"block_size 必须为 >=3 的奇数，收到 {self.block_size}"
            )
        if self.p1 < 0 or self.p2 < 0:
            raise StereoConfigError("P1/P2 必须非负")
        if self.p2 < self.p1:
            raise StereoConfigError(
                f"建议 P2 >= P1（平滑惩罚），收到 P1={self.p1}, P2={self.p2}"
            )
        if not 0 <= self.uniqueness_ratio <= 100:
            raise StereoConfigError(
                f"uniqueness_ratio 需在 [0,100]，收到 {self.uniqueness_ratio}"
            )
        if self.speckle_window_size < 0 or self.speckle_range < 0:
            raise StereoConfigError("speckle_window_size / speckle_range 需非负")
        if not 0 < self.pre_filter_cap <= 255:
            raise StereoConfigError(
                f"pre_filter_cap 需在 (0,255]，收到 {self.pre_filter_cap}"
            )
        return self

    # -------------------------------------------------------------- 属性/输出
    @property
    def disparity_range(self) -> tuple[int, int]:
        """``[d_min, d_max_exclusive)``，单位像素。"""
        return self.min_disparity, self.min_disparity + self.num_disparities

    @property
    def config_id(self) -> str:
        """稳定 ID：仅由参与计算的参数决定，便于跨运行对齐。"""
        payload = json.dumps(self.core_params(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def core_params(self) -> Dict[str, Any]:
        """参与计算的参数（不含注释/标签，也不含 config_id 自身）。"""
        keys = (
            "min_disparity",
            "num_disparities",
            "block_size",
            "p1",
            "p2",
            "disp12_max_diff",
            "uniqueness_ratio",
            "speckle_window_size",
            "speckle_range",
            "pre_filter_cap",
            "mode",
        )
        return {k: getattr(self, k) for k in keys}

    def to_dict(self, include_notes: bool = True) -> Dict[str, Any]:
        data = dict(self.core_params())
        data["name"] = self.name
        data["version"] = self.version
        if include_notes:
            data["rationale"] = self.rationale
            data["tags"] = self.tags
        data["disparity_range"] = list(self.disparity_range)
        data["config_id"] = self.config_id
        return data

    def describe(self) -> str:
        lo, hi = self.disparity_range
        return (
            f"{self.name} v{self.version} [{self.mode}] 视差搜索 {lo}..{hi - 1} px, "
            f"blockSize={self.block_size}, P1={self.p1}, P2={self.p2}, "
            f"disp12MaxDiff={self.disp12_max_diff}, uniquenessRatio={self.uniqueness_ratio}, "
            f"speckle=({self.speckle_window_size},{self.speckle_range}), "
            f"preFilterCap={self.pre_filter_cap}"
        )


# ---------------------------------------------------------------- 几何工具
def left_border_invalid_columns(config: SGBMConfig) -> int:
    """SGBM 输出中**左边界固定无效**的列数（= ``numDisparities``）。

    本机实测（OpenCV 5.0.0，图像宽度 320/640 × numDisparities 32/64/128/256
    共 8 组组合全部一致）：左起 ``numDisparities`` 列恒为无效编码
    ``(minDisparity - 1) * scale``，因为该范围内右图支撑窗口不完整，
    OpenCV 不会给出任何视差估计。

    影响：``numDisparities`` 不只是"搜索范围"，还是一笔**固定的左边界像素代价**。
    例如 320 宽的图像用 ``numDisparities=256`` 时，最多只剩 20% 的列可能有效。
    因此选参数时必须同时看深度覆盖（:func:`check_search_range_covers`）
    与图像宽度是否撑得住该范围。
    """
    return int(config.num_disparities)


def required_disparity_range(
    fx_px: float,
    baseline_m: float,
    z_max_m: float,
    z_min_m: float,
    doff_px: float = 0.0,
) -> Dict[str, float]:
    """由几何反算所需视差区间（搜索范围依据）。

    推导：``z = fx * B / (d - doff)`` ⇒ ``d = doff + fx * B / z``。
    深度越近视差越大，所以 ``z_min_m`` 决定上界，``z_max_m`` 决定下界。
    """
    if fx_px <= 0 or baseline_m <= 0:
        raise StereoConfigError("fx_px 与 baseline_m 必须为正，才能反算视差范围")
    if not (0 < z_min_m < z_max_m):
        raise StereoConfigError(
            f"需要 0 < z_min_m < z_max_m，收到 z_min_m={z_min_m}, z_max_m={z_max_m}"
        )
    d_nearest = doff_px + fx_px * baseline_m / z_min_m
    d_farthest = doff_px + fx_px * baseline_m / z_max_m
    return {
        "doff_px": float(doff_px),
        "d_required_min_px": float(d_farthest),
        "d_required_max_px": float(d_nearest),
        "z_min_m": float(z_min_m),
        "z_max_m": float(z_max_m),
    }


def check_search_range_covers(
    config: SGBMConfig, fx_px: float, baseline_m: float, z_max_m: float, z_min_m: float,
    doff_px: float = 0.0,
) -> Dict[str, Any]:
    """检查配置搜索范围是否覆盖目标深度区间；只报告，不修改配置。"""
    req = required_disparity_range(fx_px, baseline_m, z_max_m, z_min_m, doff_px)
    lo, hi = config.disparity_range
    # 搜索区间 [lo, hi) ；可表示的最大视差取 hi-1，允许半像素容差
    covers_farthest = (lo - 0.5) <= req["d_required_min_px"]
    covers_nearest = (hi - 1 + 0.5) >= req["d_required_max_px"]
    return {
        "request": req,
        "config_range": [lo, hi - 1],
        "covers_nearest": bool(covers_nearest),
        "covers_farthest": bool(covers_farthest),
        "covered": bool(covers_nearest and covers_farthest),
        "note": (
            "覆盖完整" if (covers_nearest and covers_farthest) else
            "搜索范围未完整覆盖目标深度区间：近端深度会落到 d>=min+numDisparities 一侧，"
            "远端深度会落到 d<min 一侧，二者都会成为无效像素"
        ),
    }


# ---------------------------------------------------------------- 预置配置
C01_DEFAULT = SGBMConfig(
    name="c01_default",
    num_disparities=64,
    rationale=(
        "C-01 最小基线：用于人工构造斜平面测试对（d_gt≈16..48 px）。"
        "numDisparities=64 覆盖 0..63，留出约 25% 余量；"
        "blockSize=5 与 P1/P2=200/800 为 OpenCV 常规 8 邻域设置（P1=8*bs^2, P2=32*bs^2）。"
    ),
    tags={"stage": "C-01", "scope": "artificial_test_pattern"},
)

C01_OVEN_RANGE = SGBMConfig(
    name="c01_oven_range",
    num_disparities=256,
    rationale=(
        "烤箱/食品场景的**待复核**估计：若 fx≈700 px、B≈0.1 m、目标轴向 Z∈[0.3,2.0] m，"
        "则 d=doff+fx*B/Z ∈ [35,233] px，需要覆盖 0..255 ⇒ numDisparities=256。"
        "该几何参数尚未由 B 提供，仅作范围依据记录，暂不作为任何达标声明。"
    ),
    tags={"stage": "C-01", "scope": "oven_geometry_estimate", "verified": False},
)

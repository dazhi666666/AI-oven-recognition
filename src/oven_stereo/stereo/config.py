"""SGBM 参数：从 YAML 读取、检查合法性，以及由几何推算视差搜索范围。

OpenCV 的 StereoSGBM_create 不检查参数（例如 numDisparities=0、偶数 blockSize
都会被接受，然后算出无意义的结果），所以检查放在这里。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict

import yaml

MODES = ("SGBM", "HH", "SGBM_3WAY")

# 参与计算的参数；config_id 只由它们决定
_CORE_KEYS = (
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


class StereoConfigError(ValueError):
    """匹配参数不合法。"""


@dataclass(frozen=True)
class SGBMConfig:
    name: str = "sgbm"
    min_disparity: int = 0
    num_disparities: int = 64
    block_size: int = 5
    p1: int = 200  # 常用取值 8 * 通道数 * block_size^2
    p2: int = 800  # 常用取值 32 * 通道数 * block_size^2
    disp12_max_diff: int = 1
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    pre_filter_cap: int = 63
    mode: str = "SGBM"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise StereoConfigError(f"mode 必须是 {MODES} 之一，收到 {self.mode!r}")
        if self.num_disparities <= 0 or self.num_disparities % 16:
            raise StereoConfigError(
                f"num_disparities 必须是 16 的正整数倍，收到 {self.num_disparities}"
            )
        if self.block_size < 1 or self.block_size % 2 == 0:
            raise StereoConfigError(f"block_size 必须是正奇数，收到 {self.block_size}")
        if self.p1 < 0 or self.p2 <= self.p1:
            raise StereoConfigError(f"需要 0 <= P1 < P2，收到 P1={self.p1}, P2={self.p2}")
        if not 0 <= self.uniqueness_ratio <= 100:
            raise StereoConfigError(f"uniqueness_ratio 需在 0–100，收到 {self.uniqueness_ratio}")
        if self.speckle_window_size < 0 or self.speckle_range < 0:
            raise StereoConfigError("speckle_window_size、speckle_range 不能为负")
        if self.pre_filter_cap <= 0:
            raise StereoConfigError(f"pre_filter_cap 必须为正，收到 {self.pre_filter_cap}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SGBMConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise StereoConfigError(f"配置里有未知字段：{sorted(unknown)}")
        return cls(**data)

    @property
    def max_disparity(self) -> int:
        """能搜索到的最大视差（含）。"""
        return self.min_disparity + self.num_disparities - 1

    @property
    def left_invalid_columns(self) -> int:
        """图像最左边一定无效的列数。

        实测（OpenCV 4.13）：最左边 min_disparity + num_disparities 列全部无效，
        因为在右图里找不到完整的搜索范围。
        """
        return self.min_disparity + self.num_disparities

    @property
    def config_id(self) -> str:
        core = {k: getattr(self, k) for k in _CORE_KEYS}
        return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "config_id": self.config_id}


def disparity_range_for_depth(
    f_px: float, baseline_m: float, z_min_m: float, z_max_m: float, doffs_px: float = 0.0
) -> tuple[float, float]:
    """工作距离 [z_min, z_max] 对应的视差范围 (最小, 最大)，单位像素。

    d = f * B / Z - doffs。用它来定 min_disparity 和 num_disparities，
    而不是看标准答案的视差范围。
    """
    if f_px <= 0 or baseline_m <= 0:
        raise StereoConfigError("f_px 和 baseline_m 必须为正")
    if not 0 < z_min_m < z_max_m:
        raise StereoConfigError(f"需要 0 < z_min < z_max，收到 {z_min_m}, {z_max_m}")
    return f_px * baseline_m / z_max_m - doffs_px, f_px * baseline_m / z_min_m - doffs_px

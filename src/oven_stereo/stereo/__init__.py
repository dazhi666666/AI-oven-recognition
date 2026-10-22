"""深度算法：匹配、亚像素、可信度、学习方法、脏镜头检测。（主要由 C 推进）"""

from .config import SGBMConfig, StereoConfigError, disparity_range_for_depth
from .depth import disparity_to_depth
from .pipeline import estimate_depth
from .sgbm import compute_disparity, decode_disparity

__all__ = [
    "SGBMConfig",
    "StereoConfigError",
    "compute_disparity",
    "decode_disparity",
    "disparity_range_for_depth",
    "disparity_to_depth",
    "estimate_depth",
]

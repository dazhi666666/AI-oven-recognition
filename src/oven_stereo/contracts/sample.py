"""B 提供的输入样本与独立答案，二者分开传递。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import RectifiedStereo


@dataclass(frozen=True)
class StereoSample:
    sample_id: str
    left_path: Path
    right_path: Path
    image_size_px: tuple[int, int]
    source: str
    source_version: str
    domain: str
    split: str
    conditions: dict[str, Any]
    rectified: bool
    calibration_id: str
    calibration: RectifiedStereo
    metric_depth_available: bool
    calibration_metadata: dict[str, Any]  # 完整内参及数据集允许公开的 ndisp


@dataclass(frozen=True)
class ReferenceData:
    sample_id: str
    disparity_px: np.ndarray
    disparity_valid_mask: np.ndarray
    depth_m: np.ndarray
    depth_valid_mask: np.ndarray
    source: str

"""模块输出的数据格式。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class DepthResult:
    """深度结果（C 产出）。无效点在数组里是 NaN，并由对应掩膜标出。"""

    sample_id: str
    method: str
    config_id: str
    disparity_px: np.ndarray  # float32[H, W]
    disparity_valid_mask: np.ndarray  # bool[H, W]
    depth_m: Optional[np.ndarray] = None  # float32[H, W]；没有标定时为 None
    depth_valid_mask: Optional[np.ndarray] = None
    calibration_id: Optional[str] = None
    confidence: Optional[np.ndarray] = None
    resolution_profile: str = "native"
    timings_ms: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"  # ok / disparity_only / failed
    reason: str = ""
    extra: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict:
        def valid_ratio(mask: Optional[np.ndarray]) -> Optional[float]:
            return None if mask is None else float(mask.mean())

        return {
            "sample_id": self.sample_id,
            "calibration_id": self.calibration_id,
            "method": self.method,
            "config_id": self.config_id,
            "resolution_profile": self.resolution_profile,
            "image_size": [int(self.disparity_px.shape[1]), int(self.disparity_px.shape[0])],
            "disparity_valid_ratio": valid_ratio(self.disparity_valid_mask),
            "depth_valid_ratio": valid_ratio(self.depth_valid_mask),
            "timings_ms": self.timings_ms,
            "status": self.status,
            "reason": self.reason,
            **self.extra,
        }

    def save(self, out_dir: str | Path) -> Path:
        """保存数值数组（.npy）和 result.json。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "disparity_px.npy", self.disparity_px)
        np.save(out / "disparity_valid_mask.npy", self.disparity_valid_mask)
        if self.depth_m is not None:
            np.save(out / "depth_m.npy", self.depth_m)
            np.save(out / "depth_valid_mask.npy", self.depth_valid_mask)
        if self.confidence is not None:
            np.save(out / "confidence.npy", self.confidence)
        (out / "result.json").write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out

"""已知托盘平面物理网格的测量结果。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GeometryResult:
    sample_id: str
    config_id: str
    height_m: np.ndarray
    cell_area_m2: np.ndarray
    target_mask: np.ndarray
    valid_mask: np.ndarray
    thickness_mm: float | None
    volume_ml: float | None
    coverage: float | None
    status: str
    reason: str = ""

    def summary(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "config_id": self.config_id,
            "method": "height_grid_integral",
            "grid_representation": "support_plane_cells",
            "height_reference": "support_plane_normal",
            "grid_size": [int(self.height_m.shape[1]), int(self.height_m.shape[0])],
            "roi_source": "reference",
            "tray_plane": None,  # A-02 的输入已是高度，不估计相机坐标下的平面
            "thickness_mm": self.thickness_mm,
            "thickness_definition": "max_valid_target_height",
            "volume_ml": self.volume_ml,
            "coverage": self.coverage,
            "coverage_definition": "valid_target_cells / target_cells",
            "target_cells": int(np.count_nonzero(self.target_mask)),
            "valid_target_cells": int(np.count_nonzero(self.target_mask & self.valid_mask)),
            "status": self.status,
            "reason": self.reason,
        }

    def save(self, out_dir: str | Path) -> Path:
        """保存物理网格和严格 JSON；缺失标量写为 null。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("height_m", "cell_area_m2", "target_mask", "valid_mask"):
            np.save(out / f"{name}.npy", getattr(self, name))
        (out / "result.json").write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return out

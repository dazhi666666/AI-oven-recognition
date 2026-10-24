"""托盘平面与托盘法向高度测量结果。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class TrayPlane:
    """相机坐标系中的托盘平面 n·X + d = 0，n 朝向摄像头/食物侧。"""

    normal_xyz: Tuple[float, float, float]
    offset_m: float
    fit_rms_m: float
    points_used: int

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal_xyz, dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError("normal_xyz 必须是三个有限值")
        if not math.isclose(float(np.linalg.norm(normal)), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("normal_xyz 必须是单位法向量")
        if not (math.isfinite(self.offset_m) and self.offset_m > 0):
            raise ValueError("offset_m 必须为正，表示摄像头位于托盘食物侧")
        if not (math.isfinite(self.fit_rms_m) and self.fit_rms_m >= 0):
            raise ValueError("fit_rms_m 必须是非负有限值")
        if not isinstance(self.points_used, int) or self.points_used < 3:
            raise ValueError("points_used 必须是不小于 3 的整数")

    def to_dict(self) -> dict:
        return {
            "equation": "n_dot_X_plus_d_equals_0",
            "normal_xyz": [float(value) for value in self.normal_xyz],
            "offset_m": float(self.offset_m),
            "fit_rms_m": float(self.fit_rms_m),
            "points_used": self.points_used,
            "normal_orientation": "toward_camera_and_food_side",
        }


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
    method: str = "height_grid_integral"
    roi_source: str = "reference"
    tray_plane: TrayPlane | None = None

    def summary(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "config_id": self.config_id,
            "method": self.method,
            "grid_representation": "support_plane_cells",
            "height_reference": "support_plane_normal",
            "grid_size": [int(self.height_m.shape[1]), int(self.height_m.shape[0])],
            "roi_source": self.roi_source,
            "tray_plane": None if self.tray_plane is None else self.tray_plane.to_dict(),
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

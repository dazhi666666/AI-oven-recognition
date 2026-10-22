"""已校正双目的标定参数（视差 → 距离换算所需的最少信息）。

主点差采用 Middlebury 约定：

    doffs_px = cx_right - cx_left
    Z = f_px * baseline_m / (d + doffs_px)

其中 d = 左图横坐标 - 右图横坐标。对 Middlebury 的 calib.txt，doffs 可以直接填入。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


class CalibrationError(ValueError):
    """标定参数不可用。"""


@dataclass(frozen=True)
class RectifiedStereo:
    calibration_id: str
    image_size: Tuple[int, int]  # (宽, 高)，像素
    f_px: float  # 校正后焦距，像素
    baseline_m: float  # 两个摄像头的间距，米
    doffs_px: float = 0.0  # cx_right - cx_left，像素
    rectified: bool = True

    @classmethod
    def from_projection_matrices(
        cls,
        P1: Sequence[Sequence[float]],
        P2: Sequence[Sequence[float]],
        calibration_id: str,
        image_size: Tuple[int, int],
    ) -> "RectifiedStereo":
        """由 cv2.stereoRectify 输出的 P1、P2 构造。

        P1[0,0] = f，P1[0,2] = cx_left，P2[0,2] = cx_right，P2[0,3] = -f * B。
        """
        P1 = np.asarray(P1, dtype=np.float64)
        P2 = np.asarray(P2, dtype=np.float64)
        if P1.shape != (3, 4) or P2.shape != (3, 4):
            raise CalibrationError(f"P1/P2 必须是 3x4 矩阵，收到 {P1.shape} / {P2.shape}")
        f = float(P1[0, 0])
        if f <= 0:
            raise CalibrationError(f"P1[0,0]（焦距）必须为正，收到 {f}")
        return cls(
            calibration_id=calibration_id,
            image_size=(int(image_size[0]), int(image_size[1])),
            f_px=f,
            baseline_m=float(-P2[0, 3] / f),
            doffs_px=float(P2[0, 2] - P1[0, 2]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "RectifiedStereo":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["image_size"] = tuple(data["image_size"])
        return cls(**data)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["image_size"] = list(self.image_size)
        return data

    def validate(self, image_size: Tuple[int, int] | None = None) -> None:
        """检查参数是否可用于换算；image_size 为 (宽, 高)，给出时检查是否一致。"""
        if not self.calibration_id:
            raise CalibrationError("calibration_id 不能为空")
        if not self.rectified:
            raise CalibrationError("标定未标记为已校正，不能直接按水平视差换算距离")
        if not (math.isfinite(self.f_px) and self.f_px > 0):
            raise CalibrationError(f"f_px 必须为正，收到 {self.f_px}")
        if not (math.isfinite(self.baseline_m) and self.baseline_m > 0):
            raise CalibrationError(f"baseline_m 必须为正，收到 {self.baseline_m}")
        if not math.isfinite(self.doffs_px):
            raise CalibrationError(f"doffs_px 必须是有限值，收到 {self.doffs_px}")
        if image_size is not None and tuple(image_size) != tuple(self.image_size):
            raise CalibrationError(
                f"标定分辨率 {tuple(self.image_size)} 与图像 {tuple(image_size)} 不一致"
            )

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
class PinholeIntrinsics:
    """左相机针孔内参，供深度反投影到相机坐标系。"""

    calibration_id: str
    image_size: Tuple[int, int]  # (宽, 高)，像素
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    @classmethod
    def from_matrix(
        cls,
        matrix: Sequence[Sequence[float]],
        calibration_id: str,
        image_size: Tuple[int, int],
    ) -> "PinholeIntrinsics":
        camera = np.asarray(matrix, dtype=np.float64)
        if camera.shape != (3, 3) or not np.isfinite(camera).all():
            raise CalibrationError("相机内参矩阵必须是有限的 3x3 矩阵")
        intrinsics = cls(
            calibration_id=calibration_id,
            image_size=(int(image_size[0]), int(image_size[1])),
            fx_px=float(camera[0, 0]),
            fy_px=float(camera[1, 1]),
            cx_px=float(camera[0, 2]),
            cy_px=float(camera[1, 2]),
        )
        intrinsics.validate()
        expected = np.array(
            [[intrinsics.fx_px, 0.0, intrinsics.cx_px],
             [0.0, intrinsics.fy_px, intrinsics.cy_px],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        if not np.allclose(camera, expected, atol=1e-9, rtol=0):
            raise CalibrationError("相机内参矩阵必须符合无偏斜的针孔模型")
        return intrinsics

    def validate(self, image_size: Tuple[int, int] | None = None) -> None:
        if not self.calibration_id:
            raise CalibrationError("calibration_id 不能为空")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise CalibrationError(f"image_size 必须是正的 (宽, 高)，收到 {self.image_size}")
        for name, value in (
            ("fx_px", self.fx_px), ("fy_px", self.fy_px),
            ("cx_px", self.cx_px), ("cy_px", self.cy_px),
        ):
            if not math.isfinite(value):
                raise CalibrationError(f"{name} 必须是有限值，收到 {value}")
        if self.fx_px <= 0 or self.fy_px <= 0:
            raise CalibrationError("fx_px 和 fy_px 必须大于 0")
        if image_size is not None and tuple(image_size) != tuple(self.image_size):
            raise CalibrationError(
                f"标定分辨率 {tuple(self.image_size)} 与图像 {tuple(image_size)} 不一致"
            )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["image_size"] = list(self.image_size)
        return data


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

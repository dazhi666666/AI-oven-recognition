"""已核验双目标定的**只读**表示与视差→轴向深度换算。

边界（任务书责任分解表 + "最容易越界"第 4、5 条）：

* 本模块**只读**使用标定，不修改数据集原标定，也不从别的相机借用参数。
* 输出统一为**左校正相机系的轴向 Z**（不是射线距离），参考系写入标定对象。
* 未知校正状态必须显式拒绝，不得"强行当作已校正输入"。
* 主点差 ``doff = cx_left - cx_right`` 必须参与换算，不能在推导中被省略。

换算推导（左右为已校正图像，fx、B、主点差在同一左校正系下）：

    x_l = fx * X / Z + cx_l
    x_r = fx * (X - B) / Z + cx_r
    d   = x_l - x_r = fx * B / Z + (cx_l - cx_r)
    ⇒ Z = fx * B / (d - doff),   doff = cx_l - cx_r

因此 ``d - doff > 0`` 是深度有效的前提；``d <= doff`` 一律无效（含负视差与重影区）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["StereoCalibration", "StereoCalibrationError"]

# 数值保护：d - doff 小于该值即视为无效（避免除零与溢出到无意义的巨大深度）
MIN_USABLE_DISPARITY_ABOVE_DOFF_PX = 1e-3


class StereoCalibrationError(ValueError):
    """非法/不可用标定。错误码按场景区分。"""

    def __init__(self, message: str, code: str = "E_INVALID_CALIBRATION") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StereoCalibration:
    """已核验的左校正相机系标定（不可变）。

    Parameters
    ----------
    calibration_id:
        标定来源标识（B 提供的 ID），全程写入运行元数据以对齐尺度。
    image_size:
        (width, height)，必须与被评估图像一致。
    fx_px:
        校正后左相机焦距（像素）。
    baseline_m:
        基线长度（米）。
    doff_px:
        主点差 ``cx_left - cx_right``（像素）。
    source:
        标定来源形式：``"P1P2"``（cv2.stereoRectify 投影矩阵）或 ``"KBD"``（fx/B/主点差）。
    is_rectified:
        是否已极线校正。为 ``False`` 时本模块拒绝换算（不猜）。
    frame:
        参考系说明，默认左校正相机系。
    """

    calibration_id: str
    image_size: Tuple[int, int]
    fx_px: float
    baseline_m: float
    doff_px: float
    source: str = "KBD"
    is_rectified: bool = False
    frame: str = (
        "left_rectified_camera: +X right, +Y down, +Z forward; depth = axial Z (m)"
    )
    extra: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------ 构造入口
    @classmethod
    def from_projection_matrices(
        cls,
        P1: Sequence[Sequence[float]],
        P2: Sequence[Sequence[float]],
        calibration_id: str,
        image_size: Tuple[int, int],
        is_rectified: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "StereoCalibration":
        """由 ``cv2.stereoRectify`` 的 ``P1``/``P2`` 构造（推荐）。

        ``P1[0,0] = fx``、``P1[0,2] = cx_l``、``P2[0,2] = cx_r``、``P2[0,3] = -fx*B``。
        """
        P1 = np.asarray(P1, dtype=np.float64)
        P2 = np.asarray(P2, dtype=np.float64)
        if P1.shape != (3, 4) or P2.shape != (3, 4):
            raise StereoCalibrationError(
                f"P1/P2 必须是 3x4 投影矩阵，收到 {P1.shape} / {P2.shape}"
            )
        fx = float(P1[0, 0])
        doff = float(P1[0, 2] - P2[0, 2])
        baseline = float(-P2[0, 3] / fx) if fx != 0 else float("nan")
        return cls(
            calibration_id=calibration_id,
            image_size=tuple(image_size),  # type: ignore[arg-type]
            fx_px=fx,
            baseline_m=baseline,
            doff_px=doff,
            source="P1P2",
            is_rectified=is_rectified,
            extra=extra,
        )

    @classmethod
    def from_intrinsics(
        cls,
        fx_px: float,
        baseline_m: float,
        calibration_id: str,
        image_size: Tuple[int, int],
        doff_px: float = 0.0,
        cx_left: Optional[float] = None,
        cx_right: Optional[float] = None,
        is_rectified: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "StereoCalibration":
        """由 fx、基线、主点差（或左右主点）构造。"""
        if (cx_left is not None) and (cx_right is not None):
            doff_px = float(cx_left) - float(cx_right)
        return cls(
            calibration_id=calibration_id,
            image_size=tuple(image_size),  # type: ignore[arg-type]
            fx_px=float(fx_px),
            baseline_m=float(baseline_m),
            doff_px=float(doff_px),
            source="KBD",
            is_rectified=is_rectified,
            extra=extra,
        )

    # ---------------------------------------------------------------- 校验
    def validate(self, expected_image_size: Optional[Tuple[int, int]] = None) -> None:
        if not self.calibration_id:
            raise StereoCalibrationError("calibration_id 不能为空")
        if not self.is_rectified:
            raise StereoCalibrationError(
                "标定未标记为已校正（is_rectified=False）：不得作为已校正输入强行换算，"
                "请向 B 索取校正后的 P1/P2 与校正状态",
                code="E_NOT_RECTIFIED",
            )
        w, h = self.image_size
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            raise StereoCalibrationError(f"image_size 非法：{self.image_size}")
        if expected_image_size is not None and tuple(expected_image_size) != tuple(
            self.image_size
        ):
            raise StereoCalibrationError(
                "标定分辨率与图像尺寸不一致："
                f"标定 {self.image_size} vs 图像 {tuple(expected_image_size)}；"
                "分辨率不符时内参与主点差都不可移植",
                code="E_CALIB_SIZE_MISMATCH",
            )
        if not (math.isfinite(self.fx_px) and self.fx_px > 0):
            raise StereoCalibrationError(f"fx_px 必须为正有限值，收到 {self.fx_px}")
        if not (math.isfinite(self.baseline_m) and self.baseline_m > 0):
            raise StereoCalibrationError(
                f"baseline_m 必须为正有限值，收到 {self.baseline_m}"
            )
        if not math.isfinite(self.doff_px):
            raise StereoCalibrationError(f"doff_px 必须有限，收到 {self.doff_px}")

    # ------------------------------------------------------------ 标量换算
    def depth_from_disparity_scalar(self, d_px: float) -> float:
        """单点换算，返回轴向 Z（米）；非法视差返回 ``nan``。"""
        denom = d_px - self.doff_px
        if denom <= MIN_USABLE_DISPARITY_ABOVE_DOFF_PX:
            return float("nan")
        return self.fx_px * self.baseline_m / denom

    def disparity_from_depth_scalar(self, z_m: float) -> float:
        """反算：``d = doff + fx * B / z``；非正深度返回 ``nan``。"""
        if not (math.isfinite(z_m) and z_m > 0):
            return float("nan")
        return self.doff_px + self.fx_px * self.baseline_m / z_m

    # ------------------------------------------------------------ 数组换算
    def depth_from_disparity(
        self, disparity_px: np.ndarray, valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """数组换算 → ``(depth_m, depth_valid_mask)``。

        * 分母 ``d - doff`` 必须严格大于 :data:`MIN_USABLE_DISPARITY_ABOVE_DOFF_PX`。
        * ``valid_mask``（视差有效性）与分母条件**同时**满足才算深度有效。
        * 无效位置固定为 ``NaN``，不做任何静默补全。
        """
        d = np.asarray(disparity_px, dtype=np.float64)
        denom = d - float(self.doff_px)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.fx_px * self.baseline_m / denom
        denom_ok = denom > MIN_USABLE_DISPARITY_ABOVE_DOFF_PX
        finite = np.isfinite(z)
        valid = denom_ok & finite
        if valid_mask is not None:
            valid = valid & np.asarray(valid_mask, dtype=bool)
        z = np.where(valid, z, np.nan).astype(np.float32)
        return z, valid

    # ---------------------------------------------------------------- 输出
    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "calibration_id": self.calibration_id,
            "image_size": list(self.image_size),
            "fx_px": self.fx_px,
            "baseline_m": self.baseline_m,
            "doff_px": self.doff_px,
            "source": self.source,
            "is_rectified": self.is_rectified,
            "frame": self.frame,
            "min_usable_disparity_px": self.doff_px
            + MIN_USABLE_DISPARITY_ABOVE_DOFF_PX,
            "depth_model": "Z = fx * B / (d - doff)",
            "extra": self.extra or {},
        }

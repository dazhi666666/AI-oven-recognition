"""Middlebury 2014 标定读取；不沿用其他数据集的单位假设。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oven_stereo.contracts import RectifiedStereo


@dataclass(frozen=True)
class MiddleburyCalibration:
    stereo: RectifiedStereo
    cam0: np.ndarray
    cam1: np.ndarray
    ndisp: int


def read_middlebury_calibration(path: str | Path, calibration_id: str) -> MiddleburyCalibration:
    fields = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or key in fields:
            raise ValueError(f"标定行格式错误或字段重复：{line}")
        fields[key] = value.strip()
    required = {"cam0", "cam1", "doffs", "baseline", "width", "height", "ndisp"}
    if missing := required - fields.keys():
        raise ValueError(f"标定缺少字段：{sorted(missing)}")

    cameras = []
    for key in ("cam0", "cam1"):
        value = fields[key]
        if not (value.startswith("[") and value.endswith("]")):
            raise ValueError(f"{key} 必须是 3x3 相机矩阵")
        try:
            matrix = np.array([[float(x) for x in row.split()] for row in value[1:-1].split(";")])
        except ValueError as exc:
            raise ValueError(f"{key} 不是有效矩阵") from exc
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"{key} 必须是有限的 3x3 矩阵")
        f, cx, cy = matrix[0, 0], matrix[0, 2], matrix[1, 2]
        expected = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        if f <= 0 or not np.allclose(matrix, expected, atol=1e-6, rtol=0):
            raise ValueError(f"{key} 不符合已校正 Middlebury 相机矩阵形式")
        cameras.append(matrix)
    cam0, cam1 = cameras
    if not np.allclose(cam0[[0, 1], [0, 2]], cam1[[0, 1], [0, 2]], atol=1e-3, rtol=0):
        raise ValueError("左右相机的焦距或垂直主点不一致")

    doffs = float(fields["doffs"])
    if not np.isfinite(doffs) or not np.isclose(doffs, cam1[0, 2] - cam0[0, 2], atol=0.002, rtol=0):
        raise ValueError("doffs 必须等于右主点 x 减左主点 x")
    size = (int(fields["width"]), int(fields["height"]))
    ndisp = int(fields["ndisp"])
    if min(size) <= 0 or ndisp <= 0:
        raise ValueError("图像尺寸和 ndisp 必须是正整数")
    stereo = RectifiedStereo(
        calibration_id=calibration_id,
        image_size=size,
        f_px=float(cam0[0, 0]),
        baseline_m=float(fields["baseline"]) / 1000.0,
        doffs_px=doffs,
    )
    stereo.validate()
    return MiddleburyCalibration(stereo, cam0, cam1, ndisp)

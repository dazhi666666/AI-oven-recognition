"""pytest 共享夹具：把 ``src/`` 加入路径，并提供人工构造输入。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo.calibration import StereoCalibration  # noqa: E402
from stereo.testpatterns import render_slanted_plane_pair  # noqa: E402

# 人工构造标定（非真实相机）
ARTIFICIAL_FX = 700.0
ARTIFICIAL_B = 0.1
ARTIFICIAL_CX_L = 160.5
ARTIFICIAL_CX_R = 158.5


@pytest.fixture(scope="session")
def pattern():
    """固定的斜平面人工图对（session 级，避免重复渲染）。"""
    return render_slanted_plane_pair(seed=3)


@pytest.fixture
def pair_paths(tmp_path, pattern):
    """把人工图对落盘，返回 (left_path, right_path)。"""
    left = tmp_path / "left_ARTIFICIAL.png"
    right = tmp_path / "right_ARTIFICIAL.png"
    cv2.imwrite(str(left), pattern["left"])
    cv2.imwrite(str(right), pattern["right"])
    return left, right


@pytest.fixture
def artificial_calibration(pattern):
    height, width = pattern["left"].shape[:2]
    return StereoCalibration.from_intrinsics(
        fx_px=ARTIFICIAL_FX,
        baseline_m=ARTIFICIAL_B,
        cx_left=ARTIFICIAL_CX_L,
        cx_right=ARTIFICIAL_CX_R,
        calibration_id="ARTIFICIAL-unit-test",
        image_size=(width, height),
        is_rectified=True,
    )

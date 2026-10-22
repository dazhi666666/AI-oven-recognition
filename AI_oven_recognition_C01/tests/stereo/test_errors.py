"""最小错误用例：必须显式拒绝，不得静默继续。"""

from __future__ import annotations

import numpy as np
import pytest

from stereo.calibration import StereoCalibration
from stereo.config import C01_DEFAULT, SGBMConfig, StereoConfigError
from stereo.pipeline import StereoInputError, estimate_depth
from stereo.sgbm import compute_raw_disparity

pytestmark = pytest.mark.unit


def _calib(pattern, **over):
    height, width = pattern["left"].shape[:2]
    base = dict(
        fx_px=700.0, baseline_m=0.1, doff_px=2.0,
        calibration_id="test", image_size=(width, height), is_rectified=True,
    )
    base.update(over)
    return StereoCalibration.from_intrinsics(**base)


def test_wrong_size_pair_rejected(pair_paths, pattern, tmp_path):
    """尺寸不一致必须报错，而不是自动对齐/裁切。"""
    left, right = pair_paths
    import cv2

    small = tmp_path / "left_small.png"
    cv2.imwrite(str(small), pattern["left"][:, :-17])
    with pytest.raises(StereoInputError) as exc:
        estimate_depth(small, right, config=C01_DEFAULT)
    assert exc.value.code == "E_SIZE_MISMATCH"


def test_unreadable_and_missing_inputs(pair_paths, tmp_path):
    left, right = pair_paths
    with pytest.raises(StereoInputError) as exc:
        estimate_depth(tmp_path / "missing.png", right, config=C01_DEFAULT)
    assert exc.value.code == "E_READ_FAIL"

    not_image = tmp_path / "not_an_image.png"
    not_image.write_text("这不是图像", encoding="utf-8")
    with pytest.raises(StereoInputError) as exc:
        estimate_depth(not_image, right, config=C01_DEFAULT)
    assert exc.value.code == "E_READ_FAIL"

    with pytest.raises(StereoInputError) as exc:
        estimate_depth(tmp_path, right, config=C01_DEFAULT)
    assert exc.value.code == "E_READ_FAIL"


def test_16bit_input_rejected_explicitly(pair_paths, tmp_path, pattern):
    """位深缩放会改变视差尺度，未与 B 确认前必须拒绝而不是猜测。"""
    import cv2

    left, right = pair_paths
    sixteen = tmp_path / "left16.png"
    cv2.imwrite(str(sixteen), (pattern["left"].astype(np.uint16) * 257))
    with pytest.raises(StereoInputError) as exc:
        estimate_depth(sixteen, right, config=C01_DEFAULT)
    assert exc.value.code in ("E_UNSUPPORTED_DEPTH", "E_UNSUPPORTED_CHANNELS")


def test_illegal_search_params_rejected_before_opencv(pair_paths):
    left, right = pair_paths
    for kwargs, code in (
        ({"num_disparities": 30}, "E_INVALID_SEARCH_PARAMS"),
        ({"block_size": 4}, "E_INVALID_SEARCH_PARAMS"),
        ({"min_disparity": -8}, "E_INVALID_SEARCH_PARAMS"),
    ):
        with pytest.raises((StereoConfigError, StereoInputError)) as exc:
            estimate_depth(left, right, config=SGBMConfig(name="bad", **kwargs))
        assert getattr(exc.value, "code", None) == code


def test_raw_compute_rejects_non_gray_and_mismatched_shapes(pattern):
    left = pattern["left"]
    right = pattern["right"]
    with pytest.raises(ValueError):
        compute_raw_disparity(left, right[:-5], C01_DEFAULT)  # 形状不同
    color = np.dstack([left] * 3)
    with pytest.raises(ValueError):
        compute_raw_disparity(color, color, C01_DEFAULT)  # 非单通道


def test_unrectified_calibration_rejected(pair_paths, pattern):
    left, right = pair_paths
    with pytest.raises(Exception) as exc:
        estimate_depth(left, right, config=C01_DEFAULT,
                       calibration=_calib(pattern, is_rectified=False))
    assert getattr(exc.value, "code", None) == "E_NOT_RECTIFIED"


def test_calibration_resolution_mismatch_rejected(pair_paths, pattern):
    left, right = pair_paths
    with pytest.raises(Exception) as exc:
        estimate_depth(left, right, config=C01_DEFAULT,
                       calibration=_calib(pattern, image_size=(640, 480)))
    assert getattr(exc.value, "code", None) == "E_CALIB_SIZE_MISMATCH"


@pytest.mark.parametrize("field,value", [("fx_px", 0.0), ("fx_px", -700.0),
                                         ("baseline_m", 0.0), ("baseline_m", -0.1)])
def test_illegal_calibration_values_rejected(pair_paths, pattern, field, value):
    left, right = pair_paths
    with pytest.raises(Exception) as exc:
        estimate_depth(left, right, config=C01_DEFAULT,
                       calibration=_calib(pattern, **{field: value}))
    assert getattr(exc.value, "code", None) == "E_INVALID_CALIBRATION"


def test_bad_depth_range_argument(pair_paths, pattern):
    left, right = pair_paths
    with pytest.raises(StereoInputError) as exc:
        estimate_depth(left, right, config=C01_DEFAULT,
                       calibration=_calib(pattern), depth_range_m=(1.0,))
    assert exc.value.code == "E_BAD_DEPTH_RANGE"


def test_errors_do_not_leave_partial_outputs(pair_paths, pattern, tmp_path):
    """出错时不写任何半成品输出（结果对象根本没生成）。"""
    left, right = pair_paths
    out = tmp_path / "should_not_exist"
    with pytest.raises(Exception):
        estimate_depth(left, right, config=C01_DEFAULT,
                       calibration=_calib(pattern, fx_px=-1.0))
    assert not out.exists()

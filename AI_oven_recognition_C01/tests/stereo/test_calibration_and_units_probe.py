"""缩放实测核验与标定换算的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from stereo.calibration import StereoCalibration, StereoCalibrationError
from stereo.config import C01_DEFAULT
from stereo.reference import handcheck_depth_m, handcheck_disparity_px
from stereo.sgbm import clear_unit_verification_cache, verify_disparity_units

pytestmark = pytest.mark.unit


def test_unit_verification_uses_real_api_output():
    clear_unit_verification_cache()
    report = verify_disparity_units(C01_DEFAULT, force=True)
    assert report.verified is True
    # 本机实测：整型输出为定点（×16）
    assert report.raw_dtype in ("int16", "uint16")
    assert report.verified_scale == 16.0
    assert report.dtype_prior_scale == 16.0
    chosen_err = report.evidence["chosen_median_abs_err_px"]
    assert chosen_err is not None and chosen_err < 0.5
    # 错误假设（不除 16）必须明显更差，否则说明核验没有区分力
    rejected = report.evidence["rejected_median_abs_err_px"]
    assert rejected is None or rejected > 4 * chosen_err
    assert report.evidence["pattern"]["provenance"] == "artificial_test_pattern"


def test_unit_verification_is_cached_but_consistent():
    clear_unit_verification_cache()
    a = verify_disparity_units(C01_DEFAULT)
    b = verify_disparity_units(C01_DEFAULT)
    assert a is b
    c = verify_disparity_units(C01_DEFAULT, force=True)
    assert c.verified_scale == a.verified_scale
    assert c.evidence["chosen_median_abs_err_px"] == pytest.approx(
        a.evidence["chosen_median_abs_err_px"], abs=1e-9
    )


# ------------------------------------------------------------------ 标定
def test_hand_computable_depth_conversion(artificial_calibration):
    calib = artificial_calibration
    # Z = fx*B/(d-doff) = 700*0.1/(d-2) = 70/(d-2)
    for d in (22.0, 27.0, 42.0):
        expected = 70.0 / (d - 2.0)
        assert calib.depth_from_disparity_scalar(d) == pytest.approx(expected, rel=1e-12)
        assert handcheck_depth_m(700.0, 0.1, 2.0, d) == pytest.approx(expected, rel=1e-12)


def test_forgetting_principal_point_difference_is_detectable():
    """漏掉主点差会得到明显不同的深度，这个测试固定住该行为的可见性。"""
    calib = StereoCalibration.from_intrinsics(
        fx_px=700.0, baseline_m=0.1, cx_left=160.5, cx_right=158.5,
        calibration_id="x", image_size=(320, 240),
    )
    with_doff = calib.depth_from_disparity_scalar(22.0)
    without_doff = 700.0 * 0.1 / 22.0
    assert with_doff == pytest.approx(3.5)
    assert without_doff == pytest.approx(3.1818, rel=1e-4)
    assert abs(with_doff - without_doff) > 0.3


def test_depth_and_disparity_roundtrip():
    calib = StereoCalibration.from_intrinsics(
        fx_px=700.0, baseline_m=0.1, doff_px=2.0,
        calibration_id="x", image_size=(320, 240),
    )
    for z in (0.5, 1.0, 2.5, 10.0):
        d = calib.disparity_from_depth_scalar(z)
        assert calib.depth_from_disparity_scalar(d) == pytest.approx(z, rel=1e-9)
        assert handcheck_disparity_px(700.0, 0.1, 2.0, z) == pytest.approx(d, rel=1e-12)


def test_array_conversion_marks_invalid_and_never_fills():
    calib = StereoCalibration.from_intrinsics(
        fx_px=700.0, baseline_m=0.1, doff_px=2.0,
        calibration_id="x", image_size=(4, 2),
    )
    d = np.array([[2.0, 1.999, 0.0, np.nan], [22.0, np.nan, -3.0, 2.006]], dtype=np.float32)
    z, valid = calib.depth_from_disparity(d)
    assert valid.tolist() == [[False, False, False, False], [True, False, False, True]]
    assert np.isnan(z[0, :]).all()
    np.testing.assert_allclose(z[1, 0], 70.0 / 20.0, rtol=1e-6)
    assert np.isnan(z[1, 1])


def test_p1p2_construction_matches_intrinsics():
    fx, cx_l, cx_r, B = 700.0, 160.5, 158.5, 0.1
    P1 = [[fx, 0, cx_l, 0], [0, fx, 120.0, 0], [0, 0, 1, 0]]
    P2 = [[fx, 0, cx_r, -fx * B], [0, fx, 120.0, 0], [0, 0, 1, 0]]
    a = StereoCalibration.from_projection_matrices(P1, P2, "P", (320, 240))
    b = StereoCalibration.from_intrinsics(fx, B, cx_left=cx_l, cx_right=cx_r, calibration_id="K",
                                          image_size=(320, 240))
    assert a.fx_px == pytest.approx(b.fx_px)
    assert a.baseline_m == pytest.approx(b.baseline_m)
    assert a.doff_px == pytest.approx(b.doff_px)
    assert a.depth_from_disparity_scalar(22.0) == pytest.approx(
        b.depth_from_disparity_scalar(22.0)
    )


@pytest.mark.parametrize(
    "factory,code",
    [
        (dict(fx_px=0.0, baseline_m=0.1), "E_INVALID_CALIBRATION"),
        (dict(fx_px=-700.0, baseline_m=0.1), "E_INVALID_CALIBRATION"),
        (dict(fx_px=700.0, baseline_m=0.0), "E_INVALID_CALIBRATION"),
        (dict(fx_px=700.0, baseline_m=-0.1), "E_INVALID_CALIBRATION"),
        (dict(fx_px=700.0, baseline_m=0.1, is_rectified=False), "E_NOT_RECTIFIED"),
    ],
)
def test_illegal_calibration_rejected(factory, code):
    calib = StereoCalibration.from_intrinsics(
        calibration_id="bad", image_size=(320, 240), **factory
    )
    with pytest.raises(StereoCalibrationError) as exc:
        calib.validate()
    assert exc.value.code == code


def test_calibration_size_mismatch_rejected(artificial_calibration):
    with pytest.raises(StereoCalibrationError) as exc:
        artificial_calibration.validate(expected_image_size=(640, 480))
    assert exc.value.code == "E_CALIB_SIZE_MISMATCH"


def test_bad_projection_matrix_shape_rejected():
    with pytest.raises(StereoCalibrationError):
        StereoCalibration.from_projection_matrices(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0, 0]] * 3, "P", (320, 240)
        )

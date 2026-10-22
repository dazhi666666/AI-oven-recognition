"""参考换算与误差核验工具的行为测试（参考侧，不参与预测链路）。"""

from __future__ import annotations

import numpy as np
import pytest

from stereo.calibration import StereoCalibration
from stereo.reference import (
    depth_from_reference_disparity,
    handcheck_depth_m,
    handcheck_disparity_px,
    reference_error_metrics,
)

pytestmark = pytest.mark.unit


def test_handcheck_matches_calibration_module(artificial_calibration):
    calib = artificial_calibration
    for d in (3.0, 22.0, 100.0):
        assert handcheck_depth_m(
            calib.fx_px, calib.baseline_m, calib.doff_px, d
        ) == pytest.approx(calib.depth_from_disparity_scalar(d), rel=1e-12)


def test_handcheck_invalid_disparities_return_nan():
    assert np.isnan(handcheck_depth_m(700.0, 0.1, 2.0, 2.0))   # d == doff
    assert np.isnan(handcheck_depth_m(700.0, 0.1, 2.0, 1.0))   # d < doff
    assert np.isnan(handcheck_depth_m(700.0, 0.1, 2.0, 0.0))
    with pytest.raises(ValueError):
        handcheck_depth_m(0.0, 0.1, 0.0, 10.0)
    assert np.isnan(handcheck_disparity_px(700.0, 0.1, 2.0, -1.0))


def test_reference_conversion_is_labelled_reference():
    ref_d = np.array([[22.0, 2.0, np.nan]], dtype=np.float32)
    z, valid = depth_from_reference_disparity(ref_d, 700.0, 0.1, 2.0)
    assert valid.tolist() == [[True, False, False]]
    assert z[0, 0] == pytest.approx(3.5)
    assert np.isnan(z[0, 1]) and np.isnan(z[0, 2])


def test_error_metrics_report_error_and_coverage_together():
    pred = np.array([[10.0, 12.0, 20.0, np.nan]], dtype=np.float32)
    ref = np.array([[10.0, 14.0, 20.0, 30.0]], dtype=np.float32)
    m = reference_error_metrics(pred, ref, bad_thresholds=(1.0, 2.0))
    assert m["eval_pixels"] == 3            # 第 4 个像素预测无效
    assert m["eval_coverage"] == pytest.approx(3 / 4)
    assert m["prediction_coverage"] == pytest.approx(3 / 4)
    assert m["reference_coverage"] == pytest.approx(1.0)
    assert m["mae_px"] == pytest.approx((0 + 2 + 0) / 3)
    assert m["bad1"] == pytest.approx(1 / 3)   # >1 px 的比例
    assert m["bad2"] == pytest.approx(0.0)     # >2 px 的比例
    assert m["bad_thresholds_px"] == [1.0, 2.0]
    # 覆盖率与误差必须同时出现在结果里
    assert "eval_coverage" in m and "mae_px" in m


def test_error_metrics_respect_explicit_mask():
    pred = np.array([[10.0, 11.0]], dtype=np.float32)
    ref = np.array([[10.0, 20.0]], dtype=np.float32)
    mask = np.array([[False, True]])
    m = reference_error_metrics(pred, ref, eval_mask=mask)
    assert m["eval_mask_source"] == "explicit"
    assert m["eval_pixels"] == 1
    assert m["mae_px"] == pytest.approx(9.0)


def test_error_metrics_without_common_pixels_do_not_fake_success():
    pred = np.array([[np.nan]], dtype=np.float32)
    ref = np.array([[1.0]], dtype=np.float32)
    m = reference_error_metrics(pred, ref)
    assert m["eval_pixels"] == 0
    assert "note" in m and "mae_px" not in m


def test_error_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        reference_error_metrics(np.zeros((2, 2)), np.zeros((2, 3)))

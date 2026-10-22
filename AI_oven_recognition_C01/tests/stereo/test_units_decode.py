"""定点缩放与无效编码解码的单元测试（用受控原始数组，不依赖 SGBM 精度）。"""

from __future__ import annotations

import numpy as np
import pytest

from stereo.config import SGBMConfig
from stereo.sgbm import (
    StereoUnitError,
    decode_disparity,
    expected_scale_from_dtype,
    invalid_raw_sentinel,
)

pytestmark = pytest.mark.unit

CFG = SGBMConfig()


def test_dtype_prior_scale_rules():
    assert expected_scale_from_dtype(np.dtype(np.int16)) == 16.0
    assert expected_scale_from_dtype(np.dtype(np.uint16)) == 16.0
    assert expected_scale_from_dtype(np.dtype(np.float32)) == 1.0
    assert expected_scale_from_dtype(np.dtype(np.float64)) == 1.0
    with pytest.raises(StereoUnitError) as exc:
        expected_scale_from_dtype(np.dtype(np.uint8))
    assert exc.value.code == "E_UNSUPPORTED_RAW_DTYPE"


def test_invalid_sentinel_follows_min_disparity():
    assert invalid_raw_sentinel(CFG, 16.0, np.dtype(np.int16)) == -16
    cfg16 = SGBMConfig(min_disparity=16, num_disparities=64)
    assert invalid_raw_sentinel(cfg16, 16.0, np.dtype(np.int16)) == 240  # (16-1)*16
    # 无符号类型按位回绕
    assert invalid_raw_sentinel(CFG, 16.0, np.dtype(np.uint16)) == 65520


def test_fixed_point_decode_divides_once_and_keeps_nan():
    raw = np.array([[16 * 8, 16 * 20, -16], [16 * 0, 16 * 33, 16 * 1]], dtype=np.int16)
    d, valid, info = decode_disparity(raw, CFG, scale=16.0)
    assert d.dtype == np.float32
    np.testing.assert_allclose(
        d[valid], np.array([8.0, 20.0, 0.0, 33.0, 1.0], dtype=np.float32)
    )
    assert valid.tolist() == [[True, True, False], [True, True, True]]
    # 无效位置必须是 NaN，不能被填成 0 或 min-1
    assert np.isnan(d[0, 2])
    assert info["scale_applied"] == 16.0
    assert info["invalid_raw_value"] == -16.0
    assert info["valid_pixels"] == 5 and info["total_pixels"] == 6


def test_dtype_prior_applied_when_scale_not_given():
    raw = np.full((3, 3), 16 * 12, dtype=np.int16)
    d, valid, info = decode_disparity(raw, CFG)
    assert info["scale_source"] == "dtype_prior"
    np.testing.assert_allclose(d[valid], 12.0)
    assert valid.all()


def test_float_output_must_not_be_divided_again():
    """已是像素浮点值的输出再除 16 是典型越界错误，必须不触发。"""
    raw = np.array([[8.0, 20.0, -1.0]], dtype=np.float32)
    d, valid, info = decode_disparity(raw, CFG, scale=1.0)
    np.testing.assert_allclose(d[0, :2], np.array([8.0, 20.0], dtype=np.float32))
    assert valid.tolist() == [[True, True, False]]  # -1 = minDisparity-1
    assert info["scale_applied"] == 1.0


def test_uint16_fixed_point_decode():
    raw = np.array([[16 * 5, 65520]], dtype=np.uint16)  # 65520 = 回绕后的 -16
    d, valid, info = decode_disparity(raw, CFG, scale=16.0)
    assert valid.tolist() == [[True, False]]
    np.testing.assert_allclose(d[0, 0], 5.0)
    assert np.isnan(d[0, 1])


def test_nan_and_inf_in_float_output_are_invalid():
    raw = np.array([[np.nan, np.inf, 12.0]], dtype=np.float32)
    _, valid, _ = decode_disparity(raw, CFG, scale=1.0)
    assert valid.tolist() == [[False, False, True]]


def test_nonzero_min_disparity_decoding():
    cfg = SGBMConfig(min_disparity=16, num_disparities=64)
    raw = np.array([[16 * 16, 16 * 40, 240]], dtype=np.int16)  # 240 = (16-1)*16
    d, valid, _ = decode_disparity(raw, cfg, scale=16.0)
    assert valid.tolist() == [[True, True, False]]
    np.testing.assert_allclose(d[0, :2], np.array([16.0, 40.0], dtype=np.float32))

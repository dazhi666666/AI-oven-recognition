import numpy as np
import pytest

from oven_stereo.stereo import SGBMConfig, compute_disparity, decode_disparity


def test_decode_divides_by_16_and_marks_invalid():
    raw = np.array([[-16, 0, 16 * 20 + 8]], dtype=np.int16)
    disparity, valid = decode_disparity(raw, min_disparity=0)
    assert valid.tolist() == [[False, True, True]]
    assert np.isnan(disparity[0, 0])
    assert disparity[0, 1:].tolist() == [0.0, 20.5]
    assert disparity.dtype == np.float32


def test_decode_uses_min_disparity_for_invalid_code():
    # min_disparity=4 时无效编码为 (4 - 1) * 16 = 48
    raw = np.array([[48, 64]], dtype=np.int16)
    disparity, valid = decode_disparity(raw, min_disparity=4)
    assert valid.tolist() == [[False, True]]
    assert disparity[0, 1] == 4.0


def test_decode_rejects_non_fixed_point_input():
    with pytest.raises(TypeError):
        decode_disparity(np.zeros((2, 2), dtype=np.float32), min_disparity=0)


def test_constant_shift_is_recovered(make_shift_pair, c01_config):
    left, right = make_shift_pair(shift=20)
    disparity, valid = compute_disparity(left, right, c01_config)
    region = valid[:, c01_config.left_invalid_columns :]
    values = disparity[:, c01_config.left_invalid_columns :][region]
    assert region.mean() > 0.9
    assert np.median(values) == pytest.approx(20.0, abs=0.25)


def test_slanted_plane_subpixel_accuracy(slanted_pair, c01_config):
    left, right, d_gt, valid_gt = slanted_pair
    disparity, valid = compute_disparity(left, right, c01_config)
    n = c01_config.left_invalid_columns
    sel = valid & valid_gt
    sel[:, :n] = False
    err = np.abs(disparity[sel] - d_gt[sel])
    assert sel.sum() > 0.9 * valid_gt[:, n:].sum()
    assert np.median(err) < 0.25


@pytest.mark.parametrize("min_d,num_d", [(0, 32), (0, 64), (4, 32)])
def test_left_border_is_invalid(make_shift_pair, min_d, num_d):
    config = SGBMConfig(min_disparity=min_d, num_disparities=num_d)
    left, right = make_shift_pair(shift=20)
    _, valid = compute_disparity(left, right, config)
    n = config.left_invalid_columns
    assert n == min_d + num_d
    assert not valid[:, :n].any()
    assert valid[:, n].any()


def test_same_input_gives_same_output(slanted_pair, c01_config):
    left, right, _, _ = slanted_pair
    a, va = compute_disparity(left, right, c01_config)
    b, vb = compute_disparity(left, right, c01_config)
    assert np.array_equal(va, vb)
    assert np.array_equal(a[va], b[vb])


def test_input_checks(c01_config):
    gray = np.zeros((40, 80), dtype=np.uint8)
    with pytest.raises(ValueError, match="尺寸"):
        compute_disparity(gray, gray[:, :70], c01_config)
    with pytest.raises(ValueError, match="单通道"):
        color = np.zeros((40, 80, 3), dtype=np.uint8)
        compute_disparity(color, color, c01_config)
    with pytest.raises(ValueError, match="8 位"):
        compute_disparity(gray.astype(np.float32), gray.astype(np.float32), c01_config)

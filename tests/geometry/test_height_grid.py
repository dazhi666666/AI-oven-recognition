import json

import numpy as np
import pytest

from oven_stereo.geometry import measure_height_grid


def _measure(heights, areas=1e-6, target=None, valid=None):
    return measure_height_grid(
        heights, areas,
        np.ones(heights.shape, dtype=bool) if target is None else target,
        np.ones(heights.shape, dtype=bool) if valid is None else valid,
        sample_id="test", config_id="a02_test_v1",
    )


@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 1e-9), (np.float32, 1e-5)])
def test_block_dimensions(dtype, tolerance):
    # 预期值独立于被测函数：100×60×20 mm = 120 mL。
    result = _measure(np.full((60, 100), 0.02, dtype=dtype))
    assert result.status == "ok"
    assert result.thickness_mm == pytest.approx(20.0, rel=0, abs=tolerance)
    assert result.volume_ml == pytest.approx(120.0, rel=0, abs=tolerance)
    assert result.coverage == 1.0
    assert result.height_m.dtype == np.float64


def test_nonuniform_areas_and_max_not_percentile():
    # 0.01×1e-4 + 0.03×2e-4 = 7e-6 m³ = 7 mL。
    result = _measure(np.array([[0.01, 0.03]]), np.array([[1e-4, 2e-4]]))
    assert result.thickness_mm == pytest.approx(30.0)
    assert result.volume_ml == pytest.approx(7.0)


def test_background_excluded_and_inputs_unchanged():
    heights = np.array([[0.02, np.nan, 100.0]])
    target = np.array([[True, False, False]])
    valid = np.array([[True, False, True]])
    before = heights.copy()
    result = _measure(heights, target=target, valid=valid)
    assert result.status == "ok"
    assert result.volume_ml == pytest.approx(0.02)
    np.testing.assert_equal(heights, before)
    np.testing.assert_array_equal(valid, [[True, False, True]])


def test_empty_is_not_missing_data():
    result = _measure(np.full((2, 3), np.nan), target=np.zeros((2, 3), dtype=bool))
    assert result.status == "empty"
    assert result.volume_ml == 0.0
    assert result.thickness_mm == 0.0
    assert result.coverage is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01])
def test_unreliable_target_rejected(value):
    result = _measure(np.array([[0.02, value]]))
    assert result.status == "rejected"
    assert result.volume_ml is None and result.thickness_mm is None
    assert result.coverage == 0.5
    assert not result.valid_mask[0, 1]
    assert np.isnan(result.height_m[0, 1])


def test_false_valid_flag_rejects_finite_target():
    result = _measure(np.array([[0.02, 0.02]]), valid=np.array([[True, False]]))
    assert result.status == "rejected"
    assert result.volume_ml is None
    assert np.isnan(result.height_m[0, 1])


@pytest.mark.parametrize("areas", [-1e-6, 0.0, np.nan, np.inf, np.ones((2, 2)), True])
def test_invalid_area_raises(areas):
    with pytest.raises(ValueError, match="cell_area_m2"):
        _measure(np.ones((2, 3)), areas)


@pytest.mark.parametrize("name", ["target", "valid"])
@pytest.mark.parametrize("mask", [np.ones((3, 2), dtype=bool), np.ones((2, 3))])
def test_mask_shape_and_dtype(name, mask):
    with pytest.raises(ValueError, match="mask"):
        _measure(np.ones((2, 3)), **{name: mask})


@pytest.mark.parametrize("heights", [np.ones(3), np.empty((0, 3)), np.ones((1, 1), dtype=complex)])
def test_invalid_height_array(heights):
    with pytest.raises(ValueError, match="height_m"):
        _measure(heights)


def test_overflow_rejected():
    result = _measure(np.array([[1e308]]), areas=1e308)
    assert result.status == "rejected"
    assert result.thickness_mm is None and result.volume_ml is None


def test_zero_is_valid_height_not_missing():
    result = _measure(np.zeros((1, 2)))
    assert result.status == "ok"  # 目标掩膜非空，不推断成空托盘
    assert result.coverage == 1.0
    assert result.volume_ml == 0.0


@pytest.mark.parametrize("value", ["", " ", 123, None])
def test_invalid_identifiers(value):
    with pytest.raises(ValueError, match="非空字符串"):
        measure_height_grid(np.ones((1, 1)), 1e-6, np.ones((1, 1), dtype=bool),
                            np.ones((1, 1), dtype=bool), sample_id=value, config_id="test")


@pytest.mark.parametrize("state", ["ok", "empty", "rejected"])
def test_saved_contract_and_null(tmp_path, state):
    heights = np.array([[0.02, 0.02]])
    target = np.full((1, 2), state != "empty", dtype=bool)
    valid = np.array([[True, state != "rejected"]])
    result = _measure(heights, target=target, valid=valid)
    result.save(tmp_path)
    summary = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert summary["status"] == state
    assert summary["volume_ml"] == result.volume_ml
    assert summary["thickness_mm"] == result.thickness_mm
    assert summary["grid_representation"] == "support_plane_cells"
    for name in ("height_m", "cell_area_m2", "target_mask", "valid_mask"):
        np.testing.assert_equal(np.load(tmp_path / f"{name}.npy"), getattr(result, name))
    if state == "rejected":
        assert '"volume_ml": null' in (tmp_path / "result.json").read_text()

import pytest

from oven_stereo.calibration.budget import estimate_disparity_budget


def _budget(**overrides):
    params = dict(image_width_px=320, horizontal_fov_deg=90.0, baseline_m=0.075,
                  distance_m=1.0, relative_depth_error=0.05)
    params.update(overrides)
    return estimate_disparity_budget(**params)


def test_budget_independent_hand_calculation():
    # tan(45°)=1：f=160 px，d=12 px；一阶0.6，严格界限4/7。
    result = _budget()
    assert result["f_px"] == pytest.approx(160.0)
    assert result["disparity_px"] == pytest.approx(12.0)
    assert result["disparity_error_linear_px"] == pytest.approx(0.6)
    assert result["disparity_error_symmetric_px"] == pytest.approx(4 / 7)


def test_width_scaling():
    assert _budget(image_width_px=160)["disparity_px"] == pytest.approx(6.0)


def test_symmetric_bound_meets_depth_error_for_both_signs():
    result = _budget()
    for sign in (-1, 1):
        estimated_z = 12.0 / (12.0 + sign * result["disparity_error_symmetric_px"])
        assert abs(estimated_z - 1.0) <= 0.05 + 1e-14


@pytest.mark.parametrize("overrides", [
    {"image_width_px": 0}, {"image_width_px": 320.5}, {"image_width_px": True},
    {"horizontal_fov_deg": 0}, {"horizontal_fov_deg": 180},
    {"baseline_m": -0.1}, {"distance_m": 0}, {"distance_m": float("nan")},
    {"relative_depth_error": 0}, {"relative_depth_error": 1},
])
def test_invalid_budget_parameters(overrides):
    with pytest.raises(ValueError):
        _budget(**overrides)

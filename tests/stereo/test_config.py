import pytest

from oven_stereo.stereo import SGBMConfig, StereoConfigError, disparity_range_for_depth


def test_yaml_config_loads(c01_config):
    assert c01_config.num_disparities == 64
    assert c01_config.max_disparity == 63


def test_config_id_depends_only_on_parameters():
    a = SGBMConfig(name="a")
    assert SGBMConfig(name="b").config_id == a.config_id
    assert SGBMConfig(name="a", block_size=7).config_id != a.config_id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_disparities": 30},
        {"num_disparities": 0},
        {"block_size": 4},
        {"p1": 800, "p2": 200},
        {"mode": "BM"},
        {"uniqueness_ratio": 101},
        {"pre_filter_cap": 0},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(StereoConfigError):
        SGBMConfig(**kwargs)


def test_unknown_yaml_field_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("num_disparities: 64\nblocksize: 5\n", encoding="utf-8")
    with pytest.raises(StereoConfigError, match="blocksize"):
        SGBMConfig.from_yaml(path)


def test_disparity_range_hand_calc():
    # f = 114 px，B = 0.07 m：1.5 m 处 d = 7.98/1.5 = 5.32，0.3 m 处 d = 7.98/0.3 = 26.6
    lo, hi = disparity_range_for_depth(114, 0.07, 0.3, 1.5)
    assert lo == pytest.approx(5.32)
    assert hi == pytest.approx(26.6)
    lo2, hi2 = disparity_range_for_depth(114, 0.07, 0.3, 1.5, doffs_px=2.0)
    assert (lo2, hi2) == pytest.approx((3.32, 24.6))


def test_disparity_range_rejects_bad_input():
    with pytest.raises(StereoConfigError):
        disparity_range_for_depth(114, 0.07, 1.5, 0.3)
    with pytest.raises(StereoConfigError):
        disparity_range_for_depth(0, 0.07, 0.3, 1.5)

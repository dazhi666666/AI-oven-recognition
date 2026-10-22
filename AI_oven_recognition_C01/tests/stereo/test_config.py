"""固定配置与视差搜索范围依据的单元测试。"""

from __future__ import annotations

import pytest

from stereo.config import (
    C01_DEFAULT,
    C01_OVEN_RANGE,
    SGBMConfig,
    StereoConfigError,
    check_search_range_covers,
    required_disparity_range,
)

pytestmark = pytest.mark.unit


def test_default_config_is_valid_and_has_stable_id():
    cfg = SGBMConfig()
    assert cfg.disparity_range == (0, 64)
    assert cfg.config_id == SGBMConfig().config_id
    # config_id 只由参与计算的参数决定，注释/标签不参与
    tagged = SGBMConfig(tags={"stage": "x"}, rationale="不同说明")
    assert tagged.config_id == cfg.config_id
    changed = SGBMConfig(num_disparities=32)
    assert changed.config_id != cfg.config_id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_disparities": 30},          # 非 16 倍数
        {"num_disparities": 0},           # 非正
        {"block_size": 4},                # 偶数
        {"block_size": 1},                # 过小
        {"min_disparity": -16},           # 负搜索下界
        {"uniqueness_ratio": 101},        # 越界
        {"p1": 800, "p2": 200},           # P2 < P1
        {"pre_filter_cap": 0},            # 越界
        {"mode": "NOPE"},                 # 未知模式
    ],
)
def test_illegal_config_rejected_locally(kwargs):
    """OpenCV 不会拒绝这些值，必须由本模块拒绝（实测 cv2 5.0.0 会照收）。"""
    with pytest.raises(StereoConfigError) as exc:
        SGBMConfig(**kwargs)
    assert exc.value.code == "E_INVALID_SEARCH_PARAMS"


def test_required_disparity_range_matches_geometry():
    req = required_disparity_range(fx_px=700.0, baseline_m=0.1, z_max_m=2.0, z_min_m=0.5)
    # d = fx*B/z → 0.5 m ⇒ 140 px；2.0 m ⇒ 35 px
    assert req["d_required_max_px"] == pytest.approx(140.0, rel=1e-9)
    assert req["d_required_min_px"] == pytest.approx(35.0, rel=1e-9)
    assert req["d_required_min_px"] < req["d_required_max_px"]


def test_search_range_coverage_reports_without_modifying():
    req = required_disparity_range(700.0, 0.1, 2.0, 0.5)
    assert req["d_required_max_px"] == pytest.approx(140.0)
    narrow = check_search_range_covers(C01_DEFAULT, 700.0, 0.1, 2.0, 0.5)
    assert narrow["covered"] is False
    assert narrow["covers_nearest"] is False   # 140 px 超出 0..63
    assert narrow["covers_farthest"] is True
    # 只报告，不修改配置
    assert C01_DEFAULT.disparity_range == (0, 64)

    wide = check_search_range_covers(C01_OVEN_RANGE, 700.0, 0.1, 2.0, 0.5)
    assert wide["covered"] is True


def test_geometry_helpers_validate_inputs():
    with pytest.raises(StereoConfigError):
        required_disparity_range(0.0, 0.1, 2.0, 0.5)
    with pytest.raises(StereoConfigError):
        required_disparity_range(700.0, 0.0, 2.0, 0.5)
    with pytest.raises(StereoConfigError):
        required_disparity_range(700.0, 0.1, 0.5, 2.0)


def test_oven_range_config_is_marked_unverified():
    assert C01_OVEN_RANGE.tags["verified"] is False
    assert C01_DEFAULT.tags["scope"] == "artificial_test_pattern"

"""人工构造图样自身的正确性测试。

如果图样和它的解析真值不自洽，后面所有"用已知视差核验单位"的结论都不可信。
这里用**独立重实现**的采样逻辑交叉检查渲染约定（y 方向不翻转、x 映射正确）。
"""

from __future__ import annotations

import numpy as np
import pytest

from stereo.testpatterns import (
    ARTIFICIAL_PROVENANCE,
    render_constant_shift_pair,
    render_slanted_plane_pair,
)

pytestmark = pytest.mark.unit


def test_slanted_plane_gt_follows_analytic_formula(pattern):
    h, w = pattern["left"].shape[:2]
    _, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    meta = pattern["meta"]
    expected = meta["d0_px"] + meta["slope_px_per_px"] * xx
    np.testing.assert_allclose(pattern["d_gt_px"], expected, rtol=1e-6)


def test_slanted_plane_right_image_matches_independent_resampling():
    p = render_slanted_plane_pair(height=64, width=96, d0=8.0, slope=0.1, seed=5)
    h, w = p["left"].shape[:2]
    left = p["left"].astype(np.float64)
    _, xx = np.mgrid[0:h, 0:w]
    x_l = (xx + 8.0) / (1.0 - 0.1)
    # 独立实现：逐行 np.interp（两端夹取），与图样内部的双线性采样应等价
    rebuilt = np.empty_like(left)
    cols = np.arange(w, dtype=np.float64)
    for row in range(h):
        rebuilt[row] = np.interp(np.clip(x_l[row], 0, w - 1), cols, left[row])
    assert np.max(np.abs(rebuilt - p["right"].astype(np.float64))) <= 1.0
    # 真值有效域与右图采样越界域一致
    assert np.array_equal(p["valid_gt"], x_l <= (w - 1))


def test_slanted_plane_meta_is_marked_artificial():
    p = render_slanted_plane_pair(height=32, width=48)
    assert p["meta"]["provenance"] == ARTIFICIAL_PROVENANCE
    assert p["meta"]["kind"] == "slanted_plane"
    assert "不能用于任何精度达标声明" in p["meta"]["note"]
    assert p["left"].dtype == np.uint8 and p["right"].dtype == np.uint8
    assert p["left"].shape == p["right"].shape == p["d_gt_px"].shape


def test_constant_shift_pair_is_shifted_not_reversed():
    p = render_constant_shift_pair(height=32, width=64, shift_px=12)
    # 左坐标 x 的内容出现在右图 x-12 处（即右图整体向右平移）
    assert np.array_equal(p["right"][:, 12:], p["left"][:, : 64 - 12])
    assert p["valid_gt"][0, 12] and not p["valid_gt"][0, 5]
    assert "病态" in p["meta"]["note"] or "非退化" in p["meta"]["note"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 0},
        {"width": -1},
        {"slope": 1.0},
        {"slope": -0.2},
    ],
)
def test_slanted_plane_rejects_bad_geometry(kwargs):
    with pytest.raises(ValueError):
        render_slanted_plane_pair(**kwargs)


def test_constant_shift_rejects_out_of_range_shift():
    with pytest.raises(ValueError):
        render_constant_shift_pair(height=8, width=8, shift_px=8)

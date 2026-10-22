"""SGBM 封装层回归测试。

两条实测规律必须被测试固定住，否则后续换 OpenCV 版本时会静默退化：

1. **构造入口**：本机 OpenCV 5.0.0 下 ``cv2.StereoSGBM(...)`` 类构造器一 ``compute``
   就段错误（EXIT=-11），``cv2.StereoSGBM_create(...)`` 正常 ⇒ 封装层必须优先用工厂函数。
2. **左边界代价**：左起 ``numDisparities`` 列恒为无效编码，与图像宽度、参数无关。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from stereo import config as config_mod
from stereo.config import C01_DEFAULT, SGBMConfig, check_search_range_covers
from stereo.sgbm import _matcher_factory, build_matcher, decode_disparity, invalid_raw_sentinel
from stereo.testpatterns import render_slanted_plane_pair

pytestmark = pytest.mark.unit


def test_matcher_factory_prefers_create_function():
    """回归：类构造器在本机 OpenCV 上段错误，必须走 *_create 工厂函数。"""
    assert cv2.__version__.split(".")[0] == "5", "此断言记录的是 OpenCV 5.x 的行为"
    assert hasattr(cv2, "StereoSGBM_create")
    assert _matcher_factory() is cv2.StereoSGBM_create


def test_build_matcher_returns_working_object():
    matcher = build_matcher(C01_DEFAULT)
    assert hasattr(matcher, "compute")


def test_invalid_sentinel_wraps_for_unsigned_dtypes():
    cfg = C01_DEFAULT
    assert invalid_raw_sentinel(cfg, 16.0, np.int16) == -16
    assert invalid_raw_sentinel(cfg, 16.0, np.uint16) == 65520  # numpy 2 不允许负值写入 uint
    cfg2 = SGBMConfig(name="min4", num_disparities=32, block_size=5, p1=200, p2=800, min_disparity=4)
    assert invalid_raw_sentinel(cfg2, 16.0, np.uint16) == 48


@pytest.mark.integration
@pytest.mark.parametrize("width", [320, 640])
@pytest.mark.parametrize("ndisp", [32, 64, 128])
def test_left_border_is_exactly_num_disparities_columns(width, ndisp):
    """左起 numDisparities 列恒无效（实测规律，跨宽度/参数成立）。"""
    cfg = SGBMConfig(
        name=f"ndisp{ndisp}", num_disparities=ndisp, block_size=5, p1=200, p2=800
    )
    pat = render_slanted_plane_pair(width=width)
    raw = build_matcher(cfg).compute(
        np.ascontiguousarray(pat["left"]), np.ascontiguousarray(pat["right"])
    )
    sentinel = invalid_raw_sentinel(cfg, 16.0, raw.dtype)
    assert bool((raw[:, :ndisp] == sentinel).all()), "左边界应全部为无效编码"
    _, valid, _ = decode_disparity(raw, cfg)
    assert not bool(valid[:, :ndisp].any()), "解码后左边界仍必须全部无效"
    assert config_mod.left_border_invalid_columns(cfg) == ndisp
    # 有效比例不能超过 1 - ndisp/width 这一几何上限
    assert valid.mean() <= 1.0 - ndisp / width + 1e-9


@pytest.mark.integration
@pytest.mark.parametrize("width", [320, 640])
@pytest.mark.parametrize("ndisp", [32, 64, 128, 256])
def test_valid_region_matches_coverage_prediction(width, ndisp):
    """``check_search_range_covers`` 的预测必须与真实有效区域一致。

    斜平面真值视差跨 ``[16, 16+0.1*(width-1)]``：搜索范围不够时，图右侧
    （大视差端）会整片失效；足够时，去掉左边界后应几乎全有效。
    这把"参数选择依据"和"实际输出"钉在同一个断言上。
    """
    cfg = SGBMConfig(
        name=f"ndisp{ndisp}", num_disparities=ndisp, block_size=5, p1=200, p2=800
    )
    pat = render_slanted_plane_pair(width=width)
    d_lo, d_hi = pat["meta"]["d_gt_range_px"]

    # 人工标定：把真值视差区间换算成深度区间，再用配置去问"覆盖没覆盖"
    fx, baseline, doff = 700.0, 0.1, 2.0
    z_max = fx * baseline / (d_lo - doff)   # 视差最小 => 最远
    z_min = fx * baseline / (d_hi - doff)   # 视差最大 => 最近
    verdict = check_search_range_covers(cfg, fx, baseline, z_max, z_min, doff)

    raw = build_matcher(cfg).compute(
        np.ascontiguousarray(pat["left"]), np.ascontiguousarray(pat["right"])
    )
    _, valid, _ = decode_disparity(raw, cfg)
    valid_beyond_border = valid[:, ndisp:].mean()

    if verdict["covered"]:
        assert valid_beyond_border > 0.99, (
            f"覆盖检查说够用，但左侧边界之外仅 {valid_beyond_border:.3f} 有效"
        )
    else:
        assert valid_beyond_border < 0.99, (
            f"覆盖检查说不够用，但左侧边界之外仍 {valid_beyond_border:.3f} 有效"
        )


def test_no_silent_fill_on_invalid_region():
    """无效区域必须是 NaN + False 掩膜，不得被补 0 或插值掩盖。"""
    cfg = C01_DEFAULT
    pat = render_slanted_plane_pair()
    raw = build_matcher(cfg).compute(
        np.ascontiguousarray(pat["left"]), np.ascontiguousarray(pat["right"])
    )
    d, valid, _ = decode_disparity(raw, cfg)
    assert np.isnan(d[~valid]).all()
    assert np.isfinite(d[valid]).all()

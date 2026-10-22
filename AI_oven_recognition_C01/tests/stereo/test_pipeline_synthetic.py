"""端到端链路测试（人工构造斜平面输入）。

注意：本文件的所有结论只针对**人工构造**输入，用于验证链路、单位与接口，
不构成对真实场景或任何精度目标的达标声明。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from stereo.config import C01_DEFAULT, C01_OVEN_RANGE
from stereo.pipeline import (
    PROVENANCE_ARTIFICIAL,
    PROVENANCE_B_PROVIDED,
    StereoInputError,
    estimate_depth,
)

pytestmark = pytest.mark.integration


def test_end_to_end_disparity_and_depth(pair_paths, artificial_calibration):
    left, right = pair_paths
    result = estimate_depth(
        left, right, config=C01_DEFAULT, calibration=artificial_calibration
    )
    assert result.mode == "disparity_and_depth"
    assert result.provenance == PROVENANCE_ARTIFICIAL
    assert result.is_real_scene_chain() is False  # 不得声称真实链路完成

    stats = result.stats()
    # 人工斜平面输入下的合理性下界（不是精度指标）
    assert stats["disparity_valid_ratio"] > 0.5
    assert 10.0 < stats["disparity_px_median"] < 60.0

    # 深度必须严格等于按同一标定对已输出视差的换算结果
    d = result.disparity_px
    z, valid = artificial_calibration.depth_from_disparity(d, result.disparity_valid_mask)
    assert np.array_equal(valid, result.depth_valid_mask)
    idx = result.depth_valid_mask
    np.testing.assert_allclose(result.depth_m[idx], z[idx], rtol=1e-6, atol=1e-6)


def test_depth_is_axial_z_not_ray_length(pair_paths, artificial_calibration):
    """轴向 Z 与射线距离不同：Z 必须满足 Z = fx*B/(d-doff) 的解析关系。"""
    left, right = pair_paths
    result = estimate_depth(left, right, config=C01_DEFAULT, calibration=artificial_calibration)
    idx = result.depth_valid_mask
    d = result.disparity_px[idx].astype(np.float64)
    expected = artificial_calibration.fx_px * artificial_calibration.baseline_m / (
        d - artificial_calibration.doff_px
    )
    ratio = result.depth_m[idx] / expected
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-6)


def test_determinism_same_input_same_config(pair_paths, artificial_calibration):
    left, right = pair_paths
    a = estimate_depth(left, right, config=C01_DEFAULT, calibration=artificial_calibration)
    b = estimate_depth(left, right, config=C01_DEFAULT, calibration=artificial_calibration)
    assert a.raw_disparity.dtype == b.raw_disparity.dtype
    assert np.array_equal(a.raw_disparity, b.raw_disparity)
    assert np.array_equal(a.disparity_valid_mask, b.disparity_valid_mask)
    assert np.array_equal(
        np.nan_to_num(a.disparity_px, nan=-1.0), np.nan_to_num(b.disparity_px, nan=-1.0)
    )
    assert np.array_equal(
        np.nan_to_num(a.depth_m, nan=-1.0), np.nan_to_num(b.depth_m, nan=-1.0)
    )
    # 计时只记录，不作为稳定性证明
    assert a.timings_ms["total"] > 0 and "sgbm_compute" in a.timings_ms


def test_without_calibration_returns_disparity_only(pair_paths):
    left, right = pair_paths
    result = estimate_depth(left, right, config=C01_DEFAULT, calibration=None)
    assert result.mode == "disparity_only"
    assert result.depth_m is None
    assert result.depth_valid_mask is None
    assert result.calibration is None
    assert any("disparity_only" in w for w in result.warnings)
    with pytest.raises(StereoInputError) as exc:
        result.visualize_depth()
    assert exc.value.code == "E_NO_DEPTH"


def test_search_range_check_reports_coverage(pair_paths, artificial_calibration):
    left, right = pair_paths
    covered = estimate_depth(
        left, right, config=C01_DEFAULT, calibration=artificial_calibration,
        depth_range_m=(1.5, 5.0),
    )
    assert covered.search_range_check is not None
    assert covered.search_range_check["covered"] is True
    assert covered.search_range_check["request"]["d_required_min_px"] == pytest.approx(16.0)

    narrow = estimate_depth(
        left, right, config=C01_DEFAULT, calibration=artificial_calibration,
        depth_range_m=(0.3, 2.0),
    )
    assert narrow.search_range_check["covered"] is False
    assert any("未完整覆盖" in w for w in narrow.warnings)
    # 只报告，配置不变
    assert narrow.config.disparity_range == C01_DEFAULT.disparity_range


def test_artificial_provenance_produces_explicit_warning(pair_paths, artificial_calibration):
    left, right = pair_paths
    result = estimate_depth(left, right, config=C01_DEFAULT, calibration=artificial_calibration)
    assert any("人工构造" in w and "不得" in w for w in result.warnings)
    meta = result.metadata()
    assert meta["real_scene_chain_complete"] is False
    assert "人工构造" in meta["claim_scope"]
    assert meta["stage"] == "C-01"
    assert meta["matcher"].startswith("cv2.StereoSGBM")


def test_b_provided_provenance_is_recorded(pair_paths, artificial_calibration):
    left, right = pair_paths
    result = estimate_depth(
        left, right, config=C01_DEFAULT, calibration=artificial_calibration,
        provenance=PROVENANCE_B_PROVIDED,
    )
    assert result.metadata()["real_scene_chain_complete"] is True
    # 不接受未知来源标记
    with pytest.raises(StereoInputError) as exc:
        estimate_depth(left, right, config=C01_DEFAULT, provenance="whatever")
    assert exc.value.code == "E_BAD_PROVENANCE"


def test_save_writes_numeric_outputs_and_masks(pair_paths, artificial_calibration, tmp_path):
    left, right = pair_paths
    result = estimate_depth(left, right, config=C01_DEFAULT, calibration=artificial_calibration)
    out = tmp_path / "run"
    files = result.save(out)

    for name in (
        "disparity_raw.npy",
        "disparity_px.npy",
        "disparity_valid_mask.npy",
        "disparity_valid_mask.png",
        "depth_m.npy",
        "depth_valid_mask.png",
        "run_metadata.json",
        "calibration.json",
        "config.json",
        "PROVENANCE.txt",
    ):
        assert name in files and (out / name).exists(), name

    d = np.load(out / "disparity_px.npy")
    mask = np.load(out / "disparity_valid_mask.npy").astype(bool)
    z = np.load(out / "depth_m.npy")
    # NaN 与掩膜严格一致：无效位置不能是 0，也不能被补全
    assert np.isnan(d[~mask]).all()
    assert np.isfinite(d[mask]).all()
    assert np.isnan(z[~result.depth_valid_mask]).all()
    assert np.isfinite(z[result.depth_valid_mask]).all()

    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["input_provenance"] == PROVENANCE_ARTIFICIAL
    assert meta["config"]["config_id"] == C01_DEFAULT.config_id
    assert "timings_ms" in meta and "environments" in meta
    assert meta["input"]["left"]["sha256"] and meta["input"]["right"]["sha256"]
    assert meta["decode"]["scale_verified_by_api_probe"] is True
    assert meta["disparity_units"]["verified_scale"] == 16.0


def test_disparity_only_save_has_no_depth_files(pair_paths, tmp_path):
    left, right = pair_paths
    result = estimate_depth(left, right, config=C01_DEFAULT, calibration=None)
    out = tmp_path / "run_disp"
    files = result.save(out)
    assert not (out / "depth_m.npy").exists()
    assert not (out / "depth_valid_mask.png").exists()
    calib_doc = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert calib_doc["status"] == "absent"


def test_oven_range_config_runs_on_same_input(pair_paths, artificial_calibration):
    """更大搜索范围也能跑通；仅说明配置可替换，不构成精度结论。

    注意：``numDisparities`` 是一笔固定的**左边界像素代价**（左起 ndisp 列恒无效），
    所以宽 320 的图上 ndisp=256 时有效比例上限只有 20%——这里按几何上限断言，
    而不是按"有效像素很多"断言。
    """
    left, right = pair_paths
    result = estimate_depth(left, right, config=C01_OVEN_RANGE, calibration=artificial_calibration)
    assert result.mode == "disparity_and_depth"
    stats = result.stats()
    assert stats["left_border_invalid_columns"] == C01_OVEN_RANGE.num_disparities
    assert stats["theoretical_max_valid_ratio"] == pytest.approx(
        1.0 - C01_OVEN_RANGE.num_disparities / 320
    )
    assert stats["disparity_valid_ratio"] <= stats["theoretical_max_valid_ratio"] + 1e-9
    # 去掉左边界后，中部区域的解码结果仍应与真值一致（配置换了不影响正确性）
    row = result.disparity_valid_mask[120, C01_OVEN_RANGE.num_disparities :]
    assert row.mean() > 0.9


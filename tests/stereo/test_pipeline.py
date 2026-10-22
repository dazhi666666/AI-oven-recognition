import json

import cv2
import numpy as np
import pytest

from oven_stereo.contracts import CalibrationError, RectifiedStereo
from oven_stereo.data.images import load_gray, load_gray_pair
from oven_stereo.stereo import estimate_depth
from oven_stereo.stereo.__main__ import main


@pytest.fixture
def calib(slanted_pair):
    h, w = slanted_pair[0].shape
    return RectifiedStereo("artificial", (w, h), f_px=700.0, baseline_m=0.1, doffs_px=-2.0)


def test_with_calibration_outputs_depth(slanted_pair, c01_config, calib):
    left, right, _, _ = slanted_pair
    result = estimate_depth(left, right, c01_config, calib, sample_id="s1")
    assert result.status == "ok"
    assert result.calibration_id == "artificial"
    v = result.depth_valid_mask
    assert v.any()
    assert np.isnan(result.depth_m[~v]).all()
    expected = 700.0 * 0.1 / (result.disparity_px[v].astype(np.float64) - 2.0)
    assert np.allclose(result.depth_m[v], expected, rtol=1e-6)


def test_without_calibration_is_disparity_only(slanted_pair, c01_config):
    left, right, _, _ = slanted_pair
    result = estimate_depth(left, right, c01_config)
    assert result.status == "disparity_only"
    assert result.depth_m is None and result.depth_valid_mask is None
    assert np.isnan(result.disparity_px[~result.disparity_valid_mask]).all()


def test_calibration_size_mismatch_rejected(slanted_pair, c01_config):
    left, right, _, _ = slanted_pair
    wrong = RectifiedStereo("wrong", (640, 480), f_px=700.0, baseline_m=0.1)
    with pytest.raises(CalibrationError):
        estimate_depth(left, right, c01_config, wrong)


def test_save_writes_arrays_and_summary(slanted_pair, c01_config, calib, tmp_path):
    left, right, _, _ = slanted_pair
    out = estimate_depth(left, right, c01_config, calib, sample_id="s1").save(tmp_path / "run")
    mask = np.load(out / "disparity_valid_mask.npy")
    assert mask.dtype == bool
    assert np.load(out / "depth_m.npy").dtype == np.float32
    summary = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert summary["sample_id"] == "s1"
    assert summary["config_id"] == c01_config.config_id
    assert 0 < summary["depth_valid_ratio"] <= 1


def test_command_line_end_to_end(slanted_pair, calib, c01_config_path, tmp_path):
    left, right, _, _ = slanted_pair
    cv2.imwrite(str(tmp_path / "L.png"), left)
    cv2.imwrite(str(tmp_path / "R.png"), right)
    (tmp_path / "calib.json").write_text(json.dumps(calib.to_dict()), encoding="utf-8")
    out = tmp_path / "out"
    code = main(
        ["--left", str(tmp_path / "L.png"), "--right", str(tmp_path / "R.png"),
         "--config", str(c01_config_path), "--calib", str(tmp_path / "calib.json"), "--out", str(out)]
    )
    assert code == 0
    for name in ("disparity_px.npy", "depth_m.npy", "result.json", "config.json", "calibration.json"):
        assert (out / name).exists(), name


def test_load_gray_checks(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gray(tmp_path / "missing.png")
    cv2.imwrite(str(tmp_path / "16bit.png"), np.zeros((10, 10), dtype=np.uint16))
    with pytest.raises(ValueError, match="8 位"):
        load_gray(tmp_path / "16bit.png")
    cv2.imwrite(str(tmp_path / "color.png"), np.zeros((10, 12, 3), dtype=np.uint8))
    assert load_gray(tmp_path / "color.png").shape == (10, 12)
    cv2.imwrite(str(tmp_path / "small.png"), np.zeros((10, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="尺寸"):
        load_gray_pair(tmp_path / "color.png", tmp_path / "small.png")

import json
import shutil

import cv2
import numpy as np
import pytest

from oven_stereo.data.calibration import read_middlebury_calibration
from oven_stereo.data.middlebury import load_middlebury_reference, load_middlebury_sample, main
from oven_stereo.data.pfm import read_disparity_pfm

from .conftest import write_pfm


@pytest.mark.parametrize("endian", ["<", ">"])
def test_pfm_endian_orientation_and_invalid(tmp_path, endian):
    path = tmp_path / "test.pfm"
    write_pfm(path, [[1, 2], [np.inf, 8]], endian)
    actual = read_disparity_pfm(path)
    np.testing.assert_array_equal(actual, [[1, 2], [np.inf, 8]])
    assert actual.dtype == np.float32


@pytest.mark.parametrize("payload", [
    b"PF\n1 1\n-1\n1234", b"Pf\n0 1\n-1\n", b"Pf\n1\n-1\n1234",
    b"Pf\n1 1\n0\n1234", b"Pf\n1 1\nnan\n1234", b"Pf\n1 1\ninf\n1234",
    b"Pf\n2 1\n-1\n1234", b"Pf\n1 1\n-1\n12345678",
])
def test_bad_pfm_rejected(tmp_path, payload):
    path = tmp_path / "bad.pfm"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        read_disparity_pfm(path)


@pytest.mark.parametrize("scale,endian", [("-0.003922", "<"), ("0.003922", ">")])
def test_middlebury_nonunit_scale_does_not_rescale_disparity(tmp_path, scale, endian):
    path = tmp_path / "disp.pfm"
    path.write_bytes(f"Pf\n1 1\n{scale}\n".encode() + np.array([50.0], dtype=endian + "f4").tobytes())
    assert read_disparity_pfm(path)[0, 0] == 50.0


def test_windows_chinese_image_path(tmp_path):
    from oven_stereo.data.images import load_gray

    path = tmp_path / "烤箱原图.png"
    expected = np.array([[10, 80], [100, 255]], dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", expected)
    assert ok
    path.write_bytes(encoded.tobytes())
    np.testing.assert_array_equal(load_gray(path), expected)


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_empty_or_corrupt_image_rejected(tmp_path, payload):
    from oven_stereo.data.images import load_gray

    path = tmp_path / "bad.png"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_gray(path)


def test_calibration_units_and_principal_point_sign(scene):
    calib = read_middlebury_calibration(scene / "inputs/calib.txt", "test")
    assert calib.stereo.baseline_m == 0.1
    assert calib.stereo.f_px == 100
    assert calib.stereo.doffs_px == 2
    assert calib.stereo.image_size == (3, 2)


@pytest.mark.parametrize("before,after", [
    ("baseline=100", "baseline=-1"), ("baseline=100", "baseline=nan"),
    ("doffs=2", "doffs=-2"), ("width=3", "width=0"), ("ndisp=16", "ndisp=0"),
    ("doffs=2\n", ""), ("cam0=[100 0 1; 0 100 0; 0 0 1]", "cam0=[100 0 1]"),
    ("cam1=[100 0 3; 0 100 0; 0 0 1]", "cam1=[100 0 3; 0 100 1; 0 0 1]"),
])
def test_bad_calibration_rejected(scene, before, after):
    path = scene / "inputs/calib.txt"
    path.write_text(path.read_text().replace(before, after))
    with pytest.raises(ValueError):
        read_middlebury_calibration(path, "test")


def test_reference_independent_hand_values(scene):
    sample = load_middlebury_sample(scene / "input.json", scene)
    ref = load_middlebury_reference(scene / "reference.json", sample, scene)
    # 手算：10/(0+2)=5，10/(2+2)=2.5，10/(8+2)=1；负分母无效。
    np.testing.assert_allclose(ref.depth_m, [[5, 2.5, np.nan], [1, np.nan, np.nan]], equal_nan=True)
    np.testing.assert_array_equal(ref.disparity_valid_mask, [[1, 1, 0], [1, 1, 0]])
    np.testing.assert_array_equal(ref.depth_valid_mask, [[1, 1, 0], [1, 0, 0]])
    assert np.isnan(ref.disparity_px[~ref.disparity_valid_mask]).all()
    assert ref.depth_m.dtype == np.float32
    assert ref.disparity_valid_mask.dtype == bool


def test_input_loads_after_all_answers_deleted(scene):
    shutil.rmtree(scene / "reference")
    (scene / "reference.json").unlink()
    sample = load_middlebury_sample(scene / "input.json", scene)
    assert sample.image_size_px == (3, 2)
    assert sample.calibration_metadata["cam0"][0][2] == 1


def test_missing_image_rejected(scene):
    (scene / "inputs/im1.png").unlink()
    with pytest.raises(FileNotFoundError):
        load_middlebury_sample(scene / "input.json", scene)


def test_image_size_mismatch_rejected(scene):
    cv2.imwrite(str(scene / "inputs/im1.png"), np.zeros((2, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="尺寸"):
        load_middlebury_sample(scene / "input.json", scene)


@pytest.mark.parametrize("key,value", [
    ("image_size_px", [4, 2]), ("rectified", False), ("calibration_id", ""),
    ("right_path", "inputs/im0.png"), ("source_version", "unknown"),
])
def test_wrong_input_metadata_rejected(scene, key, value):
    path = scene / "input.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_middlebury_sample(path, scene)


@pytest.mark.parametrize("key,value", [("sample_id", "wrong"), ("calibration_id", "wrong"), ("disparity_units", "mm")])
def test_reference_metadata_mismatch_rejected(scene, key, value):
    path = scene / "reference.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    sample = load_middlebury_sample(scene / "input.json", scene)
    with pytest.raises(ValueError):
        load_middlebury_reference(path, sample, scene)


def test_missing_reference_field_rejected(scene):
    path = scene / "reference.json"
    data = json.loads(path.read_text())
    del data["disparity_path"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="缺少字段"):
        load_middlebury_reference(path, load_middlebury_sample(scene / "input.json", scene), scene)


def test_reference_wrong_shape_rejected(scene):
    write_pfm(scene / "reference/disp0.pfm", [[1, 2]])
    sample = load_middlebury_sample(scene / "input.json", scene)
    with pytest.raises(ValueError, match="尺寸"):
        load_middlebury_reference(scene / "reference.json", sample, scene)


def test_prepare_cli_separates_inputs_from_answers(scene):
    out = scene / "prepared"
    assert main(["--manifest", str(scene / "input.json"), "--reference-manifest", str(scene / "reference.json"),
                 "--root", str(scene), "--out", str(out)]) == 0
    assert sorted(p.name for p in (out / "inputs").iterdir()) == ["calibration.json", "sample.json"]
    np.testing.assert_allclose(np.load(out / "reference/depth_m.npy")[0, :2], [5, 2.5])
    with pytest.raises(FileExistsError):
        main(["--manifest", str(scene / "input.json"), "--root", str(scene), "--out", str(out)])


def test_prepare_without_answers(scene):
    shutil.rmtree(scene / "reference")
    assert main(["--manifest", str(scene / "input.json"), "--root", str(scene), "--out", str(scene / "only_input")]) == 0
    assert not (scene / "only_input/reference").exists()

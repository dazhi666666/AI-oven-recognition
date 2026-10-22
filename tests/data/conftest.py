import hashlib
import json

import cv2
import numpy as np
import pytest


def write_pfm(path, values, endian="<"):
    values = np.asarray(values, dtype=np.float32)
    h, w = values.shape
    scale = "-1.0" if endian == "<" else "1.0"
    path.write_bytes(f"Pf\n{w} {h}\n{scale}\n".encode() + np.flipud(values).astype(endian + "f4").tobytes())


@pytest.fixture
def scene(tmp_path):
    inputs, reference = tmp_path / "inputs", tmp_path / "reference"
    inputs.mkdir()
    reference.mkdir()
    # 人工已校正相机：f=100 px、B=0.1 m、doffs=2 px。
    calib_text = (
        "cam0=[100 0 1; 0 100 0; 0 0 1]\n"
        "cam1=[100 0 3; 0 100 0; 0 0 1]\n"
        "doffs=2\nbaseline=100\nwidth=3\nheight=2\nndisp=16\n"
    )
    (inputs / "calib.txt").write_text(calib_text, encoding="utf-8")
    for name in ("im0.png", "im1.png"):
        cv2.imwrite(str(inputs / name), np.array([[11, 22, 33], [44, 55, 66]], dtype=np.uint8))
    write_pfm(reference / "disp0.pfm", [[0, 2, np.inf], [8, -3, np.nan]])

    def asset(path):
        data = path.read_bytes()
        return {"path": path.relative_to(tmp_path).as_posix(), "url": "https://example.invalid/" + path.name,
                "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    input_meta = {
        "schema_version": 1, "sample_id": "artificial_b01", "source": "Middlebury",
        "source_version": "2014/perfect", "domain": "public", "split": "development",
        "conditions": {}, "left_path": "inputs/im0.png", "right_path": "inputs/im1.png",
        "calibration_path": "inputs/calib.txt", "calibration_id": "artificial_calib",
        "image_size_px": [3, 2], "rectified": True, "metric_depth_available": True,
        "assets": [asset(inputs / name) for name in ("im0.png", "im1.png", "calib.txt")],
    }
    reference_meta = {
        "schema_version": 1, "sample_id": "artificial_b01", "calibration_id": "artificial_calib",
        "disparity_path": "reference/disp0.pfm", "disparity_units": "px", "depth_units": "m",
        "source_url": "artificial://test-only", "assets": [asset(reference / "disp0.pfm")],
    }
    for name, meta in (("input.json", input_meta), ("reference.json", reference_meta)):
        (tmp_path / name).write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path

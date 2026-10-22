import numpy as np
import pytest

from oven_stereo.contracts import CalibrationError, RectifiedStereo
from oven_stereo.stereo import disparity_to_depth


def _calib(**overrides) -> RectifiedStereo:
    params = dict(
        calibration_id="test", image_size=(4, 1), f_px=1000.0, baseline_m=0.2, doffs_px=10.0
    )
    params.update(overrides)
    return RectifiedStereo(**params)


def test_depth_hand_calc_with_doffs():
    # Z = 1000 * 0.2 / (d + 10)：d=90 → 2.0 m；d=190 → 1.0 m；d=-10 → 分母为 0，无效
    d = np.array([[90.0, 190.0, -10.0, np.nan]], dtype=np.float32)
    valid = np.array([[True, True, True, False]])
    depth, depth_valid = disparity_to_depth(d, valid, _calib())
    assert depth_valid.tolist() == [[True, True, False, False]]
    assert depth[0, :2].tolist() == pytest.approx([2.0, 1.0])
    assert np.isnan(depth[0, 2:]).all()


def test_invalid_disparity_stays_invalid():
    d = np.array([[90.0, 90.0, 90.0, 90.0]], dtype=np.float32)
    valid = np.array([[True, False, True, False]])
    _, depth_valid = disparity_to_depth(d, valid, _calib())
    assert depth_valid.tolist() == valid.tolist()


def test_from_projection_matrices():
    # cx_left = 160，cx_right = 162，P2[0,3] = -f * B = -70
    P1 = [[700, 0, 160, 0], [0, 700, 120, 0], [0, 0, 1, 0]]
    P2 = [[700, 0, 162, -70], [0, 700, 120, 0], [0, 0, 1, 0]]
    calib = RectifiedStereo.from_projection_matrices(P1, P2, "p", (320, 240))
    assert calib.f_px == 700
    assert calib.baseline_m == pytest.approx(0.1)
    assert calib.doffs_px == pytest.approx(2.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rectified": False},
        {"f_px": 0.0},
        {"baseline_m": -0.1},
        {"doffs_px": float("nan")},
        {"calibration_id": ""},
    ],
)
def test_invalid_calibration_rejected(overrides):
    with pytest.raises(CalibrationError):
        _calib(**overrides).validate()


def test_calibration_size_must_match_image():
    with pytest.raises(CalibrationError, match="分辨率"):
        _calib().validate(image_size=(640, 480))


def test_calibration_json_round_trip(tmp_path):
    calib = _calib()
    path = tmp_path / "calib.json"
    import json

    path.write_text(json.dumps(calib.to_dict()), encoding="utf-8")
    assert RectifiedStereo.from_json(path) == calib

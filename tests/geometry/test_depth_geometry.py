import json

import numpy as np
import pytest

from oven_stereo.contracts import CalibrationError, PinholeIntrinsics
from oven_stereo.geometry import (
    backproject_depth,
    measure_depth_geometry,
)


def _intrinsics(shape=(6, 8), fx=100.0, fy=80.0):
    height, width = shape
    return PinholeIntrinsics(
        calibration_id="artificial",
        image_size=(width, height),
        fx_px=fx,
        fy_px=fy,
        cx_px=(width - 1) / 2,
        cy_px=(height - 1) / 2,
    )


def test_backproject_depth_hand_values_and_invalid_point():
    intrinsics = PinholeIntrinsics("hand", (2, 1), 2.0, 4.0, 0.0, 0.0)
    depth = np.array([[2.0, 4.0]])
    points, valid = backproject_depth(depth, np.array([[True, False]]), intrinsics)
    np.testing.assert_allclose(points[0, 0], [0.0, 0.0, 2.0])
    assert np.isnan(points[0, 1]).all()
    np.testing.assert_array_equal(valid, [[True, False]])


def test_intrinsics_from_matrix_and_resolution_must_match():
    intrinsics = PinholeIntrinsics.from_matrix(
        [[100.0, 0.0, 1.5], [0.0, 80.0, 1.0], [0.0, 0.0, 1.0]],
        calibration_id="known_camera",
        image_size=(4, 3),
    )
    assert intrinsics.to_dict() == {
        "calibration_id": "known_camera",
        "image_size": [4, 3],
        "fx_px": 100.0,
        "fy_px": 80.0,
        "cx_px": 1.5,
        "cy_px": 1.0,
    }
    with pytest.raises(CalibrationError, match="分辨率"):
        backproject_depth(
            np.ones((2, 4)), np.ones((2, 4), dtype=bool), intrinsics,
        )


def test_front_parallel_plane_height_area_and_volume():
    shape = (6, 8)
    intrinsics = _intrinsics(shape)
    target = np.zeros(shape, dtype=bool)
    target[2:4, 3:5] = True
    tray = ~target
    depth = np.full(shape, 1.0)
    depth[target] = 0.8

    result = measure_depth_geometry(
        depth,
        np.ones(shape, dtype=bool),
        target,
        tray,
        intrinsics,
        sample_id="fronto_parallel_block",
        config_id="a03_test_v1",
        plane_max_rms_m=1e-12,
    )

    # z=1 m 的正视托盘上，每个像素格面积 = (1/fx)×(1/fy) m²。
    expected_cell_area = 1.0 / (100.0 * 80.0)
    expected_volume_ml = 0.2 * expected_cell_area * 4 * 1_000_000.0
    assert result.status == "ok"
    assert result.thickness_mm == pytest.approx(200.0, abs=1e-9)
    assert result.volume_ml == pytest.approx(expected_volume_ml, abs=1e-9)
    np.testing.assert_allclose(result.height_m[target], 0.2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.cell_area_m2, expected_cell_area, atol=1e-15, rtol=0)
    assert result.tray_plane is not None
    np.testing.assert_allclose(result.tray_plane.normal_xyz, [0, 0, -1], atol=1e-12)
    assert result.tray_plane.offset_m == pytest.approx(1.0, abs=1e-12)


def test_tilted_tray_uses_plane_normal_not_z_subtraction(tmp_path):
    shape = (7, 9)
    intrinsics = _intrinsics(shape, fx=90.0, fy=100.0)
    normal = np.array([0.15, -0.08, -1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    offset = 1.1
    height_m = 0.04
    target = np.zeros(shape, dtype=bool)
    target[2:5, 3:6] = True
    tray = ~target

    rows, cols = np.indices(shape, dtype=np.float64)
    rays = np.stack(
        ((cols - intrinsics.cx_px) / intrinsics.fx_px,
         (rows - intrinsics.cy_px) / intrinsics.fy_px,
         np.ones(shape)),
        axis=2,
    )
    denominator = rays @ normal
    tray_depth = -offset / denominator
    surface_depth = (height_m - offset) / denominator
    depth = np.where(target, surface_depth, tray_depth)

    result = measure_depth_geometry(
        depth,
        np.ones(shape, dtype=bool),
        target,
        tray,
        intrinsics,
        sample_id="tilted_tray",
        config_id="a03_test_v1",
        plane_max_rms_m=1e-12,
    )
    assert result.status == "ok"
    assert result.thickness_mm == pytest.approx(40.0, abs=1e-9)
    np.testing.assert_allclose(result.height_m[target], height_m, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.tray_plane.normal_xyz, normal, atol=1e-12, rtol=0)
    assert result.tray_plane.offset_m == pytest.approx(offset, abs=1e-12)
    assert result.tray_plane.fit_rms_m < 1e-12

    result.save(tmp_path)
    summary = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert summary["method"] == "depth_plane_height_integral"
    assert summary["tray_plane"]["normal_orientation"] == "toward_camera_and_food_side"
    assert summary["tray_plane"]["points_used"] == int(np.count_nonzero(tray))


def test_missing_target_depth_rejects_measurement():
    shape = (5, 7)
    intrinsics = _intrinsics(shape)
    target = np.zeros(shape, dtype=bool)
    target[2, 3] = True
    tray = ~target
    depth = np.full(shape, 1.0)
    valid = np.ones(shape, dtype=bool)
    valid[target] = False
    result = measure_depth_geometry(
        depth, valid, target, tray, intrinsics,
        sample_id="missing", config_id="a03_test_v1",
    )
    assert result.status == "rejected"
    assert result.coverage == 0.0
    assert result.thickness_mm is None and result.volume_ml is None


def test_too_few_or_collinear_tray_points_are_rejected():
    shape = (4, 5)
    intrinsics = _intrinsics(shape)
    depth = np.ones(shape)
    target = np.zeros(shape, dtype=bool)
    tray = np.zeros(shape, dtype=bool)
    tray[0, :2] = True
    with pytest.raises(ValueError, match="少于 3"):
        measure_depth_geometry(
            depth, np.ones(shape, dtype=bool), target, tray, intrinsics,
            sample_id="few", config_id="a03_test_v1",
        )

    tray[0, 2] = True
    with pytest.raises(ValueError, match="共线"):
        measure_depth_geometry(
            depth, np.ones(shape, dtype=bool), target, tray, intrinsics,
            sample_id="line", config_id="a03_test_v1",
        )


def test_noisy_tray_over_rms_limit_is_rejected():
    shape = (5, 7)
    intrinsics = _intrinsics(shape)
    depth = np.ones(shape)
    depth[0, 0] = 1.05
    with pytest.raises(ValueError, match="超过阈值"):
        measure_depth_geometry(
            depth,
            np.ones(shape, dtype=bool),
            np.zeros(shape, dtype=bool),
            np.ones(shape, dtype=bool),
            intrinsics,
            sample_id="noisy_tray",
            config_id="a03_test_v1",
            plane_max_rms_m=1e-4,
        )


def test_masks_must_match_and_not_overlap():
    shape = (3, 4)
    intrinsics = _intrinsics(shape)
    depth = np.ones(shape)
    mask = np.ones(shape, dtype=bool)
    with pytest.raises(ValueError, match="不能重叠"):
        measure_depth_geometry(
            depth, mask, mask, mask, intrinsics,
            sample_id="overlap", config_id="a03_test_v1",
        )
    with pytest.raises(ValueError, match="target_mask"):
        measure_depth_geometry(
            depth, mask, np.ones((2, 4), dtype=bool), np.zeros(shape, dtype=bool), intrinsics,
            sample_id="shape", config_id="a03_test_v1",
        )

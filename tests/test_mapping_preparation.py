import copy

import numpy as np
import pytest

from jev_mujoco.perception_mapping import filter_mapping_views


@pytest.mark.parametrize("reject", [False, True])
def test_mapping_preparation_preserves_every_valid_source_and_preview_alignment(monkeypatch, reject):
    from jev_mujoco import pointcloud_filter

    depth = np.array([[1., 2., np.nan], [3., -1., 4.]], dtype=np.float32)
    valid = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    transform = np.eye(4)
    transform[:3, 3] = [1., 2., 3.]
    views = {camera: {"depth_left_m": depth.copy(), "valid_left_mask": valid.copy(),
        "left_intrinsics": np.eye(3), "T_base_left_camera": transform.copy(), "stereo_sequence": 2}
        for camera in ("1", "2", "3")}
    original = copy.deepcopy(views)
    decoded = {"views": views, "camera_capture_metadata": {"2": {"letterbox_offset_rc": [1, 2]}}}
    expected_points = np.tile([[1., 2., 4.], [1., 5., 6.], [9., 6., 7.]], (3, 1))
    expected_pixels = np.tile([[0, 0], [1, 0], [1, 2]], (3, 1))
    expected_cameras = [camera for camera in ("1", "2", "3") for _ in range(3)]
    actual_filter = pointcloud_filter.filter_observed_points

    def filter_points(points, sources, *, settings):
        np.testing.assert_array_equal(points, expected_points)
        np.testing.assert_array_equal(sources["pixels_rc"], expected_pixels)
        assert list(sources["camera_ids"]) == expected_cameras
        result = actual_filter(points, sources, settings={})
        if reject:
            result["rejected_mask"][[1, 5, 6]] = True
            result["counts"] = {"input": 9, "kept": 6, "rejected": 3}
        return result

    monkeypatch.setattr(pointcloud_filter, "filter_observed_points", filter_points)
    result = filter_mapping_views(decoded, {})
    preview = result["preview"]
    assert preview["points_world_m"] == expected_points.tolist()
    assert preview["original_point_indices"] == list(range(9))
    assert preview["camera_ids"] == expected_cameras
    expected_pixels[3:6] -= [1, 2]
    assert preview["pixels_rc"] == expected_pixels.tolist()
    for camera, row, column in (("1", 1, 0), ("2", 1, 2), ("3", 0, 0)):
        if reject:
            original[camera]["depth_left_m"][row, column] = 0
            original[camera]["valid_left_mask"][row, column] = 0
    for camera in views:
        np.testing.assert_array_equal(views[camera]["depth_left_m"], original[camera]["depth_left_m"])
        np.testing.assert_array_equal(views[camera]["valid_left_mask"], original[camera]["valid_left_mask"])
    assert set(result["timings_ms"]) == {
        "depth_projection_sources_wall", "optional_filter_wall", "raw_pointcloud_preview_wall",
        "rejected_depth_apply_wall", "filter_total_wall"}
    assert all(value >= 0 for value in result["timings_ms"].values())


def test_mapping_preparation_preserves_empty_views():
    view = {"depth_left_m": np.zeros((2, 3)), "valid_left_mask": np.ones((2, 3)),
        "left_intrinsics": np.eye(3), "T_base_left_camera": np.eye(4), "stereo_sequence": 2}
    result = filter_mapping_views({"views": {"3": view}}, {})
    assert result["counts"] == {"input": 0, "kept": 0, "rejected": 0}
    assert result["preview"]["pixels_rc"] == []
    assert result["preview"]["cameras"]["3"]["valid_depth_count"] == 0

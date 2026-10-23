from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_semantic_depth import CameraObservation, prepared

from luna_mujoco.perception_mapping import CaptureSegmenter


def context(*, sequence=8, valid=None, source="selected", surface=None):
    depth = np.ones((2, 4), dtype=np.float32)
    matrix = np.eye(4)
    matrix[:3, 3] = [1, 2, 3]
    return {"confirmed_instance_id": "selected", "observation_set_id": "obs", "selection_generation": 2,
        "selected_surface_world_m": [[2, 2, 4]] if surface is None else surface,
        "views": {"3": {"camera_id": "3", "stereo_sequence": sequence,
            "depth_left_m": depth, "valid_left_mask": np.ones_like(depth) if valid is None else valid,
            "left_intrinsics": np.eye(3), "T_base_left_camera": matrix,
            "rgb_intrinsics": np.diag([2., 1., 1.]),
            "left_to_rgb": np.array([[1, 0, 0, .5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            "instance_masks": {source: np.array([[0, 0, 0, 1], [0, 0, 0, 0]])}}}}


def capture():
    value = prepared(np.zeros((2, 4)))
    from tianji_curobo.robot_segmenter import PreparedFfsDepth
    value = PreparedFfsDepth(**vars(value))
    return CaptureSegmenter(None), value


def test_restore_current_registered_supported_source_synchronizes_all_fields():
    adapter, value = capture()
    adapter.begin_capture(context())
    result = adapter.restore_selected_returns(value, camera_observation_type=CameraObservation)
    expected = np.zeros((2, 4)); expected[0, 1] = 1
    assert np.array_equal(result.filtered_depth_m.numpy()[0], expected)
    assert np.array_equal(result.camera_observation.depth_image.numpy()[0], expected)
    assert result.base_pointcloud_m.numpy()[0, 0, 1].tolist() == [2, 2, 4]
    assert adapter.source_restoration["3"]["restored_point_count"] == 1
    assert np.count_nonzero(value.filtered_depth_m.numpy()) == 0


@pytest.mark.parametrize("kind", ["sequence", "foreign", "support", "rejected"])
def test_source_restoration_rejects_stale_foreign_unsupported_or_rejected(kind):
    adapter, value = capture()
    values = context(sequence=9 if kind == "sequence" else 8,
                     source="foreign" if kind == "foreign" else "selected",
                     surface=[[200, 2, 4]] if kind == "support" else None)
    if kind == "rejected":
        values["views"]["3"]["valid_left_mask"][0, 1] = 0
        values["views"]["3"]["depth_left_m"][0, 1] = 0
    adapter.begin_capture(values)
    result = adapter.restore_selected_returns(value, camera_observation_type=CameraObservation)
    assert np.count_nonzero(result.filtered_depth_m.numpy()) == 0


def test_capture_replacement_cannot_reuse_old_selected_support():
    adapter, value = capture()
    adapter.begin_capture(context())
    adapter.restore_selected_returns(value, camera_observation_type=CameraObservation)
    adapter.begin_capture(context(source="foreign"))
    assert not adapter.source_restoration
    result = adapter.restore_selected_returns(replace(value, stereo_sequence=8), camera_observation_type=CameraObservation)
    assert np.count_nonzero(result.filtered_depth_m.numpy()) == 0


@pytest.mark.parametrize("field", ["intrinsics_tensor", "pose_matrix_tensor"])
def test_same_sequence_wrong_calibration_cannot_restore_source(field):
    from test_semantic_depth import Tensor

    adapter, value = capture()
    values = getattr(value, field).numpy().copy()
    values.flat[0] += .1
    value = replace(value, **{field: Tensor(values)})
    adapter.begin_capture(context())
    result = adapter.restore_selected_returns(value, camera_observation_type=CameraObservation)
    assert np.count_nonzero(result.filtered_depth_m.numpy()) == 0
    assert not adapter.source_restoration


def test_public_raw_robot_mask_receives_unchanged_valid_depth_and_fresh_calibration(monkeypatch):
    import sys

    from test_semantic_depth import Tensor

    from luna_mujoco.raw_robot_depth import RawRobotDepthBoundary

    class RawTensor(Tensor):
        def unsqueeze(self, axis):
            return RawTensor(np.expand_dims(self.values, axis))

    calls = []
    captures = []

    class RobotSegmenter:
        def __init__(self, *args, **kwargs):
            calls.append(("create", kwargs))

        def update_camera_projection(self, observation):
            calls.append(("calibration", observation.intrinsics.numpy().copy()))

        def get_robot_mask(self, observation, state):
            calls.append(("raw", observation.depth_image.numpy().copy()))
            captures.append((observation.pose.copy(), state.numpy().copy()))
            free = observation.depth_image.numpy().copy()
            free[0, 0, 0] = 0
            return SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1)), RawTensor(free)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    monkeypatch.setitem(sys.modules, "curobo.perception", SimpleNamespace(RobotSegmenter=RobotSegmenter))
    monkeypatch.setitem(sys.modules, "curobo.types", SimpleNamespace(CameraObservation=CameraObservation,
        Pose=SimpleNamespace(from_matrix=lambda matrix: matrix.numpy()),
        JointState=SimpleNamespace(from_position=lambda position, **kwargs: position)))
    boundary = RawRobotDepthBoundary.__new__(RawRobotDepthBoundary)
    boundary.device_cfg = SimpleNamespace(to_device=RawTensor)
    boundary.kinematics = object()
    boundary.robot_segmenters = {}
    boundary.gripper_feedback = {"A": .55, "B": .2}
    raw = np.array([[.001, 4., np.nan, 1.123456], [2., -1., 0., 3.14]], dtype=np.float32)
    request = SimpleNamespace(camera_id="1", stereo_sequence=8, depth_left_m=raw,
        valid_left_mask=np.array([[1, 1, 1, 1], [1, 1, 1, 0]]),
        left_intrinsics=np.eye(3), T_base_left_camera=np.eye(4),
        joint_positions_deg={"A": [0]*7, "B": [0]*7})
    result = boundary.prepare_ffs_observation(request)
    expected = np.array([[.001, 4., 0., 1.123456], [2., 0., 0., 0.]], dtype=np.float32)
    assert np.array_equal(calls[2][1][0], expected)
    assert np.array_equal(result.filtered_depth_m.numpy()[0, 0, 1:], expected[0, 1:])
    request.camera_id = "2"
    request.left_intrinsics[0, 0] = 2
    boundary.prepare_ffs_observation(request)
    assert len(boundary.robot_segmenters) == 2
    assert calls[4][0] == "calibration" and calls[4][1][0, 0] == 2
    request.T_base_left_camera[0, 3] = .25
    request.joint_positions_deg["A"][0] = 10
    request.depth_left_m[0, 1] = 3.5
    boundary.prepare_ffs_observation(request)
    assert [kind for kind, _ in calls].count("calibration") == 2
    assert calls[-1][0] == "raw" and calls[-1][1][0, 0, 1] == 3.5
    assert captures[-1][0][0, 3] == .25
    assert captures[-1][1][0] == pytest.approx(np.deg2rad(10))
    assert captures[-1][1][-2:].tolist() == pytest.approx([.55, .2])
    request.left_intrinsics[0, 0] = 3
    boundary.prepare_ffs_observation(request)
    assert calls[-2][0] == "calibration" and calls[-2][1][0, 0] == 3
    request.depth_left_m = np.ones((3, 4), dtype=np.float32)
    request.valid_left_mask = np.ones((3, 4), dtype=np.uint8)
    boundary.prepare_ffs_observation(request)
    assert [kind for kind, _ in calls].count("create") == 3
    assert calls[-2][0] == "calibration"


def test_capture_stage_timings_reset_with_the_capture(monkeypatch):
    from test_semantic_depth import projection

    from luna_mujoco import perception_mapping, semantic_depth

    value = prepared(np.ones((2, 4)))
    adapter = CaptureSegmenter(SimpleNamespace(prepare_ffs_observation=lambda request: value))
    monkeypatch.setattr(semantic_depth, "partition_semantic_depth", lambda *args, **kwargs: "partitioned")
    ticks = iter(index * .001 for index in range(20))
    monkeypatch.setattr(perception_mapping.time, "perf_counter", lambda: next(ticks))
    request = SimpleNamespace(camera_id="3")
    for _ in range(2):
        adapter.prepare_ffs_observation(request)
        assert adapter.remove_semantic_instances(value, projection({})) == "partitioned"
    assert adapter.timings_ms == pytest.approx({
        "robot_depth_prepare_wall": 2., "selected_return_restore_wall": 2.,
        "filtered_depth_download_wall": 2., "semantic_partition_wall": 2.})
    adapter.begin_capture(None)
    assert adapter.timings_ms == {}

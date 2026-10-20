import numpy as np
import pytest


def wire():
    from tianji_curobo.semantic_esdf_transport import encode_semantic_esdf_request

    eye = np.eye(4).tolist()
    views = {camera: {"camera_id": camera, "stereo_sequence": 4,
             "depth_left_m": np.ones((2, 3), dtype=np.float32),
             "valid_left_mask": np.ones((2, 3), dtype=np.uint8),
             "left_intrinsics": [[4, 0, 1], [0, 4, .5], [0, 0, 1]],
             "T_base_left_camera": eye, "robot_joints_at_capture": {"A": [0.] * 7, "B": [0.] * 7},
             "rgb_size": [3, 2], "rgb_intrinsics": [[4, 0, 1], [0, 4, .5], [0, 0, 1]],
             "left_to_rgb": eye, "instance_masks": {"source": np.ones((2, 3), dtype=np.uint8)}}
             for camera in ("1", "2", "3")}
    return encode_semantic_esdf_request({"observation_set_id": "rgbd-4", "sam3_generation": 4,
                                       "confirmed_instance_id": "source", "selection_generation": 4, "views": views}) | {
                                           "gripper_joint_positions_rad": {"A": 0., "B": 0.}}


def test_registered_transport_preserves_pixels_and_release_enables_all_layers():
    from jev_mujoco.perception_mapping import decode_mapping_payload

    decoded, identity = decode_mapping_payload(wire() | {"planning_settings": {}, "exclusion_state": "released"})
    assert decoded["views"]["3"]["depth_left_m"].shape == (2, 3)
    assert decoded["views"]["3"]["depth_left_m"].tolist() == [[1., 1., 1.], [1., 1., 1.]]
    assert identity["observation_id"] == "rgbd-4" and identity["generation"] == 4
    assert identity["disabled_obstacle_names"] == ()


def test_empty_head_sidecar_decodes_original_capture_and_exact_filter_identity():
    from jev_mujoco.perception_mapping import decode_mapping_payload
    from jev_mujoco.semantic_transport import decode_project_semantic_request, encode_project_semantic_request

    original = decode_project_semantic_request({key: value for key, value in wire().items()
                                               if key != "gripper_joint_positions_rad"})
    original["views"]["3"]["instance_masks"] = {}
    payload = encode_project_semantic_request(original, exclusion_state="held") | {
        "gripper_joint_positions_rad": {"A": .55, "B": 0.}, "exclusion_state": "held"}
    decoded, identity = decode_mapping_payload(payload)
    assert decoded["views"]["3"]["instance_masks"] == {}
    assert decoded["views"]["1"]["instance_masks"]["source"].all()
    assert decoded["exclusion_state"] == "held"
    assert identity["target_id"] == "source" and identity["generation"] == 4
    _, changed = decode_mapping_payload(payload | {"pointcloud_filter": {"enabled": True}})
    assert changed["scene_token"] != identity["scene_token"]
    with pytest.raises(ValueError, match="exclusion"):
        decode_mapping_payload(payload | {"exclusion_state": "invalid"})


@pytest.mark.parametrize("state,empty", [("held", True), ("released", True),
    ("held", False), ("selected", True), ("observation", True)])
def test_project_builder_receives_every_capture_with_original_head_semantics(state, empty):
    from types import SimpleNamespace

    from jev_mujoco.perception_mapping import CaptureSegmenter, PerceptionMapper, decode_mapping_payload

    decoded, _ = decode_mapping_payload(wire())
    decoded["exclusion_state"] = state
    decoded["views"]["3"]["instance_masks"] = {} if empty else {"source": np.ones((2, 3))}
    candidate = SimpleNamespace(scene_model={"voxel": {}})
    calls = []
    class ProjectBuilder:
        def __init__(self):
            self.diagnostics = {"semantic_only_active": False}
            self.scene_voxel_grids = []
        def build(self, payload):
            calls.append("project")
            assert payload["confirmed_instance_id"] == "source"
            assert bool(payload["views"]["3"]["instance_masks"]) == (not empty)
            return candidate
    mapper = PerceptionMapper.__new__(PerceptionMapper)
    mapper.capture_segmenter = CaptureSegmenter(None)
    mapper.segmenter_boundary = SimpleNamespace(robot_spheres=lambda joints: [])
    resets = []
    mapper.mapper = SimpleNamespace(reset_capture=lambda: resets.append(True), scene_voxel_grids=[])
    mapper.builder = ProjectBuilder()
    mapper.build(decoded)
    assert calls == ["project"]
    assert resets == []  # Builder resets raw maps itself; known-empty grids must survive.


def test_mapping_policy_is_validated_and_bound_to_scene_identity():
    from jev_mujoco.perception_mapping import decode_mapping_payload
    payload = wire() | {"full_observation_complete": True, "mapping_max_sphere_radius_m": .2,
        "planning_settings": {"experimental_semantic_only_mapping": True,
                              "optimizer_collision_activation_distance_m": .06}}
    decoded, identity = decode_mapping_payload(payload)
    assert decoded["mapping_collision_policy"]["optimizer_collision_activation_distance_m"] == .06
    assert decoded["mapping_max_sphere_radius_m"] == .2
    assert decoded["full_observation_complete"] is True
    assert decoded["experimental_semantic_only_mapping"] is True
    _, changed = decode_mapping_payload(payload | {"full_observation_complete": False})
    assert changed["scene_token"] != identity["scene_token"]
    with pytest.raises(ValueError, match="sphere radius"):
        decode_mapping_payload(payload | {"mapping_max_sphere_radius_m": -1})


@pytest.mark.parametrize("state", ["held", "released"])
def test_empty_head_semantic_builder_preserves_genuine_zero_valid_head_view(state):
    from jev_mujoco.perception_mapping import decode_mapping_payload, mapping_builder_payload

    decoded, _ = decode_mapping_payload(wire())
    decoded["exclusion_state"] = state
    head = decoded["views"]["3"]
    head["instance_masks"] = {}
    head["depth_left_m"][:] = 0
    head["valid_left_mask"][:] = 0
    result = mapping_builder_payload(decoded)
    assert result is decoded
    assert tuple(result["views"]) == ("1", "2", "3")
    assert result["confirmed_instance_id"] == "source"
    assert not np.any(result["views"]["3"]["depth_left_m"])


def test_mapping_rejects_missing_selected_instance_and_mismatched_capture():
    from jev_mujoco.perception_mapping import decode_mapping_payload

    payload = wire()
    payload["confirmed_instance_id"] = "hidden-truth-id"
    with pytest.raises(ValueError, match="selected"):
        decode_mapping_payload(payload)


def test_targetless_background_map_is_allowed_only_for_observation_motion():
    from jev_mujoco.perception_mapping import decode_mapping_payload
    from jev_mujoco.semantic_transport import decode_project_semantic_request, encode_project_semantic_request

    original = decode_project_semantic_request({k: v for k, v in wire().items() if k != 'gripper_joint_positions_rad'})
    original.update(mapping_mode='background', confirmed_instance_id='', sam3_generation=0)
    for view in original['views'].values():
        view['instance_masks'] = {}
    request = encode_project_semantic_request(original, exclusion_state='observation') | {
        'exclusion_state': 'observation', 'gripper_joint_positions_rad': {'A': 0., 'B': 0.}}
    _, identity = decode_mapping_payload(request)
    assert not identity['disabled_obstacle_names'] and identity['target_id'] == ''
    with pytest.raises(ValueError, match='selected instance'):
        decode_mapping_payload(request | {'exclusion_state': 'selected'})
    payload = wire()
    payload["views"]["1"]["stereo_sequence"] = 5
    with pytest.raises(ValueError, match="capture"):
        decode_mapping_payload(payload)


def test_mapping_binds_token_to_finite_actual_gripper_feedback():
    from jev_mujoco.perception_mapping import decode_mapping_payload

    payload = wire()
    _, first = decode_mapping_payload(payload)
    _, held = decode_mapping_payload(payload | {"gripper_joint_positions_rad": {"A": .55, "B": 0.}})
    assert first["scene_token"] != held["scene_token"]
    assert held["gripper_joint_positions_rad"] == {"A": .55, "B": 0.}
    for feedback in (None, {"A": .55}, {"A": float("nan"), "B": 0.}):
        with pytest.raises(ValueError, match="gripper"):
            decode_mapping_payload(payload | {"gripper_joint_positions_rad": feedback})


def test_segmenter_factory_constructs_one_articulated_boundary(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from jev_mujoco import raw_robot_depth
    from jev_mujoco.perception_mapping import create_feedback_segmenter

    root = Path(__file__).resolve().parents[1]
    from tianji_curobo import robot_segmenter

    original = (root / "resources/curobo/marvin_gui_dual.yml").read_bytes()
    observed = []
    def factory(configured, **kwargs):
        observed.append((configured, kwargs["feedback"]))
        return SimpleNamespace(sdk_boundary=object())
    def duplicate_factory(**kwargs):
        raise AssertionError("must not construct a second native robot model")
    monkeypatch.setattr(robot_segmenter, "create_robot_segmenter_adapter", duplicate_factory)
    monkeypatch.setattr(raw_robot_depth, "RawRobotDepthBoundary", factory)
    create_feedback_segmenter(robot_config_path=root / "resources/curobo/marvin_gui_dual.yml",
        urdf_path=root / "assets/marvin_urdf/marvin_with_grippers.urdf", asset_root_path=root / "assets",
        feedback={"A": .55, "B": .2})
    assert len(observed) == 1
    assert observed[0][0] == yaml.safe_load(original)
    assert observed[0][1] == {"A": .55, "B": .2}
    assert (root / "resources/curobo/marvin_gui_dual.yml").read_bytes() == original


def test_held_and_released_mapping_updates_feedback_without_replacing_segmenter(monkeypatch):
    from types import SimpleNamespace

    from jev_mujoco import geometry_service

    feedback_configs = []
    feedback_updates = []
    def factory(**kwargs):
        feedback_configs.append(kwargs["feedback"])
        return SimpleNamespace(sdk_boundary=SimpleNamespace(
            robot_spheres=lambda joints: np.array([[0., 0.,
                (feedback_updates[-1] if feedback_updates else kwargs["feedback"])["A"], .01]]),
            update_feedback=lambda feedback: feedback_updates.append(dict(feedback))))
    class Mapper:
        def __init__(self, boundary):
            self.diagnostics = {'semantic_only_active': False}
            self.segmenter_boundary = boundary
            self.mapper = SimpleNamespace(reset_capture=lambda: None)
            self.capture_segmenter = SimpleNamespace(filtered_depths={}, source_restoration={}, source_limits={})
        def update_segmenter(self, boundary):
            self.segmenter_boundary = boundary
        def build(self, decoded):
            self.capture_segmenter.filtered_depths = {key: view["depth_left_m"] for key, view in decoded["views"].items()}
            candidate = SimpleNamespace(selected_instance_id="source", instance_metrics={"source": {}}, camera_metrics={},
                                        table=SimpleNamespace(model={}), timings_ms={})
            return candidate, [], {"voxel": {}}
    monkeypatch.setattr(geometry_service, "create_feedback_segmenter", factory)
    monkeypatch.setattr(geometry_service, "PerceptionMapper", Mapper)
    runtime = geometry_service.GeometryRuntime()
    payload = wire() | {"planning_settings": {}, "gripper_joint_positions_rad": {"A": .55, "B": 0.}, "exclusion_state": "held"}
    runtime.build_perception(payload)
    mapper = runtime.perception_mapper
    boundary = mapper.segmenter_boundary
    runtime.build_perception(payload)
    runtime.build_perception(payload | {"gripper_joint_positions_rad": {"A": 0., "B": 0.}, "exclusion_state": "released"})
    assert feedback_configs == [{"A": .55, "B": 0.}]
    assert feedback_updates == [{"A": .55, "B": 0.}, {"A": .55, "B": 0.}, {"A": 0., "B": 0.}]
    assert runtime.perception_mapper is mapper and mapper.segmenter_boundary is boundary
    assert runtime.observed_free_guard.capture_spheres[0, 2] == 0.


def test_segmentation_config_unlocks_only_measured_drivers_and_preserves_geometry():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    from jev_mujoco.raw_robot_depth import articulated_segmentation_config

    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root / "resources/curobo/marvin_gui_dual.yml").read_bytes())
    before = deepcopy(original)
    result = articulated_segmentation_config(original,
        urdf_path=root / "assets/marvin_urdf/marvin_with_grippers.urdf", asset_root_path=root / "assets")
    kin = result["robot_cfg"]["kinematics"]
    assert not kin["lock_joints"]
    assert len(kin["cspace"]["joint_names"]) == 16
    assert kin["cspace"]["joint_names"][-2:] == ["gripL_finger_joint", "gripR_finger_joint"]
    assert len(kin["cspace"]["position_limit_clip"]) == 16
    assert kin["cspace"]["position_limit_clip"][-2:] == [0., 0.]
    assert kin["collision_spheres"] == before["robot_cfg"]["kinematics"]["collision_spheres"]
    assert original == before


def test_filtered_png_boundary_preserves_depth_units_and_scales_intrinsics(monkeypatch):
    import base64
    import io
    import sys
    from types import SimpleNamespace

    from PIL import Image
    from test_semantic_depth import Tensor

    from jev_mujoco.raw_robot_depth import RawRobotDepthBoundary

    monkeypatch.setitem(sys.modules, "curobo.types", SimpleNamespace(CameraObservation=SimpleNamespace,
        Pose=SimpleNamespace(from_matrix=lambda matrix: matrix.numpy())))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(uint8=np.uint8,
        zeros=lambda shape, device, dtype: SimpleNamespace(numpy=lambda: np.zeros(shape, dtype=dtype))))
    boundary = RawRobotDepthBoundary.__new__(RawRobotDepthBoundary)
    boundary.device_cfg = SimpleNamespace(to_device=Tensor, device='cpu')
    encoded = io.BytesIO()
    Image.fromarray(np.array([[1, 1000], [0, 65535]], dtype=np.uint16)).save(encoded, format="PNG")
    request = SimpleNamespace(depth_png_base64=base64.b64encode(encoded.getvalue()).decode(),
        depth_calibration_resolution=[4, 4], depth_intrinsics=[[8, 0, 2], [0, 8, 2], [0, 0, 1]],
        camera_pose_base_m=np.eye(4))
    result = boundary.prepare_filtered_observation(request)
    np.testing.assert_allclose(result.filtered_depth_m.numpy(), [[[.001, 1.], [0., 65.535]]])
    assert np.asarray(result.intrinsics).tolist() == [[4., 0., 1.], [0., 4., 1.], [0., 0., 1.]]
    assert result.removed_point_count == 0
    assert result.camera_observation.rgb_image.numpy().shape == (1, 2, 2, 3)
    assert result.camera_observation.rgb_image.numpy().dtype == np.uint8


def test_articulated_feedback_validates_before_replacing_measured_state():
    from pathlib import Path

    import yaml

    from jev_mujoco.raw_robot_depth import RawRobotDepthBoundary

    root = Path(__file__).resolve().parents[1]
    boundary = RawRobotDepthBoundary.__new__(RawRobotDepthBoundary)
    boundary.robot_config = yaml.safe_load((root / "resources/curobo/marvin_gui_dual.yml").read_bytes())
    boundary.urdf_path = root / "assets/marvin_urdf/marvin_with_grippers.urdf"
    boundary.update_feedback({"A": .55, "B": .2})
    for invalid in (None, {"A": .5}, {"A": float("nan"), "B": .2}, {"A": -1., "B": .2}):
        with pytest.raises(ValueError, match="gripper"):
            boundary.update_feedback(invalid)
        assert boundary.gripper_feedback == {"A": .55, "B": .2}
    boundary.update_feedback({"A": -1e-5, "B": .2})
    assert boundary.gripper_feedback == {"A": 0., "B": .2}


def test_free_space_guard_rejects_unknown_occluded_invalid_and_partial_sphere_coverage():
    from jev_mujoco.perception_mapping import ObservedFreeSpaceGuard

    view = {"depth_left_m": np.ones((32, 32)), "valid_left_mask": np.ones((32, 32)),
            "left_intrinsics": [[20, 0, 15.5], [0, 20, 15.5], [0, 0, 1]], "T_base_left_camera": np.eye(4)}
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": np.ones((32, 32))}, "rgbd:test")
    spheres = np.array([[[0, 0, .5, .05], [0, 0, 1.05, .05], [1, 0, .5, .05], [0, 0, .98, .05]]])
    valid, proof = guard.check(spheres)
    assert valid.tolist() == [False] and proof["unobserved_sphere_count"] == 2
    assert proof["outside_observation_scope_sphere_count"] == 1
    assert proof["outside_observation_scope_status"] == "unknown_not_certified_free"
    assert guard.check(spheres[:, :1])[0].tolist() == [True]
    filtered = np.ones((32, 32))
    filtered[15, 15] = 0
    invalid_view = view | {"valid_left_mask": np.ones((32, 32))}
    invalid_view["valid_left_mask"][15, 15] = 0  # rejected returns provide no free-space evidence
    masked = ObservedFreeSpaceGuard({"3": invalid_view}, {"3": filtered}, "rgbd:test")
    assert masked.check(spheres[:, :1])[0].tolist() == [False]


def test_source_outer_volume_does_not_certify_hidden_occupied_space():
    from jev_mujoco.perception_mapping import ObservedFreeSpaceGuard

    view = {"depth_left_m": np.ones((32, 32)), "valid_left_mask": np.ones((32, 32)),
            "left_intrinsics": [[20, 0, 15.5], [0, 20, 15.5], [0, 0, 1]], "T_base_left_camera": np.eye(4),
            "instance_masks": {"source": np.ones((32, 32), dtype=np.uint8)}}
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": np.ones((32, 32))}, "rgbd:test")
    pose = np.eye(4)
    pose[2, 3] = 1.05
    guard.source_estimate = {"T_world_object": pose, "size_m": [.2, .2, .2],
                             "shape_prior_provenance": {"instance_id": "source"}}
    sphere = np.array([[[0, 0, 1.05, .02]]])
    assert not guard.check(sphere)[0][0]
    assert not guard.check(sphere, source_indices=[0], source_contact_samples=[True])[0][0]
    assert not guard.check(sphere, source_indices=[0], source_contact_samples=[False])[0][0]
    sphere[0, 0, 2] = 1.2
    assert not guard.check(sphere, source_indices=[0])[0][0]


def test_conservative_capture_sphere_is_not_moving_robot_occupied_interior():
    from jev_mujoco.perception_mapping import ObservedFreeSpaceGuard

    view = {"depth_left_m": np.zeros((32, 32)), "valid_left_mask": np.zeros((32, 32)),
            "left_intrinsics": [[20, 0, 15.5], [0, 20, 15.5], [0, 0, 1]], "T_base_left_camera": np.eye(4)}
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": np.zeros((32, 32))}, "rgbd:test")
    guard.capture_spheres = np.array([[0, 0, 0, .1]])
    assert not guard.check(np.array([[[.04, 0, 0, .03]]]))[0][0]


@pytest.mark.parametrize("masked", [True, False])
def test_source_occupied_interval_requires_selected_unmasked_depth(masked):
    from jev_mujoco.perception_mapping import ObservedFreeSpaceGuard

    view = {"depth_left_m": np.ones((32, 32)), "valid_left_mask": np.ones((32, 32)),
            "left_intrinsics": [[20, 0, 15.5], [0, 20, 15.5], [0, 0, 1]], "T_base_left_camera": np.eye(4),
            "instance_masks": {"source": np.zeros((32, 32), dtype=np.uint8) if not masked else np.ones((32, 32), dtype=np.uint8)}}
    filtered = np.zeros((32, 32)) if masked else np.ones((32, 32))
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": filtered}, "rgbd:test")
    pose = np.eye(4)
    pose[2, 3] = 1.05
    guard.source_estimate = {"T_world_object": pose, "size_m": [.2, .2, .2],
                             "shape_prior_provenance": {"instance_id": "source"}}
    assert not guard.check(np.array([[[0, 0, 1.05, .02]]]), source_indices=[0])[0][0]



def test_explicit_held_map_uses_background_without_current_target_segmentation():
    from tianji_curobo.semantic_esdf_transport import (
        decode_semantic_esdf_request,
        encode_semantic_esdf_request,
    )

    from jev_mujoco.perception_mapping import decode_mapping_payload
    original = wire()
    feedback = original.pop('gripper_joint_positions_rad')
    decoded = decode_semantic_esdf_request(original)
    decoded.update(mapping_mode='background', confirmed_instance_id='', sam3_generation=0)
    for view in decoded['views'].values(): view['instance_masks'] = {}
    payload = encode_semantic_esdf_request(decoded)
    payload.update(exclusion_state='held', held_instance_id='source', gripper_joint_positions_rad=feedback)
    native, identity = decode_mapping_payload(payload)
    assert identity['target_id'] == 'source'
    assert native['mapping_mode'] == 'background'
    assert all(not view['instance_masks'] for view in native['views'].values())

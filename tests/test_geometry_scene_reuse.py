from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco import geometry_service


class Metadata(np.ndarray):
    def round_(self):
        np.rint(self, out=self)
        return self


class DistanceBackend:
    """CPU boundary double; reads uploaded field values instead of canned collision answers."""
    def __init__(self):
        self.destroyed = False
        self.features = np.array([0.])
        params = np.zeros((1, 1, 4)).view(Metadata)
        self.motion_planner = SimpleNamespace(scene_collision_checker=SimpleNamespace(
            data=SimpleNamespace(voxels=SimpleNamespace(params=params))))

    def update_voxel_scene(self, grids, model, **kwargs):
        if self.destroyed:
            raise RuntimeError("destroyed backend")
        self.features[:] = grids[0].feature_tensor

    def destroy(self):
        self.destroyed = True


def mapped_runtime(monkeypatch):
    from test_perception_mapping import wire

    class Mapper:
        def __init__(self, boundary):
            self.diagnostics = {'semantic_only_active': False}
            self.segmenter_boundary = boundary
            self.mapper = SimpleNamespace(reset_capture=lambda: None)
            self.capture_segmenter = SimpleNamespace(filtered_depths={}, source_restoration={}, source_limits={})
            self.value = 0.4
            self.size = 1.
            self.fail = False

        def build(self, decoded):
            if self.fail:
                raise ValueError("mapper failure")
            self.capture_segmenter.filtered_depths = {key: view["depth_left_m"]
                                                     for key, view in decoded["views"].items()}
            candidate = SimpleNamespace(selected_instance_id="source", instance_metrics={"source": {}},
                                        camera_metrics={}, table=SimpleNamespace(model={}), timings_ms={})
            return candidate, [SimpleNamespace(feature_tensor=np.array([self.value]),
                dims=[self.size] * 3, voxel_size=.01)], {
                "voxel": {"background": {"dims": [self.size] * 3, "voxel_size": .01}}}

    boundary = SimpleNamespace(robot_spheres=lambda joints: np.array([[0., 0., 0., .01]]),
                               update_feedback=lambda feedback: None)
    monkeypatch.setattr(geometry_service, "create_feedback_segmenter", lambda **kwargs: SimpleNamespace(sdk_boundary=boundary))
    monkeypatch.setattr(geometry_service, "PerceptionMapper", Mapper)
    runtime = geometry_service.GeometryRuntime()
    payload = wire()
    runtime.build_perception(payload)
    runtime._prepare_scene_cache_unlocked()
    return runtime, payload


def test_new_rgbd_world_reuses_solver_but_uploads_new_distances_and_rejects_old_token(monkeypatch):
    runtime, payload = mapped_runtime(monkeypatch)
    backend = DistanceBackend()
    feedback = {"A": 0., "B": 0.}
    with runtime.carrying_scope(None, feedback):
        runtime._batch_backends[("A", 16, .01, .03)] = backend
        old_token = runtime._refresh_scene_unlocked(arm="A", backend=backend)
    assert backend.features[0] == .4
    runtime.perception_mapper.value = -.2
    new_payload = payload | {"selected_surface_world_m": [[0.1, 0.1, 0.7], [0.11, 0.1, 0.7]]}
    runtime.build_perception(new_payload)
    with runtime.carrying_scope(None, feedback):
        assert runtime._batch_backends[("A", 16, .01, .03)] is backend
        runtime._prepare_scene_cache_unlocked()
        new_token = runtime._refresh_scene_unlocked(arm="A", backend=backend)
    assert new_token != old_token and backend.features[0] == -.2 and not backend.destroyed
    with pytest.raises(geometry_service.CuroboToolError) as error:
        runtime._require_scene_token_unchanged_unlocked(old_token)
    assert error.value.code == "CUROBO_SCENE_CHANGED"


def test_new_capture_invalidates_scene_peer_and_command_caches_in_both_modes(monkeypatch):
    runtime, payload = mapped_runtime(monkeypatch)
    runtime._select_planner_configuration_unlocked("open-key", "open")
    runtime._scene_snapshot_key_by_arm["A"] = "old-world"
    runtime._peer_collision_snapshot_by_arm["A"] = "old-peer"
    runtime._command_checkers["old"] = object()
    open_state = runtime._planner_configurations["open"][1]
    runtime._select_planner_configuration_unlocked("held-key", "held")
    runtime._scene_snapshot_key_by_arm["A"] = "old-world"
    runtime._peer_collision_snapshot_by_arm["A"] = "old-peer"
    runtime._command_checkers["old"] = object()
    held_state = runtime._planner_configurations["held"][1]
    backend = DistanceBackend()
    runtime._batch_backends["held"] = backend
    runtime.build_perception(payload)
    assert runtime._planner_configurations["held"][1]["_batch_backends"]["held"] is backend
    for saved in (open_state, held_state):
        for field in ("_scene_snapshot_key_by_arm", "_peer_collision_snapshot_by_arm", "_command_checkers"):
            assert not saved[field]


def test_failed_rebuild_never_publishes_old_world_even_when_solver_retained(monkeypatch):
    runtime, payload = mapped_runtime(monkeypatch)
    old_token = runtime.geometry_scene["scene_token"]
    backend = DistanceBackend()
    runtime._batch_backends["cached"] = backend
    runtime.perception_mapper.fail = True
    with pytest.raises(ValueError, match="mapper failure"):
        runtime.build_perception(payload)
    assert runtime.geometry_scene is None and runtime.geometry_grids is None
    assert runtime._batch_backends["cached"] is backend
    with pytest.raises(ValueError, match="build a registered RGBD scene"):
        runtime._require_scene_token_unchanged_unlocked(old_token)


def test_capacity_growth_still_destroys_solver_after_retained_scene_rebuild(monkeypatch):
    runtime, payload = mapped_runtime(monkeypatch)
    backend = DistanceBackend()
    runtime._batch_backends["cached"] = backend
    runtime.perception_mapper.size = 4.
    runtime.build_perception(payload)
    assert not backend.destroyed
    runtime._prepare_scene_cache_unlocked()
    assert backend.destroyed and not runtime._batch_backends


def test_changed_gripper_geometry_keeps_resident_solver(monkeypatch):
    runtime, payload = mapped_runtime(monkeypatch)
    backend = DistanceBackend()
    with runtime.carrying_scope(None, {"A": 0., "B": 0.}):
        runtime._batch_backends["cached"] = backend
    runtime.build_perception(payload)
    with runtime.carrying_scope(None, {"A": .2, "B": 0.}):
        assert not backend.destroyed and runtime._batch_backends['cached'] is backend


@pytest.mark.parametrize("background,source_layer,disabled,excluded", [
    (True, False, (), []),
    (True, True, ("semantic_instance::source",), ["source"]),
    (False, True, ("semantic_instance::source",), ["source"]),
])
def test_held_map_excludes_only_an_existing_source_layer(monkeypatch, background, source_layer, disabled, excluded):
    import struct

    from tianji_curobo.semantic_esdf_transport import encode_semantic_esdf_request

    from luna_mujoco.perception_mapping import decode_mapping_payload

    runtime, payload = mapped_runtime(monkeypatch)
    if background:
        decoded, _ = decode_mapping_payload(payload)
        payload = encode_semantic_esdf_request({
            "observation_set_id": "rgbd-4", "selection_generation": 4,
            "sam3_generation": 0, "confirmed_instance_id": "", "mapping_mode": "background",
            "views": {key: view | {"instance_masks": {}} for key, view in decoded["views"].items()},
        }) | {"held_instance_id": "source", "gripper_joint_positions_rad": {"A": 0., "B": 0.}}
    payload["exclusion_state"] = "held"
    table = {"pose": [.5, 0., .45, 1., 0., 0., 0.], "dims": [.8, .6, .305]}
    names = ["block_sparse_esdf_grid", "semantic_instance::destination"]
    if source_layer:
        names.append("semantic_instance::source")
    grids = [SimpleNamespace(name=name, feature_tensor=np.full((2, 2, 2), -.04),
        pose=[.5, 0., .7, 1., 0., 0., 0.], dims=[.02] * 3, voxel_size=.01) for name in names]
    model = {"voxel": {name: {"centers_base_m": [[.5, 0., .7]]} for name in names},
             "cuboid": {"semantic_table_forbidden": table}}
    candidate = SimpleNamespace(selected_instance_id="" if background else "source",
        instance_metrics={"destination": {}}, camera_metrics={"3": {}}, timings_ms={},
        integration=SimpleNamespace(pointcloud_data=struct.pack("<I", 0), surface_voxel_size_m=.01),
        table=SimpleNamespace(model=table, height_m=.6, normal_xyz=(0., 0., 1.), inlier_count=10))

    def build(decoded):
        runtime.perception_mapper.capture_segmenter.filtered_depths = {
            camera: view["depth_left_m"] for camera, view in decoded["views"].items()}
        return candidate, grids, model

    monkeypatch.setattr(runtime.perception_mapper, "build", build)
    held_binding = object()
    runtime.logical_carrying_binding = held_binding
    result = runtime.build_perception(payload)
    assert result["target_id"] == "source" and result["exclusion_state"] == "held"
    assert result["disabled_obstacle_names"] == disabled
    assert result["excluded_object_ids"] == excluded
    assert runtime.geometry_scene["disabled_obstacle_names"] == disabled
    assert runtime.logical_carrying_binding is held_binding
    assert runtime.geometry_model is model and runtime.geometry_grids is grids
    preview = runtime.pointcloud_preview["collision_scene"]
    assert preview["status"] == "available" and preview["table"]["enabled"]
    layers = {layer["name"]: layer for layer in preview["voxel_layers"]}
    assert layers["block_sparse_esdf_grid"]["enabled"]
    assert layers["semantic_instance::destination"]["enabled"]
    if source_layer:
        assert not layers["semantic_instance::source"]["enabled"]
    uploaded = []
    backend = DistanceBackend()
    backend.update_voxel_scene = lambda grids, model, **kwargs: uploaded.append(kwargs["disabled_obstacle_names"])
    runtime._refresh_scene_unlocked(arm="A", backend=backend)
    assert uploaded == [disabled]

import importlib.util
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_semantic_depth import CameraObservation, Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))


def request(state="held"):
    source = np.zeros((2, 4), dtype=np.uint8)
    source[0, 0] = 1
    other = np.zeros_like(source)
    other[0, 1] = 1
    views = {}
    for camera in ("1", "2", "3"):
        pose = np.eye(4)
        pose[0, 3] = int(camera)
        views[camera] = {"camera_id": camera, "stereo_sequence": 42,
            "depth_left_m": np.ones((2, 4), dtype=np.float32),
            "valid_left_mask": np.ones((2, 4), dtype=np.uint8),
            "left_intrinsics": np.eye(3).tolist(), "T_base_left_camera": pose.tolist(),
            "robot_joints_at_capture": {"A": [0.] * 7, "B": [0.] * 7},
            "rgb_size": [4, 2], "rgb_intrinsics": np.eye(3).tolist(), "left_to_rgb": np.eye(4).tolist(),
            "instance_masks": {} if camera == "3" else {"source": source.copy(), "other": other.copy()}}
    return {"observation_set_id": "capture-42", "confirmed_instance_id": "source", "selection_generation": 4,
            "sam3_generation": 4, "exclusion_state": state, "views": views}


class Segmenter:
    def __init__(self):
        self.requests = []
        self.separated = []
        self.erase_source = False

    def prepare_ffs_observation(self, value):
        from tianji_curobo.robot_segmenter import PreparedFfsDepth
        self.requests.append(value)
        depth = Tensor(np.where(value.valid_left_mask, value.depth_left_m, 0)[None])
        intrinsics = Tensor(np.asarray(value.left_intrinsics)[None])
        pose = np.asarray(value.T_base_left_camera)
        camera = CameraObservation(depth_image=depth, intrinsics=intrinsics, pose=pose)
        return PreparedFfsDepth(value.camera_id, value.stereo_sequence, camera, depth,
            camera.get_pointcloud(project_to_pose=True), intrinsics, Tensor(pose), 0)

    def remove_semantic_instances(self, value, projection, **kwargs):
        from jev_mujoco.semantic_depth import partition_semantic_depth
        return partition_semantic_depth(value, projection, camera_observation_type=CameraObservation)

    def crop_semantic_observation(self, value, table):
        if self.erase_source and "source" in value.instance_points_base_m:
            return replace(value, instance_points_base_m={**value.instance_points_base_m,
                                                         "source": np.empty((0, 3))})
        return value

    def filter_isolated_surfaces(self, values):
        return tuple(values)

    def separate_support_observation(self, value, plane):
        self.separated.append((value.camera_id, plane))
        return value


class Mapper:
    def __init__(self):
        self.instances = {}
        self.scene_voxel_grids = ["background"]
        self.integrated = None
        self.cleared = False
        self.timings_ms = {"mapper_measured": 1.}
        self.esdf_grid = None
        self.integration_count = 0

    def clear_instance_layers(self):
        self.cleared = True

    def build_instance_esdf(self, prepared, name, points, *, padding_m, table_model):
        from tianji_curobo.mixed_esdf_scene import InstanceGeometry
        assert self.cleared
        self.instances[name] = (points, padding_m)
        layer = "semantic_instance::" + name
        self.scene_voxel_grids.append(layer)
        count = sum(len(value) for value in points.values())
        return InstanceGeometry(layer, name, "voxel", {"centers_base_m": [[1, 2, 3]]}, count, count)

    def configure_mapping_bounds(self, bounds):
        self.bounds = bounds

    def integrate_multiview(self, values, *, table_model):
        self.integration_count += 1
        self.esdf_grid = object()
        self.integrated = tuple(values)
        return SimpleNamespace(pointcloud_data=struct.pack("<I", 0))


class Scene:
    def __init__(self, *, cuboid, mesh, voxel):
        self.cuboid, self.mesh, self.voxel = cuboid, mesh, voxel

    @staticmethod
    def create(model):
        return SimpleNamespace(cuboid=list(model["cuboid"]), mesh=list(model["mesh"]))


def builder(*, support=False):
    assert importlib.util.find_spec("jev_mujoco.semantic_builder"), "empty-head semantic builder missing"
    from tianji_curobo.mixed_esdf_scene import TableCompletion

    from jev_mujoco.semantic_builder import ProjectSemanticBuilder
    segmenter, mapper, table_inputs = Segmenter(), Mapper(), []
    def table_builder(groups, spheres, **kwargs):
        table_inputs.append((groups, spheres))
        return TableCompletion({"pose": [0, 0, .5, 1, 0, 0, 0], "dims": [10, 10, .1]},
            .55, (0, 0, 1), sum(len(group) for group in groups), .05,
            {"normal": [0, 0, 1]} if support else None)
    value = ProjectSemanticBuilder(robot_segmenter_boundary=segmenter, mapper_boundary=mapper,
        scene_type=Scene, robot_spheres=lambda joints: [[0, 0, 0, .04], [0, 1, 0, .09]],
        table_builder=table_builder)
    return value, segmenter, mapper, table_inputs


@pytest.mark.parametrize("state", ["held", "released"])
def test_empty_head_preserves_semantic_layers_original_table_inputs_and_robot_spheres(state):
    value, segmenter, mapper, table_inputs = builder()
    result = value.build(request(state))
    assert result.selected_instance_id == "source" and result.selection_generation == 4
    assert result.observation_set_id == "capture-42"
    assert result.scene_data.voxel == ["background", "semantic_instance::source", "semantic_instance::other"]
    assert set(result.scene_model["voxel"]) == {"semantic_instance::source", "semantic_instance::other"}
    groups, spheres = table_inputs[0]
    assert [len(group) for group in groups] == [6, 6, 8]
    assert spheres.shape == (6, 4) and spheres[:, 3].tolist() == [.04, .09] * 3
    assert [value.camera_id for value in segmenter.requests] == ["1", "2", "3"]
    assert [value.T_base_left_camera[0][3] for value in segmenter.requests] == [1, 2, 3]
    assert set(mapper.instances) == {"source", "other"}
    assert mapper.instances["source"][1] == pytest.approx(.16)
    assert mapper.instances["source"][0]["1"].tolist() == [[1, 0, 1]]
    assert mapper.instances["source"][0]["2"].tolist() == [[2, 0, 1]]
    assert result.instance_metrics["source"]["camera_count"] == 2
    assert result.instance_metrics["source"]["input_point_count"] == 2
    assert [view.camera_id for view in mapper.integrated] == ["1", "2", "3"]
    assert mapper.integrated[2].instance_points_base_m == {}
    assert result.camera_metrics["3"]["background_point_count"] == 8
    assert result.timings_ms["mapper_measured"] == 1


def test_zero_valid_individual_head_keeps_wrists_and_support_separation():
    value, segmenter, mapper, table_inputs = builder(support=True)
    payload = request()
    payload["views"]["3"]["valid_left_mask"][:] = 0
    payload["views"]["3"]["depth_left_m"][:] = 0
    result = value.build(payload)
    assert [len(group) for group in table_inputs[0][0]] == [6, 6, 0]
    assert mapper.integrated[2].background_point_count == 0
    assert [camera for camera, _ in segmenter.separated] == ["1", "2", "3"]
    assert result.camera_metrics["3"]["remaining_background_point_count"] == 0


def test_selected_point_loss_after_workspace_crop_is_still_a_global_error():
    from tianji_curobo.contracts import CuroboToolError
    value, segmenter, mapper, _ = builder()
    segmenter.erase_source = True
    with pytest.raises(CuroboToolError) as caught:
        value.build(request())
    assert caught.value.code == "mapping_primary_depth_empty"
    assert mapper.integrated is None and mapper.instances == {}


@pytest.mark.parametrize("change", ["state", "mode", "camera", "selected", "sequence"])
def test_builder_rejects_out_of_scope_or_invalid_original_capture_before_mapping(change):
    value, _, mapper, _ = builder()
    payload = request()
    if change == "state":
        payload["exclusion_state"] = "invalid"
    elif change == "head":
        payload["views"]["3"]["instance_masks"] = payload["views"]["1"]["instance_masks"]
    elif change == "mode":
        payload["mapping_mode"] = "background"
    elif change == "camera":
        payload["views"]["3"]["camera_id"] = "2"
    elif change == "selected":
        payload["confirmed_instance_id"] = "missing"
    else:
        payload["views"]["1"]["stereo_sequence"] = 99
    with pytest.raises(ValueError):
        value.build(payload)
    assert not mapper.cleared


def test_head_semantics_and_actual_collision_thresholds_control_padding():
    value, _, mapper, _ = builder()
    payload = request("selected")
    payload["views"]["3"]["instance_masks"] = payload["views"]["1"]["instance_masks"]
    payload["mapping_collision_policy"] = {"trajectory_collision_clearance_m": .08,
        "optimizer_collision_activation_distance_m": .04, "ik_collision_activation_distance_m": .12}
    result = value.build(payload)
    assert mapper.instances["source"][1] == pytest.approx(.24)
    assert result.instance_metrics["source"]["collision_distance_m"] == .12
    assert result.instance_metrics["source"]["boundary_margin_m"] == .02


def test_padding_rounds_outward_without_losing_actual_radius_coverage():
    value, _, mapper, _ = builder()
    payload = request() | {"mapping_max_sphere_radius_m": .201,
        "mapping_collision_policy": {"optimizer_collision_activation_distance_m": .033}}
    result = value.build(payload)
    assert mapper.instances["source"][1] == pytest.approx(.26)
    assert result.instance_metrics["source"]["minimum_padding_m"] == pytest.approx(.254)
    assert result.instance_metrics["source"]["padding_m"] >= .254


def test_explicit_semantic_only_removes_background_only_after_full_observation():
    value, _, mapper, _ = builder()
    payload = request("selected") | {"experimental_semantic_only_mapping": True,
                                    "full_observation_complete": True}
    result = value.build(payload)
    assert all(view.background_point_count == 0 for view in mapper.integrated)
    assert all(not view.camera_observation.depth_image.numpy().any() for view in mapper.integrated)
    assert set(mapper.instances) == {"source", "other"}
    assert result.camera_metrics["1"]["removed_unclassified_background_point_count"] == 6


def test_masked_rebuild_reuses_table_and_empty_background_but_rebuilds_instances():
    value, _, mapper, tables = builder()
    payload = request('selected') | {'experimental_semantic_only_mapping': True,
                                    'full_observation_complete': True}
    value.build(payload)
    value.build(payload | {'observation_set_id': 'new-frame'})
    assert len(tables) == 1
    assert mapper.integration_count == 1
    assert value.diagnostics['empty_background_cache_hit']


@pytest.mark.parametrize("changes", [{}, {"experimental_semantic_only_mapping": True},
    {"full_observation_complete": True}, {"experimental_semantic_only_mapping": True,
      "full_observation_complete": True, "observation_only": True}])
def test_background_is_retained_without_both_explicit_experimental_requirements(changes):
    value, _, mapper, _ = builder()
    value.build(request() | changes)
    assert [view.background_point_count for view in mapper.integrated] == [6, 6, 8]

import json
from copy import deepcopy
from dataclasses import dataclass
from inspect import Parameter, Signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco.geometry_service import GeometryRuntime, run_operation


@dataclass
class Grid:
    name: str
    pose: list
    feature_tensor: object
    dims: tuple = (.04, .04, .04)
    voxel_size: float = .02


def mapped():
    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'old', 'generation': 1, 'observation_id': 'initial',
        'target_id': 'one', 'exclusion_state': 'selected', 'disabled_obstacle_names': ('semantic_instance::one',)}
    runtime.geometry_grids = [Grid('semantic_instance::' + name, [x, 0, 0, 1, 0, 0, 0], np.ones(2))
                              for name, x in [('one', 0), ('two', 1)]]
    runtime.geometry_model = {'voxel': {grid.name: {'pose': grid.pose, 'centers_base_m': [[grid.pose[0], 0, 0]]}
                                      for grid in runtime.geometry_grids}, 'cuboid': {'semantic_table::table': {'pose': [0]}}}
    runtime.observed_free_guard = SimpleNamespace(scene_token='old', observation_id='initial', generation=1,
        views={'fixed': 'rays'}, selected_instance_id='one', selected_surface_world_m=[[0, 0, 0]])
    runtime.pointcloud_preview = None
    delta = np.eye(4)
    delta[0, 3] = .5
    request = {'scene_token': 'old', 'generation': 1, 'operation': 'transfer', 'target_id': 'one',
        'T_new_world_from_old_world': delta.tolist(), 'after_capture_id': 'released',
        'source': 'release_feedback_rigid_prediction', 'exclusion_state': 'released'}
    return runtime, request




def test_update_moves_only_selected_layer_reuses_tensor_and_publishes_new_token():
    runtime, request = mapped()
    first, other = runtime.geometry_grids
    result = run_operation(runtime, '/perception/update', request)
    assert result['scene_token'] != 'old'
    assert runtime.geometry_grids[0].pose[:3] == [.5, 0, 0]
    assert runtime.geometry_grids[0].feature_tensor is first.feature_tensor
    assert runtime.geometry_grids[1] is other
    assert first.pose[:3] == [0, 0, 0]
    assert runtime.geometry_model['voxel'][first.name]['centers_base_m'] == [[.5, 0, 0]]
    assert result['layer_evidence']['position_source'] == 'action_prediction'
    assert result['disabled_obstacle_names'] == ()


def test_update_does_not_move_or_relabel_free_space_capture():
    runtime, request = mapped()
    rays = runtime.observed_free_guard.views
    run_operation(runtime, '/perception/update', request)
    assert runtime.observed_free_guard.views is rays
    assert runtime.observed_free_guard.observation_id == 'initial'
    assert runtime.geometry_scene['observation_id'] == 'initial'


@pytest.mark.parametrize('field,value', [('scene_token', 'stale'), ('generation', 2), ('target_id', 'absent')])
def test_invalid_update_is_atomic(field, value):
    runtime, request = mapped()
    initial = deepcopy(runtime.geometry_scene)
    with pytest.raises(ValueError):
        run_operation(runtime, '/perception/update', request | {field: value})
    assert runtime.geometry_scene == initial
    assert runtime.geometry_grids[0].pose[:3] == [0, 0, 0]


def test_next_target_exclusion_recomputed_without_remapping():
    runtime, request = mapped()
    moved = run_operation(runtime, '/perception/update', request)
    selected = run_operation(runtime, '/perception/update', {'operation': 'select',
        'scene_token': moved['scene_token'], 'generation': 1, 'target_id': 'two',
        'exclusion_state': 'selected', 'selected_surface_world_m': [[1, 0, 0]]})
    assert selected['disabled_obstacle_names'] == ('semantic_instance::two',)
    assert selected['scene_token'] != moved['scene_token']
    assert runtime.geometry_grids[0].pose[:3] == [.5, 0, 0]


def test_transfer_invalidates_full_bounds_and_select_retokens_unchanged_bounds():
    runtime, request = mapped()
    layer = "semantic_instance::one"
    metadata = runtime.geometry_model["voxel"][layer]
    metadata.update(surface_cell_bounds_world_m=[[-.01] * 3, [.01] * 3],
                    surface_cell_bounds_source="all_native_surface_cells_before_preview_sampling")
    runtime.geometry_scene["collision_layer_bounds"] = {
        "scene_token": "old", "observation_id": "initial", "generation": 1,
        "layers": {layer: {"kind": "voxel", "enabled": False,
                           "minimum_world_m": [-.01] * 3, "maximum_world_m": [.01] * 3},
                   "semantic_instance::two": {"kind": "voxel", "enabled": True,
                                               "minimum_world_m": [.99, -.01, -.01],
                                               "maximum_world_m": [1.01, .01, .01]}}}
    moved = run_operation(runtime, "/perception/update", request)
    assert "surface_cell_bounds_world_m" not in runtime.geometry_model["voxel"][layer]
    bounds = moved["collision_layer_bounds"]
    assert bounds["scene_token"] == moved["scene_token"]
    assert bounds["layers"][layer]["status"] == "unavailable"
    assert "maximum_world_m" not in bounds["layers"][layer]
    selected = run_operation(runtime, "/perception/update", {
        "operation": "select", "scene_token": moved["scene_token"], "generation": 1,
        "target_id": "two", "exclusion_state": "selected", "selected_surface_world_m": [[1., 0., 0.]]})
    assert selected["collision_layer_bounds"]["scene_token"] == selected["scene_token"]
    assert selected["collision_layer_bounds"]["layers"]["semantic_instance::two"]["enabled"] is False
    assert selected["collision_layer_bounds"]["layers"][layer]["enabled"] is True


def test_old_planners_cannot_reuse_cached_scene_after_update():
    runtime, request = mapped()
    runtime._scene_snapshot_key_by_arm['A'] = 'old'
    runtime._command_checkers['old'] = object()
    run_operation(runtime, '/perception/update', request)
    assert not runtime._scene_snapshot_key_by_arm and not runtime._command_checkers




def test_obsolete_completion_is_rejected_atomically():
    runtime, request = mapped()
    name = "semantic_instance::one::height_completion"
    runtime.geometry_model["cuboid"][name] = {"pose": [0., 0., .3, 1., 0., 0., 0.],
                                             "dims": [.06, .04, .1]}
    original = deepcopy(runtime.geometry_model)
    with pytest.raises(ValueError, match="scene_geometry_policy_requires_rebuild"):
        run_operation(runtime, "/perception/update", request | {"support_floor_z_m": float("nan")})
    assert runtime.geometry_model == original


def sdk_scene_boundary(model):
    """Enforce the pinned public constructors, even on hosts without CUDA/cuRobo."""
    fields = json.loads((Path(__file__).parent / "fixtures" /
                         "curobo_scene_constructor_fields.json").read_text(encoding="utf-8"))
    for kind in ("cuboid", "mesh"):
        signature = Signature([Parameter(name, Parameter.KEYWORD_ONLY, default=None)
                               for name in fields[kind]])
        for name, geometry in model.get(kind, {}).items():
            signature.bind(name=name, **geometry)
    return deepcopy(model)








def test_geometry_adapter_does_not_hide_unknown_sdk_geometry_arguments():
    from luna_mujoco.backend_builder import create_solver_configs
    from luna_mujoco.planning_settings import PlanningSettings

    class Config:
        @staticmethod
        def create(**kwargs):
            return sdk_scene_boundary(kwargs["scene_model"])

    with pytest.raises(TypeError, match="dimension_typo"):
        create_solver_configs(PlanningSettings(), robot={}, batch_size=1, collision_cache={},
            ik_config_type=Config, scene_model={
                "cuboid": {"bad": {"pose": [0., 0., 0., 1., 0., 0., 0.],
                                    "dims": [.1] * 3, "dimension_typo": [.2] * 3}}})




def test_transfer_moves_every_retained_surface_once():
    runtime, request = mapped()
    from dataclasses import replace
    old = replace(runtime.geometry_grids[0], name='semantic_instance::one::history::prior',
                  pose=[.1, 0., 0., 1., 0., 0., 0.])
    runtime.geometry_grids.append(old)
    runtime.geometry_model['voxel'][old.name] = {'pose': old.pose}
    runtime.update_perception(request | {'support_floor_z_m': -.1})
    assert runtime.geometry_grids[0].pose[0] == .5
    assert runtime.geometry_grids[-1].pose[0] == .6
    assert old.pose[0] == .1




def test_release_preserves_native_grid_but_rejects_motion_until_new_measurement():
    from test_motion_coverage import guard

    from luna_mujoco.inventory_map import inventory_admission, reconcile_inventory

    runtime, request = mapped()
    runtime.geometry_scene, runtime.geometry_grids, runtime.geometry_model = reconcile_inventory(
        None, runtime.geometry_scene, runtime.geometry_grids, runtime.geometry_model)
    runtime.motion_coverage = guard()
    runtime.motion_coverage.scene_token = 'old'
    result = runtime.update_perception(request | {'support_floor_z_m': -.1})
    assert not inventory_admission(runtime.geometry_scene)['accepted']
    assert result['inventory_map']['entities']['one']['status'] == 'stale'
    assert set(runtime.geometry_model['cuboid']) == {'semantic_table::table'}
    proof = runtime.motion_coverage.check(np.array([[[0., 0., 1., .02]]]), np.zeros(1),
                                         scene_token=result['scene_token'])
    assert not proof['covered']

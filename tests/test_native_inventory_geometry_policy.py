"""Scene geometry is measured native voxels plus measured table cuboids."""
from copy import deepcopy

import numpy as np
import pytest
from test_demo_mapping import table_evidence
from test_perception_mapping import wire
from test_semantic_builder import builder, request

from luna_mujoco.inventory_map import apply_inventory_update, reconcile_inventory
from luna_mujoco.perception_mapping import decode_mapping_payload
from luna_mujoco.scene_update import prepare_scene_update


@pytest.mark.parametrize('field', ['demo_bin_priors', 'demo_completions'])
@pytest.mark.parametrize('value', [{}, {'source': {'source': 'known_cad_measured_registration'}}])
def test_wire_rejects_removed_geometry_inputs_even_when_empty(field, value):
    with pytest.raises(ValueError, match='synthetic_scene_geometry_unsupported'):
        decode_mapping_payload(wire() | {field: value})


@pytest.mark.parametrize('field', ['demo_bin_priors', 'demo_completions'])
def test_builder_rejects_legacy_geometry_before_touching_mapper(field):
    value, _, mapper, _ = builder()
    with pytest.raises(ValueError, match='synthetic_scene_geometry_unsupported'):
        value.build(request('selected') | {field: {}})
    assert not mapper.cleared


def test_current_builder_keeps_native_measurements_and_only_table_cuboids():
    value, _, mapper, _ = builder()
    candidate = value.build(request('selected') | {
        'demo_mapping': True, 'demo_tables': table_evidence()})
    assert not candidate.scene_model['mesh']
    assert set(candidate.scene_model['cuboid']) == {'semantic_table::table', 'semantic_table::side'}
    assert set(candidate.scene_model['voxel']) == {'semantic_instance::source', 'semantic_instance::other'}
    assert sum(len(points) for points in mapper.instances['source'][0].values()) == 2


def test_frozen_registered_container_points_all_reach_native_mapper_without_shape_completion():
    from dataclasses import replace
    from pathlib import Path

    value, segmenter, mapper, _ = builder()
    with np.load(Path(__file__).parent / 'fixtures/cad_bin_registered_depth.npz') as arrays:
        clouds = {name: arrays[key].copy() for name, key in [('source', 'source'), ('other', 'right')]}
    segmenter.filter_isolated_surfaces = lambda views: tuple(replace(view, instance_points_base_m={
        name: cloud if view.camera_id == camera else np.empty((0, 3))
        for name, cloud, camera in [('source', clouds['source'], '2'), ('other', clouds['other'], '1')]})
        for view in views)
    result = value.build(request('observation') | {'demo_mapping': True, 'demo_tables': table_evidence()})
    assert not result.scene_model['mesh']
    assert len(result.scene_model['voxel']) == 2
    np.testing.assert_array_equal(mapper.instances['source'][0]['2'], clouds['source'])
    np.testing.assert_array_equal(mapper.instances['other'][0]['1'], clouds['other'])


def test_planner_cache_and_upload_cannot_reuse_obsolete_scene_geometry():
    from types import SimpleNamespace

    from luna_mujoco.geometry_service import GeometryRuntime

    runtime = GeometryRuntime()
    runtime.geometry_scene = scene()
    runtime.geometry_model = native_model()
    runtime.geometry_model['mesh']['semantic_instance::source::bin_shell::cad_surface'] = {}
    with pytest.raises(ValueError, match='scene_geometry_policy_requires_rebuild'):
        runtime._prepare_scene_cache_unlocked()
    with pytest.raises(ValueError, match='scene_geometry_policy_requires_rebuild'):
        runtime._refresh_scene_unlocked(arm='A', backend=SimpleNamespace())


def scene():
    return {'scene_token': 'old', 'observation_id': 'capture', 'generation': 1,
        'target_id': 'source', 'exclusion_state': 'selected',
        'disabled_obstacle_names': ('semantic_instance::source',),
        'inventory_map': {'entities': {'source': {'status': 'observed', 'last_observation_id': 'capture'}},
                          'required_entity_ids': ['source']},
        'coverage': {'inventory_complete': True, 'missing_entity_ids': [], 'stale_entity_ids': []}}


def native_model():
    return {'voxel': {'semantic_instance::source': {'pose': [0, 0, 0, 1, 0, 0, 0]}},
            'cuboid': {'semantic_table::table': {'dims': [1, 1, .3], 'pose': [0, 0, 0, 1, 0, 0, 0]}},
            'mesh': {}}


@pytest.mark.parametrize('group,name', [
    ('mesh', 'semantic_instance::source::bin_shell::cad_surface'),
    ('cuboid', 'semantic_instance::source::height_completion'),
    ('cuboid', 'local_object::source'), ('cuboid', 'action_envelope::source'),
    ('cuboid', 'semantic_instance::source::bin_shell::floor'),
    ('mesh', 'semantic_instance::source::bin_shell::cad_surface::history::capture')])
@pytest.mark.parametrize('old', [False, True])
def test_reconcile_requires_rebuild_instead_of_retaining_or_dropping_forbidden_layers(group, name, old):
    clean = native_model()
    forbidden = deepcopy(clean)
    forbidden[group][name] = {'dims': [1, 1, 1]}
    previous = (scene(), [], forbidden) if old else None
    model = clean if old else forbidden
    before = deepcopy(model)
    with pytest.raises(ValueError, match='scene_geometry_policy_requires_rebuild'):
        reconcile_inventory(previous, scene(), [], model)
    assert model == before
    if previous:
        assert name in previous[2][group]


def test_action_effect_invalidates_without_creating_or_removing_occupied_geometry():
    initial, model = scene(), native_model()
    grid = object()
    payload = {'scene_token': 'old', 'generation': 1, 'target_id': 'source',
               'operation': 'invalidate', 'exclusion_state': 'released',
               'source': 'verified_drop_action_bounds', 'action_id': 'drop-1'}
    identity, grids, updated = prepare_scene_update(initial, [grid], model, payload)
    apply_inventory_update(identity, payload, updated)
    assert updated == model
    assert grids == [grid]
    assert identity['coverage']['inventory_complete'] is False
    assert identity['coverage']['stale_entity_ids'] == ['source']
    assert identity['layer_evidence']['geometry_invalidated'] is True
    assert identity['disabled_obstacle_names'] == ()
    assert initial['coverage']['inventory_complete'] is True


def test_action_envelope_legacy_operation_is_rejected_without_mutation():
    initial, model = scene(), native_model()
    with pytest.raises(ValueError, match='scene_update_operation_invalid'):
        prepare_scene_update(initial, [], model, {
            'scene_token': 'old', 'generation': 1, 'target_id': 'source',
            'operation': 'action_envelope', 'exclusion_state': 'released'})
    assert initial == scene()
    assert model == native_model()


def test_simulation_geometry_build_endpoint_is_rejected_without_reading_payload_or_mutating_scene():
    from luna_mujoco.geometry_service import GeometryRuntime, run_operation

    runtime = GeometryRuntime()
    runtime.geometry_scene, runtime.geometry_model = scene(), native_model()
    with pytest.raises(ValueError, match='simulation_geometry_mapping_unsupported'):
        run_operation(runtime, '/geometry/build', {})
    assert runtime.geometry_scene == scene()
    assert runtime.geometry_model == native_model()


def test_old_local_aabb_cannot_replace_measured_native_voxels():
    initial, model = scene(), native_model()
    payload = {'scene_token': 'old', 'generation': 1, 'target_id': 'product',
               'operation': 'local_object', 'exclusion_state': 'released',
               'replaces_entity_id': 'source', 'camera_id': '3', 'after_capture_id': 'after',
               'source': 'head_rgbd_object_surface_with_height_prior',
               'bounds': {'minimum_world_m': [0, 0, 0], 'maximum_world_m': [1, 1, 1]}}
    with pytest.raises(ValueError, match='local_object_requires_registered_rgbd_mapping'):
        prepare_scene_update(initial, [], model, payload)


def test_release_with_support_preserves_native_tensor_and_marks_stale_without_envelope():
    from test_scene_update import Grid

    initial, model = scene(), native_model()
    grid = Grid('semantic_instance::source', [0, 0, 0, 1, 0, 0, 0], np.ones(2))
    transform = np.eye(4)
    transform[2, 3] = .5
    payload = {'scene_token': 'old', 'generation': 1, 'target_id': 'source',
               'operation': 'transfer', 'exclusion_state': 'released',
               'source': 'release_feedback_rigid_prediction', 'after_capture_id': 'after',
               'support_floor_z_m': 0., 'T_new_world_from_old_world': transform.tolist()}
    identity, grids, updated = prepare_scene_update(initial, [grid], model, payload)
    apply_inventory_update(identity, payload, updated)
    assert set(updated['cuboid']) == {'semantic_table::table'}
    assert grids[0].feature_tensor is grid.feature_tensor
    assert identity['coverage']['stale_entity_ids'] == ['source']
    assert identity['layer_evidence']['geometry_invalidated'] is True

"""CPU boundary regressions keep modeled geometry distinct from measured points."""
from copy import deepcopy
from unittest.mock import AsyncMock

import numpy as np
import pytest
from test_current_grasp_attachment import fixture as grasp_fixture
from test_resolve_placement import setup
from test_review_place_coverage_scope import command_boundary
from test_sequential_placement_occupancy import resolve, retained_scene

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.carrying import check_command_path
from luna_mujoco.placement_constraints import release_configuration_clearance
from luna_mujoco.robot_skills.placement_resolution import occupancy_envelopes


def hybrid_layer(scene, kind):
    retained_scene(scene)
    layer = scene['collision_layer_bounds']['layers']['semantic_instance::already-delivered']
    layer.update(kind=kind, instance_id='already-delivered', observation_id='previous-placement',
        generation=scene['generation'], geometry_source='registered_cad' if kind == 'mesh' else 'observed_obb',
        source='published_cad_mesh_bounds' if kind == 'mesh' else 'published_cuboid_bounds')
    for key in ('bounds_source', 'cell_count', 'voxel_size_m'):
        del layer[key]
    return layer


@pytest.mark.parametrize('kind', ['mesh', 'cuboid'])
@pytest.mark.parametrize('occupied', [False, True])
def test_retained_published_models_are_available_occupancy_without_inventing_measurements(monkeypatch, kind, occupied):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    before = deepcopy(obs)
    layer = hybrid_layer(scene, kind)
    if occupied:
        service.world_boxes = [([.3, -.5, .60], [.9, .2, .75])]
        with pytest.raises(AlgorithmError, match='No') as error:
            resolve(algorithm, obs, scene, held, request, zone)
        assert error.value.code == 'placement_no_candidate'
        evidence = error.value.details['selection_evidence']
        assert evidence['occupied_candidate_count'] > 0
    else:
        requests, evidence = resolve(algorithm, obs, scene, held, request, zone)
        assert requests
    assert evidence['retained_occupancy_ids'] == ['already-delivered']
    assert 'already-delivered' not in evidence['missing_geometry_ids']
    assert evidence['occupancy_geometry_sources'] == [layer['source']]
    assert evidence['occupancy_bounds_are_raw_measurements'] is False
    assert all(entity['id'] != 'already-delivered' for entity in obs['objects'])
    for old, new in zip(before['objects'], obs['objects'], strict=True):
        np.testing.assert_array_equal(old['points_world_m'], new['points_world_m'])
    assert {path for path, _ in service.calls} == {'/tools/resolve_release_height'}


@pytest.mark.parametrize('kind', ['mesh', 'cuboid'])
@pytest.mark.parametrize('invalid', ['instance_id', 'source', 'geometry_source', 'disabled', 'bounds',
                                    'observation_id', 'generation'])
def test_published_model_occupancy_rejects_mismatched_or_disabled_bounds(monkeypatch, kind, invalid):
    _, _, obs, scene, _, _, zone = setup(monkeypatch)
    layer = hybrid_layer(scene, kind)
    if invalid == 'disabled':
        scene['disabled_obstacle_names'] = ['semantic_instance::already-delivered']
    elif invalid == 'bounds':
        layer['maximum_world_m'] = layer['minimum_world_m']
    else:
        layer[invalid] = 'unrelated'
    _, missing, retained = occupancy_envelopes(obs, scene,
        {'seen-red', 'demo-destination', zone['support_entity_id']}, zone['floor_z_m'])
    assert 'already-delivered' in missing
    assert retained == []


@pytest.mark.parametrize('kind,scope', [
    (None, 'observed_surface_only'), ('obb_tool', 'observed_obb_volume'),
    ('cad_tool', 'registered_cad_volume')])
async def test_postlift_result_reports_the_bound_attachment_geometry(tmp_path, kind, scope):
    runner, context, current, _, _, _ = grasp_fixture(tmp_path)
    runner.state['current_step'] = {'skill': 'execute_grasp'}
    q = current['arms']['A']['joints_rad']
    context['plan'] = {'T_world_tcp': current['arms']['A']['tcp'], 'phases': {
        phase: {'positions_rad': [q, q], 'dt': .02} for phase in ('approach', 'grasp', 'lift')}}
    runner.verify_scene = AsyncMock()
    runner.validate_motion = AsyncMock(return_value={'collision_free': True})
    runner.simulation.execute_trajectory = lambda *args: None
    runner.simulation.dwell = lambda *args: None

    async def stage(context, name, *args, **kwargs):
        if name == 'verify_holding_after_lift':
            return {'success': True}
        if name == 'prepare_attachment':
            # Lower-boundary geometry validation is exercised by CAD/OBB binding
            # tests; here the runner receives the successfully prepared binding.
            context['holding'].pop('cad_tool', None)
            context['holding'].pop('obb_tool', None)
            if kind:
                context['holding'][kind] = {'instance_id': context['holding']['instance_id']}
            context['holding'].update(attached=True, verified=True)
            return {'attached': True}
        return {}

    runner.execution_stage = stage
    result = await runner.execute_grasp(context)
    assert result['carried_geometry_scope'] == scope
    assert result['lift_carried_collision_checked'] is False
    assert result['lift_closed_gripper_collision_checked'] is True


@pytest.mark.parametrize('world_source,scope,coverage', [
    (None, 'measured_obstacles_and_both_robot_arms', 'visible_surface_only'),
    ('measured_semantic_esdf', 'measured_obstacles_and_both_robot_arms', 'visible_surface_only'),
    ('inventory_cad_meshes_obb_and_measured_table_cuboids',
     'published_inventory_geometry_and_both_robot_arms', 'registered_cad_and_observed_obb_models')])
def test_command_audit_names_the_checked_world_without_certifying_unseen_space(
        monkeypatch, world_source, scope, coverage):
    runtime, planner, payload, binding, calls = command_boundary(monkeypatch, 'observation')
    runtime.geometry_scene = {} if world_source is None else {'collision_world_source': world_source}
    result = check_command_path(runtime, planner, payload, binding)
    assert result['collision_free']
    assert {kind for _, kind in calls} == {'world', 'self', 'bounds'}
    assert result['collision_scope'] == scope
    assert result['geometry_coverage'] == coverage
    assert result.get('collision_world_source') == world_source
    assert result['whole_robot_unknown_space_certified'] is False
    assert result['full_object_coverage'] is False


@pytest.mark.parametrize('kind,source,assumption', [
    (None, 'measured_carried_surface_and_own_robot_kinematics', 'observed_surface_only'),
    ('obb_tool', 'observed_obb_carried_volume_and_own_robot_kinematics', 'observed_obb_solid_approximation'),
    ('cad_tool', 'registered_cad_carried_volume_and_own_robot_kinematics', 'registered_cad_model_and_validated_pose')])
def test_configuration_screen_identifies_its_geometry_and_keeps_none_voxel_support(
        monkeypatch, kind, source, assumption):
    monkeypatch.setattr('luna_mujoco.placement_constraints.configuration_spheres',
        lambda value: (np.array([[0., 0., .5, .03]]), ('fixture_link',)))
    goal = {'T_world_tcp': np.eye(4).tolist(), 'release_reference': {
        'attachment_spheres_tool_m': [[0., 0., 1., .02]], 'floor_z_m': 0.,
        'required_clearance_m': .01, 'target_reference_world_xy_m': [0., 0.]}}
    holding = {'points_tool_m': [[0., 0., 1.]]}
    if kind:
        holding[kind] = {'instance_id': 'part'}
    result = release_configuration_clearance(goal, holding, [{'id': 'posture'}], surface_voxel_size_m=None)
    assert result['source'] == source
    assert result['assumptions'] == ['rigid_object_vertical_settling', assumption]
    assert not result['compatible']
    assert result['configurations'][0]['surface_cell_reserve_m'] == 0.
    assert result['motion_collision_certified'] is False

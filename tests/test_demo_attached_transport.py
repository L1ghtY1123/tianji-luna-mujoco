"""Ordinary attached pose planning uses the retained world and actual command."""
from copy import deepcopy

import numpy as np
import pytest
from test_demo_planning import GraspClient, placement_inputs
from test_grasp_fk_goal import fk_evidence, inputs, pose

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.carrying import validate_phase_request
from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.planning_settings import PlanningSettings


def workspace(arm, positions):
    return [{'reachable': True, 'position_world_m': value} for value in positions]


def test_transport_phase_requires_a_measured_binding():
    with pytest.raises(ValueError, match='requires'):
        validate_phase_request({'phase': 'transport'})
    assert validate_phase_request({'phase': 'transport', 'carrying_binding': {'state': 'held'}}) == 'transport'
    with pytest.raises(ValueError, match='held'):
        validate_phase_request({'phase': 'transport', 'carrying_binding': {'state': 'preflight'}})


def test_container_reachability_uses_selected_kind_not_global_stage():
    observation, _, _, _ = placement_inputs()
    next(item for item in observation['objects'] if item['id'] == 'demo-destination')['kind'] = 'bin'
    queries = []
    def lookup(arm, positions):
        queries.append((arm, positions))
        return workspace(arm, positions)
    result = DemoAlgorithms(workspace_lookup=lookup).evaluate_reachability(
        observation, 'seen-red', 'demo-destination', requested_arm='B')
    assert len(queries[0][1]) == 1 and all(arm == 'B' and len(points) <= 2 for arm, points in queries)
    assert result['B']['destination_workspace_context']['tested_count'] == 3
    assert result['B']['destination']['reachable'] is None and result['B']['reachable']
    assert result['B']['destination_candidates'] == []
    assert result['B']['destination_id'] == 'demo-destination'
    assert result['B']['destination_status'] == 'pending_above_container_pose'
    assert result['B']['feasibility_scope'] == 'source_position_workspace_only'


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_explicit_transport_request_has_no_table_slot_requirement_but_keeps_direction_and_insertion(arm):
    observation, _, _, _ = placement_inputs()
    client = GraspClient()
    request = {'object_id': 'seen-red', 'transport_region': {
        'center_world_xy_m': [.3, .2 if arm == 'A' else -.2],
        'occupied_top_z_m': .65, 'observation_id': observation['observation_id']}}
    result = DemoAlgorithms(graspgen_client=client).generate_grasps(
        observation, 'seen-red', arm=arm, placement_request=request, planning_settings={'grasp_insertion_m': .01, 'grasp_maximum_tilt_deg': 45., 'grasp_require_away_from_robot': True})
    assert len(client.calls) == 3
    assert result['fixed_slot_reach_filter']['applied'] is True
    assert result['fixed_slot_reach_filter']['screening_scope'] == 'nominal_transport_wrist_reach'
    assert result['downward_filter']['maximum_tilt_deg'] == 45.
    assert len(result['candidates']) == 3
    assert [candidate['index'] for candidate in result['candidates']] == [6, 7, 8]


def test_verified_product_source_and_final_fk_skip_width_prediction_but_keep_nominal_bounds(monkeypatch):
    observation, scene, grasps, item = inputs('B')
    bind_verified_product(observation)
    monkeypatch.setattr(DemoAlgorithms, '_gripper_quality', lambda *args: pytest.fail('legacy width heuristic'))
    algorithm = DemoAlgorithms(curobo_request=lambda *args: {
        'scene_token': scene['scene_token'], 'batch_size': 1, 'items': [item]})
    candidate = algorithm.evaluate_arms(observation, 'seen-red', grasps, scene, requested_arm='B')['B']['valid_candidates'][0]
    phases = {'grasp': {'positions_rad': [[0.] * 7, [.1] * 7]}}
    response = {'grasp_endpoint_fk': fk_evidence(pose(.504), 'B', joints=[.1] * 7)}
    proof = algorithm._grasp_plan_evidence(candidate, response, phases, observation, 'seen-red', 'B', scene,
                                         PlanningSettings().model_dump())
    assert proof['final_grasp_validation']['accepted']
    response['grasp_endpoint_fk'] = fk_evidence(pose(.506), 'B', joints=[.1] * 7)
    with pytest.raises(AlgorithmError, match='original nominal'):
        algorithm._grasp_plan_evidence(candidate, response, phases, observation, 'seen-red', 'B', scene,
                                      PlanningSettings().model_dump())


def test_dual_vla_pose_uses_retained_scene_and_validates_both_commands():
    observation, scene, _, _ = placement_inputs()
    scene.update(exclusion_state='released', return_after_place=True)
    goals = {'A': [-86, 76, 88, -90, 2, 14, -24], 'B': [106, 61, -113, -78, 27, 36, 37]}
    calls = []

    def service(path, payload):
        calls.append((path, deepcopy(payload)))
        if path == '/tools/plan_dual_observation':
            assert payload['goal_joints_deg'] == goals
            assert 'goal_transforms_world' not in payload
            assert payload['return_after_place'] is True
            return {'scene_token': scene['scene_token'], 'attached': False,
                    'trajectories': {arm: [[0.] * 7, goals[arm]] for arm in ('A', 'B')},
                    'sample_dt_s': .02}
        assert path == '/tools/validate_dual_observation'
        return {'scene_token': scene['scene_token'], 'collision_free': True}

    plan = DemoAlgorithms(curobo_request=service).plan_dual_vla_motion(
        observation, scene, goals, planning_settings={'time_scale': 2.5})
    assert len(calls) == 2 and plan['dt'] == .02
    np.testing.assert_allclose(plan['paths']['B'][-1], np.deg2rad(goals['B']))
    assert plan['command_check']['collision_free']
    assert calls[0][1]['planning_settings']['time_scale'] == 2.5


def test_scene_invalidation_retokens_without_inventing_geometry():
    _, scene, _, _ = placement_inputs()
    before = deepcopy(scene)
    calls = []
    def service(path, body):
        calls.append((path, deepcopy(body)))
        return {'scene_token': 'updated', 'exclusion_state': 'released', 'rgbd_remapped': False}
    result = DemoAlgorithms(curobo_request=service).invalidate_scene_entity(
        scene, 'seen-red', 'drop-4', source='verified_drop_action_bounds')
    assert scene == before
    assert result['scene_token'] == 'updated' and result['rgbd_remapped'] is False
    assert calls[0][0] == '/perception/update'
    assert calls[0][1]['operation'] == 'invalidate'
    assert 'bounds' not in calls[0][1]
    assert calls[0][1]['action_id'] == 'drop-4'
    assert calls[0][1]['source'] == 'verified_drop_action_bounds'


@pytest.mark.parametrize('target,action,source', [('', 'action-1', 'mock_vla'),
    ('seen-red', '', 'mock_vla'), ('seen-red', 'action-1', 'invented')])
def test_scene_invalidation_requires_identity_and_action_provenance(target, action, source):
    _, scene, _, _ = placement_inputs()
    algorithm = DemoAlgorithms(curobo_request=lambda *args: pytest.fail('invalid request submitted'))
    with pytest.raises(ValueError):
        algorithm.invalidate_scene_entity(scene, target, action, source=source)


def bind_verified_product(observation):
    observation['inventory_state'] = {'entities': {'seen-red': {
        'id': 'seen-red', 'processing_skill': 'vla_unpack', 'process_id': 'process-1',
        'input_entity_id': 'wrapped-source', 'identity_source': 'confirmed_current_product_creation',
        'creation_evidence': {'source': 'confirmed_unpack_process_output', 'output_entity_id': 'seen-red',
            'input_entity_id': 'wrapped-source', 'process_id': 'process-1', 'processing_skill': 'vla_unpack',
            'generation': observation['generation'], 'observation_id': 'creation-observation',
            'action_id': 'effect-1', 'request_hash': 'confirmed-effect'}}}}


@pytest.mark.parametrize('mutation', ['none', 'unrelated_target', 'wrong_generation', 'missing_action', 'wrong_input', 'no_metadata'])
def test_only_bound_verified_product_metadata_can_skip_width_prediction(monkeypatch, mutation):
    observation, scene, grasps, item = inputs('B')
    bind_verified_product(observation)
    entity = observation['inventory_state']['entities']['seen-red']
    if mutation == 'unrelated_target':
        entity['creation_evidence']['output_entity_id'] = 'other'
    elif mutation == 'wrong_generation':
        entity['creation_evidence']['generation'] = observation['generation'] + 1
    elif mutation == 'missing_action':
        entity['creation_evidence'].pop('action_id')
    elif mutation == 'wrong_input':
        entity['creation_evidence']['input_entity_id'] = 'other'
    elif mutation == 'no_metadata':
        observation.pop('inventory_state')
    observation['stage'] = 'unpacked'  # Obsolete external tags cannot grant exemptions.
    predictions = []
    def predict(*args):
        predictions.append(True)
        return {'accepted': True}
    monkeypatch.setattr(DemoAlgorithms, '_gripper_quality', predict)
    algorithm = DemoAlgorithms(curobo_request=lambda *args: {
        'scene_token': scene['scene_token'], 'batch_size': 1, 'items': [item]})
    candidates = algorithm.evaluate_arms(observation, 'seen-red', grasps, scene, requested_arm='B')['B']['valid_candidates']
    assert candidates
    assert bool(predictions) is (mutation != 'none')


def test_old_stage_tag_cannot_disable_reachability_for_a_support_destination(monkeypatch):
    from luna_mujoco.algorithms import Algorithms
    observation = {'stage': 'unpacked', 'objects': [{'id': 'support', 'kind': 'marker'}]}
    calls = []
    def evaluate(self, snapshot, target, destination_id=None, **kwargs):
        calls.append(destination_id)
        return {'A': {'reachable': True}}
    monkeypatch.setattr(Algorithms, 'evaluate_reachability', evaluate)
    with pytest.raises(ValueError, match='placement_requires_measured_floor'):
        DemoAlgorithms().evaluate_reachability(observation, 'item', 'support')
    assert not calls

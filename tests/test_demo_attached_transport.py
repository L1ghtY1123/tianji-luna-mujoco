"""Ordinary attached pose planning uses the retained world and actual command."""
import hashlib
import json
import threading
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


class PoseService:
    def __init__(self, observation, arm, *, bad=None, cancelled=None):
        self.observation, self.arm = observation, arm
        self.bad, self.cancelled, self.calls = bad, cancelled, []

    def __call__(self, path, payload):
        self.calls.append((path, deepcopy(payload)))
        if path == '/tools/plan_carried_pose':
            q = np.asarray(self.observation['arms'][self.arm]['joints_rad'])
            values = np.array([q, q + .03, q + .06])
            side = 'L' if self.arm == 'A' else 'R'
            trajectory = {'joint_names': [f'Joint{i}_{side}' for i in range(1, 8)],
                'positions_rad': values.tolist(), 'velocities_rad_s': np.zeros_like(values).tolist(),
                'accelerations_rad_s2': np.zeros_like(values).tolist(), 'interpolation_dt_s': .02}
            if self.bad == 'derivatives':
                trajectory.pop('velocities_rad_s')
            if self.cancelled is not None:
                self.cancelled.set()
            return {'ok': True, 'attached': self.bad != 'detached',
                'scene_token': 'stale' if self.bad == 'token' else payload['scene_token'],
                'trajectory': trajectory,
                'collision_evidence': {'optimizer_attachment': self.bad != 'optimizer'}}
        assert path == '/carrying/validate'
        assert payload['phase'] == 'transport'
        validate_phase_request(payload)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                          allow_nan=False).encode()).hexdigest()
        return {'scene_token': payload['scene_token'], 'binding_sha256': digest,
                'attached': True, 'collision_free': self.bad != 'collision'}


@pytest.mark.parametrize('arm', ['A', 'B'])
@pytest.mark.parametrize('scale', [1., 2.5])
def test_single_attached_pose_preserves_goal_and_checks_quintic_command(arm, scale):
    observation, scene, holding, _ = placement_inputs()
    holding['arm'] = arm
    original = deepcopy(holding)
    goal = pose(.36)
    service = PoseService(observation, arm)
    result = DemoAlgorithms(curobo_request=service, workspace_lookup=workspace).plan_attached_pose(
        observation, 'seen-red', arm, scene, holding, goal,
        planning_settings={'time_scale': scale})
    assert [p for p, _ in service.calls] == ['/tools/plan_carried_pose', '/carrying/validate']
    payload = service.calls[0][1]
    np.testing.assert_allclose(payload['goal_position_m'], goal[:3, 3])
    assert payload['carrying_binding']['arm'] == arm
    assert payload['planning_settings']['trajectory_collision_clearance_m'] == .01
    assert 'trajectory_point_scale' not in payload
    assert len(result['command']['positions_rad']) == int(2 * scale) + 1
    assert result['command']['dt'] == .02
    assert service.calls[1][1]['positions_rad'] == result['command']['positions_rad']
    np.testing.assert_allclose(result['goal'], goal)
    assert result['scene_token'] == scene['scene_token'] and holding == original


@pytest.mark.parametrize('fault', ['token', 'detached', 'optimizer', 'derivatives', 'collision'])
def test_untrusted_pose_result_cannot_produce_an_executable_command(fault):
    observation, scene, holding, _ = placement_inputs()
    service = PoseService(observation, 'A', bad=fault)
    with pytest.raises(AlgorithmError):
        DemoAlgorithms(curobo_request=service, workspace_lookup=workspace).plan_attached_pose(
            observation, 'seen-red', 'A', scene, holding, pose())
    if fault != 'collision':
        assert len(service.calls) == 1


@pytest.mark.parametrize('fault', ['unverified', 'detached', 'wrong_arm', 'wrong_target'])
def test_transport_rejects_invalid_holding_before_backend(fault):
    observation, scene, holding, _ = placement_inputs()
    if fault == 'unverified': holding['verified'] = False
    elif fault == 'detached': holding['attached'] = False
    elif fault == 'wrong_arm': holding['arm'] = 'B'
    else: holding['instance_id'] = 'another'
    service = PoseService(observation, 'A')
    with pytest.raises(AlgorithmError):
        DemoAlgorithms(curobo_request=service, workspace_lookup=workspace).plan_attached_pose(
            observation, 'seen-red', 'A', scene, holding, pose())
    assert not service.calls


def test_cancel_after_pose_service_stops_before_command_validation():
    observation, scene, holding, _ = placement_inputs()
    event = threading.Event()
    service = PoseService(observation, 'A', cancelled=event)
    with pytest.raises(AlgorithmError):
        DemoAlgorithms(curobo_request=service, workspace_lookup=workspace).plan_attached_pose(
            observation, 'seen-red', 'A', scene, holding, pose(), cancelled=event)
    assert len(service.calls) == 1


def test_transport_phase_requires_a_measured_binding():
    with pytest.raises(ValueError, match='requires'):
        validate_phase_request({'phase': 'transport'})
    assert validate_phase_request({'phase': 'transport', 'carrying_binding': {'state': 'held'}}) == 'transport'
    with pytest.raises(ValueError, match='held'):
        validate_phase_request({'phase': 'transport', 'carrying_binding': {'state': 'preflight'}})


def test_unpacked_reachability_does_not_query_the_right_box_floor():
    observation, _, _, _ = placement_inputs()
    observation['stage'] = 'unpacked'
    queries = []
    def lookup(arm, positions):
        queries.append((arm, positions))
        return workspace(arm, positions)
    result = DemoAlgorithms(workspace_lookup=lookup).evaluate_reachability(
        observation, 'seen-red', 'demo-destination', requested_arm='B')
    assert len(queries) == 1 and len(queries[0][1]) == 1
    assert result['B']['destination'] is None and result['B']['reachable']
    assert result['B']['destination_id'] == 'demo-destination'
    assert result['B']['destination_status'] == 'pending_actual_lift_transport_goal'


@pytest.mark.parametrize('answer', [False, 'malformed'])
def test_actual_transport_goal_must_pass_workspace_before_planning(answer):
    observation, scene, holding, _ = placement_inputs()
    queries = []
    def lookup(arm, positions):
        queries.append((arm, positions))
        return [{'reachable': answer, 'position_world_m': positions[0]}]
    algorithm = DemoAlgorithms(workspace_lookup=lookup,
        curobo_request=lambda *args: pytest.fail('unreachable goal was submitted'))
    goal = pose(.36)
    with pytest.raises(AlgorithmError, match='workspace'):
        algorithm.plan_attached_pose(observation, 'seen-red', 'A', scene, holding, goal)
    assert queries == [('A', [goal[:3, 3].tolist()])]


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_unpacked_generation_has_no_table_slot_requirement_but_keeps_direction_and_insertion(arm):
    observation, _, _, _ = placement_inputs()
    observation['stage'] = 'unpacked'
    client = GraspClient()
    request = {'object_id': 'seen-red', 'transport_region': {
        'center_world_xy_m': [.3, .2 if arm == 'A' else -.2],
        'occupied_top_z_m': .65, 'observation_id': observation['observation_id']}}
    result = DemoAlgorithms(graspgen_client=client).generate_grasps(
        observation, 'seen-red', arm=arm, placement_request=request)
    assert len(client.calls) == 3
    assert result['fixed_slot_reach_filter']['applied'] is True
    assert result['fixed_slot_reach_filter']['screening_scope'] == 'nominal_transport_wrist_reach'
    assert result['downward_filter']['maximum_tilt_deg'] == 45.
    assert len(result['candidates']) == 3
    assert [candidate['index'] for candidate in result['candidates']] == [6, 7, 8]


def test_unpacked_source_and_final_fk_skip_width_prediction_but_keep_nominal_bounds(monkeypatch):
    observation, scene, grasps, item = inputs('B')
    observation['stage'] = 'unpacked'
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


def test_action_envelope_is_explicitly_derived_and_retokens_retained_scene():
    _, scene, _, _ = placement_inputs()
    before = deepcopy(scene)
    bounds = {'minimum_world_m': [.2, -.7, .4], 'maximum_world_m': [.3, -.5, .7]}
    calls = []
    def service(path, body):
        calls.append((path, deepcopy(body)))
        return {'scene_token': 'updated', 'exclusion_state': 'released', 'rgbd_remapped': False}
    result = DemoAlgorithms(curobo_request=service).publish_action_envelope(
        scene, 'seen-red', bounds, 'drop-4', source='verified_drop_action_bounds')
    assert scene == before
    assert result['scene_token'] == 'updated' and result['rgbd_remapped'] is False
    assert calls[0][0] == '/perception/update'
    assert calls[0][1]['operation'] == 'action_envelope'
    assert calls[0][1]['bounds'] == bounds and calls[0][1]['action_id'] == 'drop-4'
    assert calls[0][1]['source'] == 'verified_drop_action_bounds'


@pytest.mark.parametrize('bounds', [None, {'minimum_world_m': [0, 0, 0], 'maximum_world_m': [1, 0, 1]}])
def test_action_envelope_rejects_malformed_geometry_without_sending(bounds):
    _, scene, _, _ = placement_inputs()
    algorithm = DemoAlgorithms(curobo_request=lambda *args: pytest.fail('invalid envelope was submitted'))
    with pytest.raises((TypeError, ValueError)):
        algorithm.publish_action_envelope(scene, 'seen-red', bounds, 'action-1')

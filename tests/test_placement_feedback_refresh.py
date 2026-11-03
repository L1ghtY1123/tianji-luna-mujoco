"""A cached pose is a proposal; current checked paths authorize movement."""
from copy import deepcopy

import numpy as np
import pytest
from test_resolve_placement import setup

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.robot_skills.placement_resolution import digest


@pytest.mark.parametrize('changed', ['gripper', 'peer_arm'])
def test_feedback_change_replans_same_pose_with_current_robot_and_checks_all_phases(monkeypatch, changed):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    obs['arms']['A']['gripper_position_rad'] = .43867188930007794
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    frozen = deepcopy(resolution)
    if changed == 'gripper':
        obs['arms']['A']['gripper_position_rad'] = .43868042376435484  # Actual A16 snapshots.
    else:
        obs['arms']['B']['joints_rad'][0] += .01
    service.calls.clear()
    result = algorithm.plan_place(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_resolution=resolution)
    assert digest(resolution) == digest(frozen)
    assert result['resolution_feedback_check']['consistent'] is False
    assert result['resolution_feedback_check']['authorization'] == 'current_three_phase_trajectory_checks'
    assert result['resolution_feedback_check']['reference_endpoint_proof_current'] is False
    assert set(result['command_checks']) == {'approach', 'place', 'retreat'}
    assert set(result['phases']) == {'approach', 'place', 'retreat'}
    np.testing.assert_array_equal(result['goal']['T_world_tcp'], frozen['candidates'][0]['goal']['T_world_tcp'])
    _path, payload = next(item for item in service.calls if item[0] == '/tools/plan_place')
    assert payload['gripper_joint_positions_rad']['A'] == obs['arms']['A']['gripper_position_rad']
    np.testing.assert_allclose(payload['other_joints_deg'], np.rad2deg(obs['arms']['B']['joints_rad']))
    assert '/tools/resolve_release_height' not in [path for path, _ in service.calls]


def test_feedback_refresh_cannot_turn_a_current_path_collision_into_success(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    obs['arms']['A']['gripper_position_rad'] += 8.5e-6
    service.calls.clear()
    service.failure = 'collision'
    with pytest.raises(AlgorithmError) as caught:
        algorithm.plan_place(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_resolution=resolution)
    assert caught.value.code == 'place_no_valid_candidate'
    assert service.calls and '/tools/plan_place' in [path for path, _ in service.calls]
    assert all(item['code'] == 'execution_collision' for item in caught.value.details['candidate_failures'])

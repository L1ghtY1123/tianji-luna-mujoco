import json

import numpy as np
import pytest

from luna_mujoco.demo_grasp import prepare_grasp_candidates
from luna_mujoco.planning_settings import PlanningSettings, load_settings


def test_noncontact_clearance_is_configurable_and_saved_value_is_preserved(tmp_path):
    values = {'trajectory_collision_clearance_m': .004,
              'optimizer_collision_activation_distance_m': .012}
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(values))
    assert PlanningSettings(**values).trajectory_collision_clearance_m == .004
    assert load_settings(path).trajectory_collision_clearance_m == .004
    with pytest.raises(ValueError):
        PlanningSettings(trajectory_collision_clearance_m=-.001)


def test_extra_phase_holds_default_to_zero_and_gripper_feedback_is_configurable():
    settings = PlanningSettings()
    for name, value in settings.model_dump().items():
        if name.startswith('dwell_'):
            assert value == 0, name
    assert settings.gripper_stable_velocity_rad_s > 0
    assert settings.gripper_timeout_s > settings.holding_stable_window_s
    assert settings.motion_settle_timeout_s > settings.motion_stable_window_s


def test_candidates_keep_side_grasps_and_apply_configured_insertion_once():
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    pose[:3, 3] = [.4, 0, .8]
    candidates = [{'index': 0, 'score': .9, 'T_world_tcp': pose.tolist()}]
    settings = PlanningSettings(grasp_insertion_m=.003)
    kwargs = {'robot_base_world_m': [0, 0, 0], 'target_reference_world_m': [.4, 0, .8],
              'planning_settings': settings.model_dump()}
    shifted, accepted, evidence = prepare_grasp_candidates(candidates, **kwargs)
    assert len(accepted) == 1
    np.testing.assert_allclose(np.asarray(accepted[0]['T_world_tcp'])[:3, 3], [.397, 0, .8])
    repeated, _, _ = prepare_grasp_candidates(shifted, **kwargs)
    np.testing.assert_allclose(repeated[0]['T_world_tcp'], accepted[0]['T_world_tcp'])
    assert candidates[0]['T_world_tcp'] == pose.tolist()
    assert evidence['insertion_m'] == .003


def test_direction_constraints_are_optional_explicit_settings():
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    settings = PlanningSettings(grasp_maximum_tilt_deg=45, grasp_require_away_from_robot=True)
    _, accepted, evidence = prepare_grasp_candidates(
        [{'index': 0, 'T_world_tcp': pose.tolist()}],
        robot_base_world_m=[0, 0, 0], target_reference_world_m=[.4, 0, .8],
        planning_settings=settings.model_dump())
    assert not accepted
    assert evidence['rejected_tilt_count'] == 1


@pytest.mark.parametrize('kind', ['ik', 'grasp'])
def test_public_request_adapter_preserves_small_clearance_and_soft_distance(kind):
    from test_geometry_service import request_payload

    from luna_mujoco.planning_contracts import PlanGraspBatchRequest, SolveIkBatchRequest

    settings = PlanningSettings(trajectory_collision_clearance_m=.004,
                                optimizer_collision_activation_distance_m=.008)
    request_type = SolveIkBatchRequest if kind == 'ik' else PlanGraspBatchRequest
    payload = request_payload(settings)
    payload.update(freq_hz=50, trajectory_collision_clearance_m=.004,
                   optimizer_collision_activation_distance_m=.008)
    request = request_type.from_mapping(payload)
    assert request.trajectory_collision_clearance_m == .004
    if kind == 'grasp':
        assert request.optimizer_collision_activation_distance_m == .008
def test_release_gap_is_independent_and_survives_validation():
    from luna_mujoco.planning_settings import PlanningSettings
    settings = PlanningSettings(trajectory_collision_clearance_m=.004, place_release_height_m=.02)
    assert settings.place_release_height_m == .02
    assert PlanningSettings.model_validate(settings.model_dump()).place_release_height_m == .02


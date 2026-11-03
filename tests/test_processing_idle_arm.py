"""The model chooses the idle arm; VLA only constrains its processing arm."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from luna_mujoco.vla_configuration import vla_unpack_configuration


def test_vla_does_not_require_the_occluding_right_arm_pose():
    config = vla_unpack_configuration()
    assert set(config['preparation_joints_deg']) == {'A'}
    required = config['input_region']['required_robot_configurations'][0]
    assert required['scope'] == 'specified_arms'
    assert set(required['joints_deg']) == set(required['gripper_positions_rad']) == {'A'}


@pytest.mark.parametrize('selection', ['current', 'initial', 'observation'])
def test_model_selected_idle_pose_is_resolved_without_changing_vla_left_pose(selection):
    from luna_mujoco.observation_motion import observation_poses
    from luna_mujoco.planning_settings import PlanningSettings
    from luna_mujoco.robot_skills.processing import preparation_joint_goals

    snapshot = {'generation': 5, 'arms': {arm: {'joints_rad': [.1] * 7} for arm in ('A', 'B')}}
    context = {'planning_settings': PlanningSettings().model_dump(), 'initial_joint_reference': {
        'generation': 5, 'joints_rad': {'A': [.2] * 7, 'B': [.3] * 7}}}
    before = deepcopy(context)
    goals = preparation_joint_goals(context, snapshot, selection)
    expected = {'current': [.1] * 7, 'initial': [.3] * 7,
                'observation': observation_poses(context['planning_settings'])['B']}
    np.testing.assert_allclose(goals['B'], np.rad2deg(expected[selection]))
    assert goals['A'] == vla_unpack_configuration()['preparation_joints_deg']['A']
    assert context == before


def test_initial_idle_pose_rejects_stale_reference():
    from luna_mujoco.robot_skills.processing import preparation_joint_goals

    with pytest.raises(ValueError, match='initial_pose_reference_stale'):
        preparation_joint_goals({'initial_joint_reference': {'generation': 4}}, {'generation': 5}, 'initial')


def test_partial_configuration_never_claims_or_includes_unspecified_arm_geometry():
    from luna_mujoco.placement_constraints import configuration_spheres

    configuration = vla_unpack_configuration()['input_region']['required_robot_configurations'][0]
    _, links = configuration_spheres(configuration)
    assert 'gripL_Base' in links
    assert not any(link.startswith('gripR_') or link in {'Link1_R', 'Link7_R', 'TCP_Link_R'} for link in links)


def test_prepare_skill_exposes_pose_choice_to_responses():
    from luna_mujoco.runtime.controller import PrepareUnpackArguments

    schema = PrepareUnpackArguments.model_json_schema()
    assert set(schema['properties']['idle_arm_pose']['enum']) == {'current', 'initial', 'observation'}
    assert PrepareUnpackArguments().idle_arm_pose == 'current'
    with pytest.raises(ValueError):
        PrepareUnpackArguments(idle_arm_pose='legacy_fixed')


@pytest.mark.parametrize('right_arm_reached', [True, False])
def test_prepare_executes_selected_pose_and_requires_actual_right_arm_arrival(right_arm_reached):
    from luna_mujoco.observation_motion import observation_poses
    from luna_mujoco.planning_settings import PlanningSettings
    from luna_mujoco.robot_skills.processing import ProcessingSkills

    runner = ProcessingSkills()
    live = {'generation': 5, 'arms': {arm: {'joints_rad': [.1] * 7} for arm in ('A', 'B')}}
    runner.simulation = SimpleNamespace(robot_snapshot=lambda: deepcopy(live), execute_dual_trajectory=Mock())
    runner.state = {'generation': 5, 'target_id': 'input', 'facts': {}}
    runner._verified_processing_release = Mock(return_value={'release_feedback_id': 'release'})
    runner._vla_input_requirements = Mock()
    runner.measured_state = Mock(return_value=live)
    runner.motor = Mock()
    received = []

    def plan(observation, scene, goals, **kwargs):
        received.append(goals)
        return {'paths': {arm: [live['arms'][arm]['joints_rad'], np.deg2rad(goals[arm]).tolist()]
                          for arm in ('A', 'B')}, 'dt': .01}

    runner.algorithms = SimpleNamespace(plan_dual_vla_motion=plan, validate_dual_observation=Mock())

    async def call(function, *args, **kwargs):
        return function(*args, **kwargs)

    async def execute(context, stage, motor, function, paths, dt):
        for arm in ('A', 'B'):
            live['arms'][arm]['joints_rad'] = paths[arm][-1]
        if not right_arm_reached:
            live['arms']['B']['joints_rad'][0] += .1
        return {'success': True}

    runner.call_algorithm = call
    runner.execution_stage = execute
    context = {'planning_settings': PlanningSettings().model_dump(), 'scene': {}}
    if right_arm_reached:
        result = asyncio.run(runner._prepare_unpack(context, idle_arm_pose='observation'))
        assert result['success'] and context['vla_pose_execution']['idle_arm_pose'] == 'observation'
    else:
        with pytest.raises(ValueError, match='vla_preparation_goal_unconfirmed:B'):
            asyncio.run(runner._prepare_unpack(context, idle_arm_pose='observation'))
        assert 'vla_pose_execution' not in context
    np.testing.assert_allclose(received[0]['B'], np.rad2deg(observation_poses(context['planning_settings'])['B']))
    runner.algorithms.validate_dual_observation.assert_called_once()

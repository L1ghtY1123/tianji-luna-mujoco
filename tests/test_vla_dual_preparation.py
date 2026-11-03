"""Both arms must reach the VLA capability's calibrated start configuration."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from luna_mujoco.vla_configuration import vla_unpack_configuration


def test_vla_requires_both_configured_preparation_arms():
    config = vla_unpack_configuration()
    assert config['preparation_joints_deg'] == {
        'A': [-86., 76., 88., -90., 2., 14., -24.],
        'B': [106., 61., -113., -78., 27., 36., 37.]}
    required = config['input_region']['required_robot_configurations'][0]
    assert set(required['joints_deg']) == set(required['gripper_positions_rad']) == {'A', 'B'}


def test_vla_input_geometry_includes_both_required_arms():
    from luna_mujoco.placement_constraints import configuration_spheres

    configuration = vla_unpack_configuration()['input_region']['required_robot_configurations'][0]
    _, links = configuration_spheres(configuration)
    assert 'gripL_Base' in links and 'gripR_Base' in links


def test_prepare_does_not_allow_an_idle_arm_to_skip_vla_start_pose():
    from luna_mujoco.runtime.controller import PrepareUnpackArguments

    assert 'idle_arm_pose' not in PrepareUnpackArguments.model_json_schema()['properties']
    with pytest.raises(ValueError):
        PrepareUnpackArguments(idle_arm_pose='current')


@pytest.mark.parametrize('right_arm_reached', [True, False])
def test_prepare_executes_selected_pose_and_requires_actual_right_arm_arrival(right_arm_reached):
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
        result = asyncio.run(runner._prepare_unpack(context))
        assert result['success'] and set(context['vla_pose_execution']['goal_joints_deg']) == {'A', 'B'}
    else:
        with pytest.raises(ValueError, match='vla_preparation_goal_unconfirmed:B'):
            asyncio.run(runner._prepare_unpack(context))
        assert 'vla_pose_execution' not in context
    np.testing.assert_allclose(received[0]['B'], vla_unpack_configuration()['preparation_joints_deg']['B'])
    runner.algorithms.validate_dual_observation.assert_called_once()

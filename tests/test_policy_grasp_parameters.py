from copy import deepcopy

import pytest

from luna_mujoco.planning_settings import PlanningSettings


def test_model_grasp_parameters_are_scoped_and_keep_external_constraints():
    from luna_mujoco.runtime.parameters import GraspArguments, apply_grasp_parameters

    base = PlanningSettings().model_dump()
    original = deepcopy(base)
    context = {'planning_settings': base}
    choice = GraspArguments(parameters={'grasp_insertion_m': .01}, reason='Visible fingers should engage deeper')
    evidence = apply_grasp_parameters(context, choice)
    assert context['planning_settings']['grasp_insertion_m'] == .01
    assert context['planning_settings']['trajectory_collision_clearance_m'] == .005
    assert base == original
    assert evidence['source'] == 'responses_skill_arguments'
    apply_grasp_parameters(context, GraspArguments())
    assert context['planning_settings'] == original


@pytest.mark.parametrize('parameters', [
    {'trajectory_collision_clearance_m': 0.}, {'grasp_insertion_m': .2},
    {'grasp_maximum_tilt_deg': 181.}, {'grasp_require_away_from_robot': 'yes'},
])
def test_model_cannot_override_external_constraints_or_invalid_values(parameters):
    from luna_mujoco.runtime.parameters import GraspArguments

    with pytest.raises(ValueError):
        GraspArguments(parameters=parameters, reason='test')


def test_model_must_explain_parameter_override():
    from luna_mujoco.runtime.parameters import GraspArguments

    with pytest.raises(ValueError, match='reason'):
        GraspArguments(parameters={'grasp_insertion_m': .01})


def test_target_switch_does_not_promote_previous_model_override_to_ui_default(tmp_path):
    from policy_processing_fixture import runner_and_context

    from luna_mujoco.runtime.parameters import GraspArguments, apply_grasp_parameters

    runner, context, _ = runner_and_context(tmp_path)
    configured = deepcopy(context['planning_settings'])
    apply_grasp_parameters(context, GraspArguments(parameters={'grasp_insertion_m': .01}, reason='deeper engagement'))
    runner.reset_transfer(context)
    apply_grasp_parameters(context, GraspArguments())
    assert context['planning_settings']['grasp_insertion_m'] == configured['grasp_insertion_m']


async def test_policy_captures_are_not_capped_by_target_count(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.runtime.controller import PolicyController

    robot = DemoRobot()
    runner = PolicyController(robot, DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.state['generation'] = robot.robot_snapshot()['generation']
    context = {'normal_capture_count': 10000, 'recovery_capture_count': 10000}
    for recovery in (False, True):
        result = await runner.capture_observation(context, recovery=recovery)
        assert result['observation_id']
    assert runner.task_budgets(context)['normal'] == {'used': 10001, 'limit': 0}
    assert runner.task_budgets(context)['recovery'] == {'used': 10001, 'limit': 0}

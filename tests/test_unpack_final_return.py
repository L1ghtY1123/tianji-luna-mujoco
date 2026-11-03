"""The graph owns final return; completion independently checks actual feedback."""
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest
from test_agent_model_layer import task_data
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.agent.schema import TaskSpec
from luna_mujoco.runtime.controller import PolicyController
from luna_mujoco.runtime.execution_state import UnhandledSkillResult
from luna_mujoco.runtime.inventory import PolicyTaskProgress


def controller(tmp_path, *, complete=True, at_observation=False):
    robot = DemoRobot()
    value = PolicyController(robot, DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    value.task_inventory = {'selected_ids': ['pink']}
    value.task_spec = TaskSpec.model_validate(task_data() | {'scope': 'ids',
        'completion_conditions': ['All delivered products in the destination']})
    value.task_progress = PolicyTaskProgress('Deliver')
    value.task_progress.records = {'pink': {'id': 'pink', 'status': 'completed' if complete else 'observed',
        'current_region': 'tray' if complete else 'bench', 'confirmed_observation_id': 'delivery'}}
    value.task_progress.configure_task(value.task_spec, ['pink'])
    value.state['status'] = 'running'
    value.state['facts'].update(scene_available=True, grasp_held=False, observation_pose_ready=True,
                               observation_motion_scene_ready=True)
    feedback = {'arms': {arm: {'joints_rad': [0. if at_observation else .2] * 7} for arm in ('A', 'B')}}
    robot.robot_snapshot = Mock(side_effect=lambda: deepcopy(feedback))
    value.capture_observation = AsyncMock(return_value={'observation_id': 'completion'})
    value.record_observation = Mock()
    value._judge = AsyncMock(return_value={'judgments': {'goal_satisfied': {'status': 'yes'}}})
    context = {'planning_settings': TEST_SETTINGS.model_dump(mode='json'),
               'pending_transfer': {'target_id': 'black', 'verified': complete}}
    return value, context, feedback


async def explicit_return(value, context):
    async def invoke(name, arguments):
        return await value.registry.execute(name, value.state['facts'], context, arguments)

    from graph_contract_fixtures import linear_graph
    from luna_mujoco.runtime.native_graph import NativeGraphExecutor

    return await NativeGraphExecutor(invoke, target_ids=[], bindings={}).run(linear_graph([
        {'type': 'skill', 'id': 'final_return', 'skill': 'move_to_observe', 'arguments': {}}]))


async def test_last_delivery_graph_explicit_return_then_completion(tmp_path):
    value, context, feedback = controller(tmp_path)

    async def move(ctx):
        for arm in ('A', 'B'):
            feedback['arms'][arm]['joints_rad'] = [0.] * 7
        return {'success': True}

    value.move_to_observe = AsyncMock(side_effect=move)
    await explicit_return(value, context)
    await value._complete(context)
    value.move_to_observe.assert_awaited_once_with(context)
    assert value.state['status'] == 'succeeded'


async def test_ready_feedback_completion_does_not_insert_return(tmp_path):
    value, context, _ = controller(tmp_path, at_observation=True)
    value.move_to_observe = AsyncMock()
    await value._complete(context)
    value.move_to_observe.assert_not_awaited()
    assert value.state['status'] == 'succeeded'


async def test_completion_does_not_impose_unrequested_return(tmp_path):
    value, context, _ = controller(tmp_path)
    value.move_to_observe = AsyncMock()
    await value._complete(context)
    value.move_to_observe.assert_not_awaited()
    assert value.state['status'] == 'succeeded'


async def test_requested_completion_pose_rejects_missing_move_despite_visual_yes(tmp_path):
    value, context, _ = controller(tmp_path)
    value.task_spec = value.task_spec.model_copy(update={'completion_pose': 'observation'})
    with pytest.raises(ValueError, match='completion_pose_unconfirmed'):
        await value._complete(context)
    assert value.state['status'] != 'succeeded'


async def test_requested_completion_pose_accepts_measured_arrival_without_inserted_motion(tmp_path):
    value, context, _ = controller(tmp_path, at_observation=True)
    value.task_spec = value.task_spec.model_copy(update={'completion_pose': 'observation'})
    value.move_to_observe = AsyncMock()
    await value._complete(context)
    assert value.state['status'] == 'succeeded'
    value.move_to_observe.assert_not_awaited()


async def test_successful_motion_receipt_without_joint_feedback_cannot_complete(tmp_path):
    value, context, _ = controller(tmp_path)
    value.move_to_observe = AsyncMock(return_value={'success': True})
    with pytest.raises(ValueError, match='mapping_requires_observation_pose'):
        await explicit_return(value, context)
    value._judge.assert_not_awaited()


async def test_failed_explicit_return_is_a_graph_failure(tmp_path):
    value, context, _ = controller(tmp_path)
    value.move_to_observe = AsyncMock(return_value={'success': False, 'reason': 'return planning failed'})
    with pytest.raises(UnhandledSkillResult, match='return planning failed'):
        await explicit_return(value, context)
    value._judge.assert_not_awaited()
    assert value.state['status'] != 'succeeded'


async def test_pending_delivery_cannot_complete_even_at_observation_pose(tmp_path):
    value, context, _ = controller(tmp_path, complete=False, at_observation=True)
    with pytest.raises(ValueError, match='policy_finished_with_pending_targets'):
        await value._complete(context)
    value.capture_observation.assert_not_awaited()

from copy import deepcopy
from unittest.mock import AsyncMock

import numpy as np
import pytest

from luna_mujoco.robot_skills.return_pose import (
    initial_joint_reference,
    initial_pose_verdict,
    return_to_initial_pose,
)


def snapshot():
    return {'generation': 3, 'time': 1., 'arms': {
        'A': {'joints_rad': [.1] * 7}, 'B': {'joints_rad': [-.2] * 7}}}


def test_initial_pose_freezes_both_arms_and_rejects_stale_feedback():
    before = snapshot()
    reference = initial_joint_reference(before)
    before['arms']['A']['joints_rad'][0] = 1.
    assert reference['joints_rad']['A'][0] == .1
    assert initial_pose_verdict(reference, snapshot())['success']
    assert not initial_pose_verdict(reference, before)['success']
    assert not initial_pose_verdict(reference, snapshot() | {'generation': 4})['success']


async def test_return_uses_saved_joints_and_requires_actual_arrival():
    from types import SimpleNamespace

    original = snapshot()
    feedback = deepcopy(original)
    feedback['arms']['A']['joints_rad'][2] += .2
    runner = SimpleNamespace(simulation=SimpleNamespace(robot_snapshot=lambda: feedback),
                             move_to_observe=AsyncMock(return_value={'movements': {'success': True}}))
    context = {'initial_joint_reference': initial_joint_reference(original)}
    with pytest.raises(ValueError, match='initial_pose_unconfirmed'):
        await return_to_initial_pose(runner, context)
    np.testing.assert_array_equal(runner.move_to_observe.call_args.kwargs['joint_goals']['B'], [-.2] * 7)
    feedback['arms']['A']['joints_rad'] = [.1] * 7
    runner.move_to_observe.reset_mock()
    result = await return_to_initial_pose(runner, context)
    assert result['success'] and result['already_at_initial_pose']
    runner.move_to_observe.assert_not_awaited()


@pytest.mark.parametrize('displaced', [False, True])
async def test_public_return_only_requires_a_fresh_map_when_motion_is_needed(displaced):
    from types import SimpleNamespace

    from luna_mujoco.robot_skills.operations import RobotOperations
    reference = initial_joint_reference(snapshot())
    feedback = snapshot()
    if displaced:
        feedback['arms']['B']['joints_rad'][0] += .2
    runner = SimpleNamespace(simulation=SimpleNamespace(robot_snapshot=lambda: feedback),
        holding_state=None, move_to_observe=AsyncMock())
    context = {'initial_joint_reference': reference, 'scene_refresh_required': True}
    if displaced:
        with pytest.raises(ValueError, match='observation_motion_requires_current_map'):
            await RobotOperations.execute(runner, 'return_to_initial_pose', context)
    else:
        result = await RobotOperations.execute(runner, 'return_to_initial_pose', context)
        assert result['success'] and result['already_at_initial_pose']
        assert context['scene_refresh_required']  # No fabricated map refresh.
    runner.move_to_observe.assert_not_awaited()


async def test_return_noop_cannot_hide_unverified_release():
    from types import SimpleNamespace
    runner = SimpleNamespace(simulation=SimpleNamespace(robot_snapshot=snapshot))
    with pytest.raises(ValueError, match='return_requires_verified_release'):
        await return_to_initial_pose(runner, {'initial_joint_reference': initial_joint_reference(snapshot()),
                                             'pending_transfer': {'verified': False}})


async def test_completion_requires_initial_feedback_without_inserting_motion(tmp_path):
    from test_unpack_final_return import controller

    value, context, feedback = controller(tmp_path)
    context['initial_joint_reference'] = initial_joint_reference(feedback | {'generation': 1})
    original_reference = deepcopy(context['initial_joint_reference'])
    value.reset_transfer(context)
    assert context['initial_joint_reference'] == original_reference
    value.task_spec = value.task_spec.model_copy(update={'completion_pose': 'initial'})
    feedback['generation'] = 1
    feedback['arms']['A']['joints_rad'][0] += .2
    value.move_to_observe = AsyncMock()
    with pytest.raises(ValueError, match='completion_pose_unconfirmed'):
        await value._complete(context)
    value.move_to_observe.assert_not_awaited()
    feedback['arms']['A']['joints_rad'][0] -= .2
    await value._complete(context)
    assert value.state['status'] == 'succeeded'
    assert value.state['facts']['at_initial_pose'] is True


@pytest.mark.parametrize('held_in', ['context', 'runner'])
async def test_already_at_initial_pose_cannot_hide_provisional_holding(held_in):
    from types import SimpleNamespace

    runner = SimpleNamespace(simulation=SimpleNamespace(robot_snapshot=snapshot),
        holding_state={'state': 'provisional'} if held_in == 'runner' else None,
        move_to_observe=AsyncMock())
    context = {'initial_joint_reference': initial_joint_reference(snapshot())}
    if held_in == 'context':
        context['holding'] = {'state': 'provisional'}
    with pytest.raises(ValueError, match='requires_empty_hands'):
        await return_to_initial_pose(runner, context)
    runner.move_to_observe.assert_not_awaited()



"""Completing the last delivery still requires a successful collision-planned return."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from luna_mujoco.demo_runner import DemoRunner


def runner(*, ready=False, complete=True, actual_ready=None):
    task = {'transport_ids': ['pink'], 'process_records': {
        'process': {'input_entity_id': 'pink', 'output_entity_id': 'black'}},
        'item_lifecycle': {'black': {'phase': 'completed' if complete else 'unpacked_confirmed',
                                   'to_table_arm': 'A', 'to_right_bin_arm': 'B'}}}
    actual_ready = ready if actual_ready is None else actual_ready
    check = Mock() if actual_ready else Mock(side_effect=[ValueError('mapping_requires_observation_pose'), None])
    return SimpleNamespace(demo_task=task, state={'facts': {'observation_pose_ready': ready}},
                           require_observation_pose=check, _step=AsyncMock())


@pytest.mark.asyncio
async def test_last_delivery_returns_even_without_next_pink():
    value, context = runner(), {'scene': {'scene_token': 'latest'}}
    await DemoRunner.finish_unpack_at_initial_pose(value, context)
    value._step.assert_awaited_once_with('move_to_observe', context)


@pytest.mark.asyncio
async def test_existing_successful_return_is_not_repeated():
    value = runner(ready=True)
    await DemoRunner.finish_unpack_at_initial_pose(value, {})
    value._step.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_ready_flag_cannot_skip_actual_final_return():
    value, context = runner(ready=True, actual_ready=False), {}
    await DemoRunner.finish_unpack_at_initial_pose(value, context)
    value._step.assert_awaited_once_with('move_to_observe', context)
    assert context['use_retained_scene_for_return'] is True
    assert value.require_observation_pose.call_count == 2


@pytest.mark.asyncio
async def test_return_completion_requires_actual_joint_feedback():
    value = runner()
    value.require_observation_pose.side_effect = ValueError('mapping_requires_observation_pose')
    with pytest.raises(ValueError, match='mapping_requires_observation_pose'):
        await DemoRunner.finish_unpack_at_initial_pose(value, {})


@pytest.mark.asyncio
async def test_failed_return_cannot_be_reported_as_task_success():
    value = runner()
    value._step.side_effect = RuntimeError('return planning failed')
    with pytest.raises(RuntimeError, match='return planning failed'):
        await DemoRunner.finish_unpack_at_initial_pose(value, {})


@pytest.mark.asyncio
async def test_pending_delivery_cannot_enter_final_return():
    value = runner(complete=False)
    with pytest.raises(ValueError, match='pending_targets'):
        await DemoRunner.finish_unpack_at_initial_pose(value, {})
    value._step.assert_not_awaited()

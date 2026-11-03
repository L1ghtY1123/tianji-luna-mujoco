"""Explicit initial observation motion must use current-map safety, not release history."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_observation_motion import ObservationAlgorithms, context

from luna_mujoco.robot_skills.operations import RobotOperations
from luna_mujoco.runtime.controller import PolicyController


def controller():
    runner = object.__new__(PolicyController)
    runner.registry = SimpleNamespace(handlers={'move_to_observe': None})
    runner.move_to_observe = AsyncMock(return_value={'status': 'success'})
    runner.require_observation_pose = Mock()
    return runner


@pytest.mark.parametrize('stale_flag', [False, True])
async def test_initial_motion_dispatch_does_not_claim_retained_return(stale_flag):
    runner = controller()
    state = {'scene': {'scene_token': 'partial-map'}, 'use_retained_scene_for_return': stale_flag}
    await runner.execute('move_to_observe', state)
    runner.move_to_observe.assert_awaited_once_with(state)
    assert not state.get('use_retained_scene_for_return')
    assert 'pending_transfer' not in state
    runner.require_observation_pose.assert_called_once_with(state)


@pytest.mark.parametrize('transfer', [{}, {'verified': False}, {'verified': 'yes'}])
@pytest.mark.parametrize('post_effect_ready', [False, True])
async def test_unverified_transfer_never_authorizes_observation_motion(transfer, post_effect_ready):
    runner = controller()
    state = {'pending_transfer': transfer, 'unpack_return_ready': post_effect_ready}
    with pytest.raises(ValueError, match='return_requires_verified_release'):
        await runner.execute('move_to_observe', state)
    runner.move_to_observe.assert_not_awaited()


async def test_processing_effect_without_measured_geometry_still_blocks_motion():
    runner = controller()
    state = {'processing_action': {'process_id': 'effect'}, 'pending_transfer': {'verified': True}, 'scene_refresh_required': True}
    with pytest.raises(ValueError, match='observation_motion_requires_current_map'):
        await runner.execute('move_to_observe', state)
    runner.move_to_observe.assert_not_awaited()


@pytest.mark.parametrize('processing', [False, True])
async def test_verified_release_keeps_retained_geometry_motion(processing):
    runner = controller()
    state = {'pending_transfer': {'verified': True}}
    if processing:
        state.update(processing_action={'process_id': 'effect'}, scene_refresh_required=False)
    await runner.execute('move_to_observe', state)
    runner.move_to_observe.assert_awaited_once_with(state)
    assert state['use_retained_scene_for_return'] is True
    assert not state.get('scene_refresh_required')


@pytest.mark.parametrize('failure', ['dirty', 'stale', 'selected', 'holding', 'coverage'])
async def test_initial_dispatch_preserves_runner_map_empty_hand_and_coverage_checks(tmp_path, failure):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    await runner.build_scene(state, 'observation')
    if failure == 'dirty':
        state['scene_refresh_required'] = True
    elif failure == 'stale':
        state['scene']['observation_id'] = 'old-capture'
    elif failure == 'selected':
        state['scene']['exclusion_state'] = 'selected'
    elif failure == 'holding':
        state['holding'] = {'owner_id': 'held'}
    checks = []
    def reject_coverage(observation, scene, goals, **kwargs):
        checks.append(scene)
        assert scene['return_after_place'] is False
        raise ValueError('motion_coverage_unknown')
    algorithms.plan_dual_observation_motion = reject_coverage
    message = {'holding': 'observation_motion_requires_empty_hands',
               'coverage': 'motion_coverage_unknown'}.get(failure, 'observation_motion_requires_current_map')
    with pytest.raises(ValueError, match=message):
        await RobotOperations.execute(runner, 'move_to_observe', state)
    assert len(checks) == int(failure == 'coverage')
    assert algorithms.moves == []

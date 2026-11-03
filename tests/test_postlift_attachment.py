from unittest.mock import AsyncMock

import pytest
from test_current_grasp_attachment import fixture


@pytest.mark.parametrize('failure', [None, 'collision', 'motion', 'lost_hold'])
async def test_first_lift_precedes_verification_and_attachment(tmp_path, failure):
    runner, context, current, prepared, _, _ = fixture(tmp_path)
    runner.run_id = 'owned-session'
    runner.state['current_step'] = {'skill': 'execute_grasp'}
    q = current['arms']['A']['joints_rad']
    context['plan'] = {'T_world_tcp': current['arms']['A']['tcp'], 'phases': {
        name: {'positions_rad': [q, q], 'dt': .02} for name in ('approach', 'grasp', 'lift')}}
    runner.verify_scene = AsyncMock()
    runner.gripper = AsyncMock()
    calls = []

    async def stage(context, name, function, *args, **kwargs):
        calls.append(name)
        if name in ('capture_reference', 'validate_lift', 'verify_holding_after_lift', 'prepare_attachment'):
            return await function(*args, **kwargs)
        if name == 'lift':
            assert not prepared and not context['holding']['attached']
            if failure == 'motion':
                raise ValueError('lift_motion_failed')
            if failure == 'lost_hold':
                current['arms']['A'].update(gripper_closed_fraction=1., gripper_position_rad=.8,
                    gripper_position_samples=[[t, .8] for t in (0., .05, .1, .15, .2, .25)])
        return {}

    async def validate(context, phase, command, binding=None, **kwargs):
        if phase in ('lift', 'grasp_lift'):
            assert phase == 'grasp_lift'
            assert not prepared and not context['holding']['attached']
            assert binding is None
            calls.append('check_detached_lift')
            if failure == 'collision':
                raise ValueError('lift_collision')
        return {'collision_free': True, 'attached': binding is not None}

    runner.execution_stage = stage
    runner.validate_motion = validate
    runner.simulation.execute_trajectory = lambda *a: None
    runner.simulation.dwell = lambda *a: None
    if failure:
        with pytest.raises(ValueError, match={
            'collision': 'lift_collision', 'motion': 'lift_motion_failed',
            'lost_hold': 'grasp_not_held_after_lift'}[failure]):
            await runner.execute_grasp(context)
        assert not prepared and not runner.holding_state['attached']
        assert runner.holding_state['state'] == 'provisional'
        if failure == 'collision':
            assert 'lift' not in calls
    else:
        result = await runner.execute_grasp(context)
        assert calls.index('check_detached_lift') < calls.index('lift')
        assert calls.index('lift') < calls.index('verify_holding_after_lift') < calls.index('prepare_attachment')
        assert prepared and runner.holding_state['attached']
        assert result['lift_carried_collision_checked'] is False
        assert result['lift_closed_gripper_collision_checked'] is True

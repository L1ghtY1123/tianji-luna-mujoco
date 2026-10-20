"""Planning evidence cannot hide a later loss of grip or changed peer fingers."""

import copy

import pytest
from test_review_regressions import verified_visual_holding
from test_rgbd_workflow import Choices


@pytest.mark.parametrize('phase', ['approach', 'place'])
@pytest.mark.parametrize('status', ['empty', 'unknown'])
async def test_fresh_nonheld_feedback_blocks_each_carried_phase(tmp_path, phase, status):
    runner, algorithms, context = await verified_visual_holding(tmp_path)
    await runner.execute('plan_place', context, Choices('place'))
    original = runner.simulation.robot_snapshot
    checks = []
    algorithms.validate_execution = lambda *args, **kwargs: checks.append(args[3])

    def changed_feedback():
        feedback = copy.deepcopy(original())
        active = feedback['arms']['A']
        if status == 'empty':
            active.update(gripper_closed_fraction=1., gripper_position_rad=.8,
                          gripper_position_samples=[[t, .8] for t in (0., .05, .1, .15, .2, .25)])
        else:
            active.pop('gripper_position_samples')
        return feedback

    runner.simulation.robot_snapshot = changed_feedback
    motors = list(runner.simulation.motor_calls)
    command = context['place_plan']['phases'][phase]
    with pytest.raises(ValueError, match='holding_feedback_' + status):
        await runner.validate_motion(context, phase, command, context['place_plan']['carrying_binding'])
    assert checks == [] and runner.simulation.motor_calls == motors
    assert not context['holding']['verified']
    assert context['holding']['attached']  # No detach or release was commanded.
    assert context['holding']['last_position_evidence']['status'] == status
    assert not runner.state['facts'].get('released')


async def test_peer_gripper_feedback_is_not_replaced_by_grasp_time_angle(tmp_path):
    runner, algorithms, context = await verified_visual_holding(tmp_path)
    await runner.execute('plan_place', context, Choices('place'))
    original = runner.simulation.robot_snapshot
    observed = []

    def changed_feedback():
        feedback = copy.deepcopy(original())
        feedback['arms']['B']['gripper_position_rad'] = .3
        return feedback

    runner.simulation.robot_snapshot = changed_feedback
    algorithms.validate_execution = lambda observation, *args, **kwargs: observed.append(observation)
    await runner.validate_motion(context, 'approach', context['place_plan']['phases']['approach'],
                                 context['place_plan']['carrying_binding'])
    assert observed[0]['arms']['B']['gripper_position_rad'] == .3


async def test_wrong_attached_identity_rejected_before_command_validation(tmp_path):
    runner, algorithms, context = await verified_visual_holding(tmp_path)
    await runner.execute('plan_place', context, Choices('place'))
    context['place_plan']['carrying_binding']['instance_id'] = 'different-object'
    algorithms.validate_execution = lambda *args, **kwargs: pytest.fail('invalid identity reached validator')
    with pytest.raises(ValueError, match='holding_binding_mismatch'):
        await runner.execute_place(context)


async def test_loss_after_approach_stops_before_place_and_keeps_attachment(tmp_path):
    runner, _, context = await verified_visual_holding(tmp_path)
    await runner.execute('plan_place', context, Choices('place'))
    execute = runner.simulation.execute_trajectory

    def lose_object(*args):
        result = execute(*args)
        runner.simulation.closed = False
        return result

    runner.simulation.execute_trajectory = lose_object
    with pytest.raises(ValueError, match='holding_feedback_unknown'):
        await runner.execute_place(context)
    assert runner.simulation.stage == 3
    assert context['holding']['attached'] and not context['holding']['verified']


async def test_detach_failure_keeps_open_release_and_pending_attachment(tmp_path):
    runner, algorithms, context = await verified_visual_holding(tmp_path)
    await runner.execute('plan_place', context, Choices('place'))

    def fail_detach():
        assert not runner.simulation.closed
        raise RuntimeError('detach_service_failed')

    algorithms.detach_carrying = fail_detach
    with pytest.raises(RuntimeError, match='detach_service_failed'):
        await runner.execute_place(context)
    assert runner.state['facts']['released'] and not runner.state['facts']['grasp_held']
    assert runner.holding_state['attached'] and runner.holding_state['attachment_detach_pending']
    assert runner.simulation.stage == 4


async def test_failed_open_never_detaches_or_retreats(tmp_path):
    runner, algorithms, context = await verified_visual_holding(tmp_path)
    await runner.execute('plan_place', context, Choices('place'))
    runner.simulation.set_gripper = lambda *args: {'completed': True}
    algorithms.detach_carrying = lambda: pytest.fail('unconfirmed release must not detach')
    with pytest.raises(ValueError, match='release_not_confirmed_by_gripper'):
        await runner.execute_place(context)
    assert runner.simulation.stage == 4 and runner.simulation.closed
    assert runner.holding_state['attached'] and not runner.state['facts'].get('released')

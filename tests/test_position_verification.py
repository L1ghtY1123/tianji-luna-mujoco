import pytest

from luna_mujoco import verification


def feedback(position=.48, samples=None, commanded=True):
    return {'gripper_position_rad': position, 'gripper_limits_rad': [0., .8],
            'gripper_closed_command': commanded,
            'gripper_position_samples': samples if samples is not None else [[t, position] for t in (0., .05, .1, .15, .2, .25)]}


@pytest.mark.parametrize('position, status', [(.48, 'held'), (.8, 'empty'), (.795, 'empty'), (.77, 'held')])
def test_position_verdict_uses_empty_close_baseline(position, status):
    assert verification.verify_gripper_position(feedback(position))['status'] == status


@pytest.mark.parametrize('value', [feedback(samples=[[0., .5], [.3, .4]]), feedback(commanded=False),
                                   feedback(samples=[]), feedback(samples=[[0., .4], [.1, .4]])])
def test_unstable_missing_or_not_fully_commanded_is_unknown(value):
    assert verification.verify_gripper_position(value)['status'] == 'unknown'


def test_occluded_vision_cannot_cancel_position_held():
    result = verification.combine_holding_evidence({'status': 'held'}, {'status': 'unknown'})
    assert result['success'] and result['status'] == 'held'


def test_conflicting_evidence_is_unknown_and_visual_can_support_uncertain_position():
    assert verification.combine_holding_evidence({'status': 'held'}, {'status': 'empty'})['status'] == 'unknown'
    assert verification.combine_holding_evidence({'status': 'unknown'}, {'status': 'held'})['status'] == 'held'








async def test_reverification_unknown_clears_prior_verified_fact(tmp_path):
    from test_current_grasp_attachment import fixture
    runner, context, current, _, _, _ = fixture(tmp_path)
    await runner.execute('verify_grasp', context)
    current['arms']['A']['gripper_position_samples'] = []
    current['objects'] = []
    result = await runner.execute('verify_grasp', context)
    assert result['status'] == 'unknown'
    assert not context['holding']['verified']
    assert not runner.state['facts']['grasp_held']


def test_real_empty_gripper_reaches_empty_baseline():
    from luna_mujoco.simulation import Simulation
    simulation = Simulation()
    simulation.start()
    try:
        generation = simulation.robot_snapshot()['generation']
        simulation.set_gripper('A', True, generation)
        simulation.dwell(.3, generation)
        result = verification.verify_gripper_position(simulation.robot_snapshot()['arms']['A'])
        assert result['status'] == 'empty', result
    finally:
        simulation.close()




def test_visual_conflict_cannot_be_erased_by_outer_position_vote():
    visual = verification.combine_holding_evidence({'status': 'held'}, {'status': 'empty'})
    result = verification.combine_holding_evidence({'status': 'held'}, visual)
    assert result['status'] == 'unknown'
    assert result['conflicting_evidence']

import pytest


@pytest.mark.parametrize('next_status', ['observed', 'pending', 'completed', 'deferred'])
async def test_release_feedback_never_selects_next_observation_or_map(tmp_path, next_status):
    from test_release_verification import release_case
    runner, context = release_case(tmp_path)
    runner.task_progress.records['other'] = {'id': 'other', 'status': next_status}
    async def forbidden(*args, **kwargs):
        pytest.fail('release feedback selected a follow-up action')
    runner.observe = runner.build_scene = forbidden
    result = await runner.verify_place(context)
    assert result['success']
    assert runner.task_progress.completed_ids == []






async def test_repeated_release_feedback_cannot_complete_a_transfer(tmp_path):
    from test_release_verification import release_case
    runner, state = release_case(tmp_path)
    for _ in range(2):
        assert (await runner.verify_place(state))['success']
        runner.simulation.state['time'] += 1
    assert runner.task_progress.completed_ids == []
    assert runner.task_progress.records['target'].get('transfers', []) == []

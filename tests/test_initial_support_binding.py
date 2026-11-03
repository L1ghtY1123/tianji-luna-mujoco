"""A13: missing initial_parent_id must trigger a measured support judgment."""
from unittest.mock import AsyncMock

import pytest
from test_observation_reuse import prepared
from test_support_context import example
from test_unpack_final_return import controller


@pytest.mark.parametrize('answer', ['table', 'bin', 'unknown'])
async def test_absent_initial_parent_is_resolved_without_guessing_from_task_or_cad(tmp_path, answer):
    runner, context, frame = await prepared(tmp_path)
    template, _, _ = controller(tmp_path)
    runner.task_spec, runner.task_progress = template.task_spec, template.task_progress
    runner.task_inventory = template.task_inventory
    runner.task_progress.scene_entities = {
        'pink': {'id': 'pink', 'kind': 'object', 'initial_parent_id': None},
        'table': {'id': 'table', 'kind': 'table'}, 'bin': {'id': 'bin', 'kind': 'bin'}}
    context['visual_inventory_reference'] = {}
    context['observation'] = example()[1] | {'observation_id': frame['observation_id'],
        'generation': frame.get('generation'), 'inventory_state': runner.task_progress.inventory_state()}
    context['observation']['objects'][0]['id'] = 'pink'
    decisions = {}
    for index, support in enumerate(['table', 'bin']):
        status = 'unknown' if answer == 'unknown' else 'yes' if support == answer else 'no'
        decisions['support_' + str(index)] = {'status': status, 'probability': {'yes': .99, 'no': .01, 'unknown': .5}[status]}
    runner._judge = AsyncMock(return_value={'judgments': decisions,
        'evidence': {'request_hash': 'current-support', 'observation_id': frame['observation_id']}})
    if answer == 'unknown':
        with pytest.raises(ValueError, match='observed_support_unknown'):
            await runner._refresh_observed_support(context, 'pink')
        assert 'observed_support' not in runner.task_progress.scene_entities['pink']
    else:
        await runner._refresh_observed_support(context, 'pink')
        proof = runner.task_progress.scene_entities['pink']['observed_support']
        assert proof['support_entity_id'] == answer
        assert proof['observation_id'] == frame['observation_id']
        assert proof['request_hash'] == 'current-support'
        assert context['observation']['inventory_state']['entities']['pink']['observed_support'] == proof
    runner._judge.assert_awaited_once()
    assert runner._judge.await_args.kwargs['include_task'] is False
    assert context['observation_count'] == 1

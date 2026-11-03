"""Only exact repeated recovery evidence may become reversible references."""
import importlib
import json
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def api():
    return importlib.import_module('luna_mujoco.recovery_context_compaction')


def frozen_request():
    path = Path(__file__).parent / 'fixtures/black_b3_recovery_request.json'
    return json.loads(path.read_text(encoding='utf-8'))['request']


def test_frozen_b3_context_expands_to_identical_json_and_leaves_other_input_bytes():
    request = frozen_request()
    original = deepcopy(request)
    primary = json.loads(request['input'][0]['content'][0]['text'])
    context = primary['context']
    compacted = api().compact_recovery_context(context)
    expanded = api().expand_recovery_context(compacted)
    assert json.dumps(expanded, ensure_ascii=False) == json.dumps(context, ensure_ascii=False)
    assert len(json.dumps(compacted, ensure_ascii=False)) < .45 * len(json.dumps(context, ensure_ascii=False))
    assert compacted['verification_scope'] == 'graph_recovery'
    assert compacted['task']['instruction'] == context['task']['instruction']
    assert compacted['failed_result'] == context['failed_result']
    assert compacted['facts'] == context['facts']
    primary['context'] = compacted
    request['input'][0]['content'][0]['text'] = json.dumps(primary, ensure_ascii=False)
    assert request['questions'] == original['questions']
    assert request['input'][0]['content'][1:] == original['input'][0]['content'][1:]


def test_changed_values_and_numeric_types_are_not_merged():
    common = {'instruction': '必须保留中文、自然语言与永久 ID。' * 45, 'value': 1}
    context = {'verification_scope': 'graph_recovery', 'first': common,
               'same': deepcopy(common), 'changed': {**common, 'value': 2},
               'different_type': {**common, 'value': 1.0}}
    result = api().compact_recovery_context(context)
    assert '$context_ref' in result['same']
    assert result['changed'] == context['changed']
    assert result['different_type'] == context['different_type']
    assert api().expand_recovery_context(result) == context


def test_reference_paths_escape_keys_and_validate_hashes():
    common = {'description': 'literal words ' * 100}
    context = {'verification_scope': 'graph_recovery_continuation', 'a/b~c': common, 'same': deepcopy(common)}
    result = api().compact_recovery_context(context)
    assert result['same']['$context_ref'] == '#/a~1b~0c'
    assert api().expand_recovery_context(result) == context
    result['a/b~c']['description'] = 'changed after compaction'
    with pytest.raises(ValueError, match='recovery_context'):
        api().expand_recovery_context(result)


def test_expanded_hash_detects_changed_unique_safety_fact():
    common = {'description': 'retained source evidence ' * 100}
    context = {'verification_scope': 'graph_recovery', 'first': common, 'same': common,
               'facts': {'released': False, 'attached': True}}
    compacted = api().compact_recovery_context(context)
    compacted['facts']['released'] = True
    with pytest.raises(ValueError, match='recovery_context_expanded_hash_mismatch'):
        api().expand_recovery_context(compacted)


def test_non_recovery_and_reference_shaped_user_data_are_unchanged():
    common = {'description': 'must remain unchanged ' * 100}
    for scope in ['task_completion', 'current_support', 'graph_recovery']:
        context = {'verification_scope': scope, 'first': common, 'same': common,
                   'literal': {'$context_ref': 'a user supplied dictionary, not our reference'}}
        assert api().compact_recovery_context(context) == context


@pytest.mark.parametrize('scope', ['graph_recovery', 'graph_recovery_continuation', 'task_completion'])
@pytest.mark.asyncio
async def test_controller_hook_preserves_capture_requests_reference_and_probability(scope):
    from luna_mujoco.runtime.controller import PolicyController

    requests = frozen_request()['questions']
    primary = json.loads(frozen_request()['input'][0]['content'][0]['text'])
    extra = primary['context']
    extra['verification_scope'] = scope
    before = deepcopy(extra)
    rgb = np.arange(18, dtype=np.uint8).reshape((2, 3, 3))
    capture = {'observation_id': 'current', 'views': {'3': {'rgb': rgb}}}
    reference = {'entities': {'permanent-id': {'description': '中文参考，不代表当前像素'}}}
    answer = {'judgments': {'recovery_capability': {'status': 'proposed', 'value': 'stop', 'confidence': .18}}}
    received = {}

    def judge(actual_capture, **kwargs):
        received.update(kwargs)
        assert actual_capture is capture
        return answer

    async def call_algorithm(function, *args, **kwargs):
        return function(*args, **kwargs)

    completed = threading.Event()
    controller = SimpleNamespace(observation_epoch=4,
        visual_audit_sink=lambda: (None, completed), judgments=SimpleNamespace(judge=judge),
        call_algorithm=call_algorithm)
    result = await PolicyController._judge(controller,
        {'latest_capture': capture, 'capture_epochs': {'current': 4}, 'visual_inventory_reference': reference},
        requests, extra=extra, include_task=False)
    assert result is answer and result['judgments']['recovery_capability']['confidence'] == .18
    assert received['requests'] is requests and received['reference'] is reference
    if scope.startswith('graph_recovery'):
        assert 'context_compaction' in received['context']
        assert api().expand_recovery_context(received['context']) == before
    else:
        assert received['context'] == before
    np.testing.assert_array_equal(rgb, np.arange(18, dtype=np.uint8).reshape((2, 3, 3)))
    assert extra == before and completed.is_set()

from copy import deepcopy

import numpy as np

from luna_mujoco.runtime.recovery_evidence import blocked_operations, operation_evidence, operation_signature


def state():
    return {'latest_capture': {'views': {'1': {'rgb': np.zeros((16, 16, 3), dtype=np.uint8)}}},
        'observation': {'objects': [{'id': 'target'}]}, 'scene': {'coverage': {'target': 'current'}}}


def test_identical_failed_observation_options_are_removed_before_model_selection():
    evidence = operation_evidence(state(), 'observe', 'identity_unknown', {'observed': False}, [])
    proposals = [{'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': mode}}
        for mode in ('fresh', 'reuse_current')]
    seen = {operation_signature(evidence, proposal) for proposal in proposals}
    assert {row['arguments']['capture_mode'] for row in blocked_operations(seen, evidence)} == {'fresh', 'reuse_current'}
    assert len(seen) == 2


def test_changed_evidence_restores_operation_choices():
    context = state()
    evidence = operation_evidence(context, 'observe', 'identity_unknown', {'observed': False}, [])
    seen = {operation_signature(evidence, {'action': 'retry'})}
    for changed in [dict(evidence, facts={'observed': True}), dict(evidence, node='other'),
                    dict(evidence, reason='other'), dict(evidence, completed_inputs=['target'])]:
        assert blocked_operations(seen, changed) == []
    updated = deepcopy(context)
    updated['latest_capture']['views']['1']['rgb'][:] = 255
    assert blocked_operations(seen, operation_evidence(updated, 'observe', 'identity_unknown', {'observed': False}, [])) == []


def test_expired_capture_cannot_remove_a_choice_before_recovery_acquires_current_pixels():
    context = state()
    old = operation_evidence(context, 'observe', 'identity_unknown', {'observed': False}, [])
    proposal = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': 'fresh'}}
    seen = {operation_signature(old, proposal)}
    assert blocked_operations(seen, old, capture_current=False) == []
    context['latest_capture']['views']['1']['rgb'][:] = 255
    current = operation_evidence(context, 'observe', 'identity_unknown', {'observed': False}, [])
    assert operation_signature(current, proposal) not in seen


def test_map_outputs_cannot_make_identical_build_inputs_new_evidence():
    context = state()
    context['observation']['observation_id'] = 'current'
    proposal = {'action': 'skill', 'skill': 'build_scene',
                'arguments': {'scope': 'global', 'exclusion_state': 'observation'}}
    before = operation_evidence(context, 'zone', 'support_missing', {'scene_ready': False}, [])
    seen = {operation_signature(before, proposal)}
    context['scene']['coverage'] = {'new_diagnostic': 'new_scene_token'}
    after = operation_evidence(context, 'zone', 'support_missing', {'scene_ready': True}, [])
    assert blocked_operations(seen, after)[0]['skill'] == 'build_scene'
    context['observation']['observation_id'] = 'new-measured-input'
    newer = operation_evidence(context, 'zone', 'support_missing', {'scene_ready': True}, [])
    assert blocked_operations(seen, newer) == []


def test_no_progress_options_are_unavailable_without_forcing_a_recovery_action():
    import asyncio

    from luna_mujoco.agent.recovery import RecoveryPlanner

    questions = []

    async def judge(requests, context):
        questions.extend(requests)
        assert requests[0].choices == ('stop', 'unknown')
        return {'judgments': {'recovery_capability': {'status': 'proposed', 'value': 'unknown'}}}

    catalog = {'observe_scene': {'recovery_retry_safe': True, 'requires': {}, 'arguments': {
        'type': 'object', 'properties': {'capture_mode': {'enum': ['fresh', 'reuse_current']}}}}}
    blocked = [{'skill': 'observe_scene', 'arguments': {'capture_mode': mode}}
        for mode in ('fresh', 'reuse_current')]
    blocked.append({'reason': 'identical_evidence_and_operation_no_progress'})
    result = asyncio.run(RecoveryPlanner(judge, catalog).propose(
        {'id': 'observe', 'type': 'skill', 'skill': 'observe_scene'}, {}, [], {},
        facts={}, inventory=[], context={'rejected_proposals': blocked}))
    assert result['reason'] == 'decisions_recovery_unknown'
    assert len(questions) == 1

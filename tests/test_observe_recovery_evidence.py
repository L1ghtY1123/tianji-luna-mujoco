"""Observation no-progress depends on sensing and identity inputs, not planning facts."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.runtime.recovery_evidence import blocked_operations, operation_evidence, operation_signature

FIXTURE = Path(__file__).parent / 'fixtures' / 'black_b5_observe_recovery.json'


def frozen_context():
    data = json.loads(FIXTURE.read_text(encoding='utf-8'))
    capture = {**data['capture'], 'views': {key: {**view, 'rgb': np.zeros((16, 16, 3), dtype=np.uint8)}
               for key, view in data['views'].items()}}
    measured = data['current_measurements']
    context = {'latest_capture': capture, 'visual_inventory_reference': data['reference'],
        'capture_epochs': {capture['observation_id']: 7},
        'observation': {'objects': measured['measured_entities'],
                        'visual_inventory': {'missing_geometry_ids': measured['missing_geometry_ids']}},
        'scene': {'coverage': measured['map_coverage']}}
    return context, data['before_facts'], data['after_facts']


def evidence(context, facts, completed=()):
    return operation_evidence(context, 'transfer_verify_place', 'visual_destination_unconfirmed', facts, completed)


def test_new_map_or_geometry_identity_does_not_authorize_identical_rgb_release_recheck():
    context, before, after = frozen_context()
    proposal = {'action': 'skill', 'skill': 'verify_place', 'arguments': {}}
    original = operation_signature(evidence(context, before), proposal)
    assert operation_signature(evidence(context, after), proposal) == original
    context['observation']['objects'].append({'id': 'newly-confirmed-target'})
    assert operation_signature(evidence(context, after), proposal) == original
    target = next(iter(context['visual_inventory_reference']['entities'].values()))
    target['description'] = 'corrected target appearance'
    assert operation_signature(evidence(context, after), proposal) != original


def test_depth_only_change_is_not_new_evidence_for_rgb_only_release():
    context, facts, _ = frozen_context()
    view = context['latest_capture']['views']['3']
    view['depth_m'] = np.ones((16, 16), dtype=np.float32)
    proposal = {'action': 'skill', 'skill': 'verify_place', 'arguments': {}}
    original = operation_signature(evidence(context, facts), proposal)
    view['depth_m'][5, 5] = .8
    assert operation_signature(evidence(context, facts), proposal) == original
    view['rgb'][:] = 255
    assert operation_signature(evidence(context, facts), proposal) != original


def test_successful_geometry_observation_clearing_recovery_flag_cannot_reopen_release():
    context, facts, _ = frozen_context()
    context['release_execution_evidence'] = {'sha256': 'frozen-release'}
    proposal = {'action': 'skill', 'skill': 'verify_place', 'arguments': {}}
    original = operation_signature(evidence(context, facts), proposal)
    facts['recovery_needed'] = not facts.get('recovery_needed')
    context['observation']['objects'] = []
    assert operation_signature(evidence(context, facts), proposal) == original
    context['release_execution_evidence']['sha256'] = 'different-execution'
    assert operation_signature(evidence(context, facts), proposal) != original


@pytest.mark.parametrize('mode', ['fresh', 'reuse_current'])
def test_b5_build_scene_facts_do_not_reopen_same_observation(mode):
    context, before, after = frozen_context()
    proposal = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': mode}}
    seen = {operation_signature(evidence(context, before), proposal)}
    rows = blocked_operations(seen, evidence(context, after))
    assert len(rows) == 1
    assert rows[0]['arguments'] == {'capture_mode': mode}
    assert operation_signature(evidence(context, after), proposal) in seen


@pytest.mark.parametrize('change', ['rgb', 'viewpoint', 'calibration', 'intrinsics', 'generation', 'command', 'action_epoch',
    'requirements', 'entity_description', 'entity_added', 'entity_retired', 'transformation', 'release',
    'completed'])
def test_observation_relevant_changes_restore_observe_choice(change):
    context, facts, _ = frozen_context()
    proposal = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': 'fresh'}}
    seen = {operation_signature(evidence(context, facts), proposal)}
    updated, updated_facts, completed = deepcopy(context), deepcopy(facts), []
    capture = updated['latest_capture']
    reference = updated['visual_inventory_reference']
    entity_id = next(iter(reference['entities']))
    if change == 'rgb':
        capture['views']['3']['rgb'][:] = 255
    elif change == 'viewpoint':
        capture['views']['3']['T_world_camera'][0][3] += .01
    elif change == 'calibration':
        capture['views']['3']['calibration_id'] = 'new-calibration'
    elif change == 'intrinsics':
        capture['views']['3']['intrinsics']['fx'] += 1
    elif change == 'generation':
        capture['generation'] += 1
    elif change == 'command':
        capture['command_revision'] += 1
    elif change == 'action_epoch':
        updated['capture_epochs'][capture['observation_id']] += 1
    elif change == 'requirements':
        reference['task_spec']['target_selector'] = 'a newly specified subject'
    elif change == 'entity_description':
        reference['entities'][entity_id]['description'] = 'corrected identity description'
    elif change == 'entity_added':
        reference['entities']['new-id'] = {'id': 'new-id', 'kind': 'object'}
    elif change == 'entity_retired':
        reference['inventory_state']['retired_entity_ids'].append(entity_id)
    elif change == 'transformation':
        reference['inventory_state']['transformations']['process-1'] = {
            'id': 'process-1', 'input_entity_id': entity_id, 'output_entity_id': 'product-1', 'skill': 'vla_unpack'}
    elif change == 'release':
        updated_facts['released'] = not updated_facts['released']
    else:
        completed = ['prior-input']
    assert blocked_operations(seen, evidence(updated, updated_facts, completed)) == []


def test_capture_id_and_clock_alone_do_not_reopen_observe():
    context, facts, _ = frozen_context()
    proposal = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': 'fresh'}}
    seen = {operation_signature(evidence(context, facts), proposal)}
    context['latest_capture'].update(observation_id='new-id-only', time=100, captured_monotonic_s=200)
    context['capture_epochs']['new-id-only'] = 7
    assert len(blocked_operations(seen, evidence(context, facts))) == 1


@pytest.mark.parametrize('change', ['display_label', 'measured_ids', 'missing_geometry_ids'])
def test_same_sensor_input_is_not_new_evidence_when_interpretation_or_display_changes(change):
    context, facts, _ = frozen_context()
    proposal = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': 'fresh'}}
    seen = {operation_signature(evidence(context, facts), proposal)}
    if change == 'display_label':
        entity = next(iter(context['visual_inventory_reference']['entities'].values()))
        entity['display_id'] = 900
    elif change == 'measured_ids':
        context['observation']['objects'] = []
    else:
        context['observation']['visual_inventory']['missing_geometry_ids'] = ['another-verdict']
    assert len(blocked_operations(seen, evidence(context, facts))) == 1


def test_changed_depth_remains_new_measurement_even_with_identical_rgb():
    context, facts, _ = frozen_context()
    view = context['latest_capture']['views']['3']
    view['depth_m'] = np.ones((16, 16), dtype=np.float32)
    view['valid_depth'] = np.ones((16, 16), dtype=bool)
    proposal = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': 'fresh'}}
    seen = {operation_signature(evidence(context, facts), proposal)}
    view['depth_m'][5, 5] = .8
    assert blocked_operations(seen, evidence(context, facts)) == []


def test_micrometre_render_noise_does_not_reopen_observe_but_cumulative_change_does():
    context, facts, _ = frozen_context()
    view = context['latest_capture']['views']['3']
    view['depth_m'] = np.ones((16, 16), dtype=np.float32)
    proposal = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': 'fresh'}}
    seen = {operation_signature(evidence(context, facts), proposal)}
    for delta in (1e-6, 3e-6, 6e-6):
        view['depth_m'][:] = 1. + delta
        assert len(blocked_operations(seen, evidence(context, facts))) == 1
    view['depth_m'][:] = 1.001
    assert blocked_operations(seen, evidence(context, facts)) == []


def test_fresh_observe_cannot_repeat_as_reuse_current_on_unchanged_evidence():
    from luna_mujoco.runtime.recovery_evidence import operation_signatures
    context, facts, _ = frozen_context()
    fresh = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': 'fresh'}}
    seen = operation_signatures(evidence(context, facts), fresh)
    context['latest_capture'].update(observation_id='another-capture', time=200)
    context['capture_epochs']['another-capture'] = 7
    blocked = blocked_operations(seen, evidence(context, facts))
    assert {json.dumps(row['arguments'], sort_keys=True) for row in blocked} == {
        '{}', '{"capture_mode": "fresh"}', '{"capture_mode": "reuse_current"}'}
    # A changed view can justify a new sensing operation.
    context['latest_capture']['views']['3']['T_world_camera'][0][3] += .01
    assert blocked_operations(seen, evidence(context, facts)) == []


def test_reuse_current_failure_can_still_request_one_new_acquisition():
    from luna_mujoco.runtime.recovery_evidence import operation_signatures
    context, facts, _ = frozen_context()
    reuse = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': 'reuse_current'}}
    seen = operation_signatures(evidence(context, facts), reuse)
    assert all(row['arguments'].get('capture_mode') != 'fresh'
               for row in blocked_operations(seen, evidence(context, facts)))


def test_retry_of_failed_observation_uses_same_observation_boundary():
    context, before, after = frozen_context()
    proposal = {'action': 'retry'}
    previous = operation_evidence(context, 'observe', 'identity_unknown', before, [], failed_skill='observe_scene')
    current = operation_evidence(context, 'observe', 'identity_unknown', after, [], failed_skill='observe_scene')
    seen = {operation_signature(previous, proposal)}
    assert len(blocked_operations(seen, current)) == 1
    assert operation_signature(current, proposal) in seen


def test_map_publication_and_collision_settings_do_not_reopen_same_observation():
    context, facts, _ = frozen_context()
    proposal = {'action': 'skill', 'skill': 'observe_scene', 'arguments': {'capture_mode': 'fresh'}}
    seen = {operation_signature(evidence(context, facts), proposal)}
    context['scene'].update(scene_token='rebuilt', coverage={'inventory_complete': True})
    context['planning_settings'] = {'trajectory_collision_clearance_m': .01}
    assert len(blocked_operations(seen, evidence(context, facts))) == 1


def test_other_recovery_keeps_original_facts_dependency():
    context, before, after = frozen_context()
    for proposal in ({'action': 'retry'}, {'action': 'skill', 'skill': 'choose_arm', 'arguments': {}}):
        seen = {operation_signature(evidence(context, before), proposal)}
        assert blocked_operations(seen, evidence(context, after)) == []

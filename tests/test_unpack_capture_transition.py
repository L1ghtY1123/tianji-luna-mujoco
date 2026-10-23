"""A confirmed natural-pose result frame precedes independent observation mapping."""
import hashlib
from copy import deepcopy

import pytest
from test_demo_lifecycle_perception import before_kwargs, capture_and_action, client, record, work_area

from luna_mujoco.demo_lifecycle_perception import verify_unpacking
from luna_mujoco.demo_lifecycle_tracking import _transition
from luna_mujoco.demo_process import create_product, initialize_processes


def transition_case():
    capture, action = capture_and_action()
    pending = record('unpack_pending')
    pending['mock_action'] = action
    visual, _ = client()
    result = verify_unpacking(visual, capture, target_id='pink', lifecycle=pending,
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    task = {'transport_ids': ['pink'], 'item_lifecycle': {'pink': pending}}
    initialize_processes(task, {'pink': {'display_id': 1}})
    product = create_product(task, 'pink', result, capture, owner_id='run')
    reference = {'owner_id': 'run', 'demo_task': task, 'product_references': {product['id']: {
        'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'evidence': deepcopy(product['creation_evidence']), 'views': deepcopy(capture['views']),
        'verification_capture': deepcopy(result.get('verification_capture'))}}}
    later = deepcopy(capture)
    later.update(observation_id='at-observation-pose', time=3., captured_perf_counter_ns=300)
    return product['id'], task['item_lifecycle'][product['id']], later, reference, capture


def test_verification_capture_binds_actual_rgb_and_time_without_requiring_observation_pose():
    identifier, lifecycle, later, reference, original = transition_case()
    result = lifecycle['unpacking_evidence']
    summary = result['verification_capture']
    assert summary == result['visual_evidence']['verification_capture']
    assert summary['source'] == 'post_unpack_current_rgb'
    assert {key: summary[key] for key in ('observation_id', 'generation', 'time', 'captured_perf_counter_ns')} == {
        key: original[key] for key in ('observation_id', 'generation', 'time', 'captured_perf_counter_ns')}
    for camera, image in summary['images'].items():
        rgb = original['views'][camera]['rgb']
        assert image['rgb_sha256'] == hashlib.sha256(rgb.tobytes()).hexdigest()
        assert image['shape'] == list(rgb.shape)
    accepted = _transition(identifier, lifecycle, later, reference)
    assert accepted['observation_id'] == original['observation_id'] != later['observation_id']
    assert accepted['request_hash'] == result['request_hash']


@pytest.mark.parametrize('change', ['missing_summary', 'visual_summary', 'reference_summary', 'reference_rgb',
    'before_mock', 'later_before_result', 'perf_before_result', 'missing_perf', 'wrong_owner',
    'wrong_generation', 'wrong_action', 'creation_wrong_frame', 'reference_wrong_frame', 'wrong_request',
    'creation_wrong_action', 'empty_request', 'release_perf_after_action'])
def test_cross_frame_transition_rejects_broken_result_capture_chain(change):
    identifier, lifecycle, later, reference, _ = transition_case()
    result = lifecycle['unpacking_evidence']
    summary = result.get('verification_capture', {})
    product_reference = reference['product_references'][identifier]
    if change == 'missing_summary':
        result.pop('verification_capture', None)
        result['visual_evidence'].pop('verification_capture', None)
    elif change == 'visual_summary':
        result['visual_evidence']['verification_capture'] = {'observation_id': 'other'}
    elif change == 'reference_summary':
        product_reference['verification_capture'] = {'observation_id': 'other'}
    elif change == 'reference_rgb':
        product_reference['views']['1']['rgb'][0, 0, 0] ^= 1
    elif change == 'before_mock':
        summary['time'] = .5
        result['visual_evidence']['verification_capture'] = deepcopy(summary)
        product_reference['verification_capture'] = deepcopy(summary)
    elif change == 'later_before_result':
        later['time'] = 1.5
    elif change == 'perf_before_result':
        later['captured_perf_counter_ns'] = 150
    elif change == 'missing_perf':
        later.pop('captured_perf_counter_ns')
    elif change == 'wrong_owner':
        reference['owner_id'] = 'other'
    elif change == 'wrong_generation':
        later['generation'] += 1
    elif change == 'wrong_action':
        result['action_id'] = 'other'
    elif change == 'creation_wrong_frame':
        lifecycle['product_creation']['observation_id'] = 'other'
    elif change == 'reference_wrong_frame':
        product_reference['observation_id'] = 'other'
    elif change == 'creation_wrong_action':
        lifecycle['product_creation']['action_id'] = 'other'
        product_reference['evidence']['action_id'] = 'other'
    elif change == 'empty_request':
        result['request_hash'] = result['visual_evidence']['request_hash'] = ''
        lifecycle['product_creation']['request_hash'] = product_reference['evidence']['request_hash'] = ''
    elif change == 'release_perf_after_action':
        result['visual_evidence']['table_release_reference']['captured_perf_counter_ns'] = 150
    else:
        result['request_hash'] = 'other'
    with pytest.raises(ValueError):
        _transition(identifier, lifecycle, later, reference)


def test_missing_summary_only_supports_original_strict_same_frame_contract():
    identifier, lifecycle, _, reference, original = transition_case()
    result = lifecycle['unpacking_evidence']
    result.pop('verification_capture', None)
    result['visual_evidence'].pop('verification_capture', None)
    reference['product_references'][identifier].pop('verification_capture', None)
    assert _transition(identifier, lifecycle, original, reference)['observation_id'] == original['observation_id']

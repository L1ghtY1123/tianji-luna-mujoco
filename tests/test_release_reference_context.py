"""Current release results bind inventory identity and historical reference images."""
import json
from copy import deepcopy

import pytest
from test_demo_release_result import release_inputs

from luna_mujoco.demo_transfer_verification import release_result_payload
from luna_mujoco.visual_inventory import image_part


def test_place_supplies_bound_appearance_without_unrelated_task_or_historical_locations():
    capture, binding, reference = release_inputs()
    reference['inventory_state'] = {'transformations': {'p1': {
        'input_entity_id': 'original-input', 'output_entity_id': 'pink-1',
        'processing_skill': 'example_process', 'status': 'verified', 'mock_result': 'do-not-send'}}}
    reference['entities']['pink-1'].update(display_id=1, description='pink foam wrapped workpiece',
                                          simulation_body='do-not-send')
    for view in reference['views'].values():
        view['rgb'] = view['rgb'].copy()
        view['rgb'][:] = [180, 80, 130]
    original = deepcopy(reference['inventory_state'])
    payload = release_result_payload(capture, binding, reference, 'unwrap each item')
    context = json.loads(payload['input'][0]['content'][0]['text'])
    assert context['target_id'] == 'pink-1'
    assert context['appearance'] == {'kind': 'cylinder', 'color': 'pink'}
    assert context['destination_id'] == binding['destination_id']
    assert all(k not in context for k in ('task_spec', 'inventory_state', 'current_inventory'))
    content = payload['input'][0]['content']
    images = [p for p in content if p['type'] == 'input_image']
    assert len(images) == 4
    assert all(image_part(v['rgb']) in images for v in capture['views'].values())
    assert 'do-not-send' not in json.dumps(payload)
    assert reference['inventory_state'] == original
    assert [q['name'] for q in payload['questions']] == ['place_result', 'workpiece_still_held']


@pytest.mark.parametrize('fault', ['old_capture', 'wrong_generation', 'before_release', 'missing_slot'])
def test_release_context_rejects_stale_or_missing_current_binding(fault):
    capture, binding, reference = release_inputs()
    if fault == 'old_capture':
        capture['observation_id'] = binding['observation_id']
    elif fault == 'wrong_generation':
        capture['generation'] += 1
    elif fault == 'before_release':
        capture['time'] = binding['release_feedback_time'] - 1
    else:
        binding.pop('assigned_slot')
    with pytest.raises(ValueError, match='release_verification_'):
        release_result_payload(capture, binding, reference)


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_initial_history_precedes_current_views_with_head_last(arm):
    capture, binding, reference = release_inputs()
    binding['arm'] = arm
    payload = release_result_payload(capture, binding, reference)
    labels = [p['text'] for p in payload['input'][0]['content'] if p['type'] == 'input_text']
    initial = [i for i,t in enumerate(labels) if t.startswith('REFERENCE target')]
    current = [i for i,t in enumerate(labels) if t.startswith('CURRENT RGB camera')]
    assert max(initial) < min(current)
    assert labels[current[-1]].startswith('CURRENT RGB camera 3')


def test_release_sends_actual_open_feedback_without_claiming_object_location():
    capture, binding, reference = release_inputs()
    payload = release_result_payload(capture, binding, reference)
    context = json.loads(payload['input'][0]['content'][0]['text'])
    feedback = context['measured_gripper_state']
    assert feedback['source'] == 'actual_robot_joint_feedback_at_current_capture'
    assert feedback['position_rad'] == capture['arms'][binding['arm']]['gripper_position_rad']
    assert feedback['open_confirmed'] is True
    assert feedback['observation_id'] == capture['observation_id']
    assert 'does not establish object location' in feedback['meaning']
    held = payload['questions'][1]['instructions']
    assert 'physically retained' in held and 'not evidence of retention' in held
    assert 'main grey' not in json.dumps(payload['questions'])


@pytest.mark.parametrize('fault', ['retired', 'missing_target', 'wrong_camera', 'invalid_box'])
def test_appearance_reference_rejects_unbound_or_invalid_identity(fault):
    from luna_mujoco.visual_api import VisualInventoryError
    capture, binding, reference = release_inputs()
    if fault == 'retired':
        reference['inventory_state'] = {'retired_entity_ids': ['pink-1']}
    elif fault == 'missing_target':
        reference['entities'].clear()
    elif fault == 'wrong_camera':
        reference['entities']['pink-1']['reference_camera_id'] = 'missing'
    else:
        reference['entities']['pink-1']['bbox'] = [.4, .4, .2, .2]
    with pytest.raises(VisualInventoryError, match='release_.*reference'):
        release_result_payload(capture, binding, reference)

"""Current unpack results must receive the actual initial identity context."""
import json
from copy import deepcopy

import pytest
from test_demo_release_result import release_inputs

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.demo_transfer_verification import release_result_payload
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import image_part


def test_place_supplies_initial_inventory_active_id_and_three_reference_images():
    capture, binding, reference = release_inputs()
    reference['demo_task'].update(workflow='unpack_transfer', transport_ids=['pink-1'],
        item_lifecycle={'pink-1': {'phase': 'wrapped_source', 'appearance': 'wrapped',
                                 'simulator_truth': 'do-not-send'}},
        process_records={'p1': {'input_entity_id': 'pink-1', 'output_entity_id': None, 'status': 'pending',
                                'mock_result': 'do-not-send'}})
    reference['entities']['pink-1'].update(display_id=1, description='pink foam wrapped workpiece',
                                          simulation_body='do-not-send')
    for view in reference['views'].values():
        view['rgb'] = view['rgb'].copy()
        view['rgb'][:] = [180, 80, 130]
    original = deepcopy(reference['demo_task'])
    payload = release_result_payload(capture, binding, reference, 'unwrap each item')
    context = json.loads(payload['input'][0]['content'][0]['text'])['task_context']
    assert context['active_target_id'] == 'pink-1'
    assert context['initial_inventory']['pink-1']['display_id'] == 1
    assert context['current_stage'] == 'awaiting_table_release_verification_before_unpack'
    assert context['process_records']['p1']['input_entity_id'] == 'pink-1'
    content = payload['input'][0]['content']
    images = [p for p in content if p['type'] == 'input_image']
    assert all(image_part(v['rgb']) in images for v in reference['views'].values())
    assert all(image_part(v['rgb']) in images for v in capture['views'].values())
    assert 'do-not-send' not in json.dumps(payload)
    assert reference['demo_task'] == original
    assert [q['name'] for q in payload['questions']] == ['place_result', 'workpiece_still_held']


@pytest.mark.parametrize('lifecycle', [None, {}, {'other': {'phase': 'wrapped_source'}}])
def test_missing_unpack_lifecycle_cannot_fall_into_old_transfer_path(lifecycle):
    reference = {'demo_task': {'workflow': 'unpack_transfer', 'source_object_ids': ['pink'],
                               'transport_ids': ['pink'], 'item_lifecycle': lifecycle}}
    with pytest.raises(VisualInventoryError, match='unpack_lifecycle_required'):
        DemoVisualInventory(api_key='unused').resolve({}, [], entities={}, reference=reference,
                                                       purpose='associate_inventory')


def test_transfer_workflow_cannot_carry_unpack_lifecycle():
    reference = {'demo_task': {'workflow': 'transfer', 'item_lifecycle': {'pink': {}}}}
    with pytest.raises(VisualInventoryError, match='unpack_workflow_lifecycle_conflict'):
        DemoVisualInventory(api_key='unused').resolve({}, [], reference=reference)


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_initial_history_precedes_current_views_with_head_last(arm):
    capture, binding, reference = release_inputs()
    binding['arm'] = arm
    reference['demo_task']['workflow'] = 'unpack_transfer'
    payload = release_result_payload(capture, binding, reference)
    labels = [p['text'] for p in payload['input'][0]['content'] if p['type'] == 'input_text']
    initial = [i for i,t in enumerate(labels) if t.startswith('INITIAL REFERENCE RGB')]
    current = [i for i,t in enumerate(labels) if t.startswith('CURRENT camera')]
    assert max(initial) < min(current)
    assert labels[current[-1]].startswith('CURRENT camera 3')


def test_release_sends_actual_open_feedback_without_claiming_object_location():
    capture, binding, reference = release_inputs()
    reference['demo_task']['workflow'] = 'unpack_transfer'
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

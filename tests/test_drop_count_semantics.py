"""Delivery is a verified count increment, not appearance-based individual tracing."""
import json

import httpx
import pytest
from test_demo_lifecycle_perception import record
from test_drop_region_presentation import measured_drop_inputs, request

from luna_mujoco.demo_lifecycle_perception import verify_right_bin_drop
from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError


@pytest.mark.parametrize('completed,expected', [(0, 1), (1, 2)])
def test_first_and_second_delivery_require_current_visible_count(completed, expected):
    capture, action, reference = measured_drop_inputs()
    items = reference['demo_task']['item_lifecycle']
    items['previous']['phase'] = 'completed' if completed else 'wrapped_source'
    items['unverified_drop'] = record('right_bin_released')
    result, payload = request(capture, action, reference)
    context = json.loads(payload['input'][0]['content'][0]['text'])
    count = context['delivery_count']
    assert count == {'initial_right_bin_black_count': 0, 'source': 'user_scene_prior',
                     'already_completed_count': completed, 'expected_current_count': expected}
    assert context['already_completed_task_ids'] == (['previous'] if completed else [])
    assert [q['name'] for q in payload['questions']] == ['current_item_added']
    question = payload['questions'][0]
    assert question['name'] == 'current_item_added'
    assert 'additional black workpiece' in question['instructions']
    assert 'overlap or stack' in question['instructions']
    assert 'Compare BEFORE and CURRENT' in context['rules']
    assert 'separately counting every completed item is not required' in context['rules']
    assert 'action records alone are insufficient' in context['rules']
    assert result['visual_evidence']['delivery_count'] == count
    assert result['visual_evidence']['completion_semantics'] == 'released_in_bin_and_visible_addition'
    assert sum(p['type'] == 'input_image' for p in payload['input'][0]['content']) == 10
    assert 'all CURRENT views together' in question['instructions']


@pytest.mark.parametrize('value,confidence', [('no', .97), ('unknown', .99), ('yes', .77)])
def test_previous_object_or_obscured_count_cannot_complete_second_delivery(value, confidence):
    inputs = measured_drop_inputs()
    result, payload = request(*inputs, answers={'current_item_added': (value, confidence)})
    assert 'additional black workpiece' in payload['questions'][0]['instructions']
    assert result['success'] is False
    expected = confidence if value == 'yes' else 1-confidence if value == 'no' else .5
    assert result['visual_evidence']['answers'][0]['probability'] == expected


def test_presence_and_count_are_one_complete_result():
    result, payload = request(*measured_drop_inputs())
    assert result['success'] is True
    assert result['visual_evidence']['judgments'] == {'current_item_added': 'yes'}
    assert 'inside the same designated BIN REGION' in payload['questions'][0]['instructions']
    assert 'No exact pose or identity of individual dropped pieces is required' in payload['questions'][0]['instructions']


def test_old_three_answer_response_is_not_silently_reinterpreted_under_two_question_contract():
    capture, action, reference = measured_drop_inputs()
    answers = [{'name': name, 'type': 'choice', 'choice': 'yes', 'confidence': .99}
               for name in ('in_right_bin', 'separated', 'current_item_added')]
    visual = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={'answers': answers}))))
    with pytest.raises(VisualInventoryError, match='invalid_response'):
        verify_right_bin_drop(visual, capture, target_id='pink', lifecycle=record('right_bin_released'),
            action_evidence=action, reference=reference, owner_id='run')


def test_count_change_never_changes_current_or_before_image_pixels():
    capture, action, reference = measured_drop_inputs()
    _, second = request(capture, action, reference)
    reference['demo_task']['item_lifecycle']['previous']['phase'] = 'wrapped_source'
    _, first = request(capture, action, reference)
    def images(payload):
        return [part for part in payload['input'][0]['content'] if part['type'] == 'input_image']
    assert images(first) == images(second)

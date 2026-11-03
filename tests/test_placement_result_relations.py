import json

from test_demo_release_result import release_inputs
from test_placement_zones import intent

from luna_mujoco.demo_transfer_verification import release_result_payload


def test_container_result_question_checks_inside_instead_of_tabletop():
    capture, binding, reference = release_inputs()
    binding['placement_zone'] = {'intent': intent(relation='inside'), 'support_entity_id': binding['destination_id']}
    payload = release_result_payload(capture, binding, reference)
    question = next(q['instructions'] for q in payload['questions'] if q['name'] == 'place_result')
    assert 'inside' in question and 'tabletop' not in question
    context = json.loads(payload['input'][0]['content'][0]['text'])
    assert context['placement_intent']['relation'] == 'inside'


def test_candidate_selected_placement_preserves_original_language_in_release_verification():
    capture, binding, reference = release_inputs()
    original = {'instruction': '把物体放在杯子左侧的桌面上'}
    binding['placement_zone'] = {'intent': intent(), 'support_entity_id': binding['destination_id'],
        'semantic_scope': 'candidate_selection', 'original_requirement': original,
        'active_object_requirement': {'destination_ref': binding['destination_id']}}
    payload = release_result_payload(capture, binding, reference)
    context = json.loads(payload['input'][0]['content'][0]['text'])
    assert context['original_placement_requirement'] == original
    question = next(q['instructions'] for q in payload['questions'] if q['name'] == 'place_result')
    assert 'original instruction' in question
    assert 'relative' in question


def test_intermediate_release_checks_current_transfer_without_claiming_final_destination():
    capture, binding, reference = release_inputs()
    transfer = {'bound_destination_ref': 'work', 'final_destination_ref': 'box',
                'work_area_ref': 'work', 'scope': 'declared_work_area'}
    binding['placement_zone'] = {'intent': intent(), 'support_entity_id': binding['destination_id'],
        'original_requirement': {'instruction': '拆包后把产物放进盒子'}, 'current_transfer': transfer}
    payload = release_result_payload(capture, binding, reference)
    context = json.loads(payload['input'][0]['content'][0]['text'])
    assert context['current_transfer'] == transfer
    question = next(q['instructions'] for q in payload['questions'] if q['name'] == 'place_result')
    assert 'applicable' in question and 'intermediate' in question
    assert 'does not establish final task completion' in question



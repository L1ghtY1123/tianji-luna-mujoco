"""Unique remaining role/count never establishes an original pink identity."""
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from decision_fixtures import decision_answer
from test_lifecycle_role_inventory import inputs, reference_for

from luna_mujoco.demo_lifecycle_inventory import resolve_lifecycle_inventory
from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import image_part


def run_case(answer=('yes', .95), mutate=None):
    capture, detections, entities, tracking, containers = inputs()
    reference = reference_for(capture, entities)
    if mutate:
        mutate(reference, entities)
    before = deepcopy(reference)
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = [decision_answer(question,
                    answer[0] if question['name'].startswith('wrapped_identity_') else 'yes',
                    answer[1] if question['name'].startswith('wrapped_identity_') else .95)
                   for question in payload['questions']]
        return httpx.Response(200, json={'answers': answers})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
                                        instruction='unpack every original item', reference=reference)
    return result, calls, reference, before


def test_wrapped_source_request_compares_same_original_id_and_keeps_ordinal():
    result, calls, reference, before = run_case()
    assert len(calls) == 1
    assert [q['name'] for q in calls[0]['questions']] == ['identity_2', 'identity_3', 'wrapped_identity_2']
    content = calls[0]['input'][0]['content']
    images = [part for part in content if part['type'] == 'input_image']
    for view in reference['views'].values():
        assert image_part(view['rgb']) in images
    context = json.loads(content[0]['text'])
    assert context['purpose'] == 'associate_inventory'
    assert context['wrapped_identity']['wrapped_identity_references']['pink']['display_id'] == 2
    assert {part['text'] for part in content if part['type'] == 'input_text'
            and part['text'].startswith('CURRENT RGB camera ')} == {
                'CURRENT RGB camera 1', 'CURRENT RGB camera 2', 'CURRENT RGB camera 3'}
    question = next(q for q in calls[0]['questions'] if q['name'] == 'wrapped_identity_2')
    assert 'same original physical' in question['instructions']
    assert 'pink' in question['instructions'] and '#2' in question['instructions']
    assert 'Do not reassign IDs by remaining count' in question['instructions']
    assert 'combined' in question['instructions']
    assert 'Unique intrinsic markings are not required' in question['instructions']
    black_question = next(q for q in calls[0]['questions'] if q['name'] == 'identity_3')
    assert 'C3 in camera 3 shows a black workpiece.' in black_question['instructions']
    assert 'Use all CURRENT views together' in black_question['instructions']
    assert 'Robot parts, container and background do not count.' in black_question['instructions']
    entry = next(group[0] for group in result['groups'] if group[0]['entity_id'] == 'pink')
    assert entry['identity_evidence'][0]['method'] == 'initial_reference_visual_same_physical_identity'
    assert entry['identity_evidence'][0]['request_hash'] == result['wrapped_identity_verification']['request_hash']
    assert entry['identity_evidence'][0]['role_request_hash'] == result['evidence']['request_hash']
    assert result['wrapped_identity_verification']['request_hash'] == result['evidence']['request_hash']
    assert result['wrapped_identity_verification']['role_request_hash'] == result['evidence']['request_hash']
    assert reference['entities'] == before['entities']
    for camera in reference['views']:
        np.testing.assert_array_equal(reference['views'][camera]['rgb'], before['views'][camera]['rgb'])


@pytest.mark.parametrize('answer', [('unknown', .99), ('yes', .79), ('no', .99)])
def test_unique_remaining_pink_role_cannot_promote_unknown_original_identity(answer):
    with pytest.raises(VisualInventoryError, match='visual_lifecycle_original_identity_unconfirmed'):
        run_case(answer)


@pytest.mark.parametrize('mutation', ['missing_view', 'missing_box', 'changed_display_id', 'unknown_id'])
def test_missing_or_changed_original_reference_stops_before_identity_acceptance(mutation):
    def mutate(reference, entities):
        if mutation == 'missing_view': reference['views'].pop('2')
        elif mutation == 'missing_box': reference['entities']['pink'].pop('bbox')
        elif mutation == 'changed_display_id': entities['pink']['display_id'] = 1
        else: reference['entities'].pop('pink')
    with pytest.raises(VisualInventoryError, match='visual_lifecycle_reference_'):
        run_case(mutate=mutate)


def test_initial_reference_pixels_participate_in_request_cache():
    capture, detections, entities, tracking, containers = inputs()
    reference = reference_for(capture, entities)
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={'answers': [decision_answer(q) for q in payload['questions']]})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    first = resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
                                       instruction='unpack', reference=reference)
    repeated = resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
                                          instruction='unpack', reference=reference)
    assert len(calls) == 1
    assert repeated['evidence']['request_hash'] == first['evidence']['request_hash']
    assert repeated['wrapped_identity_verification'] == first['wrapped_identity_verification']
    reference['views']['2']['rgb'][:] = 5
    second = resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
                                        instruction='unpack', reference=reference)
    assert len(calls) == 2
    assert first['evidence']['request_hash'] != second['evidence']['request_hash']
    assert first['wrapped_identity_verification']['request_hash'] != second['wrapped_identity_verification']['request_hash']
    for result in (first, second):
        assert result['wrapped_identity_verification']['request_hash'] == result['evidence']['request_hash']


def test_production_resolve_passes_initial_reference_to_lifecycle_request(monkeypatch):
    import importlib
    module = importlib.import_module('luna_mujoco.demo_perception')
    capture, detections, entities, tracking, containers = inputs()
    reference = reference_for(capture, entities)
    reference['demo_task']['item_lifecycle'] = {'pink': {'phase': 'wrapped_source', 'appearance': 'wrapped',
        'appearance_revision': 0, 'to_table_arm': 'A', 'to_right_bin_arm': 'B'}}
    monkeypatch.setattr(module, 'container_correspondence', lambda *args: containers)
    monkeypatch.setattr(module, 'lifecycle_candidate_scopes', lambda *args, **kwargs: tracking)
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={'answers': [decision_answer(q) for q in payload['questions']]})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = client.resolve(capture, detections, entities=entities, reference=reference,
                            purpose='associate_inventory', instruction='unpack')
    assert len(calls) == 1
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])['wrapped_identity']
    assert context['initial_reference_observation_id'] == reference['observation_id']
    wrapped_id = next(iter(context['wrapped_identity_references']))
    assert context['wrapped_identity_references'][wrapped_id]['display_id'] == reference['entities'][wrapped_id]['display_id']
    assert wrapped_id in result['identity_coverage']['accepted_left_entity_ids']


def test_strict_unknown_never_builds_partial_identity_groups(monkeypatch):
    from luna_mujoco.visual_inventory import VisualInventory
    calls = []
    monkeypatch.setattr(VisualInventory, '_groups', lambda *args, **kwargs: calls.append(args))
    with pytest.raises(VisualInventoryError, match='visual_lifecycle_original_identity_unconfirmed'):
        run_case(('unknown', .99))
    assert calls == []


@pytest.mark.parametrize('role_answer,strict_answer,expected_error,call_count', [
    (('unknown', .99), ('yes', .95), 'visual_lifecycle_role_unconfirmed', 1),
    (('yes', .95), ('unknown', .99), 'visual_lifecycle_original_identity_unconfirmed', 1),
    (('yes', .95), ('yes', .79), 'visual_lifecycle_original_identity_unconfirmed', 1),
])
def test_each_failure_is_cached_and_no_partial_groups_escape(role_answer, strict_answer, expected_error, call_count):
    capture, detections, entities, tracking, containers = inputs()
    reference = reference_for(capture, entities)
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for question in payload['questions']:
            value, confidence = (strict_answer if question['name'].startswith('wrapped_identity_') else role_answer)
            answers.append(decision_answer(question, value, confidence))
        return httpx.Response(200, json={'answers': answers})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    for _ in range(2):
        with pytest.raises(VisualInventoryError, match=expected_error):
            resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
                instruction='unpack', reference=reference)
    assert len(calls) == call_count


def test_strict_supplementary_unknown_is_excluded_after_left_original_identity_passes():
    capture, detections, entities, tracking, containers = inputs()
    reference = reference_for(capture, entities)
    detections.append({**deepcopy(detections[2]), 'camera_id': '1'})
    tracking['candidate_scopes'][4] = {**tracking['candidate_scopes'][2], 'camera_id': '1'}
    def handle(request):
        payload = json.loads(request.content)
        return httpx.Response(200, json={'answers': [decision_answer(q, 'unknown' if q['name'] == 'wrapped_identity_4' else 'yes')
            for q in payload['questions']]})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
        instruction='unpack', reference=reference)
    pink = next(group for group in result['groups'] if group[0]['entity_id'] == 'pink')
    assert [entry['camera_id'] for entry in pink] == ['2']
    assert result['wrapped_identity_verification']['accepted_candidate_ids'] == ['2']
    assert result['unassociated_candidates'][0]['candidate'] == 'candidate-4'


def test_black_only_association_never_sends_initial_pink_reference():
    capture, detections, entities, tracking, containers = inputs()
    reference = reference_for(capture, entities)
    tracking['bindings'].pop('pink')
    tracking['candidate_scopes'].pop(2)
    entities.pop('pink')
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={'answers': [decision_answer(q) for q in payload['questions']]})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
        instruction='unpack', reference=reference)
    assert len(calls) == 1
    assert not any(part.get('text', '').startswith('INITIAL') for part in calls[0]['input'][0]['content'])
    assert all(not q['name'].startswith('wrapped_identity_') for q in calls[0]['questions'])
    assert 'wrapped_identity_verification' not in result


def test_temporal_context_shows_all_competing_references_and_sanitized_history():
    def mutate(reference, entities):
        reference['owner_id'] = 'run'
        reference['demo_task']['item_lifecycle'] = {
            'pink': {'phase': 'wrapped_source', 'appearance': 'wrapped', 'appearance_revision': 0},
            'black': {'phase': 'unpacked_confirmed', 'appearance': 'unpacked', 'appearance_revision': 1,
                'mock_action': {'target_id': 'black', 'owner_id': 'run', 'generation': 2,
                    'action_id': 'action-1', 'completed': True, 'source': 'mock_vla',
                    'ground_truth': 'SECRET_POSE', 'commanded_pose': 'SECRET_COMMAND'},
                'table_release_feedback_id': 'feedback-1', 'physical_id': 'SECRET_SIM_ID'}}
    _, calls, _, _ = run_case(mutate=mutate)
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])['wrapped_identity']
    proposal = context['temporal_correspondence_proposals'][0]
    assert {row['entity_id'] for row in proposal['initial_source_references']} == {'pink', 'black'}
    assert all(row['box_source'] == 'initial_responses_rgb_annotation_not_sam_measurement'
               for row in proposal['initial_source_references'])
    assert proposal['identity_assignment'] is False
    assert proposal['camera_comparison']['status'] == 'unavailable'
    history = context['recorded_task_history']
    assert history['pink']['recorded_actions'] == []
    assert history['black']['recorded_actions'][0]['target_id'] == 'black'
    text = json.dumps(context)
    assert 'SECRET_' not in text and 'ground_truth' not in text and 'commanded_pose' not in text
    assert 'absence of recorded action does not guarantee no physical motion' in context['rules']


@pytest.mark.parametrize('change', [None, 'missing_pose', 'different_reference'])
def test_camera_comparison_uses_only_available_registered_anchor(change):
    from luna_mujoco.demo_lifecycle_inventory import _temporal_evidence
    capture, detections, entities, tracking, containers = inputs()
    reference = reference_for(capture, entities)
    view = {'T_world_camera': np.eye(4).tolist(), 'calibration_id': 'registered-calibration',
            'intrinsics': {'fx': 500, 'fy': 500, 'cx': 50, 'cy': 40}}
    capture['views']['2'].update(deepcopy(view))
    capture['views']['2']['T_world_camera'][0][3] = .0002
    reference['container_identity_tracks'] = {'reference_observation_id': 'initial', 'generation': 2,
        'containers': {'left': {'views': {'2': deepcopy(view)}}}}
    if change == 'missing_pose':
        capture['views']['2'].pop('T_world_camera')
    elif change == 'different_reference':
        reference['container_identity_tracks']['reference_observation_id'] = 'other'
    evidence = _temporal_evidence(capture, detections, reference, containers,
                                  {2: tracking['candidate_scopes'][2]})
    comparison = evidence['temporal_correspondence_proposals'][0]['camera_comparison']
    if change is None:
        assert comparison['translation_delta_m'] == pytest.approx(.0002)
        assert comparison['rotation_delta_rad'] == pytest.approx(0.)
        assert comparison['same_calibration_id'] is True and comparison['same_intrinsics'] is True
    else:
        assert comparison == {'status': 'unavailable'}

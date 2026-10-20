import httpx
import numpy as np
import pytest

from jev_mujoco.visual_inventory import VisualInventory, VisualInventoryError


def fixture():
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[4:16, 3:14, 0] = 255
    mask = rgb[:, :, 0] > 0
    view = {'rgb': rgb, 'depth_m': np.ones((24, 32)), 'valid_depth': np.ones((24, 32), bool)}
    capture = {'source': 'rgbd', 'generation': 1, 'observation_id': 'frame-1', 'views': {'3': view}}
    detection = {'camera_id': '3', 'mask': mask, 'silhouette_mask': mask,
                 'points': np.array([[.6, .1, .8]] * 132), 'pixels': rgb[mask],
                 'pixels_rc': np.column_stack(np.nonzero(mask)), 'kind': 'cube',
                 'color': 'red', 'prompt': 'cube', 'score': .95, 'view': view}
    return capture, detection


def client_with_answers(answers, calls):
    def handle(request):
        import json
        payload = json.loads(request.content)
        calls.append(payload)
        assert str(request.url) == 'https://api.openai.com/v1/decisions'
        output = []
        for question in payload['questions']:
            name = question['name']
            choice = answers.get(name, 'unknown')
            if question['type'] == 'predicate':
                output.append({'name': name, 'type': 'predicate', 'probability': choice})
            else:
                output.append({'name': name, 'type': 'choice', 'choice': choice, 'confidence': .99,
                               'probabilities': [{'value': choice, 'probability': .99}]})
        return httpx.Response(200, json={'model': 'gpt-6-luna', 'answers': output, 'usage': {}})
    return httpx.Client(transport=httpx.MockTransport(handle))


def fixed_entities():
    return {'red': {'kind': 'cube', 'color': 'red', 'visual_membership': 'required'}}


def test_duplicate_masks_bind_one_fixed_identity_and_cache():
    capture, detection = fixture()
    calls = []
    inventory = VisualInventory(api_key='test-key', client=client_with_answers(
        {'identity_0': 'red', 'identity_1': 'red'}, calls))
    result = inventory.resolve(capture, [detection, dict(detection)], instruction='move red', entities=fixed_entities())
    assert len(result['groups']) == len(result['groups'][0]) == 1
    assert result['groups'][0][0]['visual_membership'] == 'required'
    inventory.resolve(capture, [detection, dict(detection)], instruction='move red', entities=fixed_entities())
    assert len(calls) == 1 and calls[0]['model'] == 'gpt-6-luna'


def test_decisions_cannot_create_an_inventory_from_masks():
    capture, detection = fixture()
    with pytest.raises(VisualInventoryError, match='visual_inventory_initial_rgb_required'):
        VisualInventory().resolve(capture, [detection], instruction='move red')


def test_http_error_is_cached_and_never_exposes_provider_body():
    capture, detection = fixture()
    calls = []
    def fail(request):
        calls.append(request)
        return httpx.Response(401, text='secret-provider-message')
    inventory = VisualInventory(api_key='test-key', client=httpx.Client(transport=httpx.MockTransport(fail)))
    for _ in range(2):
        with pytest.raises(VisualInventoryError, match='visual_decisions_http_401') as error:
            inventory.resolve(capture, [detection], instruction='move red', entities=fixed_entities())
        assert 'secret-provider-message' not in str(error.value)
    assert len(calls) == 1


def test_identical_pixels_in_new_capture_do_not_reuse_old_observation_id():
    capture, detection = fixture()
    calls = []
    inventory = VisualInventory(api_key='test-key', client=client_with_answers({'identity_0': 'red'}, calls))
    inventory.resolve(capture, [detection], instruction='move red', entities=fixed_entities())
    capture['observation_id'] = 'new-frame'
    result = inventory.resolve(capture, [detection], instruction='move red', entities=fixed_entities())
    assert result['evidence']['observation_id'] == 'new-frame' and len(calls) == 2


def test_unknown_membership_is_not_promoted_to_required_by_segmentation():
    capture, detection = fixture()
    inventory = VisualInventory(api_key='test-key', client=client_with_answers({'identity_0': 'red'}, []))
    result = inventory.resolve(capture, [detection], instruction='move one red', entities={'red': {'kind': 'cube'}})
    assert result['membership'] == {'red': 'unknown'}


@pytest.mark.parametrize('body', [[], {'answers': [None] * 2}])
def test_malformed_success_is_sanitized_and_cached(body):
    capture, detection = fixture()
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=body)
    inventory = VisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    for _ in range(2):
        with pytest.raises(VisualInventoryError, match='visual_decisions_invalid_response'):
            inventory.resolve(capture, [detection], instruction='move red', entities=fixed_entities())
    assert len(calls) == 1


def test_mask_quality_is_not_an_allowed_identity_answer():
    capture, detection = fixture()
    calls = []
    inventory = VisualInventory(api_key='test', client=client_with_answers({'identity_0': 'invalid_mask'}, calls))
    with pytest.raises(VisualInventoryError, match='visual_decisions_invalid_response'):
        inventory.resolve(capture, [detection], instruction='move red', entities=fixed_entities())
    assert sum(part['type'] == 'input_image' for part in calls[0]['input'][0]['content']) == 2


def test_identity_questions_only_offer_entities_reject_or_unknown():
    capture, detection = fixture()
    detection.update(kind='bin', prompt='blue open bin')
    calls = []
    inventory = VisualInventory(api_key='test', client=client_with_answers({'identity_0': 'box'}, calls))
    result = inventory.resolve(capture, [detection], instruction='move contents',
        entities={**fixed_entities(), 'box': {'kind': 'bin', 'color': 'blue'}})
    question = next(item for item in calls[0]['questions'] if item['name'] == 'identity_0')
    assert {item['value'] for item in question['choices']} == {'red', 'box', 'reject', 'unknown'}
    assert result['groups'][0][0]['entity_id'] == 'box'


@pytest.mark.parametrize(('selected', 'confidence'), [('red', .58), ('unknown', .99)])
def test_unknown_identity_reports_candidate_and_original_decision(selected, confidence):
    capture, detection = fixture()
    def respond(request):
        return httpx.Response(200, json={'model': 'gpt-6-luna', 'usage': {}, 'answers': [
            {'type': 'choice', 'name': 'identity_0', 'choice': selected, 'confidence': confidence,
             'probabilities': [{'value': selected, 'probability': .64}]},
            {'type': 'choice', 'name': 'visibility_red', 'choice': 'visible', 'confidence': 1.,
             'probabilities': [{'value': 'visible', 'probability': 1.}]}]})
    inventory = VisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(VisualInventoryError, match='visual_identity_unknown') as failure:
        inventory.resolve(capture, [detection], instruction='move red', entities=fixed_entities())
    evidence = failure.value.evidence
    assert evidence['candidate'] == 'candidate-0' and evidence['camera_id'] == '3'
    assert evidence['choice'] == selected and evidence['confidence'] == confidence
    assert evidence['minimum_confidence'] == .8
    assert evidence['answers'][0]['choice'] == selected


def test_container_pixels_do_not_include_confirmed_contents_and_prompt_is_preserved():
    capture, detection = fixture()
    mask = detection['mask'].copy()
    mask[4:10] = False
    small = {**detection, 'mask': mask, 'silhouette_mask': mask,
             'pixels_rc': np.column_stack(np.nonzero(mask)), 'pixels': detection['view']['rgb'][mask],
             'points': detection['points'][:int(mask.sum())]}
    inventory = VisualInventory(api_key='test', client=client_with_answers({'identity_0': 'box', 'identity_1': 'red'}, []))
    result = inventory.resolve(capture, [detection, small], instruction='move contents',
        entities={**fixed_entities(), 'box': {'kind': 'bin'}})
    box, part = [group[0] for group in result['groups']]
    assert not np.any(box['mask'] & part['mask'])
    assert box['mask'].sum() + part['mask'].sum() == detection['mask'].sum()
    assert box['prompt'] == 'cube' and len(box['pixels_rc']) == box['mask'].sum()

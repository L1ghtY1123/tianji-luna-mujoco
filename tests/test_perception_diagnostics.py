"""Failure diagnostics preserve the measured boundary without changing decisions."""
import json

import httpx
import pytest
from geometry_perception_fixture import GeometryPerception
from test_demo_partial_identity import inputs
from test_perception import capture_fixture
from test_visual_inventory import fixed_entities, fixture

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.perception import PerceptionError
from luna_mujoco.visual_inventory import VisualInventory, VisualInventoryError


@pytest.mark.parametrize('inventory_type', [VisualInventory, DemoVisualInventory])
def test_identity_failure_keeps_question_score_threshold_and_candidate(inventory_type):
    capture, detection = fixture()
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = [{'name': question['name'], 'type': 'predicate', 'probability': .59}
                   if question['type'] == 'predicate' else {'name': question['name'], 'type': 'choice',
                    'choice': 'red' if question['name'].startswith('identity_') else
                    'none' if question['name'] == 'scene_inventory_closure' else 'visible',
                    'confidence': .59 if question['name'].startswith('identity_') else .99}
                   for question in payload['questions']]
        return httpx.Response(200, json={'answers': answers})

    inventory = inventory_type(api_key='secret-key', client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(VisualInventoryError, match='visual_identity_unknown') as caught:
        inventory.resolve(capture, [detection], instruction='move red', entities=fixed_entities(),
                          purpose='initialize_inventory')
    evidence = caught.value.evidence
    assert evidence['component'] == 'decisions'
    assert evidence['candidate_id'] == 'candidate-0'
    assert evidence['prompt'] == 'cube' and evidence['sam3_score'] == .95
    initial = evidence
    if inventory_type is DemoVisualInventory:
        assert len(calls) == 2
        initial = evidence['initial_identity_failure']['evidence']
        assert evidence['request_hash'] != initial['request_hash']
        current = next(item for item in evidence['diagnostic_checks'] if item['name'] == 'match_0_0')
        assert current['metric'] == 'predicate_probability'
        assert current['actual'] == .59 and current['threshold'] == .8 and current['negative_threshold'] == .2
        assert current['status'] == 'unknown' and current['answer']['type'] == 'predicate'
        assert current['question'] and current['camera_id'] == '3'
        derived = next(item for item in evidence['diagnostic_checks'] if item['name'] == 'identity_0')
        assert derived['metric'] == 'minimum_pairwise_agreement'
        assert derived['actual'] == 0. and derived['answer']['choice'] == 'unknown'
    else:
        assert len(calls) == 1
    check = next(item for item in initial['diagnostic_checks'] if item['name'] == 'identity_0')
    assert check['metric'] == 'decisions_confidence'
    assert check['actual'] == .59 and check['threshold'] == .8
    assert check['status'] == 'unknown' and check['answer']['choice'] == 'red'
    assert check['question'] and check['camera_id'] == '3'
    assert evidence['observation_id'] == 'frame-1' and evidence['request_hash']
    assert 'secret-key' not in json.dumps(evidence) and 'base64' not in json.dumps(evidence)


def test_closure_low_confidence_records_original_answer_without_rejecting_identity():
    capture, detections, entities, reference = inputs(('2',))

    def respond(request):
        questions = json.loads(request.content)['questions']
        return httpx.Response(200, json={'answers': [
            {'type': 'choice', 'name': question['name'],
             'choice': 'none' if question['name'] == 'scene_inventory_closure' else
             'pink' if question['name'].startswith('identity_') else 'visible',
             'confidence': .69 if question['name'] == 'scene_inventory_closure' else .99}
            for question in questions]})

    inventory = DemoVisualInventory(api_key='secret', client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = inventory.resolve(capture, detections, instruction='move all', entities=entities,
                               reference=reference, purpose='initialize_inventory')
    check = result['demo_inventory_check']
    assert check['status'] == 'unknown' and check['choice'] == 'none'
    assert check['minimum_confidence'] == .8
    assert check['diagnostic_checks'][0]['actual'] == .69
    assert check['diagnostic_checks'][0]['question']


def test_sam_failure_records_attempt_before_call_and_preserves_details():
    capture, _ = capture_fixture()

    class FailedSegmenter:
        def segment(self, rgb, text, timestamp_us):
            raise PerceptionError('sam3_service_unavailable', details={'http_status': 503})

    perception = GeometryPerception(FailedSegmenter(), identities=('first',), prompts={'first': 'red package'})
    with pytest.raises(PerceptionError, match='sam3_service_unavailable') as caught:
        perception.observe(capture, source_hint={'kind': 'cube', 'color': 'red'})
    details = caught.value.details
    assert details['http_status'] == 503 and details['component'] == 'sam3'
    assert details['camera_id'] == '3' and details['prompt'] == 'red package'
    assert details['observation_id'] == 'capture-one'
    assert len(caught.value.sam3_queries) == 1
    assert details['sam3_queries'][0]['status'] == 'error'


def test_zero_sam_candidates_are_not_reported_as_low_scores():
    capture, _ = capture_fixture()

    class EmptySegmenter:
        def segment(self, rgb, text, timestamp_us):
            return []

    perception = GeometryPerception(EmptySegmenter(), identities=('first',), prompts={'first': 'cube'})
    result = perception.observe(capture, source_hint={'kind': 'cube', 'color': 'any'})
    for query in result['sam3_queries']:
        assert query['status'] == 'no_candidates' and query['candidate_count'] == 0
        assert query['candidate_scores'] == []
        assert 'sam3_internal_filter_threshold' in query['unknowns']
        assert 'sam3_internal_rejection_reason' in query['unknowns']


def test_depth_rejection_is_separate_from_sam_score():
    capture, masks = capture_fixture()
    mask = masks[0].copy()
    mask[37:] = False

    class ShallowSegmenter:
        def segment(self, rgb, text, timestamp_us):
            return [{'mask': mask, 'score': .99}]

    perception = GeometryPerception(ShallowSegmenter(), identities=('first',), prompts={'first': 'cube'})
    result = perception.observe(capture, source_hint={'kind': 'cube', 'color': 'any'})
    check = result['sam3_queries'][0]['diagnostic_checks'][0]
    assert check['metric'] == 'valid_depth_pixels'
    assert check['actual'] == 9 and check['threshold'] == 30 and check['status'] == 'fail'
    assert result['sam3_queries'][0]['candidate_scores'] == [.99]


def test_predicate_unknown_has_question_and_both_probability_thresholds():
    from test_agent_model_layer import Client, capture

    from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest

    client = Client([{'answers': [{'type': 'predicate', 'name': 'support', 'probability': .61}]}])
    result = DecisionsJudgments(api_key='secret', client=client).judge(capture(),
        requests=[JudgmentRequest(name='support', instructions='Is the support visible?')], context={})
    check = result['evidence']['diagnostic_checks'][0]
    assert result['judgments']['support'] == {'status': 'unknown', 'probability': .61}
    assert check['metric'] == 'predicate_probability' and check['actual'] == .61
    assert check['threshold'] == .8 and check['negative_threshold'] == .2
    assert check['question'] == 'Is the support visible?'


def test_malformed_model_answer_cannot_leak_image_or_key_into_error_evidence():
    capture, detection = fixture()
    body = {'answers': [{'name': 'identity_0', 'type': 'choice', 'choice': 'red',
                        'confidence': 'secret-key', 'image_url': 'data:image/jpeg;base64,AAAA'},
                       {'name': 'visibility_red', 'type': 'choice', 'choice': 'visible', 'confidence': .99}]}
    inventory = VisualInventory(api_key='secret-key', client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body))))
    with pytest.raises(VisualInventoryError, match='visual_decisions_invalid_response') as caught:
        inventory.resolve(capture, [detection], instruction='move red', entities=fixed_entities())
    text = json.dumps(caught.value.evidence)
    assert 'secret-key' not in text and 'base64' not in text


def test_sam_transport_error_exposes_status_and_request_hash_without_request_image(monkeypatch):
    from contextlib import contextmanager

    from luna_mujoco.perception import Sam3Client

    @contextmanager
    def stream(*args, **kwargs):
        yield httpx.Response(503)

    monkeypatch.setattr(httpx, 'stream', stream)
    capture, _ = capture_fixture()
    with pytest.raises(PerceptionError, match='sam3_invalid_output') as caught:
        Sam3Client().segment(capture['views']['3']['rgb'], 'red package', 2000000)
    details = caught.value.details
    assert details['http_status'] == 503 and details['request_hash']
    assert details['component'] == 'sam3' and details['prompt'] == 'red package'
    assert 'rgb_jpeg' not in details


def test_invalid_choice_diagnostic_cannot_claim_known_from_high_confidence():
    from luna_mujoco.visual_api import decision_diagnostic_checks

    questions = [{'name': 'identity_0', 'type': 'choice', 'instructions': 'Which entity?',
                  'choices': [{'value': 'red'}, {'value': 'unknown'}]}]
    answers = [{'name': 'identity_0', 'type': 'choice', 'choice': 'not_an_option', 'confidence': .99}]
    check = decision_diagnostic_checks(questions, answers, .8)[0]
    assert check['status'] == 'unknown'
    assert check['reason'] == 'invalid_answer_choice'
    assert check['actual'] == .99 and check['answer'] == answers[0]


def test_successful_parse_sanitizes_evidence_before_failed_gate_and_cached_return():
    from test_agent_model_layer import Client, capture

    from luna_mujoco.agent.decisions import ChoiceRequest, DecisionsJudgments

    body = {'answers': [{'name': 'arm', 'type': 'choice', 'choice': 'B', 'confidence': .61,
                        'debug': 'secret-key', 'image_url': 'data:image/jpeg;base64,AAAA',
                        'note': 'provider image data:image/jpeg;base64,BBBB'}]}
    transport = Client([body])
    client = DecisionsJudgments(api_key='secret-key', client=transport)
    request = ChoiceRequest(name='arm', instructions='Pick an arm', choices=('A', 'B', 'unknown'))
    for _ in range(2):
        result = client.judge(capture(), requests=[request], context={})
        assert result['judgments']['arm'] == {'status': 'unknown', 'value': 'unknown', 'confidence': .61}
        encoded = json.dumps(result['evidence'])
        assert 'secret-key' not in encoded and 'base64' not in encoded
        assert result['evidence']['answers'][0]['choice'] == 'B'
        assert result['evidence']['answers'][0]['confidence'] == .61
    assert len(transport.requests) == 1


def test_nonfinite_response_remains_rejected_with_json_serializable_error_evidence():
    from test_agent_model_layer import capture

    from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest

    body = '{"answers":[{"name":"visible","type":"predicate","probability":NaN}]}'
    client = DecisionsJudgments(api_key='secret', client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=body))))
    with pytest.raises(VisualInventoryError, match='decisions_invalid_answer_probability') as caught:
        client.judge(capture(), requests=[JudgmentRequest(name='visible', instructions='Visible?')], context={})
    check = caught.value.evidence['diagnostic_checks'][0]
    assert check['status'] == 'unknown' and check['actual'] == 'NaN'
    assert check['answer']['probability'] == 'NaN'
    json.dumps(caught.value.evidence, allow_nan=False)

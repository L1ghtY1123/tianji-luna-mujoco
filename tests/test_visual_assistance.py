import json

import httpx
import pytest
from test_visual_scene_assistance import OBJECTS, capture, scene_reply

from luna_mujoco import visual_assistance


def test_disabled_assistance_never_sends():
    service = visual_assistance.VisualAssistance()
    assert service.inspect_scene(capture(), OBJECTS, 'Describe scene')['status'] == 'unknown'


def test_images_are_bounded_and_same_scene_capture_is_cached():
    calls = []
    def handle(request):
        payload = json.loads(request.content); calls.append(payload)
        return httpx.Response(200, json={'output': [{'content': [{'type': 'output_text', 'text': json.dumps(scene_reply())}]}],
            'usage': {'input_tokens': 42, 'output_tokens': 9, 'attribution': {'large_tree': 'x' * 10000}}})
    service = visual_assistance.VisualAssistance(enabled=True, model='configured-vision', api_key='test',
                                                client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = service.inspect_scene(capture(), OBJECTS, 'Describe scene', reference=capture('before'))
    assert result['status'] == 'observed'
    assert service.inspect_scene(capture(), OBJECTS, 'Describe scene', reference=capture('before')) == result
    assert len(calls) == 1
    images = [v for v in calls[0]['input'][0]['content'] if v['type'] == 'input_image']
    assert len(images) == 4 and calls[0]['store'] is False
    assert 'test' not in json.dumps(result)
    assert result['usage'] == {'input_tokens': 42, 'output_tokens': 9}


@pytest.mark.parametrize('error_type', [httpx.ReadTimeout, httpx.ConnectError])
def test_network_failure_is_unknown_and_not_retried(error_type):
    calls = []
    def handle(request):
        calls.append(1)
        raise error_type('private provider details')
    service = visual_assistance.VisualAssistance(enabled=True, model='configured-vision', api_key='test',
                                                client=httpx.Client(transport=httpx.MockTransport(handle)))
    first = service.inspect_scene(capture(), OBJECTS, 'Describe scene')
    assert first['status'] == 'unknown' and 'private' not in str(first)
    assert first['reason'] == error_type.__name__ and 'provider_error_code' not in first
    assert service.inspect_scene(capture(), OBJECTS, 'Describe scene') == first
    assert len(calls) == 1


@pytest.mark.parametrize('arrival_s,expected', [(65., 'observed'), (91., 'unknown')])
def test_nonstreaming_response_obeys_same_total_deadline(monkeypatch, arrival_s, expected):
    clock = [0.]
    monkeypatch.setattr(visual_assistance.time, 'monotonic', lambda: clock[0])

    def handle(request):
        clock[0] = arrival_s
        return httpx.Response(200, json={'status': 'completed', 'output': [{'content': [
            {'type': 'output_text', 'text': json.dumps(scene_reply())}]}]})

    service = visual_assistance.VisualAssistance(enabled=True, model='configured', api_key='test',
        client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = service.inspect_scene(capture(), OBJECTS, 'Describe scene')
    assert result['status'] == expected
    assert result['timeout_s'] == 90.
    if expected == 'unknown':
        assert result['reason'] == 'TimeoutError'


def test_responses_stream_uses_completed_event_and_low_reasoning():
    def handle(request):
        payload = json.loads(request.content)
        assert payload['stream'] is True and payload['reasoning'] == {'effort': 'low'}
        body = {'status': 'completed', 'output': [{'content': [{'type': 'output_text', 'text': json.dumps(scene_reply())}]}]}
        events = [{'type': 'response.created'}, {'type': 'codex.rate_limits'},
                  {'type': 'response.completed', 'response': body}]
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text=''.join('data: ' + json.dumps(event) + '\n\n' for event in events))
    service = visual_assistance.VisualAssistance(enabled=True, model='configured-vision', api_key='test',
                                                client=httpx.Client(transport=httpx.MockTransport(handle)))
    assert service.inspect_scene(capture(), OBJECTS, 'Describe scene')['status'] == 'observed'


def test_incomplete_response_never_accepts_partial_scene():
    def handle(request):
        body = {'status': 'incomplete', 'output': [{'content': [{'type': 'output_text', 'text': json.dumps(scene_reply())}]}]}
        return httpx.Response(200, json=body)
    service = visual_assistance.VisualAssistance(enabled=True, model='configured-vision', api_key='test',
                                                client=httpx.Client(transport=httpx.MockTransport(handle)))
    assert service.inspect_scene(capture(), OBJECTS, 'Describe scene')['status'] == 'unknown'

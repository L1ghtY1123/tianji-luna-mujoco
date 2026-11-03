import json

import httpx
import numpy as np
import pytest
from test_agent_model_layer import capture

from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest
from luna_mujoco.visual_api import VisualInventoryError


def test_decisions_transient_service_failure_retries_only_the_identical_request(monkeypatch):
    calls, events = [], []
    monkeypatch.setattr('luna_mujoco.visual_api.time.sleep', lambda _: None)

    def respond(request):
        calls.append(request.content)
        return httpx.Response(503, json={'error': {'type': 'server_error'}}) if len(calls) < 3 else (
            httpx.Response(200, json={'answers': [{'name': 'visible', 'type': 'predicate', 'probability': .99}]}))

    judge = DecisionsJudgments(api_key='test', audit=events.append,
        client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = judge.judge(capture(), requests=[JudgmentRequest(name='visible', instructions='Visible?')], context={})
    assert result['judgments']['visible']['status'] == 'yes'
    assert len(calls) == 3 and len(set(calls)) == 1
    assert [event['http_status'] for event in events] == [503, 503, 200]
    assert len({event['call_id'] for event in events}) == 3


def test_unknown_and_permanent_errors_never_become_retry_until_yes(monkeypatch):
    monkeypatch.setattr('luna_mujoco.visual_api.time.sleep', lambda _: None)
    for status in (200, 401):
        calls = []
        def respond(request, calls=calls, status=status):
            calls.append(request)
            return httpx.Response(status, json={'answers': [
                {'name': 'visible', 'type': 'predicate', 'probability': .6}]})
        judge = DecisionsJudgments(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
        if status == 200:
            result = judge.judge(capture(), requests=[JudgmentRequest(name='visible', instructions='Visible?')], context={})
            assert result['judgments']['visible']['status'] == 'unknown'
        else:
            with pytest.raises(VisualInventoryError, match='http_401'):
                judge.judge(capture(), requests=[JudgmentRequest(name='visible', instructions='Visible?')], context={})
        assert len(calls) == 1


def measured_release(x=.4):
    from test_demo_release_result import release_inputs
    current, binding, reference = release_inputs()
    xx, yy = np.meshgrid(np.linspace(x-.01, x+.01, 10), np.linspace(-.01, .01, 10))
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, .65)))
    measured = {'observation_id': current['observation_id'], 'generation': current['generation'],
        'objects': [{'id': binding['target_id'], 'points_world_m': points}],
        'visual_inventory': {'evidence': {'source': 'openai_decisions', 'request_hash': 'identity-proof'}}}
    return current, binding, reference, measured


def test_operator_square_size_is_persisted_and_used_by_capability(tmp_path):
    from luna_mujoco.planning_settings import PlanningSettings, load_settings, save_settings
    from luna_mujoco.vla_configuration import vla_unpack_configuration
    settings = PlanningSettings(vla_unpack={'input_x_m': .45, 'input_y_m': .1, 'input_size_m': .16})
    save_settings(tmp_path/'settings.json', settings)
    region = vla_unpack_configuration(load_settings(tmp_path/'settings.json'))['input_region']
    assert region['size_m'] == .16 and region['physical_marker'] is False
    np.testing.assert_allclose(region['region_polygon_world_xy_m'], [[.37,.02],[.53,.02],[.53,.18],[.37,.18]])


@pytest.mark.parametrize('x', [.4, .55])
def test_release_skill_uses_visual_result_without_post_release_metric_measurement(x):
    from luna_mujoco.demo_perception import DemoPerception, DemoVisualInventory
    current, binding, reference, _ = measured_release(x)
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={'answers': [{'type': 'predicate', 'name': q['name'],
            'probability': .01 if q['name'] == 'workpiece_still_held' else .99} for q in payload['questions']]})

    engine = DemoPerception(detector=object(), visual_inventory=DemoVisualInventory(api_key='test',
        client=httpx.Client(transport=httpx.MockTransport(respond))))
    def observe(frame, **kwargs):
        raise AssertionError('qualitative release verification must not call observe')
    engine.observe = observe
    result = engine.verify_release(current, binding, reference)
    assert result['visual_evidence']['status'] == 'at_destination'
    assert 'measured_object_position' not in result['visual_evidence']
    assert result['visual_evidence']['post_release_metric_pose_measured'] is False
    assert result['geometry_updated'] is False
    assert len(calls) == 1

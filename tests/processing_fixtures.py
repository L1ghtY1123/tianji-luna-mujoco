"""Current RGB and verified-release fixtures for independent skill tests."""
import json
from copy import deepcopy

import httpx
from decision_fixtures import decision_answer
from test_demo_partial_identity import inputs

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.robot_skills.evidence import table_release_reference


def work_area():
    return {'center_world_xy_m': [.36, 0.],
            'region_polygon_world_xy_m': [[.18, -.18], [.54, -.18], [.54, .18], [.18, .18]],
            'floor_z_m': .56, 'source': 'configured_xy_measured_support'}


def capture_and_action(source='mock_vla'):
    capture, _, _, _ = inputs()
    capture.update(observation_id='after', generation=3, time=2., captured_perf_counter_ns=200)
    capture['views']['3'].update(T_world_camera=[[1, 0, 0, .36], [0, -1, 0, 0],
        [0, 0, -1, 1.56], [0, 0, 0, 1]], intrinsics={'fx': 30., 'fy': 30., 'cx': 16., 'cy': 12.})
    action = {'source': source, 'owner_id': 'run', 'target_id': 'pink', 'generation': 3,
              'action_id': 'action-1', 'completed': True, 'completed_time': 1.,
              'completed_perf_counter_ns': 100, 'table_release_feedback_id': 'table-release'}
    return capture, action


def client(answers=None):
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        rows = []
        for question in payload['questions']:
            negative = question['name'].endswith('_conflict') or question['name'] == 'workpiece_still_held'
            value, confidence = (answers or {}).get(question['name'], ('no' if negative else 'yes', .95))
            rows.append(decision_answer(question, value, confidence))
        return httpx.Response(200, json={'model': 'gpt-6-luna', 'answers': rows})

    return DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond))), calls




def release_evidence():
    return {'target_id': 'pink', 'status': 'at_destination', 'confidence': .95,
        'identity_source': 'execution_chain', 'release_feedback_id': 'table-release',
        'binding_id': 'table-binding', 'request_hash': 'table-request',
        'observation_id': 'table-before', 'source': 'openai_decisions'}


def before_kwargs(capture=None, release=None, *, owner='run', action=None):
    current, default_action = capture_and_action()
    capture, action = capture or current, action or default_action
    release = release or release_evidence()
    before = deepcopy(capture)
    before.update(observation_id='table-before', time=min(capture['time'], action['completed_time'])-.5,
                  captured_perf_counter_ns=action['completed_perf_counter_ns']-1)
    return {'active_target_ids': ['pink'], 'table_release_capture': before,
        'table_release_reference': table_release_reference(before, release, owner)}

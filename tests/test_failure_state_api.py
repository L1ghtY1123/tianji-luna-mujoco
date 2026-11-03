"""A solver error carrying replay arrays must not break HTTP or live updates."""
import numpy as np
from fastapi.testclient import TestClient
from test_api import Algorithms, Simulation

from luna_mujoco.app import create_app


def test_solver_replay_failure_is_serializable_without_losing_report_or_graph(tmp_path):
    app = create_app(Simulation(), Algorithms(), None, tmp_path)
    replay = {'candidate_replay': {'measured_points_world_m': np.zeros((646, 3)),
                                  'T_world_cloud': np.eye(4),
                                  'candidates_before_ik': [{'index': i} for i in range(300)]}}
    report = {'code': 'grasp_no_downward_candidate', 'checks': [{'name': str(i)} for i in range(24)]}
    graph = {'steps': [{'id': str(i)} for i in range(30)]}
    with TestClient(app) as client:
        runner = app.state.runner
        runner.state.update(status='needs_attention', error_details=replay,
                            policy_failure={'details': replay}, failure_report=report, policy_graph=graph)
        response = client.get('/api/state')
        assert response.status_code == 200
        state = response.json()['run']
        assert state['failure_report'] == report and state['policy_graph'] == graph
        for value in (state['error_details'], state['policy_failure']['details']):
            assert value['candidate_replay']['T_world_cloud'] == np.eye(4).tolist()
            assert value['candidate_replay']['candidates_before_ik']['count'] == 300
            assert value['candidate_replay']['measured_points_world_m']['count'] == 646
        with client.websocket_connect('/ws') as socket:
            packet = socket.receive_json()
            assert packet['run']['error_details'] == state['error_details']
        # Publishing does not destroy replay evidence retained by the runner.
        assert isinstance(runner.state['error_details']['candidate_replay']['measured_points_world_m'], np.ndarray)

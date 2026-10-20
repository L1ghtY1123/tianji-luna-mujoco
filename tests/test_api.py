import asyncio

import numpy as np
from fastapi.testclient import TestClient

from jev_mujoco.app import create_app


class Simulation:
    def __init__(self):
        self.generation = 1
        self.confirmed = False

    def start(self):
        pass

    def close(self):
        pass

    def snapshot(self):
        return {"generation": self.generation, "confirmed": self.confirmed, "time": 1.,
                "objects": [], "arms": {arm: {"tcp": np.eye(4).tolist()} for arm in ("A", "B")}}

    def robot_snapshot(self):
        return {"generation": self.generation, "confirmed": self.confirmed, "time": 1.,
                "arms": {arm: {"tcp": np.eye(4).tolist()} for arm in ("A", "B")}}

    def confirm(self):
        self.confirmed = True
        return self.snapshot()

    def reset(self):
        self.generation += 1
        return self.snapshot()

    def configure(self, scene):
        self.last_scene = scene
        self.generation += 1
        self.confirmed = False
        return self.snapshot()


class Algorithms:
    def health(self):
        return {"graspgen": {"ready": False}, "curobo": {"ready": False}}

    def clear_carrying(self):
        return {"ok": True, "attached": False}


def test_workspace_view_is_read_only_and_reports_missing_atlas(tmp_path, monkeypatch):
    from jev_mujoco.workspace import WorkspaceAtlas
    payload = {'source': 'offline_single_arm_workspace', 'arms': []}
    monkeypatch.setattr(WorkspaceAtlas, 'preview', lambda self: payload)
    simulation = Simulation()
    with TestClient(create_app(simulation, Algorithms(), lambda record: None, tmp_path)) as client:
        assert client.get('/api/workspaces').json() == payload
        scene = client.get('/api/workspace-scene')
        assert scene.status_code == 200
        assert set(scene.json()) == {'simulation', 'server_instance_id'}
        assert scene.json()['simulation'] == simulation.snapshot()
        assert client.post('/api/workspaces').status_code == 405
        assert simulation.generation == 1
        def missing(self):
            raise ValueError('workspace_file_missing:arm_A.npz')
        monkeypatch.setattr(WorkspaceAtlas, 'preview', missing)
        response = client.get('/api/workspaces')
        assert response.status_code == 503
        assert 'workspace_file_missing' in response.json()['detail']


def test_http_lifecycle_and_missing_key_error_is_recorded(tmp_path):
    def no_model(record):
        raise ValueError("TYPESAFE_API_KEY_not_configured")
    app = create_app(Simulation(), Algorithms(), no_model, tmp_path)
    with TestClient(app) as client:
        assert client.post("/api/start", json={"instruction": "抓取黑色物体"}).status_code == 409
        assert client.post("/api/confirm").status_code == 200
        assert client.post("/api/start", json={"instruction": "抓取黑色物体"}).status_code == 200
        state = client.get("/api/state").json()
        assert state["run"]["status"] == "failed"
        assert state["run"]["error"] == "TYPESAFE_API_KEY_not_configured"
        assert client.post("/api/reset").json()["generation"] == 2
        reset = client.get("/api/state").json()
        assert reset["run"]["status"] == "idle"
        assert reset["events"] == []


def test_websocket_initial_packet_and_reset_event_mode(tmp_path):
    app = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path)
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        packet = socket.receive_json()
        assert packet["events_mode"] == "replace"
        assert packet["simulation"]["time"] == 1.
        assert packet["server_time"] > 0


def test_websocket_does_not_copy_events_already_delivered(tmp_path):
    copies = []

    class RecordedEvent(dict):
        def __deepcopy__(self, memo):
            copies.append(self['id'])
            return dict(self)

    app = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path)
    app.state.runner.events = [RecordedEvent(id=1, kind='jev', data={'output': 'observe_scene'})]
    with TestClient(app) as client, client.websocket_connect('/ws') as socket:
        assert socket.receive_json()['events'][0]['id'] == 1
        assert socket.receive_json()['events'] == []
        assert socket.receive_json()['events'] == []
        assert copies == [1]
        app.state.runner.events.append(RecordedEvent(id=2, kind='skill_start', data={}))
        for _ in range(10):
            packet = socket.receive_json()
            if packet['events']:
                break
        assert [event['id'] for event in packet['events']] == [2]
        assert copies == [1, 2]
        assert len(client.get('/api/state').json()['events']) == 2
        assert copies == [1, 2, 1, 2]


def test_server_instance_identity_is_shared_by_http_and_websocket_and_changes_on_restart(tmp_path):
    from uuid import UUID

    instance_ids = []
    for _ in range(2):
        app = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path)
        with TestClient(app) as client:
            instance_id = client.get("/api/state").json()["server_instance_id"]
            assert UUID(instance_id).version == 4
            with client.websocket_connect("/ws") as socket:
                assert socket.receive_json()["server_instance_id"] == instance_id
            client.post("/api/reset").raise_for_status()
            assert client.get("/api/state").json()["server_instance_id"] == instance_id
            instance_ids.append(instance_id)
    assert instance_ids[0] != instance_ids[1]


def test_only_scene_apply_commits_edits_and_clears_previous_run(tmp_path):
    simulation = Simulation()
    app = create_app(simulation, Algorithms(), lambda record: None, tmp_path)
    with TestClient(app) as client:
        app.state.runner.state["status"] = "succeeded"
        scene = {"objects": [{"id": "cube", "position": [.4, .1, .7]}]}
        assert client.post("/api/scene/objects/cube/position", json={"position": [.4, .1, .7], "generation": 1}).status_code == 404
        assert simulation.generation == 1
        result = client.post("/api/scene", json=scene)
        assert result.status_code == 200
        assert simulation.last_scene == scene
        assert simulation.generation == 2
        assert client.get("/api/state").json()["run"]["status"] == "idle"


def test_scene_apply_is_rejected_while_task_is_waiting_for_jev(tmp_path):
    class WaitingDecisions:
        async def bind_request(self, instruction):
            await asyncio.Event().wait()

    simulation = Simulation()
    app = create_app(simulation, Algorithms(), lambda record: WaitingDecisions(), tmp_path)
    with TestClient(app) as client:
        client.post("/api/confirm").raise_for_status()
        client.post("/api/start", json={"instruction": "抓取黑色物体"}).raise_for_status()
        result = client.post("/api/scene", json={"objects": [{"id": "cube", "position": [.4, .1, .7]}]})
        assert result.status_code == 409
        assert result.json()["detail"] == "run_active_reset_first"
        assert not hasattr(simulation, "last_scene")
        client.post("/api/reset").raise_for_status()


def test_start_sent_before_reset_cannot_restart_the_reset_scene(tmp_path):
    app = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path)
    with TestClient(app) as client:
        client.post("/api/confirm").raise_for_status()
        old_generation = client.get("/api/state").json()["simulation"]["generation"]
        client.post("/api/reset").raise_for_status()
        result = client.post("/api/start", json={"instruction": "抓取黑色物体", "generation": old_generation})
        assert result.status_code == 409
        assert result.json()["detail"] == "stale_start_after_reset"
        state = client.get("/api/state").json()
        assert state["run"]["status"] == "idle"
        assert state["events"] == []


def test_reset_reports_pending_planner_and_rejects_overlapping_start(tmp_path):
    from test_runner import Decisions, Sim
    from test_runner_failures import WaitingPlanner

    class Scene(Sim):
        def start(self):
            pass

        def close(self):
            pass

        def reset(self):
            self.state["generation"] += 1
            return self.snapshot()

    algorithms = WaitingPlanner()
    app = create_app(Scene(), algorithms, lambda record: Decisions(), tmp_path)
    with TestClient(app) as client:
        try:
            client.post("/api/start", json={"instruction": "抓取黑色物体"}).raise_for_status()
            assert algorithms.entered.wait(2)
            state = client.get("/api/state").json()
            assert state["run"]["current_step"]["skill"] == "plan_grasp"
            assert state["run"]["algorithm_pending"]
            client.post("/api/reset").raise_for_status()
            reset = client.get("/api/state").json()
            assert reset["run"]["status"] == "idle"
            assert reset["run"]["algorithm_pending"]
            assert reset["events"] == []
            start = client.post("/api/start", json={"instruction": "抓取黑色物体"})
            assert start.status_code == 409
            assert start.json()["detail"] == "algorithm_request_pending"
        finally:
            algorithms.release.set()


def test_reset_clears_remote_attachment_without_blocking_physics_or_allowing_a_new_run(tmp_path):
    import threading

    class HeldAlgorithms(Algorithms):
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.attached = True

        def clear_carrying(self):
            self.entered.set()
            assert self.release.wait(5)
            self.attached = False
            return {"ok": True, "attached": False}

    algorithms = HeldAlgorithms()
    app = create_app(Simulation(), algorithms, lambda record: None, tmp_path)
    with TestClient(app) as client:
        try:
            client.post("/api/confirm").raise_for_status()
            assert client.post("/api/reset").json()["generation"] == 2
            assert algorithms.entered.wait(1)
            assert client.get("/api/state").json()["run"]["algorithm_pending"]
            response = client.post("/api/start", json={"instruction": "抓取红色方块"})
            assert response.status_code == 409
            assert response.json()["detail"] == "algorithm_request_pending"
        finally:
            algorithms.release.set()
            client.portal.call(lambda: app.state.runner.algorithm_cleanup_worker)
        assert not algorithms.attached
        assert not client.get("/api/state").json()["run"]["algorithm_pending"]


def test_state_publishes_bounded_holding_evidence_without_cloud_or_capture_payloads(tmp_path):
    app = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path)
    with TestClient(app) as client:
        app.state.runner.holding_state = {
            "state": "held", "arm": "A", "verified": True, "attached": True,
            "points_tool_m": [[1, 2, 3]] * 100, "observation": {"views": {"rgb": [1] * 100}},
            "verification": {"status": "held", "success": True},
            "attachment_evidence": {"prepare_wall_ms": 12.5},
        }
        run = client.get("/api/state").json()["run"]
        assert run["holding"] == {
            "state": "held", "arm": "A", "verified": True, "attached": True,
            "verification": {"status": "held", "success": True},
            "attachment_evidence": {"prepare_wall_ms": 12.5},
        }

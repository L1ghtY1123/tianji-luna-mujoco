"""API locks are exercised with the current program runner actually in flight."""
from fastapi.testclient import TestClient
from test_demo_async_safety import RecordedRobot, WaitingVision

from luna_mujoco.app import create_app


class ApiRobot(RecordedRobot):
    def start(self):
        pass

    def close(self):
        pass

    def snapshot(self):
        return self.robot_snapshot() | {"objects": []}

    def configure(self, scene, *, initial_joints=None):
        self.configured_scene = scene
        self.generation += 1
        return self.snapshot()

    def reset(self):
        self.generation += 1
        self.joints = [0.] * 7
        self.peer_joints = [0.] * 7
        return self.snapshot()


def make_app(tmp_path, monkeypatch):
    monkeypatch.setattr("luna_mujoco.demo_scene.demo_observation_poses",
                        lambda: {"A": [0.] * 7, "B": [0.] * 7})
    robot, algorithms = ApiRobot(), WaitingVision("plan")
    app = create_app(robot, algorithms, data_dir=tmp_path / "runs")
    return app, robot, algorithms


def start_request(client):
    response = client.post("/api/start", json={
        "instruction": "把一个粉色物体放到指定区域",
        "experiment": {"save_mode": "none", "record_video": False}})
    response.raise_for_status()
    return response


def test_current_demo_run_blocks_scene_and_settings_changes(tmp_path, monkeypatch):
    app, robot, algorithms = make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        try:
            start_request(client)
            assert algorithms.entered.wait(2)
            settings = client.get("/api/planning-settings").json()
            scene = client.post("/api/scene", json={"objects": []})
            assert scene.status_code == 409
            assert scene.json()["detail"] == "run_active_reset_first"
            changed = client.post("/api/planning-settings", json={
                "settings": {**settings["settings"], "time_scale": 4},
                "expected_revision": settings["revision"]})
            assert changed.status_code == 409
            assert client.get("/api/planning-settings").json() == settings
            assert not hasattr(robot, "configured_scene")
            assert client.get("/api/state").json()["run"]["status"] == "running"
            assert client.post("/api/reset").status_code == 200
        finally:
            algorithms.release.set()
    assert robot.motor_commands == []


def test_demo_reset_keeps_pending_request_locked_and_rejects_stale_start(tmp_path, monkeypatch):
    app, robot, algorithms = make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        try:
            start_request(client)
            assert algorithms.entered.wait(2)
            old_generation = robot.generation
            assert client.post("/api/reset").status_code == 200
            assert algorithms.cancelled.is_set()
            state = client.get("/api/state").json()
            assert state["run"]["status"] == "idle"
            assert state["run"]["algorithm_pending"] is True
            late_start = client.post("/api/start", json={"instruction": "搬运一个"})
            assert late_start.status_code == 409
            assert late_start.json()["detail"] == "algorithm_request_pending"
            stale_start = client.post("/api/start", json={
                "instruction": "搬运一个", "generation": old_generation})
            assert stale_start.status_code == 409
            assert stale_start.json()["detail"] == "stale_start_after_reset"
        finally:
            algorithms.release.set()
    assert robot.motor_commands == []
    assert app.state.runner.events == []

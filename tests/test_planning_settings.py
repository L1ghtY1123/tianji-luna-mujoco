import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_api import Algorithms, Simulation

from jev_mujoco.app import create_app
from jev_mujoco.planning_settings import PlanningSettings


@pytest.mark.parametrize("changes", [
    {"trajectory_collision_clearance_m": .04},
    {"approach_offset_m": 0}, {"lift_offset_m": float("nan")},
    {"ik_num_seeds": 0}, {"ik_num_seeds": 1.5},
    {"time_scale": .5}, {"finetune_dt_scale": 0}, {"unknown": 1},
    {"ik_num_seeds": 1, "trajopt_num_seeds": 2},
    {"ik_num_seeds": 33, "ik_seed_solver_num_seeds": 32},
])
def test_invalid_or_inconsistent_settings_rejected(changes):
    with pytest.raises(ValidationError):
        PlanningSettings(**changes)


def test_settings_apply_persist_and_reset_preserves_algorithm_configuration(tmp_path):
    app = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path / "runs")
    with TestClient(app) as client:
        original = client.get("/api/planning-settings").json()
        values = {**original["settings"], "time_scale": 4, "ik_num_seeds": 24, "plan_timeout_s": 360}
        response = client.post("/api/planning-settings", json={
            "settings": values, "expected_revision": original["revision"]})
        response.raise_for_status()
        applied = response.json()
        assert applied["settings"] == values
        assert app.state.runner.plan_timeout_s == 360
        assert applied["revision"] != original["revision"]
        assert client.get("/api/state").json()["planning_settings"]["revision"] == applied["revision"]
        client.post("/api/reset").raise_for_status()
        assert client.get("/api/planning-settings").json()["settings"] == values
        stale = client.post("/api/planning-settings", json={
            "settings": original["settings"], "expected_revision": original["revision"]})
        assert stale.status_code == 409
        stale_start = client.post("/api/start", json={
            "instruction": "抓取黑色方块", "settings_revision": original["revision"]})
        assert stale_start.status_code == 409
        assert stale_start.json()["detail"] == "planning_settings_changed"
    restarted = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path / "runs")
    with TestClient(restarted) as client:
        assert client.get("/api/planning-settings").json()["settings"] == values


def test_settings_cannot_change_during_run_and_snapshot_is_recorded(tmp_path):
    class Waiting:
        async def bind_request(self, instruction):
            await asyncio.Event().wait()

    app = create_app(Simulation(), Algorithms(), lambda record: Waiting(), tmp_path / "runs")
    with TestClient(app) as client:
        config = client.get("/api/planning-settings").json()
        client.post("/api/confirm").raise_for_status()
        client.post("/api/start", json={"instruction": "抓取", "settings_revision": config["revision"]}).raise_for_status()
        result = client.post("/api/planning-settings", json={
            "settings": {**config["settings"], "time_scale": 4}, "expected_revision": config["revision"]})
        assert result.status_code == 409
        state = client.get("/api/state").json()
        assert state["run"]["planning_settings"] == config["settings"]
        assert state["run"]["settings_revision"] == config["revision"]
        assert state["events"][0]["data"]["planning_settings"] == config["settings"]
        client.post("/api/reset").raise_for_status()


def test_failed_settings_save_does_not_change_running_configuration(tmp_path, monkeypatch):
    app = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path / "runs")
    with TestClient(app) as client:
        original = client.get("/api/planning-settings").json()
        def fail_save(*args):
            raise OSError("disk full")
        monkeypatch.setattr("jev_mujoco.app.save_settings", fail_save)
        response = client.post("/api/planning-settings", json={
            "settings": {**original["settings"], "time_scale": 4}, "expected_revision": original["revision"]})
        assert response.status_code == 503
        assert client.get("/api/planning-settings").json()["revision"] == original["revision"]


import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_api import Algorithms, Simulation

from luna_mujoco.app import create_app
from luna_mujoco.planning_settings import PlanningSettings, load_settings


@pytest.mark.parametrize("old_clearance", [.005, .0075, .00999])
def test_only_saved_legacy_clearance_migrates_explicitly_to_ten_mm(tmp_path, old_clearance):
    path = tmp_path / "planning-settings.json"
    saved = {"trajectory_collision_clearance_m": old_clearance,
             "place_release_height_m": old_clearance, "time_scale": 4.,
             "ik_position_threshold_m": .005, "attachment": {"sphere_budget": 24}}
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.warns(UserWarning, match="clearance.*migrated.*0.01"):
        settings = load_settings(path)
    assert settings.trajectory_collision_clearance_m == .01
    assert settings.place_release_height_m == .01
    assert settings.time_scale == 4.
    assert settings.ik_position_threshold_m == .005
    assert settings.attachment.sphere_budget == 24
    assert json.loads(path.read_text()) == saved
    with pytest.raises(ValidationError):
        PlanningSettings.model_validate(saved)


def test_saved_clearance_above_policy_is_preserved_and_invalid_old_value_rejected(tmp_path):
    path = tmp_path / "planning-settings.json"
    path.write_text(json.dumps({"trajectory_collision_clearance_m": .015}), encoding="utf-8")
    assert load_settings(path).place_release_height_m == .015
    path.write_text(json.dumps({"trajectory_collision_clearance_m": .0049}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(path)


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
        values = {**original["settings"], "time_scale": 4, "ik_num_seeds": 24}
        response = client.post("/api/planning-settings", json={
            "settings": values, "expected_revision": original["revision"]})
        response.raise_for_status()
        applied = response.json()
        assert applied["settings"] == values
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




def test_failed_settings_save_does_not_change_running_configuration(tmp_path, monkeypatch):
    app = create_app(Simulation(), Algorithms(), lambda record: None, tmp_path / "runs")
    with TestClient(app) as client:
        original = client.get("/api/planning-settings").json()
        def fail_save(*args):
            raise OSError("disk full")
        monkeypatch.setattr("luna_mujoco.app.save_settings", fail_save)
        response = client.post("/api/planning-settings", json={
            "settings": {**original["settings"], "time_scale": 4}, "expected_revision": original["revision"]})
        assert response.status_code == 503
        assert client.get("/api/planning-settings").json()["revision"] == original["revision"]

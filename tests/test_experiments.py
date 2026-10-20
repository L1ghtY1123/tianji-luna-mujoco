import json
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import pytest

from jev_mujoco.experiments import ExperimentOptions, ExperimentStore


def begin(store, mode="auto", name="实验一"):
    return store.start(uuid4().hex, "抓红色到蓝盒", ExperimentOptions(
        name=name, record_video=False, save_mode=mode), {"planning_settings": {"time_scale": 3}})


def test_exact_arrays_events_and_settings_survive_archive(tmp_path):
    store = ExperimentStore(tmp_path)
    session = begin(store)
    points = np.arange(60, dtype=np.float32).reshape(20, 3)
    session.event({"id": 1, "kind": "skill_complete", "title": "observe", "t_s": .25,
                   "invocation_id": "skill-1", "data": {"count": 20}}, {"points": points})
    points[:] = -1  # The archive must own the instant-of-event data.
    session.finish({"status": "failed", "error": "test failure"})
    manifest = store.get(session.id)
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "failed"
    assert manifest["video_status"] == "disabled"
    assert manifest["size_bytes"] > 0
    event = store.events(session.id)[0]
    assert event["t_s"] == .25 and event["invocation_id"] == "skill-1"
    full = json.loads(store.file(session.id, event["artifact"]).read_text())
    ref = full["points"]
    arrays = np.load(store.file(session.id, ref["path"]), allow_pickle=False)
    np.testing.assert_array_equal(arrays[ref["key"]], np.arange(60, dtype=np.float32).reshape(20, 3))
    config = json.loads(store.file(session.id, "config.json").read_text())
    assert config["planning_settings"]["time_scale"] == 3


def test_pending_save_rename_and_discard_are_independent(tmp_path):
    store = ExperimentStore(tmp_path)
    first, second = begin(store, "ask"), begin(store, "ask")
    first.finish({"status": "cancelled"})
    second.finish({"status": "succeeded"})
    store.save(first.id, "左臂/测试")
    assert store.get(first.id)["retention"] == "saved"
    assert store.get(first.id)["name"] == "左臂/测试"
    assert first.summary()["retention"] == "saved"
    assert first.summary()["name"] == "左臂/测试"
    store.discard(second.id)
    assert [item["id"] for item in store.list()] == [first.id]
    with pytest.raises(ValueError, match="saved_experiment"):
        store.discard(first.id)


def test_no_save_creates_no_experiment(tmp_path):
    store = ExperimentStore(tmp_path)
    assert begin(store, "none") is None
    assert store.list() == []
    assert not list(tmp_path.iterdir())


def test_observation_annotation_archive_retains_immutable_frame_status_and_rgb(tmp_path):
    store = ExperimentStore(tmp_path)
    session = begin(store)
    capture = {'views': {'3': {'rgb': np.zeros((48, 64, 3), np.uint8)}}}
    labels = {'observation_id': 'frame1', 'generation': 1, 'analysis_status': 'unprocessed', 'items': []}
    paths = session.observation_images('observations/labels-1', capture, labels)
    labels['observation_id'] = 'later'
    session.finish({'status': 'succeeded'})
    text = store.file(session.id, paths['3']).read_text(encoding='utf-8')
    assert 'frame1' in text and '未分析' in text and 'data:image/png;base64,' in text
    assert 'later' not in text


def test_paths_and_export_cannot_escape_experiment(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    session = begin(store)
    session.finish({"status": "succeeded"})
    for path in ("../outside.txt", "/etc/passwd", "C:\\secret", "..\\outside.txt"):
        with pytest.raises(ValueError):
            store.file(session.id, path)
    with pytest.raises(ValueError):
        store.get("../../etc")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    try:
        (store.file(session.id, "manifest.json").parent / "escape.json").symlink_to(outside)
    except OSError:
        return  # Windows may not grant symlink creation to this process.
    with pytest.raises(ValueError):
        store.file(session.id, "escape.json")
    export = store.export(session.id, "full")
    try:
        with ZipFile(export) as package:
            assert "escape.json" not in package.namelist()
    finally:
        export.unlink()


def test_review_export_declares_omitted_raw_data(tmp_path):
    store = ExperimentStore(tmp_path)
    session = begin(store)
    session.artifact("observations/frame", {"depth_m": np.ones((3, 3))})
    session.finish({"status": "succeeded"})
    export = store.export(session.id, "review")
    try:
        with ZipFile(export) as package:
            assert not any(name.endswith(".npz") for name in package.namelist())
            manifest = json.loads(package.read("manifest.json"))
            assert manifest["export_mode"] == "review"
            assert any(name.endswith(".npz") for name in manifest["omitted_files"])
    finally:
        export.unlink()


def test_active_experiment_cannot_move_or_be_discarded(tmp_path):
    store = ExperimentStore(tmp_path)
    session = begin(store, "ask")
    try:
        with pytest.raises(ValueError, match="experiment_busy"):
            store.save(session.id)
        with pytest.raises(ValueError, match="experiment_busy"):
            store.discard(session.id)
    finally:
        session.finish({"status": "cancelled"})


@pytest.mark.asyncio
async def test_runner_failure_is_archived_with_causal_ids(tmp_path):
    from test_runner import Sim

    from jev_mujoco.runner import Runner

    class Decisions:
        async def choose_visual_plan(self, state, skill, requests):
            return dict(requests)

        async def bind_request(self, instruction):
            return {"task": "grasp", "arm": "auto", "scope": "single",
                    "color": "red", "kind": "cube", "position": "any"}

        async def choose_skill(self, state, registry):
            return "recover_task" if state["facts"].get("recovery_needed") else "observe_scene"

        async def choose_recovery(self, state):
            return "stop"

    class Algorithms:
        def observe(self, *args, **kwargs):
            raise ValueError("recorded_perception_failure")

    store = ExperimentStore(tmp_path / "experiments")
    runner = Runner(Sim(), Algorithms(), lambda record: Decisions(), tmp_path / "runs", experiments=store)
    await runner.start("抓取红色方块", experiment=ExperimentOptions(record_video=False))
    await runner.worker
    store.close()
    saved = store.get(runner.run_id)
    events = store.events(runner.run_id)
    assert saved["status"] == "failed"
    start = next(e for e in events if e["kind"] == "skill_start")
    failure = next(e for e in events if e["kind"] == "skill_error")
    assert start["invocation_id"] == failure["invocation_id"]
    assert start["invocation_id"]
    assert events[-1]["kind"] == "run_end"
    assert any(item["path"].startswith("observations/") for item in saved["files"])


def test_http_saved_archive_supports_video_range_and_pending_actions(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from jev_mujoco.experiment_api import experiment_router

    store = ExperimentStore(tmp_path)
    session = begin(store, "ask")
    session.finish({"status": "succeeded"})
    (session.directory / "videos").mkdir()
    (session.directory / "videos/scene.mp4").write_bytes(bytes(range(100)))
    app = FastAPI()
    app.include_router(experiment_router(store))
    with TestClient(app) as client:
        base = f"/api/experiments/{session.id}"
        response = client.get(base + "/files/videos/scene.mp4", headers={"Range": "bytes=10-19"})
        assert response.status_code == 206 and response.content == bytes(range(10, 20))
        assert client.post(base + "/save", json={"name": "成功样本"}).json()["retention"] == "saved"
        deleted = client.delete(base)
        assert deleted.status_code == 200 and deleted.json()["recoverable"]
        assert client.get(base).status_code == 404


def test_disk_error_does_not_leave_video_sampler_running(tmp_path, monkeypatch):
    import jev_mujoco.experiments as module

    session = begin(ExperimentStore(tmp_path))

    class Video:
        running = True

        def stop(self):
            self.running = False

        def finish(self):
            assert not self.running
            (session.directory / "finished-video.txt").write_text("joined")
            return {"video_status": "ready", "errors": []}

    session.video = Video()
    original = module.archive_value

    def fail_result(directory, name, value):
        if name == "result":
            raise OSError("disk error")
        return original(directory, name, value)

    monkeypatch.setattr(module, "archive_value", fail_result)
    session.finish({"status": "failed"})
    assert not session.busy
    assert (session.directory / "finished-video.txt").read_text() == "joined"
    assert session.manifest["recording_complete"] is False


@pytest.mark.parametrize("failure_stage", ["config", "events"])
def test_initialization_failure_is_readable_terminal_archive(tmp_path, monkeypatch, failure_stage):
    import jev_mujoco.experiments as module

    store = ExperimentStore(tmp_path)

    def fail_config(*args):
        raise OSError("cannot write config")

    if failure_stage == "config":
        monkeypatch.setattr(module, "archive_value", fail_config)
    else:
        monkeypatch.setattr(module.Path, "touch", fail_config)
    with pytest.raises(OSError):
        begin(store)
    record = store.list()[0]
    assert record["status"] == "failed"
    assert record["finalized"] and record["errors"]
    assert store.events(record["id"]) == []


def test_robot_sampling_keeps_feedback_without_repeating_rolling_history(tmp_path):
    import threading

    sampled = threading.Event()

    class Simulation:
        calls = 0

        def robot_snapshot(self):
            self.calls += 1
            if self.calls > 1:
                sampled.set()
            return {"generation": 1, "time": 3.0, "arms": {"A": {
                "joints_rad": [0.1] * 7, "tcp": np.eye(4).tolist(),
                "gripper_position_rad": 0.4, "gripper_closed_command": True,
                "gripper_position_samples": [[2.0, 0.3], [3.0, 0.4]],
            }}}

    store = ExperimentStore(tmp_path)
    session = store.start(uuid4().hex, "test", ExperimentOptions(record_video=False), {}, Simulation())
    assert sampled.wait(2)
    session.finish({"status": "succeeded"})
    row = json.loads(store.file(session.id, "robot.jsonl").read_text().splitlines()[0])
    assert row["arms"]["A"]["gripper_position_rad"] == 0.4
    assert row["arms"]["A"]["joints_rad"] == [0.1] * 7
    assert "gripper_position_samples" not in row["arms"]["A"]

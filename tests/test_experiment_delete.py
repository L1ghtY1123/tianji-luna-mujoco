import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from jev_mujoco.experiments import ExperimentOptions, ExperimentStore


def recorded(store, mode="auto"):
    session = store.start(uuid4().hex, "test", ExperimentOptions(record_video=False, save_mode=mode), {})
    session.event({"id": 1, "kind": "run_end", "t_s": 1}, {"original": [1, 2, 3]})
    session.finish({"status": "succeeded"})
    return session


@pytest.mark.parametrize("mode", ["auto", "ask"])
def test_delete_moves_only_selected_archive_into_recoverable_trash(tmp_path, mode):
    store = ExperimentStore(tmp_path)
    first, second = recorded(store, mode), recorded(store)
    original = {p.relative_to(first.directory): p.read_bytes() for p in first.directory.rglob("*") if p.is_file()}
    result = store.delete(first.id)
    assert result["deleted"] and result["recoverable"]
    assert [item["id"] for item in store.list()] == [second.id]
    trash = tmp_path / ".trash" / result["trash_id"]
    assert json.loads((trash / "deletion.json").read_text())["id"] == first.id
    for name, content in original.items():
        assert (trash / "archive" / name).read_bytes() == content
    with pytest.raises(FileNotFoundError):
        store.file(first.id, "manifest.json")
    with pytest.raises(FileNotFoundError):
        store.delete(first.id)
    assert [item["id"] for item in ExperimentStore(tmp_path).list()] == [second.id]
    # Local recovery uses the recorded original location; no file content changes.
    original_directory = json.loads((trash / "deletion.json").read_text())["original_directory"]
    (trash / "archive").rename(tmp_path / original_directory)
    restored = ExperimentStore(tmp_path)
    assert restored.events(first.id)[0]["id"] == 1
    assert restored.get(first.id)["retention"] == ("pending" if mode == "ask" else "saved")


def test_delete_rejects_active_and_unregistered_unfinalized_archive(tmp_path):
    store = ExperimentStore(tmp_path)
    session = store.start(uuid4().hex, "test", ExperimentOptions(record_video=False), {})
    try:
        with pytest.raises(ValueError, match="experiment_busy"):
            store.delete(session.id)
        store.sessions.pop(session.id)
        with pytest.raises(ValueError, match="experiment_busy"):
            store.delete(session.id)
        assert session.directory.exists()
    finally:
        session.finish({"status": "cancelled"})


def test_delete_rejects_recording_until_video_finalize_returns(tmp_path):
    store = ExperimentStore(tmp_path)
    session = store.start(uuid4().hex, "test", ExperimentOptions(record_video=False), {})
    entered, release = threading.Event(), threading.Event()

    class Video:
        def stop(self):
            pass

        def finish(self):
            entered.set()
            assert release.wait(5)
            return {"video_status": "ready"}

    session.video = Video()
    session.stop({"status": "succeeded"})
    try:
        assert entered.wait(5)
        with pytest.raises(ValueError, match="experiment_busy"):
            store.delete(session.id)
    finally:
        release.set()
        store.close()
    assert store.delete(session.id)["deleted"]


def test_file_lease_prevents_delete_and_releases_after_failure(tmp_path):
    store = ExperimentStore(tmp_path)
    session = recorded(store)
    with pytest.raises(RuntimeError, match="reader interrupted"), store.read_file(session.id, "manifest.json") as path:
        assert path.read_bytes()
        with pytest.raises(ValueError, match="experiment_in_use"):
            store.delete(session.id)
        raise RuntimeError("reader interrupted")
    assert store.delete(session.id)["deleted"]


def test_delete_rejects_invalid_id_and_trash_symlink_without_touching_source(tmp_path):
    store = ExperimentStore(tmp_path / "records")
    session = recorded(store)
    for identifier in ("../outside", "A" * 32, session.id + "/.."):
        with pytest.raises(ValueError, match="invalid_experiment_id"):
            store.delete(identifier)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (store.root / ".trash").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="invalid_trash_directory"):
        store.delete(session.id)
    assert session.directory.exists()
    assert list(outside.iterdir()) == []


def test_export_holds_archive_until_package_is_complete(tmp_path, monkeypatch):
    from zipfile import ZipFile

    store = ExperimentStore(tmp_path)
    session = recorded(store)
    entered, release = threading.Event(), threading.Event()
    original = ZipFile.write

    def blocked_write(package, *args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(package, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "write", blocked_write)
    with ThreadPoolExecutor() as pool:
        export = pool.submit(store.export, session.id, "full")
        try:
            assert entered.wait(5)
            with pytest.raises(ValueError, match="experiment_in_use"):
                store.delete(session.id)
        finally:
            release.set()
        package = export.result(5)
    try:
        assert store.delete(session.id)["deleted"]
        with ZipFile(package) as archive:
            assert json.loads(archive.read("manifest.json"))["id"] == session.id
    finally:
        package.unlink()


def test_failed_rename_preserves_record(tmp_path, monkeypatch):
    from pathlib import Path

    store = ExperimentStore(tmp_path)
    session = recorded(store)
    original = Path.rename

    def fail_source(path, target):
        if path == session.directory:
            raise OSError("locked file")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail_source)
    with pytest.raises(OSError, match="locked file"):
        store.delete(session.id)
    assert store.get(session.id)["id"] == session.id
    assert list((tmp_path / ".trash").iterdir()) == []


@pytest.mark.asyncio
async def test_http_file_response_pins_archive_until_interrupted_send_ends(tmp_path):
    from jev_mujoco.experiment_api import experiment_router

    store = ExperimentStore(tmp_path)
    session = recorded(store)
    router = experiment_router(store)
    route = next(route for route in router.routes if route.path.endswith("/files/{relative:path}"))
    response = await route.endpoint(session.id, "manifest.json")

    async def send(message):
        with pytest.raises(ValueError, match="experiment_in_use"):
            store.delete(session.id)
        if message["type"] == "http.response.body":
            raise RuntimeError("disconnected")

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(RuntimeError, match="disconnected"):
        await response({"type": "http", "method": "GET", "headers": [], "extensions": {}}, receive, send)
    assert store.delete(session.id)["deleted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("send_error", [RuntimeError, asyncio.CancelledError])
async def test_http_export_pins_archive_through_transmission_and_cleans_interrupted_zip(tmp_path, send_error):
    from jev_mujoco.experiment_api import experiment_router

    store = ExperimentStore(tmp_path)
    session = recorded(store)
    router = experiment_router(store)
    route = next(route for route in router.routes if route.path.endswith("/export"))
    response = await route.endpoint(session.id, "full")

    async def send(message):
        with pytest.raises(ValueError, match="experiment_in_use"):
            store.delete(session.id)
        if message["type"] == "http.response.body":
            raise send_error("download interrupted")

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(send_error, match="download interrupted"):
        await response({"type": "http", "method": "GET", "headers": [], "extensions": {}}, receive, send)
    assert not response.path.exists()
    assert store.delete(session.id)["deleted"]


@pytest.mark.asyncio
async def test_cancelled_export_construction_cleans_package_after_writer_finishes(tmp_path, monkeypatch):
    from zipfile import ZipFile

    from jev_mujoco.experiment_api import experiment_router

    store = ExperimentStore(tmp_path)
    session = recorded(store)
    route = next(route for route in experiment_router(store).routes if route.path.endswith("/export"))
    response = await route.endpoint(session.id, "full")
    entered, release = threading.Event(), threading.Event()
    packages = []
    original_write, original_export = ZipFile.write, store.export

    def blocked_write(package, *args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_write(package, *args, **kwargs)

    def tracked_export(*args):
        package = original_export(*args)
        packages.append(package)
        return package

    async def send(message):
        pytest.fail("cancelled construction must not send a response")

    async def receive():
        return {"type": "http.disconnect"}

    monkeypatch.setattr(ZipFile, "write", blocked_write)
    monkeypatch.setattr(store, "export", tracked_export)
    request = asyncio.create_task(response(
        {"type": "http", "method": "GET", "headers": [], "extensions": {}}, receive, send))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        with pytest.raises(ValueError, match="experiment_in_use"):
            store.delete(session.id)
    finally:
        release.set()

    async def wait_until_cleaned():
        while not packages or packages[0].exists():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_until_cleaned(), 5)
    assert store.delete(session.id)["deleted"]

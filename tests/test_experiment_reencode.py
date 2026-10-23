"""Re-encoding changes video metadata, never the recorded task outcome or inputs."""
import importlib
import json

import pytest


def archive(tmp_path, *, errors=None, finalized=True):
    directory = tmp_path / ("a" * 32)
    (directory / "videos").mkdir(parents=True)
    previous_errors = ["scene encoding: RuntimeError: recorded geometry IDs do not match the rendering model"]
    recording = {"duration_s": 2.5, "streams": {"scene": {}, "cameras": {}},
                 "errors": list(errors or [])}
    manifest = {"id": directory.name, "status": "failed", "finalized": finalized,
                "record_video": True, "recording_complete": False, "video_status": "partial",
                "errors": ["archive_write:OSError", *recording["errors"], *previous_errors]}
    for name, value in [
        ("manifest.json", manifest), ("videos/recording.json", recording),
        ("video-metadata.json", {"video_status": "partial", "errors": [*recording["errors"], *previous_errors]}),
        ("videos/result.json", {"video_status": "partial", "errors": previous_errors}),
    ]:
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    (directory / "frames.jsonl.gz").write_bytes(b"original frozen geometry")
    (directory / "events.jsonl").write_bytes(b"original failed task events")
    (directory / "videos/encoding.log").write_text("original encoder evidence")
    return directory


def successful_encoder(directory):
    videos = {}
    for kind in ["scene", "cameras"]:
        relative = f"videos/{kind}.mp4"
        (directory / relative).write_bytes(b"unit test encoded output")
        videos[kind] = {"path": relative, "duration_s": 2.5}
    return {"video_status": "ready", "errors": [], "videos": videos}


def test_reencode_updates_video_metadata_preserving_failure_and_source_evidence(tmp_path, monkeypatch):
    repair = importlib.import_module("luna_mujoco.experiment_reencode")
    directory = archive(tmp_path)
    protected = {name: (directory / name).read_bytes()
                 for name in ["frames.jsonl.gz", "events.jsonl", "videos/recording.json"]}
    monkeypatch.setattr(repair, "encode_videos", successful_encoder)
    result = repair.reencode_archive(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    metadata = json.loads((directory / "video-metadata.json").read_text())
    assert result["video_status"] == manifest["video_status"] == metadata["video_status"] == "ready"
    assert manifest["status"] == "failed"
    assert manifest["errors"] == ["archive_write:OSError"]
    assert manifest["recording_complete"] is False
    assert {row["path"] for row in manifest["files"]} >= {"videos/scene.mp4", "videos/cameras.mp4"}
    for name, content in protected.items():
        assert (directory / name).read_bytes() == content
    history = directory / result["audit_path"]
    assert json.loads((history / "manifest-before.json").read_text())["video_status"] == "partial"
    assert (history / "encoding-before.log").read_text() == "original encoder evidence"


def test_reencode_keeps_capture_gaps_partial_even_after_both_encoders_succeed(tmp_path, monkeypatch):
    repair = importlib.import_module("luna_mujoco.experiment_reencode")
    directory = archive(tmp_path, errors=["scene capture gap of 5.00s exceeded 1.0s"])
    monkeypatch.setattr(repair, "encode_videos", successful_encoder)
    result = repair.reencode_archive(directory)
    assert result["video_status"] == "partial"
    assert "scene capture gap" in result["errors"][0]
    assert json.loads((directory / "manifest.json").read_text())["recording_complete"] is False


def test_failed_reencode_does_not_promote_video_or_task_status(tmp_path, monkeypatch):
    repair = importlib.import_module("luna_mujoco.experiment_reencode")
    directory = archive(tmp_path)
    monkeypatch.setattr(repair, "encode_videos",
                        lambda _: {"video_status": "failed", "errors": ["ffmpeg unavailable"], "videos": {}})
    result = repair.reencode_archive(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert result["video_status"] == manifest["video_status"] == "failed"
    assert manifest["status"] == "failed"
    assert manifest["errors"] == ["archive_write:OSError", "ffmpeg unavailable"]


def test_complete_video_archive_does_not_turn_failed_task_into_completed(tmp_path, monkeypatch):
    repair = importlib.import_module("luna_mujoco.experiment_reencode")
    directory = archive(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["errors"].remove("archive_write:OSError")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(repair, "encode_videos", successful_encoder)
    repair.reencode_archive(directory)
    repaired = json.loads((directory / "manifest.json").read_text())
    assert repaired["video_status"] == "ready"
    assert repaired["recording_complete"] is True
    assert repaired["status"] == "failed"


def test_reencode_refuses_unfinalized_archives_before_calling_encoder(tmp_path, monkeypatch):
    repair = importlib.import_module("luna_mujoco.experiment_reencode")
    directory = archive(tmp_path, finalized=False)
    monkeypatch.setattr(repair, "encode_videos", lambda _: pytest.fail("encoder must not start"))
    with pytest.raises(ValueError, match="finalized"):
        repair.reencode_archive(directory)

"""Offline repair of finalized video archives, preserving the recorded task result."""
import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .experiment_video import encode_videos
from .experiments import write_json


def _read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def reencode_archive(directory):
    """Run only on an idle, finalized archive; no live simulation or model calls."""
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    if (not manifest or manifest.get("finalized") is not True
            or manifest.get("status") == "running"
            or manifest.get("video_status") in {"recording", "encoding"}):
        raise ValueError("video repair requires an idle finalized experiment")
    if not manifest.get("record_video"):
        raise ValueError("experiment did not request video recording")
    recording = _read_json(directory / "videos/recording.json")
    if recording is None:
        raise ValueError("frozen videos/recording.json is missing")
    previous = _read_json(directory / "video-metadata.json", {})
    lock = directory / ".video-reencode.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("offline video re-encoding\n")
    try:
        audit = directory / "videos" / (
            "reencode-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        audit.mkdir()
        for source, backup in [
            ("manifest.json", "manifest-before.json"),
            ("video-metadata.json", "video-metadata-before.json"),
            ("videos/result.json", "result-before.json"),
            ("videos/encoding.log", "encoding-before.log"),
            ("videos/scene.ffmpeg.log", "scene-before.ffmpeg.log"),
            ("videos/cameras.ffmpeg.log", "cameras-before.ffmpeg.log"),
        ]:
            if (directory / source).is_file():
                shutil.copyfile(directory / source, audit / backup)
        try:
            encoded = encode_videos(directory)
        except (OSError, ValueError) as error:
            encoded = {"video_status": "failed", "videos": {},
                       "errors": [f"video re-encoding: {type(error).__name__}: {error}"]}
        capture_errors = list(recording.get("errors", []))
        errors = [*capture_errors, *encoded.get("errors", [])]
        status = encoded["video_status"]
        if status == "ready" and errors:
            status = "partial"
        result = {**encoded, "video_status": status, "errors": errors,
                  "duration_s": recording["duration_s"], "streams": recording["streams"],
                  "audit_path": audit.relative_to(directory).as_posix()}
        write_json(directory / "videos/result.json", encoded)
        write_json(directory / "video-metadata.json", result)
        write_json(audit / "result.json", {
            **result, "recorded_task_status": manifest["status"],
            "encoder_source_sha256": hashlib.sha256(
                Path(__file__).with_name("experiment_video.py").read_bytes()).hexdigest(),
        })
        # Replace only the previous video errors; unrelated archive errors remain.
        remaining = list(manifest.get("errors", []))
        for error in previous.get("errors", []):
            if error in remaining:
                remaining.remove(error)
        manifest.update(video_status=status, errors=[*remaining, *errors])
        manifest["recording_complete"] = not manifest["errors"]
        manifest["files"] = [
            {"path": path.relative_to(directory).as_posix(), "size_bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != manifest_path and path != lock
        ]
        write_json(manifest_path, manifest)
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path,
                        help="Finalized experiment directories; use while archive operations are idle.")
    args = parser.parse_args()
    failed = False
    for directory in args.directories:
        result = reencode_archive(directory)
        print(json.dumps({"directory": str(directory), **result}, ensure_ascii=False))
        failed |= result["video_status"] != "ready"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

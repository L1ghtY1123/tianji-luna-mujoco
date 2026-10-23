"""Versioned experiment archives, independent of the live controller and browser."""
import base64
import copy
import hashlib
import json
import queue
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ExperimentOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="", max_length=120)
    record_video: bool = True
    save_mode: Literal["auto", "ask", "none"] = "auto"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def archive_value(directory, name, value):
    """JSON structure plus non-pickle arrays. References are relative to the experiment."""
    arrays = {}
    array_path = name + ".npz"

    def convert(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                return convert(item.tolist())
            key = f"array_{len(arrays)}"
            arrays[key] = item
            return {"type": "ndarray", "path": array_path, "key": key,
                    "dtype": str(item.dtype), "shape": list(item.shape)}
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, dict):
            return {str(key): convert(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(child) for child in item]
        if isinstance(item, float) and not np.isfinite(item):
            return {"type": "nonfinite", "value": str(item)}
        if isinstance(item, Path):
            return str(item)
        return item

    converted = convert(value)
    path = directory / (name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez_compressed(directory / array_path, **arrays)
    write_json(path, converted)
    return name + ".json"


class ExperimentSession:
    def __init__(self, directory, manifest, simulation=None, video_factory=None):
        self.directory = directory
        self.id = manifest["id"]
        self.manifest = manifest
        self._visual_calls = []
        self.started_at = time.monotonic()
        self.simulation = simulation
        self.generation = None
        self.video = None
        self.error = None
        self._queue = queue.Queue(maxsize=128)
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._finish_lock = threading.Lock()
        self._writer = threading.Thread(target=self._write, name="experiment-writer", daemon=True)
        self._sampler = None
        self._finish_thread = None
        if simulation is not None:
            self.generation = simulation.robot_snapshot()["generation"]
        self._writer.start()
        if simulation is not None:
            self._sampler = threading.Thread(target=self._sample, name="experiment-robot", daemon=True)
            self._sampler.start()
        if manifest["record_video"] and simulation is not None:
            try:
                if video_factory is None:
                    from .experiment_video import VideoRecorder
                    video_factory = VideoRecorder
                self.video = video_factory(directory, simulation, self.started_at, self.generation)
                self.video.start()
            except Exception as error:  # noqa: BLE001 — recording errors never change robot decisions
                self._error("video_start", error)
                self.manifest["video_status"] = "failed"
        elif manifest["record_video"]:
            self.manifest["video_status"] = "unavailable"

    @property
    def busy(self):
        return not self._finished.is_set()

    def summary(self):
        return {key: self.manifest.get(key) for key in
                ("id", "name", "retention", "video_status", "status")} | {
                    "error": self.error or "; ".join(self.manifest["errors"]) or None}

    def _error(self, stage, error):
        message = stage + ":" + type(error).__name__
        self.error = message
        self.manifest["errors"].append(message)

    def _submit(self, kind, payload):
        if self._finished.is_set():
            return
        try:
            self._queue.put_nowait((kind, copy.deepcopy(payload)))
        except queue.Full as error:
            if self.error != "record_queue_full:Full":
                self._error("record_queue_full", error)

    def event(self, event, data):
        artifact = f"artifacts/event-{event['id']:05d}.json"
        event["artifact"] = artifact
        self._submit("event", (event, data))

    def artifact(self, name, data):
        self._submit("artifact", (name, data))
        return name + ".json"

    def observation_images(self, name, capture, labels):
        views = {key: value['rgb'] for key, value in capture.get('views', {}).items() if 'rgb' in value}
        self._submit('observation_images', (name, views, labels))
        return {camera: f'{name}-camera-{camera}.svg' for camera in views}

    def visual_decision(self, evidence):
        """Archive actual network calls; reconstruct images without inline base64 in JSON."""
        key = evidence['request_hash']
        if not re.fullmatch(r'[a-f0-9]{64}', key):
            raise ValueError('invalid_visual_request_hash')
        name = 'decisions/' + key
        self._submit('visual_decision', (name, evidence))
        return name + '.json'

    def retain_visual_call(self):
        """Let an in-flight image request finish archiving after cancellation."""
        done = threading.Event()
        self._visual_calls.append(done)
        return done

    def _write(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                if kind == "event":
                    event, data = payload
                    archive_value(self.directory, event["artifact"][:-5], data)
                    with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                elif kind == "artifact":
                    archive_value(self.directory, *payload)
                elif kind == 'observation_images':
                    from .observation_labels import annotated_rgb_svg
                    name, views, labels = payload
                    write_json(self.directory / (name + '.json'), labels)
                    for camera, rgb in views.items():
                        path = self.directory / f'{name}-camera-{camera}.svg'
                        path.write_text(annotated_rgb_svg(rgb, labels, camera), encoding='utf-8')
                elif kind == 'visual_decision':
                    name, evidence = payload
                    count = 0
                    for message in evidence['request']['input']:
                        for part in message['content']:
                            if part['type'] != 'input_image':
                                continue
                            prefix, data = part['image_url'].split(',', 1)
                            if prefix != 'data:image/jpeg;base64':
                                raise ValueError('unsupported_visual_archive_image')
                            image = base64.b64decode(data, validate=True)
                            path = f'{name}-image-{count:03d}.jpg'
                            target = self.directory / path
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(image)
                            part['image_url'] = {'path': path, 'sha256': hashlib.sha256(image).hexdigest(),
                                                 'media_type': 'image/jpeg'}
                            count += 1
                    write_json(self.directory / (name + '.json'), evidence)
                    summary = {key: evidence.get(key) for key in
                               ('observation_id', 'request_hash', 'endpoint', 'purpose', 'elapsed_s', 'http_status', 'error')}
                    with (self.directory / 'decisions/calls.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(summary | {'artifact': name + '.json'}, ensure_ascii=False) + '\n')
                elif kind == "robot":
                    with (self.directory / "robot.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as error:  # noqa: BLE001 — keep draining the queue after a disk failure
                self._error("archive_write", error)
            finally:
                self._queue.task_done()

    def _sample(self):
        while not self._stop.is_set():
            try:
                state = self.simulation.robot_snapshot()
                if state["generation"] != self.generation:
                    return
                # The rolling 1.1s window belongs to verification inputs, archived
                # with those events. Repeating it at 15Hz bloats telemetry.
                arms = {arm: {key: value for key, value in feedback.items()
                              if key != "gripper_position_samples"}
                        for arm, feedback in state["arms"].items()}
                self._submit("robot", {"t_s": time.monotonic() - self.started_at,
                                       "sim_time_s": state["time"], "arms": arms})
            except Exception as error:  # noqa: BLE001 — report background sampler failure
                self._error("robot_sample", error)
                return
            self._stop.wait(1 / 15)

    def stop(self, state):
        """Finalize in the background. A new task does not wait for video encoding."""
        if self._finish_thread is None:
            self._stop.set()
            self.manifest.update(status=state["status"], duration_s=time.monotonic() - self.started_at)
            if self.video is not None:
                self.video.stop()
                self.manifest["video_status"] = "encoding"
            self._finish_thread = threading.Thread(target=self.finish, args=(copy.deepcopy(state),),
                                                   name="experiment-finalize", daemon=True)
            self._finish_thread.start()

    def finish(self, state):
        with self._finish_lock:
            if self._finished.is_set():
                return
            self._stop.set()
            if self.video is not None:
                self.video.stop()
            self.manifest.update(status=state["status"], duration_s=self.manifest.get(
                "duration_s", time.monotonic() - self.started_at))
            if self._sampler is not None:
                self._sampler.join()
            for done in self._visual_calls:
                done.wait()
            self._queue.put(None)
            self._writer.join()
            try:
                archive_value(self.directory, "result", state)
            except Exception as error:  # noqa: BLE001 — video resources must close even after disk errors
                self._error("archive_result", error)
            if self.video is not None:
                self.manifest["video_status"] = "encoding"
                try:
                    write_json(self.directory / "manifest.json", self.manifest)
                except OSError as error:
                    self._error("archive_manifest", error)
                try:
                    result = self.video.finish()
                    self.manifest["video_status"] = result["video_status"]
                    self.manifest["errors"].extend(result.get("errors", []))
                    write_json(self.directory / "video-metadata.json", result)
                except Exception as error:  # noqa: BLE001 — preserve data if encoding fails
                    self._error("video_finish", error)
                    self.manifest["video_status"] = "failed"
            try:
                self.manifest["recording_complete"] = not self.manifest["errors"]
                self.manifest["finalized"] = True
                self.manifest["files"] = [{"path": path.relative_to(self.directory).as_posix(),
                                           "size_bytes": path.stat().st_size}
                                          for path in sorted(self.directory.rglob("*"))
                                          if path.is_file() and path.name != "manifest.json"]
                write_json(self.directory / "manifest.json", self.manifest)
            except Exception as error:  # noqa: BLE001 — finalizer always publishes completion/failure
                self._error("archive_finalize", error)
                self.manifest["recording_complete"] = False
                try:
                    write_json(self.directory / "manifest.json", self.manifest)
                except OSError:
                    pass
            finally:
                self._finished.set()


class ExperimentStore:
    def __init__(self, root, *, video_factory=None):
        self.root = Path(root).resolve()
        self.video_factory = video_factory
        self.sessions = {}
        self._lock = threading.RLock()
        self._readers = {}
        self._recover_interrupted()

    def _recover_interrupted(self):
        for item in self.list():
            if item.get("finalized"):
                continue
            if item["status"] == "running":
                item["status"] = "interrupted"
            if item["video_status"] in {"recording", "encoding"}:
                item["video_status"] = "interrupted"
            item["errors"].append("server_interrupted_recording")
            item.update(finalized=True, recording_complete=False)
            for key in ("size_bytes", "busy"):
                item.pop(key, None)
            write_json(self._directory(item["id"]) / "manifest.json", item)

    def start(self, run_id, instruction, options, config, simulation=None):
        # Publish the directory and its active session together to archive readers.
        with self._lock:
            return self._start(run_id, instruction, options, config, simulation)

    def _start(self, run_id, instruction, options, config, simulation=None):
        if options.save_mode == "none":
            return None
        self._validate_id(run_id)
        created = datetime.now(UTC).isoformat()
        pending = options.save_mode == "ask"
        directory = self.root / ".pending" / run_id if pending else self.root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": 1, "id": run_id, "name": options.name.strip() or instruction,
                    "instruction": instruction, "created_at": created, "status": "running",
                    "retention": "pending" if pending else "saved", "record_video": options.record_video,
                    "video_status": "recording" if options.record_video else "disabled", "errors": [],
                    "time_basis": "monotonic_seconds_since_task_start", "recording_complete": False}
        write_json(directory / "manifest.json", manifest)
        try:
            (directory / "events.jsonl").touch()
            source_root = Path(__file__).resolve().parents[2]
            source_paths = list((source_root / "src/luna_mujoco").glob("*.py"))
            source_paths.extend((source_root / "skills").glob("*/*"))
            source_paths.extend((source_root / "resources").rglob("*.json"))
            config = {**config, "source_sha256": {path.relative_to(source_root).as_posix():
                      hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths if path.is_file()}}
            calibration = source_root / "resources/cameras/calibration.json"
            if calibration.exists():
                config["camera_calibration"] = json.loads(calibration.read_text(encoding="utf-8"))
            archive_value(directory, "config", config)
            session = ExperimentSession(directory, manifest, simulation, self.video_factory)
        except Exception as error:
            manifest.update(status="failed", finalized=True, video_status="failed" if options.record_video else "disabled")
            manifest["errors"].append("experiment_start_failed:" + type(error).__name__)
            write_json(directory / "manifest.json", manifest)
            raise
        self.sessions[run_id] = session
        return session

    @staticmethod
    def _validate_id(identifier):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("invalid_experiment_id")

    def _directory(self, identifier):
        self._validate_id(identifier)
        for path in (self.root / identifier, self.root / ".pending" / identifier):
            if path.is_dir() and not path.is_symlink() and path.resolve().is_relative_to(self.root):
                return path
        raise FileNotFoundError("experiment_not_found")

    def file(self, identifier, relative):
        with self._lock:
            if "\\" in relative or ":" in relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise ValueError("invalid_experiment_path")
            directory = self._directory(identifier)
            path = directory / relative
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("invalid_experiment_path")
            if not path.is_file():
                raise FileNotFoundError("experiment_file_not_found")
            return path

    def get(self, identifier):
        with self._lock:
            directory = self._directory(identifier)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            session = self.sessions.get(identifier)
            if session is not None:
                manifest = copy.deepcopy(session.manifest)
            files = []
            for path in sorted(directory.rglob("*")):
                if (path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory.resolve())
                        and not path.name.endswith(".tmp")):
                    files.append({"path": path.relative_to(directory).as_posix(), "size_bytes": path.stat().st_size})
            return {**manifest, "files": files, "size_bytes": sum(item["size_bytes"] for item in files),
                    "busy": bool(session and session.busy)}

    def list(self):
        with self._lock:
            result = []
            for parent in (self.root, self.root / ".pending"):
                if not parent.exists():
                    continue
                for path in parent.iterdir():
                    if re.fullmatch(r"[a-f0-9]{32}", path.name) and (path / "manifest.json").is_file():
                        try:
                            item = self.get(path.name)
                            item.pop("files")
                            result.append(item)
                        except (OSError, ValueError):
                            continue
            return sorted(result, key=lambda item: item["created_at"], reverse=True)

    def events(self, identifier):
        with self._lock:
            result = []
            try:
                content = self.file(identifier, "events.jsonl").read_text(encoding="utf-8")
            except FileNotFoundError:
                manifest = self.get(identifier)
                if manifest.get("finalized") and manifest["status"] == "failed":
                    return result
                raise
            for line in content.splitlines():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    # An active writer may have an incomplete final line.
                    continue
            return result

    def _require_finished(self, identifier):
        if (session := self.sessions.get(identifier)) is not None and session.busy:
            raise ValueError("experiment_busy")
        manifest = self.get(identifier)
        if (not manifest.get("finalized") or manifest["status"] == "running"
                or manifest.get("video_status") in {"recording", "encoding"}):
            raise ValueError("experiment_busy")

    def _require_unused(self, identifier):
        if self._readers.get(identifier, 0):
            raise ValueError("experiment_in_use")

    @contextmanager
    def read_archive(self, identifier):
        """Keep a record in place for the full lifetime of an archive reader."""
        with self._lock:
            self._directory(identifier)
            self._readers[identifier] = self._readers.get(identifier, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                remaining = self._readers[identifier] - 1
                if remaining:
                    self._readers[identifier] = remaining
                else:
                    self._readers.pop(identifier)

    @contextmanager
    def read_file(self, identifier, relative):
        """Pin the archive until a file response finishes, including disconnects."""
        with self.read_archive(identifier):
            yield self.file(identifier, relative)

    def save(self, identifier, name=None):
        with self._lock:
            self._require_finished(identifier)
            self._require_unused(identifier)
            directory = self._directory(identifier)
            manifest = self.get(identifier)
            if name is not None:
                manifest["name"] = name.strip() or manifest["instruction"]
            if manifest["retention"] == "pending":
                destination = self.root / identifier
                directory.rename(destination)
                directory = destination
            manifest["retention"] = "saved"
            for key in ("size_bytes", "busy"):
                manifest.pop(key, None)
            write_json(directory / "manifest.json", manifest)
            if (session := self.sessions.get(identifier)) is not None:
                session.directory = directory
                session.manifest = manifest
            return self.get(identifier)

    def discard(self, identifier):
        with self._lock:
            if self.get(identifier)["retention"] != "pending":
                raise ValueError("saved_experiment_cannot_be_discarded")
            return self.delete(identifier)

    def delete(self, identifier):
        """Remove one finalized record from the library without erasing its files."""
        with self._lock:
            directory = self._directory(identifier)
            self._require_finished(identifier)
            self._require_unused(identifier)
            if (directory.resolve() != directory
                    or directory.resolve().parent not in {self.root, self.root / ".pending"}):
                raise ValueError("invalid_experiment_directory")
            trash = self.root / ".trash"
            if trash.is_symlink() or trash.resolve() != trash:
                raise ValueError("invalid_trash_directory")
            trash.mkdir(exist_ok=True)
            trash_id = identifier + "-" + uuid4().hex
            entry = trash / trash_id
            entry.mkdir()
            try:
                write_json(entry / "deletion.json", {
                    "id": identifier, "deleted_at": datetime.now(UTC).isoformat(),
                    "original_directory": directory.relative_to(self.root).as_posix(),
                    "archive_directory": "archive",
                })
                directory.rename(entry / "archive")
            except Exception:
                # Only these newly created metadata files belong to this transaction.
                for name in ("deletion.json", "deletion.json.tmp"):
                    (entry / name).unlink(missing_ok=True)
                entry.rmdir()
                raise
            self.sessions.pop(identifier, None)
            return {"deleted": True, "recoverable": True, "trash_id": trash_id}

    def export(self, identifier, mode):
        with self.read_archive(identifier):
            return self._export(identifier, mode)

    def _export(self, identifier, mode):
        self._require_finished(identifier)
        if mode not in {"review", "full"}:
            raise ValueError("invalid_export_mode")
        manifest = self.get(identifier)
        included, omitted = [], []
        for item in manifest["files"]:
            name = item["path"]
            if mode == "review" and (name.endswith((".npz", ".gz")) or name == "robot.jsonl"
                                     or name.startswith(("camera-frames/", "videos/camera_frames/"))):
                omitted.append(name)
            else:
                included.append(item)
        manifest.update(export_mode=mode, omitted_files=omitted, files=included)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
            path = Path(temporary.name)
        try:
            with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=1) as package:
                for item in included:
                    name = item["path"]
                    if name != "manifest.json":
                        package.write(self.file(identifier, name), name,
                                      compress_type=ZIP_STORED if name.endswith((".mp4", ".npz")) else ZIP_DEFLATED)
                package.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def close(self):
        for session in list(self.sessions.values()):
            if session._finish_thread is not None:
                session._finish_thread.join()
            elif session.busy:
                session.finish({"status": "cancelled"})

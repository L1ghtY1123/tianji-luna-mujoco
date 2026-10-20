"""Record display-only geometry and shared RGB previews; encode without replaying tasks.

Only the encoding subprocess imports MuJoCo, so its GL context stays on one
thread and cannot interfere with the live simulation's sensor renderer.
"""

import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCENE_FPS = 15
CAMERA_FPS = 2
CAMERA_ORDER = ("2", "3", "1")


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_packets(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def _interpolate_geometry(first, second, fraction, timestamp):
    following = {pose["id"]: pose for pose in second["geometry_poses"]}
    poses = []
    for pose in first["geometry_poses"]:
        target = following[pose["id"]]
        position = np.asarray(pose["position"], dtype=float)
        position += fraction * (np.asarray(target["position"]) - position)
        start = np.asarray(pose["quaternion"], dtype=float)
        end = np.asarray(target["quaternion"], dtype=float)
        start /= np.linalg.norm(start)
        end /= np.linalg.norm(end)
        dot = float(start @ end)
        if dot < 0:
            end = -end
            dot = -dot
        if dot > .9995:
            quaternion = start + fraction * (end - start)
        else:
            angle = math.acos(min(1., dot))
            quaternion = (math.sin((1 - fraction) * angle) * start
                          + math.sin(fraction * angle) * end) / math.sin(angle)
        quaternion /= np.linalg.norm(quaternion)
        poses.append({"id": pose["id"], "position": position.tolist(), "quaternion": quaternion.tolist()})
    return {**first, "t_s": timestamp, "source_t_s": [first["t_s"], second["t_s"]],
            "sim_time": first["sim_time"] + fraction * (second["sim_time"] - first["sim_time"]),
            "geometry_poses": poses}


def resample_frames(packets, duration_s, fps, *, interpolate=False):
    """Stream two source frames at a time onto a wall-clock video timeline.

    Before the first capture the frame is absent (black). After the last capture
    the last frame is held. Camera pixels are never interpolated. Geometry can
    use linear position interpolation and shortest-arc quaternion SLERP.
    """
    if not math.isfinite(duration_s) or duration_s < 0 or fps <= 0:
        raise ValueError("duration must be finite and nonnegative, and fps positive")
    packets = iter(packets)
    previous = None
    following = next(packets, None)
    for index in range(max(1, math.ceil(duration_s * fps))):
        timestamp = index / fps
        while following is not None and following["t_s"] <= timestamp:
            previous = following
            following = next(packets, None)
            if following is not None and following["t_s"] < previous["t_s"]:
                raise ValueError("capture timestamps must be ordered")
        frame = previous
        if interpolate and previous is not None and following is not None:
            fraction = (timestamp - previous["t_s"]) / (following["t_s"] - previous["t_s"])
            frame = _interpolate_geometry(previous, following, fraction, timestamp)
        yield timestamp, frame


def stitch_cameras(views):
    """Left wrist 2, head 3, right wrist 1; preserve aspect with black padding."""
    strip = Image.new("RGB", (1920, 480))
    for index, camera_id in enumerate(CAMERA_ORDER):
        rgb = np.asarray(views[camera_id]["rgb"])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"camera {camera_id} must supply uint8 RGB pixels")
        tile = ImageOps.contain(Image.fromarray(rgb), (640, 480), Image.Resampling.LANCZOS)
        strip.paste(tile, (index * 640 + (640 - tile.width) // 2, (480 - tile.height) // 2))
    draw = ImageDraw.Draw(strip)
    font = ImageFont.load_default(size=22)
    for index, label in enumerate(("LEFT WRIST / 2", "HEAD / 3", "RIGHT WRIST / 1")):
        draw.rectangle((index * 640, 0, (index + 1) * 640, 34), fill="black")
        draw.text((index * 640 + 16, 5), label, font=font, fill="white")
    return strip


def _elapsed_overlay(image, timestamp):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, image.height - 40, 280, image.height), fill="black")
    draw.text((16, image.height - 32), f"Task {timestamp:08.2f} s", font=ImageFont.load_default(size=22),
              fill="white")
    return image


class VideoRecorder:
    """A run owns its files, two sampling threads, and a blocking finish boundary."""

    def __init__(self, directory: Path, simulation, started_at: float, generation: int):
        self.directory = Path(directory)
        self.simulation = simulation
        self.started_at = started_at
        self.generation = generation
        self._stop = threading.Event()
        self._threads = []
        self._lock = threading.Lock()
        self._finish_lock = threading.Lock()
        self._errors = []
        self._ended_at = None
        self._result = None
        self._recording = None
        self._started = False
        self._stats = {kind: {"sample_count": 0, "first_t_s": None, "last_t_s": None, "max_gap_s": 0.}
                       for kind in ("scene", "cameras")}

    def _error(self, stage, error):
        with self._lock:
            self._errors.append(f"{stage}: {type(error).__name__}: {error}")

    def start(self):
        if self._started:
            raise RuntimeError("video recorder has already started")
        self._started = True
        try:
            (self.directory / "videos" / "camera_frames").mkdir(parents=True, exist_ok=True)
            snapshot = self.simulation.snapshot()
            self._check_generation(snapshot)
            self._recording = {"schema_version": 1, "generation": self.generation,
                               "scene": {"table": snapshot["table"], "objects": snapshot["objects"]},
                               "geometry_manifest": self.simulation.manifest(),
                               "camera_order": list(CAMERA_ORDER), "clock": "monotonic_seconds_since_run_start",
                               "scene_resampling": "linear_position_shortest_arc_quaternion_slerp",
                               "camera_resampling": "previous_frame_hold", "before_first_capture": "black",
                               "after_last_capture": "last_frame_hold", "geometry_target_fps": SCENE_FPS,
                               "camera_max_fps": CAMERA_FPS}
            _write_json(self.directory / "videos" / "recording.json", self._recording)
            for name, target in (("geometry", self._sample_geometry), ("preview", self._sample_preview)):
                thread = threading.Thread(target=target, name=f"experiment-{name}", daemon=True)
                thread.start()
                self._threads.append(thread)
        except Exception as error:  # noqa: BLE001 - recording failures must not stop the task.
            self._error("start recording", error)
            self.stop()

    def _check_generation(self, snapshot):
        if snapshot["generation"] != self.generation:
            raise RuntimeError("simulation generation changed during recording")

    def _observe_time(self, kind, timestamp):
        stats = self._stats[kind]
        if stats["last_t_s"] is not None:
            stats["max_gap_s"] = max(stats["max_gap_s"], timestamp - stats["last_t_s"])
        else:
            stats["first_t_s"] = timestamp
        stats["last_t_s"] = timestamp
        stats["sample_count"] += 1

    def _sample_geometry(self):
        try:
            with gzip.open(self.directory / "frames.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as output:
                deadline = time.monotonic()
                while not self._stop.is_set():
                    snapshot = self.simulation.snapshot()
                    captured_at = time.monotonic()
                    if self._stop.is_set():
                        break
                    self._check_generation(snapshot)
                    timestamp = max(0., captured_at - self.started_at)
                    packet = {"t_s": timestamp, "sim_time": snapshot["time"], "generation": self.generation,
                              "geometry_poses": snapshot["geometry_poses"], "arms": snapshot["arms"]}
                    output.write(json.dumps(packet, separators=(",", ":")) + "\n")
                    output.flush()
                    self._observe_time("scene", timestamp)
                    deadline = max(deadline + 1 / SCENE_FPS, time.monotonic())
                    self._stop.wait(max(0., deadline - time.monotonic()))
        except Exception as error:  # noqa: BLE001 - thread failures are reported in the run manifest.
            self._error("geometry sampling", error)

    def _sample_preview(self):
        try:
            with (self.directory / "videos" / "camera_frames.jsonl").open("w", encoding="utf-8") as output:
                previous_id = None
                while not self._stop.is_set():
                    requested_at = time.monotonic()
                    preview = self.simulation.preview_rgb()
                    if self._stop.is_set():
                        break
                    if preview is not None and preview["preview_frame_id"] != previous_id:
                        self._check_generation(preview)
                        timestamp = preview["captured_monotonic_s"] - self.started_at
                        # The shared cache can still contain a frame from before this run.
                        if timestamp >= 0:
                            index = self._stats["cameras"]["sample_count"]
                            path = f"videos/camera_frames/{index:08d}.jpg"
                            stitch_cameras(preview["views"]).save(self.directory / path, quality=90)
                            packet = {"t_s": timestamp, "sim_time": preview["time"],
                                      "received_t_s": time.monotonic() - self.started_at,
                                      "generation": self.generation, "frame_id": preview["preview_frame_id"],
                                      "path": path, "camera_order": list(CAMERA_ORDER)}
                            output.write(json.dumps(packet, separators=(",", ":")) + "\n")
                            output.flush()
                            self._observe_time("cameras", timestamp)
                        previous_id = preview["preview_frame_id"]
                    self._stop.wait(max(0., requested_at + 1 / CAMERA_FPS - time.monotonic()))
        except Exception as error:  # noqa: BLE001 - thread failures are reported in the run manifest.
            self._error("preview sampling", error)

    def stop(self):
        """Freeze the run end time and signal sampling without waiting for GL or disk."""
        with self._lock:
            if self._ended_at is None:
                self._ended_at = time.monotonic()
            self._stop.set()

    def finish(self):
        """Join sampling and encode; call in a background thread after stop()."""
        with self._finish_lock:
            if self._result is not None:
                return self._result
            self.stop()
            for thread in self._threads:
                thread.join()
            duration = max(0., self._ended_at - self.started_at)
            for kind, limit in (("scene", 1.), ("cameras", 5.)):
                stats = self._stats[kind]
                if stats["sample_count"]:
                    gap = max(stats["max_gap_s"], stats["first_t_s"], duration - stats["last_t_s"])
                    stats["max_unobserved_s"] = gap
                    if gap > limit:
                        self._errors.append(f"{kind} capture gap of {gap:.2f}s exceeded {limit:.1f}s")
            encoded = {"video_status": "failed", "errors": []}
            try:
                if self._recording is None:
                    raise RuntimeError("recording did not initialize")
                self._recording.update(duration_s=duration, streams=self._stats, errors=list(self._errors))
                _write_json(self.directory / "videos" / "recording.json", self._recording)
                encoded = encode_videos(self.directory)
            except Exception as error:  # noqa: BLE001 - the experiment keeps data after encoding failures.
                self._error("finish recording", error)
            errors = [*self._errors, *encoded.get("errors", [])]
            status = encoded["video_status"]
            if status == "ready" and errors:
                status = "partial"
            self._result = {**encoded, "video_status": status, "errors": errors,
                            "duration_s": duration, "streams": self._stats}
            return self._result


def encode_videos(directory):
    """Use a fresh process to own all offline GL work; never hide a missing encoder."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"video_status": "failed", "errors": ["ffmpeg executable is unavailable"]}
    directory = Path(directory).resolve()
    environment = os.environ.copy()
    if sys.platform.startswith("linux"):
        environment["MUJOCO_GL"] = "osmesa"
        environment["PYOPENGL_PLATFORM"] = "osmesa"
        environment["LP_NUM_THREADS"] = "2"
    # -m must also work when the app was started with an in-process sys.path entry.
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]),
                                               environment.get("PYTHONPATH", "")])
    log_path = directory / "videos" / "encoding.log"
    with log_path.open("wb") as log:
        process = subprocess.run([sys.executable, "-m", "jev_mujoco.experiment_video", "--encode",
                                  str(directory), ffmpeg], env=environment, stdout=log, stderr=log, check=False)
    if process.returncode:
        return {"video_status": "failed", "errors": [f"video worker exited {process.returncode}; see videos/encoding.log"]}
    return json.loads((directory / "videos" / "result.json").read_text(encoding="utf-8"))


def _encode_rgb(frames, destination, fps, size, ffmpeg):
    temporary = destination.with_suffix(".mp4.tmp")
    log_path = destination.with_suffix(".ffmpeg.log")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{size[0]}x{size[1]}", "-framerate", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(temporary)]
    count = 0
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for rgb in frames:
                    pixels = np.asarray(rgb, dtype=np.uint8)
                    if pixels.shape != (size[1], size[0], 3):
                        raise ValueError("RGB frame dimensions do not match encoder dimensions")
                    process.stdin.write(pixels.tobytes())
                    count += 1
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed; see {log_path.name}")
            except BaseException:
                process.kill()
                process.wait()
                process.stdin.close()
                raise
        if not count or not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced no video frames")
        temporary.replace(destination)
        return count
    finally:
        temporary.unlink(missing_ok=True)


def _scene_images(directory, recording):
    import mujoco

    from .model_builder import ModelBuilder, normalize_scene

    builder = ModelBuilder()
    try:
        model = builder.build(normalize_scene(recording["scene"]))
        for geometry in recording["geometry_manifest"]["geometries"]:
            if model.geom(int(geometry["id"])).name != geometry["name"]:
                raise RuntimeError("recorded geometry IDs do not match the rendering model")
        model.vis.global_.offwidth = 1280
        model.vis.global_.offheight = 720
        model.vis.quality.offsamples = 4
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [.30, .03, .77]
        camera.distance = 2.1
        camera.azimuth = 210
        camera.elevation = -35
        packets = _read_packets(directory / "frames.jsonl.gz")
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            for timestamp, packet in resample_frames(packets, recording["duration_s"], SCENE_FPS, interpolate=True):
                if packet is None:
                    yield _elapsed_overlay(Image.new("RGB", (1280, 720)), timestamp)
                    continue
                renderer.update_scene(data, camera=camera, scene_option=options)
                poses = {int(pose["id"]): pose for pose in packet["geometry_poses"]}
                for geometry in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if geometry.objtype == mujoco.mjtObj.mjOBJ_GEOM and geometry.objid in poses:
                        pose = poses[geometry.objid]
                        geometry.pos[:] = pose["position"]
                        rotation = np.empty(9)
                        mujoco.mju_quat2Mat(rotation, np.asarray(pose["quaternion"]))
                        geometry.mat[:] = rotation.reshape(3, 3)
                yield _elapsed_overlay(Image.fromarray(renderer.render()), timestamp)
    finally:
        builder.close()


def _camera_images(directory, recording):
    packets = _read_packets(directory / "videos" / "camera_frames.jsonl")
    last_path = None
    image = Image.new("RGB", (1920, 480))
    for timestamp, packet in resample_frames(packets, recording["duration_s"], CAMERA_FPS):
        if packet is not None and packet["path"] != last_path:
            with Image.open(directory / packet["path"]) as source:
                image = source.convert("RGB")
            last_path = packet["path"]
        yield _elapsed_overlay(image, timestamp)


def _encode_recording(directory, ffmpeg):
    recording = json.loads((directory / "videos" / "recording.json").read_text(encoding="utf-8"))
    errors = []
    videos = {}
    for kind, size, fps, factory in (("scene", (1280, 720), SCENE_FPS, _scene_images),
                                     ("cameras", (1920, 480), CAMERA_FPS, _camera_images)):
        try:
            stats = recording["streams"][kind]
            if not stats["sample_count"]:
                raise RuntimeError(f"no {kind} samples were captured")
            last_frame_time = (max(1, math.ceil(recording["duration_s"] * fps)) - 1) / fps
            if stats["first_t_s"] > last_frame_time:
                raise RuntimeError(f"no captured {kind} sample reaches the video timeline")
            frames = factory(directory, recording)
            try:
                count = _encode_rgb(frames, directory / "videos" / f"{kind}.mp4", fps, size, ffmpeg)
            finally:
                frames.close()
            videos[kind] = {"path": f"videos/{kind}.mp4", "fps": fps, "width": size[0], "height": size[1],
                            "frame_count": count, "duration_s": count / fps,
                            "initial_blank_s": math.ceil(stats["first_t_s"] * fps) / fps,
                            "last_frame_hold_s": max(0., recording["duration_s"] - stats["last_t_s"]),
                            "end_padding_s": max(0., count / fps - recording["duration_s"])}
        except Exception as error:  # noqa: BLE001 - preserve the other independently encoded video.
            errors.append(f"{kind} encoding: {type(error).__name__}: {error}")
    status = "ready" if len(videos) == 2 else "partial" if videos else "failed"
    return {"video_status": status, "errors": errors, "videos": videos}


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--encode":
        raise SystemExit("usage: python -m jev_mujoco.experiment_video --encode DIRECTORY FFMPEG")
    if sys.platform.startswith("linux"):
        os.nice(5)
    destination = Path(sys.argv[2])
    _write_json(destination / "videos" / "result.json", _encode_recording(destination, sys.argv[3]))

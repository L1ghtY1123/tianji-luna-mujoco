import gzip
import io
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from jev_mujoco import experiment_video as video


def pose_frame(t_s, x, quaternion=(1, 0, 0, 0)):
    return {"t_s": t_s, "sim_time": t_s + 10, "generation": 1,
            "geometry_poses": [{"id": "0", "position": [x, 0, 0], "quaternion": quaternion}]}


def test_resampling_keeps_wall_time_and_holds_last_frame():
    samples = [pose_frame(.25, 1), pose_frame(.75, 3)]
    frames = list(video.resample_frames(iter(samples), duration_s=1.6, fps=2))
    assert [t for t, _ in frames] == [0, .5, 1, 1.5]
    assert frames[0][1] is None
    assert [frame["geometry_poses"][0]["position"][0] for _, frame in frames[1:]] == [1, 3, 3]
    assert frames[-1][1]["t_s"] == .75


def test_geometry_interpolation_uses_shortest_quaternion_rotation():
    samples = [pose_frame(0, 0), pose_frame(1, 2, (-math.sqrt(.5), 0, 0, -math.sqrt(.5)))]
    frames = list(video.resample_frames(iter(samples), duration_s=1.5, fps=2, interpolate=True))
    middle = frames[1][1]
    assert middle["geometry_poses"][0]["position"] == [1, 0, 0]
    assert middle["sim_time"] == 10.5
    np.testing.assert_allclose(middle["geometry_poses"][0]["quaternion"],
                               [math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8)])
    assert frames[-1][1]["geometry_poses"][0]["position"] == [2, 0, 0]


def test_streaming_resampler_does_not_read_all_frames_in_advance():
    def samples():
        yield pose_frame(0, 0)
        yield pose_frame(1, 1)
        raise AssertionError("read beyond the two frames needed for the current output")

    frames = video.resample_frames(samples(), duration_s=3, fps=15)
    assert next(frames)[0] == 0
    assert next(frames)[0] == pytest.approx(1 / 15)


def test_camera_strip_has_left_head_right_order_and_fixed_dimensions():
    views = {camera: {"rgb": np.full((360, 640, 3), color, dtype=np.uint8)}
             for camera, color in [("1", (0, 0, 255)), ("2", (255, 0, 0)), ("3", (0, 255, 0))]}
    strip = np.asarray(video.stitch_cameras(views))
    assert strip.shape == (480, 1920, 3)
    assert strip[240, 320].tolist() == [255, 0, 0]
    assert strip[240, 960].tolist() == [0, 255, 0]
    assert strip[240, 1600].tolist() == [0, 0, 255]
    assert strip[0, 320].tolist() == [0, 0, 0]
    assert np.any(strip[5:30, 16:220] > 0)
    assert np.any(strip[5:30, 656:850] > 0)
    assert np.any(strip[5:30, 1296:1550] > 0)


class RecordingSimulation:
    def __init__(self):
        self.preview_entered = threading.Event()
        self.release_preview = threading.Event()
        self.generation = 1
        self.snapshots = 0

    def snapshot(self):
        self.snapshots += 1
        return {**pose_frame(0, self.snapshots), "generation": self.generation, "time": float(self.snapshots),
                "table": {"position": [0, 0, .5], "size": [1, 1, .1]}, "objects": [], "arms": {}}

    def manifest(self):
        return {"geometries": [], "quaternion_order": "wxyz"}

    def preview_rgb(self):
        self.preview_entered.set()
        self.release_preview.wait(timeout=3)


def test_slow_preview_does_not_block_geometry_and_stop_prevents_next_run(tmp_path, monkeypatch):
    sim = RecordingSimulation()
    recorder = video.VideoRecorder(tmp_path, sim, time.monotonic(), 1)
    monkeypatch.setattr(video, "encode_videos", lambda _: {"video_status": "failed", "errors": ["test encoder"]})
    recorder.start()
    try:
        assert sim.preview_entered.wait(timeout=1)
        time.sleep(.24)
        recorder.stop()
        sim.generation = 2
    finally:
        sim.release_preview.set()
    result = recorder.finish()
    with gzip.open(tmp_path / "frames.jsonl.gz", "rt", encoding="utf-8") as source:
        frames = [json.loads(line) for line in source]
    assert len(frames) >= 3
    assert {frame["generation"] for frame in frames} == {1}
    assert result["duration_s"] < 1
    assert result["video_status"] == "failed"
    assert recorder.finish() == result


def test_missing_ffmpeg_is_failed_even_when_recorded_data_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda _: None)
    result = video.encode_videos(tmp_path)
    assert result["video_status"] == "failed"
    assert any("ffmpeg" in error for error in result["errors"])
    assert not (tmp_path / "videos" / "scene.mp4").exists()


def test_recording_write_failure_is_reported_not_ready(tmp_path, monkeypatch):
    directory = tmp_path / "blocked"
    directory.write_text("a file cannot be used as a directory", encoding="utf-8")
    recorder = video.VideoRecorder(directory, RecordingSimulation(), time.monotonic(), 1)
    recorder.start()
    result = recorder.finish()
    assert result["video_status"] == "failed"
    assert result["errors"]


def test_preview_uses_capture_time_and_does_not_duplicate_shared_cache(tmp_path, monkeypatch):
    started_at = time.monotonic() - .25
    sim = RecordingSimulation()
    preview = {"preview_frame_id": "preview-1", "generation": 1, "time": 12.5,
               "captured_monotonic_s": started_at + .125,
               "views": {camera: {"rgb": np.zeros((48, 64, 3), dtype=np.uint8)} for camera in ("1", "2", "3")}}
    monkeypatch.setattr(sim, "preview_rgb", lambda: preview)
    monkeypatch.setattr(video, "encode_videos", lambda _: {"video_status": "failed", "errors": ["test encoder"]})
    recorder = video.VideoRecorder(tmp_path, sim, started_at, 1)
    recorder.start()
    time.sleep(.6)
    recorder.stop()
    time.sleep(.1)
    result = recorder.finish()
    packets = list(video._read_packets(tmp_path / "videos" / "camera_frames.jsonl"))
    assert len(packets) == 1
    assert packets[0]["t_s"] == pytest.approx(.125)
    assert packets[0]["sim_time"] == 12.5
    assert packets[0]["received_t_s"] > packets[0]["t_s"]
    assert (tmp_path / packets[0]["path"]).is_file()
    assert result["duration_s"] < time.monotonic() - started_at - .09


def test_encoder_failure_does_not_leave_an_apparently_ready_file(tmp_path):
    # Python rejects ffmpeg flags, exercising an actual failed encoder process.
    destination = tmp_path / "scene.mp4"
    with pytest.raises((BrokenPipeError, OSError, RuntimeError)):
        video._encode_rgb(iter([np.zeros((720, 1280, 3), dtype=np.uint8)]),
                          destination, 15, (1280, 720), sys.executable)
    assert not destination.exists()
    assert not destination.with_suffix(".tmp.mp4").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_empty_recording_cannot_encode_a_blank_video_as_ready(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "recording.json").write_text(
        json.dumps({"streams": {"scene": {"sample_count": 0}, "cameras": {"sample_count": 0}}}),
        encoding="utf-8")
    result = video._encode_recording(tmp_path, sys.executable)
    assert result["video_status"] == "failed"
    assert result["videos"] == {}
    assert len(result["errors"]) == 2


def test_camera_video_elapsed_time_advances_while_capture_is_held(tmp_path):
    from PIL import Image

    (tmp_path / "videos").mkdir()
    Image.new("RGB", (1920, 480), color=(50, 50, 50)).save(tmp_path / "capture.jpg")
    (tmp_path / "videos" / "camera_frames.jsonl").write_text(
        json.dumps({"t_s": 0, "path": "capture.jpg"}) + "\n", encoding="utf-8")
    frames = list(video._camera_images(tmp_path, {"duration_s": 1.0}))
    assert len(frames) == 2
    first, second = [np.asarray(frame) for frame in frames]
    np.testing.assert_array_equal(first[:400], second[:400])
    assert not np.array_equal(first[440:], second[440:])


def test_large_capture_gap_marks_successfully_encoded_videos_partial(tmp_path, monkeypatch):
    sim = RecordingSimulation()
    sim.release_preview.set()
    monkeypatch.setattr(video, "encode_videos", lambda _: {"video_status": "ready", "errors": []})
    recorder = video.VideoRecorder(tmp_path, sim, time.monotonic() - 10, 1)
    recorder.start()
    time.sleep(.1)
    result = recorder.finish()
    assert result["video_status"] == "partial"
    assert any("scene capture gap" in error for error in result["errors"])


@pytest.mark.parametrize(("duration", "capture_time", "visible"), [(.2, .1, False), (.5, .5, False),
                                                                  (.51, .5, True)])
def test_short_camera_recording_requires_capture_on_encoded_timeline(tmp_path, monkeypatch,
                                                                    duration, capture_time, visible):
    from PIL import Image

    (tmp_path / "videos").mkdir()
    Image.new("RGB", (1920, 480), color=(50, 50, 50)).save(tmp_path / "capture.jpg")
    (tmp_path / "videos" / "camera_frames.jsonl").write_text(
        json.dumps({"t_s": capture_time, "path": "capture.jpg"}) + "\n", encoding="utf-8")
    (tmp_path / "videos" / "recording.json").write_text(json.dumps({"duration_s": duration, "streams": {
        "scene": {"sample_count": 0},
        "cameras": {"sample_count": 1, "first_t_s": capture_time, "last_t_s": capture_time}}}), encoding="utf-8")

    def encode_frames(frames, destination, *_):
        count = sum(1 for _ in frames)
        destination.write_bytes(b"encoded frames")
        return count

    monkeypatch.setattr(video, "_encode_rgb", encode_frames)
    result = video._encode_recording(tmp_path, "unused-encoder")
    assert result["video_status"] == ("partial" if visible else "failed")
    assert ("cameras" in result["videos"]) == visible
    assert (tmp_path / "videos" / "cameras.mp4").exists() == visible
    if not visible:
        assert any("no captured cameras sample reaches the video timeline" in error for error in result["errors"])


def test_encoder_publishes_mp4_only_after_successful_completion(tmp_path, monkeypatch):
    class EncoderProcess:
        def __init__(self, command, **_):
            self.stdin = io.BytesIO()
            Path(command[-1]).write_bytes(b"unfinished video")

        def wait(self):
            return 0

        def kill(self):
            pass

    def frames():
        assert not list(tmp_path.glob("*.mp4"))
        yield np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(video.subprocess, "Popen", EncoderProcess)
    destination = tmp_path / "scene.mp4"
    assert video._encode_rgb(frames(), destination, 15, (2, 2), "unused-encoder") == 1
    assert destination.is_file()
    assert not list(tmp_path.glob("*.tmp"))

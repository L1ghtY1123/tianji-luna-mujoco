"""Actual MuJoCo RGBD -> SAM3 + Decisions -> measured perception acceptance probe.

Run with PYTHONPATH=src python tests/rgbd_perception_smoke.py --sam3-url URL.
Requires Decisions credentials; running the probe calls both inference services.
The simulator snapshot is saved only after perception as an independent oracle.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from jev_mujoco.cameras import CALIBRATION_PATH
from jev_mujoco.perception import (
    SAM3_MODEL_IDENTITY,
    Perception,
    PerceptionError,
    Sam3Client,
    public_observation,
)
from jev_mujoco.simulation import Simulation

SCENE = {
    "table": {"position": [.55, 0., .56], "size": [.7, 1., .08]},
    "objects": [
        {"id": "oracle_red_cube", "kind": "cube", "color": "red", "position": [.43, .15, .70],
         "size": [.04, .04, .04], "yaw_deg": 0.},
        {"id": "oracle_blue_bin", "kind": "bin", "color": "blue", "position": [.63, -.18, .70],
         "size": [.22, .16, .10], "wall_thickness_m": .01, "yaw_deg": 0.},
    ],
}


class RecordingDetector:
    def __init__(self, url, output):
        self.client = Sam3Client(url)
        self.output = output
        self.records = []
        self.arrays = {}

    def segment(self, rgb, text, timestamp_us):
        index = len(self.records)
        record = {"index": index, "prompt": text, "timestamp_us": timestamp_us,
                  "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        self.records.append(record)
        started = time.monotonic()
        print(json.dumps({"sam3_request": index, "prompt": text}), flush=True)
        try:
            result = self.client.segment(rgb, text, timestamp_us)
            record["detections"] = [{"score": float(item["score"]),
                                     "pixels": int(np.count_nonzero(item["mask"]))} for item in result]
            for detection, item in enumerate(result):
                self.arrays[f"request_{index}_mask_{detection}"] = np.asarray(item["mask"], dtype=np.uint8)
            return result
        except PerceptionError as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["duration_s"] = time.monotonic() - started
            print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-url", default=os.getenv("JEV_SAM3_URL", "http://127.0.0.1:8322"))
    parser.add_argument("--output", type=Path, default=Path("tests/artifacts/rgbd/perception"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    simulation = Simulation()
    simulation.start()
    detector = RecordingDetector(args.sam3_url, args.output)
    calibration_bytes = CALIBRATION_PATH.read_bytes()
    calibration = json.loads(calibration_bytes)
    (args.output / "calibration-snapshot.json").write_bytes(calibration_bytes)
    result = {"input_boundary": "Perception receives only actual registered RGBD and robot telemetry",
              "sam3_model": dict(SAM3_MODEL_IDENTITY),
              "model_identity_validation": "Sam3Client checks the pinned model identity in nonempty inference results",
              "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
              "calibration_id": calibration["calibration_id"],
              "runtime": {"mujoco_version": mujoco.__version__,
                          "configured_gl_backend": os.getenv("MUJOCO_GL", "platform_default")}}
    try:
        simulation.configure(SCENE)
        simulation.confirm()
        started = time.monotonic()
        capture = simulation.capture_rgbd()
        result["capture_duration_s"] = time.monotonic() - started
        arrays, camera_summary = {}, {}
        for identifier, view in capture["views"].items():
            for field in ("rgb", "depth_m", "valid_depth"):
                arrays[f"camera_{identifier}_{field}"] = view[field]
            Image.fromarray(view["rgb"]).save(args.output / f"camera-{identifier}.png")
            measured = view["depth_m"][view["valid_depth"]]
            camera_summary[identifier] = {
                key: value for key, value in view.items() if key not in {"rgb", "depth_m", "valid_depth"}}
            camera_summary[identifier].update(rgb_shape=list(view["rgb"].shape),
                valid_depth_pixels=len(measured), depth_range_m=[float(measured.min()), float(measured.max())],
                rgb_sha256=hashlib.sha256(view["rgb"].tobytes()).hexdigest(),
                depth_sha256=hashlib.sha256(view["depth_m"].tobytes()).hexdigest())
        np.savez_compressed(args.output / "registered-rgbd.npz", **arrays)
        result.update(observation_id=capture["observation_id"], generation=capture["generation"],
                      capture_time=capture["time"], cameras=camera_summary, arms=capture["arms"])
        started = time.monotonic()
        try:
            observation = Perception(detector).observe(capture, instruction="抓取红色方块到蓝色盒子里",
                source_hint={'kind': 'cube', 'color': 'red', 'include_containers': True,
                    'destination': {'kind': 'bin', 'color': 'blue'},
                    'visual_decision': {'source': 'jev', 'purpose': 'initialize_inventory'}})
            result["observation"] = public_observation(observation)
            result["perception_duration_s"] = time.monotonic() - started
            result["passed"] = (any(obj["kind"] == "cube" and obj["color"] == "red"
                                    for obj in observation["objects"])
                                and any(obj["kind"] == "bin" and obj["color"] == "blue" and obj.get("placement_reference")
                                        for obj in observation["objects"]))
        except PerceptionError as error:
            result.update(passed=False, error=f"{type(error).__name__}: {error}",
                          perception_duration_s=time.monotonic() - started)
        # Truth is deliberately outside the observation and was never passed to the detector/perception.
        oracle = simulation.snapshot()
        (args.output / "independent-oracle.json").write_text(json.dumps(
            {"objects": oracle["objects"], "table": oracle["table"]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    finally:
        simulation.close()
        result["sam3_calls"] = detector.records
        np.savez_compressed(args.output / "sam3-masks.npz", **detector.arrays)
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"cameras", "arms", "sam3_calls"}},
                     ensure_ascii=False), flush=True)
    if not result.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

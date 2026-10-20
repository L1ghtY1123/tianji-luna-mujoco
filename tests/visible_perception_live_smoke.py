"""Real SAM3 on offline horizontal and calibrated observation-pose RGBD.

Own joint qpos changes here are camera fixtures, never evidence of executed motion.
Only rendered RGBD and own telemetry enter Perception.observe. Fixture truth is
used for independent accuracy checks after perception has returned.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from jev_mujoco.cameras import CALIBRATION_PATH, RgbdRenderer, clone_render_model
from jev_mujoco.model_builder import ARM_JOINTS, ModelBuilder, normalize_scene
from jev_mujoco.perception import Perception, PerceptionError, Sam3Client, public_observation

ROOT = Path(__file__).resolve().parents[1]
SCENE = {"objects": [
    {"id": "oracle_source", "kind": "cube", "color": "red", "position": [.43, .15, .625],
     "size": [.05, .05, .05]},
    {"id": "oracle_destination", "kind": "bin", "color": "blue", "position": [.65, -.15, .65],
     "size": [.18, .16, .10], "wall_thickness_m": .01},
    {"id": "oracle_unmentioned", "kind": "cube", "color": "black", "position": [.60, .15, .62],
     "size": [.04, .04, .04]},
]}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class RecordingDetector:
    def __init__(self, url, capture, output):
        self.client = Sam3Client(url)
        self.output = output
        self.records = []
        self.camera_by_hash = {
            hashlib.sha256(view["rgb"].tobytes()).hexdigest(): camera_id
            for camera_id, view in capture["views"].items()
        }

    def segment(self, rgb, text, timestamp_us):
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        index = len(self.records)
        record = {"index": index, "camera_id": self.camera_by_hash[digest],
                  "prompt": text, "timestamp_us": timestamp_us, "rgb_sha256": digest}
        self.records.append(record)
        started = time.monotonic()
        try:
            result = self.client.segment(rgb, text, timestamp_us)
            record["detections"] = [{"score": item["score"], "pixels": int(item["mask"].sum())}
                                    for item in result]
            np.savez_compressed(self.output / f"sam3-request-{index:02d}.npz",
                                **{f"mask_{n}": item["mask"] for n, item in enumerate(result)})
            return result
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["duration_s"] = time.monotonic() - started
            save_json(self.output / "sam3-calls.json", self.records)
            print(json.dumps(record), flush=True)


def save_capture(capture, output):
    metadata = {key: value for key, value in capture.items() if key != "views"}
    metadata["views"] = {}
    for camera_id, view in capture["views"].items():
        Image.fromarray(view["rgb"]).save(output / f"camera-{camera_id}.png")
        arrays = {key: view[key] for key in ("rgb", "depth_m", "valid_depth")}
        np.savez_compressed(output / f"camera-{camera_id}-rgbd.npz", **arrays)
        metadata["views"][camera_id] = {key: value for key, value in view.items() if key not in arrays}
    save_json(output / "capture.json", metadata)


def evaluate(observation, capture, scene, checkpoint, output):
    objects = observation["objects"]
    checks = {}
    for fixture in scene["objects"]:
        kind, color = fixture["kind"], fixture["color"]
        matches = [item for item in objects if item["kind"] == kind and item["color"] == color]
        checks[f"unique_{color}_{kind}"] = len(matches) == 1
        checks[f"head_sees_{color}_{kind}"] = any("3" in item["coverage"]["camera_ids"] for item in matches)
        if checkpoint == "fine":
            checks[f"all_cameras_merged_{color}_{kind}"] = (len(matches) == 1 and
                set(matches[0]["coverage"]["camera_ids"]) == {"1", "2", "3"})
    checks["no_extra_instances"] = len(objects) == len(scene['objects'])
    forbidden = {"size", "yaw_deg", "quaternion", "T_world_object", "cavity", "shape", "hidden_surface"}
    checks["no_hidden_pose_size_cavity"] = all(not forbidden.intersection(item) for item in objects)
    checks["measured_table_height_3mm"] = abs(observation["table"]["surface_height_m"] - .6) <= .003
    details = []
    for index, item in enumerate(objects):
        points = item["points_world_m"]
        sources = item["point_sources"]
        maximum_error = 0.
        source_depth_valid = True
        for camera_id in set(sources["camera_ids"].tolist()):
            selected = sources["camera_ids"] == camera_id
            rows, columns = sources["pixels_rc"][selected].T
            view = capture["views"][camera_id]
            depth = view["depth_m"][rows, columns]
            intrinsics = view["intrinsics"]
            camera_points = np.column_stack(((columns - intrinsics["cx"]) * depth / intrinsics["fx"],
                                             (rows - intrinsics["cy"]) * depth / intrinsics["fy"], depth))
            transform = np.asarray(view["T_world_camera"])
            reconstructed = camera_points @ transform[:3, :3].T + transform[:3, 3]
            maximum_error = max(maximum_error, float(np.linalg.norm(points[selected] - reconstructed, axis=1).max()))
            source_depth_valid &= bool(view["valid_depth"][rows, columns].all())
        checks[f"instance_{index}_measured_point_provenance"] = source_depth_valid and maximum_error < 1e-6
        checks[f"instance_{index}_visible_only"] = item["coverage"]["observed_surface_only"]
        record = {"id": item["id"], "kind": item["kind"], "color": item["color"],
                  "point_count": len(points), "camera_ids": item["coverage"]["camera_ids"],
                  "min_world_m": points.min(axis=0).tolist(), "max_world_m": points.max(axis=0).tolist(),
                  "max_reconstruction_error_m": maximum_error}
        matching_oracle = [obj for obj in scene["objects"] if obj["kind"] == item["kind"] and
                           obj["color"] == item["color"]]
        if len(matching_oracle) == 1:
            oracle = matching_oracle[0]
            low = np.asarray(oracle["position"]) - np.asarray(oracle["size"]) / 2
            high = np.asarray(oracle["position"]) + np.asarray(oracle["size"]) / 2
            outside = np.maximum(np.maximum(low - points, points - high), 0.)
            record["fraction_inside_oracle_bounds_3mm"] = float(np.mean(np.max(outside, axis=1) < .003))
            checks[f"instance_{index}_calibrated_world_bounds"] = record["fraction_inside_oracle_bounds_3mm"] > .98
        if item["kind"] == "bin":
            reference = item.get("placement_reference")
            checks[f"instance_{index}_measured_floor_reference"] = reference is not None
            if reference:
                record["floor_z_m"] = reference["floor_z_m"]
                checks[f"instance_{index}_floor_height_3mm"] = abs(reference["floor_z_m"] - .61) < .003
                camera_id = reference["measurement_sources"]["reference_camera_id"]
                pixel = reference["measurement_sources"]["reference_pixel_rc"]
                checks[f"instance_{index}_reference_is_measured_point"] = bool(np.any(
                    (sources["camera_ids"] == camera_id) & np.all(sources["pixels_rc"] == pixel, axis=1) &
                    (np.linalg.norm(points - reference["position_world_m"], axis=1) < 1e-8)))
        details.append(record)
        np.savez_compressed(output / f"instance-{index}.npz", points_world_m=points,
                            camera_ids=sources["camera_ids"], pixels_rc=sources["pixels_rc"])
    return {"checks": checks, "passed": all(checks.values()), "instances": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-url", default="http://192.168.3.4:8322")
    parser.add_argument("--destination-color", default="blue", choices=("blue", "green", "red"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/visible-cloud-work/perception-live-smoke")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = normalize_scene(SCENE)
    scene["objects"][1]["color"] = args.destination_color
    poses = json.loads((ROOT / "resources/observation_poses.json").read_text())["joints_rad"]
    builder = ModelBuilder()
    result = {"scope": "offline camera fixtures; no execution, no live web mutation",
              "mujoco_version": mujoco.__version__, "sam3_url": args.sam3_url,
              "calibration_sha256": hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest(),
              "input_boundary": "registered RGBD and own joint telemetry only", "checkpoints": {}}
    observations = {}
    try:
        model = builder.build(scene)
        data = mujoco.MjData(model)
        renderer = RgbdRenderer(clone_render_model(model))
        try:
            for checkpoint, joints in (("initial", {"A": [0.] * 7, "B": [0.] * 7}), ("fine", poses)):
                output = args.output / checkpoint
                output.mkdir(exist_ok=True)
                for arm, values in joints.items():
                    for name, value in zip(ARM_JOINTS[arm], values, strict=True):
                        data.qpos[model.joint(name).qposadr[0]] = value
                mujoco.mj_forward(model, data)
                arms = {arm: {"joints_rad": values} for arm, values in joints.items()}
                capture = renderer.capture(data, 1, arms)
                save_capture(capture, output)
                detector = RecordingDetector(args.sam3_url, capture, output)
                started = time.monotonic()
                try:
                    reference = observations.get('initial')
                    hint = {'kind': 'cube', 'color': 'red',
                            'destination': {'kind': 'bin', 'color': args.destination_color},
                            'include_containers': True,
                            'visual_decision': {'source': 'jev', 'purpose':
                                'associate_inventory' if reference else 'initialize_inventory'}}
                    if reference:
                        hint['visual_reference'] = reference
                        hint['scene_entities'] = {obj['id']: {key: obj[key] for key in ('id', 'kind', 'color')}
                                                  for obj in reference['objects']}
                    observation = Perception(detector).observe(capture, instruction="抓取红色方块到蓝色盒子里",
                        checkpoint=checkpoint, source_hint=hint)
                    save_json(output / "observation.json", public_observation(observation))
                    observations[checkpoint] = observation
                    evidence = evaluate(observation, capture, scene, checkpoint, output)
                except PerceptionError as error:
                    evidence = {"passed": False, "error": f"{type(error).__name__}: {error}"}
                evidence["duration_s"] = time.monotonic() - started
                result["checkpoints"][checkpoint] = evidence
                save_json(args.output / "result.json", result)
                print(json.dumps({"checkpoint": checkpoint, **evidence}), flush=True)
        finally:
            renderer.close()
        # Independent test oracle is deliberately read/saved after detector/perception.
        save_json(args.output / "independent-oracle.json", scene)
        if len(observations) == 2:
            initial_ids = {item['id'] for item in observations['initial']['objects']}
            current_ids = {item['id'] for item in observations['fine']['objects']}
            result['checkpoint_association'] = {'passed': current_ids == initial_ids,
                'source': 'openai_decisions', 'initial_ids': sorted(initial_ids), 'current_ids': sorted(current_ids)}
        result["passed"] = all(item["passed"] for item in result["checkpoints"].values()) and \
            result.get("checkpoint_association", {}).get("passed", False)
    finally:
        builder.close()
        save_json(args.output / "result.json", result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

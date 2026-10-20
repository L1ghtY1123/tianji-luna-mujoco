"""Offline fixed-pose calibration; reference geometry is never runtime evidence.

Uses public MuJoCo 3.10 FK, signed geometry distance and registered RGBD.
Only a new resource/evidence file is written; physical startup remains untouched.
"""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image
from scipy.optimize import least_squares

from jev_mujoco.cameras import RgbdRenderer, camera_pose, clone_render_model, deproject_view
from jev_mujoco.model_builder import ARM_JOINTS, ModelBuilder, normalize_scene

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCENE = {"objects": [
    {"id": "source", "kind": "cube", "color": "red", "position": [.43, .15, .625],
     "size": [.05, .05, .05]},
    {"id": "destination", "kind": "bin", "color": "green", "position": [.65, -.15, .65],
     "size": [.18, .16, .10]},
]}


def set_joints(model, data, poses):
    for arm, values in poses.items():
        for name, value in zip(ARM_JOINTS[arm], values):
            data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)


@lru_cache(maxsize=1)
def sphere_config():
    return yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_text())["robot_cfg"]["kinematics"]


def clearance(model, data):
    config = sphere_config()
    ignored = config["self_collision_ignore"]
    robot = {}
    for link, spheres in config["collision_spheres"].items():
        centers = np.asarray([sphere["center"] for sphere in spheres])
        radii = np.asarray([sphere["radius"] for sphere in spheres])
        radii += config["collision_sphere_buffer"].get(link, 0.)
        body = data.body(link)
        robot[link] = (centers @ body.xmat.reshape(3, 3).T + body.xpos, radii)
    minima = {"table": [1., None], "peer": [1., None], "self": [1., None]}
    names = list(robot)
    for first, body1 in enumerate(names):
        centers1, radii1 = robot[body1]
        if body1 not in ("Link_Base", "Link_Stand", "Base_L", "Base_R"):
            offsets = np.abs(centers1 - [.55, 0., .56]) - [.35, .50, .04]
            signed = np.linalg.norm(np.maximum(offsets, 0.), axis=1) + np.minimum(offsets.max(axis=1), 0.)
            distance = np.min(signed - radii1)
            if distance < minima["table"][0]:
                minima["table"] = [float(distance), [body1, "table"]]
        for second, body2 in enumerate(names):
            if second <= first:
                continue
            if body1 == body2 or body2 in ignored.get(body1, []) or body1 in ignored.get(body2, []):
                continue
            left1 = body1.startswith("gripL") or (body1.startswith(("Link", "Base", "TCP")) and body1.endswith("_L"))
            left2 = body2.startswith("gripL") or (body2.startswith(("Link", "Base", "TCP")) and body2.endswith("_L"))
            category = "peer" if left1 != left2 else "self"
            centers2, radii2 = robot[body2]
            distances = np.linalg.norm(centers1[:, None] - centers2[None, :], axis=2)
            if category == "self":
                distances -= config["self_collision_buffer"].get(body1, 0.)
                distances -= config["self_collision_buffer"].get(body2, 0.)
            distance = np.min(distances - radii1[:, None] - radii2[None, :])
            if distance < minima[category][0]:
                minima[category] = [float(distance), [body1, body2]]
    return minima


def search(model, data):
    rng = np.random.default_rng(20261009)
    candidates = {}
    for arm, camera_id, sign in (("A", "2", 1), ("B", "1", -1)):
        indices = [model.joint(name).qposadr[0] for name in ARM_JOINTS[arm]]
        bounds = np.asarray([model.joint(name).range for name in ARM_JOINTS[arm]])
        options = []
        for position in ([.25, sign * .42, 1.05], [.35, sign * .45, 1.10], [.20, sign * .40, .95]):
            position = np.asarray(position)
            forward = np.asarray([.55, 0., .63]) - position
            forward /= np.linalg.norm(forward)
            def residual(values, indices=indices, camera_id=camera_id, forward=forward, position=position):
                data.qpos[indices] = values
                mujoco.mj_fwdPosition(model, data)
                pose = camera_pose(model, data, camera_id)
                direction = pose[:3, 2] - forward
                wrist_bend = max(1.00 - abs(values[5]), 0.)
                return np.concatenate((4 * (pose[:3, 3] - position), direction, [wrist_bend]))

            for attempt in range(12):
                seed = np.zeros(7) if attempt == 0 else rng.uniform(bounds[:, 0] * .8, bounds[:, 1] * .8)
                if attempt == 0:
                    seed[5] = sign * 1.0
                solution = least_squares(residual, seed, bounds=(bounds[:, 0], bounds[:, 1]),
                                         max_nfev=130, ftol=1e-6)
                if np.linalg.norm(solution.fun) < .08:
                    options.append(solution.x.tolist())
        candidates[arm] = options
        data.qpos[indices] = 0
        print(arm, "camera IK candidates", len(options), flush=True)
    best = None
    for first in candidates["A"]:
        for second in candidates["B"]:
            poses = {"A": first, "B": second}
            set_joints(model, data, poses)
            distances = clearance(model, data)
            score = min(item[0] for item in distances.values())
            if best is None or score > best[0]:
                best = (score, poses, distances)
            if distances["table"][0] >= .01 and distances["peer"][0] >= .01 and distances["self"][0] > 0.:
                print("clear endpoints", distances, flush=True)
                return poses
    print("best rejected endpoint", best, flush=True)
    raise RuntimeError("no endpoint clears own collision model and reference table")


def validate(model, data, poses, output):
    set_joints(model, data, {"A": poses["A"], "B": [0.] * 7})
    a_only_distances = clearance(model, data)
    if a_only_distances["table"][0] < .01 or a_only_distances["peer"][0] < .01 or a_only_distances["self"][0] <= 0.:
        raise ValueError(f"A endpoint with B horizontal failed: {a_only_distances}")
    set_joints(model, data, poses)
    distances = clearance(model, data)
    if distances["table"][0] < .01 or distances["peer"][0] < .01 or distances["self"][0] <= 0.:
        raise ValueError(f"endpoint collision clearance failed: {distances}")
    for contact in data.contact:
        if model.geom_group[contact.geom1] == 3 or model.geom_group[contact.geom2] == 3:
            raise ValueError("MuJoCo articulated robot geometry contacts reference scene")
    renderer = RgbdRenderer(clone_render_model(model))
    evidence = {}
    try:
        arms = {arm: {"joints_rad": values} for arm, values in poses.items()}
        capture = renderer.capture(data, 1, arms)
        for camera_id, view in capture["views"].items():
            rgb = view["rgb"].astype(float)
            red = (rgb[..., 0] > 1.7 * rgb[..., 1]) & (rgb[..., 0] > 1.7 * rgb[..., 2])
            green = (rgb[..., 1] > 1.5 * rgb[..., 0]) & (rgb[..., 1] > 1.3 * rgb[..., 2])
            points = deproject_view(view)
            table = (np.abs(points[:, 2] - .60) < .004) & (points[:, 0] > .30) & (points[:, 0] < .80)
            table &= np.abs(points[:, 1]) < .30
            counts = {"source_red_pixels": int(np.count_nonzero(red & view["valid_depth"])),
                      "destination_green_pixels": int(np.count_nonzero(green & view["valid_depth"])),
                      "tabletop_measured_points": int(np.count_nonzero(table))}
            if counts["source_red_pixels"] <= 100 or counts["destination_green_pixels"] <= 100:
                raise ValueError(f"camera {camera_id} cannot see reference source/destination: {counts}")
            if counts["tabletop_measured_points"] <= 3000:
                raise ValueError(f"camera {camera_id} insufficient reference tabletop RGBD: {counts}")
            output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(view["rgb"]).save(output / f"camera-{camera_id}.png")
            np.savez_compressed(output / f"camera-{camera_id}-rgbd.npz", rgb=view["rgb"],
                                depth_m=view["depth_m"], valid_depth=view["valid_depth"],
                                T_world_camera=view["T_world_camera"])
            evidence[camera_id] = {**counts, "T_world_camera": view["T_world_camera"]}
    finally:
        renderer.close()
    return {"mujoco_version": mujoco.__version__, "endpoint_clearance_m": distances,
            "A_endpoint_B_horizontal_clearance_m": a_only_distances,
            "mujoco_robot_scene_contacts": 0,
            "rgbd_views": evidence, "reference_scene": normalize_scene(REFERENCE_SCENE),
            "minimum_clearance_m": {"table": .01, "peer": .01, "self": 0.},
            "scope": "offline endpoint geometry and rendered RGBD; runtime path must be collision planned",
            "collision_geometry": "shipped planner link-local spheres with world/self buffers",
            "self_collision_exclusions": "existing marvin_gui_dual.yml self_collision_ignore",
            "table_exclusions": ["Link_Base", "Link_Stand", "Base_L", "Base_R"],
            "environment_truth_in_runtime": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="replay saved endpoint instead of searching")
    args = parser.parse_args()
    path = ROOT / "resources/observation_poses.json"
    builder = ModelBuilder()
    try:
        model = builder.build(normalize_scene(REFERENCE_SCENE))
        data = mujoco.MjData(model)
        poses = json.loads(path.read_text())["joints_rad"] if args.verify else search(model, data)
        evidence = validate(model, data, poses, ROOT / "data/visible-cloud-work/observation-poses")
        paths = ["assets/marvin_urdf/marvin_with_grippers.urdf", "resources/cameras/calibration.json",
                 "resources/curobo/marvin_gui_dual.yml", "src/jev_mujoco/model_builder.py",
                 "src/jev_mujoco/cameras.py", "scripts/calibrate_observation_poses.py"]
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
        record = {"schema_version": 1, "model_validated": True, "joints_rad": poses,
                  "validation": evidence, "source_sha256": hashes, "search_seed": 20261009,
                  "provenance": "own robot model/calibration offline camera-pose IK; actual RGBD verification",
                  "startup_joints_rad": {"A": [0.] * 7, "B": [0.] * 7}}
        if not args.verify:
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
    finally:
        builder.close()


if __name__ == "__main__":
    main()

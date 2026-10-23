"""CPU own-FK and raw measured-obstacle provenance for the exact A15 hard rejection."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from luna_mujoco.carrying import configure_gripper_feedback

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "data/visible-cloud-work"
recording = folder / "grasp-place-left-15"
probe = json.loads((folder / "a15-exact-path-probe.json").read_text(encoding="utf-8"))
payload = json.loads((recording / "gpu-requests/021-tools-plan_place.json").read_text(encoding="utf-8"))["body"]
mapping = json.loads((recording / "gpu-requests/020-perception-build.json").read_text(encoding="utf-8"))["body"]
frame = mapping["observation_set_id"]
observation = np.load(recording / f"perception-{frame}.npz", allow_pickle=True)["observation"].item()
capture = np.load(recording / f"rgbd-{frame}.npz", allow_pickle=True)["capture"].item()
spec = importlib.util.spec_from_file_location("own_robot_audit", ROOT / "scripts/audit_collision_model.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
urdf = ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf"
robot = ET.parse(urdf).getroot()
config = configure_gripper_feedback(yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_bytes()),
                                    payload["gripper_joint_positions_rad"], urdf)["robot_cfg"]["kinematics"]
goal = np.eye(4)
goal[:3, :3] = Rotation.from_quat([*payload["goal_quaternion_wxyz"][1:], payload["goal_quaternion_wxyz"][0]]).as_matrix()
goal[:3, 3] = payload["goal_position_m"]
query = next(x for x in probe["world_queries"] if x["phase"] == "place")
report = {"gpu_used": False, "oracle_used": False, "phase": "place", "samples": []}
for positive in query["positive"]:
    index = positive["sample"]
    q = np.asarray(query["joints_rad"])[0, index]
    joints = {f"Joint{i}_L": float(v) for i, v in enumerate(q, 1)}
    joints.update({f"Joint{i}_R": float(v) for i, v in enumerate(np.deg2rad(payload["other_joints_deg"]), 1)})
    transforms = audit.forward_kinematics(robot, config, joints)
    tcp = transforms["TCP_Link_L"]
    sphere = np.asarray(positive["sphere_world"])
    base = audit.sphere_modes(config, positive["link"])["world_buffered"]
    link = transforms[positive["link"]]
    base[:, :3] = base[:, :3] @ link[:3, :3].T + link[:3, 3]
    fk_error = float(np.linalg.norm(base[:, :3]-sphere[:3], axis=1).min())
    assert fk_error < 1e-6
    entry = {**positive, "actual_tcp": tcp.tolist(), "requested_release_tcp": goal.tolist(),
        "position_error_world_m": (tcp[:3, 3]-goal[:3, 3]).tolist(),
        "position_error_norm_m": float(np.linalg.norm(tcp[:3, 3]-goal[:3, 3])),
        "rotation_error_rad": float(Rotation.from_matrix(goal[:3, :3].T @ tcp[:3, :3]).magnitude()),
        "native_sphere_center_fk_error_m": fk_error, "measured_object_neighbors": []}
    for obj in observation["objects"]:
        points = np.asarray(obj["points_world_m"])
        distance = np.linalg.norm(points-sphere[:3], axis=1)
        nearest = int(np.argmin(distance))
        sources = obj["point_sources"]
        camera_id = str(sources["camera_ids"][nearest])
        row, column = sources["pixels_rc"][nearest]
        view = capture["views"][camera_id]
        raw_depth = float(view["depth_m"][row, column])
        k = view["intrinsics"]
        point_camera = np.array([(column-k["cx"])*raw_depth/k["fx"],
                                 (row-k["cy"])*raw_depth/k["fy"], raw_depth])
        transform = np.asarray(view["T_world_camera"])
        point_world = transform[:3, :3] @ point_camera + transform[:3, 3]
        entry["measured_object_neighbors"].append({"instance_id": obj["id"], "color": obj["color"],
            "point_index": nearest, "point_world_m": points[nearest].tolist(), "distance_to_center_m": float(distance[nearest]),
            "sphere_surface_gap_m": float(distance[nearest]-sphere[3]), "camera_id": camera_id,
            "pixel_rc": [int(row), int(column)], "raw_depth_m": raw_depth,
            "raw_valid": bool(view["valid_depth"][row, column]), "raw_rgb": view["rgb"][row, column].tolist(),
            "raw_deprojection_error_m": float(np.linalg.norm(point_world-points[nearest]))})
    report["samples"].append(entry)
    print(json.dumps({k: entry[k] for k in ("sample", "index", "link", "position_error_world_m", "position_error_norm_m",
                                           "rotation_error_rad", "measured_object_neighbors")}), flush=True)
report["limits"] = ["Positive cost is not a penetration distance.",
                    "Nearest raw object returns support obstacle provenance; aggregate ESDF query does not attribute a layer.",
                    "Own CPU TCP FK is checked against native sphere FK; no new target or planning variant was evaluated."]
(folder / "a15-path-cpu-diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

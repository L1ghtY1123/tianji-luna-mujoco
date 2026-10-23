"""Trace A17 measured near-Link4 returns using own robot meshes and recorded RGBD."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yaml

from luna_mujoco.carrying import configure_gripper_feedback

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "data/visible-cloud-work"
recording = folder / "grasp-place-left-17"
probe = json.loads((folder / "a17-held-state-probe.json").read_text(encoding="utf-8"))
payload = json.loads((recording / "gpu-requests/021-tools-plan_place.json").read_text(encoding="utf-8"))["body"]
stage = json.loads((recording / "stage-1.json").read_text(encoding="utf-8"))
grasp = np.asarray(stage["holding"]["grasp_support_reference"]["T_world_tcp_at_grasp"])
spec = importlib.util.spec_from_file_location("own_robot_audit", ROOT / "scripts/audit_collision_model.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
urdf = ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf"
robot = ET.parse(urdf).getroot()
config = configure_gripper_feedback(yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_bytes()),
                                    payload["gripper_joint_positions_rad"], urdf)["robot_cfg"]["kinematics"]
joints = {f"Joint{i}_L": float(v) for i, v in enumerate(np.deg2rad(payload["start_joints_deg"]), 1)}
joints.update({f"Joint{i}_R": float(v) for i, v in enumerate(np.deg2rad(payload["other_joints_deg"]), 1)})
transforms = audit.forward_kinematics(robot, config, joints)
tool = transforms["TCP_Link_L"]
report = {"gpu_used": False, "oracle_used": False, "points": []}
for point_index, link, body_index in ((8067, "gripL_Link4_L", 232), (675, "gripL_Link4_R", 236)):
    point_tool = np.asarray(payload["carrying_binding"]["points_tool_m"])[point_index]
    world = point_tool @ tool[:3, :3].T + tool[:3, 3]
    native = np.asarray(probe["measured_world_points"])[point_index]
    assert np.linalg.norm(world-native) < 1e-6
    spheres = audit.sphere_modes(config, link)["self_effective"]
    center = spheres[-1, :3] @ transforms[link][:3, :3].T + transforms[link][:3, 3]
    sphere_error = np.linalg.norm(center-np.asarray(probe["world_spheres"])[body_index, :3])
    assert sphere_error < 1e-6
    entry = {"union_index": point_index, "body_link": link, "tool_m": point_tool.tolist(), "world_m": world.tolist(),
             "native_body_sphere_fk_error_m": float(sphere_error), "mesh": {}, "pixel_matches": []}
    link_node = next(node for node in robot.findall("link") if node.get("name") == link)
    for mode in ("visual", "collision"):
        node = link_node.find(mode)
        mesh_node = node.find("geometry/mesh")
        path = urdf.parent / mesh_node.get("filename")
        mesh = trimesh.load_mesh(path, process=False)
        mesh.vertices *= np.asarray([float(v) for v in mesh_node.get("scale", "1 1 1").split()])
        mesh.apply_transform(transforms[link] @ audit.origin_transform(node))
        mesh.vertices *= 1000
        nearest, distance, _ = trimesh.proximity.closest_point_naive(mesh, world[None] * 1000)
        entry["mesh"][mode] = {"distance_m": float(distance[0]/1000),
            "nearest_surface_world_m": (nearest[0]/1000).tolist(), "file": path.relative_to(ROOT).as_posix()}
    for observation_path in recording.glob("perception-*.npz"):
        observation = np.load(observation_path, allow_pickle=True)["observation"].item()
        for obj in observation["objects"]:
            if obj["color"] != "red" or obj["kind"] != "cube":
                continue
            points = np.asarray(obj["points_world_m"])
            for reference_name, reference in (("actual_grasp_tcp", grasp), ("actual_capture_tcp", np.asarray(observation["arms"]["A"]["tcp"]))):
                local = (points-reference[:3, 3]) @ reference[:3, :3]
                distances = np.linalg.norm(local-point_tool, axis=1)
                index = int(np.argmin(distances))
                if distances[index] > 1e-5:
                    continue
                frame = observation["observation_id"]
                capture = np.load(recording / f"rgbd-{frame}.npz", allow_pickle=True)["capture"].item()
                camera = str(obj["point_sources"]["camera_ids"][index])
                row, column = obj["point_sources"]["pixels_rc"][index]
                view = capture["views"][camera]
                entry["pixel_matches"].append({"observation_id": frame, "time": observation["time"],
                    "reference": reference_name, "source_id": obj["id"], "source_point_index": index,
                    "tool_match_error_m": float(distances[index]), "raw_world_m": points[index].tolist(),
                    "camera_id": camera, "pixel_rc": [int(row), int(column)],
                    "rgb": view["rgb"][row, column].tolist(), "raw_depth_m": float(view["depth_m"][row, column]),
                    "raw_valid": bool(view["valid_depth"][row, column])})
    report["points"].append(entry)
    print(json.dumps(entry), flush=True)
report["limits"] = ["Own mesh distance diagnoses return ownership, not unseen object-volume clearance.",
                    "No measured source points, radii, margins, collision pairs or settings were changed."]
(folder / "a17-attachment-cpu-diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

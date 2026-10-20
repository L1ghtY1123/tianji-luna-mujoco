"""Read-only own-robot mesh evidence for six actual trial14 measured returns."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yaml

from jev_mujoco.carrying import configure_gripper_feedback

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("own_robot_audit", ROOT / "scripts/audit_collision_model.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
folder = ROOT / "data/visible-cloud-work"
request = json.loads((folder / "grasp-place-left-14/gpu-requests/019-carrying-prepare.json").read_text(encoding="utf-8"))["body"]
proof = json.loads((folder / "trial14-self-pair-probe.json").read_text(encoding="utf-8"))
indices = np.array([4691, 4693, 4694, 4695, 4696, 4697])
urdf = ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf"
robot = ET.parse(urdf).getroot()
config = configure_gripper_feedback(yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_bytes()),
                                    request["gripper_joint_positions_rad"], urdf)["robot_cfg"]["kinematics"]
joints = {f"Joint{i}_L": float(v) for i, v in enumerate(np.deg2rad(request["start_joints_deg"]), 1)}
joints.update({f"Joint{i}_R": float(v) for i, v in enumerate(np.deg2rad(request["other_joints_deg"]), 1)})
transforms = audit.forward_kinematics(robot, config, joints)
tool = transforms["TCP_Link_L"]
points = np.asarray(request["carrying_binding"]["points_tool_m"])[indices]
world = points @ tool[:3, :3].T + tool[:3, 3]
native_points = np.asarray(proof["measured_world_points"])[indices]
point_error = np.linalg.norm(world - native_points, axis=1)
assert point_error.max() < 1e-6
result = {"gpu_used": False, "oracle_used": False, "physical_motion": False,
    "native_measured_point_fk_max_error_m": float(point_error.max()), "native_body_sphere_fk_errors_m": {},
    "points": [{"union_index": int(i), "world_m": p.tolist(), "mesh_candidates": []} for i, p in zip(indices, world)]}
for link, index in (("gripL_Link2_L", 219), ("gripL_Link4_L", 232)):
    spheres = audit.sphere_modes(config, link)["self_effective"]
    center = spheres[-1, :3] @ transforms[link][:3, :3].T + transforms[link][:3, 3]
    error = float(np.linalg.norm(center - np.asarray(proof["world_spheres"])[index, :3]))
    assert error < 1e-6
    result["native_body_sphere_fk_errors_m"][link] = error
for link_node in robot.findall("link"):
    link = link_node.get("name")
    if not link.startswith("gripL_") and link != "TCP_Link_L":
        continue
    for mode in ("collision", "visual"):
        for node in link_node.findall(mode):
            mesh_node = node.find("geometry/mesh")
            if mesh_node is None:
                continue
            path = urdf.parent / mesh_node.get("filename")
            mesh = trimesh.load_mesh(path, process=False)
            mesh.vertices *= np.asarray([float(v) for v in mesh_node.get("scale", "1 1 1").split()])
            mesh.apply_transform(transforms[link] @ audit.origin_transform(node))
            query = mesh.copy()
            query.vertices *= 1000
            nearest, distance, faces = trimesh.proximity.closest_point_naive(query, world * 1000)
            for i, point in enumerate(result["points"]):
                point["mesh_candidates"].append({"link": link, "mode": mode,
                    "mesh": path.relative_to(ROOT).as_posix(), "distance_m": float(distance[i]/1000),
                    "nearest_surface_world_m": (nearest[i]/1000).tolist(), "face_index": int(faces[i])})
for point in result["points"]:
    point["mesh_candidates"].sort(key=lambda x: x["distance_m"])
    point["closest_collision_mesh"] = next(x for x in point["mesh_candidates"] if x["mode"] == "collision")
    point["closest_visual_mesh"] = next(x for x in point["mesh_candidates"] if x["mode"] == "visual")
    print(point["union_index"], json.dumps({"collision": point["closest_collision_mesh"],
                                           "visual": point["closest_visual_mesh"]}), flush=True)
result["limits"] = ["Nearest robot-mesh surface evidence is measured-return ownership diagnosis, not an object-volume certificate.",
                    "No raw return or native collision geometry was removed by this probe."]
(folder / "trial14-robot-return-mesh-evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

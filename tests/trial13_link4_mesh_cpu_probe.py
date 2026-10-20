"""Read-only measured-source versus own Link4 model clearance, no GPU or oracle."""

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
request = json.loads((folder / "grasp-place-left-13/gpu-requests/023-tools-plan_place.json").read_text(encoding="utf-8"))["body"]
proof = json.loads((folder / "trial13-self-pair-probe.json").read_text(encoding="utf-8"))
urdf = ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf"
robot = ET.parse(urdf).getroot()
config = configure_gripper_feedback(yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_bytes()),
                                    request["gripper_joint_positions_rad"], urdf)["robot_cfg"]["kinematics"]
joints = {f"Joint{i}_L": float(v) for i, v in enumerate(np.deg2rad(request["start_joints_deg"]), 1)}
joints.update({f"Joint{i}_R": float(v) for i, v in enumerate(np.deg2rad(request["other_joints_deg"]), 1)})
transforms = audit.forward_kinematics(robot, config, joints)
tool = transforms["TCP_Link_L"]
points = np.asarray(request["carrying_binding"]["points_tool_m"])
world_points = points @ tool[:3, :3].T + tool[:3, 3]
result = {"physical_motion": False, "gpu_used": False, "oracle_used": False,
          "point_count": len(points), "geometry": "own_URDF_collision_STL", "links": {}}
for link, global_index in (("gripL_Link4_L", 232), ("gripL_Link4_R", 236)):
    node = next(n for n in robot.findall("link") if n.get("name") == link).find("collision")
    mesh_node = node.find("geometry/mesh")
    mesh = trimesh.load_mesh(urdf.parent / mesh_node.get("filename"), process=False)
    scale = np.asarray([float(v) for v in mesh_node.get("scale", "1 1 1").split()])
    mesh.vertices = np.asarray(mesh.vertices) * scale
    mesh.apply_transform(audit.origin_transform(node))
    local_bounds = mesh.bounds.copy()
    points_link = (world_points - transforms[link][:3, 3]) @ transforms[link][:3, :3]
    in_bounds = np.all((points_link >= local_bounds[0]) & (points_link <= local_bounds[1]), axis=1)
    mesh.apply_transform(transforms[link])
    mesh_tool = (mesh.vertices - tool[:3, 3]) @ tool[:3, :3]
    # Millimetre coordinates avoid trimesh's absolute tiny-triangle tolerance.
    query = mesh.copy()
    query.vertices *= 1000
    distance, closest = [], []
    for start in range(0, len(points), 32):
        p, d, _ = trimesh.proximity.closest_point_naive(query, world_points[start:start+32] * 1000)
        closest.extend(p / 1000)
        distance.extend(d / 1000)
    distance = np.asarray(distance)
    nearest = int(np.argmin(distance))
    spheres = audit.sphere_modes(config, link)["self_effective"]
    spheres[:, :3] = spheres[:, :3] @ transforms[link][:3, :3].T + transforms[link][:3, 3]
    native = np.asarray(proof["world_spheres"])[global_index]
    fk_error = float(np.linalg.norm(spheres[-1, :3] - native[:3]))
    assert fk_error < 1e-6, (link, fk_error)
    gaps = audit.sphere_gaps(world_points, spheres)
    result["links"][link] = {"closest_measured_point_index": nearest,
        "closest_measured_point_world_m": world_points[nearest].tolist(),
        "nearest_mesh_point_world_m": np.asarray(closest)[nearest].tolist(),
        "minimum_unsigned_mesh_distance_m": float(distance[nearest]),
        "minimum_measured_point_to_configured_self_sphere_gap_m": float(gaps.min()),
        "native_sphere_center_fk_error_m": fk_error, "mesh_faces": len(mesh.faces),
        "measured_points_inside_mesh_local_AABB_count": int(in_bounds.sum()),
        "measured_minimum_tool_z_m": float(points[:, 2].min()),
        "link_mesh_maximum_tool_z_m": float(mesh_tool[:, 2].max()),
        "distance_quantiles_m": np.quantile(distance, [0, .25, .5, .75, 1]).tolist()}
    print(link, json.dumps(result["links"][link]), flush=True)
result["limits"] = ["Unsigned measured-point distances do not certify unseen object volume or physical clearance.",
                    "Native collision spheres and attachment uncertainty envelopes remain unchanged."]
(folder / "trial13-link4-mesh-clearance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

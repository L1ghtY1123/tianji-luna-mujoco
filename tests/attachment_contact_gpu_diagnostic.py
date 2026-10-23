"""Read-only own-hand contact audit using real RGBD candidate-local geometry."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def mesh_separating_gap(urdf, link, link_matrix_tool, source_vertices, source_axes):
    from scipy.spatial.transform import Rotation
    collision = ET.parse(urdf).getroot().find(f"link[@name='{link}']/collision")
    if collision is None or collision.find("geometry/mesh") is None:
        return None
    filename = collision.find("geometry/mesh").attrib["filename"]
    path = Path(filename)
    if not path.is_absolute():
        path = urdf.parent / path
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], "little")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    if len(data) != 84 + count * 50:
        return {"unsupported_mesh_format": str(path)}
    triangles = np.frombuffer(data, dtype=dtype, offset=84)["vertices"].astype(float)
    origin = collision.find("origin")
    fixed = np.eye(4)
    if origin is not None:
        fixed[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        fixed[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    matrix = link_matrix_tool @ fixed
    triangles = triangles @ matrix[:3, :3].T + matrix[:3, 3]
    edges = triangles[:, 1] - triangles[:, 0]
    normals = np.cross(edges, triangles[:, 2] - triangles[:, 0])
    axes = np.vstack((np.eye(3), source_axes.T, normals))
    axes = axes[np.linalg.norm(axes, axis=1) > 1e-10]
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    vertices = triangles.reshape(-1, 3)
    maximum = -float("inf")
    # Any separating projection is a valid lower-bound gap; absence of a
    # separating axis here does not prove mesh collision for a concave link.
    for chunk in np.array_split(axes, max(1, len(axes) // 64)):
        a, b = vertices @ chunk.T, source_vertices @ chunk.T
        maximum = max(maximum, float(np.maximum(a.min(axis=0)-b.max(axis=0), b.min(axis=0)-a.max(axis=0)).max()))
    return {"mesh_projection_gap_m": maximum, "positive_gap_proves_separation": maximum > 0,
            "source": "raw_collision_STL_and_RGBD_prior_vertices_in_TCP_frame"}


def main():
    import torch
    import yaml
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    from scipy.spatial.transform import Rotation
    from tianji_curobo.curobo_robot_config import create_dual_arm_robot_config
    from tianji_curobo.curobo_service import materialize_urdf

    from luna_mujoco.carrying import CarryingBinding, carrying_spheres, configure_gripper_feedback
    from luna_mujoco.geometry_service import ROOT, sdk_identity

    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    original = yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_bytes())
    inputs = json.loads(Path("/opt/jev/tests/artifacts/task3/attachment-contact-input.json").read_text())
    report = {"sdk": sdk_identity()[0], "input": "688 measured RGBD points plus separately estimated volume",
              "pose_limitation": "nominal solved candidate TCP; not a postclosure measured held pose",
              "arm_configuration": "zero joints for relative own-hand geometry; no task planning/execution", "arms": {}}
    for arm in ("A", "B"):
        value = inputs[arm]
        scene = {"scene_token": "contact-audit", "observation_id": "rgbd-audit", "generation": 1, "target_id": "source"}
        binding = CarryingBinding.from_mapping({"arm": arm, "owner_id": "audit", "observation_id": "rgbd-audit",
            "generation": 1, "scene_token": "contact-audit", "instance_id": "source", "frame": "tool", "state": "preflight",
            "points_tool_m": value["points_tool_m"], "estimated_bounds_tool_m": value["estimated_bounds_tool_m"],
            "shape_prior_provenance": {"source": "registered_rgbd_shape_estimate", "reference_observation_id": "rgbd-audit",
                                       "instance_id": "source", "assumptions": ["recorded real RGBD candidate-local box prior"]},
            "uncertainty_m": .005, "coverage": .5}, scene=scene, arm=arm)
        body = np.asarray(carrying_spheres(binding))
        prefix, tool = ("gripL_", "TCP_Link_L") if arm == "A" else ("gripR_", "TCP_Link_R")
        links = [tool, *[link for link in original["robot_cfg"]["kinematics"]["collision_link_names"] if link.startswith(prefix)]]
        results = {}
        for driver in (0., .55, .8):
            robot = create_dual_arm_robot_config(original, urdf_path=urdf, asset_root_path=urdf.parent)
            robot = configure_gripper_feedback(robot, {"A": driver if arm == "A" else 0., "B": driver if arm == "B" else 0.}, urdf)
            model = Kinematics(KinematicsCfg.from_data_dict(robot, tool_frames=links))
            q = torch.zeros((1, len(model.joint_names)), **model.device_cfg.as_torch_dict())
            state = model.compute_kinematics(JointState.from_position(q, joint_names=model.joint_names))
            poses = model.get_link_poses(q, links)
            matrices = np.repeat(np.eye(4)[None], len(links), axis=0)
            matrices[:, :3, 3] = poses.position.detach().cpu().numpy().reshape(-1, 3)
            quaternions = poses.quaternion.detach().cpu().numpy().reshape(-1, 4)
            matrices[:, :3, :3] = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
            inverse = np.linalg.inv(matrices[0])
            spheres = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
            current = {}
            for index, link in enumerate(links):
                indices = model.kinematics_config.get_sphere_index_from_link_name(link).detach().cpu().tolist()
                hand = spheres[indices].copy()
                hand[:, :3] = hand[:, :3] @ inverse[:3, :3].T + inverse[:3, 3]
                gap = np.linalg.norm(body[:, None, :3]-hand[None, :, :3], axis=-1)-body[:, None, 3]-hand[None, :, 3]
                current[link] = {"minimum_sphere_gap_m": float(gap.min()),
                    "currently_exempted": True, "actual_pad_link": link in (prefix+"Link3_L", prefix+"Link3_R")}
                if gap.min() <= 0:
                    current[link]["mesh_diagnostic"] = mesh_separating_gap(urdf, link, inverse @ matrices[index],
                        np.asarray(value["source_vertices_tool_m"]), np.asarray(value["source_axes_tool"]))
            results[str(driver)] = current
        report["arms"][arm] = {"bounds_tool_m": value["estimated_bounds_tool_m"], "driver_results": results}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

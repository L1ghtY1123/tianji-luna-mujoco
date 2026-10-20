"""Exact recorded trial14 held configuration, no target reconstruction or motion."""

import argparse
import json

import numpy as np
import torch
import yaml
from curobo.types import JointState
from tianji_curobo.curobo_service import materialize_urdf

from jev_mujoco.carrying import CarryingBinding
from jev_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity

parser = argparse.ArgumentParser()
parser.add_argument("--input-folder", default="data/visible-cloud-work/grasp-place-left-14/gpu-requests")
parser.add_argument("--mapping-name", default="018-perception-build.json")
parser.add_argument("--request-name", default="019-carrying-prepare.json")
options = parser.parse_args()
folder = ROOT / options.input_folder
mapping = json.loads((folder / options.mapping_name).read_text())
preparing = json.loads((folder / options.request_name).read_text())
payload = preparing["body"]
carried_link = f"jev_carried_object_{payload['arm']}"
urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
    marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
try:
    scene = run_operation(runtime, mapping["path"], mapping["body"])
    assert scene["scene_token"] == payload["scene_token"]
    prepared = run_operation(runtime, "/carrying/prepare", payload)
    binding = CarryingBinding.from_mapping(payload["carrying_binding"], scene=scene, arm=payload["arm"])
    with runtime.carrying_scope(binding, payload["gripper_joint_positions_rad"]):
        backend = next(iter(runtime._compound_backends.values()))
        planner, checker = backend.motion_planner, backend._motion_clearance_checker
        positions = torch.as_tensor(np.deg2rad(payload["start_joints_deg"])[None, None],
                                    **planner.device_cfg.as_torch_dict())
        state = checker.get_kinematics(positions)
        spheres = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
        world = checker.get_collision_distance(state).detach().cpu().numpy().reshape(-1)
        self_cost = checker.get_self_collision_distance(state.robot_spheres).detach().cpu().numpy()
        indices = checker.kinematics.kinematics_config.get_sphere_index_from_link_name(carried_link)
        attached = spheres[indices.detach().cpu().numpy().reshape(-1)]
        kin_config = checker.kinematics.kinematics_config
        robot = yaml.safe_load(runtime._robot_config_path.read_bytes())["robot_cfg"]["kinematics"]
        labels = {}
        for link in [*robot["collision_link_names"], carried_link]:
            try:
                link_indices = kin_config.get_sphere_index_from_link_name(link).detach().cpu().numpy().reshape(-1)
            except KeyError:
                continue
            labels.update({int(index): link for index in link_indices})
        self_config = checker.kinematics.get_self_collision_config()
        pairs = self_config.collision_pairs.detach().cpu().numpy().astype(int)
        padding = self_config.sphere_padding.detach().cpu().numpy()
        gaps = np.linalg.norm(spheres[pairs[:, 0], :3] - spheres[pairs[:, 1], :3], axis=-1)
        gaps -= spheres[pairs[:, 0], 3] + spheres[pairs[:, 1], 3] + padding[pairs[:, 0]] + padding[pairs[:, 1]]
        joint_state = JointState.from_position(positions[:, 0], joint_names=planner.joint_names)
        tcp_frame = "TCP_Link_L" if payload["arm"] == "A" else "TCP_Link_R"
        tcp = planner.compute_kinematics(joint_state).tool_poses.get_link_pose(tcp_frame)
        measured = tcp.transform_points(torch.as_tensor(binding.points_tool_m, **planner.device_cfg.as_torch_dict()))
        measured_world = measured.detach().cpu().numpy().reshape(-1, 3)
        negative = []
        for pair_index in np.flatnonzero(gaps < 0):
            pair = pairs[pair_index]
            body_point_gaps = {}
            for index in pair:
                if labels.get(int(index)) == carried_link:
                    continue
                point_gaps = np.linalg.norm(measured_world - spheres[index, :3], axis=-1)
                point_gaps -= spheres[index, 3] + padding[index]
                nearest = int(np.argmin(point_gaps))
                body_point_gaps[str(index)] = {"measured_point_index": nearest,
                    "measured_point_world_m": measured_world[nearest].tolist(),
                    "minimum_measured_point_to_self_sphere_gap_m": float(point_gaps[nearest])}
            negative.append({"indices": pair.tolist(), "links": [labels.get(int(k)) for k in pair],
                "sphere_world": spheres[pair].tolist(), "sphere_padding": padding[pair].tolist(),
                "self_surface_gap_m": float(gaps[pair_index]), "measured_point_gaps": body_point_gaps})
        result = {"sdk": sdk_identity()[0], "physical_motion": False, "oracle_used": False,
            "exact_requests": [options.mapping_name, options.request_name], "arm": payload["arm"],
            "scene_token": scene["scene_token"], "prepared": prepared,
            "measured_point_count": len(binding.points_tool_m),
            "gripper_feedback_rad": payload["gripper_joint_positions_rad"],
            "sphere_count": len(spheres), "attachment_sphere_count": len(attached),
            "attachment_radius_min_m": float(attached[:, 3].min()),
            "attachment_radius_max_m": float(attached[:, 3].max()),
            "world_positive_count": int(np.sum(world > 0)), "world_max_cost": float(world.max()),
            "self_positive_count": int(np.sum(self_cost > 0)), "self_max_cost": float(self_cost.max())}
        result.update(world_spheres=spheres.tolist(), sphere_labels=labels, measured_world_points=measured_world.tolist(),
            public_self_config={"collision_pairs": pairs.tolist(), "sphere_padding": padding.tolist()},
            negative_self_pairs=negative)
        result.update(world_costs=world.tolist(), world_layer_names=list(runtime.geometry_model.get("voxel", {})),
            positive_world_spheres=[{"index": int(index), "link": labels.get(int(index)),
                "sphere_world": spheres[index].tolist(), "world_cost": float(world[index])}
                for index in np.flatnonzero(world > 0)])
        print("WORLD_POSITIVE " + json.dumps(result["positive_world_spheres"]), flush=True)
        print("SELF_PAIRS " + json.dumps(negative), flush=True)
        print("RESULT " + json.dumps(result), flush=True)
finally:
    runtime.clear_carrying()

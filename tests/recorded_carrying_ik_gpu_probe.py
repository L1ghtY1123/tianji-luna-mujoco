"""Replay a complete recorded held-IK failure in an isolated GPU process.

No HTTP service or MuJoCo process is accessed. Inputs must include the complete
RGBD mapping requests and measured attachment, not a truncated event summary.
The public cuRobo IK/checker result is recorded alongside the project adapter.
This is a diagnostic entry point, not a pytest test or physical task acceptance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from curobo.types import GoalToolPose, Pose
from scipy.spatial.transform import Rotation
from tianji_curobo.curobo_service import materialize_urdf

from jev_mujoco.carrying import CarryingBinding
from jev_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-folder", type=Path, required=True,
                    help="RecordedAlgorithms directory containing complete gpu-requests, including RGBD and attachment")
parser.add_argument("--robot-config", type=Path, required=True)
parser.add_argument("--request-index", type=int, default=0, help="Index among recorded held IK calls")
parser.add_argument("--minimum-success", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
folder = args.run_folder
requests = [json.loads(file.read_text()) for file in sorted((folder / "gpu-requests").glob("*.json"))]
mappings = [row for row in requests if row["path"] == "/perception/build"]
prepare = next(row for row in requests if row["path"] == "/carrying/prepare")
request = [row for row in requests if row["path"] == "/tools/solve_ik_batch"
           and row["body"].get("carrying_binding")][args.request_index]
payload = request["body"]
urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
    marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent, robot_config_path=args.robot_config)
for mapping in mappings:
    scene = run_operation(runtime, mapping["path"], mapping["body"])
run_operation(runtime, prepare["path"], prepare["body"])
result = run_operation(runtime, request["path"], payload)
binding = CarryingBinding.from_mapping(payload["carrying_binding"], scene=scene, arm="A")
with runtime.carrying_scope(binding, payload["gripper_joint_positions_rad"]):
    backend = next(iter(runtime._ik_batch_backends.values()))
    ik, checker = backend.motion_planner, backend._motion_clearance_checker
    matrices = payload["goal_matrices_world"]
    n = len(matrices)
    matrices = np.asarray(matrices + [matrices[-1]] * (100 - n))
    target = Pose(position=torch.as_tensor(matrices[:, :3, 3], **ik.device_cfg.as_torch_dict()),
        quaternion=torch.as_tensor(Rotation.from_matrix(matrices[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]],
            **ik.device_cfg.as_torch_dict()))
    goal = GoalToolPose.from_poses({"TCP_Link_L": target}, ordered_tool_frames=["TCP_Link_L"], num_goalset=1)
    raw = ik.solve_pose(goal, return_seeds=16)
    state = checker.get_kinematics(raw.solution[:n])
    spheres = state.robot_spheres.detach().cpu().numpy()
    world = checker.get_collision_distance(state).detach().cpu().numpy()
    self_cost = checker.get_self_collision_distance(state.robot_spheres).detach().cpu().numpy()
    robot = yaml.safe_load(runtime._robot_config_path.read_bytes())["robot_cfg"]["kinematics"]
    labels = {}
    for name in [*robot["collision_link_names"], "jev_carried_object_A"]:
        indices = checker.kinematics.kinematics_config.get_sphere_index_from_link_name(name)
        labels.update({int(i): name for i in indices.detach().cpu().numpy().reshape(-1)})
    self_cfg = checker.kinematics.get_self_collision_config()
    pairs = self_cfg.collision_pairs.detach().cpu().numpy().astype(int)
    padding = self_cfg.sphere_padding.detach().cpu().numpy()
    negative_pairs = []
    positive_world = []
    for index in range(n):
        balls = spheres[index, 0]
        gaps = np.linalg.norm(balls[pairs[:, 0], :3] - balls[pairs[:, 1], :3], axis=-1)
        gaps -= balls[pairs[:, 0], 3] + balls[pairs[:, 1], 3] + padding[pairs[:, 0]] + padding[pairs[:, 1]]
        negative_pairs.append([{"links": [labels[int(k)] for k in pairs[i]], "gap_m": float(gaps[i]),
                                "spheres": balls[pairs[i]].tolist()} for i in np.flatnonzero(gaps < 0)])
        positive_world.append([{"link": labels[int(i)], "cost": float(world[index, 0, i]),
                                "sphere": balls[i].tolist()} for i in np.flatnonzero(world[index, 0] > 0)])
    report = {"exact_replay_of_reproduction_requests": True, "production_original": False, "physical_motion": False, "live_service_access": False,
        "adapter_ik": result, "public_success": raw.success[:n].detach().cpu().tolist(),
        "position_error_m": raw.position_error[:n].detach().cpu().tolist(),
        "rotation_error_rad": raw.rotation_error[:n].detach().cpu().tolist(),
        "world_max": world.max(axis=-1).tolist(), "self_cost": self_cost.tolist(),
        "seed0_world_positive": positive_world, "seed0_negative_self_pairs": negative_pairs}
    args.output.write_text(json.dumps(report, indent=2))
    successes = sum(bool(item["success"]) for item in result["items"])
    print(json.dumps({"candidate_count": n, "adapter_success_count": successes,
                      "public_success_count": int(raw.success[:n].any(dim=1).sum().item()),
                      "seed0_negative_pair_count": sum(map(len, negative_pairs)),
                      "output": str(args.output)}), flush=True)
    if successes < args.minimum_success:
        raise SystemExit(1)

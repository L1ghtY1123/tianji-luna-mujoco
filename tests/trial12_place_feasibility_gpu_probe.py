"""Bounded actual-configuration and collision-checked pose diagnostics."""

import json
from copy import deepcopy

import numpy as np
import torch
import yaml
from tianji_curobo.curobo_service import materialize_urdf

from jev_mujoco.carrying import CarryingBinding
from jev_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation
from jev_mujoco.placement import release_goal


def main():
    folder = ROOT / "data/visible-cloud-work/trial12-place-diagnostic"
    stage = json.loads((folder / "stage-1.json").read_text(encoding="utf-8"))
    mapping = json.loads((folder / "gpu-requests/017-perception-build.json").read_text(encoding="utf-8"))
    payload = json.loads((folder / "gpu-requests/018-carrying-prepare.json").read_text(encoding="utf-8"))["body"]
    holding, observation = stage["holding"], stage["holding"]["observation"]
    holding["points_tool_m"] = payload["carrying_binding"]["points_tool_m"]
    source = next(obj for obj in observation["objects"] if obj["id"] == holding["instance_id"])
    destination = next(obj for obj in observation["objects"] if obj["kind"] == "bin")
    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    scene = run_operation(runtime, mapping["path"], mapping["body"])
    run_operation(runtime, "/carrying/prepare", payload)
    binding = CarryingBinding.from_mapping(payload["carrying_binding"], scene=scene, arm="A")
    goal = np.asarray(release_goal(source, destination, holding, release_height_m=.04)["T_world_tcp"])
    goals = [np.asarray(observation["arms"]["A"]["tcp"]), goal, goal.copy(), goal.copy()]
    goals[2][2, 3] += .1
    goals[3][2, 3] += .05
    report = {"physical_motion": False, "oracle_used": False, "pose_labels": ["actual_held", "release40mm", "pre_place100mm", "pre_place50mm"],
              "goals": [matrix.tolist() for matrix in goals], "identity_proof": stage["identity_assumptions"]}
    with runtime.carrying_scope(binding, payload["gripper_joint_positions_rad"]):
        backend = next(iter(runtime._compound_backends.values()))
        planner, checker = backend.motion_planner, backend._motion_clearance_checker
        positions = torch.as_tensor(np.deg2rad(payload["start_joints_deg"])[None, None], **planner.device_cfg.as_torch_dict())
        state = checker.get_kinematics(positions)
        spheres = state.robot_spheres.detach().cpu().numpy()[0, 0]
        world = checker.get_collision_distance(state).detach().cpu().numpy()
        self_cost = checker.get_self_collision_distance(state.robot_spheres).detach().cpu().numpy()
        report["actual_held_validity"] = {"robot_and_attachment_sphere_count": len(spheres),
            "world_positive_count": int(np.sum(world > 0)), "world_max_cost": float(world.max()),
            "self_positive_count": int(np.sum(self_cost > 0)), "self_max_cost": float(self_cost.max())}
        print(json.dumps({"actual_held_validity": report["actual_held_validity"]}), flush=True)
        config = checker.kinematics.kinematics_config
        attached = config.get_sphere_index_from_link_name("jev_carried_object_A").detach().cpu().numpy().reshape(-1)
        overlaps = {}
        configured = yaml.safe_load(runtime._robot_config_path.read_bytes())
        for link in configured["robot_cfg"]["kinematics"]["collision_link_names"]:
            if link == "jev_carried_object_A":
                continue
            try:
                indices = config.get_sphere_index_from_link_name(link).detach().cpu().numpy().reshape(-1)
            except KeyError:
                continue  # Peer links may instead be represented in world collision.
            if not len(indices):
                continue
            distance = np.linalg.norm(spheres[attached, None, :3]-spheres[None, indices, :3], axis=-1)
            gap = distance-spheres[attached, None, 3]-spheres[None, indices, 3]
            if np.any(gap < 0):
                overlaps[link] = {"overlap_pair_count": int(np.sum(gap < 0)), "minimum_sphere_gap_m": float(gap.min())}
        report["geometric_attachment_overlap_links"] = overlaps
        print(json.dumps({"actual_held_validity": report["actual_held_validity"], "overlap_links": overlaps}), flush=True)
        ik_payload = {key: payload[key] for key in ("arm", "start_joints_deg", "other_joints_deg", "scene_token", "planning_settings")}
        ik_payload["goal_matrices_world"] = [matrix.tolist() for matrix in goals]
        report["collision_checked_ik"] = run_operation(runtime, "/tools/solve_ik_batch", ik_payload)
    print("IK " + json.dumps(report["collision_checked_ik"]), flush=True)
    if (report["collision_checked_ik"]["items"][3]["success"]
            and not report["collision_checked_ik"]["items"][2]["success"]):
        report["shorter_approach_authorized_condition_met"] = True
        # Run the one explicitly authorized50mm approach with all other settings unchanged.
        from scipy.spatial.transform import Rotation
        from tianji_curobo.contracts import CuroboToolError

        settings = deepcopy(payload["planning_settings"])
        settings["place_release_height_m"], settings["place_approach_offset_m"] = .04, .05
        quaternion = Rotation.from_matrix(goal[:3, :3]).as_quat()
        request = deepcopy(payload) | {"planning_settings": settings, "goal_position_m": goal[:3, 3].tolist(),
            "goal_quaternion_wxyz": [float(quaternion[3]), *quaternion[:3].tolist()],
            "approach_axis": "z", "approach_offset_m": .05, "approach_in_tool_frame": False,
            "lift_axis": settings["place_retreat_axis"], "lift_offset_m": settings["place_retreat_offset_m"],
            "lift_in_tool_frame": settings["place_retreat_in_tool_frame"]}
        try:
            result = run_operation(runtime, "/tools/plan_place", request)
            report["explicit50mm_native"] = {"success": True, "phase_counts": {name: len(phase["positions_rad"]) for name, phase in result["phases"].items()}}
        except CuroboToolError as error:
            report["explicit50mm_native"] = {"success": False, "code": error.code, "details": error.details}
    runtime.clear_carrying()
    print("FINAL " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

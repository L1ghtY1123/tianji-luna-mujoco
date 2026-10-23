"""Isolated public IK diagnosis for the six recorded 2026-10-16 release poses.

The released pose, joint feedback and fitted table are exact recorded values.
Only the first three of sixteen attachment spheres survived the public log;
the partial-attachment mode must not be described as an exact scene replay.
No service calls or simulation object poses are used.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main(folder, output, partial_attachment):
    import torch
    import yaml
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import GoalToolPose, Pose
    from scipy.spatial.transform import Rotation
    from tianji_curobo.curobo_robot_config import create_frozen_peer_arm_robot_config
    from tianji_curobo.curobo_service import materialize_urdf

    from luna_mujoco.backend_builder import create_solver_configs
    from luna_mujoco.carrying import configure_gripper_feedback
    from luna_mujoco.geometry_service import ROOT, sdk_identity
    from luna_mujoco.planning_settings import PlanningSettings

    recorded = json.loads((folder / "pre-change-state.json").read_text())
    cpu = json.loads((folder / "cpu-kinematic-probe.json").read_text())
    settings = PlanningSettings.model_validate(recorded["planning_settings"]["settings"])
    # Robot joint/gripper feedback is legitimate proprioception. Other fields
    # under simulation (object poses/geometry) are deliberately never read.
    arms = recorded["simulation"]["arms"]
    feedback = {arm: value["gripper_closed_fraction"] * .8 for arm, value in arms.items()}
    feedback["A"] = recorded["run"]["holding"]["verification"]["position_rad"]
    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    combined = yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_bytes())
    robot = create_frozen_peer_arm_robot_config(combined, "A", urdf_path=urdf, asset_root_path=urdf.parent)
    robot = configure_gripper_feedback(robot, feedback, urdf)
    kin = robot["robot_cfg"]["kinematics"]
    kin["lock_joints"].update({f"Joint{index + 1}_R": angle for index, angle in enumerate(arms["B"]["joints_rad"])})
    if partial_attachment:
        known = recorded["run"]["holding"]["attachment_evidence"]["attachment_spheres_tool_m"]["first"]
        link = "jev_carried_object_A"
        kin.setdefault("extra_links", {})[link] = {
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_name": "jev_carry_joint_A",
            "joint_type": "FIXED", "link_name": link, "parent_link_name": "TCP_Link_L"}
        kin["collision_link_names"] = [*kin["collision_link_names"], link]
        kin["collision_spheres"][link] = [{"center": row[:3], "radius": row[3]} for row in known]
        kin.setdefault("self_collision_buffer", {})[link] = 0.
        kin.setdefault("self_collision_ignore", {})[link] = ["gripL_Link3_L", "gripL_Link3_R"]
        for contact in kin["self_collision_ignore"][link]:
            kin["self_collision_ignore"].setdefault(contact, []).append(link)
        kin.setdefault("collision_sphere_buffer", {})[link] = 0.
    # Table values come from the RANSAC replacement cuboid in the run log.
    scene = {"cuboid": {"measured_table": {
        "pose": [.551113060216109, .000062578490802212, .452099766816746, 1, 0, 0, 0],
        "dims": [.7975285389026007, 1.0959697595664433, .3]}}}
    grasp_rotation = np.array([[.7827407931525575, -.5738393678944513, .24088426804049407],
        [-.619543156261017, -.6817728771499817, .3890398713663886],
        [-.059018033388492055, -.4537555771967279, -.8891696957824317]])
    matrices = []
    for candidate in cpu:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("z", candidate["yaw"], degrees=True).as_matrix() @ grasp_rotation
        matrix[:3, 3] = [*candidate["xy"], .8338956085]
        matrices.append(matrix)
    matrices.extend([matrices[-1]] * (100 - len(matrices)))
    matrices = np.asarray(matrices)
    started = time.perf_counter()
    config, _ = create_solver_configs(settings, robot=robot, scene_model=scene,
        collision_cache={"obb": 20, "mesh": 0}, batch_size=100, ik_config_type=InverseKinematicsCfg)
    ik = InverseKinematics(config)
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot, scene_model=scene, device_cfg=ik.device_cfg,
        collision_activation_distance=settings.trajectory_collision_clearance_m))
    setup_s = time.perf_counter() - started
    target = Pose(position=torch.as_tensor(matrices[:, :3, 3], **ik.device_cfg.as_torch_dict()),
        quaternion=torch.as_tensor(Rotation.from_matrix(matrices[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]],
            **ik.device_cfg.as_torch_dict()))
    goals = GoalToolPose.from_poses({"TCP_Link_L": target}, ordered_tool_frames=["TCP_Link_L"], num_goalset=1)
    started = time.perf_counter()
    raw = ik.solve_pose(goals, return_seeds=ik.config.num_seeds)
    solve_s = time.perf_counter() - started
    solutions = raw.solution[:6].reshape(-1, 1, 7)
    world, self_cost = checker.get_scene_self_collision_distance_from_joints(solutions)
    world = world.reshape(6, 16, -1).detach().cpu().numpy()
    self_cost = self_cost.reshape(6, 16, -1).detach().cpu().numpy()
    cpu_q = torch.as_tensor(np.asarray([row["q"] for row in cpu])[:, None], **ik.device_cfg.as_torch_dict())
    cpu_world, cpu_self = checker.get_scene_self_collision_distance_from_joints(cpu_q)
    report = {"sdk": sdk_identity()[0], "physical_execution": False,
        "fixture": "exact_six_release_poses_and_robot_feedback_with_measured_table",
        "missing_production_inputs": ["complete_instance_ESDF", "13_of_16_attachment_spheres"],
        "partial_attachment": partial_attachment, "setup_s": setup_s, "solve_s": solve_s,
        "success": raw.success[:6].detach().cpu().tolist(),
        "position_error_m": raw.position_error[:6].detach().cpu().tolist(),
        "rotation_error_rad": raw.rotation_error[:6].detach().cpu().tolist(),
        "solutions": raw.solution[:6].detach().cpu().tolist(),
        "raw_solution_world_max": world.max(axis=-1).tolist(),
        "raw_solution_self_max": self_cost.max(axis=-1).tolist(),
        "cpu_solution_world_max": cpu_world.reshape(6, -1).max(dim=1).values.detach().cpu().tolist(),
        "cpu_solution_self_max": cpu_self.reshape(6, -1).max(dim=1).values.detach().cpu().tolist()}
    # Map positive sphere costs to authored link names using the public model.
    model = checker.kinematics
    link_indices = {name: model.kinematics_config.get_sphere_index_from_link_name(name).detach().cpu().tolist()
                    for name in kin["collision_link_names"]}
    report["link_sphere_indices"] = link_indices
    report["cpu_spheres_world"] = checker.get_kinematics(cpu_q).robot_spheres.detach().cpu().tolist()
    report["cpu_world_costs"] = cpu_world.detach().cpu().tolist()
    report["cpu_self_costs"] = cpu_self.detach().cpu().tolist()
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key in
        ("sdk", "partial_attachment", "setup_s", "solve_s", "success", "raw_solution_world_max",
         "raw_solution_self_max", "cpu_solution_world_max", "cpu_solution_self_max")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--partial-attachment", action="store_true")
    args = parser.parse_args()
    main(args.folder, args.output, args.partial_attachment)

"""Scene-independent cuRobo position IK at the calibrated gripper reference."""

import time

import numpy as np

from .geometry import TOOL_PROFILE


def validate_query(payload):
    arm = payload.get('arm')
    points = np.asarray(payload.get('positions_world_m'), dtype=float)
    if (arm not in ('A', 'B') or points.ndim != 2 or points.shape[1] != 3
            or not 1 <= len(points) <= 2 or not np.isfinite(points).all()):
        raise ValueError('reachability_requires_arm_and_one_or_two_finite_world_positions')
    return arm, points


class PositionReachability:
    """One fixed two-target GPU solver per arm; no map, attachment or motion."""

    def __init__(self, robot_config_path, urdf_path, asset_root_path, arm):
        import torch
        import yaml
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import GoalToolPose, Pose, ToolPoseCriteria
        from tianji_curobo.curobo_robot_config import create_frozen_peer_arm_robot_config

        self.torch, self.Pose, self.GoalToolPose = torch, Pose, GoalToolPose
        robot = create_frozen_peer_arm_robot_config(
            yaml.safe_load(robot_config_path.read_text(encoding='utf8')), arm,
            urdf_path=urdf_path, asset_root_path=asset_root_path)
        kin = robot['robot_cfg']['kinematics']
        self.parent_frame = kin['tool_frames'][0]
        self.tool_frame = 'jev_reachability_center_' + arm
        self.offset_m = [0., 0., TOOL_PROFILE[arm]['tcp_ee_z_m']]
        # TCP_Link is at the wrist. The existing grasp calibration defines the
        # gripper EE reference 185 mm along its local Z. Query that point, not
        # a wrist origin incorrectly forced to touch the table/object center.
        kin.setdefault('extra_links', {})[self.tool_frame] = {
            'fixed_transform': self.offset_m + [1., 0., 0., 0.],
            'joint_name': self.tool_frame + '_joint', 'joint_type': 'FIXED',
            'link_name': self.tool_frame, 'parent_link_name': self.parent_frame}
        kin['tool_frames'] = [self.tool_frame]
        config = InverseKinematicsCfg.create(robot=robot, num_seeds=32, seed_solver_num_seeds=64,
            max_batch_size=2, self_collision_check=False, scene_model=None,
            position_tolerance=.005, orientation_tolerance=3.15, random_seed=123)
        self.solver = InverseKinematics(config)
        self.solver.update_tool_pose_criteria({self.tool_frame: ToolPoseCriteria.track_position()})

    def solve(self, points):
        torch = self.torch
        positions = np.asarray(points, dtype=float)
        if len(positions) == 1:
            positions = np.repeat(positions, 2, axis=0)
        tensor = torch.as_tensor(positions, device=self.solver.device_cfg.device,
                                 dtype=self.solver.device_cfg.dtype)
        pose = self.Pose(position=tensor,
            quaternion=torch.tensor([[1., 0., 0., 0.]] * 2, device=tensor.device, dtype=tensor.dtype))
        goal = self.GoalToolPose.from_poses({self.tool_frame: pose}, num_goalset=1)
        started = time.monotonic()
        result = self.solver.solve_pose(goal)
        active = self.solver.get_active_js(result.js_solution)
        actual = self.solver.compute_kinematics(active).tool_poses[self.tool_frame].position.reshape(2, 3)
        errors = torch.linalg.vector_norm(actual - tensor, dim=-1).detach().cpu().tolist()
        success = result.success.flatten().detach().cpu().tolist()
        joints = active.position.reshape(2, -1).detach().cpu().tolist()
        duration = time.monotonic() - started
        return [{'reachable': bool(success[index] and errors[index] <= .005),
            'position_error_m': errors[index], 'position_world_m': list(point), 'joints_rad': joints[index],
            'collision_checked': False, 'orientation_constrained': False,
            'reference_frame': self.tool_frame, 'reference_parent_frame': self.parent_frame,
            'reference_offset_m': list(self.offset_m), 'solver': 'curobo_batched_position_ik',
            'seed_count': 32, 'seed_solver_num_seeds': 64, 'batch_solve_s': duration}
            for index, point in enumerate(points)]

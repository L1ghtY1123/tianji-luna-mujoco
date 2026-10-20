"""Final trajectory proof for measured two-gripper geometry through public cuRobo."""

import hashlib
import json
from dataclasses import asdict

import numpy as np

from .contracts import CuroboToolError, TrajectoryValidationResult
from .curobo_robot_config import ARM_CONFIG_SPECS, create_dual_arm_robot_config
from .grasp_lift import collision_samples, gripper_robot_config
from .grasp_lift_compute import collision_verdict
from .gripper_collision_envelope import create_gripper_closing_envelope


def trajectory_binding(request):
    return hashlib.sha256(
        json.dumps(asdict(request), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require_measured_result(result, request):
    if (
        result.gripper_joint_positions_rad != request.gripper_joint_positions_rad
        or result.gripper_configuration_source != "measured_feedback"
        or result.binding_sha256 != trajectory_binding(request)
    ):
        raise ValueError(
            "final trajectory proof does not bind the measured two-gripper configuration"
        )
    return result


def trajectory_robot_configs(original, request, urdf_path, asset_root_path):
    robot = create_dual_arm_robot_config(
        original, urdf_path=urdf_path, asset_root_path=asset_root_path
    )
    exact = gripper_robot_config(robot, request.gripper_joint_positions_rad, urdf_path)
    # The selected gripper may close at grasp. Its time-union envelope is queried
    # against the world only, never used as cross-time self-collision geometry.
    envelope = None
    if request.grasp_phase is not None:
        envelope = create_gripper_closing_envelope(exact, request.arm, urdf_path)
    return exact, envelope


def validate_measured_trajectory_unlocked(runtime, request, planner, scene_token):
    """Caller owns the runtime scene/GPU lock through the final scene-token check."""
    try:
        import torch
        import yaml
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

        if not torch.cuda.is_available():
            raise ValueError("CUDA is required for measured trajectory validation")
        original = yaml.safe_load(runtime._robot_config_path.read_bytes())
        exact, envelope = trajectory_robot_configs(
            original, request, runtime._urdf_path, runtime._asset_root_path
        )

        def build(config):
            return RobotCollisionChecker(
                RobotCollisionCheckerCfg.load_from_config(
                    robot_config=config,
                    device_cfg=planner.device_cfg,
                    scene_collision_checker=planner.scene_collision_checker,
                    collision_activation_distance=request.trajectory_collision_clearance_m,
                )
            )

        checker = build(exact)
        world_envelope = None if envelope is None else build(envelope)
        if world_envelope is not None and world_envelope.collision_cost is None:
            raise ValueError("required closing world collision term is missing")
        if checker.collision_cost is None or checker.self_collision_cost is None:
            raise ValueError("required exact world/self collision terms are missing")
        samples, source_indices = collision_samples(np.deg2rad(request.points_deg))
        peer = np.repeat([np.deg2rad(request.other_joints_deg)], len(samples), axis=0)
        both = np.concatenate((samples, peer) if request.arm == "A" else (peer, samples), axis=1)
        names = ARM_CONFIG_SPECS["A"]["joint_names"] + ARM_CONFIG_SPECS["B"]["joint_names"]
        if set(checker.kinematics.joint_names) != set(names):
            raise ValueError("measured checker must retain both seven-joint arm chains")
        ordered = both[:, [names.index(name) for name in checker.kinematics.joint_names]]
        valid = []
        with torch.no_grad():
            for offset in range(0, len(ordered), 128):
                chunk = ordered[offset : offset + 128]
                q = torch.as_tensor(chunk[None], **planner.device_cfg.as_torch_dict()).contiguous()
                state = checker.get_kinematics(q)
                world = checker.get_collision_distance(state).detach().cpu().numpy().copy()
                self_cost = (
                    checker.get_self_collision_distance(state.robot_spheres)
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
                bounds = checker.get_bound(q).detach().cpu().numpy().copy()
                free = np.asarray(collision_verdict(world, self_cost, bounds, len(chunk)))
                if world_envelope is not None:
                    if world_envelope.kinematics.joint_names != checker.kinematics.joint_names:
                        raise ValueError("closing world checker joint order differs")
                    envelope_cost = (
                        world_envelope.get_collision_distance(world_envelope.get_kinematics(q))
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )
                    zeros = np.zeros((1, len(chunk), 1))
                    free &= collision_verdict(envelope_cost, zeros, zeros, len(chunk))
                valid.extend(free.tolist())
        invalid = next((index for index, free in enumerate(valid) if not free), None)
        return TrajectoryValidationResult.create(
            collision_free=invalid is None,
            checked_point_count=len(samples),
            first_invalid_index=None if invalid is None else source_indices[invalid],
            scene_token=scene_token,
            gripper_joint_positions_rad=dict(request.gripper_joint_positions_rad),
            gripper_configuration_source="measured_feedback",
            binding_sha256=trajectory_binding(request),
        )
    except CuroboToolError:
        raise
    except Exception as error:
        raise CuroboToolError(
            "measured trajectory collision computation failed",
            stage="sdk",
            code="CUROBO_TRAJECTORY_CHECK_FAILED",
            status=500,
            details={"reason": str(error)},
        ) from error

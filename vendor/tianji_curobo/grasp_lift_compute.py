"""GPU carried-object checks through public cuRobo configuration/checker APIs.

The Tianji runtime owns the lock and the published world. Its SDK and original
robot YAML stay unchanged; a dedicated checker owns the extra fixed object link.
"""

import hashlib

import numpy as np

from .contracts import CuroboToolError
from .curobo_robot_config import ARM_CONFIG_SPECS, create_dual_arm_robot_config
from .grasp_lift import (
    CARRIED_LINK,
    MAX_JOINT_STEP_RAD,
    attached_robot_config,
    binding_digest,
    collision_samples,
    gripper_robot_config,
    require_scene_binding,
)
from .support_contact import SUPPORT_NAME, SupportDeparture


def collision_verdict(world_cost, self_cost, bound_cost, count):
    valid = np.ones(count, dtype=bool)
    for values in (world_cost, self_cost, bound_cost):
        values = np.asarray(values)
        if (
            values.ndim < 2
            or values.shape[:2] != (1, count)
            or values.size == 0
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("collision costs must be finite nonnegative [1, samples, ...] values")
        valid &= np.all(values.reshape(count, -1) == 0, axis=1)
    return valid.tolist()


def spheres_inside_volume(spheres, bounds):
    centers = np.asarray(bounds.get("center_m"), dtype=np.float64)
    extent = np.asarray(bounds.get("extent_m"), dtype=np.float64)
    spheres = np.asarray(spheres)
    if (
        centers.shape != (3,)
        or extent.shape != (3,)
        or not np.isfinite(centers).all()
        or not np.isfinite(extent).all()
        or np.any(extent <= 0)
        or spheres.ndim != 4
        or spheres.shape[0] != 1
        or spheres.shape[-1] != 4
        or not np.isfinite(spheres).all()
        or np.any(spheres[..., 3] <= 0)
    ):
        raise ValueError("published ESDF bounds and carried spheres must be finite")
    distance = np.abs(spheres[0, ..., :3] - centers) + spheres[0, ..., 3, None]
    return np.all(distance <= extent / 2, axis=(1, 2)).tolist()


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def world_collision_valid(
    checker, scene_checker, spheres, carried_indices, exemptions, *, to_numpy=_numpy
):
    """Query disjoint geometries, never subtract support from an aggregate cost."""
    count = spheres.shape[1]
    carried_set = {int(index) for index in carried_indices}
    robot_indices = [index for index in range(spheres.shape[2]) if index not in carried_set]
    zeros = np.zeros((1, count, 1))
    # Full world remains enabled for the original robot in every sample.
    robot_cost = to_numpy(checker.get_collision_distance(spheres[:, :, robot_indices, :])).copy()
    valid = np.asarray(collision_verdict(robot_cost, zeros, zeros, count))
    for exempt in (False, True):
        indices = [index for index, value in enumerate(exemptions) if value == exempt]
        if not indices:
            continue
        try:
            if exempt:
                scene_checker.enable_obstacle(SUPPORT_NAME, enable=False)
            cost = to_numpy(
                checker.get_collision_distance(
                    spheres[:, indices, :, :][:, :, sorted(carried_set), :]
                )
            ).copy()
            empty = np.zeros((1, len(indices), 1))
            valid[indices] &= collision_verdict(cost, empty, empty, len(indices))
        finally:
            if exempt:
                scene_checker.enable_obstacle(SUPPORT_NAME, enable=True)
    return valid.tolist()


def validate_grasp_lift(runtime, payload, *, lock_timeout_s):
    """Fail closed if scene binding, actual CUDA computation or public APIs fail."""
    if not runtime.gpu_lock.acquire(timeout=lock_timeout_s):
        raise CuroboToolError(
            "cuRobo is busy", stage="runtime", code="CUROBO_BUSY", status=409, retryable=True
        )
    try:
        request = payload["request"]
        store = runtime._fixed_esdf_store
        if store is None or not store.published_status().ready:
            raise CuroboToolError(
                "published ESDF is unavailable",
                stage="scene",
                code="CUROBO_ESDF_NOT_READY",
                status=409,
            )
        # This is project-owned state, not a private SDK API. Hold the same lock
        # used by scene publication and selection until every query completes.
        metadata = store._published_metadata or {}
        actual_token = store.published_status().scene_token
        require_scene_binding(payload, actual_token, metadata)
        snapshot = runtime._prepare_scene_cache_unlocked()
        backend = runtime._backend_unlocked(request["arm"])
        token = runtime._refresh_scene_unlocked(
            snapshot=snapshot,
            arm=request["arm"],
            backend=backend,
        )
        require_scene_binding(payload, token, metadata)
        try:
            import curobo
            import torch
            import yaml
            from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        except ImportError as error:
            raise CuroboToolError(
                "public cuRobo collision checker is unavailable",
                stage="sdk",
                code="CUROBO_ATTACHMENT_UNAVAILABLE",
                status=503,
            ) from error
        if not torch.cuda.is_available():
            raise CuroboToolError(
                "CUDA carried-object validation is unavailable",
                stage="sdk",
                code="CUROBO_ATTACHMENT_UNAVAILABLE",
                status=503,
            )
        original_bytes = runtime._robot_config_path.read_bytes()
        original = yaml.safe_load(original_bytes)
        robot = create_dual_arm_robot_config(
            original,
            urdf_path=runtime._urdf_path,
            asset_root_path=runtime._asset_root_path,
        )
        robot = gripper_robot_config(
            robot, request["gripper_joint_positions_rad"], runtime._urdf_path
        )
        robot, geometry = attached_robot_config(robot, payload["attachment"], request["arm"])
        planner = backend.motion_planner
        scene_checker = planner.scene_collision_checker
        if scene_checker is None:
            raise ValueError("cuRobo has no published scene collision checker")
        config = RobotCollisionCheckerCfg.load_from_config(
            robot_config=robot,
            device_cfg=planner.device_cfg,
            scene_collision_checker=scene_checker,
            collision_activation_distance=0.0,
        )
        checker = RobotCollisionChecker(config)
        if (
            checker.collision_cost is None
            or checker.self_collision_cost is None
            or checker.collision_constraint is None
        ):
            raise ValueError("cuRobo checker is missing required collision terms")
        samples, source_indices = collision_samples(request["positions_rad"])
        peer = np.repeat([request["other_joints_rad"]], len(samples), axis=0)
        both = np.concatenate((samples, peer) if request["arm"] == "A" else (peer, samples), axis=1)
        joint_names = ARM_CONFIG_SPECS["A"]["joint_names"] + ARM_CONFIG_SPECS["B"]["joint_names"]
        if set(checker.kinematics.joint_names) != set(joint_names):
            raise ValueError("cuRobo checker must expose exactly both seven-joint arm chains")
        ordered = both[:, [joint_names.index(name) for name in checker.kinematics.joint_names]]
        kinematics_config = checker.kinematics.config.kinematics_config
        carried_indices = kinematics_config.get_sphere_index_from_link_name(CARRIED_LINK)
        if carried_indices.numel() != geometry["attachment_sphere_count"]:
            raise ValueError("cuRobo did not load every carried-object collision sphere")
        support = None
        if request.get("support_contact_policy") is not None:
            plane = metadata.get("tabletop", {}).get("support_plane")
            if not plane:
                raise CuroboToolError(
                    "observed support plane is unavailable",
                    stage="scene",
                    code="CUROBO_SUPPORT_UNVERIFIED",
                    status=409,
                    details={"missing_inputs": ["tabletop.support_plane"]},
                )
            if SUPPORT_NAME not in scene_checker.get_obstacle_names():
                raise ValueError("the bound support obstacle is absent from the active scene")
            if (store.published_scene_model or {}).get("cuboid", {}).get(SUPPORT_NAME) != plane[
                "replacement_cuboid"
            ]:
                raise ValueError("support plane does not match the published replacement cuboid")
            support = SupportDeparture(plane, geometry["observed_bounds_ee_m"])
        valid = []
        # Bound GPU memory while retaining every supplied and intermediate sample.
        with torch.no_grad():
            for offset in range(0, len(samples), 128):
                chunk = ordered[offset : offset + 128]
                q = torch.as_tensor(chunk[None], **planner.device_cfg.as_torch_dict()).contiguous()
                state = checker.get_kinematics(q)
                carried = _numpy(state.robot_spheres[:, :, carried_indices, :])
                support_valid = [True] * len(chunk)
                if support is None:
                    world = _numpy(checker.get_collision_distance(state)).copy()
                    world_valid = collision_verdict(
                        world,
                        np.zeros((1, len(chunk), 1)),
                        np.zeros((1, len(chunk), 1)),
                        len(chunk),
                    )
                else:
                    poses = _numpy(state.tool_poses.get_link_pose(CARRIED_LINK).clone().get_matrix())
                    exemptions, support_valid = support.evaluate(poses, carried[0])
                    world_valid = world_collision_valid(
                        checker,
                        scene_checker,
                        state.robot_spheres,
                        _numpy(carried_indices).tolist(),
                        exemptions,
                    )
                self_collision = checker.get_self_collision_distance(state.robot_spheres)
                joint_bounds = checker.get_bound(q)
                batch_valid = collision_verdict(
                    np.zeros((1, len(chunk), 1)),
                    _numpy(self_collision),
                    _numpy(joint_bounds),
                    len(chunk),
                )
                contained = spheres_inside_volume(
                    carried,
                    metadata.get("bounds", {}),
                )
                valid.extend(
                    all(parts)
                    for parts in zip(
                        batch_valid, contained, world_valid, support_valid, strict=True
                    )
                )
        require_scene_binding(
            payload, store.published_status().scene_token, store._published_metadata or {}
        )
        if support is not None and not support.departed:
            valid[-1] = False
            support.reason = support.reason or "support_departure_incomplete"
        invalid = next((i for i, free in enumerate(valid) if not free), None)
        return {
            "ok": True,
            "collision_free": invalid is None,
            "first_invalid_index": None if invalid is None else source_indices[invalid],
            "checked_point_count": len(samples),
            "command_point_count": len(request["positions_rad"]),
            "plan_id": request["plan_id"],
            "target_id": request["target_id"],
            "profile_revision": request["profile_revision"],
            "gripper_joint_positions_rad": request["gripper_joint_positions_rad"],
            "gripper_configuration_source": request["gripper_configuration_source"],
            "scene_token": token,
            "binding_sha256": binding_digest(payload),
            "robot_config_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "curobo_version": str(curobo.__version__),
            "validation_method": "public_RobotCollisionChecker_with_fixed_carried_link",
            "max_joint_check_step_rad": MAX_JOINT_STEP_RAD,
            "path_coverage": "sampled_linear_joint_segments",
            "source_observation_id": payload["attachment"]["source_observation_id"],
            "support_contact": None if support is None else support.proof(),
            **geometry,
        }
    except CuroboToolError:
        raise
    except Exception as error:
        raise CuroboToolError(
            "carried-object collision computation failed",
            stage="sdk",
            code="CUROBO_ATTACHMENT_CHECK_FAILED",
            status=500,
            details={"reason": str(error)},
        ) from error
    finally:
        runtime.gpu_lock.release()

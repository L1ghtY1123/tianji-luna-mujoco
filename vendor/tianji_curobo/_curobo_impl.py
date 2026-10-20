# ========================================
# MODIFICATION START: implement the private cuRobo runtime
# ========================================
"""Private native cuRobo runtime with process-wide CUDA serialization."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import replace
import gc
import logging
from inspect import signature
from math import isfinite
from numbers import Real
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

# ========================================
# MODIFICATION START: consume structured strict-line validation failures
# ========================================
from .axis_linear import AxisLinearValidationError
# ========================================
# MODIFICATION END: consume structured strict-line validation failures
# ========================================
from .axis_linear import (
    quaternion_wxyz_to_matrix,
    validate_axis_linear_poses,
)
# ========================================
# MODIFICATION START: import final trajectory validation contracts
# ========================================
from .contracts import (
    SolveIkBatchRequest,
    NativeIkBatchResult,
    _collision_clearance_profile,
    CuroboToolError,
    FixedEsdfCandidateValidationResult,
    HealthResult,
    NativeGraspBatchItem,
    NativeGraspBatchPlan,
    NativeGraspPlan,
    NativePlacePlan,
    NativeIkResult,
    NativeTrajectory,
    PlanGraspRequest,
    PlanGraspBatchRequest,
    PlanPlaceRequest,
    PlanAxisLinearRequest,
    PlanToJointRequest,
    PlanToPoseRequest,
    SolveIkRequest,
    TrajectoryValidationResult,
    ValidateTrajectoryRequest,
)
# ========================================
# MODIFICATION END: import final trajectory validation contracts
# ========================================
from .curobo_robot_config import (
    _exclude_collision_links,
    collision_links_for_arm,
    create_frozen_peer_arm_robot_config,
    create_dual_arm_robot_config,
)
from .gripper_collision_envelope import create_gripper_closing_envelope
from .curobo_scene import (
    build_curobo_scene_model,
    collision_cache_for_scene,
)
POINT_DIMENSION = 7
LOGGER = logging.getLogger(__name__)
# ========================================
# MODIFICATION START: bound continuous-path collision sampling
# ========================================
COLLISION_CHECK_MAX_JOINT_STEP_DEG = 0.25
MAX_TRAJECTORY_VALIDATION_SAMPLES = 10_000
_COLLINEAR_PATH_TOLERANCE_DEG = 1e-9
# ========================================
# MODIFICATION END: bound continuous-path collision sampling
# ========================================
# ========================================
# MODIFICATION START: separate trajectory and optimizer clearances
# ========================================
TRAJECTORY_COLLISION_CLEARANCE_M = 0.01
OPTIMIZER_COLLISION_CLEARANCE_M = 0.03
CLEARANCE_COST_EPSILON = 1.0e-9
# ========================================
# MODIFICATION END: separate trajectory and optimizer clearances
# ========================================


def _call_profile_factory(factory, legacy_args, profile_args):
    """Support legacy injected factories alongside profile-aware factories."""

    try:
        signature(factory).bind(*profile_args)
    except TypeError:
        return factory(*legacy_args)
    return factory(*profile_args)
# ========================================
# MODIFICATION START: bound strict-line candidate weights
# ========================================
AXIS_LINEAR_NON_TERMINAL_SCALES = (2.0, 5.0, 10.0)
_AXIS_LINEAR_RETRYABLE_CODES = frozenset(
    {
        "CUROBO_PLANNING_FAILED",
        "CUROBO_TRAJOPT_FAILED",
    }
)
# ========================================
# MODIFICATION END: bound strict-line candidate weights
# ========================================
DEFAULT_ROBOT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "robot_config"
    / "marvin_gui_dual.yml"
)
_CUDA_LOCK = threading.Lock()

def _rotation_matrix_to_quaternion_wxyz(rotation):
    """Convert one validated rotation matrix without changing candidate order."""

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        if diagonal == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif diagonal == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


# ========================================
# MODIFICATION START: support cuRobo v0.8.0 ToolPoseCriteria location
# ========================================
def _load_tool_pose_criteria_type():
    """Load ToolPoseCriteria across the cuRoboV2 v0.8 API boundary."""

    import curobo.types as curobo_types

    public_type = getattr(curobo_types, "ToolPoseCriteria", None)
    if public_type is not None:
        return public_type

    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    return ToolPoseCriteria
# ========================================
# MODIFICATION END: support cuRobo v0.8.0 ToolPoseCriteria location
# ========================================


# ========================================
# MODIFICATION START: enforce real-world scene clearance in graph validation
# ========================================
def _graph_planner_rollout_config():
    """Load graph metrics for physical collision truth."""

    from curobo.config_io import join_path, load_yaml
    from curobo.content import get_task_configs_path

    rollout = load_yaml(
        join_path(get_task_configs_path(), "metrics_base.yml")
    )
    scene_constraint = rollout["rollout"]["constraint_cfg"][
        "scene_collision_cfg"
    ]
    scene_constraint["activation_distance"] = 0.0
    return rollout


def _grasp_optimizer_config():
    """Use the installed official task with stronger tool tracking for grasp phases."""
    from curobo.config_io import join_path, load_yaml
    from curobo.content import get_task_configs_path

    task = load_yaml(join_path(
        get_task_configs_path(), "trajopt/lbfgs_bspline_trajopt.yml",
    ))
    pose = task["rollout"]["cost_cfg"]["tool_pose_cfg"]
    pose["weight"] = [float(value) * 50.0 for value in pose["weight"]]
    return task


# ========================================
# MODIFICATION END: enforce real-world scene clearance in graph validation
# ========================================


# ========================================
# MODIFICATION START: preserve official CUDA graph acceleration for live worlds
# ========================================
def _create_live_world_planner_config(
    planner_config_type,
    *,
    ik_config_type=None,
    ik_collision_activation_distance_m=None,
    **kwargs,
):
    """Create graph-accelerated solvers with an optional precise-goal IK clearance."""

    # SDK world/peer updates copy into the allocated buffers. Runtime capacity
    # changes recreate the planners; disabling graphs for every solve is not
    # required for live scenes and repeats CPU dispatch for each optimizer step.
    planner_config = planner_config_type.create(
        use_cuda_graph=True,
        **kwargs,
    )
    if ik_config_type is None:
        return planner_config

    ik_config_kwargs = {
        "robot": kwargs["robot"],
        "use_cuda_graph": True,
        "optimizer_collision_activation_distance": (
            kwargs["optimizer_collision_activation_distance"]
        ),
    }
    if ik_collision_activation_distance_m is not None:
        # A grasp goal can satisfy the required clearance inside TrajOpt's
        # wider soft avoidance zone. Do not push its IK seed away from it.
        ik_config_kwargs["optimizer_collision_activation_distance"] = (
            ik_collision_activation_distance_m
        )
        # Match MotionPlannerCfg.create in the pinned SDK. Rebuilding IK for
        # this clearance must not add ParticleIK or double its batch seeds.
        ik_config_kwargs["optimizer_configs"] = ["ik/lbfgs_ik.yml"]
        ik_config_kwargs["num_seeds"] = (
            16 if kwargs.get("max_batch_size", 1) > 1 else 32
        )
    for name in ("scene_model", "collision_cache", "max_batch_size", "multi_env"):
        if name in kwargs:
            ik_config_kwargs[name] = kwargs[name]
    planner_config.ik_solver_config = ik_config_type.create(
        **ik_config_kwargs,
    )
    return planner_config


def _create_world_clearance_checker(
    *,
    checker_config_type,
    checker_type,
    robot_config,
    motion_planner,
    activation_distance_m,
    gripper_envelope_arm=None,
    urdf_path=None,
):
    """Create a public cuRobo checker sharing the planner's live scene."""

    scene_checker = getattr(
        motion_planner,
        "scene_collision_checker",
        None,
    )
    if scene_checker is None:
        raise _error(
            "cuRobo motion planner has no scene collision checker",
            stage="config",
            code="CUROBO_INVALID_CONFIG",
            status=500,
        )
    if gripper_envelope_arm is not None:
        # The time-union envelope belongs only to world clearance. Using it in
        # self-collision would invent overlap between fingers at different times.
        robot_config = create_gripper_closing_envelope(
            robot_config, gripper_envelope_arm, urdf_path,
        )
    checker_config = checker_config_type.load_from_config(
        robot_config=robot_config,
        device_cfg=motion_planner.device_cfg,
        scene_collision_checker=scene_checker,
        collision_activation_distance=activation_distance_m,
    )
    # The pinned cuRobo config loader leaves both scene terms at zero spheres.
    sphere_count = int(checker_config.kinematics.total_spheres)
    for collision_term in (
        checker_config.collision_cost,
        checker_config.collision_constraint,
    ):
        collision_term.config.update_num_spheres(sphere_count)
    return checker_type(checker_config)


# ========================================
# MODIFICATION END: preserve official CUDA graph acceleration for live worlds
# ========================================


def _error(
    message,
    *,
    stage,
    code,
    status,
    retryable=False,
    details=None,
):
    return CuroboToolError(
        message,
        stage=stage,
        code=code,
        retryable=retryable,
        details={} if details is None else details,
        status=status,
    )


def _is_cuda_out_of_memory(torch_module, error):
    error_type = getattr(torch_module, "OutOfMemoryError", None)
    return isinstance(error_type, type) and isinstance(error, error_type)


# ========================================
# MODIFICATION START: validate trajectory world-clearance profiles
# ========================================
def _validate_world_clearance_profile(
    trajectory_feasibility,
    active_costs,
    peer_costs,
    *,
    arm,
):
    """Require the full robot to keep the requested world clearance at every sample."""

    feasibility = np.asarray(trajectory_feasibility)
    active = np.asarray(active_costs, dtype=np.float64)
    peer = np.asarray(peer_costs, dtype=np.float64)
    sample_count = feasibility.size if feasibility.ndim == 1 else 0
    if (
        sample_count < 1
        or active.ndim != 2
        or peer.ndim != 2
        or active.shape[0] != sample_count
        or peer.shape[0] != sample_count
        or active.shape[1] < 1
        or not np.issubdtype(feasibility.dtype, np.bool_)
        or not np.isfinite(active).all()
        or not np.isfinite(peer).all()
        or np.any(active < 0.0)
        or np.any(peer < 0.0)
    ):
        raise _error(
            "cuRobo returned an invalid world-clearance profile",
            stage="validation",
            code="CUROBO_INVALID_RESULT",
            status=500,
            details={"arm": arm},
        )

    invalid_trajectory = np.flatnonzero(~feasibility)
    if invalid_trajectory.size:
        invalid_index = int(invalid_trajectory[0])
        raise _error(
            "cuRobo rejected a colliding final command state",
            stage="validation",
            code="CUROBO_TRAJECTORY_COLLISION",
            status=422,
            details={
                "arm": arm,
                "reason": "trajectory_collision",
                "first_invalid_index": invalid_index,
                "checked_point_count": sample_count,
            },
        )

    invalid_peer = np.flatnonzero(
        np.any(peer > CLEARANCE_COST_EPSILON, axis=1)
    )
    if invalid_peer.size:
        invalid_index = int(invalid_peer[0])
        raise _error(
            "the frozen peer arm violates the motion world clearance",
            stage="validation",
            code="CUROBO_TRAJECTORY_COLLISION",
            status=422,
            details={
                "arm": arm,
                "reason": "peer_world_clearance",
                "first_invalid_index": invalid_index,
                "checked_point_count": sample_count,
            },
        )

    invalid_active = np.flatnonzero(
        np.any(active > CLEARANCE_COST_EPSILON, axis=1)
    )
    if invalid_active.size:
        invalid_index = int(invalid_active[0])
        raise _error(
            "the active arm violates the motion world clearance",
            stage="validation",
            code="CUROBO_TRAJECTORY_COLLISION",
            status=422,
            details={
                "arm": arm,
                "reason": "motion_world_clearance",
                "first_invalid_index": invalid_index,
                "checked_point_count": sample_count,
            },
        )
    return 0


# ========================================
# MODIFICATION END: validate trajectory world-clearance profiles
# ========================================


def _busy_error():
    return _error(
        "cuRobo planner is busy",
        stage="transport",
        code="CUROBO_BUSY",
        retryable=True,
        status=503,
    )


def _validated_timeout(lock_timeout_s):
    if isinstance(lock_timeout_s, bool) or not isinstance(
        lock_timeout_s,
        Real,
    ):
        raise _error(
            "lock_timeout_s must be a finite non-negative number",
            stage="input",
            code="CUROBO_INVALID_TIMEOUT",
            status=400,
        )
    timeout = float(lock_timeout_s)
    if (
        not isfinite(timeout)
        or timeout < 0.0
        or timeout > threading.TIMEOUT_MAX
    ):
        raise _error(
            "lock_timeout_s exceeds the supported finite non-negative range",
            stage="input",
            code="CUROBO_INVALID_TIMEOUT",
            status=400,
        )
    return timeout


def _remaining_timeout(deadline):
    return max(0.0, deadline - time.monotonic())


def _acquire_cuda_until(deadline):
    if not _CUDA_LOCK.acquire(timeout=_remaining_timeout(deadline)):
        raise _busy_error()


def _acquire_operation_until(deadline, timeout):
    if timeout > 0.0 and time.monotonic() >= deadline:
        raise _busy_error()
    _acquire_cuda_until(deadline)


def _tensor_to_numpy(values, *, message, code):
    try:
        detached = values.detach()
        values = detached.cpu().numpy()
    except AttributeError:
        pass
    try:
        return np.asarray(values)
    except (TypeError, ValueError) as error:
        raise _error(
            message,
            stage="validation",
            code=code,
            status=500,
        ) from error


def _success_matrix(result):
    success = getattr(result, "success", None)
    if success is None:
        raise _error(
            "cuRobo result is missing its success flag",
            stage="validation",
            code="CUROBO_INVALID_RESULT",
            status=500,
        )
    values = _tensor_to_numpy(
        success,
        message="cuRobo returned an invalid success flag",
        code="CUROBO_INVALID_RESULT",
    )
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim != 2:
        raise _error(
            "cuRobo success flag must have batch and seed dimensions",
            stage="validation",
            code="CUROBO_INVALID_RESULT",
            status=500,
        )
    if values.size == 0:
        raise _error(
            "cuRobo returned an empty success flag",
            stage="validation",
            code="CUROBO_INVALID_RESULT",
            status=500,
        )
    if np.issubdtype(values.dtype, np.number) and not np.isfinite(
        values
    ).all():
        raise _error(
            "cuRobo returned a non-finite success flag",
            stage="validation",
            code="CUROBO_INVALID_RESULT",
            status=500,
        )
    return values.astype(bool)


def _success_flag_is_true(success):
    if success is None:
        return False
    if isinstance(success, (bool, np.bool_)):
        return bool(success)
    try:
        return bool(success.any().item())
    except (AttributeError, TypeError, ValueError):
        return bool(success)


def _validated_joint_names(values, *, stage="config", code="CUROBO_INVALID_CONFIG", dimension=POINT_DIMENSION):
    if isinstance(values, (str, bytes)):
        names = ()
    else:
        try:
            names = tuple(values)
        except TypeError:
            names = ()
    if (
        len(names) != dimension
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != dimension
    ):
        raise _error(
            "cuRobo must expose exactly seven unique joint names",
            stage=stage,
            code=code,
            status=500,
        )
    return names


def _metric_matrix(result, field, success_shape, *, code):
    raw_value = getattr(result, field, None)
    if raw_value is None:
        raise _error(
            f"cuRobo result is missing {field}",
            stage="validation",
            code=code,
            status=500,
        )
    values = _tensor_to_numpy(
        raw_value,
        message=f"cuRobo returned invalid {field}",
        code=code,
    )
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise _error(
            f"cuRobo returned invalid {field}",
            stage="validation",
            code=code,
            status=500,
        ) from error
    if values.shape != tuple(success_shape):
        raise _error(
            f"cuRobo {field} shape does not match success",
            stage="validation",
            code=code,
            status=500,
        )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise _error(
            f"cuRobo returned invalid {field}",
            stage="validation",
            code=code,
            status=500,
        )
    return values


def _planning_time(result):
    raw_value = getattr(result, "total_time", None)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise _error(
            "cuRobo returned an invalid total_time",
            stage="validation",
            code="CUROBO_INVALID_RESULT",
            status=500,
        ) from error
    if not isfinite(value) or value < 0.0:
        raise _error(
            "cuRobo returned an invalid total_time",
            stage="validation",
            code="CUROBO_INVALID_RESULT",
            status=500,
        )
    return value


def _planner_interpolation_dt(motion_planner):
    try:
        value = float(
            motion_planner.trajopt_solver.config.interpolation_dt
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise _error(
            "cuRobo planner does not expose a valid interpolation_dt",
            stage="validation",
            code="CUROBO_INVALID_TIMING",
            status=500,
        ) from error
    if not isfinite(value) or value <= 0.0:
        raise _error(
            "cuRobo interpolation_dt must be positive and finite",
            stage="validation",
            code="CUROBO_INVALID_TIMING",
            status=500,
        )
    return value


def _trajectory_values(joint_state, fallback_joint_names, field="position"):
    position = getattr(joint_state, field, None)
    if position is None:
        raise _error(
            f"cuRobo interpolated trajectory has no {field}",
            stage="validation",
            code="CUROBO_INVALID_TRAJECTORY",
            status=500,
        )
    values = _tensor_to_numpy(
        position,
        message="cuRobo returned invalid trajectory positions",
        code="CUROBO_INVALID_TRAJECTORY",
    )
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise _error(
            "cuRobo returned invalid trajectory positions",
            stage="validation",
            code="CUROBO_INVALID_TRAJECTORY",
            status=500,
        ) from error
    if values.ndim == 4 and values.shape[:2] == (1, 1):
        values = values[0, 0]
    elif values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    elif values.ndim != 2:
        raise _error(
            "cuRobo trajectory must include time and joint dimensions",
            stage="validation",
            code="CUROBO_INVALID_TRAJECTORY",
            status=500,
        )
    active_names = _validated_joint_names(
        fallback_joint_names,
        stage="validation",
        code="CUROBO_INVALID_TRAJECTORY",
        dimension=len(fallback_joint_names),
    )
    output_names = getattr(joint_state, "joint_names", None) or active_names
    try:
        output_names = tuple(output_names)
    except TypeError:
        output_names = ()
    if (
        not output_names
        or any(not isinstance(name, str) or not name for name in output_names)
        or len(set(output_names)) != len(output_names)
        or values.shape[1] != len(output_names)
    ):
        raise _error(
            "cuRobo trajectory joint names do not match its positions",
            stage="validation",
            code="CUROBO_INVALID_TRAJECTORY",
            status=500,
        )
    active_name_set = set(active_names)
    active_columns = [
        (index, name)
        for index, name in enumerate(output_names)
        if name in active_name_set
    ]
    if values.shape[0] < 1 or len(active_columns) != len(active_names):
        raise _error(
            "cuRobo trajectory must contain all seven active joints",
            stage="validation",
            code="CUROBO_INVALID_TRAJECTORY",
            status=500,
        )
    values = values[:, [index for index, _name in active_columns]]
    names = tuple(name for _index, name in active_columns)
    if not np.isfinite(values).all():
        raise _error(
            "cuRobo trajectory contains non-finite values",
            stage="validation",
            code="CUROBO_INVALID_TRAJECTORY",
            status=500,
        )
    return values, names


def _hermite_command_positions(joint_state, joint_names, positions, point_scale):
    if point_scale is None:
        return None
    from .trajectory_interpolation import quintic_hermite_points

    velocity, velocity_names = _trajectory_values(joint_state, joint_names, "velocity")
    acceleration, acceleration_names = _trajectory_values(joint_state, joint_names, "acceleration")
    dt = _tensor_to_numpy(
        getattr(joint_state, "dt", None),
        message="cuRobo Hermite interpolation requires the dense trajectory dt",
        code="CUROBO_INVALID_TRAJECTORY",
    ).reshape(-1)
    if (dt.size != 1 or velocity_names != joint_names or acceleration_names != joint_names):
        raise _error(
            "cuRobo trajectory derivatives or dt do not match its positions",
            stage="validation", code="CUROBO_INVALID_TRAJECTORY", status=500,
        )
    try:
        points = quintic_hermite_points(
            positions, velocity[:len(positions)], acceleration[:len(positions)],
            sample_dt_s=float(dt[0]), point_scale=point_scale,
        )
    except (TypeError, ValueError) as error:
        raise _error(
            f"cuRobo Hermite interpolation failed: {error}",
            stage="validation", code="CUROBO_INVALID_TRAJECTORY", status=500,
        ) from error
    return tuple(tuple(float(value) for value in row) for row in points)


# ========================================
# MODIFICATION START: resample final commands at collision-check resolution
# ========================================
def _is_redundant_path_point(start, middle, end):
    """Return whether ``middle`` lies on the same directed 7-D segment."""

    full_delta = end - start
    full_norm_squared = float(np.dot(full_delta, full_delta))
    tolerance = _COLLINEAR_PATH_TOLERANCE_DEG
    if full_norm_squared <= tolerance * tolerance:
        return bool(np.linalg.norm(middle - start) <= tolerance)
    fraction = float(np.dot(middle - start, full_delta) / full_norm_squared)
    if fraction < 0.0 or fraction > 1.0:
        return False
    projected = start + fraction * full_delta
    return bool(np.linalg.norm(middle - projected) <= tolerance)


def _simplify_geometric_joint_path(points_deg):
    """Drop only duplicate or numerically collinear timing samples."""

    if len(points_deg) <= 2:
        return np.ascontiguousarray(points_deg, dtype=np.float64)
    simplified = [np.asarray(points_deg[0], dtype=np.float64)]
    for index in range(1, len(points_deg) - 1):
        middle = np.asarray(points_deg[index], dtype=np.float64)
        end = np.asarray(points_deg[index + 1], dtype=np.float64)
        if _is_redundant_path_point(simplified[-1], middle, end):
            continue
        simplified.append(middle)
    final_point = np.asarray(points_deg[-1], dtype=np.float64)
    if np.linalg.norm(final_point - simplified[-1]) > (
        _COLLINEAR_PATH_TOLERANCE_DEG
    ):
        simplified.append(final_point)
    return np.ascontiguousarray(simplified, dtype=np.float64)


def _collision_validation_points_deg(points_deg, *, arm):
    """Preserve the geometric path and cap each 7-D joint-space step."""

    geometric_points = _simplify_geometric_joint_path(points_deg)
    dense_points = [geometric_points[0].copy()]
    for goal in geometric_points[1:]:
        start = dense_points[-1]
        delta = goal - start
        subdivision_count = max(
            1,
            int(
                np.ceil(
                    np.linalg.norm(delta)
                    / COLLISION_CHECK_MAX_JOINT_STEP_DEG
                )
            ),
        )
        requested_count = len(dense_points) + subdivision_count
        if requested_count > MAX_TRAJECTORY_VALIDATION_SAMPLES:
            raise _error(
                "final command trajectory exceeds collision validation capacity",
                stage="validation",
                code="CUROBO_TRAJECTORY_TOO_COMPLEX",
                status=422,
                details={
                    "arm": arm,
                    "required_sample_count": requested_count,
                    "maximum_sample_count": (
                        MAX_TRAJECTORY_VALIDATION_SAMPLES
                    ),
                },
            )
        for step_index in range(1, subdivision_count + 1):
            dense_points.append(
                start
                + delta * (float(step_index) / subdivision_count)
            )
    return np.ascontiguousarray(dense_points, dtype=np.float64)
# ========================================
# MODIFICATION END: resample final commands at collision-check resolution
# ========================================


class _NativeCuroboBackend:
    """Translate typed native requests to one real MotionPlanner."""

    def __init__(
        self,
        *,
        arm,
        motion_planner,
        torch_module,
        joint_state_type,
        pose_type,
        goal_tool_pose_type,
        tool_pose_criteria_type,
        obstacle_cuboid_type=None,
        peer_baseline_inverse_by_link=None,
        peer_kinematics=None,
        motion_clearance_checker=None,
        active_collision_links=(),
        peer_collision_links=(),
        warmup_iterations=3,
        warmup_joint_space=False,
        device=None,
    ):
        self.arm = arm
        self.motion_planner = motion_planner
        self._peer_kinematics = peer_kinematics
        self._torch = torch_module
        self._joint_state_type = joint_state_type
        self._pose_type = pose_type
        self._obstacle_cuboid_type = obstacle_cuboid_type
        self._goal_tool_pose_type = goal_tool_pose_type
        self._tool_pose_criteria_type = tool_pose_criteria_type
        self._motion_clearance_checker = motion_clearance_checker
        self._warmup_iterations = warmup_iterations
        self._warmup_joint_space = warmup_joint_space
        # ========================================
        # MODIFICATION START: cache frozen peer-arm reference transforms
        # ========================================
        self._peer_baseline_inverse_by_link = dict(
            peer_baseline_inverse_by_link or {}
        )
        # ========================================
        # MODIFICATION END: cache frozen peer-arm reference transforms
        # ========================================

        planner_names = getattr(motion_planner, "joint_names", None)
        if not planner_names:
            planner_names = getattr(
                getattr(motion_planner, "ik_solver", None),
                "joint_names",
                None,
            )
        self.joint_names = _validated_joint_names(planner_names, dimension=14 if arm == "AB" else 7)

        try:
            tool_frames = tuple(motion_planner.tool_frames)
        except (AttributeError, TypeError) as error:
            raise _error(
                "cuRobo must expose exactly one tool frame",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
            ) from error
        if (
            len(tool_frames) != (2 if arm == "AB" else 1)
            or not isinstance(tool_frames[0], str)
            or not tool_frames[0]
        ):
            raise _error(
                "cuRobo must expose exactly one tool frame",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
            )
        self.tool_frames = tool_frames
        self.tool_frame = tool_frames[0]

        device_cfg = getattr(motion_planner, "device_cfg", None)
        self._device = getattr(device_cfg, "device", None)
        self._dtype = getattr(device_cfg, "dtype", None)
        if self._device is None:
            self._device = "cuda" if device is None else device
        if self._dtype is None:
            self._dtype = getattr(torch_module, "float32", None)

        self._active_clearance_indices_by_link = {
            link_name: self._clearance_sphere_indices(
                motion_clearance_checker,
                (link_name,),
            )
            for link_name in active_collision_links
        }
        self._active_clearance_indices = np.asarray(
            [
                index
                for indices in self._active_clearance_indices_by_link.values()
                for index in indices
            ],
            dtype=np.int64,
        )
        self._peer_clearance_indices = self._clearance_sphere_indices(
            motion_clearance_checker,
            peer_collision_links,
        )
        if motion_clearance_checker is not None:
            if self._active_clearance_indices.size < 1:
                raise _error(
                    "cuRobo active-arm clearance spheres are unavailable",
                    stage="config",
                    code="CUROBO_INVALID_CONFIG",
                    status=500,
                    details={"arm": self.arm},
                )
            overlap = np.intersect1d(
                self._active_clearance_indices,
                self._peer_clearance_indices,
            )
            if overlap.size:
                raise _error(
                    "cuRobo active and peer clearance spheres overlap",
                    stage="config",
                    code="CUROBO_INVALID_CONFIG",
                    status=500,
                    details={"arm": self.arm},
                )

    @staticmethod
    def _clearance_sphere_indices(clearance_checker, link_names):
        """Return stable sphere indices for the selected collision links."""

        if clearance_checker is None:
            return np.empty(0, dtype=np.int64)
        params = clearance_checker.kinematics.config.kinematics_config
        indices = []
        for link_name in link_names:
            link_indices = params.get_sphere_index_from_link_name(link_name)
            indices.extend(
                int(index)
                for index in _tensor_to_numpy(
                    link_indices,
                    message="cuRobo collision sphere indices are invalid",
                    code="CUROBO_INVALID_CONFIG",
                ).reshape(-1)
            )
        return np.asarray(indices, dtype=np.int64)

    def _clearance_cost_profile(self, checker, position_tensor):
        """Query one public cuRobo checker for every robot sphere."""

        if checker is None:
            raise _error(
                "cuRobo world-clearance checker is unavailable",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
                details={"arm": self.arm},
            )
        if position_tensor.ndim == 1:
            position_tensor = position_tensor.unsqueeze(0)
        if (
            position_tensor.ndim != 2
            or position_tensor.shape[1] != len(self.joint_names)
        ):
            raise _error(
                "cuRobo world-clearance query has invalid joint shape",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
                details={"arm": self.arm},
            )

        sample_count = int(position_tensor.shape[0])
        try:
            checker.setup_batch_tensors(1, sample_count)
            state = checker.get_kinematics(
                position_tensor.reshape(
                    1,
                    sample_count,
                    len(self.joint_names),
                )
            )
            costs = _tensor_to_numpy(
                # The pinned SceneCollisionCost reads KinematicsState.robot_spheres.
                checker.get_collision_distance(state),
                message="cuRobo world-clearance result is invalid",
                code="CUROBO_INVALID_RESULT",
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo world-clearance query failed",
                stage="validation",
                code="CUROBO_BACKEND_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error

        profile = np.asarray(costs, dtype=np.float64)
        if (
            profile.ndim != 3
            or profile.shape[0] != 1
            or profile.shape[1] != sample_count
            or profile.shape[2] < 1
            or not np.isfinite(profile).all()
            or np.any(profile < 0.0)
        ):
            raise _error(
                "cuRobo world-clearance result has invalid shape or values",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
                details={"arm": self.arm},
            )
        return profile[0]

    def _motion_world_clearance_costs(self, position_tensor):
        """Return clearance costs for shared, active-arm, and peer-arm spheres."""

        profile = self._clearance_cost_profile(
            self._motion_clearance_checker,
            position_tensor,
        )
        if (
            self._active_clearance_indices.max(initial=-1)
            >= profile.shape[1]
            or self._peer_clearance_indices.max(initial=-1)
            >= profile.shape[1]
        ):
            raise _error(
                "cuRobo clearance sphere indices exceed the result size",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
                details={"arm": self.arm},
            )
        return (
            profile[:, self._active_clearance_indices],
            profile[:, self._peer_clearance_indices],
        )

    def warmup(self):
        try:
            self.motion_planner.warmup(
                enable_graph=False,
                num_warmup_iterations=self._warmup_iterations,
            )
            if self._warmup_joint_space:
                # Official warmup covers plan_pose only. READY uses the public
                # joint-space solver; compile/capture it before service readiness.
                current = self.motion_planner.default_joint_state.clone().unsqueeze(0)
                goal = current.clone()
                goal.position[..., 0] += 0.2
                self.motion_planner.plan_cspace(goal, current, max_attempts=1)
        except CuroboToolError:
            raise
        except Exception as error:
            if _is_cuda_out_of_memory(self._torch, error):
                raise _error(
                    "cuRobo GPU memory is insufficient for planner warmup",
                    stage="resource",
                    code="CUROBO_RESOURCE_EXHAUSTED",
                    status=503,
                    retryable=True,
                    details={"arm": self.arm},
                ) from error
            raise _error(
                "cuRobo planner warmup failed",
                stage="config",
                code="CUROBO_WARMUP_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error

    def destroy(self):
        """Release public cuRobo resources owned by this backend."""

        planner = self.motion_planner
        if planner is None:
            return
        destroy = None
        try:
            destroy = getattr(planner, "destroy", None)
            if callable(destroy):
                destroy()
        finally:
            destroy = None
            self.motion_planner = None
            self._peer_kinematics = None
            self._motion_clearance_checker = None
            self._peer_baseline_inverse_by_link.clear()
            del planner
            gc.collect()
            cuda = getattr(self._torch, "cuda", None)
            is_available = getattr(cuda, "is_available", None)
            if callable(is_available) and is_available():
                cuda.synchronize()
                cuda.empty_cache()

    # ========================================
    # MODIFICATION START: update the cuRobo V2 collision scene
    # ========================================
    def update_scene_model(self, scene_model):
        """Load a complete V2 Scene into this arm's MotionPlanner."""

        from curobo.scene import Scene

        try:
            self.motion_planner.update_world(Scene.create(scene_model))
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo obstacle scene update failed",
                stage="scene",
                code="CUROBO_SCENE_UPDATE_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error

    def update_voxel_grid(self, voxel_grid):
        """Load one ESDF VoxelGrid and no primitive obstacles."""

        from curobo.scene import Scene

        try:
            self.motion_planner.update_world(Scene(voxel=[voxel_grid]))
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo ESDF scene update failed",
                stage="scene",
                code="CUROBO_ESDF_SCENE_UPDATE_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error

    def update_voxel_scene(
        self,
        voxel_grid,
        scene_model,
        *,
        disabled_obstacle_names=(),
    ):
        """Load the background and object ESDF layers, switching only named objects."""

        from curobo.scene import Scene

        from .mixed_esdf_scene import create_mixed_esdf_scene, update_mixed_esdf_scene

        try:
            update_mixed_esdf_scene(
                self.motion_planner,
                create_mixed_esdf_scene(Scene, voxel_grid, scene_model),
            )
            semantic_names = []
            for geometry_type in ("cuboid", "mesh", "voxel"):
                geometries = scene_model.get(geometry_type, {})
                if not isinstance(geometries, Mapping):
                    continue
                semantic_names.extend(
                    name
                    for name in geometries
                    if isinstance(name, str)
                    and name.startswith("semantic_instance::")
                )
            disabled = frozenset(disabled_obstacle_names)
            unknown = disabled.difference(semantic_names)
            if unknown:
                raise _error(
                    "cuRobo semantic obstacle selection is invalid",
                    stage="scene",
                    code="SCENE_OBJECT_UNSTABLE",
                    status=409,
                    details={"unknown_obstacles": sorted(unknown)},
                )
            if not semantic_names:
                return
            scene_checker = getattr(
                self.motion_planner,
                "scene_collision_checker",
                None,
            )
            scene_data = getattr(scene_checker, "data", None)
            if scene_data is None:
                raise _error(
                    "cuRobo scene checker cannot switch named obstacles",
                    stage="configuration",
                    code="CUROBO_INVALID_CONFIG",
                    status=500,
                    details={"arm": self.arm},
                )
            # Address each semantic obstacle through its known public type.
            for geometry_type, geometry_data in (
                ("cuboid", scene_data.cuboids),
                ("mesh", scene_data.meshes),
                ("voxel", scene_data.voxels),
            ):
                for name in sorted(scene_model.get(geometry_type, {})):
                    if name in semantic_names:
                        geometry_data.set_enabled(
                            name, enabled=name not in disabled, env_idx=0
                        )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo fixed semantic ESDF scene update failed",
                stage="scene",
                code="CUROBO_ESDF_SCENE_UPDATE_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error
    # ========================================
    # MODIFICATION END: update the cuRobo V2 collision scene
    # ========================================

    # ========================================
    # MODIFICATION START: refresh the frozen peer-arm collision spheres
    # ========================================
    def update_peer_collision_model(
        self,
        peer_backend,
        active_joints_deg,
        other_joints_deg,
    ):
        """Move frozen peer-arm spheres to the latest feedback pose.

        The target planner keeps the peer joints locked at zero.  World-space
        spheres computed by the peer arm's active planner are therefore
        transformed into the target planner's zero-pose link coordinates.
        Updating the shared kinematics tensor makes IK, graph search, and
        trajectory optimization see the same stationary peer geometry.
        """

        if getattr(peer_backend, "arm", None) == self.arm:
            raise _error(
                "peer collision model requires the opposite arm backend",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
                details={"arm": self.arm},
            )
        if not self._peer_baseline_inverse_by_link:
            raise _error(
                "peer collision reference transforms are unavailable",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
                details={"arm": self.arm},
            )

        try:
            # Validate the active seed at the same boundary as normal solves.
            self._state(active_joints_deg)
            if peer_backend is None and self._peer_kinematics is not None:
                peer_position = self._torch.as_tensor(
                    np.radians(other_joints_deg).reshape(1, 7), device=self._device, dtype=self._dtype,
                )
                peer_state = self._joint_state_type.from_position(peer_position, joint_names=list(self._peer_kinematics.joint_names))
                peer_kinematics_state = self._peer_kinematics.compute_kinematics(peer_state)
                peer_params = self._peer_kinematics.config.kinematics_config
            else:
                peer_state = peer_backend._state(other_joints_deg)
                peer_kinematics_state = peer_backend.motion_planner.compute_kinematics(peer_state)
                peer_params = peer_backend.motion_planner.kinematics.config.kinematics_config
            peer_world_spheres = peer_kinematics_state.robot_spheres[0, 0]

            target_param_sets = [
                self.motion_planner.kinematics.config.kinematics_config
            ]
            ik_solver = getattr(self.motion_planner, "ik_solver", None)
            ik_kinematics = getattr(ik_solver, "kinematics", None)
            if ik_kinematics is not None:
                ik_params = ik_kinematics.config.kinematics_config
                if all(
                    ik_params is not current
                    for current in target_param_sets
                ):
                    target_param_sets.append(ik_params)
            if self._motion_clearance_checker is not None:
                motion_checker_kinematics = (
                    self._motion_clearance_checker.kinematics
                )
                motion_checker_params = (
                    motion_checker_kinematics.config.kinematics_config
                )
                if all(
                    motion_checker_params is not current
                    for current in target_param_sets
                ):
                    target_param_sets.append(motion_checker_params)
            for link_name, baseline_inverse in (
                self._peer_baseline_inverse_by_link.items()
            ):
                peer_indices = peer_params.get_sphere_index_from_link_name(
                    link_name
                )
                for target_params in target_param_sets:
                    target_indices = (
                        target_params.get_sphere_index_from_link_name(
                            link_name
                        )
                    )
                    if peer_indices.numel() != target_indices.numel():
                        raise ValueError(
                            "collision sphere count differs for "
                            f"{link_name}"
                        )

                world_spheres = peer_world_spheres[peer_indices]
                rotation = baseline_inverse[:3, :3]
                translation = baseline_inverse[:3, 3]
                local_positions = (
                    world_spheres[:, :3].to(dtype=rotation.dtype)
                    @ rotation.transpose(0, 1)
                    + translation
                ).to(dtype=world_spheres.dtype)
                local_spheres = self._torch.cat(
                    (local_positions, world_spheres[:, 3:4]),
                    dim=-1,
                )
                for target_params in target_param_sets:
                    target_params.update_link_spheres(
                        link_name,
                        local_spheres,
                    )

            graph_planner = getattr(
                self.motion_planner,
                "graph_planner",
                None,
            )
            reset_buffer = getattr(graph_planner, "reset_buffer", None)
            if callable(reset_buffer):
                reset_buffer()
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo peer-arm collision update failed",
                stage="sdk",
                code="CUROBO_PEER_COLLISION_UPDATE_FAILED",
                status=500,
                details={
                    "arm": self.arm,
                    "peer_arm": getattr(peer_backend, "arm", None),
                },
            ) from error
    # ========================================
    # MODIFICATION END: refresh the frozen peer-arm collision spheres
    # ========================================

    def _state(self, joints_deg):
        try:
            position = self._torch.as_tensor(
                np.deg2rad(np.asarray(joints_deg, dtype=np.float64)),
                device=self._device,
                dtype=self._dtype,
            ).unsqueeze(0)
            current_state = self._joint_state_type.from_position(
                position,
                joint_names=list(self.joint_names),
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo request tensor construction failed",
                stage="sdk",
                code="CUROBO_BACKEND_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error
        return current_state

    def _goal(self, goal_position_m, goal_quaternion_wxyz):
        try:
            target_pose = self._pose_type(
                position=self._torch.as_tensor(
                    goal_position_m,
                    device=self._device,
                    dtype=self._dtype,
                ).unsqueeze(0),
                quaternion=self._torch.as_tensor(
                    goal_quaternion_wxyz,
                    device=self._device,
                    dtype=self._dtype,
                ).unsqueeze(0),
            )
            goal = self._goal_tool_pose_type.from_poses(
                {self.tool_frame: target_pose},
                ordered_tool_frames=[self.tool_frame],
                num_goalset=1,
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo request tensor construction failed",
                stage="sdk",
                code="CUROBO_BACKEND_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error
        return goal

    def _state_and_goal(
        self,
        joints_deg,
        goal_position_m,
        goal_quaternion_wxyz,
    ):
        return (
            self._state(joints_deg),
            self._goal(goal_position_m, goal_quaternion_wxyz),
        )

    def _tool_pose_arrays(self, current_state, *, sample_count):
        try:
            kinematics = self.motion_planner.compute_kinematics(current_state)
            tool_poses = kinematics.tool_poses
            pose = tool_poses.get_link_pose(self.tool_frame)
            positions = np.asarray(
                _tensor_to_numpy(
                    pose.position,
                    message="cuRobo FK position is invalid",
                    code="CUROBO_FORWARD_KINEMATICS_FAILED",
                ),
                dtype=np.float64,
            )
            quaternions = np.asarray(
                _tensor_to_numpy(
                    pose.quaternion,
                    message="cuRobo FK quaternion is invalid",
                    code="CUROBO_FORWARD_KINEMATICS_FAILED",
                ),
                dtype=np.float64,
            )
            if positions.shape != (sample_count, 3) or quaternions.shape != (
                sample_count,
                4,
            ):
                raise ValueError("unexpected FK pose shape")
            if not np.isfinite(positions).all() or not np.isfinite(
                quaternions
            ).all():
                raise ValueError("non-finite FK pose")
            return positions, quaternions
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo forward kinematics failed",
                stage="sdk",
                code="CUROBO_FORWARD_KINEMATICS_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error

    def _forward_tool_pose(self, current_state):
        positions, quaternions = self._tool_pose_arrays(
            current_state,
            sample_count=1,
        )
        return positions[0], quaternions[0]

    def _trajectory_tool_poses(self, trajectory):
        checked_positions = (
            trajectory.command_positions_rad
            if trajectory.command_positions_rad is not None else trajectory.positions_rad
        )
        positions = np.asarray(checked_positions, dtype=np.float64)
        column_by_name = {
            name: index for index, name in enumerate(trajectory.joint_names)
        }
        # ========================================
        # MODIFICATION START: make reordered trajectory FK joints contiguous
        # ========================================
        ordered_positions = np.ascontiguousarray(
            positions[
                :,
                [column_by_name[name] for name in self.joint_names],
            ]
        )
        # ========================================
        # MODIFICATION END: make reordered trajectory FK joints contiguous
        # ========================================
        try:
            state = self._joint_state_type.from_position(
                self._torch.as_tensor(
                    ordered_positions,
                    device=self._device,
                    dtype=self._dtype,
                ),
                joint_names=list(self.joint_names),
            )
        except Exception as error:
            raise _error(
                "cuRobo trajectory FK state construction failed",
                stage="sdk",
                code="CUROBO_FORWARD_KINEMATICS_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error
        return self._tool_pose_arrays(
            state,
            sample_count=positions.shape[0],
        )

    # ========================================
    # MODIFICATION START: validate every final command sample for collision
    # ========================================
    def validate_trajectory_native(self, request):
        """Return the number of collision-free command samples checked."""

        graph_planner = getattr(
            self.motion_planner,
            "graph_planner",
            None,
        )
        checker = getattr(
            graph_planner,
            "check_samples_feasibility",
            None,
        )
        if not callable(checker):
            raise _error(
                "cuRobo graph feasibility checker is unavailable",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
                details={"arm": self.arm},
            )
        try:
            validation_points_deg = _collision_validation_points_deg(
                request.points_deg,
                arm=self.arm,
            )
            point_count = len(validation_points_deg)
            positions_rad = np.ascontiguousarray(
                np.deg2rad(validation_points_deg)
            )
            position_tensor = self._torch.as_tensor(
                positions_rad,
                device=self._device,
                dtype=self._dtype,
            )
            feasibility = _tensor_to_numpy(
                checker(position_tensor),
                message="cuRobo trajectory feasibility result is invalid",
                code="CUROBO_GRAPH_FAILED",
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo trajectory feasibility check failed",
                stage="graph",
                code="CUROBO_GRAPH_FAILED",
                status=500,
                details={"arm": self.arm},
            ) from error
        flat_feasibility = np.asarray(feasibility).reshape(-1)
        if flat_feasibility.size != point_count:
            raise _error(
                "cuRobo trajectory feasibility result has invalid size",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
                details={
                    "arm": self.arm,
                    "expected_point_count": point_count,
                    "actual_result_count": int(flat_feasibility.size),
                },
            )
        if not np.issubdtype(flat_feasibility.dtype, np.bool_):
            raise _error(
                "cuRobo trajectory feasibility result must be boolean",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
                details={"arm": self.arm},
            )
        if np.issubdtype(flat_feasibility.dtype, np.number) and not np.isfinite(
            flat_feasibility
        ).all():
            raise _error(
                "cuRobo trajectory feasibility result is non-finite",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
                details={"arm": self.arm},
            )
        active_costs, peer_costs = self._motion_world_clearance_costs(
            position_tensor
        )
        _validate_world_clearance_profile(
            flat_feasibility,
            active_costs,
            peer_costs,
            arm=self.arm,
        )
        return point_count
    # ========================================
    # MODIFICATION END: validate every final command sample for collision
    # ========================================

    def plan_to_pose_native(self, request):
        current_state, goal = self._state_and_goal(
            request.start_joints_deg,
            request.goal_position_m,
            request.goal_quaternion_wxyz,
        )
        return self._plan_goal_native(request, current_state, goal)

    def plan_grasp_native(self, request):
        """Call the official cuRobo three-phase grasp planner."""

        if not isinstance(request, PlanGraspRequest):
            raise _error(
                "request must be a PlanGraspRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        current_state, goal = self._state_and_goal(
            request.start_joints_deg,
            request.goal_position_m,
            request.goal_quaternion_wxyz,
        )
        try:
            result = self.motion_planner.plan_grasp(
                goal,
                current_state,
                grasp_approach_axis=request.approach_axis,
                grasp_approach_offset=request.approach_offset_m,
                grasp_approach_in_tool_frame=request.approach_in_tool_frame,
                grasp_lift_axis=request.lift_axis,
                grasp_lift_offset=request.lift_offset_m,
                grasp_lift_in_tool_frame=request.lift_in_tool_frame,
                plan_approach_to_grasp=True,
                plan_grasp_to_lift=True,
                disable_collision_links=[],
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo grasp trajectory optimization failed",
                stage="trajopt",
                code="CUROBO_GRASP_FAILED",
                status=422,
                details={"arm": self.arm},
            ) from error
        finally:
            try:
                # cuRobo normally restores the tool criteria itself. This
                # outer restoration also covers an exception raised inside
                # an intermediate plan_pose call.
                self.motion_planner.update_tool_pose_criteria(
                    {self.tool_frame: self._tool_pose_criteria_type()}
                )
            except Exception as error:
                raise _error(
                    "cuRobo grasp planner criteria restoration failed",
                    stage="validation",
                    code="CUROBO_GRASP_RESTORE_FAILED",
                    status=500,
                    details={"arm": self.arm},
                ) from error

        if result is None:
            raise _error(
                "cuRobo grasp planning returned no result",
                stage="trajopt",
                code="CUROBO_GRASP_FAILED",
                status=422,
                details={"arm": self.arm},
            )
        failed_phase = None
        failure_status = ""
        for phase in ("approach", "grasp", "lift"):
            if not _success_flag_is_true(
                getattr(result, f"{phase}_success", None)
            ):
                failed_phase = phase
                failure_status = str(
                    getattr(result, "status", "") or ""
                )
                break
        if failed_phase is not None:
            # The official result owns large GPU tensors. Drop it before
            # raising so the exception traceback cannot retain those tensors
            # past request-scoped backend cleanup.
            result = None
            raise _error(
                f"cuRobo {failed_phase} planning failed",
                stage="trajopt",
                code="CUROBO_GRASP_FAILED",
                status=422,
                details={
                    "arm": self.arm,
                    "phase": failed_phase,
                    "status": failure_status,
                },
            )

        interpolation_dt_s = _planner_interpolation_dt(self.motion_planner)
        try:
            planning_time_s = float(getattr(result, "planning_time", 0.0))
        except (TypeError, ValueError):
            planning_time_s = 0.0
        if not isfinite(planning_time_s) or planning_time_s < 0.0:
            planning_time_s = 0.0

        def phase_trajectory(name):
            phase_state = getattr(result, f"{name}_interpolated_trajectory", None)
            positions, joint_names = _trajectory_values(
                phase_state,
                self.joint_names,
            )
            last_tstep = _tensor_to_numpy(
                getattr(
                    result,
                    f"{name}_interpolated_last_tstep",
                    None,
                ),
                message=(
                    f"cuRobo returned invalid {name} interpolation length"
                ),
                code="CUROBO_INVALID_TRAJECTORY",
            ).reshape(-1)
            if last_tstep.size != 1:
                raise _error(
                    f"cuRobo returned invalid {name} interpolation length",
                    stage="validation",
                    code="CUROBO_INVALID_TRAJECTORY",
                    status=500,
                    details={"arm": self.arm, "phase": name},
                )
            try:
                point_count = int(last_tstep[0])
            except (TypeError, ValueError, OverflowError) as error:
                raise _error(
                    f"cuRobo returned invalid {name} interpolation length",
                    stage="validation",
                    code="CUROBO_INVALID_TRAJECTORY",
                    status=500,
                    details={"arm": self.arm, "phase": name},
                ) from error
            if (
                point_count < 1
                or point_count > positions.shape[0]
                or float(last_tstep[0]) != float(point_count)
            ):
                raise _error(
                    f"cuRobo returned invalid {name} interpolation length",
                    stage="validation",
                    code="CUROBO_INVALID_TRAJECTORY",
                    status=500,
                    details={"arm": self.arm, "phase": name},
                )
            positions = positions[:point_count]
            return NativeTrajectory(
                command_positions_rad=_hermite_command_positions(
                    phase_state, joint_names, positions, request.trajectory_point_scale,
                ),
                trajectory_point_scale=request.trajectory_point_scale,
                positions_rad=tuple(
                    tuple(float(value) for value in row) for row in positions
                ),
                joint_names=joint_names,
                interpolation_dt_s=interpolation_dt_s,
                planning_time_s=planning_time_s,
                position_error_m=0.0,
                rotation_error_rad=0.0,
            )

        return NativeGraspPlan.create(
            approach=phase_trajectory("approach"),
            grasp=phase_trajectory("grasp"),
            lift=phase_trajectory("lift"),
        )

    def plan_grasp_batch_native(self, request):
        """Map N independent rigid goals to one official batched grasp call."""

        if not isinstance(request, PlanGraspBatchRequest):
            raise _error(
                "request must be a PlanGraspBatchRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        batch_size = len(request.goal_matrices_world)
        try:
            current = self._state(request.start_joints_deg)
            repeated_position = current.position.repeat(batch_size, 1)
            current_state = self._joint_state_type.from_position(
                repeated_position,
                joint_names=list(self.joint_names),
            )
            matrices = np.asarray(request.goal_matrices_world, dtype=np.float64)
            positions = matrices[:, :3, 3]
            quaternions = np.stack(
                [_rotation_matrix_to_quaternion_wxyz(matrix[:3, :3]) for matrix in matrices],
                axis=0,
            )
            target_pose = self._pose_type(
                position=self._torch.as_tensor(positions, device=self._device, dtype=self._dtype),
                quaternion=self._torch.as_tensor(quaternions, device=self._device, dtype=self._dtype),
            )
            goal = self._goal_tool_pose_type.from_poses(
                {self.tool_frame: target_pose},
                ordered_tool_frames=[self.tool_frame],
                num_goalset=1,
            )
        except CuroboToolError:
            raise
        except Exception as error:
            if _is_cuda_out_of_memory(self._torch, error):
                raise _error(
                    "cuRobo GPU memory is insufficient for batch tensor construction",
                    stage="resource",
                    code="CUROBO_RESOURCE_EXHAUSTED",
                    status=503,
                    retryable=True,
                    details={"arm": self.arm},
                ) from error
            raise _error(
                "cuRobo batch request tensor construction failed",
                stage="sdk",
                code="CUROBO_BACKEND_FAILED",
                status=500,
                details={"arm": self.arm, "batch_size": batch_size},
            ) from error

        try:
            result = self.motion_planner.plan_grasp(
                goal,
                current_state,
                grasp_approach_axis=request.approach_axis,
                grasp_approach_offset=request.approach_offset_m,
                grasp_approach_in_tool_frame=request.approach_in_tool_frame,
                grasp_lift_axis=request.lift_axis,
                grasp_lift_offset=request.lift_offset_m,
                grasp_lift_in_tool_frame=request.lift_in_tool_frame,
                plan_approach_to_grasp=True,
                plan_grasp_to_lift=True,
                disable_collision_links=[],
            )
        except CuroboToolError:
            raise
        except Exception as error:
            if _is_cuda_out_of_memory(self._torch, error):
                raise _error(
                    "cuRobo GPU memory is insufficient for batched grasp planning",
                    stage="resource",
                    code="CUROBO_RESOURCE_EXHAUSTED",
                    status=503,
                    retryable=True,
                    details={"arm": self.arm, "batch_size": batch_size},
                ) from error
            raise _error(
                "cuRobo batched grasp trajectory optimization failed",
                stage="trajopt",
                code="CUROBO_GRASP_FAILED",
                status=422,
                details={"arm": self.arm, "batch_size": batch_size},
            ) from error
        finally:
            try:
                self.motion_planner.update_tool_pose_criteria(
                    {self.tool_frame: self._tool_pose_criteria_type()}
                )
            except Exception as error:
                raise _error(
                    "cuRobo batched grasp planner criteria restoration failed",
                    stage="validation",
                    code="CUROBO_GRASP_RESTORE_FAILED",
                    status=500,
                    details={"arm": self.arm},
                ) from error

        if result is None:
            raise _error(
                "cuRobo returned an invalid batched grasp result",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
                details={"arm": self.arm},
            )

        def batch_flags(value, name):
            values = _tensor_to_numpy(
                value,
                message=f"cuRobo returned invalid {name}",
                code="CUROBO_INVALID_RESULT",
            )
            is_bool = np.issubdtype(values.dtype, np.bool_)
            is_numeric = np.issubdtype(values.dtype, np.number)
            if not is_bool and (
                not is_numeric
                or not np.isfinite(values).all()
                or not np.logical_or(values == 0, values == 1).all()
            ):
                raise _error(
                    f"cuRobo returned invalid {name}",
                    stage="validation",
                    code="CUROBO_INVALID_RESULT",
                    status=500,
                    details={"arm": self.arm},
                )
            if values.ndim == 1 and values.shape == (batch_size,):
                flattened = values
            elif values.ndim == 2 and values.shape[0] == batch_size and values.shape[1] > 0:
                flattened = values.astype(bool).any(axis=1)
            else:
                raise _error(
                    f"cuRobo returned invalid {name}",
                    stage="validation",
                    code="CUROBO_INVALID_RESULT",
                    status=500,
                    details={"arm": self.arm},
                )
            return np.asarray(flattened, dtype=bool)

        status = getattr(result, "status", None)
        if (
            not isinstance(status, str)
            or not status
            or len(status) > 256
            or any(ord(character) < 32 for character in status)
        ):
            raise _error(
                "cuRobo returned an invalid batched grasp status",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
                details={"arm": self.arm},
            )

        approach_success = batch_flags(
            getattr(result, "approach_success", None),
            "approach success",
        )
        grasp_success = batch_flags(
            getattr(result, "grasp_success", None),
            "grasp success",
        )
        lift_success = batch_flags(
            getattr(result, "lift_success", None),
            "lift success",
        )
        success = batch_flags(getattr(result, "success", None), "success")
        goalset_result = getattr(result, "goalset_result", None)
        if goalset_result is None:
            phase_fields = tuple(
                getattr(result, f"{name}_{suffix}", None)
                for name in ("approach", "grasp", "lift")
                for suffix in (
                    "interpolated_trajectory",
                    "interpolated_last_tstep",
                )
            )
            if (
                status != "Goalset planning returned None."
                or success.any()
                or approach_success.any()
                or grasp_success.any()
                or lift_success.any()
                or any(value is not None for value in phase_fields)
            ):
                raise _error(
                    "cuRobo returned an invalid batched grasp terminal result",
                    stage="validation",
                    code="CUROBO_INVALID_RESULT",
                    status=500,
                    details={"arm": self.arm},
                )
            goalset_success = np.zeros(batch_size, dtype=bool)
        else:
            goalset_success = batch_flags(
                getattr(goalset_result, "success", None),
                "goalset success",
            )
        conjunction = goalset_success & approach_success & grasp_success & lift_success
        if (
            not np.array_equal(success, conjunction)
            or np.any(approach_success & ~goalset_success)
            or np.any(grasp_success & ~approach_success)
            or np.any(lift_success & ~grasp_success)
        ):
            raise _error(
                "cuRobo returned an invalid batched grasp stage result",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
                details={"arm": self.arm},
            )

        try:
            planning_time_s = float(getattr(result, "planning_time", 0.0))
        except (TypeError, ValueError) as error:
            raise _error(
                "cuRobo returned an invalid batched grasp planning time",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
            ) from error
        if not isfinite(planning_time_s) or planning_time_s < 0.0:
            raise _error(
                "cuRobo returned an invalid batched grasp planning time",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
            )
        interpolation_dt_s = _planner_interpolation_dt(self.motion_planner)

        def phase_values(name):
            trajectory = getattr(result, f"{name}_interpolated_trajectory", None)
            raw_positions = getattr(trajectory, "position", None)
            positions = _tensor_to_numpy(
                raw_positions,
                message=f"cuRobo returned invalid {name} batch trajectory",
                code="CUROBO_INVALID_TRAJECTORY",
            )
            try:
                positions = np.asarray(positions, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise _error(
                    f"cuRobo returned invalid {name} batch trajectory",
                    stage="validation",
                    code="CUROBO_INVALID_TRAJECTORY",
                    status=500,
                ) from error
            # Official TrajOpt keeps the single returned seed axis: [B, 1, T, D].
            if positions.ndim == 4 and positions.shape[1] == 1:
                positions = positions[:, 0, :, :]
            if positions.ndim != 3 or positions.shape[0] != batch_size:
                raise _error(
                    f"cuRobo returned invalid {name} batch trajectory",
                    stage="validation",
                    code="CUROBO_INVALID_TRAJECTORY",
                    status=500,
                )
            joint_names = getattr(trajectory, "joint_names", None) or self.joint_names
            last_tstep = _tensor_to_numpy(
                getattr(result, f"{name}_interpolated_last_tstep", None),
                message=f"cuRobo returned invalid {name} batch interpolation length",
                code="CUROBO_INVALID_TRAJECTORY",
            ).reshape(-1)
            if last_tstep.size != batch_size:
                raise _error(
                    f"cuRobo returned invalid {name} batch interpolation length",
                    stage="validation",
                    code="CUROBO_INVALID_TRAJECTORY",
                    status=500,
                )
            return positions, joint_names, last_tstep

        phase_flags = {
            "approach": approach_success,
            "grasp": grasp_success,
            "lift": lift_success,
        }
        phase_data = {
            name: phase_values(name) if flags.any() else None
            for name, flags in phase_flags.items()
        }

        def phase_trajectory(name, candidate_index):
            positions, joint_names, last_tstep = phase_data[name]
            try:
                point_count = int(last_tstep[candidate_index])
            except (TypeError, ValueError, OverflowError) as error:
                raise _error(
                    f"cuRobo returned invalid {name} batch interpolation length",
                    stage="validation",
                    code="CUROBO_INVALID_TRAJECTORY",
                    status=500,
                ) from error
            if point_count < 1 or point_count > positions.shape[1] or float(last_tstep[candidate_index]) != float(point_count):
                raise _error(
                    f"cuRobo returned invalid {name} batch interpolation length",
                    stage="validation",
                    code="CUROBO_INVALID_TRAJECTORY",
                    status=500,
                )
            active_positions, active_names = _trajectory_values(
                SimpleNamespace(
                    position=positions[candidate_index, :point_count],
                    joint_names=joint_names,
                ),
                self.joint_names,
            )
            return NativeTrajectory(
                positions_rad=tuple(tuple(float(value) for value in row) for row in active_positions),
                joint_names=active_names,
                interpolation_dt_s=interpolation_dt_s,
                planning_time_s=planning_time_s,
                position_error_m=0.0,
                rotation_error_rad=0.0,
                scene_token=request.scene_token,
            )

        items = []
        for candidate_index in range(batch_size):
            trajectories = {}
            for name, flags in (
                ("approach", approach_success),
                ("grasp", grasp_success),
                ("lift", lift_success),
            ):
                trajectories[name] = phase_trajectory(name, candidate_index) if flags[candidate_index] else None
            grasp_end = None if trajectories["grasp"] is None else trajectories["grasp"].positions_rad[-1]
            items.append(
                NativeGraspBatchItem.create(
                    candidate_index=candidate_index,
                    success=bool(success[candidate_index]),
                    goalset_success=bool(goalset_success[candidate_index]),
                    approach_success=bool(approach_success[candidate_index]),
                    grasp_success=bool(grasp_success[candidate_index]),
                    lift_success=bool(lift_success[candidate_index]),
                    status=status,
                    approach=trajectories["approach"],
                    grasp=trajectories["grasp"],
                    lift=trajectories["lift"],
                    grasp_end_joints_rad=grasp_end,
                )
            )
        return NativeGraspBatchPlan.create(
            items=tuple(items),
            batch_size=batch_size,
            batch_capacity=batch_size,
            planning_time_s=planning_time_s,
            scene_token=request.scene_token,
        )

    def plan_place_native(self, request):
        """Use official grasp phases with place names; execution owns gripper release."""
        if not isinstance(request, PlanPlaceRequest):
            raise _error(
                "request must be a PlanPlaceRequest",
                stage="input", code="CUROBO_INVALID_REQUEST", status=400,
            )
        result = self.plan_grasp_native(request)
        return NativePlacePlan.create(
            approach=result.approach,
            place=result.grasp,
            retreat=result.lift,
        )

    def _plan_goal_native(self, request, current_state, goal):
        try:
            result = self.motion_planner.plan_pose(
                goal,
                current_state,
                use_implicit_goal=True,
                max_attempts=request.max_attempts,
                enable_graph_attempt=1,
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo trajectory optimization failed",
                stage="trajopt",
                code="CUROBO_TRAJOPT_FAILED",
                status=422,
                details={"arm": self.arm},
            ) from error

        if result is None:
            raise _error(
                "cuRobo planning returned no result",
                stage="trajopt",
                code="CUROBO_PLANNING_FAILED",
                status=422,
                details={
                    "arm": self.arm,
                    "reason": "plan_pose returned no result",
                    "cause": "cuRobo does not expose the IK or graph cause",
                },
            )

        success = _success_matrix(result)
        if success.shape != (1, 1):
            raise _error(
                "cuRobo plan result must contain one batch and one seed",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
            )

        successful_indices = np.argwhere(success)
        if successful_indices.size == 0:
            raise _error(
                "cuRobo trajectory optimization found no valid trajectory",
                stage="trajopt",
                code="CUROBO_TRAJOPT_FAILED",
                status=422,
                details={"arm": self.arm},
            )
        batch_index, seed_index = (
            int(successful_indices[0, 0]),
            int(successful_indices[0, 1]),
        )

        get_plan = getattr(result, "get_interpolated_plan", None)
        if not callable(get_plan):
            raise _error(
                "cuRobo result has no interpolated trajectory",
                stage="validation",
                code="CUROBO_INVALID_TRAJECTORY",
                status=500,
            )
        try:
            interpolated_plan = get_plan()
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo interpolated trajectory extraction failed",
                stage="validation",
                code="CUROBO_INVALID_TRAJECTORY",
                status=500,
            ) from error
        if interpolated_plan is None:
            raise _error(
                "cuRobo returned no interpolated trajectory",
                stage="validation",
                code="CUROBO_INVALID_TRAJECTORY",
                status=500,
            )

        positions, joint_names = _trajectory_values(
            interpolated_plan,
            self.joint_names,
        )
        position_errors = _metric_matrix(
            result,
            "position_error",
            success.shape,
            code="CUROBO_INVALID_RESULT",
        )
        rotation_errors = _metric_matrix(
            result,
            "rotation_error",
            success.shape,
            code="CUROBO_INVALID_RESULT",
        )

        try:
            return NativeTrajectory(
                command_positions_rad=_hermite_command_positions(
                    interpolated_plan, joint_names, positions, request.trajectory_point_scale,
                ),
                trajectory_point_scale=request.trajectory_point_scale,
                positions_rad=tuple(
                    tuple(float(value) for value in row)
                    for row in positions
                ),
                joint_names=joint_names,
                interpolation_dt_s=_planner_interpolation_dt(
                    self.motion_planner
                ),
                planning_time_s=_planning_time(result),
                position_error_m=float(
                    position_errors[batch_index, seed_index]
                ),
                rotation_error_rad=float(
                    rotation_errors[batch_index, seed_index]
                ),
            )
        except CuroboToolError:
            raise
        except ValueError as error:
            raise _error(
                "cuRobo returned invalid trajectory data",
                stage="validation",
                code="CUROBO_INVALID_TRAJECTORY",
                status=500,
            ) from error

    def plan_to_joint_native(self, request):
        current_state = self._state(request.start_joints_deg)
        goal_state = self._state(request.goal_joints_deg)
        try:
            result = self.motion_planner.plan_cspace(
                goal_state,
                current_state,
                max_attempts=request.max_attempts,
                enable_graph_attempt=1,
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo joint-space trajectory optimization failed",
                stage="trajopt",
                code="CUROBO_TRAJOPT_FAILED",
                status=422,
                details={"arm": self.arm},
            ) from error
        if result is None:
            raise _error(
                "cuRobo joint-space planning returned no result",
                stage="trajopt",
                code="CUROBO_PLANNING_FAILED",
                status=422,
                details={"arm": self.arm},
            )
        success = _success_matrix(result)
        if success.shape != (1, 1) or not bool(success[0, 0]):
            raise _error(
                "cuRobo joint-space planning found no valid trajectory",
                stage="trajopt",
                code="CUROBO_TRAJOPT_FAILED",
                status=422,
                details={"arm": self.arm},
            )
        get_plan = getattr(result, "get_interpolated_plan", None)
        if not callable(get_plan):
            raise _error(
                "cuRobo joint result has no interpolated trajectory",
                stage="validation",
                code="CUROBO_INVALID_TRAJECTORY",
                status=500,
            )
        try:
            interpolated_plan = get_plan()
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo joint trajectory extraction failed",
                stage="validation",
                code="CUROBO_INVALID_TRAJECTORY",
                status=500,
            ) from error
        if interpolated_plan is None:
            raise _error(
                "cuRobo returned no interpolated joint trajectory",
                stage="validation",
                code="CUROBO_INVALID_TRAJECTORY",
                status=500,
            )
        positions, joint_names = _trajectory_values(
            interpolated_plan,
            self.joint_names,
        )
        goal_by_name = dict(zip(self.joint_names, request.goal_joints_deg))
        expected_goal_rad = np.asarray(
            [np.deg2rad(goal_by_name[name]) for name in joint_names],
            dtype=np.float64,
        )
        endpoint_error_rad = float(
            np.max(np.abs(positions[-1] - expected_goal_rad))
        )
        if endpoint_error_rad > 1.0e-5:
            raise _error(
                "cuRobo joint trajectory did not end at the requested goal",
                stage="validation",
                code="CUROBO_INVALID_TRAJECTORY",
                status=500,
                details={
                    "arm": self.arm,
                    "endpoint_error_rad": endpoint_error_rad,
                },
            )
        try:
            return NativeTrajectory(
                command_positions_rad=_hermite_command_positions(
                    interpolated_plan, joint_names, positions, request.trajectory_point_scale,
                ),
                trajectory_point_scale=request.trajectory_point_scale,
                positions_rad=tuple(
                    tuple(float(value) for value in row)
                    for row in positions
                ),
                joint_names=joint_names,
                interpolation_dt_s=_planner_interpolation_dt(
                    self.motion_planner
                ),
                planning_time_s=_planning_time(result),
                position_error_m=0.0,
                rotation_error_rad=0.0,
            )
        except ValueError as error:
            raise _error(
                "cuRobo returned invalid joint trajectory data",
                stage="validation",
                code="CUROBO_INVALID_TRAJECTORY",
                status=500,
            ) from error

    # ========================================
    # MODIFICATION START: isolate strict-line candidate planning
    # ========================================
    def _plan_axis_linear_candidate(
        self,
        request,
        current_state,
        goal,
        non_terminal_scale,
    ):
        criteria_type = self._tool_pose_criteria_type
        try:
            try:
                criteria = criteria_type.linear_motion(
                    axis=request.axis,
                    non_terminal_scale=non_terminal_scale,
                    project_distance_to_goal=request.reference_frame == "tool",
                )
                self.motion_planner.update_tool_pose_criteria(
                    {self.tool_frame: criteria}
                )
            except Exception as error:
                raise _error(
                    "cuRobo axis-linear criteria update failed",
                    stage="sdk",
                    code="CUROBO_CRITERIA_UPDATE_FAILED",
                    status=500,
                    details={"arm": self.arm},
                ) from error
            return self._plan_goal_native(request, current_state, goal)
        finally:
            try:
                self.motion_planner.update_tool_pose_criteria(
                    {self.tool_frame: criteria_type()}
                )
            except Exception as error:
                raise _error(
                    "cuRobo default tool-pose criteria restore failed",
                    stage="sdk",
                    code="CUROBO_CRITERIA_RESET_FAILED",
                    status=500,
                    details={"arm": self.arm},
                ) from error
    # ========================================
    # MODIFICATION END: isolate strict-line candidate planning
    # ========================================

    # ========================================
    # MODIFICATION START: search bounded validated strict-line candidates
    # ========================================
    def plan_axis_linear_native(self, request):
        current_state = self._state(request.start_joints_deg)
        start_position, start_quaternion = self._forward_tool_pose(current_state)
        start_rotation = quaternion_wxyz_to_matrix(start_quaternion)
        axis_index = {"x": 0, "y": 1, "z": 2}[request.axis]
        direction = np.eye(3)[axis_index]
        if request.reference_frame == "tool":
            direction = start_rotation @ direction
        goal = self._goal(
            start_position + request.distance_m * direction,
            start_quaternion,
        )

        criteria_type = self._tool_pose_criteria_type
        if criteria_type is None:
            raise _error(
                "cuRobo tool-pose criteria type is unavailable",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
            )

        attempts = []
        validation_failures = []
        planning_failure = None
        for scale in AXIS_LINEAR_NON_TERMINAL_SCALES:
            try:
                trajectory = self._plan_axis_linear_candidate(
                    request,
                    current_state,
                    goal,
                    scale,
                )
            except CuroboToolError as error:
                if error.code not in _AXIS_LINEAR_RETRYABLE_CODES:
                    raise
                planning_failure = error
                attempts.append(
                    {
                        "scale": scale,
                        "outcome": "planning_failed",
                        "code": error.code,
                    }
                )
                continue

            positions, quaternions = self._trajectory_tool_poses(trajectory)
            rotations = np.asarray(
                [quaternion_wxyz_to_matrix(value) for value in quaternions]
            )
            try:
                validate_axis_linear_poses(
                    positions_m=positions,
                    rotations=rotations,
                    start_position_m=start_position,
                    start_rotation=start_rotation,
                    reference_frame=request.reference_frame,
                    axis=request.axis,
                    distance_m=request.distance_m,
                )
            except AxisLinearValidationError as error:
                validation_failures.append(error)
                attempts.append(
                    {
                        "scale": scale,
                        "outcome": "constraint_failed",
                        **error.to_details(),
                    }
                )
                continue
            except ValueError as error:
                raise _error(
                    "cuRobo trajectory validation data is invalid",
                    stage="validation",
                    code="CUROBO_INVALID_TRAJECTORY",
                    status=500,
                    details={"arm": self.arm, "reason": str(error)},
                ) from error
            return trajectory

        if validation_failures:
            best = min(
                validation_failures,
                key=lambda error: error.normalized_excess,
            )
            raise _error(
                "cuRobo trajectory violates the axis-linear constraint",
                stage="validation",
                code="CUROBO_LINEAR_CONSTRAINT_FAILED",
                status=422,
                details={
                    "arm": self.arm,
                    "reference_frame": request.reference_frame,
                    "axis": request.axis,
                    "distance_m": request.distance_m,
                    **best.to_details(),
                    "attempts": attempts,
                },
            ) from best
        if planning_failure is not None:
            raise planning_failure
        raise _error(
            "cuRobo axis-linear candidate search returned no result",
            stage="trajopt",
            code="CUROBO_PLANNING_FAILED",
            status=422,
            details={"arm": self.arm, "attempts": attempts},
        )
    # ========================================
    # MODIFICATION END: search bounded validated strict-line candidates
    # ========================================

    def solve_ik_batch_native(self, request):
        """One public IK solve; candidates are independent configurations, not a path."""
        count = len(request.goal_matrices_world)
        matrices = np.asarray(request.goal_matrices_world, dtype=np.float64)
        target = self._pose_type(
            position=self._torch.as_tensor(matrices[:, :3, 3], device=self._device, dtype=self._dtype),
            quaternion=self._torch.as_tensor(np.stack([
                _rotation_matrix_to_quaternion_wxyz(matrix[:3, :3]) for matrix in matrices
            ]), device=self._device, dtype=self._dtype),
        )
        goal = self._goal_tool_pose_type.from_poses(
            {self.tool_frame: target}, ordered_tool_frames=[self.tool_frame], num_goalset=1,
        )
        started = time.monotonic()
        # Use the public IK seed generation. A shared start state is not a path
        # constraint here; the SDK pads goal batches to its configured capacity.
        seed_count = self.motion_planner.config.num_seeds
        result = self.motion_planner.solve_pose(goal, return_seeds=seed_count)
        success = _success_matrix(result)
        if success.shape != (count, seed_count):
            raise _error("IK batch success shape mismatch", stage="validation", code="CUROBO_INVALID_IK_RESULT", status=500)
        position_error = _metric_matrix(result, "position_error", success.shape, code="CUROBO_INVALID_IK_RESULT")
        rotation_error = _metric_matrix(result, "rotation_error", success.shape, code="CUROBO_INVALID_IK_RESULT")
        solutions = _tensor_to_numpy(result.solution, message="IK batch solution is invalid", code="CUROBO_INVALID_IK_RESULT")
        if solutions.shape != (count, seed_count, 7) or not np.isfinite(solutions).all():
            raise _error("IK batch solution shape mismatch", stage="validation", code="CUROBO_INVALID_IK_RESULT", status=500)
        # Preserve all already-optimized seeds until the independent hard-clearance
        # check finishes; the SDK's first-ranked seed need not pass this check.
        passed = success.copy()
        candidate_indices, seed_indices = np.nonzero(passed)
        if len(candidate_indices):
            positions = self._torch.as_tensor(solutions[candidate_indices, seed_indices], device=self._device, dtype=self._dtype)
            active_cost, peer_cost = self._motion_world_clearance_costs(positions)
            passed[candidate_indices, seed_indices] &= np.all(active_cost <= 0.0, axis=1) & np.all(peer_cost <= 0.0, axis=1)
        reachable = passed.any(axis=1)
        # argmax selects the first passing seed, or seed zero for failure metrics.
        selected_seeds = passed.argmax(axis=1)
        return NativeIkBatchResult(
            tuple(NativeIkResult(
                reachable=bool(reachable[index]),
                joints_rad=tuple(solutions[index, selected_seeds[index]]) if reachable[index] else None,
                joint_names=self.joint_names,
                position_error_m=float(position_error[index, selected_seeds[index]]),
                rotation_error_rad=float(rotation_error[index, selected_seeds[index]]),
            ) for index in range(count)),
            request.scene_token, time.monotonic() - started,
        )

    def solve_ik_native(self, request):
        current_state, goal = self._state_and_goal(
            request.current_joints_deg,
            request.goal_position_m,
            request.goal_quaternion_wxyz,
        )
        try:
            seed_count = int(
                getattr(
                    self.motion_planner.trajopt_solver.config,
                    "num_seeds",
                    1,
                )
            )
        except (AttributeError, TypeError, ValueError):
            seed_count = 1
        seed_count = max(1, seed_count)

        solve_pose = getattr(
            getattr(self.motion_planner, "ik_solver", None),
            "solve_pose",
            None,
        )
        if not callable(solve_pose):
            raise _error(
                "cuRobo IK solver is unavailable",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
            )
        try:
            result = solve_pose(
                goal,
                current_state=current_state,
                return_seeds=seed_count,
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo IK solve failed",
                stage="ik",
                code="CUROBO_IK_FAILED",
                status=422,
                details={"arm": self.arm},
            ) from error

        if result is None:
            raise _error(
                "cuRobo IK solver returned no result",
                stage="validation",
                code="CUROBO_INVALID_RESULT",
                status=500,
            )
        success = _success_matrix(result)
        position_errors = _metric_matrix(
            result,
            "position_error",
            success.shape,
            code="CUROBO_INVALID_IK_RESULT",
        )
        rotation_errors = _metric_matrix(
            result,
            "rotation_error",
            success.shape,
            code="CUROBO_INVALID_IK_RESULT",
        )

        successful_indices = np.argwhere(success)
        if successful_indices.size == 0:
            combined_errors = position_errors + rotation_errors
            best_flat_index = int(np.argmin(combined_errors))
            batch_index, seed_index = np.unravel_index(
                best_flat_index,
                success.shape,
            )
            return NativeIkResult(
                reachable=False,
                joints_rad=None,
                joint_names=self.joint_names,
                position_error_m=float(
                    position_errors[batch_index, seed_index]
                ),
                rotation_error_rad=float(
                    rotation_errors[batch_index, seed_index]
                ),
            )

        batch_index, seed_index = (
            int(successful_indices[0, 0]),
            int(successful_indices[0, 1]),
        )
        raw_solution = getattr(result, "solution", None)
        solution = _tensor_to_numpy(
            raw_solution,
            message="cuRobo returned invalid IK solutions",
            code="CUROBO_INVALID_IK_RESULT",
        )
        try:
            solution = np.asarray(solution, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise _error(
                "cuRobo returned invalid IK solutions",
                stage="validation",
                code="CUROBO_INVALID_IK_RESULT",
                status=500,
            ) from error
        expected_shape = (
            success.shape[0],
            success.shape[1],
            POINT_DIMENSION,
        )
        if solution.shape != expected_shape:
            raise _error(
                "cuRobo IK solution shape does not match success",
                stage="validation",
                code="CUROBO_INVALID_IK_RESULT",
                status=500,
            )
        selected_solution = solution[batch_index, seed_index]
        if not np.isfinite(selected_solution).all():
            raise _error(
                "cuRobo IK solution contains non-finite values",
                stage="validation",
                code="CUROBO_INVALID_IK_RESULT",
                status=500,
            )

        try:
            return NativeIkResult(
                reachable=True,
                joints_rad=tuple(
                    float(value) for value in selected_solution
                ),
                joint_names=self.joint_names,
                position_error_m=float(
                    position_errors[batch_index, seed_index]
                ),
                rotation_error_rad=float(
                    rotation_errors[batch_index, seed_index]
                ),
            )
        except ValueError as error:
            raise _error(
                "cuRobo returned invalid IK data",
                stage="validation",
                code="CUROBO_INVALID_IK_RESULT",
                status=500,
            ) from error


class _NativeDualCuroboBackend(_NativeCuroboBackend):
    """One official MotionPlanner with both TCPs and fourteen active joints."""

    def _ordered_dual_joints(self, joints):
        from .dual_contracts import JOINT_NAMES
        values = {name: value for arm in ("A", "B")
                  for name, value in zip(JOINT_NAMES[arm], joints[arm])}
        if set(self.joint_names) != set(values):
            raise _error("dual planner joint names do not match the robot", stage="config",
                         code="CUROBO_INVALID_CONFIG", status=500)
        return [values[name] for name in self.joint_names]

    def _dual_goal(self, matrices):
        from scipy.spatial.transform import Rotation
        poses = {}
        for arm, frame in (("A", "TCP_Link_L"), ("B", "TCP_Link_R")):
            matrix = np.asarray(matrices[arm], dtype=np.float64)
            xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            poses[frame] = self._pose_type(
                position=self._torch.as_tensor(matrix[:3, 3] / 1000.,
                    device=self._device, dtype=self._dtype).unsqueeze(0),
                quaternion=self._torch.as_tensor(xyzw[[3, 0, 1, 2]],
                    device=self._device, dtype=self._dtype).unsqueeze(0),
            )
        if set(self.tool_frames) != set(poses):
            raise _error("dual planner must expose both TCP frames", stage="config",
                         code="CUROBO_INVALID_CONFIG", status=500)
        return self._goal_tool_pose_type.from_poses(
            poses, ordered_tool_frames=list(self.tool_frames), num_goalset=1,
        )

    def validate_dual_native(self, request):
        from types import SimpleNamespace
        combined = [self._ordered_dual_joints({"A": a, "B": b})
                    for a, b in zip(request.points_deg["A"], request.points_deg["B"])]
        # The inherited checker queries all fourteen joints together, including
        # inter-arm self collision and clearance of every robot sphere.
        return self.validate_trajectory_native(SimpleNamespace(points_deg=combined))

    def plan_dual_native(self, request):
        from .dual_contracts import JOINT_NAMES, DualValidationRequest
        current = self._state(self._ordered_dual_joints(request.start_joints_deg))
        if request.goal_joints_deg is not None:
            goal = self._state(self._ordered_dual_joints(request.goal_joints_deg))
            result = self.motion_planner.plan_cspace(
                goal, current, max_attempts=request.max_attempts, enable_graph_attempt=1,
            )
        else:
            result = self.motion_planner.plan_pose(
                self._dual_goal(request.goal_transforms_world), current,
                max_attempts=request.max_attempts, enable_graph_attempt=1,
            )
        if result is None or not bool(np.all(_success_matrix(result))):
            raise _error("cuRobo found no collision-free dual-arm trajectory",
                         stage="trajopt", code="CUROBO_TRAJOPT_FAILED", status=422)
        plan = result.get_interpolated_plan()
        positions, names = _trajectory_values(plan, self.joint_names)
        command = np.asarray(_hermite_command_positions(
            plan, names, positions, request.trajectory_point_scale,
        ))
        start = dict(zip(self.joint_names, self._ordered_dual_joints(request.start_joints_deg)))
        if np.max(np.abs(np.rad2deg(positions[0]) - [start[name] for name in names])) > .001:
            raise _error("dual trajectory start does not match request", stage="validation",
                         code="CUROBO_INVALID_TRAJECTORY", status=500)
        if request.goal_joints_deg is not None:
            target = dict(zip(self.joint_names, self._ordered_dual_joints(request.goal_joints_deg)))
            if np.max(np.abs(np.rad2deg(positions[-1]) - [target[name] for name in names])) > .001:
                raise _error("dual trajectory end does not match request", stage="validation",
                             code="CUROBO_INVALID_TRAJECTORY", status=500)
        raw_paths, command_paths = {}, {}
        for arm in ("A", "B"):
            columns = [names.index(name) for name in JOINT_NAMES[arm]]
            raw_paths[arm] = np.rad2deg(positions[:, columns]).tolist()
            command_paths[arm] = np.rad2deg(command[:, columns]).tolist()
        checked = self.validate_dual_native(DualValidationRequest.from_mapping({
            "points_deg": command_paths,
            "trajectory_collision_clearance_m": request.trajectory_collision_clearance_m,
            "optimizer_collision_activation_distance_m": request.optimizer_collision_activation_distance_m,
        }))
        return {"ok": True, "planner": "curobo_dual", "trajectories": command_paths,
                "raw_trajectories": raw_paths, "joint_names": {a: list(JOINT_NAMES[a]) for a in ("A", "B")},
                "sample_dt_s": 1. / request.freq_hz,
                "raw_sample_dt_s": _planner_interpolation_dt(self.motion_planner),
                "trajectory_point_scale": request.trajectory_point_scale,
                "time_parameterization": "quintic_hermite",
                "planning_time_s": _planning_time(result), "checked_point_count": checked,
                "point_count": len(command)}


class CuroboRuntime:
    """Own lazy A/B backends and serialize every CUDA operation."""

    def __init__(
        self,
        *,
        scene="pickplace",
        planner_factory=None,
        batch_planner_factory=None,
        warmup=True,
        robot_config_path=DEFAULT_ROBOT_CONFIG_PATH,
        urdf_path=None,
        asset_root_path=None,
        curobo_source="not-loaded",
        curobo_version="not-loaded",
        gpu_available=False,
        scene_state_provider: Callable[[], Mapping | None] | None = None,
    ):
        try:
            scene_model = build_curobo_scene_model(scene)
            collision_cache = collision_cache_for_scene(scene_model)
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "invalid cuRobo scene configuration",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
            ) from error

        self.scene = scene
        self._scene_model = scene_model
        self._collision_cache = collision_cache
        self._excluded_collision_links = ()
        self._planner_factory = planner_factory
        self._batch_planner_factory = batch_planner_factory
        self._warmup = bool(warmup)
        if (
            scene_state_provider is not None
            and not callable(scene_state_provider)
        ):
            raise _error(
                "scene_state_provider must be callable",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
            )
        self._scene_state_provider = scene_state_provider
        self._backends = {}
        self._dual_backends = {}
        # Compound grasp/place planners stay isolated from ordinary pose/IK
        # planners, but are warmed once per arm instead of once per request.
        self._compound_backends = OrderedDict()
        self._batch_backends = OrderedDict()
        self._ik_batch_backends = OrderedDict()
        self._robot_segmenter = None
        self._fixed_esdf_store = None
        # ========================================
        # MODIFICATION START: cache each arm's applied scene snapshot
        # ========================================
        self._scene_snapshot_key_by_arm = {}
        # ========================================
        # MODIFICATION END: cache each arm's applied scene snapshot
        # ========================================
        # ========================================
        # MODIFICATION START: cache installed peer collision snapshots
        # ========================================
        self._peer_collision_snapshot_by_arm = {}
        # ========================================
        # MODIFICATION END: cache installed peer collision snapshots
        # ========================================
        self._robot_config_path = Path(
            robot_config_path
        ).expanduser().resolve()
        self._urdf_path = (
            None
            if urdf_path is None
            else Path(urdf_path).expanduser().resolve()
        )
        self._asset_root_path = (
            None
            if asset_root_path is None
            else Path(asset_root_path).expanduser().resolve()
        )
        self._curobo_source = str(curobo_source)
        self._curobo_version = str(curobo_version)
        self._gpu_available = bool(gpu_available)

    @classmethod
    def from_backends(cls, backends, **kwargs):
        runtime = cls(
            planner_factory=lambda arm: backends[arm],
            warmup=False,
            **kwargs,
        )
        try:
            items = dict(backends).items()
        except (TypeError, ValueError) as error:
            raise _error(
                "backends must map arm names to native backends",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
            ) from error
        for raw_arm, backend in items:
            arm = runtime._validated_arm(raw_arm)
            if arm in runtime._backends:
                raise _error(
                    f"backend for arm {arm} is configured twice",
                    stage="config",
                    code="CUROBO_INVALID_CONFIG",
                    status=500,
                )
            runtime._backends[arm] = backend
        return runtime

    @property
    def gpu_lock(self):
        return _CUDA_LOCK

    @staticmethod
    def _validated_arm(arm):
        if isinstance(arm, str):
            normalized = arm.strip().upper()
        else:
            normalized = ""
        if normalized not in {"A", "B"}:
            raise _error(
                "arm must be 'A' or 'B'",
                stage="input",
                code="CUROBO_INVALID_ARM",
                status=400,
            )
        return normalized

    def _create_real_backend_unlocked(
        self,
        arm,
        batch_size=None,
        compound_planner=False,
        trajectory_collision_clearance_m=TRAJECTORY_COLLISION_CLEARANCE_M,
        optimizer_collision_activation_distance_m=OPTIMIZER_COLLISION_CLEARANCE_M,
        ik_collision_activation_distance_m=None,
        ik_only=False,
    ):
        import curobo
        import torch
        import yaml
        from curobo.collision_checking import (
            RobotCollisionChecker,
            RobotCollisionCheckerCfg,
        )
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.scene import Cuboid
        # ========================================
        # MODIFICATION START: support cuRobo v0.8.0 ToolPoseCriteria location
        # ========================================
        from curobo.types import GoalToolPose, JointState, Pose

        if ik_only:
            planner_type = InverseKinematics
            planner_config_type = InverseKinematicsCfg
        elif batch_size is None:
            if compound_planner:
                from .grasp_planner import GraspMotionPlanner
                planner_type = GraspMotionPlanner
            else:
                planner_type = MotionPlanner
            planner_config_type = MotionPlannerCfg
        else:
            from curobo.batch_motion_planner import (
                BatchMotionPlanner,
                MotionPlannerCfg as BatchMotionPlannerCfg,
            )

            from .grasp_planner import GraspBatchMotionPlanner
            planner_type = GraspBatchMotionPlanner
            planner_config_type = BatchMotionPlannerCfg

        ToolPoseCriteria = _load_tool_pose_criteria_type()
        # ========================================
        # MODIFICATION END: support cuRobo v0.8.0 ToolPoseCriteria location
        # ========================================

        if not self._robot_config_path.is_file():
            raise _error(
                "cuRobo robot configuration is missing",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
                details={"path": str(self._robot_config_path)},
            )
        if self._urdf_path is None or not self._urdf_path.is_file():
            raise _error(
                "the model capability did not provide a valid cuRobo URDF",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
                details={"path": str(self._urdf_path)},
            )
        if (
            self._asset_root_path is None
            or not self._asset_root_path.is_dir()
        ):
            raise _error(
                "the model capability did not provide a valid asset root",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
                details={"path": str(self._asset_root_path)},
            )

        with self._robot_config_path.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            combined_config = yaml.safe_load(config_file)
        # ========================================
        # MODIFICATION START: retain and reference the frozen peer arm
        # ========================================
        if arm == "AB":
            robot_config = create_dual_arm_robot_config(
                combined_config, urdf_path=self._urdf_path,
                asset_root_path=self._asset_root_path,
            )
            active_collision_links = list(robot_config["robot_cfg"]["kinematics"]["collision_link_names"])
            peer_collision_links = []
        else:
            peer_arm = "B" if arm == "A" else "A"
            robot_config = create_frozen_peer_arm_robot_config(
                combined_config,
                arm,
                urdf_path=self._urdf_path,
                asset_root_path=self._asset_root_path,
            )
            robot_config = _exclude_collision_links(
                robot_config,
                self._excluded_collision_links,
            )
            configured_collision_links = set(
                robot_config["robot_cfg"]["kinematics"][
                    "collision_link_names"
                ]
            )
            active_collision_links = [
                link_name
                for link_name in collision_links_for_arm(
                    combined_config,
                    arm,
                    include_shared=True,
                )
                if link_name in configured_collision_links
            ]
            peer_collision_links = [
                link_name
                for link_name in collision_links_for_arm(
                    combined_config,
                    peer_arm,
                )
                if link_name in configured_collision_links
            ]
        config_kwargs = {"robot": robot_config}
        if self._scene_model is not None:
            config_kwargs["scene_model"] = {key: value for key, value in self._scene_model.items() if key != "voxel"}
        if self._collision_cache is not None:
            config_kwargs["collision_cache"] = self._collision_cache

        # ========================================
        # MODIFICATION START: separate optimizer clearance from graph collision truth
        # ========================================
        config_kwargs["optimizer_collision_activation_distance"] = (
            optimizer_collision_activation_distance_m
        )
        config_kwargs["graph_planner_rollout"] = (
            _graph_planner_rollout_config()
        )
        if compound_planner:
            config_kwargs["trajopt_optimizer_configs"] = [_grasp_optimizer_config()]
        if batch_size is not None:
            config_kwargs["max_batch_size"] = int(batch_size)
            config_kwargs["multi_env"] = False
        # ========================================
        # MODIFICATION END: separate optimizer clearance from graph collision truth
        # ========================================
        if ik_only:
            config_kwargs.pop("graph_planner_rollout")
            config_kwargs.setdefault("scene_model", {"cuboid": {}})
            config_kwargs["optimizer_collision_activation_distance"] = trajectory_collision_clearance_m
            planner_config = InverseKinematicsCfg.create(**config_kwargs, use_cuda_graph=True)
        else:
            planner_config = _create_live_world_planner_config(
                planner_config_type,
                ik_config_type=InverseKinematicsCfg,
                ik_collision_activation_distance_m=ik_collision_activation_distance_m,
                **config_kwargs,
            )
        motion_planner = planner_type(planner_config)
        motion_clearance_checker = _create_world_clearance_checker(
            checker_config_type=RobotCollisionCheckerCfg,
            checker_type=RobotCollisionChecker,
            robot_config=robot_config,
            motion_planner=motion_planner,
            activation_distance_m=trajectory_collision_clearance_m,
            gripper_envelope_arm=(
                arm if compound_planner or (batch_size is not None and not ik_only)
                else None
            ),
            urdf_path=self._urdf_path,
        )

        if arm == "AB":
            return _NativeDualCuroboBackend(
                arm="AB", motion_planner=motion_planner, torch_module=torch,
                joint_state_type=JointState, pose_type=Pose,
                goal_tool_pose_type=GoalToolPose, tool_pose_criteria_type=ToolPoseCriteria,
                motion_clearance_checker=motion_clearance_checker,
                active_collision_links=active_collision_links, peer_collision_links=(),
                warmup_joint_space=True,
            )

        peer_reference_config = create_frozen_peer_arm_robot_config(
            combined_config,
            arm,
            urdf_path=self._urdf_path,
            asset_root_path=self._asset_root_path,
        )
        peer_reference_config = _exclude_collision_links(
            peer_reference_config,
            self._excluded_collision_links,
        )
        peer_reference_kinematics = Kinematics(
            KinematicsCfg.from_data_dict(
                peer_reference_config,
                tool_frames=peer_collision_links,
                device_cfg=motion_planner.device_cfg,
            )
        )
        peer_zero_position = torch.zeros(
            (
                1,
                len(peer_reference_kinematics.joint_names),
            ),
            device=motion_planner.device_cfg.device,
            dtype=motion_planner.device_cfg.dtype,
        )
        peer_zero_poses = peer_reference_kinematics.get_link_poses(
            peer_zero_position,
            peer_collision_links,
        )
        peer_zero_matrices = Pose(
            position=peer_zero_poses.position.reshape(-1, 3),
            quaternion=peer_zero_poses.quaternion.reshape(-1, 4),
        ).get_matrix()
        peer_baseline_inverse_by_link = {
            link_name: torch.linalg.inv(
                peer_zero_matrices[index].to(dtype=torch.float64)
            ).detach().clone()
            for index, link_name in enumerate(peer_collision_links)
        }
        # ========================================
        # MODIFICATION END: retain and reference the frozen peer arm
        # ========================================

        module_path = Path(curobo.__file__).resolve()
        self._curobo_source = str(module_path.parents[1])
        self._curobo_version = str(
            getattr(curobo, "__version__", "unknown")
        )
        self._gpu_available = bool(torch.cuda.is_available())

        peer_kinematics = None
        if ik_only:
            # IK needs only FK of the stationary arm, not another MotionPlanner.
            peer_config = create_frozen_peer_arm_robot_config(
                combined_config, peer_arm, urdf_path=self._urdf_path, asset_root_path=self._asset_root_path,
            )
            peer_config = _exclude_collision_links(peer_config, self._excluded_collision_links)
            peer_kinematics = Kinematics(KinematicsCfg.from_data_dict(peer_config, device_cfg=motion_planner.device_cfg))

        return _NativeCuroboBackend(
            arm=arm,
            motion_planner=motion_planner,
            peer_kinematics=peer_kinematics,
            torch_module=torch,
            joint_state_type=JointState,
            pose_type=Pose,
            obstacle_cuboid_type=Cuboid,
            goal_tool_pose_type=GoalToolPose,
            tool_pose_criteria_type=ToolPoseCriteria,
            motion_clearance_checker=motion_clearance_checker,
            active_collision_links=active_collision_links,
            peer_collision_links=peer_collision_links,
            peer_baseline_inverse_by_link=(
                peer_baseline_inverse_by_link
            ),
            warmup_iterations=1 if batch_size is not None else 3,
            warmup_joint_space=batch_size is None and not compound_planner,
        )

    def _create_backend_unlocked(self, arm):
        if self._planner_factory is None:
            return self._create_real_backend_unlocked(arm)
        return self._planner_factory(arm)

    def _backend_unlocked(self, arm):
        arm = self._validated_arm(arm)
        cached = self._backends.get(arm)
        if cached is not None:
            return cached
        try:
            candidate = self._create_backend_unlocked(arm)
            if self._warmup:
                warmup = getattr(candidate, "warmup", None)
                if not callable(warmup):
                    raise _error(
                        "cuRobo backend does not support warmup",
                        stage="config",
                        code="CUROBO_INVALID_CONFIG",
                        status=500,
                    )
                warmup()
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo backend construction failed",
                stage="config",
                code="CUROBO_CONFIG_FAILED",
                status=500,
                details={"arm": arm},
            ) from error
        self._backends[arm] = candidate
        return candidate

    def warmup(self, lock_timeout_s=240.0):
        """Warm all startup profiles; the timeout only bounds acquiring the CUDA lock."""

        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            for arm in ("A", "B"):
                for profile in ("ordinary", "grasp/place", "grasp-batch-2"):
                    started = time.monotonic()
                    LOGGER.info("cuRobo startup warmup: arm=%s profile=%s", arm, profile)
                    if profile == "ordinary":
                        self._backend_unlocked(arm)
                    elif profile == "grasp/place":
                        self._compound_backend_unlocked(arm)
                    else:
                        self._batch_backend_unlocked(arm, 2)
                    LOGGER.info(
                        "cuRobo startup warmup complete: arm=%s profile=%s wall_s=%.3f",
                        arm, profile, time.monotonic() - started,
                    )
            started = time.monotonic()
            LOGGER.info("cuRobo startup warmup: arm=AB profile=dual")
            self._dual_backend_unlocked()
            LOGGER.info(
                "cuRobo startup warmup complete: arm=AB profile=dual wall_s=%.3f",
                time.monotonic() - started,
            )
        finally:
            _CUDA_LOCK.release()

    def _destroy_compound_backend_unlocked(self, *, arm, backend):
        """Destroy one compound backend with a stable error boundary."""

        destroy = getattr(backend, "destroy", None)
        if not callable(destroy):
            return
        try:
            destroy()
        except Exception as error:
            raise _error(
                "cuRobo compound backend cleanup failed",
                stage="cleanup",
                code="CUROBO_BACKEND_CLEANUP_FAILED",
                status=500,
                details={"arm": arm},
            ) from error

    def _create_compound_backend_unlocked(
        self,
        arm,
        trajectory_collision_clearance_m,
        optimizer_collision_activation_distance_m,
    ):
        if self._planner_factory is None:
            return self._create_real_backend_unlocked(
                arm,
                compound_planner=True,
                ik_collision_activation_distance_m=trajectory_collision_clearance_m,
                trajectory_collision_clearance_m=(
                    trajectory_collision_clearance_m
                ),
                optimizer_collision_activation_distance_m=(
                    optimizer_collision_activation_distance_m
                ),
            )
        return _call_profile_factory(
            self._planner_factory,
            (arm,),
            (
                arm,
                trajectory_collision_clearance_m,
                optimizer_collision_activation_distance_m,
            ),
        )

    def _compound_backend_unlocked(
        self,
        arm,
        trajectory_collision_clearance_m=TRAJECTORY_COLLISION_CLEARANCE_M,
        optimizer_collision_activation_distance_m=OPTIMIZER_COLLISION_CLEARANCE_M,
    ):
        """Return one profile-exact compound backend isolated from pose and IK."""

        arm = self._validated_arm(arm)
        key = (
            arm,
            trajectory_collision_clearance_m,
            optimizer_collision_activation_distance_m,
        )
        cached = self._compound_backends.get(key)
        if cached is not None:
            self._compound_backends.move_to_end(key)
            return cached
        candidate = None
        try:
            candidate = self._create_compound_backend_unlocked(
                arm,
                trajectory_collision_clearance_m,
                optimizer_collision_activation_distance_m,
            )
            if any(
                candidate is backend
                for backend in (
                    *self._backends.values(),
                    *self._compound_backends.values(),
                )
            ):
                raise _error(
                    "cuRobo factory returned a cached backend",
                    stage="config",
                    code="CUROBO_INVALID_CONFIG",
                    status=500,
                    details={"arm": arm},
                )
            if self._warmup:
                warmup = getattr(candidate, "warmup", None)
                if not callable(warmup):
                    raise _error(
                        "cuRobo backend does not support warmup",
                        stage="config",
                        code="CUROBO_INVALID_CONFIG",
                        status=500,
                        details={"arm": arm},
                    )
                warmup()
            self._compound_backends[key] = candidate
            arm_keys = [
                cached_key
                for cached_key in self._compound_backends
                if cached_key[0] == arm
            ]
            if len(arm_keys) > 2:
                evicted_key = arm_keys[0]
                evicted = self._compound_backends.pop(evicted_key)
                self._destroy_compound_backend_unlocked(
                    arm=arm,
                    backend=evicted,
                )
            return candidate
        except Exception as error:
            is_cached = candidate is not None and any(
                candidate is backend
                for backend in (
                    *self._backends.values(),
                    *self._compound_backends.values(),
                )
            )
            if candidate is not None and not is_cached:
                self._destroy_compound_backend_unlocked(
                    arm=arm,
                    backend=candidate,
                )
            if isinstance(error, CuroboToolError):
                raise
            raise _error(
                "cuRobo compound backend construction failed",
                stage="config",
                code="CUROBO_CONFIG_FAILED",
                status=500,
                details={"arm": arm},
            ) from error

    def _discard_compound_backend_unlocked(
        self,
        *,
        arm,
        trajectory_collision_clearance_m,
        optimizer_collision_activation_distance_m,
        backend,
    ):
        """Evict a compound backend after an unsuccessful request."""

        key = (
            arm,
            trajectory_collision_clearance_m,
            optimizer_collision_activation_distance_m,
        )
        if self._compound_backends.get(key) is backend:
            self._compound_backends.pop(key, None)
        self._destroy_compound_backend_unlocked(arm=arm, backend=backend)

    def _create_batch_backend_unlocked(
        self,
        arm,
        batch_size,
        trajectory_collision_clearance_m,
        optimizer_collision_activation_distance_m,
    ):
        if self._batch_planner_factory is not None:
            return _call_profile_factory(
                self._batch_planner_factory,
                (arm, batch_size),
                (
                    arm,
                    batch_size,
                    trajectory_collision_clearance_m,
                    optimizer_collision_activation_distance_m,
                ),
            )
        return self._create_real_backend_unlocked(
            arm,
            batch_size=batch_size,
            ik_collision_activation_distance_m=trajectory_collision_clearance_m,
            trajectory_collision_clearance_m=trajectory_collision_clearance_m,
            optimizer_collision_activation_distance_m=(
                optimizer_collision_activation_distance_m
            ),
        )

    def _batch_backend_unlocked(
        self,
        arm,
        batch_size,
        trajectory_collision_clearance_m=TRAJECTORY_COLLISION_CLEARANCE_M,
        optimizer_collision_activation_distance_m=OPTIMIZER_COLLISION_CLEARANCE_M,
    ):
        """Reuse fixed-capacity official batches within a bounded allocation budget."""

        arm = self._validated_arm(arm)
        key = (
            arm,
            int(batch_size),
            trajectory_collision_clearance_m,
            optimizer_collision_activation_distance_m,
        )
        cached = self._batch_backends.get(key)
        if cached is not None:
            self._batch_backends.move_to_end(key)
            return cached
        # A 100-item trajectory backend alone uses tens of GB. Evict before
        # allocation, not after warmup, to avoid retaining several such graphs.
        while self._batch_backends and (
            sum(item[1] for item in self._batch_backends) + batch_size > 100
            or sum(item[0] == arm for item in self._batch_backends) >= 2
        ):
            evicted_key = next(iter(self._batch_backends))
            evicted = self._batch_backends.pop(evicted_key)
            self._destroy_compound_backend_unlocked(arm=evicted_key[0], backend=evicted)
        candidate = None
        try:
            candidate = self._create_batch_backend_unlocked(
                arm,
                batch_size,
                trajectory_collision_clearance_m,
                optimizer_collision_activation_distance_m,
            )
            if any(
                candidate is backend
                for backend in (
                    *self._backends.values(),
                    *self._compound_backends.values(),
                    *self._batch_backends.values(),
                )
            ):
                raise _error(
                    "cuRobo batch factory returned a cached backend",
                    stage="config",
                    code="CUROBO_INVALID_CONFIG",
                    status=500,
                    details={"arm": arm, "batch_size": batch_size},
                )
            if self._warmup:
                warmup = getattr(candidate, "warmup", None)
                if not callable(warmup):
                    raise _error(
                        "cuRobo batch backend does not support warmup",
                        stage="config",
                        code="CUROBO_INVALID_CONFIG",
                        status=500,
                        details={"arm": arm, "batch_size": batch_size},
                    )
                warmup()
            self._batch_backends[key] = candidate
            return candidate
        except Exception as error:
            is_cached = candidate is not None and any(
                candidate is backend
                for backend in (
                    *self._backends.values(),
                    *self._compound_backends.values(),
                    *self._batch_backends.values(),
                )
            )
            if candidate is not None and not is_cached:
                self._destroy_compound_backend_unlocked(arm=arm, backend=candidate)
            if isinstance(error, CuroboToolError):
                raise
            torch_module = sys.modules.get("torch")
            if (
                torch_module is not None
                and _is_cuda_out_of_memory(torch_module, error)
            ):
                raise _error(
                    "cuRobo GPU memory is insufficient for batch backend construction",
                    stage="resource",
                    code="CUROBO_RESOURCE_EXHAUSTED",
                    status=503,
                    retryable=True,
                    details={"arm": arm, "batch_size": batch_size},
                ) from error
            raise _error(
                "cuRobo batch backend construction failed",
                stage="config",
                code="CUROBO_CONFIG_FAILED",
                status=500,
                details={"arm": arm, "batch_size": batch_size},
            ) from error

    def _discard_batch_backend_unlocked(
        self,
        *,
        arm,
        batch_size,
        trajectory_collision_clearance_m,
        optimizer_collision_activation_distance_m,
        backend,
    ):
        key = (
            arm,
            batch_size,
            trajectory_collision_clearance_m,
            optimizer_collision_activation_distance_m,
        )
        if self._batch_backends.get(key) is backend:
            self._batch_backends.pop(key, None)
        self._destroy_compound_backend_unlocked(arm=arm, backend=backend)

    def invalidate_grasp_batch_native(
        self,
        arm,
        batch_size,
        trajectory_collision_clearance_m=TRAJECTORY_COLLISION_CLEARANCE_M,
        optimizer_collision_activation_distance_m=OPTIMIZER_COLLISION_CLEARANCE_M,
        lock_timeout_s=120.0,
    ):
        arm = self._validated_arm(arm)
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
            or batch_size > 100
        ):
            raise _error(
                "batch_size must be an integer from 1 to 100",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        trajectory, optimizer = _collision_clearance_profile(
            trajectory_collision_clearance_m,
            optimizer_collision_activation_distance_m,
        )
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            backend = self._batch_backends.pop(
                (arm, next(size for size in (2, 4, 8, 16, 50, 100) if size >= batch_size), trajectory, optimizer),
                None,
            )
            if backend is None:
                return False
            self._destroy_compound_backend_unlocked(arm=arm, backend=backend)
            return True
        finally:
            _CUDA_LOCK.release()

    # ========================================
    # MODIFICATION START: refresh peer geometry before every native solve
    # ========================================
    def _refresh_peer_collision_unlocked(
        self,
        *,
        arm,
        backend,
        active_joints_deg,
        other_joints_deg,
        cache_snapshot=True,
    ):
        peer_arm = "B" if arm == "A" else "A"
        peer_backend = self._backend_unlocked(peer_arm)
        update_peer = getattr(
            backend,
            "update_peer_collision_model",
            None,
        )
        if not callable(update_peer):
            raise _error(
                "cuRobo backend cannot refresh peer collision geometry",
                stage="config",
                code="CUROBO_INVALID_CONFIG",
                status=500,
                details={"arm": arm, "peer_arm": peer_arm},
            )
        # ========================================
        # MODIFICATION START: reuse an unchanged peer collision snapshot
        # ========================================
        peer_joint_snapshot = tuple(
            float(value) for value in other_joints_deg
        )
        cached_peer_snapshot = self._peer_collision_snapshot_by_arm.get(
            arm
        )
        if cache_snapshot and (
            cached_peer_snapshot is not None
            and cached_peer_snapshot[0] is backend
            and cached_peer_snapshot[1] is peer_backend
            and cached_peer_snapshot[2] == peer_joint_snapshot
        ):
            return False
        update_peer(
            peer_backend,
            active_joints_deg,
            other_joints_deg,
        )
        if cache_snapshot:
            self._peer_collision_snapshot_by_arm[arm] = (
                backend,
                peer_backend,
                peer_joint_snapshot,
            )
        return True
        # ========================================
        # MODIFICATION END: reuse an unchanged peer collision snapshot
        # ========================================
    # ========================================
    # MODIFICATION END: refresh peer geometry before every native solve
    # ========================================

    # ========================================
    # MODIFICATION START: refresh canonical scene inside CUDA serialization
    # ========================================
    def _scene_snapshot_unlocked(self, *, allow_unstable_object=False):
        if self._scene_state_provider is None:
            return None
        try:
            snapshot = self._scene_state_provider()
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo scene snapshot is unavailable",
                stage="scene",
                code="CUROBO_SCENE_SNAPSHOT_FAILED",
                status=500,
            ) from error

        if snapshot is None:
            if self.scene == "none":
                return None
            raise _error(
                "cuRobo scene snapshot is unavailable",
                stage="scene",
                code="CUROBO_SCENE_SNAPSHOT_FAILED",
                status=500,
            )
        if not isinstance(snapshot, Mapping):
            raise _error(
                "cuRobo scene snapshot is invalid",
                stage="scene",
                code="CUROBO_SCENE_SNAPSHOT_FAILED",
                status=500,
            )
        collision_world_source = str(
            snapshot.get("collision_world_source") or "none"
        ).strip().lower()
        if collision_world_source == "none":
            scene_token = snapshot.get("scene_token")
            if scene_token != "none:0":
                raise _error(
                    "cuRobo empty scene snapshot is invalid",
                    stage="scene",
                    code="CUROBO_SCENE_SNAPSHOT_FAILED",
                    status=500,
                )
            return {
                "collision_world_source": "none",
                "scene_token": scene_token,
            }
        if collision_world_source == "esdf":
            status = self._esdf_status_unlocked()
            if not status.ready or not status.scene_token:
                raise _error(
                    "cuRobo ESDF has no integrated camera frame",
                    stage="scene",
                    code="CUROBO_ESDF_NOT_READY",
                    status=409,
                    retryable=True,
                )
            return {
                "collision_world_source": "esdf",
                "scene_token": status.scene_token,
            }
        if collision_world_source != "manual":
            raise _error(
                "collision world source must be none, manual or esdf",
                stage="scene",
                code="CUROBO_INVALID_COLLISION_WORLD_SOURCE",
                status=422,
            )
        config_revision = snapshot.get("config_revision")
        applied_revision = snapshot.get("applied_revision")
        physics_sequence = snapshot.get("physics_sequence")
        scene_token = snapshot.get("scene_token")
        if (
            isinstance(config_revision, bool)
            or not isinstance(config_revision, int)
            or config_revision < 1
            or applied_revision != config_revision
            or isinstance(physics_sequence, bool)
            or not isinstance(physics_sequence, int)
            or physics_sequence < 0
            or not isinstance(scene_token, str)
            or not scene_token
        ):
            raise _error(
                "cuRobo scene snapshot is invalid",
                stage="scene",
                code="CUROBO_SCENE_SNAPSHOT_FAILED",
                status=500,
            )
        if (
            not allow_unstable_object
            and snapshot.get("physics_seen")
            and not snapshot.get("object_stable")
        ):
            raise _error(
                "dynamic obstacle has not settled",
                stage="scene",
                code="SCENE_OBJECT_UNSTABLE",
                status=409,
                retryable=True,
            )
        return {**snapshot, "collision_world_source": "manual"}

    @staticmethod
    def _raise_scene_changed(expected_scene_token, actual_scene_token):
        raise _error(
            "obstacle scene changed during cuRobo operation",
            stage="scene",
            code="CUROBO_SCENE_CHANGED",
            status=409,
            retryable=True,
            details={
                "expected_scene_token": expected_scene_token,
                "actual_scene_token": actual_scene_token,
            },
        )

    def _require_scene_token_unchanged_unlocked(
        self,
        expected_scene_token,
        *,
        allow_unstable_object=False,
    ):
        if expected_scene_token is None:
            return
        snapshot = self._scene_snapshot_unlocked(
            allow_unstable_object=allow_unstable_object
        )
        actual_scene_token = (
            None if snapshot is None else snapshot["scene_token"]
        )
        if actual_scene_token != expected_scene_token:
            self._raise_scene_changed(
                expected_scene_token,
                actual_scene_token,
            )

    def _configure_scene_cache_unlocked(self, scene_model, *rollback_models):
        """Recreate SDK buffers when scene capacity changes, under the CUDA lock."""

        cache = collision_cache_for_scene(scene_model, *rollback_models)
        # Clearing/replacing results does not shrink live CUDA allocations.
        # Scene loading still disables every obstacle absent from the new world.
        previous = self._collision_cache
        if previous is not None:
            for kind in ("mesh", "cuboid"):
                cache[kind] = max(cache[kind], previous[kind])
            cache["voxel"]["layers"] = max(
                cache["voxel"]["layers"], previous["voxel"]["layers"],
            )
            if np.prod(previous["voxel"]["dims"]) >= np.prod(cache["voxel"]["dims"]):
                cache["voxel"]["dims"] = list(previous["voxel"]["dims"])
        if cache != self._collision_cache:
            backends = [
                *self._backends.values(),
                *self._dual_backends.values(),
                *self._compound_backends.values(),
                *self._batch_backends.values(),
                *self._ik_batch_backends.values(),
            ]
            # No cached graph or snapshot identity may outlive its buffers.
            self._backends.clear()
            self._dual_backends.clear()
            self._compound_backends.clear()
            self._batch_backends.clear()
            self._ik_batch_backends.clear()
            self._scene_snapshot_key_by_arm.clear()
            self._peer_collision_snapshot_by_arm.clear()
            destroyed = set()
            errors = []
            for backend in backends:
                if id(backend) in destroyed:
                    continue
                destroyed.add(id(backend))
                destroy = getattr(backend, "destroy", None)
                if callable(destroy):
                    try:
                        destroy()
                    except Exception as error:
                        errors.append(error)
            if errors:
                raise _error(
                    "cuRobo scene capacity cleanup failed",
                    stage="cleanup",
                    code="CUROBO_BACKEND_CLEANUP_FAILED",
                    status=500,
                ) from errors[0]
            self._collision_cache = cache
        self._scene_model = scene_model

    def _prepare_scene_cache_unlocked(self):
        """Bind allocation and subsequent world loading to the same snapshot."""

        snapshot = self._scene_snapshot_unlocked()
        if snapshot is None:
            model = build_curobo_scene_model(self.scene)
        elif snapshot["collision_world_source"] == "esdf":
            model = self._fixed_esdf_store.published_scene_model
        elif snapshot["collision_world_source"] == "none":
            model = {}
        else:
            model = build_curobo_scene_model(self.scene, snapshot=snapshot)
        self._configure_scene_cache_unlocked(model)
        return snapshot

    def _refresh_scene_unlocked(
        self,
        *,
        arm,
        backend,
        cache_snapshot=True,
        allow_unstable_object=False,
        snapshot=None,
    ):
        if snapshot is None:
            snapshot = self._scene_snapshot_unlocked(
                allow_unstable_object=allow_unstable_object
            )
        if snapshot is None:
            return None
        scene_token = snapshot["scene_token"]
        cached_scene_snapshot = self._scene_snapshot_key_by_arm.get(arm)
        if cache_snapshot and (
            cached_scene_snapshot is not None
            and cached_scene_snapshot[0] is backend
            and cached_scene_snapshot[1] == scene_token
        ):
            return scene_token
        snapshot_key = (backend, scene_token)
        try:
            if snapshot["collision_world_source"] == "esdf":
                store = self._fixed_esdf_store
                mapper = None if store is None else store.published_mapper
                voxel_grid = (
                    None
                    if mapper is None
                    else getattr(mapper.sdk_boundary, "scene_voxel_grids", mapper.sdk_boundary.esdf_grid)
                )
                update_voxel = getattr(backend, "update_voxel_grid", None)
                if voxel_grid is None:
                    raise _error(
                        "cuRobo ESDF has no integrated camera frame",
                        stage="scene",
                        code="CUROBO_ESDF_NOT_READY",
                        status=409,
                        retryable=True,
                    )
                fixed_scene_model = (
                    None if store is None else store.published_scene_model
                )
                update_mixed = getattr(backend, "update_voxel_scene", None)
                if fixed_scene_model is not None and not callable(update_mixed):
                    raise _error(
                        "cuRobo backend cannot load semantic ESDF geometry",
                        stage="config",
                        code="CUROBO_INVALID_CONFIG",
                        status=500,
                        details={"arm": arm},
                    )
                if fixed_scene_model is None and not callable(update_voxel):
                    raise _error(
                        "cuRobo backend cannot load an ESDF voxel grid",
                        stage="config",
                        code="CUROBO_INVALID_CONFIG",
                        status=500,
                        details={"arm": arm},
                    )
                if fixed_scene_model is None:
                    update_voxel(voxel_grid)
                else:
                    selection = store.published_selection
                    selected_instance_id = selection.get("selected_instance_id")
                    disabled = (
                        ()
                        if not selected_instance_id
                        else (f"semantic_instance::{selected_instance_id}",)
                    )
                    update_mixed(
                        voxel_grid,
                        fixed_scene_model,
                        disabled_obstacle_names=disabled,
                    )
                scene_model = None
            else:
                scene_model = (
                    {}
                    if snapshot["collision_world_source"] == "none"
                    else build_curobo_scene_model(
                        self.scene,
                        snapshot=snapshot,
                    )
                )
                update_scene = getattr(
                    backend,
                    "update_scene_model",
                    None,
                )
                if not callable(update_scene):
                    raise _error(
                        "cuRobo backend cannot refresh obstacle geometry",
                        stage="config",
                        code="CUROBO_INVALID_CONFIG",
                        status=500,
                        details={"arm": arm},
                    )
                update_scene(scene_model)
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo obstacle scene update failed",
                stage="scene",
                code="CUROBO_SCENE_UPDATE_FAILED",
                status=500,
                details={"arm": arm},
            ) from error
        if scene_model is not None:
            self._scene_model = scene_model
        if cache_snapshot:
            self._scene_snapshot_key_by_arm[arm] = snapshot_key
        return scene_token
    # ========================================
    # MODIFICATION END: refresh canonical scene inside CUDA serialization
    # ========================================

    def _backend_until(self, arm, deadline):
        _acquire_cuda_until(deadline)
        try:
            return self._backend_unlocked(arm)
        finally:
            _CUDA_LOCK.release()

    def backend(self, arm, lock_timeout_s=120.0):
        timeout = _validated_timeout(lock_timeout_s)
        return self._backend_until(
            arm,
            time.monotonic() + timeout,
        )

    def _dual_backend_unlocked(self, request=None):
        key = (TRAJECTORY_COLLISION_CLEARANCE_M, OPTIMIZER_COLLISION_CLEARANCE_M)
        if request is not None:
            key = (request.trajectory_collision_clearance_m,
                   request.optimizer_collision_activation_distance_m)
        backend = self._dual_backends.get(key)
        if backend is not None:
            return backend
        # One resident full-robot planner. Parameter changes replace only this
        # owned backend; ordinary single-arm and mapper caches stay intact.
        for old in self._dual_backends.values():
            old.destroy()
        self._dual_backends.clear()
        backend = self._create_real_backend_unlocked(
            "AB", trajectory_collision_clearance_m=key[0],
            optimizer_collision_activation_distance_m=key[1],
        )
        try:
            if self._warmup:
                backend.warmup()
        except Exception:
            backend.destroy()
            raise
        self._dual_backends[key] = backend
        return backend

    def dual_operation_native(self, request, *, validate=False, lock_timeout_s=120.):
        from .dual_contracts import DualPlanRequest, DualValidationRequest
        expected_type = DualValidationRequest if validate else DualPlanRequest
        if not isinstance(request, expected_type):
            raise _error("invalid dual planning request", stage="validation",
                         code="CUROBO_INVALID_REQUEST", status=422)
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            backend = self._dual_backend_unlocked(request)
            if timeout > 0 and time.monotonic() >= deadline:
                raise _busy_error()
            token = self._refresh_scene_unlocked(arm="AB", backend=backend, snapshot=snapshot)
            if validate:
                if request.expected_scene_token is not None and request.expected_scene_token != token:
                    self._raise_scene_changed(request.expected_scene_token, token)
                checked = backend.validate_dual_native(request)
                result = {"ok": True, "collision_free": True, "checked_point_count": checked,
                          "first_invalid_index": None}
            else:
                result = backend.plan_dual_native(request)
            self._require_scene_token_unchanged_unlocked(token)
            return {**result, "scene_token": token}
        finally:
            _CUDA_LOCK.release()

    def plan_to_pose_native(
        self,
        request,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        if not isinstance(request, PlanToPoseRequest):
            raise _error(
                "request must be a PlanToPoseRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            backend = self._backend_unlocked(request.arm)
            if timeout > 0.0 and time.monotonic() >= deadline:
                raise _busy_error()
            # ========================================
            # MODIFICATION START: bind pose planning to one stable scene
            # ========================================
            scene_token = self._refresh_scene_unlocked(
                snapshot=snapshot,
                arm=request.arm,
                backend=backend,
            )
            self._refresh_peer_collision_unlocked(
                arm=request.arm,
                backend=backend,
                active_joints_deg=request.start_joints_deg,
                other_joints_deg=request.other_joints_deg,
            )
            if timeout > 0.0 and time.monotonic() >= deadline:
                raise _busy_error()
            try:
                result = backend.plan_to_pose_native(request)
                self._require_scene_token_unchanged_unlocked(scene_token)
                return replace(result, scene_token=scene_token)
            except CuroboToolError:
                raise
            except Exception as error:
                raise _error(
                    "cuRobo backend planning failed",
                    stage="sdk",
                    code="CUROBO_BACKEND_FAILED",
                    status=500,
                    details={"arm": request.arm},
                ) from error
            # ========================================
            # MODIFICATION END: bind pose planning to one stable scene
            # ========================================
        finally:
            _CUDA_LOCK.release()

    def plan_to_joint_native(
        self,
        request,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        if not isinstance(request, PlanToJointRequest):
            raise _error(
                "request must be a PlanToJointRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            backend = self._backend_unlocked(request.arm)
            if timeout > 0.0 and time.monotonic() >= deadline:
                raise _busy_error()
            scene_token = self._refresh_scene_unlocked(
                snapshot=snapshot,
                arm=request.arm,
                backend=backend,
            )
            self._refresh_peer_collision_unlocked(
                arm=request.arm,
                backend=backend,
                active_joints_deg=request.start_joints_deg,
                other_joints_deg=request.other_joints_deg,
            )
            if timeout > 0.0 and time.monotonic() >= deadline:
                raise _busy_error()
            try:
                result = backend.plan_to_joint_native(request)
                self._require_scene_token_unchanged_unlocked(scene_token)
                return replace(result, scene_token=scene_token)
            except CuroboToolError:
                raise
            except Exception as error:
                raise _error(
                    "cuRobo backend joint planning failed",
                    stage="sdk",
                    code="CUROBO_BACKEND_FAILED",
                    status=500,
                    details={"arm": request.arm},
                ) from error
        finally:
            _CUDA_LOCK.release()

    def plan_grasp_native(
        self,
        request,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        if not isinstance(request, PlanGraspRequest):
            raise _error(
                "request must be a PlanGraspRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            backend = self._compound_backend_unlocked(
                request.arm,
                request.trajectory_collision_clearance_m,
                request.optimizer_collision_activation_distance_m,
            )
            try:
                if timeout > 0.0 and time.monotonic() >= deadline:
                    raise _busy_error()
                scene_token = self._refresh_scene_unlocked(
                    snapshot=snapshot,
                    arm=request.arm,
                    backend=backend,
                    cache_snapshot=False,
                )
                self._refresh_peer_collision_unlocked(
                    arm=request.arm,
                    backend=backend,
                    active_joints_deg=request.start_joints_deg,
                    other_joints_deg=request.other_joints_deg,
                    cache_snapshot=False,
                )
                if timeout > 0.0 and time.monotonic() >= deadline:
                    raise _busy_error()
                result = backend.plan_grasp_native(request)
                self._require_scene_token_unchanged_unlocked(scene_token)
                return NativeGraspPlan.create(
                    approach=replace(
                        result.approach,
                        scene_token=scene_token,
                    ),
                    grasp=replace(
                        result.grasp,
                        scene_token=scene_token,
                    ),
                    lift=replace(
                        result.lift,
                        scene_token=scene_token,
                    ),
                )
            except CuroboToolError as error:
                if error.code == "CUROBO_SCENE_CHANGED":
                    raise
                self._discard_compound_backend_unlocked(
                    arm=request.arm,
                    trajectory_collision_clearance_m=(
                        request.trajectory_collision_clearance_m
                    ),
                    optimizer_collision_activation_distance_m=(
                        request.optimizer_collision_activation_distance_m
                    ),
                    backend=backend,
                )
                raise
            except Exception as error:
                self._discard_compound_backend_unlocked(
                    arm=request.arm,
                    trajectory_collision_clearance_m=(
                        request.trajectory_collision_clearance_m
                    ),
                    optimizer_collision_activation_distance_m=(
                        request.optimizer_collision_activation_distance_m
                    ),
                    backend=backend,
                )
                raise _error(
                    "cuRobo backend grasp planning failed",
                    stage="sdk",
                    code="CUROBO_BACKEND_FAILED",
                    status=500,
                    details={"arm": request.arm},
                ) from error
        finally:
            _CUDA_LOCK.release()

    def solve_ik_batch_native(self, request, lock_timeout_s=120.0):
        if not isinstance(request, SolveIkBatchRequest):
            raise _error("invalid IK batch request", stage="input", code="CUROBO_INVALID_REQUEST", status=400)
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        capacity = next(size for size in (50, 100, 200) if size >= len(request.goal_matrices_world))
        key = (request.arm, request.trajectory_collision_clearance_m, capacity)
        backend = None
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            backend = self._ik_batch_backends.get(key)
            if backend is None:
                # Bound retained IK graph buffers as well as trajectory buffers.
                while self._ik_batch_backends and (
                    sum(item[2] for item in self._ik_batch_backends) + capacity > 200
                    or sum(item[0] == request.arm for item in self._ik_batch_backends) >= 2
                ):
                    _old_key, old_backend = self._ik_batch_backends.popitem(last=False)
                    old_backend.destroy()
                backend = self._create_real_backend_unlocked(
                    request.arm, batch_size=capacity, ik_only=True,
                    trajectory_collision_clearance_m=request.trajectory_collision_clearance_m,
                )
                self._ik_batch_backends[key] = backend
            self._ik_batch_backends.move_to_end(key)
            scene_token = self._refresh_scene_unlocked(snapshot=snapshot, arm=request.arm, backend=backend, cache_snapshot=False)
            if scene_token != request.scene_token:
                self._raise_scene_changed(request.scene_token, scene_token)
            backend.update_peer_collision_model(None, request.start_joints_deg, request.other_joints_deg)
            if timeout > 0.0 and time.monotonic() >= deadline:
                raise _busy_error()
            result = backend.solve_ik_batch_native(request)
            self._require_scene_token_unchanged_unlocked(scene_token)
            if not isinstance(result, NativeIkBatchResult) or len(result.items) != len(request.goal_matrices_world):
                raise _error("invalid IK batch result", stage="validation", code="CUROBO_INVALID_IK_RESULT", status=500)
            return replace(result, scene_token=scene_token)
        except Exception as error:
            if (backend is not None and self._ik_batch_backends.get(key) is backend
                    and not (isinstance(error, CuroboToolError) and error.code == "CUROBO_SCENE_CHANGED")):
                self._ik_batch_backends.pop(key)
                backend.destroy()
            if isinstance(error, CuroboToolError):
                raise
            if backend is not None and _is_cuda_out_of_memory(backend._torch, error):
                raise _error("cuRobo GPU memory is insufficient for batched IK", stage="resource", code="CUROBO_RESOURCE_EXHAUSTED", status=503, retryable=True) from error
            raise _error("cuRobo batched IK failed", stage="ik", code="CUROBO_IK_FAILED", status=500) from error
        finally:
            _CUDA_LOCK.release()

    def plan_grasp_batch_native(
        self,
        request,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        if not isinstance(request, PlanGraspBatchRequest):
            raise _error(
                "request must be a PlanGraspBatchRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        batch_size = len(request.goal_matrices_world)
        capacity = next(size for size in (2, 4, 8, 16, 50, 100) if size >= batch_size)
        padded_request = replace(
            request,
            goal_matrices_world=(
                request.goal_matrices_world
                + (request.goal_matrices_world[-1],) * (capacity - batch_size)
            ),
        )
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            backend = self._batch_backend_unlocked(
                request.arm,
                capacity,
                request.trajectory_collision_clearance_m,
                request.optimizer_collision_activation_distance_m,
            )
            try:
                if timeout > 0.0 and time.monotonic() >= deadline:
                    raise _busy_error()
                scene_token = self._refresh_scene_unlocked(
                    snapshot=snapshot,
                    arm=request.arm,
                    backend=backend,
                    cache_snapshot=False,
                )
                if scene_token != request.scene_token:
                    self._raise_scene_changed(request.scene_token, scene_token)
                self._refresh_peer_collision_unlocked(
                    arm=request.arm,
                    backend=backend,
                    active_joints_deg=request.start_joints_deg,
                    other_joints_deg=request.other_joints_deg,
                    cache_snapshot=False,
                )
                if timeout > 0.0 and time.monotonic() >= deadline:
                    raise _busy_error()
                result = backend.plan_grasp_batch_native(padded_request)
                self._require_scene_token_unchanged_unlocked(scene_token)
                if (
                    not isinstance(result, NativeGraspBatchPlan)
                    or result.batch_size != capacity
                    or result.batch_capacity != capacity
                ):
                    raise _error(
                        "cuRobo returned an invalid native grasp batch",
                        stage="validation",
                        code="CUROBO_INVALID_RESULT",
                        status=500,
                        details={"arm": request.arm, "batch_size": batch_size},
                    )

                def with_scene(trajectory):
                    return None if trajectory is None else replace(trajectory, scene_token=scene_token)

                items = tuple(
                    replace(
                        item,
                        approach=with_scene(item.approach),
                        grasp=with_scene(item.grasp),
                        lift=with_scene(item.lift),
                    )
                    for item in result.items[:batch_size]
                )
                return NativeGraspBatchPlan.create(
                    items=items,
                    batch_size=batch_size,
                    batch_capacity=capacity,
                    planning_time_s=result.planning_time_s,
                    scene_token=scene_token,
                )
            except CuroboToolError as error:
                if error.code == "CUROBO_SCENE_CHANGED":
                    raise
                self._discard_batch_backend_unlocked(
                    arm=request.arm,
                    batch_size=capacity,
                    trajectory_collision_clearance_m=(
                        request.trajectory_collision_clearance_m
                    ),
                    optimizer_collision_activation_distance_m=(
                        request.optimizer_collision_activation_distance_m
                    ),
                    backend=backend,
                )
                raise
            except Exception as error:
                self._discard_batch_backend_unlocked(
                    arm=request.arm,
                    batch_size=capacity,
                    trajectory_collision_clearance_m=(
                        request.trajectory_collision_clearance_m
                    ),
                    optimizer_collision_activation_distance_m=(
                        request.optimizer_collision_activation_distance_m
                    ),
                    backend=backend,
                )
                raise _error(
                    "cuRobo backend batched grasp planning failed",
                    stage="sdk",
                    code="CUROBO_BACKEND_FAILED",
                    status=500,
                    details={"arm": request.arm, "batch_size": batch_size},
                ) from error
        finally:
            _CUDA_LOCK.release()

    def plan_place_native(
        self,
        request,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        if not isinstance(request, PlanPlaceRequest):
            raise _error(
                "request must be a PlanPlaceRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            backend = self._compound_backend_unlocked(
                request.arm,
                request.trajectory_collision_clearance_m,
                request.optimizer_collision_activation_distance_m,
            )
            try:
                if timeout > 0.0 and time.monotonic() >= deadline:
                    raise _busy_error()
                scene_token = self._refresh_scene_unlocked(
                    snapshot=snapshot,
                    arm=request.arm,
                    backend=backend,
                    cache_snapshot=False,
                )
                self._refresh_peer_collision_unlocked(
                    arm=request.arm,
                    backend=backend,
                    active_joints_deg=request.start_joints_deg,
                    other_joints_deg=request.other_joints_deg,
                    cache_snapshot=False,
                )
                if timeout > 0.0 and time.monotonic() >= deadline:
                    raise _busy_error()
                result = backend.plan_place_native(request)
                self._require_scene_token_unchanged_unlocked(scene_token)
                return NativePlacePlan.create(
                    approach=replace(
                        result.approach,
                        scene_token=scene_token,
                    ),
                    place=replace(
                        result.place,
                        scene_token=scene_token,
                    ),
                    retreat=replace(
                        result.retreat,
                        scene_token=scene_token,
                    ),
                )
            except CuroboToolError as error:
                if error.code == "CUROBO_SCENE_CHANGED":
                    raise
                self._discard_compound_backend_unlocked(
                    arm=request.arm,
                    trajectory_collision_clearance_m=(
                        request.trajectory_collision_clearance_m
                    ),
                    optimizer_collision_activation_distance_m=(
                        request.optimizer_collision_activation_distance_m
                    ),
                    backend=backend,
                )
                raise
            except Exception as error:
                self._discard_compound_backend_unlocked(
                    arm=request.arm,
                    trajectory_collision_clearance_m=(
                        request.trajectory_collision_clearance_m
                    ),
                    optimizer_collision_activation_distance_m=(
                        request.optimizer_collision_activation_distance_m
                    ),
                    backend=backend,
                )
                raise _error(
                    "cuRobo backend place planning failed",
                    stage="sdk",
                    code="CUROBO_BACKEND_FAILED",
                    status=500,
                    details={"arm": request.arm},
                ) from error
        finally:
            _CUDA_LOCK.release()

    def plan_axis_linear_native(
        self,
        request,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        if not isinstance(request, PlanAxisLinearRequest):
            raise _error(
                "request must be a PlanAxisLinearRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            backend = self._backend_unlocked(request.arm)
            if timeout > 0.0 and time.monotonic() >= deadline:
                raise _busy_error()
            # ========================================
            # MODIFICATION START: bind axis-linear planning to one stable scene
            # ========================================
            scene_token = self._refresh_scene_unlocked(
                snapshot=snapshot,
                arm=request.arm,
                backend=backend,
            )
            self._refresh_peer_collision_unlocked(
                arm=request.arm,
                backend=backend,
                active_joints_deg=request.start_joints_deg,
                other_joints_deg=request.other_joints_deg,
            )
            if timeout > 0.0 and time.monotonic() >= deadline:
                raise _busy_error()
            try:
                result = backend.plan_axis_linear_native(request)
                self._require_scene_token_unchanged_unlocked(
                    scene_token
                )
                return replace(result, scene_token=scene_token)
            except CuroboToolError as error:
                if error.code in {
                    "CUROBO_CRITERIA_UPDATE_FAILED",
                    "CUROBO_CRITERIA_RESET_FAILED",
                } and self._backends.get(request.arm) is backend:
                    self._backends.pop(request.arm, None)
                    self._scene_snapshot_key_by_arm.pop(
                        request.arm,
                        None,
                    )
                    self._peer_collision_snapshot_by_arm.pop(
                        request.arm,
                        None,
                    )
                raise
            except Exception as error:
                raise _error(
                    "cuRobo backend axis-linear planning failed",
                    stage="sdk",
                    code="CUROBO_BACKEND_FAILED",
                    status=500,
                    details={"arm": request.arm},
                ) from error
            # ========================================
            # MODIFICATION END: bind axis-linear planning to one stable scene
            # ========================================
        finally:
            _CUDA_LOCK.release()

    def solve_ik_native(
        self,
        request,
        lock_timeout_s=0.02,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        if not isinstance(request, SolveIkRequest):
            raise _error(
                "request must be a SolveIkRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            # ========================================
            # MODIFICATION START: scope live IK timeout to CUDA-lock acquisition
            # ========================================
            backend = self._backend_unlocked(request.arm)
            scene_token = self._refresh_scene_unlocked(
                snapshot=snapshot,
                arm=request.arm,
                backend=backend,
            )
            self._refresh_peer_collision_unlocked(
                arm=request.arm,
                backend=backend,
                active_joints_deg=request.current_joints_deg,
                other_joints_deg=request.other_joints_deg,
            )
            # ========================================
            # MODIFICATION END: scope live IK timeout to CUDA-lock acquisition
            # ========================================
            try:
                return backend.solve_ik_native(request)
            except CuroboToolError:
                raise
            except Exception as error:
                raise _error(
                    "cuRobo backend IK failed",
                    stage="sdk",
                    code="CUROBO_BACKEND_FAILED",
                    status=500,
                    details={"arm": request.arm},
                ) from error
        finally:
            _CUDA_LOCK.release()

    # ========================================
    # MODIFICATION START: bind final trajectory validation to one scene token
    # ========================================
    def validate_trajectory_native(
        self,
        request,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        if not isinstance(request, ValidateTrajectoryRequest):
            raise _error(
                "request must be a ValidateTrajectoryRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            if request.grasp_phase is None:
                backend = self._backend_unlocked(request.arm)
                cache_snapshot = True
            else:
                backend = self._compound_backend_unlocked(
                    request.arm,
                    request.trajectory_collision_clearance_m,
                    request.optimizer_collision_activation_distance_m,
                )
                cache_snapshot = False
            if timeout > 0.0 and time.monotonic() >= deadline:
                raise _busy_error()
            scene_token = self._refresh_scene_unlocked(
                snapshot=snapshot,
                arm=request.arm,
                backend=backend,
                cache_snapshot=cache_snapshot,
            )
            if (
                request.expected_scene_token is not None
                and request.expected_scene_token != scene_token
            ):
                self._raise_scene_changed(
                    request.expected_scene_token,
                    scene_token,
                )
            if request.gripper_joint_positions_rad is not None:
                from .trajectory_validation import validate_measured_trajectory_unlocked
                result = validate_measured_trajectory_unlocked(
                    self, request, backend.motion_planner, scene_token,
                )
                self._require_scene_token_unchanged_unlocked(scene_token)
                return result
            self._refresh_peer_collision_unlocked(
                arm=request.arm,
                backend=backend,
                active_joints_deg=request.points_deg[0],
                other_joints_deg=request.other_joints_deg,
                cache_snapshot=cache_snapshot,
            )
            if timeout > 0.0 and time.monotonic() >= deadline:
                raise _busy_error()
            try:
                checked_point_count = backend.validate_trajectory_native(request)
                self._require_scene_token_unchanged_unlocked(
                    scene_token,
                )
                return TrajectoryValidationResult.create(
                    collision_free=True,
                    checked_point_count=checked_point_count,
                    first_invalid_index=None,
                    scene_token=scene_token,
                )
            except CuroboToolError:
                raise
            except Exception as error:
                raise _error(
                    "cuRobo backend trajectory validation failed",
                    stage="sdk",
                    code="CUROBO_BACKEND_FAILED",
                    status=500,
                    details={"arm": request.arm},
                ) from error
        finally:
            _CUDA_LOCK.release()
    # ========================================
    # MODIFICATION END: bind final trajectory validation to one scene token
    # ========================================

    def _ensure_robot_segmenter_unlocked(self):
        if self._robot_segmenter is not None:
            return self._robot_segmenter
        if self._urdf_path is None or self._asset_root_path is None:
            raise _error(
                "robot segmentation resources are not configured",
                stage="configuration",
                code="CUROBO_ROBOT_SEGMENTER_NOT_CONFIGURED",
                status=503,
            )
        from .robot_segmenter import create_robot_segmenter_adapter

        self._robot_segmenter = create_robot_segmenter_adapter(
            robot_config_path=self._robot_config_path,
            urdf_path=self._urdf_path,
            asset_root_path=self._asset_root_path,
        )
        return self._robot_segmenter

    def _ensure_fixed_esdf_store_unlocked(self):
        if self._fixed_esdf_store is not None:
            return self._fixed_esdf_store
        from .fixed_esdf_store import FixedEsdfStore
        from .volumetric_mapper import create_volumetric_mapper_adapter

        segmenter = self._ensure_robot_segmenter_unlocked()
        self._fixed_esdf_store = FixedEsdfStore(
            mapper_factory=lambda: create_volumetric_mapper_adapter(
                robot_segmenter_boundary=segmenter.sdk_boundary
            ),
            selection_validator=self._validate_semantic_selection_unlocked,
        )
        return self._fixed_esdf_store

    def _validate_semantic_selection_unlocked(
        self,
        mapper,
        scene_model,
        _metadata,
        disabled_obstacle_name,
        *,
        raise_on_collision=False,
    ):
        """Load the immutable scene on both arms before publishing a selection."""

        boundary = getattr(mapper, "sdk_boundary", None)
        voxel_grid = getattr(boundary, "scene_voxel_grids", getattr(boundary, "esdf_grid", None))
        if voxel_grid is None:
            raise _error(
                "cuRobo ESDF has no integrated camera frame",
                stage="scene",
                code="CUROBO_ESDF_NOT_READY",
                status=409,
                retryable=True,
            )
        store = self._fixed_esdf_store
        previous = None if store is None else store.published_selection
        previous_mapper = None if store is None else store.published_mapper
        previous_boundary = getattr(previous_mapper, "sdk_boundary", None)
        previous_grid = getattr(previous_boundary, "scene_voxel_grids", getattr(previous_boundary, "esdf_grid", None))
        previous_scene = None if store is None else store.published_scene_model
        # A failed candidate must still fit the previously published world.
        self._configure_scene_cache_unlocked(scene_model, previous_scene)
        previous_instance_id = (
            None if previous is None else previous.get("selected_instance_id")
        )
        previous_disabled = (
            ()
            if not previous_instance_id
            else (f"semantic_instance::{previous_instance_id}",)
        )
        updated = []

        def restore_previous_selection():
            errors = []
            for arm, backend in updated:
                try:
                    backend.update_voxel_scene(
                        previous_grid if previous_grid is not None else voxel_grid,
                        previous_scene if previous_grid is not None else scene_model,
                        disabled_obstacle_names=previous_disabled if previous_grid is not None else (),
                    )
                except Exception as error:
                    errors.append((arm, error))
                finally:
                    self._scene_snapshot_key_by_arm.pop(arm, None)
            if errors:
                raise _error(
                    "cuRobo semantic selection rollback failed",
                    stage="scene",
                    code="CUROBO_SCENE_UPDATE_FAILED",
                    status=500,
                    details={"arms": [arm for arm, _error_value in errors]},
                ) from errors[0][1]

        try:
            for arm in ("A", "B"):
                backend = self._backend_unlocked(arm)
                update_mixed = getattr(backend, "update_voxel_scene", None)
                if not callable(update_mixed):
                    raise _error(
                        "cuRobo backend cannot load semantic ESDF geometry",
                        stage="configuration",
                        code="CUROBO_INVALID_CONFIG",
                        status=500,
                        details={"arm": arm},
                    )
                updated.append((arm, backend))
                update_mixed(
                    voxel_grid,
                    scene_model,
                    disabled_obstacle_names=(disabled_obstacle_name,) if disabled_obstacle_name else (),
                )
            captured_joints = (
                _metadata.get("robot_joints_at_capture")
                if isinstance(_metadata, Mapping)
                else None
            )
            if not isinstance(captured_joints, Mapping):
                raise _error(
                    "semantic ESDF has no captured robot state",
                    stage="configuration",
                    code="CUROBO_INVALID_CONFIG",
                    status=500,
                )
            for arm, backend in updated:
                peer_arm = "B" if arm == "A" else "A"
                active_joints = captured_joints.get(arm)
                other_joints = captured_joints.get(peer_arm)
                self._refresh_peer_collision_unlocked(
                    arm=arm,
                    backend=backend,
                    active_joints_deg=active_joints,
                    other_joints_deg=other_joints,
                )
                request = ValidateTrajectoryRequest.create(
                    arm=arm,
                    points_deg=[active_joints] * 6,
                    other_joints_deg=other_joints,
                )
                try:
                    backend.validate_trajectory_native(request)
                except CuroboToolError as error:
                    if error.code != "CUROBO_TRAJECTORY_COLLISION":
                        raise
                    if raise_on_collision:
                        raise _error(
                            "semantic ESDF candidate collides with the robot at the captured pose",
                            stage="scene",
                            code="fixed_esdf_initial_collision",
                            status=409,
                            details=error.details,
                        ) from error
                    restore_previous_selection()
                    return False
            return True
        except Exception as error:
            try:
                restore_previous_selection()
            except Exception as rollback_error:
                raise rollback_error from error
            raise
        finally:
            self._scene_snapshot_key_by_arm.clear()

    def _esdf_status_unlocked(self):
        if self._fixed_esdf_store is not None:
            fixed_status = self._fixed_esdf_store.published_status()
            if fixed_status.ready:
                return fixed_status
        from .volumetric_mapper import EsdfStatusResult

        return EsdfStatusResult(
            ready=False,
            revision=0,
            integrated_frame_count=0,
            point_count=0,
            scene_token=None,
        )

    def begin_fixed_esdf_candidate_native(self, build_id, lock_timeout_s=120.0):
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            self._ensure_fixed_esdf_store_unlocked().begin_candidate(build_id)
        finally:
            _CUDA_LOCK.release()

    def prepare_semantic_fixed_esdf_candidate_native(
        self,
        build_id,
        payload,
        lock_timeout_s=120.0,
    ):
        """Build and validate one private three-camera semantic candidate."""

        from curobo.scene import Scene

        from .fixed_esdf_mapping import SemanticEsdfCandidateBuilder

        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        store = None
        try:
            store = self._ensure_fixed_esdf_store_unlocked()
            segmenter = self._ensure_robot_segmenter_unlocked()
            store.begin_candidate(build_id)
            mapper = store.candidate_mapper_for(build_id)
            segmenter_boundary = segmenter.sdk_boundary
            robot_spheres = getattr(segmenter_boundary, "robot_spheres", None)
            if not callable(robot_spheres):
                raise _error(
                    "cuRobo robot collision sphere projection is unavailable",
                    stage="configuration",
                    code="CUROBO_INVALID_CONFIG",
                    status=500,
                )
            candidate = SemanticEsdfCandidateBuilder(
                robot_segmenter_boundary=segmenter_boundary,
                mapper_boundary=mapper.sdk_boundary,
                scene_type=Scene,
                robot_spheres=robot_spheres,
            ).build(payload)
            primary_view = payload.get("views", {}).get("3", {})
            metadata = {
                "mapping_mode": payload.get("mapping_mode", "semantic"),
                "observation_set_id": candidate.observation_set_id,
                "surface_voxel_size_m": candidate.integration.surface_voxel_size_m,
                "surface_color_source": "frozen_rgb" if any(
                    "colors_rgb" in model
                    for model in candidate.scene_model.get("voxel", {}).values()
                ) else "uniform",
                "selected_instance_id": candidate.selected_instance_id,
                "selection_generation": candidate.selection_generation,
                "robot_joints_at_capture": primary_view.get(
                    "robot_joints_at_capture"
                ),
                "bounds": {
                    "center_m": list(candidate.bounds.center_m),
                    "extent_m": list(candidate.bounds.extent_m),
                },
                "tabletop": {
                    "height_m": float(candidate.table.height_m),
                    "normal_xyz": list(candidate.table.normal_xyz),
                    "inlier_count": int(candidate.table.inlier_count),
                    "robot_clearance_m": float(
                        candidate.table.robot_clearance_m
                    ),
                    "support_plane": getattr(candidate.table, "support_plane", None),
                },
                "camera_metrics": {
                    key: dict(value)
                    for key, value in candidate.camera_metrics.items()
                },
                "instance_metrics": {
                    key: dict(value)
                    for key, value in candidate.instance_metrics.items()
                },
                "timings_ms": dict(candidate.timings_ms),
            }
            scene_model = {
                key: dict(value) for key, value in candidate.scene_model.items()
            }
            store.stage_semantic_candidate(
                build_id,
                candidate.integration,
                scene_model,
                metadata,
            )
            disabled_name = (f"semantic_instance::{candidate.selected_instance_id}"
                             if candidate.selected_instance_id else None)
            if self._validate_semantic_selection_unlocked(
                mapper,
                scene_model,
                metadata,
                disabled_name,
                raise_on_collision=True,
            ) is not True:
                store.discard_candidate(build_id)
                raise _error(
                    "semantic ESDF candidate collides with the robot at the captured pose",
                    stage="scene",
                    code="fixed_esdf_initial_collision",
                    status=409,
                )
            return candidate
        except Exception:
            if store is not None:
                try:
                    store.discard_candidate(build_id)
                except Exception:
                    pass
            raise
        finally:
            self._scene_snapshot_key_by_arm.clear()
            _CUDA_LOCK.release()

    def integrate_fixed_esdf_candidate_native(
        self,
        build_id,
        request,
        *,
        finalize,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            return self._ensure_fixed_esdf_store_unlocked().integrate_candidate(
                build_id,
                request,
                finalize=finalize,
            )
        finally:
            _CUDA_LOCK.release()

    def set_fixed_esdf_candidate_scene_native(
        self,
        build_id,
        scene_model,
        metadata,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            self._ensure_fixed_esdf_store_unlocked().set_candidate_scene(
                build_id,
                scene_model,
                metadata,
            )
        finally:
            _CUDA_LOCK.release()

    def validate_fixed_esdf_candidate_native(
        self,
        build_id,
        request,
        lock_timeout_s=120.0,
    ):
        """Check both arms at the captured pose in the private candidate world."""

        from .robot_segmenter import RobotSegmentationRequest

        if not isinstance(request, RobotSegmentationRequest):
            raise _error(
                "request must be a RobotSegmentationRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        checked_arms = []
        try:
            store = self._ensure_fixed_esdf_store_unlocked()
            mapper = store.candidate_mapper_for(build_id)
            scene_model = store.candidate_scene_model_for(build_id)
            candidate_metadata_for = getattr(
                store,
                "candidate_metadata_for",
                None,
            )
            metadata = (
                candidate_metadata_for(build_id)
                if callable(candidate_metadata_for)
                else {}
            )
            selected_instance_id = (
                metadata.get("selected_instance_id")
                if isinstance(metadata, Mapping)
                else None
            )
            disabled = (
                ()
                if not selected_instance_id
                else (f"semantic_instance::{selected_instance_id}",)
            )
            voxel_grid = (
                None if mapper is None else getattr(mapper.sdk_boundary, "scene_voxel_grids", mapper.sdk_boundary.esdf_grid)
            )
            if voxel_grid is None or scene_model is None:
                raise _error(
                    "fixed ESDF candidate is not ready for validation",
                    stage="scene",
                    code="CUROBO_FIXED_ESDF_CANDIDATE_NOT_READY",
                    status=409,
                )
            self._configure_scene_cache_unlocked(
                scene_model, store.published_scene_model
            )
            for arm in ("A", "B"):
                backend = self._backend_unlocked(arm)
                update_mixed = getattr(backend, "update_voxel_scene", None)
                if not callable(update_mixed):
                    raise _error(
                        "cuRobo backend cannot validate semantic ESDF geometry",
                        stage="config",
                        code="CUROBO_INVALID_CONFIG",
                        status=500,
                        details={"arm": arm},
                    )
                update_mixed(
                    voxel_grid,
                    scene_model,
                    disabled_obstacle_names=disabled,
                )
                self._scene_snapshot_key_by_arm.pop(arm, None)
                active_joints = request.joint_positions_deg[arm]
                peer_arm = "B" if arm == "A" else "A"
                other_joints = request.joint_positions_deg[peer_arm]
                self._refresh_peer_collision_unlocked(
                    arm=arm,
                    backend=backend,
                    active_joints_deg=active_joints,
                    other_joints_deg=other_joints,
                )
                validation_request = ValidateTrajectoryRequest.create(
                    arm=arm,
                    points_deg=[active_joints] * 6,
                    other_joints_deg=other_joints,
                )
                checked_arms.append(arm)
                try:
                    backend.validate_trajectory_native(validation_request)
                except CuroboToolError as error:
                    if error.code != "CUROBO_TRAJECTORY_COLLISION":
                        raise
                    return FixedEsdfCandidateValidationResult.create(
                        collision_free=False,
                        checked_arms=checked_arms,
                        colliding_arm=arm,
                    )
            return FixedEsdfCandidateValidationResult.create(
                collision_free=True,
                checked_arms=checked_arms,
            )
        finally:
            self._scene_snapshot_key_by_arm.clear()
            _CUDA_LOCK.release()

    def publish_fixed_esdf_candidate_native(self, build_id, lock_timeout_s=120.0):
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            store = self._ensure_fixed_esdf_store_unlocked()
            result = store.publish_candidate(build_id)
            self._scene_snapshot_key_by_arm.clear()
            return result
        finally:
            _CUDA_LOCK.release()

    def reconcile_fixed_esdf_candidate_native(self, build_id, lock_timeout_s=120.0):
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            if self._fixed_esdf_store is None:
                return None
            return self._fixed_esdf_store.reconcile_candidate(build_id)
        finally:
            _CUDA_LOCK.release()

    def select_fixed_esdf_instance_native(
        self,
        *,
        expected_revision,
        selected_instance_id,
        selection_generation,
        lock_timeout_s=120.0,
    ):
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            result = self._ensure_fixed_esdf_store_unlocked().select_instance(
                expected_revision=expected_revision,
                selected_instance_id=selected_instance_id,
                selection_generation=selection_generation,
            )
            self._scene_snapshot_key_by_arm.clear()
            return result
        finally:
            _CUDA_LOCK.release()

    def discard_fixed_esdf_candidate_native(self, build_id, lock_timeout_s=120.0, *, expected_scene_token=None):
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            if self._fixed_esdf_store is not None:
                if expected_scene_token is not None:
                    cleared = self._fixed_esdf_store.clear_published(build_id, expected_scene_token)
                    if cleared:
                        self._scene_snapshot_key_by_arm.clear()
                    return cleared
                self._fixed_esdf_store.discard_candidate(build_id)
            return False
        finally:
            _CUDA_LOCK.release()

    def segment_robot_native(
        self,
        request,
        *,
        include_base_pointcloud=False,
        lock_timeout_s=1.0,
    ):
        """Run robot depth segmentation under the planner's CUDA lock."""

        from .robot_segmenter import RobotSegmentationRequest

        if not isinstance(request, RobotSegmentationRequest):
            raise _error(
                "request must be a RobotSegmentationRequest",
                stage="input",
                code="CUROBO_INVALID_REQUEST",
                status=400,
            )
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        _acquire_operation_until(deadline, timeout)
        try:
            segmenter = self._ensure_robot_segmenter_unlocked()
            if include_base_pointcloud:
                return segmenter.segment_robot(
                    request,
                    include_base_pointcloud=True,
                )
            return segmenter.segment_robot(request)
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo robot segmentation failed",
                stage="sdk",
                code="CUROBO_ROBOT_SEGMENTATION_FAILED",
                status=500,
            ) from error
        finally:
            _CUDA_LOCK.release()

    def health_native(self):
        ready_arms = tuple(
            arm for arm in ("A", "B") if arm in self._backends
        )
        external_obstacle_count = 0
        if self._scene_model is not None:
            cuboids = self._scene_model.get("cuboid", {})
            if isinstance(cuboids, Mapping):
                external_obstacle_count = len(cuboids)
        try:
            return HealthResult(
                ready=True,
                arms_ready=ready_arms,
                scene=self.scene,
                external_obstacle_count=external_obstacle_count,
                curobo_source=self._curobo_source,
                curobo_version=self._curobo_version,
                gpu_available=self._gpu_available,
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "cuRobo health data is invalid",
                stage="sdk",
                code="CUROBO_BACKEND_FAILED",
                status=500,
            ) from error


__all__ = ["CuroboRuntime"]
# ========================================
# MODIFICATION END: implement the private cuRobo runtime
# ========================================

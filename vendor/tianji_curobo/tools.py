"""Typed process-local facade around the private cuRobo runtime.

Only the isolated planner service imports this module.  Core and ordinary
Extensions communicate through :mod:`curobo_service_client`, so importing the
JODELL host never imports Torch, CUDA, or cuRobo.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from .contracts import (
    SolveIkBatchRequest,
    NativeIkBatchResult,
    CuroboToolError,
    FixedEsdfCandidateValidationResult,
    HealthResult,
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
from .robot_segmenter import RobotSegmentationRequest, RobotSegmentationResult
from .volumetric_mapper import EsdfIntegrationResult


@dataclass(frozen=True)
class _ToolContext:
    runtime: Any
    formal_lock_timeout_s: float
    tracker_lock_timeout_s: float
    segmentation_lock_timeout_s: float


_context: _ToolContext | None = None


def _configuration_error(message: str, code: str) -> CuroboToolError:
    return CuroboToolError(
        message,
        stage="configuration",
        code=code,
        status=503 if code == "CUROBO_NOT_CONFIGURED" else 500,
    )


def _timeout(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _configuration_error(
            f"{name} must be a finite positive number",
            "CUROBO_INVALID_CONFIGURATION",
        )
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise _configuration_error(
            f"{name} must be a finite positive number",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return result


def _runtime_protocol(runtime: object) -> Any:
    required = (
        "plan_to_pose_native",
        "plan_to_joint_native",
        "plan_axis_linear_native",
        "solve_ik_native",
        "validate_trajectory_native",
        "health_native",
    )
    if any(not callable(getattr(runtime, name, None)) for name in required):
        raise _configuration_error(
            "runtime does not implement the cuRobo tool protocol",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return runtime


def configure_runtime(
    *,
    robot_config_path: Path,
    urdf_path: Path,
    asset_root_path: Path,
    runtime: Any | None = None,
    scene: str = "pickplace",
    scene_state_provider: Any | None = None,
    warmup: bool = True,
    formal_lock_timeout_s: float = 120.0,
    tracker_lock_timeout_s: float = 0.02,
    segmentation_lock_timeout_s: float = 1.0,
) -> None:
    """Configure one process runtime from explicit Extension-owned resources."""

    global _context
    formal_timeout = _timeout(formal_lock_timeout_s, "formal_lock_timeout_s")
    tracker_timeout = _timeout(tracker_lock_timeout_s, "tracker_lock_timeout_s")
    segmentation_timeout = _timeout(
        segmentation_lock_timeout_s,
        "segmentation_lock_timeout_s",
    )
    if runtime is None:
        if not isinstance(scene, str) or not scene.strip():
            raise _configuration_error(
                "scene must be a non-empty string", "CUROBO_INVALID_CONFIGURATION"
            )
        if not isinstance(warmup, bool):
            raise _configuration_error(
                "warmup must be a boolean", "CUROBO_INVALID_CONFIGURATION"
            )
        try:
            from ._curobo_impl import CuroboRuntime

            runtime = CuroboRuntime(
                scene=scene,
                warmup=warmup,
                robot_config_path=robot_config_path,
                urdf_path=urdf_path,
                asset_root_path=asset_root_path,
                scene_state_provider=scene_state_provider,
            )
            if warmup:
                runtime.warmup()
        except CuroboToolError:
            raise
        except Exception as error:
            raise CuroboToolError(
                "failed to configure cuRobo runtime",
                stage="configuration",
                code="CUROBO_RUNTIME_CONFIGURATION_FAILED",
                status=500,
            ) from error
    _context = _ToolContext(
        runtime=_runtime_protocol(runtime),
        formal_lock_timeout_s=formal_timeout,
        tracker_lock_timeout_s=tracker_timeout,
        segmentation_lock_timeout_s=segmentation_timeout,
    )


def reset_runtime() -> None:
    """Drop the process context; primarily useful to isolate service tests."""

    global _context
    _context = None


def _configured() -> _ToolContext:
    if _context is None:
        raise _configuration_error(
            "cuRobo tools are not configured", "CUROBO_NOT_CONFIGURED"
        )
    return _context


def plan_dual_joint(request):
    from .dual_contracts import DualPlanRequest
    if not isinstance(request, DualPlanRequest) or request.goal_joints_deg is None:
        raise ValueError("plan_dual_joint requires a dual joint goal")
    context = _configured()
    return context.runtime.dual_operation_native(request, lock_timeout_s=context.formal_lock_timeout_s)


def plan_dual_pose(request):
    from .dual_contracts import DualPlanRequest
    if not isinstance(request, DualPlanRequest) or request.goal_transforms_world is None:
        raise ValueError("plan_dual_pose requires two TCP world poses")
    context = _configured()
    return context.runtime.dual_operation_native(request, lock_timeout_s=context.formal_lock_timeout_s)


def validate_dual_trajectory(request):
    context = _configured()
    return context.runtime.dual_operation_native(request, validate=True, lock_timeout_s=context.formal_lock_timeout_s)


def plan_to_pose(request: PlanToPoseRequest) -> NativeTrajectory:
    if not isinstance(request, PlanToPoseRequest):
        raise CuroboToolError(
            "request must be a PlanToPoseRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    return context.runtime.plan_to_pose_native(
        request, lock_timeout_s=context.formal_lock_timeout_s
    )


def plan_grasp(request: PlanGraspRequest) -> NativeGraspPlan:
    if not isinstance(request, PlanGraspRequest):
        raise CuroboToolError(
            "request must be a PlanGraspRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    operation = getattr(context.runtime, "plan_grasp_native", None)
    if not callable(operation):
        raise _configuration_error(
            "cuRobo runtime does not implement plan_grasp_native",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return operation(request, lock_timeout_s=context.formal_lock_timeout_s)


def solve_ik_batch(request: SolveIkBatchRequest) -> NativeIkBatchResult:
    if not isinstance(request, SolveIkBatchRequest):
        raise CuroboToolError("invalid IK batch request", stage="validation", code="CUROBO_INVALID_REQUEST", status=422)
    context = _configured()
    return context.runtime.solve_ik_batch_native(request, lock_timeout_s=context.formal_lock_timeout_s)


def plan_grasp_batch(request: PlanGraspBatchRequest) -> NativeGraspBatchPlan:
    if not isinstance(request, PlanGraspBatchRequest):
        raise CuroboToolError(
            "request must be a PlanGraspBatchRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    operation = getattr(context.runtime, "plan_grasp_batch_native", None)
    if not callable(operation):
        raise _configuration_error(
            "cuRobo runtime does not implement plan_grasp_batch_native",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return operation(request, lock_timeout_s=context.formal_lock_timeout_s)


def invalidate_grasp_batch(
    arm: str,
    batch_size: int,
    trajectory_collision_clearance_m: float = 0.01,
    optimizer_collision_activation_distance_m: float = 0.03,
) -> bool:
    context = _configured()
    operation = getattr(context.runtime, "invalidate_grasp_batch_native", None)
    if not callable(operation):
        raise _configuration_error(
            "cuRobo runtime does not implement invalidate_grasp_batch_native",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return operation(
        arm,
        batch_size,
        trajectory_collision_clearance_m,
        optimizer_collision_activation_distance_m,
        lock_timeout_s=context.formal_lock_timeout_s,
    )


def plan_place(request: PlanPlaceRequest) -> NativePlacePlan:
    if not isinstance(request, PlanPlaceRequest):
        raise CuroboToolError(
            "request must be a PlanPlaceRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    operation = getattr(context.runtime, "plan_place_native", None)
    if not callable(operation):
        raise _configuration_error(
            "cuRobo runtime does not implement plan_place_native",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return operation(request, lock_timeout_s=context.formal_lock_timeout_s)


def plan_to_joint(request: PlanToJointRequest) -> NativeTrajectory:
    if not isinstance(request, PlanToJointRequest):
        raise CuroboToolError(
            "request must be a PlanToJointRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    return context.runtime.plan_to_joint_native(
        request,
        lock_timeout_s=context.formal_lock_timeout_s,
    )


def plan_axis_linear(request: PlanAxisLinearRequest) -> NativeTrajectory:
    if not isinstance(request, PlanAxisLinearRequest):
        raise CuroboToolError(
            "request must be a PlanAxisLinearRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    return context.runtime.plan_axis_linear_native(
        request,
        lock_timeout_s=context.formal_lock_timeout_s,
    )


def validate_trajectory(
    request: ValidateTrajectoryRequest,
) -> TrajectoryValidationResult:
    if not isinstance(request, ValidateTrajectoryRequest):
        raise CuroboToolError(
            "request must be a ValidateTrajectoryRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    return context.runtime.validate_trajectory_native(
        request,
        lock_timeout_s=context.formal_lock_timeout_s,
    )


def solve_ik(request: SolveIkRequest) -> NativeIkResult:
    if not isinstance(request, SolveIkRequest):
        raise CuroboToolError(
            "request must be a SolveIkRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    return context.runtime.solve_ik_native(
        request, lock_timeout_s=context.tracker_lock_timeout_s
    )


def segment_robot(
    request: RobotSegmentationRequest,
    *,
    include_base_pointcloud: bool = False,
) -> RobotSegmentationResult:
    """Run the internal robot-depth filter for fixed-map construction."""

    if not isinstance(request, RobotSegmentationRequest):
        raise CuroboToolError(
            "request must be a RobotSegmentationRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    operation = getattr(context.runtime, "segment_robot_native", None)
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement robot depth segmentation",
            "CUROBO_INVALID_CONFIGURATION",
        )
    if not isinstance(include_base_pointcloud, bool):
        raise CuroboToolError(
            "include_base_pointcloud must be a bool",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    if include_base_pointcloud:
        return operation(
            request,
            include_base_pointcloud=True,
            lock_timeout_s=context.segmentation_lock_timeout_s,
        )
    return operation(
        request,
        lock_timeout_s=context.segmentation_lock_timeout_s,
    )


def begin_fixed_esdf_candidate(build_id: str) -> None:
    context = _configured()
    operation = getattr(context.runtime, "begin_fixed_esdf_candidate_native", None)
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement fixed ESDF mapping",
            "CUROBO_INVALID_CONFIGURATION",
        )
    operation(build_id, lock_timeout_s=context.formal_lock_timeout_s)


def prepare_semantic_fixed_esdf_candidate(build_id: str, payload: Mapping):
    """Build and validate one private three-camera semantic candidate."""

    context = _configured()
    operation = getattr(
        context.runtime,
        "prepare_semantic_fixed_esdf_candidate_native",
        None,
    )
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement semantic fixed ESDF mapping",
            "CUROBO_INVALID_CONFIGURATION",
        )
    if not isinstance(payload, Mapping):
        raise TypeError("semantic fixed ESDF payload must be a mapping")
    return operation(
        build_id,
        payload,
        lock_timeout_s=context.formal_lock_timeout_s,
    )


def integrate_fixed_esdf_candidate(
    build_id: str,
    request: RobotSegmentationRequest,
    *,
    finalize: bool,
) -> EsdfIntegrationResult:
    if not isinstance(request, RobotSegmentationRequest):
        raise CuroboToolError(
            "request must be a RobotSegmentationRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    operation = getattr(context.runtime, "integrate_fixed_esdf_candidate_native", None)
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement fixed ESDF mapping",
            "CUROBO_INVALID_CONFIGURATION",
        )
    if not isinstance(finalize, bool):
        raise CuroboToolError(
            "finalize must be a bool",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    return operation(
        build_id,
        request,
        finalize=finalize,
        lock_timeout_s=context.formal_lock_timeout_s,
    )


def set_fixed_esdf_candidate_scene(build_id: str, scene_model, metadata) -> None:
    if not isinstance(scene_model, Mapping) or not isinstance(metadata, Mapping):
        raise CuroboToolError(
            "scene_model and metadata must be mappings",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    operation = getattr(context.runtime, "set_fixed_esdf_candidate_scene_native", None)
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement fixed ESDF semantic scenes",
            "CUROBO_INVALID_CONFIGURATION",
        )
    operation(
        build_id,
        scene_model,
        metadata,
        lock_timeout_s=context.formal_lock_timeout_s,
    )


def validate_fixed_esdf_candidate(
    build_id: str,
    request: RobotSegmentationRequest,
) -> FixedEsdfCandidateValidationResult:
    if not isinstance(request, RobotSegmentationRequest):
        raise CuroboToolError(
            "request must be a RobotSegmentationRequest",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    operation = getattr(
        context.runtime,
        "validate_fixed_esdf_candidate_native",
        None,
    )
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement fixed ESDF candidate validation",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return operation(
        build_id,
        request,
        lock_timeout_s=context.formal_lock_timeout_s,
    )


def publish_fixed_esdf_candidate(build_id: str) -> EsdfIntegrationResult:
    context = _configured()
    operation = getattr(context.runtime, "publish_fixed_esdf_candidate_native", None)
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement fixed ESDF mapping",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return operation(build_id, lock_timeout_s=context.formal_lock_timeout_s)


def reconcile_fixed_esdf_candidate(build_id: str) -> EsdfIntegrationResult | None:
    context = _configured()
    operation = getattr(context.runtime, "reconcile_fixed_esdf_candidate_native", None)
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement fixed ESDF reconciliation",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return operation(build_id, lock_timeout_s=context.formal_lock_timeout_s)


def discard_fixed_esdf_candidate(build_id: str, *, expected_scene_token=None):
    context = _configured()
    operation = getattr(context.runtime, "discard_fixed_esdf_candidate_native", None)
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement fixed ESDF mapping",
            "CUROBO_INVALID_CONFIGURATION",
        )
    if expected_scene_token is not None:
        return operation(build_id, lock_timeout_s=context.formal_lock_timeout_s,
                         expected_scene_token=expected_scene_token)
    return operation(build_id, lock_timeout_s=context.formal_lock_timeout_s)


def select_fixed_esdf_instance(
    expected_revision: int,
    selected_instance_id: str,
    selection_generation: int,
) -> EsdfIntegrationResult:
    if (
        type(expected_revision) is not int
        or expected_revision < 1
        or not isinstance(selected_instance_id, str)
        or not selected_instance_id.strip()
        or type(selection_generation) is not int
        or selection_generation < 1
    ):
        raise CuroboToolError(
            "semantic ESDF selection payload is invalid",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    context = _configured()
    operation = getattr(
        context.runtime,
        "select_fixed_esdf_instance_native",
        None,
    )
    if not callable(operation):
        raise _configuration_error(
            "runtime does not implement semantic ESDF selection",
            "CUROBO_INVALID_CONFIGURATION",
        )
    return operation(
        expected_revision=expected_revision,
        selected_instance_id=selected_instance_id.strip(),
        selection_generation=selection_generation,
        lock_timeout_s=context.formal_lock_timeout_s,
    )


def validate_grasp_lift(payload):
    from .grasp_lift_compute import validate_grasp_lift as compute

    context = _configured()
    return compute(context.runtime, payload, lock_timeout_s=context.formal_lock_timeout_s)


def health() -> HealthResult:
    return _configured().runtime.health_native()


__all__ = [
    "begin_fixed_esdf_candidate",
    "configure_runtime",
    "discard_fixed_esdf_candidate",
    "health",
    "integrate_fixed_esdf_candidate",
    "plan_axis_linear",
    "plan_place",
    "plan_to_joint",
    "plan_to_pose",
    "prepare_semantic_fixed_esdf_candidate",
    "publish_fixed_esdf_candidate",
    "reconcile_fixed_esdf_candidate",
    "reset_runtime",
    "segment_robot",
    "select_fixed_esdf_instance",
    "set_fixed_esdf_candidate_scene",
    "solve_ik",
    "validate_fixed_esdf_candidate",
    "validate_trajectory",
]

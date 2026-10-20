"""Small HTTP boundary for the isolated cuRobo GPU process."""

from __future__ import annotations

import argparse
import base64
from contextvars import ContextVar
import hashlib
import json
import logging
import os
import sys
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

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
from . import tools
from .dual_contracts import DualPlanRequest, DualValidationRequest, validate_plan_result
from .fixed_esdf_mapping import SemanticEsdfCandidate
from .robot_segmenter import RobotSegmentationRequest, RobotSegmentationResult
from .semantic_esdf_transport import decode_semantic_esdf_request
from .volumetric_mapper import EsdfIntegrationResult
MAX_REQUEST_BYTES = 64 * 1024 * 1024
SERVICE_NAME = "jodell-curobo-planner"
LOGGER = logging.getLogger(__name__)


def _error(message: str, *, stage: str, code: str, status: int, retryable=False):
    return CuroboToolError(
        message, stage=stage, code=code, status=status, retryable=retryable
    )


def _invalid_request(error: Exception) -> CuroboToolError:
    return _error(
        str(error),
        stage="validation",
        code="CUROBO_INVALID_REQUEST",
        status=422,
    )


def _request_from_mapping(request_type: Any, payload: object):
    try:
        return request_type.from_mapping(payload)
    except CuroboToolError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise _invalid_request(error) from None


def materialize_urdf(
    source: Path,
    *,
    marvin_meshes: Path,
    rag_meshes: Path,
    cache_root: Path | None = None,
) -> Path:
    """Create a deterministic URDF whose mesh references are explicit paths."""

    source = Path(source).expanduser().resolve()
    marvin_meshes = Path(marvin_meshes).expanduser().resolve()
    rag_meshes = Path(rag_meshes).expanduser().resolve()
    if not source.is_file() or not marvin_meshes.is_dir() or not rag_meshes.is_dir():
        raise FileNotFoundError("model capability returned an incomplete URDF resource set")
    text = source.read_text(encoding="utf-8")
    source_root = source.parent.as_posix().rstrip("/") + "/"
    text = text.replace('filename="./', f'filename="{source_root}')
    text = text.replace(
        "package://Marvin_M6_RAG100P_description/",
        source_root,
    )
    text = text.replace(
        "package://Marvin M6-S-CCS-696-V4.0_Base_and_Stand_Asm urdf/meshes/",
        marvin_meshes.as_posix().rstrip("/") + "/",
    ).replace(
        "package://rag100_050/meshes/",
        rag_meshes.as_posix().rstrip("/") + "/",
    )
    if "package://" in text:
        raise ValueError("URDF contains an unsupported package resource")
    # The pinned cuRobo parser requires exact signed cardinal axes. USD exports
    # introduce ~1e-8 roundoff; normalize only that noise in this derived copy.
    document = ET.fromstring(text)
    changed = False
    cardinal_axes = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for axis in document.findall("./joint/axis"):
        values = [float(value) for value in axis.get("xyz", "").split()]
        for candidate in cardinal_axes:
            if len(values) == 3 and all(abs(value - exact) <= 1e-6 for value, exact in zip(values, candidate)):
                canonical = " ".join(str(value) for value in candidate)
                if axis.get("xyz") != canonical:
                    axis.set("xyz", canonical)
                    changed = True
                break
    if changed:
        text = ET.tostring(document, encoding="unicode")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    root = Path(
        cache_root
        or os.environ.get("JODELL_CUROBO_CACHE")
        or (Path(tempfile.gettempdir()) / "jodell-curobo")
    ).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"marvin-{digest}.urdf"
    if not target.is_file():
        temporary = target.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
    return target


class PlannerRegistry:
    """Typed operations; a fake tool API can be injected without CUDA."""

    def __init__(
        self,
        *,
        tool_api: Any = tools,
        robot_config: Path | None = None,
        urdf: Path | None = None,
        asset_root: Path | None = None,
        scene: str = "pickplace",
        scene_state_provider: Any | None = None,
        warmup: bool = True,
        configure: bool = True,
    ) -> None:
        required = (
            "health",
            "plan_to_pose",
            "plan_to_joint",
            "plan_axis_linear",
            "solve_ik",
            "validate_trajectory",
        )
        if any(not callable(getattr(tool_api, name, None)) for name in required):
            raise TypeError("tool_api does not implement the cuRobo service contract")
        self._tools = tool_api
        self._external_scene_state_provider = scene_state_provider
        self._request_scene_snapshot = ContextVar(
            "jodell_curobo_request_scene_snapshot",
            default=None,
        )
        if configure:
            configure_runtime = getattr(tool_api, "configure_runtime", None)
            if not callable(configure_runtime):
                raise TypeError("tool_api has no configure_runtime operation")
            if robot_config is None or urdf is None or asset_root is None:
                raise ValueError("robot_config, urdf and asset_root are required")
            configure_runtime(
                robot_config_path=Path(robot_config),
                urdf_path=Path(urdf),
                asset_root_path=Path(asset_root),
                scene=scene,
                scene_state_provider=self._current_scene_snapshot,
                warmup=warmup,
            )

    def _current_scene_snapshot(self):
        snapshot = self._request_scene_snapshot.get()
        if snapshot is not None:
            return snapshot
        if self._external_scene_state_provider is None:
            return None
        return self._external_scene_state_provider()

    def _with_request_scene(self, payload: object, operation):
        if not isinstance(payload, Mapping):
            return operation()
        snapshot = payload.get("scene_snapshot")
        if snapshot is not None and not isinstance(snapshot, Mapping):
            raise _invalid_request(ValueError("scene_snapshot must be an object"))
        copied = None
        if snapshot is not None:
            copied = json.loads(json.dumps(snapshot, allow_nan=False))
        token = self._request_scene_snapshot.set(copied)
        try:
            return operation()
        finally:
            self._request_scene_snapshot.reset(token)

    def health(self) -> dict[str, Any]:
        result = self._tools.health()
        if not isinstance(result, HealthResult):
            raise TypeError("health returned an invalid result")
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "ready": result.ready,
            "arms_ready": list(result.arms_ready),
            "scene": result.scene,
            "external_obstacle_count": result.external_obstacle_count,
            "curobo_source": result.curobo_source,
            "curobo_version": result.curobo_version,
            "gpu_available": result.gpu_available,
        }

    def plan_dual_joint(self, payload):
        request = _request_from_mapping(DualPlanRequest, payload)
        return validate_plan_result(self._with_request_scene(
            payload, lambda: self._tools.plan_dual_joint(request)))

    def plan_dual_pose(self, payload):
        request = _request_from_mapping(DualPlanRequest, payload)
        return validate_plan_result(self._with_request_scene(
            payload, lambda: self._tools.plan_dual_pose(request)))

    def validate_dual_trajectory(self, payload):
        request = _request_from_mapping(DualValidationRequest, payload)
        return self._with_request_scene(payload, lambda: self._tools.validate_dual_trajectory(request))

    def plan_to_pose(self, payload: object) -> dict[str, Any]:
        request = _request_from_mapping(PlanToPoseRequest, payload)
        result = self._with_request_scene(
            payload,
            lambda: self._tools.plan_to_pose(request),
        )
        if not isinstance(result, NativeTrajectory):
            raise TypeError("plan_to_pose returned an invalid result")
        return {
            "ok": True,
            "planner": "curobo",
            "positions_rad": [list(row) for row in result.positions_rad],
            "joint_names": list(result.joint_names),
            "interpolation_dt_s": result.interpolation_dt_s,
            "trajectory_point_scale": result.trajectory_point_scale,
            "command_positions_rad": (
                None if result.command_positions_rad is None
                else [list(row) for row in result.command_positions_rad]
            ),
            "planning_time_s": result.planning_time_s,
            "position_error_m": result.position_error_m,
            "rotation_error_rad": result.rotation_error_rad,
            "scene_token": result.scene_token,
        }

    def plan_to_joint(self, payload: object) -> dict[str, Any]:
        request = _request_from_mapping(PlanToJointRequest, payload)
        result = self._with_request_scene(
            payload,
            lambda: self._tools.plan_to_joint(request),
        )
        if not isinstance(result, NativeTrajectory):
            raise TypeError("plan_to_joint returned an invalid result")
        return {
            "ok": True,
            "planner": "curobo",
            "positions_rad": [list(row) for row in result.positions_rad],
            "joint_names": list(result.joint_names),
            "interpolation_dt_s": result.interpolation_dt_s,
            "trajectory_point_scale": result.trajectory_point_scale,
            "command_positions_rad": (
                None if result.command_positions_rad is None
                else [list(row) for row in result.command_positions_rad]
            ),
            "planning_time_s": result.planning_time_s,
            "position_error_m": result.position_error_m,
            "rotation_error_rad": result.rotation_error_rad,
            "scene_token": result.scene_token,
        }

    def solve_ik(self, payload: object) -> dict[str, Any]:
        request = _request_from_mapping(SolveIkRequest, payload)
        result = self._with_request_scene(
            payload,
            lambda: self._tools.solve_ik(request),
        )
        if not isinstance(result, NativeIkResult):
            raise TypeError("solve_ik returned an invalid result")
        return {
            "ok": True,
            "planner": "curobo",
            "reachable": result.reachable,
            "joints_rad": None if result.joints_rad is None else list(result.joints_rad),
            "joint_names": list(result.joint_names),
            "position_error_m": result.position_error_m,
            "rotation_error_rad": result.rotation_error_rad,
        }

    def plan_axis_linear(self, payload: object) -> dict[str, Any]:
        request = _request_from_mapping(PlanAxisLinearRequest, payload)
        result = self._with_request_scene(
            payload,
            lambda: self._tools.plan_axis_linear(request),
        )
        if not isinstance(result, NativeTrajectory):
            raise TypeError("plan_axis_linear returned an invalid result")
        return {
            "ok": True,
            "planner": "curobo_linear",
            "positions_rad": [list(row) for row in result.positions_rad],
            "joint_names": list(result.joint_names),
            "interpolation_dt_s": result.interpolation_dt_s,
            "trajectory_point_scale": result.trajectory_point_scale,
            "command_positions_rad": (
                None if result.command_positions_rad is None
                else [list(row) for row in result.command_positions_rad]
            ),
            "planning_time_s": result.planning_time_s,
            "position_error_m": result.position_error_m,
            "rotation_error_rad": result.rotation_error_rad,
            "scene_token": result.scene_token,
        }

    def validate_trajectory(self, payload: object) -> dict[str, Any]:
        request = _request_from_mapping(ValidateTrajectoryRequest, payload)
        if request.gripper_joint_positions_rad is None:
            raise _invalid_request(ValueError("final validation requires measured positions of both grippers"))
        result = self._with_request_scene(
            payload,
            lambda: self._tools.validate_trajectory(request),
        )
        if not isinstance(result, TrajectoryValidationResult):
            raise TypeError("validate_trajectory returned an invalid result")
        return {
            "ok": True,
            "collision_free": result.collision_free,
            "checked_point_count": result.checked_point_count,
            "first_invalid_index": result.first_invalid_index,
            "scene_token": result.scene_token,
            "gripper_joint_positions_rad": result.gripper_joint_positions_rad,
            "gripper_configuration_source": result.gripper_configuration_source,
            "binding_sha256": result.binding_sha256,
        }

    def validate_grasp_lift(self, payload):
        from .grasp_lift import normalize_lift

        try:
            if not isinstance(payload, dict) or set(payload) != {"request", "scene_snapshot", "attachment"}:
                raise ValueError("lift payload requires request, scene_snapshot and attachment")
            validated = normalize_lift(payload["request"], payload["scene_snapshot"], payload["attachment"])
        except (ValueError, TypeError, KeyError) as error:
            raise _invalid_request(error) from None
        return self._with_request_scene(validated, lambda: self._tools.validate_grasp_lift(validated))

    @staticmethod
    def _trajectory_payload(result: NativeTrajectory) -> dict[str, Any]:
        return {
            "positions_rad": [list(row) for row in result.positions_rad],
            "joint_names": list(result.joint_names),
            "interpolation_dt_s": result.interpolation_dt_s,
            "trajectory_point_scale": result.trajectory_point_scale,
            "command_positions_rad": (
                None if result.command_positions_rad is None
                else [list(row) for row in result.command_positions_rad]
            ),
            "planning_time_s": result.planning_time_s,
            "position_error_m": result.position_error_m,
            "rotation_error_rad": result.rotation_error_rad,
            "scene_token": result.scene_token,
        }

    def plan_grasp(self, payload: object) -> dict[str, Any]:
        request = _request_from_mapping(PlanGraspRequest, payload)
        operation = getattr(self._tools, "plan_grasp", None)
        if not callable(operation):
            raise TypeError("tool_api does not implement plan_grasp")
        result = self._with_request_scene(
            payload,
            lambda: operation(request),
        )
        if not isinstance(result, NativeGraspPlan):
            raise TypeError("plan_grasp returned an invalid result")
        return {
            "ok": True,
            "planner": "curobo_grasp",
            "phases": {
                "approach": self._trajectory_payload(result.approach),
                "grasp": self._trajectory_payload(result.grasp),
                "lift": self._trajectory_payload(result.lift),
            },
            "scene_token": result.scene_token,
        }

    def solve_ik_batch(self, payload: object) -> dict[str, Any]:
        request = _request_from_mapping(SolveIkBatchRequest, payload)
        result = self._with_request_scene(payload, lambda: self._tools.solve_ik_batch(request))
        if not isinstance(result, NativeIkBatchResult):
            raise TypeError("solve_ik_batch returned an invalid result")
        return result.to_payload()

    def plan_grasp_batch(self, payload: object) -> dict[str, Any]:
        request = _request_from_mapping(PlanGraspBatchRequest, payload)
        operation = getattr(self._tools, "plan_grasp_batch", None)
        if not callable(operation):
            raise TypeError("tool_api does not implement plan_grasp_batch")
        result = self._with_request_scene(payload, lambda: operation(request))
        if not isinstance(result, NativeGraspBatchPlan):
            raise TypeError("plan_grasp_batch returned an invalid result")
        items = []
        for item in result.items:
            items.append(
                {
                    "candidate_index": item.candidate_index,
                    "success": item.success,
                    "goalset_success": item.goalset_success,
                    "approach_success": item.approach_success,
                    "grasp_success": item.grasp_success,
                    "lift_success": item.lift_success,
                    "status": item.status,
                    "phases": {
                        phase: None if getattr(item, phase) is None else self._trajectory_payload(getattr(item, phase))
                        for phase in ("approach", "grasp", "lift")
                    },
                    "grasp_end_joints_rad": None if item.grasp_end_joints_rad is None else list(item.grasp_end_joints_rad),
                }
            )
        return {
            "ok": True,
            "planner": "curobo_grasp_batch",
            "items": items,
            "batch_size": result.batch_size,
            "batch_capacity": result.batch_capacity,
            "planning_time_s": result.planning_time_s,
            "scene_token": result.scene_token,
        }

    def invalidate_grasp_batch(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise _invalid_request(
                TypeError("batch invalidation payload must be a mapping")
            )
        arm = payload.get("arm")
        batch_size = payload.get("batch_size")
        trajectory_collision_clearance_m = payload.get(
            "trajectory_collision_clearance_m",
            0.02,
        )
        optimizer_collision_activation_distance_m = payload.get(
            "optimizer_collision_activation_distance_m",
            0.05,
        )
        if not isinstance(arm, str) or not arm.strip():
            raise _invalid_request(ValueError("arm is required"))
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise _invalid_request(ValueError("batch_size must be an integer"))
        operation = getattr(self._tools, "invalidate_grasp_batch", None)
        if not callable(operation):
            raise TypeError("tool_api does not implement invalidate_grasp_batch")
        invalidated = operation(
            arm.strip().upper(),
            batch_size,
            trajectory_collision_clearance_m,
            optimizer_collision_activation_distance_m,
        )
        if not isinstance(invalidated, bool):
            raise TypeError("invalidate_grasp_batch returned an invalid result")
        return {"ok": True, "invalidated": invalidated}

    def plan_place(self, payload: object) -> dict[str, Any]:
        request = _request_from_mapping(PlanPlaceRequest, payload)
        operation = getattr(self._tools, "plan_place", None)
        if not callable(operation):
            raise TypeError("tool_api does not implement plan_place")
        result = self._with_request_scene(payload, lambda: operation(request))
        if not isinstance(result, NativePlacePlan):
            raise TypeError("plan_place returned an invalid result")
        return {
            "ok": True,
            "planner": "curobo_place",
            "phases": {
                "approach": self._trajectory_payload(result.approach),
                "place": self._trajectory_payload(result.place),
                "retreat": self._trajectory_payload(result.retreat),
            },
            "scene_token": result.scene_token,
        }

    def segment_robot(self, payload: object) -> dict[str, Any]:
        operation = getattr(self._tools, "segment_robot", None)
        if not callable(operation):
            raise _error(
                "cuRobo robot segmentation is unavailable",
                stage="configuration",
                code="CUROBO_ROBOT_SEGMENTER_NOT_CONFIGURED",
                status=503,
            )
        request = _request_from_mapping(RobotSegmentationRequest, payload)
        include_base_pointcloud = payload.get("include_base_pointcloud", False)
        if not isinstance(include_base_pointcloud, bool):
            raise _error(
                "include_base_pointcloud must be a bool",
                stage="validation",
                code="CUROBO_INVALID_REQUEST",
                status=422,
            )
        result = operation(
            request,
            include_base_pointcloud=include_base_pointcloud,
        )
        if not isinstance(result, RobotSegmentationResult):
            raise TypeError("segment_robot returned an invalid result")
        return {
            "ok": True,
            "data_base64": base64.b64encode(result.pointcloud_data).decode("ascii"),
            "filtered_depth_png_base64": base64.b64encode(
                result.filtered_depth_png
            ).decode("ascii"),
            "base_pointcloud_xyz_base64": base64.b64encode(
                result.base_pointcloud_xyz
            ).decode("ascii"),
            "point_count": result.point_count,
            "removed_point_count": result.removed_point_count,
        }

    def begin_fixed_esdf_candidate(self, payload: object) -> dict[str, Any]:
        operation = getattr(self._tools, "begin_fixed_esdf_candidate", None)
        if not callable(operation):
            raise _error(
                "fixed ESDF mapping is unavailable",
                stage="configuration",
                code="CUROBO_FIXED_ESDF_NOT_CONFIGURED",
                status=503,
            )
        build_id = _fixed_esdf_build_id(payload)
        operation(build_id)
        return {"ok": True, "build_id": build_id}

    def prepare_semantic_fixed_esdf_candidate(
        self,
        payload: object,
    ) -> dict[str, Any]:
        operation = getattr(
            self._tools,
            "prepare_semantic_fixed_esdf_candidate",
            None,
        )
        if not callable(operation):
            raise _error(
                "semantic fixed ESDF mapping is unavailable",
                stage="configuration",
                code="CUROBO_FIXED_ESDF_NOT_CONFIGURED",
                status=503,
            )
        build_id = _fixed_esdf_build_id(payload)
        if not isinstance(payload, Mapping) or set(payload) != {"build_id", "request"}:
            raise _invalid_request(
                ValueError("semantic fixed ESDF payload is invalid")
            )
        try:
            request_payload = decode_semantic_esdf_request(payload.get("request"))
        except (TypeError, ValueError) as error:
            raise _invalid_request(error) from None
        result = operation(build_id, request_payload)
        if not isinstance(result, SemanticEsdfCandidate):
            raise TypeError("semantic fixed ESDF preparation returned an invalid result")
        integration = result.integration
        if not isinstance(integration, EsdfIntegrationResult) or integration.ready is not True:
            raise TypeError("semantic fixed ESDF candidate is not ready")
        primary_view = request_payload["views"]["3"]
        return {
            "ok": True,
            "build_id": build_id,
            "candidate_ready": True,
            "timings_ms": dict(result.timings_ms),
            "observation_set_id": result.observation_set_id,
            "selected_instance_id": result.selected_instance_id,
            "selection_generation": result.selection_generation,
            "point_count": integration.point_count,
            "removed_point_count": integration.removed_point_count,
            "integrated_frame_count": integration.integrated_frame_count,
            "scene_model": {
                key: dict(value) for key, value in result.scene_model.items()
            },
            "robot_joints_at_capture": primary_view["robot_joints_at_capture"],
            "bounds": {
                "center_m": list(result.bounds.center_m),
                "extent_m": list(result.bounds.extent_m),
            },
            "tabletop": {
                "height_m": float(result.table.height_m),
                "normal_xyz": list(result.table.normal_xyz),
                "inlier_count": int(result.table.inlier_count),
                "robot_clearance_m": float(result.table.robot_clearance_m),
                "support_plane": getattr(result.table, "support_plane", None),
            },
            "camera_metrics": {
                key: dict(value) for key, value in result.camera_metrics.items()
            },
            "instance_metrics": {
                key: dict(value) for key, value in result.instance_metrics.items()
            },
        }

    def integrate_fixed_esdf_candidate(self, payload: object) -> dict[str, Any]:
        operation = getattr(self._tools, "integrate_fixed_esdf_candidate", None)
        if not callable(operation):
            raise _error(
                "fixed ESDF mapping is unavailable",
                stage="configuration",
                code="CUROBO_FIXED_ESDF_NOT_CONFIGURED",
                status=503,
            )
        build_id = _fixed_esdf_build_id(payload)
        if "finalize" not in payload:
            raise _error(
                "finalize is required",
                stage="validation",
                code="CUROBO_INVALID_REQUEST",
                status=422,
            )
        request = _request_from_mapping(RobotSegmentationRequest, payload)
        finalize = payload.get("finalize")
        if not isinstance(finalize, bool):
            raise _error(
                "finalize must be a bool",
                stage="validation",
                code="CUROBO_INVALID_REQUEST",
                status=422,
            )
        result = operation(build_id, request, finalize=finalize)
        if not isinstance(result, EsdfIntegrationResult):
            raise TypeError("fixed ESDF integration returned an invalid result")
        return {
            "ok": True,
            "ready": result.ready,
            "data_base64": base64.b64encode(result.pointcloud_data).decode("ascii"),
            "point_count": result.point_count,
            "removed_point_count": result.removed_point_count,
            "revision": result.revision,
            "integrated_frame_count": result.integrated_frame_count,
            "scene_token": result.scene_token,
            "coordinate_frame": "base",
        }

    def set_fixed_esdf_candidate_scene(self, payload: object) -> dict[str, Any]:
        operation = getattr(self._tools, "set_fixed_esdf_candidate_scene", None)
        if not callable(operation):
            raise _error(
                "fixed ESDF semantic scene is unavailable",
                stage="configuration",
                code="CUROBO_FIXED_ESDF_NOT_CONFIGURED",
                status=503,
            )
        build_id = _fixed_esdf_build_id(payload)
        scene_model = payload.get("scene_model")
        metadata = payload.get("metadata")
        if not isinstance(scene_model, Mapping) or not isinstance(metadata, Mapping):
            raise _invalid_request(
                TypeError("fixed ESDF semantic scene payload is invalid")
            )
        operation(build_id, scene_model, metadata)
        return {"ok": True}

    def validate_fixed_esdf_candidate(self, payload: object) -> dict[str, Any]:
        operation = getattr(self._tools, "validate_fixed_esdf_candidate", None)
        if not callable(operation):
            raise _error(
                "fixed ESDF candidate validation is unavailable",
                stage="configuration",
                code="CUROBO_FIXED_ESDF_NOT_CONFIGURED",
                status=503,
            )
        build_id = _fixed_esdf_build_id(payload)
        request = _request_from_mapping(RobotSegmentationRequest, payload)
        result = operation(build_id, request)
        if not isinstance(result, FixedEsdfCandidateValidationResult):
            raise TypeError("fixed ESDF candidate validation returned an invalid result")
        return {
            "ok": True,
            "collision_free": result.collision_free,
            "checked_arms": list(result.checked_arms),
            "colliding_arm": result.colliding_arm,
        }

    def publish_fixed_esdf_candidate(self, payload: object) -> dict[str, Any]:
        operation = getattr(self._tools, "publish_fixed_esdf_candidate", None)
        if not callable(operation):
            raise _error(
                "fixed ESDF mapping is unavailable",
                stage="configuration",
                code="CUROBO_FIXED_ESDF_NOT_CONFIGURED",
                status=503,
            )
        build_id = _fixed_esdf_build_id(payload)
        result = operation(build_id)
        if not isinstance(result, EsdfIntegrationResult):
            raise TypeError("fixed ESDF publish returned an invalid result")
        return {
            "ok": True,
            "ready": result.ready,
            "surface_voxel_size_m": result.surface_voxel_size_m,
            "data_base64": base64.b64encode(result.pointcloud_data).decode("ascii"),
            "point_count": result.point_count,
            "removed_point_count": result.removed_point_count,
            "revision": result.revision,
            "integrated_frame_count": result.integrated_frame_count,
            "scene_token": result.scene_token,
            "coordinate_frame": "base",
            "build_id": build_id,
        }

    def reconcile_fixed_esdf_candidate(self, payload: object) -> dict[str, Any]:
        operation = getattr(self._tools, "reconcile_fixed_esdf_candidate", None)
        if not callable(operation):
            raise _error(
                "fixed ESDF reconciliation is unavailable",
                stage="configuration",
                code="CUROBO_FIXED_ESDF_NOT_CONFIGURED",
                status=503,
            )
        build_id = _fixed_esdf_build_id(payload)
        result = operation(build_id)
        if result is None:
            return {"ok": True, "published": False, "build_id": build_id}
        if not isinstance(result, EsdfIntegrationResult):
            raise TypeError("fixed ESDF reconciliation returned an invalid result")
        return {
            "ok": True,
            "published": True,
            "ready": result.ready,
            "surface_voxel_size_m": result.surface_voxel_size_m,
            "data_base64": base64.b64encode(result.pointcloud_data).decode("ascii"),
            "point_count": result.point_count,
            "removed_point_count": result.removed_point_count,
            "revision": result.revision,
            "integrated_frame_count": result.integrated_frame_count,
            "scene_token": result.scene_token,
            "coordinate_frame": "base",
            "build_id": build_id,
        }

    def discard_fixed_esdf_candidate(self, payload: object) -> dict[str, Any]:
        operation = getattr(self._tools, "discard_fixed_esdf_candidate", None)
        if not callable(operation):
            raise _error(
                "fixed ESDF mapping is unavailable",
                stage="configuration",
                code="CUROBO_FIXED_ESDF_NOT_CONFIGURED",
                status=503,
            )
        build_id = _fixed_esdf_build_id(payload)
        if "expected_scene_token" in payload:
            token = payload["expected_scene_token"]
            if not isinstance(token, str) or not token.startswith("fixed-esdf:"):
                raise ValueError("expected_scene_token must identify a published fixed ESDF")
            cleared = operation(build_id, expected_scene_token=token)
            return {"ok": True, "build_id": build_id, "cleared_published": cleared is True}
        operation(build_id)
        return {"ok": True, "build_id": build_id}

    def select_fixed_esdf_instance(self, payload: object) -> dict[str, Any]:
        operation = getattr(self._tools, "select_fixed_esdf_instance", None)
        if not callable(operation):
            raise _error(
                "semantic ESDF selection is unavailable",
                stage="configuration",
                code="CUROBO_FIXED_ESDF_NOT_CONFIGURED",
                status=503,
            )
        if not isinstance(payload, Mapping):
            raise _invalid_request(
                TypeError("semantic ESDF selection must be a mapping")
            )
        expected_revision = payload.get("expected_revision")
        selected_instance_id = payload.get("selected_instance_id")
        selection_generation = payload.get("selection_generation")
        if (
            type(expected_revision) is not int
            or expected_revision < 1
            or not isinstance(selected_instance_id, str)
            or not selected_instance_id.strip()
            or type(selection_generation) is not int
            or selection_generation < 1
        ):
            raise _invalid_request(
                ValueError("semantic ESDF selection payload is invalid")
            )
        result = operation(
            expected_revision,
            selected_instance_id.strip(),
            selection_generation,
        )
        if not isinstance(result, EsdfIntegrationResult):
            raise TypeError("semantic ESDF selection returned an invalid result")
        return {
            "ok": True,
            "ready": result.ready,
            "revision": result.revision,
            "integrated_frame_count": result.integrated_frame_count,
            "point_count": result.point_count,
            "scene_token": result.scene_token,
            "coordinate_frame": "base",
            "selected_instance_id": selected_instance_id.strip(),
            "selection_generation": selection_generation,
        }


def _caused_by_cuda_oom(error: BaseException) -> bool:
    # The GPU runtime loads torch; error reporting must not initialize CUDA.
    oom_type = getattr(sys.modules.get("torch"), "OutOfMemoryError", None)
    if not isinstance(oom_type, type):
        return False
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, oom_type):
            return True
        error = error.__cause__ or error.__context__
    return False


def _run(operation):
    try:
        return 200, operation()
    except Exception as error:
        if _caused_by_cuda_oom(error):
            LOGGER.exception("cuRobo operation exhausted CUDA memory")
            typed = _error(
                "GPU 显存不足，cuRobo 未完成计算；请释放显存后重新计划。这不代表目标位姿无解。",
                stage="resources", code="CUROBO_CUDA_OUT_OF_MEMORY", status=503,
            )
            return typed.status, {"ok": False, **typed.to_dict()}
        if isinstance(error, CuroboToolError):
            if error.__cause__ is not None:
                LOGGER.exception("cuRobo operation returned a wrapped internal error")
            else:
                LOGGER.warning("cuRobo operation rejected: %s", error.to_dict())
            return error.status, {"ok": False, **error.to_dict()}
        LOGGER.exception("unhandled cuRobo service operation error")
        typed = _error(
            "cuRobo operation failed",
            stage="internal",
            code="CUROBO_INTERNAL_ERROR",
            status=500,
        )
        return typed.status, {"ok": False, **typed.to_dict()}


def _fixed_esdf_build_id(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise _error(
            "fixed ESDF payload must be a mapping",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    build_id = payload.get("build_id")
    if (
        not isinstance(build_id, str)
        or len(build_id) != 32
        or any(character not in "0123456789abcdef" for character in build_id)
    ):
        raise _error(
            "build_id must be 32 lowercase hexadecimal characters",
            stage="validation",
            code="CUROBO_INVALID_REQUEST",
            status=422,
        )
    return build_id


def _handler_type(registry: PlannerRegistry):
    class Handler(BaseHTTPRequestHandler):
        server_version = "JodellCuroboPlanner/1.0"

        def do_GET(self):
            if self.path.rstrip("/") != "/health":
                self._send(404, {"ok": False, "error": {"code": "NOT_FOUND"}})
                return
            self._send(*_run(registry.health))

        def do_POST(self):
            operation = {
                "/tools/plan_to_pose": registry.plan_to_pose,
                "/tools/plan_grasp": registry.plan_grasp,
                "/tools/plan_grasp_batch": registry.plan_grasp_batch,
                "/tools/solve_ik_batch": registry.solve_ik_batch,
                "/tools/plan_grasp_batch/invalidate": registry.invalidate_grasp_batch,
                "/tools/plan_place": registry.plan_place,
                "/tools/plan_to_joint": registry.plan_to_joint,
                "/tools/plan_dual_joint": registry.plan_dual_joint,
                "/tools/plan_dual_pose": registry.plan_dual_pose,
                "/tools/validate_dual_trajectory": registry.validate_dual_trajectory,
                "/tools/plan_axis_linear": registry.plan_axis_linear,
                "/tools/solve_ik": registry.solve_ik,
                "/tools/validate_trajectory": registry.validate_trajectory,
                "/tools/validate_grasp_lift": registry.validate_grasp_lift,
                "/tools/segment_robot": registry.segment_robot,
                "/tools/fixed_esdf/begin": registry.begin_fixed_esdf_candidate,
                "/tools/fixed_esdf/semantic": registry.prepare_semantic_fixed_esdf_candidate,
                "/tools/fixed_esdf/integrate": registry.integrate_fixed_esdf_candidate,
                "/tools/fixed_esdf/scene": registry.set_fixed_esdf_candidate_scene,
                "/tools/fixed_esdf/validate": registry.validate_fixed_esdf_candidate,
                "/tools/fixed_esdf/publish": registry.publish_fixed_esdf_candidate,
                "/tools/fixed_esdf/reconcile": registry.reconcile_fixed_esdf_candidate,
                "/tools/fixed_esdf/discard": registry.discard_fixed_esdf_candidate,
                "/tools/fixed_esdf/select": registry.select_fixed_esdf_instance,
            }.get(self.path.rstrip("/"))
            if operation is None:
                self._send(404, {"ok": False, "error": {"code": "NOT_FOUND"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self._send(400, {"ok": False, "error": {"code": "INVALID_JSON"}})
                return
            self._send(*_run(lambda: operation(payload)))

        def _send(self, status: int, payload: Mapping[str, Any]):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    return Handler


def serve(registry: PlannerRegistry, *, host="127.0.0.1", port=8010):
    server = ThreadingHTTPServer((host, int(port)), _handler_type(registry))
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger(__package__ + "._curobo_impl").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="JODELL cuRobo planning service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--scene", default="pickplace")
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--marvin-meshes", type=Path, required=True)
    parser.add_argument("--rag-meshes", type=Path, required=True)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    urdf = materialize_urdf(
        args.urdf, marvin_meshes=args.marvin_meshes, rag_meshes=args.rag_meshes
    )
    registry = PlannerRegistry(
        robot_config=args.robot_config,
        urdf=urdf,
        asset_root=urdf.parent,
        scene=args.scene,
        warmup=not args.no_warmup,
    )
    print(f"{SERVICE_NAME} ready on http://{args.host}:{args.port}", flush=True)
    serve(registry, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""GPU-free client for the isolated cuRobo service."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen as stdlib_urlopen

from .contracts import (
    SolveIkBatchRequest,
    NativeIkBatchResult,
    FixedEsdfCandidateValidationResult,
    NativeGraspBatchItem,
    NativeGraspBatchPlan,
    NativeIkResult,
    NativeGraspPlan,
    NativePlacePlan,
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
from .semantic_esdf_transport import encode_semantic_esdf_request
from .dual_contracts import validate_plan_result


DEFAULT_PLANNER_URL = "http://127.0.0.1:8010"
DEFAULT_TIMEOUT_S = 120.0


class CuroboServiceError(RuntimeError):
    def __init__(self, message: str, *, code="CUROBO_SERVICE_ERROR", status=None,
                 stage="service", retryable=False, details=None):
        super().__init__(message)
        self.code, self.status, self.stage = str(code), status, str(stage)
        self.retryable = bool(retryable)
        self.details = dict(details) if isinstance(details, Mapping) else {}


def _protocol(message: str, status=None):
    return CuroboServiceError(
        message, code="CUROBO_INVALID_RESPONSE", stage="protocol", status=status
    )


def _validated_build_id(build_id: object) -> str:
    value = str(build_id) if isinstance(build_id, str) else ""
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("build_id must be 32 lowercase hexadecimal characters")
    return value


class CuroboPlannerClient:
    def __init__(self, *, base_url: str | None = None, timeout_s=DEFAULT_TIMEOUT_S,
                 urlopen: Callable[..., Any] = stdlib_urlopen):
        self.base_url = str(
            base_url
            or os.environ.get("JODELL_CUROBO_URL")
            or os.environ.get("CUROBO_PLANNER_URL")
            or DEFAULT_PLANNER_URL
        ).rstrip("/")
        self.timeout_s = float(timeout_s)
        if not self.base_url or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("cuRobo service URL and a positive timeout are required")
        self._urlopen = urlopen

    def health(self) -> Mapping[str, Any]:
        return self._request("GET", "/health", timeout_s=min(self.timeout_s, 1.0))

    @staticmethod
    def _with_scene(payload: dict[str, Any], scene_snapshot) -> dict[str, Any]:
        if scene_snapshot is None:
            return payload
        if not isinstance(scene_snapshot, Mapping):
            raise TypeError("scene_snapshot must be a mapping")
        payload["scene_snapshot"] = json.loads(
            json.dumps(scene_snapshot, allow_nan=False)
        )
        return payload

    def plan_dual_joint(self, request, *, scene_snapshot=None):
        body = self._request("POST", "/tools/plan_dual_joint",
                             self._with_scene(request.to_mapping(), scene_snapshot))
        try:
            if body.get("trajectory_point_scale") != request.trajectory_point_scale or body.get("sample_dt_s") != 1. / request.freq_hz:
                raise ValueError("dual result timing does not match request")
            return validate_plan_result(body)
        except (TypeError, ValueError) as error:
            raise _protocol("invalid synchronized dual trajectory") from error

    def plan_dual_pose(self, request, *, scene_snapshot=None):
        body = self._request("POST", "/tools/plan_dual_pose",
                             self._with_scene(request.to_mapping(), scene_snapshot))
        try:
            if body.get("trajectory_point_scale") != request.trajectory_point_scale or body.get("sample_dt_s") != 1. / request.freq_hz:
                raise ValueError("dual result timing does not match request")
            return validate_plan_result(body)
        except (TypeError, ValueError) as error:
            raise _protocol("invalid synchronized dual trajectory") from error

    def validate_dual_trajectory(self, request, *, scene_snapshot=None):
        body = self._request("POST", "/tools/validate_dual_trajectory",
                             self._with_scene(request.to_mapping(), scene_snapshot))
        result = TrajectoryValidationResult.create(
            collision_free=body.get("collision_free"),
            checked_point_count=body.get("checked_point_count"),
            first_invalid_index=body.get("first_invalid_index"),
            scene_token=body.get("scene_token"),
        )
        return {"ok": True, "collision_free": result.collision_free,
                "checked_point_count": result.checked_point_count,
                "first_invalid_index": result.first_invalid_index, "scene_token": result.scene_token}

    def plan_pose(
        self,
        request: PlanToPoseRequest,
        *,
        scene_snapshot=None,
    ) -> NativeTrajectory:
        if not isinstance(request, PlanToPoseRequest):
            raise TypeError("request must be a PlanToPoseRequest")
        payload = self._with_scene({
            "arm": request.arm,
            "start_joints_deg": list(request.start_joints_deg),
            "other_joints_deg": list(request.other_joints_deg),
            "goal_position_m": list(request.goal_position_m),
            "goal_quaternion_wxyz": list(request.goal_quaternion_wxyz),
            "freq_hz": request.freq_hz,
            "max_attempts": request.max_attempts,
            "trajectory_point_scale": request.trajectory_point_scale,
        }, scene_snapshot)
        body = self._request("POST", "/tools/plan_to_pose", payload)
        if body.get("planner") != "curobo":
            raise _protocol("cuRobo service returned an invalid planner")
        try:
            return NativeTrajectory(
                positions_rad=body.get("positions_rad"),
                joint_names=body.get("joint_names"),
                interpolation_dt_s=body.get("interpolation_dt_s"),
                trajectory_point_scale=body.get("trajectory_point_scale"),
                command_positions_rad=body.get("command_positions_rad"),
                planning_time_s=body.get("planning_time_s"),
                position_error_m=body.get("position_error_m"),
                rotation_error_rad=body.get("rotation_error_rad"),
                scene_token=body.get("scene_token"),
            )
        except (TypeError, ValueError) as error:
            raise _protocol("cuRobo service returned an invalid trajectory") from error

    @staticmethod
    def _native_trajectory(value: object, *, phase: str) -> NativeTrajectory:
        if not isinstance(value, Mapping):
            raise _protocol(f"cuRobo service returned an invalid {phase} trajectory")
        try:
            return NativeTrajectory(
                positions_rad=value.get("positions_rad"),
                joint_names=value.get("joint_names"),
                interpolation_dt_s=value.get("interpolation_dt_s"),
                trajectory_point_scale=value.get("trajectory_point_scale"),
                command_positions_rad=value.get("command_positions_rad"),
                planning_time_s=value.get("planning_time_s"),
                position_error_m=value.get("position_error_m"),
                rotation_error_rad=value.get("rotation_error_rad"),
                scene_token=value.get("scene_token"),
            )
        except (TypeError, ValueError) as error:
            raise _protocol(
                f"cuRobo service returned an invalid {phase} trajectory"
            ) from error

    def plan_grasp(
        self,
        request: PlanGraspRequest,
        *,
        scene_snapshot=None,
    ) -> NativeGraspPlan:
        if not isinstance(request, PlanGraspRequest):
            raise TypeError("request must be a PlanGraspRequest")
        payload = self._with_scene(
            {
                "arm": request.arm,
                "start_joints_deg": list(request.start_joints_deg),
                "other_joints_deg": list(request.other_joints_deg),
                "goal_position_m": list(request.goal_position_m),
                "goal_quaternion_wxyz": list(request.goal_quaternion_wxyz),
                "freq_hz": request.freq_hz,
                "max_attempts": request.max_attempts,
                "trajectory_point_scale": request.trajectory_point_scale,
                "approach_axis": request.approach_axis,
                "approach_offset_m": request.approach_offset_m,
                "approach_in_tool_frame": request.approach_in_tool_frame,
                "lift_axis": request.lift_axis,
                "lift_offset_m": request.lift_offset_m,
                "lift_in_tool_frame": request.lift_in_tool_frame,
                "trajectory_collision_clearance_m": request.trajectory_collision_clearance_m,
                "optimizer_collision_activation_distance_m": request.optimizer_collision_activation_distance_m,
            },
            scene_snapshot,
        )
        body = self._request("POST", "/tools/plan_grasp", payload)
        if body.get("planner") != "curobo_grasp":
            raise _protocol("cuRobo service returned an invalid grasp planner")
        phases = body.get("phases")
        if not isinstance(phases, Mapping):
            raise _protocol("cuRobo service returned invalid grasp phases")
        return NativeGraspPlan.create(
            approach=self._native_trajectory(phases.get("approach"), phase="approach"),
            grasp=self._native_trajectory(phases.get("grasp"), phase="grasp"),
            lift=self._native_trajectory(phases.get("lift"), phase="lift"),
        )

    def solve_ik_batch(self, request: SolveIkBatchRequest, *, scene_snapshot=None):
        if not isinstance(request, SolveIkBatchRequest):
            raise TypeError("request must be a SolveIkBatchRequest")
        payload = self._with_scene({
            "arm": request.arm, "start_joints_deg": list(request.start_joints_deg),
            "other_joints_deg": list(request.other_joints_deg),
            "goal_matrices_world": [[list(row) for row in matrix] for matrix in request.goal_matrices_world],
            "scene_token": request.scene_token,
            "trajectory_collision_clearance_m": request.trajectory_collision_clearance_m,
        }, scene_snapshot)
        body = self._request("POST", "/tools/solve_ik_batch", payload)
        try:
            result = NativeIkBatchResult.from_mapping(body)
            if body.get("planner") != "curobo_ik_batch" or len(result.items) != len(request.goal_matrices_world) or result.scene_token != request.scene_token:
                raise ValueError("IK batch metadata mismatch")
            return result
        except (TypeError, ValueError) as error:
            raise _protocol("cuRobo service returned an invalid IK batch result") from error

    def plan_grasp_batch(
        self,
        request: PlanGraspBatchRequest,
        *,
        scene_snapshot=None,
    ) -> NativeGraspBatchPlan:
        if not isinstance(request, PlanGraspBatchRequest):
            raise TypeError("request must be a PlanGraspBatchRequest")
        payload = self._with_scene(
            {
                "arm": request.arm,
                "start_joints_deg": list(request.start_joints_deg),
                "other_joints_deg": list(request.other_joints_deg),
                "goal_matrices_world": [
                    [list(row) for row in matrix]
                    for matrix in request.goal_matrices_world
                ],
                "freq_hz": request.freq_hz,
                "approach_axis": request.approach_axis,
                "approach_offset_m": request.approach_offset_m,
                "approach_in_tool_frame": request.approach_in_tool_frame,
                "lift_axis": request.lift_axis,
                "lift_offset_m": request.lift_offset_m,
                "lift_in_tool_frame": request.lift_in_tool_frame,
                "scene_token": request.scene_token,
                "trajectory_collision_clearance_m": request.trajectory_collision_clearance_m,
                "optimizer_collision_activation_distance_m": request.optimizer_collision_activation_distance_m,
            },
            scene_snapshot,
        )
        body = self._request("POST", "/tools/plan_grasp_batch", payload)
        if body.get("planner") != "curobo_grasp_batch":
            raise _protocol("cuRobo service returned an invalid batch grasp planner")
        raw_items = body.get("items")
        if not isinstance(raw_items, list) or len(raw_items) > 200:
            raise _protocol("cuRobo service returned invalid batch grasp items")
        try:
            items = []
            for raw_item in raw_items:
                if not isinstance(raw_item, Mapping):
                    raise ValueError("invalid batch item")
                phases = raw_item.get("phases")
                if not isinstance(phases, Mapping):
                    raise ValueError("invalid batch phases")
                parsed_phases = {}
                for phase in ("approach", "grasp", "lift"):
                    value = phases.get(phase)
                    parsed_phases[phase] = None if value is None else self._native_trajectory(value, phase=phase)
                items.append(
                    NativeGraspBatchItem.create(
                        candidate_index=raw_item.get("candidate_index"),
                        success=raw_item.get("success"),
                        goalset_success=raw_item.get("goalset_success"),
                        approach_success=raw_item.get("approach_success"),
                        grasp_success=raw_item.get("grasp_success"),
                        lift_success=raw_item.get("lift_success"),
                        status=raw_item.get("status"),
                        approach=parsed_phases["approach"],
                        grasp=parsed_phases["grasp"],
                        lift=parsed_phases["lift"],
                        grasp_end_joints_rad=raw_item.get("grasp_end_joints_rad"),
                    )
                )
            return NativeGraspBatchPlan.create(
                items=tuple(items),
                batch_size=body.get("batch_size"),
                batch_capacity=body.get("batch_capacity"),
                planning_time_s=body.get("planning_time_s"),
                scene_token=body.get("scene_token"),
            )
        except (TypeError, ValueError, CuroboServiceError) as error:
            if isinstance(error, CuroboServiceError):
                raise
            raise _protocol("cuRobo service returned an invalid batch grasp result") from error

    def invalidate_grasp_batch(
        self,
        arm: str,
        batch_size: int,
        trajectory_collision_clearance_m: float = 0.01,
        optimizer_collision_activation_distance_m: float = 0.03,
    ) -> bool:
        if not isinstance(arm, str) or not arm.strip():
            raise TypeError("arm must be a non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        body = self._request(
            "POST",
            "/tools/plan_grasp_batch/invalidate",
            {
                "arm": arm.strip().upper(),
                "batch_size": batch_size,
                "trajectory_collision_clearance_m": trajectory_collision_clearance_m,
                "optimizer_collision_activation_distance_m": optimizer_collision_activation_distance_m,
            },
        )
        invalidated = body.get("invalidated")
        if not isinstance(invalidated, bool):
            raise _protocol("cuRobo service returned invalid batch invalidation")
        return invalidated

    def plan_place(
        self,
        request: PlanPlaceRequest,
        *,
        scene_snapshot=None,
    ) -> NativePlacePlan:
        if not isinstance(request, PlanPlaceRequest):
            raise TypeError("request must be a PlanPlaceRequest")
        payload = self._with_scene(
            {
                "arm": request.arm,
                "start_joints_deg": list(request.start_joints_deg),
                "other_joints_deg": list(request.other_joints_deg),
                "goal_position_m": list(request.goal_position_m),
                "goal_quaternion_wxyz": list(request.goal_quaternion_wxyz),
                "freq_hz": request.freq_hz,
                "max_attempts": request.max_attempts,
                "trajectory_point_scale": request.trajectory_point_scale,
                "approach_axis": request.approach_axis,
                "approach_offset_m": request.approach_offset_m,
                "approach_in_tool_frame": request.approach_in_tool_frame,
                "lift_axis": request.lift_axis,
                "lift_offset_m": request.lift_offset_m,
                "lift_in_tool_frame": request.lift_in_tool_frame,
                "trajectory_collision_clearance_m": request.trajectory_collision_clearance_m,
                "optimizer_collision_activation_distance_m": request.optimizer_collision_activation_distance_m,
            },
            scene_snapshot,
        )
        body = self._request("POST", "/tools/plan_place", payload)
        if body.get("planner") != "curobo_place":
            raise _protocol("cuRobo service returned an invalid place planner")
        phases = body.get("phases")
        if not isinstance(phases, Mapping):
            raise _protocol("cuRobo service returned invalid place phases")
        return NativePlacePlan.create(
            approach=self._native_trajectory(
                phases.get("approach"), phase="approach"
            ),
            place=self._native_trajectory(phases.get("place"), phase="place"),
            retreat=self._native_trajectory(
                phases.get("retreat"), phase="retreat"
            ),
        )

    def plan_joint(
        self,
        request: PlanToJointRequest,
        *,
        scene_snapshot=None,
    ) -> NativeTrajectory:
        if not isinstance(request, PlanToJointRequest):
            raise TypeError("request must be a PlanToJointRequest")
        payload = self._with_scene(
            {
                "arm": request.arm,
                "start_joints_deg": list(request.start_joints_deg),
                "other_joints_deg": list(request.other_joints_deg),
                "goal_joints_deg": list(request.goal_joints_deg),
                "freq_hz": request.freq_hz,
                "max_attempts": request.max_attempts,
                "trajectory_point_scale": request.trajectory_point_scale,
            },
            scene_snapshot,
        )
        body = self._request("POST", "/tools/plan_to_joint", payload)
        if body.get("planner") != "curobo":
            raise _protocol("cuRobo service returned an invalid planner")
        try:
            return NativeTrajectory(
                positions_rad=body.get("positions_rad"),
                joint_names=body.get("joint_names"),
                interpolation_dt_s=body.get("interpolation_dt_s"),
                trajectory_point_scale=body.get("trajectory_point_scale"),
                command_positions_rad=body.get("command_positions_rad"),
                planning_time_s=body.get("planning_time_s"),
                position_error_m=body.get("position_error_m"),
                rotation_error_rad=body.get("rotation_error_rad"),
                scene_token=body.get("scene_token"),
            )
        except (TypeError, ValueError) as error:
            raise _protocol(
                "cuRobo service returned an invalid joint trajectory"
            ) from error

    def plan_axis_linear(
        self,
        request: PlanAxisLinearRequest,
        *,
        scene_snapshot=None,
    ) -> NativeTrajectory:
        if not isinstance(request, PlanAxisLinearRequest):
            raise TypeError("request must be a PlanAxisLinearRequest")
        body = self._request(
            "POST",
            "/tools/plan_axis_linear",
            self._with_scene({
                "arm": request.arm,
                "start_joints_deg": list(request.start_joints_deg),
                "other_joints_deg": list(request.other_joints_deg),
                "reference_frame": request.reference_frame,
                "axis": request.axis,
                "distance_m": request.distance_m,
                "freq_hz": request.freq_hz,
                "max_attempts": request.max_attempts,
                "trajectory_point_scale": request.trajectory_point_scale,
            }, scene_snapshot),
        )
        if body.get("planner") != "curobo_linear":
            raise _protocol("cuRobo service returned an invalid linear planner")
        try:
            return NativeTrajectory(
                positions_rad=body.get("positions_rad"),
                joint_names=body.get("joint_names"),
                interpolation_dt_s=body.get("interpolation_dt_s"),
                trajectory_point_scale=body.get("trajectory_point_scale"),
                command_positions_rad=body.get("command_positions_rad"),
                planning_time_s=body.get("planning_time_s"),
                position_error_m=body.get("position_error_m"),
                rotation_error_rad=body.get("rotation_error_rad"),
                scene_token=body.get("scene_token"),
            )
        except (TypeError, ValueError) as error:
            raise _protocol(
                "cuRobo service returned an invalid linear trajectory"
            ) from error

    def solve_ik(
        self,
        request: SolveIkRequest,
        *,
        scene_snapshot=None,
    ) -> NativeIkResult:
        if not isinstance(request, SolveIkRequest):
            raise TypeError("request must be a SolveIkRequest")
        payload = self._with_scene({
            "arm": request.arm,
            "current_joints_deg": list(request.current_joints_deg),
            "other_joints_deg": list(request.other_joints_deg),
            "goal_position_m": list(request.goal_position_m),
            "goal_quaternion_wxyz": list(request.goal_quaternion_wxyz),
        }, scene_snapshot)
        body = self._request("POST", "/tools/solve_ik", payload)
        if body.get("planner") != "curobo":
            raise _protocol("cuRobo service returned an invalid planner")
        try:
            return NativeIkResult(
                reachable=body.get("reachable"),
                joints_rad=body.get("joints_rad"),
                joint_names=body.get("joint_names"),
                position_error_m=body.get("position_error_m"),
                rotation_error_rad=body.get("rotation_error_rad"),
            )
        except (TypeError, ValueError) as error:
            raise _protocol("cuRobo service returned an invalid IK result") from error

    def validate_trajectory(
        self,
        request: ValidateTrajectoryRequest,
        *,
        scene_snapshot=None,
    ) -> TrajectoryValidationResult:
        if not isinstance(request, ValidateTrajectoryRequest):
            raise TypeError("request must be a ValidateTrajectoryRequest")
        payload = {
            "arm": request.arm,
            "points_deg": [list(point) for point in request.points_deg],
            "other_joints_deg": list(request.other_joints_deg),
            "gripper_joint_positions_rad": request.gripper_joint_positions_rad,
            "gripper_configuration_source": request.gripper_configuration_source,
        }
        if request.expected_scene_token is not None:
            payload["expected_scene_token"] = request.expected_scene_token
        if request.grasp_phase is not None:
            payload["grasp_phase"] = request.grasp_phase
        payload["trajectory_collision_clearance_m"] = (
            request.trajectory_collision_clearance_m
        )
        payload["optimizer_collision_activation_distance_m"] = (
            request.optimizer_collision_activation_distance_m
        )
        payload = self._with_scene(payload, scene_snapshot)
        body = self._request(
            "POST",
            "/tools/validate_trajectory",
            payload,
        )
        try:
            result = TrajectoryValidationResult.create(
                collision_free=body.get("collision_free"),
                checked_point_count=body.get("checked_point_count"),
                first_invalid_index=body.get("first_invalid_index"),
                scene_token=body.get("scene_token"),
                gripper_joint_positions_rad=body.get("gripper_joint_positions_rad"),
                gripper_configuration_source=body.get("gripper_configuration_source"),
                binding_sha256=body.get("binding_sha256"),
            )
            if request.gripper_joint_positions_rad is not None:
                from .trajectory_validation import require_measured_result
                require_measured_result(result, request)
            return result
        except (TypeError, ValueError) as error:
            raise _protocol(
                "cuRobo service returned an invalid trajectory validation result"
            ) from error

    def validate_grasp_lift(self, payload):
        """GaP SI payload; response binding is checked by the GaP adapter."""
        return self._request("POST", "/tools/validate_grasp_lift", payload)

    def segment_robot(
        self,
        payload: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        body = self._request(
            "POST",
            "/tools/segment_robot",
            dict(payload),
            timeout_s=timeout_s,
        )
        data_base64 = body.get("data_base64")
        filtered_depth_png_base64 = body.get("filtered_depth_png_base64")
        base_pointcloud_xyz_base64 = body.get("base_pointcloud_xyz_base64")
        point_count = body.get("point_count")
        removed_point_count = body.get("removed_point_count")
        if (
            not isinstance(data_base64, str)
            or not isinstance(filtered_depth_png_base64, str)
            or not isinstance(base_pointcloud_xyz_base64, str)
            or not isinstance(point_count, int)
            or point_count < 0
            or not isinstance(removed_point_count, int)
            or removed_point_count < 0
        ):
            raise _protocol(
                "cuRobo service returned an invalid robot segmentation result"
            )
        return {
            "ok": True,
            "data_base64": data_base64,
            "filtered_depth_png_base64": filtered_depth_png_base64,
            "base_pointcloud_xyz_base64": base_pointcloud_xyz_base64,
            "point_count": point_count,
            "removed_point_count": removed_point_count,
        }

    @staticmethod
    def _esdf_response(body: Mapping[str, Any], *, include_data: bool) -> dict[str, Any]:
        ready = body.get("ready")
        revision = body.get("revision")
        frame_count = body.get("integrated_frame_count")
        point_count = body.get("point_count")
        scene_token = body.get("scene_token")
        if (
            not isinstance(ready, bool)
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
            or isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 0
            or isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count < 0
            or (scene_token is not None and not isinstance(scene_token, str))
        ):
            raise _protocol("cuRobo service returned an invalid ESDF status")
        result = {
            "ok": True,
            "ready": ready,
            "revision": revision,
            "integrated_frame_count": frame_count,
            "point_count": point_count,
            "scene_token": scene_token,
            "coordinate_frame": body.get("coordinate_frame", "base"),
        }
        if include_data:
            encoded = body.get("data_base64")
            removed = body.get("removed_point_count")
            if not isinstance(encoded, str) or not isinstance(removed, int) or removed < 0:
                raise _protocol("cuRobo service returned an invalid ESDF snapshot")
            result["data_base64"] = encoded
            result["removed_point_count"] = removed
            voxel_size = body.get("surface_voxel_size_m")
            if voxel_size is not None:
                if (isinstance(voxel_size, bool) or not isinstance(voxel_size, (int, float))
                        or not math.isfinite(voxel_size) or voxel_size <= 0):
                    raise _protocol("cuRobo service returned an invalid surface voxel size")
                result["surface_voxel_size_m"] = float(voxel_size)
        return result

    def begin_fixed_esdf_candidate(
        self,
        build_id: str,
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        value = _validated_build_id(build_id)
        return self._request(
            "POST",
            "/tools/fixed_esdf/begin",
            {"build_id": value},
            timeout_s=timeout_s,
        )

    def prepare_semantic_fixed_esdf_candidate(
        self,
        build_id: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        value = _validated_build_id(build_id)
        encoded = encode_semantic_esdf_request(payload)
        body = self._request(
            "POST",
            "/tools/fixed_esdf/semantic",
            {"build_id": value, "request": encoded},
            timeout_s=timeout_s,
        )
        if (
            body.get("build_id") != value
            or body.get("candidate_ready") is not True
            or body.get("observation_set_id") != payload.get("observation_set_id")
            or body.get("selected_instance_id")
            != payload.get("confirmed_instance_id")
            or body.get("selection_generation")
            != payload.get("selection_generation")
            or type(body.get("point_count")) is not int
            or body["point_count"] < 0
            or type(body.get("removed_point_count")) is not int
            or body["removed_point_count"] < 0
            or body.get("integrated_frame_count") != 3
            or not isinstance(body.get("scene_model"), Mapping)
            or not isinstance(body.get("bounds"), Mapping)
            or not isinstance(body.get("tabletop"), Mapping)
            or not isinstance(body.get("camera_metrics"), Mapping)
            or not isinstance(body.get("instance_metrics"), Mapping)
        ):
            raise _protocol(
                "cuRobo service returned an invalid semantic ESDF candidate"
            )
        return dict(body)

    def integrate_fixed_esdf_candidate(
        self,
        build_id: str,
        payload: Mapping[str, Any],
        *,
        finalize: bool,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if not isinstance(finalize, bool):
            raise TypeError("finalize must be a bool")
        request = dict(payload)
        request["build_id"] = _validated_build_id(build_id)
        request["finalize"] = finalize
        return self._esdf_response(
            self._request(
                "POST",
                "/tools/fixed_esdf/integrate",
                request,
                timeout_s=timeout_s,
            ),
            include_data=True,
        )

    def set_fixed_esdf_candidate_scene(
        self,
        build_id: str,
        scene_model: Mapping[str, Any],
        metadata: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(scene_model, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("scene_model and metadata must be mappings")
        return self._request(
            "POST",
            "/tools/fixed_esdf/scene",
            {
                "build_id": _validated_build_id(build_id),
                "scene_model": dict(scene_model),
                "metadata": dict(metadata),
            },
            timeout_s=timeout_s,
        )

    def validate_fixed_esdf_candidate(
        self,
        build_id: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        request = dict(payload)
        request["build_id"] = _validated_build_id(build_id)
        result = self._request(
            "POST",
            "/tools/fixed_esdf/validate",
            request,
            timeout_s=timeout_s,
        )
        try:
            validated = FixedEsdfCandidateValidationResult.create(
                collision_free=result.get("collision_free"),
                checked_arms=result.get("checked_arms"),
                colliding_arm=result.get("colliding_arm"),
            )
        except (TypeError, ValueError) as error:
            raise _protocol(
                "cuRobo service returned an invalid fixed ESDF validation result"
            ) from error
        return {
            **result,
            "collision_free": validated.collision_free,
            "checked_arms": list(validated.checked_arms),
            "colliding_arm": validated.colliding_arm,
        }

    def publish_fixed_esdf_candidate(
        self,
        build_id: str,
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        return self._esdf_response(
            self._request(
                "POST",
                "/tools/fixed_esdf/publish",
                {"build_id": _validated_build_id(build_id)},
                timeout_s=timeout_s,
            ),
            include_data=True,
        )

    def reconcile_fixed_esdf_candidate(
        self,
        build_id: str,
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        value = _validated_build_id(build_id)
        body = self._request(
            "POST",
            "/tools/fixed_esdf/reconcile",
            {"build_id": value},
            timeout_s=timeout_s,
        )
        published = body.get("published")
        if not isinstance(published, bool):
            raise _protocol("cuRobo service returned an invalid reconciliation result")
        if not published:
            return {"ok": True, "published": False, "build_id": value}
        return {
            **self._esdf_response(body, include_data=True),
            "published": True,
            "build_id": value,
        }

    def discard_fixed_esdf_candidate(
        self,
        build_id: str,
        *,
        timeout_s: float | None = None,
        expected_scene_token: str | None = None,
    ) -> Mapping[str, Any]:
        payload = {"build_id": _validated_build_id(build_id)}
        if expected_scene_token is not None:
            payload["expected_scene_token"] = expected_scene_token
        return self._request(
            "POST",
            "/tools/fixed_esdf/discard",
            payload,
            timeout_s=timeout_s,
        )

    def select_fixed_esdf_instance(
        self,
        *,
        expected_revision: int,
        selected_instance_id: str,
        selection_generation: int,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        if (
            type(expected_revision) is not int
            or expected_revision < 1
            or not isinstance(selected_instance_id, str)
            or not selected_instance_id.strip()
            or type(selection_generation) is not int
            or selection_generation < 1
        ):
            raise ValueError("semantic ESDF selection payload is invalid")
        selected = selected_instance_id.strip()
        body = self._request(
            "POST",
            "/tools/fixed_esdf/select",
            {
                "expected_revision": expected_revision,
                "selected_instance_id": selected,
                "selection_generation": selection_generation,
            },
            timeout_s=timeout_s,
        )
        result = self._esdf_response(body, include_data=False)
        if (
            body.get("selected_instance_id") != selected
            or body.get("selection_generation") != selection_generation
        ):
            raise _protocol("cuRobo service returned an invalid ESDF selection")
        result["selected_instance_id"] = selected
        result["selection_generation"] = selection_generation
        return result

    def _request(self, method: str, path: str, payload=None, timeout_s=None) -> Mapping[str, Any]:
        request_timeout_s = self.timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0.0:
            raise ValueError("request timeout_s must be positive and finite")
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with self._urlopen(
                request, timeout=request_timeout_s
            ) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read()
        except HTTPError as error:
            status, raw = int(error.code), error.read()
        except (URLError, TimeoutError, OSError) as error:
            if isinstance(error, TimeoutError) or (
                isinstance(error, URLError) and isinstance(error.reason, TimeoutError)
            ):
                raise CuroboServiceError(
                    "cuRobo planning service response timed out",
                    code="CUROBO_SERVICE_TIMEOUT", stage="transport", status=504,
                    retryable=True, details={"timeout_s": request_timeout_s},
                ) from error
            raise CuroboServiceError(
                "cuRobo planning service is unavailable",
                code="CUROBO_SERVICE_UNAVAILABLE", stage="transport", retryable=True,
            ) from error
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _protocol("cuRobo service returned invalid JSON", status) from error
        if not isinstance(body, Mapping):
            raise _protocol("cuRobo service response must be an object", status)
        if status >= 400 or body.get("ok") is not True:
            detail = body.get("error") if isinstance(body.get("error"), Mapping) else {}
            raise CuroboServiceError(
                str(detail.get("message") or f"cuRobo service HTTP {status}"),
                code=str(detail.get("code") or "CUROBO_HTTP_ERROR"),
                stage=str(detail.get("stage") or "service"), status=status,
                retryable=detail.get("retryable") is True,
                details=detail.get("details"),
            )
        return body


__all__ = ["CuroboPlannerClient", "CuroboServiceError", "DEFAULT_PLANNER_URL"]

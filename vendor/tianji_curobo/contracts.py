# ========================================
# MODIFICATION START: add lightweight typed cuRobo contracts
# ========================================
"""Typed, backend-independent contracts for cuRobo tool boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, hypot, inf, isfinite, nextafter
from numbers import Integral, Real


MAX_PLAN_ATTEMPTS = 100
MAX_GRASP_BATCH_SIZE = 100
MIN_TRAJECTORY_COLLISION_CLEARANCE_M = 0.01
MIN_OPTIMIZER_COLLISION_ACTIVATION_DISTANCE_M = 0.03
MAX_COLLISION_CLEARANCE_M = 0.5
MIN_AXIS_LINEAR_DISTANCE_M = 0.002
MAX_AXIS_LINEAR_DISTANCE_M = 0.5
# ========================================
# MODIFICATION START: bound execution trajectory validation requests
# ========================================
MIN_TRAJECTORY_VALIDATION_POINTS = 6
MAX_TRAJECTORY_VALIDATION_POINTS = 100_001
# ========================================
# MODIFICATION END: bound execution trajectory validation requests
# ========================================

_VECTOR_ERROR = "{name} must contain exactly {length} finite numbers"


def _as_tuple(values: object, *, name: str) -> tuple[object, ...]:
    if isinstance(values, (str, bytes, Mapping, set, frozenset)):
        raise ValueError(f"{name} must be an ordered sequence")
    if not isinstance(values, Sequence):
        raise ValueError(f"{name} must be an ordered sequence")
    return tuple(values)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _finite_vector(
    values: object,
    *,
    name: str,
    length: int,
) -> tuple[float, ...]:
    items = _as_tuple(values, name=name)
    if len(items) != length:
        raise ValueError(_VECTOR_ERROR.format(name=name, length=length))
    return tuple(
        _finite_float(value, name=f"{name}[{index}]")
        for index, value in enumerate(items)
    )


def _finite_matrix(
    values: object,
    *,
    name: str,
    columns: int,
    minimum_rows: int,
) -> tuple[tuple[float, ...], ...]:
    rows = _as_tuple(values, name=name)
    if len(rows) < minimum_rows:
        raise ValueError(
            f"{name} must contain at least {minimum_rows} rows of "
            f"{columns} finite numbers"
        )
    return tuple(
        _finite_vector(row, name=f"{name}[{index}]", length=columns)
        for index, row in enumerate(rows)
    )


def _positive_float(value: object, *, name: str) -> float:
    converted = _finite_float(value, name=name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _non_negative_float(value: object, *, name: str) -> float:
    converted = _finite_float(value, name=name)
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _collision_clearance_profile(
    trajectory_collision_clearance_m: object,
    optimizer_collision_activation_distance_m: object,
) -> tuple[float, float]:
    trajectory = _finite_float(
        trajectory_collision_clearance_m,
        name="trajectory_collision_clearance_m",
    )
    optimizer = _finite_float(
        optimizer_collision_activation_distance_m,
        name="optimizer_collision_activation_distance_m",
    )
    if not (
        MIN_TRAJECTORY_COLLISION_CLEARANCE_M
        <= trajectory
        <= MAX_COLLISION_CLEARANCE_M
    ):
        raise ValueError(
            "trajectory_collision_clearance_m must be between 0.01 and 0.5 m"
        )
    if not (
        MIN_OPTIMIZER_COLLISION_ACTIVATION_DISTANCE_M
        <= optimizer
        <= MAX_COLLISION_CLEARANCE_M
    ):
        raise ValueError(
            "optimizer_collision_activation_distance_m must be between 0.03 and 0.5 m"
        )
    if optimizer < trajectory:
        raise ValueError(
            "optimizer_collision_activation_distance_m must be greater than or equal to "
            "trajectory_collision_clearance_m"
        )
    return trajectory, optimizer


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _plan_attempts(value: object) -> int:
    attempts = _positive_integer(value, name="max_attempts")
    if attempts > MAX_PLAN_ATTEMPTS:
        raise ValueError(
            f"max_attempts must be between 1 and {MAX_PLAN_ATTEMPTS}"
        )
    return attempts


def _non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return converted


def _required_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalized_arm(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("arm must be 'A' or 'B'")
    arm = value.strip().upper()
    if arm not in {"A", "B"}:
        raise ValueError("arm must be 'A' or 'B'")
    return arm


def _normalized_choice(
    value: object,
    *,
    name: str,
    choices: frozenset[str],
) -> str:
    options = ", ".join(sorted(choices))
    if not isinstance(value, str):
        raise ValueError(f"{name} must be one of: {options}")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of: {options}")
    return normalized


def _bounded_nonzero_distance(value: object) -> float:
    distance = _finite_float(value, name="distance_m")
    if not MIN_AXIS_LINEAR_DISTANCE_M <= abs(distance) <= MAX_AXIS_LINEAR_DISTANCE_M:
        raise ValueError(
            f"distance_m magnitude must be between {MIN_AXIS_LINEAR_DISTANCE_M} "
            f"and {MAX_AXIS_LINEAR_DISTANCE_M} m"
        )
    return distance


def _normalized_quaternion(values: object) -> tuple[float, ...]:
    quaternion = _finite_vector(
        values,
        name="goal_quaternion_wxyz",
        length=4,
    )
    scale = max(abs(component) for component in quaternion)
    if scale == 0.0:
        raise ValueError("goal_quaternion_wxyz must have a non-zero norm")
    scaled = tuple(component / scale for component in quaternion)
    norm = hypot(*scaled)
    return tuple(component / norm for component in scaled)


def _rigid_matrix(values: object, *, name: str) -> tuple[tuple[float, ...], ...]:
    rows = _as_tuple(values, name=name)
    if len(rows) != 4:
        raise ValueError(f"{name} must be a rigid homogeneous 4x4 matrix")
    matrix = tuple(
        _finite_vector(row, name=f"{name}[{index}]", length=4)
        for index, row in enumerate(rows)
    )
    tolerance = 1.0e-5
    if any(abs(value - expected) > tolerance for value, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0))):
        raise ValueError(f"{name} must be a rigid homogeneous 4x4 matrix")
    rotation = tuple(tuple(matrix[row][column] for column in range(3)) for row in range(3))
    for row in range(3):
        for other in range(3):
            dot = sum(rotation[row][column] * rotation[other][column] for column in range(3))
            expected = 1.0 if row == other else 0.0
            if abs(dot - expected) > tolerance:
                raise ValueError(f"{name} must be a rigid homogeneous 4x4 matrix")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise ValueError(f"{name} must be a rigid homogeneous 4x4 matrix")
    return matrix


def _normalized_pose_wxyz(values: object, *, name: str) -> tuple[float, ...]:
    pose = _finite_vector(values, name=name, length=7)
    quaternion = pose[3:]
    scale = max(abs(component) for component in quaternion)
    if scale == 0.0:
        raise ValueError(f"{name} quaternion must have a non-zero norm")
    scaled = tuple(component / scale for component in quaternion)
    norm = hypot(*scaled)
    return (*pose[:3], *(component / norm for component in scaled))


def _joint_names(values: object) -> tuple[str, ...]:
    names = _as_tuple(values, name="joint_names")
    if len(names) != 7:
        raise ValueError("joint_names must contain exactly 7 names")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("joint_names must contain exactly 7 non-empty names")
    normalized = tuple(names)
    if len(set(normalized)) != 7:
        raise ValueError("joint_names must contain 7 unique names")
    return normalized


def _ready_arms(values: object) -> tuple[str, ...]:
    arms = _as_tuple(values, name="arms_ready")
    normalized = tuple(_normalized_arm(arm) for arm in arms)
    if len(set(normalized)) != len(normalized):
        raise ValueError("arms_ready must not contain duplicate arms")
    return normalized


def _mapping_value(values: Mapping[str, object], name: str) -> object:
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"{name} is required") from error


def _validated_mapping(values: object) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError("request payload must be a mapping")
    return values


def _json_details_snapshot(values: object) -> str:
    if values is None:
        copied = {}
    elif isinstance(values, Mapping):
        copied = dict(values)
    else:
        raise ValueError("details must be a mapping")
    try:
        return json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("details must be JSON-safe") from error


def _trajectory_point_scale(value: object) -> float | None:
    if value is None:
        return None
    scale = _finite_float(value, name="trajectory_point_scale")
    if not 1.0 <= scale <= 10.0:
        raise ValueError("trajectory_point_scale must be between 1 and 10")
    return scale


@dataclass(frozen=True)
class PlanToPoseRequest:
    """Validated normalized request for a formal pose trajectory."""

    # ========================================
    # MODIFICATION START: remove MOVLA timing from cuRobo pose contract
    # ========================================
    arm: str
    start_joints_deg: tuple[float, ...]
    other_joints_deg: tuple[float, ...]
    goal_position_m: tuple[float, ...]
    goal_quaternion_wxyz: tuple[float, ...]
    freq_hz: int
    max_attempts: int

    trajectory_point_scale: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_point_scale", _trajectory_point_scale(self.trajectory_point_scale),
        )
        object.__setattr__(self, "arm", _normalized_arm(self.arm))
        object.__setattr__(
            self,
            "start_joints_deg",
            _finite_vector(
                self.start_joints_deg,
                name="start_joints_deg",
                length=7,
            ),
        )
        object.__setattr__(
            self,
            "other_joints_deg",
            _finite_vector(
                self.other_joints_deg,
                name="other_joints_deg",
                length=7,
            ),
        )
        object.__setattr__(
            self,
            "goal_position_m",
            _finite_vector(
                self.goal_position_m,
                name="goal_position_m",
                length=3,
            ),
        )
        object.__setattr__(
            self,
            "goal_quaternion_wxyz",
            _normalized_quaternion(self.goal_quaternion_wxyz),
        )
        object.__setattr__(
            self,
            "freq_hz",
            _positive_integer(self.freq_hz, name="freq_hz"),
        )
        object.__setattr__(
            self,
            "max_attempts",
            _plan_attempts(self.max_attempts),
        )

    @classmethod
    def create(
        cls,
        *,
        arm: object,
        start_joints_deg: object,
        other_joints_deg: object,
        goal_position_m: object,
        goal_quaternion_wxyz: object,
        freq_hz: object,
        max_attempts: object,
        trajectory_point_scale: object = None,
    ) -> PlanToPoseRequest:
        return cls(
            arm=arm,
            start_joints_deg=start_joints_deg,
            other_joints_deg=other_joints_deg,
            goal_position_m=goal_position_m,
            goal_quaternion_wxyz=goal_quaternion_wxyz,
            freq_hz=freq_hz,
            max_attempts=max_attempts,
            trajectory_point_scale=trajectory_point_scale,
        )

    @classmethod
    def from_mapping(cls, values: object) -> PlanToPoseRequest:
        mapping = _validated_mapping(values)
        return cls.create(
            arm=_mapping_value(mapping, "arm"),
            start_joints_deg=_mapping_value(mapping, "start_joints_deg"),
            other_joints_deg=_mapping_value(mapping, "other_joints_deg"),
            goal_position_m=_mapping_value(mapping, "goal_position_m"),
            goal_quaternion_wxyz=_mapping_value(
                mapping,
                "goal_quaternion_wxyz",
            ),
            freq_hz=_mapping_value(mapping, "freq_hz"),
            max_attempts=_mapping_value(mapping, "max_attempts"),
            trajectory_point_scale=mapping.get("trajectory_point_scale"),
        )
    # ========================================
    # MODIFICATION END: remove MOVLA timing from cuRobo pose contract
    # ========================================


@dataclass(frozen=True)
class PlanGraspRequest(PlanToPoseRequest):
    """Validated request for the official three-phase cuRobo grasp planner."""

    approach_axis: str = "z"
    approach_offset_m: float = -0.10
    approach_in_tool_frame: bool = True
    lift_axis: str = "z"
    lift_offset_m: float = -0.10
    lift_in_tool_frame: bool = True
    trajectory_collision_clearance_m: float = 0.01
    optimizer_collision_activation_distance_m: float = 0.03

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in ("approach_axis", "lift_axis"):
            object.__setattr__(
                self,
                field_name,
                _normalized_choice(
                    getattr(self, field_name),
                    name=field_name,
                    choices=frozenset({"x", "y", "z"}),
                ),
            )
        for field_name in ("approach_offset_m", "lift_offset_m"):
            distance = _finite_float(getattr(self, field_name), name=field_name)
            if distance == 0.0 or abs(distance) > 0.5:
                raise ValueError(f"{field_name} must be non-zero and within 0.5 m")
            object.__setattr__(self, field_name, distance)
        for field_name in ("approach_in_tool_frame", "lift_in_tool_frame"):
            object.__setattr__(
                self,
                field_name,
                _required_bool(getattr(self, field_name), name=field_name),
            )

        trajectory, optimizer = _collision_clearance_profile(
            self.trajectory_collision_clearance_m,
            self.optimizer_collision_activation_distance_m,
        )
        object.__setattr__(self, "trajectory_collision_clearance_m", trajectory)
        object.__setattr__(
            self,
            "optimizer_collision_activation_distance_m",
            optimizer,
        )

    @classmethod
    def create(
        cls,
        *,
        arm: object,
        start_joints_deg: object,
        other_joints_deg: object,
        goal_position_m: object,
        goal_quaternion_wxyz: object,
        freq_hz: object,
        max_attempts: object,
        trajectory_point_scale: object = None,
        approach_axis: object = "z",
        approach_offset_m: object = -0.10,
        approach_in_tool_frame: object = True,
        lift_axis: object = "z",
        lift_offset_m: object = -0.10,
        lift_in_tool_frame: object = True,
        trajectory_collision_clearance_m: object = 0.01,
        optimizer_collision_activation_distance_m: object = 0.03,
    ) -> PlanGraspRequest:
        return cls(
            arm=arm,
            start_joints_deg=start_joints_deg,
            other_joints_deg=other_joints_deg,
            goal_position_m=goal_position_m,
            goal_quaternion_wxyz=goal_quaternion_wxyz,
            freq_hz=freq_hz,
            max_attempts=max_attempts,
            trajectory_point_scale=trajectory_point_scale,
            approach_axis=approach_axis,
            approach_offset_m=approach_offset_m,
            approach_in_tool_frame=approach_in_tool_frame,
            lift_axis=lift_axis,
            lift_offset_m=lift_offset_m,
            lift_in_tool_frame=lift_in_tool_frame,
            trajectory_collision_clearance_m=trajectory_collision_clearance_m,
            optimizer_collision_activation_distance_m=(
                optimizer_collision_activation_distance_m
            ),
        )

    @classmethod
    def from_mapping(cls, values: object) -> PlanGraspRequest:
        mapping = _validated_mapping(values)
        return cls.create(
            arm=_mapping_value(mapping, "arm"),
            start_joints_deg=_mapping_value(mapping, "start_joints_deg"),
            other_joints_deg=_mapping_value(mapping, "other_joints_deg"),
            goal_position_m=_mapping_value(mapping, "goal_position_m"),
            goal_quaternion_wxyz=_mapping_value(mapping, "goal_quaternion_wxyz"),
            freq_hz=_mapping_value(mapping, "freq_hz"),
            max_attempts=_mapping_value(mapping, "max_attempts"),
            trajectory_point_scale=mapping.get("trajectory_point_scale"),
            approach_axis=mapping.get("approach_axis", "z"),
            approach_offset_m=mapping.get("approach_offset_m", -0.10),
            approach_in_tool_frame=mapping.get("approach_in_tool_frame", True),
            lift_axis=mapping.get("lift_axis", "z"),
            lift_offset_m=mapping.get("lift_offset_m", -0.10),
            lift_in_tool_frame=mapping.get("lift_in_tool_frame", True),
            trajectory_collision_clearance_m=mapping.get(
                "trajectory_collision_clearance_m",
                0.01,
            ),
            optimizer_collision_activation_distance_m=mapping.get(
                "optimizer_collision_activation_distance_m",
                0.03,
            ),
        )


@dataclass(frozen=True)
class PlanGraspBatchRequest:
    """Validated independent grasp problems for one native CUDA batch."""

    arm: str
    start_joints_deg: tuple[float, ...]
    other_joints_deg: tuple[float, ...]
    goal_matrices_world: tuple[tuple[tuple[float, ...], ...], ...]
    freq_hz: int
    approach_axis: str = "z"
    approach_offset_m: float = -0.10
    approach_in_tool_frame: bool = True
    lift_axis: str = "z"
    lift_offset_m: float = -0.10
    lift_in_tool_frame: bool = True
    trajectory_collision_clearance_m: float = 0.01
    optimizer_collision_activation_distance_m: float = 0.03
    scene_token: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _normalized_arm(self.arm))
        object.__setattr__(self, "start_joints_deg", _finite_vector(self.start_joints_deg, name="start_joints_deg", length=7))
        object.__setattr__(self, "other_joints_deg", _finite_vector(self.other_joints_deg, name="other_joints_deg", length=7))
        matrices = _as_tuple(self.goal_matrices_world, name="goal_matrices_world")
        if not matrices:
            raise ValueError("goal_matrices_world must contain between 1 and 100 matrices")
        if len(matrices) > MAX_GRASP_BATCH_SIZE:
            raise CuroboToolError(
                "native grasp batch capacity exceeded",
                stage="validation",
                code="curobo_batch_capacity_exceeded",
                status=422,
                details={
                    "requested_batch_size": len(matrices),
                    "maximum_batch_size": MAX_GRASP_BATCH_SIZE,
                },
            )
        object.__setattr__(
            self,
            "goal_matrices_world",
            tuple(_rigid_matrix(matrix, name=f"goal_matrices_world[{index}]") for index, matrix in enumerate(matrices)),
        )
        object.__setattr__(self, "freq_hz", _positive_integer(self.freq_hz, name="freq_hz"))
        for field_name in ("approach_axis", "lift_axis"):
            object.__setattr__(self, field_name, _normalized_choice(getattr(self, field_name), name=field_name, choices=frozenset({"x", "y", "z"})))
        for field_name in ("approach_offset_m", "lift_offset_m"):
            distance = _finite_float(getattr(self, field_name), name=field_name)
            if distance == 0.0 or abs(distance) > 0.5:
                raise ValueError(f"{field_name} must be non-zero and within 0.5 m")
            object.__setattr__(self, field_name, distance)
        for field_name in ("approach_in_tool_frame", "lift_in_tool_frame"):
            object.__setattr__(self, field_name, _required_bool(getattr(self, field_name), name=field_name))
        object.__setattr__(self, "scene_token", _required_text(self.scene_token, name="scene_token"))
        trajectory, optimizer = _collision_clearance_profile(
            self.trajectory_collision_clearance_m,
            self.optimizer_collision_activation_distance_m,
        )
        object.__setattr__(self, "trajectory_collision_clearance_m", trajectory)
        object.__setattr__(
            self,
            "optimizer_collision_activation_distance_m",
            optimizer,
        )

    @classmethod
    def create(cls, **values) -> PlanGraspBatchRequest:
        return cls(**values)

    @classmethod
    def from_mapping(cls, values: object) -> PlanGraspBatchRequest:
        mapping = _validated_mapping(values)
        return cls.create(
            arm=_mapping_value(mapping, "arm"),
            start_joints_deg=_mapping_value(mapping, "start_joints_deg"),
            other_joints_deg=_mapping_value(mapping, "other_joints_deg"),
            goal_matrices_world=_mapping_value(mapping, "goal_matrices_world"),
            freq_hz=_mapping_value(mapping, "freq_hz"),
            approach_axis=mapping.get("approach_axis", "z"),
            approach_offset_m=mapping.get("approach_offset_m", -0.10),
            approach_in_tool_frame=mapping.get("approach_in_tool_frame", True),
            lift_axis=mapping.get("lift_axis", "z"),
            lift_offset_m=mapping.get("lift_offset_m", -0.10),
            lift_in_tool_frame=mapping.get("lift_in_tool_frame", True),
            scene_token=_mapping_value(mapping, "scene_token"),
            trajectory_collision_clearance_m=mapping.get(
                "trajectory_collision_clearance_m",
                0.01,
            ),
            optimizer_collision_activation_distance_m=mapping.get(
                "optimizer_collision_activation_distance_m",
                0.03,
            ),
        )

@dataclass(frozen=True)
class SolveIkBatchRequest:
    """Independent world-frame tool goals; no path or execution settings."""

    arm: str
    start_joints_deg: tuple[float, ...]
    other_joints_deg: tuple[float, ...]
    goal_matrices_world: tuple
    scene_token: str
    trajectory_collision_clearance_m: float = 0.01

    def __post_init__(self):
        object.__setattr__(self, "arm", _normalized_arm(self.arm))
        for name in ("start_joints_deg", "other_joints_deg"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), name=name, length=7))
        matrices = _as_tuple(self.goal_matrices_world, name="goal_matrices_world")
        if not 1 <= len(matrices) <= 200:
            raise ValueError("IK batch must contain between 1 and 200 matrices")
        object.__setattr__(self, "goal_matrices_world", tuple(
            _rigid_matrix(matrix, name=f"goal_matrices_world[{index}]")
            for index, matrix in enumerate(matrices)
        ))
        object.__setattr__(self, "scene_token", _required_text(self.scene_token, name="scene_token"))
        clearance = _finite_float(self.trajectory_collision_clearance_m, name="trajectory_collision_clearance_m")
        if not 0.01 <= clearance <= 0.5:
            raise ValueError("trajectory_collision_clearance_m must be between 0.01 and 0.5 m")
        object.__setattr__(self, "trajectory_collision_clearance_m", clearance)

    @classmethod
    def from_mapping(cls, values):
        mapping = _validated_mapping(values)
        return cls(
            arm=_mapping_value(mapping, "arm"),
            start_joints_deg=_mapping_value(mapping, "start_joints_deg"),
            other_joints_deg=_mapping_value(mapping, "other_joints_deg"),
            goal_matrices_world=_mapping_value(mapping, "goal_matrices_world"),
            scene_token=_mapping_value(mapping, "scene_token"),
            trajectory_collision_clearance_m=mapping.get("trajectory_collision_clearance_m", 0.01),
        )


@dataclass(frozen=True)
class NativeIkBatchResult:
    items: tuple
    scene_token: str
    solve_time_s: float

    def __post_init__(self):
        if not 1 <= len(self.items) <= 200 or any(not isinstance(item, NativeIkResult) for item in self.items):
            raise ValueError("IK batch items must contain 1 to 200 NativeIkResult values")
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "scene_token", _required_text(self.scene_token, name="scene_token"))
        duration = _finite_float(self.solve_time_s, name="solve_time_s")
        if duration < 0:
            raise ValueError("solve_time_s must be nonnegative")
        object.__setattr__(self, "solve_time_s", duration)

    def to_payload(self):
        return {
            "ok": True, "planner": "curobo_ik_batch", "diagnostic_only": True,
            "batch_size": len(self.items), "scene_token": self.scene_token,
            "solve_time_s": self.solve_time_s,
            "items": [{
                "candidate_index": index, "success": item.reachable,
                "joints_rad": list(item.joints_rad) if item.joints_rad is not None else None,
                "joint_names": list(item.joint_names),
                "position_error_m": item.position_error_m,
                "rotation_error_rad": item.rotation_error_rad,
            } for index, item in enumerate(self.items)],
        }

    @classmethod
    def from_mapping(cls, values):
        mapping = _validated_mapping(values)
        items = _as_tuple(_mapping_value(mapping, "items"), name="items")
        if type(mapping.get("batch_size")) is not int or len(items) != mapping["batch_size"]:
            raise ValueError("IK batch result size mismatch")
        parsed = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping) or type(item.get("candidate_index")) is not int or item["candidate_index"] != index:
                raise ValueError("IK batch result ordinal mismatch")
            parsed.append(NativeIkResult(
                reachable=item.get("success"), joints_rad=item.get("joints_rad"),
                joint_names=item.get("joint_names"), position_error_m=item.get("position_error_m"),
                rotation_error_rad=item.get("rotation_error_rad"),
            ))
        return cls(tuple(parsed), mapping.get("scene_token"), mapping.get("solve_time_s"))


@dataclass(frozen=True)
class PlanPlaceRequest(PlanGraspRequest):
    """Three-phase place request with open-loop gripper release."""


@dataclass(frozen=True)
class PlanToJointRequest:
    """Validated collision-aware cuRobo joint-goal request."""

    arm: str
    start_joints_deg: tuple[float, ...]
    other_joints_deg: tuple[float, ...]
    goal_joints_deg: tuple[float, ...]
    freq_hz: int
    max_attempts: int

    trajectory_point_scale: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_point_scale", _trajectory_point_scale(self.trajectory_point_scale),
        )
        object.__setattr__(self, "arm", _normalized_arm(self.arm))
        for field_name in (
            "start_joints_deg",
            "other_joints_deg",
            "goal_joints_deg",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_vector(
                    getattr(self, field_name),
                    name=field_name,
                    length=7,
                ),
            )
        object.__setattr__(
            self,
            "freq_hz",
            _positive_integer(self.freq_hz, name="freq_hz"),
        )
        object.__setattr__(
            self,
            "max_attempts",
            _plan_attempts(self.max_attempts),
        )

    @classmethod
    def create(
        cls,
        *,
        arm: object,
        start_joints_deg: object,
        other_joints_deg: object,
        goal_joints_deg: object,
        freq_hz: object,
        max_attempts: object,
        trajectory_point_scale: object = None,
    ) -> PlanToJointRequest:
        return cls(
            arm=arm,
            start_joints_deg=start_joints_deg,
            other_joints_deg=other_joints_deg,
            goal_joints_deg=goal_joints_deg,
            freq_hz=freq_hz,
            max_attempts=max_attempts,
            trajectory_point_scale=trajectory_point_scale,
        )

    @classmethod
    def from_mapping(cls, values: object) -> PlanToJointRequest:
        mapping = _validated_mapping(values)
        return cls.create(
            arm=_mapping_value(mapping, "arm"),
            start_joints_deg=_mapping_value(mapping, "start_joints_deg"),
            other_joints_deg=_mapping_value(mapping, "other_joints_deg"),
            goal_joints_deg=_mapping_value(mapping, "goal_joints_deg"),
            freq_hz=_mapping_value(mapping, "freq_hz"),
            max_attempts=_mapping_value(mapping, "max_attempts"),
            trajectory_point_scale=mapping.get("trajectory_point_scale"),
        )


@dataclass(frozen=True)
class PlanAxisLinearRequest:
    """Validated request for fixed-orientation motion along one frame axis."""

    # ========================================
    # MODIFICATION START: remove MOVLA timing from cuRobo line contract
    # ========================================
    arm: str
    start_joints_deg: tuple[float, ...]
    other_joints_deg: tuple[float, ...]
    reference_frame: str
    axis: str
    distance_m: float
    freq_hz: int
    max_attempts: int

    trajectory_point_scale: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_point_scale", _trajectory_point_scale(self.trajectory_point_scale),
        )
        object.__setattr__(self, "arm", _normalized_arm(self.arm))
        object.__setattr__(
            self,
            "start_joints_deg",
            _finite_vector(
                self.start_joints_deg,
                name="start_joints_deg",
                length=7,
            ),
        )
        object.__setattr__(
            self,
            "other_joints_deg",
            _finite_vector(
                self.other_joints_deg,
                name="other_joints_deg",
                length=7,
            ),
        )
        object.__setattr__(
            self,
            "reference_frame",
            _normalized_choice(
                self.reference_frame,
                name="reference_frame",
                choices=frozenset({"world", "tool"}),
            ),
        )
        object.__setattr__(
            self,
            "axis",
            _normalized_choice(
                self.axis,
                name="axis",
                choices=frozenset({"x", "y", "z"}),
            ),
        )
        object.__setattr__(
            self,
            "distance_m",
            _bounded_nonzero_distance(self.distance_m),
        )
        object.__setattr__(
            self,
            "freq_hz",
            _positive_integer(self.freq_hz, name="freq_hz"),
        )
        object.__setattr__(
            self,
            "max_attempts",
            _plan_attempts(self.max_attempts),
        )

    @classmethod
    def create(
        cls,
        *,
        arm: object,
        start_joints_deg: object,
        other_joints_deg: object,
        reference_frame: object,
        axis: object,
        distance_m: object,
        freq_hz: object,
        max_attempts: object,
        trajectory_point_scale: object = None,
    ) -> PlanAxisLinearRequest:
        return cls(
            arm=arm,
            start_joints_deg=start_joints_deg,
            other_joints_deg=other_joints_deg,
            reference_frame=reference_frame,
            axis=axis,
            distance_m=distance_m,
            freq_hz=freq_hz,
            max_attempts=max_attempts,
            trajectory_point_scale=trajectory_point_scale,
        )

    @classmethod
    def from_mapping(cls, values: object) -> PlanAxisLinearRequest:
        mapping = _validated_mapping(values)
        return cls.create(
            arm=_mapping_value(mapping, "arm"),
            start_joints_deg=_mapping_value(mapping, "start_joints_deg"),
            other_joints_deg=_mapping_value(mapping, "other_joints_deg"),
            reference_frame=_mapping_value(mapping, "reference_frame"),
            axis=_mapping_value(mapping, "axis"),
            distance_m=_mapping_value(mapping, "distance_m"),
            freq_hz=_mapping_value(mapping, "freq_hz"),
            max_attempts=_mapping_value(mapping, "max_attempts"),
            trajectory_point_scale=mapping.get("trajectory_point_scale"),
        )
    # ========================================
    # MODIFICATION END: remove MOVLA timing from cuRobo line contract
    # ========================================


@dataclass(frozen=True)
class SolveIkRequest:
    """Validated normalized request for reachability and inverse kinematics."""

    arm: str
    current_joints_deg: tuple[float, ...]
    other_joints_deg: tuple[float, ...]
    goal_position_m: tuple[float, ...]
    goal_quaternion_wxyz: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _normalized_arm(self.arm))
        object.__setattr__(
            self,
            "current_joints_deg",
            _finite_vector(
                self.current_joints_deg,
                name="current_joints_deg",
                length=7,
            ),
        )
        object.__setattr__(
            self,
            "other_joints_deg",
            _finite_vector(
                self.other_joints_deg,
                name="other_joints_deg",
                length=7,
            ),
        )
        object.__setattr__(
            self,
            "goal_position_m",
            _finite_vector(
                self.goal_position_m,
                name="goal_position_m",
                length=3,
            ),
        )
        object.__setattr__(
            self,
            "goal_quaternion_wxyz",
            _normalized_quaternion(self.goal_quaternion_wxyz),
        )

    @classmethod
    def create(
        cls,
        *,
        arm: object,
        current_joints_deg: object,
        other_joints_deg: object,
        goal_position_m: object,
        goal_quaternion_wxyz: object,
    ) -> SolveIkRequest:
        return cls(
            arm=arm,
            current_joints_deg=current_joints_deg,
            other_joints_deg=other_joints_deg,
            goal_position_m=goal_position_m,
            goal_quaternion_wxyz=goal_quaternion_wxyz,
        )

    @classmethod
    def from_mapping(cls, values: object) -> SolveIkRequest:
        mapping = _validated_mapping(values)
        return cls.create(
            arm=_mapping_value(mapping, "arm"),
            current_joints_deg=_mapping_value(
                mapping,
                "current_joints_deg",
            ),
            other_joints_deg=_mapping_value(mapping, "other_joints_deg"),
            goal_position_m=_mapping_value(mapping, "goal_position_m"),
            goal_quaternion_wxyz=_mapping_value(
                mapping,
                "goal_quaternion_wxyz",
            ),
        )


# ========================================
# MODIFICATION START: add typed execution trajectory validation contracts
# ========================================
@dataclass(frozen=True)
class ValidateTrajectoryRequest:
    """Validated final Tianji trajectory checked against the current scene."""

    arm: str
    points_deg: tuple[tuple[float, ...], ...]
    other_joints_deg: tuple[float, ...]
    expected_scene_token: str | None = None
    grasp_phase: str | None = None
    trajectory_collision_clearance_m: float = 0.01
    optimizer_collision_activation_distance_m: float = 0.03
    gripper_joint_positions_rad: dict[str, float] | None = None
    gripper_configuration_source: str | None = None

    def __post_init__(self) -> None:
        if self.gripper_joint_positions_rad is not None:
            positions = self.gripper_joint_positions_rad
            if not isinstance(positions, dict) or set(positions) != {"A", "B"}:
                raise ValueError("both gripper driver positions are required")
            values = _finite_vector([positions["A"], positions["B"]], name="gripper_joint_positions_rad", length=2)
            if self.gripper_configuration_source != "measured_feedback":
                raise ValueError("final trajectory validation requires measured_feedback geometry")
            object.__setattr__(self, "gripper_joint_positions_rad", dict(zip(("A", "B"), values)))
        elif self.gripper_configuration_source is not None:
            raise ValueError("gripper geometry source requires both driver positions")
        object.__setattr__(self, "arm", _normalized_arm(self.arm))
        rows = _as_tuple(self.points_deg, name="points_deg")
        row_count = len(rows)
        if not (
            MIN_TRAJECTORY_VALIDATION_POINTS
            <= row_count
            <= MAX_TRAJECTORY_VALIDATION_POINTS
        ):
            raise ValueError(
                "points_deg must contain between "
                f"{MIN_TRAJECTORY_VALIDATION_POINTS} and "
                f"{MAX_TRAJECTORY_VALIDATION_POINTS} rows of 7 finite numbers"
            )
        object.__setattr__(
            self,
            "points_deg",
            tuple(
                _finite_vector(
                    row,
                    name=f"points_deg[{index}]",
                    length=7,
                )
                for index, row in enumerate(rows)
            ),
        )
        object.__setattr__(
            self,
            "other_joints_deg",
            _finite_vector(
                self.other_joints_deg,
                name="other_joints_deg",
                length=7,
            ),
        )
        if self.expected_scene_token is None:
            token = None
        else:
            token = _required_text(
                self.expected_scene_token,
                name="expected_scene_token",
            )
        object.__setattr__(self, "expected_scene_token", token)
        if self.grasp_phase is None:
            grasp_phase = None
        else:
            grasp_phase = _normalized_choice(
                self.grasp_phase,
                name="grasp_phase",
                choices=frozenset(
                    {"approach", "grasp", "lift", "place", "retreat"}
                ),
            )
        object.__setattr__(self, "grasp_phase", grasp_phase)
        trajectory, optimizer = _collision_clearance_profile(
            self.trajectory_collision_clearance_m,
            self.optimizer_collision_activation_distance_m,
        )
        object.__setattr__(self, "trajectory_collision_clearance_m", trajectory)
        object.__setattr__(
            self,
            "optimizer_collision_activation_distance_m",
            optimizer,
        )

    @classmethod
    def create(
        cls,
        *,
        arm: object,
        points_deg: object,
        other_joints_deg: object,
        expected_scene_token: object = None,
        grasp_phase: object = None,
        trajectory_collision_clearance_m: object = 0.01,
        optimizer_collision_activation_distance_m: object = 0.03,
        gripper_joint_positions_rad: object = None,
        gripper_configuration_source: object = None,
    ) -> ValidateTrajectoryRequest:
        return cls(
            arm=arm,
            points_deg=points_deg,
            other_joints_deg=other_joints_deg,
            expected_scene_token=expected_scene_token,
            grasp_phase=grasp_phase,
            gripper_joint_positions_rad=gripper_joint_positions_rad,
            gripper_configuration_source=gripper_configuration_source,
            trajectory_collision_clearance_m=trajectory_collision_clearance_m,
            optimizer_collision_activation_distance_m=(
                optimizer_collision_activation_distance_m
            ),
        )

    @classmethod
    def from_mapping(cls, values: object) -> ValidateTrajectoryRequest:
        mapping = _validated_mapping(values)
        return cls.create(
            arm=_mapping_value(mapping, "arm"),
            points_deg=_mapping_value(mapping, "points_deg"),
            other_joints_deg=_mapping_value(mapping, "other_joints_deg"),
            expected_scene_token=mapping.get("expected_scene_token"),
            grasp_phase=mapping.get("grasp_phase"),
            gripper_joint_positions_rad=mapping.get("gripper_joint_positions_rad"),
            gripper_configuration_source=mapping.get("gripper_configuration_source"),
            trajectory_collision_clearance_m=mapping.get(
                "trajectory_collision_clearance_m",
                0.01,
            ),
            optimizer_collision_activation_distance_m=mapping.get(
                "optimizer_collision_activation_distance_m",
                0.03,
            ),
        )


@dataclass(frozen=True)
class TrajectoryValidationResult:
    """Typed collision result for one final execution trajectory."""

    collision_free: bool
    checked_point_count: int
    first_invalid_index: int | None = None
    scene_token: str | None = None
    gripper_joint_positions_rad: dict[str, float] | None = None
    gripper_configuration_source: str | None = None
    binding_sha256: str | None = None

    def __post_init__(self) -> None:
        if any(value is not None for value in (self.gripper_joint_positions_rad,
                                               self.gripper_configuration_source,
                                               self.binding_sha256)):
            positions = self.gripper_joint_positions_rad
            if not isinstance(positions, dict) or set(positions) != {"A", "B"}:
                raise ValueError("measured proof requires both gripper positions")
            values = _finite_vector([positions["A"], positions["B"]], name="gripper_joint_positions_rad", length=2)
            if self.gripper_configuration_source != "measured_feedback":
                raise ValueError("measured proof source is invalid")
            digest = self.binding_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
                raise ValueError("measured proof requires a SHA-256 binding")
            object.__setattr__(self, "gripper_joint_positions_rad", dict(zip(("A", "B"), values)))
        collision_free = _required_bool(
            self.collision_free,
            name="collision_free",
        )
        checked_point_count = _positive_integer(
            self.checked_point_count,
            name="checked_point_count",
        )
        if self.first_invalid_index is None:
            first_invalid_index = None
        else:
            first_invalid_index = _non_negative_integer(
                self.first_invalid_index,
                name="first_invalid_index",
            )
            if first_invalid_index >= checked_point_count:
                raise ValueError(
                    "first_invalid_index must identify a checked point"
                )
        if collision_free and first_invalid_index is not None:
            raise ValueError(
                "first_invalid_index must be omitted when collision_free is true"
            )
        if self.scene_token is None:
            scene_token = None
        else:
            scene_token = _required_text(
                self.scene_token,
                name="scene_token",
            )
        object.__setattr__(self, "collision_free", collision_free)
        object.__setattr__(
            self,
            "checked_point_count",
            checked_point_count,
        )
        object.__setattr__(
            self,
            "first_invalid_index",
            first_invalid_index,
        )
        object.__setattr__(self, "scene_token", scene_token)

    @classmethod
    def create(
        cls,
        *,
        collision_free: object,
        checked_point_count: object,
        first_invalid_index: object = None,
        scene_token: object = None,
        gripper_joint_positions_rad: object = None,
        gripper_configuration_source: object = None,
        binding_sha256: object = None,
    ) -> TrajectoryValidationResult:
        return cls(
            collision_free=collision_free,
            checked_point_count=checked_point_count,
            first_invalid_index=first_invalid_index,
            scene_token=scene_token,
            gripper_joint_positions_rad=gripper_joint_positions_rad,
            gripper_configuration_source=gripper_configuration_source,
            binding_sha256=binding_sha256,
        )


@dataclass(frozen=True)
class FixedEsdfCandidateValidationResult:
    """Collision result for the robot pose captured with a fixed-map frame."""

    collision_free: bool
    checked_arms: tuple[str, ...]
    colliding_arm: str | None = None

    def __post_init__(self) -> None:
        collision_free = _required_bool(
            self.collision_free,
            name="collision_free",
        )
        checked_arms = tuple(
            _normalized_arm(arm) for arm in _as_tuple(
                self.checked_arms,
                name="checked_arms",
            )
        )
        if not checked_arms or len(set(checked_arms)) != len(checked_arms):
            raise ValueError("checked_arms must contain unique arm names")
        colliding_arm = (
            None
            if self.colliding_arm is None
            else _normalized_arm(self.colliding_arm)
        )
        if collision_free and colliding_arm is not None:
            raise ValueError(
                "colliding_arm must be omitted when collision_free is true"
            )
        if not collision_free and colliding_arm not in checked_arms:
            raise ValueError(
                "colliding_arm must identify one of the checked arms"
            )
        object.__setattr__(self, "collision_free", collision_free)
        object.__setattr__(self, "checked_arms", checked_arms)
        object.__setattr__(self, "colliding_arm", colliding_arm)

    @classmethod
    def create(
        cls,
        *,
        collision_free: object,
        checked_arms: object,
        colliding_arm: object = None,
    ) -> FixedEsdfCandidateValidationResult:
        return cls(
            collision_free=collision_free,
            checked_arms=checked_arms,
            colliding_arm=colliding_arm,
        )
# ========================================
# MODIFICATION END: add typed execution trajectory validation contracts
# ========================================


@dataclass(frozen=True)
class NativeTrajectory:
    """Native cuRobo trajectory before Tianji joint-order conversion."""

    positions_rad: tuple[tuple[float, ...], ...]
    joint_names: tuple[str, ...]
    interpolation_dt_s: float
    planning_time_s: float
    position_error_m: float
    rotation_error_rad: float
    scene_token: str | None = None
    command_positions_rad: tuple[tuple[float, ...], ...] | None = None

    trajectory_point_scale: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_point_scale", _trajectory_point_scale(self.trajectory_point_scale),
        )
        if (self.command_positions_rad is None) != (self.trajectory_point_scale is None):
            raise ValueError("command_positions_rad and trajectory_point_scale must be supplied together")
        if self.command_positions_rad is not None:
            object.__setattr__(
                self, "command_positions_rad",
                _finite_matrix(self.command_positions_rad, name="command_positions_rad",
                               columns=7, minimum_rows=2),
            )
        object.__setattr__(
            self,
            "positions_rad",
            _finite_matrix(
                self.positions_rad,
                name="positions_rad",
                columns=7,
                minimum_rows=1,
            ),
        )
        if self.command_positions_rad is not None:
            expected_count = ceil(nextafter(
                (len(self.positions_rad) - 1) * self.trajectory_point_scale, -inf,
            )) + 1
            if len(self.command_positions_rad) != expected_count or expected_count > 100_001:
                raise ValueError("command_positions_rad count does not match trajectory_point_scale")
            if (self.command_positions_rad[0] != self.positions_rad[0]
                    or self.command_positions_rad[-1] != self.positions_rad[-1]):
                raise ValueError("command_positions_rad must preserve the original endpoints")
        object.__setattr__(self, "joint_names", _joint_names(self.joint_names))
        object.__setattr__(
            self,
            "interpolation_dt_s",
            _positive_float(
                self.interpolation_dt_s,
                name="interpolation_dt_s",
            ),
        )
        object.__setattr__(
            self,
            "planning_time_s",
            _non_negative_float(
                self.planning_time_s,
                name="planning_time_s",
            ),
        )
        object.__setattr__(
            self,
            "position_error_m",
            _non_negative_float(
                self.position_error_m,
                name="position_error_m",
            ),
        )
        object.__setattr__(
            self,
            "rotation_error_rad",
            _non_negative_float(
                self.rotation_error_rad,
                name="rotation_error_rad",
            ),
        )
        if self.scene_token is not None:
            object.__setattr__(
                self,
                "scene_token",
                _required_text(self.scene_token, name="scene_token"),
            )


@dataclass(frozen=True)
class NativeGraspPlan:
    """The three native trajectories returned by cuRobo ``plan_grasp``."""

    approach: NativeTrajectory
    grasp: NativeTrajectory
    lift: NativeTrajectory
    scene_token: str | None = field(init=False)

    def __post_init__(self) -> None:
        phases = (self.approach, self.grasp, self.lift)
        if any(not isinstance(phase, NativeTrajectory) for phase in phases):
            raise ValueError("approach, grasp and lift must be NativeTrajectory values")
        tokens = {phase.scene_token for phase in phases}
        if len(tokens) != 1:
            raise ValueError("all grasp phases must use the same scene_token")
        object.__setattr__(self, "scene_token", phases[0].scene_token)

    @classmethod
    def create(
        cls,
        *,
        approach: object,
        grasp: object,
        lift: object,
    ) -> NativeGraspPlan:
        return cls(approach=approach, grasp=grasp, lift=lift)


@dataclass(frozen=True)
class NativeGraspBatchItem:
    """One independently planned grasp problem from the official batch result."""

    candidate_index: int
    success: bool
    goalset_success: bool
    approach_success: bool
    grasp_success: bool
    lift_success: bool
    status: str
    approach: NativeTrajectory | None = None
    grasp: NativeTrajectory | None = None
    lift: NativeTrajectory | None = None
    grasp_end_joints_rad: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        index = _non_negative_integer(self.candidate_index, name="candidate_index")
        object.__setattr__(self, "candidate_index", index)
        for field_name in ("success", "goalset_success", "approach_success", "grasp_success", "lift_success"):
            object.__setattr__(self, field_name, _required_bool(getattr(self, field_name), name=field_name))
        expected_success = self.goalset_success and self.approach_success and self.grasp_success and self.lift_success
        if self.success != expected_success:
            raise ValueError("success must agree with the official stage conjunction")
        if not isinstance(self.status, str) or not self.status or len(self.status) > 256 or any(ord(character) < 32 for character in self.status):
            raise ValueError("status must be a safe non-empty string of at most 256 characters")
        for field_name, stage_success in (
            ("approach", self.approach_success),
            ("grasp", self.grasp_success),
            ("lift", self.lift_success),
        ):
            trajectory = getattr(self, field_name)
            if trajectory is not None and not isinstance(trajectory, NativeTrajectory):
                raise ValueError(f"{field_name} must be a NativeTrajectory or None")
            if not stage_success and trajectory is not None:
                raise ValueError(f"unsuccessful {field_name} must not expose a trajectory")
        if self.grasp_end_joints_rad is not None:
            if not self.grasp_success or self.grasp is None:
                raise ValueError("grasp_end_joints_rad requires a valid grasp trajectory")
            joints = _finite_vector(self.grasp_end_joints_rad, name="grasp_end_joints_rad", length=7)
            if joints != self.grasp.positions_rad[-1]:
                raise ValueError("grasp_end_joints_rad must match the grasp trajectory endpoint")
            object.__setattr__(self, "grasp_end_joints_rad", joints)

    @classmethod
    def create(cls, **values) -> NativeGraspBatchItem:
        return cls(**values)


@dataclass(frozen=True)
class NativeGraspBatchPlan:
    """Bounded ordered results from one official ``BatchMotionPlanner`` call."""

    items: tuple[NativeGraspBatchItem, ...]
    batch_size: int
    batch_capacity: int
    planning_time_s: float
    scene_token: str

    def __post_init__(self) -> None:
        items = _as_tuple(self.items, name="items")
        if any(not isinstance(item, NativeGraspBatchItem) for item in items):
            raise ValueError("items must contain NativeGraspBatchItem values")
        batch_size = _positive_integer(self.batch_size, name="batch_size")
        batch_capacity = _positive_integer(self.batch_capacity, name="batch_capacity")
        if len(items) != batch_size or tuple(item.candidate_index for item in items) != tuple(range(batch_size)):
            raise ValueError("items must contain candidate indices 0..N-1 exactly once in order")
        if not batch_size <= batch_capacity <= MAX_GRASP_BATCH_SIZE:
            raise ValueError("batch_capacity must cover batch_size and not exceed the limit")
        scene_token = _required_text(self.scene_token, name="scene_token")
        for item in items:
            for trajectory in (item.approach, item.grasp, item.lift):
                if trajectory is not None and trajectory.scene_token != scene_token:
                    raise ValueError("all batch trajectories must use the plan scene_token")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "batch_capacity", batch_capacity)
        object.__setattr__(self, "planning_time_s", _non_negative_float(self.planning_time_s, name="planning_time_s"))
        object.__setattr__(self, "scene_token", scene_token)

    @classmethod
    def create(cls, **values) -> NativeGraspBatchPlan:
        return cls(**values)


@dataclass(frozen=True)
class NativePlacePlan:
    """The three native trajectories composing an attached-object place."""

    approach: NativeTrajectory
    place: NativeTrajectory
    retreat: NativeTrajectory
    scene_token: str | None = field(init=False)

    def __post_init__(self) -> None:
        phases = (self.approach, self.place, self.retreat)
        if any(not isinstance(phase, NativeTrajectory) for phase in phases):
            raise ValueError("approach, place and retreat must be NativeTrajectory values")
        tokens = {phase.scene_token for phase in phases}
        if len(tokens) != 1:
            raise ValueError("all place phases must use the same scene_token")
        object.__setattr__(self, "scene_token", phases[0].scene_token)

    @classmethod
    def create(
        cls,
        *,
        approach: object,
        place: object,
        retreat: object,
    ) -> NativePlacePlan:
        return cls(approach=approach, place=place, retreat=retreat)


@dataclass(frozen=True)
class NativeIkResult:
    """Native cuRobo IK output before Tianji joint-order conversion."""

    reachable: bool
    joints_rad: tuple[float, ...] | None
    joint_names: tuple[str, ...]
    position_error_m: float
    rotation_error_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reachable",
            _required_bool(self.reachable, name="reachable"),
        )
        if self.joints_rad is None:
            joints = None
        else:
            joints = _finite_vector(
                self.joints_rad,
                name="joints_rad",
                length=7,
            )
        if self.reachable and joints is None:
            raise ValueError("joints_rad is required when reachable is true")
        object.__setattr__(self, "joints_rad", joints)
        object.__setattr__(self, "joint_names", _joint_names(self.joint_names))
        object.__setattr__(
            self,
            "position_error_m",
            _non_negative_float(
                self.position_error_m,
                name="position_error_m",
            ),
        )
        object.__setattr__(
            self,
            "rotation_error_rad",
            _non_negative_float(
                self.rotation_error_rad,
                name="rotation_error_rad",
            ),
        )


@dataclass(frozen=True)
class TrajectoryResult:
    """cuRobo source and execution trajectories in Tianji degree units."""

    # ========================================
    # MODIFICATION START: retain cuRobo source trajectory for plotting
    # ========================================
    curobo_points_deg: tuple[tuple[float, ...], ...]
    curobo_sample_dt_s: float
    # ========================================
    # MODIFICATION END: retain cuRobo source trajectory for plotting
    # ========================================
    points_deg: tuple[tuple[float, ...], ...]
    joint_names: tuple[str, ...]
    sample_dt_s: float
    planning_time_s: float
    position_error_m: float
    rotation_error_rad: float
    scene_token: str | None = None
    # ========================================
    # MODIFICATION START: derive cuRobo source point count
    # ========================================
    curobo_point_count: int = field(init=False)
    # ========================================
    # MODIFICATION END: derive cuRobo source point count
    # ========================================
    point_count: int = field(init=False)
    planner: str = field(default="curobo", init=False)

    trajectory_point_scale: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_point_scale", _trajectory_point_scale(self.trajectory_point_scale),
        )
        # ========================================
        # MODIFICATION START: validate cuRobo source trajectory metadata
        # ========================================
        curobo_points = _finite_matrix(
            self.curobo_points_deg,
            name="curobo_points_deg",
            columns=7,
            minimum_rows=1,
        )
        object.__setattr__(
            self,
            "curobo_points_deg",
            curobo_points,
        )
        object.__setattr__(
            self,
            "curobo_sample_dt_s",
            _positive_float(
                self.curobo_sample_dt_s,
                name="curobo_sample_dt_s",
            ),
        )
        object.__setattr__(
            self,
            "curobo_point_count",
            len(curobo_points),
        )
        # ========================================
        # MODIFICATION END: validate cuRobo source trajectory metadata
        # ========================================
        points = _finite_matrix(
            self.points_deg,
            name="points_deg",
            columns=7,
            minimum_rows=6,
        )
        object.__setattr__(self, "points_deg", points)
        object.__setattr__(self, "joint_names", _joint_names(self.joint_names))
        object.__setattr__(
            self,
            "sample_dt_s",
            _positive_float(self.sample_dt_s, name="sample_dt_s"),
        )
        object.__setattr__(
            self,
            "planning_time_s",
            _non_negative_float(
                self.planning_time_s,
                name="planning_time_s",
            ),
        )
        object.__setattr__(
            self,
            "position_error_m",
            _non_negative_float(
                self.position_error_m,
                name="position_error_m",
            ),
        )
        object.__setattr__(
            self,
            "rotation_error_rad",
            _non_negative_float(
                self.rotation_error_rad,
                name="rotation_error_rad",
            ),
        )
        if self.scene_token is not None:
            object.__setattr__(
                self,
                "scene_token",
                _required_text(self.scene_token, name="scene_token"),
            )
        object.__setattr__(self, "point_count", len(points))

    @classmethod
    def create(
        cls,
        *,
        # ========================================
        # MODIFICATION START: accept cuRobo source trajectory metadata
        # ========================================
        curobo_points_deg: object,
        curobo_sample_dt_s: object,
        # ========================================
        # MODIFICATION END: accept cuRobo source trajectory metadata
        # ========================================
        points_deg: object,
        joint_names: object,
        sample_dt_s: object,
        planning_time_s: object,
        position_error_m: object,
        rotation_error_rad: object,
        scene_token: object = None,
        trajectory_point_scale: object = None,
    ) -> TrajectoryResult:
        return cls(
            # ========================================
            # MODIFICATION START: construct cuRobo source trajectory metadata
            # ========================================
            curobo_points_deg=curobo_points_deg,
            curobo_sample_dt_s=curobo_sample_dt_s,
            # ========================================
            # MODIFICATION END: construct cuRobo source trajectory metadata
            # ========================================
            points_deg=points_deg,
            joint_names=joint_names,
            sample_dt_s=sample_dt_s,
            planning_time_s=planning_time_s,
            position_error_m=position_error_m,
            rotation_error_rad=rotation_error_rad,
            scene_token=scene_token,
            trajectory_point_scale=trajectory_point_scale,
        )


@dataclass(frozen=True)
class IkResult:
    """IK result in Tianji joint order and degree units."""

    reachable: bool
    joints_deg: tuple[float, ...] | None
    position_error_m: float
    rotation_error_rad: float
    planner: str = field(default="curobo", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reachable",
            _required_bool(self.reachable, name="reachable"),
        )
        if self.joints_deg is None:
            joints = None
        else:
            joints = _finite_vector(
                self.joints_deg,
                name="joints_deg",
                length=7,
            )
        if self.reachable and joints is None:
            raise ValueError("joints_deg is required when reachable is true")
        object.__setattr__(self, "joints_deg", joints)
        object.__setattr__(
            self,
            "position_error_m",
            _non_negative_float(
                self.position_error_m,
                name="position_error_m",
            ),
        )
        object.__setattr__(
            self,
            "rotation_error_rad",
            _non_negative_float(
                self.rotation_error_rad,
                name="rotation_error_rad",
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        reachable: object,
        joints_deg: object,
        position_error_m: object,
        rotation_error_rad: object,
    ) -> IkResult:
        return cls(
            reachable=reachable,
            joints_deg=joints_deg,
            position_error_m=position_error_m,
            rotation_error_rad=rotation_error_rad,
        )


@dataclass(frozen=True)
class HealthResult:
    """Service readiness without backend objects or credentials."""

    ready: bool
    arms_ready: tuple[str, ...]
    scene: str
    external_obstacle_count: int
    curobo_source: str
    curobo_version: str
    gpu_available: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ready",
            _required_bool(self.ready, name="ready"),
        )
        object.__setattr__(
            self,
            "arms_ready",
            _ready_arms(self.arms_ready),
        )
        object.__setattr__(
            self,
            "scene",
            _required_text(self.scene, name="scene"),
        )
        object.__setattr__(
            self,
            "external_obstacle_count",
            _non_negative_integer(
                self.external_obstacle_count,
                name="external_obstacle_count",
            ),
        )
        object.__setattr__(
            self,
            "curobo_source",
            _required_text(self.curobo_source, name="curobo_source"),
        )
        object.__setattr__(
            self,
            "curobo_version",
            _required_text(self.curobo_version, name="curobo_version"),
        )
        object.__setattr__(
            self,
            "gpu_available",
            _required_bool(self.gpu_available, name="gpu_available"),
        )


class CuroboToolError(RuntimeError):
    """Stable tool error with an HTTP status kept outside its JSON envelope."""

    def __init__(
        self,
        message: object,
        *,
        stage: object,
        code: object,
        retryable: object = False,
        details: object = None,
        status: object = 500,
    ) -> None:
        normalized_message = _required_text(message, name="message")
        super().__init__(normalized_message)
        self.message = normalized_message
        self.stage = _required_text(stage, name="stage")
        self.code = _required_text(code, name="code")
        self.retryable = _required_bool(retryable, name="retryable")
        self._details_json = _json_details_snapshot(details)
        normalized_status = _positive_integer(status, name="status")
        if normalized_status < 100 or normalized_status > 599:
            raise ValueError("status must be an HTTP status from 100 to 599")
        self.status = normalized_status

    @property
    def details(self) -> dict[str, object]:
        return json.loads(self._details_json)

    def to_dict(self) -> dict[str, object]:
        return {
            "error": {
                "stage": self.stage,
                "code": self.code,
                "message": str(self),
                "retryable": self.retryable,
                "details": self.details,
            }
        }


__all__ = [
    "CuroboToolError",
    "FixedEsdfCandidateValidationResult",
    "HealthResult",
    "IkResult",
    "MAX_AXIS_LINEAR_DISTANCE_M",
    "MAX_PLAN_ATTEMPTS",
    "NativeIkResult",
    "NativeTrajectory",
    "PlanAxisLinearRequest",
    "PlanToPoseRequest",
    "SolveIkRequest",
    "TrajectoryValidationResult",
    "TrajectoryResult",
    "ValidateTrajectoryRequest",
]
# ========================================
# MODIFICATION END: add lightweight typed cuRobo contracts
# ========================================

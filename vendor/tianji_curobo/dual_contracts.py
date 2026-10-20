"""GPU-free contracts for a synchronized fourteen-joint planning problem."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from .contracts import (_finite_vector, _finite_float, _plan_attempts,
                        _collision_clearance_profile, _rigid_matrix)

ARMS = ("A", "B")
JOINT_NAMES = {"A": tuple(f"Joint{i}_L" for i in range(1, 8)),
               "B": tuple(f"Joint{i}_R" for i in range(1, 8))}


def arm_values(value, name, convert):
    if not isinstance(value, Mapping) or set(value) != set(ARMS):
        raise ValueError(f"{name} must contain exactly A and B")
    return {arm: convert(value[arm], f"{name}.{arm}") for arm in ARMS}


def joint_vector(value, name):
    return _finite_vector(value, name=name, length=7)


def paths(value, name):
    def path(rows, label):
        if not isinstance(rows, (list, tuple)) or not 2 <= len(rows) <= 100001:
            raise ValueError(f"{label} must contain 2..100001 points")
        return tuple(joint_vector(row, label) for row in rows)
    result = arm_values(value, name, path)
    if len(result["A"]) != len(result["B"]):
        raise ValueError("dual trajectories must have identical sample counts")
    return result


@dataclass(frozen=True)
class DualPlanRequest:
    start_joints_deg: dict
    goal_joints_deg: dict | None
    goal_transforms_world: dict | None
    trajectory_point_scale: float
    freq_hz: int
    max_attempts: int
    trajectory_collision_clearance_m: float
    optimizer_collision_activation_distance_m: float

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("dual plan request must be an object")
        start = arm_values(value.get("start_joints_deg"), "start_joints_deg", joint_vector)
        joint, pose = value.get("goal_joints_deg"), value.get("goal_transforms_world")
        if (joint is None) == (pose is None):
            raise ValueError("provide exactly one dual joint or dual pose goal")
        if joint is not None:
            joint = arm_values(joint, "goal_joints_deg", joint_vector)
        if pose is not None:
            pose = arm_values(pose, "goal_transforms_world", lambda v, n: _rigid_matrix(v, name=n))
        scale = _finite_float(value.get("trajectory_point_scale", 3.), name="trajectory_point_scale")
        if not 1 <= scale <= 10:
            raise ValueError("trajectory_point_scale must be 1..10")
        freq = value.get("freq_hz", 50)
        if isinstance(freq, bool) or not isinstance(freq, int) or not 1 <= freq <= 1000:
            raise ValueError("freq_hz must be an integer in 1..1000")
        clearance, activation = _collision_clearance_profile(
            value.get("trajectory_collision_clearance_m", .01),
            value.get("optimizer_collision_activation_distance_m", .03))
        return cls(start, joint, pose, scale, freq, _plan_attempts(value.get("max_attempts", 5)), clearance, activation)

    def to_mapping(self):
        return asdict(self)


@dataclass(frozen=True)
class DualValidationRequest:
    points_deg: dict
    expected_scene_token: str | None
    trajectory_collision_clearance_m: float
    optimizer_collision_activation_distance_m: float

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("dual validation request must be an object")
        token = value.get("expected_scene_token")
        if token is not None and (not isinstance(token, str) or not token):
            raise ValueError("expected_scene_token must be a non-empty string")
        clearance, activation = _collision_clearance_profile(
            value.get("trajectory_collision_clearance_m", .01),
            value.get("optimizer_collision_activation_distance_m", .03))
        return cls(paths(value.get("points_deg"), "points_deg"), token, clearance, activation)

    def to_mapping(self):
        return asdict(self)


def validate_plan_result(value):
    if not isinstance(value, Mapping):
        raise ValueError("dual plan response must be an object")
    command = paths(value.get("trajectories"), "trajectories")
    raw = paths(value.get("raw_trajectories"), "raw_trajectories")
    scale = _finite_float(value.get("trajectory_point_scale"), name="trajectory_point_scale")
    if not 1 <= scale <= 10:
        raise ValueError("invalid dual result point scale")
    expected_count = math.ceil(math.nextafter((len(raw["A"]) - 1) * scale, -math.inf)) + 1
    if len(command["A"]) != expected_count or value.get("point_count") != expected_count:
        raise ValueError("dual result has inconsistent Hermite point count")
    if value.get("joint_names") != {arm: list(JOINT_NAMES[arm]) for arm in ARMS}:
        raise ValueError("dual result joint names do not match execution order")
    if value.get("ok") is not True or value.get("planner") != "curobo_dual":
        raise ValueError("dual result must be a successful joint problem")
    for name in ("sample_dt_s", "raw_sample_dt_s"):
        if _finite_float(value.get(name), name=name) <= 0:
            raise ValueError(f"{name} must be positive")
    if value.get("time_parameterization") != "quintic_hermite":
        raise ValueError("dual trajectory must use shared quintic Hermite timing")
    token = value.get("scene_token")
    if token is not None and (not isinstance(token, str) or not token):
        raise ValueError("invalid dual scene token")
    return dict(value)

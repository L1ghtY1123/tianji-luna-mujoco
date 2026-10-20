# ========================================
# MODIFICATION START: add strict axis-linear geometry validation
# ========================================
"""Frame math and strict validation for fixed-orientation axis motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_REFERENCE_FRAMES = frozenset({"world", "tool"})


@dataclass(frozen=True)
class AxisLinearMetrics:
    """Worst geometric deviations measured across a Cartesian pose path."""

    start_error_m: float
    max_line_error_m: float
    max_orientation_error_rad: float
    endpoint_error_m: float
    max_progress_violation_m: float


# ========================================
# MODIFICATION START: expose structured strict-line validation failures
# ========================================
class AxisLinearValidationError(ValueError):
    """A geometric constraint failure with stable machine-readable values."""

    def __init__(self, *, metric, measured, limit, metrics, reason):
        self.metric = str(metric)
        self.measured = float(measured)
        self.limit = float(limit)
        self.metrics = metrics
        self.reason = str(reason)
        super().__init__(self.reason)

    @property
    def normalized_excess(self):
        denominator = max(abs(self.limit), np.finfo(float).eps)
        return max(0.0, (self.measured - self.limit) / denominator)

    def to_details(self):
        return {
            "metric": self.metric,
            "measured": self.measured,
            "limit": self.limit,
            "reason": self.reason,
        }


def _constraint_failure(*, metric, measured, limit, metrics, label, unit):
    reason = (
        f"{label} {measured:.6f} {unit} exceeds "
        f"{limit:.6f} {unit}"
    )
    return AxisLinearValidationError(
        metric=metric,
        measured=measured,
        limit=limit,
        metrics=metrics,
        reason=reason,
    )
# ========================================
# MODIFICATION END: expose structured strict-line validation failures
# ========================================


def _finite_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numbers") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numbers")
    return array


def _normalized_choice(
    value: object,
    *,
    name: str,
    choices: frozenset[str],
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} is invalid")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} is invalid")
    return normalized


def _finite_non_negative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative number") from error
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return converted


def quaternion_wxyz_to_matrix(quaternion: object) -> np.ndarray:
    """Convert a finite WXYZ quaternion to a normalized rotation matrix."""

    values = _finite_array(quaternion, name="quaternion", shape=(4,))
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("quaternion must have a non-zero norm")
    w, x, y, z = values / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def validate_axis_linear_poses(
    *,
    positions_m: object,
    rotations: object,
    start_position_m: object,
    start_rotation: object,
    reference_frame: object,
    axis: object,
    distance_m: object,
    start_tolerance_m: object = 0.001,
    line_tolerance_m: object = 0.001,
    orientation_tolerance_rad: object = np.deg2rad(0.5),
    endpoint_tolerance_m: object = 0.001,
    progress_tolerance_m: object = 0.001,
) -> AxisLinearMetrics:
    """Validate that sampled poses form the requested strict straight line."""

    try:
        positions = np.asarray(positions_m, dtype=float)
        orientation_samples = np.asarray(rotations, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("pose samples must contain finite numbers") from error
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("positions_m must have shape (N, 3) with N >= 2")
    if orientation_samples.shape != (positions.shape[0], 3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(orientation_samples)):
        raise ValueError("pose samples must contain finite numbers")

    start_position = _finite_array(
        start_position_m,
        name="start_position_m",
        shape=(3,),
    )
    initial_rotation = _finite_array(
        start_rotation,
        name="start_rotation",
        shape=(3, 3),
    )
    frame = _normalized_choice(
        reference_frame,
        name="reference_frame",
        choices=_REFERENCE_FRAMES,
    )
    normalized_axis = _normalized_choice(
        axis,
        name="axis",
        choices=frozenset(_AXIS_INDEX),
    )
    if isinstance(distance_m, bool):
        raise ValueError("distance_m must be a finite non-zero number")
    try:
        distance = float(distance_m)
    except (TypeError, ValueError) as error:
        raise ValueError("distance_m must be a finite non-zero number") from error
    if not np.isfinite(distance) or distance == 0.0:
        raise ValueError("distance_m must be a finite non-zero number")

    start_tolerance = _finite_non_negative(
        start_tolerance_m,
        name="start_tolerance_m",
    )
    line_tolerance = _finite_non_negative(line_tolerance_m, name="line_tolerance_m")
    orientation_tolerance = _finite_non_negative(
        orientation_tolerance_rad,
        name="orientation_tolerance_rad",
    )
    endpoint_tolerance = _finite_non_negative(
        endpoint_tolerance_m,
        name="endpoint_tolerance_m",
    )
    progress_tolerance = _finite_non_negative(
        progress_tolerance_m,
        name="progress_tolerance_m",
    )

    basis = np.eye(3)[_AXIS_INDEX[normalized_axis]]
    direction = basis if frame == "world" else initial_rotation @ basis
    direction = direction / np.linalg.norm(direction)
    displacement = positions - start_position
    axial_distance = displacement @ direction
    lateral = displacement - axial_distance[:, None] * direction[None, :]
    line_errors = np.linalg.norm(lateral, axis=1)

    orientation_errors = np.array(
        [
            _rotation_angle(initial_rotation.T @ sample_rotation)
            for sample_rotation in orientation_samples
        ]
    )
    directed_progress = np.sign(distance) * axial_distance
    progress_steps = np.diff(directed_progress)
    backward_violation = max(0.0, float(np.max(-progress_steps)))
    before_start_violation = max(0.0, float(-np.min(directed_progress)))
    overshoot_violation = max(
        0.0,
        float(np.max(directed_progress) - abs(distance)),
    )
    expected_endpoint = start_position + distance * direction

    metrics = AxisLinearMetrics(
        start_error_m=float(np.linalg.norm(positions[0] - start_position)),
        max_line_error_m=float(np.max(line_errors)),
        max_orientation_error_rad=float(np.max(orientation_errors)),
        endpoint_error_m=float(np.linalg.norm(positions[-1] - expected_endpoint)),
        max_progress_violation_m=max(
            backward_violation,
            before_start_violation,
            overshoot_violation,
        ),
    )

    effective_endpoint_tolerance = min(endpoint_tolerance, abs(distance) * 0.25)
# ========================================
# MODIFICATION START: raise structured strict-line validation failures
# ========================================
    if metrics.start_error_m > start_tolerance:
        raise _constraint_failure(
            metric="start_error_m",
            measured=metrics.start_error_m,
            limit=start_tolerance,
            metrics=metrics,
            label="start error",
            unit="m",
        )
    if metrics.max_line_error_m > line_tolerance:
        raise _constraint_failure(
            metric="max_line_error_m",
            measured=metrics.max_line_error_m,
            limit=line_tolerance,
            metrics=metrics,
            label="line error",
            unit="m",
        )
    if metrics.max_orientation_error_rad > orientation_tolerance:
        raise _constraint_failure(
            metric="max_orientation_error_rad",
            measured=metrics.max_orientation_error_rad,
            limit=orientation_tolerance,
            metrics=metrics,
            label="orientation error",
            unit="rad",
        )
    if metrics.endpoint_error_m > effective_endpoint_tolerance:
        raise _constraint_failure(
            metric="endpoint_error_m",
            measured=metrics.endpoint_error_m,
            limit=effective_endpoint_tolerance,
            metrics=metrics,
            label="endpoint error",
            unit="m",
        )
    if metrics.max_progress_violation_m > progress_tolerance:
        raise _constraint_failure(
            metric="max_progress_violation_m",
            measured=metrics.max_progress_violation_m,
            limit=progress_tolerance,
            metrics=metrics,
            label="progress violation",
            unit="m",
        )
# ========================================
# MODIFICATION END: raise structured strict-line validation failures
# ========================================
    return metrics


__all__ = [
    "AxisLinearMetrics",
# ========================================
# MODIFICATION START: export structured strict-line validation failures
# ========================================
    "AxisLinearValidationError",
# ========================================
# MODIFICATION END: export structured strict-line validation failures
# ========================================
    "quaternion_wxyz_to_matrix",
    "validate_axis_linear_poses",
]
# ========================================
# MODIFICATION END: add strict axis-linear geometry validation
# ========================================

"""Measured grasp evidence, independent of model decisions and planner success."""
import numpy as np

GRIPPER_LIMIT_NOISE_TOLERANCE_RAD = 1e-4


def verify_gripper_position(feedback, *, tolerance_fraction=.02, stable_window_s=.2):
    """A close command stopped short of its actual target indicates holding."""
    unknown = {"success": False, "status": "unknown", "source": "gripper_position"}
    samples = np.asarray(feedback.get("gripper_position_samples", []), dtype=float)
    limits = np.asarray(feedback.get("gripper_limits_rad", []), dtype=float)
    closing = feedback.get('gripper_closing_command', feedback.get('gripper_closed_command'))
    target = feedback.get('gripper_command_position_rad', float(limits[1]) if limits.shape == (2,) else None)
    if (closing is not True or samples.ndim != 2
            or samples.shape[1] != 2 or len(samples) < 2 or limits.shape != (2,)
            or not np.isfinite(samples).all() or not np.isfinite(limits).all() or limits[1] <= limits[0]):
        return {**unknown, "reason": "position_feedback_unavailable_or_no_full_close_command"}
    if (type(target) not in (int, float) or not np.isfinite(target)
            or not limits[0] < target <= limits[1]):
        return {**unknown, 'reason': 'invalid_closing_command_target'}
    start = max(0, np.searchsorted(samples[:, 0], samples[-1, 0] - stable_window_s, side="right") - 1)
    window = samples[start:]
    if (window[:, 1].min() < limits[0] - GRIPPER_LIMIT_NOISE_TOLERANCE_RAD
            or window[:, 1].max() > limits[1] + GRIPPER_LIMIT_NOISE_TOLERANCE_RAD):
        return {**unknown, 'reason': 'gripper_position_outside_authored_limits',
                'position_rad': float(window[-1, 1]), 'limits_rad': limits.tolist()}
    tolerance = (limits[1] - limits[0]) * tolerance_fraction
    if window[-1, 0] - window[0, 0] < stable_window_s - 1e-6 or np.ptp(window[:, 1]) > tolerance:
        return {**unknown, "reason": "position_not_stable"}
    position = float(np.median(window[:, 1]))
    status = "held" if target - position > tolerance else "empty"
    return {"success": status == "held", "status": status, "source": "gripper_position",
            "position_rad": position, "empty_closed_rad": float(target), "tolerance_rad": float(tolerance),
            "stable_duration_s": float(window[-1, 0] - window[0, 0])}


def gripper_open_confirmed(feedback, tolerance_fraction=.02):
    limits = np.asarray(feedback.get("gripper_limits_rad", []), dtype=float)
    position = feedback.get("gripper_position_rad")
    return bool(limits.shape == (2,) and position is not None and np.isfinite(position)
                and abs(position - limits[0]) <= (limits[1] - limits[0]) * tolerance_fraction)


def combine_holding_evidence(position, visual=None):
    visual = visual or {"status": "unknown", "reason": "not_requested"}
    statuses = {item["status"] for item in (position, visual) if item["status"] != "unknown"}
    conflict = len(statuses) > 1 or any(item.get("conflicting_evidence") for item in (position, visual))
    status = next(iter(statuses)) if len(statuses) == 1 and not conflict else "unknown"
    return {"success": status == "held", "status": status, "position": position, "visual": visual,
            "source": "position_and_optional_visual", "conflicting_evidence": conflict}

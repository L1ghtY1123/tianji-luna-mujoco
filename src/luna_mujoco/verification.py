"""Low-level gripper actuator settling diagnostics."""
import numpy as np

GRIPPER_LIMIT_NOISE_TOLERANCE_RAD = 1e-4


def gripper_settling_evidence(samples, *, stable_window_s, position_tolerance_rad, velocity_tolerance_rad_s):
    """Evaluate measured position and velocity over a complete command-local window."""
    values = np.asarray(samples, dtype=float)
    unavailable = {'stable': False, 'reason': 'settling_window_incomplete'}
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) < 2
            or not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0)):
        return unavailable
    start = max(0, np.searchsorted(values[:, 0], values[-1, 0] - stable_window_s, side='right') - 1)
    window = values[start:]
    duration = float(window[-1, 0] - window[0, 0])
    if duration < stable_window_s - 1e-6:
        return unavailable
    dt = np.diff(window[:, 0])
    squared_speed = np.square(window[:, 2])
    rms_speed = float(np.sqrt(np.sum(dt * (squared_speed[:-1] + squared_speed[1:]) / 2) / duration))
    span = float(np.ptp(window[:, 1]))
    return {'stable': span <= position_tolerance_rad and rms_speed <= velocity_tolerance_rad_s,
            'window_duration_s': duration, 'position_span_rad': span,
            'rms_velocity_rad_s': rms_speed, 'maximum_velocity_rad_s': float(np.max(np.abs(window[:, 2]))),
            'position_tolerance_rad': position_tolerance_rad, 'velocity_tolerance_rad_s': velocity_tolerance_rad_s,
            'velocity_statistic': 'time_weighted_rms', 'sample_count': len(window)}

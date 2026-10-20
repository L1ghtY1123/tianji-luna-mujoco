"""CPU interpolation of one cuRobo trajectory; SDK planning remains unchanged."""
from __future__ import annotations

import math

import numpy as np


def quintic_hermite_points(
    positions, velocities, accelerations, *, sample_dt_s, point_scale
):
    """Match endpoint q/v/a and expand point intervals independently of command Hz."""
    if isinstance(point_scale, bool) or isinstance(sample_dt_s, bool):
        raise ValueError("point_scale and sample_dt_s must be finite numbers")
    scale, dt = float(point_scale), float(sample_dt_s)
    if not math.isfinite(scale) or not 1.0 <= scale <= 10.0:
        raise ValueError("point_scale must be between 1 and 10")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("sample_dt_s must be positive and finite")
    q = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    a = np.asarray(accelerations, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] not in (7, 14) or len(q) < 2:
        raise ValueError("positions must contain at least two seven- or fourteen-joint points")
    if v.shape != q.shape or a.shape != q.shape:
        raise ValueError("positions, velocities and accelerations must have matching shapes")
    if not all(np.isfinite(values).all() for values in (q, v, a)):
        raise ValueError("trajectory positions and derivatives must be finite")
    intervals = math.ceil(math.nextafter((len(q) - 1) * scale, -math.inf))
    if intervals + 1 > 100_001:
        raise ValueError("interpolated trajectory exceeds 100001 points")
    if scale == 1.0:
        return q.copy()

    from scipy.interpolate import BPoly

    # Derivatives belong to the same dense SDK state as q. Index is the curve
    # parameter here; device frequency must not change the output point count.
    derivatives = np.stack((q, v * dt, a * dt * dt), axis=1)
    curve = BPoly.from_derivatives(
        np.arange(len(q), dtype=np.float64), derivatives, extrapolate=False,
    )
    points = curve(np.linspace(0.0, len(q) - 1.0, intervals + 1))
    if scale.is_integer():
        points[::int(scale)] = q
    points[0], points[-1] = q[0], q[-1]
    if not np.isfinite(points).all():
        raise ValueError("interpolated trajectory must be finite")
    return points

"""A bounded, observed support surface and a pure normal departure policy.

Only identified background plane returns are represented by the named table
cuboid. Other returns remain in the ESDF, including near-plane residuals.
"""

import hashlib
import itertools
import json

import numpy as np
from scipy.spatial import ConvexHull, QhullError

SUPPORT_NAME = "semantic_table_forbidden"
SUPPORT_POLICY = "observed_plane_normal_departure"
CLASSIFICATION_TOLERANCE_M = 0.001
MOTION_TOLERANCE_M = 1e-6


def revision(value):
    data = {key: item for key, item in value.items() if key != "revision"}
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_support_plane(points, normal, offset, table_model):
    """Return no exemption for slopes that the horizontal completion cannot model."""
    points, normal = np.asarray(points, dtype=float), np.asarray(normal, dtype=float)
    if not np.isfinite(points).all() or normal.shape != (3,) or normal[2] < 0.999999:
        return None
    residual = points @ normal + offset
    inliers = points[np.abs(residual) <= CLASSIFICATION_TOLERANCE_M]
    # A raw return must lie inside the replacement cuboid, including its height.
    center = np.asarray(table_model["pose"][:3])
    half = np.asarray(table_model["dims"]) / 2
    inliers = inliers[np.all(np.abs(inliers - center) <= half, axis=1)]
    if len(inliers) < 3:
        return None
    try:
        hull = ConvexHull(inliers[:, :2])
    except QhullError:
        return None
    result = {
        "schema_version": 1,
        "surface_id": SUPPORT_NAME,
        "normal_xyz": normal.tolist(),
        "offset_m": float(offset),
        "height_m": float(-offset / normal[2]),
        # Never exempt a bump that was absorbed by the plane fit's tolerance.
        # The uninflated object must clear the upper envelope of classified returns.
        "contact_offset_m": float(offset - np.max(inliers @ normal + offset)),
        "inlier_count": len(inliers),
        "source_inliers_sha256": hashlib.sha256(
            np.ascontiguousarray(inliers, dtype="<f8").tobytes()
        ).hexdigest(),
        "observed_polygon_xy_m": inliers[hull.vertices, :2].tolist(),
        "classification_tolerance_m": CLASSIFICATION_TOLERANCE_M,
        "replacement_cuboid": table_model,
        "classification": "background_only_observed_plane_inliers",
    }
    result["revision"] = revision(result)
    return result


def inside_polygon(points_xy, polygon):
    polygon = np.asarray(polygon, dtype=float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    delta = np.asarray(points_xy)[..., None, :] - polygon
    cross = edges[:, 0] * delta[..., 1] - edges[:, 1] * delta[..., 0]
    return np.all(cross >= -1e-12, axis=-1)


def support_mask(points, plane):
    points = np.asarray(points, dtype=float)
    residual = points @ np.asarray(plane["normal_xyz"]) + plane["offset_m"]
    return (
        np.isfinite(points).all(axis=-1)
        & (np.abs(residual) <= plane["classification_tolerance_m"])
        & inside_polygon(points[..., :2], plane["observed_polygon_xy_m"])
    )


class SupportDeparture:
    """A contact exemption ends irrevocably once the sphere cover clears support.

    The object AABB must start nonpenetrating within the observed polygon. While
    its padded sphere cover touches the support cuboid, the rigid object may only
    translate monotonically along the observed upward normal, with no rotation.
    All robot/world, object/other-world and object/robot checks remain mandatory.
    """

    def __init__(self, plane, observed_bounds_ee_m):
        if plane.get("revision") != revision(plane) or plane.get("surface_id") != SUPPORT_NAME:
            raise ValueError("support plane revision is invalid")
        self.plane = plane
        self.normal = np.asarray(plane["normal_xyz"], dtype=float)
        if not np.isfinite(self.normal).all() or not np.isclose(np.linalg.norm(self.normal), 1):
            raise ValueError("support normal must be a finite unit vector")
        lower, upper = np.asarray(observed_bounds_ee_m, dtype=float)
        self.corners = np.asarray(list(itertools.product(*zip(lower, upper))))
        self.initial = None
        self.previous_displacement = 0.0
        self.departed = False
        self.failed = False
        self.reason = None
        self.exempt_sample_count = 0

    def evaluate(self, poses, spheres):
        poses, spheres = np.asarray(poses, dtype=float), np.asarray(spheres, dtype=float)
        if (
            poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or spheres.ndim != 3
            or spheres.shape[0] != len(poses)
            or spheres.shape[2] != 4
            or not np.isfinite(poses).all()
            or not np.isfinite(spheres).all()
        ):
            raise ValueError("support departure requires finite poses and collision spheres")
        exemptions, valid = [], []
        table = self.plane["replacement_cuboid"]
        table_top = table["pose"][2] + table["dims"][2] / 2
        for pose, balls in zip(poses, spheres, strict=True):
            corners = self.corners @ pose[:3, :3].T + pose[:3, 3]
            if self.initial is None:
                self.initial = pose.copy()
                initial_distance = np.min(corners @ self.normal + self.plane["contact_offset_m"])
                # This policy applies only to an observed initial support contact.
                if not -MOTION_TOLERANCE_M <= initial_distance <= CLASSIFICATION_TOLERANCE_M:
                    self.failed, self.reason = True, "initial_support_contact_unverified"
            touching = np.min(balls[:, 2] - balls[:, 3]) <= table_top
            delta = pose[:3, 3] - self.initial[:3, 3]
            displacement = float(delta @ self.normal)
            safe_contact = (
                np.min(corners @ self.normal + self.plane["contact_offset_m"])
                >= -MOTION_TOLERANCE_M
                and inside_polygon(corners[:, :2], self.plane["observed_polygon_xy_m"]).all()
                and np.linalg.norm(delta - displacement * self.normal) <= MOTION_TOLERANCE_M
                and np.allclose(pose[:3, :3], self.initial[:3, :3], atol=1e-6, rtol=0)
                and displacement >= self.previous_displacement - MOTION_TOLERANCE_M
            )
            if touching and (self.departed or not safe_contact):
                self.failed, self.reason = (
                    True,
                    "support_departure_requires_nonpenetrating_monotonic_normal_motion",
                )
            # Require the first clear sample to follow the same departure motion;
            # after it, every collision query uses the full world again.
            if not self.departed and not safe_contact:
                self.failed, self.reason = (
                    True,
                    "support_departure_requires_nonpenetrating_monotonic_normal_motion",
                )
            exempt = touching and not self.failed and not self.departed
            self.exempt_sample_count += int(exempt)
            exemptions.append(bool(exempt))
            valid.append(not self.failed)
            if not touching:
                self.departed = True
            self.previous_displacement = displacement
        return exemptions, valid

    def proof(self):
        return {
            "policy": SUPPORT_POLICY,
            "plane": self.plane,
            "exempt_sample_count": self.exempt_sample_count,
            "departure_complete": self.departed,
            "failure_reason": self.reason,
            "motion_tolerance_m": MOTION_TOLERANCE_M,
            "coverage": "observed_polygon_only",
        }

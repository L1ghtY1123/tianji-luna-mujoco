"""Positive-only release endpoint evidence from the configured hard world checker."""

import numpy as np
from scipy.spatial.transform import Rotation

# Match the existing trajectory validator's numerical activation-cost epsilon.
CLEARANCE_COST_EPSILON = 1e-9


def release_clearance_evidence(spheres_tool_m, request, query_world_cost, *, scene_token):
    """Query the exact uploaded surface spheres at the requested TCP pose.

    The public checker returns activation costs, not signed metric distances.
    A zero cost proves neither reachability nor observed free-space coverage.
    """
    spheres = np.asarray(spheres_tool_m, dtype=float)
    position = np.asarray(request.goal_position_m, dtype=float)
    quaternion = np.asarray(request.goal_quaternion_wxyz, dtype=float)
    clearance = float(request.trajectory_collision_clearance_m)
    if (spheres.ndim != 2 or spheres.shape[1] != 4 or len(spheres) == 0
            or not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0)):
        raise ValueError("release_requires_uploaded_tool_spheres")
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) == 0 or not np.isfinite(clearance) or clearance < 0):
        raise ValueError("release_requires_finite_world_goal_and_clearance")
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    world = spheres.copy()
    world[:, :3] = spheres[:, :3] @ rotation.T + position
    # Radius already includes voxel discretization, measurement uncertainty and
    # attachment margin. The configured checker adds hard clearance exactly once.
    costs = np.asarray(query_world_cost(world[None, None]), dtype=float)
    if (costs.shape != (1, 1, len(spheres)) or not np.isfinite(costs).all() or np.any(costs < 0)):
        raise ValueError("release_world_cost_requires_finite_nonnegative_per_sphere_values")
    indices = np.flatnonzero(costs[0, 0] > CLEARANCE_COST_EPSILON)
    conflict = bool(len(indices))
    return {"conflict": conflict, "status": "conflict" if conflict else "no_conflict_detected",
            "scope": "attached_surface_spheres_at_requested_release_goal", "frame": "world",
            "arm": request.arm, "scene_token": scene_token,
            "goal_position_m": position.tolist(), "goal_quaternion_wxyz": quaternion.tolist(),
            "checked_sphere_count": len(spheres), "required_clearance_m": clearance,
            "conflicting_sphere_indices": indices.tolist(),
            "conflicting_spheres_world_m": world[indices].tolist(),
            "world_collision_costs": costs[0, 0].tolist(),
            "numerical_cost_epsilon": CLEARANCE_COST_EPSILON,
            "collision_source": "configured_hard_world_checker",
            "min_clearance_m": None, "distance_reporting": "activation_cost_only",
            "geometry_coverage": "measured_surface_voxels_with_uncertainty", "full_object_coverage": False}

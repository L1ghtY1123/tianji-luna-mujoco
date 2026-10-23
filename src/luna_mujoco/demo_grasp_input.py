"""Bounded inference-only sampling of the existing labeled side surface."""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

SURFACE_SPACING_M = .002
MAX_INPUT_POINTS = 12000


def sample_completed_side_surface(measured_points, completed_points):
    """Keep source points and add a uniform perimeter/height grid for GraspGen.

    The official 20-neighbour, 14 mm L1 filter otherwise deletes sparse prior
    rings. These are inference samples, not measured depth or mapping voxels.
    """
    measured = np.asarray(measured_points, dtype=float)
    completed = np.asarray(completed_points, dtype=float)
    try:
        vertices = measured[ConvexHull(measured[:, :2]).vertices, :2]
    except QhullError as error:
        raise ValueError("graspgen_surface_footprint_invalid") from error
    loop = np.vstack((vertices, vertices[:1]))
    arc = np.r_[0., np.linalg.norm(np.diff(loop, axis=0), axis=1).cumsum()]
    bottom, top = completed[:, 2].min(), completed[:, 2].max()
    perimeter_count = max(3, int(np.ceil(arc[-1] / SURFACE_SPACING_M)))
    height_count = max(2, int(np.ceil((top - bottom) / SURFACE_SPACING_M)) + 1)
    if len(completed) + perimeter_count * height_count > MAX_INPUT_POINTS:
        raise ValueError("graspgen_surface_sampling_budget_exceeded")
    distances = np.arange(perimeter_count) * arc[-1] / perimeter_count
    boundary = np.column_stack([np.interp(distances, arc, loop[:, axis]) for axis in (0, 1)])
    heights = np.linspace(bottom, top, height_count)
    side = np.column_stack((np.tile(boundary, (height_count, 1)),
                            np.repeat(heights, perimeter_count)))
    return np.vstack((completed, side))


def surface_sampling_evidence(source_count, actual_count):
    return {"source": "measured_convex_hull_side_surface_sampling",
            "spacing_m": SURFACE_SPACING_M, "maximum_point_count": MAX_INPUT_POINTS,
            "source_geometry_point_count": source_count, "actual_point_count": actual_count,
            "added_surface_point_count": actual_count - source_count,
            "added_points_are_measured": False,
            "scope": "graspgen_object_input_only"}

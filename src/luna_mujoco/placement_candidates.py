"""Bounded support candidates; containment is not collision or visibility evidence."""
import numpy as np
from scipy.spatial import ConvexHull


def subdivision_cell_centers(polygon, *, offsets_world_xy_m, cell_count):
    """One complete level of long-axis bisection: 4, 8 or 16 cell centers."""
    if cell_count not in (4, 8, 16):
        raise ValueError('placement_subdivision_level_invalid')
    vertices = np.asarray(polygon, dtype=float)
    offsets = np.asarray(offsets_world_xy_m, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1:] != (2,) or len(vertices) < 3
            or offsets.ndim != 2 or offsets.shape[1:] != (2,) or not len(offsets)
            or not np.isfinite(vertices).all() or not np.isfinite(offsets).all()):
        raise ValueError('placement_subdivision_input_invalid')
    cells = [(vertices.min(0), vertices.max(0))]
    axis = int(np.argmax(vertices.max(0) - vertices.min(0)))
    while len(cells) < cell_count:
        children = []
        for lower, upper in cells:
            middle = (lower[axis] + upper[axis]) / 2
            left_upper, right_lower = upper.copy(), lower.copy()
            left_upper[axis] = right_lower[axis] = middle
            children.extend(((lower.copy(), left_upper), (right_lower, upper.copy())))
        cells = children
        axis = 1 - axis
    offset = (offsets.min(0) + offsets.max(0)) / 2
    return np.array([(lower + upper) / 2 - offset for lower, upper in cells])


def subdivision_xy_candidates(polygon, *, offsets_world_xy_m, max_candidates=15):
    """Breadth-first long-axis bisection centers; both children remain eligible.

    These are proposals, not free-space proofs. The caller checks containment,
    collision and IK and stops when enough candidates pass.
    """
    vertices = np.asarray(polygon, dtype=float)
    offsets = np.asarray(offsets_world_xy_m, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1:] != (2,) or len(vertices) < 3
            or offsets.ndim != 2 or offsets.shape[1:] != (2,) or not len(offsets)
            or not np.isfinite(vertices).all() or not np.isfinite(offsets).all()
            or type(max_candidates) is not int or not 1 <= max_candidates <= 15):
        raise ValueError('placement_subdivision_input_invalid')
    # Center the full held model, including an off-center grasp, in each cell.
    offset = (offsets.min(0) + offsets.max(0)) / 2
    cells = [(vertices.min(0), vertices.max(0))]
    result = []
    for lower, upper in cells:
        center = (lower + upper) / 2
        result.append(center - offset)
        if len(result) == max_candidates:
            break
        axis = int(np.argmax(upper - lower))
        left_upper, right_lower = upper.copy(), lower.copy()
        left_upper[axis] = right_lower[axis] = center[axis]
        cells.extend(((lower.copy(), left_upper), (right_lower, upper.copy())))
    return np.asarray(result)


def support_xy_candidates(polygon, *, offsets_world_xy_m=None, clearance_m=.01, spacing_m=.02):
    vertices = np.asarray(polygon, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3
            or not np.isfinite(vertices).all() or not np.isfinite(spacing_m) or spacing_m <= 0):
        raise ValueError('placement_requires_measured_region_polygon')
    hull = ConvexHull(vertices)
    area = .5 * abs(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                          - vertices[:, 1] * np.roll(vertices[:, 0], -1)))
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    counts = np.minimum(25, np.maximum(2, np.ceil((upper - lower) / spacing_m).astype(int) + 1))
    axes = [np.linspace(lower[i], upper[i], counts[i]) for i in range(2)]
    grid = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 2)
    points = np.unique(np.vstack((vertices.mean(axis=0), grid)), axis=0)
    offsets = np.zeros((1, 2)) if offsets_world_xy_m is None else np.asarray(offsets_world_xy_m, float)
    if (offsets.ndim != 2 or offsets.shape[1] != 2 or not len(offsets)
            or not np.isfinite(offsets).all() or not np.isfinite(clearance_m) or clearance_m < 0):
        raise ValueError('placement_requires_finite_footprint')
    if not np.isclose(area, hull.volume, atol=1e-8, rtol=1e-6):
        # CAD sections can contain concave details. Never fill them with a hull:
        # a circle enclosing the entire footprint must fit the original polygon.
        end = np.roll(vertices, -1, axis=0)
        edge = end - vertices
        relative = points[:, None, :] - vertices
        length2 = np.sum(edge * edge, axis=1)
        fraction = np.clip(np.sum(relative * edge, axis=2) / np.maximum(length2, 1e-20), 0, 1)
        distance = np.linalg.norm(relative - fraction[:, :, None] * edge, axis=2).min(axis=1)
        crossings = ((vertices[:, 1] > points[:, None, 1]) != (end[:, 1] > points[:, None, 1]))
        crossing_x = (vertices[:, 0] + (points[:, None, 1] - vertices[:, 1])
                      * edge[:, 0] / np.where(edge[:, 1] == 0, 1., edge[:, 1]))
        inside = np.sum(crossings & (points[:, None, 0] < crossing_x), axis=1) % 2 == 1
        radius = np.linalg.norm(offsets, axis=1).max() + clearance_m
        return points[inside & (distance >= radius - 1e-9)]
    # Convex hull equations have unit normals. The largest footprint projection
    # proves every footprint point remains inside each measured boundary.
    extent = np.max(offsets @ hull.equations[:, :2].T, axis=0)
    distances = points @ hull.equations[:, :2].T + hull.equations[:, 2] + extent
    return points[np.all(distances <= -clearance_m + 1e-9, axis=1)]


def measured_region_candidates(destination, *, spacing_m=.02):
    reference = destination.get('placement_reference', {})
    floor = float(reference.get('floor_z_m', float('nan')))
    if not np.isfinite(floor):
        raise ValueError('placement_requires_measured_floor')
    points = support_xy_candidates(reference.get('region_polygon_world_xy_m'), spacing_m=spacing_m)
    return [{'candidate_id': index, 'position_world_m': [*xy.tolist(), floor],
             'source': reference.get('source'), 'observation_id': reference.get('observation_id'),
             'collision_checked': False, 'coverage_checked': False}
            for index, xy in enumerate(points)]

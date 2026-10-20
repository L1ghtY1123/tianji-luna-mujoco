"""A rectangle inside the measured horizontal floor's convex extent."""

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


def _largest_supported_rectangle(mask, dx, dy):
    heights = np.zeros(mask.shape[1], dtype=int)
    best, rectangle = 0., None
    for row, supported in enumerate(mask):
        heights = np.where(supported, heights + 1, 0)
        stack = []
        for column in range(len(heights) + 1):
            height = heights[column] if column < len(heights) else 0
            start = column
            while stack and stack[-1][1] > height:
                left, previous = stack.pop()
                width_m, height_m = (column - left - 1) * dx, (previous - 1) * dy
                area = width_m * height_m
                if min(width_m, height_m) >= .03 - 1e-9 and area > best:
                    best, rectangle = area, (left, column - 1, row - previous + 1, row)
                start = left
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    return rectangle


def supported_floor_rectangle(points_xy):
    """Use the measured plane extent, independent of camera sampling density.

    Occlusion and sparse sampling do not cut holes in a box's planar floor.
    Obstacles remain in the three-dimensional collision map. The rectangle
    stays inside the observed hull; no unobserved outer boundary is added.
    """
    points = np.unique(np.asarray(points_xy, dtype=float), axis=0)
    if len(points) < 30:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    vertices = points[hull.vertices]
    edges = np.roll(vertices, -1, axis=0) - vertices
    angles = np.unique(np.round(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2), 10))
    choices = []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        local = points @ rotation
        choices.append((float(np.prod(np.ptp(local, axis=0))), angle, rotation, local))
    _, _, rotation, local = min(choices, key=lambda item: (round(item[0], 10), item[1]))
    lower, upper = local.min(axis=0), local.max(axis=0)
    spacing = float(np.median(cKDTree(local).query(local, k=2)[0][:, 1]))
    if not np.isfinite(spacing) or spacing <= 0 or spacing > .015:
        return None
    count = np.clip(np.ceil((upper - lower) / spacing).astype(int) + 1, 2, 256)
    axes = [np.linspace(lower[i], upper[i], count[i]) for i in range(2)]
    x, y = np.meshgrid(*axes)
    nodes = np.column_stack((x.ravel(), y.ravel()))
    world = nodes @ rotation.T
    supported = np.all(world @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9, axis=1)
    dx, dy = [axis[1] - axis[0] for axis in axes]
    rectangle = _largest_supported_rectangle(supported.reshape(x.shape), dx, dy)
    if rectangle is None:
        return None
    left, right, top, bottom = rectangle
    corners = np.array([[axes[0][left], axes[1][top]], [axes[0][right], axes[1][top]],
                        [axes[0][right], axes[1][bottom]], [axes[0][left], axes[1][bottom]]]) @ rotation.T
    return {'polygon_world_xy_m': corners.tolist(), 'method': 'plane_hull_inscribed_rectangle',
            'sample_spacing_m': spacing,
            'grid_step_xy_m': [float(dx), float(dy)], 'minimum_side_m': .03,
            'measured_floor_point_count': len(points)}

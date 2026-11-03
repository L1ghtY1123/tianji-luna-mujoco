"""Conservative sphere coverage of spatially partitioned convex CAD volume."""
from functools import lru_cache

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from .geometry import rigid_matrix


def partition_convex_volume(vertices, *, maximum_cells, minimum_size_m):
    """Bisect convex cells; each child includes every cut edge intersection."""
    points = np.asarray(vertices, dtype=float)
    if (type(maximum_cells) is not int or not 1 <= maximum_cells <= 31
            or not np.isfinite(minimum_size_m) or minimum_size_m <= 0
            or points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 4
            or not np.isfinite(points).all()):
        raise ValueError('cad_partition_requires_finite_volume_and_bounded_budget')
    hull = ConvexHull(points)
    cells = [points[hull.vertices]]
    while len(cells) < maximum_cells:
        widths = np.array([np.ptp(cell, axis=0) for cell in cells])
        index = int(np.argmax(widths.max(axis=1)))
        axis = int(np.argmax(widths[index]))
        if widths[index, axis] <= minimum_size_m:
            break
        cell = cells[index]
        cut = (cell[:, axis].min() + cell[:, axis].max()) / 2
        faces = ConvexHull(cell).simplices
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        first, second = cell[edges[:, 0]], cell[edges[:, 1]]
        crossing = (first[:, axis] - cut) * (second[:, axis] - cut) < 0
        first, second = first[crossing], second[crossing]
        fraction = (cut - first[:, axis]) / (second[:, axis] - first[:, axis])
        intersections = first + fraction[:, None] * (second - first)
        intersections[:, axis] = cut
        left = np.unique(np.vstack([cell[cell[:, axis] <= cut], intersections]), axis=0)
        right = np.unique(np.vstack([cell[cell[:, axis] >= cut], intersections]), axis=0)
        cells[index:index + 1] = [left, right]
    return cells


@lru_cache(maxsize=16)
def _canonical_sphere_partition(vertices_bytes, maximum_spheres, minimum_size_m):
    """Reuse only immutable model geometry; pose and padding are never cached."""
    vertices = np.frombuffer(vertices_bytes, dtype=np.float64).reshape(-1, 3)
    cells = partition_convex_volume(vertices, maximum_cells=maximum_spheres, minimum_size_m=minimum_size_m)
    centers = []
    for cell in cells:
        center, _ = trimesh.nsphere.minimum_nsphere(cell)
        center.setflags(write=False)
        cell.setflags(write=False)
        centers.append(center)
    source_volume = float(ConvexHull(vertices).volume)
    partition_volume = sum(float(ConvexHull(cell).volume) for cell in cells)
    if not np.isclose(source_volume, partition_volume, atol=1e-15, rtol=1e-8):
        raise ValueError('cad_partition_volume_not_preserved')
    return cells, centers, source_volume, partition_volume


def convex_volume_spheres(vertices, *, maximum_spheres, minimum_size_m, padding_m, transform):
    """Enclose every convex cell; containing its vertices contains its whole solid."""
    matrix = rigid_matrix(transform)
    points = np.asarray(vertices, dtype=np.float64)
    if (type(maximum_spheres) is not int or not 1 <= maximum_spheres <= 31
            or not np.isfinite(minimum_size_m) or minimum_size_m <= 0
            or points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 4
            or not np.isfinite(points).all()):
        raise ValueError('cad_partition_requires_finite_volume_and_bounded_budget')
    if not np.isfinite(padding_m) or padding_m < 0:
        raise ValueError('cad_partition_requires_nonnegative_padding')
    cells, centers, source_volume, partition_volume = _canonical_sphere_partition(
        points.tobytes(), maximum_spheres, minimum_size_m)
    spheres = []
    for cell, center in zip(cells, centers, strict=True):
        # Recheck all transformed vertices on every pose, including near-rigid
        # float32 transforms. Only the CAD-local partition/centre fitting is reused.
        center = matrix[:3, :3] @ center + matrix[:3, 3]
        transformed = cell @ matrix[:3, :3].T + matrix[:3, 3]
        radius = float(np.linalg.norm(transformed - center, axis=1).max()) + padding_m
        if not np.isfinite(center).all() or not np.isfinite(radius):
            raise ValueError('cad_partition_requires_finite_enclosing_sphere')
        spheres.append([*center.tolist(), max(.002, radius)])
    return spheres, {'fit_method': 'cad_convex_volume_partition_cover',
        'covers_model_volume': True, 'partition_count': len(cells),
        'source_convex_volume_m3': source_volume, 'partition_volume_m3': partition_volume,
        'partition_frame': 'registered_cad_canonical',
        'component_policy': 'all_components_inside_registered_source_convex_hull'}

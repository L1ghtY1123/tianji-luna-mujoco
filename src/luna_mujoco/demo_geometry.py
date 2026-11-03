"""Measured inventory geometry without hidden-surface completion."""
import numpy as np
from scipy.ndimage import binary_propagation
from scipy.spatial import cKDTree

from .cameras import deproject_view
from .perception import PerceptionError


def supported_surface_points(points):
    """Require four nearby samples within 2 cm, including the sample itself."""
    points = np.asarray(points)
    keep = np.isfinite(points).all(axis=1)
    if not keep.any():
        return keep
    selected = points[keep]
    supported = cKDTree(selected).query_ball_point(selected, r=.02, return_length=True) >= 4
    keep[np.flatnonzero(keep)] = supported
    return keep


def _locally_supported_plane(points, *, normal, offset):
    """Check neighbourhoods before selecting a thin band of plane inliers."""
    points = np.asarray(points)
    local = points[np.abs(points @ normal + offset) <= .03]
    if len(local) < 16:
        return np.empty((0, 3))
    distance, indices = cKDTree(local).query(local, k=16)
    neighbours = local[indices]
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centred, centred) / 16
    values, vectors = np.linalg.eigh(covariance)
    supported = ((distance[:, -1] <= .02) & (values[:, 1] > 1e-6)
                 & (values[:, 0] <= .1 * values[:, 1])
                 & (np.abs(vectors[:, :, 0] @ normal) >= np.cos(np.deg2rad(5)))
                 & (np.abs(local @ normal + offset) <= .006))
    return local[supported]


def fuse_table_plane(seed, point_groups, *, normal, offset):
    """Fuse locally horizontal, connected measured returns within one 2cm edge band."""
    low, high = seed[:, :2].min(axis=0), seed[:, :2].max(axis=0)
    lower, upper = low - .02, high + .02
    groups = {}
    for camera, points in point_groups.items():
        points = np.asarray(points)
        local = points[np.all((points[:, :2] >= lower) & (points[:, :2] <= upper), axis=1)]
        groups[camera] = _locally_supported_plane(local, normal=normal, offset=offset)
    shape = np.ceil((upper - lower) / .02).astype(int) + 1
    occupied, starts = np.zeros(shape, bool), np.zeros(shape, bool)
    seed_cells = np.floor((seed[:, :2] - lower) / .02).astype(int)
    starts[tuple(seed_cells.T)] = True
    occupied |= starts
    cells = {}
    for camera, points in groups.items():
        cells[camera] = np.floor((points[:, :2] - lower) / .02).astype(int)
        occupied[tuple(cells[camera].T)] = True
    connected = binary_propagation(starts, mask=occupied)
    kept = {camera: points[connected[tuple(cells[camera].T)]] for camera, points in groups.items()}
    return np.vstack([seed, *kept.values()]), {
        'source': 'same_plane_connected_multicamera_measured_returns',
        'maximum_footprint_growth_m': .02, 'local_support_radius_m': .02,
        'local_support_neighbours': 16, 'normal_tolerance_deg': 5.,
        'accepted_point_count_by_camera': {camera: len(points) for camera, points in kept.items()}}


def fit_inventory_table(view, bbox, excluded, *, supplemental_points=None):
    """Bounded RANSAC on measured depth inside an initial RGB table rectangle."""
    rows, columns = np.indices(view['depth_m'].shape)
    height, width = rows.shape
    selected = ((columns >= bbox[0] * width) & (columns < bbox[2] * width)
                & (rows >= bbox[1] * height) & (rows < bbox[3] * height) & ~excluded)
    points = deproject_view(view, selected)
    points = points[(points[:, 0] >= 0) & (points[:, 0] <= 1.1)
                    & (np.abs(points[:, 1]) <= 1.) & (points[:, 2] > .1) & (points[:, 2] <= 1.6)]
    if len(points) < 100:
        raise PerceptionError('demo_inventory_table_unobservable')
    rng = np.random.default_rng(0)
    sample = points if len(points) <= 6000 else points[rng.choice(len(points), 6000, replace=False)]
    best_count, best = 0, None
    for _ in range(256):
        triplet = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(triplet[1] - triplet[0], triplet[2] - triplet[0])
        magnitude = np.linalg.norm(normal)
        if magnitude < 1e-8:
            continue
        normal = normal / magnitude * (1 if normal[2] >= 0 else -1)
        if normal[2] < np.cos(np.deg2rad(5)):
            continue
        offset = -float(normal @ triplet[0])
        count = int(np.count_nonzero(np.abs(sample @ normal + offset) <= .006))
        if count > best_count:
            best_count, best = count, (normal, offset)
    if best is None:
        raise PerceptionError('demo_inventory_table_unobservable')
    normal, offset = best
    # Near-height returns can be a vertical container wall. The main camera
    # needs the same measured local surface support as supplemental cameras.
    surface = _locally_supported_plane(points, normal=normal, offset=offset)
    if len(surface) < 100:
        raise PerceptionError('demo_inventory_table_unobservable')
    seed_support = {'source': 'measured_local_plane_neighbourhood',
        'input_point_count': len(points), 'accepted_point_count': len(surface),
        'local_support_radius_m': .02, 'local_support_neighbours': 16,
        'normal_tolerance_deg': 5., 'distance_threshold_m': .006}
    fusion = None
    if callable(supplemental_points):
        supplemental_points = supplemental_points(surface, normal, offset)
    if supplemental_points:
        surface, fusion = fuse_table_plane(surface, supplemental_points, normal=normal, offset=offset)
    low, high = surface.min(axis=0), surface.max(axis=0)
    if np.any(high[:2] - low[:2] < .02):
        raise PerceptionError('demo_inventory_table_footprint_insufficient')
    z = float(np.median(surface[:, 2]))
    center = (low[:2] + high[:2]) / 2
    return {'model': {'pose': [*center.tolist(), z - .15, 1., 0., 0., 0.],
                      'dims': [*(high[:2] - low[:2]).tolist(), .30]},
            'surface_height_m': z, 'normal': normal.tolist(), 'inlier_count': len(surface),
            'source': 'initial_inventory_ransac_table',
            'fit': {'iterations': 256, 'distance_threshold_m': .006, 'maximum_tilt_deg': 5.,
                    'input_point_count': len(points), 'sample_limit': 6000,
                    'seed_surface_support': seed_support, 'multicamera_fusion': fusion},
            'downward_extent_m': .30, 'downward_extent_source': 'bounded_cuboid_completion'}

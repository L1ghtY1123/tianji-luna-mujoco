"""Measured inventory geometry and explicitly labelled user-prior completion."""
import numpy as np
from scipy.ndimage import binary_propagation
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .cad_assets import cad_asset
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


def fuse_table_plane(seed, point_groups, *, normal, offset):
    """Fuse locally horizontal, connected measured returns within one 2cm edge band."""
    low, high = seed[:, :2].min(axis=0), seed[:, :2].max(axis=0)
    lower, upper = low - .02, high + .02
    groups = {}
    for camera, points in point_groups.items():
        points = np.asarray(points)
        local = points[np.all((points[:, :2] >= lower) & (points[:, :2] <= upper), axis=1)
                       & (np.abs(points @ normal + offset) <= .03)]
        if len(local) < 16:
            groups[camera] = np.empty((0, 3))
            continue
        distance, indices = cKDTree(local).query(local, k=16)
        neighbours = local[indices]
        centred = neighbours - neighbours.mean(axis=1, keepdims=True)
        covariance = np.einsum('nki,nkj->nij', centred, centred) / 16
        values, vectors = np.linalg.eigh(covariance)
        supported = ((distance[:, -1] <= .02) & (values[:, 1] > 1e-6)
                     & (values[:, 0] <= .1 * values[:, 1])
                     & (np.abs(vectors[:, :, 0] @ normal) >= np.cos(np.deg2rad(5)))
                     & (np.abs(local @ normal + offset) <= .006))
        groups[camera] = local[supported]
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
    surface = points[np.abs(points @ normal + offset) <= .006]
    if len(surface) < 100:
        raise PerceptionError('demo_inventory_table_unobservable')
    fusion = None
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
                    'input_point_count': len(points), 'sample_limit': 6000, 'multicamera_fusion': fusion},
            'downward_extent_m': .30, 'downward_extent_source': 'bounded_cuboid_completion'}


def complete_source_height(instance, *, support_z_m, support_source, observation_id, height_prior_m=None):
    """Keep original samples; add a bounded collision envelope, never sensor depth."""
    points = np.asarray(instance['points_world_m'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise PerceptionError('demo_height_measured_points_required')
    priors = {cad_asset(asset)['size_m'][2]: asset for asset in ('pink_wrapped', 'black_core')}
    priors[cad_asset('black_core')['size_m'][0]] = 'black_core:flat_local_x_vertical'
    support = float(support_z_m)
    height = float(cad_asset('pink_wrapped')['size_m'][2] if height_prior_m is None else height_prior_m)
    tolerance = .015
    # Old explicit dimensions remain readable for archived callers. Active demo
    # perception always supplies the catalog height and records the asset ID.
    if height not in priors and height not in (.06, .10):
        raise PerceptionError('demo_height_prior_unauthorized')
    prior_source = 'user_cad_asset:' + priors[height] if height in priors else f'user_specified_{round(height * 1000)}mm'
    if (not np.isfinite(support) or points[:, 2].min() < support - tolerance
            or points[:, 2].max() > support + height + tolerance
            or points[:, 2].max() < support + height - tolerance):
        raise PerceptionError('demo_height_prior_inconsistent')
    low, high = points.min(axis=0), points.max(axis=0)
    if high[2] - low[2] >= height - tolerance and low[2] <= support + tolerance:
        instance.pop('completed_geometry_points_world_m', None)
        instance['geometry_completion'] = {
            'status': 'measured_height_covered', 'source': 'measured_surface_with_user_height_consistency',
            'height_prior_m': height, 'height_prior_source': prior_source,
            'support_z_m': support, 'support_source': support_source, 'observation_id': observation_id,
            'measurement_tolerance_m': tolerance, 'measured_point_count': len(points),
            'measured_bounds_world_m': [low.tolist(), high.tolist()],
            'synthetic_point_count': 0, 'synthetic_points_are_measured': False}
        return
    bottom, top = min(support, low[2]), max(support + height, high[2])
    try:
        hull = ConvexHull(points[:, :2])
    except QhullError as error:
        raise PerceptionError('demo_height_measured_footprint_required') from error
    boundary = points[hull.vertices, :2]
    # At most seven measured-hull rings, all at 2 cm or smaller spacing. This
    # is a separately labelled attachment envelope, not added observed voxels.
    heights = np.linspace(bottom, top, int(np.ceil((top - bottom) / .02)) + 1)
    envelope = np.array([[x, y, z] for z in heights for x, y in boundary], dtype=np.float32)
    instance['completed_geometry_points_world_m'] = np.vstack((points, envelope)).astype(np.float32)
    instance['geometry_completion'] = {
        'source': 'measured_surface_and_user_height_prior', 'height_prior_m': height,
        'height_prior_source': prior_source, 'support_z_m': support,
        'support_source': support_source, 'observation_id': observation_id,
        'measurement_tolerance_m': tolerance, 'measured_point_count': len(points),
        'measured_bounds_world_m': [low.tolist(), high.tolist()],
        'completed_bounds_world_m': [[*low[:2].tolist(), bottom], [*high[:2].tolist(), top]],
        'synthetic_point_count': len(envelope), 'synthetic_points_are_measured': False,
        'xy_policy': 'measured_convex_hull_extrusion', 'map_voxel_size_m': .02}

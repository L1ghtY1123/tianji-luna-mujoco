"""Calibrated partial-surface comparisons. Metrics never assign an identity."""
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .association_depth import head_wrist_pair, spatial_relation
from .cameras import deproject_view
from .observation_overlap import mask_pair_metrics

DEPTH_TOLERANCE_M = .015
CROSS_VIEW_TOLERANCE_M = .025  # Head/wrist correspondence only, in calibrated world metres.
TOLERANCE_SOURCE = 'application_initial_depth_allowance_not_certified_sensor_error'


SURFACE_MATCH = {'minimum_common_pixels': 30, 'minimum_projected_iou': .5, 'minimum_surface_precision': .55,
                 'minimum_partial_surface_agreement': .65, 'minimum_voxel_overlap': .55, 'minimum_bounds_iou': .5, 'minimum_depth_consistency': .55, 'minimum_target_coverage': .5}



def strong_surface_match(evidence):
    """Strict bidirectional measured-surface support, never CAD or old-position similarity."""
    directions = [evidence.get(key, {}) for key in ('left_to_right', 'right_to_left')]
    if not all(d.get('status') == 'available'
               and d['depth_supported_pixels'] >= SURFACE_MATCH['minimum_common_pixels']
               and d['projected_surface_in_target_fraction'] >= SURFACE_MATCH['minimum_surface_precision']
               and d.get('visible_depth_consistent_fraction', d['depth_consistent_fraction']) >= SURFACE_MATCH['minimum_depth_consistency'] for d in directions):
        return False
    covered = [d['projected_mask_iou'] >= SURFACE_MATCH['minimum_projected_iou']
               and d['target_surface_coverage'] >= SURFACE_MATCH['minimum_target_coverage'] for d in directions]
    # A close wrist view and a distant head view sample the surface at
    # different densities. Require independent 3D overlap for asymmetric coverage.
    # A sparse head projection and a dense wrist mask can have little pixel IoU even
    # on the same physical surface. Require purity in both images and independent
    # 3-D agreement before accepting this partial overlap.
    median_distances = list((evidence.get('nearest_surface_distance_m') or {}).values())
    partial = (len(median_distances) == 2
               and all(d['projected_surface_in_target_fraction'] >= SURFACE_MATCH['minimum_partial_surface_agreement']
                       and d.get('visible_depth_consistent_fraction', 0.) >= SURFACE_MATCH['minimum_partial_surface_agreement']
                       and d['depth_supported_pixels'] * d['projected_surface_in_target_fraction']
                           >= SURFACE_MATCH['minimum_common_pixels'] for d in directions)
               and (max(median_distances) <= CROSS_VIEW_TOLERANCE_M
                    or ((evidence.get('visible_bounds_iou') or 0.) >= SURFACE_MATCH['minimum_bounds_iou']
                        and min(median_distances) <= CROSS_VIEW_TOLERANCE_M)))
    return (partial or all(covered)
            or any(covered) and (evidence.get('visible_bounds_iou') or 0) >= SURFACE_MATCH['minimum_bounds_iou']
            or evidence.get('shared_voxel_fraction_of_smaller', 0) >= SURFACE_MATCH['minimum_voxel_overlap'])


def table_contact(table_points, part_points):
    """Return three-valued evidence for current horizontal support in metres."""
    if len(part_points) < 30 or len(table_points) < 30:
        return 'unknown', {}
    foot = np.quantile(part_points[:, 2], .05)
    low, high = np.quantile(part_points[:, :2], [.05, .95], axis=0)
    nearby = table_points[np.all((table_points[:, :2] >= low - .03) & (table_points[:, :2] <= high + .03), axis=1)]
    evidence = {'object_bottom_m': float(foot), 'local_table_points': len(nearby)}
    if np.quantile(table_points[:, 2], .99) < foot - .05:
        return 'below', evidence
    if len(nearby) < 30:
        return 'unknown', evidence
    height = np.median(nearby[:, 2])
    planar = np.mean(np.abs(nearby[:, 2] - height) <= .015)
    evidence.update(local_table_height_m=float(height), horizontal_inlier_fraction=float(planar))
    if planar < .75 or abs(height - foot) > .025:
        return 'unknown', evidence
    # Table points may surround the occluded footprint. Require most of the
    # measured object footprint inside the hull of local horizontal returns.
    plane = nearby[np.abs(nearby[:, 2] - height) <= .015, :2]
    try:
        hull = ConvexHull(plane[::max(1, len(plane) // 2048)])
    except QhullError:
        return 'unknown', evidence
    xy = part_points[::max(1, len(part_points) // 2048), :2]
    inside = np.all(xy @ hull.equations[:, :2].T + hull.equations[:, 2] <= .002, axis=1)
    evidence['footprint_coverage'] = float(inside.mean())
    return ('supports' if inside.mean() >= .75 else 'unknown'), evidence


def bounds_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    overlap = np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]))
    intersection = float(np.prod(overlap))
    union = float(np.prod(a[1] - a[0]) + np.prod(b[1] - b[0]) - intersection)
    return intersection / union if union > 0 else None


def projected_depth_masks(points, view, tolerance_m=DEPTH_TOLERANCE_M):
    depth = np.asarray(view['depth_m'])
    height, width = depth.shape
    transform = np.asarray(view['T_world_camera'])
    local = (np.asarray(points) - transform[:3, 3]) @ transform[:3, :3]
    local = local[np.isfinite(local).all(axis=1) & (local[:, 2] > 0)]
    k = view['intrinsics']
    columns = np.rint(local[:, 0] * k['fx'] / local[:, 2] + k['cx']).astype(int)
    rows = np.rint(local[:, 1] * k['fy'] / local[:, 2] + k['cy']).astype(int)
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    raster = np.full(height * width, np.inf)
    np.minimum.at(raster, rows[inside] * width + columns[inside], local[inside, 2])
    raster = raster.reshape(depth.shape)
    observed = view['valid_depth'] & np.isfinite(depth) & (depth > 0)
    if 'robot_mask' in view:
        observed &= ~np.asarray(view['robot_mask'], dtype=bool)
    projected = np.isfinite(raster) & observed
    visible = projected & (raster <= depth + tolerance_m)
    shared = visible & (np.abs(raster - depth) <= tolerance_m)
    return projected, visible, shared, observed


def reproject_surface(points, view, target_mask, *, tolerance_m=DEPTH_TOLERANCE_M):
    """Z-buffer world samples, then remove occluded and depth-inconsistent pixels."""
    projected, visible, shared, observed = projected_depth_masks(points, view, tolerance_m)
    evidence = {'status': 'available' if shared.any() else 'unavailable',
        'reason': None if shared.any() else 'no_common_depth_supported_surface',
        'projected_pixels': int(projected.sum()), 'occluded_pixels': int((projected & ~visible).sum()),
        'depth_supported_pixels': int(shared.sum()), 'tolerance_m': tolerance_m, 'tolerance_source': TOLERANCE_SOURCE,
        'comparison': 'world_points_projected_with_current_calibration_and_target_z_buffer',
        'identity_proven': False}
    if not shared.any():
        return {**evidence, 'projected_mask_iou': None, 'projected_surface_in_target_fraction': None}
    target = np.asarray(target_mask, dtype=bool) & observed
    intersection = int((shared & target).sum())
    # Sparse surface IoU is soft evidence only; unobserved rear faces are not filled.
    return {**evidence, 'projected_mask_iou': intersection / int((shared | target).sum()),
            'projected_surface_in_target_fraction': intersection / int(shared.sum()),
            'depth_consistent_fraction': int(shared.sum()) / int(projected.sum()),
            'visible_depth_consistent_fraction': int(shared.sum()) / int(visible.sum()),
            'target_surface_coverage': intersection / int(target.sum()) if target.any() else 0.,
            'scope': 'partial_surface_sparse_projection_not_complete_silhouette'}


def pair_geometry(left, right):
    if left['camera_id'] != right['camera_id'] and not head_wrist_pair(left['camera_id'], right['camera_id']):
        raise ValueError('cross_camera_link_requires_head_and_wrist')
    if (left['observation_id'], left['generation']) != (right['observation_id'], right['generation']):
        raise ValueError('same_capture_pair_required')
    evidence = {'spatial': spatial_relation(left['depth_evidence'], right['depth_evidence']),
                'frame': 'Link_Base', 'identity_proven': False}
    if left['camera_id'] == right['camera_id']:
        evidence['same_image_overlap'] = mask_pair_metrics(left['silhouette_mask'], right['silhouette_mask'])
    a, b = np.asarray(left['points']), np.asarray(right['points'])
    if not len(a) or not len(b):
        return {**evidence, 'surface_comparison': 'unavailable'}
    # Deterministic sampling bounds CPU cost without changing candidate eligibility.
    a, b = a[::max(1, len(a) // 2048)], b[::max(1, len(b) // 2048)]
    # The overlap coefficient uses the smaller visible surface, so a wrist
    # camera can see a subset of a tabletop measured by the head camera.
    def surface_cells(points):
        cells = np.floor(points / .01).astype(np.int64)
        return set(map(tuple, np.unique(cells, axis=0)))
    cells_a, cells_b = surface_cells(left['points']), surface_cells(right['points'])
    evidence['shared_voxel_fraction_of_smaller'] = len(cells_a & cells_b) / min(len(cells_a), len(cells_b))
    evidence['surface_voxel_size_m'] = .01
    tolerance = DEPTH_TOLERANCE_M if left['camera_id'] == right['camera_id'] else CROSS_VIEW_TOLERANCE_M
    evidence.update(visible_bounds_iou=bounds_iou([a.min(0), a.max(0)], [b.min(0), b.max(0)]),
        nearest_surface_distance_m={'a_to_b_median': float(np.median(cKDTree(b).query(a)[0])),
                                       'b_to_a_median': float(np.median(cKDTree(a).query(b)[0]))},
        left_to_right=reproject_surface(left['points'], right['view'], right['silhouette_mask'], tolerance_m=tolerance),
        right_to_left=reproject_surface(right['points'], left['view'], left['silhouette_mask'], tolerance_m=tolerance))
    poses = [row.get('pose_hypothesis', {}) for row in (left, right)]
    evidence['pose_hypotheses'] = poses
    for name, source, target in (('cad_left_to_right', left, right), ('cad_right_to_left', right, left)):
        if '_pose_world_points' in source:
            evidence[name] = reproject_surface(source['_pose_world_points'], target['view'], target['silhouette_mask'], tolerance_m=tolerance)
    if all(p.get('status') == 'estimated' for p in poses) and poses[0]['asset_id'] == poses[1]['asset_id']:
        evidence['cad_world_bounds_iou'] = bounds_iou(*(p['bounds_world_m'] for p in poses))
        evidence['cad_comparison_scope'] = 'same_model_current_pose_hypotheses_symmetries_not_excluded'
    return evidence


def temporal_overlap_evidence(members, reference):
    """Dated depth projections are positive evidence, never a motion veto."""
    evidence = {}
    for identifier, dated in reference.get('identity_references', {}).items():
        rows = []
        for member in members:
            for camera, view in dated.get('views', {}).items():
                if camera != member['camera_id'] and not head_wrist_pair(camera, member['camera_id']):
                    continue
                if any(key not in view for key in ('mask', 'depth_m', 'valid_depth', 'intrinsics', 'T_world_camera')):
                    continue
                silhouette = np.asarray(view['mask'], dtype=bool)
                mask = silhouette & view['valid_depth']
                points = deproject_view(view, mask)
                points = points[::max(1, len(points) // 4096)]
                projection = reproject_surface(points, member['view'], member['silhouette_mask'])
                row = {'reference_camera': camera, 'current_camera': member['camera_id'],
                    'projection': projection, 'same_view_mask_iou': None}
                # Pixel IoU is comparable only when the projection is EXACTLY
                # unchanged. Otherwise the calibrated depth projection applies.
                if (camera == member['camera_id'] and mask.shape == member['silhouette_mask'].shape
                        and view['intrinsics'] == member['view']['intrinsics']
                        and np.array_equal(view['T_world_camera'], member['view']['T_world_camera'])):
                    row['same_view_mask_iou'] = mask_pair_metrics(silhouette, member['silhouette_mask'])['mask_iou']
                rows.append(row)
        evidence[identifier] = {'reference_observation_id': dated.get('observation_id'), 'comparisons': rows,
            'interpretation': 'Historical geometry projected into the current view; low overlap cannot reject a moved object.'}
    return evidence


def measured_mask_color(row):
    """Name a dominant RGB color only when enough valid subject pixels agree."""
    return dominant_rgb_color(row['view']['rgb'][row['mask']])


def dominant_rgb_color(rgb):
    if len(rgb) < 30:
        return 'unknown'
    rgb = rgb[::max(1, len(rgb) // 2048)].astype(float) / 255.
    r, g, b = rgb.T
    value, low = rgb.max(axis=1), rgb.min(axis=1)
    delta = value - low
    neutral = delta < np.maximum(.04, .19 * value)
    chromatic = (~neutral) & (value > .12)
    # Leave color boundaries and brown/purple shades unresolved.
    tests = {'black': neutral & (value < .19), 'white': neutral & (value > .76),
             'gray': neutral & (value >= .23) & (value <= .72),
             'blue': chromatic & (b > 1.22*r) & (b > 1.18*g),
             'pink': chromatic & (r > 1.45*g) & (b > 1.35*g) & (b > .28*r) & (b < .88*r),
             'red': chromatic & (r > 1.55*g) & (r > 3.7*b),
             'green': chromatic & (g > 1.28*r) & (g > 1.25*b)}
    return next((name for name, test in tests.items() if float(test.mean()) >= .75), 'unknown')


def measured_table_top(points, *, tolerance_m=DEPTH_TOLERANCE_M):
    xyz = np.asarray(points)
    if len(xyz) < 80:
        return None
    height = float(np.median(xyz[:, 2]))
    plane = xyz[np.abs(xyz[:, 2] - height) <= tolerance_m]
    if len(plane) < .7 * len(xyz):
        return None
    bounds = np.quantile(plane[:, :2], [.02, .98], axis=0)
    if np.min(bounds[1] - bounds[0]) < .10:
        return None
    return height, bounds

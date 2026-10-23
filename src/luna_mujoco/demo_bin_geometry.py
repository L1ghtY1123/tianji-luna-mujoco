"""Hollow bin completion from clean registered planes and explicit outer dimensions.

No scene poses, simulator geometry, RGB-derived depth, or contact locations enter
the fit. Call only with the mapper's post-robot-removal inventory points.
"""
import hashlib
from itertools import combinations

import numpy as np

SOURCE = 'measured_planes_and_user_bin_dimensions'
OUTER_DIMS_M = (.30, .40, .20)
PLANE_TOLERANCE_M = .002
ANGLE_TOLERANCE_RAD = np.deg2rad(3.)


def validate_bin_priors(priors, instance_ids, observation_id, generation):
    if not isinstance(priors, dict) or set(priors)-set(instance_ids):
        raise ValueError('demo_bin_shell_prior_requires_current_inventory')
    for identifier, prior in priors.items():
        if (not isinstance(prior, dict) or prior.get('kind') != 'bin' or prior.get('color') != 'blue'
                or prior.get('instance_id') != identifier or prior.get('observation_id') != observation_id
                or type(prior.get('generation')) is not int or prior['generation'] != generation):
            raise ValueError('demo_bin_shell_prior_invalid_or_stale')
        if prior.get('source') == 'user_supplied_cad_shape':
            from .cad_assets import cad_asset
            asset = cad_asset('blue_box')
            valid = (prior.get('asset_id') == 'blue_box' and prior.get('cad_sha256') == asset['source']['sha256']
                     and prior.get('outer_dimensions_m') == asset['size_m'])
        else:
            valid = (prior.get('source') == 'user_specified_bin_outer_dimensions'
                     and prior.get('outer_dimensions_m') == list(OUTER_DIMS_M))
        if not valid:
            raise ValueError('demo_bin_shell_prior_invalid_or_stale')


def validate_published_bin_components(components, identifier, observation_id, generation):
    """Validate the exact declared shape, not an arbitrary count of derived layers."""
    from .cad_assets import cad_asset
    from .demo_bin_cad import SOURCE as CAD_SOURCE

    if not isinstance(components, dict) or not components:
        raise ValueError('demo_bin_shell_bounds_incomplete_or_stale')
    prefix = f'semantic_instance::{identifier}::bin_shell::'
    sources = {layer.get('geometry_source') for layer in components.values() if isinstance(layer, dict)}
    cad = sources == {CAD_SOURCE}
    asset = cad_asset('blue_box') if cad else None
    parts = ({f'cad_part_{i:02d}': record['sha256'] for i, record in enumerate(asset['collision'])}
             if cad else dict.fromkeys(('x_min', 'x_max', 'y_min', 'y_max', 'floor')))
    if cad and set(components) == {prefix + 'cad_surface'}:
        parts = {'cad_surface': asset['visual']['sha256']}
    if sources != {CAD_SOURCE if cad else SOURCE} or set(components) != {prefix+part for part in parts}:
        raise ValueError('demo_bin_shell_bounds_incomplete_or_stale')
    for name, layer in components.items():
        if not isinstance(layer, dict):
            raise TypeError('demo_bin_shell_bounds_invalid_disabled_or_stale')
        low, high = np.asarray(layer.get('minimum_world_m'), float), np.asarray(layer.get('maximum_world_m'), float)
        if (layer.get('kind') != ('mesh' if cad else 'cuboid')
                or layer.get('source') != ('published_mesh_bounds' if cad else 'published_cuboid_bounds')
                or layer.get('instance_id') != identifier or layer.get('enabled') is not True
                or layer.get('synthetic_geometry_is_measured') is not False
                or layer.get('observation_id') != observation_id or layer.get('generation') != generation
                or low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
                or np.any(high <= low)
                or (cad and (layer.get('asset_id') != 'blue_box'
                    or layer.get('cad_sha256') != asset['source']['sha256']
                    or layer.get('part_sha256') != parts[name[len(prefix):]]))):
            raise ValueError('demo_bin_shell_bounds_invalid_disabled_or_stale')
    return components


def published_bin_shell_layers(scene, identifier):
    """Exact-ID shell components, including every face or rejecting stale/partial state."""
    header = scene.get('collision_layer_bounds', {})
    layers = header.get('layers', {})
    prefix = f'semantic_instance::{identifier}::bin_shell::'
    actual = {name for name in layers if name.startswith(prefix)}
    completion = scene.get('mapping_diagnostics', {}).get('bin_shell_completions', {}).get(identifier)
    if completion is None and not actual:
        return {}  # Legacy recorded maps have no declared completion.
    from .demo_bin_cad import SOURCE as CAD_SOURCE
    if (not isinstance(completion, dict) or completion.get('source') not in {SOURCE, CAD_SOURCE}
            or completion.get('instance_id') != identifier
            or any(header.get(key) != scene.get(key) for key in ('scene_token', 'observation_id', 'generation'))
            or any(completion.get(key) != scene.get(key) for key in ('observation_id', 'generation'))):
        raise ValueError('demo_bin_shell_bounds_incomplete_or_stale')
    if actual & set(scene.get('disabled_obstacle_names', ())):
        raise ValueError('demo_bin_shell_bounds_invalid_disabled_or_stale')
    result = {name: layers[name] for name in sorted(actual)}
    if any(layer.get('geometry_source') != completion['source'] for layer in result.values()):
        raise ValueError('demo_bin_shell_bounds_incomplete_or_stale')
    return validate_published_bin_components(result, identifier, scene['observation_id'], scene['generation'])


def _planes(points):
    """Eight deterministic RANSAC planes, 256 hypotheses each, at most 8000 samples."""
    rng = np.random.default_rng(0)
    remaining = points if len(points) <= 8000 else points[rng.choice(len(points), 8000, replace=False)]
    planes = []
    for _ in range(8):
        if len(remaining) < 80:
            break
        triples = remaining[rng.integers(len(remaining), size=(256, 3))]
        normals = np.cross(triples[:, 1]-triples[:, 0], triples[:, 2]-triples[:, 0])
        length = np.linalg.norm(normals, axis=1)
        valid = length > 1e-8
        normals = normals[valid]/length[valid, None]
        if not len(normals):
            break
        offsets = np.einsum('ij,ij->i', normals, triples[valid, 0])
        distances = np.abs(remaining @ normals.T-offsets)
        counts = (distances <= PLANE_TOLERANCE_M).sum(axis=0)
        best = int(counts.argmax())
        if counts[best] < 80:
            break
        inliers = remaining[distances[:, best] <= PLANE_TOLERANCE_M]
        center = inliers.mean(axis=0)
        _, values, vectors = np.linalg.svd(inliers-center, full_matrices=False)
        normal = vectors[-1]
        if values[1]/np.sqrt(len(inliers)) < .008:
            break  # A line/rim edge alone cannot establish a wall plane.
        offset = float(normal @ center)
        mask = np.abs(points @ normal-offset) <= PLANE_TOLERANCE_M
        planes.append({'normal': normal, 'offset': offset, 'points': points[mask],
                       'point_count': int(mask.sum())})
        remaining = remaining[np.abs(remaining @ normal-offset) > PLANE_TOLERANCE_M]
    return planes


def _wall_pairs(planes, outer_width):
    result = []
    for first, second in combinations(planes, 2):
        a, b = first['normal'], second['normal']
        if abs(a @ b) < np.cos(ANGLE_TOLERANCE_RAD):
            continue
        axis = a[:2]/np.linalg.norm(a[:2])
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        offsets = sorted(float(np.median(plane['points'][:, :2] @ axis)) for plane in (first, second))
        thickness = (outer_width - (offsets[1]-offsets[0]))/2
        if not .003 <= thickness <= .025:
            continue
        if any(np.ptp(plane['points'][:, 2]) < .03 for plane in (first, second)):
            continue
        result.append((axis, offsets, thickness, (first, second)))
    return result


def _model(points, planes):
    horizontal = [p for p in planes if abs(p['normal'][2]) >= np.cos(ANGLE_TOLERANCE_RAD)]
    vertical = [p for p in planes if abs(p['normal'][2]) <= np.sin(ANGLE_TOLERANCE_RAD)]
    models = []
    for floor, rim in combinations(sorted(horizontal, key=lambda p: np.median(p['points'][:, 2])), 2):
        floor_z, rim_z = [float(np.median(p['points'][:, 2])) for p in (floor, rim)]
        base_thickness = OUTER_DIMS_M[2]-(rim_z-floor_z)
        if not .003 <= base_thickness <= .025:
            continue
        for short in _wall_pairs(vertical, OUTER_DIMS_M[0]):
            for long in _wall_pairs(vertical, OUTER_DIMS_M[1]):
                if abs(short[0] @ long[0]) > np.sin(ANGLE_TOLERANCE_RAD):
                    continue
                if np.ptp([short[2], long[2], base_thickness]) > .003:
                    continue
                # Enforce an orthogonal, upright box after measured normal agreement.
                u = short[0]
                v = np.array([-u[1], u[0]])
                u_bounds = sorted(float(np.median(p['points'][:, :2] @ u)) for p in short[3])
                v_bounds = sorted(float(np.median(p['points'][:, :2] @ v)) for p in long[3])
                center_xy = u*np.mean(u_bounds)+v*np.mean(v_bounds)
                local = np.column_stack(((points[:, :2]-center_xy) @ u,
                                         (points[:, :2]-center_xy) @ v, points[:, 2]))
                inner = np.array([np.diff(u_bounds)[0], np.diff(v_bounds)[0]])/2
                thickness = np.array(OUTER_DIMS_M[:2])/2-inner
                residual = np.minimum.reduce((abs(local[:, 0]-inner[0]), abs(local[:, 0]+inner[0]),
                    abs(local[:, 1]-inner[1]), abs(local[:, 1]+inner[1]),
                    abs(local[:, 2]-floor_z), abs(local[:, 2]-rim_z)))
                inside = ((abs(local[:, 0]) <= .15+.003) & (abs(local[:, 1]) <= .20+.003)
                          & (local[:, 2] >= rim_z-.20-.003) & (local[:, 2] <= rim_z+.003))
                agreement = float(np.mean(inside & (residual <= .003)))
                if agreement < .85:
                    continue
                models.append({'u': u, 'center_xy': center_xy, 'floor_z': floor_z, 'rim_z': rim_z,
                    'thickness': thickness, 'base_thickness': base_thickness, 'agreement': agreement,
                    'plane_counts': [p['point_count'] for p in (floor, rim, *short[3], *long[3])]})
    if len(models) != 1:
        raise ValueError('demo_bin_shell_plane_evidence_ambiguous_or_inconsistent')
    return models[0]


def fit_bin_shell(points, *, instance_id, observation_id, generation, prior=None):
    """Return five separately labeled cuboids; never modify or replace measured points."""
    original = np.asarray(points)
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 400
            or not np.isfinite(points).all() or not instance_id or not observation_id
            or type(generation) is not int or generation < 1):
        raise ValueError('demo_bin_shell_registered_points_required')
    if prior is not None:
        validate_bin_priors({instance_id: prior}, {instance_id}, observation_id, generation)
        if prior['source'] == 'user_supplied_cad_shape':
            from .demo_bin_cad import fit_cad_bin
            return fit_cad_bin(points, original=original, planes=_planes(points), instance_id=instance_id,
                               observation_id=observation_id, generation=generation)
    model = _model(points, _planes(points))
    tx, ty = model['thickness']
    tz, top = model['base_thickness'], model['rim_z']
    yaw = float(np.arctan2(model['u'][1], model['u'][0]))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                         [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    center = np.array([*model['center_xy'], top-.10])
    shapes = {'x_min': ([-.15+tx/2, 0., 0.], [tx, .40, .20]),
              'x_max': ([.15-tx/2, 0., 0.], [tx, .40, .20]),
              'y_min': ([0., -.20+ty/2, 0.], [.30, ty, .20]),
              'y_max': ([0., .20-ty/2, 0.], [.30, ty, .20]),
              'floor': ([0., 0., -.10+tz/2], [.30, .40, tz])}
    cuboids = {}
    for face, (position, dims) in shapes.items():
        cuboids[face] = {'pose': [*(center+rotation @ position).tolist(), np.cos(yaw/2), 0., 0., np.sin(yaw/2)],
                         'dims': list(dims)}
    half = np.abs(rotation) @ (np.array(OUTER_DIMS_M)/2)
    return {'source': SOURCE, 'synthetic_geometry_is_measured': False,
        'instance_id': instance_id, 'observation_id': observation_id, 'generation': generation,
        'outer_dimensions_prior_m': list(OUTER_DIMS_M), 'prior_source': 'user_specified_bin_outer_dimensions',
        'shape_assumption': 'upright_open_rectangular_bin_equal_wall_and_base_thickness',
        'measurement_source': 'registered_inventory_points_after_robot_and_surface_filter',
        'measured_point_count': len(points), 'measured_points_sha256': hashlib.sha256(original.tobytes()).hexdigest(),
        'floor_height_m': model['floor_z'], 'rim_height_m': top,
        'wall_thickness_xy_m': model['thickness'].tolist(), 'base_thickness_m': tz,
        'plane_inlier_counts': model['plane_counts'], 'plane_tolerance_m': PLANE_TOLERANCE_M,
        'angle_tolerance_deg': 3., 'minimum_model_agreement_fraction': .85,
        'model_agreement_fraction': model['agreement'], 'cuboids': cuboids,
        'completed_bounds_world_m': [(center-half).tolist(), (center+half).tolist()]}

"""Measured placement geometry; semantic choices are supplied by Decisions."""
from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from ..core.placement import PlacementIntent
from ..geometry import rigid_matrix
from ..placement import observed_object

SURFACE_SAMPLE_SPACING_M = .01
SURFACE_SUPPORT_DISTANCE_M = .015


def placement_frames(capture):
    """Freeze measured horizontal directions; Decisions still selects the language frame."""
    source = {'source_observation_id': capture.get('observation_id'), 'generation': capture.get('generation')}
    frames = {'robot_base': {**source, 'axes_world_xy': {'left': [0., 1.], 'front': [1., 0.]}}}
    for camera, view in capture.get('views', {}).items():
        if 'T_world_camera' not in view:
            continue
        transform = rigid_matrix(view['T_world_camera'])
        axes = {}
        for name, column in (('left', 0), ('front', 2)):
            axis = -transform[:2, column]
            length = float(np.linalg.norm(axis))
            if length >= .1:
                axes[name] = (axis / length).tolist()
        frames['camera_' + camera] = {**source, 'axes_world_xy': axes,
            'T_world_camera': transform.tolist(), 'calibration_id': view.get('calibration_id')}
    return frames


def _points(entity, reason):
    points = np.asarray(entity.get('points_world_m', []), float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError(reason + ':' + entity['id'])
    return points


def measured_object_region(observation, identifier):
    """An observed horizontal top, with point coverage; never modify its collision entity."""
    entity = observed_object(observation, identifier)
    points = _points(entity, 'placement_support_geometry_missing')
    # The highest well-populated horizontal band supplies a support proposal.
    # Its bounding rectangle is only a search domain; the interior coverage test
    # below prevents holes and unseen areas from becoming synthetic support.
    bands = np.floor(points[:, 2] / .008).astype(int)
    counts = {key: int(np.count_nonzero(bands == key)) for key in np.unique(bands)}
    populated = [key for key, count in counts.items() if count >= 100]
    if not populated:
        raise ValueError('placement_horizontal_surface_unobserved:' + identifier)
    key = max(populated)
    height = float(np.median(points[bands == key, 2]))
    surface = points[np.abs(points[:, 2] - height) <= .004]
    centered = surface - surface.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    normal = vectors[-1]
    if abs(normal[2]) < np.cos(np.deg2rad(5)):
        raise ValueError('placement_horizontal_surface_unobserved:' + identifier)
    lower, upper = surface[:, :2].min(0), surface[:, :2].max(0)
    if np.any(upper - lower < .02):
        raise ValueError('placement_surface_too_small:' + identifier)
    polygon = [[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]]
    position = [*((lower + upper) / 2).tolist(), height]
    region_id = 'surface-region:' + identifier
    return {'id': region_id, 'kind': 'marker', 'parent_entity_id': identifier,
        'source': 'measured_object_surface', 'visual_membership': 'excluded',
        'roles': ['target_region'], 'position': position, 'color': 'unmarked',
        'placement_reference': {'source': 'measured_object_surface',
            'support_entity_id': identifier, 'observation_id': observation['observation_id'],
            'generation': observation['generation'], 'floor_z_m': height, 'position_world_m': position,
            'region_polygon_world_xy_m': polygon,
            'measured_support_xy_m': surface[:, :2].tolist(),
            'support_sample_distance_m': SURFACE_SUPPORT_DISTANCE_M,
            'support_scope': 'observed_horizontal_surface_only_not_mass_or_stability_certificate'}}


def _direction(observation, frame, direction, frame_axes=None):
    if frame_axes is not None:
        vector = np.asarray(frame_axes[direction], float)
    elif frame == 'robot_base':
        vector = np.array([0., 1.]) if direction == 'left' else np.array([1., 0.])
    else:
        camera = frame.removeprefix('camera_') if frame else None
        view = observation.get('views', {}).get(camera, {})
        if 'T_world_camera' not in view:
            raise ValueError('placement_frame_calibration_missing')
        rotation = rigid_matrix(view['T_world_camera'])[:3, :3]
        vector = -rotation[:2, 0 if direction == 'left' else 2]
    length = float(np.linalg.norm(vector))
    if length < .1:
        raise ValueError('placement_frame_direction_ambiguous')
    return vector / length


def perceive_placement_zone(observation, destination_id, intent, *, frame_axes=None):
    intent = PlacementIntent.model_validate(intent).model_dump(mode='json')
    destination = observed_object(observation, destination_id)
    if destination.get('placement_geometry_issue'):
        raise ValueError(destination['placement_geometry_issue']['reason'] + ':' + destination_id)
    reference = destination.get('placement_reference')
    if not reference:
        raise ValueError('placement_support_geometry_missing:' + destination_id)
    if (reference.get('observation_id') != observation['observation_id']
            or reference.get('generation', observation['generation']) != observation['generation']):
        raise ValueError('placement_support_geometry_stale:' + destination_id)
    if (intent['relation'] == 'inside') != (destination.get('kind') == 'bin'):
        raise ValueError('placement_relation_support_mismatch')
    polygon = np.asarray(reference.get('region_polygon_world_xy_m'), float)
    floor = float(reference.get('floor_z_m', float('nan')))
    if (polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3
            or not np.isfinite(polygon).all() or not np.isfinite(floor)):
        raise ValueError('placement_support_geometry_missing:' + destination_id)
    # Never downgrade a location requirement to a preference with off-center fallbacks.
    if intent['zone'] == 'center' and (not reference.get('precise_xy') or not np.allclose(
            reference['position_world_m'][:2], polygon.mean(0), atol=1e-9, rtol=0)):
        raise ValueError('placement_center_extent_required')
    if frame_axes is None:
        directions = set()
        if intent['relative'] in {'left_of', 'right_of'} or intent['zone'] in {'left_half', 'right_half'}:
            directions.add('left')
        if intent['relative'] in {'in_front_of', 'behind'} or intent['zone'] in {'front_half', 'back_half'}:
            directions.add('front')
        frame_axes = {direction: _direction(observation, intent['frame'], direction).tolist()
                      for direction in sorted(directions)}
    halfplanes, reference_geometry = spatial_constraints(observation, intent, polygon, frame_axes=frame_axes)
    result = {'destination_id': destination_id, 'support_entity_id': reference.get('support_entity_id', destination_id),
        'observation_id': observation['observation_id'], 'generation': observation['generation'],
        'intent': intent, 'floor_z_m': floor, 'polygon_world_xy_m': polygon.tolist(),
        'frame_axes_world_xy': deepcopy(frame_axes),
        'halfplanes_world_xy': np.asarray(halfplanes).reshape(-1, 3).tolist(),
        'reference_geometry': reference_geometry, 'collision_checked': False, 'unknown_space_is_free': False,
        'geometry_source': reference.get('source'),
        'source_observation_id': reference.get('source_observation_id', reference['observation_id'])}
    for name in ('measured_support_xy_m', 'support_sample_distance_m'):
        if name in reference:
            result[name] = deepcopy(reference[name])
    return result


def placement_constraint_mask(positions, footprint, zone, *, clearance_m=0.):
    """Translate the actual held footprint, including off-center grasp offset."""
    positions, footprint = np.asarray(positions, float).reshape(-1, 2), np.asarray(footprint, float).reshape(-1, 2)
    hull = ConvexHull(np.asarray(zone['polygon_world_xy_m'], float))
    planes = np.vstack((hull.equations, np.asarray(zone['halfplanes_world_xy'], float).reshape(-1, 3)))
    extents = np.max(footprint @ planes[:, :2].T, axis=0)
    keep = np.all(positions @ planes[:, :2].T + planes[:, 2] + extents <= -clearance_m + 1e-9, axis=1)
    if 'measured_support_xy_m' in zone:
        tree = cKDTree(np.asarray(zone['measured_support_xy_m'], float))
        lower, upper = footprint.min(0), footprint.max(0)
        axes = [np.linspace(lower[i], upper[i], max(2, int(np.ceil((upper[i] - lower[i]) /
                 SURFACE_SAMPLE_SPACING_M)) + 1)) for i in range(2)]
        samples = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 2)
        for index in np.flatnonzero(keep):
            distance, _ = tree.query(samples + positions[index])
            keep[index] = bool(np.max(distance) <= zone['support_sample_distance_m'])
    return keep


def spatial_constraints(observation, intent, polygon, *, frame_axes=None):
    halfplanes, reference_geometry = [], {}
    for identifier in intent['reference_ids']:
        entity = next((item for item in observation['objects'] if item['id'] == identifier), None)
        if entity is None:
            raise ValueError('placement_reference_geometry_missing:' + identifier)
        points = _points(entity, 'placement_reference_geometry_missing')
        reference_geometry[identifier] = {'minimum_world_m': points.min(0).tolist(),
            'maximum_world_m': points.max(0).tolist(), 'point_count': len(points),
            'observation_id': observation['observation_id']}

    relative = intent['relative']
    if relative not in {'none', 'between'}:
        axis = _direction(observation, intent['frame'], 'left' if relative in {'left_of', 'right_of'} else 'front', frame_axes)
        if relative in {'right_of', 'behind'}:
            axis = -axis
        bounds = reference_geometry[intent['reference_ids'][0]]
        low, high = np.asarray(bounds['minimum_world_m']), np.asarray(bounds['maximum_world_m'])
        corners = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])
        halfplanes.append([float(-axis[0]), float(-axis[1]), float(np.max(corners @ axis))])
    elif relative == 'between':
        bounds = [reference_geometry[key] for key in intent['reference_ids']]
        centers = [np.mean([b['minimum_world_m'], b['maximum_world_m']], axis=0)[:2] for b in bounds]
        axis = centers[1] - centers[0]
        if np.linalg.norm(axis) < .001:
            raise ValueError('placement_between_references_overlap')
        axis /= np.linalg.norm(axis)
        extents = []
        for b in bounds:
            lo, hi = b['minimum_world_m'], b['maximum_world_m']
            corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            extents.append((float(np.min(corners @ axis)), float(np.max(corners @ axis))))
        if extents[0][1] >= extents[1][0]:
            raise ValueError('placement_between_references_overlap')
        halfplanes.extend([[-axis[0], -axis[1], extents[0][1]], [axis[0], axis[1], -extents[1][0]]])
        # Keep the result in the corridor between the two observed objects,
        # rather than the infinite strip between their projected centers.
        side = np.array([-axis[1], axis[0]])
        corners = np.array([[x, y] for b in bounds
                            for x in (b['minimum_world_m'][0], b['maximum_world_m'][0])
                            for y in (b['minimum_world_m'][1], b['maximum_world_m'][1])])
        halfplanes.extend([[side[0], side[1], -float(np.max(corners @ side))],
                           [-side[0], -side[1], float(np.min(corners @ side))]])
    zone = intent['zone']
    if zone not in {'whole', 'center'}:
        axis = _direction(observation, intent['frame'], 'left' if zone in {'left_half', 'right_half'} else 'front', frame_axes)
        if zone in {'right_half', 'back_half'}:
            axis = -axis
        bound = float((np.min(polygon @ axis) + np.max(polygon @ axis)) / 2)
        halfplanes.append([-axis[0], -axis[1], bound])
    return halfplanes, reference_geometry

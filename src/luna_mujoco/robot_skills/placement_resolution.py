"""Bounded numerical candidates and provenance, independent of task policy."""
import hashlib
import json
from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull

from ..algorithms import AlgorithmError
from ..carrying import held_collision_binding
from ..geometry import rigid_matrix
from ..inventory_map import layer_entity
from ..object_obb import held_geometry_points
from ..observation_labels import project_placement_regions
from ..placement import visible_reference_tool_m
from ..placement_candidates import subdivision_xy_candidates, subdivision_cell_centers
from .placement_selection import rank_support_candidates
from .placement_zone import placement_constraint_mask

RETRYABLE_PLACEMENT_ERRORS = {'CUROBO_TRAJOPT_FAILED', 'CUROBO_TRAJECTORY_COLLISION',
    'CUROBO_PLANNING_FAILED', 'CUROBO_NO_VALID_TRAJECTORY', 'CUROBO_GRASP_FAILED',
    'execution_collision', 'release_height_no_valid_goal', 'release_height_no_held_ik_goal', 'placement_configuration_conflict',
    'place_goal_workspace_rejected', 'placement_release_above_container', 'placement_release_too_high'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
        default=lambda item: item.tolist()).encode()).hexdigest()


def resolution_input(observation, target_id, arm, scene, holding, destination_id, request, settings):
    return {'target_id': target_id, 'destination_id': destination_id, 'arm': arm,
        'observation_id': observation['observation_id'], 'generation': observation['generation'],
        'scene_token': scene['scene_token'], 'placement_request': request, 'planning_settings': settings,
        'robot_feedback': {name: {key: feedback[key] for key in ('joints_rad', 'gripper_position_rad')}
                           for name, feedback in observation['arms'].items()},
        'occupancy_bounds': scene.get('collision_layer_bounds'),
        'carrying_binding': held_collision_binding(holding, scene, arm, target_id),
        'holding': {key: holding.get(key) for key in ('instance_id', 'arm', 'owner_id', 'frame',
            'verified', 'attached', 'points_tool_m', 'uncertainty_m', 'measurement_provenance',
            'grasp_support_reference', 'attachment_evidence')},
        'destination': next(item for item in observation['objects'] if item['id'] == destination_id)}


def validate_resolution(resolution, inputs, *, require_same_feedback=True):
    """Validate immutable pose bindings; stale feedback never authorizes a cached path."""
    from ..release_height_search import feedback_consistency

    unsigned = {key: value for key, value in resolution.items() if key != 'resolution_id'}
    check = feedback_consistency(resolution.get('robot_feedback'), inputs['robot_feedback'])
    bound_inputs = {**inputs, 'robot_feedback': resolution.get('robot_feedback')}
    if (set(check['arms']) != {'A', 'B'} or (require_same_feedback and not check['consistent'])
            or resolution.get('input_digest') != digest(bound_inputs)
            or resolution.get('resolution_id') != digest(unsigned) or not resolution.get('candidates')):
        raise ValueError('placement_resolution_changed')
    for candidate in resolution['candidates']:
        reference = candidate.get('goal', {}).get('release_reference', {})
        proof = reference.get('height_resolution', {})
        if (not reference.get('height_search') or proof.get('scope') != 'held_ik_release_endpoint'
                or proof.get('status') != 'first_world_and_held_ik_valid_sample'):
            raise ValueError('placement_resolution_requires_held_ik')
    return check


def revalidation_candidate_indices(previous, inputs, zone, requests):
    """Return only search seeds when feedback alone invalidated a signed resolution."""
    from ..release_height_search import feedback_consistency

    if not previous:
        return []
    try:
        # Validate the original proof against its own feedback, while binding
        # everything else to the current inputs. This does not authorize a plan.
        validate_resolution(previous, {**inputs, 'robot_feedback': previous.get('robot_feedback')})
        if (feedback_consistency(previous['robot_feedback'], inputs['robot_feedback'])['consistent']
                or digest(previous.get('placement_zone')) != digest(zone)):
            return []
    except ValueError:
        return []
    selection = previous.get('semantic_selection', {})
    candidates = previous['candidates']
    if (selection.get('status') != 'success' or len(candidates) > 12
            or any(item['candidate_id'] not in selection.get('candidate_ids', []) for item in candidates)):
        return []
    current_indices = {digest(request): index for index, request in enumerate(requests)}
    indices = []
    for candidate in candidates:
        index = current_indices.get(digest(candidate.get('request')))
        if index is None or candidate['candidate_id'] != 'placement-' + str(index):
            return []
        if index not in indices:
            indices.append(index)
    return indices


def occupancy_envelopes(observation, scene, ignored, floor_z):
    """Bounds from measured points and published collision geometry, not free-space proof."""
    bounds, missing, retained, _ = _occupancy_context(observation, scene, ignored, floor_z)
    return bounds, missing, retained


def _occupancy_context(observation, scene, ignored, floor_z):
    bounds, missing, retained = [], [], []
    model_sources = set()
    for entity in observation.get('objects', []):
        if entity['id'] in ignored or entity.get('kind') in {'marker', 'table'}:
            continue
        points = np.asarray(entity.get('points_world_m', []), float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
            missing.append(entity['id'])
            continue
        points = points[points[:, 2] > floor_z + .006]
        if len(points):
            bounds.append((entity['id'], points[:, :2].min(0), points[:, :2].max(0)))
    scene = scene or {}
    header = scene.get('collision_layer_bounds')
    if header is not None:
        if (any(header.get(key) != scene.get(key) for key in ('scene_token', 'observation_id', 'generation'))
                or any(observation.get(key) != scene.get(key) for key in ('observation_id', 'generation'))):
            raise AlgorithmError('Placement occupancy belongs to another scene', code='planning_stale_scene')
        records = scene.get('inventory_map', {}).get('entities', {})
        for name, layer in header.get('layers', {}).items():
            identifier = layer_entity(name)
            if identifier is None or identifier in ignored:
                continue
            if records.get(identifier, {}).get('status') == 'stale':
                raise AlgorithmError('Placement occupancy was invalidated', code='planning_stale_scene')
            lower = np.asarray(layer.get('minimum_world_m', []), float)
            upper = np.asarray(layer.get('maximum_world_m', []), float)
            native = (layer.get('kind') == 'voxel'
                and layer.get('source') == 'native_surface_voxel_cell_bounds'
                and layer.get('bounds_source') == 'all_native_surface_cells_before_preview_sampling'
                and isinstance(layer.get('cell_count'), int) and layer['cell_count'] > 0)
            model = ((layer.get('kind'), layer.get('source'), layer.get('geometry_source')) in {
                ('mesh', 'published_cad_mesh_bounds', 'registered_cad'),
                ('cuboid', 'published_cuboid_bounds', 'observed_obb')}
                and name == 'semantic_instance::' + identifier
                and layer.get('instance_id') == identifier
                and layer.get('generation') == scene.get('generation')
                and layer.get('observation_id') == records.get(identifier, {}).get(
                    'last_observation_id', scene.get('observation_id')))
            if (layer.get('enabled') is not True or name in scene.get('disabled_obstacle_names', [])
                    or not (native or model)
                    or lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
                    or np.any(upper <= lower)):
                missing.append(identifier)
                continue
            if model:
                model_sources.add(layer['source'])
            if upper[2] > floor_z + .006:
                bounds.append((identifier, lower[:2], upper[:2]))
                if records.get(identifier, {}).get('status') == 'retained':
                    retained.append(identifier)
    return bounds, missing, sorted(set(retained)), sorted(model_sources)


def candidate_requests(observation, target_id, holding, request, zone, settings, *, scene=None,
                       subdivision_cells=None):
    if any(zone.get(key) != observation.get(key) for key in ('observation_id', 'generation')):
        raise ValueError('placement_support_geometry_stale')
    if zone['destination_id'] != request.get('destination_id', zone['destination_id']):
        raise ValueError('placement_destination_changed')
    rotation = rigid_matrix(holding['grasp_support_reference']['T_world_tcp_at_grasp'])[:3, :3]
    footprint = held_geometry_points(holding) @ rotation.T
    cell = next(cell for cell in request['layout']['cells'] if cell['slot_id'] == request['slot_id'])
    precise = request.get('precise_xy') or not request.get('free_placement')
    if precise:
        cell = next(cell for cell in request['layout']['cells'] if cell['slot_id'] == request['slot_id'])
        positions = np.array([request.get('selected_reference_world_xy_m', cell['center_world_xy_m'])])
        destination = next(item for item in observation['objects'] if item['id'] == zone['destination_id'])
        fixed = destination['placement_reference'].get('precise_xy') or destination['placement_reference'].get('fixed_xy')
        offset = (rotation @ visible_reference_tool_m(holding))[:2] if fixed else np.zeros(2)
    else:
        positions = (subdivision_xy_candidates(zone['polygon_world_xy_m'],
            offsets_world_xy_m=footprint[:, :2]) if subdivision_cells is None else
            subdivision_cell_centers(cell['polygon_world_xy_m'],
                offsets_world_xy_m=footprint[:, :2], cell_count=subdivision_cells))
        offset = np.zeros(2)
    generated = len(positions)
    keep = placement_constraint_mask(positions - offset, footprint[:, :2], zone,
        clearance_m=settings['trajectory_collision_clearance_m'] + holding.get('uncertainty_m', .005))
    # The release-goal contract also binds its reference to the assigned cell.
    # Filter every subdivided proposal here, before any workspace/world query.
    cell = next(cell for cell in request['layout']['cells'] if cell['slot_id'] == request['slot_id'])
    for polygon in (cell['polygon_world_xy_m'], request['layout']['region_polygon_world_xy_m']):
        hull = ConvexHull(np.asarray(polygon, float))
        keep &= np.all(positions @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9, axis=1)
    positions = positions[keep]
    constrained_count = len(positions)
    # Bounds remain provenance/ranking context. Only the existing 3D world
    # checker may reject occupancy: an XY overlap can be an empty overhang.
    ignored = {target_id, zone['destination_id'], zone['support_entity_id']}
    margin = settings['trajectory_collision_clearance_m'] + holding.get('uncertainty_m', .005)
    _, missing, retained, model_sources = _occupancy_context(observation, scene, ignored, zone['floor_z_m'])
    ranked_observation = {**observation, 'objects': [obj for obj in observation['objects'] if obj['id'] not in ignored]}
    ranking = None
    if len(positions):
        _, ranking = rank_support_candidates(positions, ranked_observation, target_id,
            footprint_offsets=footprint[:, :2] - offset, floor_z_m=zone['floor_z_m'],
            preference=request.get('placement_preference', zone['intent']['preference']),
            desired_clearance_m=settings['optimizer_collision_activation_distance_m'])
    evidence = {'generated_count': generated, 'constraint_accepted_count': constrained_count,
        'occupied_candidate_count': 0,
        'retained_occupancy_ids': retained,
        'missing_geometry_ids': sorted(set(missing) | set(observation.get('visual_inventory', {}).get('missing_geometry_ids', [])) - ignored),
        'ranking': ranking, 'occupancy_rule': 'existing_scene_attachment_and_gripper_spheres',
        'occupancy_margin_m': margin, 'unknown_space_is_free': False, 'collision_checked': False}
    if model_sources:
        evidence.update(occupancy_geometry_sources=model_sources, occupancy_bounds_are_raw_measurements=False)
    if not len(positions):
        raise AlgorithmError('No sampled pose satisfies current measured placement constraints',
            code='placement_no_candidate', details={**evidence, 'whole_destination_infeasible': False})
    primary = positions
    evidence.update(primary_candidate_count=len(primary), remaining_candidate_count=0,
        xy_search_method='configured_xy' if precise else 'long_axis_binary_subdivision',
        maximum_xy_candidates=1 if precise else subdivision_cells or 15,
        subdivision_cells=subdivision_cells,
        accepted_candidate_limit=1 if precise or subdivision_cells is not None else 3,
        subdivision_failure_does_not_prune_children=True)
    return [deepcopy(request) | {'selected_reference_world_xy_m': position.tolist()}
            for position in primary], evidence


def project_placement_candidates(capture, resolution, holding, *, geometry_binding=None):
    """Project measured held-footprint proposals without altering captured RGB."""
    zone = resolution['placement_zone']
    separate_capture = (geometry_binding is not None
        and geometry_binding.get('source') == 'explicit_current_capture_over_verified_scene'
        and geometry_binding.get('scene_token')
        and geometry_binding.get('capture_observation_id') == capture.get('observation_id')
        and geometry_binding.get('geometry_observation_id') == zone.get('observation_id')
        and geometry_binding.get('generation') == capture.get('generation') == zone.get('generation')
        and geometry_binding.get('placement_zone_sha256') == digest(zone))
    if (any(capture.get(key) != zone.get(key) for key in ('observation_id', 'generation'))
            and not separate_capture):
        raise ValueError('placement_candidate_capture_stale')
    held = held_geometry_points(holding)
    if held.ndim != 2 or held.shape[1:] != (3,) or not len(held) or not np.isfinite(held).all():
        raise ValueError('placement_held_geometry_missing')
    cells, candidates = [], []
    for candidate in resolution['candidates']:
        pose = rigid_matrix(candidate['goal']['T_world_tcp'])
        points = held @ pose[:3, :3].T + pose[:3, 3]
        lower, upper = points[:, :2].min(0), points[:, :2].max(0)
        polygon = [[lower[0], lower[1]], [upper[0], lower[1]],
                   [upper[0], upper[1]], [lower[0], upper[1]]]
        identifier = candidate['candidate_id']
        cells.append({'slot_id': identifier, 'center_world_xy_m': ((lower + upper) / 2).tolist(),
                      'polygon_world_xy_m': np.asarray(polygon).tolist()})
        candidates.append({'candidate_id': identifier, 'visible_camera_ids': [],
            'footprint_world_xy_m': np.asarray(polygon).tolist(),
            'tcp_position_world_m': pose[:3, 3].tolist(), 'orientation_policy': 'preserve_grasp'})
    layout = {'support_entity_id': zone['support_entity_id'], 'floor_z_m': zone['floor_z_m'],
        'source': 'measured_candidate_footprint_proposal', 'observation_id': capture['observation_id'],
        'cells': cells}
    projected = project_placement_regions(capture, [{'region_id': 'placement', 'layout': layout}],
                                          require_visible_support=False)
    projections = {}
    by_id = {candidate['candidate_id']: candidate for candidate in candidates}
    for projection in projected:
        vertices = np.asarray(projection['polygon'])
        if not np.isfinite(vertices).all() or np.any(vertices < 0) or np.any(vertices > 1):
            continue
        camera = projection['camera_id']
        projections.setdefault(camera, []).append(projection)
        by_id[projection['slot_id']]['visible_camera_ids'].append(camera)
    return {'candidates': candidates, 'projections': projections,
        'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'geometry_observation_id': zone['observation_id'],
        **({'geometry_binding': deepcopy(geometry_binding)} if separate_capture else {}),
        'scope': 'Projected footprint proposals on the measured support plane; not observed objects or free-space proof.'}

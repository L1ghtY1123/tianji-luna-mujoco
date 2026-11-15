"""Require a common arm workspace for the measured source and fixed release TCP."""
from copy import deepcopy

import numpy as np

from .demo_placement import release_height_reference
from .placement import grasp_support_reference, observed_object


def fixed_release_position(destination, settings, *, target, scene=None, observation=None, release_mode=None):
    """Estimate release TCP from the source anchor before the actual grasp is available."""
    reference = destination['placement_reference']
    position = np.asarray(reference['position_world_m'], dtype=float).copy()
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('fixed_release_position_unmeasured')
    surface = float(reference['floor_z_m'])
    gap = settings['place_bottom_clearance_m']
    if destination['kind'] == 'bin':
        from .container_release import configure_release_region, container_release_reference

        destination = deepcopy(destination)
        configure_release_region(destination, settings)
        position[:2] = destination['placement_reference']['position_world_m'][:2]
        if release_mode == 'container_release':
            if scene is None or observation is None:
                raise ValueError('common_arm_requires_current_container_geometry')
            rim = container_release_reference(destination, scene, observation, settings)
            surface = rim['rim_z_m']
            gap = settings['container_release_bottom_clearance_m']
    anchor = np.eye(4)
    anchor[:3, 3] = np.asarray(target['points_world_m'], dtype=float).mean(axis=0)
    source = grasp_support_reference(observation, target['id'], anchor)
    points = np.asarray(target['points_world_m'], dtype=float)
    height = float(points[:, 2].max() - min(points[:, 2].min(), source['support_height_m']))
    position[2] = release_height_reference(source, surface, gap, object_height_m=height)['tcp_world_z_m']
    return position.tolist()


def evaluate_transfer_workspace(lookup, snapshot, target_id, destination_id, *, settings,
                                requested_arm=None, scene=None, release_mode=None):
    target = observed_object(snapshot, target_id)
    points = np.asarray(target['points_world_m'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('source_workspace_requires_current_geometry')
    destination = observed_object(snapshot, destination_id)
    goal = fixed_release_position(destination, settings, target=target, scene=scene, observation=snapshot,
                                  release_mode=release_mode)
    positions = [points.mean(axis=0).tolist(), goal]
    result = {}
    for arm in ((requested_arm,) if requested_arm else ('A', 'B')):
        if arm not in {'A', 'B'}:
            raise ValueError('invalid_requested_arm')
        rows = lookup(arm, positions)
        if (not isinstance(rows, list) or len(rows) != 2 or any(
                not isinstance(row, dict) or type(row.get('reachable')) is not bool
                or row.get('position_world_m') != position for row, position in zip(rows, positions))):
            raise ValueError('invalid_workspace_lookup_response')
        result[arm] = {'arm': arm, 'target_id': target_id, 'destination_id': destination_id,
            'source': rows[0], 'destination': rows[1], 'estimated_release_tcp_world_m': goal,
            'reachable': rows[0]['reachable'] and rows[1]['reachable'],
            'feasibility_scope': 'same_arm_source_and_estimated_release_tcp_workspace',
            'release_height_source': 'measured_visible_height_before_held_cad_or_tsdf_recheck',
            'actual_grasp_tcp_requires_release_goal_recheck': True,
            'collision_checked': False, 'complete_path_status': 'not_planned'}
    return result

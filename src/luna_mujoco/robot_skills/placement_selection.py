"""Measured support-goal ranking, separate from model policy and path validation."""
import numpy as np

from ..algorithms import AlgorithmError


def rank_support_candidates(candidates, observation, target_id, *, footprint_offsets,
                            floor_z_m, preference, desired_clearance_m):
    """Prefer separation from measured obstacles; never infer unobserved free space.

    Input candidates are already footprint-inset and ordered by arm reach. Observed
    bounds are a conservative ranking heuristic only, not collision geometry or a
    reason to reject a candidate. TSDF/ESDF and carried-body checks remain mandatory.
    """
    if preference not in {'clearance', 'reach'}:
        raise ValueError('unsupported_placement_preference')
    points = np.asarray(candidates, float).reshape(-1, 2)
    offsets = np.asarray(footprint_offsets, float).reshape(-1, 2)
    lower, upper = points + offsets.min(axis=0), points + offsets.max(axis=0)
    gaps = np.full(len(points), np.inf)
    obstacle_ids = []
    missing = set(observation.get('visual_inventory', {}).get('missing_geometry_ids', [])) - {target_id}
    for entity in observation.get('objects', []):
        if entity['id'] == target_id or entity.get('kind') in {'table', 'marker'}:
            continue
        cloud = np.asarray(entity.get('points_world_m', []), float)
        if cloud.ndim != 2 or cloud.shape[1:] != (3,) or not len(cloud) or not np.isfinite(cloud).all():
            missing.add(entity['id'])
            continue
        cloud = cloud[cloud[:, 2] >= floor_z_m]
        if not len(cloud):
            continue
        obstacle_ids.append(entity['id'])
        distance = np.maximum(np.maximum(cloud[:, :2].min(axis=0) - upper,
                                         lower - cloud[:, :2].max(axis=0)), 0.)
        gaps = np.minimum(gaps, np.linalg.norm(distance, axis=1))
    # Once the configured soft-clearance distance is reached, favor arm reach.
    # This avoids chasing the farthest table edge for an insignificant extra gap.
    order = (np.argsort(-np.minimum(gaps, desired_clearance_m), kind='stable')
             if preference == 'clearance' else np.arange(len(points)))
    evidence = {'preference': preference, 'observation_id': observation.get('observation_id'),
        'source': 'measured_inventory_bounds_and_carried_footprint',
        'desired_clearance_m': float(desired_clearance_m), 'obstacle_ids': obstacle_ids,
        'missing_geometry_ids': sorted(missing), 'collision_proof': False,
        'unknown_space_is_free': False,
        'candidates': [{'position_world_xy_m': points[i].tolist(), 'reach_rank': int(i),
            'measured_obstacle_gap_m': float(gaps[i]) if np.isfinite(gaps[i]) else None}
            for i in order]}
    return points[order], evidence


def check_goal_workspace(lookup, arm, position):
    """Check the final gripper-front XYZ, after release height has been resolved."""
    rows = lookup(arm, [position])
    if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
            or rows[0].get('position_world_m') != position or type(rows[0].get('reachable')) is not bool):
        raise AlgorithmError('invalid place workspace result', code='algorithm_invalid_response')
    if not rows[0]['reachable']:
        raise AlgorithmError('actual place goal is outside the calibrated workspace',
            code='place_goal_workspace_rejected', details={'workspace': rows})
    return rows[0]


def check_height_workspace(lookup, arm, position, *, positions=None):
    """Reject XY only when none of the bounded TCP height samples is reachable."""
    from ..demo_placement import MAX_RELEASE_HEIGHT_INCREASE_M, RELEASE_HEIGHT_STEP_M

    if positions is None:
        count = round(MAX_RELEASE_HEIGHT_INCREASE_M / RELEASE_HEIGHT_STEP_M) + 1
        positions = [[*position[:2], position[2] + i * RELEASE_HEIGHT_STEP_M] for i in range(count)]
    rows = []
    for start in range(0, len(positions), 2):
        values = positions[start:start + 2]
        batch = lookup(arm, values)
        if (not isinstance(batch, list) or len(batch) != len(values) or any(
                not isinstance(row, dict) or row.get('position_world_m') != point
                or type(row.get('reachable')) is not bool for row, point in zip(batch, values))):
            raise AlgorithmError('invalid place workspace result', code='algorithm_invalid_response')
        rows.extend(batch)
    if not any(row['reachable'] for row in rows):
        raise AlgorithmError('bounded place heights are outside the calibrated workspace',
            code='place_goal_workspace_rejected', details={'workspace': rows})
    return rows

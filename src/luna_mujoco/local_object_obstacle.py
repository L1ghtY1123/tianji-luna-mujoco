"""Measure one verified tabletop object from a selected calibrated RGBD camera."""
import time
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from .cameras import deproject_view
from .robot_surface import own_robot_surface_pixels


class LocalObjectDepthError(ValueError):
    def __init__(self, details):
        super().__init__('local_object_depth_missing')
        self.details = details


def select_unpack_component(points, work_area):
    """Require one new measured black component; never merge separate identities."""
    from .workspace import WorkspaceAtlas

    cells, inverse = np.unique(np.floor(points / .01).astype(np.int64), axis=0, return_inverse=True)
    pending = {tuple(cell): index for index, cell in enumerate(cells)}
    labels = np.full(len(cells), -1, dtype=int)
    neighbors = list(product((-1, 0, 1), repeat=3))
    component_count = 0
    while pending:
        start, index = pending.popitem()
        labels[index] = component_count
        queue = [start]
        while queue:
            current = queue.pop()
            for delta in neighbors:
                adjacent = tuple(current[axis] + delta[axis] for axis in range(3))
                index = pending.pop(adjacent, None)
                if index is not None:
                    labels[index] = component_count
                    queue.append(adjacent)
        component_count += 1
    candidates, excluded = [], []
    atlas = WorkspaceAtlas()
    for label in range(component_count):
        component = points[labels[inverse] == label]
        if len(component) < 30:
            continue
        known = False
        for previous in work_area.get('known_black_objects', []):
            lower = np.asarray(previous['minimum_world_m']) - .01
            upper = np.asarray(previous['maximum_world_m']) + .01
            overlap = np.all((component >= lower) & (component <= upper), axis=1)
            if overlap.all():
                excluded.append(previous['id'])
                known = True
                break
            if overlap.any():
                raise LocalObjectDepthError({'reason': 'black_candidate_identity_overlap',
                                             'known_entity_id': previous['id']})
        if known:
            continue
        if atlas.query('B', [component.mean(axis=0).tolist()])[0]['reachable']:
            candidates.append(component)
    proof = {'component_count': component_count, 'new_candidate_count': len(candidates),
             'excluded_known_entity_ids': excluded, 'workspace_arm': 'B',
             'workspace_scope': 'position_prefilter_not_grasp_or_path_certificate'}
    if len(candidates) != 1:
        reason = 'multiple_unbound_black_candidates' if candidates else 'new_black_candidate_missing'
        raise LocalObjectDepthError({**proof, 'reason': reason})
    return candidates[0], proof


def measure_object_obstacle(capture, detector, work_area, appearance, *, camera_id='3'):
    started = time.perf_counter()
    if appearance not in ('wrapped', 'unpacked'):
        raise ValueError('local_object_appearance_invalid')
    view = capture['views'][camera_id]
    polygon = np.asarray(work_area['region_polygon_world_xy_m'], float)
    floor = float(work_area['floor_z_m'])
    prompts = (['pink object', 'pink foam wrapped object on the table'] if appearance == 'wrapped'
               else ['black object', 'black object on the table', 'black workpiece'])
    mask = np.zeros(view['depth_m'].shape, bool)
    attempts = []
    for prompt in prompts:
        proposals = detector.segment(view['rgb'], prompt, round(capture['time'] * 1_000_000))
        attempts.append({'prompt': prompt, 'proposal_count': len(proposals)})
        for proposal in proposals:
            proposed = np.asarray(proposal['mask'], bool)
            if proposed.shape != mask.shape:
                raise ValueError('local_object_mask_shape_invalid')
            if float(proposal['score']) >= .5:
                mask |= proposed
    counts = {'segmented_pixels': int(mask.sum())}
    mask &= view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)
    counts['valid_depth_pixels'] = int(mask.sum())
    ownership = own_robot_surface_pixels(capture, {camera_id: mask})
    if ownership['status'] != 'measured':
        raise ValueError('local_object_robot_feedback_missing')
    mask &= ~ownership['masks'][camera_id]
    counts['non_robot_pixels'] = int(mask.sum())
    pixels = np.flatnonzero(mask)
    points = deproject_view(view, mask)
    keep = ((points[:, :2] >= polygon.min(axis=0)).all(axis=1)
            & (points[:, :2] <= polygon.max(axis=0)).all(axis=1)
            & (points[:, 2] > floor + .005))
    points, pixels = points[keep], pixels[keep]
    counts['table_points'] = len(points)
    diagnostics = {'camera_id': camera_id, 'observation_id': capture['observation_id'],
        'segmentation_attempts': attempts, 'point_counts': counts}
    if len(points) < 30:
        raise LocalObjectDepthError(diagnostics)
    # Remove isolated returns, not surfaces based on a simulator object pose.
    neighbors = cKDTree(points).query_ball_point(points, .01, return_length=True)
    points, pixels = points[neighbors >= 4], pixels[neighbors >= 4]
    counts['dense_table_points'] = len(points)
    if len(points) < 30:
        raise LocalObjectDepthError(diagnostics)
    component_evidence = None
    if appearance == 'unpacked' and work_area.get('source') == 'measured_cached_main_table_support':
        selected, component_evidence = select_unpack_component(points, work_area)
        keep = cKDTree(selected).query(points)[0] <= 1e-9
        points, pixels = points[keep], pixels[keep]
    mask = np.zeros(mask.shape, bool)
    mask.flat[pixels] = True
    return { 'camera_id': camera_id, 'observation_id': capture['observation_id'],
            'generation': capture['generation'], 'source': 'current_rgbd_object_surface',
            'appearance': appearance, 'segmentation_attempts': attempts, 'point_counts': counts, 'mask': mask,
            'points_world_m': points, 'component_evidence': component_evidence,
            'wall_s': time.perf_counter() - started}

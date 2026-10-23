"""One verified tabletop object's head RGBD obstacle; never remap the background."""
import time

import numpy as np
from scipy.spatial import cKDTree

from .cad_assets import workpiece_height_prior
from .cameras import deproject_view
from .robot_surface import own_robot_surface_pixels


class LocalObjectDepthError(ValueError):
    def __init__(self, details):
        super().__init__('local_object_depth_missing')
        self.details = details


def object_bounds(points, lower_xy, upper_xy, floor, height):
    points = np.asarray(points, float).reshape(-1, 3)
    valid = (np.isfinite(points).all(axis=1)
             & (points[:, :2] >= lower_xy).all(axis=1)
             & (points[:, :2] <= upper_xy).all(axis=1)
             & (points[:, 2] > floor + .005) & (points[:, 2] < floor + height + .03))
    points = points[valid]
    if not len(points):
        raise ValueError('local_object_depth_missing')
    # Local RGBD cuboids are independent of the 2 cm global voxel resolution.
    # Padding is measurement allowance; path clearance adds a separate 10 mm.
    lower, upper = points.min(axis=0) - .005, points.max(axis=0) + .005
    lower[2] = floor
    upper[2] = max(float(upper[2]), floor + height + .01)
    return {'minimum_world_m': lower.tolist(), 'maximum_world_m': upper.tolist(),
            'point_count': len(points), 'height_prior_m': height,
            'support_floor_z_m': floor, 'padding_m': .005}


def measure_object_obstacle(capture, detector, work_area, appearance, *, camera_id='3'):
    started = time.perf_counter()
    if appearance not in ('wrapped', 'unpacked'):
        raise ValueError('local_object_appearance_invalid')
    view = capture['views'][camera_id]
    polygon = np.asarray(work_area['region_polygon_world_xy_m'], float)
    floor = float(work_area['floor_z_m'])
    height = workpiece_height_prior('pink_wrapped' if appearance == 'wrapped' else 'black_core')
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
    points = deproject_view(view, mask)
    keep = ((points[:, :2] >= polygon.min(axis=0)).all(axis=1)
            & (points[:, :2] <= polygon.max(axis=0)).all(axis=1)
            & (points[:, 2] > floor + .005) & (points[:, 2] < floor + height + .03))
    points = points[keep]
    counts['table_points'] = len(points)
    diagnostics = {'camera_id': camera_id, 'observation_id': capture['observation_id'],
        'segmentation_attempts': attempts, 'point_counts': counts}
    if len(points) < 30:
        raise LocalObjectDepthError(diagnostics)
    # Remove isolated returns, not surfaces based on a simulator object pose.
    neighbors = cKDTree(points).query_ball_point(points, .01, return_length=True)
    points = points[neighbors >= 4]
    counts['dense_table_points'] = len(points)
    if len(points) < 30:
        raise LocalObjectDepthError(diagnostics)
    bounds = object_bounds(points, polygon.min(axis=0), polygon.max(axis=0), floor, height)
    return {'bounds': bounds, 'camera_id': camera_id, 'observation_id': capture['observation_id'],
            'generation': capture['generation'], 'source': ('head_rgbd_object_surface_with_height_prior'
                if camera_id == '3' else 'wrist_rgbd_object_surface_with_height_prior'),
            'appearance': appearance, 'segmentation_attempts': attempts, 'point_counts': counts, 'mask': mask,
            'points_world_m': points, 'wall_s': time.perf_counter() - started}

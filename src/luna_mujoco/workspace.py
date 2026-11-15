"""CPU lookup of versioned, offline single-arm position workspaces."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import TOOL_PROFILE
from .reachability import validate_query

ROOT = Path(__file__).resolve().parents[2]
MODEL_POLICY = 'single_arm_open_gripper_and_fixed_body_v1'


def model_fingerprint(root=ROOT):
    digest = hashlib.sha256(MODEL_POLICY.encode())
    for name in ('resources/robot/marvin_gui_dual.yml', 'assets/marvin_urdf/marvin_with_grippers.urdf'):
        digest.update((root / name).read_bytes())
    digest.update(json.dumps(TOOL_PROFILE, sort_keys=True).encode())
    return digest.hexdigest()


class WorkspaceAtlas:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.maps = {}
        self.surface_preview = None

    def contains(self, positions):
        """Vectorized union of the two offline workspace volumes in Link_Base."""
        points = np.asarray(positions, dtype=float).reshape(-1, 3)
        inside = np.zeros(len(points), dtype=bool)
        for arm in ('A', 'B'):
            _, grid, origin, resolution = self._load(arm)
            indices = np.floor((points - origin) / resolution + .5).astype(np.int64)
            bounded = np.all((indices >= 0) & (indices < grid.shape), axis=1)
            inside[bounded] |= grid[tuple(indices[bounded].T)]
        return inside

    def mask_capture(self, capture):
        """Project the reusable workspace volume using each frame's calibrated pose."""
        from .cameras import deproject_view

        capture['raw_views'] = {camera: dict(view) for camera, view in capture['views'].items()}
        for view in capture['views'].values():
            depth = np.asarray(view['depth_m'])
            valid = np.asarray(view['valid_depth']) & np.isfinite(depth) & (depth > 0)
            mask = np.zeros(depth.shape, dtype=bool)
            mask[valid] = self.contains(deproject_view(view))
            rgb = np.asarray(view['rgb']).copy()
            rgb[~mask] = 0
            depth = depth.copy()
            depth[~mask] = 0
            for key, value in (('rgb', rgb), ('depth_m', depth), ('valid_depth', mask),
                               ('workspace_mask', mask), ('effective_mask', mask)):
                value.setflags(write=False)
                view[key] = value
        capture['workspace_filter'] = {
            'source': 'offline_arm_workspace_union', 'frame': 'Link_Base',
            'model_fingerprint': self._load('A')[0]['model_fingerprint'],
            'outside_rgb': 0, 'outside_depth_m': 0,
            'camera_pose_source': 'current_capture_calibration'}
        return capture

    def _load(self, arm):
        if arm in self.maps:
            return self.maps[arm]
        path = self.root / f'resources/workspaces/arm_{arm}.npz'
        if not path.is_file():
            raise ValueError('workspace_file_missing:' + str(path))
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            reachable = data['reachable']
        if metadata.get('model_fingerprint') != model_fingerprint(self.root):
            raise ValueError('workspace_model_mismatch:regenerate_offline_workspace')
        origin = np.asarray(metadata.get('origin_m'), dtype=float)
        resolution = metadata.get('resolution_m', 0)
        if (metadata.get('schema_version') != 1 or metadata.get('arm') != arm or metadata.get('complete') is not True
                or metadata.get('frame') != 'Link_Base' or reachable.ndim != 3 or reachable.dtype != bool
                or not all(reachable.shape) or origin.shape != (3,) or not np.isfinite(origin).all()
                or not np.isfinite(resolution) or resolution <= 0):
            raise ValueError('invalid_workspace_artifact')
        self.maps[arm] = (metadata, reachable, origin, resolution)
        return self.maps[arm]

    def preview(self):
        """Read-only boundary faces of the exact lookup cells, cached for display."""
        if self.surface_preview is not None:
            return self.surface_preview
        arms = []
        for arm in ('A', 'B'):
            metadata, grid, origin, resolution = self._load(arm)
            faces = []
            for axis in range(3):
                for sign in (-1, 1):
                    neighbor = np.zeros_like(grid)
                    destination, source = [slice(None)] * 3, [slice(None)] * 3
                    destination[axis] = slice(1, None) if sign == -1 else slice(None, -1)
                    source[axis] = slice(None, -1) if sign == -1 else slice(1, None)
                    neighbor[tuple(destination)] = grid[tuple(source)]
                    faces.append({'axis': axis, 'sign': sign,
                                  'indices': np.flatnonzero(grid & ~neighbor).tolist()})
            arms.append({'arm': arm, 'origin_m': origin.tolist(), 'resolution_m': resolution,
                'shape': list(grid.shape), 'reachable_points': int(grid.sum()),
                'model_fingerprint': metadata['model_fingerprint'], 'faces': faces})
        self.surface_preview = {'source': 'offline_single_arm_workspace', 'frame': 'Link_Base',
                                'surface_policy': 'exposed_voxel_faces', 'arms': arms}
        return self.surface_preview

    def query(self, arm, positions):
        _, points = validate_query({'arm': arm, 'positions_world_m': positions})
        metadata, grid, origin, resolution = self._load(arm)
        results = []
        for point in points:
            index = np.floor((point - origin) / resolution + .5).astype(np.int64)
            in_bounds = bool(np.all(index >= 0) and np.all(index < grid.shape))
            reachable = bool(in_bounds and grid[tuple(index)])
            center = origin + index * resolution
            results.append({'reachable': reachable, 'position_world_m': point.tolist(),
                'method': 'offline_workspace_lookup', 'workspace_arm': arm,
                'workspace_model_fingerprint': metadata['model_fingerprint'],
                'grid_center_m': center.tolist(), 'grid_distance_m': float(np.linalg.norm(center - point)),
                'resolution_m': resolution, 'orientation_constrained': False,
                'collision_scope': 'single_arm_and_fixed_robot_body', 'environment_collision_checked': False,
                'peer_arm_collision_checked': False, 'reference_offset_m': metadata['reference_offset_m'],
                'status': 'inside_sampled_workspace' if reachable else 'outside_sampled_workspace',
                'limitation': 'finite_grid_position_prefilter_not_a_pose_or_path_guarantee'})
        return results


def retain_instance_masks(capture, masks_by_camera):
    """Register current masks within the immutable workspace pixel boundary."""
    if not capture.get('raw_views'):
        return capture
    registered = capture.setdefault('registered_instance_masks', {})
    for camera, masks in masks_by_camera.items():
        view = capture['views'][camera]
        target = registered.setdefault(camera, {})
        for identifier, mask in masks.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != view['workspace_mask'].shape:
                raise ValueError('registered_instance_mask_shape_mismatch')
            target[identifier] = (np.asarray(target.get(identifier, np.zeros_like(mask))) | mask) & view['workspace_mask']
    for camera, view in capture['views'].items():
        raw = capture['raw_views'][camera]
        keep = np.array(view['workspace_mask'], dtype=bool, copy=True)
        valid = keep & raw['valid_depth'] & np.isfinite(raw['depth_m']) & (raw['depth_m'] > 0)
        for name, value in (('rgb', np.where(keep[..., None], raw['rgb'], 0)),
                            ('depth_m', np.where(valid, raw['depth_m'], 0)),
                            ('valid_depth', valid), ('effective_mask', keep)):
            value.setflags(write=False)
            view[name] = value
    capture.setdefault('workspace_filter', {}).update(
        source='offline_arm_workspace_union',
        registered_entity_ids=sorted({identifier for masks in registered.values() for identifier in masks}),
        identity_assigned=True, mask_observation_id=capture['observation_id'])
    return capture

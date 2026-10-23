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
    for name in ('resources/curobo/marvin_gui_dual.yml', 'assets/marvin_urdf/marvin_with_grippers.urdf'):
        digest.update((root / name).read_bytes())
    digest.update(json.dumps(TOOL_PROFILE, sort_keys=True).encode())
    return digest.hexdigest()


class WorkspaceAtlas:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.maps = {}
        self.surface_preview = None

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

"""Project table policy around the frozen public plane-fitting adapter."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np


def table_surface_crop_model(table):
    """The measured top is the lower bound for object points and TSDF cells."""
    model = dict(table.model)
    model["pose"] = [*model["pose"][:2], float(table.height_m) + .5, *model["pose"][3:]]
    model["dims"] = [*model["dims"][:2], 1.]
    return model


def _cache_key(payload):
    cameras = {}
    metadata = payload.get("camera_capture_metadata", {})
    for camera in ('3',):
        view = payload['views'][camera]
        cameras[camera] = {name: np.asarray(view.get(name, [])).tolist() for name in
                           ("left_intrinsics", "rgb_intrinsics", "left_to_rgb", "T_base_left_camera")}
        cameras[camera].update(shape=list(np.asarray(view["depth_left_m"]).shape),
            calibration_id=metadata.get(camera, {}).get("calibration_id"))
    value = {"cameras": cameras}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _robot_clearance(table, spheres):
    spheres = np.asarray(spheres, dtype=float).reshape(-1, 4)
    if not len(spheres):
        return float("inf")
    center = np.asarray(table.model["pose"][:3])
    half = np.asarray(table.model["dims"]) / 2
    outside = np.maximum(np.abs(spheres[:, :3] - center) - half, 0)
    return float(np.min(np.linalg.norm(outside, axis=1) - spheres[:, 3]))


def _with_downward_depth(table, depth):
    model = dict(table.model)
    model['pose'] = [*model['pose'][:2], table.height_m - depth / 2, *model['pose'][3:]]
    model['dims'] = [*model['dims'][:2], depth]
    return replace(table, model=model)


class MeasuredTableCache:
    """Freeze measured static support until explicit reset or head calibration change."""

    def __init__(self, *, table_builder=None):
        if table_builder is None:
            from tianji_curobo.mixed_esdf_scene import build_table_completion
            table_builder = build_table_completion
        self.table_builder = table_builder
        self.reset()

    def reset(self):
        self.table = None
        self.key = None
        self.configuration = None
        self.diagnostics = {"cache_hit": False, "invalidation_reason": "reset"}
        self.demo_tables = None
        self.demo_signature = None

    def build_demo(self, evidence, spheres, *, hard_clearance_m):
        """Reuse each initial inventory table; never fit arbitrary background."""
        from tianji_curobo.mixed_esdf_scene import TableCompletion

        if (not isinstance(evidence, dict) or not evidence.get('source_observation_id')
                or not isinstance(evidence.get('tables'), dict)
                or evidence.get('primary_table_id') not in evidence['tables']):
            raise ValueError('demo_initial_table_evidence_required')
        signature = hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if self.demo_tables is not None:
            if signature != self.demo_signature:
                raise ValueError('demo_table_inventory_changed_requires_reset')
            for table in self.demo_tables.values():
                if _robot_clearance(table, spheres) < hard_clearance_m - 1e-8:
                    raise ValueError('tabletop_robot_clearance_violated')
            self.diagnostics.update(cache_hit=True)
            return self.table, {identifier: dict(table.model) for identifier, table in self.demo_tables.items()}
        tables = {}
        for identifier, item in evidence['tables'].items():
            model = deepcopy(item['model'])
            pose, dims = np.asarray(model['pose'], float), np.asarray(model['dims'], float)
            normal = np.asarray(item['normal'], float)
            z = float(item['surface_height_m'])
            if (pose.shape != (7,) or dims.shape != (3,) or not np.isfinite([*pose, *dims, z]).all()
                    or np.any(dims <= 0) or not np.allclose(pose[3:], [1, 0, 0, 0])
                    or normal.shape != (3,) or not np.isfinite(normal).all()
                    or abs(np.linalg.norm(normal) - 1) > 1e-5 or normal[2] < np.cos(np.deg2rad(5))
                    or abs(pose[2] + dims[2] / 2 - z) > 1e-6
                    or item.get('source') != 'initial_inventory_ransac_table' or item['inlier_count'] < 100):
                raise ValueError('demo_table_evidence_invalid')
            table = TableCompletion(model, z, tuple(item['normal']), int(item['inlier_count']), hard_clearance_m)
            if _robot_clearance(table, spheres) < hard_clearance_m - 1e-8:
                # Only inferred downward thickness can shrink; measured XY/top stay fixed.
                low, high = .001, float(dims[2])
                if _robot_clearance(_with_downward_depth(table, low), spheres) < hard_clearance_m - 1e-8:
                    raise ValueError('tabletop_robot_clearance_violated')
                for _ in range(40):
                    middle = (low + high) / 2
                    if _robot_clearance(_with_downward_depth(table, middle), spheres) >= hard_clearance_m:
                        low = middle
                    else:
                        high = middle
                table = _with_downward_depth(table, low)
            tables[identifier] = table
        self.demo_tables, self.demo_signature = tables, signature
        self.table = tables[evidence['primary_table_id']]
        self.diagnostics = {'cache_hit': False, 'source_observation_id': evidence['source_observation_id'],
                            'table_ids': list(tables), 'source': 'initial_inventory_ransac_table',
                            'tabletop_voxels': False, 'table_revision': signature[:24]}
        return self.table, {identifier: dict(table.model) for identifier, table in tables.items()}

    def build(self, groups, spheres, payload, *, hard_clearance_m, open3d_runtime=None):
        key = _cache_key(payload)
        configuration = {'hard_clearance_m': hard_clearance_m,
                         'pointcloud_filter': payload.get('planning_settings', {}).get('pointcloud_filter', {})}
        reason = "empty" if self.table is None else "head_calibration_changed"
        if (self.key != key and payload.get('scene_policy') in {'all_instances', 'selected_instances'}):
            raise ValueError('initial_head_table_required')
        if self.table is not None and self.key == key:
            if configuration != self.configuration:
                raise ValueError('table_configuration_changed_requires_scene_reset')
            gap = _robot_clearance(self.table, spheres)
            if gap >= hard_clearance_m - 1e-8:
                self.diagnostics = {**self.diagnostics, "cache_hit": True, "invalidation_reason": None,
                                    "current_robot_clearance_m": gap if np.isfinite(gap) else None}
                return self.table
            self.diagnostics = {**self.diagnostics, 'cache_hit': True,
                                'current_robot_clearance_m': gap,
                                'collision_error': 'tabletop_robot_clearance_violated'}
            raise ValueError('tabletop_robot_clearance_violated')
        self.table, self.key = None, None
        table = self.table_builder(groups, spheres, open3d_runtime=open3d_runtime,
                                   robot_clearance_m=max(.02, hard_clearance_m))
        model = dict(table.model)
        depth = .30
        model["pose"] = [*model["pose"][:2], table.height_m - depth / 2, *model["pose"][3:]]
        model["dims"] = [*model["dims"][:2], depth]
        candidate = replace(table, model=model)
        if _robot_clearance(candidate, spheres) < hard_clearance_m - 1e-8:
            # Only the inferred downward extent may shrink. The measured top
            # and fitted XY footprint remain fixed; an actual top conflict fails.
            low, high = .001, depth
            def at_depth(value):
                return replace(table, model={**model,
                    'pose': [*model['pose'][:2], table.height_m - value / 2, *model['pose'][3:]],
                    'dims': [*model['dims'][:2], value]})
            if _robot_clearance(at_depth(low), spheres) < hard_clearance_m - 1e-8:
                raise ValueError('tabletop_robot_clearance_violated')
            for _ in range(40):
                middle = (low + high) / 2
                if _robot_clearance(at_depth(middle), spheres) >= hard_clearance_m:
                    low = middle
                else:
                    high = middle
            depth = low
            model = at_depth(depth).model
        plane = table.support_plane
        if plane is not None:
            from tianji_curobo.support_contact import revision
            plane = {**plane, "replacement_cuboid": model}
            plane["revision"] = revision(plane)
        table = replace(table, model=model, support_plane=plane)
        gap = _robot_clearance(table, spheres)
        if gap < hard_clearance_m - 1e-8:
            raise ValueError("tabletop_robot_clearance_violated")
        self.table, self.key = table, key
        self.configuration = deepcopy(configuration)
        table_revision = hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
        self.diagnostics = {"cache_hit": False, "invalidation_reason": reason, "key": key,
            "source_observation_id": payload.get("observation_set_id"),
            'initial_head_observation_id': payload.get('observation_set_id'), 'table_revision': table_revision,
            "measured_height_m": float(table.height_m), "top_m": float(table.height_m),
            "bottom_m": float(table.height_m - depth), "downward_extent_m": depth,
            "robot_fitting_clearance_m": max(.02, hard_clearance_m),
            "hard_collision_clearance_m": hard_clearance_m,
            "current_robot_clearance_m": gap if np.isfinite(gap) else None}
        return table

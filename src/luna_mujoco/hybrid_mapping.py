"""Registered CAD, measured table cuboids and unknown TSDF/ESDF through public scene inputs."""
import hashlib
import struct
import time
from copy import deepcopy
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import rigid_matrix
from .measured_table_mapping import MeasuredTableCache, table_surface_crop_model
from .perception_mapping import CaptureSegmenter
from .sdk_scene import sdk_scene_model
from .semantic_builder import _requests, crop_instance_samples


def _require(condition, message="geometry_object_invalid"):
    if not condition:
        raise ValueError(message)


def _current(value, observation_id, generation):
    _require(isinstance(value, dict) and value.get("observation_id") == observation_id
             and type(value.get("generation")) is int and value["generation"] == generation,
             "geometry_object_stale")


def _pose(transform):
    transform = rigid_matrix(transform)
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)
    return [*transform[:3, 3].tolist(), *quaternion.tolist()]


def build_geometry_model(rows, observation_id, generation, *, catalog=None):
    """Validate current geometry bindings and return public mesh/cuboid inputs.

    CAD file paths come only from the verified project catalog. Its meshes are
    canonical metres; source-CAD transforms are checked but never used as the
    canonical mesh pose. Raw registered observations remain separate evidence.
    """
    _require(isinstance(rows, list) and isinstance(observation_id, str) and bool(observation_id)
             and type(generation) is int and generation >= 1)
    model = {"cuboid": {}, "mesh": {}, "voxel": {}}
    identifiers = set()
    try:
        for row in rows:
            _current(row, observation_id, generation)
            identifier = row["id"]
            _require(isinstance(identifier, str) and bool(identifier) and "::" not in identifier
                     and identifier not in identifiers and isinstance(row.get("kind"), str)
                     and bool(row["kind"]) and row["kind"] not in ("table", "marker"))
            identifiers.add(identifier)
            has_cad = row.get("cad") is not None
            unknown = row.get("geometry_type") == "tsdf_esdf"
            _require(has_cad != unknown, "geometry_object_requires_cad_or_tsdf_esdf")
            metadata = {"instance_id": identifier, "observation_id": observation_id, "generation": generation}
            name = "semantic_instance::" + identifier
            if has_cad:
                cad = row["cad"]
                _current(cad, observation_id, generation)
                _require(cad.get("instance_id") == identifier, "cad_instance_binding_changed")
                _require(isinstance(cad.get("pose_validation"), dict) and cad["pose_validation"].get("accepted") is True,
                         "cad_pose_validation_required")
                if catalog is None:
                    from .cad_perception import CadCatalog
                    catalog = CadCatalog.from_file()
                record = catalog.record(cad["asset_id"])
                _require(record["mesh_key"] == cad["mesh_key"], "cad_model_binding_changed")
                geometry = catalog.geometry(cad["mesh_key"])
                _require(geometry["asset_id"] == record["asset_id"]
                         and geometry["source_sha256"] == cad["source_sha256"]
                         and geometry["mesh_sha256"] == cad.get("mesh_sha256"), "cad_model_binding_changed")
                canonical = rigid_matrix(cad["T_world_object"])
                source = rigid_matrix(cad["T_world_source"])
                source_from_canonical = rigid_matrix(geometry["T_source_canonical_m"])
                _require(np.allclose(source @ source_from_canonical, canonical, atol=1e-6, rtol=0),
                         "cad_pose_frame_mismatch")
                model["mesh"][name] = {"file_path": str(geometry["path"]), "pose": _pose(canonical),
                    **metadata, "geometry_source": "registered_cad", "asset_id": cad["asset_id"],
                    "source_sha256": geometry["source_sha256"], "mesh_sha256": geometry["mesh_sha256"]}
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("geometry_object_invalid") from error
    return model


class HybridGeometryMapper:
    """Compose per-instance unknown ESDF grids with registered CAD and measured table cuboids."""

    def __init__(self, segmenter_boundary, *, scene_type=None, catalog=None):
        if scene_type is None:
            from curobo.scene import Scene
            scene_type = Scene
        self.segmenter_boundary = segmenter_boundary
        self.capture_segmenter = CaptureSegmenter(segmenter_boundary)
        self.table_cache = MeasuredTableCache()
        self.scene_type = scene_type
        self.catalog = catalog
        self.previous_instances = {}
        self.pending_instances = {}
        from .instance_mapping import InstanceMapper

        self.instance_mapper = InstanceMapper(self.capture_segmenter)
        from .workspace_mapping import WorkspaceMapper

        self.workspace_mapper = WorkspaceMapper(self.capture_segmenter)
        self.tsdf_surfaces = {}
        self.diagnostics = {}

    def commit_observation(self):
        self.previous_instances = self.pending_instances
        self.workspace_mapper.previous_blocks = self.workspace_mapper.pending_blocks

    def reset(self):
        self.previous_instances = {}
        self.pending_instances = {}
        self.capture_segmenter.begin_capture(None)
        self.table_cache.reset()
        self.instance_mapper.cache.clear()
        self.workspace_mapper.reset()
        self.tsdf_surfaces = {}
        self.diagnostics = {}

    def update_segmenter(self, segmenter_boundary, *, scene_type=None):
        self.capture_segmenter.begin_capture(None)
        self.segmenter_boundary = segmenter_boundary
        self.capture_segmenter.boundary = segmenter_boundary
        if scene_type is not None:
            self.scene_type = scene_type

    def build(self, decoded):
        self.capture_segmenter.begin_capture(decoded)
        self.diagnostics = {}
        try:
            return self._build(decoded)
        except Exception:
            self.capture_segmenter.begin_capture(None)
            raise

    def _build(self, decoded):
        from tianji_curobo.fixed_esdf_mapping import SemanticEsdfCandidate
        from tianji_curobo.mixed_esdf_scene import (
            MappingBounds,
            create_mixed_esdf_scene,
            fixed_mapping_bounds,
        )
        from tianji_curobo.volumetric_mapper import EsdfIntegrationResult

        started = time.perf_counter()
        requests = _requests(decoded)
        demo = decoded.get("demo_mapping", False)
        _require(type(demo) is bool and (not demo or tuple(decoded["views"]) == ("1", "2", "3")),
                 "demo_mapping_requires_three_registered_views")
        _require(decoded.get("mapping_mode", "semantic") == "semantic", "hybrid_semantic_mapping_required")
        hard_clearance = float(decoded.get("mapping_collision_policy", {}).get("trajectory_collision_clearance_m", .01))
        _require(np.isfinite(hard_clearance) and hard_clearance >= 0, "invalid_mapping_clearance")
        model = build_geometry_model(decoded["geometry_objects"], decoded["observation_set_id"],
                                     decoded["selection_generation"], catalog=self.catalog)
        requested = set().union(*(set(projection.instance_masks) for _, projection in requests))
        identifiers = {row["id"] for row in decoded["geometry_objects"]}
        _require(requested == identifiers, "geometry_instance_masks_mismatch")
        prepared, background, robots, camera_metrics = [], [], [], {}
        for depth, projection in requests:
            raw = self.capture_segmenter.prepare_ffs_observation(depth)
            view = self.capture_segmenter.remove_semantic_instances(raw, projection)
            prepared.append(view)
            background.append(view.background_points_base_m)
            robots.append(np.asarray(self.segmenter_boundary.robot_spheres(depth.joint_positions_deg)).reshape(-1, 4))
            camera_metrics[view.camera_id] = {"input_point_count": int(view.input_point_count),
                "filtered_point_count": int(view.filtered_point_count),
                "background_point_count": int(view.background_point_count),
                "removed_robot_point_count": int(view.removed_point_count),
                "removed_unclassified_background_point_count": 0}
        timings = {"depth_filter_projection": (time.perf_counter() - started) * 1000}
        stage = time.perf_counter()
        spheres = np.concatenate(robots, axis=0)
        if demo:
            table, table_models = self.table_cache.build_demo(decoded.get("demo_tables"), spheres,
                                                              hard_clearance_m=hard_clearance)
            model["cuboid"].update({"semantic_table::" + name: value for name, value in table_models.items()})
        else:
            table_payload, table_groups = decoded, background
            if self.table_cache.table is None and decoded.get("scene_policy") in ("all_instances", "selected_instances"):
                head = next(view for view in prepared if view.camera_id == "3")
                table_groups = [head.background_points_base_m]
                table_payload = decoded | {"scene_policy": "head_all_instances", "views": {"3": decoded["views"]["3"]}}
            table = self.table_cache.build(table_groups, spheres, table_payload, hard_clearance_m=hard_clearance)
            model["cuboid"]["semantic_table_forbidden"] = dict(table.model)
            crop = table_surface_crop_model(table)
            prepared = [crop_instance_samples(view, crop) for view in prepared]
        timings["table"] = (time.perf_counter() - stage) * 1000
        metrics = {}
        unknown = {row['id'] for row in decoded['geometry_objects'] if row.get('geometry_type') == 'tsdf_esdf'}
        instance_points = {identifier: {view.camera_id: view.instance_points_base_m.get(identifier, np.empty((0, 3)))
                                      for view in prepared} for identifier in unknown}
        radius = max(float(np.max(spheres[:, 3])), float(decoded.get('mapping_max_sphere_radius_m', 0.)))
        padding = radius + hard_clearance + self.instance_mapper.esdf_voxel_size_m
        prior = self.previous_instances if decoded.get('mapping_lifecycle', {}).get('previous_scene_token') else {}
        signatures, reuse = {}, {}
        for identifier in sorted(unknown):
            # The per-instance TSDF depends on measured rays, calibration and grid padding.
            # Equal IDs or equal centroids cannot establish equal geometry.
            digest = hashlib.sha256(np.float64(padding).tobytes())
            for view in prepared:
                array = np.ascontiguousarray(instance_points[identifier][view.camera_id])
                digest.update(str((view.camera_id, array.shape, array.dtype.str)).encode())
                digest.update(array.tobytes())
                camera = decoded['views'][view.camera_id]
                for field in ('left_intrinsics', 'rgb_intrinsics', 'left_to_rgb', 'T_base_left_camera'):
                    digest.update(repr(camera.get(field)).encode() if isinstance(camera.get(field),dict)
                                  else np.ascontiguousarray(camera.get(field, [])).tobytes())
                samples = (view.instance_depth_samples or {}).get(identifier)
                if samples is not None:
                    for sample in samples:
                        digest.update(np.ascontiguousarray(sample).tobytes())
            signatures[identifier] = digest.hexdigest()
            if identifier in prior and prior[identifier][0] == signatures[identifier]:
                reuse[identifier] = prior[identifier]
        changed = {key: value for key, value in instance_points.items() if key not in reuse}
        geometries = self.instance_mapper.build(prepared, changed, padding_m=padding, table_model=None)
        grids = list(self.instance_mapper.grids)
        by_name = {grid.name: grid for grid in grids}
        self.pending_instances = {}
        self.tsdf_surfaces = {}
        for row in decoded['geometry_objects']:
            identifier = row['id']
            groups = [view.instance_points_base_m.get(identifier, np.empty((0, 3))) for view in prepared]
            count = sum(len(points) for points in groups)
            _require(count > 0 and all(np.isfinite(points).all() for points in groups), 'semantic_instance_depth_empty')
            kind = 'mesh' if row.get('cad') is not None else 'voxel'
            if identifier in unknown:
                if identifier in reuse:
                    _, geometry, grid, surface_points = reuse[identifier]
                    grids.append(grid)
                else:
                    geometry = geometries[identifier]
                    grid = by_name[geometry.name]
                    surface_points = self.instance_mapper.surfaces[identifier]
                self.pending_instances[identifier] = (signatures[identifier], geometry, grid, surface_points)
                model['voxel'][geometry.name] = {**dict(geometry.model), 'instance_id': identifier,
                    'observation_id': decoded['observation_set_id'], 'generation': decoded['selection_generation'],
                    'geometry_source': 'curobo_tsdf_esdf'}
                self.tsdf_surfaces[identifier] = {'source': 'curobo_tsdf_surface', 'frame': 'world',
                    'observation_id': decoded['observation_set_id'], 'generation': decoded['selection_generation'],
                    'points_world_m': surface_points,
                    'tsdf_voxel_size_m': self.instance_mapper.voxel_size_m,
                    'esdf_voxel_size_m': self.instance_mapper.esdf_voxel_size_m}
            metrics[identifier] = {'geometry': kind, 'input_point_count': count, 'filtered_point_count': count,
                'camera_count': sum(bool(len(points)) for points in groups),
                'geometry_source': model[kind]['semantic_instance::' + identifier]['geometry_source']}
        bounds = fixed_mapping_bounds()
        if demo:
            bounds = MappingBounds(center_m=(.55, 0., .8), extent_m=(1.1, 2., 1.6),
                                   minimum_m=(0., -1., 0.), maximum_m=(1.1, 1., 1.6))
        from .workspace_mapping import BACKGROUND_LAYER

        background_grid, background_model = self.workspace_mapper.build(prepared, decoded, model)
        grids.append(background_grid)
        model['voxel'][BACKGROUND_LAYER] = background_model
        scene = create_mixed_esdf_scene(self.scene_type, grids, sdk_scene_model(model))
        integration = EsdfIntegrationResult(struct.pack("<I", 0), 0, 0, 1, 0, None)
        timings["build_total"] = (time.perf_counter() - started) * 1000
        self.diagnostics = {"semantic_only_active": False, "background_grid_omitted": False,
            "geometry_policy": "cuboid_cad_and_unknown_esdf", "grids": [grid.name for grid in grids], "demo_mapping": demo,
            "scene_policy": decoded.get("scene_policy"), "camera_ids": list(camera_metrics),
            "included_instance_ids": list(metrics), "table_cache": deepcopy(self.table_cache.diagnostics),
            "background_drop_policy": "discard_robot_table_and_outside_workspace", "unobserved_space": "unknown",
            "workspace_mapping": deepcopy(self.workspace_mapper.diagnostics),
            "instance_reuse": {"reused_tsdf_ids": sorted(reuse), "rebuilt_tsdf_ids": sorted(changed)},
            "workspace_bounds_m": {"minimum": list(bounds.minimum_m), "maximum": list(bounds.maximum_m)}}
        candidate = SemanticEsdfCandidate(observation_set_id=decoded["observation_set_id"],
            selected_instance_id=decoded["confirmed_instance_id"], selection_generation=decoded["selection_generation"],
            scene_data=scene, scene_model=MappingProxyType(model), bounds=bounds, table=table, integration=integration,
            camera_metrics=MappingProxyType(camera_metrics), instance_metrics=MappingProxyType(metrics),
            timings_ms=MappingProxyType(timings))
        return candidate, grids, model

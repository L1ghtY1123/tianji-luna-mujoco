"""Project scene policy composed from frozen public semantic mapping components."""

import struct
import time
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import numpy as np
from tianji_curobo.contracts import CuroboToolError
from tianji_curobo.fixed_esdf_mapping import SemanticEsdfCandidate
from tianji_curobo.mixed_esdf_scene import (
    build_table_completion,
    create_mixed_esdf_scene,
    fixed_mapping_bounds,
    table_footprint_mask,
)
from tianji_curobo.robot_segmenter import FfsDepthObservationRequest, SemanticProjectionRequest
from tianji_curobo.surface_colors import SurfaceColorizer

from .measured_table_mapping import MeasuredTableCache, table_surface_crop_model


def crop_instance_samples(view, table_model):
    """Crop measured object points/pixels; leave unused background untouched."""
    instances, samples = {}, {}
    for name, points in view.instance_points_base_m.items():
        keep = table_footprint_mask(points, table_model)
        instances[name] = np.ascontiguousarray(points[keep], dtype=np.float32)
        if name in (view.instance_depth_samples or {}):
            indices, depths = view.instance_depth_samples[name]
            samples[name] = (indices[keep], depths[keep])
    return replace(view, instance_points_base_m=MappingProxyType(instances),
                   instance_depth_samples=MappingProxyType(samples))


def _requests(payload):
    if not isinstance(payload, Mapping):
        raise TypeError("mapping capture must be a mapping")
    background = payload.get("mapping_mode", "semantic") == "background"
    targetless = payload.get('exclusion_state') == 'observation' and payload.get('confirmed_instance_id') == ''
    if (not isinstance(payload, Mapping) or payload.get("exclusion_state") not in ("selected", "observation", "held", "released")
            or payload.get("mapping_mode", "semantic") not in ("semantic", "background")
            or not isinstance(payload.get("observation_set_id"), str) or not payload["observation_set_id"]
            or not isinstance(payload.get("confirmed_instance_id"), str)
            or (bool(payload["confirmed_instance_id"]) == background and not targetless)
            or type(payload.get("selection_generation")) is not int or payload["selection_generation"] < 1):
        raise ValueError("mapping capture identity is invalid")
    views = payload.get("views")
    allowed = (("3",), ("1", "3"), ("2", "3"), ("1", "2", "3")) if background else (("3",), ("1", "2", "3")) if targetless else (("1", "2", "3"),)
    if not isinstance(views, Mapping) or tuple(views) not in allowed:
        raise ValueError("mapping requires ordered registered views containing the head")
    requests, instances = [], set()
    for camera_id, view in views.items():
        if not isinstance(view, Mapping) or view.get("camera_id") != camera_id:
            raise ValueError("semantic camera identity changed")
        if background and view.get("instance_masks") != {}:
            raise ValueError("background mapping must retain all valid instance depth")
        depth = FfsDepthObservationRequest.from_mapping(view)
        projection = SemanticProjectionRequest.from_mapping(view, allow_empty=True)
        if requests and (depth.stereo_sequence != requests[0][0].stereo_sequence
                         or depth.joint_positions_deg != requests[0][0].joint_positions_deg):
            raise ValueError("semantic cameras must share frozen capture feedback and sequence")
        instances.update(projection.instance_masks)
        requests.append((depth, projection))
    if not background and not targetless and payload["confirmed_instance_id"] not in instances:
        raise ValueError("selected instance is absent from the original semantic masks")
    return requests


def rounded_collision_padding(minimum_padding_m, voxel_size_m=.02):
    """Round outward to whole cells, avoiding ambiguous native scatter boundaries.

    Native object allocation uses 2 cm bounds and 2 cm TSDF cells. Matching
    ESDF cell centers keeps those TSDF centers away from scatter cell edges.
    A 1 nm rounding tolerance prevents a floating-point ceil adding an odd
    cell; it is included before rounding so required coverage never shrinks.
    """
    tolerance_m = 1e-9
    return float(np.ceil((minimum_padding_m + tolerance_m) / voxel_size_m) * voxel_size_m - tolerance_m)


def _color_preview(views, geometries, integration):
    colorizer = SurfaceColorizer(views, voxel_size_m=.02)
    if not colorizer.views:
        return geometries, integration
    colored = []
    for geometry in geometries:
        if geometry.kind == "voxel":
            model = dict(geometry.model)
            model["colors_rgb"] = colorizer.colors(model["centers_base_m"], geometry.instance_id).tolist()
            geometry = replace(geometry, model=model)
        colored.append(geometry)
    data = integration.pointcloud_data
    count = struct.unpack_from("<I", data)[0]
    centers = np.frombuffer(data, dtype="<f4", count=count * 3, offset=4).reshape(-1, 3)
    return colored, replace(integration, pointcloud_data=data[:4 + count * 12] + colorizer.colors(centers).tobytes())


class ProjectSemanticBuilder:
    def __init__(self, *, robot_segmenter_boundary, mapper_boundary, scene_type, robot_spheres,
                 open3d_runtime=None, table_builder=build_table_completion, instance_mapper=None):
        self.segmenter = robot_segmenter_boundary
        self.mapper = mapper_boundary
        self.scene_type = scene_type
        self.robot_spheres = robot_spheres
        self.open3d_runtime = open3d_runtime
        self.table_builder = table_builder
        self.table_cache = MeasuredTableCache(table_builder=table_builder)
        self.diagnostics = {}
        self.empty_background_grid = None
        self.empty_background_result = None
        self.instance_mapper = instance_mapper
        self.scene_voxel_grids = []

    def build(self, payload):
        started = time.perf_counter()
        self.diagnostics = {}
        self.scene_voxel_grids = []
        requests = _requests(payload)
        if payload.get('scene_policy') in {'all_instances', 'selected_instances'} and self.table_cache.table is None:
            raise ValueError('initial_head_table_required')
        policy = payload.get("mapping_collision_policy", {})
        hard_clearance = float(policy.get("trajectory_collision_clearance_m", .01))
        distances = [hard_clearance, float(policy.get("optimizer_collision_activation_distance_m", .03))]
        if policy.get("ik_collision_activation_distance_m") is not None:
            distances.append(float(policy["ik_collision_activation_distance_m"]))
        if not np.isfinite(distances).all() or min(distances) < 0:
            raise ValueError("mapping collision distances must be finite and nonnegative")
        semantic_only = (payload.get('scene_policy') in {'head_all_instances', 'all_instances', 'selected_instances'} or (
                         payload.get("experimental_semantic_only_mapping") is True
                         and payload.get("full_observation_complete") is True
                         and not payload.get("observation_only")
                         and payload.get("mapping_mode", "semantic") == "semantic"))
        omit_background = semantic_only and self.instance_mapper is not None
        self.mapper.clear_instance_layers()
        prepared, background_groups, robot_groups = [], [], []
        camera_metrics, timings = {}, {}
        for depth, projection in requests:
            raw = self.segmenter.prepare_ffs_observation(depth)
            view = self.segmenter.remove_semantic_instances(raw, projection, open3d_runtime=self.open3d_runtime)
            prepared.append(view)
            background_groups.append(view.background_points_base_m)
            robot_groups.append(np.asarray(self.robot_spheres(depth.joint_positions_deg)).reshape(-1, 4))
            camera_metrics[view.camera_id] = {
                "input_point_count": int(view.input_point_count),
                "filtered_point_count": int(view.filtered_point_count),
                "background_point_count": int(view.background_point_count),
                "removed_robot_point_count": int(view.removed_point_count),
            }
        timings["depth_filter_projection"] = (time.perf_counter() - started) * 1000
        stage = time.perf_counter()
        spheres = np.concatenate(robot_groups, axis=0)
        table = self.table_cache.build(background_groups, spheres, payload,
                                      hard_clearance_m=hard_clearance, open3d_runtime=self.open3d_runtime)
        crop_model = table_surface_crop_model(table)
        timings["table"] = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        cropped = []
        for view in prepared:
            result = (crop_instance_samples(view, crop_model) if omit_background else
                      self.segmenter.crop_semantic_observation(view, crop_model))
            camera_metrics[view.camera_id].update(
                background_point_count=int(result.background_point_count),
                removed_workspace_background_point_count=int(view.background_point_count - result.background_point_count),
                removed_table_penetrating_background_point_count=int(np.count_nonzero(
                    np.asarray(view.background_points_base_m)[:, 2] < table.height_m)),
                removed_table_penetrating_instance_point_count=sum(
                    int(np.count_nonzero(np.asarray(points)[:, 2] < table.height_m))
                    for points in view.instance_points_base_m.values()))
            cropped.append(result)
        timings["table_crop"] = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        # The project boundary keeps this explicit policy disabled by default.
        prepared = self.segmenter.filter_isolated_surfaces(cropped)
        instance_points = {}
        for view in prepared:
            metrics = camera_metrics[view.camera_id]
            metrics["removed_isolated_background_point_count"] = int(metrics["background_point_count"] - view.background_point_count)
            metrics["background_point_count"] = int(view.background_point_count)
            for name, points in view.instance_points_base_m.items():
                instance_points.setdefault(name, {})[view.camera_id] = points
        timings["isolated_surfaces"] = (time.perf_counter() - stage) * 1000
        selected = payload["confirmed_instance_id"]
        if payload.get('scene_policy') in {'head_all_instances', 'all_instances', 'selected_instances'}:
            requested_instances = set().union(*(set(view['instance_masks']) for view in payload['views'].values()))
            empty_instances = sorted(name for name in requested_instances
                if not any(len(points) for points in instance_points.get(name, {}).values()))
            if empty_instances:
                raise CuroboToolError('Measured semantic instance has no usable depth.', stage='scene',
                    code='semantic_instance_depth_empty', status=422, details={'instance_ids': empty_instances})
        if selected and not any(len(points) for points in instance_points.get(selected, {}).values()):
            raise CuroboToolError("The selected object has no valid depth inside the table footprint.",
                stage="scene", code="mapping_primary_depth_empty", status=422, details={"instance_id": selected})
        stage = time.perf_counter()
        radius = max(float(np.max(spheres[:, 3])), float(payload.get("mapping_max_sphere_radius_m", 0.)))
        minimum_padding_m = radius + max(distances) + .02
        padding_m = rounded_collision_padding(minimum_padding_m)
        geometries, instance_metrics = [], {}
        parallel = (self.instance_mapper.build(prepared, instance_points, padding_m=padding_m, table_model=crop_model,
                                              floor_heights_m=payload.get('instance_floor_heights_m', {}))
                    if self.instance_mapper is not None else None)
        for name, points_by_camera in instance_points.items():
            if not any(len(points) for points in points_by_camera.values()):
                instance_metrics[name] = {"skipped_reason": "empty_depth", "input_point_count": 0,
                                          "filtered_point_count": 0, "camera_count": 0}
                continue
            geometry = (parallel[name] if parallel is not None else
                        self.mapper.build_instance_esdf(prepared, name, points_by_camera,
                                                       padding_m=padding_m, table_model=crop_model))
            geometries.append(geometry)
            instance_metrics[name] = {"geometry": geometry.kind,
                "input_point_count": int(geometry.input_point_count),
                "filtered_point_count": int(geometry.filtered_point_count),
                "camera_count": sum(bool(len(points)) for points in points_by_camera.values()),
                "padding_m": padding_m, "max_sphere_radius_m": radius,
                "minimum_padding_m": minimum_padding_m, "padding_rounding_m": padding_m - minimum_padding_m,
                "collision_distance_m": max(distances), "boundary_margin_m": .02,
                "tsdf_grid_policy": "native_allocation_phase_and_measured_rays",
                "esdf_dims_m": geometry.model.get("dims")}
            if self.instance_mapper is not None:
                instance_metrics[name].update(self.instance_mapper.metrics.get(name, {}))
        timings["instances"] = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        bounds = fixed_mapping_bounds()
        self.mapper.configure_mapping_bounds(bounds)
        if table.support_plane is not None and not omit_background:
            remaining = []
            for view in prepared:
                result = self.segmenter.separate_support_observation(view, table.support_plane)
                camera_metrics[view.camera_id].update(
                    classified_support_point_count=int(view.background_point_count - result.background_point_count),
                    remaining_background_point_count=int(result.background_point_count))
                remaining.append(result)
            prepared = remaining
        for view in prepared:
            camera_metrics[view.camera_id]["removed_unclassified_background_point_count"] = (
                int(view.background_point_count) if semantic_only else 0)
        if semantic_only and not omit_background:
            empty = []
            for view in prepared:
                source = view.camera_observation
                depth = source.depth_image
                observation = type(source)(depth_image=depth.new_tensor(np.zeros(depth.shape)),
                    intrinsics=source.intrinsics, pose=source.pose, depth_to_meter=1.)
                empty.append(replace(view, camera_observation=observation,
                    background_points_base_m=np.empty((0, 3), dtype=np.float32), background_point_count=0))
            prepared = empty
        # The frozen public adapter exposes instance grids only alongside its
        # background grid. Build an empty one once, then reuse that exact empty
        # allocation. A raw/background build or reset invalidates it by identity.
        empty_cache_hit = bool(semantic_only and self.empty_background_result is not None
                               and self.mapper.esdf_grid is self.empty_background_grid)
        if omit_background:
            from tianji_curobo.volumetric_mapper import EsdfIntegrationResult
            integration = EsdfIntegrationResult(struct.pack('<I', 0), 0, 0, 1, 1, None, surface_voxel_size_m=.02)
        elif empty_cache_hit:
            integration = self.empty_background_result
        else:
            integration = self.mapper.integrate_multiview(prepared, table_model=crop_model)
            self.empty_background_result = integration if semantic_only else None
            self.empty_background_grid = self.mapper.esdf_grid if semantic_only else None
        self.diagnostics = {"semantic_only_active": semantic_only, 'background_grid_omitted': omit_background,
            'scene_policy': payload.get('scene_policy', 'legacy'), 'camera_ids': list(payload['views']),
            'included_instance_ids': list(instance_metrics),
            'background_drop_policy': 'all_nonsemantic_including_support' if omit_background else 'separate_measured_support',
            "empty_background_cache_hit": empty_cache_hit,
            "table_cache": dict(self.table_cache.diagnostics), "instance_padding_m": padding_m,
            "minimum_padding_m": minimum_padding_m, "padding_rounding_m": padding_m - minimum_padding_m,
            "max_sphere_radius_m": radius, "collision_distance_m": max(distances), "boundary_margin_m": .02,
            "crop_minimum_z_m": float(table.height_m),
            "removed_unclassified_background_point_count": sum(
                item["removed_unclassified_background_point_count"] for item in camera_metrics.values())}
        timings["background"] = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        geometries, integration = _color_preview(payload["views"], geometries, integration)
        timings["preview_rgb"] = (time.perf_counter() - stage) * 1000
        scene_model = {"cuboid": {"semantic_table_forbidden": dict(table.model)}, "mesh": {}, "voxel": {}}
        for geometry in geometries:
            scene_model[geometry.kind][geometry.name] = dict(geometry.model)
        if self.instance_mapper is None:
            self.scene_voxel_grids = list(self.mapper.scene_voxel_grids)
        else:
            self.scene_voxel_grids = ([] if omit_background else [self.mapper.esdf_grid]) + list(self.instance_mapper.grids)
        scene_data = create_mixed_esdf_scene(self.scene_type, self.scene_voxel_grids, scene_model)
        timings["build_total"] = (time.perf_counter() - started) * 1000
        if not empty_cache_hit and not omit_background:
            timings.update(self.mapper.timings_ms)
        return SemanticEsdfCandidate(
            observation_set_id=payload["observation_set_id"], selected_instance_id=selected,
            selection_generation=payload["selection_generation"], scene_data=scene_data,
            scene_model=MappingProxyType({key: MappingProxyType(values) for key, values in scene_model.items()}),
            bounds=bounds, table=table, integration=integration,
            camera_metrics=MappingProxyType({key: MappingProxyType(values) for key, values in camera_metrics.items()}),
            instance_metrics=MappingProxyType({key: MappingProxyType(values) for key, values in instance_metrics.items()}),
            timings_ms=MappingProxyType(timings))

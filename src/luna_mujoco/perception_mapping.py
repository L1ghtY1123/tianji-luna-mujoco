"""Registered RGBD transport into the frozen semantic GPU Mapper adapter."""

import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))

MAX_CAPTURE_SURFACE_POINTS = 3 * 640 * 480


def mapping_filter_settings(payload):
    settings = payload.get("planning_settings", {}).get("pointcloud_filter", payload.get("pointcloud_filter", {}))
    if not isinstance(settings, dict):
        raise TypeError("pointcloud_filter settings must be a mapping")
    return settings


def decode_mapping_payload(payload):
    from .planning_settings import PlanningSettings
    from .semantic_transport import decode_project_semantic_request

    if payload.get('retained_bin_ids') or payload.get('completed_transfer'):
        raise ValueError('predicted_scene_reuse_disabled')
    settings = PlanningSettings.model_validate(payload.get("planning_settings", {})).model_dump()
    full_observation = payload.get("full_observation_complete", False)
    if type(full_observation) is not bool:
        raise ValueError("full_observation_complete must be a boolean")
    radius = float(payload.get("mapping_max_sphere_radius_m", 0.))
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("mapping maximum sphere radius must be finite and nonnegative")
    demo_mapping = payload.get('demo_mapping', False)
    if type(demo_mapping) is not bool:
        raise ValueError('demo_mapping must be a boolean')
    demo_tables = payload.get('demo_tables')
    demo_completions = payload.get('demo_completions', {})
    demo_bin_priors = payload.get('demo_bin_priors', {})
    if demo_mapping and payload.get('mapping_mode', 'semantic') != 'semantic':
        raise ValueError('demo_inventory_semantic_mapping_required')
    if demo_mapping and not isinstance(demo_tables, dict):
        raise ValueError('demo_initial_table_evidence_required')
    values = {name: value for name, value in payload.items() if name not in ("planning_settings", "exclusion_state", "gripper_joint_positions_rad", "selected_surface_world_m", "pointcloud_filter", "checkpoint", "held_instance_id", "camera_capture_metadata", "retained_bin_ids", "completed_transfer", "full_observation_complete", "mapping_max_sphere_radius_m", "instance_floor_heights_m", "demo_mapping", "demo_tables", "demo_completions", "demo_bin_priors")}
    decoded = decode_project_semantic_request(values, exclusion_state=payload.get("exclusion_state", "selected"))
    feedback = payload.get("gripper_joint_positions_rad")
    if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
        raise ValueError("mapping requires measured A and B gripper driver feedback")
    try:
        feedback = {arm: float(feedback[arm]) for arm in ("A", "B")}
        if not np.isfinite(list(feedback.values())).all():
            raise ValueError("mapping gripper driver feedback must be finite")
    except (ValueError, TypeError) as error:
        raise ValueError("mapping gripper driver feedback must be finite") from error
    selected = decoded["confirmed_instance_id"]
    held_id = payload.get("held_instance_id")
    if held_id is not None:
        if (payload.get("exclusion_state") != "held" or decoded.get("mapping_mode") != "background"
                or not isinstance(held_id, str) or not held_id):
            raise ValueError("held background map requires explicit held identity")
        selected = held_id
    observation = decoded["observation_set_id"]
    generation = decoded["selection_generation"]
    if not isinstance(observation, str) or not observation or type(generation) is not int or generation < 1:
        raise ValueError("mapping observation identity or generation is invalid")
    views = decoded["views"]
    if demo_mapping and tuple(views) != ('1', '2', '3'):
        raise ValueError('demo_mapping requires all three registered views')
    sequences = {view.get("stereo_sequence") for view in views.values()}
    joints = [view.get("robot_joints_at_capture") for view in views.values()]
    if len(sequences) != 1 or any(value != joints[0] for value in joints[1:]):
        raise ValueError("mapping views must share one frozen capture")
    instance_ids = set().union(*(set(view["instance_masks"]) for view in views.values()))
    if demo_mapping:
        from .demo_bin_geometry import validate_bin_priors
        validate_bin_priors(demo_bin_priors, instance_ids, observation, generation)
        # Missing/unlisted returns are invalid observations, never free-space rays.
        for view in views.values():
            depth = np.asarray(view['depth_left_m'])
            registered = registered_source_masks(view, depth)
            retained = np.logical_or.reduce(list(registered.values())) if registered else np.zeros_like(depth, bool)
            valid = retained & (np.asarray(view['valid_left_mask']) != 0) & np.isfinite(depth) & (depth > 0)
            view['depth_left_m'] = np.where(valid, depth, 0)
            view['valid_left_mask'] = valid.astype(np.uint8)
        if not isinstance(demo_completions, dict) or set(demo_completions) - instance_ids:
            raise ValueError('demo_completion_requires_current_inventory_instance')
    floors = payload.get('instance_floor_heights_m', {})
    if (not isinstance(floors, dict) or set(floors) - instance_ids
            or any(type(value) not in (int, float) or not np.isfinite(value) for value in floors.values())):
        raise ValueError('instance_floor_heights_m_requires_visible_instances_and_finite_heights')
    targetless_observation = payload.get('exclusion_state') == 'observation' and not selected
    if held_id is None and selected not in instance_ids and not targetless_observation:
        raise ValueError("selected instance is absent from registered RGBD masks")
    exclusion = payload.get("exclusion_state", "selected")
    if exclusion not in ("selected", "held", "released", "observation"):
        raise ValueError("invalid mapping exclusion_state")
    surface = payload.get("selected_surface_world_m", [])
    points = np.asarray(surface, dtype=float)
    if len(surface) and (points.ndim != 2 or points.shape[1] != 3
                        or len(points) > MAX_CAPTURE_SURFACE_POINTS or not np.isfinite(points).all()):
        raise ValueError("selected_surface_world_m must contain finite measured Nx3 points")
    content = json.dumps(values | {"gripper_joint_positions_rad": feedback, "selected_surface_world_m": surface,
                                  "pointcloud_filter": mapping_filter_settings(payload), "held_instance_id": held_id,
                                  "planning_settings": settings, "full_observation_complete": full_observation,
                                  "mapping_max_sphere_radius_m": radius, "instance_floor_heights_m": floors,
                                  "demo_mapping": demo_mapping, "demo_tables": demo_tables,
                                  "demo_completions": demo_completions,
                                  'demo_bin_priors': demo_bin_priors},
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    token = "rgbd:" + hashlib.sha256(content + exclusion.encode()).hexdigest()[:24]
    identity = {"scene_token": token, "observation_id": observation, "generation": generation,
                "target_id": selected, "exclusion_state": exclusion,
                "disabled_obstacle_names": () if exclusion in ("released", "observation") else (f"semantic_instance::{selected}",),
                "gripper_joint_positions_rad": feedback,
                "selected_surface_world_m": surface,
                "collision_world_source": "measured_semantic_esdf", "source": "registered_rgbd"}
    decoded["observation_only"] = exclusion == "observation"
    decoded["exclusion_state"] = exclusion
    decoded['scene_policy'] = (('head_all_instances' if tuple(views) == ('3',) else 'all_instances') if targetless_observation
                               and decoded.get('mapping_mode', 'semantic') == 'semantic'
                               else 'selected_instances' if exclusion == 'selected' else 'legacy')
    decoded['demo_mapping'] = demo_mapping
    if demo_mapping:
        decoded['scene_policy'] = 'demo_inventory'
        decoded['demo_tables'], decoded['demo_completions'] = demo_tables, demo_completions
        decoded['demo_bin_priors'] = demo_bin_priors
        identity.update(demo_mapping=True, collision_world_source='inventory_semantic_esdf_and_cached_table_cuboids',
                        unobserved_space='unknown', observed_free_motion_gate=False,
                        unlisted_depth_policy='discard_without_free_space_claim')
        if selected in demo_completions and exclusion in ('selected', 'held'):
            identity['disabled_obstacle_names'] += (f'semantic_instance::{selected}::height_completion',)
    decoded["selected_surface_world_m"] = surface
    decoded["camera_capture_metadata"] = payload.get("camera_capture_metadata", {})
    decoded["planning_settings"] = settings
    decoded["mapping_collision_policy"] = {name: settings[name] for name in
        ("trajectory_collision_clearance_m", "optimizer_collision_activation_distance_m")}
    decoded["experimental_semantic_only_mapping"] = settings.get("experimental_semantic_only_mapping", False)
    decoded["full_observation_complete"] = full_observation
    decoded["mapping_max_sphere_radius_m"] = radius
    decoded['instance_floor_heights_m'] = floors
    return decoded, identity


def filter_mapping_views(decoded, settings):
    """Invalidate rejected measured returns before Mapper and ray evidence."""
    from .pointcloud_filter import filter_observed_points
    from .pointcloud_preview import pointcloud_preview
    started = time.perf_counter()
    timings = {}
    points, camera_ids, pixels = [], [], []
    cameras = {}
    for camera, view in decoded["views"].items():
        depth = np.asarray(view["depth_left_m"])
        valid = (np.asarray(view["valid_left_mask"]) != 0) & np.isfinite(depth) & (depth > 0)
        cameras[camera] = {
            "image_shape_hw": list(depth.shape), "valid_depth_count": int(valid.sum()),
            "invalid_depth_count": int(valid.size - valid.sum()),
            "depth_range_m": [float(depth[valid].min()), float(depth[valid].max())] if valid.any() else None,
            "depth_sha256": hashlib.sha256(np.ascontiguousarray(depth).tobytes()).hexdigest(),
            "intrinsics": np.asarray(view["left_intrinsics"]).tolist(),
            "T_world_camera": np.asarray(view["T_base_left_camera"]).tolist(),
            "stereo_sequence": view["stereo_sequence"],
            "captured_sensor": decoded.get("camera_capture_metadata", {}).get(camera),
        }
        rows, columns = np.where(valid)
        intrinsics = np.asarray(view["left_intrinsics"])
        z = depth[rows, columns]
        local = np.column_stack(((columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
                                 (rows - intrinsics[1, 2]) * z / intrinsics[1, 1], z))
        matrix = np.asarray(view["T_base_left_camera"])
        points.append(local @ matrix[:3, :3].T + matrix[:3, 3])
        camera_ids.extend([camera] * len(rows))
        pixels.append(np.column_stack((rows, columns)))
    # Keep full-resolution measurements as arrays; only the bounded display
    # sample needs Python lists for serialization.
    points = np.concatenate(points, axis=0) if points else np.empty((0, 3))
    pixels = np.concatenate(pixels, axis=0) if pixels else np.empty((0, 2), dtype=int)
    timings["depth_projection_sources_wall"] = (time.perf_counter() - started) * 1000
    stage = time.perf_counter()
    result = filter_observed_points(points, {"camera_ids": camera_ids, "pixels_rc": pixels}, settings=settings)
    timings["optional_filter_wall"] = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    preview = pointcloud_preview(points, ~result["rejected_mask"],
                                 result.get("cluster_labels"))
    indices = preview["original_point_indices"]
    source_pixels = []
    for index in indices:
        captured = cameras[camera_ids[index]]["captured_sensor"] or {}
        offset = captured.get("letterbox_offset_rc", [0, 0])
        source_pixels.append([int(pixels[index, axis]) - offset[axis] for axis in (0, 1)])
    preview.update(camera_ids=[camera_ids[index] for index in indices],
                   pixels_rc=source_pixels, cameras=cameras,
                   pixel_coordinate_frame="original_sensor_when_capture_metadata_present_else_mapper_image",
                   stage="registered_depth_before_robot_semantic_and_workspace_removal")
    timings["raw_pointcloud_preview_wall"] = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    for index in np.flatnonzero(result["rejected_mask"]):
        camera = camera_ids[index]
        row, column = pixels[index]
        decoded["views"][camera]["depth_left_m"][row, column] = 0
        decoded["views"][camera]["valid_left_mask"][row, column] = 0
    timings["rejected_depth_apply_wall"] = (time.perf_counter() - stage) * 1000
    timings["filter_total_wall"] = (time.perf_counter() - started) * 1000
    return {"enabled": bool(settings.get("enabled", False)), "counts": result["counts"],
            "rejected_depth_policy": "invalid_return_no_free_space_carving",
            "dbscan_preview": bool(settings.get("dbscan_preview", False)), "preview": preview,
            "timings_ms": timings}


def create_feedback_segmenter(*, robot_config_path, urdf_path, asset_root_path, feedback):
    """Create one articulated public model; driver feedback is a per-capture input."""
    import yaml

    from .raw_robot_depth import RawRobotDepthBoundary

    original = yaml.safe_load(Path(robot_config_path).read_bytes())
    return RawRobotDepthBoundary(original, urdf_path=urdf_path, asset_root_path=asset_root_path, feedback=feedback)


def registered_source_masks(view, raw_depth):
    """Project actual depth returns into RGB masks using measured registration."""
    rows, columns = np.indices(raw_depth.shape)
    depth_intrinsics = np.asarray(view["left_intrinsics"], dtype=float)
    points = np.stack(((columns - depth_intrinsics[0, 2]) * raw_depth / depth_intrinsics[0, 0],
                      (rows - depth_intrinsics[1, 2]) * raw_depth / depth_intrinsics[1, 1], raw_depth), axis=-1)
    transform = np.asarray(view.get("left_to_rgb", np.eye(4)), dtype=float)
    rgb_points = points @ transform[:3, :3].T + transform[:3, 3]
    intrinsics = np.asarray(view.get("rgb_intrinsics", depth_intrinsics), dtype=float)
    z = rgb_points[..., 2]
    projectable = (raw_depth > 0) & (z > 0)
    u, v = np.zeros(raw_depth.shape, dtype=int), np.zeros(raw_depth.shape, dtype=int)
    u[projectable] = np.rint(intrinsics[0, 0] * rgb_points[..., 0][projectable] / z[projectable] + intrinsics[0, 2]).astype(int)
    v[projectable] = np.rint(intrinsics[1, 1] * rgb_points[..., 1][projectable] / z[projectable] + intrinsics[1, 2]).astype(int)
    masks = {}
    for name, mask in view.get("instance_masks", {}).items():
        mask = np.asarray(mask, dtype=bool)
        inside = projectable & (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
        registered = np.zeros(raw_depth.shape, dtype=bool)
        registered[inside] = mask[v[inside], u[inside]]
        masks[name] = registered
    return masks


class ObservedFreeSpaceGuard:
    """Certify full sphere footprints before the first registered depth return."""
    def __init__(self, views, filtered_depths, scene_token, *, uncertainty_m=.01, registered_source_limits=None):
        self.scene_token = scene_token
        self.uncertainty_m = uncertainty_m
        self.capture_spheres = np.empty((0, 4))
        self.capture_joint_inputs = {}
        self.selected_surface_world_m = []
        self.support_plane = None
        self.selected_instance_id = None
        self.observation_id = None
        self.generation = None
        self.measured_table_task_scope = None
        self.source_contact_mask_policy = ("current_registered_measured_surface_support" if registered_source_limits is not None
                                           else "registered_rgb_mask_without_surface_limit")
        self.views = []
        for camera, view in views.items():
            raw = np.asarray(view["depth_left_m"], dtype=float)
            filtered = np.asarray(filtered_depths[camera], dtype=float)
            measured_raw = np.where((np.asarray(view["valid_left_mask"]) != 0) & np.isfinite(raw) & (raw > 0), raw, 0)
            retained_depth = np.where(np.isfinite(filtered) & (filtered > 0), np.minimum(measured_raw, filtered), measured_raw)
            masks = registered_source_masks(view, measured_raw)
            if registered_source_limits is not None:
                limits = registered_source_limits.get(camera, {})
                for name, mask in masks.items():
                    limit = limits.get(name)
                    if limit is None:
                        masks[name] = np.zeros_like(mask)
                    else:
                        limit = np.asarray(limit)
                        if limit.shape != raw.shape or limit.dtype != np.bool_:
                            raise ValueError("source contact limit must match current depth as a boolean image")
                        masks[name] = mask & limit
            self.views.append((retained_depth,
                np.asarray(view["left_intrinsics"], dtype=float), np.linalg.inv(view["T_base_left_camera"]),
                measured_raw, masks))

    def set_measured_table_task_scope(self, table_model, views, filtered_depths):
        """Bound task observation checks by actual pre-exclusion environment returns."""
        from scipy.spatial import cKDTree

        self.measured_table_task_scope = None
        if table_model is None or "pose" not in table_model or "dims" not in table_model:
            return
        center = np.asarray(table_model["pose"][:3], dtype=float)
        dimensions = np.asarray(table_model["dims"], dtype=float)
        if center.shape != (3,) or dimensions.shape != (3,) or not np.isfinite([center, dimensions]).all() or np.any(dimensions <= 0):
            return
        lower, upper = center[:2] - dimensions[:2] / 2, center[:2] + dimensions[:2] / 2
        nonrobot_heights, selected_raw_points = [], []
        for camera, view in views.items():
            raw = np.asarray(view["depth_left_m"], dtype=float)
            filtered = np.asarray(filtered_depths[camera], dtype=float)
            valid = (np.asarray(view["valid_left_mask"]) != 0) & np.isfinite(raw) & (raw > 0)
            rows, columns = np.indices(raw.shape)
            intrinsics = np.asarray(view["left_intrinsics"])
            local = np.stack(((columns - intrinsics[0, 2]) * raw / intrinsics[0, 0],
                              (rows - intrinsics[1, 2]) * raw / intrinsics[1, 1], raw), axis=-1)
            matrix = np.asarray(view["T_base_left_camera"])
            world = local @ matrix[:3, :3].T + matrix[:3, 3]
            inside = valid & np.all(world[..., :2] >= lower, axis=-1) & np.all(world[..., :2] <= upper, axis=-1)
            nonrobot = inside & np.isfinite(filtered) & (filtered > 0)
            nonrobot_heights.extend(world[..., 2][nonrobot].tolist())
            source_mask = registered_source_masks(view, np.where(valid, raw, 0)).get(self.selected_instance_id)
            if source_mask is not None:
                selected_raw_points.extend(world[inside & source_mask].tolist())
        selected = np.asarray(self.selected_surface_world_m, dtype=float)
        matched = np.empty((0, 3))
        if len(selected_raw_points) and selected.ndim == 2 and selected.shape[1] == 3 and np.isfinite(selected).all():
            raw_selected = np.asarray(selected_raw_points)
            inside = np.all(selected[:, :2] >= lower, axis=-1) & np.all(selected[:, :2] <= upper, axis=-1)
            matches = cKDTree(raw_selected).query(selected)[0] <= .01
            matched = selected[inside & matches]
        # A redundant cloud can contain centroids computed before a return was
        # rejected. Only currently valid registered rays may raise this bound.
        heights = np.asarray([*nonrobot_heights, *np.asarray(selected_raw_points).reshape(-1, 3)[:, 2].tolist()])
        heights = heights[np.isfinite(heights)]
        if not len(heights):
            return
        maximum = float(heights.max())
        self.measured_table_task_scope = {
            "source": "pre_semantic_registered_nonrobot_and_selected_measured_returns",
            "table_xy_min_m": lower.tolist(), "table_xy_max_m": upper.tolist(),
            "measured_max_z_m": maximum, "measurement_margin_m": self.uncertainty_m,
            "upper_z_m": maximum + self.uncertainty_m,
            "nonrobot_return_count": len(nonrobot_heights),
            "selected_raw_return_count": len(selected_raw_points),
            "matched_current_selected_surface_count": len(matched),
            "selected_surface_matching_tolerance_m": .01,
        }

    def held_source_certificate(self, binding, capture_points_world_m):
        """Tie held contact intent to this capture without requiring unseen faces."""
        from scipy.spatial import cKDTree

        if binding is None:
            return False, "no_held_binding"
        if (binding.state != "held" or binding.scene_token != self.scene_token
                or binding.instance_id != self.selected_instance_id
                or binding.observation_id != self.observation_id or binding.generation != self.generation):
            return False, "held_binding_capture_identity_differs"
        observed = np.asarray(self.selected_surface_world_m, dtype=float)
        captured = np.asarray(capture_points_world_m, dtype=float)
        if (observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 2
                or captured.shape != (len(binding.points_tool_m), 3)
                or not np.isfinite(observed).all() or not np.isfinite(captured).all()):
            return False, "held_binding_capture_points_unavailable"
        tolerance = .01 + binding.uncertainty_m
        # The union can include previously observed faces that are occluded now.
        # Current visible points must match that union, and the later pixel test
        # still limits every permitted sphere to actual visible source returns.
        distance = cKDTree(captured).query(observed)[0]
        matched = distance <= tolerance
        if np.sum(matched) < 2 or np.mean(matched) < .8:
            return False, "held_binding_first_command_fk_differs_from_capture"
        return True, "current_measured_source_matches_held_union"

    def check(self, spheres, *, stationary_indices=(), captured_geometry_mask=None, source_indices=(),
              source_contact_samples=None, held_binding=None, held_capture_points_world_m=None):
        spheres = np.asarray(spheres, dtype=float)
        if spheres.ndim != 3 or spheres.shape[2] != 4 or not np.isfinite(spheres).all() or np.any(spheres[..., 3] <= 0):
            raise ValueError("observed-free guard requires finite [samples,spheres,4] geometry")
        covered = np.zeros(spheres.shape[:2], dtype=bool)
        outside_scope = np.zeros(spheres.shape[:2], dtype=bool)
        outside_above = np.zeros(spheres.shape[:2], dtype=bool)
        if self.measured_table_task_scope is not None:
            task = self.measured_table_task_scope
            lower, upper = np.asarray(task["table_xy_min_m"]), np.asarray(task["table_xy_max_m"])
            intersects_xy = np.all(spheres[..., :2] + spheres[..., 3, None] >= lower, axis=-1) & np.all(spheres[..., :2] - spheres[..., 3, None] <= upper, axis=-1)
            outside_above = spheres[..., 2] - spheres[..., 3] > task["upper_z_m"]
            outside_scope = ~intersects_xy | outside_above
        elif self.support_plane is not None:
            polygon = np.asarray(self.support_plane["observed_polygon_xy_m"], dtype=float)
            edges = np.roll(polygon, -1, axis=0) - polygon
            relative = spheres[..., None, :2] - polygon
            distance = (edges[:, 0] * relative[..., 1] - edges[:, 1] * relative[..., 0]) / np.linalg.norm(edges, axis=1)
            intersects_xy = np.all(distance >= -spheres[..., 3, None], axis=-1)
            normal = np.asarray(self.support_plane["normal_xyz"])
            signed = spheres[..., :3] @ normal + self.support_plane["contact_offset_m"]
            outside_scope = ~(intersects_xy & (signed + spheres[..., 3] >= 0))
        enabled_views = any(np.any(raw > 0) for _depth, _intrinsics, _inverse, raw, _masks in self.views)
        projectable = np.zeros(spheres.shape[:2], dtype=bool)
        fully_projectable = np.zeros(spheres.shape[:2], dtype=bool)
        for depth, intrinsics, inverse, raw, _masks in self.views:
            if not np.any(raw > 0):
                continue  # A disabled view cannot define observation scope.
            centers = spheres[..., :3] @ inverse[:3, :3].T + inverse[:3, 3]
            x, y, z = centers[..., 0], centers[..., 1], centers[..., 2]
            radius = spheres[..., 3]
            positive = z > radius + self.uncertainty_m
            denominator = np.where(positive, z * z - radius * radius, 1.)
            horizontal = radius * np.sqrt(np.maximum(0., z * z + x * x - radius * radius))
            vertical = radius * np.sqrt(np.maximum(0., z * z + y * y - radius * radius))
            left = intrinsics[0, 0] * (x * z - horizontal) / denominator + intrinsics[0, 2]
            right = intrinsics[0, 0] * (x * z + horizontal) / denominator + intrinsics[0, 2]
            top = intrinsics[1, 1] * (y * z - vertical) / denominator + intrinsics[1, 2]
            bottom = intrinsics[1, 1] * (y * z + vertical) / denominator + intrinsics[1, 2]
            fully_projectable |= positive & (left >= 0) & (top >= 0) & (right <= depth.shape[1] - 1) & (bottom <= depth.shape[0] - 1)
            projectable |= positive & (right >= 0) & (bottom >= 0) & (left <= depth.shape[1] - 1) & (top <= depth.shape[0] - 1)
        if enabled_views:
            outside_scope |= ~projectable
        covered[:, list(stationary_indices)] = True
        if captured_geometry_mask is not None:
            certificate = np.asarray(captured_geometry_mask, dtype=bool)
            if certificate.shape != covered.shape:
                raise ValueError("captured rigid geometry certificate shape differs")
            covered |= certificate
        captured_count = int(covered.sum())
        source_pixel_count = 0
        contact_intent_spheres = np.zeros(spheres.shape[:2], dtype=bool)
        held_contact_spheres = np.zeros(spheres.shape[:2], dtype=bool)
        partial_certificates = np.zeros(spheres.shape[:2], dtype=bool)
        held_valid, held_reason = self.held_source_certificate(held_binding, held_capture_points_world_m)
        for depth, intrinsics, inverse, raw, masks in self.views:
            for sample, sphere_index in zip(*np.where(~covered & ~outside_scope), strict=True):
                sphere = spheres[sample, sphere_index]
                x, y, z = inverse[:3, :3] @ sphere[:3] + inverse[:3, 3]
                radius = sphere[3]
                if z <= radius + self.uncertainty_m:
                    continue
                denominator = z * z - radius * radius
                horizontal = (x * z + np.array([-1, 1]) * radius * np.sqrt(z * z + x * x - radius * radius)) / denominator
                vertical = (y * z + np.array([-1, 1]) * radius * np.sqrt(z * z + y * y - radius * radius)) / denominator
                left, right = intrinsics[0, 0] * horizontal + intrinsics[0, 2]
                top, bottom = intrinsics[1, 1] * vertical + intrinsics[1, 2]
                projected = (left, top, right, bottom)
                left, top = int(np.floor(left)), int(np.floor(top))
                right, bottom = int(np.ceil(right)), int(np.ceil(bottom))
                if right < 0 or bottom < 0 or left >= depth.shape[1] or top >= depth.shape[0]:
                    continue
                partial = left < 0 or top < 0 or right >= depth.shape[1] or bottom >= depth.shape[0]
                if partial and fully_projectable[sample, sphere_index]:
                    continue  # An available full view must certify the whole sphere.
                left, top = max(0, left), max(0, top)
                right, bottom = min(depth.shape[1] - 1, right), min(depth.shape[0] - 1, bottom)
                footprint = depth[top:bottom + 1, left:right + 1]
                certified = footprint >= z + radius + self.uncertainty_m
                covered[sample, sphere_index] = bool(np.all(certified))
                partial_certificates[sample, sphere_index] = partial and covered[sample, sphere_index]
                contact_requested = (source_contact_samples is not None and source_contact_samples[sample])
                if (not covered[sample, sphere_index] and sphere_index in source_indices
                        and (contact_requested or held_valid)
                        and self.selected_instance_id in masks):
                    owned = masks[self.selected_instance_id][top:bottom + 1, left:right + 1]
                    source_depth = raw[top:bottom + 1, left:right + 1]
                    rows, columns = np.mgrid[top:bottom + 1, left:right + 1]
                    points = np.stack(((columns - intrinsics[0, 2]) * source_depth / intrinsics[0, 0],
                        (rows - intrinsics[1, 2]) * source_depth / intrinsics[1, 1], source_depth), axis=-1)
                    pixel_margin = source_depth * np.hypot(1 / intrinsics[0, 0], 1 / intrinsics[1, 1]) / 2
                    intersects = np.linalg.norm(points - [x, y, z], axis=-1) <= radius + self.uncertainty_m + pixel_margin
                    measured_source = owned & (source_depth > 0) & intersects
                    # floor/ceil includes pixel cells entirely outside the actual
                    # projected footprint; they do not carry a sphere ray.
                    required = ((columns + .5 >= projected[0]) & (columns - .5 <= projected[2])
                                & (rows + .5 >= projected[1]) & (rows - .5 <= projected[3]))
                    if np.all(certified | measured_source | ~required):
                        # Only a measured source return intersecting this sphere
                        # can authorize contact. Behind-source space stays unknown.
                        covered[sample, sphere_index] = True
                        contact_intent_spheres[sample, sphere_index] = True
                        held_contact_spheres[sample, sphere_index] = held_valid
                        partial_certificates[sample, sphere_index] = partial
                        source_pixel_count += int(np.sum(measured_source & ~certified))
        unknown = np.argwhere(~covered & ~outside_scope)
        unknown_heights = None
        unknown_indices = {}
        if len(unknown):
            geometry = spheres[unknown[:, 0], unknown[:, 1]]
            z = geometry[:, 2]
            edges = np.arange(np.floor(z.min() / .05) * .05, np.ceil(z.max() / .05) * .05 + .05001, .05)
            histogram, edges = np.histogram(z, bins=edges)
            unknown_heights = {"center_quantiles_m": np.quantile(z, [0, .25, .5, .75, 1]).tolist(),
                               "minimum_sphere_bottom_m": float(np.min(z - geometry[:, 3])),
                               "maximum_sphere_top_m": float(np.max(z + geometry[:, 3])),
                               "histogram_edges_m": edges.tolist(), "histogram_counts": histogram.tolist()}
            indices, counts = np.unique(unknown[:, 1], return_counts=True)
            unknown_indices = {int(index): int(count) for index, count in zip(indices, counts, strict=True)}
        proof = {"policy": "registered_depth_visible_sphere_footprint",
                 "observation_scope": ("measured_tabletop_height_intersect_positive_camera_frusta" if self.measured_table_task_scope is not None else ("measured_tabletop_intersect_positive_camera_frusta" if self.support_plane is not None else "positive_camera_frusta")) if enabled_views else "all_spheres_without_enabled_capture",
                 "measured_table_task_scope": self.measured_table_task_scope,
                 "outside_above_measured_task_sphere_count": int(outside_above.sum()),
                 "outside_observation_scope_sphere_count": int(outside_scope.sum()),
                 "outside_observation_scope_status": "unknown_not_certified_free",
                 "partial_observation_scope_sphere_count": int(np.sum(projectable & ~fully_projectable & ~outside_scope)),
                 "partial_footprint_certificate_sphere_count": int(partial_certificates.sum()),
                 "partial_footprint_outside_status": "unknown_not_certified_free",
                 "camera_footprint_policy": "prefer_full_footprint_else_check_clipped_visible_portion",
                 "robot_removed_depth_policy": "retain_free_interval_before_raw_first_return",
                 "collision_scope": "all_spheres_all_measured_obstacles", "scene_token": self.scene_token,
                 "checked_sphere_count": int(covered.size), "unobserved_sphere_count": len(unknown),
                 "unobserved_height_summary": unknown_heights,
                 "unobserved_sphere_index_counts": unknown_indices,
                 "first_unobserved_sample_index": None if len(unknown) == 0 else int(unknown[0, 0]),
                 "first_unobserved_sphere_index": None if len(unknown) == 0 else int(unknown[0, 1]),
                 "depth_uncertainty_m": self.uncertainty_m,
                 "stationary_captured_sphere_count": len(stationary_indices),
                 "captured_occupied_volume_sphere_count": captured_count,
                 "captured_configuration_sphere_sample_count": captured_count,
                 "capture_exemption_policy": "unchanged_capture_configuration",
                 "owned_source_occupied_pixel_count": source_pixel_count,
                 "source_occupied_policy": "selected_source_measured_pixel_contact_intent",
                 "source_contact_mask_policy": self.source_contact_mask_policy,
                 "source_contact_intent_sphere_count": int(contact_intent_spheres.sum()),
                 "held_source_contact_sphere_count": int(held_contact_spheres.sum()),
                 "held_source_binding_capture_valid": held_valid,
                 "held_source_binding_capture_reason": held_reason,
                 "held_source_contact_policy": "current_binding_measured_return_intersects_sphere",
                 "source_contact_intent_certifies_free_space": False,
                 "robot_capture_volume_exemption": False,
                 "robot_capture_configuration_exemption": bool(captured_count), "captured_volume_growth_allowed": False}
        if len(unknown):
            proof["first_unobserved_sphere_world_m"] = spheres[tuple(unknown[0])].tolist()
            proof["first_unobserved_view_diagnostics"] = self.view_diagnostics(spheres[tuple(unknown[0])])
        return (covered | outside_scope).all(axis=1), proof

    def view_diagnostics(self, sphere):
        result = []
        for index, (depth, intrinsics, inverse, _raw, _masks) in enumerate(self.views, 1):
            x, y, z = inverse[:3, :3] @ sphere[:3] + inverse[:3, 3]
            radius = sphere[3]
            item = {"camera_id": str(index), "sphere_center_camera_m": [float(x), float(y), float(z)],
                    "radius_m": float(radius), "required_free_depth_m": float(z + radius + self.uncertainty_m)}
            if z <= radius + self.uncertainty_m:
                item["reason"] = "sphere_intersects_camera_plane_or_is_behind_camera"
            else:
                denominator = z * z - radius * radius
                h = (x * z + np.array([-1, 1]) * radius * np.sqrt(z*z + x*x - radius*radius)) / denominator
                v = (y * z + np.array([-1, 1]) * radius * np.sqrt(z*z + y*y - radius*radius)) / denominator
                left, right = np.floor(intrinsics[0, 0]*h[0] + intrinsics[0, 2]), np.ceil(intrinsics[0, 0]*h[1] + intrinsics[0, 2])
                top, bottom = np.floor(intrinsics[1, 1]*v[0] + intrinsics[1, 2]), np.ceil(intrinsics[1, 1]*v[1] + intrinsics[1, 2])
                item["projected_bounds_pixel"] = [int(left), int(top), int(right), int(bottom)]
                if right < 0 or bottom < 0 or left >= depth.shape[1] or top >= depth.shape[0]:
                    item["reason"] = "sphere_footprint_outside_registered_field_of_view"
                else:
                    clipped = [max(0, int(left)), max(0, int(top)), min(depth.shape[1] - 1, int(right)), min(depth.shape[0] - 1, int(bottom))]
                    item["clipped_projected_bounds_pixel"] = clipped
                    item["partial_footprint_outside_status"] = "unknown_not_certified_free" if clipped != item["projected_bounds_pixel"] else None
                    footprint = depth[clipped[1]:clipped[3]+1, clipped[0]:clipped[2]+1]
                    item["invalid_or_robot_masked_pixel_count"] = int(np.sum(footprint <= 0))
                    item["occluded_pixel_count"] = int(np.sum((footprint > 0) & (footprint < z + radius + self.uncertainty_m)))
                    item["reason"] = "invalid_or_robot_masked_pixels" if item["invalid_or_robot_masked_pixel_count"] else "first_return_occludes_sphere"
            result.append(item)
        return result


class CaptureSegmenter:
    """Retain robot-masked depth before semantic removal changes map layers."""
    def __init__(self, boundary):
        self.boundary = boundary
        self.filtered_depths = {}
        self.capture = None
        self.source_restoration = {}
        self.source_limits = {}
        self.timings_ms = {}

    def begin_capture(self, decoded):
        self.filtered_depths.clear()
        self.source_restoration.clear()
        self.source_limits.clear()
        self.timings_ms.clear()
        self.capture = decoded

    def __getattr__(self, name):
        return getattr(self.boundary, name)

    def prepare_ffs_observation(self, request):
        started = time.perf_counter()
        prepared = self.boundary.prepare_ffs_observation(request)
        self.timings_ms["robot_depth_prepare_wall"] = (
            self.timings_ms.get("robot_depth_prepare_wall", 0.) + (time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        prepared = self.restore_selected_returns(prepared)
        self.timings_ms["selected_return_restore_wall"] = (
            self.timings_ms.get("selected_return_restore_wall", 0.) + (time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        self.filtered_depths[request.camera_id] = prepared.filtered_depth_m.detach().cpu().numpy().squeeze().copy()
        self.timings_ms["filtered_depth_download_wall"] = (
            self.timings_ms.get("filtered_depth_download_wall", 0.) + (time.perf_counter() - started) * 1000)
        return prepared

    def restore_selected_returns(self, prepared, *, camera_observation_type=None):
        from scipy.spatial import cKDTree

        capture = self.capture
        if capture is None:
            return prepared
        view = capture["views"].get(prepared.camera_id)
        if view is None or view["stereo_sequence"] != prepared.stereo_sequence:
            return prepared
        intrinsics = prepared.intrinsics_tensor.detach().cpu().numpy().reshape(3, 3)
        matrix = prepared.pose_matrix_tensor.detach().cpu().numpy().reshape(4, 4)
        if (prepared.filtered_depth_m.shape[-2:] != np.asarray(view["depth_left_m"]).shape
                or not np.array_equal(intrinsics, np.asarray(view["left_intrinsics"], dtype=intrinsics.dtype))
                or not np.array_equal(matrix, np.asarray(view["T_base_left_camera"], dtype=matrix.dtype))):
            return prepared
        selected = capture["confirmed_instance_id"]
        surface = np.asarray(capture.get("selected_surface_world_m", []), dtype=float)
        if surface.ndim != 2 or surface.shape[1] != 3 or not len(surface) or not np.isfinite(surface).all():
            return prepared
        raw = np.asarray(view["depth_left_m"], dtype=float)
        valid = (np.asarray(view["valid_left_mask"]) != 0) & np.isfinite(raw) & (raw > 0)
        raw = np.where(valid, raw, 0)
        mask = registered_source_masks(view, raw).get(selected)
        if mask is None:
            return prepared
        rows, columns = np.where(mask & valid)
        intrinsics = np.asarray(view["left_intrinsics"])
        z = raw[rows, columns]
        local = np.column_stack(((columns-intrinsics[0, 2])*z/intrinsics[0, 0],
                                 (rows-intrinsics[1, 2])*z/intrinsics[1, 1], z))
        matrix = np.asarray(view["T_base_left_camera"])
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        # Perception selects representative returns in 1mm occupied voxels.
        # The voxel diameter covers other returns in the same occupied cell.
        tolerance = np.sqrt(3) * .001
        supported = cKDTree(surface).query(world)[0] <= tolerance
        limit = np.zeros(raw.shape, dtype=bool)
        limit[rows[supported], columns[supported]] = True
        self.source_limits[prepared.camera_id] = {selected: limit}
        filtered = prepared.filtered_depth_m.detach().cpu().numpy().reshape(raw.shape).copy()
        restore = limit & (filtered <= 0)
        filtered[restore] = raw[restore]
        self.source_restoration[prepared.camera_id] = {
            "source": "current_registered_selected_pixels_supported_by_measured_surface",
            "observation_id": capture["observation_set_id"], "generation": capture["selection_generation"],
            "instance_id": selected, "stereo_sequence": prepared.stereo_sequence,
            "registered_selected_point_count": len(rows), "supported_point_count": int(supported.sum()),
            "unsupported_point_count": int((~supported).sum()), "restored_point_count": int(restore.sum()),
            "removed_robot_point_count_before_source_recovery": prepared.removed_point_count,
            "surface_support_tolerance_m": float(tolerance), "optional_rejected_returns_restored": False}
        if not np.any(restore):
            return prepared
        if camera_observation_type is None:
            from curobo.types import CameraObservation
            camera_observation_type = CameraObservation
        tensor = prepared.filtered_depth_m.new_tensor(filtered.reshape(prepared.filtered_depth_m.shape))
        original = prepared.camera_observation
        observation = camera_observation_type(depth_image=tensor, intrinsics=original.intrinsics,
                                               pose=original.pose, depth_to_meter=1.)
        return replace(prepared, filtered_depth_m=tensor, camera_observation=observation,
                       base_pointcloud_m=observation.get_pointcloud(project_to_pose=True),
                       removed_point_count=max(0, prepared.removed_point_count-int(restore.sum())))

    def remove_semantic_instances(self, prepared, projection, *, open3d_runtime=None):
        from .semantic_depth import partition_semantic_depth

        started = time.perf_counter()
        result = partition_semantic_depth(prepared, projection,
            registered_mask_limits=self.source_limits.get(prepared.camera_id, {}))
        self.timings_ms["semantic_partition_wall"] = (
            self.timings_ms.get("semantic_partition_wall", 0.) + (time.perf_counter() - started) * 1000)
        return result

    def filter_isolated_surfaces(self, observations):
        return tuple(observations)


def mapping_builder_payload(decoded):
    """Use native head-anchored background mapping for incomplete observation views."""
    views = decoded["views"]
    if decoded.get('demo_mapping'):
        if tuple(views) != ('1', '2', '3'):
            raise ValueError('demo_mapping requires all three registered views')
        return decoded
    if decoded.get('scene_policy') == 'head_all_instances':
        if tuple(views) != ('3',) or not np.any(np.asarray(views['3']['valid_left_mask'])):
            raise ValueError('head_semantic_map_requires_measured_head_depth')
        return decoded
    if decoded.get('scene_policy') == 'all_instances':
        if tuple(views) != ('1', '2', '3'):
            raise ValueError('scene_update_requires_three_cameras')
        return decoded
    if decoded.get('mapping_mode') != 'background' and not views['3']['instance_masks']:
        # The project semantic builder keeps genuine empty individual views;
        # it still requires measured source and table evidence across capture.
        return decoded
    active = {camera: view for camera, view in views.items()
              if np.any((np.asarray(view["valid_left_mask"]) != 0) & (np.asarray(view["depth_left_m"]) > 0))}
    if len(active) == len(views):
        return decoded
    if not decoded.get("observation_only") or "3" not in active:
        raise ValueError("blank registered views are supported only for head-anchored observation maps")
    # The SDK's background mode explicitly supports a head-only camera set.
    # Clearing semantic removal retains source and non-source raw returns in
    # the background ESDF; no semantic instance may be excluded in this mode.
    return decoded | {"mapping_mode": "background", "confirmed_instance_id": "",
                      "views": {camera: view | {"instance_masks": {}} for camera, view in active.items()}}


class PerceptionMapper:
    def __init__(self, segmenter_boundary):
        from curobo.scene import Scene
        from tianji_curobo.volumetric_mapper import CuroboVolumetricMapperBoundary

        from .instance_mapping import InstanceMapper
        from .semantic_builder import ProjectSemanticBuilder

        self.segmenter_boundary = segmenter_boundary
        self.capture_segmenter = CaptureSegmenter(segmenter_boundary)
        self.mapper = CuroboVolumetricMapperBoundary(robot_segmenter_boundary=self.capture_segmenter,
                                                     voxel_size_m=.02, esdf_voxel_size_m=.02)
        self.builder = ProjectSemanticBuilder(robot_segmenter_boundary=self.capture_segmenter,
                       mapper_boundary=self.mapper, scene_type=Scene, robot_spheres=segmenter_boundary.robot_spheres,
                       instance_mapper=InstanceMapper(self.capture_segmenter))
        self.diagnostics = {}

    def reset(self):
        """Invalidate measured table evidence as well as per-capture GPU state."""
        self.mapper.reset_capture()
        self.capture_segmenter.begin_capture(None)
        self.builder.table_cache.reset()
        self.diagnostics = {}

    def update_segmenter(self, segmenter_boundary, *, scene_type=None):
        """Replace measured robot geometry while retaining empty mapping allocations."""
        if scene_type is None:
            from curobo.scene import Scene
            scene_type = Scene
        self.mapper.reset_capture()
        self.capture_segmenter.begin_capture(None)
        self.segmenter_boundary = segmenter_boundary
        self.capture_segmenter.boundary = segmenter_boundary
        self.builder.scene_type = scene_type
        self.builder.robot_spheres = segmenter_boundary.robot_spheres

    def build(self, decoded):
        # Builder clears instance layers each time; raw background integration
        # resets its own mapper. Preserve the builder's proven-empty grid here.
        self.capture_segmenter.begin_capture(decoded)
        try:
            native_payload = mapping_builder_payload(decoded)
            candidate = self.builder.build(native_payload)
            for camera, view in decoded["views"].items():
                self.capture_segmenter.filtered_depths.setdefault(camera, np.zeros_like(view["depth_left_m"]))
            grids = list(self.builder.scene_voxel_grids)
            model = {name: dict(value) for name, value in candidate.scene_model.items()}
            model["voxel"] = dict(model["voxel"])
            # Capacity includes the measured background grid, even though the
            # frozen builder stores only instance grid metadata in scene_model.
            for grid in grids:
                model["voxel"].setdefault(grid.name, {"pose": list(grid.pose), "dims": list(grid.dims),
                    "voxel_size": float(grid.voxel_size)})
            self.diagnostics = {**self.builder.diagnostics, "grids": [{"name": grid.name,
                "dims_m": list(grid.dims), "voxel_size_m": float(grid.voxel_size),
                "feature_shape": list(grid.feature_tensor.shape),
                "voxel_count": int(grid.feature_tensor.numel())} for grid in grids]}
            return candidate, grids, model
        except Exception:
            self.mapper.reset_capture()
            self.capture_segmenter.begin_capture(None)
            raise

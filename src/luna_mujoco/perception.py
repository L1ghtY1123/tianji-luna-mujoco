"""SAM3 RGB masks and measured RGBD geometry for Decisions-assigned identities.

No simulator object identifiers, editor metadata, or generated primitive clouds
cross this boundary. Incomplete views remain measured partial surfaces.
"""

import colorsys
import hashlib
import os
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from sam3_service.image_client import Sam3Error, Sam3ImageClient
from sam3_service.wire import MODEL_IDENTITY as SAM3_MODEL_IDENTITY
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree

from .cameras import deproject_view
from .placement import measured_source_relationships, public_relationships
from .robot_surface import own_robot_surface_pixels
from .table_markers import detect_table_markers
from .visual_identity import VisualIdentity


class PerceptionError(RuntimeError):
    """Sensor, detector, geometry, or association lacks required evidence."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details


def _readonly(array):
    result = np.ascontiguousarray(array).copy()
    result.flags.writeable = False
    return result


def _validate_capture(capture):
    try:
        if set(capture["views"]) != {"1", "2", "3"}:
            raise PerceptionError("three registered RGBD views are required")
        calibration_ids = set()
        for camera_id, view in capture["views"].items():
            if (view["camera_id"] != camera_id or view["frame_id"] != capture["observation_id"]
                    or view["robot_generation"] != capture["generation"]
                    or not np.isfinite(view['capture_time']) or abs(view['capture_time'] - capture['time']) > .020001):
                raise PerceptionError("RGBD views must use the same capture identity and synchronized timestamps")
            if not view['calibration_id'] or view['geometry_revision'] != capture['geometry_revision']:
                raise PerceptionError('RGBD calibration revision mismatch')
            calibration_ids.add(view['geometry_revision'])
            rgb, depth, valid = view["rgb"], view["depth_m"], view["valid_depth"]
            intrinsics = view["intrinsics"]
            shape = (intrinsics["height"], intrinsics["width"])
            transform = np.asarray(view["T_world_camera"], dtype=float)
            if (rgb.dtype != np.uint8 or rgb.shape != shape + (3,) or depth.shape != shape or valid.shape != shape
                    or valid.dtype != bool or view.get("registered_to") != "rgb"
                    or transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform[3], [0, 0, 0, 1])
                    or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                    or min(intrinsics["fx"], intrinsics["fy"]) <= 0
                    or any(not np.isfinite(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))):
                raise PerceptionError("invalid registered RGBD calibration or array")
            for arm, feedback in view['arms'].items():
                if not np.allclose(view["robot_joints_deg"][arm], np.degrees(feedback["joints_rad"]), atol=1e-6):
                    raise PerceptionError("RGBD robot feedback does not match captured telemetry")
        if len(calibration_ids) != 1:
            raise PerceptionError("RGBD calibration identity mismatch")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise PerceptionError("invalid registered RGBD capture") from error


def _color(rgb_pixels):
    median = np.median(rgb_pixels, axis=0) / 255
    hue, saturation, value = colorsys.rgb_to_hsv(*median)
    # Dim saturated RGB still carries color; reserve black for near-black
    # pixels before applying neutral saturation and hue classification.
    if value < .08:
        name = "black"
    elif saturation < .18:
        name = "white" if value > .75 else "gray"
    elif hue < .04 or hue >= .94:
        name = "red"
    elif hue < .11:
        name = "orange"
    elif hue < .19:
        name = "yellow"
    elif hue < .46:
        name = "green"
    elif hue < .75:
        name = "blue"
    else:
        name = "unknown"
    return name, median.tolist()


def _voxel_points(points, resolution=.001):
    _, indices = np.unique(np.floor(points / resolution).astype(np.int64), axis=0, return_index=True)
    return np.asarray(points)[np.sort(indices)].astype(np.float32)


def measured_bin_reference(points_world_m, table, point_sources=None):
    """Locate a measured horizontal floor patch; no cavity, walls or pose fit.

    A horizontal patch needs measured higher bin context to distinguish its
    role from an isolated top rim. No complete interior or capacity is inferred.
    """
    points = np.asarray(points_world_m, dtype=float)
    if len(points) < 30:
        return None
    distances, indices = cKDTree(points).query(points, k=min(16, len(points)))
    neighbours = points[indices]
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centred, centred) / neighbours.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    horizontal = ((np.abs(eigenvectors[:, 2, 0]) >= np.cos(np.deg2rad(15.)))
                  & (eigenvalues[:, 1] > 1e-8)
                  & (eigenvalues[:, 0] <= .15 * eigenvalues[:, 1])
                  & (distances[:, -1] <= .03))
    outside_support_band = np.abs(points[:, 2] - table["surface_height_m"]) > max(.002, table.get("uncertainty_m", .003))
    candidates = np.flatnonzero(horizontal & outside_support_band)
    if len(candidates) < 30:
        return None
    # Use the lowest supported measured level, excluding thin top-rim samples.
    levels, counts = np.unique(np.round(points[candidates, 2] / .002), return_counts=True)
    supported_levels = levels[counts >= 30]
    if not len(supported_levels):
        return None
    level = supported_levels.min()
    selected = candidates[np.abs(points[candidates, 2] - level * .002) <= .003]
    floor = points[selected]
    interior_distances = np.linalg.norm(floor[:, :2] - floor[:, :2].mean(axis=0), axis=1)
    # Complementary views can leave an observation gap at the visible centre.
    # Choose a nearby actual sample with local plane density; never interpolate
    # into the gap. Higher bin context below still distinguishes a floor from
    # an isolated top rim, including a rim with a sparse central outlier.
    floor_distances = cKDTree(floor).query(floor, k=5)[0]
    support_radius = np.clip(2.5 * np.median(floor_distances[:, 1]), .005, .015)
    locally_supported = np.flatnonzero(floor_distances[:, -1] <= support_radius)
    if not len(locally_supported):
        return None
    if point_sources is not None:
        head_support = locally_supported[point_sources['camera_ids'][selected[locally_supported]] == '3']
        if len(head_support):
            locally_supported = head_support
    central_index = locally_supported[np.argmin(interior_distances[locally_supported])]
    reference_index = selected[central_index]
    position = points[reference_index].tolist()
    # A dense partial rim can also have central samples. Require a visible
    # higher wall/rim fragment from this same semantic instance; a horizontal
    # patch alone cannot establish its role as a floor. One fragment suffices,
    # and isolated high outliers do not supply the missing context.
    height_separation = max(.006, 2 * table.get("uncertainty_m", .003))
    higher_indices = np.flatnonzero(points[:, 2] > position[2] + height_separation)
    if len(higher_indices) < 10:
        return None
    higher_distances = cKDTree(points[higher_indices]).query(points[higher_indices], k=5)[0]
    supported_higher = higher_indices[higher_distances[:, -1] <= .015]
    if len(supported_higher) < 10:
        return None
    sources = {"point_count": len(selected), "point_indices": selected.tolist(),
               "higher_context_point_indices": supported_higher.tolist()}
    if point_sources is not None:
        sources["camera_ids"] = sorted(set(point_sources["camera_ids"][selected].tolist()))
        sources["reference_camera_id"] = str(point_sources["camera_ids"][reference_index])
        sources["reference_pixel_rc"] = point_sources["pixels_rc"][reference_index].tolist()
    from .measured_regions import supported_floor_rectangle

    region = supported_floor_rectangle(floor[:, :2])
    if region is None:
        return None
    from scipy.spatial import ConvexHull

    # Source relationships use the measured extent, independently of the
    # rectangle selected for the shared placement layout.
    observed_floor_hull = floor[ConvexHull(floor[:, :2]).vertices, :2].tolist()
    return {"position_world_m": position, "floor_z_m": position[2],
            'observed_floor_hull_world_xy_m': observed_floor_hull,
            "region_polygon_world_xy_m": region['polygon_world_xy_m'], 'region_measurement': region,
            "source": "sam3_bin_mask_and_measured_horizontal_floor_patch",
            "measurement_sources": sources}


def _estimate_table(points):
    if len(points) < 100:
        raise PerceptionError("table_unobservable")
    # Select the dominant horizontal level in measured workspace background.
    levels = np.round(points[:, 2] / .004).astype(np.int64)
    unique, counts = np.unique(levels, return_counts=True)
    level = unique[np.argmax(counts)]
    surface = points[np.abs(points[:, 2] - level * .004) <= .006]
    if len(surface) < 100:
        raise PerceptionError("table_unobservable")
    low, high = surface.min(axis=0), surface.max(axis=0)
    z = float(np.median(surface[:, 2]))
    return {"position": [float((low[0] + high[0]) / 2), float((low[1] + high[1]) / 2), z],
            "size": [float(high[0] - low[0]), float(high[1] - low[1]), .006],
            "surface_height_m": z, "uncertainty_m": max(.003, float(np.std(surface[:, 2])) * 3),
            "estimated": True, "source": "measured_horizontal_workspace_plane",
            "points_world_m": _readonly(_voxel_points(surface))}


def _estimate_object(group, table, *, observation_provenance=None, views=None):
    all_views = group
    masks = {}
    depth_counts = {}
    for entry in group:
        camera = entry['camera_id']
        if views is None:
            depth_counts[camera] = depth_counts.get(camera, 0) + len(entry['points'])
            continue
        mask = np.asarray(entry.get('silhouette_mask', entry['mask']), dtype=bool)
        masks[camera] = masks.get(camera, np.zeros_like(mask)) | mask
    for camera, mask in masks.items():
        view = views[camera]
        depth_counts[camera] = int(np.count_nonzero(
            mask & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)))
    selected_camera = '3'
    if depth_counts.get(selected_camera, 0) < 30:
        raise PerceptionError('instance_head_depth_insufficient')
    head = [entry for entry in group if entry['camera_id'] == selected_camera]
    group = head + [entry for entry in group if entry['camera_id'] in ('2', '1')]
    raw_points = np.vstack([entry["points"] for entry in group])
    _, indices = np.unique(np.floor(raw_points / .001).astype(np.int64), axis=0, return_index=True)
    indices = np.sort(indices)
    points = raw_points[indices].astype(np.float32)
    camera_ids = np.concatenate([np.full(len(entry["points"]), entry["camera_id"]) for entry in group])[indices]
    pixels_rc = np.vstack([entry.get("pixels_rc", np.full((len(entry["points"]), 2), -1)) for entry in group])[indices]
    if len(points) < 30:
        raise PerceptionError("instance_visible_depth_insufficient")
    kind = group[0]["kind"]
    assumptions = ["visible_surface_only"]
    if kind != "bin":
        table_band_m = max(.002, table.get("uncertainty_m", .003))
        object_samples = np.abs(points[:, 2] - table["surface_height_m"]) > table_band_m
        if not np.all(object_samples):
            points, camera_ids, pixels_rc = points[object_samples], camera_ids[object_samples], pixels_rc[object_samples]
            assumptions.append("table_plane_pixels_removed")
        if len(points) < 30:
            raise PerceptionError("instance_visible_depth_insufficient_above_support")
    pixels = np.vstack([entry["pixels"] for entry in head])
    color, median_rgb = _color(pixels)
    head_points = points[camera_ids == selected_camera]
    if len(head_points) < 30:
        raise PerceptionError('instance_head_depth_insufficient')
    centroid = head_points.mean(axis=0, dtype=np.float64).tolist()
    provenance = deepcopy(observation_provenance or {})
    point_sources = {"camera_ids": _readonly(camera_ids), "pixels_rc": _readonly(pixels_rc),
                     "observation_id": provenance.get("observation_id")}
    identifier = group[0].get('entity_id')
    if not isinstance(identifier, str) or not identifier or any(
            entry.get('entity_id') != identifier for entry in group):
        raise PerceptionError('visual_identity_entity_required')
    result = {"id": identifier, "kind": kind, "color": color,
              "visible_centroid_world": centroid, "position": list(centroid),
              "estimated": True, "confidence": float(min(entry["score"] for entry in group)),
              "uncertainty_m": .005, "source": "sam3_rgb_mask_and_registered_depth", "identity_source": "openai_decisions",
              "color_evidence": {"median_rgb": median_rgb, "source": "masked_rgb"},
              "points_world_m": _readonly(points), "point_sources": point_sources,
              "observation_provenance": provenance,
              "coverage": {"observed_surface_only": True, "camera_ids": list(dict.fromkeys(entry["camera_id"] for entry in group)),
                           "sam3_camera_ids": list(dict.fromkeys(entry["camera_id"] for entry in group
                                                                  if entry.get("sam3_detected", True))),
                           "point_count": len(points), "limitations": assumptions}}
    result['measurement_camera_selection'] = {
        'policy': 'head_primary_with_registered_wrist_surfaces',
        'selected_camera_id': selected_camera,
        'valid_depth_pixels_by_camera': depth_counts,
        'acquired_camera_ids': list(provenance.get('acquired_camera_ids', ())),
        'analyzed_camera_ids': list(provenance.get('analyzed_camera_ids', ())),
        'observation_id': provenance.get('observation_id'),
        'generation': provenance.get('generation'),
        'frame': 'world'}
    identity_evidence = [item for entry in all_views for item in entry.get('identity_evidence', [])]
    if identity_evidence:
        sources = {item['source'] for item in identity_evidence}
        result['identity_source'] = (next(iter(sources)) if len(sources) == 1 else
                                     'mixed_decisions_measured_container')
        result['identity_evidence'] = deepcopy(identity_evidence)
    geometry_filters = [entry['geometry_filter_evidence'] for entry in group
                        if 'geometry_filter_evidence' in entry]
    if geometry_filters:
        result['geometry_filter_evidence'] = deepcopy(geometry_filters)
    if group[0].get('prompt'):
        result['segmentation_prompt'] = group[0]['prompt']
    if group[0].get('visual_membership') is not None:
        result['visual_membership'] = group[0]['visual_membership']
    if kind == "bin":
        reference = measured_bin_reference(points, table, point_sources)
        if reference is not None:
            reference["observation_id"] = provenance.get("observation_id")
            result["placement_reference"] = reference
    return result


def _measured_surface_support(points, view, mask, tolerance_m=.003):
    """Check current contact points against four nearby measured RGBD samples."""
    transform = np.asarray(view["T_world_camera"], dtype=float)
    camera_points = (points - transform[:3, 3]) @ transform[:3, :3]
    intrinsics = view["intrinsics"]
    front = camera_points[:, 2] > 0
    columns = np.full(len(points), -100.)
    rows = columns.copy()
    columns[front] = intrinsics["fx"] * camera_points[front, 0] / camera_points[front, 2] + intrinsics["cx"]
    rows[front] = intrinsics["fy"] * camera_points[front, 1] / camera_points[front, 2] + intrinsics["cy"]
    height, width = view['depth_m'].shape
    inside = (front & (columns >= -.5) & (columns < width - .5)
              & (rows >= -.5) & (rows < height - .5))
    matched = np.zeros(len(points), bool)
    for pixel_rows in (np.floor(rows).astype(int), np.ceil(rows).astype(int)):
        for pixel_columns in (np.floor(columns).astype(int), np.ceil(columns).astype(int)):
            in_image = (inside & (pixel_rows >= 0) & (pixel_rows < height)
                        & (pixel_columns >= 0) & (pixel_columns < width))
            indices = np.flatnonzero(in_image)
            r, c = pixel_rows[indices], pixel_columns[indices]
            depth = view["depth_m"][r, c]
            valid = view["valid_depth"][r, c] & np.isfinite(depth) & (depth > 0)
            measured = np.column_stack(((c - intrinsics["cx"]) * depth / intrinsics["fx"],
                                        (r - intrinsics["cy"]) * depth / intrinsics["fy"], depth))
            distance = np.linalg.norm(measured - camera_points[indices], axis=1)
            matched[indices] |= valid & mask[r, c] & (distance <= tolerance_m)
    return matched


def _other_view_support(points, observation, identifier, source_camera):
    supported = np.zeros(len(points), bool)
    for camera, view in observation["views"].items():
        mask = view["instance_masks"].get(identifier)
        if camera != source_camera and mask is not None:
            supported |= _measured_surface_support(points, view, mask)
    return supported


def _remove_own_robot_surface(groups, views, capture, classifier):
    """Assign supported self returns to robot ownership without changing sensor data."""
    if classifier is None:
        return groups, {'status': 'not_applied', 'source': 'identity_bound_cad_tsdf_scene',
                        'removed_semantic_pixel_count': 0,
                        'raw_depth_policy': 'unchanged_actual_returns_no_free_space_claim'}
    images = {camera: dict(view, instance_masks={}) for camera, view in views.items()}
    union = {camera: np.zeros(view["depth_m"].shape, bool) for camera, view in views.items()}
    for index, group in enumerate(groups):
        for entry in group:
            camera = entry["camera_id"]
            images[camera]["instance_masks"][index] = entry["mask"]
            union[camera] |= entry["mask"]
    evidence = classifier(capture, union)
    masks = evidence.pop("masks")
    observation = {"views": images}
    rebuilt, removed_count, ambiguous_count, contact_count = [], 0, 0, 0
    for index, group in enumerate(groups):
        entries = []
        for entry in group:
            camera, view = entry["camera_id"], views[entry["camera_id"]]
            candidates = entry["mask"] & masks.get(camera, np.zeros(entry["mask"].shape, bool))
            pixels = np.column_stack(np.nonzero(candidates))
            points = deproject_view(view, candidates)
            supported = (_other_view_support(points, observation, index, camera)
                         if len(points) else np.zeros(0, bool))
            ambiguous_count += int(supported.sum())
            contact_pixels = {tuple(match["pixel_rc"]) for match in evidence.get("matches", {}).get(camera, [])
                              if match.get("contact_surface_ambiguous")}
            contact = np.array([tuple(pixel) in contact_pixels for pixel in pixels], dtype=bool)
            contact_count += int(contact.sum())
            # Finger pads intentionally contact the source. A model hit here
            # cannot distinguish object contact from the robot's own return.
            removed = pixels[~supported & ~contact]
            mask = entry["mask"].copy()
            mask[tuple(removed.T)] = False
            removed_count += len(removed)
            robot_mask = np.array(views[camera].get("robot_surface_mask", np.zeros(mask.shape, bool)), copy=True)
            robot_mask[tuple(removed.T)] = True
            views[camera]['robot_surface_mask'] = robot_mask
            valid = mask & view["valid_depth"] & np.isfinite(view["depth_m"]) & (view["depth_m"] > 0)
            entries.append(entry | {"mask": mask, "points": deproject_view(view, mask),
                "pixels": view["rgb"][valid], "pixels_rc": np.column_stack(np.nonzero(valid))})
        rebuilt.append([entry for entry in entries if len(entry["points"])] or entries)
    for view in views.values():
        if "robot_surface_mask" in view:
            view["robot_surface_mask"] = _readonly(view["robot_surface_mask"])
    evidence.update(removed_semantic_pixel_count=removed_count, retained_cross_view_contact_count=ambiguous_count,
                    retained_finger_pad_contact_count=contact_count,
                    raw_depth_policy="unchanged_actual_returns_no_free_space_claim")
    return rebuilt, evidence


def _source_geometry_interiors(groups, views):
    """Exclude silhouette boundaries and undersized islands; preserve raw depth."""
    rebuilt, removed_count = [], 0
    for group in groups:
        if group[0]["kind"] == "bin":
            rebuilt.append(group)
            continue
        entries = []
        for entry in group:
            view = views[entry["camera_id"]]
            # Erode only the RGB silhouette. Invalid depth or robot ownership
            # holes must not remove additional neighboring object samples.
            interior = binary_erosion(entry["silhouette_mask"], structure=np.ones((3, 3), dtype=bool))
            mask = entry["mask"] & interior
            # Apply the existing 30-pixel candidate minimum to each disconnected
            # component, so a large object cannot admit a separate tiny island.
            components, _ = label(mask, structure=np.ones((3, 3), dtype=bool))
            sizes = np.bincount(components.ravel())
            mask &= sizes[components] >= 30
            removed_count += int(np.count_nonzero(entry["mask"] & ~mask))
            valid = mask & view["valid_depth"] & np.isfinite(view["depth_m"]) & (view["depth_m"] > 0)
            entries.append(entry | {"mask": mask, "points": deproject_view(view, mask),
                "pixels": view["rgb"][valid], "pixels_rc": np.column_stack(np.nonzero(valid))})
        rebuilt.append([entry for entry in entries if len(entry["points"])] or entries)
    return rebuilt, removed_count


class Perception:
    """Detector injection bounds network I/O; all fusion uses actual captured data."""
    def __init__(self, detector=None, *, robot_surface_classifier=None, visual_identity=None, cad_resolver=None):
        self.detector = detector if detector is not None else Sam3ImageClient(os.getenv("LUNA_SAM3_URL", "http://127.0.0.1:8322"))
        self.visual_identity = visual_identity if visual_identity is not None else VisualIdentity.from_environment()
        self.segmentation_cache = OrderedDict()
        self.segmentation_evidence = []
        self.cad_resolver = cad_resolver
        self.robot_surface_classifier = (own_robot_surface_pixels if robot_surface_classifier is None
                                         else robot_surface_classifier)

    def table_robot_surface_classifier(self, capture, masks):
        if self.robot_surface_classifier is None:
            return {'status': 'not_applied', 'source': 'identity_bound_cad_tsdf_scene', 'masks': {}}
        if self.robot_surface_classifier is own_robot_surface_pixels:
            return self.robot_surface_classifier(capture, masks, mask_only=True)
        return self.robot_surface_classifier(capture, masks)

    def prepare_gripper_evidence(self, capture, arms, entities):
        """Annotate current wrist candidates without deciding holding or identity."""
        from .robot_skills.grasp_verification import WRIST_CAMERAS

        masks, attempts = {}, []
        queries = self._segmentation_queries(entities)
        for arm in arms:
            camera = WRIST_CAMERAS[arm]
            camera_masks = {}
            for query in queries:
                attempt = {'camera_id': camera, 'query': query['prompt']}
                try:
                    proposals, prompt = self._segment_query(capture, camera, query)
                except PerceptionError as error:
                    attempts.append({**attempt, 'status': 'unavailable', 'error': str(error)})
                    continue
                for proposal in proposals:
                    mask = np.asarray(proposal['mask'], dtype=bool)
                    if mask.any():
                        camera_masks[f'C{len(camera_masks)}'] = mask
                attempts.append({**attempt, 'used_query': prompt,
                                 'status': 'candidates' if proposals else 'no_match'})
            masks[camera] = camera_masks
        return {'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'visual_masks': masks, 'attempts': attempts}

    @staticmethod
    def _segmentation_queries(entities):
        """Gripper evidence uses the same complete-query deduplication rule as grounding."""
        from .candidate_observation import normalized_query

        queries = {}
        for entity in entities.values():
            query = entity.get('segmentation_query') or {}
            if query.get('scene_quote') and query.get('prompt'):
                key = normalized_query(query['prompt'])
                queries[key] = query
        return list(queries.values())

    def _segment_cached(self, capture, camera_id, prompt, fallback_for=None, *, bbox=None):
        view = capture['views'][camera_id]
        timestamp_us = round(capture['time'] * 1_000_000)
        key = (capture['generation'], capture['observation_id'], camera_id, prompt, tuple(bbox) if bbox is not None else None,
               view['calibration_id'], timestamp_us, hashlib.sha256(view['rgb'].tobytes()).digest())
        evidence = {'component': 'sam3', 'stage': 'segment_box' if bbox is not None else 'segment_text', 'bbox': bbox,
            'observation_id': capture['observation_id'], 'camera_id': camera_id,
            'prompt': prompt, 'cache_hit': key in self.segmentation_cache, 'status': 'pending',
            'unknowns': ['sam3_internal_filter_threshold', 'sam3_internal_rejection_reason']}
        if fallback_for is not None:
            evidence.update(fallback_for=fallback_for, fallback_reason='no_sam3_match')
        self.segmentation_evidence.append(evidence)
        if key not in self.segmentation_cache:
            try:
                self.segmentation_cache[key] = deepcopy(self.detector.segment(view['rgb'], prompt, timestamp_us, bbox=bbox))
            except (Sam3Error, PerceptionError) as error:
                evidence.update({**(error.details or {}), 'status': 'error', 'error': str(error)})
                raise PerceptionError(str(error), details={**evidence, **(error.details or {})}) from error
            if len(self.segmentation_cache) > 128:
                self.segmentation_cache.popitem(last=False)
        self.segmentation_cache.move_to_end(key)
        proposals = self.segmentation_cache[key]
        self._record_segmentation_attempt(evidence, view, proposals)
        return proposals

    def _segment_query(self, capture, camera, query):
        """Try one phrase at a time and stop at the first SAM3 masks."""
        primary = query['prompt']
        prompts = [primary, *query.get('aliases', [])]
        for prompt in dict.fromkeys(prompts):
            proposals = self._segment_cached(capture, camera, prompt,
                fallback_for=primary if prompt != primary else None, bbox=query.get('bbox'))
            if proposals:
                return proposals, prompt
        return [], primary

    def ground(self, capture, *, proposals, entities, reference, initial=True,
               proposal_observation_id=None, inventory=(), declared_objects, reobserved_hints=()):
        from .candidate_observation import collect_candidates
        from .core.grounding import GroundingResult
        from .scene_association import associate_scene

        _validate_capture(capture)
        self.segmentation_evidence = []
        observation = collect_candidates(capture, declared_objects, self._segment_query, inventory=inventory,
            proposal_observation_id=proposal_observation_id, reference=reference, reobserved_hints=reobserved_hints)
        initial_supports = set()
        if capture['observation_id'] == proposal_observation_id or (initial and reobserved_hints):
            tables = [d.scene_quote for d in declared_objects if d.kind == 'table']
            if len(tables) == 1:
                initial_supports = {(d.scene_quote, tables[0]) for d in declared_objects
                                    if d.kind != 'table' and d.initially_on_table}
            by_ref = {s.ref: s for s in proposals}
            for spec in proposals:
                # Only unconditional positive "on" relations (direct or conjunction).
                relations = [spec.selection.predicate]
                while relations:
                    term = relations.pop()
                    if term.op == 'and':
                        relations.extend(term.terms)
                    elif term.op == 'on' and term.ref in by_ref and by_ref[term.ref].kind == 'table':
                        initial_supports.add((spec.scene_quote, by_ref[term.ref].scene_quote))
        groups, matches, scene = associate_scene(self.visual_identity, capture, observation,
            entities=entities, reference=reference, archive_loader=self.visual_identity.load_identity_archive,
            initial_supports=initial_supports,
            pose_resolver=self.cad_resolver.candidate_pose if self.cad_resolver else None)
        return GroundingResult('success', observation, groups, history_matches=matches,
            evidence={'scene': scene, 'current_correspondence': scene['current_correspondence'],
                      'scene_coverage': observation.coverage, 'mask_pairs': observation.mask_pair_evidence,
                      'queries': observation.query_evidence, 'proposal_overlap': observation.proposal_overlap_evidence,
                      'decision_records': scene['decision_records'], 'registration_basis': 'scene_association_v3'})

    def bind_task_roles(self, capture, grounding, group_ids, entities, selectors, **options):
        from .task_grounding import bind_task_roles

        return bind_task_roles(self.visual_identity, capture, grounding, group_ids, entities, selectors, **options)

    def observe(self, capture, *, instruction="", checkpoint=None, source_hint=None):
        """Capture-scoped target and environment segmentation from measured RGBD."""
        _validate_capture(capture)
        self.segmentation_evidence = []
        hint = source_hint or {"kind": "cube", "color": "any"}
        entities = hint.get('scene_entities')
        reference = hint.get('visual_reference')
        if not entities or not reference:
            raise PerceptionError('controller_objects_required')
        try:
            return self._observe_entities(capture, instruction, checkpoint, hint, entities, reference, None)
        except PerceptionError as error:
            error.sam3_queries = deepcopy(self.segmentation_evidence)
            error.details = {'observation_id': capture['observation_id'],
                'purpose': 'associate_objects', **(error.details or {}),
                'sam3_queries': [{key: deepcopy(value) for key, value in query.items() if key != 'proposals'}
                                 for query in self.segmentation_evidence]}
            raise

    def _observe_entities(self, capture, instruction, checkpoint, hint, entities, reference, discovery_evidence):
        # Deduplicate prompts, not physical instances. SAM3 receives no entity IDs.
        timings = {}
        progress = hint.get('observation_progress')
        if progress:
            progress('observation_stage', 'SAM3 分割与深度对应', {'observation_id': capture['observation_id']})
        stage_started = time.monotonic()
        grounded = hint.get('grounded_observation') or {}
        if (grounded.get('observation_id') != capture['observation_id']
                or grounded.get('generation') != capture['generation']):
            raise PerceptionError('confirmed_rgbd_grounding_required')
        provenance = {"observation_id": capture["observation_id"], "generation": capture["generation"],
                      "capture_time": capture["time"], "checkpoint": checkpoint,
                      'acquired_camera_ids': sorted(capture['views']),
                      'analyzed_camera_ids': sorted(capture['views'])}
        detections = list(grounded['confirmed_candidates'])
        queries = grounded['sam3_queries']
        views, table_points = {}, []
        for camera_id, view in capture['views'].items():
            views[camera_id] = {key: deepcopy(value) for key, value in view.items()
                               if key not in {'rgb', 'depth_m', 'valid_depth'}}
            for key in ('rgb', 'depth_m', 'valid_depth'):
                views[camera_id][key] = _readonly(view[key])
            views[camera_id].update(instance_masks={}, identity_masks={})
            if camera_id == '3':
                table_points.append(deproject_view(view, view['valid_depth']))
        timings["segmentation_and_depth"] = time.monotonic() - stage_started
        if progress:
            progress('observation_stage', '桌面与候选几何处理', {'completed_timings_s': dict(timings)})
        stage_started = time.monotonic()
        table = self._table_for_entities(capture, hint, entities, reference, detections,
                                        background_points=table_points)
        marker_capture = {**capture, "views": {camera: view for camera, view in capture["views"].items()
                          if camera in hint.get("camera_ids", ("3", "2", "1"))}}
        try:
            markers = detect_table_markers(marker_capture, table)
        except ValueError as error:
            raise PerceptionError(str(error)) from error
        markers = [marker for marker in markers if marker['id'] in entities]
        marker_masks = {camera: np.zeros(view["depth_m"].shape, bool) for camera, view in views.items()}
        for marker in markers:
            marker["observation_provenance"] = deepcopy(provenance)
            for camera, mask in marker.pop("masks_by_camera").items():
                marker_masks[camera] |= mask
                views[camera]["instance_masks"][marker["id"]] = _readonly(mask)
        timings['table_and_detection_geometry'] = time.monotonic() - stage_started
        stage_started = time.monotonic()
        identity_api_started = time.monotonic()
        identity_cpu_started = time.thread_time()
        robot_ownership = {'source': 'rgbd_grounding', 'identity_assignment': False}
        visual_result = deepcopy(grounded['visual_result'])
        if not isinstance(visual_result, dict) or not isinstance(visual_result.get('groups'), list):
            raise PerceptionError('visual_identity_result_required')
        entities.update(visual_result.get('registered_entities', {}))
        timings["identity_resolution"] = time.monotonic() - stage_started
        timings['identity_pipeline_wall'] = time.monotonic() - identity_api_started
        timings['identity_pipeline_thread_cpu'] = time.thread_time() - identity_cpu_started
        stage_started = time.monotonic()
        geometry_timing = {'objects': []}
        if progress:
            progress('observation_stage', '物体几何过滤与 CAD 配准', {'completed_timings_s': dict(timings)})
        geometry_cpu_started = time.thread_time()
        filter_started = time.monotonic()
        groups = visual_result['groups']
        for group in groups:
            for entry in group:
                view = views[entry['camera_id']]
                valid = entry['silhouette_mask'] & view['valid_depth']
                if 'robot_mask' in view:
                    valid &= ~view['robot_mask']
                entry.update(mask=valid, points=deproject_view(view, valid),
                    pixels=view['rgb'][valid], pixels_rc=np.column_stack(np.nonzero(valid)))
        groups, boundary_count = _source_geometry_interiors(groups, views)
        groups = self._filter_geometry_groups(groups, views, entities=entities)
        robot_ownership.update(source_silhouette_band_px=1, source_component_minimum_pixels=30,
                               source_geometry_removed_pixels=boundary_count)
        geometry_timing['filters_wall_s'] = time.monotonic() - filter_started
        geometry_timing['filters_thread_cpu_s'] = time.thread_time() - geometry_cpu_started
        objects = list(markers)
        for group in groups:
            try:
                object_started, object_cpu_started = time.monotonic(), time.thread_time()
                instance = _estimate_object(group, table, observation_provenance=provenance, views=views)
                entity = entities.get(instance['id'], {})
                reference_color = entity.get('color')
                if (instance['color'] == 'unknown' and isinstance(reference_color, str)
                        and reference_color.strip().lower() not in {'', 'unknown', 'any', 'none'}):
                    # Identity is already bound by Decisions. Preserve its
                    # reference appearance when the HSV vocabulary has no bin.
                    # Keep available pixel labels; neither label rebinds identity.
                    instance['color_evidence'].update(heuristic_label=instance['color'],
                        semantic_source='identity_bound_reference')
                    instance['color'] = reference_color
                object_timing = {'instance_id': instance['id'],
                    'measured_geometry_wall_s': time.monotonic() - object_started,
                    'measured_geometry_thread_cpu_s': time.thread_time() - object_cpu_started}
                from .spatial_memory import local_geometry_change
                anchor = hint.get('published_geometry', {}).get(instance['id'])
                change = local_geometry_change(anchor, instance['points_world_m'], views) if anchor else None
                if change:
                    instance['local_geometry_change'] = change
                prior_cad = (anchor or {}).get('cad')
                if hint.get('local_geometry_only') and prior_cad and change is None:
                    # Preserve the FoundationPose capture and its actual surface evidence unchanged.
                    instance['cad'] = deepcopy(prior_cad)
                    instance.pop('obb', None)
                    if anchor.get('placement_reference'):
                        instance['placement_reference'] = deepcopy(anchor['placement_reference'])
                    instance['retained_geometry_observation_id'] = anchor['observation_id']
                elif self.cad_resolver is not None:
                    for field in ('description', 'appearance_keywords', 'geometry_policy'):
                        if field in entity:
                            instance[field] = deepcopy(entity[field])
                    instance['description_provenance'] = {
                        'source': 'responses_entity_proposal', 'cad_evidence': False,
                        'proposal_source': entity.get('identity_source', 'declared_entity'),
                        'binding_source': instance.get('identity_source', 'preconfirmed_group'),
                        'entity_id': instance['id'], 'observation_id': capture['observation_id'],
                        'generation': capture['generation']}
                    try:
                        cad_started, cad_cpu_started = time.monotonic(), time.thread_time()
                        instance = self.cad_resolver.resolve(
                            {**capture, 'cad_motion_context': hint.get('cad_motion_context')},
                            group, instance, instruction=instruction)
                        object_timing.update(cad_resolution_wall_s=time.monotonic() - cad_started,
                            cad_resolution_thread_cpu_s=time.thread_time() - cad_cpu_started)
                    except (ValueError, RuntimeError, OSError) as error:
                        raise PerceptionError(str(error), details=getattr(error, 'details', {})) from error
            except PerceptionError as error:
                if str(error).startswith("instance_visible_depth_insufficient"):
                    continue  # Semantic objects remains; map publication rejects missing geometry.
                raise
            geometry_timing['objects'].append(object_timing)
            objects.append(instance)
            for entry in group:
                views[entry["camera_id"]]["instance_masks"][instance["id"]] = _readonly(entry["mask"])
                views[entry["camera_id"]]["identity_masks"][instance["id"]] = _readonly(entry.get("raw_identity_mask", entry["silhouette_mask"]))
        measured_ids = {obj['id'] for obj in objects}
        deferred = set(hint.get('association_scope', {}).get('deferred_entity_ids', []))
        visual_result['deferred_identity_ids'] = sorted(deferred)
        missing = [identifier for identifier, entity in entities.items()
                   if 'collision' in entity.get('roles', ()) and entity.get('kind') != 'table'
                   and identifier not in measured_ids and identifier not in deferred]
        visual_result['missing_geometry_ids'] = missing
        for instance in objects:
            if instance["kind"] not in {"bin", "marker"}:
                instance["measured_relationships"] = measured_source_relationships(instance, objects)
        from .placement_layout import identify_placement_regions

        identify_placement_regions({'objects': objects, 'observation_id': capture['observation_id']})
        geometry_timing['total_thread_cpu_s'] = time.thread_time() - geometry_cpu_started
        timings["geometry_filter_and_estimation"] = time.monotonic() - stage_started
        return {"observation_id": capture["observation_id"], "generation": capture["generation"],
                "time": capture["time"], "source": "rgbd", "objects": objects, "table": table,
                "observation_provenance": provenance, "arms": _robot_feedback(capture["arms"]),
                "perception_timings_s": timings, "perception_geometry_timing": geometry_timing, "views": views, "sam3_queries": queries, "sam3_request_count": sum(not query['cache_hit'] for query in queries),
                "visual_identity": {key: value for key, value in visual_result.items() if key != 'groups'},
                "scene_entities": deepcopy(entities), "visual_reference": reference,
                "visual_discovery": discovery_evidence,
                "mapping_unresolved": [{'previous_id': identifier,
                    'status': 'unseen', 'candidates': [],
                    'reason': 'fixed_collision_entity_not_observed'} for identifier in missing],
                "robot_surface_ownership": robot_ownership, "detector_model": deepcopy(SAM3_MODEL_IDENTITY)}

    def _table_for_entities(self, capture, hint, entities, reference, detections, *, background_points=()):
        return deepcopy(hint["table"]) if hint.get("table") else _estimate_table(np.vstack(background_points))

    def _record_segmentation_attempt(self, evidence, view, proposals):
        evidence.update(candidate_count=len(proposals), candidate_scores=[], diagnostic_checks=[],
                        status='candidates_returned' if proposals else 'no_candidates',
                        rgb_sha256=hashlib.sha256(np.ascontiguousarray(view['rgb']).tobytes()).hexdigest(),
                        known_facts=['returned_candidate_scores_are_sam3_scores_not_decisions_confidence'])
        for index, proposal in enumerate(proposals):
            mask = np.asarray(proposal['mask'], dtype=bool)
            score = float(proposal['score'])
            if mask.shape != view['depth_m'].shape or not np.isfinite(score) or not 0 <= score <= 1:
                raise PerceptionError('invalid_sam3_detection', details={
                    key: deepcopy(value) for key, value in evidence.items() if key != 'proposals'})
            valid = mask & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)
            count = int(np.count_nonzero(valid))
            evidence['candidate_scores'].append(score)
            evidence['diagnostic_checks'].append({'name': 'valid_depth_' + str(index),
                'component': 'rgbd', 'stage': 'segmentation_depth_filter',
                'camera_id': evidence.get('camera_id', view.get('camera_id')), 'proposal_index': index,
                'prompt': evidence['prompt'], 'sam3_score': score, 'metric': 'valid_depth_pixels',
                'mask_pixels': int(np.count_nonzero(mask)), 'actual': count,
                'threshold': 30, 'comparison': '>=', 'status': 'pass' if count >= 30 else 'fail'})

    def _filter_geometry_groups(self, groups, views, *, entities=None):
        return groups


def _robot_feedback(arms):
    return {arm: {key: deepcopy(value) for key, value in feedback.items()
                  if key in {"joints_rad", "tcp", "gripper_closed_fraction", "gripper_position_rad",
                             "gripper_limits_rad"}}
            for arm, feedback in arms.items() if arm in {"A", "B"}}


def public_observation(observation):
    """Text-only allowlist for JEV; no RGB/depth/masks/cloud arrays or private fields."""
    result = {key: deepcopy(observation[key]) for key in ("observation_id", "generation", "time", "source")}
    result["arms"] = _robot_feedback(observation["arms"])
    result["table"] = {key: deepcopy(value) for key, value in observation["table"].items()
                       if key in {"position", "size", "surface_height_m", "uncertainty_m", "estimated", "source"}}
    object_keys = {"id", "kind", "color", "position", "visible_centroid_world", "estimated",
                   "confidence", "uncertainty_m", "source"}
    result["objects"] = []
    for instance in observation["objects"]:
        public = {key: deepcopy(value) for key, value in instance.items() if key in object_keys}
        for field, keys in {
            "coverage": {"observed_surface_only", "camera_ids", "sam3_camera_ids", "point_count", "limitations"},
            "color_evidence": {"median_rgb", "source", "heuristic_label", "semantic_source"},
            "observation_provenance": {"observation_id", "generation", "capture_time", "checkpoint"},
            "placement_reference": {"position_world_m", "floor_z_m", "region_polygon_world_xy_m", "source",
                                    "observed_floor_hull_world_xy_m",
                                    "region_id", "support_entity_id", "plane", "observation_id", "region_measurement"},
            "visual_semantics": {"label", "kind", "color", "visibility", "ambiguous", "on_table", "inside_id",
                                 "on_marker_id", "evidence", "source", "observation_id", "generation", "camera_ids"},
        }.items():
            if field in instance:
                public[field] = {key: deepcopy(value) for key, value in instance[field].items() if key in keys}
        if "placement_reference" in instance and "measurement_sources" in instance["placement_reference"]:
            sources = instance["placement_reference"]["measurement_sources"]
            public["placement_reference"]["measurement_sources"] = {
                key: deepcopy(value) for key, value in sources.items() if key in {"point_count", "camera_ids"}}
        if "measured_relationships" in instance:
            public["measured_relationships"] = public_relationships(instance["measured_relationships"])
        result["objects"].append(public)
    return result

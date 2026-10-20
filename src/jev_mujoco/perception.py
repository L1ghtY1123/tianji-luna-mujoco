"""SAM3 RGB masks and measured RGBD geometry for Decisions-assigned identities.

No simulator object identifiers, editor metadata, or generated primitive clouds
cross this boundary. Incomplete views remain measured partial surfaces.
"""

import colorsys
import math
import os
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO

import httpx
import msgpack
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from .cameras import deproject_view
from .placement import measured_source_relationships, public_relationships
from .robot_surface import own_robot_surface_pixels
from .table_markers import detect_table_markers
from .visual_discovery import VisualDiscovery
from .visual_inventory import VisualInventory, VisualInventoryError

SAM3_MODEL_IDENTITY = {
    "name": "sam3", "source_sha": "5dd401d1c5c1d5c3eedff06d41b77af824517619",
    "weight_sha256": "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
}
MAX_RESPONSE_BYTES = 96 * 1024 * 1024
MAX_ARRAY_ITEMS = 16_777_216


class PerceptionError(RuntimeError):
    """Sensor, detector, geometry, or association lacks required evidence."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details


def _decode_array(value, dtype, rank, unit, frame):
    expected_keys = {"dtype", "rank", "shape", "unit", "frame", "data"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise PerceptionError("sam3_invalid_output")
    if (value["dtype"], value["rank"], value["unit"], value["frame"]) != (dtype, rank, unit, frame):
        raise PerceptionError("sam3_invalid_output")
    shape = value["shape"]
    if not isinstance(shape, list) or len(shape) != rank or any(type(n) is not int or n < 0 for n in shape):
        raise PerceptionError("sam3_invalid_output")
    count = math.prod(shape)
    data = value["data"]
    if count > MAX_ARRAY_ITEMS or type(data) is not bytes or len(data) != count * np.dtype(dtype).itemsize:
        raise PerceptionError("sam3_invalid_output")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


class Sam3Client:
    """Existing pinned base sidecar contract; only the HTTP transport is adapted.

    A request is one unchanged English class prompt and the actual frozen RGB
    JPEG. The sidecar runs the official model; errors never invoke a substitute.
    """

    def __init__(self, url=None, timeout_s=180):
        self.url = url or os.getenv("JEV_SAM3_URL", "http://127.0.0.1:8322")
        self.timeout_s = float(timeout_s)

    def segment(self, rgb, text, timestamp_us):
        text = text.strip()
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or min(rgb.shape[:2]) < 1 or max(rgb.shape[:2]) > 4096 or rgb.shape[0] * rgb.shape[1] > 4_194_304
                or not 1 <= len(text) <= 256 or type(timestamp_us) is not int or not 0 <= timestamp_us <= 2**63 - 1):
            raise PerceptionError("sam3_invalid_request")
        image_buffer = BytesIO()
        Image.fromarray(rgb).save(image_buffer, format="JPEG", quality=95, subsampling=0)
        payload = {"rgb_jpeg": image_buffer.getvalue(), "text": text, "timestamp_us": timestamp_us}
        body = msgpack.packb({"operation": "segment_text", "payload": payload}, use_bin_type=True)
        try:
            with httpx.stream("POST", self.url, content=body, timeout=self.timeout_s,
                              headers={"Content-Type": "application/msgpack", "Accept": "application/msgpack"}) as response:
                if response.status_code != 200 or response.headers.get("content-type", "").strip() != "application/msgpack":
                    raise PerceptionError("sam3_invalid_output")
                response_bytes = bytearray()
                for chunk in response.iter_bytes():
                    response_bytes.extend(chunk)
                    if len(response_bytes) > MAX_RESPONSE_BYTES:
                        raise PerceptionError("sam3_invalid_output")
        except httpx.HTTPError as error:
            raise PerceptionError("sam3_service_unavailable") from error
        try:
            envelope = msgpack.unpackb(response_bytes, raw=False, strict_map_key=True)
            if type(envelope) is not dict or type(envelope.get("ok")) is not bool:
                raise PerceptionError("sam3_invalid_output")
            if envelope["ok"] is False:
                if set(envelope) != {"ok", "error"} or not isinstance(envelope["error"], dict):
                    raise PerceptionError("sam3_invalid_output")
                code = envelope["error"].get("code")
                if code == "sam3_no_match":
                    return []
                if code not in {"sam3_service_unavailable", "sam3_invalid_request", "sam3_invalid_output",
                                "sam3_model_identity_mismatch", "sam3_visualization_failed"}:
                    code = "sam3_invalid_output"
                raise PerceptionError(code)
            if set(envelope) != {"ok", "result"}:
                raise PerceptionError("sam3_invalid_output")
            result = envelope["result"]
            if set(result) != {"masks", "boxes", "scores", "selected_index", "model", "timing_us", "timestamp_us"}:
                raise PerceptionError("sam3_invalid_output")
            if result["model"] != SAM3_MODEL_IDENTITY:
                raise PerceptionError("sam3_model_identity_mismatch")
            masks = _decode_array(result["masks"], "uint8", 3, "mask", "image")
            boxes = _decode_array(result["boxes"], "float32", 2, "pixel", "image")
            scores = _decode_array(result["scores"], "float32", 1, "confidence", "model")
            count = len(masks)
            selected = result["selected_index"]
            timing = result["timing_us"]
            if (not 1 <= count <= 200 or masks.shape[1:] != rgb.shape[:2] or boxes.shape != (count, 4)
                    or scores.shape != (count,) or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                    or not np.isin(masks, [0, 1]).all() or np.any((scores < 0) | (scores > 1))
                    or type(selected) is not int or not 0 <= selected < count or not masks[selected].any()
                    or result["timestamp_us"] != timestamp_us or not isinstance(timing, dict)
                    or set(timing) != {"inference", "total"}
                    or any(type(value) is not int or value < 0 for value in timing.values())):
                raise PerceptionError("sam3_invalid_output")
            return [{"mask": mask.astype(bool), "score": float(score), "label": text}
                    for mask, score in zip(masks, scores, strict=True) if mask.any()]
        except PerceptionError:
            raise
        except (ValueError, TypeError, KeyError) as error:
            raise PerceptionError("sam3_invalid_output") from error


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
                    or view["simulation_generation"] != capture["generation"] or view["capture_time"] != capture["time"]):
                raise PerceptionError("RGBD views must use the same captured frame, time and generation")
            calibration_ids.add(view["calibration_id"])
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
            for arm, feedback in capture["arms"].items():
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


def _estimate_object(group, table, *, observation_provenance=None):
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
    pixels = np.vstack([entry["pixels"] for entry in group])
    color, median_rgb = _color(pixels)
    centroid = points.mean(axis=0, dtype=np.float64).tolist()
    provenance = deepcopy(observation_provenance or {})
    point_sources = {"camera_ids": _readonly(camera_ids), "pixels_rc": _readonly(pixels_rc),
                     "observation_id": provenance.get("observation_id")}
    identifier = group[0].get('entity_id')
    if not isinstance(identifier, str) or not identifier or any(
            entry.get('entity_id') != identifier for entry in group):
        raise PerceptionError('visual_inventory_entity_required')
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
    images = {camera: dict(view, instance_masks={}) for camera, view in views.items()}
    union = {camera: np.zeros(view["depth_m"].shape, bool) for camera, view in views.items()}
    for index, group in enumerate(groups):
        for entry in group:
            camera = entry["camera_id"]
            images[camera]["instance_masks"][index] = entry["mask"]
            if entry["kind"] != "bin":
                union[camera] |= entry["mask"]
    evidence = classifier(capture, union)
    masks = evidence.pop("masks")
    observation = {"views": images}
    rebuilt, removed_count, ambiguous_count, contact_count = [], 0, 0, 0
    for index, group in enumerate(groups):
        # Only graspable source geometry enters attachment. Bin obstacle
        # returns keep their existing scene robot-removal path.
        if group[0]["kind"] == "bin":
            rebuilt.append(group)
            continue
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
            robot_mask = views[camera].setdefault("robot_surface_mask", np.zeros(mask.shape, bool))
            robot_mask[tuple(removed.T)] = True
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
    """Exclude mixed silhouette depth after identity association; keep raw depth."""
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
            removed_count += int(np.count_nonzero(entry["mask"] & ~mask))
            valid = mask & view["valid_depth"] & np.isfinite(view["depth_m"]) & (view["depth_m"] > 0)
            entries.append(entry | {"mask": mask, "points": deproject_view(view, mask),
                "pixels": view["rgb"][valid], "pixels_rc": np.column_stack(np.nonzero(valid))})
        rebuilt.append([entry for entry in entries if len(entry["points"])] or entries)
    return rebuilt, removed_count


class Perception:
    """Detector injection bounds network I/O; all fusion uses actual captured data."""

    def __init__(self, detector=None, *, robot_surface_classifier=None, visual_inventory=None, visual_discovery=None):
        self.detector = detector if detector is not None else Sam3Client()
        self.visual_inventory = visual_inventory if visual_inventory is not None else VisualInventory.from_environment()
        self.visual_discovery = visual_discovery if visual_discovery is not None else VisualDiscovery.from_environment()
        self.segmentation_cache = OrderedDict()
        self.robot_surface_classifier = (own_robot_surface_pixels if robot_surface_classifier is None
                                         else robot_surface_classifier)

    def observe(self, capture, *, instruction="", checkpoint=None, source_hint=None):
        """Capture-scoped target and environment segmentation from measured RGBD."""
        _validate_capture(capture)
        hint = source_hint or {"kind": "cube", "color": "any"}
        selection = hint.get('visual_decision', {})
        if selection.get('source') != 'jev' or selection.get('purpose') not in {
                'initialize_inventory', 'associate_inventory', 'verify_transfer'}:
            raise PerceptionError('visual_decisions_selection_required')
        entities = hint.get('scene_entities')
        reference = hint.get('visual_reference')
        discovery_evidence = None
        if not entities:
            if selection['purpose'] != 'initialize_inventory':
                raise PerceptionError('visual_inventory_initial_rgb_required')
            try:
                discovered = self.visual_discovery.discover(capture, instruction=instruction)
            except VisualInventoryError as error:
                raise PerceptionError(str(error), details=error.evidence) from error
            entities, reference = discovered['entities'], discovered['reference']
            discovery_evidence = discovered['evidence']
        try:
            return self._observe_entities(capture, instruction, checkpoint, hint, entities, reference, discovery_evidence)
        except PerceptionError as error:
            if discovery_evidence is not None:
                # Transport successful RGB discovery across a later geometry/API failure.
                # Runner stores it on its own thread; no worker mutates the task ledger.
                error.initial_inventory = {'entities': entities, 'reference': reference,
                    'evidence': discovery_evidence, 'observation_id': capture['observation_id']}
            raise

    def _observe_entities(self, capture, instruction, checkpoint, hint, entities, reference, discovery_evidence):
        # Deduplicate prompts, not physical instances. SAM3 receives no entity IDs.
        fixed_queries = list({(query['kind'], query['color'], query['prompt']): query
            for entity in entities.values() if (query := entity.get('segmentation_query'))}.values())
        provenance = {"observation_id": capture["observation_id"], "generation": capture["generation"],
                      "capture_time": capture["time"], "checkpoint": checkpoint}
        timestamp_us = round(capture["time"] * 1_000_000)
        detections, views, table_points, queries = [], {}, [], []
        for camera_id in ("3", "2", "1"):
            view = capture["views"][camera_id]
            prompts = []
            selected = camera_id in hint.get("camera_ids", ("3", "2", "1"))
            if selected:
                prompts = [(query['kind'], query['color'], query['prompt']) for query in fixed_queries]
            occupied = np.zeros(view["depth_m"].shape, bool)
            for kind, color, prompt in prompts:
                key = (capture['generation'], capture['observation_id'], camera_id, prompt,
                       view['calibration_id'], timestamp_us)
                cached = key in self.segmentation_cache
                if not cached:
                    self.segmentation_cache[key] = deepcopy(self.detector.segment(view['rgb'], prompt, timestamp_us))
                    if len(self.segmentation_cache) > 128:
                        self.segmentation_cache.popitem(last=False)
                self.segmentation_cache.move_to_end(key)
                queries.append({'camera_id': camera_id, 'prompt': prompt, 'cache_hit': cached})
                for detection in self.segmentation_cache[key]:
                    mask = np.asarray(detection["mask"], dtype=bool)
                    score = float(detection["score"])
                    if mask.shape != occupied.shape or not np.isfinite(score) or not 0 <= score <= 1:
                        raise PerceptionError("invalid_sam3_detection")
                    valid = mask & view["valid_depth"] & np.isfinite(view["depth_m"]) & (view["depth_m"] > 0)
                    points = deproject_view(view, valid)
                    if len(points) < 30:
                        continue
                    pixels = view["rgb"][valid]
                    measured_color, _ = _color(pixels)
                    detections.append({"mask": valid, "silhouette_mask": mask, "points": points, "pixels": pixels, "kind": kind,
                        "color": measured_color, "requested_color": color, "score": score, "camera_id": camera_id, "view": view,
                        "pixels_rc": np.column_stack(np.nonzero(valid)), "prompt": prompt})
                    occupied |= valid
            views[camera_id] = {key: deepcopy(value) for key, value in view.items()
                                if key not in {"rgb", "depth_m", "valid_depth"}}
            for key in ("rgb", "depth_m", "valid_depth"):
                views[camera_id][key] = _readonly(view[key])
            views[camera_id]["instance_masks"] = {}
            if camera_id == "3" and not hint.get("table"):
                points = deproject_view(view, ~occupied)
                # Fixed robot workspace, not an oracle table/object pose.
                inside = ((points[:, 0] >= 0.) & (points[:, 0] <= 1.1)
                          & (np.abs(points[:, 1]) <= .6) & (points[:, 2] > .1) & (points[:, 2] <= 1.6))
                table_points.append(points[inside])
        table = deepcopy(hint["table"]) if hint.get("table") else _estimate_table(np.vstack(table_points))
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
        filtered = []
        for detection in detections:
            camera = detection["camera_id"]
            mask = detection["mask"] & ~marker_masks[camera]
            if np.count_nonzero(mask) < 30:
                continue
            pixels = capture["views"][camera]["rgb"][mask]
            color, _ = _color(pixels)
            filtered.append(detection | {"mask": mask, "points": deproject_view(capture["views"][camera], mask),
                "pixels": pixels, "pixels_rc": np.column_stack(np.nonzero(mask)), "color": color})
        selected_capture = {**capture, 'views': {camera: view for camera, view in views.items()
                            if camera in hint.get('camera_ids', ('3', '2', '1'))}}
        try:
            visual_result = self.visual_inventory.resolve(selected_capture, filtered, instruction=instruction,
                reference=reference, entities=entities,
                progress=hint.get('visual_progress'), landmarks=[{key: obj[key] for key in
                    ('id', 'kind', 'color', 'position') if key in obj} for obj in markers],
                table_scope=table, placement_regions=hint.get('placement_regions', ()),
                purpose=hint['visual_decision']['purpose'])
        except VisualInventoryError as error:
            raise PerceptionError(str(error), details=error.evidence) from error
        if not isinstance(visual_result, dict) or not isinstance(visual_result.get('groups'), list):
            raise PerceptionError('visual_inventory_result_required')
        groups = visual_result['groups']
        groups, robot_ownership = _remove_own_robot_surface(groups, views, capture, self.robot_surface_classifier)
        groups, boundary_count = _source_geometry_interiors(groups, views)
        robot_ownership.update(source_silhouette_band_px=1, source_silhouette_removed_pixels=boundary_count)
        objects = list(markers)
        for group in groups:
            try:
                instance = _estimate_object(group, table, observation_provenance=provenance)
            except PerceptionError as error:
                if str(error).startswith("instance_visible_depth_insufficient"):
                    continue  # Semantic inventory remains; map publication rejects missing geometry.
                raise
            objects.append(instance)
            for entry in group:
                views[entry["camera_id"]]["instance_masks"][instance["id"]] = _readonly(entry["mask"])
        geometry_ids = {obj['id'] for obj in objects}
        missing = [identifier for identifier, entity in entities.items()
                   if 'collision' in entity.get('roles', ()) and entity.get('kind') != 'table'
                   and identifier not in geometry_ids]
        visual_result['missing_geometry_ids'] = missing
        for instance in objects:
            if instance["kind"] not in {"bin", "marker"}:
                instance["measured_relationships"] = measured_source_relationships(instance, objects)
        from .placement_layout import identify_placement_regions

        identify_placement_regions({'objects': objects, 'observation_id': capture['observation_id']})
        return {"observation_id": capture["observation_id"], "generation": capture["generation"],
                "time": capture["time"], "source": "rgbd", "objects": objects, "table": table,
                "observation_provenance": provenance, "arms": _robot_feedback(capture["arms"]),
                "views": views, "sam3_queries": queries, "sam3_request_count": sum(not query['cache_hit'] for query in queries),
                "visual_inventory": {key: value for key, value in visual_result.items() if key != 'groups'},
                "scene_entities": deepcopy(entities), "visual_reference": reference,
                "visual_discovery": discovery_evidence,
                "mapping_unresolved": [{'previous_id': identifier, 'status': 'unseen', 'candidates': [],
                    'reason': 'fixed_collision_entity_not_observed'} for identifier in missing],
                "declared_environment": [],
                "robot_surface_ownership": robot_ownership, "detector_model": deepcopy(SAM3_MODEL_IDENTITY)}


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
    object_keys = {"id", "display_id", "kind", "color", "position", "visible_centroid_world", "estimated",
                   "confidence", "uncertainty_m", "source"}
    result["objects"] = []
    for instance in observation["objects"]:
        public = {key: deepcopy(value) for key, value in instance.items() if key in object_keys}
        for field, keys in {
            "coverage": {"observed_surface_only", "camera_ids", "sam3_camera_ids", "point_count", "limitations"},
            "color_evidence": {"median_rgb", "source"},
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

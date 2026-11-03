"""Scope fresh visual correspondence using a previously verified placement identity."""
import hashlib
import json
import math
from copy import deepcopy

import numpy as np

from .observation_labels import raster_visual_annotations
from .visual_api import VisualAPI, VisualInventoryError
from .visual_inventory import VisualInventory, choice, image_part, mask_box, valid_assigned_slot


def _require(condition, reason):
    if not condition:
        raise VisualInventoryError("completed_tracking_" + reason)


def _finite_time(value):
    return type(value) in (int, float) and math.isfinite(value)


def confirmed_capture_record(capture):
    """Bind a release photograph to its original RGBD/calibration, not just its ID."""
    digest = hashlib.sha256()
    for camera, view in sorted(capture.get('views', {}).items()):
        digest.update(json.dumps([camera, view.get('calibration_id'), view.get('intrinsics')],
                                 sort_keys=True).encode())
        for key in ('rgb', 'depth_m', 'valid_depth', 'T_world_camera'):
            digest.update(key.encode())
            if key not in view:
                digest.update(b'missing')
                continue
            array = np.asarray(view[key])
            digest.update(json.dumps([array.shape, str(array.dtype)]).encode())
            digest.update(array.tobytes())
    return {**{key: capture[key] for key in
               ('observation_id', 'generation', 'time', 'captured_perf_counter_ns') if key in capture},
            'sensor_fingerprint': digest.hexdigest()}


def completed_identity_tracks(capture, reference, progress, entities):
    """Validate a confirmed transfer, independently of the final task goal."""
    tracks = {}
    for identifier, record in (progress or {}).items():
        track = record.get("completed_identity_track")
        if track is None:
            continue  # Historical callers retain their original identity questions.
        # A verified intermediate placement remains observed in the policy ledger.
        # Its bound release/visual proof below is as valid as a final delivery.
        _require(record.get("status") in {"observed", "completed"}
                 and identifier in entities, "record_invalid")
        binding, visual, prior = track.get("binding", {}), track.get("visual_evidence", {}), track.get("capture", {})
        unsigned = {key: value for key, value in binding.items() if key != "binding_id"}
        digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        _require(binding.get("binding_id") == digest, "binding_changed")
        _require(binding.get("source") == "validated_held_attachment_place_release"
                 and binding.get("identity_source") == "execution_chain"
                 and binding.get("target_id") == binding.get("instance_id") == identifier
                 and binding.get("owner_id") == (reference or {}).get("owner_id")
                 and bool(binding.get("owner_id"))
                 and binding.get("generation") == prior.get("generation") == capture.get("generation"),
                 "runtime_binding_invalid")
        _require(valid_assigned_slot(record) and valid_assigned_slot(binding)
                 and binding["assigned_slot"] == record["assigned_slot"]
                 and binding.get("layout_revision") == record["assigned_slot"]["layout_revision"]
                 and all(binding.get(key) == record.get(key) for key in
                         ("destination_id", "region_id", "slot_id", "transfer_index", "release_feedback_id")),
                 "slot_binding_changed")
        _require(visual.get("source") == "openai_decisions"
                 and visual.get("identity_source") == "execution_chain"
                 and visual.get("status") == "at_destination"
                 and type(visual.get("confidence")) in (int, float) and .8 <= visual["confidence"] <= 1.
                 and visual.get("request_hash")
                 and visual.get("observation_id") == prior.get("observation_id")
                 and all(visual.get(key) == binding.get(key) for key in
                         ("binding_id", "target_id", "destination_id", "region_id", "slot_id",
                          "layout_revision", "release_feedback_id")), "visual_binding_invalid")
        same_capture = capture.get('observation_id') == prior.get('observation_id')
        _require(capture.get("observation_id") and prior.get("observation_id")
                 and _finite_time(prior.get("time")) and _finite_time(capture.get("time"))
                 and _finite_time(binding.get("release_feedback_time"))
                 and binding["release_feedback_time"] <= prior["time"] <= capture["time"],
                 "capture_not_after_completion")
        if same_capture:
            _require(prior.get('sensor_fingerprint') and prior == confirmed_capture_record(capture),
                     'confirmed_capture_changed')
        for key in ("captured_perf_counter_ns",):
            if key in prior:
                _require(type(prior[key]) is int and type(capture.get(key)) is int
                         and (capture[key] == prior[key] if same_capture else capture[key] > prior[key]),
                         "capture_not_after_completion")
        tracks[identifier] = deepcopy(track)
    return tracks


def _inside_slot(points, slot):
    polygon = np.asarray(slot["polygon_world_xy_m"], dtype=float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = points[:, None, :2] - polygon[None, :, :]
    cross = edges[None, :, 0] * offsets[:, :, 1] - edges[None, :, 1] * offsets[:, :, 0]
    xy_inside = np.all(cross >= -1e-9, axis=1) | np.all(cross <= 1e-9, axis=1)
    # This locating envelope neither adds geometry nor proves identity. An
    # undeclared object height cannot limit generic object correspondence.
    floor, tolerance = slot["floor_z_m"], .015
    inside = np.all(xy_inside) and np.all(points[:, 2] >= floor - tolerance)
    height = slot.get('height_prior_m')
    if height is not None:
        inside = inside and np.all(points[:, 2] <= floor + height + tolerance)
    return bool(inside)


def completed_candidate_scopes(capture, detections, tracks):
    scopes = {}
    for index, detection in enumerate(detections):
        points = np.asarray(detection.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            continue
        matches = []
        for identifier, track in tracks.items():
            slot = track["binding"]["assigned_slot"]
            # Reuse the existing 6 mm measured-support exclusion for locating
            # only. Never crop by slot XY or change the raw visual candidate.
            matching = points[np.abs(points[:, 2] - slot["floor_z_m"]) > .006]
            if len(matching) and _inside_slot(matching, slot):
                matches.append((identifier, matching))
        if len(matches) != 1:
            continue
        owner, matching = matches[0]
        binding = tracks[owner]["binding"]
        scopes[index] = {"candidate": "candidate-" + str(index), "camera_id": detection["camera_id"],
            "owner_id": owner, "slot_id": binding["slot_id"], "region_id": binding["region_id"],
            "layout_revision": binding["layout_revision"], "binding_id": binding["binding_id"],
            "observation_id": capture["observation_id"], "point_count": len(matching),
            "raw_point_count": len(points), "matching_point_count": len(matching),
            "support_plane_removed_point_count": len(points) - len(matching),
            "support_plane_tolerance_m": .006,
            "support_plane_source": "validated_assigned_slot_floor_from_cached_measured_table",
            "raw_measured_bounds_world_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
            "measured_bounds_world_m": [matching.min(axis=0).tolist(), matching.max(axis=0).tolist()],
            "location_rule": "all_current_non_support_points_in_unique_completed_slot_prism",
            "height_prior_m": binding["assigned_slot"].get("height_prior_m"), "height_tolerance_m": .015,
            "identity_source": "execution_chain_and_fresh_visual_correspondence",
            "location_is_identity_proof": False}
    return scopes


def completed_correspondence_payload(capture, detections, tracks, scopes):
    """Ask current visual correspondence only; keep execution identities local."""
    _require(bool(scopes), "current_geometry_missing")
    content = [{"type": "input_text", "text": json.dumps({
        "observation_id": capture["observation_id"], "generation": capture["generation"],
        "purpose": "associate_completed_objects",
        "binding_ids": sorted(track["binding"]["binding_id"] for track in tracks.values()),
        "instructions": "Use all CURRENT images together. Classify the physical subject marked by each candidate label. "
            "A separate movable object can have a plain solid-color surface with little visible texture; "
            "it need not show industrial machining. Robot parts, the supporting container/surface itself, "
            "empty openings and image background are not separate objects. Registered-depth location checks "
            "have already scoped a prior released object, but location alone does not prove identity. "
            "Answer unknown when the visible subject cannot be distinguished. Multiple views can show the same object."})}]
    for camera, view in capture["views"].items():
        content.extend([{"type": "input_text", "text": "CURRENT RGB camera " + camera}, image_part(view["rgb"])])
        camera_scopes = [(index, scope) for index, scope in scopes.items() if scope["camera_id"] == camera]
        if not camera_scopes:
            continue
        # Exact same rectangles share a label; no candidate/mask or identity is merged.
        boxes = {}
        for index, scope in camera_scopes:
            boxes.setdefault(tuple(mask_box(detections[index]["silhouette_mask"])), []).append(index)
        items = [{"id": "/".join("Candidate " + str(index) for index in indices), "box": list(box)}
                 for box, indices in boxes.items()]
        content.extend([{"type": "input_text", "text": json.dumps({
            "image_role": "current_completed_annotations", "camera": camera, "items": items,
            "coordinates": "normalized to original full RGB"})},
            image_part(raster_visual_annotations(view["rgb"], items,
                                                  frame_color=(235, 235, 235)))])
    questions = [choice("tracked_" + str(index),
        f"Using all CURRENT views together, does the physical subject marked Candidate {index} show a separate movable object? "
        "Plain solid color or lack of visible machining does not exclude an object. "
        "Answer no for robot parts, the supporting container/surface itself, an empty opening or background; unknown if unclear.",
        ["yes", "no", "unknown"]) for index in scopes]
    _require(len(questions) <= 200, "input_limit")
    return {"model": "gpt-6-luna", "input": [{"role": "user", "content": content}], "questions": questions}


def verify_completed_correspondence(inventory, capture, detections, tracks, scopes):
    def parse(body, evidence):
        answers = body.get("answers")
        if not isinstance(answers, list) or len(answers) != len(payload["questions"]):
            raise VisualInventoryError("visual_decisions_invalid_response")
        mapping = {}
        for (index, scope), question, answer in zip(scopes.items(), payload["questions"], answers, strict=True):
            if (not isinstance(answer, dict) or answer.get("name") != question["name"]
                    or answer.get("type") != "choice" or answer.get("choice") not in {"yes", "no", "unknown"}
                    or type(answer.get("confidence")) not in (int, float)
                    or not 0 <= answer["confidence"] <= 1):
                raise VisualInventoryError("visual_decisions_invalid_response")
            if answer["choice"] == "yes" and answer["confidence"] >= inventory.minimum_confidence:
                mapping[str(index)] = scope["owner_id"]
        return {"tracks": tracks, "candidate_scopes": list(scopes.values()),
                "observation_id": capture["observation_id"], "generation": capture["generation"],
                "request_hash": evidence["request_hash"], "response": deepcopy(body),
                "minimum_confidence": inventory.minimum_confidence,
                "mapping_source": "validated_execution_track_unique_current_slot_and_fresh_binary_yes",
                "evidence": {**evidence, "answers": answers}, "candidate_mapping": mapping}

    try:
        payload = completed_correspondence_payload(capture, detections, tracks, scopes)
        return VisualAPI.request(inventory, payload, capture, "associate_completed_objects", parse)
    except VisualInventoryError as error:
        error.evidence["completed_tracking"] = {
            "tracks": tracks, "candidate_scopes": list(scopes.values()),
            "observation_id": capture["observation_id"], "generation": capture["generation"],
            "candidate_mapping": {}, "status": "unconfirmed"}
        raise


def merge_completed_correspondence(inventory, capture, detections, entities, tracking, result, *,
                                   partial_completed_ids=()):
    """Merge only new accepted visual correspondence and current measured pixels."""
    evidence = result["evidence"]
    _require(tracking["observation_id"] == evidence["observation_id"] == capture["observation_id"]
             and tracking["generation"] == evidence.get("generation") == capture["generation"]
             and tracking["request_hash"] == evidence.get("completed_correspondence_request_hash")
             and tracking["request_hash"] != evidence["request_hash"], "request_binding_mismatch")
    mapping = tracking["candidate_mapping"]
    accepted = [int(index) for index in mapping]
    grouped = VisualInventory._groups(inventory, [detections[index] for index in accepted],
        ["candidate-" + str(index) for index in accepted], entities,
        {"identity_" + str(local): mapping[str(index)] for local, index in enumerate(accepted)},
        tracking["evidence"])
    for group in grouped["groups"]:
        for entry in group:
            entry["identity_provenance"] = {
                "source": "execution_chain_and_fresh_visual_correspondence",
                "request_hash": tracking["request_hash"],
                "binding_id": tracking["tracks"][entry["entity_id"]]["binding"]["binding_id"]}
    result["groups"].extend(grouped["groups"])
    result["membership"].update(grouped["membership"])
    inventory._assign_surface_pixels(result["groups"])
    required = {identifier for identifier, entity in entities.items() if entity["kind"] not in {"table", "marker"}}
    observed = {group[0]["entity_id"] for group in result["groups"]
                if any(len(entry["points"]) for entry in group)}
    coverage = result["identity_coverage"]
    missing = sorted(required - observed)
    result["completed_tracking"] = tracking
    coverage.update(required_entity_ids=sorted(required), accepted_entity_ids=sorted(observed),
        complete=not missing,
        request_hashes={"completed_correspondence": tracking["request_hash"],
                        "remaining_inventory": evidence["request_hash"]})
    scoped = {scope['owner_id'] for scope in tracking['candidate_scopes']}
    unresolved = scoped - observed
    if unresolved - set(partial_completed_ids):
        # Keep an unresolved scoped view unknown. Another independently accepted
        # current view may still establish the entity; never erase its evidence.
        raise VisualInventoryError("visual_identity_unknown", evidence={
            **evidence, "completed_tracking": tracking, "identity_coverage": coverage,
            "missing_required_ids": missing})
    # A verified final delivery may be occluded in a later frozen-inventory
    # observation. Keep it missing, without discarding other current measurements
    # or treating its historical pose/identity as current evidence.
    if unresolved:
        result['unobserved_completed_ids'] = sorted(unresolved)
    result["missing_geometry_ids"] = missing
    return result

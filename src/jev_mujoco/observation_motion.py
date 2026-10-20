"""Task-visible camera evidence and the configured empty-hand observation motion."""
import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry import rigid_matrix


def matching_instances(observation, attributes, *, destination=False):
    result = []
    for obj in observation.get("objects", []):
        if (obj["kind"] in {"bin", "marker"}) != destination:
            continue
        kind, color = attributes.get("kind", "any"), attributes.get("color", "any")
        if kind != "any" and obj["kind"] != kind and not (kind == "cube" and obj["kind"] == "box"):
            continue
        if color != "any" and obj["color"] != color:
            continue
        result.append(obj)
    return result


def _head_detection_present(obj):
    coverage = obj.get("coverage", {})
    measured = coverage.get("camera_ids", [])
    if obj.get("kind") == "marker":
        return "3" in measured
    # Older observations/algorithm adapters do not carry the optional semantic
    # provenance. Current Perception always records it before ownership repair.
    detected = coverage.get("sam3_camera_ids", measured)
    return "3" in measured and "3" in detected


def source_view_evidence(observation, source):
    """Judge measured geometry, independently of which arm owns a camera."""
    cameras = set(source.get("coverage", {}).get("camera_ids", []))
    frame = observation.get("observation_id")
    points = np.asarray(source.get('points_world_m', []), dtype=float)
    valid = (points.ndim == 2 and points.shape[1] == 3 and len(points) >= 16
             and np.isfinite(points).all() and bool(cameras & {'1', '2', '3'})
             and source.get('observation_provenance', {}).get('observation_id', frame) == frame
             and source.get('position_source') != 'action_prediction')
    return {'source_id': source['id'], 'camera_ids': sorted(cameras), 'views_ready': bool(valid),
            'point_count': len(points), 'observation_id': frame, 'policy': 'current_measured_geometry',
            'complete_surface': False, 'grasp_feasibility': 'not_yet_tested'}


def initial_view_evidence(observation, task, attributes, destination_attributes):
    source = matching_instances(observation, attributes)
    destination = matching_instances(observation, destination_attributes, destination=True)
    if task != "place":
        source = [obj for obj in source if _head_detection_present(obj)]
        if not source:
            raise ValueError("head_target_missing")
    if task != "grasp" and not any(_head_detection_present(obj) for obj in destination):
        raise ValueError("head_destination_missing")
    source_views = [source_view_evidence(observation, obj) for obj in source]
    wrists_ready = task == "place" or all(item["views_ready"] for item in source_views)
    return {"head_targets_present": True, "views_ready": wrists_ready,
            "source_ids": [obj["id"] for obj in source], "source_views": source_views}


def standalone_place_view_evidence(observation, destination_attributes):
    """Require a current measured destination without repeating empty-hand head scope."""
    cameras = set()
    observation_id = observation["observation_id"]
    for obj in matching_instances(observation, destination_attributes, destination=True):
        points = np.asarray(obj.get("points_world_m"), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            continue
        if any(obj.get(key, {}).get("observation_id", observation_id) != observation_id
               for key in ("point_sources", "observation_provenance")):
            continue
        for camera in obj.get("coverage", {}).get("camera_ids", []):
            view = observation.get("views", {}).get(camera)
            if camera not in {"1", "2", "3"} or not view:
                continue
            if (view.get("frame_id", observation_id) != observation_id
                    or view.get("camera_id", camera) != camera):
                continue
            try:
                intrinsics = view["intrinsics"]
                calibration = np.asarray([intrinsics[key] for key in ("fx", "fy", "cx", "cy")], dtype=float)
                rigid_matrix(view["T_world_camera"])
                depth = np.asarray(view["depth_m"], dtype=float)
                valid = np.asarray(view["valid_depth"], dtype=bool)
                mask = np.asarray(view["instance_masks"][obj["id"]], dtype=bool)
            except (KeyError, TypeError, ValueError):
                continue
            if (not np.isfinite(calibration).all() or np.any(calibration[:2] <= 0)
                    or depth.ndim != 2 or valid.shape != depth.shape or mask.shape != depth.shape):
                continue
            if np.any(mask & valid & np.isfinite(depth) & (depth > 0)):
                cameras.add(camera)
    if not cameras:
        raise ValueError("destination_missing")
    return {"views_ready": True, "destination_camera_ids": sorted(cameras)}


@lru_cache(maxsize=1)
def observation_joint_limits():
    path = Path(__file__).resolve().parents[2] / "assets/marvin_urdf/marvin_with_grippers.urdf"
    robot = ET.parse(path).getroot()
    limits = {}
    for arm, suffix in (("A", "L"), ("B", "R")):
        arm_limits = []
        for index in range(1, 8):
            limit = robot.find(f"joint[@name='Joint{index}_{suffix}']/limit")
            arm_limits.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))
        limits[arm] = tuple(arm_limits)
    return limits


def observation_poses(settings=None):
    """Copy requested poses from a frozen full settings snapshot, validating robot limits.

    Resource calibration validates the default endpoints. Edited endpoints still require
    runtime FK, collision planning and measured RGBD coverage before use.
    """
    path = Path(__file__).resolve().parents[2] / "resources/observation_poses.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("model_validated"):
        raise ValueError("observation_poses_require_model_validation")
    requested = config["joints_rad"] if settings is None else settings.get("observation", config["joints_rad"])
    result = {}
    for arm, limits in observation_joint_limits().items():
        joints = requested.get(arm, config["joints_rad"][arm])
        if not isinstance(joints, (list, tuple)) or len(joints) != 7:
            raise ValueError(f"observation_requires_seven_joints:{arm}")
        for index, (value, (lower, upper)) in enumerate(zip(joints, limits), start=1):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"observation_requires_finite_joint:{arm}:{index}")
            if not lower <= value <= upper:
                raise ValueError(f"observation_joint_out_of_range:{arm}:{index}")
        result[arm] = list(joints)
    return result

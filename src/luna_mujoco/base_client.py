"""Passive tianji_base observations and owned, idempotent execution jobs.

No controller SDK, camera device, physics engine, or mode preparation lives here.
Network failure never turns an uncertain write into a second motion request.
"""
import base64
import copy
import hashlib
import io
import json
import math
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import numpy as np
from PIL import Image

BRIDGE_PATH = "/api/extensions/jodell.robot-bridge/v1"
from .robot_contracts import CAMERA_ROLES, RGBDCaptureIncomplete, RobotFeedbackUnavailable, detached_capture, immutable


class RobotExecutionError(RuntimeError):
    def __init__(self, result):
        super().__init__('robot_execution_failed:' + str(result.get('error')))
        self.details = copy.deepcopy({key: result.get(key) for key in (
            'execution_id', 'submitted_points', 'point_count', 'message',
            'error_details', 'last_frame_timing', 'segments_timing', 'segment_index',
            'dispatched', 'dispatch_uncertain', 'cleanup')})


class RobotExecutionTimeout(TimeoutError):
    def __init__(self, result, *, phase, elapsed, budget):
        super().__init__('robot_execution_timeout')
        self.details = copy.deepcopy(result)
        self.details.update(timeout_phase=phase, timeout_elapsed_s=elapsed, timeout_budget_s=budget)


def recording_execution_timeout(grippers, point_count, dt, arrival_timeout):
    """Bound all buffered segments from the request, before any motion is sent."""
    boundaries = [0]
    previous = {arm: path[0] for arm, path in grippers.items()}
    events = 1  # The initial gripper target is confirmed before the first path.
    for index in range(1, point_count):
        current = {arm: path[index] for arm, path in grippers.items()}
        if any(abs(current[arm] - previous[arm]) > 1e-6 for arm in current):
            boundaries.append(index)
            previous = current
            events += 1
    if boundaries[-1] < point_count - 1 or point_count == 1:
        boundaries.append(point_count - 1)
    segments = len(boundaries) - 1
    duration = sum(max(5, math.ceil((end-start)*dt/.02 - 1e-9)) * .02
                   for start, end in pairwise(boundaries))
    # Initial gripper execution starts the fixed execution deadline; every later
    # segment validation therefore needs an allowance inside that same deadline.
    return duration + 120.*segments + arrival_timeout*(segments+events) + 10.




@lru_cache(maxsize=1)
def gripper_limits():
    path = Path(__file__).resolve().parents[2] / "assets/marvin_urdf/marvin_with_grippers.urdf"
    root = ET.parse(path).getroot()
    result = {}
    for arm, name in (("A", "gripL_finger_joint"), ("B", "gripR_finger_joint")):
        limit = root.find(f"joint[@name='{name}']/limit")
        result[arm] = [float(limit.attrib[key]) for key in ("lower", "upper")]
    return result


def validated_pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0., 0., 0., 1.])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5)):
        raise ValueError('invalid_tool_or_camera_pose')
    return pose


def normalize_arms(raw):
    """Normalize measured joints without invoking a kinematics provider."""
    arms = {}
    for arm in ("A", "B"):
        value = raw.get(arm, {})
        joints = np.asarray(value.get("joints_deg", []), dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("robot_joint_feedback_unavailable:" + arm)
        arms[arm] = {"joints_rad": np.deg2rad(joints).tolist(),
                     "gripper_limits_rad": list(gripper_limits()[arm])}
    return arms




def decode_image(value, *, depth=False):
    raw = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.width > 4096 or image.height > 4096:
            raise ValueError("camera_image_too_large")
        pixels = np.asarray(image if depth else image.convert("RGB")).copy()
    if depth and (pixels.ndim != 2 or pixels.dtype.kind not in "ui" or pixels.max() > 65535):
        raise ValueError("depth_must_be_uint16_png")
    return pixels


def rectify_rgbd(rgb, depth, matrix, calibration):
    """Inverse-sample registered RGB and axial depth with the measured lens model.

    Brown-Conrady/rational convention follows OpenCV calib3d's public camera
    model. Depth uses nearest measured samples; unsupported models are rejected.
    """
    if calibration.get("rectified") is True or calibration.get("lens_model") == "none":
        return rgb, depth
    model = calibration.get("lens_model")
    coefficients = calibration.get("distortion_coefficients")
    keys = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
    if model not in ("brown_conrady", "brown_conrady_k6") or not isinstance(coefficients, dict):
        raise ValueError("camera_rectification_unverified")
    try:
        values = tuple(float(coefficients[key]) for key in keys)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("camera_rectification_parameters_invalid") from error
    if not np.isfinite(values).all():
        raise ValueError("camera_rectification_parameters_invalid")
    rows, columns = rectification_map(depth.shape, tuple(matrix.ravel()), model, values)
    from scipy.ndimage import map_coordinates
    corrected_rgb = np.stack([map_coordinates(rgb[..., channel], [rows, columns], order=1,
        mode="constant", cval=0, prefilter=False) for channel in range(3)], axis=-1)
    corrected_depth = map_coordinates(depth, [rows, columns], order=0, mode="constant", cval=0,
                                      prefilter=False)
    return corrected_rgb, corrected_depth


@lru_cache(maxsize=8)
def rectification_map(shape, intrinsics, model, coefficients):
    matrix = np.asarray(intrinsics).reshape(3, 3)
    rows, columns = np.indices(shape, dtype=float)
    x, y = (columns - matrix[0, 2]) / matrix[0, 0], (rows - matrix[1, 2]) / matrix[1, 1]
    k1, k2, k3, k4, k5, k6, p1, p2 = coefficients
    radius = x * x + y * y
    radial = 1 + k1 * radius + k2 * radius ** 2 + k3 * radius ** 3
    if model == "brown_conrady_k6":
        denominator = 1 + k4 * radius + k5 * radius ** 2 + k6 * radius ** 3
        if np.any(np.abs(denominator) < 1e-9):
            raise ValueError("camera_rectification_singular")
        radial /= denominator
    distorted_x = x * radial + 2 * p1 * x * y + p2 * (radius + 2 * x * x)
    distorted_y = y * radial + p1 * (radius + 2 * y * y) + 2 * p2 * x * y
    return (distorted_y * matrix[1, 1] + matrix[1, 2], distorted_x * matrix[0, 0] + matrix[0, 2])


def decode_capture(payload, *, generation, camera_ids=None):
    """Decode the requested calibrated RGBD views and verify their shared exposure identity."""
    if payload.get("ok") is not True or payload.get("complete") is not True:
        raise RGBDCaptureIncomplete(payload.get("errors", []), readiness=payload.get("readiness"),
                                    synchronization=payload.get("synchronization"))
    arms = normalize_arms(payload["arms"])
    add_capture_grippers(arms, payload.get("grippers", {}).get("arms", {}), payload["session_id"])
    identifier = payload["capture_id"]
    selected = list(CAMERA_ROLES) if camera_ids is None else camera_ids
    if (not isinstance(selected, list) or not selected or len(selected) != len(set(selected))
            or any(camera_id not in CAMERA_ROLES for camera_id in selected)):
        raise ValueError("invalid_camera_selection")
    sync = payload.get('synchronization', {})
    if set(payload.get('cameras', {})) != {CAMERA_ROLES[c] for c in selected}:
        raise ValueError('rgbd_camera_set_mismatch')
    if len(selected) > 1:
        frame_id = sync.get('frame_set_id')
        if (sync.get('matched') is not True or sync.get('same_frame') is not True
                or not isinstance(frame_id, str) or not frame_id
                or any(v.get('frame_set_id') != frame_id for v in payload['cameras'].values())):
            raise ValueError('three_camera_same_frame_unverified')
    views = {}
    for camera_id in selected:
        role = CAMERA_ROLES[camera_id]
        view = payload["cameras"][role]
        if (view.get("ok") is not True or view.get("depth_unit") != "mm"
                or view.get("depth_alignment") != "color"):
            raise ValueError("registered_rgbd_required:" + role)
        rgb = decode_image(view["rgb_jpeg_b64"])
        depth = decode_image(view["depth_png_b64"], depth=True).astype(np.float32) / 1000.
        matrix = np.asarray(view["rgb_intrinsics"], dtype=float)
        measured = normalize_arms(view["arms_at_capture"])
        add_capture_grippers(measured, view.get("grippers_at_capture", {}), payload["session_id"])
        transform = np.asarray(view["T_base_camera_m"], dtype=float)
        width, height = view["resolution"]
        if (rgb.shape != (height, width, 3) or depth.shape != (height, width)
                or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or min(matrix[0, 0], matrix[1, 1]) <= 0
                or transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-5)):
            raise ValueError("invalid_rgbd_calibration:" + role)
        rgb, depth = rectify_rgbd(rgb, depth, matrix, view)
        if not math.isfinite(view["telemetry_offset_ms"]) or abs(view["telemetry_offset_ms"]) > 250:
            raise ValueError("camera_telemetry_not_aligned:" + role)
        calibration = {key: view.get(key) for key in
                       ("rgb_intrinsics", "depth_intrinsics", "serial_number", "session_generation",
                        "lens_model", "distortion_coefficients", "rectified", "resolution")}
        # Wrist world pose changes with measured joints; base's geometry revision
        # binds the static published extrinsics and robot model instead.
        calibration.update(camera_role=role, geometry_revision=payload.get("geometry_revision"))
        views[camera_id] = {
            "camera_id": camera_id, "frame_id": identifier, "sequence": view["sequence"], "robot_generation": generation,
            "capture_time": view["capture_us"] / 1_000_000,
            "calibration_id": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
            'geometry_revision': payload['geometry_revision'],
            "rgb": immutable(rgb), "depth_m": immutable(depth), "valid_depth": immutable(depth > 0),
            "intrinsics": {"fx": matrix[0, 0], "fy": matrix[1, 1], "cx": matrix[0, 2],
                           "cy": matrix[1, 2], "width": width, "height": height},
            "T_world_camera": transform.tolist(), "arms": measured,
            "capture_context": {"observation_id": identifier, "session_id": payload["session_id"],
                "capture_time": view["capture_us"] / 1_000_000,
                "gripper_joint_positions_rad": {arm: measured[arm]["gripper_position_rad"] for arm in measured}},
            "robot_joints_deg": {arm: list(view["arms_at_capture"][arm]["joints_deg"]) for arm in measured},
            "registered_to": "rgb", "source_metadata": {key: view.get(key) for key in
                ("sequence", "session_generation", "capture_us", "rgb_capture_us", "depth_capture_us",
                 "timestamp_source", "serial_number", "telemetry_mono_us", "telemetry_offset_ms",
                 "source", "simulation_instance_id", "scene_revision", "calibration_revision", "frame_set_id")}}
    return {"observation_id": identifier, "generation": generation, "source": "rgbd",
            "provider": "tianji_base", "session_id": payload["session_id"],
            "geometry_revision": payload.get("geometry_revision"),
            "bridge_instance_id": payload.get("bridge_instance_id"),
            "readiness": payload["readiness"], "synchronization": payload["synchronization"],
            "time": max(view['capture_time'] for view in views.values()), "arms": arms, "views": views}


def add_capture_grippers(arms, measurements, session_id):
    for arm, feedback in arms.items():
        value = measurements.get(arm, {})
        fraction = value.get("closed_fraction")
        if (value.get("fresh") is not True or value.get("session_id") != session_id
                or fraction is None or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError("capture_gripper_feedback_unavailable:" + arm)
        lower, upper = gripper_limits()[arm]
        position = value.get('joint_position_rad', lower + fraction * (upper - lower))
        if not math.isfinite(position) or not lower - 1e-5 <= position <= upper + 1e-5:
            raise ValueError('capture_gripper_position_invalid:' + arm)
        feedback.update(gripper_closed_fraction=fraction, gripper_position_rad=position)
        if type(value.get("measured_at_mono_us")) is int:
            feedback["gripper_measured_time_s"] = value["measured_at_mono_us"] / 1_000_000
        target = value.get('current_target') or {}
        target_fraction = target.get('closed_fraction')
        target_time = target.get('target_mono_us')
        measured_time = value.get('measured_at_mono_us')
        if (target.get('session_id') == session_id
                and type(target_fraction) in (int, float) and math.isfinite(target_fraction)
                and 0 <= target_fraction <= 1 and type(target_time) is int
                and type(measured_time) is int and 0 < target_time <= measured_time):
            feedback.update(gripper_command_position_rad=lower + target_fraction * (upper - lower),
                            gripper_closed_command=target_fraction == 1.)


class BaseClient:
    def __init__(self, url, *, public_url=None, transport=None, poll_interval=.05, stale_after=.75):
        self.endpoint = url.rstrip('/')
        self.public_url = (public_url or url).rstrip('/')
        self.client = httpx.Client(base_url=url.rstrip("/") + BRIDGE_PATH + "/", transport=transport,
                                   timeout=httpx.Timeout(10., connect=2.), trust_env=False)
        self.client_id = "tianji-luna-" + uuid.uuid4().hex
        self.poll_interval, self.stale_after = poll_interval, stale_after
        self._lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._motion_lock = threading.RLock()
        self._preparation_lock = threading.RLock()
        self._preparation = None
        self._motion = None
        self._vla = None
        self._motion_error = None
        self._motion_end = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None
        self._quit = threading.Event()
        self._cancelled = threading.Event()
        self._submitted = threading.Event()
        self._thread = None
        self._raw = {}
        self._received = 0.
        self._generation = 1
        self._identity = None
        self._error = "base_not_polled"
        self._capture = None
        self._pending = None
        self._history = {arm: deque(maxlen=100) for arm in ("A", "B")}
        self._gripper_commands = {}

    def _request(self, method, path, **kwargs):
        response = self.client.request(method, path, **kwargs)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("invalid_robot_bridge_response")  # noqa: TRY004 - invalid network data.
        return value

    def viewer_manifest(self):
        path = '/api/extensions/jodell.urdf-viewer/v1'
        descriptor = self._request('GET', self.endpoint + path + '/embed')
        return {'renderer': descriptor['renderer'], 'model': descriptor['model'],
                'revision': self.robot_snapshot().get('geometry_revision'),
                'source': 'tianji_base', 'base_url': self.public_url,
                'viewer_url': self.public_url + path + '/embed-view?appearance=neutral'}

    def viewer_scene(self):
        scene = self._request('GET', self.endpoint + '/api/extensions/jodell.urdf-viewer/v1/measured-scene')
        if scene.get('ok') is not True or scene.get('source') != 'measured_robot_feedback':
            raise ValueError('base_twin_feedback_unavailable')
        if scene.get('environment') or scene.get('objects'):
            raise ValueError('base_twin_must_exclude_authored_environment')
        return scene

    def viewer_asset(self, path):
        url = urljoin(self.endpoint + '/', path)
        origin, target = urlparse(self.endpoint), urlparse(url)
        if (origin.scheme, origin.netloc) != (target.scheme, target.netloc):
            raise ValueError('viewer_asset_must_be_published_by_base')
        response = self.client.get(url)
        response.raise_for_status()
        return response.content, response.headers.get('content-type', 'application/octet-stream')

    def viewer_configuration(self, joints_deg, grippers):
        """Evaluate display geometry only; never update the live robot or Base scene."""
        result = self._request('POST', self.endpoint + '/api/extensions/jodell.model.marvin/v1/scene/evaluate',
                               json={'joints': joints_deg, 'grippers': grippers})
        if result.get('ok') is False or not result.get('nodes'):
            raise ValueError('base_robot_display_configuration_unavailable')
        return {'ok': True, 'nodes': result['nodes'], 'link_transforms': result.get('link_transforms', {}),
                'joints': joints_deg, 'grippers': grippers, 'source': 'evaluated_robot_configuration',
                'ghost': None, 'preview': None, 'hidden': [], 'frames': [], 'cameras': [],
                'tools': [], 'markers': {}, 'pointclouds': [], 'environment': {}}

    def refresh(self):
        value = self._request("GET", "state")
        if value.get("schema_version") != 1 or value.get("ok") is not True:
            raise ValueError("incompatible_robot_bridge")
        identity = (value.get("bridge_instance_id"), value.get("session_id"), value.get("stop_epoch"),
                    value.get("geometry_revision"), value.get("world_revision"))
        with self._lock:
            if self._identity is not None and identity != self._identity:
                self._generation += 1
                self._capture = None
                self._gripper_commands.clear()
                for history in self._history.values():
                    history.clear()
            self._identity = identity
            self._raw, self._received, self._error = copy.deepcopy(value), time.monotonic(), None
        return value

    def start(self):
        if self._thread is not None:
            return
        def poll():
            while not self._quit.is_set():
                try:
                    self.refresh()
                except (httpx.HTTPError, ValueError, KeyError) as error:
                    with self._lock:
                        self._error = type(error).__name__ + ":" + str(error)[:200]
                self._quit.wait(self.poll_interval)
        self._thread = threading.Thread(target=poll, name="base-feedback", daemon=True)
        self._thread.start()

    def robot_snapshot(self, *, include_tcp=False):
        if include_tcp:
            raise ValueError("tcp_requires_runtime_adapter")
        with self._lock:
            raw = copy.deepcopy(self._raw)
            age = time.monotonic() - self._received
            error = self._error
            arms = {}
            if raw.get("connected") and age <= self.stale_after and not error:
                try:
                    arms = normalize_arms(raw.get("arms", {}))
                except ValueError as failure:
                    error = str(failure)
            self._add_gripper_feedback(arms, raw)
            ready = bool(arms and (raw.get("automatic_enable_ready") or raw.get("ready"))
                         and not self._pending and self._vla is None and not self._motion_error
                         and (self._motion is None or self._motion.get("state") == "enabled"))
            active_commands = [self._pending['request_id']] if self._pending else []
            if raw.get('active_execution_id'):
                active_commands.append(str(raw['active_execution_id']))
            if raw.get('current_motion'):
                active_commands.append('base_motion_active')
            return {"generation": self._generation, "time": (raw.get("mono_us") or 0) / 1_000_000,
                    "connected": bool(raw.get("connected")), "available": bool(arms), "ready": ready,
                    "feedback_id": str(raw.get("mono_us")), "arms": arms,
                    "session_id": raw.get("session_id"), "bridge_instance_id": raw.get("bridge_instance_id"),
                    "geometry_revision": raw.get("geometry_revision"), "world_revision": raw.get("world_revision"),
                    "stop_epoch": raw.get("stop_epoch"), "kinematics_sha256": raw.get('kinematics_sha256'),
                    "runtime_mode": raw.get('runtime_mode') if not error and age <= self.stale_after else 'unknown',
                    "robot_ip": raw.get('robot_ip'),
                    "error": error or ("robot_feedback_stale" if age > self.stale_after else None),
                    "readiness_errors": raw.get("readiness_errors", []),
                    "can_prepare": raw.get("can_prepare") is True and age <= self.stale_after and not error,
                    "preparation_errors": [str(item.get("message") or item.get("error")) if isinstance(item, dict) else str(item)
                                           for item in raw.get("preparation_errors", [])],
                    "preparation": copy.deepcopy(raw.get("preparation")),
                    "motion": copy.deepcopy(self._motion), "motion_error": self._motion_error,
                    "current_motion": raw.get("current_motion"), "enable_state": raw.get("enable_state"),
                    "active_execution_id": raw.get("active_execution_id"),
                    "vla_run": copy.deepcopy(self._vla) or raw.get("vla_run"),
                    "automatic_enable_ready": raw.get("automatic_enable_ready"),
                    "active_commands": sorted(set(active_commands))}

    def _add_gripper_feedback(self, arms, raw):
        # The public gripper schema is normalized here, never inferred from targets.
        for arm, feedback in arms.items():
            value = raw.get("grippers", {}).get("arms", {}).get(arm, {})
            fraction, timestamp = value.get("closed_fraction"), value.get("measured_at_mono_us")
            if (value.get("fresh") is not True or fraction is None or timestamp is None
                    or value.get("session_id") != raw.get("session_id")
                    or not math.isfinite(fraction) or not 0 <= fraction <= 1):
                continue
            lower, upper = gripper_limits()[arm]
            position = lower + fraction * (upper - lower)
            t = timestamp / 1_000_000
            history = self._history[arm]
            if not history or t > history[-1][0]:
                history.append([t, position])
            feedback.update(gripper_closed_fraction=fraction, gripper_position_rad=position, gripper_measured_time_s=t,
                            gripper_position_samples=[list(row) for row in history if t - row[0] <= 1.1],
                            gripper_closed_command=False)
            target = value.get('current_target') or {}
            target_fraction = target.get('closed_fraction')
            target_time = target.get('target_mono_us')
            if (target.get('session_id') != raw.get('session_id')
                    or type(target_fraction) not in (int, float) or not math.isfinite(target_fraction)
                    or not 0 <= target_fraction <= 1 or type(target_time) is not int
                    or not 0 < target_time <= timestamp
                    or any(raw.get(key) is None for key in ('session_id', 'bridge_instance_id', 'geometry_revision'))):
                continue
            command_position = lower + target_fraction * (upper - lower)
            # The provider's held-until-replaced target is independent of the
            # measured position. Bind its command echo to this bridge identity.
            feedback.update(gripper_command_position_rad=command_position,
                            gripper_closed_command=target_fraction == 1.,
                            gripper_closing_command=target_fraction == 1.,
                            gripper_command_evidence={
                                'source': 'robot_bridge_current_target', 'generation': self._generation,
                                **{key: raw[key] for key in ('session_id', 'bridge_instance_id', 'geometry_revision')},
                                'target_mono_us': target_time, 'measured_at_mono_us': timestamp,
                                'closed_fraction': target_fraction, 'position_rad': command_position})

    def snapshot(self, *, include_tcp=False):
        return self.robot_snapshot(include_tcp=include_tcp)

    def _capture_payload(self, camera_ids=None):
        selected = list(CAMERA_ROLES) if camera_ids is None else camera_ids
        if (not isinstance(selected, list) or not selected or len(selected) != len(set(selected))
                or any(camera_id not in CAMERA_ROLES for camera_id in selected)):
            raise ValueError("invalid_camera_selection")
        payload = {"camera_ids": [CAMERA_ROLES[c] for c in selected], "max_age_ms": 500,
                   "require_stable": False, "require_same_frame": len(selected) > 1, "max_skew_ms": 20}
        return self._request("POST", "capture", json=payload)

    def capture_rgbd(self, camera_ids=None):
        self.refresh()
        before = self.robot_snapshot(include_tcp=False)
        requested_at = time.monotonic()
        requested_perf_ns = time.perf_counter_ns()
        payload = self._capture_payload(camera_ids)
        self.refresh()
        after = self.robot_snapshot(include_tcp=False)
        if (before["generation"] != after["generation"] or payload.get("session_id") != before["session_id"]
                or payload.get("bridge_instance_id") != before["bridge_instance_id"]
                or payload.get("geometry_revision") != before["geometry_revision"]):
            raise ValueError("robot_session_changed_during_capture")
        result = decode_capture(payload, generation=before["generation"], camera_ids=camera_ids)
        # Use request start as a conservative local age, while preserving the
        # provider's exposure time for ordering against physical feedback.
        # Release verification also uses the local request-start clock; this
        # supplements, and never replaces, the sensor exposure timestamp.
        result.update(captured_monotonic_s=requested_at, captured_perf_counter_ns=requested_perf_ns,
                      active_commands=sorted(set(before['active_commands'] + after['active_commands'])))
        with self._lock:
            if self._generation != result["generation"]:
                raise ValueError("robot_session_changed_during_capture")
            if set(result["views"]) == set(CAMERA_ROLES):
                self._capture = detached_capture(result)
        return result

    def latest_rgbd(self):
        with self._lock:
            return detached_capture(self._capture)

    def preview_rgb(self):
        packet = self._request('GET', 'preview')
        views, identifiers, errors = {}, [], {}
        state = self.robot_snapshot()
        for camera, role in CAMERA_ROLES.items():
            item = packet['cameras'][role]
            if item.get('ok'):
                meta = item.get('metadata', {})
                views[camera] = {'rgb': decode_image(item['rgb_jpeg_b64']), 'source_metadata': meta}
                identifiers.append((camera, meta.get('session_generation'), meta.get('sequence'), meta.get('capture_us')))
            else:
                errors[camera] = [item]
        identifier = hashlib.sha256(json.dumps(identifiers).encode()).hexdigest()[:24]
        return {'preview_frame_id': identifier, 'generation': state['generation'],
                'captured_monotonic_s': time.monotonic(), 'time': state['time'],
                'source': 'diagnostic_rgb_preview', 'is_task_observation': False, 'views': views, 'errors': errors}

    def begin_motion(self, kind, generation, sample_dt_s=.02, mode="position", arms=None, *, request_id=None):
        """Ask base to prepare one whole movement; retain uncertain begin identity."""
        if not math.isfinite(sample_dt_s) or not .01 <= sample_dt_s <= 1:
            raise ValueError("invalid_motion_sample_dt")
        if kind not in {"observe", "grasp", "place", "recording"}:
            raise ValueError("invalid_motion_kind")
        if mode not in {"position", "impedance"}:
            raise ValueError("invalid_motion_mode")
        arms = ["A", "B"] if arms is None else arms
        if (not isinstance(arms, list) or not arms or any(arm not in ("A", "B") for arm in arms)
                or len(set(arms)) != len(arms)):
            raise ValueError("invalid_motion_arms")
        with self._motion_lock:
            with self._lock:
                if self._motion is not None or self._vla is not None or self._pending:
                    raise RuntimeError("motion_cleanup_unresolved")
            if self._quit.is_set():
                raise RuntimeError("robot_client_closed")
            self.refresh()
            with self._lock:
                state = self.check_generation(generation)
                if not state["ready"]:
                    raise ValueError("robot_not_ready")
                identity = {"client_id": self.client_id, "request_id": request_id or uuid.uuid4().hex,
                            "bridge_instance_id": self._raw["bridge_instance_id"],
                            "stop_epoch": self._raw["stop_epoch"],
                            "expected_geometry_revision": self._raw["geometry_revision"],
                            "expected_session_id": self._raw["session_id"]}
                self._motion = {**identity, "kind": kind, "state": "begin_pending"}
                self._motion_error = None
                self._motion_end = None
            try:
                try:
                    result = self._request("POST", "motions/begin",
                                           json={**identity, "kind": kind, "sample_dt_s": sample_dt_s,
                                                 "mode": mode, "arms": arms})
                except httpx.HTTPStatusError as error:
                    rejected = error.response.json()
                    if (error.response.status_code in (400, 401, 403, 404, 409, 422)
                            and rejected.get("ok") is False and not rejected.get("motion_id")):
                        with self._lock:
                            self._motion = None
                        raise RuntimeError("motion_begin_rejected:" + str(rejected.get("error"))) from error
                    result = self._lookup_motion(identity)
                except (httpx.TransportError, ValueError):
                    result = self._lookup_motion(identity)
                with self._lock:
                    if result.get("motion_id"):
                        self._motion.update(motion_id=result["motion_id"], state=result.get("state"))
                    if (result.get("disable_confirmed") is True
                            and result.get("state") in {"disabled", "failed"}):
                        self._motion = None
                        self._motion_end = copy.deepcopy(result)
                    if (result.get("ok") is not True or result.get("state") != "enabled"
                            or not result.get("motion_id")):
                        raise RuntimeError("motion_begin_failed:" + str(result.get("error")))
                    self.check_generation(generation)
                self._heartbeat_stop = threading.Event()
                self._heartbeat_thread = threading.Thread(target=self._heartbeat_motion,
                                                          args=(dict(self._motion), self._heartbeat_stop),
                                                          name="base-motion-heartbeat", daemon=True)
                self._heartbeat_thread.start()
                return result
            except Exception as error:
                with self._lock:
                    if self._motion is not None:
                        self._motion_error = str(error)
                raise

    def _lookup_motion(self, identity):
        return self._request("GET", "motions/by-request", params={
            "client_id": self.client_id, "request_id": identity["request_id"]})

    @staticmethod
    def _motion_identity(motion):
        return {key: motion[key] for key in
                ("client_id", "bridge_instance_id", "expected_session_id", "stop_epoch")}

    def _heartbeat_motion(self, motion, stop):
        while not stop.wait(1.):
            try:
                result = self._request("POST", "motions/" + motion["motion_id"] + "/heartbeat",
                                       json=self._motion_identity(motion))
                if result.get("ok") is not True or result.get("state") != "enabled":
                    raise RuntimeError("motion_lease_not_enabled")
            except (httpx.HTTPError, ValueError, RuntimeError) as error:
                with self._motion_lock:
                    with self._lock:
                        if self._motion is None or self._motion["request_id"] != motion["request_id"]:
                            return
                        self._motion_error = "motion_heartbeat_failed:" + str(error)
                    try:
                        self.stop("heartbeat_failed")
                    except (httpx.HTTPError, ValueError, RuntimeError) as cleanup_error:
                        # Base's independent watchdog remains active; retain cleanup identity.
                        with self._lock:
                            self._motion_error = str(cleanup_error)
                return

    def end_motion(self, reason="completed", *, expected_id=None, expected_request=None):
        """Close the base scope, including a begin whose response was lost."""
        with self._motion_lock:
            with self._lock:
                motion = copy.deepcopy(self._motion)
                if expected_request and (motion or {}).get("request_id") != expected_request:
                    return {"ok": True, "hardware_command_sent": False}
                if expected_id and motion is not None and motion.get("motion_id") != expected_id:
                    raise RuntimeError("motion_end_owner_changed")
            self._heartbeat_stop.set()
            if motion is None:
                return copy.deepcopy(self._motion_end) or {
                    "ok": True, "hardware_command_sent": False, "disable_confirmed": False}
            with self._lock:
                self._motion["state"] = "cleanup_pending"
                if reason != "completed" and not self._cancelled.is_set():
                    self._generation += 1
                    self._capture = None
            try:
                if not motion.get("motion_id"):
                    result = self._lookup_motion(motion)
                    if not result.get("motion_id"):
                        raise RuntimeError("motion_begin_unresolved")
                    motion["motion_id"] = result["motion_id"]
                    with self._lock:
                        self._motion["motion_id"] = result["motion_id"]
                try:
                    result = self._request("POST", "motions/" + motion["motion_id"] + "/end",
                                          json={**self._motion_identity(motion), "reason": reason})
                except (httpx.HTTPError, ValueError):
                    result = self._request("GET", "motions/" + motion["motion_id"])
                if (result.get("disable_confirmed") is not True
                        or result.get("state") not in {"disabled", "failed"}):
                    raise RuntimeError("motion_disable_unconfirmed:" + str(result.get("error")))
                with self._lock:
                    self._motion = None
                    self._pending = None
                    self._motion_error = None
                    self._motion_end = copy.deepcopy(result)
                    self._cancelled.clear()
                return result
            except Exception as error:
                with self._lock:
                    self._motion_error = str(error)
                raise

    def check_generation(self, generation):
        state = self.robot_snapshot(include_tcp=False)
        if state["generation"] != generation or self._cancelled.is_set():
            raise ValueError("robot_generation_changed")
        if not state["available"]:
            # A busy perception/recording cycle may have delayed the background read.
            # Refresh telemetry only. Never reissue enable, trajectory or gripper commands.
            try:
                self.refresh()
            except (httpx.HTTPError, ValueError, KeyError) as error:
                if self._cancelled.is_set() or self.robot_snapshot(include_tcp=False)["generation"] != generation:
                    raise ValueError("robot_generation_changed") from error
                raise RobotFeedbackUnavailable(generation, state, error) from error
            state = self.robot_snapshot(include_tcp=False)
            if state["generation"] != generation or self._cancelled.is_set():
                raise ValueError("robot_generation_changed")
            if not state["available"]:
                raise RobotFeedbackUnavailable(generation, state)
        if self._motion_error:
            raise RuntimeError(self._motion_error)
        return state

    def execute_trajectory(self, arm, positions_rad, dt, generation):
        if arm not in ("A", "B"):
            raise ValueError("arm_must_be_A_or_B")
        return self._execute_paths({arm: positions_rad}, dt, generation)

    def wait_for_gripper_feedback(self, arms, generation, *, timeout=3.):
        """Wait through mode switching for fresh, same-session measurements."""
        deadline = time.monotonic() + timeout
        while True:
            self.refresh()
            state = self.check_generation(generation)
            if all('gripper_closed_fraction' in state['arms'].get(arm, {}) for arm in arms):
                return state
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('gripper_feedback_unavailable')
            self._cancelled.wait(min(.05, remaining))

    def execute_dual_trajectory(self, paths, dt, generation):
        return self._execute_paths(paths, dt, generation)

    def execute_trajectory_program(self, paths, grippers, dt, generation, *, times_s=None, tolerance_deg=1., arrival_timeout_s=30.):
        result = self._execute_paths(paths, dt, generation, grippers=grippers, execution_profile='planned',
            times_s=times_s, tolerance_deg=tolerance_deg, arrival_timeout_s=arrival_timeout_s)
        measurements = []
        for item in result.get('gripper_feedback', []):
            arms = normalize_arms(item['arms'])
            add_capture_grippers(arms, item['grippers']['arms'], item['session_id'])
            measurements.append({'source_end_index': item['source_end_index'], 'completion': item['completion'],
                'feedback': {'generation': generation, 'session_id': item['session_id'], 'source': 'tianji_base',
                             'time': item['telemetry_mono_us'] / 1_000_000, 'arms': arms}})
        return {**result, 'gripper_feedback': measurements}

    @staticmethod
    def _check_start(arm, planned_deg, state, stage):
        from .robot_errors import CHECKED_START_TOLERANCE_DEG, CheckedStartChanged
        measured = np.rad2deg(state["arms"][arm]["joints_rad"])
        delta = measured - planned_deg
        index = int(np.argmax(np.abs(delta)))
        if abs(delta[index]) > CHECKED_START_TOLERANCE_DEG:
            raise CheckedStartChanged(CheckedStartChanged.code, {
                'stage': stage, 'arm': arm, 'joint': index + 1, 'delta_deg': delta.tolist(),
                'maximum_drift_deg': float(abs(delta[index])), 'limit_deg': CHECKED_START_TOLERANCE_DEG,
                'checked_start_deg': planned_deg.tolist(), 'measured_deg': measured.tolist(),
                'feedback_time': state['time'], 'generation': state['generation'], 'dispatched': False})

    def _execute_paths(self, paths, dt, generation, grippers=None, *, execution_profile="planned",
                       times_s=None, tolerance_deg=1., arrival_timeout_s=30.):
        if (not isinstance(paths, dict) or not paths or not set(paths) <= {"A", "B"}
                or not math.isfinite(dt) or not .01 <= dt <= 1):
            raise ValueError("invalid_trajectory_or_sample_dt")
        state = self.check_generation(generation)
        positions = {arm: np.asarray(path, dtype=float) for arm, path in paths.items()}
        for arm, path in positions.items():
            if path.ndim != 2 or path.shape[1] != 7 or not 1 <= len(path) <= 100001 or not np.isfinite(path).all():
                raise ValueError("finite_N_by_7_trajectory_required")
            self._check_start(arm, np.rad2deg(path[0]), state, "prepare_dispatch")
        point_count = len(next(iter(positions.values())))
        if any(len(path) != point_count for path in positions.values()):
            raise ValueError("dual_paths_must_have_same_length")
        command = {"points_deg": {arm: np.rad2deg(path).tolist() for arm, path in positions.items()},
                   "sample_dt_s": dt, "arrival_tolerance_deg": tolerance_deg, "arrival_timeout_s": arrival_timeout_s,
                   "execution_profile": execution_profile}
        timeout = point_count*dt+40.
        if times_s is not None:
            times = np.asarray(times_s, dtype=float)
            if times.shape != (point_count,) or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
                raise ValueError('invalid_trajectory_sample_times')
            command['sample_times_s'] = times.tolist()
            timeout = float(times[-1]) + arrival_timeout_s + 10.
        if grippers is not None:
            if not isinstance(grippers, dict) or not grippers or not set(grippers) <= set(paths):
                raise ValueError("invalid_recorded_grippers")
            values = {arm: np.asarray(path, dtype=float) for arm, path in grippers.items()}
            for path in values.values():
                if (path.shape != (point_count,) or not np.isfinite(path).all()
                        or np.any(path < 0) or np.any(path > 1)):
                    raise ValueError("invalid_recorded_gripper_path")
            command["gripper_paths"] = {arm: path.tolist() for arm, path in values.items()}
            timeout = recording_execution_timeout(values, point_count, dt,
                                                   command["arrival_timeout_s"])
            if times_s is not None:
                timeout += max(0., float(times[-1]) - (point_count - 1) * dt)
        return self._execute("trajectory", command, generation, timeout=timeout)

    def set_gripper(self, arm, closed, generation, *, arrival_timeout_s=10.):
        if arm not in ("A", "B") or not isinstance(closed, bool):
            raise ValueError("invalid_gripper_command")
        self.wait_for_gripper_feedback([arm], generation)
        targets = {arm: float(closed)}
        result = self._execute("gripper-command", {"targets": targets, "arrival_tolerance": .02,
            "arrival_timeout_s": arrival_timeout_s}, generation, timeout=arrival_timeout_s + 10.)
        with self._lock:
            self._gripper_commands[arm] = closed
        return result

    def _execute(self, path, command, generation, *, timeout):
        with self._command_lock:
            self.refresh()
            with self._lock:
                state = self.check_generation(generation)
                if (self._motion is None or self._motion.get("state") != "enabled"
                        or not self._motion.get("motion_id")):
                    raise RuntimeError("motion_scope_required")
                if not state["ready"] or self._pending:
                    raise ValueError("robot_not_ready_or_execution_unresolved")
                if path == "trajectory":
                    for arm, points in command["points_deg"].items():
                        self._check_start(arm, np.asarray(points[0]), state, "before_post")
                identity = {"client_id": self.client_id, "request_id": uuid.uuid4().hex,
                            "bridge_instance_id": self._raw["bridge_instance_id"],
                            "stop_epoch": self._raw["stop_epoch"],
                            "expected_geometry_revision": self._raw["geometry_revision"],
                            "expected_session_id": self._raw["session_id"]}
                identity["motion_id"] = self._motion["motion_id"]
                self._pending = dict(identity)
                self._submitted.clear()
            try:
                try:
                    result = self._request("POST", path, json={**identity, **command})
                except httpx.HTTPStatusError as error:
                    try:
                        rejected = error.response.json()
                    except ValueError:
                        rejected = {}
                    definite_rejection = (error.response.status_code in (400, 401, 403, 404, 409, 422)
                        or (rejected.get("accepted") is False and rejected.get("dispatched") is False))
                    if (definite_rejection
                            and rejected.get("ok") is False and rejected.get("error")
                            and not rejected.get("execution_id")):
                        with self._lock:
                            self._pending = None
                        raise RuntimeError("robot_execution_rejected:" + str(rejected["error"])) from error
                    result = self._request("GET", "executions/by-request", params={
                        "client_id": self.client_id, "request_id": identity["request_id"]})
                except httpx.TransportError:
                    # Query uncertainty. Never repeat the POST, even with an idempotency key.
                    result = self._request("GET", "executions/by-request", params={
                        "client_id": self.client_id, "request_id": identity["request_id"]})
                if result.get("ok") is False and not result.get("execution_id"):
                    with self._lock:
                        self._pending = None
                    raise RuntimeError("robot_execution_rejected:" + str(result.get("error")))
                if result.get("ok") is not True or not result.get("execution_id"):
                    raise RuntimeError("robot_execution_rejected:" + str(result.get("error")))
                with self._lock:
                    if self._pending is None:
                        raise RuntimeError("motion_ended_during_submission")
                    self._pending["execution_id"] = result["execution_id"]
                self._submitted.set()
                phase_started = time.monotonic()
                preparing = path == "trajectory" and result["state"] in {"queued", "validating"}
                budget = 120. if preparing else timeout
                deadline = phase_started + budget
                waypoint_feedback = {}
                while True:
                    self.check_generation(generation)
                    with self._lock:
                        if self._motion is None or self._motion.get("state") != "enabled":
                            raise RuntimeError("motion_scope_ended")
                    if result["state"] == "arrived":
                        if not (result.get("physical_arrival_confirmed") or result.get("physical_completion_confirmed")):
                            raise RuntimeError("execution_completion_unconfirmed")
                        with self._lock:
                            self._pending = None
                        self.refresh()
                        return {**result, "completed": True, "generation": generation,
                                'waypoint_feedback': list(waypoint_feedback.values())}
                    if result["state"] in {"failed", "stopped", "cleanup_pending"}:
                        raise RobotExecutionError(result)
                    now = time.monotonic()
                    if now >= deadline:
                        raise RobotExecutionTimeout(result, phase="validation" if preparing else "execution",
                                                    elapsed=now-phase_started, budget=budget)
                    if preparing and result["state"] not in {"queued", "validating"}:
                        # Validation consumes its own bounded allowance, once only. Running
                        # progress and arrival polling must never extend the execution budget.
                        preparing = False
                        phase_started = now
                        budget = timeout
                        deadline = now + budget
                    self._cancelled.wait(.05)
                    # Bracket the measured sample by the same Base execution segment.
                    # Only gripper-boundary holds provide intermediate waypoint evidence.
                    previous = result
                    self.refresh()
                    feedback = self.robot_snapshot(include_tcp=False)
                    result = self._request("GET", "executions/" + result["execution_id"])
                    index = result.get('submitted_points', 0) - 1
                    paths = command.get('gripper_paths', {})
                    # Base alternates joint paths (odd segments) and gripper
                    # events (even segments). A path being uploaded is not
                    # evidence that its end has already been reached.
                    segment = result.get('segment_index')
                    boundary = type(segment) is int and segment > 0 and segment % 2 == 0 and any(
                        0 < index < len(values) and values[index] != values[index-1] for values in paths.values())
                    if (boundary and previous.get('segment_index') == result.get('segment_index')
                            and previous.get('submitted_points') == result.get('submitted_points')
                            and previous.get('state') == result.get('state') == 'waiting'
                            and feedback.get('available')):
                        waypoint_feedback[index] = {'source_end_index': index, 'feedback': feedback}
            finally:
                self._submitted.set()
                # On failure retain identity until stop is acknowledged; block another run.

    def execute_vla(self, prompt, completion, generation):
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(completion, dict):
            raise ValueError("invalid_vla_request")
        with self._command_lock:
            with self._motion_lock:
                self.refresh()
                with self._lock:
                    self.check_generation(generation)
                    if self._motion is not None or self._pending is not None or self._vla is not None:
                        raise RuntimeError("robot_operation_unresolved")
                    identity = {"client_id": self.client_id, "request_id": uuid.uuid4().hex,
                        "bridge_instance_id": self._raw["bridge_instance_id"], "stop_epoch": self._raw["stop_epoch"],
                        "expected_geometry_revision": self._raw["geometry_revision"],
                        "expected_session_id": self._raw["session_id"]}
                    self._vla = {**identity, "state": "start_pending"}
                try:
                    response = self._request("POST", "vla/start", json={**identity, "prompt": prompt,
                                             "completion": completion})
                except httpx.HTTPStatusError as error:
                    try:
                        rejected = error.response.json()
                    except ValueError:
                        rejected = {}
                    definite = (error.response.status_code in (400, 401, 403, 404, 409, 422)
                                or (rejected.get('accepted') is False and rejected.get('dispatched') is False))
                    if definite and rejected.get('ok') is False and rejected.get('error') and not rejected.get('run_id'):
                        with self._lock:
                            self._vla = None
                        raise RuntimeError('vla_start_rejected:' + str(rejected['error'])) from error
                    response = self._request('GET', 'vla/by-request', params={
                        'client_id': self.client_id, 'request_id': identity['request_id']})
                except (httpx.HTTPError, ValueError):
                    response = self._request("GET", "vla/by-request", params={
                        "client_id": self.client_id, "request_id": identity["request_id"]})
                if not response.get("run_id"):
                    if response.get("ok") is False:
                        with self._lock:
                            self._vla = None
                    raise RuntimeError("vla_start_unconfirmed:"+str(response.get("error")))
                with self._lock:
                    self._vla.update(response)
                self._heartbeat_stop = threading.Event()
                self._heartbeat_thread = threading.Thread(target=self._heartbeat_vla,
                    args=(dict(self._vla), self._heartbeat_stop), name="base-vla-heartbeat", daemon=True)
                self._heartbeat_thread.start()
            deadline = time.monotonic()+float(completion.get("timeout_s", 180.))+30.
            try:
                while True:
                    self.check_generation(generation)
                    with self._lock:
                        if self._vla is None:
                            raise RuntimeError("vla_run_stopped")
                        self._vla.update(response)
                    if response.get("disable_confirmed") is True:
                        with self._lock:
                            self._vla = None
                        if response.get("ok") is True and response.get("completed") is True:
                            self.refresh()
                            self.check_generation(generation)
                            return {**response, "generation": generation}
                        raise RuntimeError("vla_failed:"+str(response.get("error")))
                    if time.monotonic() >= deadline:
                        raise TimeoutError("vla_completion_timeout")
                    self._cancelled.wait(.05)
                    self.refresh()
                    response = self._request("GET", "vla/"+response["run_id"])
            finally:
                self._heartbeat_stop.set()
                with self._lock:
                    unresolved = copy.deepcopy(self._vla)
                if unresolved is not None:
                    self._stop_vla(unresolved, "vla_failed")

    def _heartbeat_vla(self, run, stop):
        while not stop.wait(1.):
            try:
                response = self._request("POST", "vla/"+run["run_id"]+"/heartbeat",
                                         json=self._motion_identity(run))
                if response.get("disable_confirmed"):
                    return
                if response.get("state") not in {"starting", "running", "cleanup_pending"}:
                    raise RuntimeError("vla_lease_not_active")
            except (httpx.HTTPError, ValueError, RuntimeError):
                self._cancelled.set()
                return

    def _stop_vla(self, run, reason):
        self._heartbeat_stop.set()
        if not run.get("run_id"):
            found = self._request("GET", "vla/by-request", params={
                "client_id": self.client_id, "request_id": run["request_id"]})
            if not found.get("run_id"):
                raise RuntimeError("vla_start_unresolved")
            run.update(found)
        try:
            response = self._request("POST", "vla/"+run["run_id"]+"/stop",
                json={**self._motion_identity(run), "reason": reason})
        except (httpx.HTTPError, ValueError):
            response = self._request("GET", "vla/"+run["run_id"])
        if response.get("disable_confirmed") is not True:
            raise RuntimeError("vla_disable_unconfirmed:"+str(response.get("error")))
        with self._lock:
            self._vla = None
        return response

    def prepare_observation(self, *, cancelled, progress_records):
        """Prepare through base only; cancellation never starts a motion or clears faults."""
        with self._preparation_lock:
            if cancelled.is_set():
                raise RuntimeError("preparation_cancelled")
            if self._preparation is not None:
                self._cancel_preparation()
                if self._preparation is not None:
                    raise RuntimeError("preparation_pending")
            self.refresh()
            payload = {"client_id": self.client_id, "request_id": uuid.uuid4().hex,
                       "bridge_instance_id": self.robot_snapshot(include_tcp=False)["bridge_instance_id"],
                       "robot_ip": self._raw.get("robot_ip"), "require_same_frame": True}
            self._preparation = payload
            try:
                record = self._request("POST", "preparations/start", json=payload)
            except httpx.HTTPStatusError as error:
                if 400 <= error.response.status_code < 500 and error.response.status_code != 408:
                    self._preparation = None
                    failure = error.response.json()
                    raise RuntimeError(str(failure.get("error") or failure.get("message") or "preparation_rejected")) from error
                record = self._request("GET", "preparations/by-request", params={
                    key: payload[key] for key in ("client_id", "request_id")})
            except httpx.HTTPError:
                record = self._request("GET", "preparations/by-request", params={
                    key: payload[key] for key in ("client_id", "request_id")})
            if not record.get("preparation_id"):
                raise RuntimeError("preparation_submission_unresolved")
            self._preparation = {**payload, "preparation_id": record["preparation_id"]}
        deadline = time.monotonic() + 135
        last_step = None
        while True:
            if cancelled.is_set():
                self._cancel_preparation()
                raise RuntimeError("preparation_cancelled")
            if record.get("step") != last_step:
                progress_records.put(copy.deepcopy(record))
                last_step = record.get("step")
            if record.get("state") != "preparing" and not record.get("worker_pending"):
                with self._preparation_lock:
                    self._preparation = None
                if record.get("state") != "ready" or record.get("ok") is not True:
                    raise RuntimeError(str(record.get("error") or "preparation_failed"))
                self.refresh()
                snapshot = self.robot_snapshot(include_tcp=False)
                if any(snapshot.get(k) != record.get(k) for k in ("session_id", "geometry_revision")):
                    raise RuntimeError("preparation_session_changed")
                return record
            if time.monotonic() >= deadline:
                raise RuntimeError("preparation_timeout")
            cancelled.wait(.2)
            record = self._request("GET", "preparations/" + record["preparation_id"])

    def _cancel_preparation(self):
        with self._preparation_lock:
            pending = self._preparation
            if pending is None:
                return
            if not pending.get("preparation_id"):
                found = self._request("GET", "preparations/by-request", params={
                    key: pending[key] for key in ("client_id", "request_id")})
                if not found.get("preparation_id"):
                    raise RuntimeError("preparation_submission_unresolved")
                pending["preparation_id"] = found["preparation_id"]
            result = self._request("POST", "preparations/" + pending["preparation_id"] + "/cancel",
                                   json={"client_id": self.client_id})
            if result.get("ok") is not True:
                raise RuntimeError("preparation_cancel_unconfirmed")
            # Base retains admission until its provider call drains.
            if not result.get("worker_pending"):
                self._preparation = None

    def stop(self, reason="cancelled"):
        self._cancel_preparation()
        with self._lock:
            self._generation += 1
            self._capture = None
            self._cancelled.set()
            pending = copy.deepcopy(self._pending)
        # Serializes with an in-flight begin, even between execution jobs.
        with self._motion_lock:
            if self._vla is not None:
                result = self._stop_vla(copy.deepcopy(self._vla), reason)
                self._cancelled.clear()
                return result
            if self._motion is not None:
                result = self.end_motion(reason)
                self._cancelled.clear()
                return result
        if pending is None:
            self._cancelled.clear()
            return {"ok": True, "hardware_command_sent": False, "physical_stop_confirmed": False}
        if not self._submitted.wait(12.):
            raise RuntimeError("command_submission_unresolved_stop_pending")
        with self._lock:
            pending = copy.deepcopy(self._pending)
        if pending is None:
            self._cancelled.clear()
            return {"ok": True, "hardware_command_sent": False, "physical_stop_confirmed": False}
        if not pending.get("execution_id"):
            found = self._request("GET", "executions/by-request", params={
                "client_id": self.client_id, "request_id": pending["request_id"]})
            if not found.get("execution_id"):
                raise RuntimeError("command_submission_unresolved_stop_pending")
            pending["execution_id"] = found["execution_id"]
        payload = {key: value for key, value in pending.items()
                   if key not in {"request_id", "expected_geometry_revision"}}
        result = self._request("POST", "stop", json={**payload, "reason": reason})
        if result.get("ok") is not True:
            raise RuntimeError("robot_stop_unconfirmed:" + str(result.get("error")))
        with self._lock:
            self._pending = None
            self._cancelled.clear()
        return result

    def dwell(self, duration_s, generation):
        if not math.isfinite(duration_s) or not 0 <= duration_s <= 30:
            raise ValueError("invalid_dwell_duration")
        start = time.monotonic()
        while True:
            self.check_generation(generation)
            remaining = duration_s - (time.monotonic() - start)
            if remaining <= 0:
                return {"completed": True, "duration_s": time.monotonic() - start}
            self._cancelled.wait(min(.02, remaining))

    def close(self):
        self._quit.set()
        try:
            self.stop("application_closed")
        finally:
            self._heartbeat_stop.set()
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.join(timeout=12.)
            self._quit.set()
            if self._thread is not None:
                self._thread.join(timeout=12.)
            self.client.close()

"""Bounded semantic transport for a held/released source absent from head RGB.

Ordinary requests retain the native wire format. The narrow extension uses the
public background codec only to carry depth/calibration, plus exact RGB mask
bits. Decoding restores a semantic request before any mapping or identity check.
"""

import base64
from collections.abc import Mapping
from copy import deepcopy

import numpy as np

TRANSPORT = "registered-semantic-empty-head-v1"
HEAD_TRANSPORT = "registered-semantic-targetless-head-v1"
ALL_TRANSPORT = "registered-semantic-targetless-three-v1"
CAMERAS = ("1", "2", "3")
IDENTITY_FIELDS = ("observation_set_id", "sam3_generation", "confirmed_instance_id", "selection_generation")
MAX_PIXELS = 4_194_304
MAX_INSTANCES_PER_VIEW = 200
MAX_MASK_PIXELS = 64 * 1024 * 1024


def _validate_identity(identity, *, targetless=False):
    if (not isinstance(identity, Mapping) or set(identity) != set(IDENTITY_FIELDS)
            or not isinstance(identity["observation_set_id"], str) or not identity["observation_set_id"]
            or not isinstance(identity["confirmed_instance_id"], str)
            or (bool(identity["confirmed_instance_id"]) == targetless)
            or type(identity["selection_generation"]) is not int or identity["selection_generation"] < 1
            or type(identity["sam3_generation"]) is not int
            or identity["sam3_generation"] != identity["selection_generation"]):
        raise ValueError("optional-head semantic identity is invalid")


def _mask_shape(view):
    size = view.get("rgb_size")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(value) is not int or value < 1 for value in size)
            or size[0]*size[1] > MAX_PIXELS):
        raise ValueError("optional-head RGB mask size is invalid")
    return size[1], size[0]


def _validate_views(views, identity, *, targetless=False):
    allowed = (("3",), CAMERAS) if targetless else (CAMERAS,)
    if not isinstance(views, Mapping) or tuple(views) not in allowed:
        raise ValueError("optional-head mapping requires the original ordered three cameras")
    instances = set()
    joints = None
    sequence = None
    for camera, view in views.items():
        if (not isinstance(view, Mapping) or view.get("camera_id") != camera
                or type(view.get("stereo_sequence")) is not int or view["stereo_sequence"] < 0
                or (sequence is not None and view["stereo_sequence"] != sequence)):
            raise ValueError("optional-head camera identity or capture sequence differs")
        sequence = view["stereo_sequence"]
        current_joints = view.get("robot_joints_at_capture")
        if not isinstance(current_joints, Mapping) or (joints is not None and current_joints != joints):
            raise ValueError("optional-head cameras must share frozen robot feedback")
        joints = current_joints
        masks = view.get("instance_masks")
        if (not isinstance(masks, Mapping) or len(masks) > MAX_INSTANCES_PER_VIEW
                or any(not isinstance(name, str) or not name for name in masks)):
            raise ValueError("optional-head instance masks are invalid")
        instances.update(masks)
    if not targetless and views["3"]["instance_masks"]:
        raise ValueError("optional-head transport is limited to an empty head semantic view")
    if not targetless and identity["confirmed_instance_id"] not in instances:
        raise ValueError("selected instance is absent from the original measured masks")


def encode_project_semantic_request(payload, *, exclusion_state="selected"):
    from tianji_curobo.semantic_esdf_transport import encode_semantic_esdf_request

    views = payload.get("views", {}) if isinstance(payload, Mapping) else {}
    targetless = (exclusion_state == 'observation' and payload.get('confirmed_instance_id') == ''
                  and payload.get('mapping_mode', 'semantic') == 'semantic')
    empty_head = isinstance(views, Mapping) and isinstance(views.get("3"), Mapping) and views["3"].get("instance_masks") == {}
    if exclusion_state not in ("selected", "observation", "held", "released") or (not empty_head and not targetless):
        return encode_semantic_esdf_request(payload)
    if set(payload) not in (set(IDENTITY_FIELDS) | {"views"}, set(IDENTITY_FIELDS) | {"views", "mapping_mode"}):
        raise ValueError("optional-head semantic request schema is invalid")
    if payload.get("mapping_mode", "semantic") != "semantic":
        return encode_semantic_esdf_request(payload)
    identity = {key: deepcopy(payload[key]) for key in IDENTITY_FIELDS}
    _validate_identity(identity, targetless=targetless)
    _validate_views(views, identity, targetless=targetless)
    masks, total = {}, 0
    for camera, view in views.items():
        shape = _mask_shape(view)
        masks[camera] = []
        for name, values in view["instance_masks"].items():
            mask = np.asarray(values)
            if mask.shape != shape or not np.isin(mask, (0, 1)).all():
                raise ValueError("semantic mask must be binary and match its actual RGB size")
            total += mask.size
            if total > MAX_MASK_PIXELS:
                raise ValueError("semantic masks exceed the bounded sidecar limit")
            bits = np.packbits(mask.astype(np.uint8).reshape(-1), bitorder="little")
            masks[camera].append({"id": name, "bits_base64": base64.b64encode(bits.tobytes()).decode("ascii")})
    depth_request = encode_semantic_esdf_request({**identity, "mapping_mode": "background",
        "confirmed_instance_id": "", "sam3_generation": 0,
        "views": {camera: view | {"instance_masks": {}} for camera, view in views.items()}})
    transport = (HEAD_TRANSPORT if tuple(views) == ('3',) else ALL_TRANSPORT) if targetless else TRANSPORT
    return {"transport": transport, "semantic_identity": identity,
            "depth_request": depth_request, "semantic_masks": masks}


def decode_project_semantic_request(payload, *, exclusion_state="selected"):
    from tianji_curobo.semantic_esdf_transport import decode_semantic_esdf_request

    if not isinstance(payload, Mapping) or "transport" not in payload:
        return decode_semantic_esdf_request(payload)
    if (set(payload) != {"transport", "semantic_identity", "depth_request", "semantic_masks"}
            or payload["transport"] not in (TRANSPORT, HEAD_TRANSPORT, ALL_TRANSPORT)
            or exclusion_state not in ("selected", "observation", "held", "released")):
        raise ValueError("optional-head transport schema or exclusion state is invalid")
    targetless = payload['transport'] in (HEAD_TRANSPORT, ALL_TRANSPORT)
    if targetless and exclusion_state != 'observation':
        raise ValueError('targetless transport requires observation exclusion state')
    identity = payload["semantic_identity"]
    _validate_identity(identity, targetless=targetless)
    depth_request = payload["depth_request"]
    if (not isinstance(depth_request, Mapping) or depth_request.get("mapping_mode") != "background"
            or depth_request.get("observation_set_id") != identity["observation_set_id"]
            or depth_request.get("selection_generation") != identity["selection_generation"]):
        raise ValueError("depth transport and semantic capture identity differ")
    decoded = decode_semantic_esdf_request(depth_request)
    masks = payload["semantic_masks"]
    cameras = ('3',) if payload['transport'] == HEAD_TRANSPORT else CAMERAS
    if not isinstance(masks, Mapping) or tuple(masks) != cameras or tuple(decoded["views"]) != cameras:
        raise ValueError("optional-head sidecar must preserve the original three camera identities")
    total = 0
    for camera, view in decoded["views"].items():
        shape = _mask_shape(view)
        entries = masks[camera]
        if not isinstance(entries, list) or len(entries) > MAX_INSTANCES_PER_VIEW:
            raise ValueError("optional-head instance count exceeds its bound")
        restored = {}
        for entry in entries:
            if (not isinstance(entry, Mapping) or set(entry) != {"id", "bits_base64"}
                    or not isinstance(entry["id"], str) or not entry["id"] or entry["id"] in restored):
                raise ValueError("optional-head mask identity is invalid or duplicated")
            count = shape[0]*shape[1]
            total += count
            if total > MAX_MASK_PIXELS:
                raise ValueError("semantic masks exceed the bounded sidecar limit")
            byte_count = (count+7)//8
            text = entry["bits_base64"]
            if not isinstance(text, str) or len(text) != 4*((byte_count+2)//3):
                raise ValueError("semantic mask bit length is invalid")
            try:
                raw = base64.b64decode(text, validate=True)
            except (ValueError, TypeError) as error:
                raise ValueError("semantic mask bit encoding is invalid") from error
            if len(raw) != byte_count or (count % 8 and raw[-1] >> (count % 8)):
                raise ValueError("semantic mask byte count or padding bits are invalid")
            bits = np.frombuffer(raw, dtype=np.uint8)
            restored[entry["id"]] = np.unpackbits(bits, count=count, bitorder="little").reshape(shape)
        view["instance_masks"] = restored
    _validate_views(decoded["views"], identity, targetless=targetless)
    return {**deepcopy(identity), "views": decoded["views"]}

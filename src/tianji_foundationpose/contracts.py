"""Versioned RGBD wire contract. Camera axes are OpenCV x-right/y-down/z-forward."""
import base64
import binascii
import io
import re

import numpy as np
from PIL import Image

MAX_PIXELS = 4_194_304
MAX_BODY_BYTES = 40 * 1024 * 1024
SDK_COMMIT = "a1b694b83e633c2cb6115b9063d940a687759392"


class PoseError(ValueError):
    def __init__(self, code, message=""):
        self.code = code
        super().__init__(code + (": " + message if message else ""))


def model_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise PoseError("invalid_model_id")
    return value


def _png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_request(rgb, depth_m, mask, K, model, capture_id, camera_id, score):
    rgb = np.asarray(rgb)
    depth_m = np.asarray(depth_m, dtype="<f4")
    mask = np.asarray(mask)
    if rgb.dtype != np.uint8 or mask.dtype not in (np.uint8, np.bool_):
        raise PoseError("invalid_request", "RGB must be uint8; mask must be binary")
    request = {
        "version": 1, "model_id": model, "capture_id": capture_id, "camera_id": camera_id,
        "rgb_png": _png(rgb), "depth_f32_m": base64.b64encode(depth_m.tobytes()).decode("ascii"),
        "mask_png": _png((mask > 0).astype(np.uint8) * 255),
        "K": np.asarray(K, dtype=float).tolist(), "segmentation_score": float(score),
        "depth_shape": list(depth_m.shape),
    }
    decode_request(request)
    return request


def _decode_png(value, mode):
    if not isinstance(value, str):
        raise PoseError("invalid_request", "PNG must be base64")
    raw = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        w, h = image.size
        if image.format != "PNG" or w < 1 or h < 1 or max(w, h) > 4096 or w*h > MAX_PIXELS:
            raise PoseError("invalid_request", "invalid PNG dimensions")
        if image.mode != mode:
            raise PoseError("invalid_request", "invalid PNG mode")
        return np.asarray(image).copy()


def decode_request(payload):
    try:
        required = {"version", "model_id", "capture_id", "camera_id", "rgb_png", "depth_f32_m",
                    "mask_png", "K", "segmentation_score", "depth_shape"}
        if not isinstance(payload, dict) or set(payload) != required or payload["version"] != 1:
            raise PoseError("invalid_request", "invalid version or fields")
        model_id(payload["model_id"])
        for field in ("capture_id", "camera_id"):
            if not isinstance(payload[field], str) or not 1 <= len(payload[field]) <= 128:
                raise PoseError("invalid_request", field)
        rgb = _decode_png(payload["rgb_png"], "RGB")
        mask = _decode_png(payload["mask_png"], "L") > 0
        h, w = rgb.shape[:2]
        if mask.shape != (h,w) or payload["depth_shape"] != [h,w]:
            raise PoseError("invalid_request", "RGB, depth and mask sizes differ")
        depth_bytes = base64.b64decode(payload["depth_f32_m"], validate=True)
        if len(depth_bytes) != h*w*4:
            raise PoseError("invalid_request", "depth size mismatch")
        depth = np.frombuffer(depth_bytes, dtype="<f4").reshape(h,w).copy()
        if not np.isfinite(depth).all() or (depth < 0).any() or (depth > 65.535).any():
            raise PoseError("invalid_request", "depth must be finite metres in [0,65.535]")
        K = np.asarray(payload["K"], dtype=float)
        if (K.shape != (3,3) or not np.isfinite(K).all() or K[0,0] <= 0 or K[1,1] <= 0
                or not np.allclose(K[2], [0,0,1]) or K[0,1] != 0 or K[1,0] != 0):
            raise PoseError("invalid_request", "invalid pinhole intrinsics")
        score = payload["segmentation_score"]
        if type(score) not in (int,float) or not np.isfinite(score) or not 0 <= score <= 1:
            raise PoseError("invalid_request", "invalid segmentation score")
        # Require substantial observed depth support before SDK filtering.
        if np.count_nonzero(mask & (depth >= .0005)) <= 32:
            raise PoseError("insufficient_depth")
        return {**{k: payload[k] for k in ("model_id","capture_id","camera_id","segmentation_score")},
                "rgb":rgb, "depth_m":depth, "mask":mask, "K":K}
    except PoseError:
        raise
    except (ValueError, TypeError, KeyError, OSError, binascii.Error) as error:
        raise PoseError("invalid_request") from error


def validate_pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4,4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0,0,0,1], atol=1e-5)
            or not np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(pose[:3,:3]), 1, atol=1e-3)):
        raise PoseError("invalid_pose")
    return pose

import math
import urllib.error
import urllib.request
from urllib.parse import urlsplit


TIMEOUT_SECONDS = 180
ERROR_CODES = {
    "graspgen_service_unavailable",
    "graspgen_model_identity_mismatch",
    "graspgen_no_candidates",
    "graspgen_invalid_request",
    "graspgen_invalid_output",
    "graspgen_visualization_failed",
}


class GraspGenRequestError(ValueError):
    pass


def _error(code):
    return {"ok": False, "error": {"code": code}}


def _inspect_points(value, *, allow_empty):
    import numpy

    if not isinstance(value, numpy.ndarray):
        raise GraspGenRequestError("points must be an ndarray")
    if value.dtype != numpy.dtype("float32") or value.itemsize != 4:
        raise GraspGenRequestError("points must be float32")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise GraspGenRequestError("invalid point shape")
    from .wire import MAX_ARRAY_BYTES, MAX_ARRAY_ITEMS, MAX_POINTS

    if value.shape[0] > MAX_POINTS or (not allow_empty and value.shape[0] == 0):
        raise GraspGenRequestError("invalid point count")
    expected_items = value.shape[0] * 3
    expected_bytes = expected_items * 4
    if (
        value.size != expected_items
        or value.nbytes != expected_bytes
        or expected_items > MAX_ARRAY_ITEMS
        or expected_bytes > MAX_ARRAY_BYTES
    ):
        raise GraspGenRequestError("invalid point byte count")
    return value


def _materialize_points(value):
    import numpy

    if not numpy.isfinite(value).all():
        raise GraspGenRequestError("non-finite points")
    return numpy.ascontiguousarray(value)


def _candidate(value, *, allow_orientation=False):
    import numpy
    from .wire import MAX_CANDIDATES

    required = {"index", "T_camera_grasp", "score"}
    allowed = required | {"orientation_deg"} if allow_orientation else required
    if type(value) is not dict or not required <= set(value) <= allowed:
        raise ValueError("invalid candidate")
    orientation = value.get("orientation_deg", 0)
    if type(orientation) is not int or orientation not in (0, 180):
        raise ValueError("invalid candidate orientation")
    if type(value["index"]) is not int or not 0 <= value["index"] < MAX_CANDIDATES:
        raise ValueError("invalid candidate index")
    if type(value["score"]) not in (int, float) or not math.isfinite(value["score"]):
        raise ValueError("invalid candidate score")
    transform = numpy.asarray(value["T_camera_grasp"], dtype=numpy.float64)
    if transform.shape != (4, 4) or not numpy.isfinite(transform).all():
        raise ValueError("invalid candidate transform")
    rotation = transform[:3, :3]
    if (
        not numpy.allclose(transform[3], [0, 0, 0, 1], atol=1e-5)
        or not numpy.allclose(rotation.T @ rotation, numpy.eye(3), atol=1e-4)
        or not numpy.isclose(numpy.linalg.det(rotation), 1.0, atol=1e-4)
    ):
        raise ValueError("nonrigid candidate")
    result = {
        "index": value["index"],
        "T_camera_grasp": transform.tolist(),
        "score": float(value["score"]),
    }
    if "orientation_deg" in value:
        result["orientation_deg"] = orientation
    return result


def _viewer_url(value):
    if type(value) is not str or not 1 <= len(value) <= 256:
        raise ValueError("invalid viewer URL")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("invalid viewer URL") from error
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname != "127.0.0.1"
        or port != 8024
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid viewer URL")
    return value


def validate_generate_request(payload):
    if type(payload) is not dict or set(payload) != {
        "scene_points_camera_m", "object_points_camera_m", "gripper_config_id"
    }:
        raise GraspGenRequestError("invalid generate request")
    if payload["gripper_config_id"] != "robotiq_2f_140":
        raise GraspGenRequestError("invalid gripper")
    scene = _inspect_points(payload["scene_points_camera_m"], allow_empty=True)
    object_points = _inspect_points(payload["object_points_camera_m"], allow_empty=False)
    from .wire import MAX_ARRAY_BYTES

    if scene.nbytes + object_points.nbytes > 2 * MAX_ARRAY_BYTES:
        raise GraspGenRequestError("point arrays exceed aggregate limit")
    return {
        "scene_points_camera_m": _materialize_points(scene),
        "object_points_camera_m": _materialize_points(object_points),
        "gripper_config_id": "robotiq_2f_140",
    }


def validate_visualize_request(payload):
    required = {"scene_points_camera_m", "candidates"}
    allowed = required | {"scene_colors_rgb", "selected_candidate_index", "selected_orientation_deg", "output_format"}
    if type(payload) is not dict or not required <= set(payload) <= allowed:
        raise GraspGenRequestError("invalid visualize request")
    candidates = payload["candidates"]
    from .wire import MAX_VISUALIZATION_CANDIDATES

    if type(candidates) is not list or not 1 <= len(candidates) <= MAX_VISUALIZATION_CANDIDATES:
        raise GraspGenRequestError("invalid candidate count")
    scene = _inspect_points(payload["scene_points_camera_m"], allow_empty=True)
    try:
        candidates = [_candidate(candidate, allow_orientation=True) for candidate in candidates]
    except (TypeError, ValueError, OverflowError) as error:
        raise GraspGenRequestError("invalid candidate") from error
    result = {
        "scene_points_camera_m": _materialize_points(scene),
        "candidates": candidates,
    }
    if "output_format" in payload:
        if payload["output_format"] not in ("url", "recording"):
            raise GraspGenRequestError("invalid visualization output format")
        result["output_format"] = payload["output_format"]
    pairs = [(candidate["index"], candidate.get("orientation_deg", 0)) for candidate in candidates]
    if len(set(pairs)) != len(pairs):
        raise GraspGenRequestError("candidate orientation pairs must be unique")
    if "selected_orientation_deg" in payload and "selected_candidate_index" not in payload:
        raise GraspGenRequestError("selected orientation requires a candidate index")
    if "selected_candidate_index" in payload:
        selected = payload["selected_candidate_index"]
        orientation = payload.get("selected_orientation_deg", 0)
        if (
            type(selected) is not int
            or type(orientation) is not int
            or orientation not in (0, 180)
            or (selected, orientation) not in pairs
        ):
            raise GraspGenRequestError("selected pair must be one of the displayed candidates")
        result["selected_candidate_index"] = selected
        if "selected_orientation_deg" in payload:
            result["selected_orientation_deg"] = orientation
    if "scene_colors_rgb" in payload:
        import numpy

        colors = payload["scene_colors_rgb"]
        if (
            not isinstance(colors, numpy.ndarray)
            or colors.dtype != numpy.dtype("uint8")
            or colors.shape != scene.shape
        ):
            raise GraspGenRequestError("colors must be uint8 RGB matching scene points")
        result["scene_colors_rgb"] = numpy.ascontiguousarray(colors)
    return result


def serialize_envelope(envelope):
    from .wire import pack

    return pack(envelope)


class GraspGenClient:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def health(self):
        response = self._call("health", {})
        return response if response.get("ok") is False else response["health"]

    def generate(self, payload):
        try:
            request = validate_generate_request(payload)
            from .wire import encode_array

            wire_payload = {
                "scene_points_camera_m": encode_array(request["scene_points_camera_m"], unit="meter", frame="camera"),
                "object_points_camera_m": encode_array(request["object_points_camera_m"], unit="meter", frame="camera"),
                "gripper_config_id": request["gripper_config_id"],
            }
        except Exception:
            return _error("graspgen_invalid_request")
        response = self._call("generate", wire_payload)
        if response.get("ok") is False:
            return response
        try:
            result = response["result"]
            from .wire import MAX_CANDIDATES

            if type(result) is not dict or set(result) != {"candidates", "model"}:
                raise ValueError("invalid result")
            candidates = result["candidates"]
            if type(candidates) is not list or not 1 <= len(candidates) <= MAX_CANDIDATES:
                raise ValueError("invalid candidates")
            checked_candidates = [_candidate(value) for value in candidates]
            seen_indices = set()
            for position, candidate in enumerate(checked_candidates):
                index = candidate["index"]
                if index in seen_indices:
                    raise ValueError("duplicate candidate index")
                seen_indices.add(index)
                if position == 0:
                    continue
                previous = checked_candidates[position - 1]
                if candidate["score"] > previous["score"] or (
                    candidate["score"] == previous["score"]
                    and candidate["index"] < previous["index"]
                ):
                    raise ValueError("candidates are not stably score ordered")
            return {
                "candidates": checked_candidates,
                "model": result["model"],
            }
        except Exception:
            return _error("graspgen_invalid_output")

    def visualize(self, payload):
        try:
            request = validate_visualize_request(payload)
            from .wire import encode_array

            wire_payload = {
                "scene_points_camera_m": encode_array(request["scene_points_camera_m"], unit="meter", frame="camera"),
                "candidates": request["candidates"],
            }
            if "output_format" in request:
                wire_payload["output_format"] = request["output_format"]
            if "selected_candidate_index" in request:
                wire_payload["selected_candidate_index"] = request["selected_candidate_index"]
            if "selected_orientation_deg" in request:
                wire_payload["selected_orientation_deg"] = request["selected_orientation_deg"]
            if "scene_colors_rgb" in request:
                wire_payload["scene_colors_rgb"] = encode_array(
                    request["scene_colors_rgb"], unit="rgb_0_255", frame="camera"
                )
        except Exception:
            return _error("graspgen_invalid_request")
        response = self._call("visualize", wire_payload)
        if response.get("ok") is False:
            return response
        try:
            result = response["result"]
            if request.get("output_format") == "recording":
                if type(result) is not dict or set(result) != {"viewer_html", "viewer_scene", "model"}:
                    raise ValueError("invalid recorded viewer")
                if not all(type(result[key]) is bytes and result[key] for key in ("viewer_html", "viewer_scene")):
                    raise ValueError("invalid recorded viewer files")
                return {key: result[key] for key in ("viewer_html", "viewer_scene")}
            if type(result) is not dict or set(result) != {"viewer_url", "model"}:
                raise ValueError("invalid result")
            return {"viewer_url": _viewer_url(result["viewer_url"])}
        except Exception:
            return _error("graspgen_invalid_output")

    def _call(self, operation, payload):
        from .wire import (
            MAX_CANDIDATES,
            MAX_REQUEST_BYTES,
            MAX_RESPONSE_BYTES,
            MODEL_IDENTITY,
            unpack,
        )

        try:
            body = serialize_envelope({"operation": operation, "payload": payload})
        except Exception:
            return _error("graspgen_invalid_request")
        if type(body) is not bytes or len(body) > MAX_REQUEST_BYTES:
            return _error("graspgen_invalid_request")
        try:
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={"Content-Type": "application/msgpack", "Accept": "application/msgpack"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                content_type = response.headers.get("Content-Type")
                content_length_text = response.headers.get("Content-Length")
                try:
                    content_length = int(content_length_text)
                except (TypeError, ValueError):
                    return _error("graspgen_invalid_output")
                if (
                    response.status != 200
                    or type(content_type) is not str
                    or content_type.strip() != "application/msgpack"
                    or not 0 <= content_length <= MAX_RESPONSE_BYTES
                ):
                    return _error("graspgen_invalid_output")
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(response_body) != content_length or len(response_body) > MAX_RESPONSE_BYTES:
                return _error("graspgen_invalid_output")
        except urllib.error.HTTPError:
            return _error("graspgen_invalid_output")
        except (urllib.error.URLError, TimeoutError, OSError):
            return _error("graspgen_service_unavailable")
        try:
            envelope = unpack(response_body)
        except Exception:
            return _error("graspgen_invalid_output")
        if type(envelope) is not dict or type(envelope.get("ok")) is not bool:
            return _error("graspgen_invalid_output")
        if envelope["ok"] is False:
            if (
                set(envelope) != {"ok", "error"}
                or type(envelope["error"]) is not dict
                or set(envelope["error"]) != {"code"}
                or envelope["error"].get("code") not in ERROR_CODES
            ):
                return _error("graspgen_invalid_output")
            return envelope
        expected_key = "health" if operation == "health" else "result"
        if set(envelope) != {"ok", expected_key} or type(envelope[expected_key]) is not dict:
            return _error("graspgen_invalid_output")
        value = envelope[expected_key]
        if operation == "health":
            expected_health = {
                "status",
                "service_version",
                "source_sha",
                "generator_sha256",
                "discriminator_sha256",
                "gpu_ready",
                "model_ready",
                "max_candidates",
                "model",
            }
            model = value.get("model")
            if (
                set(value) != expected_health
                or type(model) is not dict
                or type(model.get("generator")) is not dict
                or type(model.get("discriminator")) is not dict
                or value["status"] != "ok"
                or value["service_version"] != "1.0.0"
                or value["source_sha"] != model.get("source_sha")
                or value["generator_sha256"] != model["generator"].get("sha256")
                or value["discriminator_sha256"]
                != model["discriminator"].get("sha256")
                or value["gpu_ready"] is not True
                or value["model_ready"] is not True
                or value["max_candidates"] != MAX_CANDIDATES
            ):
                return _error("graspgen_invalid_output")
        elif "model" not in value:
            return _error("graspgen_invalid_output")
        if value["model"] != MODEL_IDENTITY:
            return _error("graspgen_model_identity_mismatch")
        return envelope

import hashlib
import logging
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy

from . import tools
from .contracts import InvalidRequest, InvalidOutput, NoCandidates, ModelIdentityMismatch

from .client import GraspGenRequestError, validate_visualize_request
from .wire import (
    MAX_ARRAY_BYTES,
    MAX_CANDIDATES,
    MAX_POINTS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MODEL_IDENTITY,
    decode_array,
    pack,
    unpack,
)


GRASPGEN_SOURCE_SHA = "2dd8852e1be60f5f9d277fafcc621835cdf59110"
MODELS_SOURCE_SHA = "ec1ccbb5eec0680db669246ac312a3636f16ee43"
CONFIG_IDENTITY = MODEL_IDENTITY["config"]
GENERATOR_IDENTITY = MODEL_IDENTITY["generator"]
DISCRIMINATOR_IDENTITY = MODEL_IDENTITY["discriminator"]
OFFICIAL_GENERATOR_CHECKPOINT = "graspgen_robotiq_2f_140_gen.pth"
OFFICIAL_DISCRIMINATOR_CHECKPOINT = "graspgen_robotiq_2f_140_dis.pth"

SAMPLER = None
RUN_GRASPMOE = None
GPU_READY = False
SERVICE_VERSION = "1.0.0"
INTERACTIVE_VIEWER = None
INTERACTIVE_VIEWER_LOCK = threading.Lock()
LOGGER = logging.getLogger("jodell.graspgen.service")


def _error(code):
    return {"ok": False, "error": {"code": code}}


def _inspect_points(value, *, allow_empty):
    if not isinstance(value, numpy.ndarray):
        raise InvalidRequest("points must be an ndarray")
    if value.dtype != numpy.dtype("float32") or value.itemsize != 4:
        raise InvalidRequest("points must be float32")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise InvalidRequest("invalid point shape")
    if value.shape[0] > MAX_POINTS or (not allow_empty and value.shape[0] == 0):
        raise InvalidRequest("invalid point count")
    expected_items = value.shape[0] * 3
    expected_bytes = expected_items * 4
    if (
        value.size != expected_items
        or value.nbytes != expected_bytes
        or expected_bytes > MAX_ARRAY_BYTES
    ):
        raise InvalidRequest("invalid point byte count")
    return value


def _materialize_points(value):
    if not numpy.isfinite(value).all():
        raise InvalidRequest("non-finite points")
    return numpy.ascontiguousarray(value)


def validate_generate_request(payload):
    if type(payload) is not dict or set(payload) != {
        "scene_points_camera_m", "object_points_camera_m", "gripper_config_id"
    }:
        raise InvalidRequest("invalid generate request")
    if payload["gripper_config_id"] != "robotiq_2f_140":
        raise InvalidRequest("invalid gripper")
    scene = _inspect_points(payload["scene_points_camera_m"], allow_empty=True)
    object_points = _inspect_points(payload["object_points_camera_m"], allow_empty=False)
    if scene.nbytes + object_points.nbytes > 2 * MAX_ARRAY_BYTES:
        raise InvalidRequest("point arrays exceed aggregate limit")
    return {
        "scene_points_camera_m": _materialize_points(scene),
        "object_points_camera_m": _materialize_points(object_points),
        "gripper_config_id": "robotiq_2f_140",
    }


def _wire_points(value):
    if type(value) is dict:
        return decode_array(
            value,
            dtype="float32",
            rank=2,
            shape_suffix=(3,),
            unit="meter",
            frame="camera",
        )
    return value


def _git_head(source_root):
    git_path = source_root / ".git"
    if not git_path.is_dir():
        return None
    head = (git_path / "HEAD").read_text(encoding="ascii").strip()
    if head.startswith("ref: "):
        ref_path = git_path / head[5:]
        if not ref_path.is_file():
            return None
        return ref_path.read_text(encoding="ascii").strip()
    return head


def _source_identity(root, marker_name):
    identity = _git_head(root)
    marker = root / marker_name
    if identity is None and marker.is_file():
        identity = marker.read_text(encoding="ascii").strip()
    return identity


def _verify_file_identity(path, identity, *, expected_name=None):
    try:
        if expected_name is not None and path.name != expected_name:
            raise ModelIdentityMismatch("model identity mismatch")
        if path.stat().st_size != identity["size"]:
            raise ModelIdentityMismatch("model identity mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ModelIdentityMismatch("model identity mismatch") from error
    if digest.hexdigest() != identity["sha256"]:
        raise ModelIdentityMismatch("model identity mismatch")


def verify_model_identity(source_root, models_root, config_path, generator_path, discriminator_path):
    source_root = Path(source_root).resolve()
    models_root = Path(models_root).resolve()
    paths = [Path(value).resolve() for value in (config_path, generator_path, discriminator_path)]
    if _source_identity(source_root, "GRASPGEN_SOURCE_SHA") != GRASPGEN_SOURCE_SHA:
        raise ModelIdentityMismatch("source identity mismatch")
    if _source_identity(models_root, "GRASPGEN_MODELS_SOURCE_SHA") != MODELS_SOURCE_SHA:
        raise ModelIdentityMismatch("models identity mismatch")
    for path in paths:
        try:
            path.relative_to(models_root)
        except ValueError as error:
            raise ModelIdentityMismatch("model path outside verified root") from error
    _verify_file_identity(paths[0], CONFIG_IDENTITY, expected_name="graspgen_robotiq_2f_140.yml")
    _verify_file_identity(paths[1], GENERATOR_IDENTITY)
    _verify_file_identity(paths[2], DISCRIMINATOR_IDENTITY)


def _module_is_under_root(module, source_root):
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return False
    try:
        Path(module_file).resolve().relative_to(source_root)
    except (OSError, ValueError):
        return False
    return True


def load_official_model(source_root, models_root, config_path, generator_path, discriminator_path):
    verify_model_identity(source_root, models_root, config_path, generator_path, discriminator_path)
    source_root = Path(source_root).resolve()
    for name, loaded in tuple(sys.modules.items()):
        if (name == "grasp_gen" or name.startswith("grasp_gen.")) and not _module_is_under_root(loaded, source_root):
            raise ModelIdentityMismatch("cached module outside verified source")
    source_text = os.fspath(source_root)
    retained = []
    for entry in sys.path:
        try:
            if Path(entry).resolve() == source_root:
                continue
        except (OSError, TypeError, ValueError):
            pass
        retained.append(entry)
    sys.path[:] = [source_text, *retained]
    import importlib

    server_module = importlib.import_module("grasp_gen.grasp_server")
    samplers_module = importlib.import_module("grasp_gen.samplers")
    for name in (
        "grasp_gen", "grasp_gen.grasp_server", "grasp_gen.samplers",
    ):
        if not _module_is_under_root(sys.modules.get(name), source_root):
            raise ModelIdentityMismatch("imported module outside verified source")
    config = server_module.load_grasp_cfg(os.fspath(config_path))
    try:
        configured_generator = Path(config.eval.checkpoint).resolve()
        configured_discriminator = Path(config.discriminator.checkpoint).resolve()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise ModelIdentityMismatch("configured checkpoint identity mismatch") from error
    config_root = Path(config_path).resolve().parent
    if configured_generator != config_root / OFFICIAL_GENERATOR_CHECKPOINT:
        raise ModelIdentityMismatch("configured generator identity mismatch")
    if configured_discriminator != config_root / OFFICIAL_DISCRIMINATOR_CHECKPOINT:
        raise ModelIdentityMismatch("configured discriminator identity mismatch")
    config.eval.checkpoint = os.fspath(Path(generator_path).resolve())
    config.discriminator.checkpoint = os.fspath(Path(discriminator_path).resolve())
    sampler = server_module.GraspGenSampler(config)
    return sampler, samplers_module.run_graspmoe


def _publish_viewer(scene_points, candidates, *, scene_colors_rgb=None, selected_candidate_index=None, selected_orientation_deg=0, output_format="url"):
    global INTERACTIVE_VIEWER
    with INTERACTIVE_VIEWER_LOCK:
        if INTERACTIVE_VIEWER is None:
            from .visualization import InteractiveViewer

            try:
                port = int(os.environ.get("JODELL_GRASPGEN_VISER_PORT", "8024"))
            except ValueError as error:
                raise InvalidOutput("invalid Viser port") from error
            viewer_url = os.environ.get(
                "JODELL_GRASPGEN_VIEWER_URL", f"http://127.0.0.1:{port}/"
            )
            INTERACTIVE_VIEWER = InteractiveViewer(
                port=port,
                public_url=viewer_url,
            )
        options = {"scene_colors_rgb": scene_colors_rgb}
        if output_format == "recording":
            options["output_format"] = output_format
        if selected_candidate_index is not None:
            options["selected_candidate_index"] = selected_candidate_index
            options["selected_orientation_deg"] = selected_orientation_deg
        return INTERACTIVE_VIEWER.publish(scene_points, candidates, **options)


def _close_interactive_viewer():
    global INTERACTIVE_VIEWER
    with INTERACTIVE_VIEWER_LOCK:
        viewer = INTERACTIVE_VIEWER
        INTERACTIVE_VIEWER = None
    if viewer is not None:
        viewer.close()


def handle_envelope(envelope):
    if type(envelope) is not dict or set(envelope) != {"operation", "payload"}:
        return _error("graspgen_invalid_request")
    operation = envelope["operation"]
    payload = envelope["payload"]
    if operation == "health":
        if type(payload) is not dict or payload:
            return _error("graspgen_invalid_request")
        return {
            "ok": True,
            "health": {
                "status": "ok",
                "service_version": SERVICE_VERSION,
                "source_sha": GRASPGEN_SOURCE_SHA,
                "generator_sha256": GENERATOR_IDENTITY["sha256"],
                "discriminator_sha256": DISCRIMINATOR_IDENTITY["sha256"],
                "gpu_ready": GPU_READY,
                "model_ready": SAMPLER is not None and callable(RUN_GRASPMOE),
                "max_candidates": MAX_CANDIDATES,
                "model": MODEL_IDENTITY,
            },
        }
    if operation == "generate":
        try:
            if type(payload) is not dict:
                raise InvalidRequest("invalid request")
            decoded = dict(payload)
            for key in ("scene_points_camera_m", "object_points_camera_m"):
                decoded[key] = _wire_points(decoded.get(key))
            request = validate_generate_request(decoded)
            candidates = tools.generate(
                SAMPLER,
                request["object_points_camera_m"],
                RUN_GRASPMOE,
            )
            return {
                "ok": True,
                "result": {
                    "candidates": candidates,
                    "model": MODEL_IDENTITY,
                },
            }
        except NoCandidates:
            return _error("graspgen_no_candidates")
        except (InvalidRequest, GraspGenRequestError, TypeError, ValueError):
            return _error("graspgen_invalid_request")
        except ModelIdentityMismatch:
            return _error("graspgen_model_identity_mismatch")
        except Exception:
            return _error("graspgen_invalid_output")
    if operation == "visualize":
        try:
            if type(payload) is not dict:
                raise InvalidRequest("invalid request")
            decoded = dict(payload)
            decoded["scene_points_camera_m"] = _wire_points(decoded.get("scene_points_camera_m"))
            if "scene_colors_rgb" in decoded and type(decoded["scene_colors_rgb"]) is dict:
                decoded["scene_colors_rgb"] = decode_array(
                    decoded["scene_colors_rgb"], dtype="uint8", rank=2,
                    shape_suffix=(3,), unit="rgb_0_255", frame="camera",
                )
            request = validate_visualize_request(decoded)
        except (InvalidRequest, TypeError, ValueError, OverflowError):
            return _error("graspgen_invalid_request")
        try:
            options = {"scene_colors_rgb": request.get("scene_colors_rgb")}
            if request.get("output_format") == "recording":
                options["output_format"] = "recording"
            if "selected_candidate_index" in request:
                options["selected_candidate_index"] = request["selected_candidate_index"]
            if "selected_orientation_deg" in request:
                options["selected_orientation_deg"] = request["selected_orientation_deg"]
            viewer_url = _publish_viewer(
                request["scene_points_camera_m"], request["candidates"],
                **options,
            )
            if request.get("output_format") == "recording":
                return {"ok": True, "result": {**viewer_url, "model": MODEL_IDENTITY}}
            return {
                "ok": True,
                "result": {"viewer_url": viewer_url, "model": MODEL_IDENTITY},
            }
        except Exception:
            return _error("graspgen_visualization_failed")
    return _error("graspgen_invalid_request")


def pack_bounded_response(response):
    try:
        encoded = pack(response)
    except Exception:
        encoded = None
    if type(encoded) is not bytes or len(encoded) > MAX_RESPONSE_BYTES:
        encoded = pack(_error("graspgen_invalid_output"))
    if type(encoded) is not bytes or len(encoded) > MAX_RESPONSE_BYTES:
        raise InvalidOutput("stable response exceeds limit")
    return encoded


def is_exact_msgpack_content_type(value):
    return type(value) is str and value.strip() == "application/msgpack"


class GraspGenRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get("Content-Type")
        try:
            content_length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            content_length = -1
        if not is_exact_msgpack_content_type(content_type) or not 0 <= content_length <= MAX_REQUEST_BYTES:
            response = _error("graspgen_invalid_request")
        else:
            try:
                body = self.rfile.read(content_length)
                if len(body) != content_length:
                    raise ValueError("short body")
                response = handle_envelope(unpack(body, limit=MAX_REQUEST_BYTES))
            except Exception:
                response = _error("graspgen_invalid_request")
        encoded = pack_bounded_response(response)
        self.send_response(200)
        self.send_header("Content-Type", "application/msgpack")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        return


def initialize_from_environment():
    global GPU_READY, SAMPLER, RUN_GRASPMOE
    SAMPLER, RUN_GRASPMOE = load_official_model(
        os.environ["JODELL_GRASPGEN_SOURCE"],
        os.environ["JODELL_GRASPGEN_MODELS_SOURCE"],
        os.environ["JODELL_GRASPGEN_CONFIG"],
        os.environ["JODELL_GRASPGEN_GENERATOR"],
        os.environ["JODELL_GRASPGEN_DISCRIMINATOR"],
    )
    import torch

    GPU_READY = torch.cuda.is_available()


def serve():
    host = os.environ.get("GAP_GRASPGEN_HOST", "127.0.0.1")
    port = int(os.environ.get("GAP_GRASPGEN_PORT", "8123"))
    server = ThreadingHTTPServer((host, port), GraspGenRequestHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        _close_interactive_viewer()


if __name__ == "__main__":
    initialize_from_environment()
    serve()

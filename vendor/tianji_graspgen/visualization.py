import threading
from pathlib import Path


_UNSET_ORIENTATION = object()


class VisualizationFailed(RuntimeError):
    pass


def _fail():
    return VisualizationFailed("graspgen_visualization_failed")


def _default_dependencies():
    from grasp_gen.utils import viser_utils

    return {
        "create_visualizer": viser_utils.create_visualizer,
        "get_color_from_score": viser_utils.get_color_from_score,
        "visualize_grasp": viser_utils.visualize_grasp,
        "visualize_pointcloud": viser_utils.visualize_pointcloud,
    }


def _validated_inputs(scene_points, candidates, **options):
    import numpy
    from .client import validate_visualize_request

    try:
        request = validate_visualize_request({
            "scene_points_camera_m": scene_points, "candidates": candidates, **options,
        })
    except (TypeError, ValueError, OverflowError) as error:
        raise _fail() from error
    checked = []
    for value in request["candidates"]:
        transform = numpy.asarray(value["T_camera_grasp"], dtype=numpy.float64)
        checked.append((value["index"], transform, value["score"], value.get("orientation_deg", 0)))
    return request["scene_points_camera_m"], checked


def _frame_visualizer_camera(server, checked):
    import numpy

    translations = numpy.stack([item[1][:3, 3] for item in checked], axis=0)
    lower = translations.min(axis=0)
    upper = translations.max(axis=0)
    look_at = (lower + upper) / 2.0
    radius = float(numpy.linalg.norm(upper - lower) / 2.0)
    distance = max(0.35, radius * 2.5)
    scene = getattr(server, "scene", None)
    set_up_direction = getattr(scene, "set_up_direction", None)
    initial_camera = getattr(server, "initial_camera", None)
    if not callable(set_up_direction) or initial_camera is None:
        raise _fail()
    set_up_direction("-y")
    initial_camera.look_at = tuple(float(value) for value in look_at)
    initial_camera.position = (
        float(look_at[0]),
        float(look_at[1]),
        float(look_at[2] - distance),
    )


def _add_directed_axes(server, name, transform):
    """Use Viser's public line segments for short positive XYZ arrow shafts."""
    import numpy

    segments, colors = [], []
    for axis, color in enumerate(((255, 64, 64), (64, 255, 64), (64, 128, 255))):
        tip = numpy.zeros(3)
        tip[axis] = 0.04
        segments.append([numpy.zeros(3), tip])
        colors.append([color, color])
        for sign in (-1, 1):
            wing = tip.copy()
            wing[axis] -= 0.008
            wing[(axis + 1) % 3] = sign * 0.003
            segments.append([wing, tip])
            colors.append([color, color])
    points = numpy.asarray(segments) @ transform[:3, :3].T + transform[:3, 3]
    server.scene.add_line_segments(
        name, points=points.astype(numpy.float32),
        colors=numpy.asarray(colors, dtype=numpy.uint8), line_width=2.0,
    )


def _populate_scene(server, scene, checked, dependencies, scene_colors_rgb, selected_candidate_index, selected_orientation_deg):
    import numpy

    color = [128, 128, 128]
    if scene_colors_rgb is not None:
        # The pinned official helper infers the scale from max(color). Using
        # 0..1 consistently preserves even uint8 images containing only 0/1.
        color = scene_colors_rgb.astype(numpy.float32) / 255.0
    dependencies["visualize_pointcloud"](
        server, "scene_points", scene, color, size=0.002
    )

    scores = numpy.asarray([item[2] for item in checked], dtype=numpy.float32)
    colors = dependencies["get_color_from_score"](scores, use_255_scale=True)
    for position, (index, transform, _score, orientation) in enumerate(checked):
        selected = (index, orientation) == (selected_candidate_index, selected_orientation_deg)
        pair_path = f"{index:03d}/{orientation}"
        dependencies["visualize_grasp"](
            server,
            f"grasps/{pair_path}",
            transform,
            color=(255, 64, 160) if selected else colors[position],
            gripper_name="robotiq_2f_140",
            linewidth=3.0 if selected else 1.0,
        )
        _add_directed_axes(server, f"candidate_axes/{pair_path}", transform)
        # The immutable GraspGen ordinal is also used by the workbench dropdown.
        server.scene.add_label(
            f"candidate_labels/{pair_path}",
            f"#{index + 1} · {orientation}°" + (" · 已选" if selected else ""),
            # Put opposite orientations on their respective positive-X tips.
            position=tuple(float(value) for value in transform[:3, 3] + .045 * transform[:3, 0]),
            anchor="bottom-left", depth_test=False,
        )


class InteractiveViewer:
    """Own the latest official GraspGen Viser scene for a bounded sidecar."""

    def __init__(self, *, port, public_url, dependencies=None):
        if type(port) is not int or isinstance(port, bool) or not 1 <= port <= 65535:
            raise _fail()
        if type(public_url) is not str or not public_url:
            raise _fail()
        self._port = port
        self._public_url = public_url
        self._dependencies = dependencies
        self._lock = threading.RLock()
        self._server = None

    def publish(self, scene_points, candidates, *, scene_colors_rgb=None, selected_candidate_index=None, selected_orientation_deg=_UNSET_ORIENTATION, output_format="url"):
        options = {}
        if selected_candidate_index is not None:
            options["selected_candidate_index"] = selected_candidate_index
        if selected_orientation_deg is not _UNSET_ORIENTATION:
            options["selected_orientation_deg"] = selected_orientation_deg
        else:
            selected_orientation_deg = 0
        if scene_colors_rgb is not None:
            options["scene_colors_rgb"] = scene_colors_rgb
        scene, checked = _validated_inputs(scene_points, candidates, **options)
        dependencies = (
            _default_dependencies()
            if self._dependencies is None
            else self._dependencies
        )
        with self._lock:
            server = self._server
            created = server is None
            try:
                if created:
                    server = dependencies["create_visualizer"](port=self._port)
                else:
                    reset = getattr(getattr(server, "scene", None), "reset", None)
                    if not callable(reset):
                        raise _fail()
                    reset()
                _frame_visualizer_camera(server, checked)
                _populate_scene(server, scene, checked, dependencies, scene_colors_rgb, selected_candidate_index, selected_orientation_deg)
            except Exception as error:
                if created and server is not None:
                    try:
                        server.stop()
                    except Exception:
                        pass
                elif server is not None:
                    try:
                        server.scene.reset()
                    except Exception:
                        try:
                            server.stop()
                        except Exception:
                            pass
                        self._server = None
                if isinstance(error, VisualizationFailed):
                    raise
                raise _fail() from error
            self._server = server
            if output_format == "recording":
                # Public Viser offline playback API. Keep its packaged client
                # unchanged and serve both files through the workbench origin.
                import viser
                scene_bytes = server.get_scene_serializer().serialize()
                client = Path(viser.__file__).parent / "client" / "build" / "index.html"
                return {"viewer_html": client.read_bytes(), "viewer_scene": scene_bytes}
        return self._public_url

    def close(self):
        with self._lock:
            server = self._server
            self._server = None
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass

"""Explicit candidate-to-published lifecycle for a fixed ESDF scene."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
import logging
import math
import struct
import threading
import time
from types import MappingProxyType
from typing import Any, Callable
from uuid import uuid4

from .contracts import CuroboToolError
from .mixed_esdf_scene import (
    build_table_completion,
    create_mixed_esdf_scene,
    fixed_mapping_bounds,
)
from .robot_segmenter import FfsDepthObservationRequest, SemanticProjectionRequest
from .surface_colors import SurfaceColorizer


LOGGER = logging.getLogger("jodell.curobo.semantic_esdf")
LOGGER.setLevel(logging.INFO)


TABLETOP_DIAGNOSTIC_KEYS = (
    "height_m",
    "normal_xyz",
    "bounds_xy",
    "inlier_count",
    "inlier_ratio",
    "area_m2",
    "support_plane",
)
OPEN3D_DIAGNOSTIC_KEYS = (
    "backend",
    "open3d_version",
    "input_point_count",
    "filtered_point_count",
    "removed_point_count",
)
DEFAULT_BUILD_TIMEOUT_S = 180.0


class _BuildCancelled(RuntimeError):
    pass


class _BuildDeadlineExceeded(TimeoutError):
    pass


def _payload(result: Any) -> dict[str, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]
    if not isinstance(result, Mapping):
        return {
            "ok": False,
            "error": "invalid_fixed_esdf_response",
            "message": "Fixed ESDF operation returned an invalid response.",
        }
    return dict(result)


def _diagnostic_fields(value: Any, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: deepcopy(value[key]) for key in keys if key in value}


def _published_diagnostics(
    published: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not published:
        return None
    return {
        "tabletop": _diagnostic_fields(
            published.get("tabletop"),
            TABLETOP_DIAGNOSTIC_KEYS,
        ),
        "open3d": _diagnostic_fields(
            published.get("open3d"),
            OPEN3D_DIAGNOSTIC_KEYS,
        ),
        "robot_side_margin_m": published.get("robot_side_margin_m"),
    }


@dataclass(frozen=True)
class SemanticEsdfCandidate:
    observation_set_id: str
    selected_instance_id: str
    selection_generation: int
    scene_data: Any
    scene_model: Mapping[str, Any]
    bounds: Any
    table: Any
    integration: Any
    camera_metrics: Mapping[str, Mapping[str, int]]
    instance_metrics: Mapping[str, Mapping[str, int | str]]
    timings_ms: Mapping[str, float] = field(default_factory=dict)


class SemanticEsdfCandidateBuilder:
    """Build a private semantic scene or a head-anchored background subset."""

    def __init__(
        self,
        *,
        robot_segmenter_boundary: Any,
        mapper_boundary: Any,
        scene_type: Any,
        robot_spheres: Callable[[Mapping[str, Sequence[float]]], Any],
        open3d_runtime=None,
        table_builder: Callable[..., Any] = build_table_completion,
        instance_geometry_builder: Callable[..., Any] | None = None,
    ) -> None:
        self._segmenter = robot_segmenter_boundary
        self._mapper = mapper_boundary
        self._scene_type = scene_type
        self._robot_spheres = robot_spheres
        self._open3d = open3d_runtime
        self._table_builder = table_builder
        self._instance_geometry_builder = instance_geometry_builder

    @staticmethod
    def _request(payload: object):
        if not isinstance(payload, Mapping):
            raise ValueError("semantic ESDF request must be a mapping")
        observation_set_id = str(payload.get("observation_set_id") or "")
        selected = str(payload.get("confirmed_instance_id") or "")
        generation = payload.get("selection_generation")
        views = payload.get("views")
        mode = payload.get("mapping_mode", "semantic")
        if mode not in ("semantic", "background"):
            raise ValueError("unknown ESDF mapping mode")
        background = mode == "background"
        if (
            not observation_set_id
            or (not selected and not background)
            or (background and selected)
            or type(generation) is not int
            or generation < 1
            or not isinstance(views, Mapping)
            or tuple(views) not in (
                (("3",), ("1", "3"), ("2", "3"), ("1", "2", "3"))
                if background else (("1", "2", "3"),)
            )
        ):
            raise ValueError("semantic ESDF request identity or views are invalid")
        return observation_set_id, selected, generation, views

    def build(self, payload: object) -> SemanticEsdfCandidate:
        build_started = time.perf_counter()
        timings = {}
        observation_set_id, selected, generation, views = self._request(payload)
        if self._instance_geometry_builder is None:
            self._mapper.clear_instance_layers()
        prepared = []
        background_groups = []
        instance_points: dict[str, dict[str, Any]] = {}
        camera_metrics = {}
        robot_spheres = []
        for camera_id in views:
            camera_started = time.perf_counter()
            view = views[camera_id]
            if not isinstance(view, Mapping) or str(view.get("camera_id") or "") != camera_id:
                raise ValueError("semantic ESDF camera identity changed")
            background = payload.get("mapping_mode") == "background"
            if background and view.get("instance_masks") != {}:
                raise ValueError("background mapping must not exclude any instance")
            depth_request = FfsDepthObservationRequest.from_mapping(
                {
                    key: view.get(key)
                    for key in (
                        "camera_id",
                        "stereo_sequence",
                        "depth_left_m",
                        "valid_left_mask",
                        "left_intrinsics",
                        "T_base_left_camera",
                        "robot_joints_at_capture",
                    )
                },
            )
            projection = SemanticProjectionRequest.from_mapping(
                {
                    key: view.get(key)
                    for key in (
                        "rgb_size",
                        "rgb_intrinsics",
                        "left_to_rgb",
                        "instance_masks",
                    )
                },
                allow_empty=background or camera_id != "3",
            )
            filtered = self._segmenter.prepare_ffs_observation(depth_request)
            semantic = self._segmenter.remove_semantic_instances(
                filtered,
                projection,
                open3d_runtime=self._open3d,
            )
            prepared.append(semantic)
            background_groups.append(semantic.background_points_base_m)
            camera_metrics[camera_id] = MappingProxyType(
                {
                    "input_point_count": int(semantic.input_point_count),
                    "filtered_point_count": int(semantic.filtered_point_count),
                    "background_point_count": int(semantic.background_point_count),
                    "removed_robot_point_count": int(semantic.removed_point_count),
                }
            )
            robot_spheres.append(self._robot_spheres(depth_request.joint_positions_deg))
            LOGGER.info(
                "semantic ESDF camera=%s filter_s=%.3f input_points=%d kept_points=%d",
                camera_id, time.perf_counter() - camera_started,
                semantic.input_point_count, semantic.filtered_point_count,
            )
        timings["depth_filter_projection"] = (time.perf_counter() - build_started) * 1000
        np = __import__("numpy")
        spheres = np.concatenate(
            [np.asarray(value, dtype=np.float64).reshape(-1, 4) for value in robot_spheres],
            axis=0,
        )
        stage_started = time.perf_counter()
        table = self._table_builder(
            background_groups,
            spheres,
            open3d_runtime=self._open3d,
        )
        LOGGER.info("semantic ESDF table_s=%.3f", time.perf_counter() - stage_started)
        timings["table"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        cropped = []
        for semantic in prepared:
            view = self._segmenter.crop_semantic_observation(semantic, table.model)
            cropped.append(view)
            camera_metrics[view.camera_id] = MappingProxyType({
                **camera_metrics[view.camera_id],
                "background_point_count": int(view.background_point_count),
                "removed_workspace_background_point_count": int(
                    semantic.background_point_count - view.background_point_count
                ),
            })
        prepared = cropped
        timings["table_crop"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        prepared = self._segmenter.filter_isolated_surfaces(prepared)
        for view in prepared:
            removed = camera_metrics[view.camera_id]["background_point_count"] - view.background_point_count
            camera_metrics[view.camera_id] = MappingProxyType({
                **camera_metrics[view.camera_id],
                "background_point_count": int(view.background_point_count),
                "removed_isolated_background_point_count": int(removed),
            })
            for instance_id, points in view.instance_points_base_m.items():
                instance_points.setdefault(instance_id, {})[view.camera_id] = points
        timings["isolated_surfaces"] = (time.perf_counter() - stage_started) * 1000
        if selected and (selected not in instance_points or not any(
            len(points) for points in instance_points[selected].values()
        )):
            raise CuroboToolError(
                "The selected object has no valid depth inside the table footprint.",
                stage="scene", code="mapping_primary_depth_empty", status=422,
                details={"instance_id": selected},
            )
        stage_started = time.perf_counter()
        geometries = []
        # Covers the largest supported activation (0.5 m), every captured robot
        # sphere radius, and two interpolation cells at each ESDF boundary.
        instance_padding_m = float(np.max(spheres[:, 3])) + .5 + .02
        instance_metrics = {}
        for instance_id, points_by_camera in instance_points.items():
            if not any(len(points) for points in points_by_camera.values()):
                instance_metrics[instance_id] = MappingProxyType({
                    "skipped_reason": "empty_depth", "input_point_count": 0,
                    "filtered_point_count": 0, "camera_count": 0,
                })
                LOGGER.info("semantic ESDF instance=%s skipped: empty depth", instance_id)
                continue
            if self._instance_geometry_builder is not None:
                geometry = self._instance_geometry_builder(
                    instance_id, points_by_camera, open3d_runtime=self._open3d,
                )
            else:
                geometry = self._mapper.build_instance_esdf(
                    prepared, instance_id, points_by_camera, padding_m=instance_padding_m,
                    table_model=table.model,
                )
            geometries.append(geometry)
        LOGGER.info(
            "semantic ESDF instances_s=%.3f count=%d",
            time.perf_counter() - stage_started, len(geometries),
        )
        timings["instances"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        bounds = fixed_mapping_bounds()
        self._mapper.configure_mapping_bounds(bounds)
        support_plane = getattr(table, "support_plane", None)
        if support_plane is not None:
            remaining = []
            for view in prepared:
                separated = self._segmenter.separate_support_observation(view, support_plane)
                camera_metrics[view.camera_id] = MappingProxyType({
                    **camera_metrics[view.camera_id],
                    "classified_support_point_count": view.background_point_count - separated.background_point_count,
                    "remaining_background_point_count": separated.background_point_count,
                })
                remaining.append(separated)
            prepared = remaining
        integration = self._mapper.integrate_multiview(prepared, table_model=table.model)
        LOGGER.info("semantic ESDF mapper_s=%.3f", time.perf_counter() - stage_started)
        timings["background"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        colorizer = SurfaceColorizer(views, voxel_size_m=0.01)
        colored_geometries = []
        for geometry in geometries:
            if geometry.kind == "voxel" and colorizer.views:
                model = dict(geometry.model)
                model["colors_rgb"] = colorizer.colors(
                    model["centers_base_m"], geometry.instance_id,
                ).tolist()
                geometry = replace(geometry, model=model)
            colored_geometries.append(geometry)
        geometries = colored_geometries
        if colorizer.views:
            data = integration.pointcloud_data
            count = struct.unpack_from("<I", data)[0]
            centers = np.frombuffer(data, dtype="<f4", count=count * 3, offset=4).reshape(-1, 3)
            integration = replace(integration, pointcloud_data=(
                data[:4 + count * 12] + colorizer.colors(centers).tobytes()
            ))
        timings["preview_rgb"] = (time.perf_counter() - stage_started) * 1000
        timings["build_total"] = (time.perf_counter() - build_started) * 1000
        timings.update(getattr(self._mapper, "timings_ms", {}))

        scene_model: dict[str, dict[str, Any]] = {
            "cuboid": {"semantic_table_forbidden": dict(table.model)},
            "mesh": {},
            "voxel": {},
        }
        for geometry in geometries:
            scene_model[geometry.kind][geometry.name] = dict(geometry.model)
            instance_metrics[geometry.instance_id] = MappingProxyType(
                {
                    "geometry": geometry.kind,
                    "input_point_count": int(geometry.input_point_count),
                    "filtered_point_count": int(geometry.filtered_point_count),
                    "camera_count": sum(
                        bool(len(points))
                        for points in instance_points[geometry.instance_id].values()
                    ),
                }
            )
        scene_data = create_mixed_esdf_scene(
            self._scene_type,
            getattr(self._mapper, "scene_voxel_grids", self._mapper.esdf_grid),
            scene_model,
        )
        return SemanticEsdfCandidate(
            observation_set_id=observation_set_id,
            selected_instance_id=selected,
            selection_generation=generation,
            scene_data=scene_data,
            scene_model=MappingProxyType(
                {key: MappingProxyType(dict(value)) for key, value in scene_model.items()}
            ),
            bounds=bounds,
            table=table,
            integration=integration,
            camera_metrics=MappingProxyType(camera_metrics),
            instance_metrics=MappingProxyType(instance_metrics),
            timings_ms=MappingProxyType(timings),
        )


class FixedEsdfMapping:
    """Build a private candidate and retain the last complete published map."""

    def __init__(
        self,
        *,
        input_provider: Any | None,
        tabletop_provider: Any | None,
        begin_candidate: Callable[[], Any],
        filter_robot_observation: Callable[[Mapping[str, Any]], Any],
        integrate_candidate: Callable[..., Any],
        set_candidate_scene: Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
        validate_candidate: Callable[[Mapping[str, Any]], Any],
        publish_candidate: Callable[[], Any],
        reconcile_candidate: Callable[[str], Any],
        discard_candidate: Callable[[], Any],
        scene_prerequisites: Callable[[], Any] | None = None,
        prepare_scene: Callable[..., Any] | None = None,
        required_ready_frames: int = 20,
        max_observation_attempts: int = 240,
        interval_s: float = 0.05,
        build_timeout_s: float = DEFAULT_BUILD_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
        deadline_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if int(required_ready_frames) < 1:
            raise ValueError("required_ready_frames must be positive")
        if int(max_observation_attempts) < int(required_ready_frames):
            raise ValueError("max_observation_attempts must cover required_ready_frames")
        self._scene_prerequisites = scene_prerequisites
        self._prepare_scene = prepare_scene
        self._begin_candidate = begin_candidate
        self._filter_robot_observation = filter_robot_observation
        self._integrate_candidate = integrate_candidate
        self._set_candidate_scene = set_candidate_scene
        self._validate_candidate = validate_candidate
        self._publish_candidate = publish_candidate
        self._reconcile_candidate = reconcile_candidate
        self._discard_candidate = discard_candidate
        self._required_ready_frames = int(required_ready_frames)
        self._max_observation_attempts = int(max_observation_attempts)
        self._interval_s = max(0.001, float(interval_s))
        self._build_timeout_s = float(build_timeout_s)
        if not math.isfinite(self._build_timeout_s) or self._build_timeout_s <= 0.0:
            raise ValueError("build_timeout_s must be positive and finite")
        self._clock = clock
        self._deadline_clock = deadline_clock
        self._lock = threading.RLock()
        if input_provider is not None:
            if not callable(getattr(input_provider, "prerequisites", None)):
                raise TypeError("calibrated depth provider must implement prerequisites()")
            if not callable(getattr(input_provider, "snapshot", None)):
                raise TypeError("calibrated depth provider must implement snapshot()")
        if tabletop_provider is not None:
            if not callable(
                getattr(tabletop_provider, "tabletop_prerequisites", None)
            ):
                raise TypeError(
                    "tabletop provider must implement tabletop_prerequisites()"
                )
            if not callable(
                getattr(tabletop_provider, "process_depth_observations", None)
            ):
                raise TypeError(
                    "tabletop provider must implement process_depth_observations()"
                )
        self._input_provider = input_provider
        self._tabletop_provider = tabletop_provider
        self._thread: threading.Thread | None = None
        self._build_admitted = False
        self._closed = False
        self._stop_event = threading.Event()
        self._idle_event = threading.Event()
        self._idle_event.set()
        self._state = "empty"
        self._stage = "idle"
        self._last_error = ""
        self._last_sequence = None
        self._build_started_at = None
        self._build_deadline = None
        self._build_id: str | None = None
        self._captured_frame_count = 0
        self._robot_filtered_frame_count = 0
        self._integrated_frame_count = 0
        self._published: dict[str, Any] | None = None
        self._published_at: float | None = None
        self._pending_publish: dict[str, Any] | None = None

    @property
    def has_pending_publish(self) -> bool:
        with self._lock:
            return self._pending_publish is not None

    def rebind_providers(
        self,
        *,
        input_provider: Any | None,
        tabletop_provider: Any | None,
    ) -> None:
        """Refresh provider references after Extension reactivation."""

        with self._lock:
            self._input_provider = input_provider
            self._tabletop_provider = tabletop_provider

    def _prerequisites(self) -> dict[str, Any]:
        if self._scene_prerequisites is not None:
            return _payload(self._scene_prerequisites())
        with self._lock:
            input_provider = self._input_provider
            tabletop_provider = self._tabletop_provider
        if input_provider is None:
            return {
                "ok": False,
                "error": "esdf_input_unavailable",
                "message": "ESDF depth input provider is unavailable.",
            }
        result = _payload(input_provider.prerequisites())
        if result.get("ok") is False:
            return result
        if result.get("camera_running") is not True:
            return {
                "ok": False,
                "error": "camera_preview_required",
                "message": "Camera preview must be running before map construction.",
            }
        if result.get("depth_capture_enabled") is not True:
            return {
                "ok": False,
                "error": "esdf_depth_required",
                "message": "Head-camera depth must be enabled before map construction.",
            }
        if tabletop_provider is None:
            return {
                "ok": False,
                "error": "tabletop_provider_unavailable",
                "message": "Open3D tabletop perception provider is unavailable.",
            }
        tabletop = _payload(tabletop_provider.tabletop_prerequisites())
        if tabletop.get("ok") is False:
            return tabletop
        return {**result, "ok": True, "tabletop": tabletop}

    def build(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                return {
                    **self.status(),
                    "ok": False,
                    "error": "fixed_esdf_mapping_closed",
                    "message": "Fixed ESDF mapping has been stopped.",
                }
            if self._pending_publish is not None:
                pending = True
            else:
                pending = False
            if self._build_admitted:
                return {
                    **self.status(),
                    "ok": False,
                    "error": "fixed_esdf_build_in_progress",
                    "message": "A fixed ESDF map is already being built.",
                }
            if not pending:
                self._build_admitted = True
                self._stop_event.clear()
                self._idle_event.clear()
        if pending:
            self.reconcile_pending_publish()
            return {
                **self.status(),
                "ok": False,
                "error": "fixed_esdf_publish_unknown",
                "message": (
                    "The previous fixed ESDF publish is awaiting authoritative "
                    "reconciliation."
                ),
            }
        thread_started = False
        try:
            prerequisites = self._prerequisites()
            if prerequisites.get("ok") is False:
                return {**self.status(), **prerequisites}
            with self._lock:
                cancelled = self._closed or self._stop_event.is_set()
                if not cancelled:
                    self._state = "building"
                    self._stage = "reading_stereo_cache" if self._prepare_scene else "capturing_depth"
                    self._last_error = ""
                    self._last_sequence = None
                    self._build_started_at = self._clock()
                    self._build_deadline = (
                        self._deadline_clock() + self._build_timeout_s
                    )
                    self._build_id = uuid4().hex
                    self._captured_frame_count = 0
                    self._robot_filtered_frame_count = 0
                    self._integrated_frame_count = 0
                    self._thread = threading.Thread(
                        target=self._run,
                        name="curobo-fixed-esdf-builder",
                        daemon=True,
                    )
                    thread = self._thread
            if cancelled:
                self._fail(
                    "fixed_esdf_build_cancelled",
                    "Fixed ESDF map construction was cancelled.",
                )
                return {
                    **self.status(),
                    "ok": False,
                    "error": "fixed_esdf_build_cancelled",
                    "message": "Fixed ESDF map construction was cancelled.",
                }
            try:
                thread.start()
            except Exception as error:
                with self._lock:
                    self._thread = None
                self._fail("fixed_esdf_thread_start_failed", str(error))
                return {
                    **self.status(),
                    "ok": False,
                    "error": "fixed_esdf_thread_start_failed",
                    "message": str(error),
                }
            thread_started = True
            return self.status()
        finally:
            if not thread_started:
                with self._lock:
                    self._build_admitted = False
                    self._idle_event.set()

    def _raise_if_stopped(self) -> None:
        if self._stop_event.is_set():
            raise _BuildCancelled
        with self._lock:
            deadline = self._build_deadline
        if deadline is not None and self._deadline_clock() >= deadline:
            raise _BuildDeadlineExceeded

    def _remaining_timeout_s(self) -> float:
        self._raise_if_stopped()
        with self._lock:
            deadline = self._build_deadline
        if deadline is None:
            raise _BuildDeadlineExceeded
        return max(0.001, deadline - self._deadline_clock())

    def _invoke(
        self,
        operation: Callable[..., Any],
        *args,
        pass_control: bool = False,
        **kwargs,
    ) -> Any:
        timeout_s = self._remaining_timeout_s()
        if pass_control:
            kwargs["timeout_s"] = timeout_s
            kwargs["cancel_event"] = self._stop_event
            kwargs["deadline_monotonic_s"] = (
                self._deadline_clock() + timeout_s
            )
        result = operation(*args, **kwargs)
        self._raise_if_stopped()
        return result

    def _invoke_cleanup(
        self,
        operation: Callable[..., Any],
        *args,
        timeout_s: float,
        **kwargs,
    ) -> Any:
        cleanup_timeout_s = max(0.001, float(timeout_s))
        kwargs["timeout_s"] = cleanup_timeout_s
        kwargs["cancel_event"] = threading.Event()
        kwargs["deadline_monotonic_s"] = (
            self._deadline_clock() + cleanup_timeout_s
        )
        return operation(*args, **kwargs)

    def _set_pending_publish(self, build_id: str, selected: Mapping[str, Any]) -> None:
        with self._lock:
            self._pending_publish = {
                "build_id": str(build_id),
                "selected": deepcopy(dict(selected)),
            }

    def _mark_publish_unknown(self) -> None:
        with self._lock:
            self._state = "publish_unknown"
            self._stage = "publish_unknown"
            self._last_error = ""

    def _complete_publish(
        self,
        published: Mapping[str, Any],
        selected: Mapping[str, Any],
    ) -> None:
        combined = {
            **dict(published),
            **dict(selected["metadata"]),
        }
        if selected.get("geometry") is not None:
            combined["geometry"] = deepcopy(selected["geometry"])
        with self._lock:
            self._published = combined
            self._published_at = self._clock()
            self._pending_publish = None
            self._state = "published"
            self._stage = "published"
            self._last_error = ""

    def forget_publication(self, expected_scene_token: str) -> None:
        """Remove the host preview only after the GPU confirms cancellation."""
        with self._lock:
            if self._published and self._published.get("scene_token") == expected_scene_token:
                self._published = None
                self._published_at = None
                self._state = "idle"
                self._stage = "idle"
                self._last_error = ""

    def record_semantic_publication(
        self,
        published: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> None:
        """Mirror one authoritative semantic publish in the host snapshot."""

        if not isinstance(published, Mapping) or not isinstance(candidate, Mapping):
            raise TypeError("semantic publication and candidate must be mappings")
        revision = published.get("revision")
        scene_token = published.get("scene_token")
        if (
            published.get("ready") is not True
            or type(revision) is not int
            or revision < 1
            or not isinstance(scene_token, str)
            or not scene_token.startswith("fixed-esdf:")
            or published.get("integrated_frame_count") != 3
        ):
            raise ValueError("authoritative semantic ESDF publication is invalid")
        required = (
            "observation_set_id",
            "selected_instance_id",
            "selection_generation",
            "scene_model",
            "bounds",
            "tabletop",
            "camera_metrics",
            "instance_metrics",
        )
        if any(candidate.get(key) is None for key in required):
            raise ValueError("semantic ESDF candidate metadata is incomplete")
        metadata = {
            key: deepcopy(candidate[key])
            for key in required
        }
        metadata["robot_joints_at_capture"] = deepcopy(
            candidate.get("robot_joints_at_capture")
        )
        self._complete_publish(published, {"metadata": metadata})

    def _clear_pending_as_uncommitted(self, build_id: str) -> None:
        try:
            self._invoke_cleanup(
                self._discard_candidate,
                build_id,
                timeout_s=0.5,
            )
        except Exception:
            pass
        with self._lock:
            pending = self._pending_publish
            if pending is not None and pending.get("build_id") == build_id:
                self._pending_publish = None
            self._state = "failed"
            self._stage = "failed"
            self._last_error = (
                "fixed_esdf_publish_not_committed: cuRobo confirmed that the "
                "candidate was not published."
            )

    @staticmethod
    def _authoritative_publish(payload: Mapping[str, Any]) -> bool:
        token = str(payload.get("scene_token") or "")
        return (
            payload.get("ok") is True
            and payload.get("published") is True
            and token.startswith("fixed-esdf:")
        )

    def _start_pending_reconciliation(self, *, reopen: bool) -> bool:
        with self._lock:
            if self._pending_publish is None:
                return False
            if self._build_admitted:
                return True
            if self._closed and not reopen:
                return False
            if reopen:
                self._closed = False
                self._stop_event.clear()
            self._build_admitted = True
            self._idle_event.clear()
            self._stage = "reconciling_publish"
            self._thread = threading.Thread(
                target=self._run_pending_reconciliation,
                name="curobo-fixed-esdf-reconciler",
                daemon=True,
            )
            thread = self._thread
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._thread = None
                self._build_admitted = False
                self._stage = "publish_unknown"
                self._idle_event.set()
            raise
        return True

    def reconcile_pending_publish(self) -> bool:
        """Start one bounded reconciliation attempt for a pending publish."""

        return self._start_pending_reconciliation(reopen=False)

    def resume_pending_publish(self) -> bool:
        """Resume a retained pending transaction after Extension activation."""

        return self._start_pending_reconciliation(reopen=True)

    def _run_pending_reconciliation(self) -> None:
        try:
            with self._lock:
                pending = deepcopy(self._pending_publish)
            if pending is None:
                return
            build_id = str(pending["build_id"])
            reconciled = _payload(
                self._invoke_cleanup(
                    self._reconcile_candidate,
                    build_id,
                    timeout_s=2.0,
                )
            )
            if self._authoritative_publish(reconciled):
                self._complete_publish(reconciled, pending["selected"])
            elif reconciled.get("ok") is True and reconciled.get("published") is False:
                self._clear_pending_as_uncommitted(build_id)
            else:
                self._mark_publish_unknown()
        except Exception:
            self._mark_publish_unknown()
        finally:
            with self._lock:
                self._thread = None
                self._build_admitted = False
                self._idle_event.set()

    def _capture_robot_free_observations(self):
        observations = []
        attempts = 0
        while len(observations) < self._required_ready_frames:
            self._raise_if_stopped()
            attempts += 1
            if attempts > self._max_observation_attempts:
                self._fail(
                    "fixed_esdf_observation_timeout",
                    "Not enough new depth observations were received.",
                )
                return None
            started_at = self._deadline_clock()
            with self._lock:
                provider = self._input_provider
                after_sequence = self._last_sequence
            if provider is None:
                self._fail("esdf_input_unavailable", "ESDF input was removed.")
                return None
            request = {"camera_id": "head"}
            if after_sequence is not None:
                request["after_sequence"] = after_sequence
            captured = _payload(self._invoke(provider.snapshot, request))
            if captured.get("unchanged") is True:
                if self._stop_event.wait(self._interval_s):
                    raise _BuildCancelled
                continue
            if captured.get("ok") is False:
                self._fail_result(captured)
                return None
            sequence = captured.get("sequence")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
            ):
                self._fail(
                    "invalid_esdf_observation",
                    "ESDF observation sequence must be a non-negative integer.",
                )
                return None
            if after_sequence is not None and sequence <= after_sequence:
                if self._stop_event.wait(self._interval_s):
                    raise _BuildCancelled
                continue
            raw_observation = captured.get("observation")
            if not isinstance(raw_observation, Mapping):
                self._fail("invalid_esdf_observation", "ESDF observation is invalid.")
                return None
            with self._lock:
                self._stage = "removing_robot"
                self._captured_frame_count += 1
                self._last_sequence = sequence
            filtered = _payload(
                self._invoke(
                    self._filter_robot_observation,
                    raw_observation,
                    pass_control=True,
                )
            )
            if filtered.get("ok") is False:
                self._fail_result(filtered)
                return None
            robot_free = filtered.get("observation")
            if not isinstance(robot_free, Mapping):
                self._fail(
                    "invalid_robot_segmentation_response",
                    "cuRobo robot segmentation returned no filtered depth observation.",
                )
                return None
            observations.append(dict(robot_free))
            with self._lock:
                self._robot_filtered_frame_count += 1
                self._stage = "capturing_depth"
            elapsed = max(0.0, self._deadline_clock() - started_at)
            if len(observations) < self._required_ready_frames:
                if self._stop_event.wait(
                    max(0.0, self._interval_s - elapsed)
                ):
                    raise _BuildCancelled
        return observations

    def _prepare_depth_scene(self, build_id):
        started = _payload(
            self._invoke(
                self._begin_candidate,
                build_id,
                pass_control=True,
            )
        )
        if started.get("ok") is False:
            self._fail_result(started)
            return
        observations = self._capture_robot_free_observations()
        if observations is None:
            return
        self._raise_if_stopped()
        with self._lock:
            tabletop_provider = self._tabletop_provider
            self._stage = "open3d_processing"
        if tabletop_provider is None:
            self._fail(
                "tabletop_provider_unavailable",
                "Open3D tabletop perception provider was removed.",
            )
            return
        processed = _payload(
            self._invoke(
                tabletop_provider.process_depth_observations,
                observations,
                pass_control=True,
            )
        )
        self._raise_if_stopped()
        if processed.get("ok") is False:
            self._fail_result(processed)
            return
        sanitized = processed.get("observations")
        if (
            isinstance(sanitized, (str, bytes))
            or not isinstance(sanitized, Sequence)
            or len(sanitized) != len(observations)
            or not all(isinstance(item, Mapping) for item in sanitized)
        ):
            self._fail(
                "invalid_open3d_response",
                "Open3D returned invalid sanitized depth observations.",
            )
            return
        last_result = None
        with self._lock:
            self._stage = "integrating_esdf"
        for index, observation in enumerate(sanitized):
            self._raise_if_stopped()
            last_result = _payload(
                self._invoke(
                    self._integrate_candidate,
                    observation,
                    build_id=build_id,
                    finalize=index == len(sanitized) - 1,
                    pass_control=True,
                )
            )
            self._raise_if_stopped()
            if last_result.get("ok") is False:
                self._fail_result(last_result)
                return
            with self._lock:
                self._integrated_frame_count += 1
        if last_result is None or last_result.get("ready") is not True:
            self._fail(
                "fixed_esdf_candidate_not_ready",
                "cuRobo Mapper did not produce a ready ESDF candidate.",
            )
            return
        scene_candidates = processed.get("scene_candidates")
        if (
            isinstance(scene_candidates, (str, bytes))
            or not isinstance(scene_candidates, Sequence)
            or not scene_candidates
        ):
            self._fail(
                "invalid_tabletop_scene",
                "Open3D returned no expanded tabletop candidates.",
            )
            return
        selected = None
        last_observation = dict(sanitized[-1])
        for candidate in scene_candidates:
            self._raise_if_stopped()
            if not isinstance(candidate, Mapping):
                continue
            geometry = candidate.get("geometry")
            margin = candidate.get("robot_side_margin_m")
            if not isinstance(geometry, Mapping):
                continue
            metadata = {
                "coordinate_frame": "base",
                "tabletop": processed.get("tabletop"),
                "open3d": processed.get("filter"),
                "robot_side_margin_m": margin,
            }
            with self._lock:
                self._stage = "validating_initial_collision"
            staged = _payload(
                self._invoke(
                    self._set_candidate_scene,
                    geometry,
                    metadata,
                    build_id=build_id,
                    pass_control=True,
                )
            )
            if staged.get("ok") is False:
                self._fail_result(staged)
                return
            validated = _payload(
                self._invoke(
                    self._validate_candidate,
                    last_observation,
                    build_id=build_id,
                    pass_control=True,
                )
            )
            self._raise_if_stopped()
            if validated.get("ok") is False:
                self._fail_result(validated)
                return
            if validated.get("collision_free") is True:
                selected = {
                    "geometry": dict(geometry),
                    "robot_side_margin_m": margin,
                    "metadata": metadata,
                }
                break
        if selected is None:
            self._fail(
                "fixed_esdf_initial_collision",
                "Every expanded tabletop candidate collides with the robot at the captured mapping pose.",
            )
            return
        return selected

    def _scene_progress(self, stage):
        self._raise_if_stopped()
        with self._lock:
            self._stage = stage

    def _run(self) -> None:
        try:
            with self._lock:
                build_id = self._build_id
            if build_id is None:
                raise RuntimeError("fixed ESDF build has no transaction id")
            if self._prepare_scene is None:
                selected = self._prepare_depth_scene(build_id)
            else:
                prepared = _payload(self._invoke(
                    self._prepare_scene, build_id, pass_control=True,
                    progress=self._scene_progress,
                ))
                if prepared.get("ok") is False:
                    self._fail_result(prepared)
                    return
                if prepared.get("candidate_ready") is not True or prepared.get("integrated_frame_count") != 3:
                    raise ValueError("three-camera FFS candidate is not ready")
                selected = {"metadata": prepared}
                with self._lock:
                    self._captured_frame_count = 3
                    self._robot_filtered_frame_count = 3
                    self._integrated_frame_count = 3
            if selected is None:
                return
            self._raise_if_stopped()
            with self._lock:
                self._stage = "publishing"
            publish_error = None
            try:
                published = _payload(
                    self._invoke(
                        self._publish_candidate,
                        build_id,
                        pass_control=True,
                    )
                )
            except Exception as error:
                publish_error = error
                published = {}
            token = str(published.get("scene_token") or "")
            publish_valid = (
                published.get("ok") is True
                and published.get("ready") is not False
                and token.startswith("fixed-esdf:")
            )
            if not publish_valid:
                self._set_pending_publish(build_id, selected)
                with self._lock:
                    self._stage = "reconciling_publish"
                attempted_reconcile = False
                while True:
                    with self._lock:
                        deadline = self._build_deadline
                    remaining_s = (
                        0.0
                        if deadline is None
                        else deadline - self._deadline_clock()
                    )
                    if attempted_reconcile and (
                        self._stop_event.is_set() or remaining_s <= 0.0
                    ):
                        self._mark_publish_unknown()
                        return
                    try:
                        attempted_reconcile = True
                        reconciled = _payload(
                            self._invoke_cleanup(
                                self._reconcile_candidate,
                                build_id,
                                timeout_s=min(2.0, max(0.001, remaining_s)),
                            )
                        )
                        break
                    except Exception:
                        wait_s = min(self._interval_s, max(0.0, remaining_s))
                        if self._stop_event.wait(wait_s):
                            self._mark_publish_unknown()
                            return
                if (
                    self._authoritative_publish(reconciled)
                ):
                    published = reconciled
                elif publish_error is not None:
                    with self._lock:
                        self._pending_publish = None
                    raise publish_error
                elif published.get("ok") is False:
                    with self._lock:
                        self._pending_publish = None
                    self._fail_result(published)
                    return
                else:
                    with self._lock:
                        self._pending_publish = None
                    self._fail(
                        "invalid_fixed_esdf_scene_token",
                        "Published map returned an invalid scene token.",
                    )
                    return
            self._complete_publish(published, selected)
        except _BuildCancelled:
            self._fail(
                "fixed_esdf_build_cancelled",
                "Fixed ESDF map construction was cancelled.",
            )
        except _BuildDeadlineExceeded:
            self._fail(
                "fixed_esdf_build_timeout",
                "Fixed ESDF map construction exceeded its time limit.",
            )
        except Exception as error:
            self._fail(getattr(error, "code", "fixed_esdf_build_failed"), str(error))
        finally:
            with self._lock:
                self._thread = None
                self._build_admitted = False
                self._build_id = None
                self._build_deadline = None
                self._idle_event.set()

    def request_close(self) -> None:
        """Signal cancellation without waiting for in-flight owned resources."""

        with self._lock:
            self._closed = True
            self._stop_event.set()

    def close(self, *, timeout_s: float = 5.0) -> None:
        """Cancel one candidate build and wait for its worker to release ownership."""

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout_s must be a finite non-negative number")
        deadline = time.monotonic() + timeout
        self.request_close()
        with self._lock:
            admitted = self._build_admitted
            thread = self._thread
        if admitted and not self._idle_event.wait(timeout):
            raise RuntimeError("fixed ESDF builder did not stop before timeout")
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                raise RuntimeError("fixed ESDF builder did not stop before timeout")

    def _fail_result(self, result: Mapping[str, Any]) -> None:
        self._fail(
            str(result.get("error") or "fixed_esdf_build_failed"),
            str(result.get("message") or "Fixed ESDF map construction failed."),
        )

    def _fail(self, code: str, message: str) -> None:
        with self._lock:
            build_id = self._build_id
            pending = self._pending_publish
        if pending is not None and pending.get("build_id") == build_id:
            self._mark_publish_unknown()
            return
        if build_id is not None:
            try:
                self._invoke_cleanup(
                    self._discard_candidate,
                    build_id,
                    timeout_s=0.5,
                )
            except Exception:
                pass
        with self._lock:
            self._state = "failed"
            self._stage = "failed"
            self._last_error = f"{code}: {message}"

    def status(self) -> dict[str, Any]:
        with self._lock:
            published = dict(self._published or {})
            state = self._state
            stage = self._stage
            running = self._thread is not None and self._thread.is_alive()
            started_at = self._build_started_at
            last_error = self._last_error
            captured = self._captured_frame_count
            robot_filtered = self._robot_filtered_frame_count
            integrated = self._integrated_frame_count
            published_at = self._published_at
            publish_pending = self._pending_publish is not None
        current_time = (
            self._clock()
            if started_at is not None or published_at is not None
            else None
        )
        return {
            "ok": True,
            "state": state,
            "stage": stage,
            "ready": bool(published),
            "thread_running": running,
            "last_error": last_error,
            "required_frame_count": self._required_ready_frames,
            "build_age_s": (
                None if started_at is None else max(0.0, current_time - started_at)
            ),
            "published_age_s": (
                None
                if published_at is None
                else max(0.0, current_time - published_at)
            ),
            "captured_frame_count": captured,
            "robot_filtered_frame_count": robot_filtered,
            "integrated_frame_count": integrated,
            "publish_pending": publish_pending,
            "revision": int(published.get("revision", 0) or 0),
            "point_count": int(published.get("point_count", 0) or 0),
            "scene_token": published.get("scene_token"),
            "selected_instance_id": published.get("selected_instance_id"),
            "selection_generation": published.get("selection_generation"),
            "diagnostics": _published_diagnostics(published),
        }

    def apply_selection(
        self,
        published: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> None:
        """Mirror an authoritative service selection without rebuilding."""

        if not isinstance(published, Mapping):
            raise TypeError("published selection must be a mapping")
        revision = published.get("revision")
        scene_token = published.get("scene_token")
        selected_instance_id = published.get("selected_instance_id")
        selection_generation = published.get("selection_generation")
        if (
            type(expected_revision) is not int
            or type(revision) is not int
            or revision <= expected_revision
            or not isinstance(scene_token, str)
            or not scene_token.startswith("fixed-esdf:")
            or not isinstance(selected_instance_id, str)
            or not selected_instance_id
            or type(selection_generation) is not int
            or selection_generation < 1
        ):
            raise ValueError("published semantic ESDF selection is invalid")
        with self._lock:
            current = self._published
            if current is None or current.get("revision") != expected_revision:
                raise RuntimeError("fixed ESDF scene revision changed")
            current.update(
                revision=revision,
                scene_token=scene_token,
                selected_instance_id=selected_instance_id,
                selection_generation=selection_generation,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            published = dict(self._published or {})
        return {**published, **self.status()}


__all__ = [
    "FixedEsdfMapping",
    "SemanticEsdfCandidate",
    "SemanticEsdfCandidateBuilder",
]

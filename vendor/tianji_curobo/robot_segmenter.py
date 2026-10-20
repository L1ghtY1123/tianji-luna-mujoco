"""Project-side boundary for cuRobo v0.8.0 robot depth segmentation."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any

from .contracts import CuroboToolError
from .curobo_robot_config import ARM_CONFIG_SPECS, create_robot_segmentation_config
from .mixed_esdf_scene import table_footprint_mask
from .scene_config import SCENE_VOXEL_SIZE_M


def surface_component_inlier_mask(points, *, minimum_voxels=8, minimum_pixels=32):
    """Reject only small, sparsely supported components on the 10 mm grid."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if not len(points):
        return np.zeros(0, dtype=bool)
    cells, inverse = np.unique(
        np.floor(np.asarray(points) / SCENE_VOXEL_SIZE_M).astype(np.int64),
        axis=0, return_inverse=True,
    )
    # 26-neighbour connectivity includes diagonally observed surfaces.
    pairs = cKDTree(cells).query_pairs(math.sqrt(3) + 1e-6, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(cells), len(cells)))
    _, labels = connected_components(graph, directed=False)
    voxel_counts = np.bincount(labels)
    pixel_counts = np.bincount(labels[inverse])
    # A dense small top may be a real unlabelled obstacle; retain ambiguity.
    supported = (voxel_counts >= minimum_voxels) | (pixel_counts >= minimum_pixels)
    return supported[labels[inverse]]


def _error(message: str, *, code: str, stage="segmentation", status=422):
    return CuroboToolError(
        message,
        stage=stage,
        code=code,
        status=status,
    )


def _finite_numbers(value: object, *, count: int, name: str) -> tuple[float, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error(f"{name} must contain {count} numbers", code="CUROBO_INVALID_REQUEST")
    values = tuple(float(item) for item in value)
    if len(values) != count or not all(math.isfinite(item) for item in values):
        raise _error(f"{name} must contain {count} finite numbers", code="CUROBO_INVALID_REQUEST")
    return values


def _matrix(value: object, *, size: int, name: str) -> tuple[tuple[float, ...], ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error(f"{name} must be a {size}x{size} matrix", code="CUROBO_INVALID_REQUEST")
    rows = tuple(
        _finite_numbers(row, count=size, name=f"{name}[{index}]")
        for index, row in enumerate(value)
    )
    if len(rows) != size:
        raise _error(f"{name} must be a {size}x{size} matrix", code="CUROBO_INVALID_REQUEST")
    return rows


@dataclass(frozen=True)
class RobotSegmentationRequest:
    depth_png_base64: str
    depth_intrinsics: tuple[tuple[float, ...], ...]
    depth_calibration_resolution: tuple[float, float]
    camera_pose_base_m: tuple[tuple[float, ...], ...]
    depth_to_rgb_translation_mm: tuple[float, float, float]
    joint_positions_deg: Mapping[str, tuple[float, ...]]
    voxel_size_mm: float

    @classmethod
    def from_mapping(cls, payload: object) -> "RobotSegmentationRequest":
        if not isinstance(payload, Mapping):
            raise _error("request body must be an object", code="CUROBO_INVALID_REQUEST")
        encoded_depth = payload.get("depth_png_base64")
        if not isinstance(encoded_depth, str) or not encoded_depth:
            raise _error("depth_png_base64 is required", code="CUROBO_INVALID_REQUEST")
        try:
            base64.b64decode(encoded_depth, validate=True)
        except ValueError as error:
            raise _error(
                "depth_png_base64 must be valid base64",
                code="CUROBO_INVALID_REQUEST",
            ) from error

        joint_payload = payload.get("joint_positions_deg")
        if not isinstance(joint_payload, Mapping):
            raise _error(
                "joint_positions_deg must contain arms A and B",
                code="CUROBO_INVALID_REQUEST",
            )
        joint_positions_deg = {
            arm: _finite_numbers(
                joint_payload.get(arm),
                count=7,
                name=f"joint_positions_deg.{arm}",
            )
            for arm in ("A", "B")
        }
        resolution = _finite_numbers(
            payload.get("depth_calibration_resolution", (0.0, 0.0)),
            count=2,
            name="depth_calibration_resolution",
        )
        if resolution[0] <= 0.0 or resolution[1] <= 0.0:
            raise _error(
                "depth_calibration_resolution must be positive",
                code="CUROBO_INVALID_REQUEST",
            )
        voxel_size_mm = float(payload.get("voxel_size_mm", 0.0))
        if not math.isfinite(voxel_size_mm) or voxel_size_mm < 0.0:
            raise _error(
                "voxel_size_mm must be a finite non-negative number",
                code="CUROBO_INVALID_REQUEST",
            )
        return cls(
            depth_png_base64=encoded_depth,
            depth_intrinsics=_matrix(
                payload.get("depth_intrinsics"),
                size=3,
                name="depth_intrinsics",
            ),
            depth_calibration_resolution=(resolution[0], resolution[1]),
            camera_pose_base_m=_matrix(
                payload.get("camera_pose_base_m"),
                size=4,
                name="camera_pose_base_m",
            ),
            depth_to_rgb_translation_mm=_finite_numbers(
                payload.get("depth_to_rgb_translation_mm"),
                count=3,
                name="depth_to_rgb_translation_mm",
            ),
            joint_positions_deg=joint_positions_deg,
            voxel_size_mm=voxel_size_mm,
        )


@dataclass(frozen=True)
class RobotSegmentationResult:
    pointcloud_data: bytes
    filtered_depth_png: bytes
    base_pointcloud_xyz: bytes
    point_count: int
    removed_point_count: int


@dataclass(frozen=True)
class PreparedRobotDepth:
    camera_observation: Any
    filtered_depth_m: Any
    intrinsics: tuple[tuple[float, ...], ...]
    removed_point_count: int


@dataclass(frozen=True)
class FfsDepthObservationRequest:
    """One raw FFS left-depth observation for cuRobo mapping.

    RGB-aligned depth is intentionally absent: it is useful for GraspGen mask
    sampling, but resampling it again for Mapper would introduce holes.
    """

    camera_id: str
    stereo_sequence: int
    depth_left_m: Any
    valid_left_mask: Any
    left_intrinsics: tuple[tuple[float, ...], ...]
    T_base_left_camera: tuple[tuple[float, ...], ...]
    joint_positions_deg: Mapping[str, tuple[float, ...]]
    frame: str

    @classmethod
    def from_mapping(cls, payload: object) -> "FfsDepthObservationRequest":
        if not isinstance(payload, Mapping):
            raise ValueError("FFS depth observation must be a mapping")
        if any(
            name in payload
            for name in ("depth_rgb_m", "depth_m", "points_rgb_m", "valid_rgb_mask")
        ):
            raise ValueError("RGB-aligned depth must not enter cuRobo Mapper")
        import numpy as np

        camera_id = str(payload.get("camera_id") or "")
        if camera_id not in {"1", "2", "3"}:
            raise ValueError("camera_id must be 1, 2, or 3")
        stereo_sequence = payload.get("stereo_sequence")
        if type(stereo_sequence) is not int or stereo_sequence < 0:
            raise ValueError("stereo_sequence must be a non-negative integer")
        depth = np.asarray(payload.get("depth_left_m"))
        valid = np.asarray(payload.get("valid_left_mask"))
        if depth.dtype != np.float32 or depth.ndim != 2 or not depth.size:
            raise ValueError("depth_left_m must be a non-empty float32 image")
        if valid.dtype != np.uint8 or valid.shape != depth.shape:
            raise ValueError("valid_left_mask must be a uint8 image matching left depth")
        if not np.isin(valid, (0, 1)).all():
            raise ValueError("valid_left_mask must be binary")
        selected = valid != 0
        if (
            np.isinf(depth).any()
            or not np.isfinite(depth[selected]).all()
            or np.any(depth[selected] <= 0.0)
        ):
            raise ValueError("valid FFS left depth must be finite and positive")
        depth = np.ascontiguousarray(depth, dtype=np.float32).copy()
        valid = np.ascontiguousarray(valid, dtype=np.uint8).copy()
        depth.setflags(write=False)
        valid.setflags(write=False)
        joints = payload.get("robot_joints_at_capture")
        if not isinstance(joints, Mapping):
            raise ValueError("robot_joints_at_capture must contain arms A and B")
        joint_positions_deg = {
            arm: _finite_numbers(
                joints.get(arm),
                count=7,
                name=f"robot_joints_at_capture.{arm}",
            )
            for arm in ("A", "B")
        }
        return cls(
            camera_id=camera_id,
            stereo_sequence=stereo_sequence,
            depth_left_m=depth,
            valid_left_mask=valid,
            left_intrinsics=_matrix(
                payload.get("left_intrinsics"), size=3, name="left_intrinsics"
            ),
            T_base_left_camera=_matrix(
                payload.get("T_base_left_camera"),
                size=4,
                name="T_base_left_camera",
            ),
            joint_positions_deg=MappingProxyType(joint_positions_deg),
            frame=f"camera:{camera_id}:left",
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "stereo_sequence": self.stereo_sequence,
            "depth_left_m": self.depth_left_m,
            "valid_left_mask": self.valid_left_mask,
            "left_intrinsics": self.left_intrinsics,
            "T_base_left_camera": self.T_base_left_camera,
            "robot_joints_at_capture": dict(self.joint_positions_deg),
        }


@dataclass(frozen=True)
class PreparedFfsDepth:
    camera_id: str
    stereo_sequence: int
    camera_observation: Any
    filtered_depth_m: Any
    base_pointcloud_m: Any
    intrinsics_tensor: Any
    pose_matrix_tensor: Any
    removed_point_count: int


@dataclass(frozen=True)
class SemanticProjectionRequest:
    rgb_size: tuple[int, int]
    rgb_intrinsics: tuple[tuple[float, ...], ...]
    left_to_rgb: tuple[tuple[float, ...], ...]
    instance_masks: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: object, *, allow_empty: bool = False) -> "SemanticProjectionRequest":
        if not isinstance(payload, Mapping):
            raise ValueError("semantic projection must be a mapping")
        import numpy as np

        size = payload.get("rgb_size")
        if (
            isinstance(size, (str, bytes))
            or not isinstance(size, Sequence)
            or len(size) != 2
            or any(type(value) is not int or value <= 0 for value in size)
        ):
            raise ValueError("rgb_size must contain two positive integers")
        rgb_size = (int(size[0]), int(size[1]))
        masks = payload.get("instance_masks")
        if not isinstance(masks, Mapping) or (not masks and not allow_empty):
            raise ValueError("instance_masks must contain at least one instance")
        frozen_masks = {}
        expected_shape = (rgb_size[1], rgb_size[0])
        for raw_name, value in masks.items():
            name = str(raw_name)
            mask = np.asarray(value)
            if (
                not name
                or mask.dtype != np.uint8
                or mask.shape != expected_shape
                or not np.isin(mask, (0, 1)).all()
            ):
                raise ValueError("every semantic instance mask must be a binary RGB image")
            copied = np.ascontiguousarray(mask, dtype=np.uint8).copy()
            copied.setflags(write=False)
            frozen_masks[name] = copied
        transform = _matrix(payload.get("left_to_rgb"), size=4, name="left_to_rgb")
        rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5)
        ):
            raise ValueError("left_to_rgb must be a rigid transform")
        return cls(
            rgb_size=rgb_size,
            rgb_intrinsics=_matrix(
                payload.get("rgb_intrinsics"), size=3, name="rgb_intrinsics"
            ),
            left_to_rgb=transform,
            instance_masks=MappingProxyType(frozen_masks),
        )


@dataclass(frozen=True)
class PreparedSemanticDepth:
    camera_id: str
    stereo_sequence: int
    camera_observation: Any
    intrinsics_tensor: Any
    pose_matrix_tensor: Any
    instance_points_base_m: Mapping[str, Any]
    background_points_base_m: Any
    input_point_count: int
    filtered_point_count: int
    background_point_count: int
    removed_point_count: int
    instance_depth_samples: Mapping[str, Any] | None = None


class RobotSegmenterAdapter:
    """Keep SDK types behind a small binary pointcloud contract."""

    def __init__(self, *, sdk_boundary: Any) -> None:
        segment = getattr(sdk_boundary, "segment", None)
        if not callable(segment):
            raise TypeError("sdk_boundary must provide segment(request)")
        self._sdk_boundary = sdk_boundary

    @property
    def sdk_boundary(self) -> Any:
        return self._sdk_boundary

    def segment_robot(
        self,
        request: RobotSegmentationRequest,
        *,
        include_base_pointcloud: bool = False,
    ) -> RobotSegmentationResult:
        if not isinstance(request, RobotSegmentationRequest):
            raise TypeError("request must be a RobotSegmentationRequest")
        if not isinstance(include_base_pointcloud, bool):
            raise TypeError("include_base_pointcloud must be a bool")
        (
            pointcloud_data,
            filtered_depth_png,
            base_pointcloud_xyz,
            point_count,
            removed_point_count,
        ) = self._sdk_boundary.segment(
            request,
            include_base_pointcloud=include_base_pointcloud,
        )
        if not isinstance(pointcloud_data, (bytes, bytearray, memoryview)):
            raise TypeError("robot segmenter returned invalid pointcloud data")
        if not isinstance(filtered_depth_png, (bytes, bytearray, memoryview)):
            raise TypeError("robot segmenter returned invalid filtered depth data")
        if not isinstance(base_pointcloud_xyz, (bytes, bytearray, memoryview)):
            raise TypeError("robot segmenter returned invalid base pointcloud data")
        return RobotSegmentationResult(
            pointcloud_data=bytes(pointcloud_data),
            filtered_depth_png=bytes(filtered_depth_png),
            base_pointcloud_xyz=bytes(base_pointcloud_xyz),
            point_count=int(point_count),
            removed_point_count=int(removed_point_count),
        )


class CuroboRobotSegmenterBoundary:
    """Use only cuRobo v0.8.0 public perception and type exports."""

    def __init__(
        self,
        *,
        robot_config_path: Path,
        urdf_path: Path,
        asset_root_path: Path,
    ) -> None:
        try:
            import cv2
            import numpy as np
            import torch
            import yaml
            from curobo.kinematics import Kinematics, KinematicsCfg
            from curobo.perception import FilterDepth, RobotSegmenter
            from curobo.types import CameraObservation, DeviceCfg, JointState, Pose

            with Path(robot_config_path).open("r", encoding="utf-8") as config_file:
                combined_config = yaml.safe_load(config_file)
            robot_config = create_robot_segmentation_config(
                combined_config,
                urdf_path=urdf_path,
                asset_root_path=asset_root_path,
            )
            device_cfg = DeviceCfg()
            kinematics_config = KinematicsCfg.from_data_dict(
                robot_config,
                device_cfg=device_cfg,
            )
            # cuRobo v0.8.0's from_robot_file() selects bfloat16, while the
            # same release's segmentation CUDA checks accept float16/float32.
            kinematics = Kinematics(kinematics_config)
            segmenter = RobotSegmenter(
                kinematics,
                distance_threshold=0.05,
                use_cuda_graph=True,
                ops_dtype=torch.float32,
            )
        except CuroboToolError:
            raise
        except Exception as error:
            raise _error(
                "failed to initialize cuRobo RobotSegmenter",
                code="CUROBO_ROBOT_SEGMENTER_CONFIGURATION_FAILED",
                stage="configuration",
                status=500,
            ) from error

        self._cv2 = cv2
        self._np = np
        self._torch = torch
        self._camera_observation_type = CameraObservation
        self._device_cfg = device_cfg
        self._joint_state_type = JointState
        self._pose_type = Pose
        self._kinematics = kinematics
        self._segmenter = segmenter
        self._filter_depth_type = FilterDepth
        self._depth_filter = None
        self._depth_filter_shape = None
        self._joint_names = tuple(
            ARM_CONFIG_SPECS["A"]["joint_names"]
            + ARM_CONFIG_SPECS["B"]["joint_names"]
        )
        color_indices = np.arange(256, dtype=np.uint8).reshape(256, 1)
        self._color_lut = cv2.applyColorMap(
            color_indices,
            cv2.COLORMAP_TURBO,
        ).reshape(256, 3)[:, ::-1].copy()

    @property
    def device_cfg(self):
        return self._device_cfg

    @property
    def numpy(self):
        return self._np

    def robot_spheres(self, joint_positions_deg: Mapping[str, Sequence[float]]):
        """Return every official cuRobo collision sphere at one dual-arm pose."""

        if not isinstance(joint_positions_deg, Mapping):
            raise _error(
                "joint_positions_deg must contain both arms",
                code="CUROBO_INVALID_REQUEST",
            )
        positions_deg = tuple(
            value
            for arm in ("A", "B")
            for value in _finite_numbers(
                joint_positions_deg.get(arm),
                count=7,
                name=f"joint_positions_deg.{arm}",
            )
        )
        joint_positions_rad = [[math.radians(value) for value in positions_deg]]
        sphere_batches = self._kinematics.get_robot_as_spheres(
            self._device_cfg.to_device(joint_positions_rad)
        )
        if not isinstance(sphere_batches, list) or len(sphere_batches) != 1:
            raise _error(
                "cuRobo returned an invalid robot sphere batch",
                code="CUROBO_BACKEND_FAILED",
                stage="sdk",
                status=500,
            )
        result = []
        for sphere in sphere_batches[0]:
            pose = getattr(sphere, "pose", None)
            radius = getattr(sphere, "radius", None)
            center = _finite_numbers(pose[:3] if isinstance(pose, Sequence) else pose, count=3, name="robot sphere pose")
            try:
                radius_m = float(radius)
            except (TypeError, ValueError, OverflowError) as error:
                raise _error(
                    "cuRobo returned an invalid robot sphere radius",
                    code="CUROBO_BACKEND_FAILED",
                    stage="sdk",
                    status=500,
                ) from error
            if not math.isfinite(radius_m) or radius_m <= 0.0:
                raise _error(
                    "cuRobo returned an invalid robot sphere radius",
                    code="CUROBO_BACKEND_FAILED",
                    stage="sdk",
                    status=500,
                )
            result.append((*center, radius_m))
        if not result:
            raise _error(
                "cuRobo returned no valid robot collision spheres",
                code="CUROBO_BACKEND_FAILED",
                stage="sdk",
                status=500,
            )
        return self._np.asarray(result, dtype=self._np.float64)

    def _decode_depth(self, encoded_depth: str):
        compressed = base64.b64decode(encoded_depth, validate=True)
        depth = self._cv2.imdecode(
            self._np.frombuffer(compressed, dtype=self._np.uint8),
            self._cv2.IMREAD_UNCHANGED,
        )
        if depth is None or depth.ndim != 2 or depth.dtype != self._np.uint16:
            raise _error(
                "depth image must be a uint16 PNG",
                code="CUROBO_INVALID_DEPTH_IMAGE",
            )
        return depth

    @staticmethod
    def _scaled_intrinsics(request: RobotSegmentationRequest, width: int, height: int):
        source_width, source_height = request.depth_calibration_resolution
        scale_x = width / source_width
        scale_y = height / source_height
        intrinsics = [list(row) for row in request.depth_intrinsics]
        intrinsics[0][0] *= scale_x
        intrinsics[0][2] *= scale_x
        intrinsics[1][1] *= scale_y
        intrinsics[1][2] *= scale_y
        return intrinsics

    def _pointcloud(
        self,
        depth,
        intrinsics,
        translation_mm,
        voxel_size_mm,
    ) -> bytes:
        np = self._np
        height, width = depth.shape
        u_coordinates, v_coordinates = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        depth_float = depth.astype(np.float32)
        valid = depth_float > 0.0
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]
        x_depth = (u_coordinates - cx) * depth_float / fx
        y_depth = (v_coordinates - cy) * depth_float / fy
        x_rgb = (x_depth + translation_mm[0])[valid]
        y_rgb = (y_depth + translation_mm[1])[valid]
        z_rgb = (depth_float + translation_mm[2])[valid]
        points = np.stack((x_rgb, y_rgb, z_rgb), axis=-1) / 1000.0
        color_index = np.clip(
            (z_rgb - 150.0) / (1200.0 - 150.0) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        colors = self._color_lut[color_index]
        if voxel_size_mm > 0.0 and len(points):
            voxel_keys = np.floor(points * (1000.0 / voxel_size_mm)).astype(np.int64)
            _, selected_indices = np.unique(voxel_keys, axis=0, return_index=True)
            points = points[selected_indices]
            colors = colors[selected_indices]
        return (
            struct.pack("<I", len(points))
            + points.astype("<f4").tobytes()
            + colors.astype(np.uint8).tobytes()
        )

    def _depth_tensor(self, depth_mm):
        depth_m = self._torch.as_tensor(
            depth_mm,
            device=self._device_cfg.device,
            dtype=self._torch.float32,
        )
        return self._torch.nan_to_num(depth_m, nan=0.0).mul(0.001).unsqueeze(0)

    def _depth_tensor_m(self, depth_m):
        tensor = self._torch.as_tensor(
            depth_m,
            device=self._device_cfg.device,
            dtype=self._torch.float32,
        )
        return self._torch.nan_to_num(
            tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).unsqueeze(0)

    def _filter_sensor_depth(self, depth_m):
        image_shape = tuple(int(value) for value in depth_m.shape[-2:])
        if self._depth_filter is None or self._depth_filter_shape != image_shape:
            self._depth_filter = self._filter_depth_type(
                image_shape=image_shape,
                depth_minimum_distance=0.05,
                depth_maximum_distance=3.0,
                flying_pixel_threshold=0.75,
                bilateral_kernel_size=5,
                bilateral_sigma_spatial=2.0,
                bilateral_sigma_depth=0.03,
                device=str(self._device_cfg.device),
                num_batch=1,
            )
            self._depth_filter_shape = image_shape
        filtered_depth, _valid_mask = self._depth_filter(depth_m)
        return filtered_depth

    def _projected_base_pointcloud(self, prepared: PreparedRobotDepth) -> bytes:
        structured = prepared.camera_observation.get_pointcloud(
            project_to_pose=True
        )
        points = structured.reshape(-1, 3)[
            prepared.filtered_depth_m.reshape(-1) > 0.0
        ]
        if hasattr(points, "detach"):
            points = points.detach().cpu().numpy()
        points = self._np.asarray(points, dtype=self._np.float32).reshape(-1, 3)
        if not self._np.isfinite(points).all():
            raise _error(
                "cuRobo projected non-finite base-frame points",
                code="CUROBO_POINTCLOUD_PROJECTION_FAILED",
                status=500,
            )
        return struct.pack("<I", len(points)) + points.astype("<f4").tobytes()

    def _encode_depth_png(self, depth_m) -> bytes:
        depth_mm = (
            depth_m.squeeze(0)
            .mul(1000.0)
            .round()
            .clamp_(0.0, 65535.0)
            .to(device="cpu")
            .numpy()
            .astype(self._np.uint16)
        )
        encoded, payload = self._cv2.imencode(".png", depth_mm)
        if not encoded:
            raise _error(
                "failed to encode robot-filtered depth image",
                code="CUROBO_DEPTH_ENCODING_FAILED",
                stage="encoding",
                status=500,
            )
        return payload.tobytes()

    def _mapped_camera_observation(
        self,
        *,
        depth_m,
        intrinsics_tensor,
        pose_tensor,
        height: int,
        width: int,
    ):
        zero_rgb = self._torch.zeros(
            (1, height, width, 3),
            device=self._device_cfg.device,
            dtype=self._torch.uint8,
        )
        return self._camera_observation_type(
            rgb_image=zero_rgb,
            depth_image=depth_m,
            intrinsics=intrinsics_tensor.unsqueeze(0),
            pose=self._pose_type.from_matrix(pose_tensor),
            depth_to_meter=1.0,
        )

    def prepare_observation(
        self, request: RobotSegmentationRequest
    ) -> PreparedRobotDepth:
        if not isinstance(request, RobotSegmentationRequest):
            raise TypeError("request must be a RobotSegmentationRequest")
        depth = self._decode_depth(request.depth_png_base64)
        height, width = depth.shape
        intrinsics = self._scaled_intrinsics(request, width, height)
        depth_tensor = self._filter_sensor_depth(self._depth_tensor(depth))
        intrinsics_tensor = self._device_cfg.to_device(intrinsics)
        pose_tensor = self._device_cfg.to_device(request.camera_pose_base_m)
        camera_observation = self._camera_observation_type(
            depth_image=depth_tensor,
            intrinsics=intrinsics_tensor,
            pose=self._pose_type.from_matrix(pose_tensor),
            depth_to_meter=1.0,
        )
        joint_positions_rad = [
            math.radians(value)
            for arm in ("A", "B")
            for value in request.joint_positions_deg[arm]
        ]
        joint_state = self._joint_state_type.from_position(
            self._device_cfg.to_device(joint_positions_rad),
            joint_names=list(self._joint_names),
        )
        mask, filtered_depth = self._segmenter.get_robot_mask(
            camera_observation,
            joint_state,
        )
        removed_point_count = int(mask.sum().item())
        mapped_observation = self._mapped_camera_observation(
            depth_m=filtered_depth,
            intrinsics_tensor=intrinsics_tensor,
            pose_tensor=pose_tensor,
            height=height,
            width=width,
        )
        return PreparedRobotDepth(
            camera_observation=mapped_observation,
            filtered_depth_m=filtered_depth,
            intrinsics=tuple(tuple(float(value) for value in row) for row in intrinsics),
            removed_point_count=removed_point_count,
        )

    def prepare_filtered_observation(
        self,
        request: RobotSegmentationRequest,
    ) -> PreparedRobotDepth:
        """Create a Mapper observation from already cuRobo/Open3D-cleaned depth."""

        if not isinstance(request, RobotSegmentationRequest):
            raise TypeError("request must be a RobotSegmentationRequest")
        depth = self._decode_depth(request.depth_png_base64)
        height, width = depth.shape
        intrinsics = self._scaled_intrinsics(request, width, height)
        intrinsics_tensor = self._device_cfg.to_device(intrinsics)
        pose_tensor = self._device_cfg.to_device(request.camera_pose_base_m)
        depth_tensor = self._depth_tensor(depth)
        return PreparedRobotDepth(
            camera_observation=self._mapped_camera_observation(
                depth_m=depth_tensor,
                intrinsics_tensor=intrinsics_tensor,
                pose_tensor=pose_tensor,
                height=height,
                width=width,
            ),
            filtered_depth_m=depth_tensor,
            intrinsics=tuple(
                tuple(float(value) for value in row) for row in intrinsics
            ),
            removed_point_count=0,
        )

    def prepare_ffs_observation(
        self,
        request: FfsDepthObservationRequest,
    ) -> PreparedFfsDepth:
        """Filter and robot-segment one original FFS left-depth image."""

        if not isinstance(request, FfsDepthObservationRequest):
            raise TypeError("request must be an FfsDepthObservationRequest")
        depth = self._np.asarray(request.depth_left_m, dtype=self._np.float32).copy()
        depth[self._np.asarray(request.valid_left_mask) == 0] = 0.0
        filtered_depth = self._filter_sensor_depth(self._depth_tensor_m(depth))
        intrinsics_tensor = self._device_cfg.to_device(request.left_intrinsics)
        pose_matrix_tensor = self._device_cfg.to_device(request.T_base_left_camera)
        source_observation = self._camera_observation_type(
            depth_image=filtered_depth,
            intrinsics=intrinsics_tensor,
            pose=self._pose_type.from_matrix(pose_matrix_tensor),
            depth_to_meter=1.0,
        )
        joint_positions_rad = [
            math.radians(value)
            for arm in ("A", "B")
            for value in request.joint_positions_deg[arm]
        ]
        joint_state = self._joint_state_type.from_position(
            self._device_cfg.to_device(joint_positions_rad),
            joint_names=list(self._joint_names),
        )
        robot_mask, robot_free_depth = self._segmenter.get_robot_mask(
            source_observation,
            joint_state,
        )
        mapper_observation = self._camera_observation_type(
            depth_image=robot_free_depth,
            intrinsics=intrinsics_tensor.unsqueeze(0),
            pose=self._pose_type.from_matrix(pose_matrix_tensor),
            depth_to_meter=1.0,
        )
        base_points = mapper_observation.get_pointcloud(project_to_pose=True)
        return PreparedFfsDepth(
            camera_id=request.camera_id,
            stereo_sequence=request.stereo_sequence,
            camera_observation=mapper_observation,
            filtered_depth_m=robot_free_depth,
            base_pointcloud_m=base_points,
            intrinsics_tensor=intrinsics_tensor,
            pose_matrix_tensor=pose_matrix_tensor,
            removed_point_count=int(robot_mask.sum().item()),
        )

    @staticmethod
    def _numpy_array(value, np):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _open3d_inlier_mask(self, points, *, open3d_runtime=None):
        np = self._np
        if open3d_runtime is None:
            import open3d as open3d_runtime
        cloud = open3d_runtime.geometry.PointCloud()
        cloud.points = open3d_runtime.utility.Vector3dVector(
            np.asarray(points, dtype=np.float64)
        )
        radius_cloud, radius_indices = cloud.remove_radius_outlier(
            nb_points=16,
            radius=0.025,
        )
        radius_indices = np.asarray(radius_indices, dtype=np.int64)
        if not len(radius_indices):
            raise _error(
                "Open3D radius filtering rejected every FFS point",
                code="CUROBO_OPEN3D_FILTER_EMPTY",
            )
        _, statistical_indices = radius_cloud.remove_statistical_outlier(
            nb_neighbors=32,
            std_ratio=0.75,
        )
        statistical_indices = np.asarray(statistical_indices, dtype=np.int64)
        retained = radius_indices[statistical_indices]
        if not len(retained):
            raise _error(
                "Open3D statistical filtering rejected every FFS point",
                code="CUROBO_OPEN3D_FILTER_EMPTY",
            )
        keep = np.zeros(len(points), dtype=bool)
        keep[retained] = True
        return keep

    def remove_semantic_instances(
        self,
        prepared: PreparedFfsDepth,
        projection: SemanticProjectionRequest,
        *,
        open3d_runtime=None,
    ) -> PreparedSemanticDepth:
        """Remove every named semantic surface from one left-depth observation."""

        if not isinstance(prepared, PreparedFfsDepth):
            raise TypeError("prepared must be a PreparedFfsDepth")
        if not isinstance(projection, SemanticProjectionRequest):
            raise TypeError("projection must be a SemanticProjectionRequest")
        np = self._np
        left_points = self._numpy_array(
            prepared.camera_observation.get_pointcloud(project_to_pose=False),
            np,
        ).reshape(*prepared.filtered_depth_m.shape[-2:], 3)
        base_points = self._numpy_array(prepared.base_pointcloud_m, np).reshape(
            *prepared.filtered_depth_m.shape[-2:], 3
        )
        filtered_depth = self._numpy_array(prepared.filtered_depth_m, np).reshape(
            prepared.filtered_depth_m.shape[-2:]
        ).astype(np.float32, copy=True)
        valid_depth = (
            np.isfinite(filtered_depth)
            & (filtered_depth > 0.0)
            & np.isfinite(left_points).all(axis=-1)
            & np.isfinite(base_points).all(axis=-1)
        )
        valid_indices = np.flatnonzero(valid_depth.reshape(-1))
        input_point_count = int(len(valid_indices))
        if input_point_count == 0:
            raise _error(
                "FFS observation has no robot-free depth points",
                code="CUROBO_FFS_DEPTH_EMPTY",
            )
        open3d_keep = self._open3d_inlier_mask(
            base_points.reshape(-1, 3)[valid_indices],
            open3d_runtime=open3d_runtime,
        )
        retained_indices = valid_indices[open3d_keep]
        retained = np.zeros(filtered_depth.size, dtype=bool)
        retained[retained_indices] = True
        retained = retained.reshape(filtered_depth.shape)

        transform = np.asarray(projection.left_to_rgb, dtype=np.float64)
        points_rgb = (
            left_points.astype(np.float64, copy=False) @ transform[:3, :3].T
            + transform[:3, 3]
        )
        z = points_rgb[..., 2]
        intrinsics = np.asarray(projection.rgb_intrinsics, dtype=np.float64)
        projectable = retained & np.isfinite(points_rgb).all(axis=-1) & (z > 0.0)
        u = np.zeros(filtered_depth.shape, dtype=np.int64)
        v = np.zeros(filtered_depth.shape, dtype=np.int64)
        u[projectable] = np.rint(
            intrinsics[0, 0] * points_rgb[..., 0][projectable] / z[projectable]
            + intrinsics[0, 2]
        ).astype(np.int64)
        v[projectable] = np.rint(
            intrinsics[1, 1] * points_rgb[..., 1][projectable] / z[projectable]
            + intrinsics[1, 2]
        ).astype(np.int64)
        width, height = projection.rgb_size
        inside = (
            projectable
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        union = np.zeros(filtered_depth.shape, dtype=bool)
        instance_points = {}
        instance_samples = {}
        for instance_id, rgb_mask in projection.instance_masks.items():
            left_mask = np.zeros(filtered_depth.shape, dtype=bool)
            left_mask[inside] = rgb_mask[v[inside], u[inside]] != 0
            left_mask &= retained
            union |= left_mask
            points = np.ascontiguousarray(base_points[left_mask], dtype=np.float32)
            points.setflags(write=False)
            instance_points[instance_id] = points
            indices = np.flatnonzero(left_mask)
            instance_samples[instance_id] = (indices, filtered_depth.reshape(-1)[indices].copy())
        background = retained & ~union
        filtered_depth[~background] = 0.0
        background_points = np.ascontiguousarray(
            base_points[background], dtype=np.float32
        )
        background_points.setflags(write=False)
        background_depth = self._depth_tensor_m(filtered_depth)
        intrinsics = prepared.intrinsics_tensor
        if len(intrinsics.shape) == 2:
            intrinsics = (
                intrinsics.unsqueeze(0)
                if hasattr(intrinsics, "unsqueeze")
                else np.expand_dims(intrinsics, 0)
            )
        mapper_observation = self._camera_observation_type(
            depth_image=background_depth,
            intrinsics=intrinsics,
            pose=self._pose_type.from_matrix(prepared.pose_matrix_tensor),
            depth_to_meter=1.0,
        )
        return PreparedSemanticDepth(
            camera_id=prepared.camera_id,
            stereo_sequence=prepared.stereo_sequence,
            camera_observation=mapper_observation,
            intrinsics_tensor=prepared.intrinsics_tensor,
            pose_matrix_tensor=prepared.pose_matrix_tensor,
            instance_points_base_m=MappingProxyType(instance_points),
            background_points_base_m=background_points,
            input_point_count=input_point_count,
            filtered_point_count=int(np.count_nonzero(retained)),
            background_point_count=int(np.count_nonzero(background)),
            instance_depth_samples=MappingProxyType(instance_samples),
            removed_point_count=prepared.removed_point_count,
        )

    def crop_semantic_observation(
        self, prepared: PreparedSemanticDepth, table_model: Mapping[str, Any],
    ) -> PreparedSemanticDepth:
        """Crop a private depth copy before TSDF integration, and its semantic surfaces."""

        np = self._np
        source = prepared.camera_observation
        depth = self._numpy_array(source.depth_image, np)
        shape = depth.shape[-2:]
        points = self._numpy_array(
            source.get_pointcloud(project_to_pose=True), np,
        ).reshape(*shape, 3)
        cropped_depth = depth.reshape(shape).copy()
        cropped_depth[~table_footprint_mask(points, table_model)] = 0.0

        def crop_points(values):
            cropped = np.ascontiguousarray(
                values[table_footprint_mask(values, table_model)], dtype=np.float32,
            )
            cropped.setflags(write=False)
            return cropped

        background = crop_points(prepared.background_points_base_m)
        instances = {
            key: crop_points(value)
            for key, value in prepared.instance_points_base_m.items()
        }
        samples = {}
        for key, (indices, depths) in (prepared.instance_depth_samples or {}).items():
            keep = table_footprint_mask(prepared.instance_points_base_m[key], table_model)
            samples[key] = (indices[keep], depths[keep])
        return replace(
            prepared,
            instance_depth_samples=MappingProxyType(samples),
            camera_observation=self._camera_observation_type(
                depth_image=self._depth_tensor_m(cropped_depth),
                intrinsics=source.intrinsics, pose=source.pose, depth_to_meter=1.0,
            ),
            background_points_base_m=background,
            instance_points_base_m=MappingProxyType(instances),
            background_point_count=len(background),
        )

    def filter_isolated_surfaces(self, observations):
        """Remove isolated background from frozen views, preserving named objects."""
        np = self._np
        groups = []
        for view in observations:
            groups.append(view.background_points_base_m)
            groups.extend(view.instance_points_base_m.values())
        if not groups:
            return tuple(observations)
        keep = surface_component_inlier_mask(np.concatenate(groups))
        if keep.all():
            return tuple(observations)
        result, offset = [], 0
        for view in observations:
            background_keep = keep[offset:offset + len(view.background_points_base_m)]
            offset += len(background_keep)
            source = view.camera_observation
            depth = self._numpy_array(source.depth_image, np).copy()
            pixels = np.flatnonzero(depth.reshape(-1) > 0)
            depth.reshape(-1)[pixels[~background_keep]] = 0
            background = np.ascontiguousarray(view.background_points_base_m[background_keep])
            background.setflags(write=False)
            # A small disconnected top can be a real object's only visible face.
            # Semantic surfaces support background connectivity but are not culled.
            offset += sum(len(points) for points in view.instance_points_base_m.values())
            result.append(replace(
                view, background_points_base_m=background, background_point_count=len(background),
                camera_observation=self._camera_observation_type(
                    depth_image=self._depth_tensor_m(depth.reshape(depth.shape[-2:])),
                    intrinsics=source.intrinsics, pose=source.pose, depth_to_meter=1.0,
                ),
            ))
        return tuple(result)

    def separate_support_observation(self, view, plane):
        """Replace classified background support rays with the bound table cuboid.

        Semantic instance depths and every unclassified background ray are kept.
        """
        from .support_contact import support_mask

        np = self._np
        classified = support_mask(view.background_points_base_m, plane)
        source = view.camera_observation
        depth = self._numpy_array(source.depth_image, np).copy()
        pixels = np.flatnonzero(depth.reshape(-1) > 0)
        if len(pixels) != len(classified):
            raise ValueError("support classification is not aligned with background depth pixels")
        depth.reshape(-1)[pixels[classified]] = 0
        background = np.ascontiguousarray(view.background_points_base_m[~classified])
        background.setflags(write=False)
        return replace(
            view, background_points_base_m=background, background_point_count=len(background),
            camera_observation=self._camera_observation_type(
                depth_image=self._depth_tensor_m(depth.reshape(depth.shape[-2:])),
                intrinsics=source.intrinsics, pose=source.pose, depth_to_meter=1.0,
            ),
        )

    def stack_instance_observations(self, observations, instance_id):
        """Rebuild original masked rays on GPU without changing camera caches."""
        views = []
        for item in observations:
            source = item.camera_observation
            depth = self._torch.zeros_like(source.depth_image)
            samples = (item.instance_depth_samples or {}).get(instance_id)
            if samples is not None:
                indices, values = samples
                device = self._device_cfg.device
                depth.reshape(-1)[self._torch.as_tensor(indices, device=device)] = self._torch.as_tensor(
                    values, device=device, dtype=depth.dtype,
                )
            views.append(replace(item, camera_observation=self._camera_observation_type(
                depth_image=depth, intrinsics=source.intrinsics,
                pose=source.pose, depth_to_meter=1.0,
            )))
        return self.stack_prepared_observations(views)

    def stack_prepared_observations(
        self,
        observations: Sequence[PreparedFfsDepth | PreparedSemanticDepth],
    ):
        """Stack the actual head-containing views for public cuRobo Mapper."""

        values = tuple(observations)
        if tuple(item.camera_id for item in values) not in (("3",), ("1", "3"), ("2", "3"), ("1", "2", "3")):
            raise ValueError("mapping requires an ordered camera subset containing head camera 3")
        shapes = {tuple(item.camera_observation.depth_image.shape[-2:]) for item in values}
        if len(shapes) != 1:
            raise ValueError("cuRobo multiview mapping requires equal left-depth shapes")
        height, width = next(iter(shapes))
        return self._camera_observation_type(
            depth_image=self._torch.cat(
                [item.camera_observation.depth_image for item in values], dim=0
            ),
            rgb_image=self._torch.zeros(
                (len(values), height, width, 3),
                device=self._device_cfg.device,
                dtype=self._torch.uint8,
            ),
            intrinsics=self._torch.stack(
                [
                    item.camera_observation.intrinsics.squeeze(0)
                    if len(item.camera_observation.intrinsics.shape) == 3
                    else item.camera_observation.intrinsics
                    for item in values
                ],
                dim=0,
            ),
            pose=self._pose_type.from_matrix(
                self._torch.stack(
                    [item.pose_matrix_tensor for item in values],
                    dim=0,
                )
            ),
            depth_to_meter=1.0,
        )

    def segment(
        self,
        request: RobotSegmentationRequest,
        *,
        include_base_pointcloud: bool = False,
    ):
        if not isinstance(include_base_pointcloud, bool):
            raise TypeError("include_base_pointcloud must be a bool")
        prepared = self.prepare_observation(request)
        filtered = (
            prepared.filtered_depth_m.squeeze(0)
            .to(device="cpu")
            .numpy()
            .astype(self._np.float32)
            * 1000.0
        )
        pointcloud = self._pointcloud(
            filtered,
            prepared.intrinsics,
            request.depth_to_rgb_translation_mm,
            request.voxel_size_mm,
        )
        point_count = struct.unpack_from("<I", pointcloud, 0)[0]
        base_pointcloud_xyz = (
            self._projected_base_pointcloud(prepared)
            if include_base_pointcloud
            else struct.pack("<I", 0)
        )
        return (
            pointcloud,
            self._encode_depth_png(prepared.filtered_depth_m),
            base_pointcloud_xyz,
            point_count,
            prepared.removed_point_count,
        )


def create_robot_segmenter_adapter(
    *,
    robot_config_path: Path,
    urdf_path: Path,
    asset_root_path: Path,
) -> RobotSegmenterAdapter:
    return RobotSegmenterAdapter(
        sdk_boundary=CuroboRobotSegmenterBoundary(
            robot_config_path=robot_config_path,
            urdf_path=urdf_path,
            asset_root_path=asset_root_path,
        )
    )


__all__ = [
    "FfsDepthObservationRequest",
    "PreparedFfsDepth",
    "PreparedSemanticDepth",
    "RobotSegmentationRequest",
    "RobotSegmentationResult",
    "RobotSegmenterAdapter",
    "SemanticProjectionRequest",
    "PreparedRobotDepth",
    "create_robot_segmenter_adapter",
]

"""Public cuRobo Mapper boundary for robot-filtered depth fusion."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import inspect
import struct
import time
from types import SimpleNamespace
from typing import Any

from .robot_segmenter import RobotSegmentationRequest
from .contracts import CuroboToolError
from .scene_config import SCENE_VOXEL_SIZE_M


DEFAULT_EXTENT_METERS_XYZ = (2.4, 2.4, 2.0)
DEFAULT_GRID_CENTER_METERS = (0.35, 0.0, 0.8)
# One resolution for scene capture, semantic objects and collision-cache capacity.
DEFAULT_TSDF_VOXEL_SIZE_M = SCENE_VOXEL_SIZE_M
DEFAULT_ESDF_VOXEL_SIZE_M = SCENE_VOXEL_SIZE_M
DEFAULT_TRUNCATION_DISTANCE_M = 0.06
DEFAULT_MAX_PREVIEW_POINTS = 60_000


def crop_tsdf_to_table(mapper, table_model):
    """Crop exported GPU TSDF cells, then let the official importer rebuild storage."""

    import torch

    storage = mapper.tsdf
    blocks = storage.export_blocks()
    coordinates = blocks["active_block_coords"]
    if len(coordinates) == 0:
        return
    size = storage.block_size
    device = coordinates.device
    # Public block payload uses centered XYZ keys and X-fast local cells;
    # storage grid_shape is ZYX. Include the half-cell when locating centers.
    shape = torch.tensor(list(reversed(storage.config.grid_shape)), device=device)
    offsets = ((shape + size - 1) // size) // 2
    index = torch.arange(size ** 3, device=device)
    local = torch.stack((index % size, index // size % size, index // (size * size)), dim=-1)
    cells = (coordinates[:, None, :] + offsets) * size + local
    centers = storage.data.origin + (cells + .5 - shape * .5) * storage.config.voxel_size
    center = torch.tensor(table_model["pose"][:3], device=device)
    half = torch.tensor(table_model["dims"], device=device) * .5
    keep = ((centers[..., :2] >= center[:2] - half[:2]).all(dim=-1)
            & (centers[..., :2] <= center[:2] + half[:2]).all(dim=-1)
            & (centers[..., 2] >= center[2] - half[2]))
    blocks["block_data"].masked_fill_(~keep[..., None], 0.)
    active = (blocks["block_data"][..., 1] > 0).any(dim=-1)
    blocks = {name: values[active] for name, values in blocks.items()}
    mapper.reset()
    mapper.integrator.import_blocks(blocks)


def ensure_warp_torch_namespace() -> None:
    """Expose Warp's current public Torch adapter to the locked cuRobo runtime."""

    try:
        import warp
    except ModuleNotFoundError:
        return
    if hasattr(warp, "torch"):
        return
    device_from_torch = getattr(warp, "device_from_torch", None)
    if not callable(device_from_torch):
        raise RuntimeError("Warp PyTorch device adapter is unavailable")
    warp.torch = SimpleNamespace(device_from_torch=device_from_torch)


def create_esdf_preview_colors(np: Any, point_count: int):
    """Use a fixed visible color instead of the mapper's dummy RGB frame."""

    return np.tile(
        np.asarray([[0, 220, 255]], dtype=np.uint8),
        (int(point_count), 1),
    )


@dataclass(frozen=True)
class EsdfIntegrationResult:
    pointcloud_data: bytes
    point_count: int
    removed_point_count: int
    revision: int
    integrated_frame_count: int
    scene_token: str | None
    ready: bool = True
    surface_voxel_size_m: float | None = None


@dataclass(frozen=True)
class EsdfStatusResult:
    ready: bool
    revision: int
    integrated_frame_count: int
    point_count: int
    scene_token: str | None


class VolumetricMapperAdapter:
    """Validate the process-local Mapper boundary without leaking SDK objects."""

    def __init__(self, *, sdk_boundary: Any) -> None:
        for operation in ("integrate",):
            if not callable(getattr(sdk_boundary, operation, None)):
                raise TypeError(f"sdk_boundary must provide {operation}()")
        self._sdk_boundary = sdk_boundary

    @property
    def sdk_boundary(self) -> Any:
        return self._sdk_boundary

    def integrate_esdf(
        self,
        request: RobotSegmentationRequest,
        *,
        finalize: bool,
    ) -> EsdfIntegrationResult:
        if not isinstance(request, RobotSegmentationRequest):
            raise TypeError("request must be a RobotSegmentationRequest")
        if not isinstance(finalize, bool):
            raise TypeError("finalize must be a bool")
        result = self._sdk_boundary.integrate(request, finalize=finalize)
        if not isinstance(result, EsdfIntegrationResult):
            raise TypeError("volumetric mapper returned an invalid integration result")
        data = bytes(result.pointcloud_data)
        if len(data) < 4:
            raise ValueError("ESDF preview pointcloud is truncated")
        encoded_count = struct.unpack_from("<I", data, 0)[0]
        if encoded_count != result.point_count or len(data) != 4 + encoded_count * 15:
            raise ValueError("ESDF preview pointcloud count does not match its payload")
        if result.ready:
            if result.revision < 1 or result.integrated_frame_count < 1:
                raise ValueError("ready ESDF integration result requires observations")
            if result.scene_token != f"esdf:{result.revision}":
                raise ValueError("ESDF scene token does not match its revision")
        elif (
            result.revision != 0
            or result.scene_token is not None
        ):
            raise ValueError("warming ESDF integration result must be unpublished")
        return EsdfIntegrationResult(
            pointcloud_data=data,
            point_count=result.point_count,
            removed_point_count=result.removed_point_count,
            revision=result.revision,
            integrated_frame_count=result.integrated_frame_count,
            scene_token=result.scene_token,
            ready=result.ready,
            surface_voxel_size_m=result.surface_voxel_size_m,
        )

class CuroboVolumetricMapperBoundary:
    """Fuse filtered observations through public cuRobo V2 APIs only."""

    def __init__(
        self,
        *,
        robot_segmenter_boundary: Any,
        extent_meters_xyz=DEFAULT_EXTENT_METERS_XYZ,
        grid_center_meters=DEFAULT_GRID_CENTER_METERS,
        voxel_size_m=DEFAULT_TSDF_VOXEL_SIZE_M,
        esdf_voxel_size_m=DEFAULT_ESDF_VOXEL_SIZE_M,
        truncation_distance_m=DEFAULT_TRUNCATION_DISTANCE_M,
        max_preview_points=DEFAULT_MAX_PREVIEW_POINTS,
    ) -> None:
        prepare = getattr(
            robot_segmenter_boundary,
            "prepare_filtered_observation",
            None,
        )
        if not callable(prepare):
            raise TypeError("robot_segmenter_boundary must prepare observations")
        from curobo.perception import Mapper, MapperCfg

        self._mapper_type = Mapper
        self._mapper_cfg_type = MapperCfg
        self._robot_segmenter_boundary = robot_segmenter_boundary
        self._extent_meters_xyz = tuple(float(value) for value in extent_meters_xyz)
        self._grid_center_meters = tuple(float(value) for value in grid_center_meters)
        self._voxel_size_m = float(voxel_size_m)
        self._esdf_voxel_size_m = float(esdf_voxel_size_m)
        self._truncation_distance_m = float(truncation_distance_m)
        self._max_preview_points = int(max_preview_points)
        self._mapper = None
        self._mapper_shape = None
        self._esdf_grid = None
        self._revision = 0
        self._integrated_frame_count = 0
        self._pointcloud_data = struct.pack("<I", 0)
        self._point_count = 0
        self._instance_grids = []
        self._instance_mapper_cache = OrderedDict()
        self.timings_ms = {}

    @property
    def scene_voxel_grids(self):
        if self._esdf_grid is None:
            return None
        return [self._esdf_grid, *self._instance_grids]

    @property
    def esdf_grid(self) -> Any:
        return self._esdf_grid

    def clear_instance_layers(self):
        self._instance_grids = []

    def reset_capture(self):
        """Reuse allocation only; never carry observations across captures."""
        if self._mapper is not None:
            self._mapper.reset()
        self._instance_grids = []
        self._esdf_grid = None
        self._integrated_frame_count = 0
        self._revision = 0
        self._point_count = 0
        self._pointcloud_data = struct.pack("<I", 0)

    def configure_mapping_bounds(self, bounds) -> None:
        center = tuple(float(value) for value in bounds.center_m)
        extent = tuple(float(value) for value in bounds.extent_m)
        if len(center) != 3 or len(extent) != 3:
            raise ValueError("mapping bounds must contain center and extent xyz")
        if center == self._grid_center_meters and extent == self._extent_meters_xyz:
            return
        self._grid_center_meters = center
        self._extent_meters_xyz = extent
        self._mapper = None
        self._mapper_shape = None
        self._esdf_grid = None

    def _create_mapper(self, image_shape: tuple[int, int], num_cameras: int = 1):
        device_cfg = self._robot_segmenter_boundary.device_cfg
        parameters = {
            "extent_meters_xyz": self._extent_meters_xyz,
            "extent_esdf_meters_xyz": self._extent_meters_xyz,
            "voxel_size": self._voxel_size_m,
            "esdf_voxel_size": self._esdf_voxel_size_m,
            "grid_center": device_cfg.to_device(self._grid_center_meters),
            "truncation_distance": self._truncation_distance_m,
            "minimum_tsdf_weight": 0.5,
            "depth_minimum_distance": 0.05,
            "depth_maximum_distance": 3.0,
            "decay_factor": 1.0,
            "frustum_decay_factor": 0.5,
            "enable_static": False,
            "num_cameras": int(num_cameras),
            "seeding_method": "scatter",
        }
        supported = inspect.signature(self._mapper_cfg_type).parameters
        if "image_height" in supported:
            parameters["image_height"] = image_shape[0]
        if "image_width" in supported:
            parameters["image_width"] = image_shape[1]
        return self._mapper_type(self._mapper_cfg_type(**parameters))

    def _surface_pointcloud(self, *, voxel_centers: bool = False) -> tuple[bytes, int]:
        np = self._robot_segmenter_boundary.numpy
        if voxel_centers:
            # Both map viewers need voxel centers, not triangle topology.
            # Keep collision ESDF unchanged and avoid first-use marching cubes.
            vertices = self._mapper.extract_occupied_voxels(
                surface_only=True,
            ).centers
        else:
            ensure_warp_torch_namespace()
            surface_mesh = self._mapper.extract_mesh(surface_only=True)
            vertices = surface_mesh.vertices
        if vertices is None:
            return struct.pack("<I", 0), 0
        if len(vertices) > self._max_preview_points:
            stride = max(
                1,
                (len(vertices) + self._max_preview_points - 1)
                // self._max_preview_points,
            )
            vertices = vertices[::stride]
        if hasattr(vertices, "detach"):
            vertices = vertices.detach().cpu().numpy()
        points = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        colors = create_esdf_preview_colors(np, len(points))
        return (
            struct.pack("<I", len(points))
            + points.astype("<f4").tobytes()
            + colors.astype(np.uint8).tobytes(),
            len(points),
        )

    def integrate(
        self,
        request: RobotSegmentationRequest,
        *,
        finalize: bool,
    ) -> EsdfIntegrationResult:
        if not isinstance(finalize, bool):
            raise TypeError("finalize must be a bool")
        prepared = self._robot_segmenter_boundary.prepare_filtered_observation(request)
        height, width = prepared.filtered_depth_m.shape[-2:]
        image_shape = (int(height), int(width))
        if self._mapper is None or self._mapper_shape != image_shape:
            self._mapper = self._create_mapper(image_shape)
            self._mapper_shape = image_shape
            self._revision = 0
            self._integrated_frame_count = 0
            self._esdf_grid = None
            self._pointcloud_data = struct.pack("<I", 0)
            self._point_count = 0
        # Use the public positional contract: cuRobo v0.8 names this argument
        # ``observation`` while newer documentation has used different labels.
        self._mapper.integrate(prepared.camera_observation)
        self._integrated_frame_count += 1
        if not finalize:
            return EsdfIntegrationResult(
                pointcloud_data=struct.pack("<I", 0),
                point_count=0,
                removed_point_count=prepared.removed_point_count,
                revision=0,
                integrated_frame_count=self._integrated_frame_count,
                scene_token=None,
                ready=False,
            )
        self._esdf_grid = self._mapper.compute_esdf()
        self._pointcloud_data, self._point_count = self._surface_pointcloud(voxel_centers=True)
        self._revision += 1
        return EsdfIntegrationResult(
            pointcloud_data=self._pointcloud_data,
            point_count=self._point_count,
            removed_point_count=prepared.removed_point_count,
            revision=self._revision,
            integrated_frame_count=self._integrated_frame_count,
            scene_token=f"esdf:{self._revision}",
            ready=True,
            surface_voxel_size_m=self._voxel_size_m,
        )

    def integrate_multiview(self, prepared_observations, *, table_model=None) -> EsdfIntegrationResult:
        """Fuse an ordered head-containing camera batch and compute ESDF once."""

        prepared = tuple(prepared_observations)
        if tuple(item.camera_id for item in prepared) not in (("3",), ("1", "3"), ("2", "3"), ("1", "2", "3")):
            raise ValueError("ESDF requires an ordered camera subset containing head camera 3")
        camera_count = len(prepared)
        stack = getattr(
            self._robot_segmenter_boundary,
            "stack_prepared_observations",
            None,
        )
        if not callable(stack):
            raise TypeError("robot segmenter boundary cannot stack camera observations")
        observation = stack(prepared)
        height, width = observation.depth_image.shape[-2:]
        image_shape = (int(height), int(width))
        started = time.perf_counter()
        if self._mapper is None or self._mapper_shape != (camera_count, *image_shape):
            self._mapper = self._create_mapper(image_shape, num_cameras=camera_count)
        else:
            self._mapper.reset()
        self.timings_ms = {"mapper_setup": (time.perf_counter() - started) * 1000}
        self._mapper_shape = (camera_count, *image_shape)
        self._esdf_grid = None
        self._pointcloud_data = struct.pack("<I", 0)
        self._point_count = 0
        started = time.perf_counter()
        self._mapper.integrate(observation)
        self._integrated_frame_count = camera_count
        if table_model is not None:
            crop_tsdf_to_table(self._mapper, table_model)
        self._esdf_grid = self._mapper.compute_esdf()
        self.timings_ms["background_fusion_esdf"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        self._pointcloud_data, self._point_count = self._surface_pointcloud(voxel_centers=True)
        self.timings_ms["background_preview"] = (time.perf_counter() - started) * 1000
        self._revision += 1
        return EsdfIntegrationResult(
            pointcloud_data=self._pointcloud_data,
            point_count=self._point_count,
            removed_point_count=sum(item.removed_point_count for item in prepared),
            revision=self._revision,
            integrated_frame_count=self._integrated_frame_count,
            scene_token=f"esdf:{self._revision}",
            ready=True,
            surface_voxel_size_m=self._voxel_size_m,
        )

    def build_instance_esdf(self, observations, instance_id, points_by_camera, *, padding_m, table_model=None):
        """Fuse one object's masked depth using the public GPU Mapper.

        TSDF bounds cover the measured object. ESDF bounds additionally cover
        robot sphere radii and the largest supported collision activation.
        Preserve Mapper's returned world pose and XYZ distance-buffer layout.
        """
        np = self._robot_segmenter_boundary.numpy
        from .mixed_esdf_scene import InstanceGeometry

        points = np.concatenate([np.asarray(value).reshape(-1, 3) for value in points_by_camera.values()])
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        tsdf_size = self._voxel_size_m
        esdf_size = self._esdf_voxel_size_m
        observation = self._robot_segmenter_boundary.stack_instance_observations(observations, instance_id)
        height, width = observation.depth_image.shape[-2:]
        # Reuse any allocation which contains the new measurements and full
        # collision padding. Exact min/max keys churn on millimetre depth noise.
        key = None
        for cached_key in self._instance_mapper_cache:
            cached_center, cached_extent, cached_padding, cached_height, cached_width = cached_key
            if (cached_height, cached_width) != (int(height), int(width)) or cached_padding < padding_m:
                continue
            half = np.asarray(cached_extent) * .5
            if np.all(minimum >= np.asarray(cached_center) - half) and np.all(maximum <= np.asarray(cached_center) + half):
                key = cached_key
                break
        if key is None:
            # Reserve 2 cm of allocation slack, without moving measured rays.
            low = np.floor(minimum / .02) * .02 - .02
            high = np.ceil(maximum / .02) * .02 + .02
            center = (low + high) * .5
            key = (tuple(center), tuple(high - low), float(padding_m), int(height), int(width))
            config = self._mapper_cfg_type(
                extent_meters_xyz=tuple(high - low + .06),
                extent_esdf_meters_xyz=tuple(high - low + 2 * padding_m),
                voxel_size=tsdf_size, esdf_voxel_size=esdf_size,
                grid_center=self._robot_segmenter_boundary.device_cfg.to_device(center),
                truncation_distance=.03, minimum_tsdf_weight=.5,
                depth_minimum_distance=.05, depth_maximum_distance=3.,
                decay_factor=1., frustum_decay_factor=1., enable_static=False,
                num_cameras=3, image_height=int(height), image_width=int(width),
                seeding_method="scatter",
            )
            mapper = self._mapper_type(config)
        else:
            mapper = self._instance_mapper_cache.pop(key)
            mapper.reset()
        self._instance_mapper_cache[key] = mapper
        # Retain a tabletop working set across captures; a two-entry cache
        # rebuilt all three mappers on every repeated three-object capture.
        while len(self._instance_mapper_cache) > 8:
            self._instance_mapper_cache.popitem(last=False)
        mapper.integrate(observation)
        if table_model is not None:
            crop_tsdf_to_table(mapper, table_model)
        grid = mapper.compute_esdf()
        grid.name = f"semantic_instance::{instance_id}"
        # Own the distance data after this local mapper is released.
        grid.feature_tensor = grid.feature_tensor.clone()
        centers = mapper.extract_occupied_voxels(surface_only=True).centers
        if len(centers) == 0:
            raise CuroboToolError(
                "The object's masked depth produced no integrated voxel surface.",
                stage="scene", code="mapping_instance_surface_empty", status=422,
                details={"instance_id": instance_id},
            )
        if len(centers) > 20000:
            centers = centers[::(len(centers) + 19999) // 20000]
        centers = centers.detach().cpu().numpy().astype(np.float32)
        self._instance_grids.append(grid)
        return InstanceGeometry(
            name=grid.name, instance_id=instance_id, kind="voxel",
            model={"pose": list(grid.pose), "dims": list(grid.dims),
                   "voxel_size": float(grid.voxel_size), "surface_voxel_size": tsdf_size,
                   "centers_base_m": centers.tolist()},
            input_point_count=len(points), filtered_point_count=len(centers),
        )


def create_volumetric_mapper_adapter(
    *, robot_segmenter_boundary: Any
) -> VolumetricMapperAdapter:
    return VolumetricMapperAdapter(
        sdk_boundary=CuroboVolumetricMapperBoundary(
            robot_segmenter_boundary=robot_segmenter_boundary
        )
    )


__all__ = [
    "CuroboVolumetricMapperBoundary",
    "EsdfIntegrationResult",
    "EsdfStatusResult",
    "VolumetricMapperAdapter",
    "create_volumetric_mapper_adapter",
]

"""Independent semantic maps composed through the pinned public cuRobo API."""
import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np


def surface_geometry_metadata(centers, voxel_size):
    """Reduce all world cells before creating the bounded display point sample."""
    if len(centers.shape) != 2 or centers.shape[1] != 3 or not len(centers):
        raise ValueError("instance_surface_requires_nonempty_xyz")
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("instance_surface_requires_positive_voxel_size")
    count = len(centers)
    if hasattr(centers, "detach"):
        # Only six reduction values and the bounded preview cross the GPU boundary.
        lower = centers.amin(dim=0).detach().cpu().numpy().astype(float)
        upper = centers.amax(dim=0).detach().cpu().numpy().astype(float)
        preview = centers[::max(1, (count + 19999) // 20000)].detach().cpu().numpy()
    else:
        lower, upper = np.min(centers, axis=0).astype(float), np.max(centers, axis=0).astype(float)
        preview = centers[::max(1, (count + 19999) // 20000)]
    if not np.isfinite([lower, upper]).all():
        raise ValueError("instance_surface_requires_finite_xyz")
    return {"surface_voxel_size": voxel_size,
            "centers_base_m": np.asarray(preview, dtype=np.float32).tolist(),
            "surface_cell_bounds_world_m": [(lower - voxel_size / 2).tolist(), (upper + voxel_size / 2).tolist()],
            "surface_cell_bounds_source": "all_native_surface_cells_before_preview_sampling",
            "surface_cell_count": count}


def local_mapping_inputs(observation, center, table_model):
    """Translate registered camera poses, never depth, into an instance frame."""
    from curobo.types import CameraObservation, Pose

    local = CameraObservation(depth_image=observation.depth_image, rgb_image=observation.rgb_image,
        intrinsics=observation.intrinsics, depth_to_meter=observation.depth_to_meter,
        pose=Pose(position=observation.pose.position - center, quaternion=observation.pose.quaternion))
    table = deepcopy(table_model)
    if table is not None:
        offset = center.detach().cpu().numpy() if hasattr(center, 'detach') else np.asarray(center)
        table['pose'][:3] = (np.asarray(table['pose'][:3]) - offset).tolist()
    return local, table


def instance_grid_layout(points, padding_m, *, voxel_size_m=.02, floor_z_m=None):
    # Preserve the measured dtype: promoting float32 before the native 2 cm
    # quantization can move a TSDF boundary and change its sampling phase.
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('instance_points_must_be_finite_xyz')
    if not np.isfinite(padding_m) or padding_m < 0:
        raise ValueError('instance_padding_must_be_nonnegative')
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    # Preserve the native TSDF allocation and phase; remove only ESDF slack.
    low = np.floor(minimum / .02) * .02 - .02
    high = np.ceil(maximum / .02) * .02 + .02
    center = (low + high) / 2
    if floor_z_m is not None:
        if not np.isfinite(floor_z_m):
            raise ValueError('instance_floor_height_must_be_finite')
        # Align the grid, not the observed geometry: a planar release surface
        # must not be rounded upward by half a voxel during ESDF seeding.
        index = np.round((center[2] - floor_z_m) / voxel_size_m - .5)
        center[2] = floor_z_m + (index + .5) * voxel_size_m
    half = np.maximum(np.abs(minimum - center), np.abs(maximum - center)) + padding_m
    legacy_extent = high - low + 2 * padding_m
    legacy_shape = np.array([math.ceil(value / voxel_size_m) for value in legacy_extent])
    shape = np.ceil(2 * half / voxel_size_m - 1e-8).astype(int)
    # Remove whole pairs of cells so voxel centers retain the native phase.
    shape += (legacy_shape - shape) % 2
    if floor_z_m is not None:
        shape[2] += shape[2] % 2
    return {'center': center.tolist(), 'tsdf_extent': (high - low + .06).tolist(),
            'esdf_extent': (shape * voxel_size_m - 1e-9).tolist(), 'esdf_shape': shape.tolist(),
            'legacy_esdf_extent': legacy_extent.tolist(),
            'esdf_cells': int(np.prod(shape)), 'legacy_esdf_cells': int(np.prod(legacy_shape))}


class InstanceMapper:
    """One mapper/stream per layer; bounded workers, atomic publication after join."""
    def __init__(self, segmenter, *, max_workers=4, compact_esdf=True, local_frame=True, voxel_size_m=.02):
        self.segmenter = segmenter
        self.max_workers = max_workers
        self.compact_esdf = compact_esdf
        self.local_frame = local_frame
        self.voxel_size_m = voxel_size_m
        self.cache = OrderedDict()
        self.grids = []
        self.metrics = {}

    def build(self, observations, instances, *, padding_m, table_model, floor_heights_m=None):
        import torch
        from curobo.perception import Mapper, MapperCfg
        from tianji_curobo.contracts import CuroboToolError
        from tianji_curobo.mixed_esdf_scene import InstanceGeometry

        started = time.perf_counter()
        self.grids, self.metrics = [], {}
        jobs, owned_inputs = [], []
        producer = torch.cuda.current_stream()
        build_error = None
        try:
            # Construct and reserve all mappers on one thread. No mapper can be
            # shared between two instances even when their bounding boxes coincide.
            for name, points_by_camera in instances.items():
                if not any(len(value) for value in points_by_camera.values()):
                    continue
                packed_at = time.perf_counter()
                points = np.concatenate([np.asarray(value).reshape(-1, 3) for value in points_by_camera.values()])
                floor = (floor_heights_m or {}).get(name)
                voxel_size = self.voxel_size_m
                layout = instance_grid_layout(points, padding_m, voxel_size_m=voxel_size, floor_z_m=floor)
                observation = self.segmenter.stack_instance_observations(observations, name)
                center = self.segmenter.device_cfg.to_device(layout['center'])
                local_table = table_model
                if self.local_frame:
                    observation, local_table = local_mapping_inputs(observation, center, table_model)
                owned_inputs.append(observation)
                height, width = observation.depth_image.shape[-2:]
                camera_count = int(observation.depth_image.shape[0])
                extent = layout['esdf_extent']
                if not self.compact_esdf:
                    extent = layout['legacy_esdf_extent']
                tsdf_extent = tuple(np.asarray(layout['tsdf_extent'], dtype=points.dtype))
                esdf_extent = tuple(extent) if self.compact_esdf else tuple(np.asarray(extent, dtype=points.dtype))
                # The SDK compiles grid origin into its kernels. A local origin
                # lets translated instances reuse kernels, with identical voxel
                # phase. Key by actual cell counts, not float extent roundoff.
                origin = (0., 0., 0.) if self.local_frame else tuple(layout['center'])
                key = (name, origin, voxel_size, tuple(math.ceil(v / voxel_size) for v in tsdf_extent),
                       tuple(math.ceil(v / voxel_size) for v in esdf_extent), camera_count, int(height), int(width))
                cached = self.cache.pop(key, None)
                packed_ms = (time.perf_counter() - packed_at) * 1000
                allocated_at = time.perf_counter()
                if cached is None:
                    # MapperCfg computes ceil(extent / voxel_size). Retain the
                    # input scalar dtype, as in the native adapter: converting
                    # float32 to Python float here can allocate an extra cell.
                    cfg = MapperCfg(extent_meters_xyz=tsdf_extent,
                        extent_esdf_meters_xyz=esdf_extent, voxel_size=voxel_size, esdf_voxel_size=voxel_size,
                        grid_center=self.segmenter.device_cfg.to_device(origin),
                        truncation_distance=.03, minimum_tsdf_weight=.5,
                        depth_minimum_distance=.05, depth_maximum_distance=3.,
                        decay_factor=1., frustum_decay_factor=1., enable_static=False,
                        num_cameras=camera_count, image_height=int(height), image_width=int(width), seeding_method='scatter')
                    mapper, stream = Mapper(cfg), torch.cuda.Stream()
                else:
                    mapper, stream = cached
                    mapper.reset()
                self.cache[key] = (mapper, stream)
                stream.wait_stream(producer)
                self.metrics[name] = {**layout, 'cache_hit': cached is not None,
                    'voxel_size_m': voxel_size, 'aligned_floor_z_m': floor,
                    'camera_count': camera_count,
                    'integration_frame': 'instance_local' if self.local_frame else 'world',
                    'packing_ms': packed_ms, 'allocation_reset_ms': (time.perf_counter() - allocated_at) * 1000,
                    'input_point_count': len(points)}
                jobs.append((name, mapper, stream, observation, local_table, center))
            with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(jobs)))) as executor:
                futures = [executor.submit(self.integrate, mapper, stream, observation, table)
                           for _, mapper, stream, observation, table, _ in jobs]
                # The executor waits for every worker even when one result fails.
                phase_times = [future.result() for future in futures]
            geometries, grids = {}, []
            for (name, mapper, _stream, _observation, _table, center), measured in zip(jobs, phase_times):
                exported_at = time.perf_counter()
                grid = mapper.integrator.get_voxel_grid()
                grid.name = 'semantic_instance::' + name
                grid.feature_tensor = grid.feature_tensor.clone()
                centers = mapper.extract_occupied_voxels(surface_only=True).centers
                if self.local_frame:
                    grid.pose = [(float(value) + offset) for value, offset in zip(grid.pose[:3], self.metrics[name]['center'])] + list(grid.pose[3:])
                    centers = centers + center
                if not len(centers):
                    raise CuroboToolError('Masked depth produced no integrated surface.', stage='scene',
                        code='mapping_instance_surface_empty', status=422, details={'instance_id': name})
                surface = surface_geometry_metadata(centers, grid.voxel_size)
                geometries[name] = InstanceGeometry(grid.name, name, 'voxel',
                    {'pose': list(grid.pose), 'dims': list(grid.dims), 'voxel_size': float(grid.voxel_size),
                     **surface}, self.metrics[name]['input_point_count'], len(surface['centers_base_m']))
                grids.append(grid)
                self.metrics[name].update(measured, export_preview_ms=(time.perf_counter() - exported_at) * 1000,
                    actual_esdf_cells=int(grid.feature_tensor.numel()))
        except BaseException as error:
            build_error = error
            raise
        finally:
            # Attempt every stream even if one reports an asynchronous error.
            # Keep inputs and mappers owned until all cleanup has been attempted.
            cleanup_errors = []
            for name, _, stream, _, _, _ in jobs:
                try:
                    stream.synchronize()
                except BaseException as error:  # noqa: BLE001 -- Report after attempting every owned stream.
                    cleanup_errors.append((name, error))
            # A worker may report a device error even if its final sync succeeds.
            # Never reuse potentially incomplete maps after any failed build.
            if build_error is not None or cleanup_errors:
                self.cache.clear()
            if cleanup_errors:
                primary_error = build_error if build_error is not None else cleanup_errors[0][1]
                for name, error in cleanup_errors:
                    primary_error.add_note(f'Instance {name} stream cleanup failed: {type(error).__name__}: {error}')
                if build_error is None:
                    raise primary_error
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.grids = grids
        self.wall_ms = (time.perf_counter() - started) * 1000
        return geometries

    @staticmethod
    def integrate(mapper, stream, observation, table_model):
        import torch
        from tianji_curobo.volumetric_mapper import crop_tsdf_to_table

        times = {}
        with torch.cuda.stream(stream):
            marks = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            marks[0].record(stream)
            mapper.integrate(observation)
            marks[1].record(stream)
            if table_model is not None:
                crop_tsdf_to_table(mapper, table_model)
            marks[2].record(stream)
            # Public advanced API avoids the per-map CPU metadata conversion
            # inside Mapper.compute_esdf; materialize metadata only after join.
            mapper.integrator.compute_esdf()
            marks[3].record(stream)
            stream.synchronize()
            for index, name in enumerate(('tsdf_gpu_ms', 'table_crop_gpu_ms', 'esdf_gpu_ms')):
                times[name] = marks[index].elapsed_time(marks[index + 1])
        return times

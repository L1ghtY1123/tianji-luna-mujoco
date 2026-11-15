"""Rebuild anonymous workspace TSDF from each registered RGBD observation."""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from .instance_mapping import surface_geometry_metadata
from .workspace import WorkspaceAtlas

BACKGROUND_LAYER = 'workspace_environment'


class WorkspaceMapper:
    """Rebuild all workspace cells; reuse allocations without retaining old measurements."""

    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.atlas = WorkspaceAtlas()
        self.mapper = None
        self.workspace_grids = None
        self.cad_surfaces = {}
        self.diagnostics = {}

    def reset(self):
        self.mapper = None

    def _workspace(self, points):
        import torch

        if self.workspace_grids is None:
            self.workspace_grids = []
            for arm in ('A', 'B'):
                _, grid, origin, resolution = self.atlas._load(arm)
                self.workspace_grids.append((torch.as_tensor(grid, device=points.device),
                                             points.new_tensor(origin), resolution))
        valid = torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)
        for grid, origin, resolution in self.workspace_grids:
            indices = torch.floor((points - origin) / resolution + .5).long()
            bounded = ((indices >= 0) & (indices < indices.new_tensor(grid.shape))).all(dim=-1)
            cells = indices[bounded]
            valid[bounded] |= grid[cells[:, 0], cells[:, 1], cells[:, 2]]
        return valid

    def _centers(self, blocks):
        import torch

        storage = self.mapper.tsdf
        coordinates = blocks['active_block_coords']
        size = storage.block_size
        shape = coordinates.new_tensor(list(reversed(storage.config.grid_shape)))
        offsets = ((shape + size - 1) // size) // 2
        index = torch.arange(size ** 3, device=coordinates.device)
        local = torch.stack((index % size, index // size % size, index // (size * size)), dim=-1)
        cells = (coordinates[:, None, :] + offsets) * size + local
        return storage.data.origin + (cells + .5 - shape * .5) * storage.config.voxel_size

    def _anonymous_depth(self, view, decoded):
        """Keep container contents when a container silhouette covers them."""
        import open3d as o3d
        import torch
        from curobo.types import CameraObservation

        from .cad_perception import CadCatalog

        source = view.camera_observation
        depth = source.depth_image.new_tensor(self.segmenter.filtered_depths[view.camera_id]).reshape(source.depth_image.shape)
        observation = CameraObservation(depth_image=depth, intrinsics=source.intrinsics,
                                        pose=source.pose, depth_to_meter=1.)
        points = observation.get_pointcloud(project_to_pose=True).reshape(-1, 3).detach().cpu().numpy()
        valid = (depth.reshape(-1) > 0).detach().cpu().numpy()
        excluded = np.zeros(len(points), dtype=bool)
        for row in decoded['geometry_objects']:
            if row.get('cad') is None:
                samples = (view.instance_depth_samples or {}).get(row['id'])
                if samples is not None:
                    excluded[samples[0]] = True
                continue
            cad = row['cad']
            key = cad['mesh_sha256']
            if key not in self.cad_surfaces:
                geometry = CadCatalog.from_file().geometry(cad['mesh_key'])
                scene = o3d.t.geometry.RaycastingScene(nthreads=1)
                scene.add_triangles(o3d.core.Tensor(geometry['vertices_m'].astype(np.float32)),
                                    o3d.core.Tensor(geometry['faces'].astype(np.uint32)))
                vertices = np.asarray(geometry['vertices_m'])
                self.cad_surfaces[key] = (scene, vertices.min(0), vertices.max(0))
            scene, lower, upper = self.cad_surfaces[key]
            pose = np.asarray(cad['T_world_object'])
            local = (points - pose[:3, 3]) @ pose[:3, :3]
            nearby = valid & np.all((local >= lower - .005) & (local <= upper + .005), axis=1)
            indices = np.flatnonzero(nearby)
            if len(indices):
                distances = scene.compute_distance(o3d.core.Tensor(local[indices].astype(np.float32)), nthreads=1).numpy()
                excluded[indices[distances <= .005]] = True
        depth.reshape(-1)[torch.as_tensor(excluded, device=depth.device)] = 0
        return CameraObservation(depth_image=depth, intrinsics=source.intrinsics, pose=source.pose, depth_to_meter=1.)

    def build(self, prepared, decoded, model):
        from curobo.perception import Mapper, MapperCfg
        from curobo.types import CameraObservation

        views = []
        for view in prepared:
            source = self._anonymous_depth(view, decoded)
            depth = source.depth_image.clone()
            points = source.get_pointcloud(project_to_pose=True).reshape(*depth.shape[-2:], 3)
            keep = self._workspace(points)
            for table in model['cuboid'].values():
                center = points.new_tensor(table['pose'][:3])
                half = points.new_tensor(table['dims']) / 2
                keep &= ~((points >= center - half - .005) & (points <= center + half + .005)).all(dim=-1)
            depth.reshape(keep.shape)[~keep] = 0
            views.append(replace(view, camera_observation=CameraObservation(depth_image=depth,
                intrinsics=source.intrinsics, pose=source.pose, depth_to_meter=1.)))
        observation = self.segmenter.stack_prepared_observations(views)
        if self.mapper is None:
            # Anonymous geometry uses 2 cm TSDF cells and the existing 5 cm ESDF resolution.
            self.mapper = Mapper(MapperCfg(extent_meters_xyz=(2., 2.4, 2.2), voxel_size=.02,
                esdf_voxel_size=.05, grid_center=observation.depth_image.new_tensor([0., 0., 1.1]),
                num_cameras=len(views), image_height=observation.depth_image.shape[-2],
                image_width=observation.depth_image.shape[-1]))
        self.mapper.reset()
        self.mapper.integrate(camera_observation=observation)
        incoming = {key: value.clone() for key, value in self.mapper.tsdf.export_blocks().items()}
        centers = self._centers(incoming)
        keep = self._workspace(centers)
        incoming['block_data'].masked_fill_(~keep[..., None], 0.)
        active = (incoming['block_data'][..., 1] > 0).any(dim=-1)
        incoming = {key: values[active] for key, values in incoming.items()}
        self.mapper.reset()
        self.mapper.integrator.import_blocks(incoming)
        grid = self.mapper.compute_esdf()
        grid.name = BACKGROUND_LAYER
        grid.feature_tensor = grid.feature_tensor.clone()
        centers = self.mapper.extract_occupied_voxels(surface_only=True).centers
        metadata = {'pose': list(grid.pose), 'dims': list(grid.dims), 'voxel_size': float(grid.voxel_size),
            'geometry_source': 'workspace_tsdf_esdf', 'observation_id': decoded['observation_set_id'],
            'generation': decoded['selection_generation'], 'identity_scope': 'anonymous',
            **(surface_geometry_metadata(centers, .02) if len(centers) else {})}
        self.diagnostics = {'scope': 'global', 'reused': False,
            'surface_cell_count': len(centers), 'tsdf_voxel_size_m': .02,
            'observation_id': decoded['observation_set_id'], 'historical_cells_retained': False,
            'occluded_space': 'unobserved', 'outside_workspace': 'discard',
            'esdf_update': 'recompute_from_current_workspace_tsdf'}
        return grid, deepcopy(metadata)

"""Read-only display snapshots of actual Mapper products and inferred support.

No simulator or SDK imports: the GPU composition boundary passes public grids.
Distance samples retain native XYZ flat indices; they are not visibility claims.
"""
import struct
from copy import deepcopy

import numpy as np


def _bin_shell_metadata(name, identity):
    for identifier, completion in identity.get('mapping_diagnostics', {}).get('bin_shell_completions', {}).items():
        prefix = f'semantic_instance::{identifier}::bin_shell::'
        part = name[len(prefix):]
        if name.startswith(prefix) and part in {**completion.get('cuboids', {}), **completion.get('meshes', {})}:
            if (completion.get('instance_id') != identifier
                    or any(completion.get(key) != identity[key] for key in ('observation_id', 'generation'))):
                raise ValueError('demo_bin_shell_published_identity_mismatch')
            return {'geometry_source': completion['source'], 'instance_id': identifier,
                'observation_id': completion['observation_id'], 'generation': completion['generation'],
                'synthetic_geometry_is_measured': False, 'face': part,
                **({'asset_id': completion['asset_id'], 'cad_sha256': completion['cad_sha256'],
                    'part_sha256': completion['part_sha256'][part]} if 'meshes' in completion else {})}
    if '::bin_shell::' in name:
        raise ValueError('demo_bin_shell_published_provenance_missing')
    return {}


def collision_layer_bounds(grids, model, identity):
    """World bounds of published cells/cuboids, without sampling or changing them."""
    from scipy.spatial.transform import Rotation

    disabled = set(identity.get("disabled_obstacle_names", ()))
    layers = {}
    for grid in grids:
        metadata = model.get("voxel", {}).get(grid.name, {})
        bounds = np.asarray(metadata.get("surface_cell_bounds_world_m", []), dtype=float)
        voxel_size = float(metadata.get("surface_voxel_size", getattr(grid, "voxel_size", np.nan)))
        if (bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0])
                or not np.isfinite(voxel_size) or voxel_size <= 0
                or metadata.get("surface_cell_bounds_source") != "all_native_surface_cells_before_preview_sampling"
                or not isinstance(metadata.get("surface_cell_count"), int) or metadata["surface_cell_count"] <= 0):
            layers[grid.name] = {"kind": "voxel", "enabled": grid.name not in disabled,
                                 "source": "native_surface_voxel_cell_bounds", "status": "unavailable"}
            continue
        layers[grid.name] = {
            "kind": "voxel", "enabled": grid.name not in disabled,
            "source": "native_surface_voxel_cell_bounds", "voxel_size_m": voxel_size,
            "cell_count": metadata["surface_cell_count"],
            "bounds_source": metadata["surface_cell_bounds_source"],
            "minimum_world_m": bounds[0].tolist(), "maximum_world_m": bounds[1].tolist(),
        }
    for name, cuboid in model.get("cuboid", {}).items():
        pose, dims = np.asarray(cuboid["pose"], dtype=float), np.asarray(cuboid["dims"], dtype=float)
        if (pose.shape != (7,) or dims.shape != (3,) or not np.isfinite(pose).all()
                or not np.isfinite(dims).all() or np.any(dims <= 0)):
            raise ValueError("invalid_published_cuboid_bounds")
        half = np.abs(Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()) @ dims / 2
        layers[name] = {"kind": "cuboid", "enabled": name not in disabled,
                        "source": "published_cuboid_bounds",
                        "minimum_world_m": (pose[:3] - half).tolist(),
                        "maximum_world_m": (pose[:3] + half).tolist(),
                        **_bin_shell_metadata(name, identity)}
    for name, mesh in model.get('mesh', {}).items():
        vertices, pose = np.asarray(mesh.get('vertices'), float), np.asarray(mesh.get('pose'), float)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or pose.shape != (7,) or not np.isfinite(vertices).all() or not np.isfinite(pose).all()):
            raise ValueError('invalid_published_mesh_bounds')
        world = vertices @ Rotation.from_quat(pose[3:], scalar_first=True).as_matrix().T+pose[:3]
        layers[name] = {'kind': 'mesh', 'enabled': name not in disabled, 'source': 'published_mesh_bounds',
            'minimum_world_m': world.min(0).tolist(), 'maximum_world_m': world.max(0).tolist(),
            **_bin_shell_metadata(name, identity)}
    return {**{key: identity[key] for key in ("scene_token", "observation_id", "generation")}, "layers": layers}


def _sample_indices(count, maximum):
    return np.linspace(0, count - 1, min(count, maximum), dtype=int) if count else np.empty(0, dtype=int)


def collision_scene_preview(candidate, grids, model, identity, *, maximum_surface_points=15000):
    """Snapshot exactly the map products used by update_voxel_scene, without rebuilding."""
    integration = candidate.integration
    data = integration.pointcloud_data
    count = struct.unpack_from('<I', data)[0]
    background = np.frombuffer(data, dtype='<f4', count=count * 3, offset=4).reshape(-1, 3)
    layers = []
    disabled = set(identity.get('disabled_obstacle_names', ()))
    for grid in grids:
        metadata = model.get('voxel', {}).get(grid.name, {})
        # Semantic-only maps omit the background grid, so the first grid can
        # already be an instance. Its measured surface is keyed by grid name.
        is_instance = grid.name.startswith('semantic_instance::')
        centers = np.asarray(metadata.get('centers_base_m', []), dtype=float).reshape(-1, 3) if is_instance else background
        selected = _sample_indices(len(centers), maximum_surface_points)
        values = grid.feature_tensor
        shape = list(values.shape)
        flat = values.reshape(-1)
        distance_indices = _sample_indices(len(flat), 512)
        # Transfer only bounded native field samples, never allocate a second grid.
        distances = flat[distance_indices]
        if not isinstance(distances, np.ndarray):
            distances = distances.detach().cpu().numpy()
        distances = np.asarray(distances, dtype=float)
        if not np.isfinite(centers).all() or not np.isfinite(distances).all():
            raise ValueError('nonfinite_collision_preview')
        shape_xyz = shape if len(shape) == 3 else [round(value / grid.voxel_size) for value in grid.dims]
        layers.append({
            'name': grid.name, 'enabled': grid.name not in disabled,
            'role': 'semantic_instance' if is_instance else 'background',
            'provenance': 'transformed_measured_voxels' if metadata.get('position_source') == 'action_prediction' else 'mapper_tsdf_surface_voxels',
            'position_source': metadata.get('position_source', 'rgbd_measured'),
            'pose_wxyz': list(grid.pose), 'dims_m': list(grid.dims), 'voxel_size_m': float(grid.voxel_size),
            'surface_voxel_size_m': float(metadata.get('surface_voxel_size', integration.surface_voxel_size_m or grid.voxel_size)),
            'surface_centers_world_m': centers[selected].tolist(),
            'mapper_surface_preview_count': len(centers), 'displayed_surface_count': len(selected),
            'surface_sampling': 'bounded_samples_of_mapper_surface_preview',
            'esdf': {'source': 'published_grid_feature_tensor', 'shape_xyz': shape_xyz,
                     'cell_count': int(np.prod(shape)), 'layout': 'X_slowest_Z_fastest',
                     'sample_flat_indices': distance_indices.tolist(), 'sample_distances_m': distances.tolist(),
                     'sign_convention': 'negative_inside_positive_outside',
                     'visibility_certificate': False},
        })
    table = candidate.table
    return deepcopy({
        'status': 'available', 'frame': 'world', 'units': 'm',
        **{key: identity[key] for key in ('scene_token', 'observation_id', 'generation')},
        'voxel_layers': layers,
        'cuboid_layers': [{'name': name, 'enabled': name not in disabled,
            'role': 'bin_shell_completion', 'provenance': metadata['geometry_source'],
            'model': dict(value), **metadata}
            for name, value in model.get('cuboid', {}).items()
            if (metadata := _bin_shell_metadata(name, identity))],
        'table': {'name': 'semantic_table_forbidden', 'enabled': True,
                  'provenance': 'inferred_ransac_support_completion', 'model': dict(table.model),
                  'height_m': float(table.height_m), 'normal_xyz': list(table.normal_xyz),
                  'ransac_inlier_count': int(table.inlier_count),
                  'support_plane': getattr(table, 'support_plane', None),
                  'completion': {'xy_expansion_m': .05,
                                 'below_plane_m': float(table.height_m - (table.model['pose'][2] - table.model['dims'][2] / 2)),
                                 'above_plane_m': float(table.model['pose'][2] + table.model['dims'][2] / 2 - table.height_m),
                                 'robot_facing_crop': True}},
        'mapping_diagnostics': identity.get('mapping_diagnostics', {}),
        'mesh_layers': {name: dict(value) for name, value in model.get('mesh', {}).items()},
        'camera_metrics': {key: dict(value) for key, value in candidate.camera_metrics.items()},
        'coverage': {'full_environment_coverage': False, 'unknown_space_motion_gate': False,
                     'collision_scope': ('sam3_instances_and_measured_table' if identity.get('semantic_only_active')
                                         else 'measured_table_footprint_obstacles_and_inferred_support'),
                     'unknown_policy': 'unobserved_regions_not_independently_rejected',
                     'surface_policy': ('measured_tsdf_esdf_with_separately_labeled_bin_shell_geometry'
                        if identity.get('mapping_diagnostics', {}).get('bin_shell_completions') else
                        'visible_returns_with_tsdf_esdf_inference_no_hidden_object_completion')},
    })

"""Read-only display snapshots of actual Mapper products and inferred support.

No simulator or SDK imports: the GPU composition boundary passes public grids.
Distance samples retain native XYZ flat indices; they are not visibility claims.
"""
import struct
from copy import deepcopy
from pathlib import Path

import numpy as np


def _cad_mesh(geometry):
    """Verify the published path/hash against the project catalog before exporting."""
    from scipy.spatial.transform import Rotation

    from .cad_perception import CadCatalog

    catalog = CadCatalog.from_file()
    record = catalog.record(geometry['asset_id'])
    mesh = catalog.geometry(record['mesh_key'])
    if (geometry.get('geometry_source') != 'registered_cad'
            or geometry.get('source_sha256') != mesh['source_sha256']
            or geometry.get('mesh_sha256') != mesh['mesh_sha256']
            or Path(geometry['file_path']).resolve() != Path(mesh['path']).resolve()
            or any(key in geometry for key in ('vertices', 'faces', 'scale', 'file_string', 'urdf_path'))):
        raise ValueError('published_cad_mesh_binding_changed')
    pose = np.asarray(geometry['pose'], dtype=float)
    if (pose.shape != (7,) or not np.isfinite(pose).all()
            or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6, rtol=0)):
        raise ValueError('invalid_published_cad_pose')
    rotation = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
    vertices = np.asarray(mesh['vertices_m'], dtype=float)
    world = vertices @ rotation.T + pose[:3]
    return mesh, world.min(0), world.max(0)


def _geometry_metadata(geometry):
    return {key: deepcopy(geometry[key]) for key in ('geometry_source', 'instance_id', 'asset_id',
            'source_sha256', 'mesh_sha256', 'observation_id', 'generation', 'geometry_observation_ids')
            if key in geometry}


def collision_layer_bounds(grids, model, identity):
    """World bounds of published cells, cuboids and verified complete CAD meshes."""
    from scipy.spatial.transform import Rotation

    from .inventory_map import require_native_inventory_geometry

    require_native_inventory_geometry(model)
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
                        **_geometry_metadata(cuboid),
                        "minimum_world_m": (pose[:3] - half).tolist(),
                        "maximum_world_m": (pose[:3] + half).tolist()}
    for name, geometry in model.get('mesh', {}).items():
        _, lower, upper = _cad_mesh(geometry)
        layers[name] = {'kind': 'mesh', 'enabled': name not in disabled,
                        'source': 'published_cad_mesh_bounds', **_geometry_metadata(geometry),
                        'minimum_world_m': lower.tolist(), 'maximum_world_m': upper.tolist()}
    return {**{key: identity[key] for key in ("scene_token", "observation_id", "generation")}, "layers": layers}


def _sample_indices(count, maximum):
    return np.linspace(0, count - 1, min(count, maximum), dtype=int) if count else np.empty(0, dtype=int)


def collision_scene_preview(candidate, grids, model, identity, *, maximum_surface_points=15000):
    """Snapshot exactly the map products used by update_voxel_scene, without rebuilding."""
    from .inventory_map import require_native_inventory_geometry

    require_native_inventory_geometry(model)
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
    mesh_layers = {}
    for name, geometry in model.get('mesh', {}).items():
        mesh, lower, upper = _cad_mesh(geometry)
        # STL repeats vertices per triangle. Exact deduplication changes neither
        # coordinates nor triangles and avoids sending those repeated values.
        vertices, indices = np.unique(mesh['vertices_m'], axis=0, return_inverse=True)
        faces = indices[np.asarray(mesh['faces'], dtype=int)]
        mesh_layers[name] = {'name': name, 'enabled': name not in disabled, 'role': 'semantic_instance',
            **_geometry_metadata(geometry), 'pose_wxyz': list(geometry['pose']),
            'vertices_local_m': vertices.tolist(), 'faces': faces.tolist(), 'sampling': 'none',
            'vertex_count': len(vertices), 'face_count': len(faces),
            'minimum_world_m': lower.tolist(), 'maximum_world_m': upper.tolist()}
    hybrid = bool(mesh_layers) or any(item.get('geometry_source') == 'observed_obb'
                                    for item in model.get('cuboid', {}).values())
    table = candidate.table
    # Mesh arrays were converted to fresh lists above. Copying the whole result
    # again traverses hundreds of thousands of triangle indices on every build.
    # Only the small metadata borrowed from live map state needs a deep copy.
    return {
        'status': 'available', 'frame': 'world', 'units': 'm',
        **{key: identity[key] for key in ('scene_token', 'observation_id', 'generation')},
        'voxel_layers': layers,
        'cuboid_layers': [{'name': name, 'enabled': name not in disabled,
                           'role': 'semantic_instance' if value.get('geometry_source') == 'observed_obb'
                                   else 'measured_table', **_geometry_metadata(value), 'model': deepcopy(value)}
                          for name, value in model.get('cuboid', {}).items()],
        'table': {'name': 'semantic_table_forbidden', 'enabled': True,
                  'provenance': 'inferred_ransac_support_completion', 'model': deepcopy(table.model),
                  'height_m': float(table.height_m), 'normal_xyz': list(table.normal_xyz),
                  'ransac_inlier_count': int(table.inlier_count),
                  'support_plane': deepcopy(getattr(table, 'support_plane', None)),
                  'completion': {'xy_expansion_m': .05,
                                 'below_plane_m': float(table.height_m - (table.model['pose'][2] - table.model['dims'][2] / 2)),
                                 'above_plane_m': float(table.model['pose'][2] + table.model['dims'][2] / 2 - table.height_m),
                                 'robot_facing_crop': True}},
        'mapping_diagnostics': deepcopy(identity.get('mapping_diagnostics', {})),
        'mesh_layers': mesh_layers,
        'camera_metrics': deepcopy(dict(candidate.camera_metrics)),
        'coverage': {**deepcopy(identity.get('coverage', {})), 'full_environment_coverage': False,
                     'sensor_coverage_role': identity.get('sensor_coverage_role', 'diagnostic_for_model_mapping_strategy'),
                     'unknown_space_motion_gate': bool(identity.get('observed_free_motion_gate')),
                     'collision_scope': ('registered_inventory_geometry_and_measured_table' if hybrid else
                                         'sam3_instances_and_measured_table' if identity.get('semantic_only_active')
                                         else 'measured_table_footprint_obstacles_and_inferred_support'),
                     'unknown_policy': ('swept_unknown_regions_rejected' if identity.get('observed_free_motion_gate')
                                        else 'motion_coverage_not_certified'),
                     'surface_policy': ('registered_cad_meshes_and_observed_obb_solid_envelopes' if hybrid else
                                        'visible_returns_with_tsdf_esdf_inference_no_hidden_object_completion')},
    }

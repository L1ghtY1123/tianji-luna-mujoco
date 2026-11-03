"""Current published container bounds and placement support geometry."""
import copy
import re

import numpy as np

from ..runtime.diagnostics import DiagnosticError


def _valid_model_container_bounds(layer, scene, identifier, name):
    if (layer.get('kind'), layer.get('source'), layer.get('geometry_source')) not in {
            ('mesh', 'published_cad_mesh_bounds', 'registered_cad'),
            ('cuboid', 'published_cuboid_bounds', 'observed_obb')}:
        return False
    record = scene.get('inventory_map', {}).get('entities', {}).get(identifier, {})
    if (layer.get('instance_id') != identifier or type(layer.get('generation')) is not int
            or layer['generation'] != scene['generation']
            or not isinstance(layer.get('observation_id'), str) or not layer['observation_id']
            or layer.get('observation_id') != record.get('last_observation_id', scene['observation_id'])
            or record.get('status', 'observed') not in {'observed', 'retained'}
            or record.get('last_generation', scene['generation']) != scene['generation']
            or name not in record.get('layer_names', [name])):
        return False
    if layer['geometry_source'] == 'registered_cad':
        return (isinstance(layer.get('asset_id'), str) and bool(layer['asset_id'])
            and all(isinstance(layer.get(key), str) and re.fullmatch('[0-9a-f]{64}', layer[key])
                    for key in ('source_sha256', 'mesh_sha256')))
    return True


def container_occupied_bounds(destination, scene, observation):
    """Use the exact enabled bin's published geometry bounds, never raw mask outliers."""
    header = scene.get('collision_layer_bounds')
    if (not isinstance(header, dict) or not isinstance(header.get('layers'), dict)
            or not isinstance(scene.get('scene_token'), str) or not scene['scene_token']
            or not isinstance(scene.get('observation_id'), str) or not scene['observation_id']
            or type(scene.get('generation')) is not int or scene['generation'] < 0
            or any(header.get(key) != scene.get(key)
                   for key in ('scene_token', 'observation_id', 'generation'))
            or any(observation.get(key) != scene.get(key) for key in ('observation_id', 'generation'))):
        raise ValueError('drop_requires_current_bin_occupancy')
    identifier = destination.get('id')
    if destination.get('kind') != 'bin' or not isinstance(identifier, str) or not identifier:
        raise ValueError('drop_requires_complete_enabled_bin_occupancy')
    name = 'semantic_instance::' + identifier
    if any(layer.startswith(name + '::bin_shell::') for layer in header['layers']):
        raise ValueError('scene_geometry_policy_requires_rebuild:cad_shell')
    layer = header['layers'].get(name, {})
    lower = np.asarray(layer.get('minimum_world_m'), float)
    upper = np.asarray(layer.get('maximum_world_m'), float)
    voxel_size = layer.get('voxel_size_m')
    native = (layer.get('kind') == 'voxel' and layer.get('source') == 'native_surface_voxel_cell_bounds'
        and layer.get('bounds_source') == 'all_native_surface_cells_before_preview_sampling'
        and type(layer.get('cell_count')) is int and layer['cell_count'] > 0
        and type(voxel_size) in (int, float) and np.isfinite(voxel_size) and voxel_size > 0)
    model = _valid_model_container_bounds(layer, scene, identifier, name)
    if (not (native or model) or layer.get('enabled') is not True
            or name in scene.get('disabled_obstacle_names', [])
            or lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower)):
        raise ValueError('drop_requires_complete_enabled_bin_occupancy')
    evidence = {'source': 'published_inventory_container_occupancy_bounds', 'entity_id': identifier,
        'scene_token': scene['scene_token'], 'observation_id': scene['observation_id'],
        'generation': scene['generation'], 'layer_name': name, 'layer_names': [name],
        'minimum_world_m': lower.tolist(), 'maximum_world_m': upper.tolist(),
        'identity_source': destination.get('identity_source')}
    if native:
        evidence.update(bounds_source=layer['bounds_source'], cell_count=layer['cell_count'], voxel_size_m=voxel_size)
    else:
        evidence.update(bounds_source=layer['source'], geometry_source=layer['geometry_source'],
            geometry_observation_id=layer['observation_id'], geometry_kind=layer['kind'])
        evidence.update({key: copy.deepcopy(layer[key]) for key in (
            'asset_id', 'source_sha256', 'mesh_sha256', 'geometry_observation_ids') if key in layer})
    return evidence


def container_transport_region(destination, scene, observation):
    """Bind a measured region; no CAD or hidden cavity is inferred."""
    evidence = container_occupied_bounds(destination, scene, observation)
    if destination.get('placement_reference', {}).get('observation_id') != observation['observation_id']:
        raise DiagnosticError('container_placement_region_unavailable', {
            'component': 'placement_geometry', 'observation_id': observation['observation_id'],
            'destination_id': destination['id'], 'container_collision_occupancy_valid': True,
            'full_box_bottom_required': False,
            'diagnostic_checks': [{'name': 'placement_region_observation', 'status': 'unknown',
                'actual': destination.get('placement_reference', {}).get('observation_id'),
                'expected': observation['observation_id']}],
            'required_evidence': 'A usable measured placement patch, a retained measured patch with '
                'validated current correspondence, or a declared CAD support facet with a current validated pose. '
                'Seeing the whole box bottom is not required.',
            'limitation': 'Container collision geometry does not establish an interior placement region. '
                'Rebuilding the same map or recapturing unchanged views does not reveal an occluded patch. '
                'Obtaining another view requires an explicit model-selected capability; this skill does not move.'})
    reference = destination['placement_reference']
    # The inscribed rectangle is useful for a grid layout, but arbitrary
    # container delivery may use the measured floor's full convex extent.
    # This adds no unobserved outer boundary and changes no collision layer.
    extent = 'observed_floor_hull' if 'observed_floor_hull_world_xy_m' in reference else 'inscribed_rectangle'
    polygon = np.asarray(reference.get('observed_floor_hull_world_xy_m', reference['region_polygon_world_xy_m']), float)
    region_source = 'measured_visible_floor_patch'
    if destination['placement_reference'].get('source') == 'retained_measured_floor_patch':
        region_source = 'retained_measured_floor_patch'
    elif reference.get('source') in {'validated_cad_support_plane', 'validated_cad_plane_with_measured_height'}:
        region_source = reference['source']
        extent = 'registered_cad_support_facet'
        polygon = np.asarray(reference['region_polygon_world_xy_m'], float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError('drop_requires_finite_bin_and_carried_geometry')
    center = (polygon.min(axis=0) + polygon.max(axis=0)) / 2
    return {'center_world_xy_m': center.tolist(), 'target_reference': 'finger_front',
        'free_placement': True,
        'polygon_world_xy_m': polygon.tolist(), 'source': region_source, 'placement_extent': extent,
        **({'support_provenance': copy.deepcopy(reference['region_measurement'])}
           if region_source.startswith('validated_cad_') else {}),
        'source_observation_id': destination['placement_reference'].get('source_observation_id',
                                                                        observation['observation_id']),
        'occupied_top_z_m': evidence['maximum_world_m'][2],
        'observation_id': observation['observation_id'], 'occupancy': evidence}



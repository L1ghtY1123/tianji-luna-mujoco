"""Collision occupancy derived from a measured container and an entry receipt."""
import hashlib
import json
from copy import deepcopy

import numpy as np


def container_geometry(model, identifier):
    name = 'semantic_instance::' + identifier
    for group in ('cuboid', 'mesh'):
        geometry = model.get(group, {}).get(name)
        if geometry and geometry.get('geometry_source') in {'registered_cad', 'observed_obb'}:
            return group, geometry
    raise ValueError('container_occupancy_requires_measured_container')


def occupancy_geometry(model, identity, target_id, container_id, receipt_id, binding_id):
    from .collision_mapping import collision_layer_bounds

    if target_id == container_id or not receipt_id or not binding_id:
        raise ValueError('container_occupancy_binding_invalid')
    group, geometry = container_geometry(model, container_id)
    name = 'semantic_instance::' + container_id
    bounds = collision_layer_bounds([], {group: {name: geometry}}, identity)['layers'][name]
    lower, upper = np.asarray(bounds['minimum_world_m']), np.asarray(bounds['maximum_world_m'])
    return {'instance_id': target_id, 'geometry_source': 'container_occupancy',
        'pose': [*((lower + upper) / 2).tolist(), 1., 0., 0., 0.], 'dims': (upper - lower).tolist(),
        'observation_id': geometry['observation_id'], 'generation': geometry['generation'],
        'container_id': container_id, 'container_geometry_sha256': hashlib.sha256(
            json.dumps(geometry, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'container_geometry_source': geometry['geometry_source'],
        'entry_receipt_id': receipt_id, 'release_binding_id': binding_id,
        'source': 'operator_container_release_rule', 'object_pose_measured': False,
        'occupancy_scope': 'whole_measured_container_bounds'}


def refresh_container_occupancy(model, layer, identity):
    """A current container measurement updates the same declared containment bound."""
    return occupancy_geometry(model, identity, layer['instance_id'], layer['container_id'],
                              layer['entry_receipt_id'], layer['release_binding_id'])


def occupancy_record(layer, previous=None):
    return {**deepcopy(previous or {}), 'status': 'retained', 'source': 'container_occupancy',
        'last_observation_id': layer['observation_id'], 'last_generation': layer['generation'],
        'layer_names': ['semantic_instance::' + layer['instance_id']],
        'container_id': layer['container_id'], 'entry_receipt_id': layer['entry_receipt_id'],
        'object_pose_measured': False}


def published_container_occupancy(entity, container, scene, invalidated_ids=()):
    """Read a receipt-bound occupied range for sensing and product exclusion."""
    from .demo_transfer_verification import valid_container_entry
    from .robot_skills.container_geometry import container_occupied_bounds

    identifier = entity.get('id')
    entry = entity.get('container_entry') or {}
    container_id = entry.get('destination_id')
    if (not valid_container_entry(entry, identifier, container_id, entry.get('release_feedback_id'))
            or container.get('id') != container_id
            or identifier in invalidated_ids or container_id in invalidated_ids
            or entry.get('generation') != scene.get('generation')):
        return None
    name = 'semantic_instance::' + identifier
    layer = scene.get('collision_layer_bounds', {}).get('layers', {}).get(name, {})
    record = scene.get('inventory_map', {}).get('entities', {}).get(identifier, {})
    if (record.get('source') != 'container_occupancy' or record.get('status') != 'retained'
            or record.get('entry_receipt_id') != entry['receipt_id']
            or layer.get('geometry_source') != 'container_occupancy'
            or layer.get('entry_receipt_id') != entry['receipt_id']
            or layer.get('release_binding_id') != entry['binding_id']
            or layer.get('container_id') != container_id or layer.get('enabled') is not True):
        return None
    try:
        bounds = container_occupied_bounds(container, scene, scene)
    except ValueError:
        return None
    if not np.allclose([layer.get('minimum_world_m'), layer.get('maximum_world_m')],
                       [bounds['minimum_world_m'], bounds['maximum_world_m']], atol=1e-9, rtol=0):
        return None
    return {'id': identifier, 'entity_id': identifier, 'container_id': container_id,
        'entry_receipt_id': entry['receipt_id'], 'layer_name': name,
        'scene_token': scene['scene_token'], 'generation': scene['generation'],
        'observation_id': bounds['observation_id'],
        'minimum_world_m': bounds['minimum_world_m'], 'maximum_world_m': bounds['maximum_world_m'],
        'geometry_source': 'container_occupancy', 'object_pose_measured': False}

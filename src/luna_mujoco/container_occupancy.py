"""Receipt-bound containment history backed by anonymous TSDF updates."""
from copy import deepcopy


def occupancy_record(request, previous=None):
    return {**deepcopy(previous or {}), 'status': 'stale', 'source': 'container_contents_pending',
        'container_id': request['container_id'], 'entry_receipt_id': request['receipt_id'],
        'release_binding_id': request['binding_id'], 'object_pose_measured': False,
        'layer_names': []}


def published_container_occupancy(entity, container, scene, invalidated_ids=()):
    """Defer individual recognition from accepted history, independently of visibility."""
    from .demo_transfer_verification import valid_container_entry

    identifier = entity.get('id')
    entry = entity.get('container_entry') or {}
    container_id = entry.get('destination_id')
    if (not valid_container_entry(entry, identifier, container_id, entry.get('release_feedback_id'))
            or container.get('id') != container_id
            or entry.get('generation') != scene.get('generation')):
        return None
    name = 'semantic_instance::' + container_id
    region = scene.get('collision_layer_bounds', {}).get('layers', {}).get(name, {})
    return {'id': identifier, 'entity_id': identifier, 'container_id': container_id,
        'entry_receipt_id': entry['receipt_id'], 'scene_token': scene.get('scene_token'),
        'generation': scene['generation'], 'observation_id': entry['observation_id'],
        'layer_name': 'workspace_environment',
        'minimum_world_m': region.get('minimum_world_m'), 'maximum_world_m': region.get('maximum_world_m'),
        'region_source': 'container_bounds_for_identity_scope_only',
        'geometry_source': 'workspace_tsdf_esdf', 'object_pose_measured': False}

"""Versioned inputs used to reuse identical scene analysis."""
from ..memory_reference import visual_identity_state


def observation_inputs(context):
    """Inputs to observation/identity interpretation, independent of planning state."""
    capture = context.get('latest_capture', {})
    reference = context.get('identity_reference', {})
    entity_fields = ('id', 'kind', 'color', 'description', 'appearance_keywords', 'initial_parent_id',
        'roles', 'segmentation_query', 'reference_camera_id', 'reference_observation_id', 'bbox',
        'identity_request_hash', 'current_identity_proposal', 'last_position_measurement',
        'asset_id', 'asset_source', 'geometry_policy', 'geometry_policy_instruction',
        'last_motion_receipt', 'current_region', 'visual_release', 'container_entry')
    entities = reference.get('object_memory_snapshot', {}).get('entities', reference.get('entities', {}))
    return {
        'capture_version': {key: capture.get(key) for key in ('generation', 'manifest_revision', 'command_revision')},
        'action_epoch': context.get('capture_epochs', {}).get(capture.get('observation_id')),
        'cameras': {camera: {key: view.get(key) for key in
            ('camera_id', 'calibration_id', 'intrinsics', 'T_world_camera', 'registered_to')}
            for camera, view in capture.get('views', {}).items()},
        'reference_id': reference.get('observation_id'),
        'entities': {identifier: {key: entity[key] for key in entity_fields if key in entity}
                     for identifier, entity in entities.items()},
        'lifecycle': visual_identity_state(reference),
        'association_scope': context.get('association_scope', {}),
        'unresolved_appearance_proposals': [{key: row['proposal'].get(key) for key in (
            'kind', 'color', 'segmentation_prompt', 'segmentation_aliases')}
            for row in reference.get('unresolved_proposals', []) if row.get('status') == 'unknown'],
        'settings': {key: context.get('planning_settings', {}).get(key) for key in ('observation', 'pointcloud_filter')},
    }



def observation_identity_scope(object_memory, spatial_memory, task_progress, task_spec, *, target_id=None):
    """Completed container deliveries retain history; current contents use anonymous TSDF."""
    from ..container_occupancy import published_container_occupancy

    completed = task_progress.completed_input_ids() if task_progress is not None else set()
    focus = set()
    if target_id and target_id not in completed:
        focus.add(target_id)
    if task_progress is not None:
        focus.update(set(task_progress.input_ids) - task_progress.completed_input_ids())
    entities = object_memory.entities()
    scene = spatial_memory.scene
    invalidated = spatial_memory.lifecycle().get('invalidated_entity_ids', [])
    occupancies = []
    for identifier in sorted(completed - focus):
        entity = entities.get(identifier, {})
        entry = entity.get('container_entry') or {}
        container_id = entry.get('destination_id')
        if not container_id or container_id in focus:
            continue
        occupied = published_container_occupancy(entity, entities.get(container_id, {}), scene, invalidated)
        if occupied is not None:
            occupancies.append(occupied)
    return {'focus_entity_ids': sorted(focus), 'deferred_entity_ids': [row['id'] for row in occupancies],
            'container_occupancies': occupancies,
            'discover_task_inputs': task_spec is not None and task_spec.scope in {'all', 'count'},
            'target_selector': task_spec.target_selector if task_spec is not None else ''}

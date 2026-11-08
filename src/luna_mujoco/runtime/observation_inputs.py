"""Versioned inputs used to reuse identical scene analysis."""
from ..inventory_reference import visual_inventory_state


def observation_inputs(context):
    """Inputs to observation/identity interpretation, independent of planning state."""
    capture = context.get('latest_capture', {})
    reference = context.get('visual_inventory_reference', {})
    entity_fields = ('id', 'kind', 'color', 'description', 'appearance_keywords', 'initial_parent_id',
        'roles', 'segmentation_query', 'reference_camera_id', 'reference_observation_id', 'bbox',
        'identity_request_hash', 'current_identity_proposal', 'last_position_measurement',
        'asset_id', 'asset_source', 'geometry_policy', 'geometry_policy_instruction')
    entities = reference.get('inventory_state', {}).get('entities', reference.get('entities', {}))
    return {
        'capture_version': {key: capture.get(key) for key in ('generation', 'manifest_revision', 'command_revision')},
        'action_epoch': context.get('capture_epochs', {}).get(capture.get('observation_id')),
        'cameras': {camera: {key: view.get(key) for key in
            ('camera_id', 'calibration_id', 'intrinsics', 'T_world_camera', 'registered_to')}
            for camera, view in capture.get('views', {}).items()},
        'reference_id': reference.get('observation_id'),
        'entities': {identifier: {key: entity[key] for key in entity_fields if key in entity}
                     for identifier, entity in entities.items()},
        'lifecycle': visual_inventory_state(reference),
        'association_scope': context.get('association_scope', {}),
        'settings': {key: context.get('planning_settings', {}).get(key) for key in ('observation', 'pointcloud_filter')},
    }



def observation_identity_scope(object_memory, spatial_memory, task_progress, task_spec, *, target_id=None):
    """Project current responsibilities and collision bounds into a sensing-only scope."""
    from ..demo_transfer_verification import valid_container_entry
    from ..container_occupancy import published_container_occupancy

    result = {'focus_entity_ids': [], 'deferred_entity_ids': [], 'container_occupancies': []}
    if task_progress is None or task_spec is None:
        return result
    entities = object_memory.entities()
    completed = set(task_progress.completed_ids)
    pending = set(task_progress.input_ids) - task_progress.completed_input_ids()
    focus = {identifier for identifier, record in task_progress.records.items()
             if record.get('root_input_id', identifier) in pending and identifier not in completed}
    if target_id and target_id not in completed:
        focus.add(target_id)
    result['focus_entity_ids'] = sorted(focus & entities.keys())
    scene = spatial_memory.scene
    invalidated = set(spatial_memory.lifecycle()['invalidated_entity_ids'])
    protected_containers = {task_spec.source_ref, task_spec.work_area_ref}
    # Refresh the destination while processing views still expose its contents,
    # before a later preparation pose or carried object can occlude them.
    if not task_spec.processing or focus:
        protected_containers.add(task_spec.destination_ref)
    for identifier in focus:
        entity = entities.get(identifier, {})
        protected_containers.update((entity.get('current_region'), entity.get('initial_parent_id'),
                                     (entity.get('observed_support') or {}).get('support_entity_id')))
    for identifier in sorted(completed):
        entity = entities.get(identifier, {})
        entry = entity.get('container_entry') or {}
        container_id = entry.get('destination_id')
        if not valid_container_entry(entry, identifier, container_id, entry.get('release_feedback_id')):
            continue
        # An unrestricted source can include this container. Preserve full discovery there.
        if not task_spec.source_ref or container_id in protected_containers:
            continue
        result['deferred_entity_ids'].append(identifier)
        bounds = published_container_occupancy(entity, entities.get(container_id, {}), scene, invalidated)
        if bounds is not None:
            result['container_occupancies'].append(bounds)
    if result['container_occupancies']:
        result.update(scene_token=scene['scene_token'], generation=scene['generation'])
    return result

"""Execution adapters for typed memory inputs and detached presentation snapshots."""
from copy import deepcopy

import numpy as np

from ..memory_contracts import (
    GeometryMeasurement,
    GeometryObservation,
    ObjectObservation,
    PositionMeasurement,
)
from ..object_memory import restored_object_record
from ..observation_geometry import current_table_measurements, measured_position


def release_memory_changes(entity, execution, attempt, result):
    """Physical release and visual receipts, independent of any task ledger snapshot."""
    physical = attempt.get('physical_release', {})
    target = entity['id']
    if (target != execution.get('target_id') or target != attempt.get('target_id')
            or not physical.get('feedback_id') or not execution.get('destination_id')
            or not all(physical.get(key) is True for key in
                       ('success', 'release_commanded', 'released', 'execution_completed'))):
        return {}
    feedback = physical['feedback_id']
    from ..demo_transfer_verification import valid_container_entry

    entry = result.get('container_entry', {})
    if valid_container_entry(entry, target, execution['destination_id'], feedback):
        destination_ref = execution.get('destination_ref', execution['destination_id'])
        previous_entry = entity.get('container_entry') or {}
        if previous_entry.get('release_feedback_id') == feedback:
            if (previous_entry.get('receipt_id') != entry['receipt_id']
                    or previous_entry.get('binding_id') != entry['binding_id']):
                raise ValueError('conflicting_container_entry_receipt')
            return {}
        return {'container_entry': deepcopy(entry), 'current_region': destination_ref,
            'confirmed_observation_id': None, 'observed_support': None,
            'transport_status': entry['placement_outcome'], 'transfers': ({
                'destination_id': execution['destination_id'], 'destination_ref': destination_ref,
                'transfer_index': len(entity.get('transfers', [])), 'arm': execution['arm'],
                'observation_id': entry['observation_id'], 'release_feedback_id': feedback,
                'source': entry['source'], 'receipt_id': entry['receipt_id'],
                'placement_outcome': entry['placement_outcome']},)}
    previous = entity.get('visual_release') or {}
    same_release = previous.get('action_id') == feedback
    release = deepcopy(previous) if same_release else {
        'action_id': feedback, 'transfer_index': len(entity.get('transfers', [])),
        'destination_id': execution['destination_id'], 'confirmed_observation_id': None,
        'pre_release_measurement_observation_id': (entity.get('last_position_measurement') or {}).get('observation_id')}
    changes = {}
    visual = result.get('visual_evidence', {})
    from ..surface_placement import placement_evidence_id
    bound_visual = (placement_evidence_id(visual) and visual.get('observation_id')
        and visual.get('target_id') == target and visual.get('destination_id') == execution['destination_id']
        and visual.get('release_feedback_id') == feedback)
    if bound_visual:
        if not same_release:
            changes.update(current_region=None, confirmed_observation_id=None, observed_support=None)
        release['assessment'] = {key: deepcopy(visual[key]) for key in (
            'observation_id', 'generation', 'request_hash', 'measurement_hash', 'execution_evidence_id', 'source', 'judgments', 'diagnostic_checks',
            'support_entity_id', 'placement_outcome') if key in visual}
        judgments = visual.get('judgments', {})
        support = visual.get('support_entity_id')
        if (support and visual.get('placement_outcome') == 'direct_success'
                and judgments.get('place_result') == 'yes' and judgments.get('target_released') == 'yes'):
            changes['observed_support'] = {
                'support_entity_id': support, 'observation_id': visual['observation_id'],
                'generation': visual.get('generation'),
                **{k: visual[k] for k in ('request_hash', 'measurement_hash') if visual.get(k)},
                'source': visual['source'], 'judgments': {
                    'place_result': 'yes', 'target_released': 'yes'},
                'scope': 'support_only_not_requested_subregion_completion'}
    confirmed = result.get('success') is True and visual.get('status') == 'at_destination' and bound_visual
    if confirmed and not release.get('confirmed_observation_id'):
        observation_id = visual['observation_id']
        destination_ref = execution.get('destination_ref', execution['destination_id'])
        release['confirmed_observation_id'] = observation_id
        changes.update(current_region=destination_ref, confirmed_observation_id=observation_id,
            transport_status='confirmed', transfers=({'destination_id': execution['destination_id'],
                'destination_ref': destination_ref, 'transfer_index': release['transfer_index'],
                'arm': execution['arm'], 'observation_id': observation_id, 'release_feedback_id': feedback},))
    elif not release.get('confirmed_observation_id') and bound_visual:
        changes['transport_status'] = 'awaiting_visual_confirmation'
    if bound_visual and visual.get('region_id'):
        release['region_id'] = visual['region_id']
    if confirmed and visual.get('placement_region'):
        release['placement_region'] = deepcopy(visual['placement_region'])
    changes['visual_release'] = release
    return changes


def measurement_reference(measured, observation):
    """Persist the pose/source fingerprint; raw RGBD and masks stay in their database tables."""
    import hashlib
    import json

    points = np.asarray(measured.get('points_world_m', []), dtype=np.float64)
    reference = {'observation_id': observation['observation_id'], 'generation': observation['generation'],
                 'points_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
                 'point_count': len(points), 'cad': measured.get('cad')}
    if measured.get('local_geometry_change'):
        reference['local_geometry_change'] = measured['local_geometry_change']
    return json.loads(json.dumps(reference, default=lambda value: value.tolist(), allow_nan=False))


def observation_update(observation, existing):
    entities = deepcopy(existing)
    for identifier, entity in observation.get('scene_entities', {}).items():
        if identifier not in entities:
            entities[identifier] = deepcopy(entity)
    positions = {}
    visible = []
    for obj in observation['objects']:
        identifier = obj['id']
        if not identifier.isascii() or not identifier.isdecimal():
            continue  # Configured regions have their own spatial configuration.
        entity = entities.get(identifier, {})
        for key in ('id', 'kind', 'color'):
            if key in obj:
                entity.setdefault(key, deepcopy(obj[key]))
        if 'roles' not in entity:
            entity['roles'] = ['container', 'collision'] if obj['kind'] == 'bin' else ['collision']
        if 'segmentation_query' not in entity and obj.get('segmentation_prompt'):
            entity['segmentation_query'] = {
                'kind': obj['kind'], 'color': obj.get('color', 'unknown'), 'prompt': obj['segmentation_prompt']}
        entities[identifier] = entity
        visible.append(identifier)
        position = measured_position(obj)
        if position is not None:
            positions[identifier] = PositionMeasurement(observation_id=observation['observation_id'],
                position_world_m=tuple(position.tolist()), source=obj.get('position_source', 'rgbd_visible_surface'))
    return ObjectObservation(observation_id=observation['observation_id'],
        entities=tuple(restored_object_record(entity) for key, entity in entities.items()
                       if key.isascii() and key.isdecimal()),
        visible_ids=tuple(visible), positions=positions)


def geometry_update(observation):
    """Project sensor data into the execution layer's current geometry inputs."""
    measured = {}
    for obj in observation['objects']:
        position = measured_position(obj)
        item = {'observation_id': observation['observation_id'], 'geometry_status': 'measured'}
        if position is not None:
            item['position'] = position.tolist()
        measured[obj['id']] = GeometryMeasurement.model_validate(item)
    for identifier, item in current_table_measurements(observation).items():
        measured[identifier] = GeometryMeasurement.model_validate({**item, 'geometry_status': 'measured'})
    return GeometryObservation(observation_id=observation['observation_id'], measurements=measured)


def merge_geometry_measurements(previous, update: GeometryObservation):
    retained = {}
    for identifier, item in previous.items():
        status = 'cached' if item.kind == 'table' else 'missing'
        retained[identifier] = GeometryMeasurement.model_validate(item.model_dump() | {'geometry_status': status})
    return retained | {key: row.model_copy(deep=True) for key, row in update.measurements.items()}


def freeze_scene_observation(observation):
    """Keep the sensor input bound to a collision publication in execution state."""
    result = deepcopy({key: value for key, value in observation.items()
                       if key not in {'object_memory_snapshot', 'scene_entities', 'visual_reference'}})
    for key in ('registered_entities', 'membership', 'progress'):
        result.get('visual_identity', {}).pop(key, None)
    return result


def planning_observation(observation, object_memory, planning_regions):
    """Join current instance metadata to the map's frozen measurement for one skill call."""
    result = deepcopy(observation)
    result['object_memory_snapshot'] = object_memory.snapshot()
    result['scene_entities'] = deepcopy(result['object_memory_snapshot']['entities'])
    result['scene_entities'].update(deepcopy(planning_regions))
    return result


def perception_motion_context(object_memory, spatial_memory):
    """Dated physical changes select pose initialization; task progress never enters perception."""
    invalidated = set(spatial_memory.lifecycle()['invalidated_entity_ids'])
    fields = ('transport_status', 'transfers', 'processing_history', 'output_entity_id')
    return {identifier: {**{field: deepcopy(record.get(field)) for field in fields},
                         'geometry_invalidated': identifier in invalidated}
            for identifier, record in object_memory.entities().items() if record['status'] == 'active'}


def execution_snapshot(task_progress, object_memory, geometry_measurements, planning_regions):
    """Compose the workbench response without giving either memory a task dependency."""
    memory = object_memory.snapshot()
    entities = memory['entities']
    measurements = {key: row.model_dump(mode='json', exclude_none=True)
                    for key, row in geometry_measurements.items()}
    scene_entities = {key: deepcopy(entity) | deepcopy(measurements.get(key, {}))
                      for key, entity in entities.items()}
    scene_entities.update(deepcopy(planning_regions))
    return {**task_progress.snapshot(), **memory, 'scene_entities': scene_entities,
            'scene_objects_observation_id': object_memory.observation_id}

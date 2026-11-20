"""Commit physical facts through the database, independently of graph control."""
from copy import deepcopy

import numpy as np

from ..memory_contracts import ObjectOperation, ObjectRecord
from ..visual_identity import mask_box
from .memory import (
    geometry_update,
    measurement_reference,
    merge_geometry_measurements,
    observation_update,
    release_memory_changes,
)


class PhysicalMemoryWriter:
    """A proposed memory becomes visible only after its dated evidence commits."""

    def __init__(self, database):
        self.database = database

    def operation(self, memory, operation):
        proposed = deepcopy(memory)
        proposed.record_operation(operation)
        record = ObjectRecord.model_validate(proposed.entities()[operation.entity_id])
        self.database.record(operation.entity_id, operation.event_key, operation.kind,
            record.model_dump(mode='json', exclude={'id'}), operation.evidence)
        return proposed

    def observation(self, memory, geometry, *, capture, observation, run_id, step):
        if any(observation.get(key) != capture.get(key) for key in ('observation_id', 'generation')):
            raise ValueError('observation_commit_capture_mismatch')
        proposed = deepcopy(memory)
        proposed.observe(observation_update(observation, proposed.entities()))
        measurements = merge_geometry_measurements(geometry, geometry_update(observation))
        committed = self.checkpoint(proposed, skill='observe_scene', result={'status': 'success'},
            status='success', capture=capture, observation=observation, target_id=None,
            run_id=run_id, step=step)
        return committed, measurements

    def checkpoint(self, memory, *, skill, result, status, capture, observation, target_id,
                   run_id, step, release_execution=None, release_attempt=None, container_receipt=None):
        entities = memory.entities()
        operation_observation_id = observation.get('observation_id')
        if skill in {'verify_place', 'verify_grasp'}:
            proof = result.get('container_entry') or result.get('visual_evidence') or result
            operation_observation_id = proof.get('observation_id') or operation_observation_id
        observed = skill == 'observe_scene' and status == 'success'
        identifiers = {key for key, entity in entities.items() if entity['status'] == 'active'} if observed else {
            target_id, result.get('selected_id'), result.get('input_entity_id'), result.get('output_entity_id')}
        if skill == 'select_target':
            identifiers = {result.get('selected_id')}
        proposed = deepcopy(memory)
        events = []
        for identifier in sorted(identifiers & entities.keys()):
            entity = proposed.entities()[identifier]
            if observed:
                measured = next((obj for obj in observation['objects'] if obj['id'] == identifier), None)
                if measured is not None:
                    entity['geometry_reference'] = measurement_reference(measured, observation)
                head = observation.get('views', {}).get('3', {})
                mask = head.get('identity_masks', head.get('instance_masks', {})).get(identifier)
                if mask is not None and np.any(mask):
                    entity.update(reference_observation_id=observation['observation_id'],
                        reference_camera_id='3', bbox=mask_box(mask), identity_source='openai_decisions',
                        identity_request_hash=observation['visual_identity']['evidence']['request_hash'])
            proposed.update(ObjectRecord.model_validate(entity))
            event = run_id + ':' + step['invocation_id'] + (':observation' if observed else '')
            evidence = {'run_id': run_id, 'observation_id': operation_observation_id,
                'status': status, 'reason': result.get('reason'), 'step': deepcopy(step)}
            memory_result = {**result, 'container_entry': container_receipt} if container_receipt else result
            changes = (release_memory_changes(entity, release_execution or {}, release_attempt or {}, memory_result)
                       if skill == 'verify_place' else {})
            operation = ObjectOperation(entity_id=identifier, event_key=event, kind=skill,
                observation_id=operation_observation_id, evidence=evidence, **changes)
            proposed.record_operation(operation)
            current = ObjectRecord.model_validate(proposed.entities()[identifier]).model_dump(mode='json')
            events.append((identifier, operation.event_key, operation.kind,
                {key: value for key, value in current.items() if key != 'id'}, operation.evidence))
        if observed:
            self.database.commit_observation(capture, observation, events)
        else:
            self.database.record_many(events)
        return proposed

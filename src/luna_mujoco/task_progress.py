"""Measured task history. This ledger never chooses targets or recovery actions."""
import hashlib
import json
from copy import deepcopy

import numpy as np

from .observation_geometry import SOURCE_BIN_MIN_XY_FRACTION, measured_position


def confirmed_task_source(record):
    """Task membership is a bound visual judgment, never a free-space certificate."""
    initial = record.get('initial_source', {})
    evidence = initial.get('task_source_evidence', {})
    if (evidence.get('source') == 'openai_decisions' and evidence.get('status') == 'yes'
            and evidence.get('target_id') == record.get('id')
            and evidence.get('source_id') and evidence.get('source_kind')
            and evidence.get('request_hash') and evidence.get('observation_id')
            and evidence['observation_id'] == initial.get('measurement_observation_id')):
        return evidence
    return None


def original_source_bin(record):
    """A tiny floor-hull overlap is not a source-box identity."""
    initial = record.get('initial_source', {})
    if 'task_source_evidence' in initial:
        confirmed = confirmed_task_source(record)
        return confirmed['source_id'] if confirmed and confirmed['source_kind'] == 'bin' else None
    if initial.get('identity_source') == 'openai_responses' and initial.get('initial_parent_kind') == 'bin':
        return initial.get('initial_parent_id')
    candidates = {item['object_id'] for item in record.get('initial_source', {}).get('measured_relationships', [])
                  if item.get('relation') == 'above_observed_bin_floor'
                  and item.get('visible_xy_fraction', 0.) >= SOURCE_BIN_MIN_XY_FRACTION}
    return next(iter(candidates)) if len(candidates) == 1 else None


def source_region_geometry(record, objects):
    """Expose measured separation for membership judgment, not hidden box extents."""
    position = measured_position(record)
    if position is None:
        return []
    result = []
    for obj in objects:
        reference = obj.get('placement_reference', {})
        polygon = np.asarray(reference.get('observed_floor_hull_world_xy_m',
                                           reference.get('region_polygon_world_xy_m')), dtype=float)
        if (obj.get('kind') != 'bin' or polygon.ndim != 2 or polygon.shape[1] != 2
                or len(polygon) < 3 or not np.isfinite(polygon).all()):
            continue
        gap = np.maximum(np.maximum(polygon.min(axis=0) - position[:2],
                                    position[:2] - polygon.max(axis=0)), 0.)
        result.append({'bin_id': obj['id'],
            'centroid_to_observed_floor_xy_bounds_gap_m': round(float(np.linalg.norm(gap)), 6),
            'source': 'measured_centroid_and_visible_floor_polygon', 'complete_box_bounds': False})
    return result


class TaskProgress:
    def __init__(self, instruction, scope="single", requested_arm=None, *,
                 requested_count=None, selection_order='explicit'):
        if requested_count is not None and (type(requested_count) is not int or requested_count < 1):
            raise ValueError('invalid_requested_count')
        if selection_order not in {'inventory', 'explicit'}:
            raise ValueError('invalid_selection_order')
        self.requested_count = requested_count
        self.selection_order = selection_order
        self.instruction = instruction
        self.scope = scope
        self.requested_arm = requested_arm
        self.records = {}
        self.completed_ids = []
        self.attempts = {}
        self.current_id = None
        self.observation_id = None
        self.visible_ids = set()
        self.failure = None
        self.relevance = {}
        self.ledger_revision = 0
        self.inventory_initialized = False
        self.membership = {}
        self.placement_requests = {}

    def placement_request(self, destination_id, *, transfer_index=None):
        """Bind the current object to the measured destination region."""
        index = self.transfer_index() if transfer_index is None else transfer_index
        key = (self.current_id, index, destination_id)
        request = self.placement_requests.get(key, {
            'transfer_index': index, 'destination_id': destination_id,
            'region_id': 'region:' + destination_id, 'ledger_revision': self.ledger_revision})
        return deepcopy(request) | {'object_id': self.current_id,
            'release_mode': self.records[self.current_id]['release_mode']}

    def bind_shared_work_area(self, destination, observation):
        """Use the full measured destination for skill-selected or configured XY."""
        reference = destination['placement_reference']
        polygon = np.asarray(reference['region_polygon_world_xy_m'], float)
        if (polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3
                or not np.isfinite(polygon).all()):
            raise ValueError('shared_work_area_requires_polygon')
        area = float(abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1))
                         - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1))) / 2)
        if area <= 0:
            raise ValueError('shared_work_area_requires_polygon')
        layout = {'center_world_xy_m': polygon.mean(axis=0).tolist(),
            'region_polygon_world_xy_m': polygon.tolist(),
            'floor_z_m': reference['floor_z_m'], 'region_id': 'region:' + destination['id'],
            'support_entity_id': reference.get('support_entity_id', destination['id']),
            'observation_id': observation['observation_id'], 'source': reference.get('source')}
        key = (self.current_id, self.transfer_index(), destination['id'])
        self.placement_requests[key] = {'transfer_index': key[1],
            'destination_id': destination['id'], 'region_id': layout['region_id'],
            'ledger_revision': self.ledger_revision,
            'layout': layout, 'layout_revision': hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:24]}
        return self.placement_request(destination['id'])

    def visual_progress(self):
        """Expose the destination geometry bound to each object's transfer."""
        result = {}
        for identifier, record in self.records.items():
            if self.membership.get(identifier) != 'required':
                continue
            item = {key: deepcopy(record[key]) for key in
                    ('status', 'destination_id', 'transfer_index', 'release_feedback_id') if key in record}
            destination = record.get('destination_id')
            request = self.placement_requests.get((identifier, self.transfer_index(identifier), destination), {})
            layout = request.get('layout')
            if layout:
                item['region_id'] = request['region_id']
                item['placement_region'] = {
                    'polygon_world_xy_m': deepcopy(layout['region_polygon_world_xy_m']),
                    'center_world_xy_m': deepcopy(layout['center_world_xy_m']),
                    'floor_z_m': layout['floor_z_m'], 'layout_revision': request['layout_revision']}
            result[identifier] = item
        return result

    def transfer_index(self, identifier=None):
        """The selected transfer keeps its geometry index after confirmation."""
        return self.records.get(identifier or self.current_id, {}).get('transfer_index', 0)

    def bound_arm(self):
        return self.requested_arm or self.records.get(self.current_id, {}).get('execution_arm')

    def bind_arm(self, arm):
        if arm not in {'A', 'B'} or self.current_id not in self.records:
            raise ValueError('invalid_execution_arm')
        if self.requested_arm and self.requested_arm != arm:
            raise ValueError('requested_arm_cannot_switch')
        self.records[self.current_id]['execution_arm'] = arm

    def observe(self, objects, observation_id):
        before = deepcopy(self.records)
        self.observation_id = observation_id
        self.visible_ids = {obj['id'] for obj in objects}
        for record in self.records.values():
            record['visibility'] = 'visible' if record['id'] in self.visible_ids else 'unseen'
        for obj in objects:
            if obj.get('kind') in {'table', 'bin', 'marker'}:
                continue
            record = self.records.setdefault(obj['id'], {'id': obj['id'], 'status': 'observed'})
            self.membership.setdefault(obj['id'], 'unknown')
            if obj.get('visual_membership') in {'required', 'excluded'}:
                record['visual_membership'] = obj['visual_membership']
            record.update(visibility='visible', association_status='matched',
                          last_seen_observation_id=observation_id,
                          position_source=obj.get('position_source', 'rgbd_visible_surface'))
            record['observation_reference'] = {'observation_id': observation_id, 'instance_id': obj['id']}
            record['last_observed_attributes'] = {key: deepcopy(obj[key]) for key in
                ('kind', 'color', 'coverage', 'measured_relationships', 'visual_semantics') if key in obj}
            position = measured_position(obj)
            if position is not None:
                record['last_observed_position'] = position.tolist()
            if record['status'] in {'selected', 'completed', 'deferred'}:
                continue  # Preserve source-region semantics and quotas after moving the object.
            for key in ('kind', 'color', 'position', 'visual_semantics', 'measured_relationships'):
                if key in obj:
                    record[key] = deepcopy(obj[key])
            if not record.get('transfers') and not self.attempts.get(obj['id']):
                # Discovery creates a proposal before RGBD association. Add the
                # first measured source evidence before selection or movement.
                initial = record.setdefault('initial_source', {})
                initial.update({key: deepcopy(obj[key]) for key in
                    ('position', 'visual_semantics', 'measured_relationships') if key in obj})
                initial['measurement_observation_id'] = observation_id
        if self.records != before:
            self.ledger_revision += 1
        if self.records and self.scope == 'single' and self.selection_order == 'inventory':
            self.inventory_initialized = True

    def register_entities(self, entities, memberships=None):
        """Initialize task-local records from identity metadata and explicit task membership."""
        memberships = memberships or {}
        for identifier, entity in entities.items():
            if (entity['kind'] in {'table', 'bin', 'marker'} or entity.get('status') in {'consumed', 'removed'}
                    or identifier in self.records):
                continue
            parent = entities.get(entity.get('initial_parent_id'), {})
            membership = memberships.get(identifier, 'unknown')
            self.records[identifier] = {
                'id': identifier, 'status': 'observed', 'visual_membership': membership,
                **{key: deepcopy(entity[key]) for key in ('kind', 'color', 'description') if key in entity},
                'initial_source': {
                    **{key: deepcopy(entity[key]) for key in ('initial_parent_id', 'identity_source') if key in entity},
                    'initial_parent_kind': parent.get('kind')}}
            self.membership[identifier] = membership
            self.ledger_revision += 1
        self.inventory_initialized = True

    def fail(self, identifier, stage, code, evidence):
        key = identifier or 'task'
        counted = evidence.get('status') != 'unknown' and evidence.get('executed') is not False
        self.attempts[key] = self.attempts.get(key, 0) + int(counted)
        previous = self.failure or {}
        repeated = previous.get('target_id') == identifier and previous.get('code') == code
        same_code_count = previous.get('consecutive_same_code_count', 1) + 1 if repeated else 1
        self.failure = {'target_id': identifier, 'stage': stage, 'code': code,
                        'evidence': deepcopy(evidence), 'attempt_count': self.attempts[key],
                        'retry_limit': 3, 'consecutive_same_code_count': same_code_count,
                        'counts_as_attempt': counted}

    def record_release(self, identifier, *, feedback_id, action_id=None):
        """An open gripper is release evidence, not destination confirmation."""
        if not feedback_id:
            raise ValueError('release_requires_feedback')
        record = self.records[identifier]
        action_id = action_id or feedback_id
        if record.get('visual_release', {}).get('action_id') != action_id:
            visual = self.visual_progress().get(identifier, {})
            record.update(release_feedback_id=feedback_id, transport_status='released', visual_release={
                'action_id': action_id, 'transfer_index': self.transfer_index(identifier),
                'destination_id': record.get('destination_id'), 'confirmed_observation_id': None,
                **{key: visual[key] for key in ('region_id',) if visual.get(key) is not None}})
            self.ledger_revision += 1

    def confirm_visual_transfer(self, identifier, evidence):
        record = self.records[identifier]
        release = record.get('visual_release', {})
        region_matches = (not release.get('region_id')
                          or evidence.get('region_id') == release['region_id'])
        if release.get('confirmed_observation_id'):
            return (region_matches and record.get('destination_id') == release['destination_id']
                    and evidence.get('destination_id') == release['destination_id']
                    and bool(evidence.get('observation_id'))
                    and evidence.get('source') == 'openai_decisions'
                    and evidence.get('status') == 'at_destination')
        record['destination_visual_evidence'] = deepcopy(evidence)
        confirmed = (region_matches and bool(record.get('release_feedback_id'))
                     and release.get('transfer_index') == self.transfer_index(identifier)
                     and release.get('destination_id') == record.get('destination_id')
                     and evidence.get('source') == 'openai_decisions'
                     and evidence.get('status') == 'at_destination'
                     and evidence.get('destination_id') == record.get('destination_id')
                     and bool(evidence.get('observation_id')))
        if confirmed:
            self.complete(identifier, observation_id=evidence['observation_id'])
            release['confirmed_observation_id'] = evidence['observation_id']
            record['transport_status'] = 'confirmed'
        else:
            record['transport_status'] = 'awaiting_visual_confirmation'
        return confirmed

    def record_visual_progress(self, identifier, evidence):
        """Keep verified release evidence and later observations separately."""
        record = self.records[identifier]
        release = record.get('visual_release', {})
        confirmed = (release.get('confirmed_observation_id')
                     and release.get('destination_id') == record.get('destination_id')
                     and release.get('transfer_index') == self.transfer_index(identifier))
        field = ('destination_tracking_evidence' if confirmed
                 else 'destination_visual_evidence')
        record[field] = deepcopy(evidence)

    def snapshot(self):
        members = {key: item for key, item in self.records.items()
                   if self.membership.get(key, 'unknown') == 'required'}
        unresolved = [deepcopy(item) for key, item in self.records.items()
                      if self.membership.get(key) == 'unknown']
        completed = [deepcopy(self.records[identifier]) for identifier in self.completed_ids]
        deferred = [deepcopy(item) for item in self.records.values() if item['status'] == 'deferred']
        unseen = [key for key, item in members.items()
                  if key not in self.visible_ids and item['status'] not in {'completed', 'deferred'}]
        return {'instruction': self.instruction, 'scope': self.scope, 'requested_arm': self.requested_arm,
                'requested_count': self.requested_count, 'selection_order': self.selection_order,
                'transfer_index': self.transfer_index(),
                'ledger_revision': self.ledger_revision,
                'inventory_initialized': self.inventory_initialized,
                'placement_regions': deepcopy(list(self.placement_requests.values())),
                'objects': deepcopy(sorted(members.values(), key=lambda item: item['id'])),
                'membership': dict(self.membership), 'unresolved_membership': unresolved,
                'membership_pending_ids': [item['id'] for item in unresolved],
                'membership_review_ids': [item['id'] for item in unresolved],
                'current_id': self.current_id, 'current': deepcopy(self.records.get(self.current_id)),
                'completed': completed, 'deferred': deferred,
                'unseen_ids': unseen, 'attempts': dict(self.attempts),
                'unseen_objects': [deepcopy(self.records[key]) for key in unseen],
                'unseen_relevance': {key: self.relevance.get(key, 'unknown') for key in unseen},
                'observation_id': self.observation_id, 'failure': deepcopy(self.failure)}

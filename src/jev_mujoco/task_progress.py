"""Measured task history. This ledger never chooses targets or recovery actions."""
import hashlib
import json
from copy import deepcopy

import numpy as np

from .inventory_order import head_image_order, head_rgb_box_order

SOURCE_BIN_MIN_XY_FRACTION = .5


def measured_position(obj):
    value = obj.get('visible_centroid_world', obj.get('position'))
    if value is None:
        return None
    position = np.asarray(value, dtype=float)
    return position if position.shape == (3,) and np.isfinite(position).all() else None


def original_source_bin(record):
    """A tiny floor-hull overlap is not a source-box identity."""
    initial = record.get('initial_source', {})
    if initial.get('identity_source') == 'openai_responses' and initial.get('initial_parent_kind') == 'bin':
        return initial.get('initial_parent_id')
    candidates = {item['object_id'] for item in record.get('initial_source', {}).get('measured_relationships', [])
                  if item.get('relation') == 'above_observed_bin_floor'
                  and item.get('visible_xy_fraction', 0.) >= SOURCE_BIN_MIN_XY_FRACTION}
    return next(iter(candidates)) if len(candidates) == 1 else None


def source_region_confirmed(route, record):
    """Validate the declared route's source, without deciding task membership.

    Reuse the same measured overlap test as source-box identity. Missing or
    ambiguous geometry is unconfirmed, never proof that an object is outside.
    Preserve the original source when a two-leg transfer reaches the table.
    """
    if route not in {'box_table', 'box_table_box'}:
        return True
    reference = record if 'initial_source' in record else {'initial_source': record}
    return original_source_bin(reference) is not None


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
    def __init__(self, instruction, scope="single", requested_arm=None, *, route='direct',
                 requested_count=None, selection_order='explicit', quantity_groups=None):
        if requested_count is not None and (type(requested_count) is not int or requested_count < 1):
            raise ValueError('invalid_requested_count')
        if selection_order not in {'inventory', 'explicit'}:
            raise ValueError('invalid_selection_order')
        self.requested_count = requested_count
        self.selection_order = selection_order
        self.instruction = instruction
        self.scope = scope
        self.requested_arm = requested_arm
        if route not in {'direct', 'box_table', 'table_box', 'box_table_box'}:
            raise ValueError('unsupported_transfer_route')
        self.route = route
        self.stage_arms = {}
        self.records = {}
        self.completed_ids = []
        self.attempts = {}
        self.current_id = None
        self.observation_id = None
        self.visible_ids = set()
        self.failure = None
        self.relevance = {}
        self.next_display_id = 1
        self.ledger_revision = 0
        self.inventory_initialized = False
        self.membership = {}
        self.placement_requests = {}
        self.inventory_finalized = False
        self.inventory_revision = 0
        self.initial_head_observation_id = None
        self.scene_entities = {}
        self.scene_inventory_observation_id = None
        self.quantity_groups = deepcopy(quantity_groups or [])
        self.group_membership = {}
        self.initial_head_reference = None
        self.placement_support_walls = {}
        self.route_destinations = {}

    def placement_request(self, destination_id, *, stage_index=None):
        """Keep global inventory capacity and bind the current number to its slot."""
        stage = self.stage_index() if stage_index is None else stage_index
        key = (stage, destination_id)
        if key not in self.placement_requests:
            identifiers = []
            for identifier in self.records:
                if self.membership.get(identifier) != 'required':
                    continue
                identifiers.append(identifier)
            # Standalone place has a fresh task ledger but keeps the initial
            # scene's transport numbering. Slot 3 still needs that layout size.
            frozen_capacity = max((entity.get('display_id', 0) for entity in self.scene_entities.values()
                                   if 'transport' in entity.get('roles', ())), default=0)
            target_count = max(1, len(identifiers), frozen_capacity)
            self.placement_requests[key] = {'target_count': target_count,
                'target_ids': sorted(identifiers), 'stage_index': stage, 'destination_id': destination_id,
                'region_id': 'region:' + destination_id, 'inventory_revision': self.inventory_revision,
                'source': 'confirmed_task_inventory' if identifiers else 'single_held_object'}
        request = deepcopy(self.placement_requests[key])
        number = self.records.get(self.current_id, {}).get('display_id')
        if number is None and request['target_count'] == 1:
            number = 1
        return request | {'object_id': self.current_id, 'object_display_id': number, 'slot_id': number}

    def visual_progress(self):
        """The visual judge gets the same frozen slot used by placement planning."""
        result = {}
        for identifier, record in self.records.items():
            if self.membership.get(identifier) != 'required':
                continue
            item = {key: deepcopy(record[key]) for key in
                    ('status', 'destination_id', 'stage_index', 'release_feedback_id') if key in record}
            destination = record.get('destination_id')
            region = self.placement_requests.get((self.stage_index(identifier), destination), {})
            if destination:
                item.update(region_id=region.get('region_id', 'region:' + destination),
                            slot_id=record.get('display_id'))
            layout = region.get('layout', {})
            cell = next((cell for cell in layout.get('cells', [])
                         if cell['slot_id'] == item.get('slot_id')), None)
            if cell is not None:
                item['assigned_slot'] = deepcopy(cell) | {'floor_z_m': layout['floor_z_m'],
                    'layout_revision': region.get('layout_revision')}
            result[identifier] = item
        return result

    def bind_placement_region(self, destination, observation, *, stage_index=None):
        from .placement_layout import (
            measured_region_translation,
            measured_support_walls,
            partition_in_head_frame,
        )

        identifier = destination['id']
        stage = self.stage_index() if stage_index is None else stage_index
        request = self.placement_request(identifier, stage_index=stage)
        key = (stage, identifier)
        reference = destination.get('placement_reference', {})
        walls = measured_support_walls(destination)
        if 'layout' in request:
            layout = request['layout']
            delta = measured_region_translation(layout, reference, self.placement_support_walls.get(key, ()), walls)
            if delta is None:
                return request
            for cell in layout['cells']:
                cell['center_world_xy_m'] = (np.asarray(cell['center_world_xy_m']) + delta[:2]).tolist()
                cell['polygon_world_xy_m'] = (np.asarray(cell['polygon_world_xy_m']) + delta[:2]).tolist()
            layout['center_world_xy_m'] = (np.asarray(layout['center_world_xy_m']) + delta[:2]).tolist()
            layout['region_polygon_world_xy_m'] = (np.asarray(layout['region_polygon_world_xy_m']) + delta[:2]).tolist()
            layout['floor_z_m'] += float(delta[2])
            layout.update(observation_id=observation['observation_id'], measurement_time=observation.get('time'), movement_evidence={
                'method': 'independent_bin_walls_and_region_translation', 'translation_world_m': delta.tolist(),
                'boundary_residual_limit_m': .002})
            revision = hashlib.sha256(json.dumps(layout, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
            self.placement_requests[key].update(layout=layout, layout_revision=revision)
            self.placement_support_walls[key] = walls.copy()
            return self.placement_request(identifier, stage_index=stage)
        polygon = reference.get('region_polygon_world_xy_m')
        if polygon is None or 'floor_z_m' not in reference:
            raise ValueError('placement_requires_measured_region_polygon')
        head = self.initial_head_reference or observation.get('views', {}).get('3', {})
        layout = partition_in_head_frame(polygon, request['target_count'], head, reference['floor_z_m'])
        layout.update(floor_z_m=reference['floor_z_m'], region_id=request['region_id'],
                      support_entity_id=identifier, source=reference.get('source'),
                      observation_id=observation['observation_id'],
                      measurement_time=observation.get('time'),
                      initial_head_observation_id=self.initial_head_observation_id or observation['observation_id'])
        revision = hashlib.sha256(json.dumps(layout, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
        self.placement_requests[key].update(layout=layout, layout_revision=revision)
        self.placement_support_walls[key] = walls.copy()
        return self.placement_request(identifier, stage_index=stage)

    def confirm_route_destinations(self, selections, observation):
        """Validate all JEV-selected route regions before permitting first motion."""
        if not self.inventory_finalized:
            raise ValueError('route_requires_frozen_inventory')
        members = {key for key, value in self.membership.items() if value == 'required'}
        stages = {'0', '1'} if self.route == 'box_table_box' else {'0'}
        if set(selections) != members or any(set(value) != stages for value in selections.values()):
            raise ValueError('complete_route_destinations_required')
        visible = {obj['id']: obj for obj in observation['objects']}
        regions = {}
        for identifier, selected in selections.items():
            for stage, destination_id in selected.items():
                destination = visible.get(destination_id, {})
                to_table = self.route == 'box_table' or self.route == 'box_table_box' and stage == '0'
                if destination.get('kind') != ('marker' if to_table else 'bin'):
                    raise ValueError('route_destination_missing_or_wrong_kind')
                if (self.route == 'box_table_box' and stage == '1'
                        and destination_id == original_source_bin(self.records[identifier])):
                    raise ValueError('final_destination_must_differ_from_source_bin')
                reference = destination.get('placement_reference', {})
                if 'region_polygon_world_xy_m' not in reference or 'floor_z_m' not in reference:
                    raise ValueError('route_destination_region_unmeasured')
                regions[int(stage), destination_id] = destination
        for (stage, _), destination in regions.items():
            self.placement_request(destination['id'], stage_index=stage)
        self.route_destinations = deepcopy(selections)
        self.ledger_revision += 1

    def pending_route_id(self):
        return next((key for key, item in self.records.items()
                     if item.get('stage_index') == 1 and item['status'] not in
                     {'completed', 'deferred'}), None)

    def stage_index(self, identifier=None):
        identifier = identifier or self.current_id or self.pending_route_id()
        return self.records.get(identifier, {}).get('stage_index', 0)

    def bound_arm(self):
        return self.requested_arm or self.stage_arms.get(str(self.stage_index()))

    def bind_arm(self, arm):
        if arm not in {'A', 'B'}:
            raise ValueError('invalid_execution_arm')
        if self.requested_arm and self.requested_arm != arm:
            raise ValueError('requested_arm_cannot_switch')
        stage = str(self.stage_index())
        if self.stage_arms.get(stage) not in (None, arm):
            raise ValueError('stage_arm_is_fixed')
        self.stage_arms[stage] = arm

    def observe(self, objects, observation_id):
        before = deepcopy(self.records)
        self.observation_id = observation_id
        self.visible_ids = {obj['id'] for obj in objects}
        for entity in self.scene_entities.values():
            if entity.get('kind') != 'table':
                entity['visibility'] = 'visible' if entity['id'] in self.visible_ids else 'unseen'
                entity['geometry_status'] = 'measured' if entity['id'] in self.visible_ids else 'missing'
        for obj in objects:
            if self.scene_inventory_observation_id and obj['id'] not in self.scene_entities:
                continue
            entity = self.scene_entities.setdefault(obj['id'], {key: deepcopy(obj[key]) for key in
                ('id', 'kind', 'color') if key in obj})
            if 'position' in obj:
                entity['position'] = deepcopy(obj['position'])
            entity.update(observation_id=observation_id, visibility='visible')
        for record in self.records.values():
            record['visibility'] = 'visible' if record['id'] in self.visible_ids else 'unseen'
        for obj in objects:
            if obj.get('kind') in {'bin', 'marker'}:
                continue
            if self.scene_inventory_observation_id and obj['id'] not in self.scene_entities:
                continue
            if self.inventory_finalized and obj['id'] not in self.records:
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
        if self.records != before:
            self.ledger_revision += 1
        if self.records and (self.route != 'direct'
                or self.scope == 'single' and self.selection_order == 'inventory'):
            self.inventory_initialized = True

    def initialize_scene_inventory(self, observation):
        """Fix all initial entities; later captures only update their measurements."""
        if self.scene_inventory_observation_id:
            return
        semantic = observation.get('scene_entities')
        if semantic:
            self.scene_entities = deepcopy(semantic)
            for identifier, entity in self.scene_entities.items():
                entity.update(observation_id=observation['observation_id'], geometry_status='missing')
                if entity['kind'] in {'table', 'bin', 'marker'}:
                    continue
                record = self.records.setdefault(identifier, {'id': identifier, 'status': 'observed'})
                record.update({key: deepcopy(entity[key]) for key in
                    ('kind', 'color', 'description', 'visual_membership') if key in entity})
                parent = self.scene_entities.get(entity.get('initial_parent_id'), {})
                record['initial_source'] = {key: deepcopy(entity[key]) for key in
                    ('initial_parent_id', 'identity_source') if key in entity}
                record['initial_source']['initial_parent_kind'] = parent.get('kind')
                self.membership[identifier] = entity.get('visual_membership', 'unknown')
        else:
            self.scene_entities = {}
        for obj in observation['objects']:
            kind, color = obj['kind'], obj.get('color', 'any')
            entity = {key: deepcopy(obj[key]) for key in ('id', 'kind', 'color', 'position') if key in obj}
            entity.update(observation_id=observation['observation_id'], visibility='visible',
                          roles=['container', 'collision'] if kind == 'bin' else
                                ['target_region'] if kind == 'marker' else ['collision'])
            if kind != 'marker':
                prompt = obj.get('segmentation_prompt')
                if not prompt:
                    prompt = 'open box' if kind == 'bin' else kind
                    if color != 'any':
                        prompt = color + ' ' + prompt
                entity['segmentation_query'] = {'kind': kind, 'color': color, 'prompt': prompt}
            if obj.get('placement_reference'):
                entity['region_id'] = 'region:' + obj['id']
            if semantic and obj['id'] in semantic and semantic[obj['id']].get('segmentation_query'):
                entity['segmentation_query'] = deepcopy(semantic[obj['id']]['segmentation_query'])
            self.scene_entities[obj['id']] = {**self.scene_entities.get(obj['id'], {}), **entity,
                                               'geometry_status': 'measured'}
        if observation.get('table'):
            observation['table']['id'] = 'table'
            self.scene_entities['table'] = {**self.scene_entities.get('table', {}), 'id': 'table', 'kind': 'table', 'roles': ['table', 'collision'],
                'observation_id': observation['observation_id'],
                **{key: deepcopy(observation['table'][key]) for key in
                   ('position', 'size', 'surface_height_m') if key in observation['table']}}
        self.scene_inventory_observation_id = observation['observation_id']

    def scene_queries(self):
        queries = {}
        for identifier, entity in self.scene_entities.items():
            query = entity.get('segmentation_query')
            if query is None:
                continue
            key = (query['kind'], query['color'], query['prompt'])
            queries.setdefault(key, {**query, 'entity_ids': []})['entity_ids'].append(identifier)
        return list(queries.values())

    def freeze_inventory(self, observation):
        """Validate the complete measured membership before allocating any number."""
        if self.inventory_finalized:
            self.assign_display_ids(observation)
            return self.snapshot()
        if any(value == 'unknown' for value in self.membership.values()):
            raise ValueError('inventory_membership_unknown')
        identifiers = [key for key, value in self.membership.items() if value == 'required']
        if not identifiers or self.requested_count is not None and len(identifiers) != self.requested_count:
            raise ValueError('inventory_quantity_mismatch')
        visible = {**observation.get('scene_entities', {}), **{obj['id']: obj for obj in observation['objects']}}
        if any(key not in visible for key in identifiers):
            raise ValueError('inventory_required_object_unseen')
        if any(not source_region_confirmed(self.route, self.records[key]) for key in identifiers):
            raise ValueError('task_source_region_unconfirmed')
        assigned = set()
        for group in self.quantity_groups:
            judgments = self.group_membership.get(group['group_id'], {})
            members = {key for key, value in judgments.items() if value == 'required'}
            if any(value == 'unknown' for value in judgments.values()):
                raise ValueError('inventory_group_membership_unknown:' + group['group_id'])
            if len(members) != group['requested_count'] or not members.issubset(identifiers):
                raise ValueError('inventory_group_quantity_mismatch:' + group['group_id'])
            if assigned & members:
                raise ValueError('inventory_group_members_overlap:' + group['group_id'])
            assigned.update(members)
        semantic = observation.get('scene_entities', {})
        if semantic and all(semantic.get(key, {}).get('bbox') for key in identifiers):
            ordered = head_rgb_box_order([semantic[key] for key in identifiers])
        else:
            ordered = head_image_order([visible[key] for key in identifiers], observation.get('views', {}).get('3', {}))
        for number, identifier in enumerate(ordered, 1):
            self.records[identifier]['display_id'] = number
            self.records[identifier].setdefault('initial_source', {key: deepcopy(visible[identifier][key])
                for key in ('position', 'visual_semantics', 'measured_relationships') if key in visible[identifier]})
        self.next_display_id = len(ordered) + 1
        self.inventory_finalized = True
        self.inventory_initialized = True
        self.inventory_revision += 1
        self.ledger_revision += 1
        self.initial_head_observation_id = self.scene_inventory_observation_id or observation['observation_id']
        self.initial_head_reference = {key: deepcopy(value) for key, value in
            observation.get('views', {}).get('3', {}).items() if key in {'intrinsics', 'T_world_camera'}}
        self.initialize_scene_inventory(observation)
        for identifier in identifiers:
            self.scene_entities[identifier]['roles'] = ['transport', 'collision']
        for identifier, entity in self.scene_entities.items():
            number = self.records.get(identifier, {}).get('display_id')
            if number is None:
                number = self.next_display_id
                self.next_display_id += 1
            entity['display_id'] = number
            if identifier in self.records:
                self.records[identifier]['display_id'] = number
        self.assign_display_ids(observation)
        return self.snapshot()

    def assign_display_ids(self, observation):
        """Copy frozen numbers to current detections; never select or renumber members."""
        result = {}
        for obj in observation['objects']:
            identifier = obj['id']
            number = self.scene_entities.get(identifier, {}).get('display_id',
                self.records.get(identifier, {}).get('display_id'))
            if number is None:
                obj.pop('display_id', None)
                continue
            obj['display_id'] = number
            result[identifier] = number
        if observation.get('table') and self.scene_entities.get('table', {}).get('display_id'):
            observation['table']['display_id'] = self.scene_entities['table']['display_id']
        return result

    def rank_candidates(self, observation, *, arm):
        feedback = observation.get('arms', {}).get(arm, {})
        if 'tcp' not in feedback:
            return []
        origin = np.asarray(feedback['tcp'], dtype=float)[:3, 3]
        ranked = []
        for obj in observation['objects']:
            identifier = obj['id']
            record = self.records.get(identifier, {})
            membership = self.membership.get(identifier, 'unknown')
            direct_single_candidate = (self.route == 'direct' and self.scope == 'single'
                                       and self.selection_order == 'explicit' and membership == 'unknown')
            if membership != 'required' and not direct_single_candidate:
                continue
            if self.inventory_initialized and not record:
                continue
            if obj.get('kind') in {'bin', 'marker'} or record.get('status') in {'completed', 'deferred'}:
                continue
            position = measured_position(obj)
            if position is not None:
                ranked.append({'id': identifier, 'display_id': record.get('display_id'),
                               'tcp_distance_m': float(np.linalg.norm(position - origin)), 'arm': arm})
        if self.selection_order == 'inventory':
            return sorted(ranked, key=lambda item: (item.get('display_id') or 10**9, item['id']))
        return sorted(ranked, key=lambda item: (item['tcp_distance_m'], item['id']))

    def select(self, source, destination_id=None):
        identifier = source['id']
        if self.requested_count is not None and len(self.completed_ids) >= self.requested_count:
            raise ValueError('requested_quantity_already_completed')
        if self.inventory_initialized and identifier not in self.records:
            raise ValueError('target_not_in_initial_inventory')
        if identifier in self.records and self.membership.get(identifier, 'unknown') != 'required':
            raise ValueError('target_not_in_task_inventory')
        if self.pending_route_id() not in (None, identifier):
            raise ValueError('pending_route_requires_same_object')
        record = self.records.get(identifier, {})
        if record.get('status') in {'completed'}:
            raise ValueError('target_already_completed')
        if record.get('status') == 'deferred':
            raise ValueError('target_deferred_for_this_run')
        pending = [item for key, item in self.records.items()
                   if self.membership.get(key) == 'required'
                   and item['status'] not in {'completed', 'deferred'}]
        if self.selection_order == 'inventory' and pending and self.pending_route_id() is None:
            first = min(pending, key=lambda item: (item.get('display_id', 10**9), item['id']))
            if first['id'] != identifier:
                raise ValueError('target_violates_inventory_order')
        if not source_region_confirmed(self.route, {**source, **record}):
            raise ValueError('task_source_region_unconfirmed')
        self.membership[identifier] = 'required'
        self.records[identifier] = {**record, **{key: deepcopy(source[key]) for key in
            ('id', 'kind', 'color', 'position', 'visual_semantics', 'measured_relationships') if key in source},
            'status': 'selected', 'destination_id': destination_id,
            'source_observation_id': self.observation_id}
        number = self.scene_entities.get(identifier, {}).get('display_id', source.get('display_id'))
        if number is not None:
            self.records[identifier].setdefault('display_id', number)
        self.records[identifier].setdefault('stage_index', 0)
        self.records[identifier].setdefault('initial_source', {key: deepcopy(source[key]) for key in
            ('position', 'visual_semantics', 'measured_relationships') if key in source})
        self.current_id = identifier
        self.ledger_revision += 1

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
                'action_id': action_id, 'stage_index': self.stage_index(identifier),
                'destination_id': record.get('destination_id'), 'confirmed_observation_id': None,
                **{key: visual[key] for key in ('region_id', 'slot_id') if visual.get(key) is not None}})
            self.ledger_revision += 1

    def confirm_visual_transfer(self, identifier, evidence):
        record = self.records[identifier]
        release = record.get('visual_release', {})
        slot_matches = all(evidence.get(key) == release[key] for key in ('region_id', 'slot_id')
                           if key in release and release.get('slot_id') is not None)
        if release.get('confirmed_observation_id'):
            return (slot_matches and record.get('destination_id') == release['destination_id']
                    and evidence.get('destination_id') == release['destination_id']
                    and bool(evidence.get('observation_id'))
                    and evidence.get('source') == 'openai_decisions'
                    and evidence.get('status') == 'at_destination')
        record['destination_visual_evidence'] = deepcopy(evidence)
        confirmed = (slot_matches and bool(record.get('release_feedback_id'))
                     and release.get('stage_index') == self.stage_index(identifier)
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

    def complete(self, identifier, *, observation_id=None, feedback_id=None):
        record = self.records[identifier]
        stages = record.setdefault('completed_stages', [])
        evidence = {'feedback_id': feedback_id} if feedback_id else {'observation_id': observation_id}
        if not next(iter(evidence.values())):
            raise ValueError('completion_requires_release_evidence')
        key, identifier_value = next(iter(evidence.items()))
        if record.get('status') == 'completed' or any(step.get(key) == identifier_value for step in stages):
            return
        stages.append({'stage_index': self.stage_index(identifier), 'destination_id': record.get('destination_id'),
                       'arm': self.bound_arm(), **evidence})
        if self.route == 'box_table_box' and self.stage_index(identifier) == 0:
            record.update(status='pending', stage_index=1, current_region=record.get('destination_id'))
            self.current_id = None
            self.failure = None
            self.ledger_revision += 1
            return
        if identifier not in self.completed_ids:
            self.completed_ids.append(identifier)
            self.ledger_revision += 1
        record.update(status='completed', **{'confirmed_' + key: identifier_value})
        self.current_id = None
        self.failure = None

    def defer(self, identifier, reason):
        self.records[identifier].update(status='deferred', reason=reason)
        self.ledger_revision += 1
        self.current_id = None
        self.failure = None

    def assess_unseen(self, judgments):
        unseen = set(self.snapshot()['unseen_ids'])
        if any(key not in unseen or value not in {'required', 'excluded', 'unknown'}
               for key, value in judgments.items()):
            raise ValueError('invalid_unseen_relevance_judgment')
        self.relevance.update(judgments)

    def assess_membership(self, judgments):
        for identifier, value in judgments.items():
            record = self.records.get(identifier)
            if (record and record.get('visual_membership') in {'required', 'excluded'}
                    and value != record['visual_membership']):
                raise ValueError('task_membership_conflicts_with_visual_evidence')
            if (record is None or value not in {'required', 'excluded', 'unknown'}
                    or self.inventory_finalized and value != self.membership.get(identifier)
                    or record['status'] in {'selected', 'completed', 'deferred'}
                    and value != self.membership.get(identifier, 'required')):
                raise ValueError('invalid_task_membership')
            if value == 'required' and not source_region_confirmed(self.route, record):
                raise ValueError('task_source_region_unconfirmed')
        if self.requested_count is not None and self.selection_order == 'explicit':
            membership = self.membership | judgments
            required_count = sum(value == 'required'
                                 for key, value in membership.items())
            if required_count > self.requested_count:
                raise ValueError('requested_quantity_membership_ambiguous')
        if any(self.membership.get(key) != value for key, value in judgments.items()):
            self.membership.update(judgments)
            self.ledger_revision += 1

    def assess_groups(self, judgments):
        known = {group['group_id'] for group in self.quantity_groups}
        for group_id, values in judgments.items():
            if group_id not in known or any(key not in self.records or value not in
                    {'required', 'excluded', 'unknown'} for key, value in values.items()):
                raise ValueError('invalid_group_membership')
            if self.inventory_finalized and values != self.group_membership.get(group_id):
                raise ValueError('inventory_is_finalized')
        self.group_membership.update(deepcopy(judgments))

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
                'route': self.route, 'stage_index': self.stage_index(), 'stage_arms': dict(self.stage_arms),
                'pending_route_id': self.pending_route_id(),
                'ledger_revision': self.ledger_revision,
                'inventory_initialized': self.inventory_initialized,
                'inventory_finalized': self.inventory_finalized,
                'inventory_revision': self.inventory_revision,
                'initial_head_observation_id': self.initial_head_observation_id,
                'scene_entities': deepcopy(self.scene_entities),
                'scene_inventory_observation_id': self.scene_inventory_observation_id,
                'placement_regions': deepcopy(list(self.placement_requests.values())),
                'route_destinations': deepcopy(self.route_destinations),
                'quantity_groups': [{**group, 'member_ids': [key for key, value in
                    self.group_membership.get(group['group_id'], {}).items() if value == 'required']}
                    for group in self.quantity_groups],
                'objects': deepcopy(sorted(members.values(), key=lambda item: item.get('display_id', 10**9))),
                'membership': dict(self.membership), 'unresolved_membership': unresolved,
                'membership_pending_ids': [item['id'] for item in unresolved],
                'membership_review_ids': ([key for key, item in self.records.items()
                    if item['status'] == 'observed'] if not self.inventory_finalized else []),
                'current_id': self.current_id, 'current': deepcopy(self.records.get(self.current_id)),
                'completed': completed, 'deferred': deferred,
                'unseen_ids': unseen, 'attempts': dict(self.attempts),
                'unseen_objects': [deepcopy(self.records[key]) for key in unseen],
                'unseen_relevance': {key: self.relevance.get(key, 'unknown') for key in unseen},
                'observation_id': self.observation_id, 'failure': deepcopy(self.failure)}

    def terminal_status(self, judgment):
        state = self.snapshot()
        if (judgment == 'complete' and self.requested_count is not None
                and len(self.completed_ids) != self.requested_count):
            return 'needs_attention'
        if state['unresolved_membership']:
            return 'needs_attention'
        if judgment == 'complete' and self.pending_route_id() is not None:
            return 'needs_attention'
        if judgment == 'complete' and any(item['status'] not in {'completed', 'deferred'}
                                          for item in state['objects']):
            return 'needs_attention'
        unresolved = [key for key in state['unseen_ids'] if self.relevance.get(key) != 'excluded']
        if judgment == 'unknown' or unresolved:
            return 'needs_attention'
        if judgment == 'partial' or state['deferred']:
            return 'partial_success' if state['completed'] else 'failed'
        return 'succeeded' if judgment == 'complete' else 'failed'

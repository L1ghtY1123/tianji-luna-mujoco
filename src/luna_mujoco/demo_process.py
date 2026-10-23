"""Separate immutable source entities, unpack processes and visible products."""
from copy import deepcopy
from uuid import uuid4

from .task_progress import TaskProgress


def initialize_processes(task, entities):
    if task.get('process_records') is not None:
        return
    task.update(process_records={}, process_order=[], product_entities={})
    for ordinal, identifier in enumerate(task['transport_ids'], 1):
        process_id = 'process-' + uuid4().hex
        task['process_order'].append(process_id)
        task['process_records'][process_id] = {'id': process_id, 'process_id': process_id,
            'display_id': ordinal, 'input_entity_id': identifier,
            'output_entity_id': None, 'status': 'pending', 'ordinal': ordinal,
            'input_display_id': entities.get(identifier, {}).get('display_id'),
            'relationship': 'unpack_process_input_and_output_not_same_entity'}


def process_for_input(task, identifier):
    return next(row for row in task['process_records'].values() if row['input_entity_id'] == identifier)


def current_process_entity(task, input_id):
    if not task.get('process_records'):
        return input_id
    row = process_for_input(task, input_id)
    return row['output_entity_id'] or input_id


def process_snapshot(task):
    rows = deepcopy(task.get('process_records', {}))
    for row in rows.values():
        lifecycle = task['item_lifecycle'][row['output_entity_id'] or row['input_entity_id']]
        phase = lifecycle['phase']
        row.update(phase=phase, to_table_arm=lifecycle['to_table_arm'], to_right_bin_arm=lifecycle['to_right_bin_arm'])
        row['status'] = 'completed' if phase == 'completed' else 'pending' if phase == 'wrapped_source' else 'processing'
    return {'process_records': rows, 'process_order': list(task.get('process_order', [])),
        'pending_unpack_ids': [rows[key]['input_entity_id'] for key in task.get('process_order', [])
                               if rows[key]['output_entity_id'] is None],
        'not_started_input_ids': [rows[key]['input_entity_id'] for key in task.get('process_order', [])
                               if rows[key]['status'] == 'pending'],
        'pending_delivery_product_ids': [row['output_entity_id'] for row in rows.values()
                                        if row['output_entity_id'] and row['status'] != 'completed'],
        'delivered_product_ids': [row['output_entity_id'] for row in rows.values()
                                 if row['output_entity_id'] and row['status'] == 'completed'],
        'processing_ids': [key for key, row in rows.items() if row['status'] == 'processing'],
        'completed_process_ids': [key for key, row in rows.items() if row['status'] == 'completed'],
        'product_entities': deepcopy(task.get('product_entities', {})),
        'product_ids': list(task.get('product_entities', {}))}


def inventory_views(entities):
    """Derived views only; regions and historical identities are not obstacles."""
    result = {'current_entities': {}, 'historical_entities': {}, 'target_regions': {}}
    for identifier, entity in entities.items():
        if entity.get('kind') == 'marker' or 'target_region' in entity.get('roles', []):
            name = 'target_regions'
        elif entity.get('geometry_status', '').startswith('historical'):
            name = 'historical_entities'
        else:
            name = 'current_entities'
        result[name][identifier] = deepcopy(entity)
    return result


def completed_input_ids(task):
    return {row['input_entity_id'] for row in process_snapshot(task)['process_records'].values()
            if row['status'] == 'completed'}


def completed_product_collection(task):
    members = [key for key in task.get('product_entities', {})
               if task['item_lifecycle'][key]['phase'] == 'completed']
    if not members:
        return None
    identifier = 'product-collection:' + task['right_bin_id']
    return {'id': identifier, 'kind': 'collection', 'color': 'black', 'members': sorted(members),
        'description': 'one or more visible black workpieces (or their visible parts) inside the right destination bin',
        'geometry_source': 'current_segmented_product_collection', 'identity_assignment': False,
        'visual_membership': 'excluded', 'roles': ['collision', 'product_collection'],
        'lifecycle_phase': 'completed_collection', 'current_appearance': 'unpacked',
        'appearance_revision': 1, 'segmentation_query': {'kind': 'object', 'color': 'black', 'prompt': 'black object'}}


def create_product(task, input_id, result, capture, *, owner_id):
    """Mint once after visible-result confirmation; never rewrite the input ID."""
    process = process_for_input(task, input_id)
    source = task['item_lifecycle'][input_id]
    visual = result.get('visual_evidence', {})
    if (not owner_id or result.get('success') is not True or result.get('status') != 'confirmed'
            or result.get('target_id') != input_id or not result.get('request_hash')
            or result.get('observation_id') != capture.get('observation_id')
            or result.get('generation') != capture.get('generation')
            or visual.get('judgments') != {'unpack_result': 'yes'}
            or visual.get('view_scope') != 'fused_current_three_rgb'
            or type(visual.get('confidence')) not in (int, float) or not .8 <= visual['confidence'] <= 1.):
        raise ValueError('product_requires_confirmed_visible_result')
    if process['output_entity_id']:
        product = task['product_entities'][process['output_entity_id']]
        if product['creation_evidence']['request_hash'] != result['request_hash']:
            raise ValueError('product_already_bound_to_another_observation')
        return deepcopy(product)
    if source['phase'] != 'unpack_pending':
        raise ValueError('product_requires_pending_process')
    identifier = 'product-' + uuid4().hex
    evidence = {'source': 'confirmed_unpack_process_output', 'process_id': process['id'],
        'input_entity_id': input_id, 'output_entity_id': identifier, 'owner_id': owner_id,
        'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'request_hash': result['request_hash'], 'action_id': source.get('mock_action', {}).get('action_id'),
        'relationship': process['relationship']}
    product = {'id': identifier, 'kind': 'object', 'color': 'black', 'asset_id': 'black_core',
        'description': 'separate black unpacked product on the main working table',
        'display_label': 'P' + str(process['ordinal']), 'input_display_id': process['input_display_id'],
        'process_id': process['id'], 'input_entity_id': input_id,
        'identity_source': 'confirmed_current_product_creation', 'creation_evidence': evidence,
        'visual_membership': 'required', 'roles': ['product', 'collision'],
        'segmentation_query': {'kind': 'object', 'color': 'black', 'prompt': 'black object'}}
    product_record = deepcopy(source)
    product_record.update(phase='unpacked_confirmed', appearance='unpacked', appearance_revision=1,
        process_id=process['id'], input_entity_id=input_id, product_creation=deepcopy(evidence),
        unpacking_evidence=deepcopy(result))
    task['item_lifecycle'][identifier] = product_record
    task['product_entities'][identifier] = deepcopy(product)
    source.update(phase='consumed', output_entity_id=identifier, process_id=process['id'])
    process.update(output_entity_id=identifier, status='processing', creation_evidence=deepcopy(evidence))
    return product


class ProcessTaskProgress(TaskProgress):
    """Entity ledger with explicit derived-product registration, not ID aliases."""
    process_task = None

    def freeze_inventory(self, observation):
        already_frozen = self.inventory_finalized
        super().freeze_inventory(observation)
        if not already_frozen:
            source_ids = observation.get('demo_task', {}).get('source_object_ids', [])
            # The entire source catalog is numbered once, including count-excluded
            # pink subjects. Non-source entities follow those immutable ordinals.
            numbers = {key: number for number, key in enumerate(source_ids, 1)}
            next_number = len(numbers) + 1
            for identifier in self.scene_entities:
                if identifier not in numbers:
                    numbers[identifier] = next_number
                    next_number += 1
            for identifier, number in numbers.items():
                if identifier in self.scene_entities:
                    self.scene_entities[identifier]['display_id'] = number
                if identifier in self.records:
                    self.records[identifier]['display_id'] = number
            self.next_display_id = next_number
            self.assign_display_ids(observation)
            self.initial_entities = deepcopy(inventory_views(self.scene_entities)['current_entities'])
        return self.snapshot()

    def register_product(self, product):
        identifier, source_id = product['id'], product['input_entity_id']
        if identifier in self.records:
            return
        source = self.records[source_id]
        if source.get('stage_index') != 1 or not source.get('destination_visual_evidence'):
            raise ValueError('product_ledger_requires_confirmed_input_release')
        source.update(status='consumed', output_entity_id=identifier)
        self.membership[source_id] = 'excluded'
        self.scene_entities[source_id].update(visual_membership='excluded', geometry_status='historical_consumed',
                                               roles=['process_input'])
        self.scene_entities[source_id].pop('segmentation_query', None)
        self.records[identifier] = {**deepcopy(product), 'status': 'pending', 'stage_index': 1,
            'display_id': self.next_display_id,
            'initial_source': {'identity_source': 'confirmed_current_product_creation',
                               'initial_parent_kind': 'table', 'process_id': product['process_id']}}
        self.next_display_id += 1
        self.scene_entities[identifier] = {**deepcopy(product), 'display_id': self.records[identifier]['display_id']}
        if self.process_task is not None:
            self.process_task['product_entities'][identifier]['display_id'] = self.records[identifier]['display_id']
        self.membership[identifier] = 'required'
        self.current_id = None
        self.ledger_revision += 1

    def sync_process_geometry(self):
        """Keep historical entities in the ledger, not in current collision ownership."""
        task = self.process_task
        if task is None:
            return
        for identifier, lifecycle in task['item_lifecycle'].items():
            if lifecycle['phase'] == 'consumed' or (identifier in task.get('product_entities', {})
                                                    and lifecycle['phase'] == 'completed'):
                entity = self.scene_entities.get(identifier)
                if entity is not None:
                    entity.update(roles=['process_input' if lifecycle['phase'] == 'consumed' else 'completed_product'],
                                  geometry_status='historical', visual_membership='excluded')
                    entity.pop('segmentation_query', None)
        collection = completed_product_collection(task)
        if collection:
            self.scene_entities[collection['id']] = collection

    def pending_route_id(self):
        return next((key for key, row in self.records.items() if row.get('stage_index') == 1
                     and row['status'] not in {'completed', 'deferred', 'consumed'}), None)

    def select(self, source, destination_id=None):
        if source['id'] not in (self.process_task or {}).get('product_entities', {}):
            return super().select(source, destination_id)
        # Products originate on the table; do not counterfeit a source-bin identity.
        route = self.route
        try:
            self.route = 'direct'
            return super().select(source, destination_id)
        finally:
            self.route = route

    def snapshot(self):
        result = super().snapshot()
        if self.process_task is not None:
            result.update(process_snapshot(self.process_task))
            result['all_entities'] = deepcopy(self.scene_entities)
            result['initial_entities'] = deepcopy(getattr(self, 'initial_entities', {}))
            result.update(inventory_views(self.scene_entities))
            identifiers = set(self.process_task['transport_ids']) | set(self.process_task['product_entities'])
            result['objects'] = deepcopy(sorted((row for key, row in self.records.items() if key in identifiers),
                                                key=lambda row: row.get('display_id', 10**9)))
        return result

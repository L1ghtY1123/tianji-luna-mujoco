"""Permanent entity identities, verified transfers and capability effects."""
from collections import Counter
from copy import deepcopy

from ..task_progress import TaskProgress


class PolicyTaskProgress(TaskProgress):
    """The graph selects actions; the ledger records facts without a route policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = []
        self.task_spec = None
        self.object_tasks = {}
        self.transformations = {}
        self.initial_entities = {}

    def configure_task(self, task_spec, selected_ids, *, object_tasks=None):
        self.task_spec = task_spec
        self.object_tasks = dict(object_tasks or {})
        self.input_ids = list(selected_ids)
        self.initial_entities = deepcopy(self.scene_entities)

    def assess_observed_membership(self, judgments):
        """Keep sensing proposals separate from the graph's task confirmation."""
        for identifier, value in judgments.items():
            if identifier not in self.records or value not in {'required', 'excluded', 'unknown'}:
                raise ValueError('invalid_observed_membership')
        for identifier, value in judgments.items():
            record = self.records[identifier]
            if record.get('candidate_membership') != value:
                record['candidate_membership'] = value
                self.ledger_revision += 1

    def transfer_index(self, identifier=None):
        # Placement layouts need an action index, not an enumerated task phase.
        return self.records.get(identifier or self.current_id, {}).get('transfer_index', 0)

    def bind_arm(self, arm):
        if arm not in {'A', 'B'} or self.current_id not in self.records:
            raise ValueError('invalid_execution_arm')
        if self.requested_arm and self.requested_arm != arm:
            raise ValueError('requested_arm_cannot_switch')
        self.records[self.current_id]['execution_arm'] = arm

    def bound_arm(self):
        return self.records.get(self.current_id, {}).get('execution_arm')

    def select(self, source, destination_id=None):
        identifier = source['id']
        record = self.records.get(identifier)
        if record is None or self.membership.get(identifier) != 'required':
            raise ValueError('target_not_in_task_inventory')
        if record.get('status') in {'consumed', 'completed', 'deferred'}:
            raise ValueError('target_not_pending:' + identifier)
        if self.selection_order == 'inventory':
            completed = self.completed_input_ids()
            first = next((key for key in self.input_ids if key not in completed), None)
            if first is not None and record.get('root_input_id', identifier) != first:
                raise ValueError('target_violates_inventory_order')
        record.pop('execution_arm', None)
        # A new transfer has a new destination and action index. Old release
        # correspondence remains historical; it must not constrain the new one.
        record.pop('completed_identity_track', None)
        record.update(status='selected', destination_id=destination_id,
                      transfer_index=len(record.get('transfers', [])),
                      source_observation_id=self.observation_id)
        self.current_id = identifier
        self.ledger_revision += 1

    def complete(self, identifier, *, observation_id=None, feedback_id=None):
        if not observation_id:
            raise ValueError('completion_requires_visual_destination_evidence')
        record = self.records[identifier]
        transfers = record.setdefault('transfers', [])
        if any(row['observation_id'] == observation_id for row in transfers):
            return
        transfers.append({'destination_id': record['destination_id'],
                          'destination_ref': record.get('destination_ref', record['destination_id']),
                          'transfer_index': self.transfer_index(identifier), 'arm': record.get('execution_arm'),
                          'observation_id': observation_id, 'release_feedback_id': record.get('release_feedback_id')})
        record.update(current_region=record.get('destination_ref', record['destination_id']), status='observed',
                      confirmed_observation_id=observation_id)
        if identifier in self.scene_entities:
            self.scene_entities[identifier]['current_region'] = record['current_region']
        if self._entity_goal_satisfied(identifier):
            record['status'] = 'completed'
            if identifier not in self.completed_ids:
                self.completed_ids.append(identifier)
        self.current_id = None
        self.failure = None
        self.ledger_revision += 1

    def _entity_goal_satisfied(self, identifier):
        if self.task_spec is None:
            return False
        record = self.records[identifier]
        actual = Counter(record.get('processing_history', []))
        task = self.object_tasks.get(record.get('root_input_id', identifier), self.task_spec)
        required = Counter(task.processing)
        return (record.get('current_region') == task.destination_ref
                and bool(record.get('confirmed_observation_id'))
                and all(actual[name] >= count for name, count in required.items()))

    def completed_input_ids(self):
        result = set()
        for identifier in self.input_ids:
            current = identifier
            visited = set()
            while current in self.records and self.records[current].get('output_entity_id'):
                if current in visited:
                    raise ValueError('cyclic_entity_lineage')
                visited.add(current)
                current = self.records[current]['output_entity_id']
            if current in self.records and self._entity_goal_satisfied(current):
                result.add(identifier)
        return result

    def register_product(self, product):
        identifier, source_id = product['id'], product['input_entity_id']
        source = self.records[source_id]
        if identifier == source_id:
            raise ValueError('processing_output_requires_distinct_identity')
        if identifier in self.records:
            existing = self.records[identifier]
            if all(existing.get(key) == product.get(key) for key in
                   ('creation_evidence', 'input_entity_id', 'process_id', 'processing_skill')):
                return
            raise ValueError('product_identity_already_bound')
        if source.get('output_entity_id') or source.get('status') in {'consumed', 'completed'}:
            raise ValueError('processing_input_already_consumed')
        evidence = product.get('creation_evidence', {})
        if not all(evidence.get(key) for key in ('observation_id', 'request_hash', 'action_id')):
            raise ValueError('product_requires_confirmed_creation_evidence')
        skill = product['processing_skill']
        self.records[identifier] = {**deepcopy(product), 'status': 'observed',
            'display_id': self.next_display_id,
            'execution_rank': source.get('execution_rank', source.get('display_id')),
            'root_input_id': source.get('root_input_id', source_id),
            'current_region': source.get('current_region'),
            'processing_history': [*source.get('processing_history', []), skill],
            'initial_source': {'identity_source': 'confirmed_process_output',
                               'measurement_observation_id': evidence['observation_id']}}
        self.next_display_id += 1
        self.scene_entities[identifier] = deepcopy(self.records[identifier])
        self.scene_entities[identifier].update(roles=['product', 'collision'], visual_membership='required')
        source.update(status='consumed', output_entity_id=identifier)
        self.scene_entities[source_id].update(geometry_status='historical_consumed', roles=['process_input'],
                                              visual_membership='excluded')
        self.scene_entities[source_id].pop('segmentation_query', None)
        self.membership[source_id] = 'excluded'
        self.membership[identifier] = 'required'
        self.transformations[product['process_id']] = {
            'id': product['process_id'], 'skill': skill, 'input_entity_id': source_id,
            'output_entity_id': identifier, 'evidence': deepcopy(evidence)}
        self.current_id = None
        self.ledger_revision += 1

    def inventory_state(self):
        return {'inventory_finalized': self.inventory_finalized,
            'entities': deepcopy(self.scene_entities),
            'retired_entity_ids': [key for key, row in self.records.items() if row.get('status') == 'consumed'],
            'transformations': deepcopy(self.transformations)}

    def snapshot(self):
        result = super().snapshot()
        result.update(self.inventory_state(), initial_entities=deepcopy(self.initial_entities),
            object_tasks={key: value.model_dump(mode='json') for key, value in self.object_tasks.items()},
            completed_input_ids=sorted(self.completed_input_ids()), selected_input_ids=list(self.input_ids),
            objects=deepcopy(list(self.records.values())),
            product_ids=[key for key, row in self.records.items() if row.get('input_entity_id')])
        return result

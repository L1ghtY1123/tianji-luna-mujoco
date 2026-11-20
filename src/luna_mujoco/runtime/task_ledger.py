"""Task-local obligations, verified transfers and processing receipts."""
from collections import Counter
from copy import deepcopy

from ..core.placement import ReleaseLedgerSnapshot
from ..task_progress import TaskProgress


class PolicyTaskProgress(TaskProgress):
    """The graph selects actions; the ledger records facts without a route policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = []
        self.task_spec = None
        self.object_tasks = {}

    def configure_task(self, task_spec, selected_ids, *, object_tasks=None):
        self.task_spec = task_spec
        self.selection_order = task_spec.selection_order
        self.object_tasks = dict(object_tasks or {})
        self.input_ids = list(selected_ids)

    def claim_input(self, identifier):
        """Record responsibility for an input only when the graph selects it."""
        record = self.records[identifier]
        root = record.get('root_input_id', identifier)
        if root in self.input_ids:
            return
        if self.task_spec.scope == 'ids' and root not in self.task_spec.target_refs:
            raise ValueError('target_outside_requested_ids:' + root)
        if self.task_spec.quantity is not None and len(self.input_ids) >= self.task_spec.quantity:
            raise ValueError('requested_quantity_already_selected')
        self.input_ids.append(root)
        record['root_input_id'] = root
        self.ledger_revision += 1

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
        if self.bound_arm() not in (None, arm):
            raise ValueError('single_arm_transfer_cannot_switch_arm')
        self.records[self.current_id]['execution_arm'] = arm

    def bound_arm(self):
        return self.records.get(self.current_id, {}).get('execution_arm')

    def release_snapshot(self, target_id, destination_id):
        return ReleaseLedgerSnapshot(target_id=target_id, current_id=self.current_id,
            completed_ids=tuple(self.completed_ids), record={key: deepcopy(self.records[target_id].get(key))
                for key in ('id', 'status', 'release_feedback_id')},
            bound_arm=self.bound_arm(), placement_request=deepcopy(self.placement_request(destination_id)),
            visual_progress=deepcopy(self.visual_progress()[target_id]), transfer_index=self.transfer_index(target_id))

    def select(self, source, destination_id=None):
        identifier = source['id']
        record = self.records.get(identifier)
        if record is None or self.membership.get(identifier) != 'required':
            raise ValueError('target_not_in_task_bindings')
        if record.get('status') in {'consumed', 'completed', 'deferred'}:
            raise ValueError('target_not_pending:' + identifier)
        self.claim_input(identifier)
        if record.get('transfer_index') != len(record.get('transfers', [])):
            record.pop('execution_arm', None)
        # A new transfer has a new destination and action index. Old release
        # correspondence remains historical; it must not constrain the new one.
        record.update(status='selected', destination_id=destination_id,
                      transfer_index=len(record.get('transfers', [])),
                      source_observation_id=self.observation_id)
        self.current_id = identifier
        self.ledger_revision += 1

    def complete(self, identifier, *, observation_id=None, feedback_id=None, container_entry=None):
        if not observation_id:
            raise ValueError('completion_requires_visual_destination_evidence')
        record = self.records[identifier]
        transfers = record.setdefault('transfers', [])
        if any(row['observation_id'] == observation_id for row in transfers):
            return
        transfer = {'destination_id': record['destination_id'],
                          'destination_ref': record.get('destination_ref', record['destination_id']),
                          'transfer_index': self.transfer_index(identifier), 'arm': record.get('execution_arm'),
                          'observation_id': observation_id, 'release_feedback_id': record.get('release_feedback_id')}
        if container_entry is not None:
            transfer.update(source=container_entry['source'], receipt_id=container_entry['receipt_id'])
            record['container_entry'] = deepcopy(container_entry)
            record['confirmed_observation_id'] = None
        else:
            record['confirmed_observation_id'] = observation_id
        transfers.append(transfer)
        record.update(current_region=record.get('destination_ref', record['destination_id']), status='observed')
        if self._entity_goal_satisfied(identifier):
            record['status'] = 'completed'
            if identifier not in self.completed_ids:
                self.completed_ids.append(identifier)
        self.current_id = None
        self.failure = None
        self.ledger_revision += 1

    def confirm_container_transfer(self, identifier, evidence):
        """Record the target container from the bound release receipt."""
        from ..demo_transfer_verification import valid_container_entry

        record = self.records[identifier]
        if not valid_container_entry(evidence, identifier, record.get('destination_id'),
                                     record.get('release_feedback_id')):
            return False
        previous = record.get('container_entry')
        if previous and previous.get('release_feedback_id') == evidence['release_feedback_id']:
            return previous == evidence
        release = record.get('visual_release', {})
        if (release.get('transfer_index') != self.transfer_index(identifier)
                or release.get('destination_id') != record.get('destination_id')):
            return False
        self.complete(identifier, observation_id=evidence['observation_id'], container_entry=evidence)
        record['transport_status'] = evidence['placement_outcome']
        return True

    def _entity_goal_satisfied(self, identifier):
        if self.task_spec is None:
            return False
        record = self.records[identifier]
        actual = Counter(record.get('processing_history', []))
        task = self.object_tasks.get(record.get('root_input_id', identifier), self.task_spec)
        required = Counter(task.processing)
        return (record.get('current_region') == task.destination_ref
                and bool(record.get('confirmed_observation_id') or record.get('container_entry'))
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
            'execution_rank': source.get('execution_rank'),
            'root_input_id': source.get('root_input_id', source_id),
            'current_region': product.get('current_region') or product.get('observed_support', {}).get('support_entity_id'),
            'processing_history': [*source.get('processing_history', []), skill],
            'initial_source': {'identity_source': 'confirmed_process_output',
                               'measurement_observation_id': evidence['observation_id']}}
        source.update(status='consumed', output_entity_id=identifier)
        self.membership[source_id] = 'excluded'
        self.membership[identifier] = 'required'
        self.current_id = None
        self.ledger_revision += 1

    def snapshot(self):
        result = super().snapshot()
        result.update(
            object_tasks={key: value.model_dump(mode='json') for key, value in self.object_tasks.items()},
            completed_input_ids=sorted(self.completed_input_ids()), selected_input_ids=list(self.input_ids),
            objects=deepcopy(list(self.records.values())),
            product_ids=[key for key, row in self.records.items() if row.get('input_entity_id')])
        return result

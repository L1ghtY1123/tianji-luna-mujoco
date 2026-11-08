"""Instance memory with typed updates and detached reads. No task or graph dependency."""
from copy import deepcopy

from .memory_contracts import ObjectObservation, ObjectOperation, ObjectRecord, ObjectRetirement


def restored_object_record(entity):
    """Project stored legacy records at the storage boundary, including pre-split data."""
    fields = {key: value for key, value in entity.items() if key in ObjectRecord.model_fields}
    fields['status'] = entity.get('status') if entity.get('status') in {'consumed', 'removed'} else 'active'
    return ObjectRecord.model_validate(fields)


class ObjectMemory:
    def __init__(self):
        self._entities: dict[str, ObjectRecord] = {}
        self.revision = 0
        self.observation_id: str | None = None

    def restore(self, records: tuple[ObjectRecord, ...]):
        proposed = {record.id: record.model_copy(deep=True) for record in records}
        if len(proposed) != len(records):
            raise ValueError('duplicate_memory_instance')
        self._entities.update(proposed)
        self.revision += bool(proposed)

    def entities(self):
        return {key: row.model_dump(mode='json', exclude_none=True) for key, row in self._entities.items()}

    def update(self, record: ObjectRecord):
        if record.id not in self._entities:
            raise ValueError('memory_instance_not_registered:' + record.id)
        self._entities[record.id] = record.model_copy(deep=True)
        self.revision += 1

    def observe(self, update: ObjectObservation):
        incoming = {record.id: record for record in update.entities}
        if any(identifier in self._entities and self._entities[identifier].status != 'active'
               for identifier in update.visible_ids):
            raise ValueError('retired_instance_cannot_be_observed_as_active')
        proposed = dict(self._entities)
        for identifier, record in incoming.items():
            previous = proposed.get(identifier)
            if previous is None:
                proposed[identifier] = record.model_copy(deep=True)
            else:
                changes = record.model_dump(exclude_unset=True)
                changes.pop('status', None)
                proposed[identifier] = ObjectRecord.model_validate(previous.model_dump() | changes)
        for identifier, record in proposed.items():
            if record.status != 'active':
                continue
            changes = {'visibility': 'visible' if identifier in update.visible_ids else 'unseen'}
            if identifier in update.visible_ids:
                changes['observation_id'] = update.observation_id
            if identifier in update.positions:
                changes['last_position_measurement'] = update.positions[identifier].model_dump()
            proposed[identifier] = ObjectRecord.model_validate(record.model_dump() | changes)
        self._entities = proposed
        self.observation_id = update.observation_id
        self.revision += 1

    def retire(self, receipt: ObjectRetirement):
        record = self._entities[receipt.entity_id]
        if record.status == 'consumed' and receipt.reason != 'consumed':
            raise ValueError('consumed_instance_cannot_be_reactivated')
        self.update(ObjectRecord.model_validate(record.model_dump() | {
            'status': receipt.reason, 'retirement': receipt.model_dump(),
            'visibility': 'unseen', 'current_region': None, 'observed_support': None}))

    def record_operation(self, operation: ObjectOperation):
        record = self._entities[operation.entity_id]
        changes = operation.model_dump(exclude_unset=True, exclude={
            'entity_id', 'event_key', 'kind', 'evidence', 'transfers', 'observation_id'})
        transfers = list(record.transfers)
        seen = {row['observation_id']: row for row in transfers}
        for transfer in operation.transfers:
            previous = seen.get(transfer['observation_id'])
            if previous is not None and previous != transfer:
                raise ValueError('conflicting_transfer_receipt')
            if transfer['observation_id'] not in seen:
                transfers.append(deepcopy(transfer))
                seen[transfer['observation_id']] = transfer
        changes['transfers'] = transfers
        self.update(ObjectRecord.model_validate(record.model_dump() | changes))

    def register_product(self, product: ObjectRecord):
        source = self._entities[product.input_entity_id]
        if product.id == source.id:
            raise ValueError('processing_output_requires_distinct_identity')
        if product.id in self._entities:
            if self._entities[product.id] == product and source.output_entity_id == product.id:
                return
            raise ValueError('product_identity_already_bound')
        evidence = product.creation_evidence or {}
        if not product.process_id or not product.processing_skill or not all(
                evidence.get(key) for key in ('observation_id', 'request_hash', 'action_id')):
            raise ValueError('product_requires_confirmed_creation_evidence')
        if source.status != 'active' or source.output_entity_id:
            raise ValueError('processing_input_already_consumed')
        retired = ObjectRecord.model_validate(source.model_dump() | {
            'status': 'consumed', 'output_entity_id': product.id,
            'visibility': 'unseen', 'current_region': None, 'observed_support': None,
            'retirement': ObjectRetirement(entity_id=source.id, reason='consumed',
                source='openai_decisions', observation_id=evidence['observation_id'],
                request_hash=evidence['request_hash']).model_dump(),
            'roles': ['process_input']})
        self._entities.update({source.id: retired, product.id: product.model_copy(deep=True)})
        self.revision += 1

    def snapshot(self):
        entities = self.entities()
        fields = ('status', 'retirement', 'current_region', 'confirmed_observation_id', 'transport_status',
                  'last_motion_receipt',
                  'visual_release', 'container_entry', 'transfers', 'processing_history', 'input_entity_id',
                  'output_entity_id', 'observed_support', 'creation_evidence')
        transformations = {}
        for entity in entities.values():
            if entity.get('process_id') and entity.get('input_entity_id'):
                transformations[entity['process_id']] = {
                    'id': entity['process_id'], 'skill': entity['processing_skill'],
                    'input_entity_id': entity['input_entity_id'], 'output_entity_id': entity['id'],
                    'evidence': deepcopy(entity.get('creation_evidence', {}))}
        return {'entities': entities,
                'object_history': {key: {field: deepcopy(row[field]) for field in fields if field in row}
                                   for key, row in entities.items()},
                'retired_entity_ids': [key for key, row in entities.items() if row['status'] in {'consumed', 'removed'}],
                'transformations': transformations, 'inventory_revision': self.revision}

    def scene_queries(self):
        queries = {}
        for entity in self.entities().values():
            query = entity.get('segmentation_query')
            if query is None or entity['status'] != 'active':
                continue
            key = (query['kind'], query['color'], query['prompt'])
            queries.setdefault(key, {**query, 'entity_ids': []})['entity_ids'].append(entity['id'])
        return list(queries.values())

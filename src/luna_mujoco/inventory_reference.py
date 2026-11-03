"""Read current entity identity and task bindings without exposing execution truth."""
import hashlib
import json
from copy import deepcopy


def bound_identity_coverage_source(observation):
    """Validate the provenance union without relabeling geometry as a model answer."""
    visual = observation.get('visual_inventory', {})
    coverage = visual.get('identity_coverage', {})
    if coverage.get('source') == 'openai_decisions':
        return True
    if coverage.get('source') != 'mixed_decisions_measured_container':
        return False
    proof = visual.get('container_correspondence', {})
    try:
        unsigned = {key: value for key, value in proof.items() if key != 'binding_hash'}
        digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
        mapping = proof['candidate_mapping']
        entities = observation.get('scene_entities') or {item['id']: item for item in observation['objects']}
        return bool(mapping) and (
            proof == visual.get('evidence', {}).get('container_correspondence')
            and proof.get('source') == 'measured_container_correspondence'
            and proof.get('status') == 'matched'
            and proof.get('observation_id') == observation['observation_id']
            and proof.get('generation') == observation['generation']
            and proof.get('binding_hash') == coverage.get('container_correspondence_binding_hash') == digest
            and set(mapping.values()) == set(coverage.get('measured_container_entity_ids', []))
            and all(entities[identifier]['kind'] == 'bin'
                    and proof['matches'][index]['entity_id'] == identifier for index, identifier in mapping.items()))
    except (KeyError, TypeError, ValueError):
        return False


def current_inventory_entities(reference, fallback=None):
    reference = reference or {}
    state = reference.get('inventory_state', {})
    entities = state.get('entities', fallback if fallback is not None else reference.get('entities', {}))
    retired = set(state.get('retired_entity_ids', ()))
    return {identifier: deepcopy(entity) for identifier, entity in entities.items() if identifier not in retired}


def source_inventory(reference):
    reference = reference or {}
    task = reference.get('task_spec', {})
    bindings = reference.get('task_inventory', {})
    return {'source_id': bindings.get('source_id', task.get('source_ref')),
            'source_object_ids': list(bindings.get('source_object_ids', task.get('target_refs', ())))}


def visual_inventory_state(reference):
    """Only identity/lineage facts enter Decisions; physical receipts remain local."""
    state = (reference or {}).get('inventory_state', {})
    fields = ('id', 'input_entity_id', 'output_entity_id', 'process_id', 'processing_skill', 'skill', 'status')
    transformations = state.get('transformations', {})
    return {'retired_entity_ids': list(state.get('retired_entity_ids', ())),
            'transformations': {identifier: {key: deepcopy(record[key]) for key in fields if key in record}
                                for identifier, record in transformations.items()}}

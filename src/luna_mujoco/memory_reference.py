"""Read permanent entity identity and dated physical history."""
from copy import deepcopy

IDENTITY_ARCHIVE_LIMIT = 3


def select_identity_history(history, entity):
    """Bound dated references by recent evidence, another view and original identity."""
    seen = {entity.get('reference_observation_id')}
    available = []
    for event in reversed(history):
        state = event.get('state', {})
        observation_id = state.get('reference_observation_id')
        confirmed = (event.get('kind') in {'instance_registered', 'processing_result'}
                     or event.get('kind') == 'observe_scene' and event.get('evidence', {}).get('status') == 'success')
        if (not confirmed or not observation_id or observation_id in seen
                or not state.get('identity_request_hash') or not state.get('bbox')
                or state.get('world_key') != entity.get('world_key')):
            continue
        seen.add(observation_id)
        available.append(state)
    if len(available) <= IDENTITY_ARCHIVE_LIMIT:
        return available, len(available)
    selected = [available[0], available[-1]]
    cameras = {row.get('reference_camera_id') for row in selected}
    alternate = next((row for row in available[1:-1] if row.get('reference_camera_id') not in cameras),
                     available[1])
    selected.insert(1, alternate)
    return selected, len(available)


def bound_identity_coverage_source(observation):
    """Validate the provenance union without relabeling geometry as a model answer."""
    visual = observation.get('visual_identity', {})
    coverage = visual.get('identity_coverage', {})
    if (coverage.get('source') == 'current_segmentation'
            and not coverage.get('accepted_entity_ids')
            and visual.get('evidence', {}).get('identity_derivation') == 'no_current_identity_candidates'
            and not visual.get('evidence', {}).get('questions')):
        return True
    return coverage.get('source') == 'openai_decisions'


def current_memory_entities(reference, fallback=None):
    reference = reference or {}
    state = reference.get('object_memory_snapshot', {})
    entities = state.get('entities', fallback if fallback is not None else reference.get('entities', {}))
    retired = set(state.get('retired_entity_ids', ()))
    return {identifier: deepcopy(entity) for identifier, entity in entities.items()
            if identifier not in retired and entity.get('status', 'active') == 'active'}


def visual_identity_state(reference):
    """Only identity/lineage facts enter Decisions; physical receipts remain local."""
    state = (reference or {}).get('object_memory_snapshot', {})
    fields = ('id', 'input_entity_id', 'output_entity_id', 'process_id', 'processing_skill', 'skill', 'status')
    transformations = state.get('transformations', {})
    return {'retired_entity_ids': list(state.get('retired_entity_ids', ())),
            'object_history': deepcopy(state.get('object_history', {})),
            'transformations': {identifier: {key: deepcopy(record[key]) for key in fields if key in record}
                                for identifier, record in transformations.items()}}

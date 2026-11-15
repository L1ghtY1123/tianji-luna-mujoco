"""Bounded presentation of verified history; Decisions still judges completion."""
from copy import deepcopy


def source_presence_request():
    from ..core.decisions import JudgmentRequest

    return JudgmentRequest(name='source_has_remaining_input', instructions=(
        'Do the CURRENT images show any matching input object still in this source? '
        'Answer true when one remains, false when none remain. Inspect current visible contents, '
        'including additional matching inputs outside the selected IDs. Do not infer contents '
        'from task instructions, initial reference images or past processing records. '
        'If visibility is insufficient or membership ambiguous, remain uncertain. '
        'Non-target objects do not count. This question does not assess processing, '
        'CAD, product delivery or robot posture.'))


def completion_expected_status(name):
    """Remaining input is a presence question: completion requires a current no."""
    return 'no' if name == 'source_has_remaining_input' else 'yes'


def completion_failures(judgments):
    return {name for name, answer in judgments.items()
            if answer.get('status') != completion_expected_status(name)}





def completion_obligations(snapshot, memory):
    fields = ('id', 'kind', 'color', 'description', 'status', 'destination_id', 'destination_kind',
              'visual_membership', 'input_entity_id', 'output_entity_id', 'root_input_id', 'processing_history',
              'creation_evidence', 'visual_release', 'container_entry', 'transfers', 'transport_status', 'confirmed_observation_id',
              'last_seen_observation_id', 'visibility', 'association_status')
    result = {key: deepcopy(snapshot[key]) for key in
        ('object_tasks', 'selected_input_ids', 'completed_input_ids', 'retired_entity_ids', 'product_ids', 'transformations',
         'membership', 'unresolved_membership', 'deferred') if key in snapshot}
    task_ids = {record['id'] for record in snapshot.get('objects', [])}
    result['retired_entity_ids'] = [identifier for identifier in memory['retired_entity_ids']
                                    if identifier in task_ids]
    result['transformations'] = {key: deepcopy(row) for key, row in memory['transformations'].items()
                                if row['input_entity_id'] in task_ids and row['output_entity_id'] in task_ids}
    result['objects'] = []
    for record in snapshot.get('objects', []):
        summary = {key: deepcopy(record[key]) for key in fields if key in record}
        for name in ('destination_visual_evidence', 'destination_tracking_evidence'):
            if name in record:
                evidence = record[name]
                summary[name] = {key: deepcopy(evidence[key]) for key in (
                    'status', 'source', 'identity_source', 'request_hash', 'observation_id', 'generation',
                    'destination_id', 'target_id', 'region_id', 'binding_id',
                    'reason', 'confidence', 'answers', 'placement_outcome') if key in evidence}
        result['objects'].append(summary)
    return result

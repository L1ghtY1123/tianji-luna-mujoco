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
    """Remaining input is a presence question: completion requires a confident no."""
    return 'no' if name == 'source_has_remaining_input' else 'yes'


def completion_failures(judgments):
    return {name for name, answer in judgments.items()
            if answer.get('status') != completion_expected_status(name)}


COMPLETION_MILESTONE_RULE = (
    ' Judge past delivery and processing milestones from the matching input/output IDs in '
    'verified_obligations.objects and transformations, including container_entry, destination_visual_evidence, '
    'visual_release and creation_evidence. A consumed input need not remain visible or still sit '
    'at its earlier destination in the final RGB. Use CURRENT images for current physical state, '
    'and reject contradictory current evidence. Missing or unverified historical receipts remain '
    'unproven; completed status alone is not evidence. A container_entry with source '
    'operator_container_release_rule is the operator-authorized delivery receipt after checked '
    'above-container release. Accept that receipt for the requested entry milestone, including '
    'occluded or stacked stored contents. It does not claim a measured individual settled pose. '
    'Apply the same evidence standard to this '
    'requirement whether it is phrased at aggregate or individual-object scope. '
    'For aggregate processing/delivery milestones, assess the selected_input_ids and their verified '
    'lineages. The separate source_has_remaining_input question checks current source closure, '
    'including additional matching inputs. Do not add a source-closure requirement to a historical '
    'milestone that does not ask for it. With zero selected inputs, per-input milestones are vacuously '
    'satisfied and source closure is assessed separately. Do not invent a minimum quantity. '
    'If the quoted condition explicitly asks about current source '
    'contents, judge that clause from CURRENT source RGB and the original input selector, consistently '
    'with source_has_remaining_input. Initial unknown completeness is not current contrary evidence.')

COMPLETION_HISTORY_RULE = (
    ' INITIAL reference descriptions and images may show old contents; they describe historical '
    'appearances. Retired entities are confirmed consumed or removed instances; '
    'their old visible/occluded fields do not assert that they still exist. Assess remaining targets '
    'within task.source_ref from the union of CURRENT views and the original target_selector. '
    'Do not substitute the current manipulation target_id (which may be a processing output) for '
    'that original selector or source. Unknown historical completeness can be resolved by current '
    'visible evidence; current occlusion or indistinguishable subjects must still remain uncertain. '
    'Current images cannot prove past action order; use matching verified receipts for past milestones.')


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
                    'reason', 'confidence', 'answers') if key in evidence}
        result['objects'].append(summary)
    return result

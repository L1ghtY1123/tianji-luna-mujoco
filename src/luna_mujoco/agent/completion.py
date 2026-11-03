"""Bounded presentation of verified history; Decisions still judges completion."""
from copy import deepcopy


def has_confirmed_delivery(record):
    """Observation status can change; a bound delivery receipt remains historical."""
    transfers = record.get('transfers', [])
    observation_id = record.get('confirmed_observation_id')
    return bool(record.get('transport_status') == 'confirmed' and observation_id and transfers
                and transfers[-1].get('observation_id') == observation_id)


def required_observation_return(requirements, failure):
    """An explicit return obligation is independent of visual insufficiency."""
    return (requirements.get('after_each_object', {}).get('robot_pose') == 'observation'
        and str(failure).split(':', 1)[0] in {
            'processing_requires_post_delivery_observation',
            'object_post_delivery_observation_missing', 'object_observation_pose_not_reached'})

COMPLETION_MILESTONE_RULE = (
    ' Judge past delivery and processing milestones from the matching input/output IDs in '
    'verified_obligations.objects and transformations, including destination_visual_evidence, '
    'visual_release and creation_evidence. A consumed input need not remain visible or still sit '
    'at its earlier destination in the final RGB. Use CURRENT images for current physical state, '
    'and reject contradictory current evidence. Missing or unverified historical receipts remain '
    'unproven; completed status alone is not evidence. Apply the same evidence standard to this '
    'requirement whether it is phrased at aggregate or individual-object scope.')

COMPLETION_HISTORY_RULE = (
    ' INITIAL reference descriptions and images may show old contents; they describe historical '
    'appearances, not additional CURRENT objects. task.source_inventory_complete records the initial '
    'proposal status, not the answer for this current capture. Retired entities are consumed inputs; '
    'their old visible/occluded fields do not assert that they still exist. Assess remaining targets '
    'within task.source_ref from the union of CURRENT views and the original target_selector. '
    'Do not substitute the current manipulation target_id (which may be a processing output) for '
    'that original selector or source. Unknown historical completeness can be resolved by current '
    'visible evidence; current occlusion or indistinguishable subjects must still remain uncertain. '
    'For historical per-object return-and-observe requirements use each verified_obligations.objects '
    'post_delivery_observation receipts, including each intermediate entry in transfers: they record '
    'the input ID, measured dual-arm posture, completed '
    'inventory observation ID and action epoch after delivery. A current image alone cannot prove '
    'past action order. Missing receipts remain unproven; do not infer them from delivery alone.')


def completion_obligations(snapshot):
    fields = ('id', 'kind', 'color', 'description', 'status', 'destination_id', 'destination_kind',
              'visual_membership', 'input_entity_id', 'output_entity_id', 'root_input_id', 'processing_history',
              'creation_evidence', 'visual_release', 'transfers', 'transport_status', 'confirmed_observation_id',
              'last_seen_observation_id', 'visibility', 'association_status', 'post_delivery_observation')
    result = {key: deepcopy(snapshot[key]) for key in
        ('object_tasks', 'selected_input_ids', 'completed_input_ids', 'retired_entity_ids', 'product_ids', 'transformations',
         'membership', 'unresolved_membership', 'deferred') if key in snapshot}
    result['objects'] = []
    for record in snapshot.get('objects', []):
        summary = {key: deepcopy(record[key]) for key in fields if key in record}
        for name in ('destination_visual_evidence', 'destination_tracking_evidence'):
            if name in record:
                evidence = record[name]
                summary[name] = {key: deepcopy(evidence[key]) for key in (
                    'status', 'source', 'identity_source', 'request_hash', 'observation_id', 'generation',
                    'destination_id', 'target_id', 'region_id', 'slot_id', 'binding_id',
                    'reason', 'confidence', 'answers') if key in evidence}
        result['objects'].append(summary)
    return result

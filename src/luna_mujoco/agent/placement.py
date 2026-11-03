"""Ground original placement requirements by judging measured candidate IDs."""
from .decisions import JudgmentRequest


def placement_region_questions(*, configured=False):
    """Separate destination identity from restrictions inside that destination."""
    matching = (
        'For current_transfer only, is the supplied destination_id the support/container '
        'required for the CURRENT target by the original instruction? This is a destination '
        'identity and language check before placement, not proof of completed delivery. '
        'Use current observations and the confirmed per-ID task binding; future completion '
        'receipts and a prior input processing area are separate requirements.')
    if configured:
        matching = (
            'For current_transfer only, does the supplied operator-configured destination region '
            'satisfy the CURRENT target placement location required by the original instruction? '
            'This region is an external workbench setting: its ID and XY are supplied by the operator, '
            'and its support Z is measured. It need not have a visible marking, physical object, '
            'SAM mask or visible name in RGB. Compare the supplied region binding/configuration '
            'with the location clause for this transfer, including a declared processing work area. '
            'Do not require the target to be grasped or placed yet. Do not substitute the later product '
            'destination for the current input work area. This checks requested location, not completed '
            'delivery, numerical safety, or later observation/return receipts. Unknown configuration '
            'or a conflicting region binding remains uncertain; do not invent a region or waive exact XY.')
    return [JudgmentRequest(name='placement_destination_matches', instructions=matching),
        JudgmentRequest(name='placement_additional_spatial_constraint', instructions=(
        'Read the original natural-language instruction for the placement stage identified by current_transfer. '
        'Does its TEXT specify a location within the selected destination (a named subregion, a relative '
        'position, distance, or a different object orientation), beyond the supplied operator region and '
        'preserving the grasp orientation? This classifies what the language requests; it does not ask whether '
        'placement has happened, whether it is feasible, or whether its success can be seen. A left/right '
        'modifier identifying WHICH container is not a restriction within that container. An instruction simply '
        'to put an object inside the selected container or onto the selected surface has no additional internal '
        'position restriction. A configured processing INPUT area applies to that input transfer, not a later '
        'product transfer. CAD/grasp methods, collision/workspace checks, observation posture and final robot '
        'return are not internal position clauses. Answer yes for an actual additional spatial clause, no when '
        'none is requested in the text; uncertain only when the wording or current transfer cannot be '
        'interpreted. Do not infer a new language requirement from numeric geometry, camera viewpoints or '
        'current robot/object poses.'))]


async def select_placement_candidates(judge, candidates, *, context):
    """Select IDs from measured proposals in one Decisions request; never emit poses."""
    visible = [candidate for candidate in candidates if candidate['visible_camera_ids']]
    if not visible:
        return {'status': 'unknown', 'reason': 'placement_candidates_not_visible',
                'candidate_ids': [], 'evidence': []}
    requests = [JudgmentRequest(name='placement_candidate_' + str(index), instructions=(
        f'Does candidate {candidate["candidate_id"]} satisfy the location and orientation clause for the '
        'current transfer in the original instruction? Do not assess CAD choice, grasp method, '
        'processing completion, later observation or final return. Use the '
        'labelled footprint proposal, CURRENT RGB, permanent inventory IDs and measured context. '
        'The candidate is the whole held-object footprint, not just a point. Check the intended '
        'support/container, relative location, named subregion and specified viewpoint. '
        'Use semantic_frames for horizontal directions; an absent or degenerate requested axis '
        'is uncertain, never a reason to switch camera or use the robot frame instead. '
        'A visible free space elsewhere is not an acceptable substitute. The supplied on/inside '
        'mode describes the geometry, not an interpretation you must accept. If references are '
        'occluded, viewpoint is ambiguous, or distance/orientation requirements cannot be verified, '
        'remain uncertain. For an intermediate processing placement, check its supplied capability '
        'requirements and user constraints; do not claim it is the final destination. '
        'This checks semantic suitability only, not stability, IK, collision or path feasibility.'))
        for index, candidate in enumerate(visible)]
    result = await judge(requests, {**context, 'verification_scope': 'placement_candidates',
        'placement_candidates': visible})
    accepted = [candidate['candidate_id'] for candidate, request in zip(visible, requests, strict=True)
                if result.get('judgments', {}).get(request.name, {}).get('status') == 'yes']
    return {'status': 'success' if accepted else 'unknown',
        'reason': 'placement_candidates_selected' if accepted else 'placement_no_semantic_candidate',
        'candidate_ids': accepted, 'evidence': result}


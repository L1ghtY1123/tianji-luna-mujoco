"""Choose an observed destination, then compare it with the graph's proposed ID."""
import json

from .decisions import ChoiceRequest, JudgmentRequest


def target_membership_question(name, identifier, selector, instruction, *, processing=()):
    lifecycle = (' Assess only CURRENT INPUT membership in this original user instruction: '
        + json.dumps(instruction, ensure_ascii=False)
        + '. A proposed selector may describe an input/output lineage, but future processing outputs '
          'are not current input candidates. Do not require those outputs to exist yet, or require '
          'the input to already have the output appearance. Keep all original input/source/subset '
          'constraints; processing does not authorize additional input objects.' if processing else '')
    return JudgmentRequest(name=name, instructions=(
        'Does CURRENT visible canonical entity ' + identifier + ' satisfy target_selector '
        + repr(selector) + ' and the original task source/subset constraints? '
        'Confirm task membership from actual evidence, not proposed membership or action history. '
        'Distinct canonical IDs must refer to independently distinguishable physical objects.' + lifecycle))


def destination_candidates(current_entities, locations):
    aliases = {item['id']: item['image_label'] for view in locations.get('views', {}).values()
               for item in view['items']}
    candidates = []
    for number, entity in enumerate(sorted(current_entities, key=lambda row: row['id'])):
        if entity.get('entity_type') == 'configured_region' or entity.get('kind') == 'marker':
            continue
        candidate = {'value': aliases.get(entity['id'], 'ENTITY' + str(number + 1)),
                     'instance_id': entity['id'], 'kind': entity.get('kind')}
        provenance = entity.get('observation_provenance', {})
        if (locations.get('observation_id')
                and provenance.get('observation_id') == locations['observation_id']
                and entity.get('visible_centroid_world') is not None):
            candidate.update(color=entity.get('color'),
                visible_centroid_world=entity['visible_centroid_world'],
                observation_id=provenance['observation_id'])
        candidates.append(candidate)
    return candidates


def destination_question(instruction, selector, candidates, *, processing=()):
    if processing:
        return ChoiceRequest(name='destination_requirement_matches', instructions=(
            'Select only the FINAL destination entity requested for the processed product of input selector '
            + repr(selector) + '. The selector identifies the input, not its destination. '
            'Read the final-delivery clause in the original user instruction below. Do not select the input source '
            'or its intermediate processing area. Do not judge whether processing, delivery or later robot return '
            'has already happened; the product need not exist yet.\nOriginal user instruction: '
            + instruction + '\nCURRENT destination candidates: '
            + json.dumps(candidates, ensure_ascii=False)
            + '\nReturn one listed value. ID labels locate the corresponding CURRENT entities. '
            'For unqualified left/right use the measured robot/world Y coordinates: larger Y is farther LEFT, '
            'smaller Y farther RIGHT (+Y is robot LEFT; -Y is robot RIGHT). '
            'An explicitly named viewpoint in the USER instruction overrides this convention; camera-image '
            'sides and historical reference descriptions do not override it. '
            'CAD choice, processing success, free space and motion safety are separate questions. '
            'Choose none_of_listed for a clearly requested destination absent from these candidates; '
            'choose unknown only if the required destination cannot be distinguished from this evidence.'),
            choices=tuple(row['value'] for row in candidates) + ('none_of_listed', 'unknown'))
    measured = [{key: row[key] for key in ('value', 'kind', 'color', 'visible_centroid_world')}
                for row in candidates if 'visible_centroid_world' in row]
    locations = (' Current measured candidate centroids in robot world metres: '
        + json.dumps(measured, ensure_ascii=False, separators=(',', ':'))
        + '. +Y is robot LEFT; -Y is robot RIGHT. Unqualified left/right uses this convention; '
          'an explicitly named viewpoint overrides it. RGB establishes identity, not world-axis direction.'
        if measured else '')
    return ChoiceRequest(name='destination_requirement_matches', instructions=(
        'For objects matching ' + repr(selector) + ', which CURRENT visible entity is the '
        + 'requested destination in this ORIGINAL instruction: ' + json.dumps(instruction, ensure_ascii=False)
        + '? Select its value from destination_candidates; values beginning ID match current image labels. '
        'Other values map to the permanent-ID reference. Check the requested support or container and '
        'original side/viewpoint using current RGB and measured_location_rule. Proposed task bindings '
        'are hypotheses. Choose none_of_listed if a clearly requested destination is absent from the '
        'list, or unknown if ambiguous. This selects the destination entity only; empty landing space '
        'and motion feasibility are checked by placement skills.' + locations),
        choices=tuple(row['value'] for row in candidates) + ('none_of_listed', 'unknown'))


def bind_destination_choice(result, candidates, proposed_id):
    """Do not silently rebind the graph when the model selects a different entity."""
    name = 'destination_requirement_matches'
    answer = result['judgments'][name]
    value = answer.get('value')
    known = answer['status'] == 'known' and value != 'unknown'
    selected = next((row['instance_id'] for row in candidates if row['value'] == value), None) if known else None
    status = ('yes' if selected == proposed_id else 'no') if known else 'unknown'
    result['judgments'][name] = {**answer, 'selection_status': answer['status'], 'status': status,
        'selected_instance_id': selected, 'proposed_instance_id': proposed_id}
    result['evidence']['destination_binding_comparison'] = {
        'source': 'typed_choice_compared_with_proposed_graph_binding', 'status': status,
        'candidate_mapping': candidates, 'selected_instance_id': selected, 'proposed_instance_id': proposed_id}

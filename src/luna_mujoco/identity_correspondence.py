"""Decisions compares each candidate with every possible permanent identity."""
import json
import re
from copy import deepcopy

from .decision_conditions import parse_answers
from .multiview_identity import grouped_view_instruction
from .visual_api import VisualAPI, VisualInventoryError
from .visual_inventory import CURRENT_ANCHOR_IDENTITY_RULE

CURRENT_IDENTITY_ANCHOR_RULE = (
    'CURRENT confirmed identity anchors: these are the original accepted Decisions answers from '
    'this exact CURRENT frozen capture, not historical positions or new numerical identity assignments. '
    'Their identity questions are omitted because those answers remain accepted. The unchanged CURRENT '
    'images retain their Candidate labels; use the candidate, camera and box below to locate each anchor. '
    'Compare each unresolved candidate with the current accepted anchor for that proposed ID as well as '
    'the initial appearance reference. Multiple cameras can show one physical subject, but simultaneously '
    'distinct current subjects cannot share one ID. Movement between captures cannot explain two distinct '
    'subjects in this same current capture. Position, color, CAD category and provenance stamps alone do '
    'not establish a new match. Reject visibly different subjects and keep ambiguous correspondence '
    'unknown; do not revise the accepted answers.')


def metadata(part):
    if part.get('type') != 'input_text':
        return {}
    try:
        value = json.loads(part['text'])
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def has_product_reference(payload):
    # product_reference_content has already checked owner, generation and image hashes.
    return any(metadata(part).get('image_role') == 'verified_product_reference'
               for part in payload['input'][0]['content'])


def scope_identity_clarification(payload, evidence, threshold):
    """Do not re-ask an identity already accepted from this exact request's capture."""
    result = deepcopy(payload)
    context = metadata(result['input'][0]['content'][0])
    if (evidence.get('observation_id') != context.get('observation_id')
            or evidence.get('generation') != context.get('generation')
            or not evidence.get('request_hash')):
        raise VisualInventoryError('visual_identity_clarification_provenance_missing')
    answers = {answer['name']: answer for answer in evidence.get(
        'original_decisions_answers', evidence.get('answers', []))}
    retained = {}
    for question in payload['questions']:
        name = question['name']
        if not name.startswith('identity_') or name not in answers:
            continue
        judgments, raw = parse_answers({'answers': [answers[name]]}, [question], threshold)
        accepted = (judgments[name] == 'yes' if question['type'] == 'predicate'
                    else judgments[name] not in {'unknown', 'reject'})
        if accepted:
            retained[name] = {'question': deepcopy(question), 'answer': raw[0],
                'observation_id': evidence['observation_id'], 'generation': context.get('generation'),
                'request_hash': evidence['request_hash']}
    result['questions'] = [question for question in result['questions'] if question['name'] not in retained]
    candidates = {row['candidate']: row for part in result['input'][0]['content']
                  if 'candidate' in (row := metadata(part)) and 'camera' in row}
    anchors = []
    for name, record in retained.items():
        candidate = 'candidate-' + name.removeprefix('identity_')
        row = candidates.get(candidate, {})
        provenance = row.get('candidate_provenance', {})
        identifier = (record['answer'].get('choice') if record['answer']['type'] == 'choice' else
                      evidence.get('same_capture_identity_proposals', {}).get(name))
        if (identifier not in context.get('fixed_entities', {})
                or provenance.get('candidate') != candidate
                or provenance.get('camera') != row.get('camera')
                or provenance.get('observation_id') != context.get('observation_id')
                or provenance.get('generation') != context.get('generation')
                or not provenance.get('silhouette_mask') or not provenance.get('view', {}).get('rgb')
                or not row.get('box')):
            raise VisualInventoryError('visual_identity_clarification_anchor_binding_invalid')
        label = next((label for label, item in context.get('identity_display_legend', {}).items()
                      if item['entity_id'] == identifier), identifier)
        anchors.append({**deepcopy(record), 'source': 'openai_decisions', 'status': 'confirmed',
            'entity_id': identifier, 'candidate': candidate, 'camera': row['camera'], 'box': deepcopy(row['box']),
            'display_label': 'CURRENT confirmed ' + label + ' at Candidate ' + name.removeprefix('identity_'),
            'minimum_confidence': threshold, 'candidate_provenance': deepcopy(provenance)})
    if anchors:
        result['input'][0]['content'].append({'type': 'input_text', 'text': json.dumps({
            'current_confirmed_identity_anchors': anchors, 'rule': CURRENT_IDENTITY_ANCHOR_RULE}, ensure_ascii=False)})
    return result, retained


def correspondence_payload(payload):
    result = deepcopy(payload)
    context = metadata(result['input'][0]['content'][0])
    proof = context.get('same_capture_reference', {})
    current = context.get('observation_id')
    same_capture = bool(current and proof.get('initial_reference_observation_id') == current
        and proof.get('current_observation_id') == current
        and proof.get('generation') == context.get('generation'))
    candidates = {row['candidate']: row['camera'] for part in result['input'][0]['content']
                  if 'candidate' in (row := metadata(part)) and 'camera' in row}
    reference_cameras = {}
    for part in result['input'][0]['content']:
        row = metadata(part)
        if row.get('image_role') == 'initial_reference_annotations' and row.get('camera'):
            for item in row.get('items', []):
                reference_cameras.setdefault(item['id'], set()).add(row['camera'])
    questions, pairs = [], {}
    anchors = any(metadata(part).get('current_confirmed_container_anchors')
                  for part in result['input'][0]['content'])
    for question in payload['questions']:
        if not question['name'].startswith('identity_') or question['type'] != 'choice':
            questions.append(deepcopy(question))
            continue
        number = question['name'].removeprefix('identity_')
        camera = candidates['candidate-' + number]
        alternatives = {}
        for index, option in enumerate(question['choices']):
            if option['value'] in {'reject', 'unknown'}:
                continue
            name = f'match_{number}_{index}'
            alternatives[name] = option['value']
            cameras = reference_cameras.get(option['value'], set())
            stationary = (same_capture and camera in proof.get('verified_camera_ids', [])
                          and bool(cameras.intersection(proof.get('verified_camera_ids', []))))
            temporal_rule = (
                'This reference and CURRENT views use the SAME verified frozen capture with no intervening motion. '
                'Camera projections differ, but the referenced subject cannot have moved to a different place. '
                if stationary else 'Historical positions, supports and orientations may change. ')
            identical = next((row for row in context.get('identical_reference_regions', [])
                if row.get('candidate') == 'candidate-' + number and row.get('camera') == camera
                and row.get('reference_entity_id') == option['value']
                and row.get('current_observation_id') == current
                and row.get('reference_observation_id') not in (None, current)
                and row.get('generation') == context.get('generation')), None)
            if identical is not None:
                temporal_rule = (
                    'These are different acquisitions, but their same-camera union ROI '
                    + json.dumps(identical['union_roi_xyxy_px'])
                    + ' has byte-identical RGB, depth and validity with identical calibration and camera pose. '
                    'You can directly compare their 2D positions in that region. This local evidence does not '
                    'prove continuous history or identity; inspect the subjects themselves. ')
            location = question['instructions'].partition(' Current measured candidate surface: ')
            questions.append({'type': 'predicate', 'name': name, 'instructions':
                f'The physical subject marked Candidate {number} in CURRENT camera {camera} is the same '
                f"individual physical entity as reference {option['description']} (ID {option['value']}). "
                'Compare the subject itself with the INITIAL or verified PRODUCT appearance reference. '
                'An object and its surrounding container are different entities. ' + temporal_rule
                + 'Color, SAM text, location and action history alone '
                'do not prove identity. False if they are visibly different subjects; uncertain when '
                'the subject or individual correspondence cannot be distinguished.'
                + (' ' + CURRENT_ANCHOR_IDENTITY_RULE if anchors else '')
                + (location[1] + location[2] if location[1] else '')})
        if not alternatives:
            # Geometric exclusions can leave only reject/unknown. Keep that
            # explicit model choice instead of inventing an identity or verdict.
            questions.append(deepcopy(question))
            continue
        pairs[question['name']] = alternatives
    if len(questions) > 200:
        raise VisualInventoryError('visual_inventory_input_limit')
    result['questions'] = questions
    return factor_identity_instructions(result), pairs


def factor_identity_instructions(payload):
    """Present repeated rules once, with a lossless per-predicate binding."""
    result = deepcopy(payload)
    content = result['input'][0]['content']
    context = metadata(content[0])
    locations = {row['candidate']: row for row in context.get('candidate_geometric_location_hints', [])}
    prefix = re.compile(r'The physical subject marked Candidate (\d+) in CURRENT camera (\S+) '
        r'is the same individual physical entity as reference (.+) \(ID ([^()]+)\)\.', re.DOTALL)
    marker = ' Current measured candidate surface: '
    references, additional, overrides, measurements = {}, {}, {}, {}
    variants = []
    common = None
    for question in result['questions']:
        if question['type'] != 'predicate' or not question['name'].startswith('match_'):
            continue
        first, separator, rest = question['instructions'].partition(' Compare the subject itself')
        match = prefix.fullmatch(first)
        if match is None or not separator:
            raise VisualInventoryError('visual_identity_instruction_binding_invalid')
        number, camera, description, identifier = match.groups()
        if not question['name'].startswith('match_' + number + '_'):
            raise VisualInventoryError('visual_identity_instruction_binding_invalid')
        before, measured, remainder = (separator + rest).partition(marker)
        after = ''
        if measured:
            location, end = json.JSONDecoder().raw_decode(remainder)
            candidate = 'candidate-' + number
            if location.get('camera') != camera or location.get('candidate', candidate) != candidate:
                raise VisualInventoryError('visual_identity_instruction_binding_invalid')
            encoded, after = remainder[:end], remainder[end:]
            if candidate in measurements and measurements[candidate] != encoded:
                raise VisualInventoryError('visual_identity_instruction_measurement_conflict')
            measurements[candidate] = encoded
            if candidate in locations and locations[candidate] != location:
                raise VisualInventoryError('visual_identity_instruction_measurement_conflict')
            if candidate not in locations or json.dumps(location) != encoded:
                additional[candidate] = encoded
        fragments = {'before_candidate_measurement': before,
            'has_candidate_measurement': bool(measured), 'after_candidate_measurement': after}
        if common is None:
            common = fragments
        elif fragments != common:
            if fragments not in variants:
                variants.append(fragments)
            overrides[question['name']] = variants.index(fragments)
        if identifier in references and references[identifier] != description:
            raise VisualInventoryError('visual_identity_instruction_reference_conflict')
        references[identifier] = description
        question['instructions'] = (
            f'The physical subject marked Candidate {number} in CURRENT camera {camera} '
            f'is the same individual physical entity as permanent reference ID {identifier}.')
        entity = context.get('fixed_entities', {}).get(identifier, {})
        if entity.get('kind') and entity.get('color'):
            question['instructions'] += (
                f" Reference subject: {entity['color']} {entity['kind']}. "
                'Match this physical object itself, excluding contents and neighboring objects.')
        question['instructions'] += grouped_view_instruction(context, 'candidate-' + number)
    if common is not None:
        content.append({'type': 'input_text', 'text': json.dumps({'factored_identity_instructions': {
            'applies_to': 'Every predicate whose name starts with match_',
            'rule': 'Apply both shared instruction fragments in full to every match predicate. '
                'If its name has an instruction override, apply the indexed instruction variant instead. '
                'For its Candidate number and camera, use that exact row of candidate_geometric_location_hints '
                'in the original context as the Current measured candidate surface. '
                'Additional candidate measurements below preserve measurements absent from that table or '
                'their original serialization. The permanent reference ID has the original reference label '
                'and description below. Factoring repeated text does not change any identity, uncertainty, '
                'distinct-subject, geometric-evidence limitation, question type, answer or acceptance rule.',
            'shared_instruction_fragments': common, 'instruction_overrides_by_question': overrides,
            'instruction_variants': variants,
            'original_reference_labels_and_descriptions_by_id': references,
            'additional_candidate_measurements': additional,
            'candidate_measurements_source': 'original context candidate_geometric_location_hints; '
                'candidate and camera must both match the predicate'}}, ensure_ascii=False)})
    return result


def resolve_correspondences(body, request, original, pairs, threshold):
    judgments, raw = parse_answers(body, request['questions'], threshold)
    answers = {answer['name']: answer for answer in raw}
    derived, checks = [], []
    for question in original['questions']:
        name = question['name']
        if name not in pairs:
            derived.append(deepcopy(answers[name]))
            continue
        alternatives = pairs[name]
        positive = [key for key in alternatives if judgments[key] == 'yes']
        selected = positive[0] if len(positive) == 1 else None
        exclusive = selected is not None and all(
            judgments[key] == 'no' for key in alternatives if key != selected)
        probabilities = {identifier: answers[key]['probability'] for key, identifier in alternatives.items()}
        # A conservative agreement score, not a categorical confidence returned by the API.
        score = min([answers[selected]['probability']] + [
            1. - answers[key]['probability'] for key in alternatives if key != selected]) if selected else 0.
        choice = alternatives[selected] if exclusive else 'unknown'
        derived.append({'type': 'choice', 'name': name, 'choice': choice, 'confidence': score,
                        'confidence_metric': 'minimum_pairwise_agreement'})
        checks.append({'name': name, 'choice': choice, 'probabilities': probabilities,
                       'metric': 'minimum_pairwise_agreement', 'score': score,
                       'positive_threshold': threshold, 'negative_threshold': round(1. - threshold, 12),
                       'rule': 'exactly_one_positive_and_all_alternatives_negative'})
    return {**body, 'answers': derived}, {
        'original_decisions_answers': raw, 'identity_correspondence': checks,
        'identity_derivation': 'exclusive_pairwise_visual_predicates_not_numeric_identity_assignment'}


def request_identity_correspondences(client, payload, capture, purpose, parse, *, identity_subjects=None,
                                    include_proposal_subjects=True):
    request, pairs = correspondence_payload(payload)
    if identity_subjects is not None and (pairs or include_proposal_subjects):
        from .identity_subject_views import append_identity_subject_views

        questions = request['questions']
        subject_request = request if include_proposal_subjects else {
            **request, 'questions': [question for question in questions if question['name'].startswith('match_')]}
        request = {**append_identity_subject_views(subject_request, capture, identity_subjects), 'questions': questions}

    def receive(body, evidence):
        resolved, proof = resolve_correspondences(body, request, payload, pairs, client.minimum_confidence)
        resolved_answers = {answer['name']: answer for answer in resolved['answers']}
        diagnostics = [{**check, 'component': 'decisions', 'actual': check['score'],
                        'threshold': client.minimum_confidence,
                        'status': 'known' if check['choice'] != 'unknown' else 'unknown',
                        'reason': check['rule'], 'answer': resolved_answers[check['name']]}
                       for check in proof['identity_correspondence']]
        return parse(resolved, {**evidence, **proof,
                               'diagnostic_checks': diagnostics + evidence.get('diagnostic_checks', [])})

    return VisualAPI.request(client, request, capture, purpose, receive)

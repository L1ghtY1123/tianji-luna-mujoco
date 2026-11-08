"""Associate each current subject with permanent memory through atomic identity predicates."""
import hashlib
import json
from copy import deepcopy

from .decision_conditions import parse_answers
from .demo_identity_presentation import box_overlap
from .inventory_decision_input import (
    candidate_labels,
    current_bindings,
    inventory_payload,
    metadata,
    reference_labels,
    scene_questions,
)
from .visual_api import VisualAPI, VisualInventoryError, merge_decision_evidence
from .visual_inventory import reference_content


def identity_hypotheses(payload, question):
    """Order comparisons using locating proposals and appearance queries, without accepting an ID."""
    context = metadata(payload['input'][0]['content'][0])
    candidate = 'candidate-' + question['name'].removeprefix('identity_')
    subject = next(row for part in payload['input'][0]['content']
                   if (row := metadata(part)).get('candidate') == candidate)
    identifiers = [row['value'] for row in question['choices']
                   if row['value'] not in {'reject', 'unknown', 'new_instance'}]

    def priority(identifier):
        entity = context['memory_entities'][identifier]
        proposal = entity.get('current_identity_proposal', {})
        overlap = (box_overlap(subject['box'], proposal['bbox'])
                   if proposal.get('observation_id') == context['observation_id']
                   and proposal.get('camera_id') == subject['camera'] else 0.)
        query = entity.get('segmentation_query', {})
        appearance_match = subject['sam3_prompt_proposal'] in (
            query.get('prompt'), *query.get('aliases', []))
        current_creation = entity.get('creation_evidence', {}).get('observation_id') == context['observation_id']
        projected_overlap = max((row['mask_iou'] for row in context.get('projected_reference_surfaces', [])
            if row['entity_id'] == identifier and row['target_candidate'] == candidate), default=0.)
        scope = context.get('association_scope', {})
        return (identifier in scope.get('deferred_entity_ids', ()),
                -overlap, not appearance_match, -projected_overlap, not current_creation,
                identifier not in scope.get('focus_entity_ids', ()))

    return sorted(identifiers, key=priority)


def correspondence_instructions(subject, reference_label, description, current_matches=(), *, subject_description=None):
    reference_scope = (
        'The reference is already localized in this SAME capture as ' + ', '.join(current_matches)
        + '. Compare those labelled subjects directly. ' if current_matches else
        'Compare the dated reference images with the current subject and the recorded physical changes. ')
    if subject_description:
        subject += ' (proposed subject description: ' + subject_description + ')'
    return ('Judge only this identity hypothesis: CURRENT subject ' + subject + ' and reference '
        + reference_label + ' denote one physical object. '
        'Reference description (dated; image directions refer only to its original view): ' + description + ' '
        + reference_scope
        + 'A container and an object inside it are different physical objects even when their rectangles overlap. '
        'Calibrated measurements share one world frame; different simultaneously occupied locations are evidence '
        'against identity. Visible surface sampling varies by viewpoint. '
        'SAM class names are proposals, not facts. Use RGB, masks and depth evidence together. '
        'Return a low probability for different objects; remain uncertain only when correspondence evidence is insufficient.')


def identity_request(payload, question, identifier, *, bindings=None, identity_subjects=None):
    context = metadata(payload['input'][0]['content'][0])
    labels = reference_labels(context)
    subjects = candidate_labels(payload)
    candidate = 'candidate-' + question['name'].removeprefix('identity_')
    subject = subjects[candidate]['label'] + ' in camera ' + subjects[candidate]['camera']
    if identifier is None:
        instructions = ('The CURRENT subject labelled ' + subject
            + ' is a distinct external physical object that can be registered as a new instance. '
              'All listed remembered-instance comparisons were confidently negative. Robot parts, '
              'background, and a duplicate fragment of an accepted current subject are outside this predicate.')
    else:
        entity = context['memory_entities'][identifier]
        current_matches = [subjects[key]['label'] + ' in camera ' + subjects[key]['camera']
                           for key, value in (bindings or {}).items()
                           if value == identifier and key != candidate and key in subjects]
        instructions = correspondence_instructions(subject, labels[identifier]['image_label'],
                                                    entity['description'], current_matches)
    request_question = {'type': 'predicate', 'name': question['name'], 'instructions': instructions}
    # Limit historical images to this hypothesis while retaining current
    # accepted subjects so similar instances remain visually distinguishable.
    referenced_ids = [identifier] if identifier is not None else sorted(set((bindings or {}).values()))
    request = inventory_payload(payload, 'identity', [request_question], bindings=bindings,
                                candidate_ids=[candidate], entity_ids=referenced_ids, identity_subjects=identity_subjects)
    if identifier is not None:
        from .inventory_decision_input import add_correspondence_geometry
        add_correspondence_geometry(request, subjects[candidate]['label'], subjects[candidate]['camera'],
                                   labels[identifier]['image_label'], current_matches=current_matches)
    return request


def request_identity_correspondences(client, payload, capture, purpose, parse, *, subjects=None):
    context = metadata(payload['input'][0]['content'][0])
    answers, records, raw_answers, registered = {}, [], [], {}

    def ask(question, identifier):
        request = identity_request(payload, question, identifier, bindings=current_bindings(
            context, list(answers.values()), client.minimum_confidence), identity_subjects=subjects)
        body, evidence = VisualAPI.request(client, request, capture, purpose,
                                          lambda body, evidence: (body, evidence))
        judgments, _ = parse_answers(body, request['questions'], client.minimum_confidence)
        raw = body['answers'][0]
        raw_answers.append(deepcopy(raw))
        records.append({**evidence, 'answers': [deepcopy(raw)]})
        return judgments[question['name']], raw, evidence

    preferred = set(context.get('reference_view_questions', []))
    questions = sorted(payload['questions'], key=lambda question: question['name'] not in preferred)
    for question in questions:
        if not question['name'].startswith('identity_'):
            continue
        unresolved, answer = [], None
        for identifier in identity_hypotheses(payload, question):
            status, raw, evidence = ask(question, identifier)
            candidate_answer = {**deepcopy(raw), 'identity_candidate': identifier}
            if status == 'yes':
                answer = candidate_answer
                break
            if status == 'unknown':
                unresolved.append(candidate_answer)
            answer = candidate_answer
        if answer is None or not (answer.get('probability', 0.) >= client.minimum_confidence):
            if unresolved:
                answer = unresolved[0]
            elif client.register_instance is not None:
                status, raw, evidence = ask(question, None)
                if status == 'yes':
                    candidate = 'candidate-' + question['name'].removeprefix('identity_')
                    entity = client.register_instance(capture, subjects[candidate],
                        {'answer': deepcopy(raw), 'request_hash': evidence['request_hash'],
                         'compared_instance_ids': identity_hypotheses(payload, question)})
                    identifier = entity['id']
                    registered[identifier] = entity
                    context['memory_entities'][identifier] = deepcopy(entity)
                    context['identity_display_legend']['Reference ' + identifier] = {
                        'entity_id': identifier, 'image_label': 'R' + identifier,
                        'description': entity['description']}
                    for following in questions:
                        if following['name'].startswith('identity_'):
                            following['choices'].append({'value': identifier, 'description': entity['description']})
                    payload['input'][0]['content'][0]['text'] = json.dumps(context, ensure_ascii=False)
                    stored = {'observation_id': capture['observation_id'], 'time': capture.get('time'),
                        'camera_id': entity['reference_camera_id'], 'bbox': entity['bbox'],
                        'rgb': capture['views'][entity['reference_camera_id']]['rgb']}
                    payload['input'][0]['content'].extend(reference_content(
                        {'entities': {identifier: entity}, 'identity_references': {identifier: stored}},
                        current_capture=capture))
                    answer = {**deepcopy(raw), 'identity_candidate': identifier}
                else:
                    answer = {**deepcopy(raw), 'identity_candidate': 'reject', 'identity_accept_status': 'no'}
        if answer is None:
            raise VisualInventoryError('identity_reference_or_registration_required')
        answers[question['name']] = answer

    bindings = current_bindings(context, list(answers.values()), client.minimum_confidence)
    for question in scene_questions(payload, 'progress'):
        request = inventory_payload(payload, 'progress', [question], bindings=bindings)
        body, evidence = VisualAPI.request(client, request, capture, purpose + '_progress',
                                          lambda body, evidence: (body, evidence))
        parse_answers(body, [question], client.minimum_confidence)
        answers[question['name']] = body['answers'][0]
        raw_answers.append(deepcopy(body['answers'][0]))
        records.append({**evidence, 'answers': body['answers']})
    if set(answers) != {question['name'] for question in payload['questions']}:
        raise VisualInventoryError('visual_decisions_invalid_response')
    evidence = merge_decision_evidence(records, capture, purpose)
    if not records:
        evidence.update(source='current_segmentation', component='perception',
            request_hash=hashlib.sha256(capture['observation_id'].encode()).hexdigest(),
            identity_derivation='no_current_identity_candidates')
    evidence.update(original_decisions_answers=raw_answers, registered_entities=registered)
    if records:
        evidence['identity_derivation'] = 'atomic_current_instance_correspondence'
    return parse({'answers': [answers[question['name']] for question in payload['questions']]}, evidence)

"""Fresh visual evidence for the configured simulation unpack capability."""
import hashlib
import json
import math
from copy import deepcopy

from ..core.decisions import predicate_status
from ..decision_conditions import condition, parse_answers
from ..observation_requirements import assessment
from ..visual_api import VisualAPI, VisualDecisionError, merge_decision_evidence
from ..vla_configuration import vla_processing_effect
from .evidence import (
    _require,
    unpack_verification_capture,
    validate_table_release_reference,
    validate_work_area,
)


def _validate_action(capture, target_id, action, owner_id, *, source):
    _require(bool(owner_id) and action.get('owner_id') == owner_id
             and action.get('target_id') == target_id and bool(target_id)
             and action.get('generation') == capture.get('generation')
             and type(capture.get('generation')) is int
             and action.get('source') == source and action.get('completed') is True
             and bool(action.get('action_id')), 'action_binding_invalid')
    _require(set(capture.get('views', {})) == {'1', '2', '3'}
             and bool(capture.get('observation_id')), 'three_current_views_required')
    current, completed = capture.get('time'), action.get('completed_time')
    _require(all(type(value) in (int, float) and math.isfinite(value) for value in (current, completed))
             and current >= completed, 'capture_before_action')
    acquired = capture.get('captured_perf_counter_ns')
    finished = action.get('completed_perf_counter_ns')
    if finished is not None:
        _require(type(finished) is int and type(acquired) is int and acquired > finished,
                 'capture_before_action')
    else:
        _require(current > completed, 'capture_before_action')


def unpack_result_questions():
    effect = vla_processing_effect()
    return [condition('unpack_result', f"Is a new separate {effect['output']['color']} workpiece visible on the main tabletop in "
        'CURRENT, outside both blue bins, that was not present there in BEFORE? Compare the whole tabletop, '
        'not just the former wrapper location: processing may move the output elsewhere on this table. '
        'Black robot components and pre-existing workpieces do not count. One clear current view is '
        'sufficient; wrapper disappearance is checked separately.'),
        condition('wrapper_removed', f"The bound {effect['input']['color']} wrapper on the tabletop in BEFORE has disappeared in "
            'CURRENT. Compare its former tabletop area across the current camera views. Other pink objects '
            'inside the source bin are separate instances. Occlusion alone does not establish removal.')]


def unpack_result_context(question_name):
    context = {
        'subject': 'The bound pink wrapped input previously released onto the main table.',
        'image_times': 'BEFORE follows verified table release. CURRENT follows the unpacking action. '
            'Each time has three simultaneous camera views. The action record does not prove its result.',
        'evidence_rule': 'Use BEFORE to identify the bound processing area and CURRENT for the requested '
            'visible result. One clear current view is sufficient. Keep occluded or ambiguous evidence uncertain.'}
    if question_name == 'unpack_result':
        context['evidence_rule'] = (
            'Compare the whole main tabletop in BEFORE and CURRENT for the requested visible change. '
            'The processing action can relocate its output across the tabletop; its location need not '
            'match the old wrapper location. One clear current view is sufficient. '
            'Keep occluded or ambiguous evidence uncertain.')
        context['expected_appearance'] = (
            'A new separate black solid workpiece, possibly irregular with protrusions, on the main tabletop. '
            'Distinguish its body from robot fingers using the before/current views. Existing workpieces '
            'inside the bins are outside this question.')
    elif question_name == 'wrapper_removed':
        context['reference_subject'] = (
            'Only the pink wrapper on the main table in BEFORE. Other pink objects inside the source bin '
            'are different instances. A visible black part alone does not prove this wrapper disappeared.')
    else:
        raise ValueError('unknown_unpack_question')
    return context


def unpack_question_payload(capture, before, question, *, target_id, audit_marker):
    context = dict(unpack_result_context(question['name']), observation_id=capture['observation_id'],
                   before_observation_id=before['observation_id'], target_id=target_id, audit_marker=audit_marker)
    context['processing_effect'] = vla_processing_effect()
    content = [{'type': 'input_text', 'text': json.dumps(context)}]
    from ..raw_visual_input import capture_content
    content.extend(capture_content(before, modalities=('rgb',), role='before_unpacking'))
    content.extend(capture_content(capture, modalities=('rgb',)))
    return {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': [deepcopy(question)]}


def verify_unpacking(client, capture, *, target_id, release_evidence, action_evidence, owner_id, work_area,
                     table_release_capture=None, table_release_reference=None, active_target_ids=None):
    """Confirm the expected appearance transition; do not mutate state or build geometry."""
    _validate_action(capture, target_id, action_evidence, owner_id,
                     source='mock_vla')
    _require(active_target_ids == [target_id], 'single_active_target_required')
    table = release_evidence
    confidence = table.get('confidence')
    _require(bool(action_evidence.get('table_release_feedback_id'))
             and table.get('release_feedback_id') == action_evidence['table_release_feedback_id']
             and table.get('target_id') == target_id and table.get('status') == 'at_destination'
             and table.get('identity_source') == 'execution_chain'
             and predicate_status(confidence) == 'yes'
             and bool(table.get('request_hash')) and bool(table.get('binding_id')), 'table_release_required')
    validate_table_release_reference(table_release_capture, table_release_reference, capture,
                                     table, action_evidence, owner_id)
    verification_capture = unpack_verification_capture(capture)
    validate_work_area(work_area)
    # Cache isolation is an audit marker, not a demand for RGB to verify IDs/hashes.
    scope = {'target_id': target_id, 'owner_id': owner_id, 'generation': capture['generation'],
        'active_target_ids': list(active_target_ids), 'action_id': action_evidence['action_id'],
        'table_release_reference': table_release_reference}
    marker = hashlib.sha256(json.dumps({**scope, 'verification_capture': verification_capture},
                                      sort_keys=True).encode()).hexdigest()
    questions = unpack_result_questions()
    def parse(body, evidence):
        judgments, answers = parse_answers(body, questions)
        result_answer = next(answer for answer in answers if answer['name'] == 'unpack_result')
        views = {'judgments': {'unpack_result': judgments['unpack_result']},
                 'answers': [deepcopy(result_answer)], 'confidence': result_answer.get('probability'),
                 'confidence_rule': 'unpack_result_predicate_probability', 'view_scope': 'fused_current_three_rgb'}
        # These questions already require visible current evidence. A second
        # meta-question about the same evidence can contradict both concrete
        # answers without identifying any missing fact. Derive completeness
        # from the required facts; unknown or failed facts still cannot pass.
        observation = assessment(judgments, {'unpack_result': 'yes', 'wrapper_removed': 'yes'},
                                 capture['observation_id'])
        observation['sufficiency_source'] = 'required_visual_criteria'
        success = observation['status'] == 'confirmed'
        reason = None if success else ('unpack_visual_criteria_' + observation['status'] + ':'
            + ','.join(observation['failed_criteria'] or observation['missing_evidence']))
        visual = {**evidence, **views, 'status': observation['status'],
                  'target_id': target_id, 'action_id': action_evidence['action_id'],
                  'identity_source': 'verified_single_active_execution_chain', 'visibility_source': 'openai_decisions',
                  'identity_scope': 'process_input_only_not_product_identity',
                  'execution_chain': {key: deepcopy(value) for key, value in scope.items()
                                      if key != 'table_release_reference'},
                  'table_release_request_hash': table['request_hash'], 'table_release_binding_id': table['binding_id'],
                  'table_release_reference': deepcopy(table_release_reference),
                  'verification_capture': deepcopy(verification_capture),
                  'work_area_zoom': None, 'observation_assessment': observation,
                  'required_judgments': judgments}
        return {**evidence, 'success': success, 'status': visual['status'], 'reason': reason, 'target_id': target_id,
                'generation': capture['generation'], 'appearance': 'unpacked' if success else 'unconfirmed',
                'action_id': action_evidence['action_id'], 'visual_evidence': visual,
                'verification_capture': deepcopy(verification_capture),
                'observation_assessment': observation,
                'response': deepcopy(body), 'geometry_updated': False}

    answers, records = [], []
    for question in questions:
        payload = unpack_question_payload(capture, table_release_capture, question, target_id=target_id, audit_marker=marker)
        try:
            body, evidence = VisualAPI.request(client, payload, capture, 'verify_unpacking',
                                              lambda body, evidence: (body, evidence))
        except VisualDecisionError as error:
            error.evidence['completed_questions'] = merge_decision_evidence(records, capture, 'verify_unpacking')
            raise
        answers.extend(body.get('answers', []))
        records.append({**evidence, 'answers': body.get('answers', [])})
    return parse({'model': 'gpt-6-luna', 'answers': answers},
                 merge_decision_evidence(records, capture, 'verify_unpacking'))

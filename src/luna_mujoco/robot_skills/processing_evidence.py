"""Fresh visual evidence for the configured simulation unpack capability."""
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from ..decision_conditions import condition, parse_answers, result_answers
from ..observation_requirements import assessment
from ..visual_api import VisualAPI, VisualInventoryError
from ..visual_inventory import image_part
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
    return [condition('unpack_result', "In this camera's CURRENT view, a black workpiece is visible on the main "
        'tabletop, outside both blue bins. It may be partly occluded by an open gripper; judge its exposed '
        'body in this image. Black robot components and workpieces inside a blue bin do not count.'),
        condition('wrapper_removed', 'Compare the pink wrapped task object resting on the tabletop in BEFORE with CURRENT. '
            'Its former tabletop area is visible and the intact pink wrapped object is no longer there. '
            'Pink or magenta robot housings are not the task wrapper. Occlusion does not establish removal; '
            'a still-visible intact target anywhere in CURRENT contradicts removal. The separate product may '
            'be at another tabletop position. This question checks the bound pink wrapper disappearance, '
            'not the product identity or how the action occurred.')]


def unpack_result_evidence(body, minimum_confidence):
    selected = [row for row in body.get('answers', []) if row.get('name') == 'unpack_result']
    return result_answers({'answers': selected}, 'unpack_result', minimum_confidence)


def unpack_result_context():
    return {
        'task': 'Unwrap the bound pink foam-wrapped workpiece and confirm its separate black inner part on the measured support.',
        'stage': 'BEFORE images show the verified pink workpiece placed on the main table. '
                 'CURRENT images were taken after the unpacking action. Its result must be checked visually; '
                 'the action record alone does not establish success.',
        'views': 'One camera is supplied independently. BEFORE is historical; CURRENT is the scene to verify. '
                 'Judge only what this camera makes visible. Hidden or ambiguous subjects remain uncertain.',
        'scope': 'Judge workpieces on the main tabletop. Black robot or gripper parts and black workpieces '
                 'inside containers do not count.',
        'appearance': 'The expected black workpiece is an irregular solid part with protrusions, not '
                      'necessarily a rectangular block. A workpiece near or partly behind an open gripper '
                      'still counts if its exposed body is distinguishable from the robot. Do not require '
                      'a visible air gap between the workpiece and fingers.',
        'required_outcome': 'The pink foam has been removed and its black inner workpiece is separately '
            'on the main tabletop. The assigned execution arm must also reach its measured location; '
            'that geometric requirement is checked separately with calibrated RGBD and the robot workspace. '
            'Do not guess metric reachability from RGB. Report insufficient observation when the visual '
            'requirements cannot be confirmed or ruled out; the policy graph must explicitly obtain another view before reconfirmation.',
    }


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
             and type(confidence) in (int, float) and client.minimum_confidence <= confidence <= 1.
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
    context = dict(unpack_result_context(), observation_id=capture['observation_id'], audit_marker=marker)
    questions = unpack_result_questions()
    def parse(body, evidence):
        answers = body.get('answers')
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise VisualInventoryError('visual_decisions_invalid_response')
        judgments = {}
        for answer, question in zip(answers, questions, strict=True):
            if answer == {'type': 'refusal', 'name': question['name']}:
                judgments[question['name']] = 'unknown'
            else:
                current, _ = parse_answers({'answers': [answer]}, [question], client.minimum_confidence)
                judgments.update(current)
        result_answer = next(answer for answer in answers if answer['name'] == 'unpack_result')
        views = {'judgments': {'unpack_result': judgments['unpack_result']},
                 'answers': [deepcopy(result_answer)], 'confidence': result_answer.get('probability'),
                 'confidence_rule': 'unpack_result_predicate_probability', 'view_scope': 'single_current_rgb'}
        # These questions already require visible current evidence. A second
        # meta-question about the same evidence can contradict both concrete
        # answers without identifying any missing fact. Derive completeness
        # from the required facts; unknown or failed facts still cannot pass.
        observation = assessment(judgments, {'unpack_result': 'yes', 'wrapper_removed': 'yes'},
                                 capture['observation_id'], require_sufficiency=False)
        observation['sufficiency_source'] = 'required_visual_criteria'
        success = observation['status'] == 'confirmed'
        visual = {**evidence, **views, 'status': 'confirmed' if success else 'unknown',
                  'target_id': target_id, 'action_id': action_evidence['action_id'],
                  'identity_source': 'verified_single_active_execution_chain', 'visibility_source': 'openai_decisions',
                  'identity_scope': 'process_input_only_not_product_identity',
                  'execution_chain': {key: deepcopy(value) for key, value in scope.items()
                                      if key != 'table_release_reference'},
                  'table_release_request_hash': table['request_hash'], 'table_release_binding_id': table['binding_id'],
                  'table_release_reference': deepcopy(table_release_reference),
                  'verification_capture': deepcopy(verification_capture),
                  'work_area_zoom': None, 'observation_assessment': observation,
                  'camera_id': evidence['camera_ids'][0], 'required_judgments': judgments}
        return {**evidence, 'success': success, 'status': visual['status'], 'target_id': target_id,
                'generation': capture['generation'], 'appearance': 'unpacked' if success else 'unconfirmed',
                'action_id': action_evidence['action_id'], 'visual_evidence': visual,
                'verification_capture': deepcopy(verification_capture),
                'observation_assessment': observation,
                'response': deepcopy(body), 'geometry_updated': False}

    def camera_result(camera):
        content = [{'type': 'input_text', 'text': json.dumps({**context, 'camera_id': camera})},
            {'type': 'input_text', 'text': 'BEFORE UNPACKING, AFTER VERIFIED TABLE RELEASE RGB camera ' + camera},
            image_part(table_release_capture['views'][camera]['rgb']),
            {'type': 'input_text', 'text': 'CURRENT RGB camera ' + camera},
            image_part(capture['views'][camera]['rgb'])]
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': questions}
        scoped_capture = {**capture, 'views': {camera: capture['views'][camera]}}
        return VisualAPI.request(client, payload, scoped_capture, 'verify_unpacking', parse)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(camera_result, ('1', '2', '3')))
    selected = max(results, key=lambda row: (row['success'], row['visual_evidence']['confidence'] or -1.))
    result = deepcopy(selected)
    result['visual_evidence'].update(view_scope='independent_current_three_rgb',
        aggregation='one_complete_camera_witness_for_both_required_facts',
        selected_camera_id=selected['visual_evidence']['camera_id'],
        camera_evidence=[{key: row['visual_evidence'][key] for key in
            ('camera_id', 'request_hash', 'observation_id', 'required_judgments', 'confidence', 'status')}
            | {'answers': deepcopy(row['response']['answers'])} for row in results])
    return result

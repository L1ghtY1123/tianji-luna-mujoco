"""Typed semantic/visual predicate judgments, always using the Decisions endpoint."""
import json
import math
from copy import deepcopy

from ..core.decisions import ChoiceRequest, DecisionQuery, JudgmentRequest, choice_value, predicate_status
from ..visual_api import VisualAPI, VisualDecisionError, merge_decision_evidence
from .decision_state import judgment_state


class DecisionsJudgments(VisualAPI):
    endpoint = '/v1/decisions'
    source = 'openai_decisions'
    error_prefix = 'decisions_judgment'

    def judge(self, capture, *, requests, context, reference=None):
        requests = [request if isinstance(request, JudgmentRequest) else
                    ChoiceRequest.model_validate(request) if 'choices' in request else
                    JudgmentRequest.model_validate(request) for request in requests]
        names = [request.name for request in requests]
        if not names or len(names) != len(set(names)):
            raise VisualDecisionError('decisions_invalid_requests')
        if len(requests) > 1:
            judgments, records = {}, []
            for request in requests:
                result = self.judge(capture, requests=[request], context=context, reference=reference)
                judgments.update(result['judgments'])
                records.append(result['evidence'])
            return {'judgments': judgments,
                    'evidence': merge_decision_evidence(records, capture, 'semantic_judgments')}
        from ..raw_visual_input import capture_content, gripper_detail_content, reference_content

        context = deepcopy(context)
        if requests[0].name in {'object_held', 'failure_gripper_holds'}:
            from ..robot_skills.grasp_verification import gripper_feedback_state

            arms = ('A', 'B') if requests[0].name == 'failure_gripper_holds' else (context['selected_arm'],)
            context['gripper_feedback'] = [gripper_feedback_state(capture, arm) for arm in arms]
        query = DecisionQuery.model_validate_json(json.dumps({
            'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
            'question': requests[0].model_dump(mode='json'),
            'context': judgment_state(context, requests[0])}, allow_nan=False))
        primary = context.get('primary_wrist_camera_id')
        modalities = tuple(context.get('visual_modalities', ('rgb',)))
        content = [{'type': 'input_text', 'text': json.dumps({
            'observation_id': capture['observation_id'], 'context': query.context,
            'reference_entities': {key: {field: value for field, value in entity.items()
                if field in {'id', 'kind', 'color', 'description', 'reference_camera_id',
                             'confirmed_observation_id', 'reference_observation_id'}}
                for key, entity in (reference or {}).get('entities', {}).items()},
            'reference_entities_scope': 'Dated identity descriptions. Image metadata binds each drawn reference label '
                'to its capture and camera. Former locations in descriptions are historical.',
            'rules': 'Answer only this question from the supplied evidence. Each image has adjacent camera, capture and modality metadata. '
                'Full views use the effective workspace scope. Crops, when supplied, identify their source pixel rectangle. '
                'Masks and depth, when supplied, are separate images with their camera/time and encoding. '
                'Task instructions describe a request, not an observed outcome. Reference images are dated. '
                'Yellow labelled boxes identify the queried subjects. Workspace-masked regions are unobserved. '
                'Dark or black object surfaces remain valid visual evidence. '},
            ensure_ascii=False)}]
        visual_capture = capture
        camera_ids = context.get('visual_camera_ids')
        if camera_ids is not None:
            if not camera_ids or any(camera not in capture['views'] for camera in camera_ids):
                raise VisualDecisionError('decisions_required_camera_missing')
            visual_capture = {**capture, 'views': {camera: capture['views'][camera] for camera in camera_ids}}
        if requests[0].name in {'object_held', 'failure_gripper_holds'}:
            for arm in arms:
                camera = {'A': '2', 'B': '1'}[arm]
                detail = gripper_detail_content(capture, arm)
                content.extend(detail or capture_content(
                    {**capture, 'views': {camera: capture['views'][camera]}}, modalities=('rgb',)))
        else:
            content.extend(capture_content(visual_capture, modalities=modalities,
                masks=context.get('visual_masks'), boxes=context.get('visual_boxes'), camera_order=(primary,) if primary else (),
                subject_crops=context.get('subject_crops', False)))
        content.extend(reference_content(reference if requests[0].name not in {'object_held', 'failure_gripper_holds'} else None,
            modalities=tuple(context.get('reference_modalities', ('rgb',))), current_capture=capture,
            subject_crops=context.get('verification_scope') == 'grasp_result'))
        image_capture = visual_capture
        questions = []
        for request in requests:
            question = {'type': 'predicate', 'name': request.name, 'instructions': request.instructions}
            if isinstance(request, ChoiceRequest):
                question.update(type='choice', choices=[{'value': value,
                    **({'description': value + '. ' + request.descriptions[value]} if value in request.descriptions else {})}
                    for value in request.choices])
            questions.append(question)
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': questions}

        def parse(body, evidence):
            answers = body.get('answers')
            if not isinstance(answers, list) or len(answers) != len(requests):
                raise VisualDecisionError('decisions_invalid_answer_count')
            judgments = {}
            for request, answer in zip(requests, answers, strict=True):
                expected_type = 'choice' if isinstance(request, ChoiceRequest) else 'predicate'
                if (isinstance(answer, dict) and answer.get('name') == request.name
                        and answer.get('type') == 'refusal'):
                    judgments[request.name] = {'status': 'unknown', 'reason': 'model_refusal'}
                    if isinstance(request, ChoiceRequest):
                        judgments[request.name]['value'] = 'unknown'
                    for check in evidence.get('diagnostic_checks', []):
                        if check.get('name') == request.name:
                            check.update(status='unknown', reason='model_refusal')
                    continue
                if (not isinstance(answer, dict) or answer.get('name') != request.name
                        or answer.get('type') != expected_type):
                    raise VisualDecisionError('decisions_invalid_answer')
                probability = answer.get('confidence' if expected_type == 'choice' else 'probability')
                if (expected_type == 'predicate' and (type(probability) not in {int, float} or not math.isfinite(probability)
                        or not 0 <= probability <= 1)):
                    raise VisualDecisionError('decisions_invalid_answer_probability')
                if isinstance(request, ChoiceRequest):
                    try:
                        value = choice_value(answer, request.choices)
                    except ValueError as error:
                        raise VisualDecisionError('decisions_invalid_answer:' + str(error)) from None
                    if request.selection_mode == 'proposal':
                        # A proposed operation is not a certified world fact.
                        # Execution must independently validate its preconditions.
                        status = 'unknown' if value == 'unknown' else 'proposed'
                        judgments[request.name] = {'status': status, 'value': value,
                            'confidence': probability, 'selection_mode': 'proposal'}
                        for check in evidence.get('diagnostic_checks', []):
                            if check.get('name') == request.name:
                                check.update(status=status, metric='model_operation_proposal',
                                    actual=value, threshold=None, comparison='explicit_allowed_choice',
                                    confidence=probability, safety_certified=False,
                                    reason='model_selected_unknown' if value == 'unknown' else 'operation_proposal')
                        continue
                    known = value != 'unknown'
                    judgments[request.name] = {'status': 'known' if known else 'unknown',
                                              'value': value if known else 'unknown', 'confidence': probability}
                    continue
                status = predicate_status(probability)
                judgments[request.name] = {'status': status, 'probability': probability}
            for check in evidence.get('diagnostic_checks', []):
                if check['name'] in judgments:
                    judgments[check['name']]['reason'] = check['reason']
            return {'judgments': judgments, 'evidence': {**evidence, 'answers': answers}}
        return self.request(payload, image_capture, 'semantic_judgments', parse)

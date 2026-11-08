"""Typed semantic/visual predicate judgments, always using the Decisions endpoint."""
import json
import math
from copy import deepcopy

from ..core.decisions import ChoiceRequest, DecisionQuery, JudgmentRequest
from ..observation_labels import outline_masks, raster_visual_annotations
from ..task_binding_images import current_instance_location_content
from ..visual_api import VisualAPI, VisualInventoryError, merge_decision_evidence
from ..visual_inventory import image_part, product_reference_content, reference_content
from .completion import COMPLETION_HISTORY_RULE
from .decision_state import judgment_state


class DecisionsJudgments(VisualAPI):
    endpoint = '/v1/decisions'
    source = 'openai_decisions'
    error_prefix = 'decisions_judgment'

    def __init__(self, *, minimum_confidence=.8, **kwargs):
        super().__init__(**kwargs)
        if not .5 < minimum_confidence <= 1.:
            raise ValueError('decisions_invalid_confidence')
        self.minimum_confidence = minimum_confidence

    def judge(self, capture, *, requests, context, reference=None):
        requests = [request if isinstance(request, JudgmentRequest) else
                    ChoiceRequest.model_validate(request) if 'choices' in request else
                    JudgmentRequest.model_validate(request) for request in requests]
        names = [request.name for request in requests]
        if not names or len(names) != len(set(names)):
            raise VisualInventoryError('decisions_invalid_requests')
        if len(requests) > 1:
            judgments, records = {}, []
            for request in requests:
                result = self.judge(capture, requests=[request], context=context, reference=reference)
                judgments.update(result['judgments'])
                records.append(result['evidence'])
            return {'judgments': judgments,
                    'evidence': merge_decision_evidence(records, capture, 'semantic_judgments')}
        context = deepcopy(context)
        reference = deepcopy(reference)
        if context.get('verification_scope') == 'current_support':
            from .support import scope_support_evidence

            scope_support_evidence(context, reference, requests[0])
        query = DecisionQuery.model_validate_json(json.dumps({
            'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
            'question': requests[0].model_dump(mode='json'),
            'context': judgment_state(context, requests[0])}, allow_nan=False))
        scope = context['verification_scope']
        numeric_proposal = scope == 'placement_xy_proposal'
        language_only = scope == 'placement_region_scope' and requests[0].name == 'placement_additional_spatial_constraint'
        if numeric_proposal and (context.get('placement_zone', {}).get('semantic_scope') != 'entire_region'
                or any(not isinstance(request, ChoiceRequest) or request.selection_mode != 'proposal'
                       or request.name != 'placement_xy' for request in requests)):
            raise VisualInventoryError('placement_xy_requires_authorized_region_proposal')
        content = [{'type': 'input_text', 'text': json.dumps({
            'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
            'context': query.context,
            **({'operation_proposals': {request.name:
                'Propose one allowed operation; report confidence honestly. This is not factual or safety proof. '
                'Its numerical preconditions will be checked before execution. Choose unknown to abstain.'
                for request in requests if isinstance(request, ChoiceRequest) and request.selection_mode == 'proposal'}}
               if any(isinstance(request, ChoiceRequest) and request.selection_mode == 'proposal' for request in requests)
               else {}),
            'reference_entities': {identifier: {key: value for key, value in entity.items()
                if key in {'id', 'kind', 'color', 'description', 'appearance_keywords', 'reference_observation_id', 'image_label'}}
                for identifier, entity in (reference or {}).get('entities', {}).items()},
            'rules': 'Judge the single stated fact or choice. Use current RGB for current appearance, '
                     'the original instruction for language, and supplied contracts for operation constraints. '
                     'reference_entities are dated identity records, not current locations. '
                     'task and object_task describe requested outcomes, not observed facts. '
                     'Confirmed identities and dated history retain their recorded scope. SAM names are proposals. '
                     'A predicate reports P(statement is true); contrary evidence gives a low probability. '
                     'Use uncertainty for missing or ambiguous evidence. Operation proposals require no '
                     'proof of future success. Numerical skills check motion safety separately.'
                     + (' Dated verified support_memory is established historical evidence, not a proposal. '
                        'Use it with current measurements and recorded actions to assess continuity. '
                        'A missing current view alone does not erase a previously observed location; '
                        'retaining a relation does not mean it is currently visible or geometrically safe.'
                        if context.get('verification_scope') == 'current_support' else '')
                     + (COMPLETION_HISTORY_RULE if context.get('verification_scope') in
                        {'task_completion', 'policy_condition'}
                        and requests[0].name != 'source_has_remaining_input' else '')},
            ensure_ascii=False)}]
        image_capture, image_reference = capture, reference
        if language_only:
            introduction = json.loads(content[0]['text'])
            introduction.pop('reference_entities', None)
            introduction['rules'] = (
                'Read original_requirement and current_transfer to interpret only the requested placement clause. '
                'The requirements state what is requested, not what has happened. This language question '
                'does not need images, current poses, motion feasibility or completion history.')
            content[0]['text'] = json.dumps(introduction, ensure_ascii=False)
            image_capture, image_reference = {**capture, 'views': {}}, None
        if numeric_proposal:
            # Language and destination identity were already established by the
            # region skill. Requiring current pixels again contradicts this
            # operation's role, especially when the object is now in the hand.
            introduction = json.loads(content[0]['text'])
            introduction.pop('reference_entities', None)
            introduction['rules'] = (
                'Choose among the supplied XY proposals in the explicitly authorized region. '
                'The prior region judgment establishes destination identity and absence of further spatial '
                'language restrictions for this transfer. Do not rejudge the whole task or current visibility. '
                'No images are needed for this numerical proposal. For placement_preference=clearance, '
                'prefer greater measured_obstacle_gap_m capped at desired_clearance_m, then smaller travel_xy_m. '
                'For preference=reach, prefer smaller travel_xy_m. A null gap is unavailable evidence. '
                'Measured horizontal separation is a preference and needs subsequent full collision checking. '
                'Unknown remains available for inconsistent or uninterpretable '
                'inputs. The program checks Z, collision, workspace, held IK and trajectory afterward; '
                'this choice establishes none of those facts and does not mean placement is complete.')
            content[0]['text'] = json.dumps(introduction, ensure_ascii=False)
            image_capture, image_reference = {**capture, 'views': {}}, None
        # Recovery and registration compare dated subjects against current views.
        if scope in {'graph_recovery', 'instance_registration'}:
            references = reference_content(image_reference, current_capture=image_capture)
            for index, part in enumerate(references[:-1]):
                if part['type'] != 'input_text':
                    continue
                try:
                    row = json.loads(part['text'])
                except ValueError:
                    continue
                if row.get('image_role') in {'initial_reference_annotations', 'instance_reference'} and row.get('items'):
                    content.extend([part, references[index + 1]])
            content.extend(product_reference_content(image_reference, image_capture,
                (image_reference or {}).get('entities', {})))
        projected = context.get('candidate_projection', {})
        if projected and any(projected.get(key) != capture.get(key) for key in ('observation_id', 'generation')):
            raise VisualInventoryError('placement_candidate_capture_stale')
        table_regions = []
        if scope == 'current_support':
            from .support import support_table_regions

            table_regions = support_table_regions(image_capture, context)
        locations = (current_instance_location_content(image_capture, context.get('current_instance_locations', {}),
                                                       table_regions=table_regions)
                     if image_capture['views'] else [])
        located = {json.loads(locations[index]['text'])['camera_id']: locations[index:index + 2]
                   for index in range(0, len(locations), 2)}
        cameras = list(image_capture['views'])
        if scope == 'current_support':
            from .support import support_camera_order
            cameras = support_camera_order(cameras, context.get('current_instance_locations', {}),
                context['subject']['id'], context['support_candidates'], table_regions=table_regions)
        detail_camera = None
        if scope == 'current_support':
            subject_id = context['subject']['id']
            visible = [(camera, item) for camera, view in context.get('current_instance_locations', {}).get('views', {}).items()
                       for item in view['items'] if item['id'] == subject_id and camera in image_capture['views']]
            if visible:
                detail_camera, detail_item = max(visible, key=lambda row:
                    (row[1]['box'][2] - row[1]['box'][0]) * (row[1]['box'][3] - row[1]['box'][1])
                    * image_capture['views'][row[0]]['rgb'].shape[0] * image_capture['views'][row[0]]['rgb'].shape[1])
        for camera in cameras:
            view = image_capture['views'][camera]
            regions = projected.get('projections', {}).get(camera, [])
            if regions:
                content.extend([{'type': 'input_text', 'text': json.dumps({
                    'image_role': 'placement_candidate_proposals', 'camera_id': camera,
                    'observation_id': capture['observation_id'],
                    'meaning': 'CURRENT RGB with hypothetical placement footprints. Judge the named language '
                               'constraint. These outlines do not establish occupancy or motion safety.',
                    'candidate_ids': [region['region_id'] for region in regions]})},
                    image_part(raster_visual_annotations(view['rgb'], [], regions=regions))])
            elif scope == 'instance_registration' and camera == context['proposal']['reference_camera_id']:
                proposed = {'id': 'proposal', 'image_label': 'P', 'box': context['proposal']['bbox']}
                mask = context.get('proposal_mask')
                rgb = view['rgb'] if mask is None else outline_masks(view['rgb'], [mask])
                content.extend([{'type': 'input_text', 'text': json.dumps({
                    'image_role': 'current_instance_proposal', 'camera_id': camera,
                    'observation_id': capture['observation_id'], 'subject_label': 'P',
                    'box': proposed['box'], 'mask_contours_supplied': mask is not None,
                    'meaning': 'P locates the CURRENT proposed physical subject. '
                        'Its identity is being judged. An enclosing container or supporting surface '
                        'is a distinct physical entity; reference labels R identify the comparison instances.'})},
                    image_part(raster_visual_annotations(rgb, [proposed], label_layout='near_box'))])
            elif camera in located:
                content.extend(located[camera])
            else:
                content.extend([{'type': 'input_text', 'text': 'CURRENT RGB camera ' + camera}, image_part(view['rgb'])])
            if camera == detail_camera:
                from ..task_binding_images import current_subject_detail_content

                content.extend(current_subject_detail_content(view['rgb'], detail_item['box'],
                                                              detail_item['image_label'], camera))
        questions = []
        for request in requests:
            question = {'type': 'predicate', 'name': request.name, 'instructions': request.instructions}
            if (scope == 'current_support' and not isinstance(request, ChoiceRequest)
                    and context['subject'].get('image_label') == 'A'):
                support = next(iter(context['support_candidates'].values()))
                relation = ('inside container B, including contents extending above its rim'
                            if support['kind'] == 'bin' else 'resting on the tabletop marked B')
                question['instructions'] = ('The object marked A is ' + relation + '. '
                    'Assess A using its detail crop, current views, measured geometry and dated observations. '
                    'Other marked objects are separate physical subjects.')
            if context.get('verification_scope') == 'task_completion' and request.name == 'source_has_remaining_input':
                task = context['task']
                source = (reference or {}).get('entities', {}).get(task.get('source_ref'), {})
                question['instructions'] = (
                    'Source ' + source.get('description', str(task.get('source_ref')))
                    + '. Original input selector ' + task['target_selector'] + '. '
                    + request.instructions + ' Use the three CURRENT camera images.')
            if isinstance(request, ChoiceRequest):
                question.update(type='choice', choices=[{'value': value} for value in request.choices])
            questions.append(question)
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': questions}

        def parse(body, evidence):
            answers = body.get('answers')
            if not isinstance(answers, list) or len(answers) != len(requests):
                raise VisualInventoryError('decisions_invalid_answer_count')
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
                    raise VisualInventoryError('decisions_invalid_answer')
                probability = answer.get('confidence' if expected_type == 'choice' else 'probability')
                if (type(probability) not in {int, float} or not math.isfinite(probability)
                        or not 0 <= probability <= 1):
                    raise VisualInventoryError('decisions_invalid_answer_probability')
                if isinstance(request, ChoiceRequest):
                    value = answer.get('choice')
                    if value not in request.choices:
                        raise VisualInventoryError('decisions_invalid_answer_choice')
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
                    known = value != 'unknown' and probability >= self.minimum_confidence
                    judgments[request.name] = {'status': 'known' if known else 'unknown',
                                              'value': value if known else 'unknown', 'confidence': probability}
                    continue
                status = ('yes' if probability >= self.minimum_confidence else 'no'
                          if probability <= round(1 - self.minimum_confidence, 12) else 'unknown')
                judgments[request.name] = {'status': status, 'probability': probability}
            for check in evidence.get('diagnostic_checks', []):
                if check['name'] in judgments:
                    judgments[check['name']]['reason'] = check['reason']
            return {'judgments': judgments, 'evidence': {**evidence, 'answers': answers}}
        return self.request(payload, image_capture, 'semantic_judgments', parse)

"""Typed semantic/visual predicate judgments, always using the Decisions endpoint."""
import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..observation_labels import raster_visual_annotations
from ..task_binding_images import current_instance_location_content
from ..visual_api import VisualAPI, VisualInventoryError
from ..visual_inventory import image_part, product_reference_content, reference_content
from .completion import COMPLETION_HISTORY_RULE


class JudgmentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    name: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,63}$')
    instructions: str = Field(min_length=1)


class ChoiceRequest(JudgmentRequest):
    choices: tuple[str, ...] = Field(min_length=2, max_length=64)
    selection_mode: Literal['evidence', 'proposal'] = 'evidence'

    @model_validator(mode='after')
    def distinct_choices(self):
        if len(set(self.choices)) != len(self.choices) or 'unknown' not in self.choices:
            raise ValueError('choices_require_distinct_values_and_unknown')
        return self


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
        numeric_proposal = context.get('verification_scope') == 'placement_xy_proposal'
        if numeric_proposal and (context.get('placement_zone', {}).get('semantic_scope') != 'entire_region'
                or any(not isinstance(request, ChoiceRequest) or request.selection_mode != 'proposal'
                       or request.name != 'placement_xy' for request in requests)):
            raise VisualInventoryError('placement_xy_requires_authorized_region_proposal')
        if len(requests) > 200:
            judgments, batches = {}, []
            for start in range(0, len(requests), 200):
                try:
                    result = self.judge(capture, requests=requests[start:start + 200],
                                        context=context, reference=reference)
                except VisualInventoryError as error:
                    error.evidence = {**error.evidence, 'completed_batches': batches}
                    raise
                judgments.update(result['judgments'])
                batches.append(result['evidence'])
            digest = hashlib.sha256(json.dumps([batch['request_hash'] for batch in batches]).encode()).hexdigest()
            return {'judgments': judgments, 'evidence': {'source': self.source,
                'component': 'decisions', 'stage': 'semantic_judgments', 'purpose': 'semantic_judgments',
                'camera_ids': list(capture['views']),
                'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
                'diagnostic_checks': [check for batch in batches for check in batch.get('diagnostic_checks', [])],
                'request_hash': digest, 'batches': batches}}
        content = [{'type': 'input_text', 'text': json.dumps({
            'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
            'context': {key: value for key, value in context.items() if key != 'current_instance_locations'},
            **({'operation_proposals': {request.name:
                'Propose one allowed operation; report confidence honestly. This is not factual or safety proof. '
                'Its numerical preconditions will be checked before execution. Choose unknown to abstain.'
                for request in requests if isinstance(request, ChoiceRequest) and request.selection_mode == 'proposal'}}
               if any(isinstance(request, ChoiceRequest) and request.selection_mode == 'proposal' for request in requests)
               else {}),
            'reference_entities': {identifier: {key: value for key, value in entity.items()
                if key in {'id', 'kind', 'color', 'description', 'appearance_keywords', 'initial_parent_id', 'display_id'}}
                for identifier, entity in (reference or {}).get('entities', {}).items()},
            'rules': 'Answer each question only within its stated scope. '
                     'Use CURRENT RGB for current visual facts, the original instruction for language meaning, '
                     'and supplied configuration/contracts for settings and legal operation choices. '
                     'Explicitly confirmed identity, CAD-model and historical relation records are established '
                     'within their recorded scope; do not ask for those decisions again. Historical poses are '
                     'not current measurements. Unconfirmed inventory labels and SAM proposals remain proposals. '
                     'For a placement-location question, judge this transfer location, not CAD selection, '
                     'processing completion, later observation or final robot return. '
                     'Do not require current visual proof for language interpretation or a strategy proposal. '
                     'Choose one allowed strategy even if several are suitable; it does not certify numerical '
                     'safety or success. Preserve uncertainty when evidence needed by this specific question '
                     'is missing or contradictory. Do not change the task or invent metric geometry.'
                     + (' Dated verified support_memory is established historical evidence, not a proposal. '
                        'Use it with current measurements and recorded actions to assess continuity. '
                        'A missing current view alone does not erase a previously observed location; '
                        'retaining a relation does not mean it is currently visible or geometrically safe.'
                        if context.get('verification_scope') == 'current_support' else '')
                     + (COMPLETION_HISTORY_RULE if context.get('verification_scope') in
                        {'task_completion', 'policy_condition'} else '')},
            ensure_ascii=False)}]
        image_capture, image_reference = capture, reference
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
                'No images are needed for this numerical proposal. Prefer smaller travel_xy_m when candidates '
                'are otherwise equivalent. Unknown remains available for inconsistent or uninterpretable '
                'inputs. The program checks Z, collision, workspace, held IK and trajectory afterward; '
                'this choice establishes none of those facts and does not mean placement is complete.')
            content[0]['text'] = json.dumps(introduction, ensure_ascii=False)
            image_capture, image_reference = {**capture, 'views': {}}, None
        content.extend(reference_content(image_reference, current_capture=image_capture))
        content.extend(product_reference_content(image_reference, image_capture, (image_reference or {}).get('entities', {})))
        projected = context.get('candidate_projection', {})
        if projected and any(projected.get(key) != capture.get(key) for key in ('observation_id', 'generation')):
            raise VisualInventoryError('placement_candidate_capture_stale')
        for camera, view in image_capture['views'].items():
            label = ('HISTORICAL SCENE RGB, not current object/occupancy proof, camera '
                     if context.get('image_evidence_scope') == 'historical_scene_for_xy_proposal'
                     else 'CURRENT RGB camera ')
            content.extend([{'type': 'input_text', 'text': label + camera}, image_part(view['rgb'])])
            regions = projected.get('projections', {}).get(camera, [])
            if regions:
                content.extend([{'type': 'input_text', 'text': json.dumps({
                    'image_role': 'placement_candidate_proposals', 'camera_id': camera,
                    'observation_id': capture['observation_id'],
                    'meaning': 'The same current RGB with proposed footprint outlines computed from '
                               'the bound observed OBB or registered CAD geometry. '
                               'These are hypothetical placements, not observed objects or free-space proof.',
                    'candidate_ids': [region['slot_id'] for region in regions]})},
                    image_part(raster_visual_annotations(view['rgb'], [], regions=regions))])
        content.extend(current_instance_location_content(image_capture, context.get('current_instance_locations', {})))
        questions = []
        for request in requests:
            question = {'type': 'predicate', 'name': request.name, 'instructions': request.instructions}
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
                                    confidence=probability, safety_certified=False)
                        continue
                    known = value != 'unknown' and probability >= self.minimum_confidence
                    judgments[request.name] = {'status': 'known' if known else 'unknown',
                                              'value': value if known else 'unknown', 'confidence': probability}
                    continue
                status = ('yes' if probability >= self.minimum_confidence else 'no'
                          if probability <= round(1 - self.minimum_confidence, 12) else 'unknown')
                judgments[request.name] = {'status': status, 'probability': probability}
            return {'judgments': judgments, 'evidence': {**evidence, 'answers': answers}}
        return self.request(payload, image_capture, 'semantic_judgments', parse)

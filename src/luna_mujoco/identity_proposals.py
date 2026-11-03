"""Ask Decisions to verify unambiguous same-image correspondence proposals."""
import json
import math
from copy import deepcopy

from .visual_api import VisualInventoryError


def scope_same_capture_identity(payload):
    result = deepcopy(payload)
    context = json.loads(result['input'][0]['content'][0]['text'])
    proof = context.get('same_capture_reference', {})
    current = context.get('observation_id')
    if (not current or proof.get('initial_reference_observation_id') != current
            or proof.get('current_observation_id') != current
            or proof.get('generation') != context.get('generation')):
        return result, {}
    questions = {question['name']: question for question in result['questions']}
    proposals = {}
    for row in context.get('same_capture_box_correspondence_proposals', []):
        if row['camera'] not in proof.get('verified_camera_ids', []):
            continue
        ranked = sorted(row['same_camera_reference_box_overlaps'], key=lambda item: item['box_iou'], reverse=True)
        if not ranked or ranked[0]['box_iou'] < .85 or len(ranked) > 1 and ranked[1]['box_iou'] > .2:
            continue
        number = row['candidate'].removeprefix('candidate-')
        name = 'identity_' + number
        question = questions.get(name)
        if question is None or question['type'] != 'choice':
            continue
        identifier = ranked[0]['reference_entity_id']
        option = next((item for item in question['choices'] if item['value'] == identifier), None)
        if option is None:
            continue
        question.clear()
        question.update(type='predicate', name=name, instructions=(
            f"Does Candidate {number} in camera {row['camera']} depict the SAME physical subject as reference "
            f"{option['description']} in this same frozen capture? Use the RGB subject and surrounding "
            'objects. The overlap supplies a comparison proposal, not the answer. A container and its '
            'contents are different subjects. Reject a different or ambiguous subject; do not demand '
            'appearance differences between two stationary identical containers when their reference '
            'locations in these identical pixels distinguish them.'))
        proposals[name] = identifier
    return result, proposals


def resolve_identity_proposals(body, proposals, minimum_confidence):
    """Translate a positive predicate to its proposed ID; never infer an answer."""
    result = deepcopy(body)
    seen = set()
    for answer in result.get('answers', []):
        name = answer.get('name')
        if name not in proposals:
            continue
        probability = answer.get('probability')
        if (name in seen or answer.get('type') != 'predicate'
                or type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0. <= probability <= 1.):
            raise VisualInventoryError('visual_decisions_invalid_response')
        seen.add(name)
        answer.clear()
        answer.update(type='choice', name=name, confidence=probability,
                      choice=proposals[name] if probability >= minimum_confidence else 'unknown')
    if seen != set(proposals):
        raise VisualInventoryError('visual_decisions_invalid_response')
    return result

"""Parse observable predicates separately from categorical identity confidence."""
import math
from copy import deepcopy

from .visual_api import VisualInventoryError


def condition(name, statement):
    return {'type': 'predicate', 'name': name, 'instructions': statement}


def accepted_identity(answer, threshold):
    """Read an accepted identity while retaining the original model response type."""
    if answer.get('type') == 'predicate':
        identifier, score = answer.get('identity_candidate'), answer.get('probability')
    elif answer.get('type') == 'choice':
        identifier, score = answer.get('choice'), answer.get('confidence')
    else:
        return None
    if (isinstance(identifier, str) and identifier not in {'unknown', 'reject'}
            and type(score) in (int, float) and math.isfinite(score) and threshold <= score <= 1.):
        return identifier
    return None


def parse_answers(body, questions, threshold):
    answers = body.get('answers')
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise VisualInventoryError('visual_decisions_invalid_response')
    judgments = {}
    for question, answer in zip(questions, answers, strict=True):
        if not isinstance(answer, dict) or answer.get('name') != question['name']:
            raise VisualInventoryError('visual_decisions_invalid_response')
        if answer.get('type') == 'refusal':
            judgments[question['name']] = 'unknown'
            continue
        if answer.get('type') != question['type']:
            raise VisualInventoryError('visual_decisions_invalid_response')
        field = 'probability' if question['type'] == 'predicate' else 'confidence'
        value = answer.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise VisualInventoryError('visual_decisions_invalid_response')
        if question['type'] == 'predicate':
            judgment = 'yes' if value >= threshold else 'no' if value <= round(1-threshold, 12) else 'unknown'
        else:
            if answer.get('choice') not in [item['value'] for item in question['choices']]:
                raise VisualInventoryError('visual_decisions_invalid_response')
            judgment = answer['choice'] if value >= threshold else 'unknown'
        judgments[question['name']] = judgment
    return judgments, deepcopy(answers)


def result_answers(body, name, threshold):
    """Current result checks require predicate responses, including during recomputation."""
    judgments, answers = parse_answers(body, [condition(name, '')], threshold)
    return {'judgments': judgments, 'answers': answers, 'confidence': answers[0].get('probability'),
            'confidence_rule': name + '_predicate_probability',
            'view_scope': 'fused_current_three_rgb'}

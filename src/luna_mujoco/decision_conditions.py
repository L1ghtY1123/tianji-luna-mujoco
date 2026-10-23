"""Parse observable predicates separately from categorical identity confidence."""
import math
from copy import deepcopy

from .visual_api import VisualInventoryError


def condition(name, statement):
    return {'type': 'predicate', 'name': name, 'instructions': statement}


def parse_answers(body, questions, threshold):
    answers = body.get('answers')
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise VisualInventoryError('visual_decisions_invalid_response')
    judgments = {}
    for question, answer in zip(questions, answers, strict=True):
        if (not isinstance(answer, dict) or answer.get('name') != question['name']
                or answer.get('type') != question['type']):
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
    """Read current results and explicitly retained historical choice records."""
    rows = body.get('answers')
    legacy = isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict) and rows[0].get('type') == 'choice'
    question = ({'type': 'choice', 'name': name, 'choices': [{'value': v} for v in ('yes', 'no', 'unknown')]}
                if legacy else condition(name, ''))
    judgments, answers = parse_answers(body, [question], threshold)
    field = 'confidence' if legacy else 'probability'
    return {'judgments': judgments, 'answers': answers, 'confidence': answers[0][field],
            'confidence_rule': name + ('_answer_confidence' if legacy else '_predicate_probability'),
            'view_scope': 'fused_current_three_rgb'}

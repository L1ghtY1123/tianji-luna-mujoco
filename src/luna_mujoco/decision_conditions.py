"""Single interpretation of Decisions facts and categorical probability maxima."""
import math
from copy import deepcopy

from .core.decisions import choice_value, predicate_status
from .visual_api import VisualDecisionError


def condition(name, statement):
    return {'type': 'predicate', 'name': name, 'instructions': statement}


def accepted_identity(answer):
    """Read an accepted identity while retaining the original model response type."""
    if answer.get('identity_conflict') or answer.get('identity_rejected_by_workspace'):
        return None
    if answer.get('type') == 'predicate':
        identifier, score = answer.get('identity_candidate'), answer.get('probability')
        accepted = predicate_status(score) == 'yes'
    elif answer.get('type') == 'choice':
        identifier = choice_value(answer)
        if identifier == 'new_instance':
            identifier = answer.get('identity_candidate')
        accepted = True
    else:
        return None
    if (isinstance(identifier, str) and identifier not in {'unknown', 'reject', 'new_instance'}
            and accepted):
        return identifier
    return None


def parse_answers(body, questions):
    answers = body.get('answers')
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise VisualDecisionError('visual_decisions_invalid_response')
    judgments = {}
    for question, answer in zip(questions, answers, strict=True):
        if not isinstance(answer, dict) or answer.get('name') != question['name']:
            raise VisualDecisionError('visual_decisions_invalid_response')
        if answer.get('type') == 'refusal':
            judgments[question['name']] = 'unknown'
            continue
        if answer.get('type') != question['type']:
            raise VisualDecisionError('visual_decisions_invalid_response')
        if question['type'] == 'predicate':
            value = answer.get('probability')
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise VisualDecisionError('visual_decisions_invalid_response')
            judgment = predicate_status(value)
        else:
            try:
                judgment = choice_value(answer, [item['value'] for item in question['choices']])
            except ValueError as error:
                raise VisualDecisionError('visual_decisions_invalid_response:' + str(error)) from None
        judgments[question['name']] = judgment
    return judgments, deepcopy(answers)


def result_answers(body, name):
    """Current result checks require predicate responses, including during recomputation."""
    judgments, answers = parse_answers(body, [condition(name, '')])
    return {'judgments': judgments, 'answers': answers, 'confidence': answers[0].get('probability'),
            'confidence_rule': name + '_predicate_probability',
            'view_scope': 'fused_current_three_rgb'}

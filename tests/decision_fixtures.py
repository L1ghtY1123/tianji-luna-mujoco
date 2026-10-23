"""Build model test doubles with the actual question's wire type."""
def decision_answer(question, value='yes', confidence=.95):
    if question['type'] == 'predicate':
        probability = confidence if value == 'yes' else 1-confidence if value == 'no' else .5
        return {'name': question['name'], 'type': 'predicate', 'probability': probability}
    return {'name': question['name'], 'type': 'choice', 'choice': value, 'confidence': confidence}

"""Historic two-question replies cannot silently become a new fused result."""
import json
from pathlib import Path

import pytest

from luna_mujoco.demo_lifecycle_perception import drop_result_evidence
from luna_mujoco.demo_lifecycle_tracking import _visual_drop


def test_original_v12_two_answer_response_is_preserved_and_rejected_by_new_protocol():
    body = json.loads((Path(__file__).parent / 'fixtures/drop-v12-response.json').read_text('utf-8-sig'))
    assert [answer['confidence'] for answer in body['answers']] == [.79, 1.]
    with pytest.raises(ValueError, match='invalid_response'):
        drop_result_evidence(body, .8)


@pytest.mark.parametrize('mutation', [None, 'missing_rule', 'old_unknown', 'low_count', 'negative'])
def test_completed_inventory_rechecks_original_fused_drop_answer(mutation):
    body = {'answers': [{'name': 'current_item_added', 'type': 'choice', 'choice': 'yes', 'confidence': .95}]}
    visual = {**drop_result_evidence(body, .8), 'source': 'openai_decisions',
              'request_hash': 'joint-drop-reply', 'observation_id': 'after-drop'}
    if mutation == 'missing_rule':
        visual.pop('confidence_rule')
    elif mutation == 'old_unknown':
        visual['status'] = 'unknown'
    elif mutation == 'low_count':
        visual['answers'][0]['confidence'] = .79
    elif mutation == 'negative':
        visual['answers'][0].update(choice='no', confidence=.95)
    if mutation is None:
        _visual_drop(visual)
    else:
        with pytest.raises(ValueError):
            _visual_drop(visual)

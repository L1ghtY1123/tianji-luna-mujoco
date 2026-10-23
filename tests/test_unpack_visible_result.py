"""One fused visible-result judgment; execution history owns process provenance."""
import json

import pytest
from test_demo_lifecycle_perception import before_kwargs, capture_and_action, client, record, work_area

from luna_mujoco.demo_lifecycle_perception import unpack_result_evidence, verify_unpacking
from luna_mujoco.visual_api import VisualInventoryError


def run(answer=('yes', .91), **changes):
    capture, action = capture_and_action()
    visual, calls = client({'unpack_result': answer})
    kwargs = dict(target_id='pink', lifecycle=record('unpack_pending'), action_evidence=action,
        owner_id='run', work_area=work_area(), **before_kwargs())
    kwargs.update(changes)
    return verify_unpacking(visual, capture, **kwargs), calls


def test_combined_views_confirm_output_without_claiming_individual_camera_support():
    result, calls = run()
    visual = result['visual_evidence']
    assert result['success'] and result['target_id'] == 'pink'
    assert visual['judgments'] == {'unpack_result': 'yes'}
    assert visual['view_scope'] == 'fused_current_three_rgb'
    assert 'qualified_camera_ids' not in visual and 'conflicting_camera_ids' not in visual
    assert visual['confidence'] == .91
    assert visual['identity_source'] == 'verified_single_active_execution_chain'
    assert visual['execution_chain']['active_target_ids'] == ['pink']
    assert visual['execution_chain']['owner_id'] == 'run'
    payload = calls[0]
    assert sum(p['type'] == 'input_image' for p in payload['input'][0]['content']) == 2
    assert 'table_release_reference' not in json.dumps(payload)
    assert [q['name'] for q in payload['questions']] == ['unpack_result', 'wrapper_removed', 'observation_sufficient']


def test_unpack_question_allows_visible_body_partly_occluded_by_open_gripper():
    _, calls = run()
    payload = calls[0]
    content = payload['input'][0]['content']
    context = json.loads(content[0]['text'])
    question = payload['questions'][0]['instructions']
    assert 'partly occluded' in question
    assert 'outside both blue bins' in question
    assert 'not necessarily a rectangular block' in context['appearance']
    assert 'Do not require a visible air gap' in context['appearance']
    assert sum('BEFORE UNPACKING, AFTER VERIFIED TABLE RELEASE' in p.get('text', '') for p in content) == 3


@pytest.mark.parametrize('answer', [('no', .99), ('unknown', .99), ('yes', .799)])
def test_absent_ambiguous_or_low_confidence_result_never_advances(answer):
    result, _ = run(answer)
    assert not result['success'] and result['status'] == 'unknown'
    assert result['appearance_revision'] == 0


@pytest.mark.parametrize('active', [None, [], ['other'], ['pink', 'other'], ['pink', 'pink']])
def test_chain_requires_exactly_one_current_active_id(active):
    with pytest.raises(VisualInventoryError, match='single_active'):
        run(active_target_ids=active)


def test_old_per_camera_answers_cannot_be_upgraded_into_new_fused_confirmation():
    body = {'answers': [{'type': 'choice', 'name': f'camera_{camera}_{kind}',
                         'choice': 'yes' if kind == 'result' else 'no', 'confidence': 1.}
                        for camera in ('2', '1', '3') for kind in ('result', 'conflict')]}
    with pytest.raises(VisualInventoryError, match='invalid_response'):
        unpack_result_evidence(body, .8)


def test_choice_probability_does_not_replace_answer_confidence():
    body = {'answers': [{'type': 'choice', 'name': 'unpack_result', 'choice': 'yes',
        'confidence': .77, 'probabilities': [{'value': 'yes', 'probability': .99}]}]}
    assert unpack_result_evidence(body, .8)['judgments'] == {'unpack_result': 'unknown'}


@pytest.mark.parametrize('criterion', ['wrapper_removed', 'observation_sufficient'])
def test_visible_black_alone_cannot_bypass_missing_task_evidence(criterion):
    capture, action = capture_and_action()
    visual, calls = client({criterion: ('unknown', .95)})
    result = verify_unpacking(visual, capture, target_id='pink', lifecycle=record('unpack_pending'),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert not result['success']
    assert result['observation_assessment']['needs_observation']
    assert criterion in result['observation_assessment']['missing_evidence']
    assert len(calls) == 1

"""One complete camera witness; execution history owns process provenance."""
import json

import pytest
from processing_fixtures import before_kwargs, capture_and_action, client, release_evidence, work_area

from luna_mujoco.robot_skills.processing_evidence import unpack_result_evidence, verify_unpacking
from luna_mujoco.visual_api import VisualInventoryError


def run(answer=('yes', .91), **changes):
    capture, action = capture_and_action()
    visual, calls = client({'unpack_result': answer})
    kwargs = dict(target_id='pink', release_evidence=release_evidence(), action_evidence=action,
        owner_id='run', work_area=work_area(), **before_kwargs())
    kwargs.update(changes)
    return verify_unpacking(visual, capture, **kwargs), calls


def test_independent_views_retain_one_complete_camera_witness():
    result, calls = run()
    visual = result['visual_evidence']
    assert result['success'] and result['target_id'] == 'pink'
    assert visual['judgments'] == {'unpack_result': 'yes'}
    assert visual['view_scope'] == 'independent_current_three_rgb'
    assert len(calls) == 3
    assert {row['camera_id'] for row in visual['camera_evidence']} == {'1', '2', '3'}
    assert visual['selected_camera_id'] in {'1', '2', '3'}
    assert 'qualified_camera_ids' not in visual and 'conflicting_camera_ids' not in visual
    assert visual['confidence'] == .91
    assert visual['identity_source'] == 'verified_single_active_execution_chain'
    assert visual['execution_chain']['active_target_ids'] == ['pink']
    assert visual['execution_chain']['owner_id'] == 'run'
    payload = calls[0]
    assert sum(p['type'] == 'input_image' for p in payload['input'][0]['content']) == 1
    assert 'table_release_reference' not in json.dumps(payload)
    assert [q['name'] for q in payload['questions']] == ['unpack_result', 'wrapper_removed']
    assert result['observation_assessment']['sufficiency_source'] == 'required_visual_criteria'


def test_unpack_question_allows_visible_body_partly_occluded_by_open_gripper():
    _, calls = run()
    payload = calls[0]
    content = payload['input'][0]['content']
    context = json.loads(content[0]['text'])
    question = payload['questions'][0]['instructions']
    assert 'partly occluded' in question
    assert 'using the views together' not in question
    assert "this camera's CURRENT view" in question
    assert 'supplied black CAD' not in context['appearance']
    assert 'outside both blue bins' in question
    assert 'not necessarily a rectangular block' in context['appearance']
    assert 'Do not require a visible air gap' in context['appearance']
    assert sum('BEFORE UNPACKING, AFTER VERIFIED TABLE RELEASE' in p.get('text', '') for p in content) == 1


@pytest.mark.parametrize('answer', [('no', .99), ('unknown', .99), ('yes', .799)])
def test_absent_ambiguous_or_low_confidence_result_never_advances(answer):
    result, _ = run(answer)
    assert not result['success'] and result['status'] == 'unknown'
    assert result['appearance'] == 'unconfirmed'


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


def test_old_choice_probability_cannot_replace_current_predicate_probability():
    body = {'answers': [{'type': 'choice', 'name': 'unpack_result', 'choice': 'yes',
        'confidence': .77, 'probabilities': [{'value': 'yes', 'probability': .99}]}]}
    with pytest.raises(VisualInventoryError, match='invalid_response'):
        unpack_result_evidence(body, .8)


@pytest.mark.parametrize('criterion', ['wrapper_removed', 'unpack_result'])
def test_visible_black_alone_cannot_bypass_missing_task_evidence(criterion):
    capture, action = capture_and_action()
    visual, calls = client({criterion: ('unknown', .95)})
    result = verify_unpacking(visual, capture, target_id='pink', release_evidence=release_evidence(),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert not result['success']
    assert result['observation_assessment']['needs_observation']
    assert criterion in result['observation_assessment']['missing_evidence']
    assert len(calls) == 3


@pytest.mark.parametrize('probabilities,success', [((.99, .93), True), ((.04, 0.), False),
                                                  ((.99, .79), False), ((.79, .93), False)])
def test_required_visual_facts_preserve_real_visible_and_occluded_outcomes(probabilities, success):
    capture, action = capture_and_action()
    answers = {name: ('yes', probability) for name, probability in
               zip(('unpack_result', 'wrapper_removed'), probabilities)}
    visual, _ = client(answers)
    result = verify_unpacking(visual, capture, target_id='pink', release_evidence=release_evidence(),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert result['success'] is success

"""Placement language selects measured proposals, not generated coordinates."""
import importlib
import json
from copy import deepcopy

import numpy as np
import pytest
from placement_intent_fixture import Answers


def selector():
    api = importlib.import_module('luna_mujoco.agent.placement')
    assert hasattr(api, 'select_placement_candidates'), 'runtime must select grounded candidates'
    return api.select_placement_candidates


def proposals():
    return [{'candidate_id': 'placement-' + str(i), 'visible_camera_ids': ['1'],
             'footprint_world_xy_m': [[i * .1, 0], [i * .1 + .02, .02]]} for i in range(3)]


async def test_language_is_checked_in_one_batch_against_all_measured_candidates():
    judge = Answers(placement_candidate_0=False, placement_candidate_1=True, placement_candidate_2=True)
    context = {'original_requirement': {'instruction': '把盒子里的东西放到杯子左边的桌面上'},
               'target_id': 'held', 'destination_id': 'table-region:table'}
    result = await selector()(judge, proposals(), context=context)
    assert result['status'] == 'success'
    assert result['candidate_ids'] == ['placement-1', 'placement-2']
    assert len(judge.calls) == 1
    assert judge.calls[0][1]['original_requirement'] == context['original_requirement']
    assert len(judge.calls[0][0]) == 3


async def test_no_match_does_not_select_whole_support_or_first_candidate():
    result = await selector()(Answers(), proposals(), context={})
    assert result['status'] == 'unknown'
    assert result['candidate_ids'] == []


async def test_unprojected_candidate_is_not_offered_for_visual_selection():
    judge = Answers(placement_candidate_0=True)
    candidates = proposals()[:1]
    candidates[0]['visible_camera_ids'] = []
    result = await selector()(judge, candidates, context={})
    assert result['status'] == 'unknown'
    assert judge.calls == []


def projection_inputs():
    capture = {'observation_id': 'now', 'generation': 1, 'views': {'1': {
        'rgb': np.zeros((100, 100, 3), np.uint8), 'T_world_camera': np.eye(4).tolist(),
        'intrinsics': {'fx': 100., 'fy': 100., 'cx': 50., 'cy': 50.}}}}
    pose = np.eye(4)
    pose[2, 3] = 1.
    resolution = {'placement_zone': {'observation_id': 'now', 'generation': 1,
        'floor_z_m': 1., 'support_entity_id': 'table', 'intent': {'relation': 'on'}},
        'candidates': [{'candidate_id': 'placement-0', 'goal': {'T_world_tcp': pose.tolist()}}]}
    holding = {'points_tool_m': [[-.1, -.05, 0], [.1, -.05, 0], [.1, .05, 0], [-.1, .05, 0]]}
    return capture, resolution, holding


def projector():
    api = importlib.import_module('luna_mujoco.robot_skills.placement_resolution')
    assert hasattr(api, 'project_placement_candidates'), 'current calibrated footprint projections required'
    return api.project_placement_candidates


def test_projection_uses_whole_held_footprint_and_does_not_change_current_rgb():
    capture, resolution, held = projection_inputs()
    before = deepcopy(capture)
    result = projector()(capture, resolution, held)
    candidate = result['candidates'][0]
    assert candidate['visible_camera_ids'] == ['1']
    polygon = np.asarray(result['projections']['1'][0]['polygon'])
    np.testing.assert_allclose(polygon.min(0), [.4, .45])
    np.testing.assert_allclose(polygon.max(0), [.6, .55])
    np.testing.assert_array_equal(capture['views']['1']['rgb'], before['views']['1']['rgb'])


@pytest.mark.parametrize('change', ['observation_id', 'generation'])
def test_projection_rejects_stale_camera_evidence(change):
    capture, resolution, held = projection_inputs()
    capture[change] = 'other' if change == 'observation_id' else 2
    with pytest.raises(ValueError, match='placement_candidate_capture_stale'):
        projector()(capture, resolution, held)


def test_candidate_outside_all_images_has_no_visual_witness():
    capture, resolution, held = projection_inputs()
    resolution['candidates'][0]['goal']['T_world_tcp'][0][3] = 20.
    result = projector()(capture, resolution, held)
    assert result['candidates'][0]['visible_camera_ids'] == []


def test_decisions_receives_original_rgb_and_distinct_candidate_proposal_image():
    from test_agent_model_layer import Client

    from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest

    capture, resolution, held = projection_inputs()
    projected = projector()(capture, resolution, held)
    client = Client([{'answers': [{'type': 'predicate', 'name': 'candidate', 'probability': .99}]}])
    model = DecisionsJudgments(api_key='test', client=client)
    model.judge(capture, requests=[JudgmentRequest(name='candidate', instructions='Matches the original task?')],
        context={'verification_scope': 'placement_candidates', 'candidate_projection': projected})
    parts = client.requests[0][1]['input'][0]['content']
    assert len([part for part in parts if part['type'] == 'input_image']) == 2
    roles = [json.loads(part['text']) for part in parts if part['type'] == 'input_text' and part['text'].startswith('{')]
    assert any(role.get('image_role') == 'placement_candidate_proposals' for role in roles)


async def test_runtime_only_binds_semantically_accepted_candidates(tmp_path):
    from test_unified_placement_controller import controller

    from luna_mujoco.robot_skills.placement_resolution import digest
    from luna_mujoco.runtime.placement import select_for_controller

    runner = controller(tmp_path)
    capture, resolution, held = projection_inputs()
    resolution['candidates'].append(deepcopy(resolution['candidates'][0]) | {'candidate_id': 'placement-1'})
    resolution['placement_zone'].update(semantic_scope='candidate_selection',
        original_requirement={'instruction': '放在杯子左侧'}, active_object_requirement=None,
        destination_id='table', bound_destination_ref='table')
    resolution.update(status='success', selection_evidence={})
    answers = Answers(placement_candidate_0=True, placement_candidate_1=False)
    async def judge(context, requests, *, extra=None, **kwargs):
        return await answers(requests, extra)
    runner._judge = judge
    context = {'latest_capture': capture, 'capture_epochs': {'now': runner.observation_epoch}, 'holding': held}
    result = await select_for_controller(runner, context, resolution)
    assert [c['candidate_id'] for c in result['candidates']] == ['placement-0']
    assert result['resolution_id'] == digest({key: value for key, value in result.items() if key != 'resolution_id'})
    assert context['placement_resolution'] == result
    assert len(resolution['candidates']) == 2


async def test_runtime_unknown_candidate_selection_leaves_no_executable_resolution(tmp_path):
    from test_unified_placement_controller import controller

    from luna_mujoco.runtime.placement import select_for_controller

    runner = controller(tmp_path)
    capture, resolution, held = projection_inputs()
    resolution['placement_zone'].update(semantic_scope='candidate_selection',
        original_requirement={'instruction': '不可见的左侧隔间'}, active_object_requirement=None,
        destination_id='table', bound_destination_ref='table')
    resolution.update(status='success', selection_evidence={})
    answers = Answers()
    async def judge(context, requests, *, extra=None, **kwargs):
        return await answers(requests, extra)
    runner._judge = judge
    context = {'latest_capture': capture, 'capture_epochs': {'now': runner.observation_epoch}, 'holding': held}
    result = await select_for_controller(runner, context, resolution)
    assert result['status'] == 'unknown'
    assert 'placement_resolution' not in context

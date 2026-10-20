import pytest
from test_decisions import install_agent

from jev_mujoco.decisions import Decisions


async def test_request_binds_explicit_arm_and_collection_in_one_model_call(monkeypatch):
    result = {'task': 'grasp_place', 'route': 'table_box', 'kind': 'any', 'color': 'any', 'position': 'any',
              'quantity': 'unbounded', 'selection_order': 'explicit', 'environment_red': 'no', 'environment_blue': 'no',
              'destination_kind': 'bin', 'destination_color': 'blue', 'destination_position': 'any',
              'arm': 'A', 'scope': 'collection', 'visual_semantics': 'needed'}
    records = []
    install_agent(monkeypatch, result, {key: 1. for key in result})
    engine = Decisions(None, lambda *args: records.append(args[2]))
    assert await engine.bind_request('用左臂把两个红色方块和蓝盒旁的球放进蓝盒') == result
    assert len(records) == 1


@pytest.mark.parametrize('action', ['reobserve', 'retry_grasp', 'retry_place', 'defer', 'stop'])
async def test_identical_failure_can_receive_different_jev_recovery_decisions(monkeypatch, action):
    records = []
    install_agent(monkeypatch, {'recovery': action}, {'recovery': 1.})
    engine = Decisions(None, lambda *args: records.append(args[2]))
    state = {'instruction': '把物体放进去', 'failure': {'stage': 'verify_grasp', 'status': 'empty'},
             'task_progress': {'attempts': {'a': 1}}}
    assert await engine.choose_recovery(state) == action
    assert records[0]['state']['failure']['status'] == 'empty'


async def test_task_assessment_sees_original_quotas_and_confirmed_history(monkeypatch):
    records = []
    install_agent(monkeypatch, {'progress': 'continue'}, {'progress': 1.})
    state = {'instruction': '放两个红方块和一个蓝球，黑色不动',
             'task_progress': {'completed': [{'id': 'red-1', 'color': 'red', 'kind': 'cube'}]},
             'observation': {'objects': [{'id': 'red-2', 'color': 'red'}]}}
    result = await Decisions(None, lambda *args: records.append(args[2])).assess_task(state)
    assert result == 'continue'
    assert records[0]['state']['instruction'] == state['instruction']
    assert records[0]['state']['task_progress']['completed'][0]['id'] == 'red-1'


async def test_optional_visual_routing_uncertainty_does_not_reject_clear_task(monkeypatch):
    result = {'task': 'grasp_place', 'route': 'table_box', 'arm': 'A', 'scope': 'single', 'visual_semantics': 'needed',
              'quantity': '1', 'selection_order': 'explicit', 'environment_red': 'no', 'environment_blue': 'no',
              'kind': 'cube', 'color': 'red', 'position': 'any', 'destination_kind': 'bin',
              'destination_color': 'blue', 'destination_position': 'any'}
    confidence = dict.fromkeys(result, 1.)
    confidence.update(visual_semantics=.16, destination_position=.4, scope=.5)
    install_agent(monkeypatch, result, confidence)
    assert await Decisions(None, lambda *args: None).bind_request('左臂将红方块放进蓝盒') == result


async def test_inventory_assessment_is_independent_of_next_motion_map(monkeypatch):
    records = []
    install_agent(monkeypatch, {'progress': 'complete'}, {'progress': 1.})
    state = {'instruction': '把指定两个物体放进盒子',
             'task_progress': {'completed': [{'id': 'a'}, {'id': 'b'}], 'deferred': []},
             'observation': {'objects': []},
             'perception_state': {'scene_refresh_required': True,
                                  'release_endpoint_evidence': {'status': 'confirmed', 'target_id': 'b'}},
             'collision_scene': {'scene_token': 'old'}, 'arm_evaluations': {'A': {}},
             'facts': {'scene_ready': False}}
    await Decisions(None, lambda *args: records.append(args[2])).assess_task(state)
    supplied = records[0]['state']
    assert supplied['task_progress']['completed'] == state['task_progress']['completed']
    assert 'collision_scene' not in supplied and 'arm_evaluations' not in supplied
    assert 'perception_state' not in supplied and 'facts' not in supplied
    assert supplied['observation_evidence']['release_endpoint_evidence']['target_id'] == 'b'
    assert 'scene_refresh_required' not in supplied['observation_evidence']


async def test_recovery_emphasizes_budget_and_current_feedback_without_planner_arrays(monkeypatch):
    records = []
    install_agent(monkeypatch, {'recovery': 'stop'}, {'recovery': 1.})
    state = {'instruction': '把红色放进去', 'facts': {'grasp_held': True, 'attached': True},
             'failure': {'code': 'recovery_retry_budget_exhausted', 'attempt_count': 3, 'retry_limit': 3},
             'verification': {'status': 'held', 'position': {'status': 'held'},
                              'attachment': {'attached': True, 'attachment_spheres_tool_m': [[999] * 4]}},
             'task_progress': {'completed': [], 'deferred': []},
             'observation_budget': {'used': 8, 'limit': 8},
             'arm_evaluations': {'A': {'candidate_columns': ['candidate_index'], 'candidates': [[0]]}}}
    assert await Decisions(None, lambda *args: records.append(args[2])).choose_recovery(state) == 'stop'
    supplied = records[0]['state']
    assert supplied['failure'] == state['failure']
    assert supplied['verification']['position']['status'] == 'held'
    assert supplied['observation_budget'] == state['observation_budget']
    assert 'arm_evaluations' not in supplied
    assert 'attachment_spheres_tool_m' not in supplied['verification']['attachment']
    assert set(records[0]['questions']['recovery']['criteria']) == {
        'reobserve', 'retry_grasp', 'retry_place', 'retry_plan', 'defer', 'stop'}


async def test_flexible_query_attributes_do_not_offer_unsupported_for_relations(monkeypatch):
    result = {'task': 'grasp_place', 'route': 'table_box', 'arm': 'auto', 'scope': 'collection', 'visual_semantics': 'needed',
              'quantity': 'unbounded', 'selection_order': 'explicit', 'environment_red': 'no', 'environment_blue': 'no',
              'kind': 'any', 'color': 'any', 'position': 'any', 'destination_kind': 'any',
              'destination_color': 'any', 'destination_position': 'any'}
    records = []
    install_agent(monkeypatch, result, dict.fromkeys(result, 1.))
    await Decisions(None, lambda *args: records.append(args[2])).bind_request('红色放蓝盒，蓝球放标记')
    for name in ('kind', 'color', 'position', 'destination_kind', 'destination_color', 'destination_position'):
        assert 'unsupported' not in records[0]['questions'][name]['criteria']
        assert 'any' in records[0]['questions'][name]['criteria']

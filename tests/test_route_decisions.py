import pytest
from test_decisions import install_agent

from jev_mujoco.decisions import Decisions, compact_decision_state


async def test_completed_frozen_inventory_does_not_require_new_source_observation(monkeypatch):
    records = []
    install_agent(monkeypatch, {'progress': 'complete'}, {'progress': 1.})
    state = {'instruction': '把盒内物体放到桌面', 'task_progress': {
        'inventory_finalized': True, 'objects': [{'id': 'part', 'status': 'completed'}],
        'completed': [{'id': 'part', 'verification_scope': 'release_only'}]},
        'observation': {'observation_id': 'before-last-grasp', 'objects': [{'id': 'part'}]}}
    assert await Decisions(None, lambda *args: records.append(args[2])).assess_task(state) == 'complete'
    instructions = records[0]['questions']['progress']['instructions']
    assert '不要求新的源区域观察' in instructions
    assert '且当前源区域观察支持完成' not in instructions


async def test_offline_workspace_evidence_and_choice_remain_visible_to_jev(monkeypatch):
    records = []
    install_agent(monkeypatch, {'arm': 'A'}, {'arm': 1.})
    evidence = {'A': {'feasibility_scope': 'offline_position_workspace', 'reachable': True,
                     'source': {'resolution_m': .01}, 'destination': {'reachable': True}}}
    assert compact_decision_state({'arm_evaluations': evidence})['arm_evaluations'] == evidence
    assert await Decisions(None, lambda *args: records.append(args[2])).choose_arm({}, evidence) == 'A'
    assert '离线' in records[0]['questions']['arm']['instructions']
    assert '目的地同侧' in records[0]['questions']['arm']['instructions']
    assert '不能仅因上一段' in records[0]['questions']['arm']['instructions']


async def test_arm_selection_uses_current_leg_endpoints_without_previous_leg_geometry(monkeypatch):
    records = []
    install_agent(monkeypatch, {'arm': 'B'}, {'arm': 1.})
    evidence = {'B': {'feasibility_scope': 'offline_position_workspace', 'reachable': True,
                     'source': {'position_world_m': [.4, 0., .62], 'resolution_m': .01},
                     'destination': {'position_world_m': [.45, -.23, .61], 'reachable': True}}}
    state = {'instruction': '左盒经桌面到右盒', 'requested_arm': None,
             'target_id': 'part', 'destination_id': 'right-bin',
             'task_progress': {'stage_index': 1, 'stage_arms': {'0': 'A'},
                               'initial_source': {'position': [.45, .23, .62]}}}
    assert await Decisions(None, lambda *args: records.append(args[2])).choose_arm(state, evidence) == 'B'
    context = records[0]['state']
    assert context['current_transfer'] == {'stage_index': 1, 'source_id': 'part',
        'destination_id': 'right-bin', 'source_world_m': {'x': .4, 'y': 0., 'z': .62},
        'destination_world_m': {'x': .45, 'y': -.23, 'z': .61}}
    assert context['arm_evaluations'] == evidence
    assert 'task_progress' not in context
    assert state['task_progress']['stage_arms'] == {'0': 'A'}


def test_center_evidence_survives_decision_compaction_without_fake_grasp_candidates():
    evidence = {'A': {'feasibility_scope': 'center_position_ik_only', 'reachable': True,
                     'source': {'reachable': True}, 'destination': {'reachable': True}}}
    result = compact_decision_state({'arm_evaluations': evidence})
    assert result['arm_evaluations'] == evidence


def test_center_reachability_history_references_current_evidence_once():
    evidence = {'A': {'feasibility_scope': 'center_position_ik_only', 'reachable': True,
                     'source': {'reachable': True}, 'destination': {'reachable': True}}}
    result = compact_decision_state({'arm_evaluations': evidence, 'recent_results': [
        {'title': 'evaluate_reachability', 'data': {'result': evidence}}]}, skill_selection=True)
    assert result['arm_evaluations'] == evidence
    assert result['recent_results'][0]['data'] == {'result_reference': 'arm_evaluations'}


def test_completed_execution_keeps_outcome_without_repeating_motor_paths():
    event = {'title': 'execute_grasp', 'data': {'result': {'arm': 'A', 'candidate_index': 2,
        'lift_carried_collision_checked': False,
        'stages': {'grasp': {'joints_rad': [0.] * 7}, 'lift': {'completed': True}}}}}
    result = compact_decision_state({'recent_results': [event]})['recent_results'][0]['data']['result']
    assert result['arm'] == 'A' and result['candidate_index'] == 2
    assert result['stage_names'] == ['grasp', 'lift']
    assert result['lift_carried_collision_checked'] is False
    assert 'stages' not in result and 'stages' in event['data']['result']


def test_decision_numeric_projection_preserves_small_thresholds_and_raw_state():
    state = {'observation': {'position': [.123456789012, -1e-8], 'unknown': True},
             'failure': {'code': 'place_no_valid_candidate', 'residual': -1e-7}}
    result = compact_decision_state(state)
    assert result['observation']['position'] == [.123457, -1e-8]
    assert result['failure'] == state['failure']
    assert result['observation']['unknown'] is True
    assert state['observation']['position'][0] == .123456789012


def test_old_observation_keeps_identity_but_not_repeated_surface_geometry():
    previous = {'observation_id': 'old', 'objects': [{'id': 'red', 'kind': 'cube', 'color': 'red',
        'position': [.4, .1, .6], 'support_reference': {'points': [[1, 2, 3]] * 1000}}]}
    projected = compact_decision_state({'observation': {'observation_id': 'new'},
        'recent_results': [{'title': 'observe_scene', 'data': {'result': previous}}]})
    evidence = projected['recent_results'][0]['data']['result']
    assert evidence['objects'][0]['id'] == 'red'
    assert 'support_reference' not in evidence['objects'][0]
    assert previous['objects'][0]['support_reference']['points']


async def test_bind_request_exposes_two_stage_route_as_a_typed_choice(monkeypatch):
    captured = []
    answer = {'task': 'grasp_place', 'route': 'box_table_box', 'arm': 'auto', 'scope': 'single',
              'quantity': '1', 'selection_order': 'explicit', 'environment_red': 'no',
              'visual_semantics': 'optional', 'kind': 'cube', 'color': 'red', 'position': 'any',
              'destination_kind': 'any', 'destination_color': 'blue', 'destination_position': 'any'}
    install_agent(monkeypatch, answer, {'task': 1., 'arm': 1., 'route': 1.,
                                      'quantity': 1., 'selection_order': 1.}, captured)
    result = await Decisions(None, lambda *args: None).bind_request('把红色物体从左盒经桌面放到右盒')
    assert result['route'] == 'box_table_box'
    assert 'route' in captured[0]['schema']['properties']


async def test_second_leg_offers_only_final_box_not_marker_or_original_source(monkeypatch):
    records = []
    install_agent(monkeypatch, {'destination': 'right'}, {'destination': 1.})
    state = {'route': 'box_table_box', 'instruction': '左盒经桌面到右盒',
        'task_progress': {'route': 'box_table_box', 'stage_index': 1, 'current': {'id': 'red',
            'initial_source': {'measured_relationships': [
                {'relation': 'above_observed_bin_floor', 'object_id': 'left', 'visible_xy_fraction': .99},
                {'relation': 'above_observed_bin_floor', 'object_id': 'right', 'visible_xy_fraction': .01}]}}}}
    objects = [{'id': 'marker', 'kind': 'marker'}, {'id': 'left', 'kind': 'bin'}, {'id': 'right', 'kind': 'bin'}]
    await Decisions(None, lambda *args: records.append(args[2])).choose_destination(state, objects)
    criteria = records[0]['questions']['destination']['criteria']
    assert set(criteria) == {'right', 'missing', 'ambiguous'}
    assert records[0]['state']['active_transfer_stage']['index'] == 1


async def test_route_target_choices_are_limited_to_unfinished_initial_inventory(monkeypatch):
    records = []
    install_agent(monkeypatch, {'target': 'two'}, {'target': 1.})
    state = {'route': 'table_box', 'task_progress': {'scope': 'collection', 'inventory_initialized': True,
        'objects': [{'id': 'one', 'status': 'completed'}, {'id': 'two', 'status': 'observed'}]}}
    objects = [{'id': key, 'kind': 'cube'} for key in ('one', 'two', 'box-wall-alias')]
    await Decisions(None, lambda *args: records.append(args[2])).choose_target(state, objects)
    assert set(records[0]['questions']['target']['criteria']) == {'two', 'missing', 'ambiguous'}


async def test_route_assessment_uses_initial_inventory_scope_without_unknown_obstacle_targets(monkeypatch):
    records = []
    install_agent(monkeypatch, {'progress': 'complete'}, {'progress': 1.})
    state = {'instruction': '所有方块放入盒子', 'route': 'table_box',
        'task_progress': {'inventory_initialized': True,
                         'objects': [{'id': 'one', 'status': 'completed'}]},
        'observation': {'objects': [{'id': 'one', 'kind': 'cube'}, {'id': 'wall-alias', 'kind': 'cube'},
                                   {'id': 'bin', 'kind': 'bin'}]}}
    assert await Decisions(None, lambda *args: records.append(args[2])).assess_task(state) == 'complete'
    assert {obj['id'] for obj in records[0]['state']['observation']['objects']} == {'one', 'bin'}
    assert len(state['observation']['objects']) == 3


@pytest.mark.parametrize('route', ['direct', 'box_table'])
async def test_assessment_asks_jev_for_membership_using_source_relationships(monkeypatch, route):
    records = []
    install_agent(monkeypatch, {'progress': 'continue', 'member_0': 'required', 'member_1': 'excluded',
                               'member_2': 'unknown'}, dict.fromkeys(['progress', 'member_0', 'member_1', 'member_2'], 1.))
    objects = [{'id': 'inside', 'kind': 'cube', 'status': 'observed', 'measured_relationships': [
        {'relation': 'above_observed_bin_floor', 'object_id': 'bin', 'visible_xy_fraction': 1.,
         'source': 'registered_rgbd_visible_samples'}]},
        {'id': 'outside', 'kind': 'cube', 'status': 'observed', 'measured_relationships': []},
        {'id': 'unclear', 'kind': 'cube', 'status': 'observed'}]
    state = {'instruction': '把盒子中物体放到桌面黄色区域', 'route': route,
        'task_progress': {'inventory_initialized': True, 'objects': [], 'unresolved_membership': objects,
                          'membership_pending_ids': ['inside', 'outside', 'unclear']},
        'observation': {'objects': objects + [{'id': 'bin', 'kind': 'bin'}]}}
    result = await Decisions(None, lambda *args: records.append(args[2])).assess_task(state)
    assert result == {'judgment': 'continue', 'membership': {
        'inside': 'required', 'outside': 'excluded', 'unclear': 'unknown'}}
    questions = records[0]['questions']
    assert set(questions['member_0']['criteria']) == {'required', 'excluded', 'unknown'}
    for index in (1, 2):
        expected = {'excluded', 'unknown'} if route == 'box_table' else {'required', 'excluded', 'unknown'}
        assert set(questions[f'member_{index}']['criteria']) == expected
    supplied = records[0]['state']['observation']['objects'][0]
    assert supplied['measured_relationships'][0]['source'] == 'registered_rgbd_visible_samples'


def test_source_geometry_reports_measured_gap_without_claiming_complete_box_bounds():
    from jev_mujoco.task_progress import source_region_geometry

    objects = [{'id': 'bin', 'kind': 'bin', 'placement_reference': {
        'region_polygon_world_xy_m': [[.3, .2], [.6, .2], [.6, .5], [.3, .5]]}}]
    assert source_region_geometry({'position': [.45, -.15, .65]}, objects) == [
        {'bin_id': 'bin', 'centroid_to_observed_floor_xy_bounds_gap_m': .35,
         'source': 'measured_centroid_and_visible_floor_polygon', 'complete_box_bounds': False}]
    assert source_region_geometry({'position': [.45, .3, .65]}, objects)[0][
        'centroid_to_observed_floor_xy_bounds_gap_m'] == 0.
    assert source_region_geometry({}, objects) == []
    assert source_region_geometry({'position': [.45, -.15, .65]}, [{'id': 'bin', 'kind': 'bin'}]) == []


async def test_direct_collection_target_choices_require_membership(monkeypatch):
    records = []
    install_agent(monkeypatch, {'target': 'source'}, {'target': 1.})
    state = {'route': 'direct', 'task_progress': {'scope': 'collection', 'inventory_initialized': False,
        'objects': [{'id': 'source', 'status': 'observed'}],
        'membership': {'source': 'required', 'obstacle': 'excluded', 'unclear': 'unknown'}}}
    objects = [{'id': key, 'kind': 'cube'} for key in ('source', 'obstacle', 'unclear')]
    await Decisions(None, lambda *args: records.append(args[2])).choose_target(state, objects)
    assert set(records[0]['questions']['target']['criteria']) == {'source', 'missing', 'ambiguous'}

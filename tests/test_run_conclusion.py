import pytest

from jev_mujoco.run_conclusion import run_conclusion


@pytest.mark.parametrize('code,service', [('visual_discovery_http_429', 'Responses'),
                                        ('visual_decisions_http_429', 'Decisions')])
def test_visual_rate_limit_is_explained_as_service_failure(code, service):
    result = run_conclusion({'status': 'needs_attention', 'error': code,
        'failed_step': {'skill': 'observe_scene'}}, [])
    assert service in result['summary'] and '429' in result['summary']
    assert '视觉接口' in result['summary']


def test_place_all_ik_rejected_does_not_claim_trajectory_or_kinematic_failure():
    details = {'planning_summary': {'generated_goal_count': 6, 'ik_rejected_count': 6,
                                    'trajectory_rejected_count': 0}}
    result = run_conclusion({'status': 'failed', 'error': 'place_no_valid_candidate',
        'failed_step': {'skill': 'plan_place'}}, [{'kind': 'skill_error', 'data': {
            'skill': 'plan_place', 'code': 'place_no_valid_candidate', 'details': details}}])
    assert '6 个放置候选' in result['summary']
    assert '未进入轨迹规划' in result['summary']
    assert '不证明运动学不可达' in result['summary']


def test_initial_quantity_rejection_explains_decision_and_retains_evidence():
    details = {'field': 'quantity', 'selected': 'unbounded', 'confidence': .48, 'minimum_confidence': .6}
    result = run_conclusion({'status': 'failed', 'error': 'low_confidence:quantity',
        'error_code': 'low_confidence:quantity', 'failed_step': {'skill': 'bind_task'}}, [
            {'kind': 'skill_error', 'data': {'skill': 'bind_task', 'code': 'low_confidence:quantity',
                                           'details': details}}])
    assert result['root_cause']['evidence']['details'] == details
    assert '搬运数量' in result['summary']
    assert '0.48' in result['summary'] and '0.60' in result['summary']
    assert '未明确指定数量' in result['summary']
    assert '未开始观测或运动' in result['summary']
    assert '不足以确定更具体的原因' not in result['summary']


def test_low_confidence_without_provider_score_does_not_format_none_as_number():
    result = run_conclusion({'status': 'failed', 'error': 'low_confidence:target',
        'failed_step': {'skill': 'identify_target'}}, [{'kind': 'skill_error', 'data': {
            'skill': 'identify_target', 'code': 'low_confidence:target', 'details': {
                'field': 'target', 'selected': 'seen-object', 'confidence': None, 'minimum_confidence': .6}}}])
    assert '未返回置信度' in result['summary']
    assert '未开始观测或运动' not in result['summary']


async def test_initial_rejection_reaches_saved_runner_conclusion_without_observation(tmp_path, monkeypatch):
    import json

    from test_decisions import install_agent
    from test_runner import Algorithms, Sim, wait

    from jev_mujoco.decisions import Decisions
    from jev_mujoco.runner import Runner

    output = {'task': 'grasp_place', 'route': 'box_table', 'arm': 'auto', 'scope': 'collection',
              'visual_semantics': 'optional', 'kind': 'any', 'color': 'green', 'position': 'any',
              'destination_kind': 'marker', 'destination_color': 'yellow', 'destination_position': 'any',
              'quantity': 'unbounded', 'selection_order': 'explicit',
              'environment_green': 'no', 'environment_yellow': 'no'}
    install_agent(monkeypatch, output, {**{field: 1. for field in output}, 'quantity': .48})
    sim = Sim()
    runner = Runner(sim, Algorithms(), lambda record: Decisions(None, record), tmp_path)
    await runner.start('抓取盒子中一个绿色物体放到桌面黄色区域')
    await wait(runner)
    assert sim.frame == 0
    assert runner.state['error_code'] == 'low_confidence:quantity'
    assert '置信度 0.48，要求至少 0.60' in runner.state['conclusion']['summary']
    events = [json.loads(line) for line in (tmp_path / (runner.run_id + '.jsonl')).read_text(
        encoding='utf-8').splitlines()]
    assert events[-1]['kind'] == 'run_end'
    assert events[-1]['data']['conclusion'] == runner.state['conclusion']


@pytest.mark.parametrize('status', ['failed', 'needs_attention', 'partial_success', 'cancelled'])
def test_unmapped_failure_keeps_message_stage_and_uncertainty(status):
    state = {'status': status, 'error': 'specific_detail', 'error_code': 'unmapped_code',
             'failed_step': {'skill': 'build_scene'}}
    result = run_conclusion(state, [])
    assert result['root_cause']['evidence']['error'] == 'specific_detail'
    assert 'build_scene' in result['summary']
    assert '证据不足' in result['summary']
    assert result['summary'].count('unmapped_code') == 1


def test_keeps_algorithm_evidence_and_recovery_rejections_separately():
    failure = {'stage': 'choose_arm', 'code': 'no_reachable_arm',
               'evidence': {'arm_evaluations': {'A': {'reachable': False}}}}
    events = [{'kind': 'task_failure', 'data': failure}, {'kind': 'skill_error', 'data': {
        'code': 'retry_plan_requires_current_evaluation', 'details': {'selected_skill': 'recover_task'}}}]
    result = run_conclusion({'status': 'failed', 'failure': failure}, events)
    assert result['root_cause'] == failure
    assert len(result['rejected_calls']) == 1
    assert '可达性' in result['summary']


def test_distinct_terminal_error_is_visible_alongside_original_failure():
    result = run_conclusion({'status': 'failed',
        'failure': {'stage': 'plan_grasp', 'code': 'planning_no_valid_candidate'},
        'error': 'plan_place_timeout:180s', 'error_code': 'plan_place_timeout',
        'failed_step': {'skill': 'plan_place'}}, [])
    assert 'planning_no_valid_candidate' in result['summary']
    assert '最终停止' in result['summary']
    assert 'plan_place' in result['summary'] and '180s' in result['summary']


def test_deferred_target_root_is_not_hidden_by_later_empty_target_selection():
    original = {'target_id': 'source', 'stage': 'choose_arm', 'code': 'no_reachable_arm'}
    latest = {'target_id': None, 'stage': 'identify_target', 'code': 'target_missing'}
    result = run_conclusion({'status': 'failed', 'failure': latest, 'error': 'target_missing',
        'failed_step': {'skill': 'identify_target'}}, [
            {'kind': 'task_failure', 'data': original}, {'kind': 'task_failure', 'data': latest}])
    assert result['root_cause'] == original
    assert 'no_reachable_arm' in result['summary'] and 'target_missing' in result['summary']


def test_a_completed_targets_old_failure_is_not_the_remaining_tasks_root():
    old = {'target_id': 'done', 'stage': 'plan_grasp', 'code': 'planning_no_valid_candidate'}
    latest = {'target_id': 'pending', 'stage': 'choose_arm', 'code': 'no_reachable_arm'}
    result = run_conclusion({'status': 'partial_success', 'failure': latest,
        'task_progress': {'completed': [{'id': 'done'}]}}, [
            {'kind': 'task_failure', 'data': old}, {'kind': 'task_failure', 'data': latest}])
    assert result['root_cause'] == latest


def test_workspace_miss_reports_lookup_evidence_not_live_ik_failure():
    point = {'reachable': False, 'method': 'offline_workspace_lookup', 'resolution_m': .01}
    failure = {'stage': 'choose_arm', 'code': 'no_reachable_arm',
               'evidence': {'arm_evaluations': {'A': {'source': point}}}}
    result = run_conclusion({'status': 'needs_attention', 'failure': failure}, [])
    assert '离线工作空间' in result['summary'] and '未命中' in result['summary']


@pytest.mark.parametrize('code,description', [
    ('workspace_file_missing', '缺少'),
    ('workspace_model_mismatch', '模型不匹配'),
    ('invalid_workspace_artifact', '不完整或格式无效'),
])
def test_workspace_file_errors_explain_configuration_failure(code, description):
    result = run_conclusion({'status': 'failed', 'error': code,
        'failed_step': {'skill': 'evaluate_reachability'}}, [])
    assert '离线工作空间' in result['summary'] and description in result['summary']
    assert '不能判断机械臂是否可达' in result['summary']


def test_unknown_place_conclusion_explains_execution_and_visual_http_error():
    state = {'status': 'needs_attention', 'facts': {'released': True, 'place_execution_completed': True},
        'verification': {'status': 'unknown', 'reason': 'placement_not_visible', 'open_confirmed': True},
        'visual_result': {'status': 'unknown', 'source': 'openai_visual', 'reason': 'HTTPStatusError', 'http_status': 429}}
    result = run_conclusion(state, [])
    assert result['root_cause']['stage'] == 'verify_place'
    assert result['root_cause']['evidence_status'] == 'unknown'
    assert 'placement_not_visible' in result['summary']
    assert '429' in result['summary'] and 'OpenAI' in result['summary']
    assert '放置动作已执行' in result['summary']
    assert '抓放失败' not in result['summary']
    assert result['failures'] == []


def test_success_does_not_report_an_old_visual_service_error():
    result = run_conclusion({'status': 'succeeded', 'verification': {'status': 'confirmed'},
        'visual_result': {'status': 'unknown', 'http_status': 429}}, [])
    assert '429' not in result['summary']


def test_unknown_conclusion_reports_observation_gap_without_claiming_an_error():
    result = run_conclusion({'status': 'needs_attention', 'facts': {'released': True},
        'verification': {'status': 'unknown', 'observation_reason': 'instance association ambiguous'}}, [])
    assert '该步骤返回错误' not in result['summary']
    assert 'instance association ambiguous' in result['summary']


@pytest.mark.parametrize('code, explanation', [
    ('placement_region_no_free_candidate', '已占用位置'),
    ('placed_object_obstacle_unavailable', '没有可复用的体素'),
    ('placed_table_object_not_observed', '未关联到已放在桌面'),
])
def test_occupied_placement_failures_have_specific_conclusions(code, explanation):
    result = run_conclusion({'status': 'needs_attention', 'error_code': code, 'error': code}, [])
    assert explanation in result['summary']
    assert 'JEV' in result['summary']


def test_placement_failure_summary_explains_position_and_yaw_rejection_counts():
    details = {'section_count': 2, 'yaw_count_per_position': 18, 'positions': [
        {'position_index': 0, 'occupied_rejections': 18, 'boundary_rejections': 0},
        {'position_index': 1, 'occupied_rejections': 12, 'boundary_rejections': 6}]}
    failure = {'stage': 'plan_place', 'code': 'placement_region_no_free_candidate',
               'evidence': {'details': details}}
    result = run_conclusion({'status': 'needs_attention', 'failure': failure}, [])
    assert '2 个候选位置' in result['summary'] and '18 个 yaw' in result['summary']
    assert '位置 1：占用 18、边界 0' in result['summary']
    assert '位置 2：占用 12、边界 6' in result['summary']
    assert result['root_cause']['evidence']['details'] == details

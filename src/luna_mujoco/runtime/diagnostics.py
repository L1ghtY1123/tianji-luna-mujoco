"""Present recorded failure evidence without guessing a model's internal reasoning."""
import math
from collections import Counter
from copy import deepcopy

from .execution_state import PolicyStopped, UnhandledSkillResult


class DiagnosticError(ValueError):
    def __init__(self, code, details):
        super().__init__(code)
        self.code = code
        self.details = deepcopy(details)


def exception_details(error):
    details = deepcopy(getattr(error, 'details', None) or getattr(error, 'evidence', None) or {})
    if not isinstance(details, dict):
        details = {'detail': str(details)[:1000]}
    if not details:
        details['message'] = str(error)[:1000]
    return details


def judgment_details(result, requests, *, failed_names=None):
    """Keep the actual question alongside the score used by the acceptance rule."""
    details = deepcopy(result.get('evidence', {}))
    details['component'] = 'decisions'
    details['judgments'] = deepcopy(result['judgments'])
    recorded = {check['name']: check for check in details.get('diagnostic_checks', [])}
    details['diagnostic_checks'] = []
    for request in requests:
        row = result['judgments'][request.name]
        choice = 'confidence' in row
        check = recorded.get(request.name) or {
            'name': request.name, 'question': request.instructions,
            'actual': row.get('confidence') if choice else row.get('probability'),
            'answer': deepcopy(row)}
        check.update(status=row['status'], required=failed_names is None or request.name in failed_names)
        details['diagnostic_checks'].append(check)
    return details


def _bounded(value, depth=0):
    if isinstance(value, str):
        if value.startswith('data:image/'):
            return '[image omitted]'
        return value[:1200]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= 3:
        return '[see archived evidence]'
    if isinstance(value, list):
        return [_bounded(item, depth + 1) for item in value[:24]]
    if isinstance(value, dict):
        return {key: _bounded(item, depth + 1) for key, item in list(value.items())[:24]
                if key.lower() not in {'api_key', 'authorization', 'headers', 'images', 'image',
                                       'image_url', 'rgb_jpeg', 'mask', 'cloud', 'rgb', 'depth'}}
    return str(value)[:200]


def _component(code, details):
    if code == 'CUROBO_CUDA_OUT_OF_MEMORY':
        return 'curobo_resource'
    if code == 'unpacked_object_outside_execution_workspace':
        return 'motion_planning'
    if code == 'unpack_depth_unconfirmed_after_observation':
        return 'perception_geometry'
    if code == 'explicit_observation_return_required':
        return 'policy_runtime'
    if details.get('component'):
        return details['component']
    if code.startswith(('planning_', 'grasp_lift_', 'grasp_no_transport_', 'execution_observation_',
                        'execution_collision', 'no_reachable_arm')):
        return 'motion_planning'
    if code.startswith('sam3_'):
        return 'sam3'
    if code.startswith(('visual_', 'decisions_', 'task_binding_unconfirmed', 'task_completion_unconfirmed')):
        return 'decisions'
    if code.startswith('responses_'):
        return 'responses'
    return 'policy_runtime'


def _planning_evidence(details):
    failures = details.get('failures', [])
    counts = Counter((row.get('stage', 'unknown'), row.get('code') or row.get('status', 'unknown'))
                     for row in failures)
    checks = []
    for sampling in details.get('rounds', []):
        reach = sampling.get('transport_reach_filter') or {}
        if not reach:
            continue
        distances = [item['nominal_transport_wrist_distance_m'] for item in reach.get('candidate_evidence', [])
                     if type(item.get('nominal_transport_wrist_distance_m')) in (int, float)
                     and math.isfinite(item['nominal_transport_wrist_distance_m'])]
        actual = {key: deepcopy(reach[key]) for key in (
            'input_count', 'accepted_count', 'rejected_count', 'reach_screening_limit_m',
            'measurement_reserve_m', 'shoulder_world_m', 'geometry_source') if key in reach}
        if distances:
            actual['candidate_wrist_distance_range_m'] = [min(distances), max(distances)]
        checks.append({'name': 'transport_reach_filter', 'status': 'failed',
            'sampling_round': sampling.get('sampling_round'),
            'question': '保持候选抓姿时，机械臂是否能将物体送到所选目的地？',
            'actual': actual, 'expected': 'accepted_count > 0'})
    # Keep different reasons visible; several phases may reject the same candidate.
    seen = set()
    for row in failures or [{'details': details}]:
        evidence = row.get('details', {})
        if evidence.get('first_world_collision'):
            checks.append({'name': 'world_collision', 'status': 'failed',
                'candidate_id': row.get('candidate_index'), 'stage': row.get('stage'),
                'question': '完整机器人是否满足当前实测障碍的硬净距？',
                'actual': deepcopy(evidence['first_world_collision'])})
        coverage_status = 'diagnostic' if evidence.get('unknown_space_motion_gate') is False else 'failed'
        for coverage in evidence.get('observation_coverage', []):
            if coverage.get('covered') is not False:
                continue
            key = (row.get('candidate_index'), coverage.get('first_unobserved_sample_index'),
                   coverage.get('sample_offset'))
            if key in seen:
                continue
            seen.add(key)
            fields = ('first_unobserved_sample_index', 'sample_offset', 'first_unobserved_sphere_index',
                      'first_unobserved_sphere_world_m', 'first_unobserved_motion_envelope_m',
                      'closest_capture_robot_sphere_world_m', 'minimum_capture_containment_gap_m',
                      'unobserved_sphere_count', 'unobserved_sphere_count_is_lower_bound',
                      'coverage_evaluation_complete', 'first_unobserved_link', 'segment_joint_delta_rad',
                      'camera_certificates', 'diagnosis_scope', 'union_subdivision_resolution_m', 'scene_token')
            checks.append({'name': 'observed_motion_coverage', 'status': coverage_status,
                           'candidate_id': row.get('candidate_index'),
                           'question': '轨迹扫掠体积是否都有有效观测覆盖证据？',
                           'actual': {field: coverage[field] for field in fields if field in coverage},
                           'expected': 'diagnostic_only_not_motion_gate' if coverage_status == 'diagnostic' else 'covered=true'})
            for camera in coverage.get('camera_certificates', []):
                checks.append({'name': 'camera_motion_coverage', 'status': coverage_status,
                    'candidate_id': row.get('candidate_index'), 'camera_id': camera.get('camera'),
                    'question': '该相机的投影和深度是否单独证明整个扫掠球可见？',
                    'actual': deepcopy(camera), 'expected': 'whole_sphere_certified'})
            if len(checks) >= 12:
                break
        if len(checks) >= 12:
            break
    return checks, [{'stage': stage, 'code': code, 'count': count}
                    for (stage, code), count in counts.items()]


def build_failure_report(code, details, state):
    component = _component(code, details)
    checks = deepcopy(details.get('diagnostic_checks', []))
    known = list(details.get('known_facts', []))
    unknown = list(details.get('unknowns', []))
    counts = []
    # Query evidence remains useful even when the later failure is identity or depth.
    for query in details.get('sam3_queries', []):
        checks.append({key: deepcopy(query[key]) for key in (
            'component', 'stage', 'camera_id', 'prompt', 'status', 'candidate_count',
            'candidate_scores', 'error', 'request_hash') if key in query})
        checks.extend(deepcopy(query.get('diagnostic_checks', [])))
    if component == 'motion_planning':
        if details.get('stage') == 'source_clearance_lift':
            checks.append({'name': 'source_clearance_lift', 'status': 'failed',
                'question': '当前发布的来源碰撞层与目标测量是否支持抬升高度计算？',
                'actual': deepcopy(details)})
        planning_checks, counts = _planning_evidence(details)
        checks.extend(planning_checks)
        summary = '运动规划或执行前的几何检查未通过。'
        if counts:
            known.append('候选失败记录共 ' + str(sum(row['count'] for row in counts))
                         + ' 条；同一候选可能在不同阶段出现，不能当作独立候选数量。')
        if any(check['name'] == 'observed_motion_coverage' and check['status'] == 'failed'
               for check in planning_checks):
            known.append('存在扫掠体积缺少有效观测覆盖证据的记录；这不等于已证实发生碰撞。')
            unknown.append('单凭覆盖拒绝不能确定是实际遮挡、视野缺口还是数值边界；需核对坐标、包络与捕获几何。')
        next_step = '按候选、阶段和拒绝码核查轨迹；覆盖不足时核对首个缺口及数值边界，再选择显式补观测或修正几何检查。'
    elif component == 'curobo_resource':
        summary = 'cuRobo GPU 显存不足，本次计算未完成；不能据此判断任务不可达或碰撞无解。'
        if details.get('allocator_message'):
            checks.append({'name': 'cuda_memory_allocator', 'status': 'failed',
                           'actual': details['allocator_message']})
        next_step = '核对分配器原始消息、常驻规划器显存及当前步骤的临时分配峰值。'
    elif component == 'perception_mapping':
        summary = '清单碰撞地图缺少当前有效几何；观测部分成功不表示地图已完整更新。'
        for field, label in (('missing_entity_ids', '缺少地图层'), ('stale_entity_ids', '几何已过期')):
            entities = details.get(field, [])
            if entities:
                known.append(label + '：' + '、'.join(str(entity) for entity in entities))
        unknown.append('几何缺失本身不能确定是遮挡、分割零候选还是身份未确认；需核对当前采集证据。')
        next_step = '核对相应实体的三相机 SAM3 候选、Decisions 身份判断和深度证据；由任务图决定补观测或其他恢复动作。'
    elif component == 'decisions':
        summary = 'Decisions 请求、返回解析或判断验收未通过；具体环节见错误码和检查项。'
        if not unknown:
            unknown.append('接口未提供模型内部判断理由；分数本身不能证明是遮挡、颜色、提示词或分割造成。')
        next_step = '打开对应 observation_id/request_hash 的当前图像、候选标注、问题和原始回答，核对具体未通过项。'
    elif component == 'sam3':
        summary = 'SAM3 候选生成或分割证据未满足后续要求；返回零候选不等于分割低分。'
        if not unknown:
            unknown.append('没有服务返回的分数及内部过滤规则时，不能断言是分割置信度低。')
        next_step = '按相机和 prompt 检查返回候选数、分割分数、掩码面积和有效深度像素；区分服务错误、零候选和几何过滤。'
    else:
        summary = '任务在当前步骤停止，具体原因见错误码和记录证据。'
        next_step = '核查失败步骤和归档证据；未记录的原因保持未知。'
    if details.get('message'):
        checks.append({'name': 'reported_error', 'status': 'failed', 'actual': details['message']})
    if details.get('validation_errors'):
        checks.append({'name': 'policy_validation', 'status': 'failed',
            'question': 'Responses 输出是否满足类型、引用范围和技能契约？',
            'actual': deepcopy(details['validation_errors'])})
    evidence = details.get('evidence', {})
    # Long inventories must not hide the rejected candidate beyond the preview limit.
    checks.sort(key=lambda row: (
        not (row.get('candidate_id') is not None and row.get('candidate_id') == details.get('candidate_id')),
        row.get('required') is False,
        row.get('status') not in {'unknown', 'no', 'fail', 'failed', 'error', 'no_candidates'}))
    report = {'component': component, 'code': code, 'summary': summary,
              'stage': details.get('stage') or (state.get('current_step') or state.get('failed_step') or {}).get('skill'),
              'target_id': state.get('target_id'), 'arm': state.get('arm') or details.get('arm'),
              'observation_id': details.get('observation_id') or evidence.get('observation_id'),
              'request_hash': details.get('request_hash') or evidence.get('request_hash'),
              'checks': [_bounded(row) for row in checks[:32]],
              'known_facts': known, 'unknowns': unknown, 'next_step': next_step,
              'failure_counts': counts, 'evidence_refs': deepcopy(details.get('evidence_refs', []))}
    if report['request_hash']:
        report['evidence_refs'].append({'request_hash': report['request_hash']})
    return report


def failure_record(error, state):
    code = getattr(error, 'code', None) or str(error)
    details = exception_details(error)
    record = {}
    if isinstance(error, (PolicyStopped, UnhandledSkillResult)):
        record['policy_failure'] = {'code': code, 'details': deepcopy(details)}
        result = details.get('result') or details.get('last_result') or {}
        if result.get('status') in {'failure', 'failed', 'unknown', 'rejected'} or result.get('success') is False:
            code = result.get('error_code') or result.get('reason') or code
            details = result_details(result)
            if result.get('evaluations'):
                details['diagnostic_checks'] = [{'name': 'arm_reachability', 'actual': result['evaluations']}]
    record.update(error_code=code, error_details=details,
                  failure_report=build_failure_report(code, details, state))
    return record


def result_details(result):
    """Verification skills return their own evidence directly, unlike raised service errors."""
    fields = ('component', 'stage', 'purpose', 'observation_id', 'request_hash',
              'diagnostic_checks', 'known_facts', 'unknowns', 'evidence_refs', 'sam3_queries')
    details = {}
    for source in (result.get('visual_evidence') or {}, result, result.get('details') or {}):
        details.update({key: deepcopy(source[key]) for key in fields if key in source})
    details.update(deepcopy(result.get('details') or {}))
    for key in ('observation_assessment', 'workspace_evidence'):
        if result.get(key):
            details.setdefault('diagnostic_checks', []).append({'name': key, 'actual': deepcopy(result[key])})
    return details

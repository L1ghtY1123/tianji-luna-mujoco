"""Factual run conclusions from recorded evidence; never selects recovery actions."""
from copy import deepcopy

REASONS = {
    'low_confidence': '任务程序 的判断未达到当前置信度门槛，本次选择未应用；不代表物体不可达或运动规划失败。',
    'release_endpoint_not_reached': '实际 TCP 未到固定释放位置，或 attachment 碰撞球最低点到实测支撑面的间距与请求不符；未松爪，保留持物状态，由 任务程序 决定下一步。',
    'destination_support_reference_stale': '目的地支撑面来自旧观测，不能用于本次放置高度；需要当前测量证据。',
    'release_requires_uploaded_attachment_spheres': '缺少本次已上传的持物碰撞球，无法计算球底释放间距；未生成放置轨迹。',
    'result_action_capture_required': '缺少当前目标、当前代次的动作结果会话，不能补拍结果；本次调用未执行。',
    'CUROBO_CUDA_OUT_OF_MEMORY': 'cuRobo 计算时 GPU 显存不足，未得到规划结果；这不是 IK 不可达或碰撞无解。',
    'jev_input_budget_exceeded': '任务程序 决策输入超过上下文预算，未能作出下一步决定；这不代表物理动作失败。',
    'target_missing': '当前没有选出符合任务约束且仍待处理的目标；这不等于场景中没有物体。',
    'task_source_region_unconfirmed': '选中的物体没有符合本任务盒内来源约束的实测证据，已拒绝加入清单或绑定目标；没有执行搬运。来源不明不等于物体不存在。',
    'requested_quantity_membership_ambiguous': '任务程序 确认的搬运成员超过指令要求数量，本次成员修改未应用；需要重新明确要搬哪些对象。',
    'requested_quantity_already_completed': '指令规定的物体数量已完成，拒绝继续抓取额外物体；由 任务程序 核对清单并结束。',
    'target_violates_inventory_order': '默认清单顺序要求先处理更小编号的未完成物体，本次目标绑定已拒绝；未执行搬运。',
    'semantic_scene_identity_unresolved': '当前观测仍有清单物体未见或身份关联不唯一；为避免删除真实障碍，未发布碰撞图，匹配证据保留在运行记录。',
    'declared_obstacle_not_observed': '未识别到指令明确声明的环境障碍，或只有同色任务目标而无独立障碍候选；不能把该区域视为空位。',
    'tabletop_robot_clearance_violated': '实测桌面 cuboid 与当前机器人碰撞球的间距低于配置硬阈值，本次建图拒绝发布；需核对观测和位姿。',
    'no_reachable_arm': '未能选出同时满足源位置和目的位置可达性要求的机械臂。有限求解或离线图未命中不等于物理上绝对不可达。',
    'workspace_file_missing': '缺少离线工作空间文件，需要安装完整图；当前不能判断机械臂是否可达。',
    'workspace_model_mismatch': '离线工作空间与当前机器人模型不匹配，需要重新生成；当前不能判断机械臂是否可达。',
    'invalid_workspace_artifact': '离线工作空间文件不完整或格式无效，需要重新生成；当前不能判断机械臂是否可达。',
    'planning_no_valid_candidate': '已尝试的抓取候选未得到完整可执行轨迹。',
    'grasp_no_downward_candidate': '本轮抓姿均未通过向下45度以内且不朝机器人的方向筛选，尚未进入IK。',
    'grasp_no_fixed_slot_reachable_candidate': '方向合格的抓姿未通过桌面放置可达性预筛选，尚未进入IK。',
    'no_collision_free_source_ik': '三轮抓姿均未得到可用的抓取IK解，尚未进入轨迹规划。',
    'place_no_valid_candidate': '已尝试的放置位置和 yaw 未得到完整通过碰撞检查的放置及撤离轨迹。',
    'placement_region_no_supported_candidate': '实测放置区域无法容纳当前持物包络及安全余量，未进入 IK 或轨迹计算；需要核对目标点云、持物包络和区域边界。',
    'placement_region_no_free_candidate': '本次有限放置候选与已占用位置或盒壁重叠，未进入 IK 或执行放置；不会通过抬高释放位置叠放。这不证明整个区域没有空位，由 任务程序 决定下一步。',
    'placement_requires_measured_region_polygon': '缺少实测放置区域边界，无法按任务物体数量划分位置，未进入放置规划。',
    'placed_object_obstacle_unavailable': '已放置的盒内物体不可见，且没有可复用的体素几何，不能将它占用的位置当作空位；本次建图未发布，由 任务程序 决定如何补充证据。',
    'placed_table_object_not_observed': '当前观测未关联到已放在桌面的物体，不能确认其障碍位置；本次建图未执行，由 任务程序 决定如何补充观测。',
    'placed_object_not_observed': '当前观测未关联到已放置物体，不能确认其障碍位置；本次建图未执行，不用预测体素代替，由 任务程序 决定如何补充观测。',
    'placed_object_visibility_unresolved': '当前证据不能唯一关联已放置物体；本次建图未执行，由 任务程序 决定如何补充观测。',
    'grasp_verification_failed': '执行结束后的证据未通过抓取验证。',
    'place_verification_failed': '执行结束后的证据未通过放置验证。',
    'placement_not_visible': '当前视觉分析缺少目标或目的区域的对应 ID/深度，目标是否正确落位仍未知。',
    'placement_depth_insufficient': '已识别目标的有效深度不足，无法确认放置结果。',
    'destination_region_not_visible': '当前视图缺少目标与目的区域的共同可见证据，无法确认落点。',
    'destination_floor_not_visible': '缺少目的区域的实测支撑面，无法完成放置验证。',
    'sam3_service_unavailable': 'SAM3 请求失败，未取得可用于验证的分割结果。',
    'tcp_not_retreated': '当前反馈未确认末端已撤离释放位置，放置验证仍不充分。',
    'open_position_not_confirmed': '当前夹爪反馈未确认张开，放置验证仍不充分。',
    'place_execution_incomplete': '放置及撤离动作尚未完整结束，不能仅凭空爪判定放置成功。',
    'release_not_recorded': '本轮未记录成功释放，不能仅凭空爪判定放置成功。',
    'instance_association_ambiguous': '当前观测不能唯一关联物体 ID；未将不确定匹配当作同一物体。',
    'algorithm_service_unavailable': '算法服务请求失败；这不能证明目标或路径不可行。',
    'plan_grasp_timeout': '抓取规划超过本次时间预算，未取得可执行结果。',
    'plan_place_timeout': '放置规划超过本次时间预算，未取得可执行结果。',
    'skill_selection_budget_exhausted': '任务程序 连续选择不满足条件的调用，已达到调用拒绝上限。',
    'execution_collision': '轨迹未通过当前碰撞场景的复核，未允许执行该轨迹。',
    'retry_plan_requires_current_evaluation': '重试规划调用缺少当前碰撞地图或抓姿 IK 评估；中心位置 IK 不能代替抓姿评估。',
}
STATUS = {'failed': '任务未完成', 'needs_attention': '任务需要处理', 'partial_success': '任务部分完成',
          'succeeded': '任务已完成', 'cancelled': '任务已取消'}

STEPS = {'observe_scene': '观测建清单', 'bind_inventory': '确认初始清单', 'plan_grasp': '抓取规划',
         'generate_grasps': '生成抓姿', 'execute_grasp': '执行抓取', 'verify_grasp': '抓取反馈核验',
         'plan_place': '桌面放置规划', 'execute_place': '执行桌面放置', 'verify_place': '桌面放置核验',
         'prepare_unpack': '移动到拆包位姿', 'mock_unpack': '模拟拆包', 'verify_unpack': '拆包核验',
         'plan_transport': '右盒投放规划', 'execute_transport': '持物移动到右盒',
         'drop': '右盒释放', 'verify_drop': '右盒投放核验', 'move_to_observation': '返回观测位姿'}


def brief_error(code, skill=None, message=None, details=None):
    """A short explanation for operators; complete diagnostics stay in evidence."""
    code = str(code or 'unknown_error')
    key = code.split(':', 1)[0]
    reasons = {
        'drop_carried_projection_outside_bin': '持物投影超出右盒内腔，尚未开始运输。请检查抓姿和投放点。',
        'semantic_scene_identity_unresolved': '部分物体身份未确认，碰撞场景尚未更新。',
        'visual_identity_unknown': '当前图像未能确认物体身份，尚未更新碰撞场景。',
        'CUROBO_TRAJECTORY_COLLISION': '规划轨迹未通过碰撞检查，未执行该次移动。',
        'CUROBO_NO_SOLUTION': '本次规划未找到可执行轨迹，未开始移动。',
        'CUROBO_TRAJOPT_FAILED': '轨迹优化未得到可执行路径，未执行该次移动。',
        'CUROBO_NO_VALID_TRAJECTORY': '该目标位姿未得到可执行轨迹，尚未移动或松爪。',
        'program_stopped': code.partition(':')[2] or '流程成功条件未满足。',
    }
    reason = reasons.get(key) or REASONS.get(key, '').split('；')[0].split('。')[0]
    if key == 'visual_identity_unknown' and details and details.get('candidate'):
        candidate = str(details['candidate']).replace('candidate-', 'C')
        reason = f"相机 {details.get('camera_id', '?')} 的候选 {candidate} 身份未确认，尚未更新碰撞场景。"
    if not reason and ('_http_' in key or key.endswith('_transport_error')):
        reason = ('Responses' if key.startswith('visual_discovery') else 'Decisions') + ' 请求失败，请检查服务连接。'
    if not reason and 'timeout' in key:
        reason = '计算超过本次时间预算，尚未得到可执行结果。'
    if not reason:
        reason = (message if message and any('\u4e00' <= char <= '\u9fff' for char in message)
                  else '该节点报错，详细原因尚未归类')
    return f"{STEPS.get(skill, skill or '任务')}：{reason[:160].rstrip('。')}。" + (
        '' if key in reasons or key in REASONS else f'（{key[:100]}）')


def run_conclusion(state, events):
    failures = [deepcopy(event['data']) for event in events if event.get('kind') == 'task_failure']
    rejections = [deepcopy(event['data']) for event in events if event.get('kind') == 'skill_error'
                  and isinstance(event.get('data', {}).get('details'), dict)
                  and event['data']['details'].get('selected_skill')]
    progress = state.get('task_progress') or {}
    completed_ids = {item['id'] for item in progress.get('completed', [])}
    unresolved = [item for item in failures if item.get('target_id') not in completed_ids]
    root = deepcopy(unresolved[0] if unresolved else (state.get('failure') or progress.get('failure') or {}))
    if not root and state.get('error'):
        root = {'code': state.get('error_code') or state['error'],
                'stage': (state.get('failed_step') or {}).get('skill'),
                'evidence': {'error': state['error']}}
        for event in reversed(events):
            data = event.get('data') or {}
            if (event.get('kind') == 'skill_error' and data.get('skill') == root['stage']
                    and data.get('code') == root['code'] and isinstance(data.get('details'), dict)):
                root['evidence']['details'] = deepcopy(data['details'])
                break
    verification = state.get('verification') or {}
    if not root and verification.get('status') == 'unknown':
        root = {'code': verification.get('reason') or 'verification_unknown',
                'stage': 'verify_place' if state.get('facts', {}).get('released') else 'verify_grasp',
                'evidence_status': 'unknown', 'evidence': deepcopy(verification)}
    if state.get('status') == 'succeeded':
        root = {}
    code = str(root.get('code') or '')
    reason = REASONS.get(code.split(':')[0], '')
    if code.startswith(('visual_discovery_http_', 'visual_decisions_http_')):
        service = 'Responses' if code.startswith('visual_discovery_') else 'Decisions'
        reason = f'{service} 视觉接口返回 HTTP {code.rsplit("_", 1)[-1]}，未得到本次图片判断，已保留请求档案并停止；未确认物体消失或搬运完成。'
    if root and not reason:
        reason = ('当前验证证据不足，尚不能确认结果。' if root.get('evidence_status') == 'unknown'
                  else '该步骤返回错误；当前证据不足以确定更具体的原因。')
    summary = STATUS.get(state.get('status'), '任务状态未知')
    if state.get('status') == 'succeeded' and verification.get('verification_scope') == 'release_only':
        summary += '。按释放标准完成：放置动作结束且夹爪反馈确认张开；物体最终落点未核验。'
    if state.get('status') == 'succeeded' and verification.get('verification_scope') == 'release_and_visual_destination':
        summary += '。夹爪反馈确认释放，放后图像确认原 ID 物体已到达指定区域；未测量最终精确位姿。'
    if root:
        label = '证据不足步骤' if root.get('evidence_status') == 'unknown' else '失败步骤'
        summary += f"。{label}：{root.get('stage') or '未记录'}。{reason} 技术原因：{code}。"
    if any(event.get('kind') == 'scene_identity_blocked' for event in events):
        summary += ' 身份关联未确认，本轮已停止，不再自动 recover。'
    detail = root.get('evidence', {}).get('error')
    if detail and detail != code:
        summary += f' 记录详情：{str(detail)[:1000]}。'
    details = root.get('evidence', {}).get('details') or {}
    planning = details.get('planning_summary') or {}
    if (code == 'place_no_valid_candidate' and planning.get('generated_goal_count', 0) > 0
            and planning.get('generated_goal_count') == planning.get('ik_rejected_count')):
        summary += (f" 本次 {planning['generated_goal_count']} 个放置候选均未通过含持物碰撞的 IK 筛选，"
                    '未进入轨迹规划；这不证明运动学不可达。候选残差见记录；残差本身不能区分碰撞与求解器未收敛。')
    if code.startswith('low_confidence:'):
        field = details.get('field') or code.split(':', 1)[1]
        label = {'quantity': '搬运数量', 'task': '任务类型', 'arm': '指定执行臂',
                 'selection_order': '目标选择顺序', 'target': '目标身份'}.get(field, field)
        summary += f' 未确定字段：{label}。'
        if 'selected' in details:
            selected = details['selected']
            if field == 'quantity' and selected == 'unbounded':
                selected = '未明确指定数量'
            summary += f' 任务程序 返回：{selected}。'
        confidence, minimum = details.get('confidence'), details.get('minimum_confidence')
        if isinstance(minimum, (int, float)):
            if isinstance(confidence, (int, float)):
                summary += f' 置信度 {confidence:.2f}，要求至少 {minimum:.2f}。'
            else:
                summary += f' 服务未返回置信度，要求至少 {minimum:.2f}。'
        if root.get('stage') == 'bind_task':
            summary += ' 此次任务尚未开始观测或运动。'
    if (code in {'placement_region_no_free_candidate', 'placement_region_no_supported_candidate'}
            and isinstance(details, dict) and {'section_count', 'yaw_count_per_position'} <= details.keys()):
        summary += f" 本次检查 {details['section_count']} 个候选位置，每处 {details['yaw_count_per_position']} 个 yaw。"
        positions = details.get('positions', [])
        # Stored public evidence may contain a bounded prefix for large tasks.
        if isinstance(positions, dict):
            positions = positions.get('first', [])
        for position in positions:
            summary += (f" 位置 {position['position_index'] + 1}：占用 {position['occupied_rejections']}、"
                        f"边界 {position['boundary_rejections']} 个 yaw 被淘汰。")
    terminal = state.get('error_code') or state.get('error')
    if terminal and terminal != code:
        terminal_step = (state.get('failed_step') or {}).get('skill') or '未记录'
        message = state.get('error') or terminal
        summary += f' 最终停止步骤：{terminal_step}；原因：{terminal}。'
        if message != terminal:
            summary += f' {str(message)[:1000]}。'
    evaluations = root.get('evidence', {}).get('arm_evaluations') or {}
    for arm, evaluation in evaluations.items():
        if not isinstance(evaluation, dict):
            continue
        for key, label in (('source', '源位置'), ('destination', '目的位置')):
            point = evaluation.get(key) or {}
            if point.get('method') == 'offline_workspace_lookup':
                verdict = '命中' if point.get('reachable') else '未命中'
                summary += f' {arm} 臂{label}离线工作空间{verdict}，网格 {point["resolution_m"] * 1000:g} mm。'
                continue
            error = point.get('position_error_m')
            if isinstance(error, (float, int)):
                verdict = '通过' if point.get('reachable') else '未通过'
                summary += f' {arm} 臂{label} IK {verdict}，位置残差 {error * 1000:.1f} mm。'
    if rejections:
        summary += f' 有 {len(rejections)} 次调用因前提不满足被拒绝，未执行，也未计为新的物理失败。'
    if state.get('status') != 'succeeded' and verification.get('status') == 'unknown':
        if state.get('facts', {}).get('place_execution_completed'):
            summary += ' 放置动作已执行，最终落点仍待验证。'
        if verification.get('open_confirmed'):
            summary += ' 夹爪反馈已确认张开。'
        summary += ' 验证证据不足，结果仍为 unknown。'
        observation_reasons = list(dict.fromkeys(filter(None, [verification.get('observation_reason')] + [
            attempt.get('observation_reason') for attempt in verification.get('camera_attempts', [])])))
        if observation_reasons:
            summary += ' 观测缺口：' + '；'.join(observation_reasons) + '。'
        assisted = state.get('visual_result') or {}
        if assisted.get('status') == 'unknown' and assisted.get('reason'):
            summary += f" OpenAI 视觉辅助未提供有效结论：{assisted['reason']}"
            if assisted.get('http_status'):
                summary += f"（HTTP {assisted['http_status']}）"
            summary += '。'
    completed = len(progress.get('completed', []))
    if progress:
        summary += f' 清单已完成 {completed} 项；未完成项及暂缓原因见物体清单。'
    brief = brief_error(code, root.get('stage'), state.get('error'), root.get('evidence', {}).get('details')) if root else summary
    return {'status': state.get('status'), 'summary': summary, 'brief': brief, 'root_cause': root,
            'failures': failures, 'rejected_calls': rejections,
            'verification': deepcopy(verification), 'visual_assistance': deepcopy(state.get('visual_result')),
            'final_error': state.get('error'), 'completed_count': completed}

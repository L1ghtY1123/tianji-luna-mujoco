// Historical recordings may predate the backend conclusion. Preserve what is known.
export function runConclusion(run = {}, events = [], describeError = null) {
  if (['running', 'idle', 'succeeded'].includes(run.status)) return '';
  const brief = run.conclusion?.brief ?? run.error_summary;
  if (brief && !brief.includes('未归类')) return brief;
  const failures = events.filter(event => event.kind === 'task_failure').map(event => event.data);
  const root = failures.findLast(failure => failure.stage !== 'recover_task')
    ?? run.failure ?? run.task_progress?.failure ?? failures.at(-1);
  const code = root?.code ?? run.error_code ?? run.error;
  const step = root?.stage ?? run.failed_step?.skill ?? '未记录';
  if (String(code).split(':')[0] === 'drop_carried_projection_outside_bin') {
    return '右盒投放规划：持物投影超出盒内范围，尚未开始运输；抓取已完成。';
  }
  const planningErrors = { visual_identity_unknown: '当前观测未能确认物体身份，清单与碰撞场景未更新。',
    CUROBO_TRAJOPT_FAILED: '轨迹优化未得到可执行路径，未执行该次移动。',
    CUROBO_TRAJECTORY_COLLISION: '规划轨迹未通过碰撞检查，未执行该次移动。' };
  if (planningErrors[code]) return `${({ observe_scene: '观测核验', prepare_unpack: '移动到拆包位姿', plan_transport: '右盒投放规划',
    move_to_observe: '返回观测位姿', plan_grasp: '抓取规划', plan_place: '桌面放置规划' })[step] ?? step}：${planningErrors[code]}`;
  if (run.conclusion?.brief || run.error_summary) return run.conclusion?.brief ?? run.error_summary;
  if (run.conclusion?.summary) return run.conclusion.summary;
  if (!code) return run.status === 'needs_attention' ? '验证或决策证据不足，任务需要处理。' : '';
  const reason = {
    no_reachable_arm: '未能选出满足源位置和目的位置可达性要求的执行臂。',
    retry_plan_requires_current_evaluation: '重试规划缺少当前场景或抓姿 IK 评估。',
    instance_association_ambiguous: '当前观测不能唯一对应物体 ID。',
    release_endpoint_not_reached: '实际释放位置尚未到位；夹爪保持闭合，未高处释放。',
    destination_support_reference_stale: '放置支撑面来自旧观测，需要当前测量。',
    release_requires_uploaded_attachment_spheres: '缺少已上传的持物碰撞球，不能计算球底释放间距。',
    result_action_capture_required: '缺少当前动作结果会话，补拍调用未执行。',
    grasp_no_downward_candidate: '本轮抓取候选均未通过向下45度及远离机器人方向筛选。',
    grasp_no_fixed_slot_reachable_candidate: '方向合格的抓取候选均未通过固定放置区的名义可达性筛选。',
    no_collision_free_source_ik: '有界采样预算内没有找到通过碰撞检查的源抓取IK，尚未进入抓取轨迹规划。',
  }[String(code).split(':')[0]] ?? '当前证据不足以确定更具体原因。';
  const raw = root?.evidence?.error ?? run.error;
  const detail = raw && raw !== code && !String(raw).startsWith(`${code}:`) ? `${code}:${raw}` : (raw || code);
  const description = describeError?.(detail);
  let text = `任务未完成。失败步骤：${step}。` + (description && description !== detail
    ? description : `${reason} 技术原因：${detail}。`);
  const terminal = run.error_code ?? run.error;
  if (terminal && terminal !== code) text += ` 最终停止原因：${terminal}。`;
  return text;
}

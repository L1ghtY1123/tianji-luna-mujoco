const FAILURE_EVENTS = new Set(['task_failure', 'skill_error']);
const SUCCESS_STATUSES = new Set(['success', 'succeeded', 'complete', 'completed']);
const REPORT_LISTS = ['checks', 'failure_counts', 'known_facts', 'unknowns', 'evidence_refs'];

function reportItems(value) {
  return Array.isArray(value) ? value : Array.isArray(value?.first) ? value.first : [];
}

export async function loadArchivedFailureReport(events, fetchArtifact) {
  const source = events.findLast(event => event.data?.failure_report);
  const report = source?.data.failure_report;
  if (!source?.artifact || !REPORT_LISTS.some(key => report[key]?.first)) return events;
  let fullReport;
  try {
    fullReport = (await fetchArtifact(source.artifact)).failure_report;
    if (!fullReport) throw new Error('归档未包含完整失败报告');
  } catch (error) {
    fullReport = {...report, unknowns:[...reportItems(report.unknowns), `完整失败报告读取失败：${error.message}`]};
  }
  return events.map(event => event === source ? {...event, data:{...event.data, failure_report:fullReport}} : event);
}

function recordedFailure(run, events) {
  const end = events.findLast(event => event.kind === 'run_end')?.data ?? {};
  const report = run.failure_report ?? end.failure_report
    ?? events.findLast(event => event.data?.failure_report)?.data.failure_report;
  const failures = events.filter(event => FAILURE_EVENTS.has(event.kind));
  const succeeded = SUCCESS_STATUSES.has(run.status ?? end.status);
  return { report: succeeded ? null : report, end, failures, succeeded };
}

export function failureSummary(run = {}, events = []) {
  const { report, end, failures, succeeded } = recordedFailure(run, events);
  if (succeeded) return '';
  if (report) return report.summary || report.code || '失败报告未记录结论';
  const failure = run.failure ?? end.failure ?? failures.at(-1)?.data;
  const conclusion = run.conclusion?.brief || run.error_summary || run.conclusion?.summary
    || end.conclusion?.brief || end.error_summary || end.conclusion?.summary
    || '';
  if (conclusion) return conclusion;
  const reason = run.error || end.error || run.error_code || end.error_code || failure?.error || failure?.code;
  if (!reason) return '';
  const stage = run.failed_step?.skill ?? end.failed_step?.skill ?? failure?.stage;
  return `${stage ? `失败步骤：${stage}。` : ''}${display(reason)}`;
}

function display(value) {
  return typeof value === 'string' ? value : JSON.stringify(value);
}

const MEASUREMENT_LABELS = {
  covered: '覆盖检查通过',
  sample_index: '轨迹采样索引',
  first_unobserved_sample_index: '首个未覆盖的轨迹采样索引',
  first_unobserved_sphere_index: '首个未覆盖的碰撞球索引',
  first_unobserved_sphere_world_m: '首个未覆盖碰撞球（世界坐标 x,y,z 与半径，m）',
  closest_capture_robot_sphere_world_m: '采集时最近的机器人碰撞球（世界坐标 x,y,z 与半径，m）',
  maximum_motion_envelope_m: '最大运动包络（m）',
  first_unobserved_motion_envelope_m: '首个未覆盖位置的运动包络（m）',
  minimum_capture_containment_gap_m: '距采集时机器人包络的最小包含缺口（m）',
  depth_uncertainty_m: '深度不确定度（m）',
  checked_sphere_count: '已检查碰撞球数量',
  requested_sphere_count: '需要检查的碰撞球数量',
  unobserved_sphere_count: '未覆盖的碰撞球数量',
  unobserved_sphere_count_is_lower_bound: '未覆盖数量仅为下界',
  coverage_evaluation_complete: '覆盖检查已遍历全部位置',
  continuous_segment_envelope_checked: '已检查连续轨迹段包络',
  first_unobserved_link: '首个未覆盖的机器人连杆',
  segment_joint_delta_rad: '该轨迹段的关节变化量（rad）',
  projected_bounds_px: '投影边界（左、上、右、下，像素）',
  image_size_px: '图像尺寸（宽、高，像素）',
  required_free_depth_m: '证明球体可见所需的深度（m）',
  minimum_depth_m: '投影区域内最小实测深度（m）',
  invalid_depth_pixels: '无有效深度的像素数',
  insufficient_depth_pixels: '深度不足以证明球体可见的像素数',
  candidate_wrist_distance_range_m: '候选腕部距离范围（m）',
  reach_screening_limit_m: '腕部可达距离筛选上限（m）',
  target_centroid_world_m: '目标实测中心（世界坐标，m）',
  published_envelope_minimum_world_m: '已发布碰撞范围下界（世界坐标，m）',
  published_envelope_maximum_world_m: '已发布碰撞范围上界（世界坐标，m）',
};

const DIAGNOSTIC_LABELS = {
  sam3_internal_filter_threshold: 'SAM3 服务未返回内部过滤阈值',
  sam3_internal_rejection_reason: 'SAM3 服务未返回内部拒绝原因',
  model_internal_reason_not_returned: '接口未返回模型内部判断理由',
  sam3_score_not_recorded: '此候选未记录 SAM3 分数',
};

function measurementRows(value, element) {
  const list = element('ul', '');
  for (const [key, measured] of Object.entries(value)) {
    const label = MEASUREMENT_LABELS[key] ?? key;
    const nested = measured !== null && typeof measured === 'object' && !Array.isArray(measured);
    const row = element('li', `${label}：${nested ? '' : display(measured)}`);
    if (nested) row.append(measurementRows(measured, element));
    list.append(row);
  }
  return list;
}

// One read-only renderer for live state and archived events. Missing evidence stays missing.
export function renderFailureReport(container, run = {}, events = [], { document = container.ownerDocument,
  additionalErrors = [] } = {}) {
  const { report, end, failures, succeeded } = recordedFailure(run, events);
  const open = container.children?.[1]?.open ?? false;
  const summary = failureSummary(run, events);
  const hasDetails = !succeeded && (run.error_details || end.error_details || failures.length);
  container.replaceChildren();
  container.hidden = !summary && !hasDetails && !additionalErrors.length;
  if (container.hidden) return;
  const element = (tag, text) => {
    const node = document.createElement(tag);
    node.textContent = text;
    return node;
  };
  container.append(element('p', display(summary || (succeeded
    ? '任务已成功，视频或归档存在异常记录。' : '失败原因未记录，请查看原始证据。'))));
  const details = element('details', '');
  details.open = open;
  details.className = 'failure-evidence';
  details.append(element('summary', '展开具体检查与证据'));
  if (report) {
    for (const [key, label] of [['component', '组件'], ['code', '错误码'], ['stage', '步骤'],
      ['target_id', '目标 ID'], ['arm', '执行臂'], ['observation_id', '观测 ID'], ['request_hash', '请求哈希']]) {
      if (report[key] != null) details.append(element('p', `${label}：${display(report[key])}`));
    }
    if (report.checks?.count > reportItems(report.checks).length) details.append(element('p',
      `当前仅显示 ${reportItems(report.checks).length} / ${report.checks.count} 项归档摘要，完整检查见原始事件文件。`));
    for (const check of reportItems(report.checks)) {
      const item = element('section', '');
      item.append(element('strong', check.name ?? '检查项'));
      if (check.question) {
        const question = Array.isArray(check.question)
          ? check.question.filter(part => part.type === 'input_text').map(part => part.text).join('\n')
          : check.question;
        item.append(element('p', `判断问题：${question}`));
      }
      const requirement = check.threshold == null
        ? check.expected != null ? '按预期条件检查' : '阈值未记录'
        : `要求：${check.comparison ?? '比较方式未记录'} ${display(check.threshold)}`;
      if (check.actual !== null && typeof check.actual === 'object' && !Array.isArray(check.actual)) {
        item.append(element('p', '实际值：'), measurementRows(check.actual, element), element('p', requirement));
      } else {
        item.append(element('p', `实际值：${check.actual == null ? '未记录' : display(check.actual)} · ${requirement}`));
      }
      for (const [key, label] of [['camera_id', '相机'], ['candidate_id', '候选 ID'], ['entity_id', '实体 ID']]) {
        if (check[key] != null) item.append(element('p', `${label}：${display(check[key])}`));
      }
      for (const [key, label] of [['answer', '原始回答'], ['metric', '指标'], ['sam3_score', 'SAM3 分数'],
        ['prompt', '查询描述'], ['candidate_count', '候选数量'], ['candidate_scores', '候选分数'],
        ['status', '记录状态'], ['reason', '记录原因'], ['expected', '预期值']]) {
        if (check[key] != null) item.append(element('p', `${label}：${display(check[key])}`));
      }
      const displayed = new Set(['name', 'question', 'actual', 'threshold', 'comparison', 'camera_id',
        'candidate_id', 'entity_id', 'answer', 'metric', 'sam3_score', 'prompt', 'candidate_count',
        'candidate_scores', 'status', 'reason', 'expected']);
      const remaining = Object.fromEntries(Object.entries(check).filter(([key]) => !displayed.has(key)));
      if (Object.keys(remaining).length) item.append(element('pre', JSON.stringify(remaining, null, 2)));
      details.append(item);
    }
    if (reportItems(report.failure_counts).length) {
      details.append(element('h3', '按阶段记录的失败次数'));
      for (const failure of reportItems(report.failure_counts)) details.append(element('p',
        `${failure.stage ?? '阶段未记录'} · ${failure.code ?? '原因未记录'} · ${failure.count ?? '次数未记录'} 次`));
    }
    for (const [key, label] of [['known_facts', '已知事实'], ['unknowns', '尚未确认']]) {
      if (!reportItems(report[key]).length) continue;
      details.append(element('h3', label));
      const list = element('ul', '');
      for (const fact of reportItems(report[key])) list.append(element('li',
        typeof fact === 'string' ? DIAGNOSTIC_LABELS[fact] ?? fact : display(fact)));
      details.append(list);
    }
    if (reportItems(report.evidence_refs).length) {
      details.append(element('h3', '证据引用'));
      for (const ref of reportItems(report.evidence_refs)) details.append(element('p', [
        ref.event_id != null ? `事件 #${ref.event_id}` : '', ref.artifact ? `文件：${ref.artifact}` : '',
        ref.request_hash ? `请求哈希：${ref.request_hash}` : '',
      ].filter(Boolean).join(' · ')));
    }
    if (report.next_step) details.append(element('p', `下一步：${display(report.next_step)}`));
  } else if (!succeeded) {
    details.append(element('p', '未归档结构化失败报告；以下为原始错误记录。'));
    const raw = { error: run.error ?? end.error, error_code: run.error_code ?? end.error_code,
      error_details: run.error_details ?? end.error_details, failed_step: run.failed_step ?? end.failed_step,
      failure: run.failure ?? end.failure, events: failures };
    details.append(element('pre', JSON.stringify(raw, null, 2)));
  }
  for (const error of additionalErrors) details.append(element('p', display(error)));
  container.append(details);
}

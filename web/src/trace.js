import { formatJson, traceDetails, confidenceLabel, candidateCount, createEventTitleResolver } from './model.js';

const EXECUTION_STAGE_LABELS = {
  execution_stage_start: '阶段开始',
  execution_stage_complete: '阶段完成',
  execution_stage_error: '阶段失败',
};
const SKILL_EVENT_LABELS = {
  skill_start: 'skill 开始',
  skill_complete: 'skill 完成',
  skill_error: 'skill 失败',
};
const SYSTEM_EVENT_LABELS = {
  planning_attempt_failed: '规划无解，交回 JEV',
};
const CATEGORY_LABELS = { jev: 'JEV 决策', skill: 'skill 执行', algorithm: '算法计算', system: '系统事件' };
const PLANNING_BATCH_LABELS = {
  planning_batch_start: 'plan_grasp_batch 开始',
  planning_batch_complete: 'plan_grasp_batch 完成',
  ik_batch_start: 'Batched IK 开始',
  ik_batch_complete: 'Batched IK 完成',
};
const PLACEMENT_LABELS = {
  placement_layout: '放置区域分区与位置候选',
  placement_position_checked: '固定 TCP XY 的世界 yaw 候选',
  placement_candidate_selected: '选定完整放置及撤离方案',
};

function candidateIndicesLabel(indices) {
  if (Array.isArray(indices)) return indices.join(', ');
  if (Array.isArray(indices?.first) && Number.isFinite(indices.count)) {
    return `${indices.first.join(', ')}（仅展示 ${indices.first.length} 个，共 ${indices.count} 个）`;
  }
  return '未提供';
}

// Display copy only: preserve the received events and JEV's original evidence.
function summarizeIkCandidateIds(value) {
  if (Array.isArray(value)) return value.map(summarizeIkCandidateIds);
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => {
    if (key === 'candidate_indices' && candidateCount(item) !== null) {
      return ['candidate_count', candidateCount(item)];
    }
    return [key, summarizeIkCandidateIds(item)];
  }));
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addJson(parent, label, value) {
  const section = element('details', 'trace-data');
  section.append(element('summary', '', label), element('pre', '', formatJson(value)));
  parent.append(section);
}

export function createTraceList(container, counter) {
  let signature = '';
  const expanded = new Set();
  return function render(events = [], contextEvents = events) {
    const nextSignature = JSON.stringify(events === contextEvents ? events : [events, contextEvents]);
    if (nextSignature === signature) return;
    signature = nextSignature;
    counter.textContent = events.length;
    if (!events.length) {
      container.replaceChildren();
      const empty = element('div', 'trace-empty');
      empty.append(element('div', 'trace-empty-icon', '⌘'), element('strong', '', '等待第一条指令'), element('p', '', '任务开始后，这里按时间记录 JEV 决策、skill 执行、算法计算和系统事件。'));
      container.append(empty);
      return;
    }
    const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
    const titleForEvent = createEventTitleResolver(contextEvents);
    const cards = events.map((event, index) => {
      const key = String(event.id ?? index);
      const isJev = event.kind === 'jev';
      const stageLabel = EXECUTION_STAGE_LABELS[event.kind];
      const stageSkill = event.data?.skill ?? '执行';
      const skillLabel = SKILL_EVENT_LABELS[event.kind];
      const systemLabel = SYSTEM_EVENT_LABELS[event.kind];
      const failed = ['error', 'skill_error', 'execution_stage_error', 'algorithm_error'].includes(event.kind);
      const algorithmEvent = ['algorithm_start', 'algorithm_complete', 'algorithm_error'].includes(event.kind);
      const placementLabel = PLACEMENT_LABELS[event.kind];
      const batchLabel = PLANNING_BATCH_LABELS[event.kind] && titleForEvent(event,
        event.kind.startsWith('planning_batch_') ? event.title ?? PLANNING_BATCH_LABELS[event.kind] : PLANNING_BATCH_LABELS[event.kind]);
      const ikBatch = ['ik_batch_start', 'ik_batch_complete'].includes(event.kind);
      const ikSkill = skillLabel && [event.title, event.data?.skill, event.data?.id].includes('evaluate_arms');
      const category = isJev ? 'jev' : stageLabel || skillLabel ? 'skill' : batchLabel || algorithmEvent || placementLabel ? 'algorithm' : 'system';
      const batchFailed = ['planning_batch_complete', 'ik_batch_complete'].includes(event.kind) && event.data?.success_count === 0;
      const card = element('details', `trace-card ${failed ? 'trace-error' : batchFailed || event.kind === 'planning_attempt_failed' ? 'trace-warning' : ''}`);
      card.open = expanded.has(key);
      card.addEventListener('toggle', () => { if (card.open) expanded.add(key); else expanded.delete(key); });
      const summary = element('summary', 'trace-summary');
      const number = element('span', 'trace-number', String(index + 1).padStart(2, '0'));
      const text = element('div', 'trace-title');
      const title = isJev && event.title === '选择技能' ? '选择 skill' : titleForEvent(event);
      text.append(element('strong', '', batchLabel ?? placementLabel ?? systemLabel ?? title ?? event.kind ?? '事件'));
      const data = traceDetails(event.data);
      const meta = element('div', 'trace-meta');
      meta.append(element('span', `trace-kind trace-kind-${category}`, CATEGORY_LABELS[category]));
      if (!isJev) meta.append(element('span', '', stageLabel ?? skillLabel ?? batchLabel ?? systemLabel ?? event.kind ?? 'event'));
      if (stageLabel) meta.append(element('span', '', `${stageSkill} 内部阶段`));
      if (data.duration_s !== null) meta.append(element('span', '', `${data.duration_s.toFixed(2)} s`));
      if (isJev && data.confidence !== null) meta.append(element('span', 'confidence', `置信度 ${confidenceLabel(data.confidence)}`));
      text.append(meta);
      if (batchLabel && Number.isFinite(event.data?.batch_index)) {
        const batch = event.data;
        const count = candidateCount(batch.candidate_indices);
        const candidates = ikBatch ? `候选 ${count ?? '—'} 个` : `候选编号 ${candidateIndicesLabel(batch.candidate_indices)}`;
        text.append(element('p', 'trace-event-note', `批次 ${batch.batch_index} / ${batch.batch_count ?? '—'} · ${candidates} · Arm ${batch.arm ?? '—'}`));
        if (Number.isFinite(batch.success_count) && event.kind !== 'ik_batch_start') {
          const remaining = ikBatch && count !== null && batch.success_count >= 0 && batch.success_count <= count
            ? ` · 未通过 ${count - batch.success_count} 个` : '';
          text.append(element('p', 'trace-event-note', `${ikBatch ? '通过' : '成功'} ${batch.success_count} 个${remaining}`));
        }
      }
      if (event.kind === 'planning_batch_complete' && event.data?.stopped_early === true) {
        const count = event.data.unplanned_candidate_count;
        const remaining = Number.isFinite(count) ? `；${count} 个候选未规划` : '';
        text.append(element('p', 'trace-event-note', `已找到最高 score 可行轨迹，停止后续批次${remaining}。`));
      }
      if (event.kind === 'planning_attempt_failed') {
        text.append(element('p', 'trace-event-note', `Arm ${event.data?.arm ?? '—'} 本次规划无解，尚未执行运动；JEV 决定重新选臂或停止。`));
      }
      if (event.kind === 'placement_layout') {
        const layout = event.data;
        text.append(element('p', 'trace-event-note', `${layout.target_count} 个目标 · ${layout.section_count} 个分区 · 每处 ${layout.yaw_count_per_position} 个 yaw`));
      }
      if (event.kind === 'placement_position_checked') {
        const position = event.data;
        const xyz = position.position_world_m.map(value => value.toFixed(3)).join(', ');
        text.append(element('p', 'trace-event-note', `位置 ${position.position_index + 1} (${xyz}) m · 候选 ${position.accepted_yaw_count} · 后续检查 IK 和三维碰撞`));
      }
      if (event.kind === 'placement_candidate_selected') {
        const selected = event.data;
        text.append(element('p', 'trace-event-note', `位置 ${selected.position_index + 1} · yaw ${selected.yaw_delta_deg.toFixed(0)}° · 候选 ${selected.candidate_index}。放置与撤离检查均通过，停止后续候选。`));
      }
      if (event.data?.error) {
        text.append(element('p', 'trace-event-error', `${event.data.skill ? `${event.data.skill}：` : ''}${event.data.error}`));
      }
      summary.append(number, text, element('span', 'trace-chevron', '⌄'));
      card.append(summary);
      const content = element('div', 'trace-content');
      if (stageLabel) {
        content.append(element('p', 'trace-value-note', `该阶段由 ${stageSkill} skill 内部顺序执行，不调用 JEV。`));
        addJson(content, '阶段数据', event.data ?? null);
      } else if (isJev) {
        addJson(content, '输入状态 · state', data.state);
        addJson(content, '问题 · questions', data.questions);
        addJson(content, '结构化输出 · output', data.output);
        addJson(content, 'JEV 返回详情 · probabilities / confidence', data.provider_details);
        content.append(element('p', 'trace-value-note', `置信度：${confidenceLabel(data.confidence)} · 耗时：${data.duration_s === null ? '未提供' : `${data.duration_s.toFixed(3)} s`}`));
        addJson(content, '完整事件数据（含错误与用量）', event.data);
      } else {
        addJson(content, '事件数据', ikBatch || ikSkill ? summarizeIkCandidateIds(event.data ?? null) : event.data ?? null);
      }
      if (event.time !== undefined) content.append(element('p', 'trace-value-note', `事件时间：${event.time}`));
      card.append(content);
      return card;
    });
    container.replaceChildren(...cards);
    if (nearBottom) container.scrollTop = container.scrollHeight;
  };
}

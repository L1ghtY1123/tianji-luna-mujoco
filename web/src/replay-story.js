import { buildObjectTimeline } from './execution-graph.js';
import { memoryEntities, renderTaskObjects } from './task-ledger.js';

const PURPOSES = {
  description_instance: '初始描述绑定', physical_instance: '额外物体确认',
  object_held: '夹持检查', held_object_is_target: '持物目标身份',
  compile_task: '理解任务并生成总编排', revise_graph: '根据新事件修订总编排',
  current_instance_correspondence: '跨相机掩码对应', instance_temporal: '跨时刻实例对应',
  instance_archive: '历史档案身份恢复', instance_novelty: '新实例确认',
  initialize_objects: '登记初始实例与任务图', associate_objects: '关联当前观测与时间记忆',
  associate_completed_objects: '核对已放置物体的身份',
  verify_release_result: '确认放置与松爪', verify_unpacking: '确认拆包产物',
  verify_right_bin_drop: '确认目的地投放',
  verify_bin_drop: '确认物体入盒', semantic_judgments: '核验任务条件', task_completion: '核验最终完成条件',
};
const STATUS_NAMES = { complete: '完成', failed: '失败', warning: '需关注', incomplete: '未记录结束', recorded: '已记录', cancelled: '已取消' };
const PHASES = {
  observe: ['观测与时间记忆', 'observe_scene', 'observe', 'bind_objects'],
  grasp: ['抓取物体', 'identify_target', 'identify_destination', 'choose_arm', 'build_scene', 'generate_grasps', 'evaluate_arms', 'plan_grasp', 'plan_grasp_and_place', 'execute_approach', 'execute_grasp', 'close_gripper', 'execute_lift', 'verify_grasp'],
  place: ['物体放置', 'plan_place', 'execute_place', 'verify_place', 'observe_result'],
  unpack: ['VLA 拆包', 'prepare_unpack', 'vla_unpack', 'verify_unpack'],
  return: ['返回观测位姿', 'move_to_observe'],
  system: ['任务记录'],
};

export function modelCallInfo(event) {
  const data = event.data ?? event;
  const type = data.endpoint?.endsWith('/responses') ? 'Responses'
    : data.endpoint?.endsWith('/decisions') ? 'Decisions' : '模型调用';
  const questionName = (data.request?.questions ?? data.questions)?.[0]?.name;
  let purpose = PURPOSES[questionName] ?? PURPOSES[data.purpose] ?? data.purpose ?? '用途未归档';
  if (type === 'Decisions' && data.purpose === 'initialize_objects') purpose = '核对分割提议与实例记忆';
  return { type, purpose, elapsed: data.elapsed_s,
    failed: Boolean(data.error) || (Number.isFinite(data.http_status) && data.http_status >= 400) };
}

export function buildReplayStages(graph) {
  const ownership = new Map();
  for (const group of buildObjectTimeline(graph)) {
    for (const stage of group.stages) for (const node of stage.nodes) ownership.set(node.id, group);
  }
  const stages = [];
  for (const node of graph.nodes) {
    const source = node.events.find(event => ['skill_start', 'skill_complete', 'skill_error'].includes(event.kind));
    const skill = source?.data?.skill ?? source?.title;
    const phase = Object.entries(PHASES).find(([, names]) => names.slice(1).includes(skill))?.[0]
      ?? (source ? `skill:${skill}` : null);
    const group = ownership.get(node.id);
    let stage = stages.at(-1);
    // Unscoped bookkeeping stays with its chronological stage, never with a guessed target.
    const targetId = group?.targetId ?? null;
    if (!stage || (phase && (phase !== stage.phase || (targetId && stage.targetId && targetId !== stage.targetId)))) {
      stage = { id: `story:${node.id}`, phase: phase ?? 'system', title: PHASES[phase ?? 'system']?.[0] ?? skill,
        targetId, displayId: group?.displayId ?? null, nodes: [], events: [], models: [] };
      stages.push(stage);
    }
    if (!stage.targetId && targetId) { stage.targetId = targetId; stage.displayId = group.displayId; }
    stage.nodes.push(node);
    stage.events.push(...node.events);
    stage.models.push(...node.events.filter(event => event.kind === 'visual_decisions_call'));
  }
  // Initial run metadata precedes the first skill; fold it into the first actual stage.
  if (stages[0]?.phase === 'system' && stages.length > 1) {
    const first = stages.shift();
    stages[0].nodes.unshift(...first.nodes);
    stages[0].events.unshift(...first.events);
    stages[0].models.unshift(...first.models);
  }
  for (const stage of stages) {
    if (stage.phase === 'unpack' && stage.nodes.some(node => node.events.some(event => event.title === 'vla_unpack'))) stage.title = 'VLA 拆包（仿真）';
    if (stage.phase === 'grasp') {
      const progress = graph.events.findLast(event => event.data?.task_progress)?.data.task_progress;
      const entities = new Map(memoryEntities(progress).map(item => [item.id, item]));
      const targetIds = [...new Set(stage.events.map(event => event.target_id ?? event.data?.target_id).filter(id => entities.get(id)?.kind === 'object'))];
      if (targetIds.length === 1) stage.title = `抓取 ${entityLabel(entities.get(targetIds[0]))}`;
    }
    const times = stage.events.map(event => event.t_s).filter(Number.isFinite);
    stage.startTime = times.length ? Math.min(...times) : null;
    stage.endTime = times.length ? Math.max(...times) : null;
    stage.duration = times.length ? stage.endTime - stage.startTime : null;
    stage.status = ['failed', 'warning', 'incomplete', 'cancelled'].find(status => stage.nodes.some(node => node.status === status))
      ?? (stage.nodes.some(node => node.status === 'complete') ? 'complete' : 'recorded');
  }
  return stages;
}

// Read committed memory snapshots and the entity fields archived by earlier runs.
export function objectsSnapshots(events = []) {
  const snapshots = [];
  for (const event of events) {
    const source = event.data?.temporal_memory ?? event.data?.task_progress;
    if (!source?.entities || typeof source.entities !== 'object') continue;
    const progress = Object.fromEntries(['entities', 'object_history', 'retired_entity_ids',
      'transformations', 'memory_revision'].filter(key => key in source)
      .map(key => [key, structuredClone(source[key])]));
    snapshots.push({ event, progress: structuredClone(progress), provenance: 'snapshot',
      title: event.kind === 'run_end' ? '任务结束时间记忆' : event.data?.checkpoint ?? '时间记忆快照' });
  }
  return snapshots;
}

const QUESTION_NAMES = { place_result: '放置结果', workpiece_still_held: '工件仍被夹持',
  description_instance: '初始描述绑定', physical_instance: '额外物体确认',
  object_held: '夹持检查', held_object_is_target: '持物目标身份',
  unpack_result: '处理产物核验', current_item_added: '本次投放结果' };

export function modelResult(archive) {
  const answers = Array.isArray(archive.response?.answers) ? archive.response.answers : [];
  const questions = archive.request?.questions ?? [];
  const result = { answers: answers.map(answer => {
    const question = questions.find(item => item.name === answer.name);
    const choice = question?.choices?.find(item => item.value === answer.choice);
    return { ...answer, check: archive.diagnostic_checks?.find(item => item.name === answer.name),
    choiceProbability: answer.probabilities?.find(item => item.value === answer.choice)?.probability,
    optionProbabilities: (question?.choices ?? []).map(option => ({...option,
      probability: answer.probabilities?.find(item => item.value === option.value)?.probability})),
    title: QUESTION_NAMES[answer.name] ?? (/^identity_/.test(answer.name)
      ? `分割提议 ${answer.name.slice(9)} · 身份判断` : answer.name),
    instructions: Array.isArray(question?.instructions)
      ? question.instructions.filter(part => part.type === 'input_text').map(part => part.text).join('\n')
      : question?.instructions ?? '完整问题未归档',
    choiceLabel: [answer.choice, choice?.description].filter(Boolean).join(' · ') };
  }), objects: archive.graph_proposal ?? null };
  if (archive.endpoint?.endsWith('/responses') && !result.objects) {
    for (const message of archive.response?.output ?? []) {
      if (message.type !== 'message') continue;
      for (const part of message.content ?? []) {
        if (part.type !== 'output_text') continue;
        try { result.objects = JSON.parse(part.text); } catch { /* Raw response remains available. */ }
      }
    }
  }
  return result;
}

function dom(container) {
  return (tag, text = '', className = '') => {
    const item = container.ownerDocument.createElement(tag);
    item.textContent = text;
    item.className = className;
    return item;
  };
}

export function renderReplayStages(container, graph, onSelect) {
  const element = dom(container);
  container.replaceChildren();
  const stages = buildReplayStages(graph);
  const buttons = [];
  function activate(button, stage, event = null) {
    for (const item of buttons) item.setAttribute('aria-pressed', String(item === button));
    onSelect(stage, event);
  }
  for (const [index, stage] of stages.entries()) {
    const card = element('section', '', `workflow-stage status-${stage.status}`);
    const button = element('button', '', 'workflow-step');
    button.setAttribute('aria-pressed', 'false');
    button.append(element('span', String(index + 1).padStart(2, '0'), 'step-number'));
    const content = element('span', '', 'step-content');
    content.append(element('strong', stage.title), element('span', [
      stage.targetId ? `实体 ${stage.displayId ? `#${stage.displayId}` : stage.targetId}` : '共享步骤', STATUS_NAMES[stage.status],
      Number.isFinite(stage.duration) ? `${stage.duration.toFixed(1)} 秒` : '',
    ].filter(Boolean).join(' · '), 'muted'));
    button.append(content);
    button.addEventListener('click', () => activate(button, stage));
    buttons.push(button);
    card.append(button);
    if (stage.models.length) {
      const models = element('div', '', 'workflow-models');
      for (const event of stage.models) {
        const info = modelCallInfo(event);
        const call = element('button', '', `model-call ${info.type.toLowerCase()}`);
        call.setAttribute('aria-pressed', 'false');
        call.setAttribute('title', info.purpose);
        call.setAttribute('aria-label', `${info.type} · ${info.purpose}`);
        call.append(element('strong', info.type),
          element('span', `${info.failed ? '调用失败 · ' : ''}${Number.isFinite(info.elapsed) ? `${info.elapsed.toFixed(2)} 秒` : '耗时未归档'}`, 'muted'));
        call.addEventListener('click', () => activate(call, stage, event));
        buttons.push(call);
        models.append(call);
      }
      card.append(models);
    }
    container.append(card);
  }
  if (!stages.length) container.append(element('p', '本次实验未记录执行步骤。', 'muted'));
  return stages;
}

const COLOR_NAMES = { pink: '粉色', black: '黑色', blue: '蓝色', gray: '灰色', white: '白色' };
const KIND_NAMES = { object: '物体', bin: '盒子', table: '桌子', marker: '目标区域', collection: '产物集合' };
function entityLabel(entity) {
  if (!entity) return '实体信息未归档';
  return `${entity.id ? `ID ${entity.id} ` : ''}${entity.name ?? `${COLOR_NAMES[entity.color] ?? ''}${KIND_NAMES[entity.kind] ?? entity.kind ?? '物体'}`}`;
}

export function renderObjectMemorySnapshot(container, snapshot, onEvidence) {
  const element = dom(container);
  container.replaceChildren();
  if (!snapshot) { container.append(element('p', '该实验缺少已归档的时间记忆快照。', 'muted')); return; }
  const { progress, event } = snapshot;
  const provenance = element('div', '', 'objects-provenance');
  provenance.append(element('span', '数据库历史快照', 'badge'),
    element('span', `事件 #${event.id ?? '—'}`, 'muted'));
  if (event.data?.model_event_ids?.length) provenance.append(element('span',
    `模型调用：${event.data.model_event_ids.map(id => `#${id}`).join('、')}`, 'muted'));
  if (event.data?.evidence_source === 'gripper_feedback') provenance.append(element('span',
    '夹爪反馈核验 · 本节点没有调用视觉模型', 'muted'));
  const source = element('button', '查看记忆来源', 'button');
  source.addEventListener('click', () => onEvidence(event));
  provenance.append(source);
  container.append(provenance);
  if (progress.entities) {
    renderTaskObjects(container, progress);
    return;
  }
  container.append(element('p', '查看归档中的原始时间记忆记录。', 'muted'));
  const raw = element('details');
  raw.append(element('summary', '原始记忆 JSON'), element('pre', JSON.stringify(progress, null, 2), 'candidate-detail'));
  container.append(raw);
}

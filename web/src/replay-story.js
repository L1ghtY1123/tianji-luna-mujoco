import { buildObjectTimeline } from './execution-graph.js';
import { isTaskInventory, inventoryEntities, renderTaskInventory } from './task-inventory.js';

const PURPOSES = {
  compile_task: '理解任务并生成总编排', revise_graph: '根据新事件修订总编排',
  initialize_inventory: '建立初始清单与任务程序', associate_inventory: '核对场景清单与物体身份',
  associate_completed_objects: '核对已放置物体的身份',
  verify_release_result: '确认放置与松爪', verify_unpacking: '确认拆包产物',
  verify_right_bin_drop: '确认目的地投放',
  verify_bin_drop: '确认物体入盒', semantic_judgments: '核验任务条件', task_completion: '核验最终完成条件',
};
const CHECKPOINTS = { initial: '场景观测', fine: '抓取相关观测', post_grasp: '抓取后采集',
  post_place: '放置后采集', post_unpack: '处理后采集', result: '结果采集', task_completion: '任务完成检查',
  recovery_decision: '恢复判断输入' };
const CAPTURE_SKILLS = { observe_scene: '场景识别', observe: '场景识别',
  execute_place: '放置完成后采集', verify_place: '放置结果核验', verify_drop: '投放结果核验',
  verify_unpack: '拆包结果核验', vla_unpack: '拆包后采集',
  move_to_observe: '移至观测位姿后采集', return_to_initial_pose: '返回任务起始位姿后采集' };
const STATUS_NAMES = { complete: '完成', failed: '失败', warning: '需关注', incomplete: '未记录结束', recorded: '已记录', cancelled: '已取消' };
const PHASES = {
  observe: ['观察与清单', 'observe_scene', 'observe', 'bind_inventory'],
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
  let purpose = PURPOSES[data.purpose] ?? data.purpose ?? '用途未归档';
  if (type === 'Decisions' && data.purpose === 'initialize_inventory') purpose = '核对分割提议与清单身份';
  return { type, purpose, elapsed: data.elapsed_s,
    failed: Boolean(data.error) || (Number.isFinite(data.http_status) && data.http_status >= 400) };
}

export function observationImageSections(observation, frame, cameraId) {
  const labels = observation.event.data?.object_labels ?? {};
  const time = value => Number.isFinite(value) ? `${value.toFixed(3)} 秒` : '未归档';
  const items = (labels.items ?? []).filter(item => item.camera_id === cameraId);
  const boxes = new Set(items.map(item => item.box_source));
  const sources = [];
  if (boxes.has('associated_instance_mask')) sources.push('已关联 SAM3 掩码的外接矩形');
  if (boxes.has('responses_reference_proposal')) sources.push(labels.analysis_status === 'responses_proposal'
    ? '初始 Responses 输出的原始参考框，未经 SAM3 或 Decisions 确认' : 'Responses 初始参考框（缺少当前掩码）');
  if (boxes.has(undefined)) sources.push('旧归档未记录逐框来源');
  const legacy = labels.overlay_scope !== 'capture_identity_geometry';
  return [
    `RGB 来源：相机 ${cameraId} 的归档采集，拍摄于 ${time(frame.captureTime)}。标注记录：事件 #${observation.event.id}，${time(observation.time)}。`,
    `框来源：${sources.join('；') || '本版本没有已确认身份的框；不表示没有检测候选'}。`,
    `身份来源：${labels.analysis_status === 'responses_proposal' ? 'Responses 清单提议与编译后的永久 ID，尚未由 Decisions 确认，不代表实测几何。' : legacy ? '旧版叠加可能含后续任务状态，不代表拍摄时的物体状态。' : '逐框记录 Decisions 的身份判断或已验证的测量对应，程序据此绘制 ID；不含后续夹持或完成状态。'}`,
    '去向：此叠加图供人检查，不是模型输入。程序使用底层 ID、掩码与实测几何。下面的“实际模型输入”直接展示已发送的参考框和候选框，点击可查看完整问题与回答。',
  ];
}

export function observationImageSource(observation, frame, cameraId) {
  return observationImageSections(observation, frame, cameraId).join('\n\n');
}

export function groupObservationFrames(observations, events = []) {
  const captures = new Map(events.filter(event => event.kind === 'observation_recorded')
    .map(event => [event.data?.observation_id, event]));
  const frames = new Map();
  for (const observation of observations) {
    const capture = captures.get(observation.observationId);
    if (!frames.has(observation.observationId)) frames.set(observation.observationId, {
      id: observation.observationId, captureTime: capture?.t_s ?? null,
      title: CHECKPOINTS[capture?.data?.checkpoint ?? observation.checkpoint] ?? '观测（用途未归档）', revisions: [],
    });
    const labels = observation.event.data?.object_labels ?? {};
    const statuses = new Set((labels.items ?? []).map(item => item.status));
    const held = statuses.has('held');
    const selected = statuses.has('selected');
    const title = labels.analysis_status === 'responses_proposal' ? 'Responses 初始参考框（未核验提议）'
      : held ? '夹爪反馈更新：夹持中' : selected ? '选择当前抓取目标'
      : labels.analysis_status === 'unprocessed' ? '原始 RGB（无身份叠加）'
        : labels.analysis_status !== 'analyzed' ? '图像记录（标注状态未归档）'
          : !labels.items?.length ? '识别结果（无实体标注）'
            : '身份与几何结果（程序叠加）';
    frames.get(observation.observationId).revisions.push({ ...observation,
      revisionTitle: title, reusesCapture: held || selected });
  }
  return [...frames.values()].map(frame => {
    const capture = captures.get(frame.id);
    const invocation = capture?.invocation_id;
    const completion = capture?.data?.checkpoint === 'task_completion';
    const source = invocation && events.find(event => event.kind === 'skill_start' && event.invocation_id === invocation);
    const skill = source?.data?.skill ?? source?.title;
    const models = events.filter(event => event.kind === 'visual_decisions_call' && event.data?.observation_id === frame.id);
    const compilerInput = models.some(event => modelCallInfo(event).type === 'Responses'
      && ['compile_task', 'initialize_inventory'].includes(event.data?.purpose));
    let title = completion ? '任务完成检查'
      : CAPTURE_SKILLS[skill] ?? (skill ? `${skill} 采集` : compilerInput ? '任务编排参考画面' : frame.title);
    const targetId = completion ? null : capture?.target_id ?? source?.data?.target_id;
    if (targetId) {
      const label = [...frame.revisions, ...observations.filter(item => item.event.id < capture.id).reverse()]
        .flatMap(item => item.event.data?.object_labels?.items ?? [])
        .find(item => item.id === targetId);
      const name = label?.name && !label.name.includes('unknown') ? label.name : '物体';
      title += ` · ${label ? `${label.display_id != null ? `#${label.display_id} ` : ''}${name}` : targetId}`;
    }
    const hasLabels = frame.revisions.some(item => item.event.data?.object_labels?.analysis_status === 'analyzed'
      && item.event.data.object_labels.items?.length);
    const modelUses = [...new Set(models.map(event => {
      const info = modelCallInfo(event);
      return `${info.type}：${info.purpose}${info.failed ? '（调用失败）' : ''}`;
    }))];
    const reuseUses = [...new Set(events.filter(event => event.kind === 'observation_reused'
      && event.data?.observation_id === frame.id).map(event => PURPOSES[event.data.purpose] ?? event.data.purpose))];
    const analysisSummary = `${hasLabels ? '已保存本帧物体识别与身份标注。' : '未保存本帧物体识别与身份标注。'}${modelUses.length
      ? `此画面用于 ${modelUses.join('；')}。模型使用画面不代表生成了分割或身份叠加图。`
      : '未记录与此帧关联的模型调用，不能仅凭原始图判断是否做过分析。'}${reuseUses.length
      ? `复用此采集：${reuseUses.join('；')}。保留原始采集时间；复用图像不代表复用判断结果。` : ''}`;
    return { ...frame, title, analysisSummary,
      source: capture ? `${completion ? '任务完成检查 · ' : skill ? `${skill} · ${invocation} · ` : '独立采集 · '}事件 #${capture.id ?? '未归档'} · 观测 ID ${frame.id}`
        : `采集记录缺失，拍摄时间与调用来源无法核实 · 观测 ID ${frame.id}`,
      preferredId: (frame.revisions.find(item => item.status === 'analyzed' && !item.reusesCapture)
        ?? frame.revisions[0]).id };
  });
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
      const entities = new Map(inventoryEntities(progress).map(item => [item.id, item]));
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

// Only explicit task snapshots are displayed; events never synthesize a ledger.
export function inventorySnapshots(events = []) {
  const snapshots = [];
  for (const event of events) {
    const progress = event.data?.task_progress;
    if (!progress || typeof progress !== 'object') continue;
    snapshots.push({ event, progress: structuredClone(progress), provenance: 'snapshot',
      title: event.kind === 'run_end' ? '任务结束清单' : event.data?.checkpoint ?? '清单快照' });
  }
  return snapshots;
}

const QUESTION_NAMES = { place_result: '放置结果', workpiece_still_held: '工件仍被夹持',
  unpack_result: '处理产物核验', current_item_added: '本次投放结果' };

export function modelResult(archive) {
  const answers = Array.isArray(archive.response?.answers) ? archive.response.answers : [];
  const questions = archive.request?.questions ?? [];
  const result = { answers: answers.map(answer => {
    const question = questions.find(item => item.name === answer.name);
    const choice = question?.choices?.find(item => item.value === answer.choice);
    return { ...answer, title: QUESTION_NAMES[answer.name] ?? (/^identity_/.test(answer.name)
      ? `分割提议 ${answer.name.slice(9)} · 身份判断` : answer.name),
    instructions: Array.isArray(question?.instructions)
      ? question.instructions.filter(part => part.type === 'input_text').map(part => part.text).join('\n')
      : question?.instructions ?? '完整问题未归档',
    choiceLabel: ({ reject: '排除此提议', unknown: '无法确认' })[answer.choice] ?? choice?.description ?? answer.choice };
  }), inventory: null };
  if (archive.endpoint?.endsWith('/responses')) {
    for (const message of archive.response?.output ?? []) {
      if (message.type !== 'message') continue;
      for (const part of message.content ?? []) {
        if (part.type !== 'output_text') continue;
        try { result.inventory = JSON.parse(part.text); } catch { /* Raw response remains available. */ }
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
  return `${entity.display_id ? `#${entity.display_id} ` : ''}${entity.name ?? `${COLOR_NAMES[entity.color] ?? ''}${KIND_NAMES[entity.kind] ?? entity.kind ?? '物体'}`}`;
}

export function renderInventorySnapshot(container, snapshot, onEvidence) {
  const element = dom(container);
  container.replaceChildren();
  if (!snapshot) { container.append(element('p', '这条记录没有归档清单，不能从当前场景补造。', 'muted')); return; }
  const { progress, event } = snapshot;
  const provenance = element('div', '', 'inventory-provenance');
  provenance.append(element('span', '已保存的清单快照', 'badge'),
    element('span', `事件 #${event.id ?? '—'}`, 'muted'));
  if (event.data?.model_event_ids?.length) provenance.append(element('span',
    `模型调用：${event.data.model_event_ids.map(id => `#${id}`).join('、')}`, 'muted'));
  if (event.data?.evidence_source === 'gripper_feedback') provenance.append(element('span',
    '夹爪反馈核验 · 本节点没有调用视觉模型', 'muted'));
  const source = element('button', '查看清单来源', 'button');
  source.addEventListener('click', () => onEvidence(event));
  provenance.append(source);
  container.append(provenance);
  if (isTaskInventory(progress)) {
    renderTaskInventory(container, progress, event.data?.task_inventory);
    return;
  }
  container.append(element('p', '历史清单仅保留原始记录，不重建执行流程。', 'muted'));
  const raw = element('details');
  raw.append(element('summary', '原始清单 JSON'), element('pre', JSON.stringify(progress, null, 2), 'candidate-detail'));
  container.append(raw);
}

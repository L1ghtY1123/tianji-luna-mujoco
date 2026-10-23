import { buildObjectTimeline } from './execution-graph.js';

const PURPOSES = {
  initialize_inventory: '建立初始清单与任务程序', associate_inventory: '抓前确认物体身份',
  verify_release_result: '确认桌面放置与松爪', verify_unpacking: '确认拆包产物',
  verify_right_bin_drop: '确认本次黑件进入右盒',
};
const CHECKPOINTS = { initial: '初始观测', fine: '返回观测位姿 · 抓前观察',
  post_place: '桌面放置后观察', post_unpack: 'VLA 拆包后观察', result: '投放后观察' };
const STATUS_NAMES = { complete: '完成', failed: '失败', warning: '需关注', incomplete: '未记录结束', recorded: '已记录', cancelled: '已取消' };
const PHASES = {
  observe: ['观察与清单', 'observe_scene', 'observe', 'bind_inventory'],
  grasp: ['抓取物体', 'identify_target', 'identify_destination', 'choose_arm', 'build_scene', 'generate_grasps', 'evaluate_arms', 'plan_grasp', 'plan_grasp_and_place', 'execute_approach', 'execute_grasp', 'close_gripper', 'execute_lift', 'verify_grasp'],
  place: ['桌面放置', 'plan_place', 'execute_place', 'verify_place', 'observe_result'],
  unpack: ['VLA 拆包', 'prepare_unpack', 'mock_unpack', 'verify_unpack'],
  return: ['返回观测位姿', 'move_to_observe'],
  drop: ['运输与右盒投放', 'plan_transport', 'execute_transport', 'drop', 'verify_drop'],
  recovery: ['恢复处理', 'recover_task', 'inspect_visual_result'],
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

export function groupObservationFrames(observations, events = []) {
  const captures = new Map(events.filter(event => event.kind === 'observation_recorded')
    .map(event => [event.data?.observation_id, event]));
  const frames = new Map();
  for (const observation of observations) {
    const capture = captures.get(observation.observationId);
    if (!frames.has(observation.observationId)) frames.set(observation.observationId, {
      id: observation.observationId, captureTime: capture?.t_s ?? null,
      title: CHECKPOINTS[capture?.data?.checkpoint ?? observation.checkpoint] ?? '观测（时机未归档）', revisions: [],
    });
    const labels = observation.event.data?.object_labels ?? {};
    const statuses = new Set((labels.items ?? []).map(item => item.status));
    const held = statuses.has('held');
    const selected = statuses.has('selected');
    const title = held ? '夹爪反馈更新：夹持中' : selected ? '选择当前抓取目标'
      : labels.analysis_status === 'unprocessed' ? '原始观测 · 尚未叠加识别标注'
        : statuses.has('pending') ? '识别产物 · 等待抓取' : '场景识别与身份标注';
    frames.get(observation.observationId).revisions.push({ ...observation,
      revisionTitle: title, reusesCapture: held || selected });
  }
  return [...frames.values()].map(frame => ({ ...frame,
    preferredId: (frame.revisions.find(item => item.status === 'analyzed' && !item.reusesCapture)
      ?? frame.revisions[0]).id }));
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
    const phase = Object.entries(PHASES).find(([, names]) => names.slice(1).includes(skill))?.[0];
    const group = ownership.get(node.id);
    let stage = stages.at(-1);
    // Unscoped bookkeeping stays with its chronological stage, never with a guessed target.
    const processId = group?.processId ?? null;
    if (!stage || (phase && (phase !== stage.phase || (processId && stage.processId && processId !== stage.processId)))) {
      stage = { id: `story:${node.id}`, phase: phase ?? 'system', title: PHASES[phase ?? 'system'][0],
        processId, displayId: group?.displayId ?? null, nodes: [], events: [], models: [] };
      stages.push(stage);
    }
    if (!stage.processId && processId) { stage.processId = processId; stage.displayId = group.displayId; }
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
    if (stage.phase === 'unpack' && stage.nodes.some(node => node.events.some(event => event.title === 'mock_unpack'))) stage.title = 'VLA 拆包（仿真）';
    if (stage.phase === 'grasp') {
      const catalog = graph.events.findLast(event => event.data?.task_progress?.all_entities)?.data.task_progress.all_entities ?? {};
      const entities = new Map((Array.isArray(catalog) ? catalog : Object.values(catalog)).map(item => [item.id, item]));
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

// Only stored snapshots and explicitly recorded entity transitions are shown.
export function inventorySnapshots(events = []) {
  const snapshots = [];
  const hasCheckpoints = events.some(event => event.kind === 'inventory_snapshot');
  let progress = null;
  for (const event of events) {
    if (hasCheckpoints && !['inventory_snapshot', 'run_end'].includes(event.kind)) continue;
    const result = event.data?.task_progress ?? event.data?.result;
    if (result && (result.all_entities || result.process_records || result.inventory_initialized) && typeof result === 'object') {
      progress = structuredClone(result);
      const checkpoints = { bind_inventory: '初始清单', observe_scene: '抓取前观测 · 清单核对',
        verify_place: '桌面放置核验', verify_unpack: '拆包核验 · 产物登记',
        verify_grasp: '抓取反馈核验', verify_drop: '右盒投放核验' };
      snapshots.push({ event, progress, provenance: 'snapshot', title: event.data?.outcome === 'not_applied'
        ? `${checkpoints[event.data?.checkpoint] ?? '核验'} · 未通过，保留原清单` : checkpoints[event.data?.checkpoint]
        ?? (event.kind === 'run_end' ? '任务结束清单'
        : snapshots.length ? '清单快照' : '初始清单') });
    } else if (event.kind === 'product_created' && event.data?.product_entity && progress) {
      const data = event.data;
      const previous = progress.process_records?.[data.process_id];
      if (!previous || previous.input_entity_id !== data.input_entity_id) continue;
      progress = structuredClone(progress);
      if (Array.isArray(progress.all_entities)) progress.all_entities = [...progress.all_entities.filter(item => item.id !== data.output_entity_id), structuredClone(data.product_entity)];
      else progress.all_entities = { ...progress.all_entities, [data.output_entity_id]: structuredClone(data.product_entity) };
      progress.process_records[data.process_id] = { ...previous, output_entity_id: data.output_entity_id,
        phase: 'unpacked', status: 'processing' };
      snapshots.push({ event, progress, provenance: 'event', title: '拆包确认 · 新产物登记' });
    }
  }
  return snapshots;
}

const QUESTION_NAMES = { place_result: '粉件已放在桌面', workpiece_still_held: '工件仍被夹持',
  unpack_result: '桌面上可见独立黑件', current_item_added: '本次黑件已进入右盒' };

export function modelResult(archive) {
  const answers = Array.isArray(archive.response?.answers) ? archive.response.answers : [];
  const questions = archive.request?.questions ?? [];
  const result = { answers: answers.map(answer => {
    const question = questions.find(item => item.name === answer.name);
    const choice = question?.choices?.find(item => item.value === answer.choice);
    return { ...answer, title: QUESTION_NAMES[answer.name] ?? (/^identity_/.test(answer.name)
      ? `分割提议 ${answer.name.slice(9)} · 身份判断` : answer.name),
    instructions: question?.instructions ?? '完整问题未归档',
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
      stage.processId ? `工序 #${stage.displayId ?? '—'}` : '共享步骤', STATUS_NAMES[stage.status],
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
  const catalog = progress.all_entities ?? progress.objects ?? [];
  const entities = Array.isArray(catalog) ? catalog : Object.values(catalog);
  const byId = new Map(entities.map(entity => [entity.id, entity]));
  const records = Object.values(progress.process_records ?? {});
  const provenance = element('div', '', 'inventory-provenance');
  provenance.append(element('span', snapshot.provenance === 'snapshot' ? '已保存的完整清单快照' : '由已确认拆包事件更新 · 非完整快照', 'badge'),
    element('span', `事件 #${event.id ?? '—'}`, 'muted'));
  if (event.data?.model_event_ids?.length) provenance.append(element('span',
    `模型调用：${event.data.model_event_ids.map(id => `#${id}`).join('、')}`, 'muted'));
  if (event.data?.evidence_source === 'gripper_feedback') provenance.append(element('span',
    '夹爪反馈核验 · 本节点没有调用视觉模型', 'muted'));
  const source = element('button', '查看清单来源', 'button');
  source.addEventListener('click', () => onEvidence(event));
  provenance.append(source);
  container.append(provenance);
  const board = element('div', '', 'inventory-board');
  const columns = [
    ['待拆包', record => record.status !== 'completed' && !record.output_entity_id],
    ['已拆包 · 待投放', record => record.status !== 'completed' && Boolean(record.output_entity_id)],
    ['已完成', record => record.status === 'completed'],
  ];
  for (const [title, matches] of columns) {
    const selected = records.filter(matches);
    const column = element('section', '', 'inventory-column');
    column.append(element('h3', `${title} · ${selected.length}`));
    for (const record of selected) {
      const card = element('article', '', 'inventory-process');
      card.append(element('strong', `工序 #${record.display_id ?? '—'}`), element('p', entityLabel(byId.get(record.input_entity_id))));
      if (record.output_entity_id) card.append(element('p', `↓ ${entityLabel(byId.get(record.output_entity_id))}`, 'product-link'));
      card.append(element('p', `到桌面：${({ A: '左臂 A', B: '右臂 B' })[record.to_table_arm] ?? '未归档'} · 到右盒：${({ A: '左臂 A', B: '右臂 B' })[record.to_right_bin_arm] ?? '未归档'}`, 'muted'));
      const ids = element('details');
      ids.append(element('summary', '永久 ID 与工序关联'), element('p', `输入：${record.input_entity_id}`),
        element('p', `产物：${record.output_entity_id ?? '尚未登记'}`), element('p', `工序：${record.id}`));
      card.append(ids);
      column.append(card);
    }
    if (!selected.length) column.append(element('p', records.length ? '无' : '工序清单未归档', 'muted'));
    board.append(column);
  }
  container.append(board);
  const order = progress.process_order ?? [];
  if (order.length) container.append(element('p', `当时记录的工序顺序：${order.map(id => `#${progress.process_records?.[id]?.display_id ?? '—'}`).join(' → ')}`, 'inventory-order'));
  container.append(element('h3', `该时刻登记的实体 · ${entities.length} 项`));
  const tableWrap = element('div', '', 'inventory-table-wrap');
  const table = element('table', '', 'inventory-table');
  const head = element('tr');
  for (const title of ['实体', '记录性质', '永久 ID']) head.append(element('th', title));
  table.append(head);
  for (const entity of entities) {
    const row = element('tr');
    const unpackedInput = records.some(record => record.input_entity_id === entity.id && record.output_entity_id);
    const nature = entity.kind === 'marker' ? '数值区域 · 非实物' : entity.kind === 'collection' ? '盒内集合 · 不另计为一个工件'
      : unpackedInput ? '已拆包输入 · 历史身份'
        : entity.geometry_status === 'historical' ? '历史身份 · 不代表当前精确位置' : '场景实体';
    row.append(element('td', entityLabel(entity)), element('td', nature), element('td', entity.id, 'entity-id'));
    table.append(row);
  }
  tableWrap.append(table);
  container.append(tableWrap);
}

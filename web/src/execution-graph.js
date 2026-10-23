import { createEventTitleResolver } from './model.js';

const SKILL_EVENTS = new Set(['skill_start', 'skill_complete', 'skill_error']);
const CATEGORY_NAMES = { jev: '历史模型记录', skill: '技能程序', algorithm: '算法调用', system: '系统记录', subgraph: '阶段子图' };
const STATUS_NAMES = { failed: '失败', complete: '完成', warning: '需关注', recorded: '已记录', incomplete: '未记录结束', cancelled: '已取消' };
const COLORS = { jev: '#7ec3ff', skill: '#8ed2a2', algorithm: '#7ec3ff', system: '#b7c6d5', subgraph: '#c3a6f0' };
const STATUS_COLORS = { failed: '#f07e73', complete: '#80d59b', warning: '#edc778', recorded: '#9aaebe', incomplete: '#edc778', cancelled: '#b7c6d5' };
const PHASES = {
  observe: ['观察', 'observe', 'observe_scene', 'observe_geometry', 'observe_result', 'move_to_observe'],
  perceive: ['感知 / 建图', 'inspect_scene', 'check_scene_changes', 'build_scene', 'update_scene'],
  source: ['目标物体', 'identify_target', 'assess_task'],
  container: ['目的容器', 'identify_destination'],
  grasp: ['抓取', 'generate_grasps', 'evaluate_arms', 'choose_arm', 'plan_grasp', 'plan_grasp_and_place', 'execute_approach', 'execute_grasp', 'close_gripper', 'execute_lift', 'verify_grasp'],
  transport: ['运输 / 放置', 'plan_place', 'execute_place', 'verify_place'],
  unpack: ['VLA 拆包', 'prepare_unpack', 'mock_unpack', 'verify_unpack'],
  drop: ['黑件运输 / 右盒投放', 'plan_transport', 'execute_transport', 'drop', 'verify_drop'],
  recovery: ['恢复 / 重试', 'recover_task', 'inspect_visual_result'],
  system: ['任务 / 决策'],
};

function category(event) {
  if (event.kind === 'jev') return 'jev';
  if (SKILL_EVENTS.has(event.kind) || event.kind?.startsWith('execution_stage_')) return 'skill';
  if (event.kind?.startsWith('algorithm_') || event.kind?.startsWith('planning_batch_') || event.kind?.startsWith('ik_batch_')) return 'algorithm';
  return 'system';
}

function summarize(node) {
  const times = node.events.map(event => event.t_s).filter(Number.isFinite);
  node.startTime = times.length ? Math.min(...times) : null;
  node.endTime = times.length ? Math.max(...times) : null;
  const terminal = node.events.filter(event => ['skill_complete', 'skill_error', 'algorithm_complete', 'algorithm_error'].includes(event.kind)).at(-1);
  const skillTerminal = node.events.filter(event => ['skill_complete', 'skill_error'].includes(event.kind)).at(-1);
  const end = node.invocationId ? skillTerminal : terminal;
  const runEnd = node.events.findLast(event => event.kind === 'run_end');
  if (runEnd) {
    const status = runEnd.data?.status;
    node.status = ['success', 'succeeded', 'complete', 'completed'].includes(status) ? 'complete'
      : ['failed', 'error'].includes(status) ? 'failed'
        : ['cancelled', 'canceled'].includes(status) ? 'cancelled' : 'warning';
  } else if (end?.data?.result?.status === 'unknown') node.status = 'warning';
  else if (end) node.status = end.kind.endsWith('_complete') && end.data?.result?.success !== false ? 'complete' : 'failed';
  else if (node.events.some(event => ['error', 'execution_stage_error', 'algorithm_error', 'task_failure'].includes(event.kind) || event.data?.error)) node.status = 'failed';
  else if (node.events.some(event => event.kind === 'planning_attempt_failed')) node.status = 'warning';
  else node.status = node.invocationId || node.callId ? 'incomplete' : 'recorded';
  const start = node.events.find(event => ['skill_start', 'algorithm_start'].includes(event.kind));
  node.duration = end && Number.isFinite(start?.t_s) && Number.isFinite(end.t_s) ? Math.max(0, end.t_s - start.t_s) : null;
  return node;
}

// Invocation IDs own evidence. Phase grouping below changes presentation only.
export function buildExecutionGraph(events = []) {
  const titleForEvent = createEventTitleResolver(events);
  const seen = new Set();
  const records = [];
  const skills = new Map();
  const parents = new Map();
  const warnings = new Set();
  for (const [index, event] of events.entries()) {
    const key = event.id === undefined || event.id === null ? `missing-${index}` : String(event.id);
    if (seen.has(key)) continue;
    seen.add(key);
    records.push({ event, key, index });
    if (event.invocation_id && event.parent_invocation_id) parents.set(event.invocation_id, event.parent_invocation_id);
    if (SKILL_EVENTS.has(event.kind) && event.invocation_id) {
      const existing = skills.get(event.invocation_id);
      if (!existing || event.kind === 'skill_start') skills.set(event.invocation_id, event);
    }
  }
  function owner(event) {
    if (event.kind === 'jev') return null;
    let id = event.invocation_id ?? event.parent_invocation_id;
    const visited = new Set();
    while (id && !visited.has(id)) {
      if (skills.has(id)) return id;
      visited.add(id);
      id = parents.get(id) ?? (id === event.invocation_id ? event.parent_invocation_id : null);
    }
    return null;
  }
  const grouped = new Map();
  for (const { event, key, index } of records) {
    const invocationId = owner(event);
    const callId = event.kind?.startsWith('algorithm_') ? event.data?.call_id : null;
    const id = invocationId ? `skill:${invocationId}` : callId ? `algorithm:${callId}` : `event:${key}`;
    if (!grouped.has(id)) {
      const source = invocationId ? skills.get(invocationId) : event;
      grouped.set(id, {
        id, invocationId, callId: invocationId ? null : callId,
        parentInvocationId: source.parent_invocation_id ?? null,
        category: invocationId ? 'skill' : category(event),
        title: titleForEvent(source, source.title ?? source.data?.skill ?? source.kind ?? '事件'),
        events: [], firstIndex: index,
      });
    }
    grouped.get(id).events.push(event);
    if (!invocationId && category(event) === 'skill') warnings.add('部分 skill 事件缺少可关联的调用 ID，已按独立事件显示。');
    if (!Number.isFinite(event.t_s)) warnings.add('部分事件缺少任务时间，无法准确定位其视频位置。');
  }
  const nodes = [...grouped.values()].map(summarize);
  const indices = new Map(records.map(record => [record.event, record.index]));
  for (const node of nodes) {
    const calls = new Map();
    for (const event of node.events) {
      const callId = event.kind?.startsWith('algorithm_') && event.data?.call_id;
      if (!callId || node.callId) continue;
      if (!calls.has(callId)) calls.set(callId, {
        id: `${node.id}/algorithm:${callId}`, callId, invocationId: null, parentInvocationId: node.invocationId,
        category: 'algorithm', title: titleForEvent(event, event.data?.function ?? event.title ?? '算法调用'),
        firstIndex: indices.get(event), events: [],
      });
      calls.get(callId).events.push(event);
    }
    node.calls = [...calls.values()].map(summarize);
  }
  return { nodes, events: records.map(record => record.event),
    edges: nodes.slice(1).map((node, index) => ({ from: nodes[index].id, to: node.id })), warnings: [...warnings] };
}

const OBJECT_PHASES = {
  observe: ['观测 / 清单', 'observe_scene', 'observe', 'bind_inventory'],
  target: ['目标 / 放置区', 'identify_target', 'identify_destination'],
  arm: ['执行臂选择', 'choose_arm'],
  map: ['场景建图', 'build_scene'],
  return: ['返回观测位姿', 'move_to_observe'],
  generate: ['GraspGen', 'generate_grasps'],
  ik: ['Batched IK', 'evaluate_arms'],
  plan: ['IK / 抓取规划', 'plan_grasp', 'plan_grasp_and_place'],
  grasp: ['抓取 / 持物', 'execute_approach', 'execute_grasp', 'close_gripper', 'execute_lift', 'verify_grasp'],
  place: ['放置 / 松爪', 'plan_place', 'execute_place'],
  verify: ['放置核验', 'verify_place', 'observe_result'],
  unpack: ['VLA 拆包 / 核验', 'prepare_unpack', 'mock_unpack', 'verify_unpack'],
  transport: ['黑件持物运输', 'plan_transport', 'execute_transport'],
  drop: ['右盒投放 / 完成核验', 'drop', 'verify_drop'],
  system: ['程序 / 记录'],
};

export function buildObjectTimeline(graph) {
  const groups = new Map();
  const processRecords = { ...(graph.events?.findLast(event => event.data?.task_progress?.process_records)
    ?.data.task_progress.process_records ?? {}) };
  for (const event of graph.events ?? []) {
    if (event.kind !== 'product_created' || !event.data?.process_id) continue;
    const data = event.data;
    processRecords[data.process_id] = { ...data, ...processRecords[data.process_id] };
  }
  const entityProcesses = new Map();
  for (const [id, record] of Object.entries(processRecords)) {
    for (const entity of [record.input_entity_id, record.output_entity_id].filter(Boolean)) entityProcesses.set(entity, id);
  }
  const inventory = graph.events?.findLast(event => Array.isArray(event.data?.task_progress?.objects))
    ?.data.task_progress.objects ?? [];
  const displayIds = new Map(inventory.filter(object => Number.isInteger(object.display_id) && object.display_id > 0)
    .map(object => [object.id, object.display_id]));
  for (const node of graph.nodes) {
    const targets = [...new Set(node.events.map(event => event.target_id ?? event.data?.target_id
      ?? event.data?.result?.target_id).filter(value => typeof value === 'string' && value))];
    const targetId = targets.length === 1 ? targets[0] : null;
    const processIds = [...new Set([...node.events.map(event => event.process_id ?? event.data?.process_id),
      ...targets.map(id => entityProcesses.get(id))].filter(Boolean))];
    const processId = processIds.length === 1 ? processIds[0] : null;
    const process = processRecords[processId] ?? {};
    const key = processId ?? targetId ?? '__shared__';
    if (!groups.has(key)) groups.set(key, { targetId, processId,
      inputEntityId: process.input_entity_id ?? null, outputEntityId: process.output_entity_id ?? null,
      displayId: process.display_id ?? displayIds.get(targetId) ?? null, stages: [] });
    const source = node.events.find(event => SKILL_EVENTS.has(event.kind)) ?? node.events[0];
    const skill = source.data?.skill ?? source.title;
    let phase = Object.entries(OBJECT_PHASES).find(([, names]) => names.slice(1).includes(skill))?.[0];
    if (!phase && node.events.some(event => event.kind === 'product_created')) phase = 'unpack';
    if (!phase && node.events.some(event => event.kind === 'visual_decisions_call' || event.kind?.startsWith('observation_'))) phase = 'observe';
    phase ??= 'system';
    const group = groups.get(key);
    let stage = group.stages.at(-1);
    if (!stage || stage.phase !== phase) {
      stage = { id: `timeline:${node.id}`, phase, title: OBJECT_PHASES[phase][0], nodes: [], events: [] };
      group.stages.push(stage);
    }
    stage.nodes.push(node);
    stage.events.push(...node.events);
  }
  for (const group of groups.values()) {
    for (const stage of group.stages) {
      stage.startTime = stage.nodes.find(node => Number.isFinite(node.startTime))?.startTime ?? null;
      stage.status = ['failed', 'warning', 'incomplete'].find(status => stage.nodes.some(node => node.status === status))
        ?? (stage.nodes.some(node => node.status === 'complete') ? 'complete' : 'recorded');
      const paths = new Set(stage.events.flatMap(event => [
        event.artifact, event.data?.artifact, event.data?.observation_artifact,
      ]).filter(Boolean));
      stage.artifactCount = paths.size;
    }
  }
  return [...groups.values()];
}

export function renderObjectTimeline(container, graph, onSelect) {
  const document = container.ownerDocument;
  const element = (tag, text, className = '') => {
    const item = document.createElement(tag);
    item.textContent = text;
    item.className = className;
    return item;
  };
  container.replaceChildren();
  for (const group of buildObjectTimeline(graph)) {
    const section = element('section', '', 'object-route');
    section.append(element('h3', group.processId ? `工序 #${group.displayId ?? '—'}` : group.targetId
      ? group.displayId ? `物体 #${group.displayId}` : `物体 ${group.targetId}`
      : '共享观测 / 未记录目标 ID'));
    if (group.processId) section.append(element('p', `输入 ID：${group.inputEntityId ?? '未记录'} → 产物 ID：${group.outputEntityId ?? '等待拆包确认'}`, 'muted'));
    else if (group.displayId) section.append(element('p', `永久 ID：${group.targetId}`, 'muted'));
    const route = element('div', '', 'object-route-stages');
    for (const [stageIndex, stage] of group.stages.entries()) {
      const card = element('div', '', `object-stage status-${stage.status}`);
      const button = element('button', '', 'object-stage-button');
      button.type = 'button';
      button.append(element('span', `步骤 ${stageIndex + 1}`, 'muted'), element('strong', stage.title), element('span', STATUS_NAMES[stage.status]),
        element('span', `${stage.artifactCount} 个证据文件 · ${stage.events.length} 条记录`));
      button.addEventListener('click', () => onSelect(stage));
      card.append(button);
      if (stage.nodes.length > 1) {
        const details = element('details', '', 'object-stage-calls');
        details.append(element('summary', `展开 ${stage.nodes.length} 次调用`));
        for (const node of stage.nodes) {
          const child = element('button', `${node.title} · ${STATUS_NAMES[node.status]}`, 'stage-button');
          child.addEventListener('click', () => onSelect(node));
          details.append(child);
        }
        card.append(details);
      }
      route.append(card);
    }
    section.append(route);
    container.append(section);
  }
}

function phaseOf(node, skills) {
  const event = node.events[0];
  const skill = node.invocationId ? event.data?.skill ?? node.title : event.data?.output?.skill
    ?? skills.get(event.invocation_id ?? event.parent_invocation_id)?.title;
  return Object.entries(PHASES).find(([, names]) => names.slice(1).includes(skill))?.[0] ?? 'system';
}

function phaseGroups(graph) {
  const skills = new Map(graph.nodes.filter(node => node.invocationId).map(node => [node.invocationId, node]));
  const order = new Map((graph.events ?? graph.nodes.flatMap(node => node.events)).map((event, index) => [event, index]));
  const groups = [];
  for (const node of graph.nodes) {
    const phase = phaseOf(node, skills);
    let group = groups.at(-1);
    if (!group || group.phase !== phase) {
      group = { id: `group:${node.id}`, phase, title: PHASES[phase][0], nodes: [] };
      groups.push(group);
    }
    group.nodes.push(node);
  }
  for (const group of groups) {
    const events = [...new Set(group.nodes.flatMap(node => node.events))];
    events.sort((a, b) => order.get(a) - order.get(b));
    group.summary = summarize({ id: group.id, title: group.title, category: 'subgraph', events, invocationId: null });
    group.callCount = group.nodes.filter(node => node.invocationId).length;
    group.failures = group.nodes.filter(node => node.status === 'failed').length;
    group.summary.title += ` · ${group.callCount || group.nodes.length} 次${group.callCount ? '调用' : '记录'}`;
  }
  return groups;
}

const CARD_WIDTH = 284;
const CARD_HEIGHT = 158;
const COLUMN_GAP = 36;
const ROW_GAP = 44;
const COLUMN_STEP = CARD_WIDTH + COLUMN_GAP + 48;
const GROUP_COLUMNS = 3;
const BAND_COLUMNS = 6;

export function layoutExecutionGraph(graph, { expanded = new Set() } = {}) {
  const groups = phaseGroups(graph);
  const nodes = [];
  const bands = [];
  let y = 154;
  let column = 0;
  let rowHeight = 0;
  for (const group of groups) {
    const open = expanded.has(group.id);
    const primary = group.nodes.filter(node => node.invocationId || node.category === 'system' || node.callId);
    let shown = group.callCount ? primary : group.nodes;
    if (open) shown = group.nodes.flatMap(node => [node, ...(node.calls ?? [])]).sort((a, b) => a.firstIndex - b.firstIndex);
    else if (shown.length > 6) shown = [group.summary];
    const columns = Math.min(GROUP_COLUMNS, shown.length);
    if (column + columns > BAND_COLUMNS) {
      bands.push({ y, height: rowHeight });
      y += rowHeight + 58;
      column = 0;
      rowHeight = 0;
    }
    const rows = Math.ceil(shown.length / GROUP_COLUMNS);
    group.x = 24 + column * COLUMN_STEP;
    group.y = y;
    group.width = columns * COLUMN_STEP - COLUMN_GAP;
    group.height = 74 + rows * CARD_HEIGHT + (rows - 1) * ROW_GAP + 24;
    group.expanded = open;
    for (const [index, node] of shown.entries()) {
      nodes.push({ node, groupId: group.id, band: bands.length, x: group.x + 24 + (index % GROUP_COLUMNS) * COLUMN_STEP,
        y: y + 74 + Math.floor(index / GROUP_COLUMNS) * (CARD_HEIGHT + ROW_GAP), width: CARD_WIDTH, height: CARD_HEIGHT });
    }
    column += columns;
    rowHeight = Math.max(rowHeight, group.height);
  }
  bands.push({ y, height: rowHeight });
  const edges = nodes.slice(1).map((item, index) => {
    const previous = nodes[index];
    const start = { x: previous.x + previous.width, y: previous.y + previous.height / 2 };
    const end = { x: item.x - 7, y: item.y + item.height / 2 };
    let points = [start, end];
    if (previous.y !== item.y) {
      const band = bands[previous.band];
      const gutterY = previous.band === item.band ? previous.y + previous.height + ROW_GAP / 2 : band.y + band.height + 28;
      points = [start, { x: start.x + 18, y: start.y }, { x: start.x + 18, y: gutterY },
        { x: item.x - 18, y: gutterY }, { x: item.x - 18, y: end.y }, end];
    }
    return { from: previous, to: item, status: previous.node.status, points };
  });
  return { groups, nodes, edges, width: Math.max(1000, ...groups.map(group => group.x + group.width + 24)), height: Math.max(y + rowHeight + 32, 240) };
}

function timeLabel(time) { return Number.isFinite(time) ? `${time.toFixed(3)} s` : '时间未提供'; }
function xml(value) { return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&apos;'); }
function wrapText(value, width = 24) {
  const rows = [];
  let row = '';
  let used = 0;
  for (const character of String(value)) {
    const cells = character.charCodeAt(0) > 255 ? 2 : 1;
    if (used + cells > width || character === '\n') { rows.push(row); row = ''; used = 0; }
    if (character !== '\n') { row += character; used += cells; }
  }
  rows.push(row);
  return rows;
}
function shape(tag, attrs = {}, children = [], text) { return { tag, attrs, children, text }; }
function label(x, y, text, attrs = {}) { return shape('text', { x, y, fill: '#e5edf4', ...attrs }, [], text); }

function route(edge) {
  return edge.points.map((point, index) => `${index ? 'L' : 'M'} ${point.x} ${point.y}`).join(' ');
}

function drawing(graph, options = {}) {
  const layout = layoutExecutionGraph(graph, options);
  const title = options.title ?? '实验执行图';
  const content = [label(28, 38, wrapText(title, 72)[0], { 'font-size': 26 }),
    label(28, 72, '从左到右，换行继续 · 实际调用开始顺序 · 分组含完整决策证据'),
    label(28, 107, '蓝：模型 / 算法调用   绿：技能程序   紫：阶段子图'),
    label(28, 136, '路线颜色表示前一调用结果：绿 = 完成，红 = 失败，灰 = 其他', { fill: '#b3c6d5' })];
  for (const group of layout.groups) {
    content.push(shape('rect', { x: group.x, y: group.y, width: group.width, height: group.height, rx: 14, fill: '#121f2d', stroke: '#716587', 'stroke-width': 2 }));
    content.push(shape('g', { 'data-node-id': group.summary.id, role: 'button', tabindex: 0,
      'aria-label': `${group.title}，${group.summary.events.length} 条事件，查看全部证据`, cursor: 'pointer' }, [
      label(group.x + 18, group.y + 31, group.title, { fill: COLORS.subgraph }),
      label(group.x + 18, group.y + 59, `skill ${group.callCount} · ${group.summary.events.length} 条事件`, { fill: '#b4c6d4' }),
    ]));
    content.push(shape('g', { 'data-toggle-group': group.id, role: 'button', tabindex: 0,
      'aria-expanded': String(group.expanded), 'aria-label': `${group.expanded ? '收起' : '展开'} ${group.title}`, cursor: 'pointer' }, [
      shape('title', {}, [], group.expanded ? '收起记录' : '展开每次调用及内部算法'),
      shape('rect', { x: group.x + group.width - 64, y: group.y + 14, width: 44, height: 44, rx: 7, fill: '#26354b', stroke: '#8c7ba5' }),
      label(group.x + group.width - 51, group.y + 43, group.expanded ? '−' : '+'),
    ]));
  }
  for (const edge of layout.edges) {
    const color = edge.status === 'failed' ? STATUS_COLORS.failed : edge.status === 'complete' ? STATUS_COLORS.complete : STATUS_COLORS.recorded;
    content.push(shape('path', { d: route(edge), fill: 'none', stroke: color, 'stroke-width': 3,
      'marker-end': `url(#arrow-${edge.status === 'failed' ? 'failed' : edge.status === 'complete' ? 'complete' : 'recorded'})` },
      [shape('title', {}, [], `调用开始顺序；前一调用结果：${STATUS_NAMES[edge.status]}`)]));
  }
  for (const item of layout.nodes) {
    const { node, x, y, width, height } = item;
    const lines = wrapText(node.title).slice(0, 2);
    if (wrapText(node.title).length > 2) lines[1] = lines[1].slice(0, -1) + '…';
    content.push(shape('g', { 'data-node-id': node.id, role: 'button', tabindex: 0, cursor: 'pointer',
      'aria-label': `${node.title}，${STATUS_NAMES[node.status]}，${node.events.length} 条事件`, 'aria-pressed': 'false' }, [
      shape('title', {}, [], `${node.title}\n${node.id}\n${timeLabel(node.startTime)} → ${timeLabel(node.endTime)}`),
      shape('rect', { x, y, width, height, rx: 10, fill: '#1b2c3c', stroke: COLORS[node.category], 'stroke-width': 2 }),
      label(x + 14, y + 29, CATEGORY_NAMES[node.category], { fill: COLORS[node.category] }),
      ...lines.map((line, index) => label(x + 14, y + 59 + index * 25, line, { 'font-size': 22 })),
      label(x + 14, y + 113, `${STATUS_NAMES[node.status]} · ${node.events.length} 条记录`, { fill: STATUS_COLORS[node.status] }),
      label(x + 14, y + 142, `${timeLabel(node.startTime)}${node.duration !== null ? ` · ${node.duration.toFixed(2)} s` : ''}`, { fill: '#b4c6d4' }),
    ]));
  }
  if (!layout.nodes.length) content.push(label(28, 200, '这次实验尚未记录事件。'));
  const defs = shape('defs', {}, ['failed', 'complete', 'recorded'].map(status => shape('marker', {
    id: `arrow-${status}`, markerWidth: 9, markerHeight: 9, refX: 8, refY: 4, orient: 'auto', markerUnits: 'userSpaceOnUse',
  }, [shape('path', { d: 'M0,0 L0,8 L8,4 z', fill: STATUS_COLORS[status] })])));
  return { layout, tree: shape('svg', { xmlns: 'http://www.w3.org/2000/svg', width: layout.width, height: layout.height,
    viewBox: `0 0 ${layout.width} ${layout.height}`, 'font-family': 'Segoe UI,Microsoft YaHei,sans-serif', 'font-size': 20,
    'aria-label': title }, [defs, shape('rect', { width: '100%', height: '100%', fill: '#0b131c' }),
    shape('g', { 'data-graph-layer': 'true' }, content)]) };
}

function serialize(node) {
  const attributes = Object.entries(node.attrs).map(([name, value]) => ` ${name}="${xml(value)}"`).join('');
  return `<${node.tag}${attributes}>${node.text === undefined ? '' : xml(node.text)}${node.children.map(serialize).join('')}</${node.tag}>`;
}

export function graphSvg(graph, options = {}) { return serialize(drawing(graph, options).tree); }

export function createExecutionGraph(container, { onSelect }) {
  const expanded = new Set();
  let graph = { nodes: [], edges: [], warnings: [] };
  let selectedId = null;
  let layer = null;
  let layout = null;
  let pan = null;
  const viewport = { x: 0, y: 0, scale: 1 };
  function transform() { layer?.setAttribute('transform', `translate(${viewport.x} ${viewport.y}) scale(${viewport.scale})`); }
  function zoom(factor) {
    const old = viewport.scale;
    viewport.scale = Math.min(3, Math.max(0.12, old * factor));
    const x = (container.clientWidth || 1024) / 2;
    const y = (container.clientHeight || 620) / 2;
    viewport.x = x - (x - viewport.x) * viewport.scale / old;
    viewport.y = y - (y - viewport.y) * viewport.scale / old;
    transform();
  }
  function fit() {
    if (!layout) return;
    viewport.scale = Math.min(1, (container.clientWidth || 1024) / layout.width, (container.clientHeight || 620) / layout.height);
    viewport.x = ((container.clientWidth || 1024) - layout.width * viewport.scale) / 2;
    viewport.y = ((container.clientHeight || 620) - layout.height * viewport.scale) / 2;
    transform();
  }
  function reset() { viewport.x = viewport.y = 0; viewport.scale = 1; transform(); }
  function render(next = graph) {
    graph = next;
    const drawingResult = drawing(graph, { expanded });
    layout = drawingResult.layout;
    const selections = new Map(layout.nodes.map(item => [item.node.id, item.node]));
    for (const group of layout.groups) selections.set(group.id, group.summary);
    function mount(source) {
      const node = container.ownerDocument.createElementNS('http://www.w3.org/2000/svg', source.tag);
      for (const [name, value] of Object.entries(source.attrs)) node.setAttribute(name, String(value));
      if (source.text !== undefined) node.textContent = source.text;
      for (const child of source.children) node.append(mount(child));
      if (source.attrs['data-graph-layer']) layer = node;
      const id = source.attrs['data-node-id'];
      const groupId = source.attrs['data-toggle-group'];
      if (id || groupId) {
        const activate = () => {
          if (groupId) { expanded.has(groupId) ? expanded.delete(groupId) : expanded.add(groupId); render(); }
          else {
            selectedId = id;
            for (const item of container.querySelectorAll('[data-node-id]')) item.setAttribute('aria-pressed', String(item.getAttribute('data-node-id') === id));
            onSelect(selections.get(id));
          }
        };
        node.addEventListener('click', activate);
        node.addEventListener('keydown', event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); activate(); } });
        if (id) node.setAttribute('aria-pressed', String(id === selectedId));
      }
      return node;
    }
    const svg = mount(drawingResult.tree);
    svg.removeAttribute('viewBox');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.addEventListener('wheel', event => { event.preventDefault(); zoom(event.deltaY < 0 ? 1.12 : 1 / 1.12); }, { passive: false });
    svg.addEventListener('pointerdown', event => {
      if (event.button !== 0 || event.target.closest('[data-node-id], [data-toggle-group]')) return;
      pan = { x: event.clientX, y: event.clientY, originalX: viewport.x, originalY: viewport.y };
      svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener('pointermove', event => {
      if (!pan) return;
      viewport.x = pan.originalX + event.clientX - pan.x;
      viewport.y = pan.originalY + event.clientY - pan.y;
      transform();
    });
    svg.addEventListener('pointerup', () => { pan = null; });
    svg.addEventListener('pointercancel', () => { pan = null; });
    container.replaceChildren(svg);
    transform();
  }
  return { render, expanded, zoom, fit, reset };
}

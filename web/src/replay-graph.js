import { graphEvidence, initialScope, scopeVisits, authorLabel, statusLabel, statusColor } from './replay-graph-evidence.js';

const NODE_WIDTH = 248;
const NODE_HEIGHT = 124;

// Coordinates only: the node array is not the execution order.
export function layoutGraph(scope) {
  const byId = new Map(scope.nodes.map(node => [node.id, node]));
  const seen = new Set([scope.entry]);
  const queue = byId.has(scope.entry) ? [scope.entry] : [];
  for (let index = 0; index < queue.length; index += 1) {
    for (const edge of scope.edges.filter(edge => edge.source === queue[index])) {
      if (!seen.has(edge.target) && byId.has(edge.target)) { seen.add(edge.target); queue.push(edge.target); }
    }
  }
  const ordered = [...queue, ...scope.nodes.map(node => node.id).filter(id => !seen.has(id))];
  const indices = new Map(ordered.map((id, index) => [id, index]));
  const routed = scope.edges.filter(edge => indices.get(edge.target) !== indices.get(edge.source) + 1
    || scope.edges.filter(other => other.source === edge.source && other.target === edge.target).length > 1);
  const top = 52 + routed.length * 28;
  const positions = new Map(ordered.map((id, index) => [id, { x: 32 + index * 370, y: top, row: 0, column: index }]));
  return { positions, width: 64 + Math.max(1, ordered.length) * 370 - 122, height: top + NODE_HEIGHT + 22, routed };
}

function short(text, length = 32) { return text.length > length ? text.slice(0, length - 1) + '…' : text; }

export function renderExecutableGraph(container, graph, events = null, skills = {}) {
  const document = container.ownerDocument;
  const html = (tag, text = '', className = '') => {
    const result = document.createElement(tag); result.textContent = text; result.className = className; return result;
  };
  const svgElement = (tag, attributes = {}, text = '') => {
    const result = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) result.setAttribute(key, String(value));
    result.textContent = text; return result;
  };
  const evidence = graphEvidence(events);
  const ownership = scope => scope.input_id && scope.task ? '物体独立编排'
    : scope.input_id == null && scope.task == null ? '共享能力' : '归属字段无效';
  const owned = graph.subgraphs.filter(scope => ownership(scope) === '物体独立编排').length;
  const shared = graph.subgraphs.filter(scope => ownership(scope) === '共享能力').length;
  const invalid = graph.subgraphs.length - owned - shared;
  container.append(html('p', `${owned} 个物体 subgraph · ${shared} 个共享 subgraph${invalid ? ` · ${invalid} 个归属无效` : ''}。root 中的「调用 subgraph」是 node，下方 subgraph 是被调用的定义。`, 'graph-legend'));
  const legend = html('div', '', 'graph-key');
  for (const [title, entries] of [
    ['来源文字', [['Responses 初始设计'], ['Decisions 新增'], ['Runtime 路由']]],
    ['node 边框 / 结果', [['成功 / yes', 'success'], ['失败', 'failure'], ['不确定 / 中断', 'unknown'], ['执行中', 'running'], ['no / selected', 'selected'], ['未执行', 'pending']]],
    ['edge 条件 / 箭头方向', [['success / yes', 'success'], ['failure', 'failure'], ['unknown', 'unknown'], ['no', 'no'], ['selected', 'selected']]],
    ['edge 线型', [['实线：实际走过', 'success', 'solid'], ['虚线：尚未走过', 'pending', 'dashed']]],
  ]) {
    const row = html('div'); row.append(html('strong', title));
    for (const [label, status, line] of entries) {
      const item = html('span', label);
      if (status) { const swatch = html('i'); swatch.style.borderColor = statusColor(status); swatch.className = line ? `graph-key-line ${line}` : 'graph-key-swatch'; item.append(swatch); }
      row.append(item);
    }
    legend.append(row);
  }
  container.append(legend, html('p', '默认图：Responses 编排 + 当前进度。运行时恢复折叠在各 scope 下方；# 为全任务实际调用顺序，同一 node 重试保留每一次记录。node 成功不等于物体任务已完成。', 'graph-view-note'));

  for (const [index, scope] of [graph, ...graph.subgraphs].entries()) {
    const initial = initialScope(scope, evidence);
    const visits = scopeVisits(evidence, scope.id);
    const runtimeNodes = scope.nodes.filter(node => evidence.nodeAuthors.has(`${scope.id}:${node.id}`));
    const runtimeEdges = scope.edges.filter(edge => evidence.edgeAuthors.has(`${scope.id}:${edge.id}`));
    const byId = new Map(scope.nodes.map(node => [node.id, node]));
    const section = html('details', '', 'graph-scope'); section.open = true;
    section.setAttribute('data-scope-id', scope.id);
    section.append(html('summary', index === 0 ? `graph · ${scope.id} · root` : `subgraph 定义 · ${scope.id} · ${ownership(scope)}`));
    if (scope.input_id) {
      const owner = html('details', '', 'graph-owner');
      owner.append(html('summary', `清单 ID：${scope.input_id}`), html('p',
        `来源 ${scope.task?.source_ref ?? '未指定'} → 目的地 ${scope.task?.destination_ref ?? '未指定'}；处理 ${scope.task?.processing?.join(', ') || '搬运'}`)); section.append(owner);
    }

    const selection = html('details', '', 'graph-node-inspector'); selection.hidden = true;
    const selectionTitle = html('summary', 'node 详情'); const information = html('div'); selection.append(selectionTitle, information);
    function choose(node) {
      selection.hidden = false; selection.open = true; information.replaceChildren();
      const key = `${scope.id}:${node.id}`; const history = evidence.nodes.get(key) ?? [];
      selectionTitle.textContent = `node · ${node.id} · ${authorLabel(evidence.nodeAuthors.get(key))}`;
      information.append(html('p', `类型：${node.type === 'subgraph' ? '调用 subgraph 的 node' : node.type}；实际调用 ${history.length} 次；最近结果：${statusLabel(history.at(-1)?.status ?? 'pending')}`));
      if (node.type === 'subgraph') {
        const open = html('button', `查看 subgraph 定义：${node.subgraph_id}`, 'button');
        open.addEventListener('click', () => {
          const target = [...container.querySelectorAll('[data-scope-id]')].find(item => item.getAttribute('data-scope-id') === node.subgraph_id);
          if (target) { target.open = true; target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
        }); information.append(open);
      }
      const contract = skills[node.skill];
      if (contract) for (const [label, value] of [['Description', contract.capability ?? contract.description], ['Inputs', contract.inputs], ['Outputs', contract.outputs]]) {
        information.append(html('p', `${label}. ${Array.isArray(value) ? value.join(', ') : value ?? '未归档'}`));
      }
      else if (node.type === 'skill') information.append(html('p', '此历史实验未单独归档 skill 契约。'));
      const edges = scope.edges.filter(edge => edge.target === node.id || edge.source === node.id);
      information.append(html('p', `关联 edges：${edges.length} 条。以下是完整 graph 的真实记录，包含未走过的恢复分支。`));
      const list = html('ul', '', 'graph-edge-list');
      for (const edge of edges) {
        const edgeKey = `${scope.id}:${edge.id}`; const count = evidence.edges.get(edgeKey)?.length ?? 0;
        list.append(html('li', `${edge.source} → ${edge.target} · ${edge.on} · ${authorLabel(evidence.edgeAuthors.get(edgeKey))} · ${count ? `已走 ${count} 次` : '未走过'} · edge ${edge.id}`));
      }
      information.append(list);
      const raw = html('details'); raw.append(html('summary', '原始 node 参数与每次执行证据'), html('pre', JSON.stringify({ node, visits: history }, null, 2), 'candidate-detail')); information.append(raw);
    }

    const scroll = html('div', '', 'policy-canvas');
    const { positions, width, height, routed } = layoutGraph(initial);
    const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': `${index ? 'subgraph' : 'graph'} ${scope.id} Responses 编排与进度` });
    const defs = svgElement('defs'); const markerId = `arrow-${graph.id}-${index}`;
    const marker = svgElement('marker', { id: markerId, markerWidth: 10, markerHeight: 10, refX: 9, refY: 5, orient: 'auto', markerUnits: 'userSpaceOnUse' });
    marker.append(svgElement('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: 'context-stroke' })); defs.append(marker); svg.append(defs);
    for (const edge of initial.edges) {
      const start = positions.get(edge.source); const end = positions.get(edge.target); if (!start || !end) continue;
      const adjacent = !routed.includes(edge); const lane = routed.indexOf(edge);
      const x1 = start.x + (adjacent ? NODE_WIDTH : 70); const x2 = end.x + (adjacent ? 0 : 100);
      const y1 = start.y + (adjacent ? NODE_HEIGHT / 2 : 0); const y2 = end.y + (adjacent ? NODE_HEIGHT / 2 : 0); const bendY = 20 + lane * 28;
      const d = adjacent ? `M ${x1} ${y1} L ${x2} ${y2}` : `M ${x1} ${y1} C ${x1} ${bendY}, ${x2} ${bendY}, ${x2} ${y2}`;
      const count = evidence.edges.get(`${scope.id}:${edge.id}`)?.length ?? 0; const color = statusColor(edge.on);
      const line = svgElement('path', { d, fill: 'none', stroke: color, 'stroke-width': count ? 2.5 : 1.5,
        'stroke-dasharray': count ? '' : '6 4', 'marker-end': `url(#${markerId})` });
      line.append(svgElement('title', {}, `${edge.id} · Responses 初始设计 · ${edge.source} → ${edge.target} · ${edge.on} · 已走 ${count} 次`)); svg.append(line);
      svg.append(svgElement('text', { x: (x1 + x2) / 2, y: adjacent ? y1 - 10 : .25 * y1 + .75 * bendY - 6,
        fill: color, 'font-size': 12, 'text-anchor': 'middle', 'paint-order': 'stroke', stroke: '#101d2b', 'stroke-width': 5 }, `${edge.on}${count ? ` ×${count}` : ''}`));
    }
    for (const node of initial.nodes) {
      const p = positions.get(node.id); const history = evidence.nodes.get(`${scope.id}:${node.id}`) ?? [];
      const status = history.at(-1)?.status ?? 'pending'; const color = statusColor(status);
      const group = svgElement('g', { tabindex: 0, role: 'button', 'aria-label': `node ${node.id}`, style: 'cursor:pointer', 'data-node-id': node.id });
      group.append(svgElement('rect', { x: p.x, y: p.y, width: NODE_WIDTH, height: NODE_HEIGHT, rx: 9, fill: node.type === 'subgraph' ? '#25364d' : '#192b3b', stroke: color, 'stroke-width': 2 }));
      const name = node.skill ?? node.subgraph_id ?? (node.type === 'router' ? 'Decisions 条件判断' : '结束');
      group.append(svgElement('title', {}, `${name}\nnode ${node.id}\nResponses 初始设计`));
      const nameLines = [name.slice(0, 28), name.length > 28 ? short(name.slice(28), 28) : ''];
      nameLines.filter(Boolean).forEach((line, i) => group.append(svgElement('text', { x: p.x + 10, y: p.y + 24 + i * 18, fill: '#ecf5ff', 'font-size': 14 }, line)));
      const type = node.type === 'subgraph' ? '调用 subgraph' : `${node.type} node`;
      group.append(svgElement('text', { x: p.x + 10, y: p.y + 63, fill: '#b5c8d9', 'font-size': 12 }, `${type} · Responses`));
      group.append(svgElement('text', { x: p.x + 10, y: p.y + 84, fill: color, 'font-size': 12 }, `${statusLabel(status)}${history.length ? ` · #${history[0].order}${history.length > 1 ? ` 起 / ${history.length} 次` : ''}` : ''}`));
      group.append(svgElement('text', { x: p.x + 10, y: p.y + 104, fill: '#9eb4c6', 'font-size': 11 }, short(node.id, 35)));
      group.addEventListener('click', () => choose(node)); group.addEventListener('keydown', event => { if (event.key === 'Enter') choose(node); }); svg.append(group);
    }
    scroll.append(svg); section.append(scroll);
    const runtimeVisits = visits.filter(visit => evidence.nodeAuthors.has(`${scope.id}:${visit.nodeId}`));
    if (runtimeNodes.length) section.append(html('p', `已折叠运行时扩展：${runtimeNodes.length} node / ${runtimeEdges.length} edge；实际进入 ${runtimeVisits.length} 次。展开下方记录查看 Decisions 选择和恢复结果。`, 'graph-recovery-note'));
    const trace = html('details', '', 'graph-execution');
    trace.append(html('summary', `实际执行顺序 · ${visits.length} 次 node 调用${runtimeVisits.length ? `，含 ${runtimeVisits.length} 次恢复调用` : ''}`));
    trace.append(html('p', '按开始时间编号；调用 subgraph 时会进入下方定义，再返回 root。没有开始记录的 node 不在此表。selected 只表示选择了 edge，不表示恢复成功。', 'graph-view-note'));
    const table = html('table', '', 'graph-visit-table'); const header = html('tr');
    for (const label of ['顺序 / 时间', 'node / 能力', '来源', '结果', '实际下一条 edge']) header.append(html('th', label));
    const head = html('thead'); head.append(header); const body = html('tbody'); table.append(head, body);
    for (const visit of visits) {
      const node = byId.get(visit.nodeId); const row = html('tr'); row.setAttribute('data-visit-order', String(visit.order));
      row.append(html('td', `#${visit.order} · ${Number(visit.start ?? 0).toFixed(2)}s`));
      const cell = html('td'); const button = html('button', `${node?.skill ?? node?.subgraph_id ?? node?.type ?? ''}\n${visit.nodeId}`, 'graph-visit-node');
      button.addEventListener('click', () => { if (node) { choose(node); selection.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } }); cell.append(button); row.append(cell);
      row.append(html('td', authorLabel(evidence.nodeAuthors.get(`${scope.id}:${visit.nodeId}`))));
      const result = html('td', statusLabel(visit.status)); result.style.color = statusColor(visit.status);
      if (visit.reason) { const detail = html('details'); detail.append(html('summary', '原因'), html('p', visit.reason)); result.append(detail); }
      row.append(result, html('td', visit.edge ? `${visit.edge.on} → ${visit.edge.target}\n${visit.edge.edge_id}` : visit.status === 'running' ? '等待结果' : node?.type === 'end' ? '结束 / 返回调用方' : '未记录转移')); body.append(row);
    }
    const tableScroll = html('div', '', 'graph-table-scroll'); tableScroll.append(table); trace.append(tableScroll); section.append(trace);
    const complete = html('details', '', 'graph-complete'); complete.append(html('summary', `完整 graph 检查 · ${scope.nodes.length} node / ${scope.edges.length} edge`));
    complete.append(html('p', '包含 Responses 初始设计与运行时扩展。Decisions 选择能力与参数；Runtime 将选择写入 graph，并生成 success / failure / unknown 返回路由。选择 node 查看全部关联 edge、来源和执行次数；JSON 导出始终包含完整 graph。', 'graph-view-note'));
    const picker = html('select'); picker.setAttribute('aria-label', `${scope.id} 完整 node 列表`); picker.append(html('option', '选择 node…'));
    for (const node of scope.nodes) { const option = html('option', `${node.id} · ${node.skill ?? node.subgraph_id ?? node.type} · ${authorLabel(evidence.nodeAuthors.get(`${scope.id}:${node.id}`))}`); option.value = node.id; picker.append(option); }
    picker.addEventListener('change', () => { if (byId.has(picker.value)) choose(byId.get(picker.value)); }); complete.append(picker); section.append(complete, selection); container.append(section);
  }
}

// SVG/PNG export is the visible Responses projection; JSON exports the full revision.
export function executableGraphSvg(container) {
  const svgs = [...container.querySelectorAll('.policy-canvas > svg')];
  if (!svgs.length) throw new Error('当前实验没有可导出的显式 graph');
  const document = container.ownerDocument;
  const root = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); root.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const sizes = svgs.map(svg => svg.getAttribute('viewBox').split(' ').map(Number));
  const width = Math.max(1200, ...sizes.map(size => size[2])); const height = sizes.reduce((sum, size) => sum + size[3] + 78, 110);
  root.setAttribute('width', width); root.setAttribute('height', height); root.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const background = document.createElementNS(root.namespaceURI, 'rect');
  for (const [key, value] of Object.entries({ width: '100%', height: '100%', fill: '#101d2b' })) background.setAttribute(key, value); root.append(background);
  function label(text, y, color = '#ecf5ff') {
    const item = document.createElementNS(root.namespaceURI, 'text');
    for (const [key, value] of Object.entries({ x: 32, y, fill: color, 'font-size': 16 })) item.setAttribute(key, value);
    item.textContent = text; root.append(item);
  }
  label('Responses 初始编排 + 进度；运行时恢复已折叠，完整 node / edge 见 JSON。# = 实际调用顺序', 28);
  label('node：绿=成功，红=失败，黄=不确定/中断，蓝=执行中，灰=未执行。来源见文字。', 54);
  label('edge：实线=走过 / 虚线=未走过；颜色和文字表示条件，箭头表示控制方向。', 80);
  let y = 110;
  svgs.forEach((svg, index) => {
    const scope = svg.closest('.graph-scope'); label(scope.querySelector('summary').textContent, y + 22);
    const note = scope.querySelector('.graph-recovery-note'); if (note) label(note.textContent, y + 45, '#b7a5ff');
    const copy = svg.cloneNode(true); copy.setAttribute('y', y + 64); copy.setAttribute('width', sizes[index][2]); copy.setAttribute('height', sizes[index][3]);
    root.append(copy); y += sizes[index][3] + 78;
  });
  return root.outerHTML;
}

import { archivedEntityLabels, displayEntityName, SKILL_LABELS } from './replay-labels.js';
import { graphEvidence, scopeVisits, policyView, statusLabel, statusColor, recoveryEpisodes } from './replay-graph-evidence.js';
import { layoutGraph, NODE_WIDTH } from './replay-graph-layout.js';
import { graphPanels } from './replay-graph-views.js';

const outcomeLabel = value => ({success: '成功', failure: '失败', unknown: '缺证',
  empty: '无候选', yes: '成立', no: '不成立'})[value] ?? value;
const nodeName = node => SKILL_LABELS[node.skill] ?? node.skill ?? node.subgraph_id
  ?? (node.type === 'router' ? node.id : `${outcomeLabel(node.status)}结束 / 返回`);

export function fitGraphCanvas(canvas, scale = null) {
  const svg = canvas.querySelector('svg');
  if (!svg || !canvas.clientWidth || !canvas.clientHeight) return;
  const [, , width, height] = svg.getAttribute('viewBox').split(' ').map(Number);
  const factor = scale ?? Math.min(1, (canvas.clientWidth - 16) / width);
  svg.setAttribute('width', width * factor); svg.setAttribute('height', height * factor);
  canvas.dataset.scale = String(factor);
  canvas.scrollTop = canvas.scrollLeft = 0;
}

function labelLines(text) {
  const lines = ['']; let units = 0;
  for (const char of text) {
    const size = char.charCodeAt(0) > 255 ? 2 : 1;
    if (units + size > 26) { if (lines.length === 2) { lines[1] += '…'; break; } lines.push(''); units = 0; }
    lines[lines.length - 1] += char; units += size;
  }
  return lines;
}

export function renderExecutableGraph(container, graph, events = null, skills = {}, selectedScope = graph.id, callSelections = {}) {
  container.replaceChildren();
  const document = container.ownerDocument;
  const html = (tag, text = '', className = '') => {
    const result = document.createElement(tag); result.textContent = text; result.className = className; return result;
  };
  const svgElement = (tag, attributes = {}, text = '') => {
    const result = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) result.setAttribute(key, String(value));
    result.textContent = text; return result;
  };
  const evidence = graphEvidence(events, graph);
  const names = archivedEntityLabels(events ?? []);
  const scopes = [graph, ...graph.subgraphs];
  const panels = graphPanels(graph);
  const buttons = new Map();
  function showScope(id) {
    selectedScope = id;
    for (const section of container.querySelectorAll('.graph-scope')) section.hidden = section.dataset.scopeId !== id;
    for (const [key, button] of buttons) button.setAttribute('aria-pressed', String(key === id));
    const canvas = container.querySelector('.graph-scope:not([hidden]) .policy-canvas');
    if (canvas) requestAnimationFrame(() => fitGraphCanvas(canvas, Number(canvas.dataset.scale ?? 1)));
  }
  if (!panels.some(panel => panel.id === selectedScope)) selectedScope = graph.id;
  const final = evidence.runEnd;
  container.append(html('p', final ? `任务最终结果：${final.status === 'succeeded' ? '已完成' : '未完成 / 已停止'}`
    : events?.length ? '尚无任务结束记录。节点结果不代表整个任务结果。' : '编排预览：尚未执行。', 'graph-run-result'));
  if (final?.error) {
    const detail = html('details'); detail.append(html('summary', '停止原因'), html('p', final.error)); container.append(detail);
  }
  const recoveries = recoveryEpisodes(graph, evidence, panels);
  const recoverySummary = html('section', '', 'graph-recovery-summary');
  recoverySummary.append(html('h3', `实际失败恢复 · ${recoveries.length} 次`));
  if (!recoveries.length) recoverySummary.append(html('p', '没有记录到进入恢复分支的实际转移。已编排的备选分支仍在下方保留。'));
  const definitions = new Map(scopes.flatMap(scope => scope.nodes.map(node => [`${scope.id}:${node.id}`, node])));
  function visitButton(visit, prefix = '') {
    const node = definitions.get(`${visit.scopeId}:${visit.nodeId}`);
    const button = html('button', `${prefix}${nodeName(node)} · ${statusLabel(visit.status)}`, 'graph-recovery-step');
    button.style.borderColor = statusColor(visit.status);
    button.title = `${visit.scopeId} / ${visit.nodeId} · 实际次序 #${visit.order}`;
    button.addEventListener('click', () => {
      const panel = panels.find(item => item.scopeId === visit.scopeId && item.nodes.some(value => value.id === visit.nodeId));
      renderExecutableGraph(container, graph, events, skills, panel.id,
        {...callSelections, [visit.scopeId]: visit.scopeCallId ?? 'all'});
      const section = container.querySelector('.graph-scope:not([hidden])');
      section?.scrollIntoView({block: 'start', behavior: 'smooth'});
      const target = [...section.querySelectorAll('[data-node-id]')].find(item => item.dataset.nodeId === visit.nodeId);
      target?.dispatchEvent(new MouseEvent('click', {bubbles: true}));
    });
    return button;
  }
  for (const [number, episode] of recoveries.entries()) {
    const card = html('article', '', 'graph-recovery-card');
    card.setAttribute('data-recovery-number', String(number + 1));
    const object = episode.trigger.targetId ?? episode.trigger.inputId;
    card.append(html('strong', `恢复 ${number + 1}${object ? ` · 物体 ${object}` : ''} · ${episode.status === 'returned' ? '已返回主线' : episode.status === 'stopped' ? '未返回主线，已停止' : '恢复中'}`));
    card.append(visitButton(episode.trigger, '触发：'));
    if (episode.trigger.reason) card.append(html('p', episode.trigger.reason, 'graph-recovery-reason'));
    const steps = html('div', '', 'graph-recovery-steps');
    for (const visit of episode.steps) {
      const node = definitions.get(`${visit.scopeId}:${visit.nodeId}`);
      if (node.type === 'end') continue;
      const step = html('div');
      step.append(html('span', '→', 'graph-recovery-arrow'), visitButton(visit,
        node.type === 'subgraph' ? '进入子图：' : node.type === 'router' ? '选择分支：' : ''));
      if (node.type === 'router' && visit.edge) step.append(html('small', `选择 ${visit.edge.on}`));
      steps.append(step);
    }
    card.append(steps);
    if (episode.returnVisit) card.append(visitButton(episode.returnVisit, '返回后：'));
    if (episode.returnVisit?.status === 'failure') card.append(html('p', '恢复路径已执行并返回；返回后的节点失败。', 'graph-recovery-reason'));
    recoverySummary.append(card);
  }
  container.append(recoverySummary);
  container.append(html('p', '主图、任务子图、失败处理分别查看。每个框是一个原始节点；虚线表示正常路线，跨视图连接可点击进入恢复或返回主线。灰色节点尚未执行，仍完整保留。', 'graph-legend'));
  const legend = html('div', '', 'graph-key');
  for (const [title, entries] of [
    ['最近一次节点结果', [['成功 / yes', 'success'], ['本次失败', 'failure'], ['证据不足 / 中断', 'unknown'], ['执行中', 'running'], ['判断 no', 'no'], ['未执行', 'pending']]],
    ['路线', [['虚线：正常路线', 'pending', 'dashed'], ['×次数：实际走过', 'success', 'dashed'], ['→编号：分支 / 回跳', 'unknown']]],
  ]) {
    const row = html('div'); row.append(html('strong', title));
    for (const [label, status, line] of entries) {
      const item = html('span', label);
      if (status) { const swatch = html('i'); swatch.style.borderColor = statusColor(status); swatch.className = line ? `graph-key-line ${line}` : 'graph-key-swatch'; item.append(swatch); }
      row.append(item);
    }
    legend.append(row);
  }
  container.append(legend, html('p', '节点失败或缺证，可以进入恢复后继续。子图调用框只报告该次子图的返回结果；任务最终结果单独显示。选择节点查看问题、参数、历史与全部出入边。', 'graph-view-note'));

  const navigation = html('nav', '', 'graph-scope-navigation');
  navigation.setAttribute('aria-label', '任务图层级');
  for (const [category, title] of [['root', '主图'], ['task', '任务子图'], ['recovery', '失败处理 / 恢复']]) {
    const group = html('div', '', `graph-navigation-group graph-navigation-${category}`);
    group.append(html('h4', title));
    const matches = panels.filter(panel => panel.category === category);
    const options = html('div', '', 'graph-navigation-options'); group.append(options);
    for (const panel of matches) {
      const button = html('button', '', 'graph-scope-button');
      const visits = scopeVisits(evidence, panel.scopeId).filter(visit => panel.nodes.some(node => node.id === visit.nodeId));
      button.append(html('strong', panel.title), html('span', `${panel.nodes.length} 节点 · 累计执行 ${visits.length} 次`));
      button.addEventListener('click', () => showScope(panel.id)); buttons.set(panel.id, button); options.append(button);
    }
    if (!matches.length) group.append(html('p', '未编排', 'graph-view-note'));
    navigation.append(group);
  }
  container.append(navigation);
  for (const [index, panel] of panels.entries()) {
    const scope = scopes.find(item => item.id === panel.scopeId);
    const calls = evidence.calls.get(scope.id) ?? [];
    const selectionId = callSelections[scope.id] ?? calls.at(-1)?.id;
    const call = calls.find(item => item.id === selectionId) ?? null;
    const fullView = policyView(scope, evidence, call);
    const ids = new Set(panel.nodes.map(node => node.id));
    const view = {...fullView, entry: panel.entry, entries: panel.entries, category: panel.category,
      nodes: fullView.nodes.filter(node => ids.has(node.id)),
      edges: fullView.edges.filter(edge => ids.has(edge.source)).map(edge => ({...edge,
        normal: edge.normal || panel.category !== 'recovery' && !['failure', 'unknown'].includes(edge.on)}))};
    const { positions, width, height, routes, bands, connections } = layoutGraph(view);
    const visits = scopeVisits(evidence, scope.id, call).filter(visit => ids.has(visit.nodeId));
    const byId = new Map(scope.nodes.map(node => [node.id, node]));
    const section = html('section', '', 'graph-scope');
    section.setAttribute('data-scope-id', panel.id);
    section.append(html('h3', `${panel.title} · 本视图 ${view.nodes.length} 节点 · ${call ? `第 ${call.number} 次调用` : '累计记录'}`));
    section.append(html('p', `所属 ${scope.id === graph.id ? '主图' : 'subgraph'}：${scope.id}，共 ${scope.nodes.length} 节点 / ${scope.edges.length} 条边。`
      + (panel.category === 'recovery' ? '这里展示失败或缺证后的处理；成功返回主线的连接保留，恢复失败后的终止也保留。' : '失败处理放在独立视图中，通过节点下的连接进入。')
      + (panel.sharedWithRecovery ? '此子图也被恢复路径复用。' : ''), 'graph-view-note'));
    if (calls.length > 1) {
      const picker = html('select'); picker.setAttribute('aria-label', '子图调用记录');
      for (const item of calls) {
        const option = html('option', `第 ${item.number} 次调用 · ${statusLabel(item.status)}`); option.value = item.id; picker.append(option);
      }
      const all = html('option', '全部调用（累计）'); all.value = 'all'; picker.append(all); picker.value = call?.id ?? 'all';
      picker.addEventListener('change', () => renderExecutableGraph(container, graph, events, skills, panel.id,
        {...callSelections, [scope.id]: picker.value})); section.append(picker);
    }
    if (!visits.length) section.append(html('p', '暂无执行记录；以下保留完整编排。', 'graph-view-note'));
    const lastOperation = visits.findLast(visit => byId.get(visit.nodeId)?.type !== 'end');
    if (lastOperation) {
      const latestNode = byId.get(lastOperation.nodeId), next = lastOperation.edge?.target;
      section.append(html('p', `本图最近执行：${positions.get(lastOperation.nodeId).number} · ${nodeName(latestNode)} · ${statusLabel(lastOperation.status)}。`
        + (next ? positions.has(next) ? `之后转至节点 ${positions.get(next).number}。` : '之后转入另一视图中的节点。' : '')
        + `尚未执行 ${view.nodes.filter(node => !node.history.length).length} 个节点（包含备选分支）。`, 'graph-view-note'));
    }
    const toolbar = html('div', '', 'actions graph-local-controls');
    for (const [label, factor] of [['＋', 1.2], ['−', 1 / 1.2]]) {
      const button = html('button', label, 'button');
      button.setAttribute('aria-label', factor > 1 ? '放大当前子图' : '缩小当前子图');
      button.addEventListener('click', () => {
        const canvas = section.querySelector('.policy-canvas');
        fitGraphCanvas(canvas, Math.min(2, Math.max(.1, Number(canvas.dataset.scale ?? 1) * factor)));
      });
      toolbar.append(button);
    }
    for (const [label, scale] of [['适应宽度', null], ['100%', 1]]) {
      const button = html('button', label, 'button');
      button.addEventListener('click', () => fitGraphCanvas(section.querySelector('.policy-canvas'), scale)); toolbar.append(button);
    }
    section.append(toolbar);
    if (scope.input_id) {
      const owner = html('details', '', 'graph-owner');
      owner.append(html('summary', '物体任务与技术标识'), html('p',
        `来源 ${displayEntityName(scope.task?.source_ref, names, '未指定')} → 目的地 ${displayEntityName(scope.task?.destination_ref, names, '未指定')}；处理 ${scope.task?.processing?.join(', ') || '搬运'}；永久 ID ${scope.input_id}`)); section.append(owner);
    }

    const selection = html('details', '', 'graph-node-inspector'); selection.hidden = true;
    const selectionTitle = html('summary', 'node 详情'); const information = html('div'); selection.append(selectionTitle, information);
    function navigateToNode(node) {
      const destination = panels.find(item => item.scopeId === scope.id && item.nodes.some(value => value.id === node.id));
      if (!destination) return;
      renderExecutableGraph(container, graph, events, skills, destination.id,
        {...callSelections, [scope.id]: call?.id ?? 'all'});
      const element = [...container.querySelectorAll('[data-node-id]')].find(item => item.dataset.nodeId === node.id);
      element?.scrollIntoView({block: 'center', behavior: 'smooth'}); element?.focus({preventScroll: true});
    }
    function choose(node) {
      if (!positions.has(node.id)) { navigateToNode(node); return; }
      selection.hidden = false; selection.open = true; information.replaceChildren();
      const history = view.nodes.find(item => item.id === node.id)?.history ?? [];
      selectionTitle.textContent = `节点 ${positions.get(node.id).number} · ${nodeName(node)} · ${node.id}`;
      information.append(html('p', `类型：${node.type === 'subgraph' ? '调用 subgraph 的 node' : node.type}；实际调用 ${history.length} 次；最近结果：${statusLabel(history.at(-1)?.status ?? 'pending')}；历史失败 ${history.filter(visit => visit.status === 'failure').length} 次、缺证 ${history.filter(visit => visit.status === 'unknown').length} 次。`));
      const latest = history.at(-1);
      if (latest?.edge) {
        const target = byId.get(latest.edge.target);
        information.append(html('p', `最近一次实际去向：${latest.edge.on} → ${latest.edge.target}${target?.type === 'end' ? '（结束 / 返回调用方）' : '（继续执行声明分支）'}。`));
      }
      if (node.type === 'subgraph') {
        const open = html('button', `进入：${scopeTitles.get(node.subgraph_id) ?? node.subgraph_id}`, 'button');
        open.addEventListener('click', () => renderExecutableGraph(container, graph, events, skills, node.subgraph_id,
          {...callSelections, [scope.id]: call?.id ?? 'all', [node.subgraph_id]: String(history.at(-1)?.eventId ?? '')})); information.append(open);
      }
      if (node.type === 'router') information.append(html('p', `判断问题：${node.condition}`));
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
        list.append(html('li', `${edge.source} → ${edge.target} · ${edge.on} · ${count ? `已走 ${count} 次` : '未走过'} · 每次进入本图最多 ${edge.max_traversals} 次 · edge ${edge.id}`));
      }
      information.append(list);
      const raw = html('details'); raw.append(html('summary', '原始 node 参数与每次执行证据'), html('pre', JSON.stringify({ node, visits: history }, null, 2), 'candidate-detail')); information.append(raw);
    }

    const scroll = html('div', '', 'policy-canvas');
    const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': `${panel.title} Responses 编排与进度` });
    for (const band of bands) svg.append(svgElement('text', {x: 32, y: band.y, fill: '#a7becf', 'font-size': 20}, band.label));
    let pinnedNode = null;
    function focusEdges(id) {
      svg.classList.toggle('has-edge-focus', Boolean(id));
      for (const group of svg.querySelectorAll('.graph-edge')) {
        group.classList.toggle('edge-focused', group.dataset.source === id || group.dataset.target === id);
      }
    }
    const defs = svgElement('defs'); const markerId = `arrow-${graph.id}-${index}`;
    const marker = svgElement('marker', { id: markerId, markerWidth: 10, markerHeight: 10, refX: 9, refY: 5, orient: 'auto', markerUnits: 'userSpaceOnUse' });
    marker.append(svgElement('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: 'context-stroke' })); defs.append(marker); svg.append(defs);
    function destinationIsRecovery(id) {
      return panels.some(item => item.scopeId === scope.id && item.category === 'recovery'
        && item.nodes.some(node => node.id === id));
    }
    function edgeElement(edge, jump = false) {
      const start = positions.get(edge.source); const end = positions.get(edge.target); if (!start) return null;
      const {d, x, y} = routes.get(edge.id);
      const count = edge.traversals;
      const color = count ? (panel.category === 'recovery' || destinationIsRecovery(edge.target)
        ? '#f3bf55' : edge.normal ? '#62d4ab' : statusColor(edge.on)) : '#a7becf';
      const edgeGroup = svgElement('g', {class: `graph-edge ${jump ? 'graph-jump' : 'graph-direct'} ${count ? 'edge-taken' : edge.normal ? 'edge-normal' : 'edge-unused'}`,
        'data-source': edge.source, 'data-target': edge.target});
      const line = svgElement('path', {d, fill: 'none', stroke: color, 'stroke-width': count ? 3 : 1.2,
        'data-edge-id': edge.authoredId, 'data-traversals': count,
        'stroke-dasharray': edge.normal || !count ? '8 6' : '', 'marker-end': `url(#${markerId})`});
      line.append(svgElement('title', {}, `${edge.id} · ${edge.source} → ${edge.target} · ${edge.on} · ${count ? `已走 ${count} 次` : '未走过'} · 每次进入本图上限 ${edge.max_traversals}`)); edgeGroup.append(line);
      const label = svgElement('text', {x, y, fill: color, 'font-size': 20, 'text-anchor': jump ? 'start' : 'middle',
        'paint-order': 'stroke', stroke: '#101d2b', 'stroke-width': 5});
      const condition = outcomeLabel(edge.on);
      const target = byId.get(edge.target);
      if (!target) return null;
      const destinationPanel = panels.find(item => item.scopeId === scope.id && item.nodes.some(node => node.id === edge.target));
      const targetLabel = target.type === 'end' ? '结束' : target.type === 'router' ? '判断'
        : target.type === 'subgraph' ? '子图' : SKILL_LABELS[target.skill] ?? '';
      const destination = end ? `${String(end.number).padStart(2, '0')} ${targetLabel}`
        : `${destinationPanel?.category === 'recovery' ? '失败处理' : '返回主线'} ↗`;
      label.textContent = jump ? `${condition.length > 10 ? condition.slice(0, 9) + '…' : condition} → ${destination}${count ? ` ×${count}` : ''}`
        : `${condition}${count ? ` ×${count}` : ''}`;
      edgeGroup.append(label);
      if (jump) {
        edgeGroup.setAttribute('tabindex', '0'); edgeGroup.setAttribute('role', 'button');
        edgeGroup.setAttribute('aria-label', `节点 ${start.number} ${condition}，${destination}，${nodeName(target)}`);
        const navigate = event => {
          if (!end) { event.stopPropagation(); navigateToNode(target); return; }
          event.stopPropagation(); pinnedNode = edge.target; focusEdges(edge.target); choose(byId.get(edge.target));
          const scale = Number(scroll.dataset.scale ?? 1);
          scroll.scrollTo({top: Math.max(0, end.y * scale - 24), left: Math.max(0, end.x * scale - 32), behavior: 'smooth'});
          svg.querySelector(`[data-node-number="${end.number}"]`)?.focus({preventScroll: true});
        };
        edgeGroup.addEventListener('click', navigate);
        edgeGroup.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); navigate(event);} });
      }
      return edgeGroup;
    }
    for (const edge of view.edges) if (routes.get(edge.id)?.direct) svg.append(edgeElement(edge));
    for (const node of view.nodes) {
      const p = positions.get(node.id);
      const status = node.visit.status; const color = statusColor(status);
      const group = svgElement('g', { tabindex: 0, role: 'button', style: 'cursor:pointer',
        'data-node-id': node.authoredId, 'data-node-number': p.number, 'data-node-status': status });
      group.append(svgElement('rect', { x: p.x, y: p.y, width: NODE_WIDTH, height: p.height, rx: 9, fill: node.type === 'subgraph' ? '#25364d' : '#192b3b', stroke: color, 'stroke-width': 2 }));
      const name = `${String(p.number).padStart(2, '0')}  ${nodeName(node)}`;
      group.setAttribute('aria-label', `${node.authoredId} · ${name} · ${statusLabel(status)} · 调用 ${node.history.length} 次`);
      group.append(svgElement('title', {}, `${name}\nnode ${node.authoredId}\n调用 ${node.history.length} 次`));
      const nameLines = labelLines(name);
      nameLines.filter(Boolean).forEach((line, i) => group.append(svgElement('text', { x: p.x + 14, y: p.y + 29 + i * 25, fill: '#ecf5ff', 'font-size': 22 }, line)));
      const childIsRecovery = panels.find(item => item.id === node.subgraph_id)?.category === 'recovery';
      const type = node.type === 'subgraph' ? `${childIsRecovery ? '恢复子图' : '任务子图'}调用 · 点击进入 ↗` : node.type === 'router' ? (node.choices?.length ? 'Decisions · 选择分支' : 'Decisions · 判断事实') : node.type === 'end' ? '结束本图 / 返回调用方' : '技能调用';
      group.append(svgElement('text', { x: p.x + 14, y: p.y + 81, fill: '#b5c8d9', 'font-size': 20 }, type));
      group.append(svgElement('text', { x: p.x + 14, y: p.y + 110, fill: color, 'font-size': 20 },
        `${node.history.length > 1 ? '最近：' : ''}${statusLabel(status)}${node.history.length ? ` · ${node.history.length} 次` : ''}`));
      if (connections.get(node.id).length) group.append(svgElement('path', {d: `M ${p.x + 12} ${p.y + 126} H ${p.x + NODE_WIDTH - 12}`, stroke: '#3e556b'}));
      group.addEventListener('pointerenter', () => focusEdges(node.id));
      group.addEventListener('pointerleave', () => focusEdges(pinnedNode));
      const activate = () => {
        pinnedNode = node.id; focusEdges(node.id);
        if (node.type === 'subgraph' && scopes.some(scope => scope.id === node.subgraph_id))
          renderExecutableGraph(container, graph, events, skills, node.subgraph_id,
            {...callSelections, [scope.id]: call?.id ?? 'all', [node.subgraph_id]: String(node.history.at(-1)?.eventId ?? '')});
        else choose(byId.get(node.authoredId));
      };
      group.addEventListener('click', activate); group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activate(); } }); svg.append(group);
      for (const edge of connections.get(node.id)) svg.append(edgeElement(edge, true));
    }
    scroll.append(svg); section.append(scroll);
    let drag = null;
    scroll.addEventListener('pointerdown', event => {
      if (event.button !== 0 || event.target.closest('[data-node-id],.graph-jump')) return;
      pinnedNode = null; focusEdges(null);
      drag = {x: event.clientX, y: event.clientY, left: scroll.scrollLeft, top: scroll.scrollTop};
      scroll.setPointerCapture(event.pointerId);
    });
    scroll.addEventListener('pointermove', event => { if (drag) { scroll.scrollLeft = drag.left + drag.x - event.clientX; scroll.scrollTop = drag.top + drag.y - event.clientY; } });
    scroll.addEventListener('pointerup', () => { drag = null; });
    scroll.addEventListener('pointercancel', () => { drag = null; });
    const trace = html('details', '', 'graph-execution');
    trace.append(html('summary', `实际执行顺序 · ${visits.length} 次 node 调用`));
    trace.append(html('p', '按开始时间编号；调用 subgraph 时会进入下方定义，再返回 root。没有开始记录的 node 不在此表。分支选择不等于技能执行成功。', 'graph-view-note'));
    const table = html('table', '', 'graph-visit-table'); const header = html('tr');
    for (const label of ['调用次序 / 时间', '节点 / 能力', '当时物体 ID', '结果', '实际下一条 edge']) header.append(html('th', label));
    const head = html('thead'); head.append(header); const body = html('tbody'); table.append(head, body);
    for (const visit of visits) {
      const node = byId.get(visit.nodeId); const row = html('tr'); row.setAttribute('data-visit-order', String(visit.order));
      row.append(html('td', `#${visit.order} · ${Number(visit.start ?? 0).toFixed(2)}s`));
      const cell = html('td'); const button = html('button', `${node?.skill ?? node?.subgraph_id ?? node?.type ?? ''}\n${visit.nodeId}`, 'graph-visit-node');
      button.addEventListener('click', () => { if (node) { choose(node); selection.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } }); cell.append(button); row.append(cell);
      row.append(html('td', visit.targetId ?? visit.inputId ?? '未绑定'));
      const result = html('td', statusLabel(visit.status)); result.style.color = statusColor(visit.status);
      if (visit.reason || visit.outcome) { const detail = html('details'); detail.append(html('summary', '来源 / 原因'), html('p', visit.reason ?? '')); if (visit.outcome) detail.append(html('pre', JSON.stringify(visit.outcome, null, 2))); result.append(detail); }
      row.append(result, html('td', visit.edge ? `${visit.edge.on} → ${visit.edge.target}\n${visit.edge.edge_id}` : visit.status === 'running' ? '等待结果' : node?.type === 'end' ? '结束 / 返回调用方' : '未记录转移')); body.append(row);
    }
    const tableScroll = html('div', '', 'graph-table-scroll'); tableScroll.append(table); trace.append(tableScroll); section.append(trace);
    const complete = html('details', '', 'graph-complete'); complete.append(html('summary', `完整 graph 检查 · ${scope.nodes.length} node / ${scope.edges.length} edge`));
    complete.append(html('p', '全部 node / edge 由开局 Responses 编排及校验修订。Decisions 只选择已声明的分支，Runtime 不追加节点。选择 node 可查看参数、全部关联 edge 和执行证据；JSON 导出包含完整编排。', 'graph-view-note'));
    const picker = html('select'); picker.setAttribute('aria-label', `${scope.id} 完整 node 列表`); picker.append(html('option', '选择 node…'));
    for (const node of scope.nodes) { const option = html('option', `${node.id} · ${node.skill ?? node.subgraph_id ?? node.type} · ${'Responses 编排'}`); option.value = node.id; picker.append(option); }
    picker.addEventListener('change', () => { if (byId.has(picker.value)) choose(byId.get(picker.value)); }); complete.append(picker); section.append(complete, selection); container.append(section);
  }
  showScope(selectedScope);
}

// SVG/PNG exports the selected presentation view, including its boundary connectors.
export function executableGraphSvg(container) {
  const svgs = [...container.querySelectorAll('.graph-scope:not([hidden]) .policy-canvas > svg')];
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
    for (const [key, value] of Object.entries({ x: 32, y, fill: color, 'font-size': 20 })) item.setAttribute(key, value);
    item.textContent = text; root.append(item);
  }
  label('完整 graph 与执行结果；每行从左到右，编号用于定位节点。灰色节点尚未执行。', 28);
  label('node：绿=成功，红=本次失败，黄=证据不足/中断，蓝=执行中。节点失败不等于任务终止。', 54);
  label('edge：虚线=正常路线；→编号=转至该节点；×次数=实际走过。每次调用另见回放。', 80);
  let y = 110;
  svgs.forEach((svg, index) => {
    const scope = svg.closest('.graph-scope'); label(scope.querySelector('h3').textContent, y + 22);
    const copy = svg.cloneNode(true); copy.setAttribute('y', y + 64); copy.setAttribute('width', sizes[index][2]); copy.setAttribute('height', sizes[index][3]);
    copy.classList.remove('has-edge-focus');
    for (const edge of copy.querySelectorAll('.graph-edge')) edge.setAttribute('opacity', '1');
    root.append(copy); y += sizes[index][3] + 78;
  });
  return root.outerHTML;
}

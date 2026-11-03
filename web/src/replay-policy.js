import { modelResult } from './replay-story.js';
import { renderTaskBindings } from './task-inventory.js';
import { renderExecutableGraph, executableGraphSvg } from './replay-graph.js';

function element(container, tag, text = '', className = '') {
  const node = container.ownerDocument.createElement(tag);
  node.textContent = text;
  node.className = className;
  return node;
}

// Display the archived control structure. Never infer a plan from executed skills.
export function renderPolicyGraph(container, graph, executionEvents = null, skills = {}) {
  container.replaceChildren();
  if (graph?.version === 2 && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    renderExecutableGraph(container, graph, executionEvents, skills);
    return;
  }
  container.append(element(container, 'p',
    '未归档当前 graph 格式；请查看原始编排数据。', 'muted'));
}

export function createReplayPolicy(container, { request, fileUrl }) {
  let revision = 0;
  let currentRun = null;
  let selectedId = null;
  const cache = new Map();
  let displayedGraph = null;
  let scale = 1;
  function clear() {
    revision += 1;
    currentRun = selectedId = null;
    cache.clear();
    displayedGraph = null;
    container.replaceChildren();
  }
  async function show(runId, events) {
    if (currentRun !== runId) clear();
    currentRun = runId;
    const operation = ++revision;
    const accepted = events.filter(event => ['skill_program', 'policy_graph_patch'].includes(event.kind));
    let skills = {};
    const catalogEvent = events.find(event => event.kind === 'skill_catalog');
    if (catalogEvent) {
      try {
        let catalog = catalogEvent.data;
        if (catalogEvent.artifact) {
          const url = fileUrl(runId, catalogEvent.artifact);
          if (!cache.has(url)) cache.set(url, request(url).catch(failure => { cache.delete(url); throw failure; }));
          catalog = await cache.get(url);
        }
        skills = catalog?.skills ?? {};
      } catch { /* Keep graph accessible if an old contract artifact is unavailable. */ }
    }
    const rejected = events.filter(event => ['skill_error', 'algorithm_error'].includes(event.kind)
      && event.data?.details?.rejected_proposal);
    const sources = accepted.length ? accepted : rejected.length ? [rejected[0]] : events.filter(event =>
      event.kind === 'visual_decisions_call' && event.data?.endpoint?.endsWith('/responses'));
    container.replaceChildren(element(container, 'p', sources.length ? '正在读取归档编排…' : '此实验尚未归档 Responses 编排。', 'muted'));
    const versions = await Promise.all(sources.map(async (event, index) => {
      const proposed = !accepted.length;
      const path = proposed ? event.data?.artifact ?? event.artifact : event.artifact;
      let graph;
      let bindings;
      let error;
      let original;
      try {
        let value = event.data;
        if (path) {
          const url = fileUrl(runId, path);
          if (!cache.has(url)) cache.set(url, request(url).catch(failure => { cache.delete(url); throw failure; }));
          value = await cache.get(url);
        }
        const output = proposed ? value?.details?.rejected_proposal ?? modelResult(value).inventory : null;
        original = proposed ? output : value;
        graph = proposed ? output?.policy_graph ?? output?.graph : value?.policy_graph ?? (value?.nodes ? value : null);
        bindings = proposed ? output?.task_inventory : value?.task_inventory;
      } catch (failure) { error = failure.message; }
      const end = index + 1 < sources.length ? events.indexOf(sources[index + 1]) : events.length;
      return { event, path, graph, bindings, original, error, proposed,
        execution: proposed ? null : events.slice(0, end),
        title: `${proposed ? rejected.includes(event) ? 'Responses graph（校验未通过，未执行）' : '模型提议（未见采纳记录）' : event.kind === 'skill_program' ? 'Responses 初始编排 graph' : '运行时 graph revision'} · 事件 #${event.id}` };
    }));
    if (operation !== revision || !versions.length) return;
    container.replaceChildren();
    const select = element(container, 'select');
    select.setAttribute('aria-label', 'graph 版本');
    for (const version of versions) {
      const option = element(container, 'option', version.title);
      option.value = String(version.event.id);
      select.append(option);
    }
    if (!versions.some(version => String(version.event.id) === selectedId)) {
      selectedId = String(versions.at(-1).event.id);
    }
    select.value = selectedId;
    const content = element(container, 'div');
    function render() {
      selectedId = select.value;
      const version = versions.find(item => String(item.event.id) === selectedId);
      content.replaceChildren();
      displayedGraph = version.graph;
      if (version.bindings) {
        const details = element(container, 'details');
        details.append(element(container, 'summary', '清单绑定与任务要求'));
        renderTaskBindings(details, version.bindings); content.append(details);
      }
      if (version.error) content.append(element(container, 'p', `编排读取失败：${version.error}`, 'inline-warning'));
      const tree = element(container, 'div', '', 'policy-tree');
      content.append(tree);
      renderPolicyGraph(tree, version.graph, version.execution, skills);
      resize();
      if (version.original) {
        const raw = element(container, 'details');
        raw.append(element(container, 'summary', '原始编排 JSON'),
          element(container, 'pre', JSON.stringify(version.original, null, 2), 'candidate-detail'));
        content.append(raw);
      }
    }
    select.addEventListener('change', render);
    container.append(select, content);
    render();
  }
  function resize(mode) {
    if (displayedGraph?.version !== 2) return;
    for (const svg of container.querySelectorAll('.policy-canvas > svg')) {
      const [, , width, height] = svg.getAttribute('viewBox').split(' ').map(Number);
      const factor = mode === 'fit' ? Math.max(.75, Math.min(1, (svg.parentElement.clientWidth - 2) / width)) : scale;
      svg.setAttribute('width', width * factor); svg.setAttribute('height', height * factor);
    }
  }
  return { show, clear, graph: () => displayedGraph, svg: () => executableGraphSvg(container),
    zoom(factor) { scale = Math.min(2, Math.max(.2, scale * factor)); resize(); },
    fit() { resize('fit'); }, reset() { scale = 1; resize(); } };
}

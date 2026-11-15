import { modelResult } from './replay-story.js';
import { renderTaskBindings } from './task-ledger.js';
import { renderExecutableGraph, executableGraphSvg, fitGraphCanvas } from './replay-graph.js';

function element(container, tag, text = '', className = '') {
  const node = container.ownerDocument.createElement(tag);
  node.textContent = text;
  node.className = className;
  return node;
}

// Display the archived control structure. Never infer a plan from executed skills.
export function renderPolicyGraph(container, graph, executionEvents = null, skills = {}, view = {}) {
  container.replaceChildren();
  if (graph?.version === 2 && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    renderExecutableGraph(container, graph, executionEvents, skills, view.scopeId, {}, view.templates);
    return;
  }
  container.append(element(container, 'p',
    '未归档当前 graph 格式；请查看原始编排数据。', 'muted'));
}

export function archivedTemplateDeclarations(archive, previous = {}) {
  const text = (archive?.response?.output ?? []).filter(item => item.type === 'message')
    .flatMap(item => item.content ?? []).filter(part => part.type === 'output_text').map(part => part.text).join('');
  let raw;
  try { raw = JSON.parse(text); } catch { return {}; }
  const templates = raw.root !== undefined || raw.remove_subgraph_ids !== undefined ? {...previous} : {};
  for (const id of raw.remove_subgraph_ids ?? []) delete templates[id];
  for (const scope of raw.graph?.subgraphs ?? raw.subgraphs ?? []) {
    if (scope.template) templates[scope.id] = {template:scope.template, arguments:scope.arguments};
    else delete templates[scope.id];
  }
  return templates;
}

export function preferredPolicyVersion(versions) {
  return versions.findLast(version => version.event.kind === 'skill_program' && version.graph)
    ?? versions.findLast(version => version.proposed && version.graph && !version.error && !version.validationErrors)
    ?? versions.findLast(version => version.proposed && version.graph)
    ?? versions.at(-1);
}

export function createReplayPolicy(container, { request, fileUrl }) {
  let revision = 0;
  let currentRun = null;
  let selectedId = null;
  let explicitVersionSelection = false;
  const cache = new Map();
  let displayedGraph = null;
  let scale = 1;
  function clear() {
    revision += 1;
    currentRun = selectedId = null;
    explicitVersionSelection = false;
    cache.clear();
    displayedGraph = null;
    container.replaceChildren();
  }
  async function show(runId, events) {
    if (currentRun !== runId) clear();
    currentRun = runId;
    const previousGraphId = displayedGraph?.id;
    const visibleScopes = [...container.querySelectorAll('.graph-scope:not([hidden])')];
    const previousScope = visibleScopes.length > 1 ? 'all' : visibleScopes[0]?.dataset.scopeId;
    const operation = ++revision;
    const accepted = events.filter(event => ['skill_program', 'terminal_recovery_started'].includes(event.kind));
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
    const proposals = events.filter(event => event.kind === 'visual_decisions_call'
      && event.data?.endpoint?.endsWith('/responses'));
    const sources = [...(proposals.length ? proposals : rejected.slice(0, 1)), ...accepted];
    container.replaceChildren(element(container, 'p', sources.length ? '正在读取归档编排…' : '此实验尚未归档 Responses 编排。', 'muted'));
    const versions = await Promise.all(sources.map(async (event, index) => {
      const proposed = !['skill_program', 'terminal_recovery_started'].includes(event.kind);
      const path = proposed ? event.data?.artifact ?? event.artifact : event.artifact;
      let graph;
      let bindings;
      let error;
      let original;
      let templateArchive;
      let templates = {};
      let validationErrors = event.data?.validation_errors;
      try {
        let value = event.data;
        if (path) {
          const url = fileUrl(runId, path);
          if (!cache.has(url)) cache.set(url, request(url).catch(failure => { cache.delete(url); throw failure; }));
          value = await cache.get(url);
        }
        let output = proposed ? value?.details?.rejected_proposal ?? modelResult(value).objects : null;
        if (proposed && !output && event.data?.request_hash) {
          const failure = rejected.find(row => row.data.details.request_hash === event.data.request_hash);
          if (failure) {
            let details = failure.data.details;
            if (failure.artifact) {
              const url = fileUrl(runId, failure.artifact);
              if (!cache.has(url)) cache.set(url, request(url).catch(error => { cache.delete(url); throw error; }));
              details = (await cache.get(url)).details;
            }
            output = details?.rejected_proposal;
          }
        }
        original = proposed ? output : value;
        templateArchive = value;
        templates = archivedTemplateDeclarations(value);
        graph = proposed ? output?.policy_graph ?? output?.graph : value?.policy_graph ?? value?.graph ?? (value?.nodes ? value : null);
        bindings = proposed ? output?.task_bindings : value?.task_bindings;
        validationErrors ??= value?.validation_errors ?? value?.details?.validation_errors;
      } catch (failure) { error = failure.message; }
      const end = index + 1 < sources.length ? events.indexOf(sources[index + 1]) : events.length;
      const attempt = event.data?.model_call?.attempt ?? 0;
      const draft = attempt ? `开局 Responses 图修订 ${attempt}` : '开局 Responses 初稿';
      const result = validationErrors ? '校验未通过' : event.data?.error ? '编排失败' : '已校验完整编排 · 实例未绑定';
      return { event, path, graph, bindings, original, error, proposed, validationErrors, templates, templateArchive,
        execution: proposed ? null : events.slice(0, end),
        title: `${event.kind === 'terminal_recovery_started' ? '系统失败回位图' : proposed ? `${draft} · ${result}` : '执行图 · 已绑定永久 ID'} · 事件 #${event.id}` };
    }));
    if (operation !== revision || !versions.length) return;
    // Execution graphs retain the declaration labels from their archived model
    // version. Never expand historical templates using today's library.
    let declarations = {};
    for (const version of versions) {
      if (version.proposed) {
        declarations = archivedTemplateDeclarations(version.templateArchive, declarations);
        version.templates = declarations;
      } else if (version.event.kind === 'skill_program') version.templates = declarations;
    }
    container.replaceChildren();
    const executionStart = events.findIndex(event => event.kind === 'policy_node_started');
    const duringExecution = proposals.filter(event => executionStart >= 0 && events.indexOf(event) > executionStart).length;
    container.append(element(container, 'p', executionStart < 0
      ? `Responses ${proposals.length} 次；尚无工作流节点执行记录。初稿与修订版本分别保留。`
      : `Responses：开局编译 ${proposals.length - duringExecution} 次（含图校验修订），工作流开始后 ${duringExecution} 次。第一个 graph 节点从事件 #${events[executionStart].id} 开始。`, 'graph-view-note'));
    const select = element(container, 'select');
    select.setAttribute('aria-label', 'graph 版本');
    for (const version of versions) {
      const option = element(container, 'option', version.title);
      option.value = String(version.event.id);
      select.append(option);
    }
    if (!explicitVersionSelection || !versions.some(version => String(version.event.id) === selectedId)) {
      selectedId = String(preferredPolicyVersion(versions).event.id);
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
        details.append(element(container, 'summary', '实例引用与任务要求'));
        renderTaskBindings(details, version.bindings); content.append(details);
      }
      if (version.error) content.append(element(container, 'p', `编排读取失败：${version.error}`, 'inline-warning'));
      if (version.validationErrors) {
        const validation = element(container, 'details');
        validation.append(element(container, 'summary', '图校验错误'),
          element(container, 'pre', JSON.stringify(version.validationErrors, null, 2), 'candidate-detail'));
        content.append(validation);
      }
      const tree = element(container, 'div', '', 'policy-tree');
      content.append(tree);
      if (version.proposed && versions.some(item => item.event.kind === 'terminal_recovery_started'))
        content.prepend(element(container, 'p', '任务未进入这份编排的执行；以下保留归档的完整任务图。系统失败回位图可在版本列表单独查看。', 'graph-view-note'));
      renderPolicyGraph(tree, version.graph, version.execution, skills, {
        templates:version.templates, scopeId:version.graph?.id === previousGraphId ? previousScope : undefined});
      if (version.original) {
        const raw = element(container, 'details');
        raw.append(element(container, 'summary', '原始编排 JSON'),
          element(container, 'pre', JSON.stringify(version.original, null, 2), 'candidate-detail'));
        content.append(raw);
      }
    }
    select.addEventListener('change', () => { explicitVersionSelection = true; render(); });
    container.append(select, content);
    render();
  }
  function resize(mode) {
    if (displayedGraph?.version !== 2) return;
    for (const canvas of container.querySelectorAll('.graph-scope:not([hidden]) .policy-canvas'))
      fitGraphCanvas(canvas, mode === 'fit' ? null : scale);
  }
  return { show, clear, graph: () => displayedGraph, svg: () => executableGraphSvg(container),
    zoom(factor) {
      const canvas = container.querySelector('.graph-scope:not([hidden]) .policy-canvas');
      scale = Math.min(2, Math.max(.1, Number(canvas?.dataset.scale ?? 1) * factor)); resize();
    },
    fit() { resize('fit'); }, reset() { scale = 1; resize(); } };
}

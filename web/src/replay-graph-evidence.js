// Archived execution evidence is separate from graph topology. A possible edge
// is never an execution; each visit to a repeated node remains a separate row.
export function graphEvidence(events) {
  const evidence = { nodes: new Map(), edges: new Map(), visits: [], nodeAuthors: new Map(), edgeAuthors: new Map() };
  const active = new Map();
  for (const event of events ?? []) {
    const data = event.data ?? {};
    const key = `${data.subgraph_id}:${data.node_id}`;
    if (event.kind === 'policy_graph_patch') {
      for (const node of Array.isArray(data.added_nodes) ? data.added_nodes : []) evidence.nodeAuthors.set(`${data.subgraph_id}:${node.id}`, data.source);
      for (const edge of Array.isArray(data.added_edges) ? data.added_edges : []) {
        // A Decisions patch selects a capability, not all three hypothetical
        // outcomes. NativeGraphExecutor supplies these return routes itself.
        const source = data.source === 'decisions' && edge.on !== 'selected' ? 'runtime_control' : data.source;
        evidence.edgeAuthors.set(`${data.subgraph_id}:${edge.id}`, source);
      }
    }
    if (event.kind === 'policy_node_started') {
      const visit = { order: evidence.visits.length + 1, nodeId: data.node_id, scopeId: data.subgraph_id,
        start: event.t_s, eventId: event.id, status: 'running' };
      evidence.visits.push(visit);
      const history = evidence.nodes.get(key) ?? [];
      history.push(visit); evidence.nodes.set(key, history); active.set(key, visit);
    }
    if (event.kind === 'policy_node' && active.has(key)) {
      Object.assign(active.get(key), { status: data.status, reason: data.reason, end: event.t_s });
    }
    if (event.kind === 'policy_edge') {
      const edgeKey = `${data.subgraph_id}:${data.edge_id}`;
      const history = evidence.edges.get(edgeKey) ?? [];
      history.push(event); evidence.edges.set(edgeKey, history);
      const visit = active.get(`${data.subgraph_id}:${data.source}`);
      if (visit) visit.edge = data;
    }
  }
  if ((events ?? []).some(event => event.kind === 'run_end')) {
    for (const visit of evidence.visits) if (visit.status === 'running') visit.status = 'interrupted';
  }
  return evidence;
}

export function scopeVisits(evidence, scopeId) {
  return evidence.visits.filter(visit => visit.scopeId === scopeId);
}

export function initialScope(scope, evidence) {
  const nodes = scope.nodes.filter(node => !evidence.nodeAuthors.has(`${scope.id}:${node.id}`));
  const ids = new Set(nodes.map(node => node.id));
  const edges = scope.edges.filter(edge => ids.has(edge.source) && ids.has(edge.target)
    && !evidence.edgeAuthors.has(`${scope.id}:${edge.id}`));
  return { ...scope, nodes, edges };
}

export const authorLabel = author => ({ decisions: 'Decisions 新增', runtime_control: 'Runtime 路由' })[author] ?? 'Responses 初始设计';
export const statusLabel = status => ({ success: '成功', failure: '失败', unknown: '不确定',
  running: '执行中', selected: '已选择 edge', yes: '判断 yes', no: '判断 no',
  interrupted: '中断 · 无结果', pending: '未执行' })[status] ?? status;
export const statusColor = status => ({ success: '#62d4ab', failure: '#ff7c83', unknown: '#f3bf55',
  running: '#71bfff', selected: '#b7a5ff', yes: '#62d4ab', no: '#c6b6eb', interrupted: '#f3bf55' })[status] ?? '#7c96ac';

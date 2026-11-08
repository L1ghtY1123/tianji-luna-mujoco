// Archived execution evidence is separate from graph topology. A possible edge
// is never an execution; each visit to a repeated node remains a separate row.
export function graphEvidence(events, graph = null) {
  const evidence = { nodes: new Map(), edges: new Map(), visits: [], calls: new Map(), runEnd: null };
  const active = new Map();
  for (const event of events ?? []) {
    const data = event.data ?? {};
    if (event.kind === 'run_end') evidence.runEnd = data;
    const key = `${data.subgraph_id}:${data.node_id}`;
    if (event.kind === 'policy_node_started') {
      const visit = { order: evidence.visits.length + 1, nodeId: data.node_id, scopeId: data.subgraph_id,
        start: event.t_s, eventId: event.id, targetId: event.target_id, inputId: data.input_id, status: 'running' };
      evidence.visits.push(visit);
      const history = evidence.nodes.get(key) ?? [];
      history.push(visit); evidence.nodes.set(key, history); active.set(key, visit);
    }
    if (event.kind === 'policy_node' && active.has(key)) {
      Object.assign(active.get(key), { status: data.status, reason: data.reason, judgment: data.judgment,
        targetId: event.target_id ?? active.get(key).targetId, outcome: data.outcome_evidence,
        end: event.t_s, endEventId: event.id });
    }
    if (event.kind === 'policy_edge') {
      const edgeKey = `${data.subgraph_id}:${data.edge_id}`;
      const history = evidence.edges.get(edgeKey) ?? [];
      history.push(event); evidence.edges.set(edgeKey, history);
      const visit = active.get(`${data.subgraph_id}:${data.source}`);
      if (visit) visit.edge = data;
    }
    if (event.kind === 'run_end' && data.status === 'needs_attention' && data.error && data.active_policy_node) {
      const node = data.active_policy_node;
      const visit = active.get(`${node.subgraph_id}:${node.node_id}`);
      // Older runs can stop on an exception before emitting policy_node. The
      // archived active node identifies the failed call; it supplies no edge.
      if (visit?.status === 'running') Object.assign(visit, {status: 'failure', reason: data.error,
        end: event.t_s, outcome: {producer: 'run_end_error', reason: data.error,
          details: {error_code: data.error_code}}});
    }
  }
  if ((events ?? []).some(event => event.kind === 'run_end')) {
    for (const visit of evidence.visits) if (visit.status === 'running') visit.status = 'interrupted';
  }
  if (graph) {
    const scopes = [graph, ...graph.subgraphs];
    const definitions = new Map(scopes.flatMap(scope => scope.nodes.map(node => [`${scope.id}:${node.id}`, node])));
    for (const scope of graph.subgraphs) {
      const calls = evidence.visits.filter(visit => definitions.get(`${visit.scopeId}:${visit.nodeId}`)?.subgraph_id === scope.id)
        .map((visit, index) => ({id: String(visit.eventId), number: index + 1,
          startEventId: visit.eventId, endEventId: visit.endEventId ?? Infinity,
          status: visit.status, caller: `${visit.scopeId}/${visit.nodeId}`}));
      evidence.calls.set(scope.id, calls);
      for (const visit of scopeVisits(evidence, scope.id)) {
        visit.scopeCallId = calls.find(call => visit.eventId > call.startEventId && visit.eventId < call.endEventId)?.id;
      }
    }
  }
  return evidence;
}

export function scopeVisits(evidence, scopeId, call = null) {
  return evidence.visits.filter(visit => visit.scopeId === scopeId && (!call || visit.scopeCallId === call.id));
}

export function recoveryEpisodes(graph, evidence, panels) {
  const definitions = new Map([graph, ...graph.subgraphs].flatMap(scope =>
    scope.nodes.map(node => [`${scope.id}:${node.id}`, node])));
  const recovery = new Set(panels.filter(panel => panel.category === 'recovery').flatMap(panel =>
    panel.nodes.map(node => `${panel.scopeId}:${node.id}`)));
  const edges = [...evidence.edges.values()].flat().sort((a, b) => a.id - b.id);
  const episodes = [];
  for (const trigger of evidence.visits) {
    if (!['failure', 'unknown'].includes(trigger.status) || !trigger.edge
        || recovery.has(`${trigger.scopeId}:${trigger.nodeId}`)
        || definitions.get(`${trigger.scopeId}:${trigger.edge.target}`)?.type === 'end') continue;
    if (!recovery.has(`${trigger.scopeId}:${trigger.edge.target}`)) {
      const returnVisit = evidence.visits.find(visit => visit.eventId > trigger.endEventId
        && visit.scopeId === trigger.scopeId && visit.nodeId === trigger.edge.target);
      episodes.push({trigger, steps: [], returnVisit, returnEdge: trigger.edge, status: 'returned'});
      continue;
    }
    const call = evidence.calls.get(trigger.scopeId)?.find(item => item.id === trigger.scopeCallId);
    const following = edges.filter(event => event.id > trigger.eventId && event.id < (call?.endEventId ?? Infinity)
      && event.data.subgraph_id === trigger.scopeId);
    const returned = following.find(event => recovery.has(`${trigger.scopeId}:${event.data.source}`)
      && !recovery.has(`${trigger.scopeId}:${event.data.target}`)
      && definitions.get(`${trigger.scopeId}:${event.data.target}`)?.type !== 'end');
    const stop = returned?.id ?? call?.endEventId ?? Infinity;
    const steps = evidence.visits.filter(visit => visit.eventId > trigger.eventId && visit.eventId < stop);
    const returnVisit = returned && evidence.visits.find(visit => visit.eventId > returned.id
      && visit.scopeId === trigger.scopeId && visit.nodeId === returned.data.target);
    episodes.push({trigger, steps, returnVisit, returnEdge: returned?.data,
      status: returned ? 'returned' : evidence.runEnd || Number.isFinite(stop) ? 'stopped' : 'running'});
  }
  return episodes;
}

export function policyView(scope, evidence, call = null) {
  const nodes = scope.nodes.map(node => {
    const history = (evidence.nodes.get(`${scope.id}:${node.id}`) ?? []).filter(visit => !call || visit.scopeCallId === call.id);
    return {...node, authoredId: node.id, history, visit: history.at(-1) ?? {status: 'pending'}};
  });
  const normal = new Set(['success', 'empty', 'yes', 'no']);
  const edges = scope.edges.map(edge => ({...edge, authoredId: edge.id, normal: normal.has(edge.on),
    traversals: (evidence.edges.get(`${scope.id}:${edge.id}`) ?? [])
      .filter(event => !call || event.id > call.startEventId && event.id < call.endEventId).length}));
  return {...scope, nodes, edges};
}

export const statusLabel = status => ({ success: '成功', failure: '本次失败', unknown: '证据不足',
  running: '执行中', yes: '判断 yes', no: '判断 no', empty: '无候选',
  interrupted: '中断 · 无结果', pending: '未执行' })[status] ?? status;
export const statusColor = status => ({ success: '#62d4ab', failure: '#ff7c83', unknown: '#f3bf55',
  running: '#71bfff', yes: '#62d4ab', no: '#c6b6eb', empty: '#87c6e8', interrupted: '#f3bf55' })[status] ?? '#7c96ac';

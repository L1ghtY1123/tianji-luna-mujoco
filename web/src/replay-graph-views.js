// Presentation partitions only. The authored scopes, nodes and edges remain unchanged.
function normalNodes(scope) {
  const nodes = new Set(scope.nodes.map(node => node.id));
  const reached = new Set(), pending = [scope.entry];
  while (pending.length) {
    const id = pending.pop();
    if (!nodes.has(id) || reached.has(id)) continue;
    reached.add(id);
    for (const edge of scope.edges) {
      if (edge.source === id && !['failure', 'unknown'].includes(edge.on)) pending.push(edge.target);
    }
  }
  return reached;
}

export function graphPanels(graph) {
  const scopes = [graph, ...graph.subgraphs];
  const definitions = new Map(scopes.map(scope => [scope.id, scope]));
  const normal = new Map(scopes.map(scope => [scope.id, normalNodes(scope)]));
  const uses = new Map(), pending = [{id: graph.id, recovery: false}], seen = new Set();
  while (pending.length) {
    const use = pending.pop(), key = `${use.id}:${use.recovery}`;
    if (seen.has(key) || !definitions.has(use.id)) continue;
    seen.add(key);
    const modes = uses.get(use.id) ?? new Set(); modes.add(use.recovery); uses.set(use.id, modes);
    for (const node of definitions.get(use.id).nodes) if (node.type === 'subgraph') {
      pending.push({id: node.subgraph_id, recovery: use.recovery || !normal.get(use.id).has(node.id)});
    }
  }
  const panels = [];
  for (const scope of scopes) {
    const recoveryOnly = uses.get(scope.id)?.has(true) && !uses.get(scope.id)?.has(false);
    const groups = recoveryOnly ? [{ids: new Set(scope.nodes.map(node => node.id)), recovery: true, whole: true}]
      : [{ids: normal.get(scope.id), recovery: false, whole: false},
         {ids: new Set(scope.nodes.filter(node => !normal.get(scope.id).has(node.id)).map(node => node.id)), recovery: true, whole: false}];
    for (const group of groups) {
      if (!group.ids.size) continue;
      const nodes = scope.nodes.filter(node => group.ids.has(node.id));
      const incoming = [...new Set(scope.edges.filter(edge => group.ids.has(edge.target) && !group.ids.has(edge.source)).map(edge => edge.target))];
      incoming.sort((a, b) => Number(nodes.find(node => node.id === a)?.type === 'end') - Number(nodes.find(node => node.id === b)?.type === 'end'));
      const entries = group.ids.has(scope.entry) ? [scope.entry] : incoming.length ? incoming : [nodes[0].id];
      panels.push({id: group.recovery && !group.whole ? `${scope.id}::recovery` : scope.id,
        scopeId: scope.id, entry: entries[0], entries, nodes, edges: scope.edges.filter(edge => group.ids.has(edge.source)),
        category: group.recovery ? 'recovery' : scope.id === graph.id ? 'root' : 'task',
        title: `${group.recovery ? group.whole ? '恢复子图' : '失败处理 / 恢复分支' : scope.id === graph.id ? '主图' : '任务子图'} · ${scope.id}`,
        wholeScope: group.whole, sharedWithRecovery: !group.recovery && uses.get(scope.id)?.has(true)});
    }
  }
  return panels;
}

// Layout never changes the policy. Long transitions use labelled jump connectors
// instead of sharing a bundle of lines across unrelated nodes.
export const NODE_WIDTH = 320;
export const NODE_HEIGHT = 144;

export function layoutGraph(scope) {
  const byId = new Map(scope.nodes.map(node => [node.id, node]));
  const primary = [], seen = new Set(), pending = [...(scope.entries ?? [scope.entry])].reverse();
  while (pending.length) {
    const id = pending.pop();
    if (seen.has(id) || !byId.has(id)) continue;
    seen.add(id); primary.push(byId.get(id));
    const next = scope.edges.filter(edge => edge.source === id && (edge.normal
      || scope.category === 'recovery' && !['failure', 'unknown'].includes(edge.on)))
      .sort((a, b) => Number(b.on === 'success') - Number(a.on === 'success'));
    for (const edge of next.reverse()) pending.push(edge.target);
  }
  const secondary = scope.nodes.filter(node => !seen.has(node.id));
  const columns = Math.min(3, Math.max(1, scope.nodes.length));
  const columnStep = NODE_WIDTH + 96;
  const width = columns * columnStep + 16;
  const positions = new Map(), bands = [], connections = new Map(), routes = new Map();
  let offset = 0;
  const groups = [[scope.category === 'recovery' ? '失败处理与恢复路线' : '正常执行路线', primary], ['本视图中的其他节点', secondary]];
  for (const [band, [, nodes]] of groups.entries()) {
    for (const [index, node] of nodes.entries()) {
      positions.set(node.id, {x: 32 + index % columns * columnStep,
        index: offset + index, number: offset + index + 1,
        row: Math.floor(index / columns), column: index % columns, band});
      connections.set(node.id, []);
    }
    offset += nodes.length;
  }
  const directPairs = new Set();
  for (const edge of scope.edges) {
    const a = positions.get(edge.source), b = positions.get(edge.target);
    if (!a) continue;
    const pair = `${edge.source}:${edge.target}`;
    const direct = b && edge.normal && a.band === b.band && b.index === a.index + 1 && !directPairs.has(pair);
    if (direct) directPairs.add(pair);
    else connections.get(edge.source).push(edge);
    routes.set(edge.id, {direct});
  }
  let top = 64;
  for (const [label, nodes] of groups) {
    if (!nodes.length) continue;
    bands.push({label, y: top - 18});
    for (let first = 0; first < nodes.length; first += columns) {
      const row = nodes.slice(first, first + columns);
      const height = Math.max(...row.map(node => NODE_HEIGHT + connections.get(node.id).length * 28));
      for (const node of row) Object.assign(positions.get(node.id), {y: top, height});
      top += height + 88;
    }
    top += 42;
  }
  for (const edge of scope.edges) {
    const a = positions.get(edge.source), b = positions.get(edge.target), route = routes.get(edge.id);
    if (!route) continue;
    if (!route.direct) {
      const y = a.y + NODE_HEIGHT + connections.get(edge.source).indexOf(edge) * 28 + 6;
      Object.assign(route, {d: `M ${a.x + 12} ${y - 6} H ${a.x + 26}`, x: a.x + 34, y});
    } else if (a.row === b.row) {
      const y = a.y + 56;
      Object.assign(route, {d: `M ${a.x + NODE_WIDTH} ${y} H ${b.x}`,
        x: (a.x + NODE_WIDTH + b.x) / 2, y: y - 12});
    } else {
      const sx = a.x + NODE_WIDTH / 2, tx = b.x + NODE_WIDTH / 2, bend = a.y + a.height + 44;
      Object.assign(route, {d: `M ${sx} ${a.y + a.height} V ${bend} H ${tx} V ${b.y}`,
        x: (sx + tx) / 2, y: bend - 12});
    }
  }
  return {positions, routes, connections, bands, nodeWidth: NODE_WIDTH, width, height: top};
}

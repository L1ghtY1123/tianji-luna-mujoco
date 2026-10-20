import assert from 'node:assert/strict';
import test from 'node:test';
import { buildExecutionGraph, graphSvg, layoutExecutionGraph } from '../src/execution-graph.js';

test('retries remain separate calls and late stage events bind by invocation ID', () => {
  const events = [
    { id: 1, kind: 'jev', title: '选择技能', t_s: 1 },
    { id: 2, kind: 'skill_start', title: 'plan_grasp', invocation_id: 'attempt-a', t_s: 2 },
    { id: 3, kind: 'skill_error', title: 'plan_grasp', invocation_id: 'attempt-a', t_s: 3 },
    { id: 4, kind: 'jev', title: '选择技能', invocation_id: 'attempt-a', t_s: 4 },
    { id: 5, kind: 'skill_start', title: 'plan_grasp', invocation_id: 'attempt-b', t_s: 5 },
    { id: 6, kind: 'planning_batch_complete', invocation_id: 'batch-a', parent_invocation_id: 'attempt-a', t_s: 6 },
    { id: 7, kind: 'execution_stage_complete', invocation_id: 'attempt-b', t_s: 7 },
    { id: 8, kind: 'skill_complete', title: 'plan_grasp', invocation_id: 'attempt-b', t_s: 8 },
  ];
  const graph = buildExecutionGraph(events);
  assert.deepEqual(graph.nodes.map(node => node.id), ['event:1', 'skill:attempt-a', 'event:4', 'skill:attempt-b']);
  assert.deepEqual(graph.nodes[1].events.map(event => event.id), [2, 3, 6]);
  assert.deepEqual(graph.nodes[3].events.map(event => event.id), [5, 7, 8]);
  assert.equal(graph.nodes[1].status, 'failed');
  assert.equal(graph.nodes[3].status, 'complete');
  assert.equal(graph.nodes[2].category, 'jev');
  assert.deepEqual(graph.edges.at(-1), { from: 'event:4', to: 'skill:attempt-b' });
  assert.equal(events[1].events, undefined);
});

test('events without invocation IDs are not guessed into a nearby skill', () => {
  const graph = buildExecutionGraph([
    { id: 1, kind: 'skill_start', title: 'observe', t_s: 1 },
    { id: 2, kind: 'execution_stage_start', data: { skill: 'observe' }, t_s: 2 },
    { id: 3, kind: 'skill_complete', title: 'observe', t_s: 3 },
  ]);
  assert.equal(graph.nodes.length, 3);
  assert.equal(graph.warnings.length > 0, true);
  assert.equal(graph.nodes.every(node => node.events.length === 1), true);
});

test('nested skill calls remain independent and duplicate event IDs do not duplicate nodes', () => {
  const event = { id: 2, kind: 'skill_start', title: 'child', invocation_id: 'child', parent_invocation_id: 'outer', t_s: 2 };
  const graph = buildExecutionGraph([
    { id: 1, kind: 'skill_start', title: 'outer', invocation_id: 'outer', t_s: 1 },
    event,
    { id: 3, kind: 'execution_stage_complete', invocation_id: 'child', t_s: 3 },
    event,
  ]);
  assert.equal(graph.nodes.length, 2);
  assert.equal(graph.nodes[1].parentInvocationId, 'outer');
  assert.deepEqual(graph.nodes[1].events.map(item => item.id), [2, 3]);
});

test('graph exports retain actual sequence, event identity and escape text', () => {
  const graph = buildExecutionGraph([{ id: 7, kind: 'jev', title: '<x & y>', t_s: 2.5 }]);
  const svg = graphSvg(graph, { title: '实验 <7>' });
  assert.match(svg, /&lt;x &amp; y&gt;/);
  assert.match(svg, /event:7/);
  assert.match(svg, /2.500/);
  assert.doesNotMatch(svg, /<x & y>/);
});

test('standalone algorithm events are labelled as algorithm calculations', () => {
  const graph = buildExecutionGraph([{ id: 1, kind: 'algorithm_complete', title: 'sam3', t_s: 4 }]);
  assert.equal(graph.nodes[0].category, 'algorithm');
});

test('a completed skill whose result is unsuccessful shows failure and elapsed duration', () => {
  const graph = buildExecutionGraph([
    { id: 1, kind: 'skill_start', title: 'verify_grasp', invocation_id: 'verify-1', t_s: 10 },
    { id: 2, kind: 'skill_complete', title: 'verify_grasp', invocation_id: 'verify-1', t_s: 13.25, data: { result: { success: false } } },
  ]);
  assert.equal(graph.nodes[0].status, 'failed');
  assert.equal(graph.nodes[0].duration, 3.25);
});

test('algorithm call IDs pair exact start/end evidence within their owning invocation', () => {
  const graph = buildExecutionGraph([
    { id: 1, kind: 'skill_start', title: 'observe', invocation_id: 's1', t_s: 1 },
    { id: 2, kind: 'algorithm_start', title: 'sam3', invocation_id: 's1', t_s: 2, data: { call_id: 'a1' } },
    { id: 3, kind: 'algorithm_start', title: 'sam3', invocation_id: 's1', t_s: 3, data: { call_id: 'a2' } },
    { id: 4, kind: 'algorithm_error', title: 'sam3', invocation_id: 's1', t_s: 4, data: { call_id: 'a1', error: 'timeout' } },
    { id: 5, kind: 'algorithm_complete', title: 'sam3', invocation_id: 's1', t_s: 7, data: { call_id: 'a2' } },
    { id: 6, kind: 'skill_complete', title: 'observe', invocation_id: 's1', t_s: 8 },
  ]);
  assert.deepEqual(graph.nodes[0].calls.map(call => [call.callId, call.status, call.duration, call.events.map(event => event.id)]),
    [['a1', 'failed', 2, [2, 4]], ['a2', 'complete', 4, [3, 5]]]);
});

test('compact phase layout preserves retry identities and all evidence when expanded', () => {
  const events = [];
  for (let i = 0; i < 40; i += 1) {
    events.push({ id: i * 3 + 1, kind: 'jev', title: '选择技能', t_s: i * 3, data: { output: { skill: 'recover_task' } } });
    events.push({ id: i * 3 + 2, kind: 'skill_start', title: 'recover_task', invocation_id: `s${i}`, t_s: i * 3 + 1 });
    events.push({ id: i * 3 + 3, kind: 'skill_error', title: 'recover_task', invocation_id: `s${i}`, t_s: i * 3 + 2 });
  }
  const graph = buildExecutionGraph(events);
  const compact = layoutExecutionGraph(graph);
  assert.equal(graph.nodes.filter(node => node.invocationId).length, 40);
  assert.ok(compact.height < 1000);
  assert.equal(compact.groups.length, 1);
  assert.equal(compact.nodes.length, 1);
  assert.deepEqual(compact.nodes[0].node.events.map(event => event.id), events.map(event => event.id));
  const expanded = layoutExecutionGraph(graph, { expanded: new Set([compact.groups[0].id]) });
  assert.equal(expanded.nodes.filter(item => item.node.invocationId).length, 40);
  assert.ok(expanded.nodes[1].x > expanded.nodes[0].x);
  assert.equal(expanded.nodes[1].y, expanded.nodes[0].y);
});

test('only recorded routes are drawn, with explicit failure and terminal unknown', () => {
  const graph = buildExecutionGraph([
    { id: 1, kind: 'skill_start', title: 'verify_grasp', invocation_id: 's1', t_s: 1 },
    { id: 2, kind: 'skill_complete', title: 'verify_grasp', invocation_id: 's1', t_s: 2, data: { result: { success: false } } },
    { id: 3, kind: 'run_end', title: '任务结束', t_s: 3, data: { status: 'needs_attention' } },
  ]);
  const layout = layoutExecutionGraph(graph);
  assert.equal(layout.edges.length, 1);
  assert.equal(layout.edges[0].status, 'failed');
  assert.equal(graph.nodes.at(-1).status, 'warning');
  const svg = graphSvg(graph);
  assert.match(svg, /失败/);
  assert.match(svg, /需关注/);
  assert.doesNotMatch(svg, /abort|not_found/);
});

test('an unfinished skill stays incomplete when an internal algorithm has completed', () => {
  const graph = buildExecutionGraph([
    { id: 1, kind: 'skill_start', title: 'observe', invocation_id: 's1', t_s: 1 },
    { id: 2, kind: 'algorithm_complete', title: 'sam3', invocation_id: 's1', t_s: 2, data: { call_id: 'a1' } },
  ]);
  assert.equal(graph.nodes[0].status, 'incomplete');
});

test('group evidence preserves file order even when IDs or clocks are not ordered', () => {
  const events = [
    { id: 8, kind: 'skill_start', title: 'recover_task', invocation_id: 's1', t_s: 8 },
    { id: 2, kind: 'jev', title: '恢复决策', invocation_id: 's1', t_s: 2 },
    { id: 4, kind: 'skill_error', title: 'recover_task', invocation_id: 's1', t_s: 4 },
  ];
  const group = layoutExecutionGraph(buildExecutionGraph(events)).groups[0];
  assert.deepEqual(group.summary.events.map(event => event.id), [8, 2, 4]);
});

test('unknown verification is distinguished from a recorded failure', () => {
  const graph = buildExecutionGraph([
    { id: 1, kind: 'skill_start', title: 'verify_grasp', invocation_id: 's1', t_s: 1 },
    { id: 2, kind: 'skill_complete', title: 'verify_grasp', invocation_id: 's1', t_s: 2, data: { result: { success: false, status: 'unknown' } } },
  ]);
  assert.equal(graph.nodes[0].status, 'warning');
});

test('small consecutive phase subgraphs share a horizontal row', () => {
  const graph = buildExecutionGraph(['observe', 'identify_target', 'identify_destination'].flatMap((title, index) => [
    { id: index * 2, kind: 'skill_start', title, invocation_id: `s${index}`, t_s: index * 2 },
    { id: index * 2 + 1, kind: 'skill_complete', title, invocation_id: `s${index}`, t_s: index * 2 + 1 },
  ]));
  const layout = layoutExecutionGraph(graph);
  assert.equal(layout.groups.length, 3);
  assert.equal(layout.groups[0].y, layout.groups[2].y);
  assert.ok(layout.groups[0].x < layout.groups[1].x && layout.groups[1].x < layout.groups[2].x);
  assert.ok(layout.height < 600);
});

test('a rejected skill without a start record uses its explicit skill field for the phase', () => {
  const graph = buildExecutionGraph([{ id: 1, kind: 'skill_error', title: '步骤失败：recover_task',
    invocation_id: 'rejected', t_s: 1, data: { skill: 'recover_task', error: 'precondition' } }]);
  assert.equal(layoutExecutionGraph(graph).groups[0].phase, 'recovery');
});

test('a normal task wraps horizontal stages across a compact overview without losing routes', () => {
  const skills = ['observe_scene', 'assess_task', 'identify_destination', 'build_scene', 'choose_arm', 'plan_grasp', 'execute_grasp', 'verify_grasp', 'plan_place', 'execute_place', 'verify_place'];
  const graph = buildExecutionGraph(skills.flatMap((title, index) => [
    { id: index * 2, kind: 'skill_start', title, invocation_id: `s${index}`, t_s: index * 2 },
    { id: index * 2 + 1, kind: 'skill_complete', title, invocation_id: `s${index}`, t_s: index * 2 + 1 },
  ]));
  const layout = layoutExecutionGraph(graph);
  assert.ok(layout.width / layout.height > 1.5, 'default overview should be landscape rather than a tall strip');
  assert.equal(layout.nodes.length, skills.length);
  assert.deepEqual(layout.edges.map(edge => [edge.from.node.id, edge.to.node.id]), graph.edges.map(edge => [edge.from, edge.to]));
  for (const item of layout.nodes) {
    assert.ok(item.x >= 0 && item.y >= 0 && item.x + item.width <= layout.width && item.y + item.height <= layout.height);
    for (const other of layout.nodes.filter(other => other !== item)) {
      assert.ok(item.x + item.width <= other.x || other.x + other.width <= item.x || item.y + item.height <= other.y || other.y + other.height <= item.y);
    }
  }
});

test('wrapped and expanded routes avoid unrelated cards', () => {
  const graph = buildExecutionGraph(['observe_scene', 'assess_task', 'identify_destination', ...Array(9).fill('plan_grasp'), 'execute_place'].flatMap((title, index) => [
    { id: index * 2, kind: 'skill_start', title, invocation_id: `s${index}`, t_s: index * 2 },
    { id: index * 2 + 1, kind: 'skill_complete', title, invocation_id: `s${index}`, t_s: index * 2 + 1 },
  ]));
  const collapsed = layoutExecutionGraph(graph);
  const layout = layoutExecutionGraph(graph, { expanded: new Set(collapsed.groups.map(group => group.id)) });
  for (const edge of layout.edges) {
    assert.ok(edge.points?.length >= 2, 'routes expose orthogonal points used by SVG export and browser');
    for (let i = 1; i < edge.points.length; i += 1) {
      const [a, b] = [edge.points[i - 1], edge.points[i]];
      assert.ok(a.x === b.x || a.y === b.y);
      for (const card of layout.nodes.filter(item => item !== edge.from && item !== edge.to)) {
        const crosses = a.x === b.x
          ? a.x > card.x && a.x < card.x + card.width && Math.max(a.y, b.y) > card.y && Math.min(a.y, b.y) < card.y + card.height
          : a.y > card.y && a.y < card.y + card.height && Math.max(a.x, b.x) > card.x && Math.min(a.x, b.x) < card.x + card.width;
        assert.equal(crosses, false, `${edge.from.node.id} → ${edge.to.node.id} crosses ${card.node.id}`);
      }
    }
  }
});

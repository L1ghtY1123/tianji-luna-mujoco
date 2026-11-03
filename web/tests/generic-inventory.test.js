import assert from 'node:assert/strict';
import test from 'node:test';
import { createObjectList } from '../src/object-list.js';
import { runEvidence, taskProgressSummary } from '../src/run-evidence.js';
import { inventorySnapshots, renderInventorySnapshot } from '../src/replay-story.js';
import { createReplayPolicy } from '../src/replay-policy.js';
import { buildExecutionGraph, buildObjectTimeline, renderObjectTimeline } from '../src/execution-graph.js';
import { buildReplayStages } from '../src/replay-story.js';

function container() {
  const document = { createElement: tag => ({ tag, ownerDocument: document, children: [], textContent: '',
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    addEventListener() {}, setAttribute() {},
  }) };
  return document.createElement('section');
}
const text = node => [node.textContent, ...node.children.map(text)].join(' ');
const progress = {
  entities: { raw: { id: 'raw', kind: 'object', description: '原料', status: 'consumed' },
    product: { id: 'product', kind: 'object', description: '成品', status: 'completed', current_region: 'shelf',
      processing_history: ['polish'], transfers: [{ destination_id: 'shelf', verified: true }] },
    spare: { id: 'spare', kind: 'object', status: 'observed' } },
  objects: [{ id: 'raw', status: 'consumed' }, { id: 'product', status: 'completed' }, { id: 'spare', status: 'observed' }],
  initial_entities: { raw: { id: 'raw' }, spare: { id: 'spare' } }, retired_entity_ids: ['raw'],
  transformations: { 'polish-1': { id: 'polish-1', skill: 'polish', input_entity_id: 'raw', output_entity_id: 'product',
    evidence: { observation_id: 'frame-2', points_world_m: ['HIDDEN'] } } },
  selected_input_ids: ['raw'], completed_input_ids: ['raw'], product_ids: ['product'],
};

test('live inventory shows generic entities, status and lineage without fixed transport stages', () => {
  const target = container();
  createObjectList(target, { document: target.ownerDocument }).render(progress);
  const value = text(target);
  assert.match(value, /原料.*已消耗/);
  assert.match(value, /成品.*已完成.*shelf/);
  assert.match(value, /raw.*→.*product.*polish/);
  assert.match(value, /spare/);
  assert.doesNotMatch(value, /第1段|第2段|待拆包|右盒|粉色包装件/);
});

test('execution history keeps distinct entity targets and arbitrary capability names', () => {
  const graph = buildExecutionGraph([
    { id: 1, kind: 'skill_start', invocation_id: 'a', target_id: 'raw', data: { skill: 'plan_grasp' } },
    { id: 2, kind: 'skill_complete', invocation_id: 'a', target_id: 'raw', data: { skill: 'plan_grasp' } },
    { id: 3, kind: 'skill_start', invocation_id: 'b', target_id: 'raw', data: { skill: 'polish' } },
    { id: 4, kind: 'skill_start', invocation_id: 'c', target_id: 'product', data: { skill: 'plan_grasp' } },
    { id: 5, kind: 'run_end', data: { task_progress: progress } },
  ]);
  const groups = buildObjectTimeline(graph);
  assert.deepEqual(groups.filter(group => group.targetId).map(group => group.targetId), ['raw', 'product']);
  const target = container();
  renderObjectTimeline(target, graph, () => {});
  assert.match(text(target), /raw.*→.*product.*polish/);
  assert.ok(buildReplayStages(graph).some(stage => stage.title === 'polish'));
});

test('generic completion counts selected inputs and retains bounded transformation evidence', () => {
  assert.equal(taskProgressSummary(progress), '已完成 1 / 1 个目标 · 已登记产物 1 个');
  const evidence = runEvidence({ task_progress: progress, task_inventory: { selected_ids: ['raw'], destination_id: 'shelf' } });
  assert.equal(evidence.task_progress.completed_count, 1);
  assert.deepEqual(evidence.task_progress.completed_input_ids, ['raw']);
  assert.deepEqual(evidence.task_progress.transformations[0].evidence, { observation_id: 'frame-2' });
  assert.equal(evidence.task_inventory.destination_id, 'shelf');
  assert.equal(JSON.stringify(evidence).includes('HIDDEN'), false);
});

test('replay accepts explicit generic snapshots but excludes observation proposals', () => {
  const events = [
    { id: 1, kind: 'skill_complete', data: { result: { entities: { proposal: { id: 'proposal' } } } } },
    { id: 2, kind: 'inventory_snapshot', data: { task_progress: progress } },
    { id: 3, kind: 'run_end', data: { task_progress: progress } },
  ];
  const snapshots = inventorySnapshots(events);
  assert.equal(snapshots.length, 2);
  const target = container();
  renderInventorySnapshot(target, snapshots[0], () => {});
  assert.match(text(target), /raw.*→.*product.*polish/);
  assert.match(text(target), /已消耗/);
  assert.doesNotMatch(text(target), /待拆包|右盒/);
  assert.deepEqual(events[1].data.task_progress, progress);
});

test('Responses policy_graph preserves all unexecuted nodes and task inventory bindings', async () => {
  const target = container();
  const graph = { steps: [{ type: 'skill', id: 'observe', skill: 'observe_scene' },
    { type: 'skill', id: 'finish', skill: 'custom_finish', arguments: { target_id: '$target' } }] };
  const output = { policy_graph: graph, task_inventory: { source_object_ids: ['raw', 'spare'], selected_ids: ['raw'],
    source_id: 'tray', destination_id: 'shelf', work_area_id: null, support_id: 'table' } };
  const viewer = createReplayPolicy(target, { fileUrl: (_run, path) => path, request: async () => ({
    endpoint: '/v1/responses', response: { output: [{ type: 'message', content: [{ type: 'output_text', text: JSON.stringify(output) }] }] },
  }) });
  await viewer.show('run', [{ id: 1, kind: 'visual_decisions_call', data: { endpoint: '/v1/responses', artifact: 'response.json' } }]);
  assert.match(text(target), /observe_scene/);
  assert.match(text(target), /custom_finish/);
  assert.match(text(target), /shelf/);
  assert.match(text(target), /spare/);
});

test('live policy retains unexecuted nodes without assigning success to a transformed input', () => {
  const target = container();
  const pending = { ...progress, completed_input_ids: [],
    transformations: Object.values(progress.transformations),
    objects: progress.objects.map(item => item.id === 'product' ? { ...item, status: 'observed' } : item) };
  createObjectList(target, { document: target.ownerDocument }).render(pending,
    { selected_ids: ['raw'], destination_id: 'shelf' },
    { steps: [{ type: 'skill', id: 'future', skill: 'deliver', arguments: { target_id: '$product' } }] }, []);
  assert.match(text(target), /deliver/);
  assert.match(text(target), /未见执行记录/);
  assert.match(text(target), /shelf/);
  assert.equal(taskProgressSummary(pending), '已完成 0 / 1 个目标 · 已登记产物 1 个');
});

test('accepted policy_graph wrapper keeps all nodes and records only matching outcomes', async () => {
  const target = container();
  const viewer = createReplayPolicy(target, { fileUrl: (_run, path) => path, request: async () => ({
    policy_graph: { steps: [{ type: 'skill', id: 'done', skill: 'observe_scene' },
      { type: 'skill', id: 'later', skill: 'inspect' }] }, task_inventory: { selected_ids: ['raw'] },
  }) });
  await viewer.show('run', [{ id: 1, kind: 'skill_program', artifact: 'program.json' },
    { id: 2, kind: 'policy_node', data: { node_id: 'done', status: 'success' } }]);
  assert.match(text(target), /observe_scene.*已记录执行 1 次 · 最近成功/s);
  assert.match(text(target), /inspect.*未见执行记录/s);
  assert.match(text(target), /raw/);
});

test('live state refresh preserves the expanded graph and scopes outcomes to its latest revision', () => {
  const target = container();
  const list = createObjectList(target, { document: target.ownerDocument });
  const graph = { steps: [{ type: 'skill', id: 'same-id', skill: 'inspect' }] };
  const events = [{ kind: 'policy_node', data: { node_id: 'same-id', status: 'success' } },
    { kind: 'policy_revised' }];
  list.render(progress, null, graph, events);
  target.children.at(-1).open = true;
  list.render(progress, null, graph, events);
  assert.equal(target.children.at(-1).open, true);
  assert.match(text(target), /未见执行记录/);
  assert.doesNotMatch(text(target), /最近成功/);
});

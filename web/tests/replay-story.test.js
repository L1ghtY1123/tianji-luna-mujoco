import assert from 'node:assert/strict';
import test from 'node:test';

test('new observation checkpoints supersede sparse legacy result snapshots', () => {
  const progress = { all_entities: {}, process_records: {} };
  const events = [
    { id: 1, kind: 'skill_complete', data: { result: progress } },
    ...['bind_inventory', 'verify_place', 'verify_unpack', 'observe_scene', 'verify_drop'].map((checkpoint, i) =>
      ({ id: i + 2, kind: 'inventory_snapshot', data: { checkpoint, task_progress: progress } })),
  ];
  const snapshots = inventorySnapshots(events);
  assert.equal(snapshots.length, 5);
  assert.equal(snapshots[1].title, '桌面放置核验');
  assert.equal(snapshots[3].title, '抓取前观测 · 清单核对');
});
import { buildExecutionGraph } from '../src/execution-graph.js';
import { groupObservationFrames, modelCallInfo, buildReplayStages, inventorySnapshots, modelResult } from '../src/replay-story.js';

test('one capture has named annotation revisions, without inventing another capture', () => {
  const observation = (id, frame, status) => ({ id: String(id), observationId: frame, time: id,
    event: { id, data: { object_labels: { analysis_status: 'analyzed', items: [{ status }] } } } });
  const frames = groupObservationFrames([observation(10, 'a', 'observed'), observation(16, 'a', 'selected'),
    observation(73, 'a', 'held'), observation(175, 'b', 'pending')], [
    { kind: 'observation_recorded', t_s: 3, data: { observation_id: 'a', checkpoint: 'initial' } },
  ]);
  assert.equal(frames.length, 2);
  assert.equal(frames[0].captureTime, 3);
  assert.equal(frames[0].revisions.length, 3);
  assert.match(frames[0].revisions[2].revisionTitle, /夹爪反馈/);
  assert.equal(frames[0].revisions[2].reusesCapture, true);
  assert.equal(frames[0].preferredId, '10');
});

test('model type comes from actual endpoint, not the historical event kind', () => {
  assert.equal(modelCallInfo({ data: { endpoint: '/v1/responses', purpose: 'initialize_inventory' } }).type, 'Responses');
  assert.equal(modelCallInfo({ data: { endpoint: '/v1/decisions', purpose: 'verify_release_result' } }).purpose, '确认桌面放置与松爪');
  assert.equal(modelCallInfo({ data: {} }).type, '模型调用');
});

test('stages preserve chronological skills and expose model calls once', () => {
  const events = [
    { id: 1, kind: 'skill_start', title: 'observe_scene', invocation_id: 'a', t_s: 0 },
    { id: 2, kind: 'visual_decisions_call', invocation_id: 'a', t_s: 2, data: { endpoint: '/v1/responses' } },
    { id: 3, kind: 'skill_complete', title: 'observe_scene', invocation_id: 'a', t_s: 3 },
    { id: 4, kind: 'skill_start', title: 'execute_grasp', invocation_id: 'b', t_s: 4 },
    { id: 5, kind: 'skill_complete', title: 'execute_grasp', invocation_id: 'b', t_s: 6 },
    { id: 6, kind: 'skill_start', title: 'verify_place', invocation_id: 'c', t_s: 7 },
    { id: 7, kind: 'visual_decisions_call', invocation_id: 'c', t_s: 8, data: { endpoint: '/v1/decisions' } },
    { id: 8, kind: 'skill_complete', title: 'verify_place', invocation_id: 'c', t_s: 9 },
  ];
  const before = JSON.stringify(events);
  const stages = buildReplayStages(buildExecutionGraph(events));
  assert.deepEqual(stages.map(stage => stage.phase), ['observe', 'grasp', 'place']);
  assert.deepEqual(stages.flatMap(stage => stage.models.map(event => event.id)), [2, 7]);
  assert.equal(JSON.stringify(events), before);
});

test('inventory snapshots never use future IDs or fabricate intermediate full snapshots', () => {
  const initial = { all_entities: [{ id: 'pink', color: 'pink' }], process_records: { p: { id: 'p', input_entity_id: 'pink', status: 'pending' } } };
  const final = { all_entities: [{ id: 'pink' }, { id: 'black' }], process_records: { p: { id: 'p', input_entity_id: 'pink', output_entity_id: 'black', status: 'completed' } } };
  const events = [
    { id: 8, kind: 'skill_complete', title: 'observe', data: { result: { objects: [{ id: 'proposal' }] } } },
    { id: 14, kind: 'skill_complete', title: 'bind_inventory', data: { result: initial } },
    { id: 158, kind: 'product_created', data: { process_id: 'p', input_entity_id: 'pink', output_entity_id: 'black', product_entity: { id: 'black', color: 'black' } } },
    { id: 268, kind: 'run_end', data: { task_progress: final } },
  ];
  const before = JSON.stringify(events);
  const snapshots = inventorySnapshots(events);
  assert.equal(snapshots.length, 3);
  assert.equal(snapshots[0].provenance, 'snapshot');
  assert.equal(snapshots[0].progress.all_entities.length, 1);
  assert.equal(snapshots[0].progress.process_records.p.output_entity_id, undefined);
  assert.equal(snapshots[1].provenance, 'event');
  assert.equal(snapshots[1].progress.process_records.p.output_entity_id, 'black');
  assert.equal(snapshots[2].progress.process_records.p.status, 'completed');
  assert.equal(JSON.stringify(events), before);
  assert.deepEqual(inventorySnapshots([{ kind: 'run_start' }]), []);
});

test('Responses exposes structured output only and Decisions keeps negative predicates explicit', () => {
  const response = modelResult({ endpoint: '/v1/responses', response: { output: [
    { type: 'reasoning', encrypted_content: 'private' },
    { type: 'message', content: [{ type: 'output_text', text: '{"entities":[{"ref":"pink"}],"task":{"supported":true},"program":{"steps":[]}}' }] },
  ] } });
  assert.equal(response.inventory.entities.length, 1);
  assert.equal(JSON.stringify(response).includes('private'), false);
  const decision = modelResult({ endpoint: '/v1/decisions', request: { questions: [
    { name: 'workpiece_still_held', type: 'predicate', instructions: 'Is the object still held?' },
  ] }, response: { answers: [{ name: 'workpiece_still_held', type: 'predicate', probability: 0 }] } });
  assert.match(decision.answers[0].title, /仍被夹持/);
  assert.equal(decision.answers[0].probability, 0);
  assert.match(decision.answers[0].instructions, /still held/);
  assert.equal(modelResult({ response: { answers: { count: 16, first: [] } } }).answers.length, 0);
});

test('real catalog dictionaries retain entity IDs during a product transition', () => {
  const snapshots = inventorySnapshots([
    { id: 1, kind: 'skill_complete', data: { result: { all_entities: { pink: { id: 'pink' } },
      process_records: { p: { id: 'p', input_entity_id: 'pink' } } } } },
    { id: 2, kind: 'product_created', data: { process_id: 'p', input_entity_id: 'pink', output_entity_id: 'black', product_entity: { id: 'black' } } },
  ]);
  assert.deepEqual(Object.keys(snapshots[1].progress.all_entities), ['pink', 'black']);
  assert.deepEqual(Object.keys(snapshots[0].progress.all_entities), ['pink']);
});

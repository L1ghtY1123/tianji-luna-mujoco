import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

test('new observation checkpoints supersede sparse legacy result snapshots', () => {
  const progress = { all_entities: {}, process_records: {} };
  const events = [
    { id: 1, kind: 'skill_complete', data: { result: progress } },
    ...['bind_inventory', 'verify_place', 'verify_unpack', 'observe_scene', 'verify_drop'].map((checkpoint, i) =>
      ({ id: i + 2, kind: 'inventory_snapshot', data: { checkpoint, task_progress: progress } })),
  ];
  const snapshots = inventorySnapshots(events);
  assert.equal(snapshots.length, 5);
  assert.equal(snapshots[1].title, 'verify_place');
  assert.equal(snapshots[3].title, 'observe_scene');
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
  assert.equal(modelCallInfo({ data: { endpoint: '/v1/decisions', purpose: 'verify_release_result' } }).purpose, '确认放置与松爪');
  assert.equal(modelCallInfo({ data: {} }).type, '模型调用');
});

test('reused capture keeps its original time and lists independent later analysis purposes', () => {
  const observations = [{ id: 'raw', observationId: 'frame', status: 'unprocessed',
    event: { id: 3, data: { object_labels: { analysis_status: 'unprocessed', items: [] } } } }];
  const events = [
    { id: 2, kind: 'observation_recorded', t_s: 10, data: { observation_id: 'frame', checkpoint: 'post_place' } },
    { id: 4, kind: 'observation_reused', t_s: 12, data: { observation_id: 'frame', purpose: 'associate_inventory' } },
    { id: 5, kind: 'observation_reused', t_s: 13, data: { observation_id: 'frame', purpose: 'task_completion' } },
  ];
  const frames = groupObservationFrames(observations, events);
  assert.equal(frames.length, 1);
  assert.equal(frames[0].captureTime, 10);
  assert.match(frames[0].analysisSummary, /复用.*清单.*最终完成/);
  assert.match(frames[0].analysisSummary, /不代表复用判断结果/);
  assert.match(frames[0].analysisSummary, /未记录.*模型调用/);
});

test('capture titles use their actual skill and target instead of the legacy initial checkpoint', () => {
  const records = ['first', 'later'].map((frame, i) => ({ id: String(i + 10), observationId: frame,
    status: 'analyzed', event: { id: i + 10, data: { object_labels: { analysis_status: 'analyzed',
      items: [{ id: 'product', name: '黑色物体', display_id: 2, status: 'observed' }] } } } }));
  const events = [
    { id: 1, kind: 'observation_recorded', t_s: 1, data: { observation_id: 'first', checkpoint: 'initial' } },
    { id: 2, kind: 'visual_decisions_call', data: { observation_id: 'first', endpoint: '/v1/responses', purpose: 'compile_task' } },
    { id: 3, kind: 'skill_start', invocation_id: 's2', title: 'observe_scene' },
    { id: 4, kind: 'observation_recorded', t_s: 150, invocation_id: 's2', target_id: 'product',
      data: { observation_id: 'later', checkpoint: 'initial' } },
  ];
  const frames = groupObservationFrames(records, events);
  assert.match(frames[0].title, /任务编排参考/);
  assert.match(frames[1].title, /场景识别.*#2.*黑色物体/);
  assert.doesNotMatch(frames[1].title, /初始/);
  assert.match(frames[1].source, /observe_scene.*s2.*#4/);
});

test('raw result captures show Decisions usage without claiming identity segmentation', () => {
  const observations = [{ id: '3', observationId: 'release', status: 'unprocessed',
    event: { id: 3, data: { object_labels: { analysis_status: 'unprocessed', items: [] } } } }];
  const events = [
    { id: 1, kind: 'skill_start', title: 'verify_place', invocation_id: 'verify' },
    { id: 2, kind: 'observation_recorded', invocation_id: 'verify', t_s: 10,
      data: { observation_id: 'release', checkpoint: 'post_place' } },
    { id: 4, kind: 'visual_decisions_call', data: { observation_id: 'release',
      endpoint: '/v1/decisions', purpose: 'verify_release_result', http_status: 200 } },
  ];
  const [frame] = groupObservationFrames(observations, events);
  assert.match(frame.title, /放置结果核验/);
  assert.match(frame.analysisSummary, /未保存.*身份标注/);
  assert.match(frame.analysisSummary, /Decisions.*确认放置/);
  assert.doesNotMatch(frame.analysisSummary, /未分析|未使用/);
  assert.equal(frame.revisions[0].revisionTitle, '原始 RGB（无身份叠加）');
});

test('missing capture records and unknown analysis remain explicit, without guessed motion or identity', () => {
  const [frame] = groupObservationFrames([{ id: '8', observationId: 'orphan', checkpoint: 'fine', time: 123,
    event: { id: 8, data: {} } }]);
  assert.equal(frame.captureTime, null);
  assert.doesNotMatch(frame.title, /返回观测位姿|初始/);
  assert.match(frame.source, /采集记录缺失/);
  assert.match(frame.revisions[0].revisionTitle, /标注状态未归档/);
  assert.match(frame.analysisSummary, /未记录.*模型调用/);
});

test('task completion capture is distinct from the preceding return even with a retained invocation id', () => {
  const observations = [{ id: '395', observationId: 'done', status: 'unprocessed',
    event: { id: 395, data: { object_labels: { analysis_status: 'unprocessed', items: [] } } } }];
  const [frame] = groupObservationFrames(observations, [
    { id: 383, kind: 'skill_start', title: 'return_to_initial_pose', invocation_id: 's33' },
    { id: 393, kind: 'skill_complete', title: 'return_to_initial_pose', invocation_id: 's33' },
    { id: 394, kind: 'observation_recorded', invocation_id: 's33', t_s: 248,
      data: { observation_id: 'done', checkpoint: 'task_completion' } },
  ]);
  assert.equal(frame.title, '任务完成检查');
  assert.match(frame.source, /任务完成检查.*#394/);
  assert.doesNotMatch(frame.source, /return_to_initial_pose/);
});

test('experiment 103 exposes all eight actual capture purposes and preserves raw-only verification', () => {
  const events = JSON.parse(readFileSync(new URL('./fixtures/observation-run103.json', import.meta.url), 'utf8'));
  const observations = events.filter(event => event.kind === 'observation_annotated').map(event => ({
    id: String(event.id), observationId: event.data.observation_id, time: event.t_s,
    status: event.data.object_labels.analysis_status, event,
  }));
  const frames = groupObservationFrames(observations, events);
  assert.equal(frames.length, 8);
  ['任务编排参考画面', '放置结果核验', '场景识别', '拆包结果核验', '投放结果核验',
    '场景识别', '返回任务起始位姿后采集', '任务完成检查'].forEach((title, index) =>
    assert.ok(frames[index].title.startsWith(title), frames[index].title));
  for (const index of [1, 4, 7]) {
    assert.equal(frames[index].revisions.length, 1);
    assert.match(frames[index].analysisSummary, /未保存.*身份标注.*Decisions/);
  }
  assert.match(frames[6].analysisSummary, /未记录.*模型调用/);
  assert.match(frames[2].title, /#1/);
  assert.match(frames[5].title, /#7/);
  assert.doesNotMatch(frames[5].analysisSummary, /抓前/);
  assert.doesNotMatch(frames[2].title, /unknown/);
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

test('historical product events never synthesize task snapshots', () => {
  const progress = { all_entities: { raw: { id: 'raw' } }, process_records: { p: { input_entity_id: 'raw' } } };
  const events = [
    { id: 1, kind: 'skill_complete', data: { result: progress } },
    { id: 2, kind: 'inventory_snapshot', data: { task_progress: progress } },
    { id: 3, kind: 'product_created', data: { process_id: 'p', input_entity_id: 'raw', output_entity_id: 'product', product_entity: { id: 'product' } } },
  ];
  const before = JSON.stringify(events);
  const snapshots = inventorySnapshots(events);
  assert.equal(snapshots.length, 1);
  assert.deepEqual(snapshots[0].progress, progress);
  assert.equal(JSON.stringify(events), before);
});

test('retired skill names remain raw events without special execution phases', () => {
  const names = ['mock_unpack', 'recover_task', 'inspect_visual_result'];
  const events = names.map((skill, index) => ({ id: index + 1, kind: 'skill_start',
    invocation_id: `old-${index}`, title: skill, data: { skill } }));
  const stages = buildReplayStages(buildExecutionGraph(events));
  assert.deepEqual(stages.map(stage => stage.title), names);
  assert.deepEqual(stages.map(stage => stage.phase), names.map(skill => `skill:${skill}`));
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

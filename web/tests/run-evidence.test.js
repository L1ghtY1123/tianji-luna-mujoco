import assert from 'node:assert/strict';
import test from 'node:test';
import { runEvidence, runStatusSummary, taskProgressSummary } from '../src/run-evidence.js';

test('process completion counts do not double-count input and output entities', () => {
  const progress = { process_records: { first: {}, second: {} }, completed_process_ids: ['first'],
    pending_unpack_ids: ['pink-2'], processing_ids: [], product_ids: ['black-1'],
    completed: [{ id: 'pink-1' }, { id: 'black-1' }] };
  assert.equal(taskProgressSummary(progress), '已完成 1 道工序 · 待拆包 1 件 · 处理中 0 件');
  assert.equal(runEvidence({ task_progress: progress }).task_progress.completed_count, 1);
});

test('run evidence displays holding, model stage, verification and preparation timing without raw clouds', () => {
  const run = { facts: { attached: true }, verification: { status: 'held', success: true, measurements: { matched_fraction: .9 } },
    collision_model: { mode: 'actual_closed', attached: true }, holding: { state: 'held', arm: 'A', verified: true, attached: true,
      points_tool_m: [[1, 2, 3]], attachment_evidence: { prepare_wall_ms: 12, secret: 'HIDDEN',
        collision_evidence: { full_object_coverage: false, geometry_coverage: 'measured_surface_voxels_with_uncertainty' } },
      verification: { status: 'held', success: true, measurements: { matched_fraction: .9 }, points_world_m: ['HIDDEN'] } },
    metrics: { run_duration_s: 1.2 } };
  const result = runEvidence(run);
  assert.equal(result.holding.state, 'held');
  assert.equal(result.holding.prepare_wall_ms, 12);
  assert.equal(result.collision_model.mode, 'actual_closed');
  assert.equal(result.verification.status, 'held');
  assert.equal(result.metrics.run_duration_s, 1.2);
  assert.equal(JSON.stringify(result).includes('HIDDEN'), false);
  assert.equal(JSON.stringify(result).includes('points_tool_m'), false);
});

test('pending release stays visible even without terminal metrics or a new verification', () => {
  const result = runEvidence({ holding: { state: 'release_pending', arm: 'B', attached: false, verified: false },
    collision_model: { mode: 'actual_open', attached: false } });
  assert.equal(result.holding.state, 'release_pending');
  assert.equal(result.holding.attached, false);
  assert.equal(result.verification, null);
});

test('compact summary identifies measured gripper, release pending and geometry preparation time', () => {
  const run = { holding: { state: 'release_pending', arm: 'B', verified: false, attached: false },
    collision_model: { gripper: 'actual_feedback', attached: false, prepare_wall_ms: 12.4 },
    verification: { status: 'unknown', success: false } };
  const text = runStatusSummary(run);
  assert.match(text, /释放待确认/);
  assert.match(text, /B/);
  assert.match(text, /实测夹爪/);
  assert.match(text, /unknown/);
  assert.match(text, /12.4 ms/);
  assert.equal(runStatusSummary({}), '');
});

test('runner provisional holding state has a Chinese label before visual confirmation', () => {
  const text = runStatusSummary({ holding: { state: 'provisional', arm: 'A', verified: false, attached: false } });
  assert.match(text, /A 臂 暂定持物 · 未确认/);
  assert.equal(text.includes('provisional'), false);
});

test('collection summary displays measured progress and readable deferred reasons', () => {
  const progress = { completed: [{ id: 'one', points_world_m: ['HIDDEN'] }],
    deferred: [{ id: 'two', reason: 'grasp_verification_failed' }, { id: 'three', reason: 'private_backend_code' }] };
  const text = taskProgressSummary(progress);
  assert.match(text, /已完成 1 个.*暂缓 2 个/);
  assert.match(text, /抓取未通过验证/);
  assert.match(text, /需要重新评估/);
  assert.equal(text.includes('private_backend_code'), false);
  assert.match(runStatusSummary({ task_progress: progress }), /已完成 1 个/);
  const evidence = runEvidence({ task_progress: progress });
  assert.equal(evidence.task_progress.completed.length, 1);
  assert.equal(JSON.stringify(evidence).includes('HIDDEN'), false);
  assert.equal(taskProgressSummary(undefined), '');
});

test('archived progress summaries retain full counts when event arrays were summarized', () => {
  assert.match(taskProgressSummary({ completed: { count: 25, first: [{ id: 'one' }] }, deferred: [] }), /已完成 25 个.*暂缓 0 个/);
});

import assert from 'node:assert/strict';
import test from 'node:test';
import { runEvidence, runStatusSummary, taskProgressSummary } from '../src/run-evidence.js';

test('historical process progress is not converted into modern counts or stages', () => {
  const progress = { process_records: { first: {} }, completed_process_ids: ['first'], stage_index: 1 };
  assert.equal(taskProgressSummary(progress), '历史清单：请查看归档原始记录');
  assert.equal(runEvidence({ task_progress: progress }).task_progress, null);
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

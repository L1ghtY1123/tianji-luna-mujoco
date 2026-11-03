import test from 'node:test';
import assert from 'node:assert/strict';
import { runConclusion } from '../src/run-conclusion.js';

test('uses the persisted evidence conclusion in live and replay views', () => {
  assert.equal(runConclusion({ conclusion: { summary: '缺少抓姿评估，原始失败保留。' } }), '缺少抓姿评估，原始失败保留。');
});

test('historical codes do not imply a fixed right-bin workflow', () => {
  const run = { status: 'needs_attention', error_code: 'drop_carried_projection_outside_bin',
    failed_step: { skill: 'plan_transport' } };
  assert.match(runConclusion(run), /plan_transport.*drop_carried_projection_outside_bin/);
  assert.equal(runConclusion({ ...run, conclusion: { brief: '简短原因', summary: '完整诊断' } }), '简短原因');
});

test('legacy failures include stage, evidence limits and no duplicate code', () => {
  const text = runConclusion({ status: 'failed', error: 'unknown_error', error_code: 'unknown_error',
    failed_step: { skill: 'plan_place' } });
  assert.match(text, /plan_place/);
  assert.doesNotMatch(text, /右盒|SAM3|分割失败/);
  assert.equal(text.split('unknown_error').length, 2);
});

test('legacy failure summary preserves the last record without inferring a recovery workflow', () => {
  const text = runConclusion({ status: 'failed', error: 'retry_plan_requires_current_evaluation' }, [
    { kind: 'task_failure', data: { stage: 'choose_arm', code: 'no_reachable_arm' } },
    { kind: 'task_failure', data: { stage: 'recover_task', code: 'retry_plan_requires_current_evaluation' } },
  ]);
  assert.match(text, /recover_task/);
  assert.match(text, /retry_plan_requires_current_evaluation/);
});

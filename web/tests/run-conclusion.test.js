import test from 'node:test';
import assert from 'node:assert/strict';
import { runConclusion } from '../src/run-conclusion.js';

test('uses the persisted evidence conclusion in live and replay views', () => {
  assert.equal(runConclusion({ conclusion: { summary: '缺少抓姿评估，原始失败保留。' } }), '缺少抓姿评估，原始失败保留。');
});

test('legacy failures include stage, evidence limits and no duplicate code', () => {
  const text = runConclusion({ status: 'failed', error: 'unknown_error', error_code: 'unknown_error',
    failed_step: { skill: 'plan_place' } });
  assert.match(text, /plan_place/);
  assert.match(text, /证据不足/);
  assert.equal(text.split('unknown_error').length, 2);
});

test('replay recovers original failure rather than presenting the invalid recovery as its cause', () => {
  const text = runConclusion({ status: 'failed', error: 'retry_plan_requires_current_evaluation' }, [
    { kind: 'task_failure', data: { stage: 'choose_arm', code: 'no_reachable_arm' } },
    { kind: 'task_failure', data: { stage: 'recover_task', code: 'retry_plan_requires_current_evaluation' } },
  ]);
  assert.match(text, /choose_arm/);
  assert.match(text, /no_reachable_arm/);
  assert.match(text, /最终停止/);
});

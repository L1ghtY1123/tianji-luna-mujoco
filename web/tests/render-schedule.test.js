import assert from 'node:assert/strict';
import test from 'node:test';
import { createRenderSchedule } from '../src/render-schedule.js';

test('idle scene renders once, then only after invalidation', () => {
  const schedule = createRenderSchedule(30);
  assert.equal(schedule.shouldRender(0), true);
  assert.equal(schedule.shouldRender(1000), false);
  schedule.invalidate();
  assert.equal(schedule.shouldRender(1001), true);
  assert.equal(schedule.shouldRender(2000), false);
});

test('continuous simulation updates render at the chosen maximum frame rate', () => {
  const schedule = createRenderSchedule(30);
  assert.equal(schedule.shouldRender(0), true);
  schedule.invalidate();
  assert.equal(schedule.shouldRender(16), false);
  assert.equal(schedule.shouldRender(34), true);
  schedule.invalidate();
  assert.equal(schedule.shouldRender(50), false);
  assert.equal(schedule.shouldRender(68), true);
});

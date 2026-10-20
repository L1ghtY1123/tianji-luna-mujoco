import assert from 'node:assert/strict';
import test from 'node:test';
import { createTaskClock } from '../src/task-clock.js';

function clockFixture() {
  let timeMs = 5000;
  let wallTimeMs = 9_000_000;
  const clock = createTaskClock({ now: () => timeMs, wallNow: () => wallTimeMs });
  return {
    clock,
    advance(ms) { timeMs += ms; wallTimeMs += ms; },
    setWallTime(ms) { wallTimeMs = ms; },
  };
}

const idle = { status: 'idle' };
const running = { status: 'running', run_id: 'first', generation: 1 };
const startEvent = { id: 1, kind: 'run_start', time: 1000, data: {} };

test('idle remains zero until Start is clicked', () => {
  const { clock, advance } = clockFixture();
  advance(20_000);
  clock.sync(idle, [startEvent], 1020);
  assert.equal(clock.elapsedSeconds(), 0);
  clock.onStart();
  assert.equal(clock.read(), 0);
  advance(1250);
  assert.equal(clock.elapsedSeconds(), 1.25);
});

test('old idle snapshots do not stop the clock while Start is pending', () => {
  const { clock, advance } = clockFixture();
  clock.onStart();
  advance(500);
  clock.sync(idle, [], 1000);
  advance(500);
  assert.equal(clock.elapsedSeconds(), 1);
});

test('running uses the server start time despite a different browser wall clock', () => {
  const { clock, advance, setWallTime } = clockFixture();
  clock.sync(running, [startEvent], 1012);
  assert.equal(clock.elapsedSeconds(), 12);
  advance(1500);
  setWallTime(1);
  assert.equal(clock.elapsedSeconds(), 13.5);
});

test('running continues without new snapshots and calibrates when its start event arrives', () => {
  const { clock, advance } = clockFixture();
  clock.onStart();
  advance(500);
  clock.sync(running, [], 1001);
  assert.equal(clock.elapsedSeconds(), 0.5);
  clock.sync(running, [startEvent], 1001.5);
  assert.equal(clock.elapsedSeconds(), 1.5);
  advance(3000);
  clock.sync(undefined);
  assert.equal(clock.elapsedSeconds(), 4.5);
});

test('terminal metrics freeze the duration through repeated snapshots and refresh', () => {
  const { clock, advance } = clockFixture();
  clock.sync(running, [startEvent], 1009);
  const completed = { ...running, status: 'succeeded', metrics: { run_duration_s: 10.75 } };
  clock.sync(completed, [startEvent], 1020);
  advance(5000);
  clock.sync(completed, [startEvent], 1025);
  assert.equal(clock.elapsedSeconds(), 10.75);
  const refreshed = clockFixture().clock;
  refreshed.sync(completed, [startEvent], 1100);
  assert.equal(refreshed.elapsedSeconds(), 10.75);
});

test('Reset immediately holds zero while the server still reports the old run', () => {
  const { clock, advance } = clockFixture();
  clock.sync(running, [startEvent], 1012);
  clock.reset();
  advance(2000);
  clock.sync(running, [startEvent], 1014);
  assert.equal(clock.elapsedSeconds(), 0);
  clock.sync(idle, [], 1014);
  advance(2000);
  assert.equal(clock.elapsedSeconds(), 0);
  clock.onStart();
  advance(500);
  assert.equal(clock.elapsedSeconds(), 0.5);
});

test('a different run starts from its own elapsed time', () => {
  const { clock, advance } = clockFixture();
  clock.sync(running, [startEvent], 1012);
  advance(4000);
  clock.sync({ ...running, run_id: 'second' }, [{ ...startEvent, time: 1030 }], 1030.25);
  assert.equal(clock.elapsedSeconds(), 0.25);
  advance(500);
  assert.equal(clock.elapsedSeconds(), 0.75);
});

test('failed Start restores the previous result without accepting stale terminal snapshots', () => {
  const { clock, advance } = clockFixture();
  const previous = { ...running, status: 'failed', metrics: { run_duration_s: 7.25 } };
  clock.sync(previous, [startEvent], 1010);
  clock.onStart();
  advance(500);
  clock.sync(previous, [startEvent], 1011);
  assert.equal(clock.elapsedSeconds(), 0.5);
  clock.cancelPending();
  assert.equal(clock.elapsedSeconds(), 7.25);
  advance(1000);
  assert.equal(clock.elapsedSeconds(), 7.25);
});

test('a failed first Start returns to zero', () => {
  const { clock, advance } = clockFixture();
  clock.onStart();
  advance(2000);
  clock.cancelPending();
  advance(1000);
  assert.equal(clock.elapsedSeconds(), 0);
});

test('a failed HTTP response cannot undo an already confirmed running snapshot', () => {
  const { clock, advance } = clockFixture();
  clock.onStart();
  clock.sync(running, [startEvent], 1002);
  clock.cancelPending();
  advance(1000);
  assert.equal(clock.elapsedSeconds(), 3);
});

test('refresh can use wall time when an older server omits server_time', () => {
  const { clock, advance, setWallTime } = clockFixture();
  setWallTime(1_004_000);
  clock.sync(running, [startEvent]);
  assert.equal(clock.elapsedSeconds(), 4);
  advance(1000);
  assert.equal(clock.elapsedSeconds(), 5);
});

import assert from 'node:assert/strict';
import test from 'node:test';
import { buildScene, controlsForState, sceneFromSnapshot, traceDetails, mergeEvents, confidenceLabel, performanceSummary } from '../src/model.js';

test('scene edit preserves identity and metadata, sends only supported pose controls', () => {
  const scene = buildScene({
    table: { position: [0, 0, 0.5], size: [1, 1, 0.05] },
    objects: [{ id: 'cup_1', kind: 'cup', color: 'red', position: ['0.1', '0', '0.7'], size: [0.08, 0.08, 0.12], yaw_deg: '90' }],
  });
  assert.deepEqual(scene.objects[0], { id: 'cup_1', kind: 'cup', color: 'red', position: [0.1, 0, 0.7], size: [0.08, 0.08, 0.12], yaw_deg: 90 });
  assert.throws(() => buildScene({ ...scene, objects: [{ ...scene.objects[0], position: [NaN, 0, 0] }] }), /位置/);
  assert.throws(() => buildScene({ ...scene, objects: [scene.objects[0], scene.objects[0]] }), /重复/);
});

test('draft yaw is computed from backend wxyz quaternion', () => {
  const draft = sceneFromSnapshot({ table: { position: [0, 0, 0.5], size: [1, 1, 0.1] }, objects: [{ id: 'object_1', kind: 'box', color: 'blue', position: [0, 0, 0.6], size: [0.1, 0.1, 0.1], quaternion: [Math.SQRT1_2, 0, 0, Math.SQRT1_2] }] });
  assert.ok(Math.abs(draft.objects[0].yaw_deg - 90) < 0.000001);
});

test('running blocks scene mutations and Start while keeping Reset available', () => {
  assert.deepEqual(controlsForState({ connected: true, busy: false, dirty: false, confirmed: true, status: 'running', instruction: '抓取杯子' }), { editDisabled: true, startDisabled: true, resetDisabled: false, confirmDisabled: true });
  assert.equal(controlsForState({ connected: true, busy: false, dirty: true, confirmed: true, status: 'idle', instruction: '抓取杯子' }).startDisabled, true);
  assert.equal(controlsForState({ connected: true, busy: false, dirty: false, confirmed: true, status: 'idle', instruction: ' ' }).startDisabled, true);
});

test('JEV trace preserves actual state, questions, output, duration and confidence without inventing values', () => {
  const data = { state: { objects: ['a'] }, questions: ['Which object?'], output: { object_id: 'a' }, provider_details: { confidence: 0.83 }, duration_s: 0.42 };
  assert.deepEqual(traceDetails(data), { state: data.state, questions: data.questions, output: data.output, provider_details: data.provider_details, confidence: 0.83, duration_s: 0.42 });
  assert.equal(traceDetails({ output: {} }).confidence, null);
  assert.equal(traceDetails({ output: {} }).duration_s, null);
});

test('incremental events append without duplicates and reset replaces prior run events', () => {
  const first = { id: 1, title: 'task' };
  const second = { id: 2, title: 'object' };
  assert.deepEqual(mergeEvents([first], [first, second], 'append'), [first, second]);
  assert.deepEqual(mergeEvents([first, second], [], 'replace'), []);
  assert.deepEqual(mergeEvents([first], [second]), [second]);
  const history = [first, second];
  assert.equal(mergeEvents(history, [], 'append'), history);
});

test('JEV per-field provider confidence is shown verbatim without an invented aggregate', () => {
  const confidence = { task: 0.97, target: 0.85 };
  assert.deepEqual(traceDetails({ provider_details: { confidence } }).confidence, confidence);
  assert.equal(confidenceLabel(confidence), 'task 97.0% · target 85.0%');
  assert.equal(confidenceLabel(null), '未提供');
});

test('performance summary converts backend seconds to milliseconds and preserves missing values', () => {
  assert.deepEqual(performanceSummary({ decision_count: 12, jev_mean_duration_s: 0.342, jev_total_duration_s: 4.104, run_duration_s: 38.57 }), { count: '12', mean: '342 ms', total: '4.10 s', run: '38.57 s' });
  assert.deepEqual(performanceSummary({ decision_count: 0 }), { count: '0', mean: '—', total: '—', run: '—' });
});

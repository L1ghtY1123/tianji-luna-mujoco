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

test('scene editing preserves fixed CAD bins and the heading of a flat black workpiece', () => {
  const snapshot = { table: { position: [.572, -.026, .532], size: [.6, .8, .08] }, objects: [
    { id: 'bin', kind: 'cad', asset_id: 'blue_box', color: 'blue', fixed: true,
      position: [.481, .573, .571], size: [.41, .31, .145], quaternion: [1, 0, 0, 0] },
    { id: 'black', kind: 'cad', asset_id: 'black_core', color: 'black', pitch_deg: 90,
      position: [.36, -.24, .595], size: [.044, .056, .0745],
      quaternion: [Math.cos(Math.PI / 12) * Math.SQRT1_2, -Math.sin(Math.PI / 12) * Math.SQRT1_2,
        Math.cos(Math.PI / 12) * Math.SQRT1_2, Math.sin(Math.PI / 12) * Math.SQRT1_2] },
  ] };
  const scene = buildScene(sceneFromSnapshot(snapshot));
  assert.equal(scene.objects[0].fixed, true);
  assert.equal(scene.objects[1].pitch_deg, 90);
  assert.ok(Math.abs(scene.objects[1].yaw_deg - 30) < 1e-8);
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
  assert.deepEqual(performanceSummary({ run_duration_s: 38.57 }, [
    { kind: 'visual_decisions_call', data: { elapsed_s: .342 } },
    { kind: 'algorithm_complete', data: { elapsed_s: 99 } },
  ]), { count: '1', mean: '342 ms', total: '0.34 s', run: '38.57 s' });
  assert.deepEqual(performanceSummary({ decision_count: 0 }), { count: '0', mean: '—', total: '—', run: '—' });
});


test('demo scene editing preserves side support and cup appearance', () => {
  const support = [{ id: 'source-table', position: [.39, .66, .56], size: [.34, .32, .08] }];
  const snapshot = { table: { position: [.55, 0, .56], size: [.7, 1, .08] }, support_tables: support,
    objects: [{ id: 'pink', kind: 'cylinder', color: 'pink', visual_style: 'cup', position: [.4, .6, .65], size: [.048, .048, .08] }] };
  const roundtrip = buildScene(sceneFromSnapshot(snapshot));
  assert.deepEqual(roundtrip.support_tables, support);
  assert.equal(roundtrip.objects[0].visual_style, 'cup');
});

test('elliptical workpieces preserve unequal axes and cross section through scene editing', () => {
  const snapshot = { table: { position: [0, 0, .5], size: [1, 1, .1] },
    objects: [{ id: 'pink', kind: 'cylinder', color: 'pink', cross_section: 'ellipse',
      position: [.4, .6, .65], size: [.06, .04, .1] }] };
  const draft = sceneFromSnapshot(snapshot);
  assert.equal(draft.objects[0].cross_section, 'ellipse');
  draft.objects[0].size[0] = .07;
  assert.deepEqual(buildScene(draft).objects[0].size, [.07, .04, .1]);
  assert.equal(buildScene(draft).objects[0].cross_section, 'ellipse');
});

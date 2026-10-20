import assert from 'node:assert/strict';
import test from 'node:test';
import { clampReplayTime, formatReplayTime, synchronizeVideoTimes, artifactReferences, fileUrl, archivedObservations, embeddedRawRgb } from '../src/replay.js';

test('task seek times clamp to the experiment without using simulation time', () => {
  assert.equal(clampReplayTime('12.5', 40), 12.5);
  assert.equal(clampReplayTime(-2, 40), 0);
  assert.equal(clampReplayTime(55, 40), 40);
  assert.equal(clampReplayTime('bad', 40), 0);
  assert.equal(formatReplayTime(3661.25), '01:01:01.250');
  assert.equal(formatReplayTime(59.9998), '00:01:00.000');
});

test('both videos seek together and a short recording holds its last available frame', () => {
  const scene = { readyState: 1, duration: 25, currentTime: 1 };
  const cameras = { readyState: 1, duration: 12, currentTime: 1 };
  const pending = { readyState: 0, duration: NaN, currentTime: 0 };
  synchronizeVideoTimes([scene, cameras, pending], 20);
  assert.equal(scene.currentTime, 20);
  assert.equal(cameras.currentTime, 12);
  assert.equal(pending.currentTime, 0);
  synchronizeVideoTimes([scene, cameras], -1);
  assert.equal(scene.currentTime, 0);
  assert.equal(cameras.currentTime, 0);
});

test('small playback drift is left alone but larger drift is corrected', () => {
  const video = { readyState: 1, duration: 30, currentTime: 4.98 };
  synchronizeVideoTimes([video], 5, 0.12);
  assert.equal(video.currentTime, 4.98);
  synchronizeVideoTimes([video], 9, 0.12);
  assert.equal(video.currentTime, 9);
});

test('array references expose only experiment-relative NPZ downloads', () => {
  const value = { result: { points: { path: 'artifacts/event-7.npz', key: 'points', shape: [4, 3] } }, nested: [{ array_ref: 'artifacts/event-7.npz' }, { path: '../secret.npz' }] };
  assert.deepEqual(artifactReferences(value), ['artifacts/event-7.npz']);
  assert.equal(fileUrl('run /1', 'artifacts/点 云.npz'), '/api/experiments/run%20%2F1/files/artifacts/%E7%82%B9%20%E4%BA%91.npz');
  assert.throws(() => fileUrl('1', '../secret.npz'));
});

test('archived observation revisions retain exact frame, checkpoint and camera bindings', () => {
  const events = [
    { id: 1, kind: 'observation_recorded', t_s: 2, data: { observation_id: 'initial-frame', checkpoint: 'initial', observation_artifact: 'observations/frame-0001.json' } },
    { id: 2, kind: 'observation_annotated', t_s: 3, data: { observation_id: 'initial-frame', object_labels: { observation_id: 'initial-frame', analysis_status: 'unprocessed' }, annotated_rgb: { 3: 'observations/labels-00002-camera-3.svg' } } },
    { id: 3, kind: 'observation_annotated', t_s: 4, data: { observation_id: 'initial-frame', object_labels: { observation_id: 'initial-frame', analysis_status: 'analyzed' }, annotated_rgb: { 3: 'observations/labels-00003-camera-3.svg' } } },
    { id: 4, kind: 'observation_recorded', t_s: 8, data: { observation_id: 'next-frame', checkpoint: 'fine', observation_artifact: 'observations/frame-0002.json' } },
    { id: 5, kind: 'observation_annotated', t_s: 10, data: { observation_id: 'next-frame', object_labels: { observation_id: 'next-frame', analysis_status: 'analyzed' }, annotated_rgb: { 1: 'observations/right.svg', 2: 'observations/left.svg', 3: 'observations/head.svg' } } },
  ];
  const original = structuredClone(events);
  const observations = archivedObservations(events);
  assert.equal(observations.length, 3);
  assert.deepEqual(observations.map(item => [item.observationId, item.checkpoint, item.status, item.time]), [
    ['initial-frame', 'initial', 'unprocessed', 3], ['initial-frame', 'initial', 'analyzed', 4], ['next-frame', 'fine', 'analyzed', 10],
  ]);
  assert.equal(observations[1].artifact, 'observations/frame-0001.json');
  assert.deepEqual(observations[2].cameras.map(item => [item.id, item.path]), [
    ['2', 'observations/left.svg'], ['3', 'observations/head.svg'], ['1', 'observations/right.svg'],
  ]);
  assert.deepEqual(events, original);
});

test('replay refuses cross-frame annotations and untrusted image paths', () => {
  const observations = archivedObservations([
    { id: 1, kind: 'observation_annotated', data: { observation_id: 'new', object_labels: { observation_id: 'old' }, annotated_rgb: { 3: 'observations/old.svg' } } },
    { id: 2, kind: 'observation_annotated', data: { observation_id: 'new', annotated_rgb: { 1: '../private.svg', 2: 'https://remote/image.svg', 3: 'observations/current.svg', 4: 'artifacts/data.json' } } },
  ]);
  assert.equal(observations.length, 1);
  assert.deepEqual(observations[0].cameras.map(item => item.path), ['observations/current.svg']);
  assert.equal(observations[0].checkpoint, null);
  assert.equal(observations[0].status, 'unknown');
});

test('raw view uses only the original PNG embedded in the archived SVG', () => {
  assert.equal(embeddedRawRgb('<svg><image x="0" href="data:image/png;base64,aGVsbG8="/><text>label</text></svg>'), 'data:image/png;base64,aGVsbG8=');
  assert.equal(embeddedRawRgb('<svg><image href="https://remote.test/image.png"/></svg>'), null);
  assert.equal(embeddedRawRgb('<svg><image href="data:text/html;base64,aGVsbG8="/></svg>'), null);
});

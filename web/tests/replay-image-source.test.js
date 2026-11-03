import assert from 'node:assert/strict';
import test from 'node:test';
import * as story from '../src/replay-story.js';
import { extractReplayEvidence } from '../src/replay-evidence.js';

test('question-specific images and text remain inspectable in replay', () => {
  const archive = { endpoint: '/v1/decisions', observation_id: 'current', request: { input: [], questions: [
    { type: 'predicate', name: 'place_result', instructions: [
      { type: 'input_text', text: 'Is the pink object on the white table?' },
      { type: 'input_text', text: 'CURRENT RGB camera 3' },
      { type: 'input_image', image_url: { path: 'decisions/current.jpg' } },
    ] },
  ] }, response: { answers: [{ type: 'predicate', name: 'place_result', probability: .99 }] } };
  const result = extractReplayEvidence(archive);
  assert.equal(result.images.length, 1);
  assert.equal(result.images[0].cameraId, '3');
  assert.equal(result.images[0].questionName, 'place_result');
  assert.equal(result.images[0].sourceObservationId, 'current');
  assert.match(story.modelResult(archive).answers[0].instructions, /pink object.*\nCURRENT RGB/s);
});

test('old holding overlay exposes capture and later event time without asserting image holding', () => {
  const observation = { time: 97.267, event: { id: 111, data: { object_labels: {
    items: [{ id: 'pink', status: 'held' }],
  } } } };
  const info = story.observationImageSource(observation, { captureTime: .517, source: '事件 #4' }, '3');
  assert.match(info, /0.517.*97.267/s);
  assert.match(info, /旧版.*后续.*不代表拍摄时/s);
  assert.match(info, /模型输入/);
});

test('actual model image provenance distinguishes reference proposal, SAM mask box and raw RGB', () => {
  const content = ['initial_reference_annotations', 'current_annotations', 'current_rgb'].flatMap((role, index) => [
    { type: 'input_text', text: JSON.stringify({ image_role: role, camera: '3' }) },
    { type: 'input_image', image_url: { path: `decisions/image-${index}.jpg`, sha256: `sha-${index}` } },
  ]);
  const result = extractReplayEvidence({ endpoint: '/v1/decisions', purpose: 'associate_inventory',
    observation_id: 'frame', request: { input: [{ content }] } });
  assert.match(result.images[0].source, /Responses.*参考框/);
  assert.match(result.images[1].source, /SAM3.*外接矩形/);
  assert.match(result.images[2].source, /无框/);
  assert.equal(result.images[0].imageRole, 'initial_reference_annotations');
  assert.equal(result.images[0].sourceObservationId, null, 'old reference must not inherit the current capture ID');
  assert.equal(result.images[2].sourceObservationId, 'frame');
  assert.equal(result.images[1].cameraId, '3');
});

test('camera explanation separates source, identity and consumer without claiming no detection', () => {
  const value = story.observationImageSections({ event: { id: 26, data: { object_labels: {
    overlay_scope: 'capture_identity_geometry', items: [],
  } } }, time: 101 }, { captureTime: .897 }, '2');
  assert.equal(value.length, 4);
  assert.match(value[1], /不表示没有检测候选/);
  assert.match(value[2], /Decisions/);
  assert.match(value[3], /不是模型输入/);
});

test('plain raw-camera labels cannot inherit the preceding annotated role', () => {
  const result = extractReplayEvidence({ endpoint: '/v1/decisions', request: { input: [{ content: [
    { type: 'input_text', text: JSON.stringify({ image_role: 'initial_reference_annotations', camera: '3', items: [{ id: 'a' }] }) },
    { type: 'input_image', image_url: { path: 'decisions/ref.jpg' } },
    { type: 'input_text', text: 'CURRENT RGB camera 2' },
    { type: 'input_image', image_url: { path: 'decisions/raw.jpg' } },
  ] }] } });
  assert.equal(result.images[1].cameraId, '2');
  assert.equal(result.images[1].imageRole, 'current_rgb');
  assert.deepEqual(result.images[1].annotationItems, []);
});

test('PNG SAM subject views retain MIME, candidate and current capture provenance', () => {
  const result = extractReplayEvidence({ endpoint: '/v1/decisions', observation_id: 'frame', request: { input: [{ content: [
    { type: 'input_text', text: JSON.stringify({ image_role: 'current_candidate_sam_subject',
      candidate: 'candidate-6', camera: '2', observation_id: 'frame', generation: 4 }) },
    { type: 'input_image', image_url: { path: 'decisions/call-image-013.png', sha256: 'exact-png-hash', media_type: 'image/png' } },
  ] }] } });
  const image = result.images[0];
  assert.equal(image.path, 'decisions/call-image-013.png');
  assert.equal(image.media_type, 'image/png');
  assert.equal(image.sha256, 'exact-png-hash');
  assert.equal(image.frame, 'current');
  assert.equal(image.sourceObservationId, 'frame');
  assert.equal(image.cameraId, '2');
  assert.match(image.label, /SAM.*candidate-6/);
  assert.match(image.source, /SAM.*轮廓.*置灰/);
  assert.match(image.source, /不证明身份/);
});

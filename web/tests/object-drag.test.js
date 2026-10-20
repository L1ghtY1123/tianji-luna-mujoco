import assert from 'node:assert/strict';
import test from 'node:test';
import { Ray, Vector3 } from 'three';
import { createDragProjection, createObjectDrag, objectPreviewPosition } from '../src/object-drag.js';

function ray(origin, direction) {
  return new Ray(new Vector3(...origin), new Vector3(...direction).normalize());
}

function near(actual, expected) {
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-12));
}

test('horizontal drag preserves the picked offset and the initial height', () => {
  const drag = createDragProjection({ position: [1, 2, 0.5], ray: ray([1.2, 2.1, 4], [0, 0, -1]) });
  near(drag(ray([1.2, 2.1, 4], [0, 0, -1]), 0), [1, 2, 0.5]);
  near(drag(ray([2.2, 3.1, 4], [0, 0, -1]), 0), [2, 3, 0.5]);
});

test('horizontal drag ignores a ray parallel to the starting plane', () => {
  const drag = createDragProjection({ position: [0, 0, 0.5], ray: ray([0, 0, 4], [0, 0, -1]) });
  assert.equal(drag(ray([0, 0, 4], [1, 0, 0])), null);
  assert.equal(createDragProjection({ position: [0, 0, 0.5], ray: ray([0, 0, 4], [1, 0, 0]) }), null);
});

test('Shift drag changes only height with upward pointer motion increasing Z', () => {
  const drag = createDragProjection({ position: [1, 2, 0.5], vertical: true, screenY: 100, metersPerPixel: 0.01 });
  assert.deepEqual(drag(null, 80), [1, 2, 0.7]);
  assert.deepEqual(drag(null, 120), [1, 2, 0.3]);
});

test('release updates a local draft once and immediately allows another drag', () => {
  const changed = [];
  const drag = createObjectDrag({ onDragEnd: (id, position) => {
    changed.push({ id, position });
  } });
  drag.setEnabled(true);
  assert.equal(drag.begin('box', [1, 2, 3], 4), true);
  drag.move([2, 3, 3]);
  assert.deepEqual(changed, []);
  drag.finish();
  drag.finish();
  assert.equal(drag.active, null);
  assert.deepEqual(changed, [{ id: 'box', position: [2, 3, 3] }]);
  assert.equal(drag.begin('other', [0, 0, 0], 4), true);
});

test('clicking without moving selects without submitting a pose change', async () => {
  let submissions = 0;
  const drag = createObjectDrag({ onDragEnd: () => { submissions += 1; } });
  drag.setEnabled(true);
  drag.begin('box', [1, 2, 3], 4);
  await drag.finish();
  assert.equal(submissions, 0);
  assert.equal(drag.active, null);
});

test('same-generation snapshots preserve preview and a new generation cancels it', () => {
  const cancelled = [];
  const drag = createObjectDrag({ onDragCancel: id => cancelled.push(id) });
  drag.setEnabled(true);
  drag.begin('box', [1, 2, 3], 4);
  drag.move([4, 5, 3]);
  drag.updateSnapshot({ generation: 4, objects: [{ id: 'box', position: [9, 9, 9] }] });
  assert.deepEqual(drag.active.position, [4, 5, 3]);
  drag.updateSnapshot({ generation: 5, objects: [{ id: 'box', position: [1, 2, 3] }] });
  assert.equal(drag.active, null);
  assert.deepEqual(cancelled, ['box']);
});

test('cancelling an unreleased drag keeps all previous draft changes', () => {
  const cancelled = [];
  const draft = { objects: [{ id: 'box', position: [1, 2, 3] }] };
  const drag = createObjectDrag({
    onDragEnd: (id, position) => { draft.objects.find(object => object.id === id).position = position; },
    onDragCancel: id => cancelled.push(id),
  });
  drag.setEnabled(true);
  drag.begin('box', [1, 2, 3], 4);
  drag.move([2, 2, 3]);
  drag.finish();
  drag.begin('box', [2, 2, 3], 4);
  drag.move([9, 9, 9]);
  assert.deepEqual(objectPreviewPosition('box', [0, 0, 0], draft, drag.active), [9, 9, 9]);
  drag.cancel();
  assert.equal(drag.active, null);
  assert.deepEqual(cancelled, ['box']);
  assert.deepEqual(draft.objects[0].position, [2, 2, 3]);
  assert.deepEqual(objectPreviewPosition('box', [0, 0, 0], draft, drag.active), [2, 2, 3]);
});

test('draft positions override live snapshots until cleared, without hiding robot motion', () => {
  const draft = { objects: [{ id: 'box', position: ['1', '2', '.8'] }] };
  assert.deepEqual(objectPreviewPosition('box', [4, 5, 6], draft), [1, 2, .8]);
  assert.deepEqual(objectPreviewPosition(null, [4, 5, 6], draft), [4, 5, 6]);
  assert.deepEqual(objectPreviewPosition('box', [4, 5, 6], null), [4, 5, 6]);
  draft.objects[0].position = [1, NaN, .8];
  assert.deepEqual(objectPreviewPosition('box', [4, 5, 6], draft), [4, 5, 6]);
});

test('disabled interaction and missing objects cancel an active pointer drag', () => {
  const drag = createObjectDrag();
  assert.equal(drag.begin('box', [1, 2, 3], 4), false);
  drag.setEnabled(true);
  drag.begin('box', [1, 2, 3], 4);
  drag.setEnabled(false);
  assert.equal(drag.active, null);
  drag.setEnabled(true);
  drag.begin('box', [1, 2, 3], 4);
  drag.updateSnapshot({ generation: 4, objects: [] });
  assert.equal(drag.active, null);
});

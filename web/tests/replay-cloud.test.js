import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { candidateLineGeometry, replayBounds } from '../src/replay-cloud.js';

test('candidate tool glyph uses archived world rotation and translation and skips invalid poses', () => {
  const geometry = candidateLineGeometry([
    {
      status: 'passed',
      matrix: [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
    },
    { status: 'failed', matrix: [[NaN]] },
  ]);
  try {
    const position = geometry.getAttribute('position');
    assert.equal(position.count, 14);
    const actual = [position.getX(0), position.getY(0), position.getZ(0)];
    const expected = [1, 1.975, 2.965];
    expected.forEach((value, axis) => assert.ok(Math.abs(actual[axis] - value) < 1e-6));
    assert.ok(Math.abs(position.getZ(7) - 3.025) < 1e-6);
    // Local X rotates onto world +Y, local Y onto world -X; Z stays +Z.
    for (const [index, axis, direction] of [[9, 1, 1], [11, 0, -1], [13, 2, 1]]) {
      const origin = [position.getX(index - 1), position.getY(index - 1), position.getZ(index - 1)];
      const end = [position.getX(index), position.getY(index), position.getZ(index)];
      assert.deepEqual(origin, [1, 2, 3]);
      assert.ok((end[axis] - origin[axis]) * direction > 0);
      for (let other = 0; other < 3; other += 1) {
        if (other !== axis) assert.equal(end[other], origin[other]);
      }
    }
  } finally {
    geometry.dispose();
  }
});

test('camera fit includes the recorded tool mount above the object and the complete cloud', () => {
  // Real 30dd replay: the tool mount is 17.5 cm above the measured/completed top.
  const cloudGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(.17191, .62000, .411998),
    new THREE.Vector3(.22861, .65856, .511999),
  ]);
  const linesGeometry = candidateLineGeometry([{
    status: 'passed',
    matrix: [[-.5, -.866025388, 0, .200300896], [-.866025388, .5, 0, .639133108],
      [0, 0, -1, .687132053], [0, 0, 0, 1]],
  }]);
  try {
    const cloud = new THREE.Points(cloudGeometry), lines = new THREE.LineSegments(linesGeometry);
    const bounds = replayBounds(cloud, lines);
    for (const geometry of [cloudGeometry, linesGeometry]) {
      const points = geometry.getAttribute('position');
      for (let i = 0; i < points.count; i += 1) {
        assert.ok(bounds.containsPoint(new THREE.Vector3().fromBufferAttribute(points, i)));
      }
    }
    assert.ok(bounds.max.z > .72, 'tool glyph must not be cropped above the 10 cm object');
    assert.ok(bounds.min.z < .412, 'bottom of the completed object must remain visible');
  } finally {
    cloudGeometry.dispose();
    linesGeometry.dispose();
  }
});

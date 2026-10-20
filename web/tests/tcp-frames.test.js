import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { createTcpFrames } from '../src/tcp-frames.js';

test('TCP axes follow the supplied front frame without a duplicate offset', () => {
  const frames = createTcpFrames();
  assert.equal(frames.root.visible, false);
  frames.setVisible(true);
  frames.update({ arms: { A: { tcp_front: [[0, 0, 1, .4], [0, 1, 0, .2], [-1, 0, 0, .6], [0, 0, 0, 1]] } } });
  frames.root.updateMatrixWorld(true);
  const frame = frames.root.children[0];
  assert.deepEqual(new THREE.Vector3().setFromMatrixPosition(frame.matrixWorld).toArray(), [.4, .2, .6]);
  assert.deepEqual(new THREE.Vector3(0, 0, 1).transformDirection(frame.matrixWorld).toArray(), [1, 0, 0]);
  assert.equal(frames.root.children[1].visible, false);
  frames.update({ arms: {} });
  assert.equal(frame.visible, false);
  frames.setVisible(false);
  assert.equal(frames.root.visible, false);
});

import assert from 'node:assert/strict';
import test from 'node:test';
import { createWorkspaceLayer } from '../src/workspace-layer.js';

const payload = () => ({ source: 'offline_single_arm_workspace', frame: 'Link_Base',
  surface_policy: 'exposed_voxel_faces', arms: ['A', 'B'].map(arm => ({
    arm, origin_m: [.4, .1, .6], resolution_m: .01, shape: [1, 1, 1], reachable_points: 1,
    faces: [0, 1, 2].flatMap(axis => [-1, 1].map(sign => ({ axis, sign, indices: [0] }))),
  })),
});

test('renders actual voxel faces in meters with independent visibility and opacity', () => {
  const layer = createWorkspaceLayer(payload());
  assert.equal(layer.object.children.length, 2);
  const [left, right] = layer.object.children;
  assert.equal(left.geometry.getAttribute('position').count, 36);
  left.geometry.computeBoundingBox();
  for (const [axis, value] of ['x', 'y', 'z'].map((axis, i) => [axis, [.4, .1, .6][i]])) {
    assert.ok(Math.abs(left.geometry.boundingBox.min[axis] - (value - .005)) < 1e-6);
    assert.ok(Math.abs(left.geometry.boundingBox.max[axis] - (value + .005)) < 1e-6);
  }
  const positions = left.geometry.getAttribute('position');
  const normals = left.geometry.getAttribute('normal');
  for (let i = 0; i < positions.count; i++) {
    const dot = (positions.getX(i) - .4) * normals.getX(i)
      + (positions.getY(i) - .1) * normals.getY(i) + (positions.getZ(i) - .6) * normals.getZ(i);
    assert.ok(dot > 0, 'Every face must point out of the reachable voxel');
  }
  assert.notEqual(left.material.color.getHex(), right.material.color.getHex());
  layer.setVisible('A', false);
  assert.equal(left.visible, false);
  assert.equal(right.visible, true);
  layer.setOpacity(.1);
  assert.equal(right.material.opacity, .1);
  let disposed = 0;
  for (const mesh of layer.object.children) {
    mesh.geometry.addEventListener('dispose', () => disposed++);
    mesh.material.addEventListener('dispose', () => disposed++);
  }
  layer.dispose();
  assert.equal(disposed, 4);
});

test('rejects wrong coordinate frame or invalid cell index', () => {
  const frame = payload(); frame.frame = 'camera';
  assert.throws(() => createWorkspaceLayer(frame), /工作空间/);
  const index = payload(); index.arms[0].faces[0].indices = [1];
  assert.throws(() => createWorkspaceLayer(index), /工作空间/);
});

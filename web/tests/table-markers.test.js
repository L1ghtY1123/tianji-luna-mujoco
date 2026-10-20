import assert from 'node:assert/strict';
import test from 'node:test';
import { Mesh } from 'three';
import { buildScene } from '../src/model.js';
import { applyScenePreview } from '../src/scene-preview.js';

function draft() {
  return { table: { position: [.55, 0, .56], size: [.7, 1, .08] },
    objects: [{ id: 'region', kind: 'marker', color: 'yellow',
      position: [.55, -.25, 8], size: [.16, .12, .5], yaw_deg: 20 }] };
}

test('marker scene input is constrained to a flat tabletop region', () => {
  const item = buildScene(draft()).objects[0];
  assert.ok(Math.abs(item.position[2] - .6001) < 1e-10);
  assert.deepEqual(item.size, [.16, .12, .0002]);
});

test('marker colors are unique and their rotated extent stays on the table', () => {
  const scene = draft();
  scene.objects.push({ ...scene.objects[0], id: 'duplicate' });
  assert.throws(() => buildScene(scene), /颜色/);
  scene.objects.pop();
  scene.objects[0].position[0] = 1;
  assert.throws(() => buildScene(scene), /桌面/);
  scene.objects[0].position[0] = .55;
  scene.objects[0].color = 'gray';
  assert.throws(() => buildScene(scene), /颜色/);
});

test('marker preview follows table height and ignores vertical dragging', () => {
  const scene = draft();
  scene.table.position[2] = .7;
  const mesh = new Mesh();
  const definition = { name: 'object_geom_region', object_id: 'region', object_kind: 'marker',
    type: 'box', size: [.08, .06, .0001] };
  const pose = { position: [.55, -.25, .6001], quaternion: [1, 0, 0, 0] };
  applyScenePreview(mesh, definition, pose, scene, { id: 'region', position: [.6, -.1, 5] });
  assert.ok(Math.abs(mesh.position.z - .7401) < 1e-10);
  assert.deepEqual(mesh.position.toArray().slice(0, 2), [.6, -.1]);
  assert.equal(mesh.scale.z, 1);
});

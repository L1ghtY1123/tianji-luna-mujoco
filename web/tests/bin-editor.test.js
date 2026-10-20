import assert from 'node:assert/strict';
import test from 'node:test';
import { BoxGeometry, Mesh, Vector3 } from 'three';
import { buildScene, sceneFromSnapshot } from '../src/model.js';
import { applyScenePreview } from '../src/scene-preview.js';
import * as drag from '../src/object-drag.js';

const bin = { id: 'bin__blue', kind: 'bin', color: 'blue', position: [.5, .2, .66],
  size: [.24, .18, .12], yaw_deg: 0, wall_thickness_m: .01, mass_kg: .4 };
const table = { position: [.55, 0, .56], size: [.7, 1, .08] };

test('bin scene payload and snapshot conversion preserve wall thickness and mass', () => {
  const value = sceneFromSnapshot({ table, objects: [{ ...bin, quaternion: [1, 0, 0, 0] }] });
  assert.equal(value.objects[0].wall_thickness_m, .01);
  assert.equal(value.objects[0].mass_kg, .4);
  assert.deepEqual(buildScene(value).objects[0], bin);
});

for (const changes of [{ wall_thickness_m: 0 }, { wall_thickness_m: .09 }, { size: [.24, .18, .01] }, { mass_kg: 0 }]) {
  test(`invalid bin cavity or mass ${JSON.stringify(changes)} cannot be applied`, () => {
    assert.throws(() => buildScene({ table, objects: [{ ...bin, ...changes }] }), /物料盒/);
  });
}

test('all bin pieces share the object center while wall thickness, dimensions and yaw preview independently', () => {
  const definition = { name: 'object_geom_bin__blue__wall_x_plus', object_id: 'bin__blue', object_kind: 'bin',
    component: 'wall_x_plus', type: 'box', size: [.005, .09, .055], local_position_m: [.115, 0, .005] };
  const mesh = new Mesh(new BoxGeometry(.01, .18, .11));
  const pose = { position: [.615, .2, .665], quaternion: [1, 0, 0, 0] };
  const draft = { objects: [{ ...bin, position: [.6, .3, .7], size: [.3, .22, .16], wall_thickness_m: .02, yaw_deg: 90 }] };
  applyScenePreview(mesh, definition, pose, draft);
  assert.ok(mesh.position.distanceTo(new Vector3(.6, .44, .71)) < 1e-10);
  assert.deepEqual(mesh.scale.toArray(), [2, .22 / .18, .14 / .11]);
  assert.ok(mesh.localToWorld(new Vector3(.005, 0, 0)).distanceTo(new Vector3(.6, .45, .71)) < 1e-10);
  applyScenePreview(mesh, definition, pose, draft, { id: 'bin__blue', position: [.7, .4, .8] });
  assert.ok(mesh.position.distanceTo(new Vector3(.7, .54, .81)) < 1e-10);
  applyScenePreview(mesh, definition, pose, null);
  assert.deepEqual(mesh.position.toArray(), pose.position);
  assert.deepEqual(mesh.scale.toArray(), [1, 1, 1]);
});

test('grabbing a bin wall starts dragging its rigid body center rather than the wall center', () => {
  assert.equal(typeof drag.editableObjectPosition, 'function');
  const snapshot = { objects: [bin] };
  assert.deepEqual(drag.editableObjectPosition(bin.id, [.615, .2, .665], snapshot, null), [.5, .2, .66]);
  assert.deepEqual(drag.editableObjectPosition(bin.id, [.615, .2, .665], snapshot,
    { objects: [{ ...bin, position: [.6, .3, .7] }] }), [.6, .3, .7]);
});

test('dragging an applied bin without a dirty draft preserves each component offset', () => {
  const definition = { name: 'object_geom_bin__blue__wall_x_plus', object_id: 'bin__blue', object_kind: 'bin',
    component: 'wall_x_plus', type: 'box', size: [.005, .09, .055], local_position_m: [.115, 0, .005] };
  const mesh = new Mesh(new BoxGeometry(.01, .18, .11));
  applyScenePreview(mesh, definition, { position: [.615, .2, .665], quaternion: [1, 0, 0, 0] }, null,
    { id: 'bin__blue', position: [.7, .4, .8] });
  assert.ok(mesh.position.distanceTo(new Vector3(.815, .4, .805)) < 1e-10);
  assert.deepEqual(mesh.scale.toArray(), [1, 1, 1]);
});

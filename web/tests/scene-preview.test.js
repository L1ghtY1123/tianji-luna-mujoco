import assert from 'node:assert/strict';
import test from 'node:test';
import { BoxGeometry, Mesh, Vector3 } from 'three';
import { applyScenePreview } from '../src/scene-preview.js';

const pose = { position: [.4, .1, .62], quaternion: [1, 0, 0, 0] };
const box = { name: 'object_geom_block', type: 'box', size: [.02, .02, .02] };
const draft = () => ({ objects: [{ id: 'block', kind: 'box', position: [.5, .2, .7], size: [.08, .12, .16], yaw_deg: 90 }] });

function near(actual, expected) {
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-10, `${actual} != ${expected}`));
}

test('editing a flat CAD workpiece keeps its long axis horizontal', () => {
  const mesh = new Mesh();
  const definition = { name: 'black_mesh', object_id: 'block', object_kind: 'cad',
    local_position_m: [0, 0, 0], local_quaternion: [1, 0, 0, 0] };
  const value = draft();
  Object.assign(value.objects[0], { kind: 'cad', yaw_deg: 30, pitch_deg: 90 });
  applyScenePreview(mesh, definition, pose, value);
  near(new Vector3(0, 0, 1).applyQuaternion(mesh.quaternion).toArray(), [Math.sqrt(3) / 2, .5, 0]);
});

test('yaw and full dimensions immediately change the displayed mesh in world coordinates', () => {
  const mesh = new Mesh(new BoxGeometry(.04, .04, .04));
  assert.equal(applyScenePreview(mesh, box, pose, draft()), true);
  near(mesh.position.toArray(), [.5, .2, .7]);
  near(mesh.scale.toArray(), [2, 3, 4]);
  near(mesh.localToWorld(new Vector3(.02, 0, 0)).toArray(), [.5, .24, .7]);
  near(mesh.localToWorld(new Vector3(0, .02, 0)).toArray(), [.44, .2, .7]);
  assert.equal(applyScenePreview(mesh, box, pose, draft()), false);
});

test('live snapshots cannot overwrite preview yaw and size, while clearing the draft restores physics', () => {
  const mesh = new Mesh();
  const movedPose = { position: [1, 2, 3], quaternion: [.5, .5, .5, .5] };
  applyScenePreview(mesh, box, movedPose, draft());
  near(mesh.quaternion.toArray(), [0, 0, Math.SQRT1_2, Math.SQRT1_2]);
  near(mesh.scale.toArray(), [2, 3, 4]);
  applyScenePreview(mesh, box, movedPose, null);
  near(mesh.position.toArray(), movedPose.position);
  near(mesh.quaternion.toArray(), [.5, .5, .5, .5]);
  near(mesh.scale.toArray(), [1, 1, 1]);
});

test('dragging uses the active position and keeps the previously edited yaw and dimensions', () => {
  const mesh = new Mesh();
  applyScenePreview(mesh, box, pose, draft(), { id: 'block', position: [.6, .3, .8] });
  near(mesh.position.toArray(), [.6, .3, .8]);
  near(mesh.scale.toArray(), [2, 3, 4]);
  near(mesh.quaternion.toArray(), [0, 0, Math.SQRT1_2, Math.SQRT1_2]);
});

test('cylinder and sphere scale from MuJoCo radius and half-height conventions', () => {
  const cylinder = new Mesh();
  const value = draft();
  value.objects[0].size = [.08, .08, .3];
  applyScenePreview(cylinder, { ...box, type: 'cylinder', size: [.02, .05, 0] }, pose, value);
  near(cylinder.scale.toArray(), [2, 2, 3]);
  const sphere = new Mesh();
  value.objects[0].size = [.12, .12, .12];
  applyScenePreview(sphere, { ...box, type: 'sphere', size: [.02, 0, 0] }, pose, value);
  near(sphere.scale.toArray(), [3, 3, 3]);
});

test('table dimensions and position also preview; robot links retain live transforms', () => {
  const value = draft();
  value.table = { position: [.55, .1, .56], size: [.8, 1.2, .1] };
  const table = new Mesh();
  applyScenePreview(table, { name: 'table', type: 'box', size: [.4, .4, .04] }, pose, value);
  near(table.position.toArray(), [.55, .1, .56]);
  near(table.scale.toArray(), [1, 1.5, 1.25]);
  const robot = new Mesh();
  applyScenePreview(robot, { name: 'arm_link', type: 'mesh' }, pose, value);
  near(robot.position.toArray(), pose.position);
  near(robot.scale.toArray(), [1, 1, 1]);
});

test('invalid temporary size or yaw inputs never introduce invalid transforms', () => {
  const mesh = new Mesh();
  const value = draft();
  value.objects[0].size = [.08, 0, NaN];
  value.objects[0].yaw_deg = '';
  applyScenePreview(mesh, box, pose, value);
  near(mesh.scale.toArray(), [1, 1, 1]);
  near(mesh.quaternion.toArray(), [0, 0, 0, 1]);
});

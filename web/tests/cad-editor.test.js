import assert from 'node:assert/strict';
import test from 'node:test';
import { Mesh, Quaternion, Vector3 } from 'three';
import * as model from '../src/model.js';
import { applyScenePreview } from '../src/scene-preview.js';

const table = { position: [.46, 0, .52], size: [.6, .8, .08] };
const asset = { id: 'pink_wrapped', label: '粉色包装件', size_m: [.06, .04, .1], default_color: 'pink' };

test('CAD scene roundtrip preserves asset identity and an arbitrary hex color', () => {
  const item = { id: 'part_1', kind: 'cad', asset_id: asset.id, color: '#d153a8',
    position: [.4, .1, .7], size: asset.size_m, yaw_deg: 0, mass_kg: .123 };
  const draft = model.sceneFromSnapshot({ table, objects: [item] });
  assert.equal(draft.objects[0].asset_id, asset.id);
  assert.deepEqual(model.buildScene(draft).objects[0], item);
  delete draft.objects[0].asset_id;
  assert.throws(() => model.buildScene(draft), /CAD/);
});

test('adding CAD instances has no old 30-instance cap and always assigns unique scene IDs', () => {
  assert.equal(typeof model.createCadObject, 'function');
  const draft = { table, objects: [] };
  for (let index = 0; index < 40; index++) draft.objects.push(model.createCadObject(draft, asset));
  assert.equal(new Set(draft.objects.map(item => item.id)).size, 40);
  assert.equal(model.buildScene(draft).objects.length, 40);
  assert.ok(draft.objects.every(item => item.asset_id === asset.id && item.kind === 'cad'));
  assert.deepEqual(draft.objects[0].size, asset.size_m);
  assert.notEqual(draft.objects[0].size, asset.size_m);
});

test('CAD preview composes the body edit with MuJoCo mesh local translation and rotation', () => {
  const local = new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), Math.PI / 2);
  const definition = { object_id: 'part_1', object_kind: 'cad', type: 'mesh',
    local_position_m: [.01, .02, .03], local_quaternion: [local.w, local.x, local.y, local.z] };
  const pose = { position: [.41, .12, .73], quaternion: definition.local_quaternion };
  const draft = { objects: [{ id: 'part_1', kind: 'cad', position: [.5, .2, .8], yaw_deg: 90 }] };
  const mesh = new Mesh();
  applyScenePreview(mesh, definition, pose, draft);
  assert.ok(mesh.position.distanceTo(new Vector3(.48, .21, .83)) < 1e-10);
  const expected = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 2).multiply(local);
  assert.ok(mesh.quaternion.angleTo(expected) < 1e-7);
  applyScenePreview(mesh, definition, pose, null, { id: 'part_1', position: [.6, .3, .9] });
  assert.ok(mesh.position.distanceTo(new Vector3(.61, .32, .93)) < 1e-10);
  assert.ok(mesh.quaternion.angleTo(local) < 1e-7);
});


test('CAD catalog recovers after a transient restart and bounds persistent failures', async () => {
  const { loadCadAssets } = await import('../src/cad-assets.js');
  let calls = 0;
  const waits = [];
  const result = await loadCadAssets(async () => {
    if (++calls === 1) throw new Error('connection reset');
    return { assets: [asset] };
  }, async delay => waits.push(delay));
  assert.deepEqual(result, [asset]);
  assert.equal(calls, 2);
  assert.deepEqual(waits, [1000]);
  calls = 0;
  await assert.rejects(loadCadAssets(async () => { calls++; throw new Error('offline'); }, async () => {}), /offline/);
  assert.equal(calls, 3);
});

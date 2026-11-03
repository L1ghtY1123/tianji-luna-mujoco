import test from 'node:test';
import assert from 'node:assert/strict';
import {collisionStages, loadCollisionStage} from '../src/replay-collision.js';

test('stage loader binds historical scene and reuses only frozen CAD geometry', async () => {
  const asset = `collision/assets/v1-${'a'.repeat(64)}.json`;
  const event = {kind: 'collision_scene_archived', data: {scene_token: 'old', observation_id: 'capture', snapshot_artifact: 'collision/stage.json'}};
  const scene = {status: 'available', scene_token: 'old', observation_id: 'capture',
    mesh_layers: {a: {pose_wxyz: [1], mesh_artifact: asset}, b: {pose_wxyz: [2], mesh_artifact: asset}}};
  const calls = [];
  const loaded = await loadCollisionStage(event, async path => {
    calls.push(path);
    return path === event.data.snapshot_artifact ? scene : {vertices_local_m: [[0, 0, 0]], faces: []};
  });
  assert.deepEqual(calls, ['collision/stage.json', asset]);
  assert.deepEqual(loaded.mesh_layers.b.pose_wxyz, [2]);
  assert.equal(scene.mesh_layers.b.vertices_local_m, undefined);
  assert.equal(collisionStages([event, {kind: 'build_scene'}]).length, 1);
  await assert.rejects(loadCollisionStage(event, async () => ({...scene, scene_token: 'latest'})), /版本不一致/);
});

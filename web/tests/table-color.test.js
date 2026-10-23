import assert from 'node:assert/strict';
import test from 'node:test';
import { buildScene, sceneFromSnapshot } from '../src/model.js';

test('table color survives snapshot editing and submission', () => {
  const snapshot = { table: { position: [.5, 0, .5], size: [.6, .8, .08], color: '#426f81' }, objects: [] };
  const draft = sceneFromSnapshot(snapshot);
  assert.equal(draft.table.color, '#426f81');
  draft.table.color = '#ffffff';
  assert.equal(buildScene(draft).table.color, '#ffffff');
  assert.equal(snapshot.table.color, '#426f81');
});

test('old scenes without a table color default to white', () => {
  const scene = { table: { position: [.5, 0, .5], size: [.6, .8, .08] }, objects: [] };
  assert.equal(sceneFromSnapshot(scene).table.color, 'white');
  assert.equal(buildScene(scene).table.color, 'white');
});

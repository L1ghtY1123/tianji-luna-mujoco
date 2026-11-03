import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { labelsForFrame, createObjectList } from '../src/object-list.js';

test('live inventory ignores historical ledgers instead of reconstructing transport stages', () => {
  const container = { hidden: false, children: ['stale'], replaceChildren() { this.children = []; } };
  createObjectList(container, { document: {} }).render({ process_records: { old: {} },
    objects: [{ id: 'part', stage_index: 1 }], stage_arms: { '1': 'B' } });
  assert.equal(container.hidden, true);
  assert.deepEqual(container.children, []);
});

test('inventory has a distinct DOM identity from the editable scene object list', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(ids.includes('task-object-list'));
});

test('labels are accepted only for their matching task frame', () => {
  const labels = { observation_id: 'task-1', generation: 2, items: [{id: 'source', display_id: 1, x: .2, y: .3}] };
  assert.equal(labelsForFrame(labels, 'task-1', 2).length, 1);
  assert.deepEqual(labelsForFrame(labels, 'preview-1', 2), []);
  assert.deepEqual(labelsForFrame(labels, 'task-1', 3), []);
});

test('obstacles and destinations have labels on their own camera without task numbers', () => {
  const labels = {observation_id: 'frame', generation: 1, items: [
    {id: 'obstacle', camera_id: '1', display_id: null, x: .2, y: .3},
    {id: 'bin', camera_id: '2', display_id: null, x: .4, y: .5}]};
  assert.deepEqual(labelsForFrame(labels, 'frame', 1, '1').map(item => item.id), ['obstacle']);
  assert.deepEqual(labelsForFrame(labels, 'frame', 1, '3'), []);
});

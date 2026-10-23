import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { objectRows, labelsForFrame, createObjectList } from '../src/object-list.js';

test('process inventory separates pending sources from products and completed jobs', () => {
  const element = tag => ({ tag, children: [], append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; } });
  const container = element('section');
  const list = createObjectList(container, { document: { createElement: element } });
  list.render({ all_entities: { pink_a: { id: 'pink_a', color: 'pink' }, pink_b: { id: 'pink_b', color: 'pink' },
      black_a: { id: 'black_a', color: 'black' } },
    process_order: ['job_a', 'job_b'], pending_unpack_ids: ['pink_b'], processing_ids: [],
    completed_process_ids: ['job_a'], product_ids: ['black_a'],
    process_records: {
      job_a: { id: 'job_a', display_id: 1, input_display_id: 1, input_entity_id: 'pink_a', output_entity_id: 'black_a', status: 'completed', phase: 'completed' },
      job_b: { id: 'job_b', display_id: 2, input_display_id: 2, input_entity_id: 'pink_b', output_entity_id: null, status: 'pending', phase: 'wrapped_source' },
    },
  });
  const text = node => [node.textContent ?? '', ...node.children.map(text)].join(' ');
  const result = text(container);
  assert.match(result, /当前场景实体.*3/);
  assert.match(result, /初始物体档案/);
  assert.match(result, /已拆包 · 待投放/);
  assert.match(result, /待拆包.*1/);
  assert.match(result, /已完成.*1/);
  assert.match(result, /#2.*pink_b/);
  assert.match(result, /pink_a.*black_a/);
  assert.doesNotMatch(result, /搬运物体清单|同一物体/);
});

test('released objects stay pending until visual destination confirmation', () => {
  const rows = objectRows({objects: [{id: 'red', status: 'selected',
    transport_status: 'awaiting_visual_confirmation', destination_visual_evidence: {status: 'elsewhere'}}]});
  assert.equal(rows[0].status, '已释放 · 待视觉确认');
  assert.equal(rows[0].destinationVisual, '视觉显示尚未到达目标区域');
  assert.equal(objectRows({objects: [{id: 'green', status: 'completed',
    destination_visual_evidence: {status: 'at_destination'}}]})[0].destinationVisual, '去向已由视觉核对');
});

test('inventory has a distinct DOM identity from the editable scene object list', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(ids.includes('task-object-list'));
});

test('completed and unseen are separate states and numbers never compact', () => {
  const rows = objectRows({ objects: [
    { id: 'b', display_id: 2, status: 'observed', visibility: 'visible' },
    { id: 'a', display_id: 1, status: 'completed', visibility: 'unseen' },
    { id: 'c', display_id: 3, status: 'deferred', visibility: 'unseen' },
  ] });
  assert.deepEqual(rows.map(row => row.number), [1, 2, 3]);
  assert.equal(rows[0].status, '已完成');
  assert.equal(rows[0].visibility, '本帧未见');
});

test('old experiment remains readable without a ledger', () => {
  assert.deepEqual(objectRows(null), []);
  assert.equal(objectRows({ completed: [{id: 'a', status: 'completed'}] })[0].number, '—');
});

test('non-task IDs stay out of transport rows and current target keeps region and slot', () => {
  const rows = objectRows({membership: {part: 'required', obstacle: 'excluded'},
    stage_arms: {'1': 'B'}, placement_regions: [{destination_id: 'bin', region_id: 'region:bin'}],
    objects: [{id: 'part', display_id: 3, stage_index: 1, destination_id: 'bin'}, {id: 'obstacle'}]});
  assert.deepEqual(rows.map(row => row.id), ['part']);
  assert.equal(rows[0].number, 3);
  assert.equal(rows[0].stage, 2);
  assert.equal(rows[0].arm, 'B');
  assert.equal(rows[0].regionId, 'region:bin');
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

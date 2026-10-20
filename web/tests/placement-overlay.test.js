import assert from 'node:assert/strict';
import test from 'node:test';
import { regionsForFrame, appendPlacementOverlay } from '../src/placement-overlay.js';

const cell = {camera_id: '3', region_id: 'region:bin', slot_id: 2,
  source: 'measured_floor', source_observation_id: 'first', layout_revision: 'revision',
  center: [.5, .5], polygon: [[.2, .2], [.8, .2], [.8, .8], [.2, .8]]};
const labels = {observation_id: 'current', generation: 1, regions: [cell]};

test('regions render only on their current captured camera and generation', () => {
  assert.deepEqual(regionsForFrame(labels, 'current', 1, '3'), [cell]);
  assert.deepEqual(regionsForFrame(labels, 'old', 1, '3'), []);
  assert.deepEqual(regionsForFrame(labels, 'current', 2, '3'), []);
  assert.deepEqual(regionsForFrame(labels, 'current', 1, '1'), []);
});

test('overlay shows frozen slot and provenance without changing the image', () => {
  const node = name => ({name, children: [], attributes: {},
    append(...items) { this.children.push(...items); },
    setAttribute(key, value) { this.attributes[key] = value; }});
  const surface = node('surface');
  const image = node('image');
  surface.append(image);
  appendPlacementOverlay(surface, [cell], {createElementNS: (_, name) => node(name)});
  assert.equal(surface.children[0], image);
  const svg = surface.children[1];
  assert.equal(svg.name, 'svg');
  assert.equal(svg.children.find(item => item.name === 'text').textContent, '区2');
  assert.match(svg.children.find(item => item.name === 'title').textContent, /region:bin.*first/);
});

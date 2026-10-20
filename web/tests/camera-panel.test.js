import assert from 'node:assert/strict';
import test from 'node:test';

import { createCameraPanel } from '../src/camera-panel.js';

function harness(initialMode = "task", { delayImages = false } = {}) {
  const decodes = [];
  function node(tagName) {
    return { tagName, children: [], textContent: '', open: true, listeners: {}, style: {},
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      addEventListener(kind, callback) { this.listeners[kind] = callback; },
      setAttribute(key, value) { this[key] = value; },
      decode() {
        if (!delayImages) return Promise.resolve();
        return new Promise((resolve, reject) => decodes.push({ image: this, resolve, reject }));
      },
    };
  }
  const container = node('div');
  const status = node('p');
  const document = { createElement: node, createElementNS: (_, name) => node(name), visibilityState: 'visible' };
  const requests = [];
  const timers = [];
  const panel = createCameraPanel(container, status, {
    document, initialMode, request(path) { return new Promise((resolve, reject) => requests.push({ path, resolve, reject })); },
    schedule(callback, delay) { const timer = { callback, delay }; timers.push(timer); return timer; },
    cancel(timer) { timer.cancelled = true; },
    isVisible: () => document.visibilityState !== 'hidden' && container.open,
  });
  function descendants(parent) { return parent.children.flatMap(child => [child, ...descendants(child)]); }
  return { panel, container, status, document, requests, timers, decodes,
    images() { return descendants(container).filter(child => child.tagName === 'img'); },
  };
}

test('task annotations attach to their own image surface and refresh within the same capture', async () => {
  const ui = harness();
  const payload = capture();
  payload.object_labels = {observation_id: 'rgbd-1', generation: 1, analysis_status: 'analyzed',
    items: ['1', '2', '3'].map(camera_id => ({camera_id, id: 'obstacle-full-id', role: 'obstacle',
      label: '黑色方块 · 障碍物\nID obstacle-full-id', x: .2, y: .3}))};
  let pending = ui.panel.refresh();
  ui.requests.at(-1).resolve(payload);
  await pending;
  for (const card of ui.container.children) {
    const figure = card.children[1].children[0];
    const surface = figure.children[0];
    assert.equal(surface.className, 'camera-image-surface');
    assert.equal(surface.children[0].tagName, 'img');
    assert.match(surface.children[1].children[0].textContent, /obstacle-full-id/);
    assert.match(figure.children[1].textContent, /本帧/);
  }
  payload.object_labels.items.forEach(item => { item.label = '已搬运\nID target-full-id'; });
  pending = ui.panel.refresh();
  ui.requests.at(-1).resolve(payload);
  await pending;
  assert.match(ui.container.children[0].children[1].children[0].children[0].children[1].children[0].textContent, /已搬运/);
});

function preview(frame = 'preview-1', generation = 1) {
  return { preview_frame_id: frame, generation, time: 12,
    cameras: ['1', '2', '3'].map(id => ({ id, label: id, frame_id: frame,
      rgb_url: `/api/camera-preview/${id}/rgb?frame_id=${frame}` })) };
}

const settle = () => new Promise(resolve => setImmediate(resolve));

test('successive preview frames retain the same cards and image containers', async () => {
  const ui = harness('preview');
  let pending = ui.panel.refresh();
  ui.requests[0].resolve(preview());
  await pending;
  const cards = [...ui.container.children];
  const images = cards.map(card => card.children[1]);
  for (let frame = 2; frame <= 5; frame++) {
    pending = ui.panel.refresh();
    ui.requests.at(-1).resolve(preview(`preview-${frame}`));
    await pending;
    cards.forEach((card, index) => {
      assert.equal(ui.container.children[index], card);
      assert.equal(card.children[1], images[index]);
    });
    assert.ok(ui.images().every(image => image.src.endsWith(`frame_id=preview-${frame}`)));
  }
});

test('preview keeps the last complete frame until all three new images decode', async () => {
  const ui = harness('preview', { delayImages: true });
  let pending = ui.panel.refresh();
  ui.requests[0].resolve(preview());
  await settle();
  assert.equal(ui.images().length, 0);
  ui.decodes.forEach(job => job.resolve());
  await pending;
  const previous = [...ui.images()];
  pending = ui.panel.refresh();
  ui.requests[1].resolve(preview('preview-2'));
  await settle();
  ui.decodes.slice(3, 5).forEach(job => job.resolve());
  await settle();
  assert.deepEqual(ui.images(), previous);
  assert.match(ui.status.textContent, /preview-1/);
  ui.decodes[5].resolve();
  await pending;
  assert.ok(ui.images().every(image => image.src.endsWith('frame_id=preview-2')));
});

test('an image decode failure preserves the last complete preview and allows the next refresh', async () => {
  const ui = harness('preview', { delayImages: true });
  let pending = ui.panel.refresh();
  ui.requests[0].resolve(preview());
  await settle();
  ui.decodes.forEach(job => job.resolve());
  await pending;
  const previous = [...ui.images()];
  pending = ui.panel.refresh();
  ui.requests[1].resolve(preview('preview-2'));
  await settle();
  assert.equal(ui.decodes.length, 6);
  ui.decodes[3].reject(new Error('image unavailable'));
  ui.decodes.slice(4).forEach(job => job.resolve());
  await pending;
  assert.deepEqual(ui.images(), previous);
  assert.match(ui.status.textContent, /相机预览读取失败/);
  pending = ui.panel.refresh();
  ui.requests[2].resolve(preview('preview-3'));
  await settle();
  ui.decodes.slice(6).forEach(job => job.resolve());
  await pending;
  assert.ok(ui.images().every(image => image.src.endsWith('frame_id=preview-3')));
});

test('reset while images decode discards the old scene without restoring its images', async () => {
  const ui = harness('preview', { delayImages: true });
  const pending = ui.panel.refresh();
  ui.requests[0].resolve(preview());
  await settle();
  assert.equal(ui.decodes.length, 3);
  ui.panel.setGeneration(2);
  ui.decodes.forEach(job => job.resolve());
  await pending;
  assert.equal(ui.images().length, 0);
  assert.match(ui.status.textContent, /等待当前场景/);
});

test('switching away and back while a frame request is pending discards that response', async () => {
  const ui = harness('preview');
  const pending = ui.panel.refresh();
  ui.panel.setMode('task');
  ui.panel.setMode('preview');
  ui.requests[0].resolve(preview());
  await pending;
  assert.equal(ui.images().length, 0);
  assert.match(ui.status.textContent, /等待实时/);
});

function capture(generation = 1, observationId = 'rgbd-1') {
  return { observation_id: observationId, generation, time: 10.25,
    cameras: ['1', '2', '3'].map(id => ({ id, label: `相机 ${id}`, frame_id: `${observationId}-${id}`,
      rgb_url: `/api/cameras/${id}/rgb`, depth_url: `/api/cameras/${id}/depth` })) };
}

test('camera panel displays the three registered RGB and depth frames using read-only local URLs', async () => {
  const ui = harness();
  const pending = ui.panel.refresh();
  assert.equal(ui.requests[0].path, '/api/cameras');
  ui.requests[0].resolve(capture());
  await pending;
  assert.equal(ui.images().length, 6);
  assert.ok(ui.images().every(image => image.src.startsWith('/api/cameras/') && image.src.includes('frame_id=')));
  assert.match(ui.status.textContent, /rgbd-1/);
  assert.doesNotMatch(ui.status.textContent, /仿真时间|10\.25/);
});

test('the actual API bound metadata displays one correctly bound query per preview URL', async () => {
  const ui = harness();
  const pending = ui.panel.refresh();
  const payload = capture(2, 'rgbd-generation-2-observation-3');
  for (const camera of payload.cameras) {
    camera.frame_id = payload.observation_id;
    camera.rgb_url = `/api/cameras/${camera.id}/rgb?frame_id=${payload.observation_id}`;
    camera.depth_url = `/api/cameras/${camera.id}/depth?frame_id=${payload.observation_id}`;
  }
  ui.requests[0].resolve(payload);
  await pending;
  assert.equal(ui.images().length, 6);
  assert.deepEqual(ui.images().map(image => image.src), ['2', '3', '1'].flatMap(id => [
    `/api/cameras/${id}/rgb?frame_id=rgbd-generation-2-observation-3`,
    `/api/cameras/${id}/depth?frame_id=rgbd-generation-2-observation-3`,
  ]));
});

for (const url of ['/api/cameras/1/rgb?frame_id=other', '/api/cameras/1/rgb?frame_id=rgbd-1-1&other=1',
  '/api/cameras/1/rgb?frame_id=rgbd-1-1&frame_id=rgbd-1-1', '/api/cameras/2/rgb?frame_id=rgbd-1-1']) {
  test(`mismatched camera frame URL ${url} cannot populate previews`, async () => {
    const ui = harness();
    const pending = ui.panel.refresh();
    const payload = capture();
    payload.cameras[0].rgb_url = url;
    ui.requests[0].resolve(payload);
    await pending;
    assert.equal(ui.images().length, 0);
    assert.match(ui.status.textContent, /相机预览读取失败/);
  });
}

test('camera polling waits two seconds after completion, never overlaps and pauses when hidden or collapsed', async () => {
  const ui = harness();
  ui.panel.start();
  ui.timers[0].callback();
  const pending = ui.panel.refresh();
  assert.equal(ui.requests.length, 1);
  ui.requests[0].resolve(capture());
  await pending;
  assert.equal(ui.timers.at(-1).delay, 2000);
  ui.document.visibilityState = 'hidden';
  ui.timers.at(-1).callback();
  assert.equal(ui.requests.length, 1);
  ui.document.visibilityState = 'visible';
  ui.container.open = false;
  ui.timers.at(-1).callback();
  assert.equal(ui.requests.length, 1);
  ui.panel.stop();
  assert.equal(ui.timers.at(-1).cancelled, true);
});

test('a reset rejects older camera generations and empty cached captures stay explicitly unavailable', async () => {
  const ui = harness();
  const pending = ui.panel.refresh();
  ui.panel.setGeneration(2);
  ui.requests[0].resolve(capture(1));
  await pending;
  assert.equal(ui.images().length, 0);
  const next = ui.panel.refresh();
  ui.requests[1].resolve({ observation_id: null, generation: 2, time: 0, cameras: [] });
  await next;
  assert.equal(ui.images().length, 0);
  assert.match(ui.status.textContent, /等待/);
});

test('invalid remote frame URLs cannot populate the sensor preview', async () => {
  const ui = harness();
  const pending = ui.panel.refresh();
  const invalid = capture();
  invalid.cameras[0].rgb_url = 'https://example.invalid/truth.png';
  ui.requests[0].resolve(invalid);
  await pending;
  assert.equal(ui.images().length, 0);
  assert.match(ui.status.textContent, /相机预览读取失败/);
});


test('live preview shows only RGB in left head right order and switches to task frames', async () => {
  const ui = harness('preview');
  const pending = ui.panel.refresh();
  assert.equal(ui.requests[0].path, '/api/camera-preview');
  ui.requests[0].resolve({preview_frame_id: 'preview-1', generation: 1, time: 12.,
    cameras: ['1','2','3'].map(id => ({id, frame_id: 'preview-1', label: id,
      rgb_url: `/api/camera-preview/${id}/rgb?frame_id=preview-1`}))});
  await pending;
  assert.equal(ui.images().length, 3);
  assert.deepEqual(ui.images().map(image => image.src.split('/')[3]), ['2','3','1']);
  ui.panel.setMode('task');
  const task = ui.panel.refresh();
  assert.equal(ui.requests.at(-1).path, '/api/cameras');
  ui.requests.at(-1).resolve(capture());
  await task;
  assert.equal(ui.images().length, 6);
});

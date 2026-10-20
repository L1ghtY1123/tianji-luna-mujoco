import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import { createCollisionSphereLayer, validateCollisionSpheres } from '../src/collision-sphere-layer.js';

const source = (await readFile(new URL('../src/collision-sphere-page.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\r?\n/gm, '');
const settle = () => new Promise(resolve => setImmediate(resolve));
const snapshot = (revision = 1) => ({
  source: 'resources/curobo/marvin_gui_dual.yml', gripper_state: 'open',
  radius_policy: 'original_yaml_plus_link_buffer', sphere_count: 1, link_count: 1,
  spheres: [{ id: 'finger:0', link: 'finger', center_m: [.2, .1, .7], radius_m: .02 }],
  simulation: { manifest_revision: revision, generation: 1, time: 2.5,
    geometry_poses: [{ id: 1, position: [.2, .1, .7], quaternion: [1, 0, 0, 0] }], objects: [], table: { position: [0, 0, .5] } },
});

function page() {
  const elements = new Map();
  const pending = [];
  const requests = [];
  const timers = new Map();
  const windowListeners = {};
  let nextTimer = 0;
  const view = { snapshots: [], manifests: [], home() {}, setTable() {},
    async loadManifest(value) { this.manifests.push(value); },
    update(value) { this.snapshots.push(value); },
    setOverlay(value) { this.overlay = value; },
    setGeometryVisible(value) { this.geometryVisible = value; },
  };
  const element = id => {
    if (!elements.has(id)) elements.set(id, { textContent: '', hidden: false, checked: true, value: '65',
      listeners: {}, addEventListener(kind, callback) { this.listeners[kind] = callback; },
      setAttribute() {},
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    document: { getElementById: element, hidden: false },
    window: { addEventListener(kind, callback) { windowListeners[kind] = callback; } },
    createSceneView: () => view, createCollisionSphereLayer, validateCollisionSpheres,
    request(path, body) {
      requests.push({ path, body });
      return new Promise((resolve, reject) => pending.push({ path, resolve, reject }));
    },
    setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
  });
  vm.runInContext(source, context);
  return {
    view, element, pending, timers, requests,
    windowEvent(kind, event) { windowListeners[kind]?.(event); },
    click(id) { return element(id).listeners.click(); },
    async reply(path, value) {
      const index = pending.findIndex(item => item.path === path);
      assert.notEqual(index, -1, `Expected ${path}`);
      pending.splice(index, 1)[0].resolve(value);
      await settle();
    },
    async tick() {
      const [id, callback] = timers.entries().next().value;
      timers.delete(id); callback(); await settle();
    },
  };
}

test('the separate viewer uses one open-gripper snapshot for spheres and meshes and only reads APIs', async () => {
  const ui = page();
  const data = snapshot();
  await ui.reply('/api/collision-spheres', data);
  await ui.reply('/api/manifest', { revision: 1, geometries: [] });
  assert.equal(ui.view.snapshots[0], data.simulation);
  assert.equal(ui.view.overlay.children[0].count, 1);
  assert.equal(ui.element('sphere-loading').hidden, true);
  assert.match(ui.element('sphere-count').textContent, /1.*link.*1.*碰撞球/);
  assert.match(ui.element('sphere-source').textContent, /marvin_gui_dual.yml/);
  assert.equal(ui.timers.size, 1);
  await ui.tick();
  const moved = snapshot(); moved.simulation.time = 3;
  await ui.reply('/api/collision-spheres', moved);
  assert.equal(ui.view.snapshots.at(-1), moved.simulation);
  assert.equal(ui.view.manifests.length, 1);
  assert.ok(ui.requests.every(({ path, body }) => path.startsWith('/api/') && body === undefined));
});

test('pause during an in-flight read does not restart polling; refresh is serialized and remains paused', async () => {
  const ui = page();
  ui.click('toggle-follow');
  ui.click('refresh-spheres');
  assert.equal(ui.pending.length, 1);
  await ui.reply('/api/collision-spheres', snapshot());
  await ui.reply('/api/manifest', { revision: 1, geometries: [] });
  assert.equal(ui.timers.size, 0);
  assert.match(ui.element('sphere-status').textContent, /已暂停/);
  ui.click('refresh-spheres');
  await ui.reply('/api/collision-spheres', snapshot());
  assert.equal(ui.timers.size, 0);
  ui.click('toggle-follow');
  assert.equal(ui.pending.length, 1);
});

test('a rebuilt scene cannot pair old snapshot poses with new geometry; next read recovers', async () => {
  const ui = page();
  await ui.reply('/api/collision-spheres', snapshot(1));
  await ui.reply('/api/manifest', { revision: 2, geometries: [] });
  assert.equal(ui.view.snapshots.length, 0);
  assert.equal(ui.view.manifests.length, 0);
  assert.match(ui.element('sphere-status').textContent, /场景/);
  await ui.tick();
  await ui.reply('/api/collision-spheres', snapshot(2));
  await ui.reply('/api/manifest', { revision: 2, geometries: [] });
  assert.equal(ui.view.snapshots.length, 1);
  assert.equal(ui.view.manifests.length, 1);
});

test('returning from browser history retains the sphere layer and resumes a single refresh loop', async () => {
  const ui = page();
  await ui.reply('/api/collision-spheres', snapshot());
  await ui.reply('/api/manifest', { revision: 1, geometries: [] });
  const mesh = ui.view.overlay.children[0];
  ui.windowEvent('pagehide', { persisted: true });
  assert.equal(ui.timers.size, 0);
  assert.equal(ui.view.overlay.children[0], mesh);
  ui.windowEvent('pageshow', { persisted: true });
  await ui.reply('/api/collision-spheres', snapshot());
  assert.equal(ui.view.overlay.children[0], mesh);
  assert.equal(ui.timers.size, 1);
});

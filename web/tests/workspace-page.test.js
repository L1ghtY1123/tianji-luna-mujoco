import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = (await readFile(new URL('../src/workspace-page.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\r?\n/gm, '');
const settle = () => new Promise(resolve => setImmediate(resolve));
const snapshot = (revision = 1) => ({ server_instance_id: 'server-one', simulation: {
  manifest_revision: revision, time: 2.5, geometry_poses: [], table: { position: [0, 0, .6] },
} });
const atlas = { arms: [{ arm: 'A', reachable_points: 123, resolution_m: .01 },
  { arm: 'B', reachable_points: 120, resolution_m: .01 }] };

function page() {
  const elements = new Map(), pending = [], requests = [], timers = new Map();
  let timerId = 0;
  const element = id => {
    if (!elements.has(id)) elements.set(id, { textContent: '', hidden: false, checked: true, value: '22',
      listeners: {}, addEventListener(kind, callback) { this.listeners[kind] = callback; }, setAttribute() {} });
    return elements.get(id);
  };
  const view = { snapshots: [], manifests: [], home() {}, setTable() {},
    async loadManifest(value) { this.manifests.push(value); }, update(value) { this.snapshots.push(value); },
    setOverlay(value) { this.overlay = value; }, setGeometryVisible(value) { this.geometryVisible = value; } };
  const layers = [];
  const context = vm.createContext({ document: { getElementById: element },
    window: { addEventListener() {} }, createSceneView: () => view,
    createWorkspaceLayer(value) {
      const layer = { object: {}, visible: {}, payload: value, setVisible(arm, v) { this.visible[arm] = v; },
        setOpacity(v) { this.opacity = v; }, dispose() { this.disposed = true; } };
      layers.push(layer); return layer;
    },
    request(path, body) {
      requests.push({ path, body });
      return new Promise((resolve, reject) => pending.push({ path, resolve, reject }));
    }, setTimeout(callback) { timers.set(++timerId, callback); return timerId; },
    clearTimeout(id) { timers.delete(id); },
  });
  vm.runInContext(source, context);
  return { view, element, pending, requests, timers, layers,
    click(id) { return element(id).listeners.click(); },
    async reply(path, data) {
      const i = pending.findIndex(item => item.path === path); assert.notEqual(i, -1, path);
      pending.splice(i, 1)[0].resolve(data); await settle();
    }, async tick() {
      const [id, callback] = timers.entries().next().value; timers.delete(id); callback(); await settle();
    },
  };
}

test('loads static workspace once and follows read-only poses with independent controls', async () => {
  const ui = page();
  ui.element('show-A').checked = false;
  await ui.reply('/api/workspace-scene', snapshot());
  await ui.reply('/api/workspaces', atlas);
  await ui.reply('/api/manifest', { revision: 1, geometries: [] });
  assert.equal(ui.layers[0].visible.A, false);
  assert.equal(ui.layers[0].visible.B, true);
  assert.equal(ui.element('workspace-loading').hidden, true);
  ui.element('workspace-opacity').listeners.input({ target: { value: '10' } });
  assert.equal(ui.layers[0].opacity, .1);
  await ui.tick();
  await ui.reply('/api/workspace-scene', snapshot());
  assert.equal(ui.layers.length, 1);
  assert.equal(ui.view.snapshots.length, 2);
  assert.ok(ui.requests.every(item => item.body === undefined));
  assert.equal(ui.requests.filter(item => item.path === '/api/workspaces').length, 1);
});

test('serializes refresh and does not combine new geometry with stale scene poses', async () => {
  const ui = page();
  ui.click('toggle-follow'); ui.click('refresh-workspaces');
  assert.equal(ui.pending.length, 1);
  await ui.reply('/api/workspace-scene', snapshot());
  await ui.reply('/api/workspaces', atlas);
  await ui.reply('/api/manifest', { revision: 2, geometries: [] });
  assert.equal(ui.view.snapshots.length, 0);
  assert.equal(ui.timers.size, 0);
  assert.match(ui.element('workspace-status').textContent, /场景/);
  ui.click('refresh-workspaces');
  await ui.reply('/api/workspace-scene', snapshot(2));
  await ui.reply('/api/manifest', { revision: 2, geometries: [] });
  assert.equal(ui.view.snapshots.length, 1);
});

test('backend restart refreshes the atlas and releases old render geometry', async () => {
  const ui = page();
  await ui.reply('/api/workspace-scene', snapshot());
  await ui.reply('/api/workspaces', atlas);
  await ui.reply('/api/manifest', { revision: 1, geometries: [] });
  await ui.tick();
  await ui.reply('/api/workspace-scene', { ...snapshot(), server_instance_id: 'server-two' });
  await ui.reply('/api/workspaces', atlas);
  await ui.reply('/api/manifest', { revision: 1, geometries: [] });
  assert.equal(ui.layers.length, 2);
  assert.equal(ui.layers[0].disposed, true);
});

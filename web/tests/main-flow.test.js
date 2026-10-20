import { runConclusion } from '../src/run-conclusion.js';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import * as model from '../src/model.js';
import { createTaskClock } from '../src/task-clock.js';
import { createPlanningSettingsEditor } from '../src/planning-settings.js';
import { planningPayload, planningSettings } from './planning-fixture.js';
import { createCameraPanel } from '../src/camera-panel.js';
import { createObjectList } from '../src/object-list.js';
import { createPointcloudPanel } from '../src/pointcloud-panel.js';
import { runEvidence, runStatusSummary } from '../src/run-evidence.js';

// Execute the real page entry point with DOM and network boundaries replaced.
// No browser, GPU service, or physical scene is touched by these ordering tests.
const source = (await readFile(new URL('../src/main.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\r?\n/gm, '');

const settle = () => new Promise(resolve => setImmediate(resolve));

function snapshot({ generation = 1, serverTime = 1000, run = { status: 'idle' }, events = [], objects = [], settings = planningPayload(), instanceId } = {}) {
  return {
    simulation: { generation, confirmed: true, time: 9000, objects, table: { position: [0, 0, .5], size: [1, 1, .1] } },
    server_time: serverTime,
    server_instance_id: instanceId,
    run,
    events,
    events_mode: 'replace',
    planning_settings: settings,
  };
}

async function page(initial = snapshot(), { loadSettings = true } = {}) {
  const elements = new Map();
  const pending = [];
  const intervals = new Map();
  let socket;
  let timeMs = 0;
  let dragCallbacks;
  let reloadCount = 0;
  let traceRenders = 0;

  function element(tagName) {
    return {
      tagName, value: '', textContent: '', listeners: {}, style: {}, children: [],
      classList: { toggle() {} },
      addEventListener(kind, listener) { this.listeners[kind] = listener; },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      showModal() { this.open = true; },
      close() { this.open = false; },
    };
  }

  function getElement(id) {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  }
  getElement('instruction').value = '抓取黑色物体';
  getElement('record-video').checked = true;
  getElement('experiment-save-mode').value = 'auto';

  const view = {
    setInteraction(value) { this.interaction = value; },
    setDraft(value) { this.draft = value ? structuredClone(value) : null; },
    setTable() {}, update() {}, cancelInteraction() {}, home() {},
    setOverlay(value) { this.overlay = value; },
    setEnvironmentVisible(value) { this.environmentVisible = value; },
    async loadManifest() {},
  };
  const context = vm.createContext({
    ...model,
    Error,
    document: { getElementById: getElement, createElement: element, createTextNode: text => text },
    createSceneView: (container, callbacks) => { dragCallbacks = callbacks; return view; },
    createTraceList: () => () => { traceRenders += 1; },
    createTaskClock: () => createTaskClock({ now: () => timeMs }),
    createPlanningSettingsEditor,
    createCameraPanel,
    createObjectList: container => createObjectList(container, { document: { createElement: element } }),
    createPointcloudPanel,
    runConclusion,
    runEvidence,
    runStatusSummary,
    performance: { now: () => timeMs },
    location: { protocol: 'http:', host: 'test.invalid', reload() { reloadCount += 1; } },
    WebSocket: class {
      constructor() { this.listeners = {}; socket = this; }
      addEventListener(kind, listener) { this.listeners[kind] = listener; }
    },
    request(path, body) {
      if (path === '/api/manifest') return Promise.resolve({ revision: 1, geometries: [] });
      if (path === '/api/health') return Promise.resolve({});
      return new Promise((resolve, reject) => pending.push({ path, body, resolve, reject }));
    },
    setInterval(callback, milliseconds) { intervals.set(milliseconds, callback); },
    setTimeout() {},
    clearTimeout() {},
  });
  vm.runInContext(source, context, { filename: 'main.js' });

  function takeRequest(path) {
    const index = pending.findIndex(request => request.path === path);
    assert.notEqual(index, -1, `Expected request to ${path}`);
    return pending.splice(index, 1)[0];
  }
  takeRequest('/api/state').resolve(initial);
  if (loadSettings) takeRequest('/api/planning-settings').resolve(planningPayload());
  await settle();

  function vectorInput(section, vectorIndex, axis) {
    const row = getElement(section).children.filter(child => child.className === 'vector-fields')[vectorIndex];
    return row.children[axis].children.find(child => child.tagName === 'input');
  }

  function edit(input, value) {
    input.value = value;
    input.listeners.input();
  }

  return {
    click(id) { return getElement(id).listeners.click(); },
    takeRequest,
    advance(ms) { timeMs += ms; intervals.get(100)(); },
    seconds() { return Number(getElement('sim-time').textContent); },
    message(payload) { socket.listeners.message({ data: JSON.stringify(payload) }); },
    connection(kind) { socket.listeners[kind](); },
    reloads() { return reloadCount; },
    traceRenders() { return traceRenders; },
    view,
    element: getElement,
    vectorInput,
    select(id) { dragCallbacks.onSelect(id); },
    requests() { return pending.map(request => request.path); },
    drag(id, position) {
      dragCallbacks.onSelect(id);
      dragCallbacks.onDragStart(id);
      return dragCallbacks.onDragEnd(id, position);
    },
    inputPosition(index, value) {
      edit(vectorInput('object-editor', 0, index), value);
    },
    inputSize(index, value) { edit(vectorInput('object-editor', 1, index), value); },
    inputTable(vectorIndex, axis, value) { edit(vectorInput('table-fields', vectorIndex, axis), value); },
    inputYaw(value) {
      edit(vectorInput('object-editor', 0, 3), value);
    },
    metadataInput(label) {
      const fields = getElement('object-editor').children.find(child => child.className === 'metadata-fields');
      return fields.children.find(child => child.children[0]?.textContent === label)?.children[1];
    },
    inputMetadata(label, value) { edit(this.metadataInput(label), value); },
    scalarInput(label) {
      return getElement('object-editor').children.find(child => child.className === 'input-field' && child.children[0]?.textContent === label)?.children[1];
    },
    inputScalar(label, value) { edit(this.scalarInput(label), value); },
    planningInput(name) {
      function find(node) {
        if (node.name === name) return node;
        for (const child of node.children) { const result = find(child); if (result) return result; }
      }
      return find(getElement('planning-groups'));
    },
    inputPlanning(name, value) {
      assert.notEqual(getElement('planning-fields').disabled, true);
      assert.notEqual(this.planningInput(name).disabled, true);
      edit(this.planningInput(name), value);
    },
  };
}

test('live pose updates do not reserialize unchanged event history', async () => {
  const app = await page(snapshot({ events: [{ id: 1, kind: 'run_start', data: {} }] }));
  const before = app.traceRenders();
  app.message({ ...snapshot(), events_mode: 'append' });
  app.message({ ...snapshot(), events_mode: 'append' });
  assert.equal(app.traceRenders(), before);
  app.message({ ...snapshot({ events: [{ id: 2, kind: 'skill_start', data: {} }] }), events_mode: 'append' });
  assert.equal(app.traceRenders(), before + 1);
});

const sceneObjects = [
  { id: 'black_cube', kind: 'cube', color: 'black', position: [.4, .15, .62], size: [.04, .04, .04], quaternion: [1, 0, 0, 0] },
  { id: 'red_box', kind: 'box', color: 'red', position: [.5, -.15, .63], size: [.05, .04, .06], quaternion: [1, 0, 0, 0] },
];

test('cancel scene edits restores the latest physical snapshot without a mutation request', async () => {
  const ui = await page(snapshot({ objects: sceneObjects }));
  assert.equal(ui.element('cancel-scene').disabled, true);
  ui.drag('black_cube', [.8, .2, .9]);
  ui.inputSize(0, '70');
  ui.inputMetadata('颜色', 'green');
  ui.inputTable(0, 0, '100');
  ui.click('add-object');
  const updated = sceneObjects.map(object => ({ ...object, position: [...object.position] }));
  updated[0].position[2] = .61;
  ui.message(snapshot({ objects: updated, serverTime: 1001 }));
  assert.equal(ui.element('cancel-scene').disabled, false);
  ui.click('cancel-scene');
  assert.deepEqual(ui.requests(), []);
  assert.equal(ui.view.draft, null);
  assert.equal(ui.element('object-count').textContent, 2);
  assert.equal(ui.vectorInput('object-editor', 0, 2).value, 610);
  assert.equal(ui.vectorInput('object-editor', 1, 0).value, 40);
  assert.equal(ui.metadataInput('颜色').value, 'black');
  assert.equal(ui.vectorInput('table-fields', 0, 0).value, 0);
  assert.equal(ui.element('cancel-scene').disabled, true);
  assert.equal(ui.element('apply-scene').disabled, true);
  assert.equal(ui.element('start').disabled, false);
});

test('cancel invalid edits and removed objects restores the accepted scene', async () => {
  const ui = await page(snapshot({ objects: sceneObjects }));
  ui.inputPosition(0, '');
  const heading = ui.element('object-editor').children.find(child => child.className === 'selected-heading');
  heading.children[1].listeners.click();
  assert.equal(ui.element('object-count').textContent, 1);
  assert.equal(ui.element('cancel-scene').disabled, false);
  ui.click('cancel-scene');
  ui.select('black_cube');
  assert.equal(ui.vectorInput('object-editor', 0, 0).value, 400);
  assert.equal(ui.element('object-count').textContent, 2);
  assert.equal(ui.view.draft, null);
  assert.deepEqual(ui.requests(), []);
});

test('long errors keep a compact summary and full details without reopening a dismissed repeat', async () => {
  const ui = await page(snapshot());
  const failure = 'backend_failure:' + 'long_detail_'.repeat(80);
  const failed = snapshot({ serverTime: 1001, run: { status: 'failed', error: failure } });
  ui.message(failed);
  assert.ok(ui.element('error-message').textContent.length <= 120);
  assert.ok(ui.element('error-details').textContent.includes(failure));
  ui.click('dismiss-error');
  ui.message({ ...failed, server_time: 1002 });
  assert.equal(ui.element('error-banner').hidden, true);
});

test('partial success keeps collection counts and deferred reasons visible', async () => {
  const ui = await page(snapshot({ run: { status: 'partial_success', task_progress: {
    completed: [{ id: 'one' }], deferred: [{ id: 'two', reason: 'grasp_verification_failed' }],
  } } }));
  assert.equal(ui.element('run-status').textContent, '部分完成');
  const summary = ui.element('run-details').children.find(child => child.className === 'run-holding-summary');
  assert.match(summary.textContent, /已完成 1 个.*暂缓 1 个.*抓取未通过验证/);
  assert.equal(ui.element('facts-section').hidden, false);
});

test('active grasp failure presents evidence awaiting JEV recovery without reset advice', async () => {
  const ui = await page(snapshot({ run: { status: 'running', failure: { code: 'grasp_verification_failed' } } }));
  assert.match(ui.element('run-failure').textContent, /等待 JEV/);
  assert.equal(ui.element('run-failure').textContent.includes('Reset'), false);
});

test('marker editor exposes tabletop XY dimensions and submits fixed height after a drag', async () => {
  const ui = await page(snapshot({ objects: sceneObjects }));
  ui.inputMetadata('类型', 'marker');
  assert.equal(ui.view.draft.objects[0].color, 'yellow');
  assert.deepEqual(Array.from(ui.view.draft.objects[0].size), [.16, .12, .0002]);
  const rows = ui.element('object-editor').children.filter(child => child.className === 'vector-fields');
  assert.equal(rows[0].children.length, 4);
  assert.equal(rows[0].children[2].children[1].disabled, true);
  assert.equal(rows[1].children.length, 2);
  ui.drag('black_cube', [.1, .15, 1.5]);
  ui.click('apply-scene');
  const request = ui.takeRequest('/api/scene');
  assert.equal(request.body.objects[0].kind, 'marker');
  assert.ok(Math.abs(request.body.objects[0].position[2] - .5501) < 1e-10);
  assert.deepEqual(Array.from(request.body.objects[0].position).slice(0, 2), [.1, .15]);
});

test('Start includes experiment defaults and preserves existing scene and settings bindings', async () => {
  const ui = await page();
  ui.click('start');
  const body = ui.takeRequest('/api/start').body;
  assert.equal(body.instruction, '抓取黑色物体');
  assert.equal(body.generation, 1);
  assert.equal(body.settings_revision, 'settings-1');
  assert.deepEqual(JSON.parse(JSON.stringify(body.experiment)), { name: '', record_video: true, save_mode: 'auto' });
  assert.equal(ui.element('experiment-name').disabled, true);
});

test('Start sends a user experiment name, video choice and pending save policy', async () => {
  const ui = await page();
  ui.element('experiment-name').value = '  左臂放置  ';
  ui.element('record-video').checked = false;
  ui.element('experiment-save-mode').value = 'ask';
  ui.click('start');
  assert.deepEqual(JSON.parse(JSON.stringify(ui.takeRequest('/api/start').body.experiment)), {
    name: '左臂放置', record_video: false, save_mode: 'ask',
  });
});

test('no-save mode explicitly disables video and does not send a recording request', async () => {
  const ui = await page();
  ui.element('experiment-save-mode').value = 'none';
  ui.element('experiment-save-mode').listeners.change();
  assert.equal(ui.element('record-video').disabled, true);
  assert.equal(ui.element('record-video-label').textContent, '不录视频');
  ui.click('start');
  const experiment = ui.takeRequest('/api/start').body.experiment;
  assert.equal(experiment.save_mode, 'none');
  assert.equal(experiment.record_video, false);
});

test('experiment status exposes pending retention, encoding failures and a safe replay link', async () => {
  const ui = await page(snapshot({ run: { status: 'succeeded', experiment: {
    id: 'run one', name: '实验一', retention: 'pending', video_status: 'encoding',
  } } }));
  assert.equal(ui.element('experiment-result').hidden, false);
  assert.match(ui.element('experiment-status').textContent, /待决定保存.*编码/);
  assert.equal(ui.element('experiment-replay').href, '/replay.html?run=run%20one');
  ui.message(snapshot({ serverTime: 1001, run: { status: 'succeeded', experiment: {
    id: 'run one', retention: 'pending', video_status: 'error', error: 'ffmpeg failed',
  } } }));
  assert.match(ui.element('experiment-status').textContent, /失败.*ffmpeg failed/);
  ui.message(snapshot({ generation: 2, serverTime: 1002 }));
  assert.equal(ui.element('experiment-result').hidden, true);
});

test('viewer tabs keep camera polling visibility and pointcloud open lifecycle aligned', async () => {
  const ui = await page();
  assert.equal(ui.element('camera-panel').hidden, true);
  ui.click('view-tab-cameras');
  assert.equal(ui.element('camera-panel').hidden, false);
  assert.equal(ui.element('camera-panel').open, true);
  assert.equal(ui.element('viewport-shell').hidden, true);
  ui.click('view-tab-pointcloud');
  assert.equal(ui.element('camera-panel').open, false);
  assert.equal(ui.element('pointcloud-panel').open, true);
  assert.equal(ui.element('viewport-shell').hidden, false);
  assert.ok(ui.requests().includes('/api/perception-preview'));
  ui.click('view-tab-scene');
  assert.equal(ui.element('pointcloud-panel').open, false);
  assert.equal(ui.view.overlay, null);
});

test('parameters open in a dialog without clearing a pending draft', async () => {
  const ui = await page();
  ui.inputPlanning('time_scale', '2');
  ui.click('open-planning');
  assert.equal(ui.element('planning-dialog').open, true);
  ui.click('close-planning');
  assert.equal(ui.element('planning-dialog').open, false);
  assert.equal(ui.planningInput('time_scale').value, '2');
  assert.equal(ui.element('start').disabled, true);
});

test('parameter validation errors remain visible inside the open modal dialog', async () => {
  const ui = await page();
  ui.click('open-planning');
  ui.inputPlanning('time_scale', '');
  ui.click('apply-planning');
  assert.equal(ui.element('planning-error').hidden, false);
  assert.match(ui.element('planning-error').textContent, /数值/);
  assert.equal(ui.requests().includes('/api/planning-settings'), false);
});

test('holding status survives terminal runs and pending release, with internal stage timing and no raw clouds', async () => {
  const run = { status: 'running', current_step: { skill: 'execute_grasp', started_at: 990,
    stage: 'post_grasp', stage_started_at: 998 }, holding: { state: 'held', arm: 'A', verified: true,
    attached: true, points_tool_m: [[99, 99, 99]], attachment_evidence: { prepare_wall_ms: 12.4 } },
    collision_model: { gripper: 'actual_feedback', attached: true }, verification: { status: 'held', success: true } };
  const ui = await page(snapshot({ run }));
  assert.match(ui.element('run-step').textContent, /execute_grasp \/ post_grasp/);
  assert.match(ui.element('run-step-timing').textContent, /post_grasp 2.0 s/);
  const summary = ui.element('run-details').children.find(child => child.className === 'run-holding-summary');
  assert.match(summary.textContent, /A 臂 持物/);
  assert.match(summary.textContent, /几何准备 12.4 ms/);
  assert.equal(ui.element('facts-section').hidden, false);
  assert.equal(ui.element('facts-content').textContent.includes('points_tool_m'), false);
  assert.equal(JSON.parse(ui.element('facts-content').textContent).verification.status, 'held');
  ui.message(snapshot({ serverTime: 1001, run: { status: 'failed', holding: { state: 'release_pending', arm: 'A',
    verified: false, attached: false }, collision_model: { gripper: 'actual_feedback', attached: false } } }));
  assert.equal(ui.element('run-details').hidden, false);
  assert.match(summary.textContent, /释放待确认/);
  ui.message(snapshot({ generation: 2, serverTime: 1002 }));
  assert.equal(summary.hidden, true);
  assert.equal(ui.element('facts-section').hidden, true);
});

test('point-cloud data opens without an overlay and the viewport hide control preserves the open inspection panel', async () => {
  const ui = await page();
  assert.equal(ui.requests().includes('/api/perception-preview'), false);
  assert.equal(ui.element('hide-pointcloud').hidden, true);
  const panel = ui.element('pointcloud-panel');
  panel.open = true;
  panel.listeners.toggle();
  const read = ui.takeRequest('/api/perception-preview');
  assert.equal(read.body, undefined);
  read.resolve({
    source: 'registered_rgbd', generation: 1, scene_token: 'scene-1', observation_id: 'rgbd-1', checkpoint: 'initial',
    points_world_m: [[0, 0, .5]], kept_mask: [true], cluster_labels: null,
    counts: { input: 1, kept: 1, rejected: 0 }, total_point_count: 1, preview_sampling: 'uniform_original_indices',
  });
  await settle();
  assert.equal(ui.view.overlay, null);
  const controls = ui.element('pointcloud-content').children.find(child => child.className === 'pointcloud-controls');
  const toggle = controls.children.find(child => child.textContent === '显示点云');
  const environment = controls.children.find(child => child.textContent === '隐藏仿真物体');
  assert.ok(toggle);
  assert.ok(environment);
  environment.listeners.click();
  assert.equal(ui.view.environmentVisible, false);
  toggle.listeners.click();
  assert.equal(ui.view.overlay.name, 'registered-rgbd-preview');
  assert.equal(ui.element('hide-pointcloud').hidden, false);
  assert.match(ui.element('pointcloud-status').textContent, /initial/);
  ui.click('hide-pointcloud');
  assert.equal(ui.view.overlay, null);
  assert.equal(ui.element('hide-pointcloud').hidden, true);
  assert.equal(panel.open, true);
  assert.equal(toggle.textContent, '显示点云');
  assert.match(ui.element('pointcloud-status').textContent, /initial/);
  toggle.listeners.click();
  assert.equal(ui.view.overlay.name, 'registered-rgbd-preview');
  panel.open = false;
  panel.listeners.toggle();
  assert.equal(ui.view.overlay, null);
  assert.equal(ui.view.environmentVisible, true);
  assert.equal(ui.element('hide-pointcloud').hidden, true);
  ui.message(snapshot({ generation: 2, serverTime: 1001 }));
  assert.equal(ui.view.overlay, null);
  assert.match(ui.element('pointcloud-status').textContent, /当前场景/);
  assert.equal(ui.requests().includes('/api/perception-preview'), false);
});

test('bin controls edit external dimensions, wall thickness and mass in a shared local draft', async () => {
  const bin = { ...sceneObjects[1], id: 'blue_bin', kind: 'bin', size: [.24, .18, .12], wall_thickness_m: .01, mass_kg: .4 };
  const ui = await page(snapshot({ objects: [bin] }));
  assert.equal(ui.scalarInput('壁厚 / mm')?.value, 10);
  assert.equal(ui.scalarInput('壁厚 / mm').step, '1');
  ui.inputSize(0, '250');
  ui.inputScalar('壁厚 / mm', '12');
  ui.inputScalar('质量 / kg', '.5');
  assert.deepEqual(ui.view.draft.objects[0].size, [.25, .18, .12]);
  assert.equal(ui.view.draft.objects[0].wall_thickness_m, .012);
  assert.deepEqual(ui.requests(), []);
  ui.click('apply-scene');
  const sent = ui.takeRequest('/api/scene').body.objects[0];
  assert.equal(sent.wall_thickness_m, .012);
  assert.equal(sent.mass_kg, .5);
});

test('switching a solid box to a bin supplies usable independent cavity controls', async () => {
  const ui = await page(snapshot({ objects: [sceneObjects[1]] }));
  ui.inputMetadata('类型', 'bin');
  assert.equal(ui.scalarInput('壁厚 / mm')?.value, 10);
  assert.deepEqual(ui.view.draft.objects[0].size, [.05, .04, .06]);
  assert.equal(ui.view.draft.objects[0].kind, 'bin');
  assert.equal(ui.element('apply-scene').disabled, false);
  assert.match(ui.element('scene-note').textContent, /类型.*应用.*显示/);
  assert.deepEqual(ui.requests(), []);
  ui.inputMetadata('类型', 'box');
  assert.equal(ui.scalarInput('壁厚 / mm'), undefined);
});

test('place-only tasks explain the need for verified session holding and display the destination independently', async () => {
  const ui = await page(snapshot({ run: { status: 'failed', task: 'place', target_id: 'source_track',
    destination_id: 'bin_track', error: 'place_requires_verified_holding' } }));
  assert.match(ui.element('error-message').textContent, /已验证.*持物/);
  assert.equal(ui.element('target-value').textContent, 'source_track');
  assert.equal(ui.element('destination-value').textContent, 'bin_track');
});

test('a successful grasp-place run uses a task-wide success label', async () => {
  const ui = await page(snapshot({ run: { status: 'succeeded', task: 'grasp_place' } }));
  assert.equal(ui.element('run-status').textContent, '任务成功');
});

test('object and table lengths display millimeters with one millimeter arrows', async () => {
  const ui = await page(snapshot({ objects: sceneObjects }));
  for (const [section, vectorIndex, expected] of [
    ['object-editor', 0, [400, 150, 620]],
    ['object-editor', 1, [40, 40, 40]],
    ['table-fields', 0, [0, 0, 500]],
    ['table-fields', 1, [1000, 1000, 100]],
  ]) {
    expected.forEach((value, axis) => {
      assert.equal(Number(ui.vectorInput(section, vectorIndex, axis).value), value);
      assert.equal(ui.vectorInput(section, vectorIndex, axis).step, '1');
    });
    assert.ok(ui.element(section).children.filter(child => child.className === 'field-heading').every(child => child.textContent.includes('/ mm')));
  }
  ui.drag('black_cube', [.631, -.122, .805]);
  assert.deepEqual([0, 1, 2].map(axis => Number(ui.vectorInput('object-editor', 0, axis).value)), [631, -122, 805]);
});

test('millimeter positions, dimensions, and degree yaw preview in SI and wait for Apply', async () => {
  const ui = await page(snapshot({ objects: sceneObjects }));
  ui.select('red_box');
  ui.inputPosition(0, '501');
  ui.inputSize(1, '41');
  ui.inputYaw('35');
  ui.inputTable(0, 0, '25');
  ui.inputTable(1, 2, '101');
  assert.deepEqual(ui.view.draft.objects[1].position, [.501, -.15, .63]);
  assert.deepEqual(ui.view.draft.objects[1].size, [.05, .041, .06]);
  assert.equal(ui.view.draft.objects[1].yaw_deg, 35);
  assert.deepEqual(ui.view.draft.table.position, [.025, 0, .5]);
  assert.deepEqual(ui.view.draft.table.size, [1, 1, .101]);
  assert.deepEqual(ui.requests(), []);
  ui.click('apply-scene');
  const apply = ui.takeRequest('/api/scene');
  assert.deepEqual(Array.from(apply.body.objects[1].position), [.501, -.15, .63]);
  assert.deepEqual(Array.from(apply.body.objects[1].size), [.05, .041, .06]);
  assert.equal(apply.body.objects[1].yaw_deg, 35);
  assert.deepEqual(Array.from(apply.body.table.position), [.025, 0, .5]);
  assert.deepEqual(Array.from(apply.body.table.size), [1, 1, .101]);
});

for (const kind of ['cube', 'sphere', 'cylinder', 'box']) {
  test(`${kind} size arrows preserve shape constraints without replacing the focused input`, async () => {
    const ui = await page(snapshot({ objects: [{ ...sceneObjects[0], kind }] }));
    const focusedInput = ui.vectorInput('object-editor', 1, 0);
    ui.inputSize(0, '41');
    const expectedSize = kind === 'cube' || kind === 'sphere' ? [.041, .041, .041]
      : kind === 'cylinder' ? [.041, .041, .04] : [.041, .04, .04];
    assert.deepEqual(ui.view.draft.objects[0].size, expectedSize);
    assert.deepEqual([0, 1, 2].map(axis => Number(ui.vectorInput('object-editor', 1, axis).value)), expectedSize.map(value => value * 1000));
    assert.equal(ui.vectorInput('object-editor', 1, 0), focusedInput);
    ui.inputSize(2, '42');
    const afterHeight = kind === 'cube' || kind === 'sphere' ? [.042, .042, .042]
      : kind === 'cylinder' ? [.041, .041, .042] : [.041, .04, .042];
    assert.deepEqual(ui.view.draft.objects[0].size, afterHeight);
    assert.deepEqual(ui.requests(), []);
    ui.click('apply-scene');
    assert.deepEqual(Array.from(ui.takeRequest('/api/scene').body.objects[0].size), afterHeight);
  });
}

for (const invalid of ['', 'Infinity', 'NaN']) {
  for (const field of ['position', 'size', 'yaw', 'table position', 'table size']) {
    test(`invalid ${field} value ${JSON.stringify(invalid)} is rejected before Apply`, async () => {
      const ui = await page(snapshot({ objects: sceneObjects }));
      if (field === 'position') ui.inputPosition(0, invalid);
      else if (field === 'size') ui.inputSize(0, invalid);
      else if (field === 'yaw') ui.inputYaw(invalid);
      else ui.inputTable(field === 'table position' ? 0 : 1, 0, invalid);
      ui.click('apply-scene');
      assert.deepEqual(ui.requests(), []);
      assert.equal(ui.element('error-banner').hidden, false);
      assert.match(ui.element('error-message').textContent, /必须为.*数值/);
    });
  }
}

test('dragging multiple objects and editing XYZ share one draft and wait for Apply', async () => {
  const ui = await page(snapshot({ objects: sceneObjects }));
  ui.drag('black_cube', [.6, .2, .8]);
  assert.deepEqual(ui.requests(), []);
  assert.equal(ui.element('apply-scene').disabled, false);
  assert.equal(ui.element('start').disabled, true);
  assert.equal(ui.view.interaction.enabled, true);
  assert.deepEqual(ui.view.draft.objects[0].position, [.6, .2, .8]);

  ui.message(snapshot({ serverTime: 1001, objects: sceneObjects }));
  assert.deepEqual(ui.view.draft.objects[0].position, [.6, .2, .8]);
  ui.drag('red_box', [.7, -.2, .9]);
  ui.inputPosition(0, '750');
  assert.equal(Number(ui.view.draft.objects[1].position[0]), .75);
  assert.deepEqual(ui.requests(), []);

  ui.click('apply-scene');
  const apply = ui.takeRequest('/api/scene');
  assert.deepEqual(Array.from(apply.body.objects[0].position), [.6, .2, .8]);
  assert.deepEqual(Array.from(apply.body.objects[1].position), [.75, -.2, .9]);
  assert.equal(ui.view.interaction.enabled, false);
  apply.resolve({ generation: 2 });
  await settle();
  ui.takeRequest('/api/state').resolve(snapshot({ generation: 2, serverTime: 1002, objects: apply.body.objects }));
  await settle();
  assert.equal(ui.view.draft, null);
  assert.equal(ui.element('apply-scene').disabled, true);
});

for (const action of ['apply-scene', 'reset']) {
  test(`a rejected ${action} preserves draft coordinates and preview`, async () => {
    const ui = await page(snapshot({ objects: sceneObjects }));
    ui.drag('black_cube', [.6, .2, .8]);
    ui.click(action);
    ui.takeRequest(action === 'reset' ? '/api/reset' : '/api/scene').reject(new Error('request_rejected'));
    await settle();
    ui.message(snapshot({ serverTime: 1001, objects: sceneObjects }));
    assert.deepEqual(ui.view.draft.objects[0].position, [.6, .2, .8]);
    assert.equal(ui.view.interaction.enabled, true);
    assert.equal(ui.element('apply-scene').disabled, false);
  });
}

test('successful Reset clears unapplied scene changes and adopts restored positions', async () => {
  const ui = await page(snapshot({ objects: sceneObjects }));
  ui.drag('black_cube', [.6, .2, .8]);
  ui.click('reset');
  ui.takeRequest('/api/reset').resolve({ generation: 2 });
  await settle();
  ui.takeRequest('/api/state').resolve(snapshot({ generation: 2, serverTime: 1001, objects: sceneObjects }));
  await settle();
  assert.equal(ui.view.draft, null);
  assert.equal(ui.element('apply-scene').disabled, true);
  ui.inputPosition(0, '450');
  assert.deepEqual(ui.view.draft.objects[0].position.map(Number), [.45, .15, .62]);
});

for (const action of ['apply-scene', 'reset']) {
  test(`a successful ${action} clears its draft even if the state refresh fails`, async () => {
    const ui = await page(snapshot({ objects: sceneObjects }));
    ui.drag('black_cube', [.6, .2, .8]);
    ui.click(action);
    const pending = ui.takeRequest(action === 'reset' ? '/api/reset' : '/api/scene');
    const accepted = snapshot({ generation: 2, objects: action === 'reset' ? sceneObjects : pending.body.objects }).simulation;
    pending.resolve(accepted);
    await settle();
    ui.takeRequest('/api/state').reject(new Error('network_unavailable'));
    await settle();
    assert.equal(ui.view.draft, null);
    assert.equal(ui.element('apply-scene').disabled, true);
    ui.inputPosition(0, '700');
    const expected = action === 'reset' ? [.7, .15, .62] : [.7, .2, .8];
    assert.deepEqual(ui.view.draft.objects[0].position.map(Number), expected);
  });
}

test('task snapshots disable dragging while leaving physics updates active', async () => {
  const ui = await page(snapshot({ objects: sceneObjects }));
  assert.equal(ui.view.interaction.enabled, true);
  ui.message(snapshot({ serverTime: 1001, objects: sceneObjects, run: { status: 'running' } }));
  assert.equal(ui.view.interaction.enabled, false);
  assert.equal(ui.element('scene-fields').disabled, true);
});

for (const response of ['resolved', 'rejected']) {
  test(`a superseded Start ${response} response cannot undo the successful Reset clock`, async () => {
    const ui = await page();
    ui.click('start');
    const oldStart = ui.takeRequest('/api/start');
    ui.advance(2000);
    ui.click('reset');
    ui.takeRequest('/api/reset').resolve({ generation: 2 });
    await settle();
    assert.equal(ui.seconds(), 0);

    if (response === 'resolved') oldStart.resolve({ status: 'running' });
    else oldStart.reject(new Error('stale_start_after_reset'));
    await settle();
    ui.advance(1000);
    assert.equal(ui.seconds(), 0);
    ui.takeRequest('/api/state').resolve(snapshot({ generation: 2, serverTime: 1003 }));
    await settle();
    assert.equal(ui.seconds(), 0);
  });

  test(`an older Start ${response} response cannot cancel a subsequent Start after Reset`, async () => {
    const ui = await page();
    ui.click('start');
    const oldStart = ui.takeRequest('/api/start');
    ui.click('reset');
    ui.takeRequest('/api/reset').resolve({ generation: 2 });
    await settle();
    ui.takeRequest('/api/state').resolve(snapshot({ generation: 2, serverTime: 1002 }));
    await settle();

    ui.click('start');
    const newStart = ui.takeRequest('/api/start');
    assert.equal(newStart.body.generation, 2);
    ui.advance(500);
    if (response === 'resolved') oldStart.resolve({ status: 'running' });
    else oldStart.reject(new Error('stale_start_after_reset'));
    await settle();
    assert.equal(ui.seconds(), .5);
    ui.advance(500);
    assert.equal(ui.seconds(), 1);
    newStart.reject(new Error('start_rejected'));
    await settle();
    assert.equal(ui.seconds(), 0);
  });
}

test('a rejected Start restores the previously displayed terminal duration', async () => {
  const ui = await page(snapshot({ run: { status: 'failed', run_id: 'previous', metrics: { run_duration_s: 7.25 } } }));
  assert.equal(ui.seconds(), 7.25);
  ui.click('start');
  ui.advance(500);
  assert.equal(ui.seconds(), .5);
  ui.takeRequest('/api/start').reject(new Error('start_rejected'));
  await settle();
  assert.equal(ui.seconds(), 7.25);
});

test('a successful Start keeps timing when its follow-up state request fails', async () => {
  const ui = await page();
  ui.click('start');
  ui.advance(500);
  ui.takeRequest('/api/start').resolve({ status: 'running', run_id: 'new', generation: 1 });
  await settle();
  ui.takeRequest('/api/state').reject(new Error('network_unavailable'));
  await settle();
  ui.advance(1000);
  assert.equal(ui.seconds(), 1.5);
});

test('camera and viewport task clocks stay synchronized while idle, running, finished and reset', async () => {
  const ui = await page();
  ui.advance(5000);
  assert.equal(ui.element('camera-task-time').textContent, '0.00');
  const run = { status: 'running', run_id: 'clock', generation: 1 };
  const events = [{ id: 1, kind: 'run_start', time: 1000, data: {} }];
  ui.message(snapshot({ serverTime: 1002, run, events }));
  ui.advance(1500);
  assert.equal(ui.element('camera-task-time').textContent, '3.50');
  assert.equal(ui.element('camera-task-time').textContent, ui.element('sim-time').textContent);
  ui.message(snapshot({ serverTime: 1005, run: { ...run, status: 'succeeded', metrics: { run_duration_s: 4.25 } } }));
  ui.advance(5000);
  assert.equal(ui.element('camera-task-time').textContent, '4.25');
  assert.equal(ui.element('camera-task-time').textContent, ui.element('sim-time').textContent);
  ui.message(snapshot({ generation: 2, serverTime: 1010 }));
  ui.advance(5000);
  assert.equal(ui.element('camera-task-time').textContent, '0.00');
  assert.equal(ui.element('camera-task-time').textContent, ui.element('sim-time').textContent);
});

test('a rejected Reset preserves the running clock', async () => {
  const run = { status: 'running', run_id: 'previous', generation: 1 };
  const events = [{ id: 1, kind: 'run_start', time: 990, data: {} }];
  const ui = await page(snapshot({ run, events }));
  ui.click('reset');
  ui.takeRequest('/api/reset').reject(new Error('reset_rejected'));
  await settle();
  ui.advance(1000);
  assert.equal(ui.seconds(), 11);
});

test('older generation or timestamp snapshots cannot revive a Reset task clock', async () => {
  const run = { status: 'running', run_id: 'previous', generation: 1 };
  const events = [{ id: 1, kind: 'run_start', time: 990, data: {} }];
  const ui = await page(snapshot({ run, events }));
  assert.equal(ui.seconds(), 10);
  ui.click('reset');
  ui.takeRequest('/api/reset').resolve({ generation: 2 });
  await settle();
  ui.takeRequest('/api/state').resolve(snapshot({ generation: 2, serverTime: 1002 }));
  await settle();

  ui.message(snapshot({ generation: 1, serverTime: 1003, run, events }));
  ui.message(snapshot({ generation: 2, serverTime: 1001, run, events }));
  ui.advance(1000);
  assert.equal(ui.seconds(), 0);
});

test('a new backend instance reloads once before stale generation and timestamp checks', async () => {
  const ui = await page(snapshot({ generation: 5, instanceId: 'first' }));
  ui.message(snapshot({ generation: 1, serverTime: 900, instanceId: 'second', run: { status: 'failed' } }));
  assert.equal(ui.reloads(), 1);
  assert.equal(ui.element('run-status').textContent, '场景就绪');
  ui.message(snapshot({ generation: 2, instanceId: 'second' }));
  ui.message(snapshot({ generation: 99, instanceId: 'first', run: { status: 'failed' } }));
  assert.equal(ui.reloads(), 1);
  assert.equal(ui.element('run-status').textContent, '场景就绪');
});

test('reconnecting to the same backend keeps stale generation and timestamp protection', async () => {
  const ui = await page(snapshot({ generation: 5, instanceId: 'first' }));
  ui.connection('close');
  ui.connection('open');
  ui.message(snapshot({ generation: 1, serverTime: 1001, instanceId: 'first', run: { status: 'failed' } }));
  ui.message(snapshot({ generation: 5, serverTime: 999, instanceId: 'first', run: { status: 'failed' } }));
  assert.equal(ui.reloads(), 0);
  assert.equal(ui.element('run-status').textContent, '场景就绪');
  ui.message(snapshot({ generation: 6, serverTime: 1001, instanceId: 'first', run: { status: 'failed' } }));
  assert.equal(ui.element('run-status').textContent, '任务失败');
});

test('missing or invalid backend identity stays compatible without clearing a known instance', async () => {
  const ui = await page(snapshot());
  ui.message(snapshot({ instanceId: 'first' }));
  for (const instanceId of [undefined, null, '', 123]) {
    ui.message(snapshot({ instanceId, run: { status: 'failed' } }));
    assert.equal(ui.reloads(), 0);
    assert.equal(ui.element('run-status').textContent, '任务失败');
  }
  ui.message(snapshot({ instanceId: 'second' }));
  assert.equal(ui.reloads(), 1);
});

test('planning shows its current batch, elapsed wait, and total deadline without new snapshots', async () => {
  const run = { status: 'running', run_id: 'planning', current_step: {
    skill: 'plan_grasp', started_at: 990, timeout_s: 180, batch_index: 2, batch_count: 3, candidate_indices: [17, 20], arm: 'A',
    last_error: 'no feasible approach', success_count: 1,
  } };
  const ui = await page(snapshot({ run }));
  assert.equal(ui.element('run-details').hidden, false);
  assert.match(ui.element('run-step').textContent, /plan_grasp/);
  assert.match(ui.element('run-step-timing').textContent, /10\.0 s.*180 s/);
  assert.match(ui.element('run-candidate').textContent, /2 \/ 3.*17.*20.*A.*1/);
  assert.match(ui.element('run-last-error').textContent, /批次.*no feasible approach/);
  ui.advance(2500);
  assert.match(ui.element('run-step-timing').textContent, /12\.5 s/);
});

test('current and failed Batched IK status shows counts instead of candidate ID lists', async () => {
  const current_step = { skill: 'evaluate_arms', started_at: 990, batch_index: 2, batch_count: 2,
    candidate_indices: Array.from({ length: 100 }, (_, index) => index + 100), arm: 'B', success_count: 12 };
  const ui = await page(snapshot({ run: { status: 'running', current_step } }));
  assert.match(ui.element('run-candidate').textContent, /Batched IK 批次 2 \/ 2.*候选 100 个.*Arm B.*成功 12 个/);
  assert.doesNotMatch(ui.element('run-candidate').textContent, /候选编号|100, 101/);
  ui.message(snapshot({ serverTime: 1001, run: { status: 'failed', error: 'IK disconnected',
    failed_step: { ...current_step, candidate_indices: { count: 100, first: [100, 101, 102] } } } }));
  assert.match(ui.element('run-candidate').textContent, /候选 100 个/);
  assert.match(ui.element('run-failure').textContent, /失败步骤：evaluate_arms.*IK disconnected/);
});

test('grasp and place IK stages are distinguished from trajectory planning', async () => {
  for (const skill of ['plan_grasp', 'plan_place']) {
    const current_step = { skill, stage: 'ik_batch', started_at: 990, batch_index: 1,
      candidate_indices: [0, 1, 2], arm: 'A', success_count: 0 };
    const ui = await page(snapshot({ run: { status: 'running', current_step } }));
    assert.match(ui.element('run-candidate').textContent, /Batched IK 批次.*候选 3 个/);
    ui.message(snapshot({ run: { status: 'running', current_step: {
      ...current_step, stage: 'planning_batch' } } }));
    assert.match(ui.element('run-candidate').textContent, /规划批次.*候选编号 0, 1, 2/);
  }
});

test('a failed planning snapshot keeps its cause visible and freezes the clock', async () => {
  const running = { status: 'running', run_id: 'planning', current_step: { skill: 'plan_grasp', started_at: 990 } };
  const events = [{ id: 1, kind: 'run_start', time: 980, data: {} }];
  const ui = await page(snapshot({ run: running, events }));
  ui.message(snapshot({ serverTime: 1003, events, run: {
    status: 'failed', run_id: 'planning', error: 'cuRobo service disconnected',
    failed_step: { skill: 'plan_grasp', started_at: 990, duration_s: 13 },
    metrics: { run_duration_s: 23 },
  } }));
  assert.equal(ui.element('error-banner').hidden, false);
  assert.equal(ui.element('run-status').textContent, '任务失败');
  assert.match(ui.element('run-step').textContent, /plan_grasp/);
  assert.match(ui.element('run-failure').textContent, /失败步骤：plan_grasp.*cuRobo service disconnected/);
  assert.equal(ui.element('run-failure').hidden, false);
  ui.click('dismiss-error');
  assert.equal(ui.element('run-failure').hidden, false);
  ui.advance(5000);
  assert.equal(ui.seconds(), 23);
  assert.match(ui.element('run-step-timing').textContent, /13\.0 s/);
});

test('outstanding algorithm work blocks a new Start but allows Reset and scene editing', async () => {
  const ui = await page(snapshot({ run: { status: 'idle', algorithm_pending: true } }));
  assert.equal(ui.element('start').disabled, true);
  assert.equal(ui.element('reset').disabled, false);
  assert.equal(ui.element('scene-fields').disabled, false);
  assert.equal(ui.element('algorithm-pending').hidden, false);
  assert.match(ui.element('start-hint').textContent, /后台算法.*返回/);
  ui.message(snapshot({ serverTime: 1001, run: { status: 'idle', algorithm_pending: false } }));
  assert.equal(ui.element('start').disabled, false);
  assert.equal(ui.element('algorithm-pending').hidden, true);
});

test('asynchronous cleanup failure stays visible and blocks Start until Reset cleanup succeeds', async () => {
  const ui = await page(snapshot({ run: { status: 'idle', algorithm_pending: true } }));
  const cleanup = { status: 'idle', error: null, algorithm_pending: false,
    algorithm_cleanup_error: 'RuntimeError:GPU cleanup disconnected' };
  ui.message(snapshot({ serverTime: 1001, run: cleanup }));
  assert.equal(ui.element('start').disabled, true);
  assert.equal(ui.element('reset').disabled, false);
  assert.equal(ui.element('algorithm-pending').hidden, true);
  assert.equal(ui.element('run-status').textContent, '算法清理失败');
  assert.match(ui.element('run-dot').className, /failed/);
  assert.match(ui.element('start-hint').textContent, /清理失败.*Reset/);
  assert.equal(ui.element('run-details').hidden, false);
  for (const id of ['error-details', 'run-failure']) {
    assert.match(ui.element(id).textContent, /算法状态清理失败.*Reset/);
    assert.match(ui.element(id).textContent, /GPU cleanup disconnected/);
  }
  ui.click('dismiss-error');
  ui.advance(2000);
  ui.message(snapshot({ serverTime: 1003, run: cleanup }));
  assert.equal(ui.element('run-failure').hidden, false);
  assert.equal(ui.element('start').disabled, true);
  ui.click('start');
  assert.equal(ui.requests().includes('/api/start'), false);
  ui.message(snapshot({ serverTime: 1004, run: { ...cleanup, status: 'failed', error: 'prior planning failure' } }));
  assert.match(ui.element('run-failure').textContent, /prior planning failure/);
  assert.match(ui.element('run-failure').textContent, /GPU cleanup disconnected/);
  ui.click('reset');
  ui.takeRequest('/api/reset').resolve(snapshot({ generation: 2 }).simulation);
  await settle();
  ui.takeRequest('/api/state').resolve(snapshot({ generation: 2, serverTime: 1005,
    run: { status: 'idle', algorithm_pending: true, algorithm_cleanup_error: null } }));
  await settle();
  assert.equal(ui.element('start').disabled, true);
  ui.message(snapshot({ generation: 2, serverTime: 1006,
    run: { status: 'idle', algorithm_pending: false, algorithm_cleanup_error: null } }));
  assert.equal(ui.element('start').disabled, false);
  assert.equal(ui.element('reset').disabled, false);
  assert.equal(ui.element('run-failure').hidden, true);
  assert.equal(ui.element('run-status').textContent, '场景就绪');
});

test('Reset keeps the task clock at zero while an old algorithm call still finishes', async () => {
  const ui = await page(snapshot({ run: { status: 'failed', algorithm_pending: true, run_id: 'old',
    error: 'plan timeout', metrics: { run_duration_s: 180 } } }));
  ui.click('reset');
  ui.takeRequest('/api/reset').resolve(snapshot({ generation: 2 }).simulation);
  await settle();
  assert.equal(ui.element('start').disabled, true);
  ui.takeRequest('/api/state').resolve(snapshot({ generation: 2, serverTime: 1001,
    run: { status: 'idle', algorithm_pending: true } }));
  await settle();
  ui.advance(5000);
  assert.equal(ui.seconds(), 0);
  assert.equal(ui.element('algorithm-pending').hidden, false);
  assert.equal(ui.element('run-failure').hidden, true);
});

test('an active planning call shows progress without claiming that its result is discarded', async () => {
  const ui = await page(snapshot({ run: { status: 'running', algorithm_pending: true,
    current_step: { skill: 'plan_grasp', started_at: 990, timeout_s: 180 } } }));
  assert.equal(ui.element('algorithm-pending').hidden, true);
  assert.match(ui.element('run-step').textContent, /plan_grasp/);
  assert.equal(ui.element('scene-fields').disabled, true);
});

test('planning timeouts keep a Chinese summary and the original reason in error details', async () => {
  const ui = await page(snapshot({ run: { status: 'failed', algorithm_pending: true,
    error: 'plan_grasp_timeout:180s', failed_step: { skill: 'plan_grasp', duration_s: 180 } } }));
  assert.match(ui.element('error-message').textContent, /抓取规划超过总时限/);
  for (const id of ['error-details', 'run-failure']) {
    assert.match(ui.element(id).textContent, /抓取规划超过总时限，任务已停止，迟到结果不会执行/);
    assert.match(ui.element(id).textContent, /plan_grasp_timeout:180s/);
  }
  assert.equal(ui.element('algorithm-pending').hidden, false);
});

for (const [error, explanation] of [
  ['arm_already_attempted:B', /该机械臂本轮已尝试规划.*每臂最多尝试一次/],
  ['no_reachable_arm', /JEV 未选择可执行的机械臂/],
]) {
  test(`arm selection rejection explains ${error} without claiming both arms are unreachable`, async () => {
    const ui = await page(snapshot({ run: { status: 'failed', error,
      failed_step: { skill: 'choose_arm', duration_s: 1 } } }));
    assert.match(ui.element('error-message').textContent, explanation);
    for (const id of ['error-details', 'run-failure']) {
      assert.match(ui.element(id).textContent, explanation);
      assert.ok(ui.element(id).textContent.includes(error));
      assert.doesNotMatch(ui.element(id).textContent, /两个机械臂都未找到/);
    }
  });
}

test('release collision hints use structured error codes while retaining service details', async () => {
  const ui = await page(snapshot());
  ui.message(snapshot({ serverTime: 1001, run: { status: 'failed',
    error_code: 'release_clearance_conflict', error: 'cuRobo /tools/plan_place: requested pose collides' } }));
  assert.match(ui.element('run-failure').textContent, /物体离盒底高度/);
  assert.match(ui.element('run-failure').textContent, /requested pose collides/);
});

test('a pending algorithm rejection uses the backend error code to explain the wait', async () => {
  const ui = await page();
  ui.click('start');
  ui.takeRequest('/api/start').reject(new Error('algorithm_request_pending'));
  await settle();
  assert.match(ui.element('error-message').textContent, /后台算法调用尚未返回/);
  assert.match(ui.element('error-details').textContent, /algorithm_request_pending/);
});

test('settings must load before Start and a failed fetch keeps Start unavailable', async () => {
  const ui = await page(snapshot(), { loadSettings: false });
  assert.equal(ui.element('start').disabled, true);
  ui.takeRequest('/api/planning-settings').reject(new Error('offline'));
  await settle();
  assert.equal(ui.element('start').disabled, true);
  assert.equal(ui.element('reload-planning').disabled, false);
  assert.match(ui.element('error-message').textContent, /算法参数.*offline/);
});

test('planning edits require Apply, send SI values and bind the next Start to the accepted revision', async () => {
  const ui = await page();
  ui.click('toggle-advanced-planning');
  ui.inputPlanning('ik_position_threshold_m', '8');
  ui.inputPlanning('time_scale', '3');
  assert.equal(ui.element('start').disabled, true);
  assert.deepEqual(ui.requests(), []);
  ui.click('apply-planning');
  const apply = ui.takeRequest('/api/planning-settings');
  assert.equal(apply.body.settings.ik_position_threshold_m, .008);
  assert.equal(apply.body.settings.time_scale, 3);
  assert.equal(apply.body.expected_revision, 'settings-1');
  const accepted = planningPayload(apply.body.settings, 'settings-2');
  apply.resolve(accepted);
  await settle();
  ui.takeRequest('/api/state').resolve(snapshot({ serverTime: 1001, settings: accepted }));
  await settle();
  assert.equal(ui.element('start').disabled, false);
  ui.click('start');
  assert.equal(ui.planningInput('ik_num_seeds').disabled, true);
  assert.equal(ui.takeRequest('/api/start').body.settings_revision, 'settings-2');
});

test('advanced settings require explicit unlock and relocking preserves the pending draft', async () => {
  const ui = await page();
  assert.equal(ui.planningInput('ik_num_seeds').disabled, true);
  assert.equal(ui.planningInput('trajopt_num_seeds').disabled, true);
  assert.equal(ui.planningInput('approach_axis').disabled, true);
  assert.equal(ui.planningInput('time_scale').disabled, false);
  assert.equal(ui.element('toggle-advanced-planning').disabled, false);
  assert.match(ui.element('advanced-planning-status').textContent, /锁定/);
  ui.click('toggle-advanced-planning');
  ui.inputPlanning('ik_num_seeds', '16');
  ui.click('toggle-advanced-planning');
  assert.equal(ui.planningInput('ik_num_seeds').disabled, true);
  assert.equal(ui.planningInput('ik_num_seeds').value, '16');
  assert.equal(ui.element('start').disabled, true);
  assert.equal(ui.element('apply-planning').disabled, false);
  assert.match(ui.element('planning-status').textContent, /未应用/);
  assert.deepEqual(ui.requests(), []);
  ui.click('apply-planning');
  assert.equal(ui.takeRequest('/api/planning-settings').body.settings.ik_num_seeds, 16);
});

test('execution timing can be applied while advanced settings remain locked', async () => {
  const ui = await page();
  ui.inputPlanning('time_scale', '2');
  ui.click('apply-planning');
  const apply = ui.takeRequest('/api/planning-settings');
  assert.equal(apply.body.settings.time_scale, 2);
  assert.equal(apply.body.settings.ik_num_seeds, planningSettings.ik_num_seeds);
  assert.equal(ui.planningInput('ik_num_seeds').disabled, true);
});

for (const run of [{ status: 'running' }, { status: 'failed', algorithm_pending: true }]) {
  test(`${run.status} algorithm state relocks advanced settings and requires a fresh unlock after completion`, async () => {
    const ui = await page();
    ui.click('toggle-advanced-planning');
    ui.inputPlanning('ik_num_seeds', '16');
    ui.message(snapshot({ serverTime: 1001, run }));
    assert.equal(ui.element('planning-fields').disabled, true);
    assert.equal(ui.element('toggle-advanced-planning').disabled, true);
    assert.equal(ui.planningInput('ik_num_seeds').disabled, true);
    ui.click('toggle-advanced-planning');
    assert.equal(ui.planningInput('ik_num_seeds').disabled, true);
    ui.message(snapshot({ serverTime: 1002 }));
    assert.equal(ui.element('toggle-advanced-planning').disabled, false);
    assert.equal(ui.element('planning-fields').disabled, false);
    assert.equal(ui.planningInput('ik_num_seeds').disabled, true);
    assert.equal(ui.planningInput('ik_num_seeds').value, '16');
    assert.equal(ui.element('apply-planning').disabled, false);
    assert.deepEqual(ui.requests(), []);
  });
}

test('invalid settings do not send a request and backend validation errors remain readable', async () => {
  const ui = await page();
  ui.inputPlanning('time_scale', '');
  ui.click('apply-planning');
  assert.deepEqual(ui.requests(), []);
  assert.match(ui.element('error-message').textContent, /轨迹时间倍率.*数值/);
  ui.inputPlanning('time_scale', '2');
  ui.click('apply-planning');
  ui.takeRequest('/api/planning-settings').reject(new Error(JSON.stringify([
    { loc: ['body', 'settings'], msg: 'Value error, soft activation 必须不小于 hard clearance' },
  ])));
  await settle();
  assert.match(ui.element('error-message').textContent, /soft activation 必须不小于 hard clearance/);
  assert.doesNotMatch(ui.element('error-message').textContent, /\[\{"loc"/);
  assert.equal(ui.planningInput('time_scale').value, '2');
});

test('other pages update clean settings but conflicting drafts require explicit reload', async () => {
  const ui = await page();
  ui.message(snapshot({ serverTime: 1001, settings: planningPayload({ ...planningSettings, time_scale: 2 }, 'settings-2') }));
  assert.equal(ui.planningInput('time_scale').value, '2');
  ui.inputPlanning('time_scale', '3');
  ui.message(snapshot({ serverTime: 1002, settings: planningPayload({ ...planningSettings, time_scale: 4 }, 'settings-3') }));
  assert.equal(ui.planningInput('time_scale').value, '3');
  assert.equal(ui.element('apply-planning').disabled, true);
  assert.match(ui.element('planning-status').textContent, /其他页面.*重新读取/);
  ui.click('reload-planning');
  ui.takeRequest('/api/planning-settings').resolve(planningPayload({ ...planningSettings, time_scale: 4 }, 'settings-3'));
  await settle();
  assert.equal(ui.planningInput('time_scale').value, '4');
  assert.equal(ui.element('start').disabled, false);
});

test('Reset retains applied parameters and pending algorithms lock parameter editing', async () => {
  const applied = planningPayload({ ...planningSettings, time_scale: 4 }, 'settings-2');
  const ui = await page(snapshot({ settings: applied, run: { status: 'idle', algorithm_pending: true } }));
  assert.equal(ui.element('planning-fields').disabled, true);
  assert.equal(ui.element('reload-planning').disabled, true);
  ui.click('reset');
  ui.takeRequest('/api/reset').resolve(snapshot({ generation: 2 }).simulation);
  await settle();
  ui.takeRequest('/api/state').resolve(snapshot({ generation: 2, serverTime: 1001, settings: applied }));
  await settle();
  assert.equal(ui.planningInput('time_scale').value, '4');
  assert.equal(ui.element('planning-fields').disabled, false);
});

test('a run displays its own captured planning settings after current settings change', async () => {
  const captured = { ...planningSettings, time_scale: 2 };
  const ui = await page(snapshot({ settings: planningPayload({ ...planningSettings, time_scale: 4 }, 'settings-3'),
    run: { status: 'succeeded', planning_settings: captured, settings_revision: 'settings-2' } }));
  assert.equal(ui.element('run-planning-settings').hidden, false);
  assert.match(ui.element('run-planning-settings-content').textContent, /"time_scale": 2/);
  assert.equal(ui.planningInput('time_scale').value, '4');
});

test('a server revision conflict preserves the draft and blocks another Apply until reload', async () => {
  const ui = await page();
  ui.inputPlanning('time_scale', '3');
  ui.click('apply-planning');
  ui.takeRequest('/api/planning-settings').reject(new Error('planning_settings_changed'));
  await settle();
  ui.message(snapshot({ serverTime: 1001 }));
  assert.equal(ui.planningInput('time_scale').value, '3');
  assert.equal(ui.element('apply-planning').disabled, true);
  assert.match(ui.element('planning-status').textContent, /其他页面.*重新读取/);
});

import { runConclusion } from '../src/run-conclusion.js';
import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';
import { buildExecutionGraph, createExecutionGraph, graphSvg, renderObjectTimeline } from '../src/execution-graph.js';
import { createReplayEvidence } from '../src/replay-evidence.js';
import { formatJson, traceDetails, confidenceLabel, candidateCount, createEventTitleResolver } from '../src/model.js';
import { taskProgressSummary } from '../src/run-evidence.js';
import { createObjectList } from '../src/object-list.js';
import { groupObservationFrames, modelCallInfo, inventorySnapshots, renderReplayStages, renderInventorySnapshot } from '../src/replay-story.js';

const source = (await readFile(new URL('../src/replay.js', import.meta.url), 'utf8')).replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '');
const traceSource = (await readFile(new URL('../src/trace.js', import.meta.url), 'utf8')).replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '');
const settle = () => new Promise(resolve => setImmediate(resolve));

function manifest(id, extra = {}) {
  return { id, name: id, instruction: '抓取方块', created_at: '2026-10-12T08:00:00Z', status: 'completed', retention: 'saved', duration_s: 20, record_video: false, video_status: 'disabled', size_bytes: 1200, files: [], errors: [], ...extra };
}

function page(url = 'http://localhost/replay.html') {
  const pending = [];
  const elements = new Map();
  const timers = new Map();
  const document = { createElement: tag => node(tag), createElementNS: (_ns, tag) => node(tag), getElementById: id => get(id) };
  function node(tagName = 'div') {
    const attributes = new Map();
    return {
      tagName, ownerDocument: document, children: [], textContent: '', className: '', value: '', hidden: false, disabled: false, dataset: {}, listeners: {},
      readyState: 0, duration: NaN, currentTime: 0, error: null, paused: true, style: {},
      classList: { toggle() {} },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      addEventListener(kind, listener) { this.listeners[kind] = listener; },
      setAttribute(name, value) { attributes.set(name, value); },
      getAttribute(name) { return attributes.get(name) ?? null; },
      removeAttribute(name) { attributes.delete(name); },
      querySelectorAll(selector) {
        if (selector === ':scope > details') return this.children.filter(child => child.tagName === 'details');
        const attribute = /^\[([\w-]+)\]$/.exec(selector)?.[1];
        return this.children.flatMap(child => [...(attribute && child.getAttribute(attribute) ? [child] : []), ...child.querySelectorAll(selector)]);
      },
      pause() { this.paused = true; },
      play() { this.paused = false; return Promise.resolve(); },
      load() { this.readyState = 0; this.duration = NaN; this.currentTime = 0; },
      showModal() { this.open = true; },
      close() { this.open = false; this.listeners.close?.(); },
      focus() {},
    };
  }
  function get(id) {
    if (!elements.has(id)) elements.set(id, node(id.endsWith('-video') ? 'video' : 'div'));
    return elements.get(id);
  }
  document.body = node('body');
  const window = {
    location: { href: url }, history: { replaceState(_state, _unused, next) { window.location.href = String(next); } },
    addEventListener() {}, clearTimeout(id) { timers.delete(id); }, setTimeout(callback) { timers.set(timers.size + 1, callback); return timers.size; },
  };
  const context = vm.createContext({
    document, window, URL, Error, console, buildExecutionGraph, createExecutionGraph, graphSvg, renderObjectTimeline, createReplayEvidence,
    runConclusion,
    groupObservationFrames, modelCallInfo, inventorySnapshots, renderReplayStages, renderInventorySnapshot,
    formatJson, traceDetails, confidenceLabel, candidateCount, createEventTitleResolver, taskProgressSummary, createObjectList,
    request(path, body) { return new Promise((resolve, reject) => pending.push({ path, body, resolve, reject })); },
    fetch(path, options) { return new Promise((resolve, reject) => pending.push({ path, options, resolve, reject })); },
  });
  vm.runInContext(traceSource, context, { filename: 'trace.js' });
  vm.runInContext(source, context, { filename: 'replay.js' });
  function take(path) {
    const index = pending.findIndex(item => item.path === path);
    assert.notEqual(index, -1, `Expected request ${path}`);
    return pending.splice(index, 1)[0];
  }
  async function load(manifests, events = []) {
    take('/api/experiments').resolve({ root: '/experiments', experiments: manifests });
    await settle();
    const runId = new URL(url).searchParams.get('run') ?? manifests[0].id;
    take(`/api/experiments/${runId}`).resolve(manifests.find(item => item.id === runId));
    take(`/api/experiments/${runId}/events`).resolve({ events });
    await settle();
  }
  function text(item) { return [item.textContent, ...item.children.map(text)].join(' '); }
  return { get, take, load, pending, window, text };
}

test('IK replay exposes eligibility separately and labels historical raw-only results', async () => {
  const ui = page();
  await ui.load([manifest('recorded')]);
  const container = ui.get('candidate-filter-test');
  const viewer = createReplayEvidence(container, { fileUrl: (_run, path) => path });
  const replay = { stage: 'after_ik', candidate_indices: [27],
    request: { goal_matrices_world: [[[1]]] },
    response: { items: [{ candidate_index: 0, success: true }] } };
  await viewer.show('recorded', [], { values: [{ candidate_replay: replay }] });
  assert.match(ui.text(container), /仅原始 IK/);
  assert.match(ui.text(container), /最终可用性未记录/);
  await viewer.show('recorded', [], { values: [{ candidate_replay: { ...replay,
    candidate_filters: [{ candidate_index: 27, accepted: false, reason: 'predicted_gripper_collision' }],
  } }] });
  assert.match(ui.text(container), /0 通过.*1 未通过/);
  const select = container.querySelectorAll('[aria-label]').find(node =>
    node.getAttribute('aria-label') === '选择候选编号');
  select.value = '27';
  select.listeners.change();
  assert.match(ui.text(container), /"rawStatus": "passed"/);
  assert.match(ui.text(container), /"eligible": false/);
  assert.match(ui.text(container), /predicted_gripper_collision/);
  viewer.dispose();
});

test('deep links select graph and task time while busy records prevent save/discard/export', async () => {
  const ui = page('http://localhost/replay.html?run=busy&view=graph&t=12.5');
  await ui.load([manifest('busy', { retention: 'pending', busy: true, record_video: true, video_status: 'encoding' })]);
  assert.equal(ui.get('tab-graph').getAttribute('aria-selected'), 'true');
  assert.equal(ui.get('playback-time').textContent, '00:00:12.500');
  assert.equal(ui.get('save-experiment').disabled, true);
  assert.equal(ui.get('discard-experiment').disabled, true);
  assert.equal(ui.get('review-export').getAttribute('aria-disabled'), 'true');
  assert.match(ui.get('scene-video-message').textContent, /生成/);
});

test('partial success replay presents completed and deferred counts from its terminal event', async () => {
  const ui = page();
  await ui.load([manifest('partial', { status: 'partial_success' })], [
    { id: 1, kind: 'run_end', title: '任务结束', t_s: 20, data: { task_progress: {
      completed: [{ id: 'one' }], deferred: [{ id: 'two', reason: 'grasp_verification_failed' }],
    } } },
  ]);
  assert.match(ui.text(ui.get('experiment-list')), /部分完成/);
  assert.match(ui.get('experiment-meta').textContent, /部分完成.*已完成 1 个.*暂缓 1 个.*抓取未通过验证/);
});

test('a slow previous experiment cannot replace the selected experiment', async () => {
  const ui = page();
  ui.take('/api/experiments').resolve({ root: '/experiments', experiments: [manifest('first'), manifest('second')] });
  await settle();
  const firstManifest = ui.take('/api/experiments/first');
  const firstEvents = ui.take('/api/experiments/first/events');
  ui.get('experiment-list').children[1].listeners.click();
  ui.take('/api/experiments/second').resolve(manifest('second'));
  ui.take('/api/experiments/second/events').resolve({ events: [] });
  await settle();
  firstManifest.resolve(manifest('first'));
  firstEvents.resolve({ events: [] });
  await settle();
  assert.equal(ui.get('experiment-title').textContent, 'second');
  assert.match(ui.window.location.href, /run=second/);
});

test('pending save sends the name and updates the saved controls', async () => {
  const ui = page();
  await ui.load([manifest('pending', { retention: 'pending' })]);
  ui.get('experiment-name').value = ' 左臂实验 ';
  ui.get('rename-form').listeners.submit({ preventDefault() {} });
  const save = ui.take('/api/experiments/pending/save');
  assert.equal(save.body.name, '左臂实验');
  save.resolve(manifest('pending', { name: '左臂实验' }));
  await settle();
  ui.take('/api/experiments').resolve({ experiments: [manifest('pending', { name: '左臂实验' })] });
  await settle();
  assert.equal(ui.get('experiment-title').textContent, '左臂实验');
  assert.equal(ui.get('discard-experiment').hidden, false);
  assert.equal(ui.get('save-experiment').disabled, false);
});

test('observation details expose the actual RGBD artifact and can seek both tracks', async () => {
  const ui = page();
  await ui.load([manifest('recorded', { record_video: true, video_status: 'complete', files: [{ path: 'videos/scene.mp4', size_bytes: 100 }, { path: 'videos/cameras.mp4', size_bytes: 100 }] })], [
    { id: 1, kind: 'observation_recorded', title: 'RGBD 观察', t_s: 7.5, artifact: 'artifacts/event-00001.json', data: { observation_artifact: 'observations/frame.json' } },
  ]);
  for (const name of ['scene', 'cameras']) {
    const video = ui.get(`${name}-video`);
    video.duration = 20;
    video.readyState = 1;
    video.listeners.loadedmetadata();
  }
  ui.get('execution-graph').querySelectorAll('[data-node-id]').find(node => node.getAttribute('data-node-id') === 'event:1').listeners.click();
  assert.equal(ui.get('scene-video').currentTime, 7.5);
  assert.equal(ui.get('cameras-video').currentTime, 7.5);
  assert.equal(ui.get('node-diagnostics').open, false);
  assert.match(ui.text(ui.get('node-files')), /RGBD/);
  const downloads = ui.get('node-files').children.flatMap(child => child.children).filter(child => child.href);
  assert.equal(downloads.some(link => link.href.endsWith('/observations/frame.json')), true);
});

test('discard only sends deletion after confirming the current pending record', async () => {
  const ui = page();
  await ui.load([manifest('pending', { retention: 'pending' })]);
  ui.get('discard-experiment').listeners.click();
  assert.equal(ui.pending.length, 0);
  ui.get('discard-dialog').returnValue = 'cancel';
  ui.get('discard-dialog').listeners.close();
  assert.equal(ui.pending.length, 0);
  ui.get('discard-experiment').listeners.click();
  ui.get('discard-dialog').returnValue = 'discard';
  ui.get('discard-dialog').listeners.close();
  const deletion = ui.take('/api/experiments/pending');
  assert.equal(deletion.options.method, 'DELETE');
  deletion.resolve({ ok: true, json: async () => ({ deleted: true }) });
  await settle();
  ui.take('/api/experiments').resolve({ experiments: [] });
  await settle();
  assert.equal(ui.get('experiment-content').hidden, true);
  assert.equal(new URL(ui.window.location.href).searchParams.has('run'), false);
});

test('an older refresh cannot turn a newly saved experiment back into pending', async () => {
  const ui = page();
  await ui.load([manifest('pending', { retention: 'pending' })]);
  ui.get('refresh-list').listeners.click();
  ui.take('/api/experiments').resolve({ experiments: [manifest('pending', { retention: 'pending' })] });
  await settle();
  const oldManifest = ui.take('/api/experiments/pending');
  const oldEvents = ui.take('/api/experiments/pending/events');
  ui.get('experiment-name').value = '保存后名称';
  ui.get('rename-form').listeners.submit({ preventDefault() {} });
  ui.take('/api/experiments/pending/save').resolve(manifest('pending', { name: '保存后名称' }));
  await settle();
  ui.take('/api/experiments').resolve({ experiments: [manifest('pending', { name: '保存后名称' })] });
  await settle();
  oldManifest.resolve(manifest('pending', { retention: 'pending' }));
  oldEvents.resolve({ events: [] });
  await settle();
  assert.equal(ui.get('experiment-title').textContent, '保存后名称');
  assert.equal(ui.get('discard-experiment').hidden, false);
});

test('saved archive deletion requires confirmation and clears stale file previews', async () => {
  const ui = page();
  await ui.load([manifest('saved')]);
  assert.equal(ui.get('discard-experiment').hidden, false);
  ui.get('discard-experiment').listeners.click();
  assert.equal(ui.pending.length, 0);
  ui.get('discard-dialog').returnValue = 'discard';
  ui.get('discard-dialog').listeners.close();
  const deletion = ui.take('/api/experiments/saved');
  assert.equal(deletion.options.method, 'DELETE');
  deletion.resolve({ ok: true, json: async () => ({ deleted: true, recoverable: true }) });
  await settle();
  ui.take('/api/experiments').resolve({ experiments: [] });
  await settle();
  assert.equal(ui.get('experiment-content').hidden, true);
  assert.match(ui.get('page-notice').textContent, /回收区/);
});

test('a pre-delete list refresh cannot reintroduce the deleted archive', async () => {
  const ui = page();
  await ui.load([manifest('saved')]);
  ui.get('refresh-list').listeners.click();
  const staleList = ui.take('/api/experiments');
  ui.get('discard-experiment').listeners.click();
  ui.get('discard-dialog').returnValue = 'discard';
  ui.get('discard-dialog').listeners.close();
  ui.take('/api/experiments/saved').resolve({ ok: true, json: async () => ({ deleted: true, recoverable: true }) });
  await settle();
  ui.take('/api/experiments').resolve({ experiments: [] });
  await settle();
  staleList.resolve({ experiments: [manifest('saved')] });
  await settle();
  assert.match(ui.get('experiment-count').textContent, /^0 条/);
  assert.equal(new URL(ui.window.location.href).searchParams.has('run'), false);
});

test('confirmation is bound to the selected archive and cancelling sends no deletion', async () => {
  const ui = page();
  await ui.load([manifest('first'), manifest('second')]);
  ui.get('discard-experiment').listeners.click();
  ui.get('experiment-list').children[1].listeners.click();
  ui.take('/api/experiments/second').resolve(manifest('second'));
  ui.take('/api/experiments/second/events').resolve({ events: [] });
  await settle();
  ui.get('discard-dialog').returnValue = 'discard';
  ui.get('discard-dialog').listeners.close();
  assert.equal(ui.pending.length, 0);
});

test('expanded algorithm calls keep zoom and select their exact paired evidence', async () => {
  const ui = page('http://localhost/replay.html?run=recorded&view=graph');
  await ui.load([manifest('recorded')], [
    { id: 1, kind: 'skill_start', title: 'observe', invocation_id: 's1', t_s: 1 },
    { id: 2, kind: 'algorithm_start', title: 'sam3', invocation_id: 's1', t_s: 3, data: { call_id: 'a1' } },
    { id: 3, kind: 'algorithm_complete', title: 'sam3', invocation_id: 's1', t_s: 4, data: { call_id: 'a1' } },
    { id: 4, kind: 'skill_complete', title: 'observe', invocation_id: 's1', t_s: 5 },
  ]);
  const graph = ui.get('execution-graph');
  ui.get('graph-zoom-out').listeners.click();
  const transform = graph.querySelectorAll('[data-graph-layer]')[0].getAttribute('transform');
  assert.match(transform, /scale\(0.8\)/);
  graph.querySelectorAll('[data-toggle-group]')[0].listeners.click();
  assert.equal(graph.querySelectorAll('[data-graph-layer]')[0].getAttribute('transform'), transform);
  const call = graph.querySelectorAll('[data-node-id]').find(item => item.getAttribute('data-node-id') === 'skill:s1/algorithm:a1');
  assert.ok(call);
  call.listeners.click();
  assert.match(ui.get('node-meta').textContent, /算法调用 a1.*父调用 s1.*2 条记录/);
  assert.equal(ui.get('playback-time').textContent, '00:00:03.000');
});

test('place algorithm details keep parent stage context when only the call evidence is selected', async () => {
  const ui = page('http://localhost/replay.html?run=recorded&view=graph');
  const events = [
    { id: 1, kind: 'skill_start', title: 'plan_place', invocation_id: 'place', t_s: 1 },
    { id: 2, kind: 'algorithm_start', title: 'plan_grasp_batch', invocation_id: 'place', t_s: 2, data: { call_id: 'batch', function: 'plan_grasp_batch' } },
    { id: 3, kind: 'algorithm_complete', title: 'plan_grasp_batch', invocation_id: 'place', t_s: 3, data: { call_id: 'batch' } },
    { id: 4, kind: 'skill_complete', title: 'plan_place', invocation_id: 'place', t_s: 4 },
  ];
  const original = structuredClone(events);
  await ui.load([manifest('recorded')], events);
  const graph = ui.get('execution-graph');
  graph.querySelectorAll('[data-toggle-group]')[0].listeners.click();
  const call = graph.querySelectorAll('[data-node-id]').find(item => item.getAttribute('data-node-id') === 'skill:place/algorithm:batch');
  assert.match(call.getAttribute('aria-label'), /plan_place_batch/);
  call.listeners.click();
  assert.equal(ui.get('node-title').textContent, 'plan_place_batch');
  const cards = ui.get('node-trace').children;
  assert.equal(cards.length, 2);
  for (const card of cards) assert.match(ui.text(card.children[0]), /plan_place_batch/);
  assert.match(ui.text(cards[0]), /"function": "plan_grasp_batch"/);
  assert.deepEqual(events, original);
});

function observations() {
  return [
    { id: 1, kind: 'observation_recorded', t_s: 2, data: { observation_id: 'head-1', checkpoint: 'initial', observation_artifact: 'observations/frame-0001.json' } },
    { id: 2, kind: 'observation_annotated', t_s: 3, data: { observation_id: 'head-1', object_labels: { observation_id: 'head-1', analysis_status: 'unprocessed' }, annotated_rgb: { 3: 'observations/initial-raw.svg' } } },
    { id: 3, kind: 'observation_annotated', t_s: 4, data: { observation_id: 'head-1', object_labels: { observation_id: 'head-1', analysis_status: 'analyzed' }, annotated_rgb: { 3: 'observations/initial.svg' } } },
    { id: 4, kind: 'observation_recorded', t_s: 8, data: { observation_id: 'three-2', checkpoint: 'post_place', observation_artifact: 'observations/frame-0002.json' } },
    { id: 5, kind: 'observation_annotated', t_s: 9, data: { observation_id: 'three-2', object_labels: { observation_id: 'three-2', analysis_status: 'analyzed' }, annotated_rgb: { 1: 'observations/right.svg', 2: 'observations/left.svg', 3: 'observations/head.svg' } } },
    { id: 6, kind: 'visual_decisions_call', t_s: 10, data: { observation_id: 'three-2', purpose: 'verify_destination_region', artifact: 'decisions/hash.json', elapsed_s: 1, http_status: 200 } },
  ];
}

test('observation gallery defaults to analyzed head frame and selects later three-camera evidence', async () => {
  const ui = page('http://localhost/replay.html?view=observations');
  await ui.load([manifest('recorded')], observations());
  assert.equal(ui.get('tab-observations').getAttribute('aria-selected'), 'true');
  assert.equal(ui.get('observation-select').children.length, 2);
  assert.equal(ui.get('observation-revision').children.length, 2);
  assert.match(ui.get('observation-meta').textContent, /初始观测.*00:00:02.000.*2 个标注版本/);
  assert.equal(ui.get('observation-cameras').children.length, 1);
  assert.match(ui.text(ui.get('observation-cameras')), /头部.*3/);
  ui.get('observation-select').value = 'three-2';
  ui.get('observation-select').listeners.change();
  const cameras = ui.get('observation-cameras').children;
  assert.equal(cameras.length, 3);
  assert.match(ui.get('observation-meta').textContent, /桌面放置后观察/);
  assert.match(ui.text(cameras[0]), /左腕/);
  assert.match(ui.text(cameras[1]), /头部/);
  assert.match(ui.text(cameras[2]), /右腕/);
  assert.match(ui.text(ui.get('observation-artifacts')), /模型调用.*verify_destination_region/);
  ui.get('observation-seek').listeners.click();
  assert.equal(ui.get('playback-time').textContent, '00:00:08.000');
});

test('annotation viewer zooms and reads the matching embedded original without a live camera request', async () => {
  const ui = page('http://localhost/replay.html?view=observations');
  await ui.load([manifest('recorded')], observations());
  const card = ui.get('observation-cameras').children[0];
  card.children.find(item => item.tagName === 'button').listeners.click();
  assert.equal(ui.get('observation-dialog').open, true);
  assert.equal(ui.get('observation-image').src, '/api/experiments/recorded/files/observations/initial.svg');
  ui.get('observation-zoom-in').listeners.click();
  assert.equal(ui.get('observation-image').style.width, '125%');
  ui.get('observation-raw').listeners.click();
  const request = ui.take('/api/experiments/recorded/files/observations/initial.svg');
  assert.equal(request.options, undefined);
  request.resolve({ ok: true, text: async () => '<svg><image href="data:image/png;base64,aGVsbG8="/></svg>' });
  await settle();
  assert.equal(ui.get('observation-image').src, 'data:image/png;base64,aGVsbG8=');
  assert.match(ui.get('observation-image-status').textContent, /原始 RGB/);
  assert.equal(ui.pending.length, 0);
});

test('switching archives closes old annotation and ignores an outstanding raw-image response', async () => {
  const ui = page();
  await ui.load([manifest('first'), manifest('second')], observations());
  ui.get('observation-cameras').children[0].children.find(item => item.tagName === 'button').listeners.click();
  ui.get('observation-raw').listeners.click();
  const old = ui.take('/api/experiments/first/files/observations/initial.svg');
  ui.get('experiment-list').children[1].listeners.click();
  ui.take('/api/experiments/second').resolve(manifest('second'));
  ui.take('/api/experiments/second/events').resolve({ events: [] });
  await settle();
  old.resolve({ ok: true, text: async () => '<svg><image href="data:image/png;base64,aGVsbG8="/></svg>' });
  await settle();
  assert.equal(ui.get('observation-dialog').open, false);
  assert.equal(ui.get('observation-select').disabled, true);
  assert.equal(ui.get('observation-cameras').children.length, 0);
  assert.equal(ui.get('observation-image').getAttribute('src'), null);
  assert.match(ui.get('observation-meta').textContent, /未保存/);
});

test('workflow model button loads its archived questions and never performs an inference request', async () => {
  const ui = page();
  await ui.load([manifest('recorded')], [
    { id: 1, kind: 'skill_start', title: 'verify_place', invocation_id: 'verify', t_s: 1 },
    { id: 2, kind: 'visual_decisions_call', invocation_id: 'verify', t_s: 2,
      data: { endpoint: '/v1/decisions', purpose: 'verify_release_result', artifact: 'decisions/place.json' } },
    { id: 3, kind: 'skill_complete', title: 'verify_place', invocation_id: 'verify', t_s: 3 },
  ]);
  assert.equal(ui.pending.length, 0);
  const call = ui.get('object-timeline').querySelectorAll('[aria-label]').find(item => item.getAttribute('aria-label') === 'Decisions · 确认桌面放置与松爪');
  call.listeners.click();
  ui.take('/api/experiments/recorded/files/decisions/place.json').resolve({
    endpoint: '/v1/decisions', purpose: 'verify_release_result',
    request: { input: [], questions: [{ name: 'workpiece_still_held', instructions: 'Is the object still held?' }] },
    response: { answers: [{ name: 'workpiece_still_held', type: 'predicate', probability: 0 }] },
  });
  await settle();
  assert.match(ui.text(ui.get('node-evidence')), /工件仍被夹持.*概率：0.0%/);
  assert.match(ui.text(ui.get('node-evidence')), /Is the object still held/);
  assert.equal(ui.get('node-diagnostics').open, false);
  assert.equal(ui.pending.length, 0);
});

test('inventory tab switches real snapshots and keeps permanent input and independent product IDs', async () => {
  const ui = page();
  const initial = { all_entities: { pink: { id: 'pink', display_id: 1, kind: 'object', color: 'pink' } },
    process_records: { p: { id: 'p', display_id: 1, input_entity_id: 'pink', to_table_arm: 'A', to_right_bin_arm: 'B' } }, process_order: ['p'] };
  await ui.load([manifest('recorded')], [
    { id: 14, kind: 'skill_complete', title: 'bind_inventory', data: { result: initial } },
    { id: 158, kind: 'product_created', data: { process_id: 'p', input_entity_id: 'pink', output_entity_id: 'black', product_entity: { id: 'black', kind: 'object', color: 'black' } } },
  ]);
  ui.get('tab-inventory').listeners.click();
  assert.equal(ui.get('tab-inventory').getAttribute('aria-selected'), 'true');
  assert.match(ui.text(ui.get('inventory-board')), /非完整快照.*已拆包 · 待投放 · 1.*产物：black/);
  ui.get('inventory-snapshot').value = '14';
  ui.get('inventory-snapshot').listeners.change();
  assert.match(ui.text(ui.get('inventory-board')), /完整清单快照.*待拆包 · 1.*输入：pink/);
  assert.doesNotMatch(ui.text(ui.get('inventory-board')), /产物：black/);
});

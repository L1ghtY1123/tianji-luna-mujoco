import { renderFailureReport, loadArchivedFailureReport } from '../src/failure-report.js';
import { runConclusion } from '../src/run-conclusion.js';
import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';
import { buildExecutionGraph, createExecutionGraph, graphSvg, renderObjectTimeline } from '../src/execution-graph.js';
import { createReplayEvidence } from '../src/replay-evidence.js';
import { createReplayPolicy } from '../src/replay-policy.js';
import { formatJson, traceDetails, confidenceLabel, candidateCount, createEventTitleResolver } from '../src/model.js';
import { taskProgressSummary } from '../src/run-evidence.js';
import { createObjectList } from '../src/object-list.js';
import { groupObservationFrames, modelCallInfo, inventorySnapshots, renderReplayStages, renderInventorySnapshot, observationImageSource } from '../src/replay-story.js';

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
    runConclusion, renderFailureReport, loadArchivedFailureReport, createReplayPolicy,
    groupObservationFrames, modelCallInfo, inventorySnapshots, renderReplayStages, renderInventorySnapshot, observationImageSource,
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

test('replay shows the full archived policy before any planned skill executes', async () => {
  const ui = page();
  const graph = { version: 1, steps: [{ type: 'for_each_target', max_iterations: 64,
    between_targets: [{ type: 'skill', id: 'return', skill: 'move_to_observe', arguments: {} }], body: [
      { type: 'skill', id: 'pick', skill: 'execute_grasp', arguments: { target_id: '$target' } },
      { type: 'if', condition: 'pick.success', then_steps: [
        { type: 'retry', max_attempts: 2, until: 'inspect.success', body: [
          { type: 'skill', id: 'inspect', skill: 'verify_grasp', arguments: {} }] }],
      else_steps: [{ type: 'stop', reason: 'grasp failed' }] },
      { type: 'replan', reason: 'new evidence', event_key: 'changed' },
    ] }] };
  await ui.load([manifest('recorded', { status: 'needs_attention' })], [
    { id: 3, kind: 'skill_program', t_s: 5, artifact: 'artifacts/event-00003.json', data: { steps: { count: 1 } } },
    { id: 4, kind: 'skill_start', title: 'observe_scene' },
    { id: 5, kind: 'skill_error', title: 'observe_scene', data: { error: 'visual_identity_unknown' } },
  ]);
  ui.take('/api/experiments/recorded/files/artifacts/event-00003.json').resolve(graph);
  await settle();
  const text = ui.text(ui.get('policy-plan'));
  for (const expected of ['初始编排', '逐物体循环', '64', '物体之间', 'execute_grasp', '$target',
    'pick.success', '条件成立', '条件不成立', '最多 2 次', 'verify_grasp', 'grasp failed', 'new evidence', '未见执行记录']) {
    assert.ok(text.includes(expected), expected);
  }
  assert.equal(ui.pending.some(item => item.body), false);
});

test('completed replay hydrates compacted failure evidence before rendering its graph', async () => {
  const ui = page();
  const checks = Array.from({length:12}, (_, index)=>({name:`coverage-${index}`, actual:index}));
  await ui.load([manifest('recorded',{status:'needs_attention'})],[
    {id:1,kind:'skill_program',data:{version:1,steps:[{type:'skill',id:'unpack',skill:'vla_unpack',arguments:{}}]}},
    {id:99,kind:'run_end',artifact:'artifacts/event-00099.json',data:{status:'needs_attention',
      failure_report:{summary:'coverage rejected',checks:{count:12,first:checks.slice(0,3)}}}},
  ]);
  ui.take('/api/experiments/recorded/files/artifacts/event-00099.json').resolve({
    failure_report:{summary:'coverage rejected',checks}});
  await settle();
  assert.match(ui.text(ui.get('policy-plan')),/vla_unpack.*未见执行记录/s);
  assert.match(ui.text(ui.get('experiment-errors')),/coverage-11/);
  assert.equal(ui.pending.some(item=>item.body),false);
});

test('Responses evidence displays policy_graph and retains legacy output only as raw JSON', async () => {
  const ui = page();
  await ui.load([manifest('recorded')]);
  const container = ui.get('policy-evidence-test');
  const viewer = createReplayEvidence(container, { fileUrl: (_run, path) => path });
  for (const field of ['policy_graph', 'program']) {
    await viewer.show('recorded', [], { values: [{ endpoint: '/v1/responses', request: { input: [] }, response: { output: [
      { type: 'message', content: [{ type: 'output_text', text: JSON.stringify({ task: {}, entities: [],
        [field]: { version: 1, steps: [{ type: 'skill', id: 'work', skill: 'custom_skill', arguments: {} }] } }) }] },
    ] } }] });
    assert.match(ui.text(container), /custom_skill/);
    if (field === 'program') {
      assert.match(ui.text(container), /不重建历史编排/);
      assert.doesNotMatch(ui.text(container), /调用 custom_skill/);
    }
  }
  viewer.dispose();
});

test('policy revisions keep node outcomes within their version and aggregate repeated execution', async () => {
  const ui = page();
  await ui.load([manifest('recorded')]);
  const container = ui.get('policy-test');
  const viewer = createReplayPolicy(container, { request: () => { throw Error('unexpected fetch'); }, fileUrl: (_run, path) => path });
  const graph = { version: 1, steps: [{ type: 'skill', id: 'same', skill: 'custom_skill', arguments: {} }] };
  await viewer.show('recorded', [
    { id: 1, kind: 'skill_program', data: graph },
    { id: 2, kind: 'policy_node', data: { node_id: 'same', status: 'failure' } },
    { id: 3, kind: 'policy_revised', data: graph },
    { id: 4, kind: 'policy_node', data: { node_id: 'same', status: 'success' } },
    { id: 5, kind: 'policy_node', data: { node_id: 'same', status: 'unknown' } },
  ]);
  assert.match(ui.text(container), /已记录执行 1 次 · 最近失败/);
  const select = container.children[0];
  select.value = '3'; select.listeners.change();
  assert.match(ui.text(container), /已记录执行 2 次 · 最近结果不确定/);
  assert.doesNotMatch(ui.text(container), /最近失败/);
});

test('late archived policy cannot overwrite another experiment and missing plans stay explicit', async () => {
  const ui = page();
  await ui.load([manifest('recorded')]);
  const container = ui.get('policy-race-test');
  let resolve;
  const viewer = createReplayPolicy(container, { request: () => new Promise(done => { resolve = done; }), fileUrl: (_run, path) => path });
  const first = viewer.show('first', [{ id: 1, kind: 'skill_program', artifact: 'plan.json' }]);
  await viewer.show('second', []);
  resolve({ version: 1, steps: [{ type: 'skill', id: 'old', skill: 'obsolete', arguments: {} }] });
  await first;
  assert.match(ui.text(container), /尚未归档 Responses 编排/);
  assert.doesNotMatch(ui.text(container), /obsolete/);
});

test('raw Responses fallback is a proposal and cannot inherit execution status', async () => {
  const ui = page();
  await ui.load([manifest('recorded')]);
  const container = ui.get('policy-proposal-test');
  const viewer = createReplayPolicy(container, { fileUrl: (_run, path) => path, request: async () => ({
    endpoint: '/v1/responses', response: { output: [{ type: 'message', content: [{ type: 'output_text',
      text: JSON.stringify({ policy_graph: { version: 1, steps: [{ type: 'skill', id: 'planned', skill: 'inspect', arguments: {} }] } }) }] }] },
  }) });
  await viewer.show('recorded', [
    { id: 1, kind: 'visual_decisions_call', data: { endpoint: '/v1/responses', artifact: 'response.json' } },
    { id: 2, kind: 'policy_node', data: { node_id: 'planned', status: 'success' } },
  ]);
  assert.match(ui.text(container), /模型提议（未见采纳记录）.*inspect/);
  assert.doesNotMatch(ui.text(container), /最近成功/);
});

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

test('partial success replay presents generic input completion counts from its terminal event', async () => {
  const ui = page();
  await ui.load([manifest('partial', { status: 'partial_success' })], [
    { id: 1, kind: 'run_end', title: '任务结束', t_s: 20, data: { task_progress: {
      selected_input_ids: ['one', 'two'], completed_input_ids: ['one'], product_ids: [], transformations: {},
    } } },
  ]);
  assert.match(ui.text(ui.get('experiment-list')), /部分完成/);
  assert.match(ui.get('experiment-meta').textContent, /部分完成.*已完成 1 . 2 个目标/);
});

test('a slow previous experiment cannot replace the selected experiment', async () => {
  const ui = page();
  ui.take('/api/experiments').resolve({ root: '/experiments', experiments: [manifest('first'), manifest('second')] });
  await settle();
  const firstManifest = ui.take('/api/experiments/first');
  const firstEvents = ui.take('/api/experiments/first/events');
  ui.get('experiment-list').children[1].children[1].listeners.click();
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
  assert.equal(ui.get('experiment-list').querySelectorAll('[data-select-run]')[0].disabled, false);
  assert.equal(ui.get('refresh-list').disabled, false);
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

function selectArchive(ui, id) {
  const checkbox = ui.get('experiment-list').querySelectorAll('[data-select-run]')
    .find(node => node.getAttribute('data-select-run') === id);
  assert.ok(checkbox, `Selection checkbox for ${id}`);
  checkbox.checked = true;
  checkbox.listeners.change();
  return checkbox;
}

test('batch selection excludes busy records and resets when the search changes', async () => {
  const ui = page();
  await ui.load([manifest('first'), manifest('second'), manifest('busy', { busy: true })]);
  assert.equal(ui.get('delete-selected').disabled, true);
  const busy = ui.get('experiment-list').querySelectorAll('[data-select-run]')
    .find(node => node.getAttribute('data-select-run') === 'busy');
  assert.equal(busy.disabled, true);
  selectArchive(ui, 'first');
  assert.equal(ui.get('select-all-experiments').indeterminate, true);
  ui.get('select-all-experiments').checked = true;
  ui.get('select-all-experiments').listeners.change();
  assert.match(ui.get('selected-count').textContent, /2/);
  ui.get('experiment-search').value = 'second';
  ui.get('experiment-search').listeners.input();
  assert.match(ui.get('selected-count').textContent, /0/);
  ui.get('select-all-experiments').checked = true;
  ui.get('select-all-experiments').listeners.change();
  ui.get('delete-selected').listeners.click();
  assert.match(ui.text(ui.get('batch-discard-names')), /second/);
  assert.doesNotMatch(ui.text(ui.get('batch-discard-names')), /first|busy/);
  ui.get('batch-discard-dialog').returnValue = 'cancel';
  ui.get('batch-discard-dialog').listeners.close();
  assert.equal(ui.pending.length, 0);
});

test('batch deletion is serial, clears a deleted active replay and keeps failed items selected', async () => {
  const ui = page();
  await ui.load([manifest('first'), manifest('second'), manifest('third')]);
  for (const id of ['first', 'second', 'third']) selectArchive(ui, id);
  ui.get('delete-selected').listeners.click();
  assert.equal(ui.pending.length, 0);
  assert.match(ui.get('batch-discard-count').textContent, /3/);
  ui.get('batch-discard-dialog').returnValue = 'discard';
  ui.get('batch-discard-dialog').listeners.close();
  const first = ui.take('/api/experiments/first');
  assert.equal(first.options.method, 'DELETE');
  assert.equal(ui.pending.length, 0);
  assert.equal(ui.get('delete-selected').disabled, true);
  assert.equal(ui.get('experiment-search').disabled, true);
  ui.get('batch-discard-dialog').listeners.close();
  assert.equal(ui.pending.length, 0);
  first.resolve({ ok: true, json: async () => ({ deleted: true }) });
  await settle();
  ui.take('/api/experiments/second').resolve({ ok: false, json: async () => ({ detail: 'experiment_in_use' }) });
  await settle();
  ui.take('/api/experiments/third').resolve({ ok: true, json: async () => ({ deleted: true }) });
  await settle();
  ui.take('/api/experiments').resolve({ experiments: [manifest('second')] });
  await settle();
  assert.equal(ui.get('experiment-content').hidden, true);
  assert.equal(new URL(ui.window.location.href).searchParams.has('run'), false);
  assert.match(ui.get('page-notice').textContent, /2.*3/);
  assert.match(ui.get('page-error').textContent, /second.*下载或导出/);
  assert.match(ui.get('selected-count').textContent, /1/);
  assert.equal(ui.get('experiment-list').querySelectorAll('[data-select-run]')[0].checked, true);
  assert.equal(ui.get('delete-selected').disabled, false);
});

test('batch deletion leaves an unselected active replay open and rejects stale list responses', async () => {
  const ui = page();
  await ui.load([manifest('keep'), manifest('remove')]);
  ui.get('refresh-list').listeners.click();
  const stale = ui.take('/api/experiments');
  selectArchive(ui, 'remove');
  ui.get('delete-selected').listeners.click();
  ui.get('batch-discard-dialog').returnValue = 'discard';
  ui.get('batch-discard-dialog').listeners.close();
  ui.take('/api/experiments/remove').resolve({ ok: true, json: async () => ({ deleted: true }) });
  await settle();
  ui.take('/api/experiments').resolve({ experiments: [manifest('keep')] });
  await settle();
  stale.resolve({ experiments: [manifest('keep'), manifest('remove')] });
  await settle();
  // The old refresh may still refresh the open replay, but cannot restore a deleted list row.
  assert.match(ui.get('experiment-count').textContent, /^1 条/);
  assert.equal(ui.get('experiment-content').hidden, false);
  assert.equal(new URL(ui.window.location.href).searchParams.get('run'), 'keep');
});

test('batch deletion does not invalidate a different replay still loading', async () => {
  const ui = page();
  await ui.load([manifest('remove'), manifest('loading')]);
  selectArchive(ui, 'remove');
  ui.get('experiment-list').children[1].children[1].listeners.click();
  const detail = ui.take('/api/experiments/loading');
  const events = ui.take('/api/experiments/loading/events');
  ui.get('delete-selected').listeners.click();
  ui.get('batch-discard-dialog').returnValue = 'discard';
  ui.get('batch-discard-dialog').listeners.close();
  ui.take('/api/experiments/remove').resolve({ ok: true, json: async () => ({ deleted: true }) });
  await settle();
  ui.take('/api/experiments').resolve({ experiments: [manifest('loading')] });
  await settle();
  detail.resolve(manifest('loading'));
  events.resolve({ events: [] });
  await settle();
  assert.equal(ui.get('experiment-content').hidden, false);
  assert.equal(ui.get('experiment-title').textContent, 'loading');
});

test('failed batch deletion restores the selected replay if it was still loading', async () => {
  const ui = page();
  await ui.load([manifest('keep'), manifest('loading')]);
  selectArchive(ui, 'loading');
  ui.get('experiment-list').children[1].children[1].listeners.click();
  const detail = ui.take('/api/experiments/loading');
  const events = ui.take('/api/experiments/loading/events');
  ui.get('delete-selected').listeners.click();
  ui.get('batch-discard-dialog').returnValue = 'discard';
  ui.get('batch-discard-dialog').listeners.close();
  ui.take('/api/experiments/loading').resolve({ ok: false, json: async () => ({ detail: 'experiment_in_use' }) });
  await settle();
  ui.take('/api/experiments').resolve({ experiments: [manifest('keep'), manifest('loading')] });
  await settle();
  detail.resolve(manifest('loading', { name: 'stale' }));
  events.resolve({ events: [] });
  ui.take('/api/experiments/loading').resolve(manifest('loading'));
  ui.take('/api/experiments/loading/events').resolve({ events: [] });
  await settle();
  assert.equal(ui.get('experiment-content').hidden, false);
  assert.equal(ui.get('experiment-title').textContent, 'loading');
  assert.match(ui.get('page-error').textContent, /下载或导出/);
  assert.match(ui.get('selected-count').textContent, /1/);
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
  ui.get('experiment-list').children[1].children[1].listeners.click();
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
  assert.match(ui.get('observation-meta').textContent, /场景观测.*00:00:02.000.*2 个图像版本/);
  assert.match(ui.get('observation-source').textContent, /事件 #1.*head-1/);
  assert.match(ui.get('observation-analysis').textContent, /未保存.*身份标注/);
  assert.equal(ui.get('observation-cameras').children.length, 1);
  assert.match(ui.text(ui.get('observation-cameras')), /头部.*3/);
  ui.get('observation-select').value = 'three-2';
  ui.get('observation-select').listeners.change();
  const cameras = ui.get('observation-cameras').children;
  assert.equal(cameras.length, 3);
  assert.match(ui.get('observation-meta').textContent, /放置后采集/);
  assert.match(ui.text(cameras[0]), /左腕/);
  assert.match(ui.text(cameras[1]), /头部/);
  assert.match(ui.text(cameras[2]), /右腕/);
  assert.match(ui.text(ui.get('observation-artifacts')), /模型调用.*verify_destination_region/);
  ui.get('observation-seek').listeners.click();
  assert.equal(ui.get('playback-time').textContent, '00:00:08.000');
});

test('observation page distinguishes a raw visual verification from missing analysis and missing capture time', async () => {
  const ui = page('http://localhost/replay.html?view=observations');
  await ui.load([manifest('recorded')], [
    { id: 1, kind: 'skill_start', title: 'verify_place', invocation_id: 'verify' },
    { id: 2, kind: 'observation_recorded', invocation_id: 'verify', t_s: 4,
      data: { observation_id: 'release', checkpoint: 'post_place' } },
    { id: 3, kind: 'observation_annotated', t_s: 4.1, data: { observation_id: 'release',
      annotated_rgb: { 3: 'observations/raw.svg' }, object_labels: { analysis_status: 'unprocessed', items: [] } } },
    { id: 4, kind: 'visual_decisions_call', data: { observation_id: 'release', endpoint: '/v1/decisions', purpose: 'verify_release_result' } },
    { id: 5, kind: 'observation_annotated', t_s: 7, data: { observation_id: 'orphan', annotated_rgb: { 3: 'observations/unknown.svg' } } },
  ]);
  assert.match(ui.get('observation-select').children[0].textContent, /第 1 次采集.*放置结果核验/);
  assert.match(ui.get('observation-source').textContent, /verify_place.*#2/);
  assert.match(ui.get('observation-analysis').textContent, /未保存.*身份标注.*Decisions.*确认放置/);
  ui.get('observation-select').value = 'orphan';
  ui.get('observation-select').listeners.change();
  assert.equal(ui.get('observation-seek').disabled, true);
  assert.match(ui.get('observation-source').textContent, /采集记录缺失/);
});

test('annotation viewer zooms and reads the matching embedded original without a live camera request', async () => {
  const ui = page('http://localhost/replay.html?view=observations');
  await ui.load([manifest('recorded')], observations());
  const card = ui.get('observation-cameras').children[0];
  card.children.find(item => item.tagName === 'button').listeners.click();
  assert.equal(ui.get('observation-dialog').open, true);
  assert.notEqual(ui.get('observation-image').src, '/api/experiments/recorded/files/observations/initial.svg');
  ui.get('observation-zoom-in').listeners.click();
  assert.equal(ui.get('observation-image').style.width, '125%');
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
  ui.get('experiment-list').children[1].children[1].listeners.click();
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
  const call = ui.get('object-timeline').querySelectorAll('[aria-label]').find(item => item.getAttribute('aria-label') === 'Decisions · 确认放置与松爪');
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

test('historical inventory remains raw and product events do not create snapshots', async () => {
  const ui = page();
  const initial = { all_entities: { raw: { id: 'raw' } }, process_records: { p: { input_entity_id: 'raw' } } };
  await ui.load([manifest('recorded')], [
    { id: 14, kind: 'inventory_snapshot', data: { task_progress: initial } },
    { id: 158, kind: 'product_created', data: { process_id: 'p', output_entity_id: 'product', product_entity: { id: 'product' } } },
  ]);
  ui.get('tab-inventory').listeners.click();
  assert.match(ui.text(ui.get('inventory-board')), /历史清单仅保留原始记录/);
  assert.match(ui.text(ui.get('inventory-board')), /process_records/);
  assert.doesNotMatch(ui.text(ui.get('inventory-board')), /待拆包|到右盒|product/);
  assert.equal(ui.get('inventory-snapshot').children.length, 1);
});

test('replay renders the archived failure report with readable checks and original references', async () => {
  const ui = page();
  await ui.load([manifest('recorded', { status: 'failed' })], [{ id: 12, kind: 'run_end', data: { status: 'failed',
    failure_report: { summary: '对象身份未确认', component: 'Decisions',
      checks: [{ name: 'identity', actual: .4, threshold: .8, comparison: '>=', question: '物体是否匹配？' }],
      evidence_refs: [{ event_id: 7, artifact: 'decisions/7.json' }], unknowns: ['无法确认本次候选'] } } }]);
  assert.match(ui.text(ui.get('experiment-errors')), /对象身份未确认.*实际值：0.4.*要求：>= 0.8.*7.json/s);
  assert.equal(ui.pending.length, 0);
});

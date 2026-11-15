import { renderFailureReport, loadArchivedFailureReport } from './failure-report.js';
import { request } from './api.js';
import { createTraceList } from './trace.js';
import { buildExecutionGraph } from './execution-graph.js';
import { taskProgressSummary } from './run-evidence.js';
import { createObjectList } from './object-list.js';
import { createReplayEvidence } from './replay-evidence.js';
import { createReplayPolicy } from './replay-policy.js';
import { SKILL_LABELS } from './replay-labels.js';
import { modelCallInfo, objectsSnapshots, renderReplayStages, renderObjectMemorySnapshot } from './replay-story.js';

export function clampReplayTime(value, duration = Infinity) {
  const time = Number(value);
  const limit = Number.isFinite(duration) ? Math.max(0, duration) : Infinity;
  return Number.isFinite(time) ? Math.min(limit, Math.max(0, time)) : 0;
}

export function formatReplayTime(value) {
  const total = Math.round(clampReplayTime(value) * 1000);
  const hours = Math.floor(total / 3600000);
  const minutes = Math.floor(total / 60000) % 60;
  const seconds = Math.floor(total / 1000) % 60;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(total % 1000).padStart(3, '0')}`;
}

export function synchronizeVideoTimes(videos, taskTime, tolerance = 0) {
  for (const video of videos) {
    if (!video.readyState || !Number.isFinite(video.duration)) continue;
    const target = clampReplayTime(taskTime, video.duration);
    if (Math.abs(video.currentTime - target) > tolerance) video.currentTime = target;
  }
}

function validFilePath(path) {
  return typeof path === 'string' && path.length > 0 && !path.includes('\\') && !path.includes(':') && !path.includes('\0') && path.split('/').every(part => part && part !== '.' && part !== '..');
}

export function fileUrl(runId, path) {
  if (!validFilePath(path)) throw new Error('无效的实验文件路径');
  return `/api/experiments/${encodeURIComponent(runId)}/files/${path.split('/').map(encodeURIComponent).join('/')}`;
}

export function artifactReferences(value) {
  const paths = new Set();
  function visit(item) {
    if (typeof item === 'string' && item.endsWith('.npz') && validFilePath(item)) paths.add(item);
    else if (Array.isArray(item)) item.forEach(visit);
    else if (item && typeof item === 'object') Object.values(item).forEach(visit);
  }
  visit(value);
  return [...paths];
}

const REPLAY_VIEWS = ['graph', 'video', 'observations', 'visual', 'collision', 'data'];

export function decisionObservations(events = []) {
  const records = new Map(events.filter(event => event.kind === 'observation_recorded')
    .map(event => [event.data?.observation_id, event]));
  const observations = new Map();
  for (const event of events) {
    if (event.kind !== 'visual_decisions_call' || event.data?.endpoint !== '/v1/decisions'
        || !validFilePath(event.data.artifact)) continue;
    const id = event.data.observation_id ?? `request-${event.id}`;
    if (!observations.has(id)) {
      const record = records.get(id);
      observations.set(id, {id, time: record?.t_s ?? event.t_s, calls: []});
    }
    observations.get(id).calls.push(event);
  }
  return [...observations.values()];
}

const RESULT_NAMES = { running: '执行中', completed: '完成', complete: '完成', succeeded: '成功', success: '成功', partial_success: '部分完成', needs_attention: '结果待确认', failed: '失败', error: '失败', cancelled: '已取消', canceled: '已取消', interrupted: '记录已中断', stopped: '已停止', unknown: '结果待确认' };
const VIDEO_NAMES = { recording: '视频录制中', encoding: '视频编码中', pending: '等待视频处理', complete: '视频编码完成', completed: '视频编码完成', ready: '视频可用', saved: '视频已保存', failed: '视频录制或编码失败', error: '视频录制或编码失败', disabled: '本次未录制视频', off: '本次未录制视频', partial: '视频不完整', incomplete: '视频不完整', interrupted: '视频录制已中断', unavailable: '视频不可用' };

function formatSize(bytes) {
  if (!Number.isFinite(bytes)) return '大小未提供';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString('zh-CN', { hour12: false }) : '时间未提供';
}

export function mountReplayPage(document) {
  const byId = id => document.getElementById(id);
  const objectList = byId('replay-object-list') ? createObjectList(byId('replay-object-list'), {document}) : null;
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const initialUrl = new URL(window.location.href);
  const initialRun = initialUrl.searchParams.get('run');
  const initialTime = clampReplayTime(initialUrl.searchParams.get('t'));
  const state = {
    experiments: [], selectedRunIds: new Set(), deletionProgress: null, manifest: null, events: [], graph: null, runId: null,
    view: REPLAY_VIEWS.includes(initialUrl.searchParams.get('view')) ? initialUrl.searchParams.get('view') : 'graph',
    time: 0, duration: 0, playing: false, master: null, selectedNode: null,
    operation: 0, fileOperation: 0, listOperation: 0, poll: null, mutating: false,
    observations: [], observationId: null, decisionCallId: null,
    objects: [], objectsId: null,
  };
  const tracks = [
    { name: 'scene', path: 'videos/scene.mp4', label: '主场景' },
    { name: 'cameras', path: 'videos/cameras.mp4', label: '三相机' },
  ].map(track => ({ ...track, video: byId(`${track.name}-video`), available: false }));
  let trace = createTraceList(byId('node-trace'), byId('detail-event-count'));
  const evidence = createReplayEvidence(byId('node-evidence'), {
    document, request, fileUrl, onFile: path => { setView('data'); previewFile(path).catch(showError); },
  });
  const fileEvidence = createReplayEvidence(byId('file-evidence'), { document, request, fileUrl });
  const observationEvidence = createReplayEvidence(byId('observation-model-evidence'), { document, request, fileUrl });
  const policy = createReplayPolicy(byId('policy-plan'), { request, fileUrl });
  let collisionReplay = null, collisionOperation = 0;
  async function showCollision() {
    const operation = ++collisionOperation, runId = state.runId;
    const {createCollisionReplay} = await import('./replay-collision.js');
    if (operation !== collisionOperation || runId !== state.runId || state.view !== 'collision') return;
    collisionReplay?.dispose();
    collisionReplay = createCollisionReplay(document, {load: path => request(fileUrl(runId, path))});
    await collisionReplay.update(state.events);
  }
  document.body.classList.toggle('graph-only', initialUrl.searchParams.get('standalone') === '1');

  function showError(error) {
    byId('page-error').textContent = error instanceof Error ? error.message : String(error);
    byId('page-error').hidden = false;
  }

  function notify(message) {
    byId('page-notice').textContent = message;
    byId('page-notice').hidden = false;
  }

  function clearMessages() {
    byId('page-error').hidden = true;
    byId('page-notice').hidden = true;
  }

  function updateUrl() {
    const url = new URL(window.location.href);
    if (state.runId) url.searchParams.set('run', state.runId);
    else url.searchParams.delete('run');
    url.searchParams.set('view', state.view);
    url.searchParams.set('t', state.time.toFixed(3));
    window.history.replaceState(null, '', url);
    const standalone = new URL(url);
    standalone.searchParams.set('view', 'graph');
    standalone.searchParams.set('standalone', '1');
    byId('standalone-graph').href = standalone.href;
  }

  function setView(view) {
    state.view = view;
    byId('replay-timeline').hidden = view !== 'video';
    for (const name of REPLAY_VIEWS) {
      byId(`tab-${name}`).setAttribute('aria-selected', String(name === view));
      byId(`tab-${name}`).tabIndex = name === view ? 0 : -1;
      byId(`view-${name}`).hidden = name !== view;
    }
    updateUrl();
    if (view === 'collision') showCollision().catch(showError);
    else { collisionOperation++; collisionReplay?.dispose(); }
  }

  function filteredExperiments() {
    const query = byId('experiment-search').value.trim().toLocaleLowerCase();
    return state.experiments.filter(item => `${item.name ?? ''} ${item.instruction ?? ''} ${item.id}`.toLocaleLowerCase().includes(query));
  }

  function canDelete(item) {
    return !item.busy && item.finalized !== false && item.status !== 'running'
      && !['recording', 'encoding'].includes(item.video_status);
  }

  function renderSelectionControls() {
    const available = filteredExperiments().filter(canDelete);
    const count = available.filter(item => state.selectedRunIds.has(item.id)).length;
    const all = byId('select-all-experiments');
    all.checked = available.length > 0 && count === available.length;
    all.indeterminate = count > 0 && count < available.length;
    all.disabled = state.mutating || available.length === 0;
    byId('selected-count').textContent = `已选 ${state.selectedRunIds.size} 条`;
    byId('delete-selected').disabled = state.mutating || state.selectedRunIds.size === 0;
    byId('delete-selected').textContent = state.deletionProgress ? `删除中 ${state.deletionProgress}` : '删除所选';
    byId('clear-selection').disabled = state.mutating || state.selectedRunIds.size === 0;
    byId('experiment-search').disabled = state.mutating;
    byId('refresh-list').disabled = state.mutating;
  }

  function renderList() {
    const query = byId('experiment-search').value.trim();
    const filtered = filteredExperiments();
    const eligible = new Set(filtered.filter(canDelete).map(item => item.id));
    for (const id of state.selectedRunIds) if (!eligible.has(id)) state.selectedRunIds.delete(id);
    byId('experiment-count').textContent = `${filtered.length} 条记录${query ? ` / 共 ${state.experiments.length} 条` : ''}`;
    const list = byId('experiment-list');
    list.replaceChildren();
    for (const item of filtered) {
      const row = element('div', 'experiment-row');
      const checkbox = element('input');
      checkbox.type = 'checkbox';
      checkbox.setAttribute('data-select-run', item.id);
      checkbox.setAttribute('aria-label', `选择实验 ${conciseName(item)} · ${item.id}`);
      checkbox.checked = state.selectedRunIds.has(item.id);
      checkbox.disabled = state.mutating || !canDelete(item);
      if (!canDelete(item)) checkbox.title = '实验正在录制或编码，完成后可删除';
      checkbox.addEventListener('change', () => {
        if (state.mutating || !canDelete(item)) return;
        if (checkbox.checked) state.selectedRunIds.add(item.id);
        else state.selectedRunIds.delete(item.id);
        renderSelectionControls();
      });
      const button = element('button', 'experiment-item');
      button.type = 'button';
      button.disabled = state.mutating;
      button.setAttribute('aria-current', String(state.runId === item.id));
      button.append(element('strong', '', conciseName(item)));
      button.append(element('span', 'muted', formatDate(item.created_at)));
      const status = `${RESULT_NAMES[item.status] ?? item.status ?? '状态未提供'}${item.retention === 'pending' ? ' · 待保存' : ''}`;
      button.append(element('span', `result ${item.status ?? ''}`, status));
      button.append(element('span', 'muted', `${Number.isFinite(item.duration_s) ? `${item.duration_s.toFixed(2)} s` : '用时未提供'} · ${formatSize(item.size_bytes)}`));
      button.addEventListener('click', () => { if (item.id !== state.runId) selectRun(item.id).catch(showError); });
      row.append(checkbox, button);
      list.append(row);
    }
    if (!filtered.length) list.append(element('p', 'muted', query ? '没有匹配的实验。' : '暂无实验记录。请在工作台开始一次启用保存的任务。'));
    renderSelectionControls();
  }

  function stopPlayback() {
    state.playing = false;
    for (const track of tracks) track.video.pause();
    byId('play-pause').textContent = '播放';
  }

  function updateTime() {
    byId('playback-time').textContent = formatReplayTime(state.time);
    byId('playback-seek').value = String(state.time);
    byId('playback-seek').setAttribute('aria-valuetext', `任务时间 ${formatReplayTime(state.time)}`);
    for (const track of tracks) byId(`${track.name}-time`).textContent = track.available ? formatReplayTime(track.video.currentTime) : '无视频';
  }

  function seek(time, { writeUrl = true } = {}) {
    state.time = clampReplayTime(time, state.duration);
    synchronizeVideoTimes(tracks.filter(track => track.available).map(track => track.video), state.time);
    updateTime();
    if (writeUrl) updateUrl();
  }

  function updateDuration() {
    const times = state.events.map(event => event.t_s).filter(Number.isFinite);
    const durations = tracks.filter(track => track.available).map(track => track.video.duration).filter(Number.isFinite);
    state.duration = Math.max(0, state.manifest?.duration_s ?? 0, ...times, ...durations);
    byId('playback-seek').max = String(state.duration);
    byId('playback-seek').disabled = state.duration <= 0;
    byId('playback-duration').textContent = formatReplayTime(state.duration);
  }

  function videoStatus() {
    const manifest = state.manifest;
    if (!manifest) return;
    const status = typeof manifest.video_status === 'string' ? manifest.video_status : manifest.video_status?.status;
    const label = manifest.record_video === false ? '本次未录制视频，仍可查看观测图、执行图与数据。' : VIDEO_NAMES[status] ?? `视频状态：${status ?? '未提供'}`;
    const ready = tracks.filter(track => track.available && track.video.readyState > 0 && !track.video.error);
    byId('video-status').textContent = `${label}${manifest.busy ? '；正在写入实验，完成后可保存或导出。' : ''}`;
    byId('play-pause').disabled = ready.length === 0;
    for (const track of tracks) {
      const message = byId(`${track.name}-video-message`);
      const detail = byId(`${track.name}-video-status`);
      if (!track.available) {
        message.hidden = false;
        message.textContent = manifest.record_video === false ? '未启用视频录制' : ['recording', 'encoding', 'pending'].includes(status) ? '视频正在生成，请稍候…' : '没有此视频文件';
        detail.textContent = `${track.path} · ${manifest.record_video === false ? '未录制' : '文件尚不可用'}`;
      } else if (track.video.error) {
        message.hidden = false;
        message.textContent = '视频无法播放，请下载文件检查编码或完整性。';
        detail.textContent = `${track.path} · 媒体错误 ${track.video.error.code}`;
      } else if (track.video.readyState === 0) {
        message.hidden = false;
        message.textContent = '正在载入视频…';
        detail.textContent = `${track.path} · 等待媒体信息`;
      } else {
        message.hidden = true;
        detail.textContent = `${track.path} · ${Number.isFinite(track.video.duration) ? `${track.video.duration.toFixed(3)} s` : '时长未提供'}${track.video.duration + 0.1 < state.duration ? ' · 录制短于任务，末尾保持最后画面' : ''}`;
      }
    }
  }

  function loadVideos() {
    const paths = new Set((state.manifest.files ?? []).map(file => file.path));
    for (const track of tracks) {
      track.available = paths.has(track.path);
      const source = track.available ? fileUrl(state.runId, track.path) : '';
      if (track.video.getAttribute('src') !== (source || null)) {
        if (source) track.video.setAttribute('src', source);
        else track.video.removeAttribute('src');
        track.video.load();
      }
    }
    updateDuration();
    videoStatus();
    updateTime();
  }

  async function play() {
    if (state.playing) { stopPlayback(); updateUrl(); return; }
    const ready = tracks.filter(track => track.available && track.video.readyState > 0 && !track.video.error);
    if (!ready.length) return;
    state.master = ready.reduce((longest, track) => track.video.duration > longest.video.duration ? track : longest).video;
    if (state.time >= state.master.duration) seek(0);
    synchronizeVideoTimes(ready.map(track => track.video), state.time);
    state.playing = true;
    byId('play-pause').textContent = '暂停';
    const operation = state.operation;
    const results = await Promise.allSettled(ready.filter(track => track.video.currentTime < track.video.duration).map(track => track.video.play()));
    if (operation !== state.operation) return;
    const failure = results.find(result => result.status === 'rejected');
    if (failure) { stopPlayback(); showError(new Error(`视频播放失败：${failure.reason?.message ?? failure.reason}`)); }
  }

  function renderSummary({ replaceName = false } = {}) {
    const manifest = state.manifest;
    if (!manifest) return;
    byId('experiment-title').textContent = conciseName(manifest);
    byId('experiment-instruction').textContent = manifest.instruction || '未提供任务指令';
    byId('experiment-meta').textContent = `${formatDate(manifest.created_at)} · ${RESULT_NAMES[manifest.status] ?? manifest.status}`;
    const progress = state.events.findLast(event => event.kind === 'run_end')?.data?.task_progress;
    objectList?.render(progress);
    if (progress) byId('experiment-meta').textContent += ` · ${taskProgressSummary(progress)}`;
    const calls = state.events.filter(event => event.kind === 'visual_decisions_call').map(modelCallInfo);
    const metrics = byId('replay-metrics');
    metrics.replaceChildren();
    for (const [label, value] of [['任务用时', Number.isFinite(manifest.duration_s) ? `${manifest.duration_s.toFixed(1)} 秒` : '未归档'],
      ['Responses', `${calls.filter(call => call.type === 'Responses').length} 次`],
      ['Decisions', `${calls.filter(call => call.type === 'Decisions').length} 次`]]) {
      const metric = element('div', 'replay-metric');
      metric.append(element('span', 'muted', label), element('strong', '', value));
      metrics.append(metric);
    }
    const pending = manifest.retention === 'pending';
    byId('retention-badge').textContent = pending ? '待保存' : '已归档';
    byId('retention-badge').classList.toggle('pending', pending);
    if (replaceName) byId('experiment-name').value = manifest.name || '';
    byId('save-experiment').textContent = pending ? '保存实验' : '保存名称';
    byId('save-experiment').disabled = Boolean(manifest.busy || state.mutating);
    byId('discard-experiment').hidden = false;
    byId('discard-experiment').disabled = Boolean(manifest.busy || manifest.finalized === false
      || manifest.status === 'running' || ['recording', 'encoding'].includes(manifest.video_status) || state.mutating);
    byId('experiment-name').disabled = state.mutating;
    const errors = (manifest.errors ?? []).map(error => typeof error === 'string' ? error : JSON.stringify(error));
    const end = state.events.findLast(event => event.kind === 'run_end')?.data;
    renderFailureReport(byId('experiment-errors'), end ?? manifest, state.events, { additionalErrors: errors });
    for (const mode of ['review', 'full']) {
      const link = byId(`${mode}-export`);
      link.href = `/api/experiments/${encodeURIComponent(state.runId)}/export?mode=${mode}`;
      link.setAttribute('aria-disabled', String(Boolean(manifest.busy || state.mutating)));
      link.tabIndex = manifest.busy || state.mutating ? -1 : 0;
    }
  }

  function conciseName(manifest) {
    const name = manifest.name || '未命名实验';
    return name.length > 42 && name === manifest.instruction ? '机器人任务实验' : name;
  }

  function renderObjectMemory() {
    state.objects = objectsSnapshots(state.events);
    const select = byId('objects-snapshot');
    select.replaceChildren();
    select.disabled = !state.objects.length;
    for (const snapshot of state.objects) {
      const option = element('option', '', `${snapshot.title} · 事件 #${snapshot.event.id}`);
      option.value = String(snapshot.event.id);
      select.append(option);
    }
    if (!state.objects.some(snapshot => String(snapshot.event.id) === state.objectsId)) state.objectsId = String(state.objects.at(-1)?.event.id ?? '');
    select.value = state.objectsId;
    showObjects();
    const origin = byId('objects-origin');
    origin.replaceChildren();
    for (const event of state.events.filter(item => item.kind === 'visual_decisions_call' && item.data?.purpose === 'initialize_objects')) {
      const info = modelCallInfo(event);
      const button = element('button', 'button', `${info.type} · ${info.purpose}`);
      button.addEventListener('click', () => openModelEvent(event));
      origin.append(button);
    }
  }

  async function showObjects() {
    const runId = state.runId;
    const selectedId = state.objectsId;
    const board = byId('objects-board');
    let snapshot = state.objects.find(item => String(item.event.id) === selectedId);
    try {
      if (snapshot && validFilePath(snapshot.event.artifact)) {
        board.textContent = '正在读取时间记忆数据库快照…';
        const data = await request(fileUrl(runId, snapshot.event.artifact));
        snapshot = objectsSnapshots([{ ...snapshot.event, data }])[0];
      }
      if (runId !== state.runId || selectedId !== state.objectsId) return;
      renderObjectMemorySnapshot(board, snapshot, event => {
        if (validFilePath(event.artifact)) { setView('data'); previewFile(event.artifact).catch(showError); }
        else { setView('graph'); showNode({ title: '时间记忆记录', events: [event], startTime: event.t_s }, event); }
      });
    } catch (error) {
      if (runId === state.runId && selectedId === state.objectsId) {
        board.textContent = `时间记忆快照读取失败，${error.message}`;
      }
    }
  }

  function openModelEvent(event) {
    const node = state.graph.nodes.find(item => item.events.includes(event));
    if (node) { setView('graph'); showNode(node, event); }
  }

  function addDownload(parent, path, text = path) {
    if (!validFilePath(path)) return;
    const link = element('a', '', text);
    link.href = fileUrl(state.runId, path);
    link.download = path.split('/').at(-1);
    parent.append(link);
  }

  function renderObservations() {
    state.observations = decisionObservations(state.events);
    const select = byId('observation-select');
    select.replaceChildren();
    const count = state.observations.reduce((sum, observation) => sum + observation.calls.length, 0);
    byId('observation-count').textContent = `${state.observations.length} 次观测 · ${count} 次请求`;
    select.disabled = !state.observations.length;
    for (const [index, observation] of state.observations.entries()) {
      const option = element('option', '', `第 ${index + 1} 次观测 · ${formatReplayTime(observation.time)} · ${observation.calls.length} 次 Decisions`);
      option.value = observation.id;
      select.append(option);
    }
    if (!state.observations.some(item => item.id === state.observationId)) {
      state.observationId = state.observations[0]?.id ?? null;
    }
    showObservation(state.observationId);
  }

  function showObservation(id) {
    observationEvidence.clear();
    const select = byId('observation-model-call');
    select.replaceChildren();
    const observation = state.observations.find(item => item.id === id);
    select.disabled = !observation;
    byId('observation-seek').disabled = !Number.isFinite(observation?.time);
    if (!observation) {
      byId('observation-meta').textContent = '本次实验没有归档 Decisions 请求。';
      select.onchange = null;
      return;
    }
    state.observationId = id;
    byId('observation-select').value = id;
    byId('observation-meta').textContent = `${id} · ${observation.calls.length} 次 Decisions 请求`;
    for (const event of observation.calls) {
      const info = modelCallInfo(event);
      const option = element('option', '', `#${event.id} · Decisions · ${info.purpose}`);
      option.value = String(event.id);
      select.append(option);
    }
    const preferred = observation.calls.find(event => String(event.id) === state.decisionCallId) ?? observation.calls[0];
    select.value = String(preferred.id);
    const show = () => {
      const event = observation.calls.find(item => String(item.id) === select.value);
      state.decisionCallId = select.value;
      void observationEvidence.show(state.runId, [event]);
    };
    select.onchange = show;
    show();
  }

  function showNode(node, event = null) {
    state.selectedNode = { node, event };
    const events = event ? [event] : node.events;
    const model = event?.kind === 'visual_decisions_call' ? modelCallInfo(event) : null;
    byId('node-title').textContent = model ? `${model.type} · ${model.purpose}` : event?.title ?? node.title;
    byId('node-meta').textContent = `${node.callId ? `算法调用 ${node.callId}` : node.invocationId ? `调用 ${node.invocationId}` : `事件 ${events[0]?.id ?? '—'}`}${node.parentInvocationId ? ` · 父调用 ${node.parentInvocationId}` : ''} · ${events.length} 条记录`;
    const time = event ? event.t_s : node.startTime;
    byId('seek-node').hidden = !Number.isFinite(time);
    byId('seek-node').textContent = `定位视频 · ${formatReplayTime(time)}`;
    byId('node-artifacts').replaceChildren();
    byId('node-files').replaceChildren();
    byId('node-diagnostics').open = false;
    evidence.clear();
    if (model) {
      byId('node-meta').textContent = `事件 #${event.id} · ${Number.isFinite(model.elapsed) ? `${model.elapsed.toFixed(2)} 秒 · ` : ''}${model.failed ? '请求失败' : '实际归档的模型请求与回答'}`;
      void evidence.show(state.runId, [event]).catch(showError);
    }
    const calls = node.nodes ?? [];
    for (const call of node.calls ?? []) {
      const button = element('button', 'stage-button', call.title);
      button.setAttribute('data-algorithm-id', call.id);
      button.addEventListener('click', () => showNode(call));
      byId('node-artifacts').append(button);
    }
    if (!event && calls.length) {
      byId('node-meta').textContent = `${Number.isFinite(node.duration) ? `阶段用时 ${node.duration.toFixed(2)} 秒 · ` : ''}下列为实际发生的技能调用`;
      const list = element('div', 'skill-sequence');
      for (const call of calls.filter(item => item.invocationId)) {
        const button = element('button', 'stage-button', `${SKILL_LABELS[call.title] ?? call.title}${Number.isFinite(call.duration) ? ` · ${call.duration.toFixed(2)} 秒` : ''}`);
        button.title = call.title;
        button.addEventListener('click', () => showNode(call));
        list.append(button);
      }
      byId('node-artifacts').append(list);
    }
    for (const call of events.filter(item => item.kind === 'visual_decisions_call' && item !== event && !calls.length)) {
      const info = modelCallInfo(call);
      const button = element('button', `button model-link ${info.type.toLowerCase()}`, `${info.type} · ${info.purpose}`);
      button.addEventListener('click', () => showNode(node, call));
      byId('node-artifacts').append(button);
    }
    if (!event && node.phase === 'observe') {
      const objects = element('button', 'button primary', '查看时间记忆数据库');
      objects.addEventListener('click', () => setView('visual'));
      byId('node-artifacts').append(objects);
    }
    const inspect = element('button', 'button primary', '查看图片、SAM3 与抓取候选');
    inspect.addEventListener('click', () => {
      inspect.disabled = true;
      evidence.show(state.runId, events).catch(showError).finally(() => { inspect.disabled = false; });
    });
    if (!model) byId('node-artifacts').append(inspect);
    for (const item of events) {
      if (item.artifact) {
        const wrapper = element('div', 'artifact-row');
        const button = element('button', 'button', `查看事件 #${item.id} 完整数据`);
        button.addEventListener('click', () => { setView('data'); previewFile(item.artifact).catch(showError); });
        wrapper.append(button);
        addDownload(wrapper, item.artifact, '下载 JSON');
        byId('node-files').append(wrapper);
      }
      if (validFilePath(item.data?.observation_artifact)) {
        const path = item.data.observation_artifact;
        const wrapper = element('div', 'artifact-row');
        const button = element('button', 'button', `查看事件 #${item.id} RGBD 与标定`);
        button.addEventListener('click', () => { setView('data'); previewFile(path).catch(showError); });
        wrapper.append(button);
        addDownload(wrapper, path, '下载 RGBD 引用 JSON');
        byId('node-files').append(wrapper);
      }
      const observation = state.observations.find(record => record.calls.some(call => call.id === item.id));
      if (observation) {
        const button = element('button', 'button', '查看 Decisions 输入');
        button.addEventListener('click', () => {
          state.decisionCallId = String(item.id);
          showObservation(observation.id);
          setView('observations');
        });
        byId('node-artifacts').append(button);
      }
    }
    trace(events, state.events);
    for (const details of byId('node-trace').querySelectorAll(':scope > details')) details.open = false;
    if (Number.isFinite(time)) { stopPlayback(); seek(time); }
  }

  function renderFiles() {
    const query = byId('file-search').value.trim().toLocaleLowerCase();
    const files = (state.manifest?.files ?? []).filter(file => file.path.toLocaleLowerCase().includes(query));
    byId('file-list').replaceChildren();
    for (const file of files) {
      const row = element('div', 'file-row');
      if (/\.(json|png|jpe?g|webp|gif|svg|mp4|webm)$/i.test(file.path)) {
        const button = element('button', '', file.path);
        button.type = 'button';
        button.addEventListener('click', () => previewFile(file.path).catch(showError));
        row.append(button);
      } else row.append(element('span', '', file.path));
      const footer = element('div');
      footer.append(element('span', 'muted', formatSize(file.size_bytes)));
      addDownload(footer, file.path, '下载');
      row.append(footer);
      byId('file-list').append(row);
    }
    if (!files.length) byId('file-list').append(element('p', 'empty-state', '没有匹配的文件。'));
  }

  async function previewFile(path) {
    const operation = ++state.fileOperation;
    fileEvidence.clear();
    const runId = state.runId;
    const url = fileUrl(runId, path);
    byId('file-title').textContent = path;
    byId('file-download').href = url;
    byId('file-download').hidden = false;
    byId('array-references').replaceChildren();
    byId('file-content').textContent = '正在读取数据…';
    if (/\.(png|jpe?g|webp|gif|svg|mp4|webm)$/i.test(path)) {
      const isVideo = /\.(mp4|webm)$/i.test(path);
      const preview = element(isVideo ? 'video' : 'img', 'archive-media-preview');
      preview.src = url;
      if (isVideo) { preview.controls = true; preview.preload = 'auto'; preview.muted = true; }
      else preview.alt = path;
      preview.addEventListener('error', () => { byId('file-content').textContent = '媒体无法预览，可下载原文件。'; });
      byId('file-evidence').replaceChildren(preview);
      byId('file-content').textContent = '';
      return;
    }
    const file = state.manifest.files.find(item => item.path === path);
    if (file?.size_bytes > 5 * 1024 ** 2) {
      byId('file-content').textContent = '此 JSON 超过 5 MB，请下载原文件查看。';
      return;
    }
    try {
      const value = await request(url);
      if (operation !== state.fileOperation || runId !== state.runId) return;
      byId('file-content').textContent = JSON.stringify(value, null, 2);
      void fileEvidence.show(runId, [], { values: [value] });
      for (const reference of artifactReferences(value)) addDownload(byId('array-references'), reference, `数组文件 · ${reference}`);
    } catch (error) {
      if (operation !== state.fileOperation || runId !== state.runId) return;
      byId('file-content').textContent = `文件读取失败：${error.message}`;
      throw error;
    }
  }

  function scheduleRefresh() {
    window.clearTimeout(state.poll);
    const manifest = state.manifest;
    if (!manifest || (!manifest.busy && manifest.status !== 'running' && !['encoding', 'recording', 'pending'].includes(manifest.video_status))) return;
    state.poll = window.setTimeout(() => refreshSelected().catch(showError), 3000);
  }

  async function refreshSelected() {
    if (!state.runId || state.mutating) return;
    const operation = state.operation;
    const base = `/api/experiments/${encodeURIComponent(state.runId)}`;
    const [manifest, payload] = await Promise.all([request(base), request(`${base}/events`)]);
    const events = await loadArchivedFailureReport(payload.events ?? [], path => request(fileUrl(manifest.id, path)));
    if (operation !== state.operation) return;
    state.manifest = manifest;
    state.events = events;
    state.graph = buildExecutionGraph(state.events);
    state.experiments = state.experiments.map(item => item.id === manifest.id ? manifest : item);
    renderSummary();
    renderList();
    renderFiles();
    renderObservations();
    renderObjectMemory();
    renderGraph();
    loadVideos();
    scheduleRefresh();
  }

  function renderGraph() {
    policy.show(state.runId, state.events).catch(showError);
    const stages = renderReplayStages(byId('object-timeline'), state.graph, (node, event) => {
      showNode(node, event);
      if (window.matchMedia?.('(max-width: 1250px)').matches) byId('node-title').scrollIntoView?.({ behavior: 'smooth', block: 'start' });
    });
    if (!state.selectedNode && stages.length) showNode(stages[0]);
  }

  async function selectRun(runId, time = 0, { preserveMessages = false } = {}) {
    if (state.mutating) return;
    if (!preserveMessages) clearMessages();
    stopPlayback();
    window.clearTimeout(state.poll);
    const operation = ++state.operation;
    state.fileOperation += 1;
    evidence.clear();
    fileEvidence.clear();
    observationEvidence.clear();
    policy.clear();
    state.runId = runId;
    state.manifest = null;
    state.selectedNode = null;
    state.observationId = null;
    state.decisionCallId = null;
    state.objectsId = null;
    state.time = time;
    state.master = null;
    byId('experiment-content').hidden = true;
    byId('empty-selection').hidden = false;
    byId('empty-selection').replaceChildren(element('h2', '', '正在载入实验…'));
    renderList();
    const base = `/api/experiments/${encodeURIComponent(runId)}`;
    try {
      const [manifest, payload] = await Promise.all([request(base), request(`${base}/events`)]);
      const events = await loadArchivedFailureReport(payload.events ?? [], path => request(fileUrl(runId, path)));
      if (operation !== state.operation) return;
      state.manifest = manifest;
      state.events = events;
      state.graph = buildExecutionGraph(state.events);
      byId('empty-selection').hidden = true;
      byId('experiment-content').hidden = false;
      byId('node-title').textContent = '节点详情';
      byId('node-meta').textContent = '展开执行证据，选择步骤或模型调用查看归档。';
      byId('seek-node').hidden = true;
      byId('node-artifacts').replaceChildren();
      byId('node-files').replaceChildren();
      byId('node-trace').replaceChildren();
      trace = createTraceList(byId('node-trace'), byId('detail-event-count'));
      byId('detail-event-count').textContent = '0';
      byId('file-title').textContent = '数据预览';
      byId('file-content').textContent = '选择一个 JSON 文件查看。';
      byId('file-download').hidden = true;
      byId('array-references').replaceChildren();
      renderSummary({ replaceName: true });
      renderGraph();
      renderFiles();
      renderObservations();
      renderObjectMemory();
      loadVideos();
      seek(time);
      setView(state.view);
      scheduleRefresh();
    } catch (error) {
      if (operation !== state.operation) return;
      byId('empty-selection').replaceChildren(element('h2', '', '实验读取失败'), element('p', '', error.message));
      throw error;
    }
  }

  async function loadList() {
    const operation = state.operation;
    const listOperation = ++state.listOperation;
    const payload = await request('/api/experiments');
    if (operation !== state.operation || listOperation !== state.listOperation) return;
    state.experiments = [...(payload.experiments ?? [])].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
    byId('experiment-root').textContent = payload.root ?? '未提供';
    renderList();
  }

  async function saveExperiment(event) {
    event.preventDefault();
    if (!state.manifest || state.manifest.busy || state.mutating) return;
    clearMessages();
    const runId = state.runId;
    const wasPending = state.manifest.retention === 'pending';
    state.mutating = true;
    state.operation += 1;
    window.clearTimeout(state.poll);
    renderSummary();
    renderList();
    try {
      const manifest = await request(`/api/experiments/${encodeURIComponent(runId)}/save`, { name: byId('experiment-name').value.trim() });
      if (runId === state.runId) { state.manifest = manifest; renderSummary({ replaceName: true }); }
      await loadList();
      notify(wasPending ? '实验已保存。' : '实验名称已保存。');
    } catch (error) { showError(error); }
    finally { state.mutating = false; renderList(); renderSummary(); scheduleRefresh(); }
  }

  async function discardExperiment() {
    if (!state.manifest || byId('discard-experiment').disabled || state.mutating) return;
    await deleteExperiments([state.manifest]);
  }

  function clearDeletedReplay() {
    stopPlayback();
    window.clearTimeout(state.poll);
    state.operation += 1;
    state.fileOperation += 1;
    state.runId = null;
    state.manifest = null;
    state.events = [];
    state.graph = null;
    state.selectedNode = null;
    evidence.clear();
    fileEvidence.clear();
    observationEvidence.clear();
    policy.clear();
    byId('experiment-content').hidden = true;
    byId('empty-selection').hidden = false;
    byId('empty-selection').replaceChildren(element('h2', '', '选择一条实验记录'));
    updateUrl();
  }

  async function deleteExperiments(records) {
    if (state.mutating || records.length === 0) return;
    clearMessages();
    const activeRunId = state.runId;
    const releasingActive = records.some(item => item.id === activeRunId);
    if (releasingActive) {
      stopPlayback();
      // Release video range requests before the store moves the archive.
      for (const track of tracks) { track.video.removeAttribute('src'); track.video.load(); }
    }
    state.mutating = true;
    state.deletionProgress = `0/${records.length}`;
    if (releasingActive) state.operation += 1;
    state.listOperation += 1;
    window.clearTimeout(state.poll);
    renderSummary();
    renderList();
    let deleted = 0;
    const failures = [];
    try {
      for (const [index, item] of records.entries()) {
        try {
          const response = await fetch(`/api/experiments/${encodeURIComponent(item.id)}`, { method: 'DELETE' });
          const payload = await response.json();
          if (!response.ok || payload.deleted !== true) {
            const reasons = { experiment_busy: '实验仍在录制或编码，请完成后再删除。', experiment_in_use: '实验文件正在下载或导出，请结束后再删除。' };
            throw new Error(reasons[payload.detail] ?? payload.detail ?? '实验删除失败');
          }
          deleted += 1;
          state.selectedRunIds.delete(item.id);
          state.experiments = state.experiments.filter(record => record.id !== item.id);
          if (item.id === state.runId) clearDeletedReplay();
        } catch (error) {
          failures.push(`${conciseName(item)}（${item.id}）：${error.message}`);
        }
        state.deletionProgress = `${index + 1}/${records.length}`;
        renderSelectionControls();
      }
      notify(`已删除 ${deleted}/${records.length} 条实验，移入本地回收区，原始视频与数据仍保留。`);
      if (failures.length) showError(new Error(`有 ${failures.length} 条未删除：${failures.join('；')}`));
      try { await loadList(); }
      catch (error) { showError(new Error(`${failures.join('；')} 刷新列表失败：${error.message}`)); }
    } finally {
      state.mutating = false;
      state.deletionProgress = null;
      if (releasingActive && state.runId === activeRunId && state.manifest) loadVideos();
      renderList();
      renderSummary();
      scheduleRefresh();
    }
    if (releasingActive && state.runId === activeRunId && !state.manifest) {
      await selectRun(activeRunId, state.time, { preserveMessages: true });
    }
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = element('a');
    link.href = url;
    link.download = filename;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function exportGraph(format) {
    if (!state.graph || !state.manifest) return;
    const svg = policy.svg();
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const filename = `graph-${state.runId}`;
    if (format === 'svg') { downloadBlob(blob, `${filename}.svg`); return; }
    const url = URL.createObjectURL(blob);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      // Bound canvas area for long experiments; SVG always retains full resolution.
      const scale = Math.min(2, 16000 / image.width, 16000 / image.height, Math.sqrt(32000000 / (image.width * image.height)));
      const canvas = document.createElement('canvas');
      canvas.width = Math.ceil(image.width * scale);
      canvas.height = Math.ceil(image.height * scale);
      canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
      const png = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
      if (!png) throw new Error('PNG 导出失败，请使用 SVG 导出。');
      downloadBlob(png, `${filename}.png`);
    } finally { URL.revokeObjectURL(url); }
  }

  for (const name of REPLAY_VIEWS) {
    byId(`tab-${name}`).addEventListener('click', () => setView(name));
    byId(`tab-${name}`).addEventListener('keydown', event => {
      const tabs = REPLAY_VIEWS;
      let index = tabs.indexOf(name);
      if (event.key === 'ArrowRight') index = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') index = (index + tabs.length - 1) % tabs.length;
      else if (event.key === 'Home') index = 0;
      else if (event.key === 'End') index = tabs.length - 1;
      else return;
      event.preventDefault();
      setView(tabs[index]);
      byId(`tab-${tabs[index]}`).focus();
    });
  }
  for (const track of tracks) {
    const video = track.video;
    video.addEventListener('loadedmetadata', () => { updateDuration(); synchronizeVideoTimes([video], state.time); videoStatus(); updateTime(); });
    video.addEventListener('error', videoStatus);
    video.addEventListener('timeupdate', () => {
      if (video === state.master && state.playing) {
        state.time = clampReplayTime(video.currentTime, state.duration);
        synchronizeVideoTimes(tracks.filter(item => item.video !== video && item.available).map(item => item.video), state.time, 0.18);
      }
      updateTime();
    });
    video.addEventListener('ended', () => { if (state.master === video) { state.time = video.currentTime; stopPlayback(); updateTime(); updateUrl(); } });
  }
  byId('play-pause').addEventListener('click', () => play().catch(showError));
  byId('playback-seek').addEventListener('input', event => { stopPlayback(); seek(event.target.value); });
  byId('playback-speed').addEventListener('change', event => { for (const track of tracks) track.video.playbackRate = Number(event.target.value); });
  byId('seek-node').addEventListener('click', () => { const selection = state.selectedNode; if (selection) { setView('video'); seek(selection.event?.t_s ?? selection.node.startTime); } });
  byId('experiment-search').addEventListener('input', () => {
    if (state.mutating) return;
    state.selectedRunIds.clear();
    renderList();
  });
  byId('select-all-experiments').addEventListener('change', () => {
    if (state.mutating) return;
    for (const item of filteredExperiments().filter(canDelete)) {
      if (byId('select-all-experiments').checked) state.selectedRunIds.add(item.id);
      else state.selectedRunIds.delete(item.id);
    }
    renderList();
  });
  byId('clear-selection').addEventListener('click', () => {
    if (state.mutating) return;
    state.selectedRunIds.clear();
    renderList();
  });
  let batchDiscardRecords = [];
  byId('delete-selected').addEventListener('click', () => {
    if (state.mutating) return;
    batchDiscardRecords = filteredExperiments().filter(item => canDelete(item) && state.selectedRunIds.has(item.id));
    if (batchDiscardRecords.length === 0) return;
    byId('batch-discard-count').textContent = `将删除 ${batchDiscardRecords.length} 条实验：`;
    byId('batch-discard-names').replaceChildren(...batchDiscardRecords.map(item =>
      element('li', '', `${conciseName(item)} · ID ${item.id}`)));
    byId('batch-discard-dialog').returnValue = '';
    byId('batch-discard-dialog').showModal();
  });
  byId('batch-discard-dialog').addEventListener('close', () => {
    const records = batchDiscardRecords;
    batchDiscardRecords = [];
    if (byId('batch-discard-dialog').returnValue === 'discard') deleteExperiments(records).catch(showError);
  });
  byId('file-search').addEventListener('input', renderFiles);
  byId('refresh-list').addEventListener('click', async () => {
    if (state.mutating) return;
    clearMessages();
    byId('refresh-list').disabled = true;
    try { await loadList(); if (state.runId) await refreshSelected(); }
    catch (error) { showError(error); }
    finally { byId('refresh-list').disabled = state.mutating; }
  });
  byId('rename-form').addEventListener('submit', saveExperiment);
  let discardRunId = null;
  byId('discard-experiment').addEventListener('click', () => {
    if (!state.manifest || byId('discard-experiment').disabled) return;
    discardRunId = state.runId;
    byId('discard-name').textContent = `${state.manifest.name || '未命名实验'} · ${formatSize(state.manifest.size_bytes)} · ID ${state.runId}`;
    byId('discard-dialog').returnValue = '';
    byId('discard-dialog').showModal();
  });
  byId('discard-dialog').addEventListener('close', () => { if (byId('discard-dialog').returnValue === 'discard' && discardRunId === state.runId) discardExperiment().catch(showError); });
  byId('export-svg').addEventListener('click', () => exportGraph('svg').catch(showError));
  byId('export-png').addEventListener('click', () => exportGraph('png').catch(showError));
  byId('export-json').addEventListener('click', () => {
    if (policy.graph()) downloadBlob(new Blob([JSON.stringify(policy.graph(), null, 2)], { type: 'application/json' }), `graph-${state.runId}.json`);
  });
  byId('graph-zoom-in').addEventListener('click', () => policy.zoom(1.25));
  byId('graph-zoom-out').addEventListener('click', () => policy.zoom(0.8));
  byId('graph-fit').addEventListener('click', () => policy.fit());
  byId('graph-reset').addEventListener('click', () => policy.reset());
  byId('objects-snapshot').addEventListener('change', () => { state.objectsId = byId('objects-snapshot').value; showObjects(); });
  byId('observation-select').addEventListener('change', () => showObservation(byId('observation-select').value));
  byId('observation-seek').addEventListener('click', () => {
    const observation = state.observations.find(item => item.id === state.observationId);
    if (Number.isFinite(observation?.time)) { stopPlayback(); setView('video'); seek(observation.time); }
  });
  byId('standalone-graph').addEventListener('click', updateUrl);
  window.addEventListener('pagehide', () => { stopPlayback(); window.clearTimeout(state.poll); });
  loadList().then(() => {
    const id = initialRun || state.experiments[0]?.id;
    if (id) return selectRun(id, initialTime);
  }).catch(showError);
}

if (typeof document !== 'undefined') mountReplayPage(document);

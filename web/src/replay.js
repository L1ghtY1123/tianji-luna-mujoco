import { runConclusion } from './run-conclusion.js';
import { request } from './api.js';
import { createTraceList } from './trace.js';
import { buildExecutionGraph, createExecutionGraph, graphSvg } from './execution-graph.js';
import { taskProgressSummary } from './run-evidence.js';
import { createObjectList } from './object-list.js';
import { createReplayEvidence } from './replay-evidence.js';
import { groupObservationFrames, modelCallInfo, inventorySnapshots, renderReplayStages, renderInventorySnapshot } from './replay-story.js';

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

const CAMERA_NAMES = { 2: '左腕 A · 相机 2', 3: '头部 · 相机 3', 1: '右腕 B · 相机 1' };
const CHECKPOINT_NAMES = { initial: '观测位姿观察', fine: '抓前观察 / 建图', post_place: '放后观察', post_unpack: 'VLA 拆包后观察', result: '结果观察' };
const ANALYSIS_NAMES = { analyzed: '已分析', unprocessed: '未分析：尚无本帧掩码与 ID 关联', unknown: '分析状态未记录' };
const REPLAY_VIEWS = ['graph', 'inventory', 'video', 'observations', 'data'];
const SKILL_LABELS = { observe_scene: '三视角观测与识别', bind_inventory: '登记初始清单',
  identify_target: '选择当前物体', identify_destination: '确定目标区域', choose_arm: '确定执行臂',
  build_scene: '建立 TSDF / ESDF', generate_grasps: 'GraspGen 候选与预处理', plan_grasp: 'Batched IK 与抓取规划',
  execute_grasp: '执行三段抓取', verify_grasp: '夹爪反馈核验', plan_place: '规划桌面放置',
  execute_place: '放置、松爪与撤离', verify_place: '确认桌面放置', prepare_unpack: '移动到 VLA 位姿',
  mock_unpack: '仿真拆包：生成黑件', verify_unpack: '确认拆包结果', move_to_observe: '双臂返回观测位姿',
  plan_transport: '持物 plan_pose 规划', execute_transport: '运输到右盒上方', drop: '松爪投放', verify_drop: '确认右盒新增产物' };

export function archivedObservations(events = []) {
  const records = new Map(events.filter(event => event.kind === 'observation_recorded' && event.data?.observation_id)
    .map(event => [event.data.observation_id, event.data]));
  const observations = [];
  for (const [index, event] of events.entries()) {
    if (event.kind !== 'observation_annotated' || !event.data?.observation_id) continue;
    const data = event.data;
    const labels = data.object_labels ?? {};
    if (labels.observation_id && labels.observation_id !== data.observation_id) continue;
    const cameras = Object.entries(data.annotated_rgb ?? {})
      .filter(([, path]) => validFilePath(path) && /\.(svg|png|jpe?g)$/i.test(path))
      .map(([id, path]) => ({ id, path, label: CAMERA_NAMES[id] ?? `相机 ${id}` }));
    const cameraOrder = { 2: 0, 3: 1, 1: 2 };
    cameras.sort((a, b) => (cameraOrder[a.id] ?? 3) - (cameraOrder[b.id] ?? 3));
    if (!cameras.length) continue;
    const record = records.get(data.observation_id);
    observations.push({ id: String(event.id ?? `annotation-${index}`), observationId: data.observation_id,
      checkpoint: record?.checkpoint ?? null, artifact: record?.observation_artifact ?? null,
      status: labels.analysis_status ?? 'unknown', time: event.t_s, event, cameras });
  }
  return observations;
}

// Archive SVGs embed the original RGB as a PNG. Do not execute SVG or resolve external URLs.
export function embeddedRawRgb(svg) {
  return /<image\b[^>]*\bhref=["'](data:image\/png;base64,[A-Za-z0-9+/=]+)["']/i.exec(svg)?.[1] ?? null;
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
    experiments: [], manifest: null, events: [], graph: null, runId: null,
    view: REPLAY_VIEWS.includes(initialUrl.searchParams.get('view')) ? initialUrl.searchParams.get('view') : 'graph',
    time: 0, duration: 0, playing: false, master: null, selectedNode: null,
    operation: 0, fileOperation: 0, listOperation: 0, poll: null, mutating: false,
    observations: [], observationFrames: [], observationId: null, imageOperation: 0, image: null, imageZoom: 1,
    inventory: [], inventoryId: null,
  };
  const tracks = [
    { name: 'scene', path: 'videos/scene.mp4', label: '主场景' },
    { name: 'cameras', path: 'videos/cameras.mp4', label: '三相机' },
  ].map(track => ({ ...track, video: byId(`${track.name}-video`), available: false }));
  let trace = createTraceList(byId('node-trace'), byId('detail-event-count'));
  const graphView = createExecutionGraph(byId('execution-graph'), { onSelect: showNode });
  const evidence = createReplayEvidence(byId('node-evidence'), {
    document, request, fileUrl, onFile: path => { setView('data'); previewFile(path).catch(showError); },
  });
  const fileEvidence = createReplayEvidence(byId('file-evidence'), { document, request, fileUrl });
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
  }

  function renderList() {
    const query = byId('experiment-search').value.trim().toLocaleLowerCase();
    const filtered = state.experiments.filter(item => `${item.name ?? ''} ${item.instruction ?? ''} ${item.id}`.toLocaleLowerCase().includes(query));
    byId('experiment-count').textContent = `${filtered.length} 条记录${query ? ` / 共 ${state.experiments.length} 条` : ''}`;
    const list = byId('experiment-list');
    list.replaceChildren();
    for (const item of filtered) {
      const button = element('button', 'experiment-item');
      button.type = 'button';
      button.setAttribute('aria-current', String(state.runId === item.id));
      button.append(element('strong', '', conciseName(item)));
      button.append(element('span', 'muted', formatDate(item.created_at)));
      const status = `${RESULT_NAMES[item.status] ?? item.status ?? '状态未提供'}${item.retention === 'pending' ? ' · 待保存' : ''}`;
      button.append(element('span', `result ${item.status ?? ''}`, status));
      button.append(element('span', 'muted', `${Number.isFinite(item.duration_s) ? `${item.duration_s.toFixed(2)} s` : '用时未提供'} · ${formatSize(item.size_bytes)}`));
      button.addEventListener('click', () => { if (item.id !== state.runId) selectRun(item.id).catch(showError); });
      list.append(button);
    }
    if (!filtered.length) list.append(element('p', 'muted', query ? '没有匹配的实验。' : '暂无实验记录。请在工作台开始一次启用保存的任务。'));
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
    const conclusion = runConclusion(end ?? manifest, state.events);
    if (conclusion) errors.unshift(conclusion);
    byId('experiment-errors').textContent = errors.join('\n');
    byId('experiment-errors').hidden = errors.length === 0;
    for (const mode of ['review', 'full']) {
      const link = byId(`${mode}-export`);
      link.href = `/api/experiments/${encodeURIComponent(state.runId)}/export?mode=${mode}`;
      link.setAttribute('aria-disabled', String(Boolean(manifest.busy || state.mutating)));
      link.tabIndex = manifest.busy || state.mutating ? -1 : 0;
    }
  }

  function conciseName(manifest) {
    const name = manifest.name || '未命名实验';
    return name.length > 42 && name === manifest.instruction ? '拆包与搬运实验' : name;
  }

  function renderInventory() {
    state.inventory = inventorySnapshots(state.events);
    const select = byId('inventory-snapshot');
    select.replaceChildren();
    select.disabled = !state.inventory.length;
    for (const snapshot of state.inventory) {
      const option = element('option', '', `${snapshot.title} · 事件 #${snapshot.event.id}`);
      option.value = String(snapshot.event.id);
      select.append(option);
    }
    if (!state.inventory.some(snapshot => String(snapshot.event.id) === state.inventoryId)) state.inventoryId = String(state.inventory.at(-1)?.event.id ?? '');
    select.value = state.inventoryId;
    showInventory();
    const origin = byId('inventory-origin');
    origin.replaceChildren();
    for (const event of state.events.filter(item => item.kind === 'visual_decisions_call' && item.data?.purpose === 'initialize_inventory')) {
      const info = modelCallInfo(event);
      const button = element('button', 'button', `${info.type} · ${info.purpose}`);
      button.addEventListener('click', () => openModelEvent(event));
      origin.append(button);
    }
  }

  function showInventory() {
    renderInventorySnapshot(byId('inventory-board'), state.inventory.find(snapshot => String(snapshot.event.id) === state.inventoryId), event => {
      if (validFilePath(event.artifact)) { setView('data'); previewFile(event.artifact).catch(showError); }
      else { setView('graph'); showNode({ title: '清单记录', events: [event], startTime: event.t_s }, event); }
    });
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

  function observationDescription(observation) {
    return `${observation.observationId} · ${CHECKPOINT_NAMES[observation.checkpoint] ?? observation.checkpoint ?? '观测时机未记录'} · ${ANALYSIS_NAMES[observation.status] ?? observation.status}`;
  }

  function renderObservations() {
    state.observations = archivedObservations(state.events);
    state.observationFrames = groupObservationFrames(state.observations, state.events);
    const select = byId('observation-select');
    select.replaceChildren();
    const frames = new Set(state.observations.map(item => item.observationId)).size;
    byId('observation-count').textContent = `${frames} 帧 · ${state.observations.length} 次标注记录`;
    select.disabled = !state.observations.length;
    byId('observation-seek').disabled = true;
    for (const [index, frame] of state.observationFrames.entries()) {
      const option = element('option', '', `${index + 1}. ${frame.title} · ${Number.isFinite(frame.captureTime) ? formatReplayTime(frame.captureTime) : '拍摄时间未归档'}`);
      option.value = frame.id;
      select.append(option);
    }
    if (!state.observations.some(item => item.id === state.observationId)) {
      const first = state.observations[0];
      state.observationId = state.observations.find(item => item.observationId === first?.observationId && item.status === 'analyzed')?.id ?? first?.id ?? null;
    }
    showObservation(state.observationId);
  }

  function showObservation(id) {
    const observation = state.observations.find(item => item.id === id);
    byId('observation-cameras').replaceChildren();
    byId('observation-artifacts').replaceChildren();
    byId('observation-revision').replaceChildren();
    byId('observation-reuse-note').hidden = true;
    if (!observation) {
      byId('observation-meta').textContent = '本次实验未保存观测标注图。旧记录的 RGBD 数组可在数据页下载。';
      return;
    }
    state.observationId = observation.id;
    const frame = state.observationFrames.find(item => item.id === observation.observationId);
    const revision = frame.revisions.find(item => item.id === id);
    byId('observation-select').value = frame.id;
    for (const item of frame.revisions) {
      const option = element('option', '', `#${item.event.id} · ${item.revisionTitle}`);
      option.value = item.id;
      byId('observation-revision').append(option);
    }
    byId('observation-revision').value = id;
    byId('observation-meta').textContent = `${frame.title} · 拍摄于 ${Number.isFinite(frame.captureTime) ? formatReplayTime(frame.captureTime) : '时间未归档'} · ${frame.revisions.length} 个标注版本`;
    byId('observation-reuse-note').hidden = !revision.reusesCapture;
    byId('observation-reuse-note').textContent = `这是旧拍摄帧上的状态更新（事件 #${observation.event.id}，${formatReplayTime(observation.time)}），没有重新拍照。${revision.revisionTitle.includes('夹爪') ? '“夹持中”来自夹爪反馈，不是这张图的视觉判断。' : '像素沿用同次观测，仅目标标注变化。'}`;
    byId('observation-seek').disabled = !Number.isFinite(observation.time);
    for (const camera of observation.cameras) {
      const card = element('figure', 'observation-card');
      card.append(element('figcaption', '', camera.label));
      const open = element('button', 'observation-thumbnail');
      open.type = 'button';
      open.setAttribute('aria-label', `放大 ${camera.label} · ${observation.observationId}`);
      const image = element('img');
      image.src = fileUrl(state.runId, camera.path);
      image.alt = `${camera.label} · ${observationDescription(observation)}`;
      image.loading = 'lazy';
      const status = element('p', 'muted', '点击图片放大，可切换原始 RGB');
      image.addEventListener('error', () => { status.textContent = '图片读取失败：归档可能尚未写入或文件缺失。'; });
      open.append(image);
      open.addEventListener('click', () => openObservationImage(observation, camera));
      card.append(open, status);
      addDownload(card, camera.path, '下载标注图');
      byId('observation-cameras').append(card);
    }
    const artifacts = byId('observation-artifacts');
    if (validFilePath(observation.artifact)) {
      const button = element('button', 'button', '查看原始 RGBD / 标定引用');
      button.addEventListener('click', () => { setView('data'); previewFile(observation.artifact).catch(showError); });
      artifacts.append(button);
      addDownload(artifacts, observation.artifact, '下载 RGBD 引用 JSON');
    }
    for (const event of state.events.filter(item => item.kind === 'visual_decisions_call' && item.data?.observation_id === observation.observationId)) {
      if (!validFilePath(event.data.artifact)) continue;
      const info = modelCallInfo(event);
      const button = element('button', 'button', `${info.type} · ${info.purpose} · 事件 #${event.id}`);
      button.addEventListener('click', () => openModelEvent(event));
      artifacts.append(button);
    }
  }

  function imageZoom(factor = 1, reset = false) {
    state.imageZoom = reset ? 1 : Math.min(4, Math.max(0.5, state.imageZoom * factor));
    byId('observation-image').style.width = `${state.imageZoom * 100}%`;
    byId('observation-zoom-label').textContent = `${Math.round(state.imageZoom * 100)}%`;
  }

  function showAnnotatedImage() {
    state.imageOperation += 1;
    if (!state.image) return;
    byId('observation-image').src = state.image.url;
    byId('observation-image-status').textContent = '归档标注图 · 标注状态以本次记录为准';
    byId('observation-annotated').setAttribute('aria-pressed', 'true');
    byId('observation-raw').setAttribute('aria-pressed', 'false');
    byId('observation-raw').disabled = !state.image.camera.path.endsWith('.svg');
  }

  function openObservationImage(observation, camera) {
    const url = fileUrl(state.runId, camera.path);
    state.image = { url, camera, raw: null };
    byId('observation-image-title').textContent = `${camera.label} · ${observation.observationId}`;
    byId('observation-image-meta').textContent = observationDescription(observation);
    byId('observation-image').alt = `${camera.label} · ${observation.observationId}`;
    byId('observation-open-file').href = url;
    imageZoom(1, true);
    showAnnotatedImage();
    byId('observation-dialog').showModal();
  }

  async function showRawImage() {
    const image = state.image;
    if (!image) return;
    const operation = ++state.imageOperation;
    byId('observation-image-status').textContent = '正在读取归档中的原始 RGB…';
    byId('observation-raw').disabled = true;
    try {
      if (!image.raw) {
        const response = await fetch(image.url);
        if (!response.ok) throw new Error('归档图片读取失败');
        image.raw = embeddedRawRgb(await response.text());
      }
      if (operation !== state.imageOperation) return;
      if (!image.raw) throw new Error('此归档没有内嵌原始 RGB，可下载 RGBD 数组查看');
      byId('observation-image').src = image.raw;
      byId('observation-image-status').textContent = '原始 RGB · 来自同一归档帧，不含标注';
      byId('observation-annotated').setAttribute('aria-pressed', 'false');
      byId('observation-raw').setAttribute('aria-pressed', 'true');
    } catch (error) {
      if (operation === state.imageOperation) byId('observation-image-status').textContent = error.message;
    } finally {
      if (operation === state.imageOperation) byId('observation-raw').disabled = false;
    }
  }

  function clearObservationImage() {
    state.imageOperation += 1;
    state.image = null;
    byId('observation-image').removeAttribute('src');
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
      const inventory = element('button', 'button primary', '查看物体清单与工序状态');
      inventory.addEventListener('click', () => setView('inventory'));
      byId('node-artifacts').append(inventory);
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
      const observation = state.observations.find(record => record.event === item);
      if (observation && !events.some(previous => previous !== item && previous.id < item.id && previous.kind === 'observation_annotated' && previous.data?.observation_id === observation.observationId)) {
        const button = element('button', 'button', `查看观测与标注版本 · ${CHECKPOINT_NAMES[observation.checkpoint] ?? '归档图'}`);
        button.addEventListener('click', () => { showObservation(observation.id); setView('observations'); });
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
      if (file.path.endsWith('.json')) {
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
    if (operation !== state.operation) return;
    state.manifest = manifest;
    state.events = payload.events ?? [];
    state.graph = buildExecutionGraph(state.events);
    state.experiments = state.experiments.map(item => item.id === manifest.id ? manifest : item);
    renderSummary();
    renderList();
    renderFiles();
    renderObservations();
    renderInventory();
    renderGraph();
    loadVideos();
    scheduleRefresh();
  }

  function renderGraph() {
    graphView.render(state.graph);
    const stages = renderReplayStages(byId('object-timeline'), state.graph, (node, event) => {
      showNode(node, event);
      if (window.matchMedia?.('(max-width: 1250px)').matches) byId('node-title').scrollIntoView?.({ behavior: 'smooth', block: 'start' });
    });
    if (!state.selectedNode && stages.length) showNode(stages[0]);
    byId('graph-warning').textContent = state.graph.warnings.join(' ');
    byId('graph-warning').hidden = state.graph.warnings.length === 0;
    byId('export-svg').disabled = state.graph.nodes.length === 0;
    byId('export-png').disabled = state.graph.nodes.length === 0;
  }

  async function selectRun(runId, time = 0) {
    clearMessages();
    stopPlayback();
    window.clearTimeout(state.poll);
    const operation = ++state.operation;
    state.fileOperation += 1;
    evidence.clear();
    fileEvidence.clear();
    state.runId = runId;
    state.manifest = null;
    state.selectedNode = null;
    state.observationId = null;
    state.inventoryId = null;
    clearObservationImage();
    if (byId('observation-dialog').open) byId('observation-dialog').close();
    state.time = time;
    state.master = null;
    byId('experiment-content').hidden = true;
    byId('empty-selection').hidden = false;
    byId('empty-selection').replaceChildren(element('h2', '', '正在载入实验…'));
    renderList();
    const base = `/api/experiments/${encodeURIComponent(runId)}`;
    try {
      const [manifest, payload] = await Promise.all([request(base), request(`${base}/events`)]);
      if (operation !== state.operation) return;
      state.manifest = manifest;
      state.events = payload.events ?? [];
      state.graph = buildExecutionGraph(state.events);
      byId('empty-selection').hidden = true;
      byId('experiment-content').hidden = false;
      byId('node-title').textContent = '节点详情';
      byId('node-meta').textContent = '选择上方阶段或完整调用图节点查看记录。';
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
      graphView.expanded.clear();
      graphView.reset();
      renderSummary({ replaceName: true });
      renderGraph();
      renderFiles();
      renderObservations();
      renderInventory();
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
    try {
      const manifest = await request(`/api/experiments/${encodeURIComponent(runId)}/save`, { name: byId('experiment-name').value.trim() });
      if (runId === state.runId) { state.manifest = manifest; renderSummary({ replaceName: true }); }
      await loadList();
      notify(wasPending ? '实验已保存。' : '实验名称已保存。');
    } catch (error) { showError(error); }
    finally { state.mutating = false; renderSummary(); scheduleRefresh(); }
  }

  async function discardExperiment() {
    if (!state.manifest || byId('discard-experiment').disabled || state.mutating) return;
    const runId = state.runId;
    stopPlayback();
    // Release video range requests before asking the archive store to move its files.
    for (const track of tracks) { track.video.removeAttribute('src'); track.video.load(); }
    state.mutating = true;
    state.operation += 1;
    window.clearTimeout(state.poll);
    renderSummary();
    clearMessages();
    try {
      const response = await fetch(`/api/experiments/${encodeURIComponent(runId)}`, { method: 'DELETE' });
      const payload = await response.json();
      if (!response.ok || payload.deleted !== true) {
        const reasons = { experiment_busy: '实验仍在录制或编码，请完成后再删除。', experiment_in_use: '实验文件正在下载或导出，请结束后再删除。' };
        throw new Error(reasons[payload.detail] ?? payload.detail ?? '实验删除失败');
      }
      if (runId === state.runId) {
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
        byId('experiment-content').hidden = true;
        byId('empty-selection').hidden = false;
        byId('empty-selection').replaceChildren(element('h2', '', '选择一条实验记录'));
        updateUrl();
      }
      await loadList();
      notify('实验已移入本地回收区，原始视频与数据仍保留。');
    } catch (error) { showError(error); if (runId === state.runId && state.manifest) loadVideos(); }
    finally { state.mutating = false; renderSummary(); scheduleRefresh(); }
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
    const svg = graphSvg(state.graph, { title: state.manifest.name || '实验执行图', expanded: graphView.expanded });
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const filename = `execution-${state.runId}`;
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
  byId('experiment-search').addEventListener('input', renderList);
  byId('file-search').addEventListener('input', renderFiles);
  byId('refresh-list').addEventListener('click', async () => {
    clearMessages();
    byId('refresh-list').disabled = true;
    try { await loadList(); if (state.runId) await refreshSelected(); }
    catch (error) { showError(error); }
    finally { byId('refresh-list').disabled = false; }
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
  byId('discard-dialog').addEventListener('close', () => { if (byId('discard-dialog').returnValue === 'discard' && discardRunId === state.runId) discardExperiment(); });
  byId('export-svg').addEventListener('click', () => exportGraph('svg').catch(showError));
  byId('export-png').addEventListener('click', () => exportGraph('png').catch(showError));
  byId('graph-zoom-in').addEventListener('click', () => graphView.zoom(1.25));
  byId('graph-zoom-out').addEventListener('click', () => graphView.zoom(0.8));
  byId('graph-fit').addEventListener('click', () => graphView.fit());
  byId('graph-reset').addEventListener('click', () => graphView.reset());
  byId('inventory-snapshot').addEventListener('change', () => { state.inventoryId = byId('inventory-snapshot').value; showInventory(); });
  byId('observation-select').addEventListener('change', () => {
    const frame = state.observationFrames.find(item => item.id === byId('observation-select').value);
    if (frame) showObservation(frame.preferredId);
  });
  byId('observation-revision').addEventListener('change', () => showObservation(byId('observation-revision').value));
  byId('observation-seek').addEventListener('click', () => {
    const frame = state.observationFrames.find(item => item.revisions.some(revision => revision.id === state.observationId));
    if (frame && Number.isFinite(frame.captureTime)) { stopPlayback(); setView('video'); seek(frame.captureTime); }
  });
  byId('observation-zoom-in').addEventListener('click', () => imageZoom(1.25));
  byId('observation-zoom-out').addEventListener('click', () => imageZoom(0.8));
  byId('observation-zoom-fit').addEventListener('click', () => imageZoom(1, true));
  byId('observation-annotated').addEventListener('click', showAnnotatedImage);
  byId('observation-raw').addEventListener('click', showRawImage);
  byId('observation-close').addEventListener('click', () => byId('observation-dialog').close());
  byId('observation-dialog').addEventListener('close', clearObservationImage);
  byId('observation-image').addEventListener('error', () => { byId('observation-image-status').textContent = '图片加载失败，请检查归档文件。'; });
  byId('standalone-graph').addEventListener('click', updateUrl);
  window.addEventListener('pagehide', () => { stopPlayback(); window.clearTimeout(state.poll); });
  loadList().then(() => {
    const id = initialRun || state.experiments[0]?.id;
    if (id) return selectRun(id, initialTime);
  }).catch(showError);
}

if (typeof document !== 'undefined') mountReplayPage(document);

import { mountWorkbench } from './workbench.js';
import { renderFailureReport } from './failure-report.js';
import './styles.css';
import { request } from './api.js';
import { runConclusion } from './run-conclusion.js';
import { controlsForState, formatJson, mergeEvents, performanceSummary, candidateCount } from './model.js';
import { createBaseView } from './base-view.js';
import { createTraceList } from './trace.js';
import { createTaskClock } from './task-clock.js';
import { createPlanningSettingsEditor } from './planning-settings.js';
import { createCameraPanel } from './camera-panel.js';
import { createPointcloudPanel } from './pointcloud-panel.js';
import { runEvidence, runStatusSummary } from './run-evidence.js';
import { createObjectList } from './object-list.js';

const byId = id => document.getElementById(id);
const objectList = createObjectList(byId('task-object-list'));
const renderTrace = createTraceList(byId('trace-list'), byId('event-count'));
const taskClock = createTaskClock();
const state = { connected: false, busy: false, dirty: false, dragging: false, planningLoading: false, snapshot: null, run: {}, events: [], draft: null, selectedId: null, manifestRevision: null, operation: 0, lastServerTime: 0, receivedAt: 0 };
let activeView = 'scene';
let pointcloudOpen = null;
const holdingSummary = element('p', 'run-holding-summary');
byId('run-details').append(holdingSummary);
const planningEditor = createPlanningSettingsEditor(byId('planning-groups'), { document, onChange: renderControls });
const cameraPanel = createCameraPanel(byId('camera-grid'), byId('camera-status'), {
  document, request, schedule: setTimeout, cancel: clearTimeout,
  isVisible: () => document.visibilityState !== 'hidden' && !byId('workspace').hidden && activeView === 'cameras' && byId('camera-panel').open !== false,
});
cameraPanel.start();
byId("camera-mode").addEventListener("change", event => cameraPanel.setMode(event.target.value));
let view = null;
const pointcloudPanel = createPointcloudPanel(byId('pointcloud-content'), byId('pointcloud-status'), {
  document, request, onOverlay: overlay => {
    view?.setOverlay(overlay);
    byId('hide-pointcloud').hidden = !overlay;
  },
});
let manifestLoading = false;
let manifestWanted = false;
let lastRunError = null;
let serverInstanceId = null;
let serverReloading = false;
function errorMessage(error) {
  const message = error instanceof Error ? error.message : String(error);
  try {
    const details = JSON.parse(message);
    if (Array.isArray(details)) return details.map(item => `${(item.loc ?? []).filter(part => part !== 'body').join(' / ')}：${item.msg ?? '参数无效'}`).join('\n');
  } catch { /* Plain backend error codes are handled below. */ }
  const hints = {
    confirm_scene_first: '请先点击「重力稳定」，再「确认场景」。',
    run_already_active: '任务正在运行。请等待完成，或点击 Reset 取消。',
    run_active_reset_first: '任务正在运行。请点击 Reset 后再编辑场景。',
    stale_start_after_reset: '场景已经重置或修改，请确认当前场景后重新开始。',
    task_not_implemented: '任务类型未受支持。请使用抓取、放置或抓取并放置指令。',
    place_requires_verified_holding: '单独放置需要本次后端会话中已验证的持物状态。请先完成抓取；Reset 或场景重建后需重新抓取。',
    low_confidence: '当前视觉判断的置信度不足。展开模型记录查看具体答案和置信度。',
    skill_contract_not_satisfied: '技能前提未满足，技能程序已停止。展开记录查看缺失证据。',
    skill_selection_budget_exhausted: '连续无效 skill 调用已达上限，任务等待处理；物理动作未因这些调用重复执行。',
    no_reachable_arm: '源与目的未同时命中可用机械臂工作空间。展开选臂记录查看证据。',
    arm_already_attempted: '该机械臂本轮已尝试规划，任务已停止。展开选臂与规划记录查看失败证据。',
    grasp_verification_failed: '抓取未通过视觉验证。展开运行事实查看失败证据。',
    place_verification_failed: '放置未通过视觉验证。展开运行事实查看目标分离与可见放置区域的观测证据。',
    holding_state_reset_required: '当前仍有持物或待确认释放记录，请先完成放置或 Reset 后编辑/确认场景。',
    algorithm_cleanup_failed_reset_to_retry: '算法状态清理失败。请 Reset 后重试，清理成功前不会开始新任务。',
    release_clearance_conflict: '当前物体离盒底高度与携物表面球或碰撞间距冲突，任务已停止。请检查放置高度和实际可见点云。',
    destination_requires_measured_placement_reference: '相机未测得有效放置参照点，无法定位放置目标。',
    destination_requires_measured_floor_reference: '相机未测得有效盒底高度，无法计算相对释放高度。',
    release_requires_grasp_support_reference: '缺少抓取时的 TCP 与桌面支撑参照，无法计算物体离盒底高度。',
    target_missing: '未找到匹配的场景物体。请核对颜色、类型和指令。',
    target_ambiguous: '目标不唯一。请修改物体属性或给出更明确的指令。',
    scene_moved_since_planning: '物体在规划后移动。请 Reset，等待场景稳定后重试。',
    algorithm_request_pending: '后台算法调用尚未返回，请等待后再开始新任务。Reset 和场景编辑仍可使用。',
    plan_grasp_timeout: '抓取规划超过总时限，任务已停止，迟到结果不会执行。',
    plan_place_timeout: '放置规划超过总时限，任务已停止，迟到结果不会执行。',
    planning_settings_changed: '算法参数已被其他页面修改。请重新读取参数，检查后再应用或开始任务。',
  };
  const hint = hints[message.split(':')[0]];
  return hint ? `${hint}（${message}）` : `操作未完成：${message}。当前证据不足以确定更具体原因。`;
}

function showError(error, { recorded = false, source = '' } = {}) {
  const message = recorded ? String(error) : errorMessage(error);
  if (byId('planning-dialog').open) {
    byId('planning-error').textContent = message;
    byId('planning-error').hidden = false;
    return;
  }
  const summary = message.split('\n')[0].split('（')[0];
  byId('error-message').textContent = summary.length > 100 ? `${summary.slice(0, 100)}…` : summary;
  byId('error-details').textContent = message;
  byId('error-disclosure').open = false;
  byId('error-banner').hidden = false;
  byId('error-banner').dataset.source = source;
}

function runFailure() {
  const errors = [];
  const conclusion = runConclusion(state.run, state.events, errorMessage);
  if (conclusion) errors.push(conclusion);
  else if (state.run.error) errors.push(errorMessage(state.run.error));
  const failure = state.run.failure ?? state.run.task_progress?.failure;
  if (state.run.status === 'running' && failure?.code) {
    errors.push(`${errorMessage(failure.code)} 本次技能程序已停止，请查看失败证据。`);
  }
  if (state.run.algorithm_cleanup_error) {
    errors.push(errorMessage(`algorithm_cleanup_failed_reset_to_retry:${state.run.algorithm_cleanup_error}`));
  }
  return errors.join('\n');
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderControls() {
  const inputs = { connected: state.connected, busy: state.busy, dirty: state.dirty, confirmed: state.snapshot?.confirmed, status: state.run.status, instruction: byId('instruction').value };
  const controls = controlsForState({ ...inputs, busy: state.busy || state.dragging });
  byId('prepare-devices').disabled = state.busy || state.run.status === 'running';
  byId('start').disabled = controls.startDisabled || Boolean(state.run.algorithm_pending || state.run.algorithm_cleanup_error) || state.planningLoading || !planningEditor.ready || planningEditor.dirty;
  byId('reset').disabled = controls.resetDisabled;
  byId('stop-task').disabled = !state.connected || state.run.status !== 'running';
  byId('instruction').disabled = controls.editDisabled;
  for (const id of ['experiment-name', 'record-video', 'experiment-save-mode']) byId(id).disabled = controls.editDisabled;
  const saveDisabled = byId('experiment-save-mode').value === 'none';
  byId('record-video').disabled = controls.editDisabled || saveDisabled;
  byId('record-video-label').textContent = saveDisabled ? '不录视频' : '录制视频';
  byId('refresh-health').disabled = state.busy;
  const planningLocked = controls.editDisabled || state.planningLoading || Boolean(state.run.algorithm_pending);
  const planningActive = state.run.status === 'running' || Boolean(state.run.algorithm_pending);
  if (planningActive && !planningEditor.advancedLocked) planningEditor.setAdvancedLocked(true);
  byId('planning-fields').disabled = planningLocked || !planningEditor.ready;
  byId('toggle-advanced-planning').disabled = planningLocked || !planningEditor.ready;
  byId('toggle-advanced-planning').textContent = planningEditor.advancedLocked ? '解锁高级参数' : '锁定高级参数';
  byId('toggle-advanced-planning').ariaPressed = String(!planningEditor.advancedLocked);
  byId('advanced-planning-status').textContent = planningActive ? '任务运行中或后台算法尚未结束，所有参数暂不可编辑；高级参数已锁定。' : planningEditor.advancedLocked ? '实验调参已锁定，常用执行时间可编辑；系统标定始终只读。' : '实验调参已解锁。修改后仍需应用；重新锁定会保留参数草稿。';
  byId('apply-planning').disabled = planningLocked || !planningEditor.ready || !planningEditor.dirty || planningEditor.conflict;
  byId('reload-planning').disabled = planningLocked;
  byId('planning-status').textContent = state.planningLoading ? '正在读取算法参数…' : !planningEditor.ready ? '算法参数尚未加载；读取成功后才能开始任务。' : planningEditor.conflict ? '其他页面已更新算法参数。草稿已保留；请重新读取并检查后再修改。' : planningEditor.dirty ? '参数有未应用修改。点击「应用算法参数」后才能开始任务。' : '算法参数已应用。Reset 保留这些参数。';
  const confirmed = state.snapshot?.confirmed;
  byId('scene-badge').textContent = !state.connected ? '待连接' : confirmed ? '反馈就绪' : '待准备';
  byId('scene-badge').classList.toggle('confirmed', Boolean(confirmed));
  byId('scene-note').textContent = 'Base 管理设备与标定，仿真场景在 Luna 编辑。任务采集须通过三相机同步和稳定检查。';
  byId('start-hint').textContent = state.busy ? '请求处理中…' : state.run.status === 'running' ? '任务运行中' : state.run.algorithm_pending ? '等待后台算法返回' : !confirmed ? '点击准备设备，等待实测反馈和三相机就绪。' : planningEditor.dirty ? '请先应用算法参数。' : '设备就绪；每次任务观测重新检查三相机。';
  byId('open-planning').classList.toggle('has-draft', planningEditor.dirty);
}

function renderExperiment() {
  const experiment = state.run.experiment;
  byId('experiment-result').hidden = !experiment;
  if (!experiment) return;
  const retention = { saved: '已归档', pending: '待决定保存', none: '不保存' }[experiment.retention] ?? experiment.retention ?? '记录中';
  const video = { pending: '视频待处理', recording: '视频录制中', encoding: '视频编码中', ready: '视频就绪', completed: '视频就绪',
    saved: '视频就绪', disabled: '未录制视频', off: '未录制视频', skipped: '未录制视频', partial: '视频不完整',
    interrupted: '视频录制中断', unavailable: '视频不可用', failed: '视频处理失败', error: '视频处理失败' }[experiment.video_status] ?? experiment.video_status ?? '视频状态待更新';
  byId('experiment-status').textContent = `${experiment.name || '本次实验'} · ${retention} · ${video}${experiment.error ? ` · 保存/录制失败：${experiment.error}` : ''}`;
  byId('experiment-result').classList.toggle('has-error', Boolean(experiment.error) || ['failed', 'error', 'partial', 'interrupted', 'unavailable'].includes(experiment.video_status));
  byId('experiment-replay').hidden = !experiment.id;
  byId('experiment-replay').href = experiment.id ? `/#replay?run=${encodeURIComponent(experiment.id)}` : '/#replay';
}

function selectView(name) {
  activeView = name;
  for (const viewName of ['scene', 'cameras', 'pointcloud']) {
    const selected = viewName === name;
    byId(`view-tab-${viewName}`).ariaSelected = String(selected);
    byId(`view-tab-${viewName}`).tabIndex = selected ? 0 : -1;
  }
  byId('viewport-shell').hidden = name === 'cameras';
  byId('camera-panel').hidden = name !== 'cameras';
  byId('camera-panel').open = name === 'cameras';
  byId('pointcloud-panel').hidden = name !== 'pointcloud';
  byId('pointcloud-panel').open = name === 'pointcloud';
  byId('viewer-content').classList.toggle('show-pointcloud', name === 'pointcloud');
  updatePointcloudOpen();
  if (name === 'cameras') void cameraPanel.refresh();
}

function updatePointcloudOpen() {
  const open = byId('pointcloud-panel').open;
  if (pointcloudOpen === open) return;
  pointcloudOpen = open;
  void pointcloudPanel.setOpen(open);
}

function renderRunDetails() {
  const running = state.run.status === 'running';
  const step = running ? state.run.current_step : state.run.failed_step;
  const failure = runFailure();
  const pending = !running && Boolean(state.run.algorithm_pending);
  holdingSummary.textContent = runStatusSummary(state.run);
  holdingSummary.hidden = !holdingSummary.textContent;
  byId('run-details').hidden = !running && !step && !failure && !pending && !holdingSummary.textContent;
  byId('run-details').classList.toggle('has-error', Boolean(failure));
  const decisionNames = { bind_task: '绑定任务', bind_attributes: '绑定物体属性', choose_skill: '执行技能程序', select_arguments: '校验技能参数' };
  const stepName = decisionNames[step?.skill] ?? step?.skill;
  byId('run-step').textContent = stepName ? `${running ? '当前步骤' : '失败步骤'}：${stepName}${step?.stage ? ` / ${step.stage}` : ''}` : running ? '任务处理中' : '任务状态';
  let seconds = step?.duration_s;
  if (running && Number.isFinite(step?.started_at) && state.lastServerTime) {
    seconds = Math.max(0, state.lastServerTime - step.started_at + (performance.now() - state.receivedAt) / 1000);
  }
  byId('run-step-timing').hidden = !Number.isFinite(seconds);
  byId('run-step-timing').textContent = Number.isFinite(seconds)
    ? `${running ? '已等待' : '步骤耗时'} ${seconds.toFixed(1)} s${Number.isFinite(step?.timeout_s) ? ` · 总时限 ${step.timeout_s} s` : ''}` : '';
  if (running && step?.stage && Number.isFinite(step.stage_started_at) && state.lastServerTime) {
    const stageSeconds = Math.max(0, state.lastServerTime - step.stage_started_at + (performance.now() - state.receivedAt) / 1000);
    byId('run-step-timing').hidden = false;
    byId('run-step-timing').textContent += ` · ${step.stage} ${stageSeconds.toFixed(1)} s`;
  }
  byId('run-candidate').hidden = !Number.isFinite(step?.batch_index);
  const ikStep = step?.stage === 'ik_batch' || step?.skill === 'evaluate_arms';
  const candidates = ikStep ? `候选 ${candidateCount(step.candidate_indices) ?? '—'} 个`
    : `候选编号 ${(step?.candidate_indices ?? []).join(', ')}`;
  byId('run-candidate').textContent = Number.isFinite(step?.batch_index)
    ? `${ikStep ? 'Batched IK 批次' : '规划请求'} ${step.request_index ?? step.batch_index} · ${candidates} · Arm ${step.arm ?? '—'}${Number.isFinite(step.success_count) ? ` · ${ikStep ? '通过' : '完整执行检查通过'} ${step.success_count} 个` : ''}` : '';
  byId('run-last-error').hidden = !running || !step?.last_error;
  byId('run-last-error').textContent = step?.last_error ? `上一批次结果：${step.last_error}` : '';
  renderFailureReport(byId('run-failure'), state.run, state.events, { document,
    additionalErrors: state.run.algorithm_cleanup_error
      ? [errorMessage(`algorithm_cleanup_failed_reset_to_retry:${state.run.algorithm_cleanup_error}`)] : [] });
  byId('algorithm-pending').hidden = !pending;
}

function setConnection(connected) {
  state.connected = connected;
  byId('connection').classList.toggle('online', connected);
  byId('connection').replaceChildren(element('i'), document.createTextNode(connected ? '实时连接' : '连接中断 · 重连中'));
  renderControls();
}

async function refreshManifest() {
  if (!view) return;
  if (manifestLoading) { manifestWanted = true; return; }
  manifestLoading = true;
  try {
    const manifest = await request('/api/manifest');
    await view.loadManifest(manifest);
    state.manifestRevision = manifest.revision;
    byId('viewport-loading').hidden = true;
  } catch (error) {
    showError(`场景显示失败：${error.message}。请确认后端资源服务可访问后刷新页面。`);
  } finally {
    manifestLoading = false;
    if (manifestWanted) { manifestWanted = false; void refreshManifest(); }
  }
}

function applyState(payload) {
  const snapshot = payload?.simulation;
  if (!snapshot || serverReloading) return;
  const instanceId = payload.server_instance_id;
  if (typeof instanceId === 'string' && instanceId) {
    // Backend reload starts a new scene whose generation can be lower.
    if (serverInstanceId !== null && instanceId !== serverInstanceId) {
      serverReloading = true;
      location.reload();
      return;
    }
    serverInstanceId = instanceId;
  }
  if (snapshot.generation < (state.snapshot?.generation ?? 0)) return;
  if (Number.isFinite(payload.server_time) && payload.server_time < state.lastServerTime) return;
  if (Number.isFinite(payload.server_time)) {
    state.lastServerTime = payload.server_time;
    state.receivedAt = performance.now();
  }
  state.snapshot = snapshot;
  const mode = snapshot.runtime_mode;
  byId('connection-mode').textContent = `${mode === 'simulation' ? '仿真' : mode === 'real' ? '真机' : '设备未确认'} · ${snapshot.robot_ip ?? ''}`;
  byId('simulation-scene-link').hidden = mode !== 'simulation';
  cameraPanel.setGeneration(snapshot.generation);
  pointcloudPanel.setGeneration(snapshot.generation);
  state.run = payload.run ?? {};
  renderExperiment();
  const names = { idle: snapshot.confirmed ? '设备就绪' : '等待设备', running: '任务运行中', succeeded: '任务成功', success: '任务成功', completed: '任务完成', partial_success: '部分完成', needs_attention: '结果待确认', failed: '任务失败', error: '任务出错', cancelled: '任务已取消', unsupported: '暂不支持此任务' };
  byId('run-status').textContent = state.run.algorithm_cleanup_error ? '算法清理失败' : names[state.run.status] ?? state.run.status ?? '等待场景';
  byId('run-dot').className = `run-dot ${state.run.status === 'running' ? 'active' : ''} ${state.run.algorithm_cleanup_error || ['failed', 'error'].includes(state.run.status) ? 'failed' : ''}`;
  byId('task-value').textContent = ({ grasp: '抓取', place: '放置', grasp_place: '抓取并放置', unsupported: '不支持' })[state.run.task] ?? state.run.task ?? '—';
  byId('target-value').textContent = state.run.target_id ?? '—';
  byId('destination-value').textContent = state.run.destination_id ?? '—';
  byId('arm-value').textContent = state.run.arm ? `Arm ${state.run.arm}` : '—';
  const failure = runFailure();
  if (failure && (failure !== lastRunError || byId('error-banner').dataset.source === 'feedback')) showError(failure, { recorded: true });
  lastRunError = failure;
  if (snapshot.physics_error && !failure) showError(`Base 反馈不可用：${snapshot.physics_error}`, {source: 'feedback'});
  else if (byId('error-banner').dataset.source === 'feedback') {
    byId('error-banner').hidden = true;
    byId('error-banner').dataset.source = '';
  }
  const previousEvents = state.events;
  state.events = mergeEvents(state.events, payload.events, payload.events_mode);
  planningEditor.sync(payload.planning_settings);
  taskClock.sync(state.run, state.events, payload.server_time);
  renderTaskClock();
  if (state.events !== previousEvents) renderTrace(state.events);
  const facts = state.run.facts;
  objectList.render(state.run.task_progress, state.run.task_bindings,
    state.run.terminal_recovery_graph ?? state.run.policy_graph, state.events, state.run.run_id);
  const metrics = state.run.metrics;
  byId('performance-summary').hidden = !metrics;
  if (metrics) {
    const summary = performanceSummary(metrics, state.events);
    byId('metric-count').textContent = summary.count;
    byId('metric-mean').textContent = summary.mean;
    byId('metric-total').textContent = summary.total;
    byId('metric-run').textContent = summary.run;
  }
  byId('facts-section').hidden = !metrics && !state.run.holding && !state.run.verification && !state.run.collision_model && !state.run.task_progress && (!facts || !Object.keys(facts).length);
  byId('facts-content').textContent = formatJson(runEvidence(state.run));
  byId('run-planning-settings').hidden = !state.run.planning_settings;
  byId('run-planning-settings-content').textContent = state.run.planning_settings
    ? formatJson({ revision: state.run.settings_revision, settings: state.run.planning_settings }) : '';
  view?.update(snapshot);
  if (snapshot.manifest_revision !== undefined && snapshot.manifest_revision !== state.manifestRevision && !manifestLoading) void refreshManifest();
  renderControls();
}

function renderTaskClock() {
  const seconds = taskClock.elapsedSeconds().toFixed(2);
  byId('sim-time').textContent = seconds;
  byId('camera-task-time').textContent = seconds;
  renderRunDetails();
}

async function refreshState(operation = state.operation) {
  const payload = await request('/api/state');
  if (operation !== state.operation) return null;
  setConnection(true);
  applyState(payload);
  return payload;
}

async function perform(path, body = {}, options = {}) {
  const operation = ++state.operation;
  let requestSucceeded = false;
  state.busy = true;
  byId('error-banner').hidden = true;
  byId('planning-error').hidden = true;
  renderControls();
  try {
    const result = await request(path, body);
    requestSucceeded = true;
    if (operation !== state.operation) return null;
    options.afterSuccess?.(result);
    const payload = await refreshState(operation);
    if (operation !== state.operation) return null;
    if (options.manifest) await refreshManifest();
    return payload;
  } catch (error) {
    if (operation === state.operation) {
      showError(error);
      if (!requestSucceeded) options.onError?.(error);
    }
    return null;
  } finally {
    if (operation === state.operation) { state.busy = false; renderControls(); }
  }
}

async function refreshHealth() {
  try {
    const payload = await request('/api/health');
    const health = payload.services ?? payload;
    const entries = Object.entries(health).filter(([name]) => !['time', 'timestamp', 'status', 'ok'].includes(name));
    byId('services').replaceChildren(...entries.map(([name, info]) => {
      const row = element('div', 'service-row');
      const value = typeof info === 'object' && info !== null ? info : { status: info };
      const status = value.ready ?? value.status ?? value.ok ?? value.healthy ?? value.available ?? value.configured;
      const healthy = status === true || ['ok', 'ready', 'healthy', 'online', 'available', 'configured'].includes(status);
      const label = healthy ? '可用' : status === false ? '未就绪' : typeof status === 'string' ? status : '查看详情';
      const badge = element('span', `service-state ${healthy ? 'healthy' : ''}`, label);
      const details = element('details', 'service-details');
      const summary = element('summary');
      const title = name === 'luna' && value.responses_model
        ? `Responses ${value.responses_model} (${value.responses_reasoning_effort}) · Decisions ${value.decisions_model}`
        : name;
      summary.append(element('span', '', title), badge);
      details.append(summary, element('pre', '', formatJson(info)));
      row.append(details);
      return row;
    }));
  } catch (error) { byId('services').replaceChildren(element('span', 'service-error', `服务检测失败：${error.message}`)); }
}

async function refreshPlanningSettings(discardDraft = false) {
  state.planningLoading = true;
  renderControls();
  try {
    const payload = await request('/api/planning-settings');
    if (discardDraft) planningEditor.accept(payload);
    else planningEditor.load(payload);
  } catch (error) {
    showError(`无法读取算法参数：${error.message}`);
  } finally {
    state.planningLoading = false;
    renderControls();
  }
}

function connectSocket() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws`);
  socket.addEventListener('open', () => { setConnection(true); });
  socket.addEventListener('message', event => {
    try { applyState(JSON.parse(event.data)); } catch (error) {
      console.error('实时状态渲染失败', error);
      showError(`实时状态显示异常。后台任务状态请以任务结果和实验记录核对。\n${error.stack ?? error.message}`, { recorded: true });
    }
  });
  socket.addEventListener('close', () => { setConnection(false); setTimeout(connectSocket, 2000); });
  socket.addEventListener('error', () => socket.close());
}

byId('dismiss-error').addEventListener('click', () => { byId('error-banner').hidden = true; });
byId('prepare-devices').addEventListener('click', () => void perform('/api/prepare'));
for (const name of ['scene', 'cameras', 'pointcloud']) {
  byId(`view-tab-${name}`).addEventListener('click', () => selectView(name));
  byId(`view-tab-${name}`).addEventListener('keydown', event => {
    const names = ['scene', 'cameras', 'pointcloud'];
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'scene' : event.key === 'End' ? 'pointcloud'
      : names[(names.indexOf(name) + (event.key === 'ArrowRight' ? 1 : 2)) % names.length];
    selectView(next);
    byId(`view-tab-${next}`).focus();
  });
}
byId('open-planning').addEventListener('click', () => byId('planning-dialog').showModal());
byId('close-planning').addEventListener('click', () => byId('planning-dialog').close());
for (const category of ['common', 'experimental']) {
  byId(`planning-tab-${category}`).addEventListener('click', () => {
    planningEditor.setCategory(category);
    for (const name of ['common', 'experimental']) byId(`planning-tab-${name}`).ariaPressed = String(name === category);
  });
}
selectView('scene');
byId('instruction').addEventListener('input', renderControls);
byId('experiment-save-mode').addEventListener('change', renderControls);
byId('hide-pointcloud').addEventListener('click', () => pointcloudPanel.setVisible(false));
byId('pointcloud-panel').addEventListener('toggle', updatePointcloudOpen);
byId('refresh-health').addEventListener('click', () => void refreshHealth());
byId('toggle-advanced-planning').addEventListener('click', () => {
  if (byId('toggle-advanced-planning').disabled) return;
  planningEditor.setAdvancedLocked(!planningEditor.advancedLocked);
  renderControls();
});
byId('reload-planning').addEventListener('click', () => {
  if (!byId('reload-planning').disabled) void refreshPlanningSettings(true);
});
byId('apply-planning').addEventListener('click', () => {
  if (byId('apply-planning').disabled) return;
  try {
    void perform('/api/planning-settings', { settings: planningEditor.read(), expected_revision: planningEditor.revision }, {
      afterSuccess: result => planningEditor.accept(result),
      onError: error => { if (error.message.startsWith('planning_settings_changed')) planningEditor.markConflict(); },
    });
  } catch (error) { showError(error); }
});
byId('stop-task').addEventListener('click', () => void perform('/api/stop'));
byId('reset').addEventListener('click', () => {
  void perform('/api/reset', {}, { manifest: true, afterSuccess: () => { taskClock.reset(); renderTaskClock(); } });
});
byId('start').addEventListener('click', () => {
  if (byId('start').disabled) return;
  planningEditor.setAdvancedLocked(true);
  taskClock.onStart();
  renderTaskClock();
  void perform('/api/start', { instruction: byId('instruction').value.trim(), generation: state.snapshot.generation, settings_revision: planningEditor.revision,
    experiment: { name: byId('experiment-name').value.trim(), record_video: byId('experiment-save-mode').value !== 'none' && byId('record-video').checked, save_mode: byId('experiment-save-mode').value } }, {
    onError: () => { taskClock.cancelPending(); renderTaskClock(); },
  });
});

try { view = createBaseView(byId('viewport')); } catch (error) { showError(error); }
mountWorkbench({onTaskVisibility: active => view?.setActive(active)});
void refreshState().catch(error => { setConnection(false); showError(`无法连接后端：${error.message}。请检查服务是否启动。`); });
void refreshManifest();
void refreshHealth();
void refreshPlanningSettings();
connectSocket();
setInterval(() => { if (!state.connected) void refreshState().catch(() => {}); }, 5000);
setInterval(() => void refreshHealth(), 15000);
setInterval(renderTaskClock, 100);

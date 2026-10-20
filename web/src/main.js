import './styles.css';
import { request } from './api.js';
import { runConclusion } from './run-conclusion.js';
import { buildScene, controlsForState, formatJson, sceneFromSnapshot, mergeEvents, performanceSummary, candidateCount } from './model.js';
import { createSceneView } from './scene-view.js';
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
let sceneEditorOpen = true;
let pointcloudOpen = null;
const holdingSummary = element('p', 'run-holding-summary');
byId('run-details').append(holdingSummary);
const planningEditor = createPlanningSettingsEditor(byId('planning-groups'), { document, onChange: renderControls });
const cameraPanel = createCameraPanel(byId('camera-grid'), byId('camera-status'), {
  document, request, schedule: setTimeout, cancel: clearTimeout,
  isVisible: () => document.visibilityState !== 'hidden' && activeView === 'cameras' && byId('camera-panel').open !== false,
});
cameraPanel.start();
byId("camera-mode").addEventListener("change", event => cameraPanel.setMode(event.target.value));
const colorValues = { red: '#ed6a66', blue: '#648af0', green: '#62b893', yellow: '#dec267', orange: '#dd9b5f', gray: '#929ba6', white: '#dddfe6', black: '#45515c' };
const kindNames = { box: '长方体', cube: '立方体', sphere: '球体', cylinder: '圆柱体', bin: '开放物料盒', marker: '桌面放置标记' };
let view = null;
const pointcloudPanel = createPointcloudPanel(byId('pointcloud-content'), byId('pointcloud-status'), {
  document, request, onOverlay: overlay => {
    view?.setOverlay(overlay);
    byId('hide-pointcloud').hidden = !overlay;
  },
  onEnvironmentVisible: visible => view?.setEnvironmentVisible(visible),
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
    TYPESAFE_API_KEY_not_configured: '请在后端 .env 中配置 TYPESAFE_API_KEY，然后重启服务。',
    low_confidence: 'JEV 对当前选择的置信度不足。展开对应 JEV 决策，查看具体选项和置信度。',
    skill_contract_not_satisfied: 'JEV 选择的 skill 不满足执行条件，本次调用未执行。未满足的条件会返回 JEV 重新选择。',
    skill_selection_budget_exhausted: '连续无效 skill 调用已达上限，任务等待处理；物理动作未因这些调用重复执行。',
    no_reachable_arm: 'JEV 未选择可执行的机械臂。展开选臂记录，查看选择结果和两臂 IK 证据。',
    arm_already_attempted: '该机械臂本轮已尝试规划，任务已停止；每臂最多尝试一次。展开选臂与规划无解记录，查看 JEV 的选择和失败证据。',
    grasp_verification_failed: '抓取未通过相机与夹爪反馈验证。展开运行事实查看失败证据。',
    place_verification_failed: '放置未通过相机与夹爪反馈验证。展开运行事实查看目标分离与可见放置区域的观测证据。',
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

function showError(error) {
  const message = errorMessage(error);
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
}

function runFailure() {
  const errors = [];
  const conclusion = runConclusion(state.run, state.events, errorMessage);
  if (conclusion) errors.push(conclusion);
  else if (state.run.error) errors.push(errorMessage(state.run.error));
  const failure = state.run.failure ?? state.run.task_progress?.failure;
  if (state.run.status === 'running' && failure?.code) {
    errors.push(`${errorMessage(failure.code)} 等待 JEV 根据失败证据决定下一步。`);
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

function markDirty() { state.dirty = true; view?.setDraft(state.draft); renderControls(); }

function inputField(label, value, onChange, options = {}) {
  const wrapper = element('label', 'input-field');
  wrapper.append(element('span', '', label));
  let input;
  if (options.choices) {
    input = element('select');
    const choices = { ...options.choices };
    if (!(value in choices)) choices[value] = value;
    for (const [key, name] of Object.entries(choices)) {
      const option = element('option', '', name);
      option.value = key;
      input.append(option);
    }
  } else {
    input = element('input');
    input.type = options.text ? 'text' : 'number';
    if (!options.text) { input.step = options.step ?? '0.01'; input.inputMode = 'decimal'; }
  }
  const scale = options.scale ?? 1;
  input.value = typeof value === 'number' ? (Number.isFinite(value) ? Number((value * scale).toFixed(4)) : '') : value;
  input.addEventListener('input', () => {
    let changedValue = input.value;
    if (!options.text && !options.choices) {
      const number = input.value.trim() === '' ? NaN : Number(input.value);
      changedValue = Number.isFinite(number) ? number / scale : NaN;
    }
    onChange(changedValue);
    markDirty();
  });
  wrapper.append(input);
  return wrapper;
}

function vectorFields(parent, title, values, onChange, axes = ['X', 'Y', 'Z']) {
  parent.append(element('div', 'field-heading', title));
  const row = element('div', 'vector-fields');
  axes.forEach((axis, index) => row.append(inputField(axis, values[index], value => onChange(index, value), { scale: 1000, step: '1' })));
  parent.append(row);
  return row;
}

function renderEditor() {
  if (!state.draft) return;
  byId('object-count').textContent = state.draft.objects.length;
  if (!state.draft.objects.some(object => object.id === state.selectedId)) state.selectedId = state.draft.objects[0]?.id ?? null;
  const rows = state.draft.objects.map(object => {
    const row = element('button', `object-row ${object.id === state.selectedId ? 'selected' : ''}`);
    row.type = 'button';
    const swatch = element('span', 'object-swatch');
    swatch.style.background = colorValues[object.color] ?? '#8596aa';
    const description = element('span', 'object-description');
    description.append(element('strong', '', object.id), element('small', '', `${kindNames[object.kind] ?? object.kind} · ${object.color}`));
    row.append(swatch, description, element('span', 'row-arrow', '›'));
    row.addEventListener('click', () => { state.selectedId = object.id; renderEditor(); renderControls(); });
    return row;
  });
  byId('object-list').replaceChildren(...(rows.length ? rows : [element('div', 'empty-small', '场景中暂无物体，点击添加。')]));
  const editor = byId('object-editor');
  editor.replaceChildren();
  const selected = state.draft.objects.find(object => object.id === state.selectedId);
  if (selected) {
    const title = element('div', 'selected-heading');
    title.append(element('span', '', selected.id));
    const remove = element('button', 'text-button danger', '移除');
    remove.type = 'button';
    remove.addEventListener('click', () => { state.draft.objects = state.draft.objects.filter(object => object.id !== selected.id); markDirty(); renderEditor(); });
    title.append(remove);
    editor.append(title);
    const metadata = element('div', 'metadata-fields');
    metadata.append(inputField('类型', selected.kind, value => {
      selected.kind = value;
      if (value === 'bin') {
        selected.wall_thickness_m ??= .01;
        selected.mass_kg ??= .4;
      }
      if (value === 'marker') {
        const used = new Set(state.draft.objects.filter(object => object !== selected && object.kind === 'marker').map(object => object.color));
        selected.color = ['yellow', 'green', 'orange', 'red', 'blue'].find(color => !used.has(color)) ?? 'yellow';
        selected.size = [.16, .12, .0002];
        selected.position[2] = Number(state.draft.table.position[2]) + Number(state.draft.table.size[2]) / 2 + .0001;
      }
      renderEditor();
    }, { choices: kindNames }));
    const colors = { red: '红色', blue: '蓝色', green: '绿色', yellow: '黄色', orange: '橙色' };
    if (selected.kind !== 'marker') Object.assign(colors, { gray: '灰色', white: '白色', black: '黑色' });
    metadata.append(inputField('颜色', selected.color, value => { selected.color = value; }, { choices: colors }));
    editor.append(metadata);
    const positionFields = vectorFields(editor, '位置 / mm · 偏航角 / °', selected.position, (index, value) => { selected.position[index] = value; });
    if (selected.kind === 'marker') positionFields.children[2].children[1].disabled = true;
    positionFields.append(inputField('yaw / °', selected.yaw_deg, value => { selected.yaw_deg = value; }, { step: '5' }));
    const sizeFields = vectorFields(editor, selected.kind === 'bin' ? '外尺寸 / mm' : '尺寸 / mm', selected.size, (index, value) => {
      const axes = selected.kind === 'cube' || selected.kind === 'sphere' ? [0, 1, 2]
        : selected.kind === 'cylinder' && index < 2 ? [0, 1] : [index];
      for (const axis of axes) {
        selected.size[axis] = value;
        if (axis !== index) {
          // Update linked dimensions in place so holding a spin button keeps focus.
          sizeFields.children[axis].children[1].value = Number.isFinite(value) ? Number((value * 1000).toFixed(4)) : '';
        }
      }
    }, selected.kind === 'bin' ? ['外长 X', '外宽 Y', '外高 Z'] : selected.kind === 'marker' ? ['X', 'Y'] : ['X', 'Y', 'Z']);
    if (selected.kind === 'marker') {
      editor.append(element('p', 'field-note', '固定在桌面的彩色放置区域，可调整 XY、长宽和朝向。每个标记需使用不同颜色；相机观测后才能作为放置目的地。'));
    }
    if (selected.kind === 'bin') {
      editor.append(inputField('壁厚 / mm', selected.wall_thickness_m, value => { selected.wall_thickness_m = value; }, { scale: 1000, step: '1' }));
      editor.append(inputField('质量 / kg', selected.mass_kg, value => { selected.mass_kg = value; }, { step: '.01' }));
      editor.append(element('p', 'field-note', '顶部开放，底板与四壁组成一个刚体；物料盒受重力。外长、外宽必须大于两倍壁厚，外高必须大于壁厚。'));
    }
  }
  const tableFields = byId('table-fields');
  tableFields.replaceChildren();
  vectorFields(tableFields, '桌面中心 / mm', state.draft.table.position, (index, value) => { state.draft.table.position[index] = value; });
  vectorFields(tableFields, '桌面尺寸 / mm', state.draft.table.size, (index, value) => { state.draft.table.size[index] = value; });
}

function replaceDraft(snapshot) {
  if (!snapshot?.objects || !snapshot?.table) return;
  state.draft = sceneFromSnapshot(snapshot);
  state.dirty = false;
  view?.setDraft(null);
  renderEditor();
}

function renderControls() {
  const inputs = { connected: state.connected, busy: state.busy, dirty: state.dirty, confirmed: state.snapshot?.confirmed, status: state.run.status, instruction: byId('instruction').value };
  const controls = controlsForState({ ...inputs, busy: state.busy || state.dragging });
  // Lock the form during a drag without cancelling the drag itself.
  view?.setInteraction({ enabled: !controlsForState(inputs).editDisabled, selectedId: state.selectedId });
  byId('scene-fields').disabled = controls.editDisabled;
  byId('apply-scene').disabled = controls.editDisabled || !state.dirty;
  byId('cancel-scene').disabled = controls.editDisabled || !state.dirty;
  byId('settle').disabled = controls.editDisabled || state.dirty;
  byId('confirm').disabled = controls.confirmDisabled;
  byId('start').disabled = controls.startDisabled || Boolean(state.run.algorithm_pending || state.run.algorithm_cleanup_error) || state.planningLoading || !planningEditor.ready || planningEditor.dirty;
  byId('reset').disabled = controls.resetDisabled;
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
  const confirmed = state.snapshot?.confirmed && !state.dirty;
  byId('scene-badge').textContent = !state.connected ? '待连接' : state.dirty ? '有未应用修改' : confirmed ? '已确认' : '待确认';
  byId('scene-badge').classList.toggle('confirmed', Boolean(confirmed));
  byId('scene-note').textContent = state.dirty ? '位置、yaw、尺寸与已应用物料盒的壁厚即时预览。类型、颜色和新增物体在点击「应用场景修改」后显示；完成后等待重力稳定并确认。' : confirmed ? '初始场景已确认。Reset 将恢复此物理基线。' : '拖动和数值修改都需先应用，再等待重力稳定并确认场景。类型、颜色和新增物体在应用后显示。';
  byId('start-hint').textContent = state.dragging ? '松手后请应用场景修改。' : state.busy ? '请求处理中…' : state.run.status === 'running' ? '任务运行中；Reset 可取消。' : state.run.algorithm_pending ? '等待后台算法返回，仍可 Reset。' : state.run.algorithm_cleanup_error ? '算法清理失败，请 Reset 重试。' : !planningEditor.ready || state.planningLoading ? '等待算法参数加载。' : planningEditor.dirty ? '参数草稿未应用，请打开参数设置。' : state.dirty ? '请应用修改并重新确认场景。' : !state.snapshot?.confirmed ? '先确认场景，再开始任务。' : '场景就绪 · 放置需已验证持物。';
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
  byId('experiment-replay').href = experiment.id ? `/replay.html?run=${encodeURIComponent(experiment.id)}` : '/replay.html';
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
  const decisionNames = { bind_task: 'JEV 选择任务', bind_attributes: 'JEV 选择物体属性', choose_skill: 'JEV 选择下一 skill', select_arguments: 'JEV 确定 skill 参数' };
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
    ? `${ikStep ? 'Batched IK 批次' : '规划批次'} ${step.batch_index} / ${step.batch_count ?? '—'} · ${candidates} · Arm ${step.arm ?? '—'}${Number.isFinite(step.success_count) ? ` · 成功 ${step.success_count} 个` : ''}` : '';
  byId('run-last-error').hidden = !running || !step?.last_error;
  byId('run-last-error').textContent = step?.last_error ? `上一批次结果：${step.last_error}` : '';
  byId('run-failure').hidden = !failure;
  byId('run-failure').textContent = failure;
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
  const previousGeneration = state.snapshot?.generation;
  state.snapshot = snapshot;
  cameraPanel.setGeneration(snapshot.generation);
  pointcloudPanel.setGeneration(snapshot.generation);
  state.run = payload.run ?? {};
  renderExperiment();
  if (!state.draft || (!state.dirty && snapshot.generation !== previousGeneration)) replaceDraft(snapshot);
  const names = { idle: snapshot.confirmed ? '场景就绪' : '等待确认', running: '任务运行中', succeeded: '任务成功', success: '任务成功', completed: '任务完成', partial_success: '部分完成', needs_attention: '结果待确认', failed: '任务失败', error: '任务出错', cancelled: '任务已取消', unsupported: '暂不支持此任务' };
  byId('run-status').textContent = state.run.algorithm_cleanup_error ? '算法清理失败' : names[state.run.status] ?? state.run.status ?? '等待场景';
  byId('run-dot').className = `run-dot ${state.run.status === 'running' ? 'active' : ''} ${state.run.algorithm_cleanup_error || ['failed', 'error'].includes(state.run.status) ? 'failed' : ''}`;
  byId('task-value').textContent = ({ grasp: '抓取', place: '放置', grasp_place: '抓取并放置', unsupported: '不支持' })[state.run.task] ?? state.run.task ?? '—';
  byId('target-value').textContent = state.run.target_id ?? '—';
  byId('destination-value').textContent = state.run.destination_id ?? '—';
  byId('arm-value').textContent = state.run.arm ? `Arm ${state.run.arm}` : '—';
  const failure = runFailure();
  if (failure && failure !== lastRunError) showError(failure);
  lastRunError = failure;
  if (snapshot.physics_error) showError(`物理仿真出错：${snapshot.physics_error}。请检查后端日志并重启仿真。`);
  const previousEvents = state.events;
  state.events = mergeEvents(state.events, payload.events, payload.events_mode);
  planningEditor.sync(payload.planning_settings);
  taskClock.sync(state.run, state.events, payload.server_time);
  renderTaskClock();
  if (state.events !== previousEvents) renderTrace(state.events);
  const facts = state.run.facts;
  objectList.render(state.run.task_progress);
  const metrics = state.run.metrics;
  byId('performance-summary').hidden = !metrics;
  if (metrics) {
    const summary = performanceSummary(metrics);
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
  view?.setTable(snapshot.table);
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
    // Scene endpoints return the accepted snapshot. Use it even if the
    // subsequent state refresh fails, so Reset cannot leave a stale preview.
    if (options.refreshDraft && result.objects && result.table) {
      applyState({ simulation: result, run: { status: 'idle', algorithm_pending: state.run.algorithm_pending }, events: [] });
      replaceDraft(state.snapshot);
    }
    const payload = await refreshState(operation);
    if (operation !== state.operation) return null;
    if (options.refreshDraft) replaceDraft(state.snapshot);
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
      summary.append(element('span', '', name), badge);
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
    try { applyState(JSON.parse(event.data)); } catch (error) { showError(`实时数据处理失败：${error.message}`); }
  });
  socket.addEventListener('close', () => { setConnection(false); setTimeout(connectSocket, 2000); });
  socket.addEventListener('error', () => socket.close());
}

byId('dismiss-error').addEventListener('click', () => { byId('error-banner').hidden = true; });
byId('toggle-scene-editor').addEventListener('click', () => {
  sceneEditorOpen = !sceneEditorOpen;
  byId('editor-panel').hidden = !sceneEditorOpen;
  byId('workspace').classList.toggle('scene-collapsed', !sceneEditorOpen);
  byId('toggle-scene-editor').ariaExpanded = String(sceneEditorOpen);
  byId('toggle-scene-editor').textContent = sceneEditorOpen ? '收起场景' : '展开场景';
});
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
for (const category of ['common', 'experimental', 'system']) {
  byId(`planning-tab-${category}`).addEventListener('click', () => {
    planningEditor.setCategory(category);
    for (const name of ['common', 'experimental', 'system']) byId(`planning-tab-${name}`).ariaPressed = String(name === category);
  });
}
selectView('scene');
byId('instruction').addEventListener('input', renderControls);
byId('experiment-save-mode').addEventListener('change', renderControls);
byId('home-view').addEventListener('click', () => view?.home());
byId('toggle-tcp-frames').addEventListener('click', event => {
  const button = event.currentTarget;
  const visible = button.getAttribute('aria-pressed') !== 'true';
  button.setAttribute('aria-pressed', String(visible));
  button.textContent = visible ? '隐藏 TCP 坐标系' : '显示 TCP 坐标系';
  view?.setTcpFramesVisible(visible);
});
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
byId('add-object').addEventListener('click', () => {
  if (!state.draft) return;
  let number = 1;
  while (state.draft.objects.some(object => object.id === `object_${number}`)) number += 1;
  const table = state.draft.table;
  const object = { id: `object_${number}`, kind: 'box', color: 'blue', position: [Number(table.position[0]), Number(table.position[1]), Number(table.position[2]) + Number(table.size[2]) / 2 + 0.15], size: [0.06, 0.06, 0.08], yaw_deg: 0 };
  state.draft.objects.push(object);
  state.selectedId = object.id;
  markDirty();
  renderEditor();
});
byId('apply-scene').addEventListener('click', () => {
  try { void perform('/api/scene', buildScene(state.draft), { refreshDraft: true, manifest: true }); } catch (error) { showError(error); }
});
byId('cancel-scene').addEventListener('click', () => {
  if (byId('cancel-scene').disabled) return;
  view?.cancelInteraction();
  replaceDraft(state.snapshot);
  renderControls();
});
byId('settle').addEventListener('click', () => void perform('/api/settle', {}, { refreshDraft: true }));
byId('confirm').addEventListener('click', () => void perform('/api/confirm', {}, { refreshDraft: true }));
byId('reset').addEventListener('click', () => {
  view?.cancelInteraction();
  void perform('/api/reset', {}, { refreshDraft: true, manifest: true, afterSuccess: () => { taskClock.reset(); renderTaskClock(); } });
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

function finishDrag() {
  state.dragging = false;
  renderControls();
}

try { view = createSceneView(byId('viewport'), {
  onSelect(id) { state.selectedId = id; renderEditor(); },
  onDragStart() { state.dragging = true; renderControls(); },
  onDragEnd(id, position) {
    const object = state.draft.objects.find(item => item.id === id);
    if (object) {
      object.position = [...position];
      if (object.kind === 'marker') object.position[2] = Number(state.draft.table.position[2]) + Number(state.draft.table.size[2]) / 2 + .0001;
      markDirty();
      renderEditor();
    }
    finishDrag();
  },
  onDragCancel: finishDrag,
}); } catch (error) {
  byId('viewport-loading').replaceChildren(element('strong', '', '无法创建 3D 视图'), element('span', '', '请启用浏览器硬件加速，并使用支持 WebGL 2 的浏览器。'));
  showError(`WebGL 初始化失败：${error.message}`);
}
void refreshState().catch(error => { setConnection(false); showError(`无法连接后端：${error.message}。请检查服务是否启动。`); });
void refreshManifest();
void refreshHealth();
void refreshPlanningSettings();
connectSocket();
setInterval(() => { if (!state.connected) void refreshState().catch(() => {}); }, 5000);
setInterval(() => void refreshHealth(), 15000);
setInterval(renderTaskClock, 100);

const STATUS_NAMES = { active: '活动实例', observed: '已观测', selected: '已选择', consumed: '已消耗', removed: '已移除', completed: '已完成' };
const PLACEMENT_NAMES = { execution_assumed: '按执行记录确认', direct_success: '直接确认成功', inferred_success: '推测成功', failure: '失败', unknown: '证据不足' };

export function isTaskLedger(progress) {
  return Boolean(progress && Array.isArray(progress.selected_input_ids) && progress.transformations
    && typeof progress.transformations === 'object');
}

export function memoryTransformations(progress) {
  const records = progress?.transformations ?? {};
  return Array.isArray(records) ? records : Object.values(records);
}

export function memoryEntities(progress) {
  const catalog = progress?.entities ?? progress?.all_entities ?? progress?.objects ?? [];
  return (Array.isArray(catalog) ? catalog : Object.values(catalog))
    .filter(entity => entity.source !== 'configured_demo_region');
}

export function renderTaskObjects(container, progress, bindings = null) {
  const element = (tag, text = '') => {
    const node = container.ownerDocument.createElement(tag);
    node.textContent = text;
    return node;
  };
  const section = (title, rows, folded = false) => {
    const node = element(folded ? 'details' : 'section');
    node.className = 'process-objects-section';
    node.append(element(folded ? 'summary' : 'h3', `${title}（${rows.length}）`));
    const list = element('ul');
    for (const text of rows) list.append(element('li', text));
    node.append(list);
    container.append(node);
  };
  const retired = new Set(progress.retired_entity_ids ?? []);
  const objects = new Map((progress.objects ?? []).map(item => [item.id, item]));
  const entities = memoryEntities(progress).map(entity => ({ ...entity, ...objects.get(entity.id) }));
  section('登记实体', entities.map(item => {
    let text = `${item.id ? `ID ${item.id} ` : ''}${item.description ?? item.name ?? item.kind ?? '实体'}`
      + ` · ${STATUS_NAMES[item.status] ?? item.status ?? '状态未记录'} · ID ${item.id}`;
    if (item.current_region) text += ` · 当前区域 ${item.current_region}`;
    const placement = item.container_entry?.placement_outcome ?? item.visual_release?.assessment?.placement_outcome
      ?? item.destination_visual_evidence?.placement_outcome;
    if (placement) text += ` · 放置${PLACEMENT_NAMES[placement] ?? placement}`;
    if (retired.has(item.id)) text += ' · 已退役';
    if (item.processing_history?.length) text += ` · 处理记录 ${item.processing_history.join(' → ')}`;
    for (const transfer of item.transfers ?? []) {
      text += ` · 转移至 ${transfer.destination_id ?? transfer.region_id ?? '区域未记录'}`;
      if (transfer.arm) text += ` / ${transfer.arm}臂`;
    }
    return text;
  }));
  section('输入与产物关联', memoryTransformations(progress).map(item =>
    `${item.input_entity_id} → ${item.output_entity_id} · 技能 ${item.skill} · 记录 ${item.id}`));
  const tasks = { ...(bindings?.object_tasks ?? {}), ...(progress.object_tasks ?? {}) };
  if (Object.keys(tasks).length) section('每件物体的任务要求', Object.entries(tasks).map(([id, task]) =>
    `输入 ID ${id} · 来源 ${task.source_ref ?? '未指定'} · 最终目的地 ${task.destination_ref ?? '未指定'}`
    + ` · 处理 ${task.processing?.join(' → ') || '直接搬运'}`
    + ` · 完成条件 ${task.completion_conditions?.join('；') || '未记录'}`));
  if (progress.initial_entities) {
    section('初始实体档案', memoryEntities({ entities: progress.initial_entities }).map(item =>
      `${item.description ?? item.name ?? item.kind ?? '实体'} · ID ${item.id}`));
  }
  if (bindings) renderTaskBindings(container, bindings);
}

export function renderTaskBindings(container, bindings) {
  const document = container.ownerDocument;
  const details = document.createElement('details');
  const title = document.createElement('summary');
  title.textContent = '任务引用与本次已选输入';
  const content = document.createElement('pre');
  content.className = 'candidate-detail';
  content.textContent = JSON.stringify(bindings, null, 2);
  details.append(title, content);
  container.append(details);
}

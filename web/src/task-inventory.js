const STATUS_NAMES = { observed: '已观测', selected: '已选择', consumed: '已消耗', completed: '已完成' };

export function isTaskInventory(progress) {
  return Boolean(progress && Array.isArray(progress.selected_input_ids) && progress.transformations
    && typeof progress.transformations === 'object');
}

export function inventoryTransformations(progress) {
  const records = progress?.transformations ?? {};
  return Array.isArray(records) ? records : Object.values(records);
}

export function inventoryEntities(progress) {
  const catalog = progress?.entities ?? progress?.all_entities ?? progress?.objects ?? [];
  return Array.isArray(catalog) ? catalog : Object.values(catalog);
}

export function renderTaskInventory(container, progress, bindings = null) {
  const element = (tag, text = '') => {
    const node = container.ownerDocument.createElement(tag);
    node.textContent = text;
    return node;
  };
  const section = (title, rows, folded = false) => {
    const node = element(folded ? 'details' : 'section');
    node.className = 'process-inventory-section';
    node.append(element(folded ? 'summary' : 'h3', `${title}（${rows.length}）`));
    const list = element('ul');
    for (const text of rows) list.append(element('li', text));
    node.append(list);
    container.append(node);
  };
  const retired = new Set(progress.retired_entity_ids ?? []);
  const objects = new Map((progress.objects ?? []).map(item => [item.id, item]));
  const entities = inventoryEntities(progress).map(entity => ({ ...entity, ...objects.get(entity.id) }));
  section('登记实体', entities.map(item => {
    let text = `${item.display_id ? `#${item.display_id} ` : ''}${item.description ?? item.name ?? item.kind ?? '实体'}`
      + ` · ${STATUS_NAMES[item.status] ?? item.status ?? '状态未记录'} · ID ${item.id}`;
    if (item.current_region) text += ` · 当前区域 ${item.current_region}`;
    if (retired.has(item.id)) text += ' · 已退役';
    if (item.processing_history?.length) text += ` · 处理记录 ${item.processing_history.join(' → ')}`;
    for (const transfer of item.transfers ?? []) {
      text += ` · 转移至 ${transfer.destination_id ?? transfer.region_id ?? '区域未记录'}`;
      if (transfer.arm) text += ` / ${transfer.arm}臂`;
    }
    return text;
  }));
  section('输入与产物关联', inventoryTransformations(progress).map(item =>
    `${item.input_entity_id} → ${item.output_entity_id} · 技能 ${item.skill} · 记录 ${item.id}`));
  const tasks = { ...(bindings?.object_tasks ?? {}), ...(progress.object_tasks ?? {}) };
  if (Object.keys(tasks).length) section('每件物体的任务要求', Object.entries(tasks).map(([id, task]) =>
    `输入 ID ${id} · 来源 ${task.source_ref ?? '未指定'} · 最终目的地 ${task.destination_ref ?? '未指定'}`
    + ` · 处理 ${task.processing?.join(' → ') || '直接搬运'}`
    + ` · 完成条件 ${task.completion_conditions?.join('；') || '未记录'}`));
  section('初始实体档案', inventoryEntities({ entities: progress.initial_entities ?? {} }).map(item =>
    `${item.description ?? item.name ?? item.kind ?? '实体'} · ID ${item.id}`), true);
  if (bindings) renderTaskBindings(container, bindings);
}

export function renderTaskBindings(container, bindings) {
  const document = container.ownerDocument;
  const details = document.createElement('details');
  const title = document.createElement('summary');
  title.textContent = '任务清单绑定';
  const content = document.createElement('pre');
  content.className = 'candidate-detail';
  content.textContent = JSON.stringify(bindings, null, 2);
  details.append(title, content);
  container.append(details);
}

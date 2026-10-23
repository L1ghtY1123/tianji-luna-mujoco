const statuses = {observed: '待处理', selected: '当前目标', completed: '已完成', deferred: '未完成 · 暂缓', superseded: '身份未知 · 已重编号'};
const colors = {red: '红色', pink: '粉色', green: '绿色', blue: '蓝色', black: '黑色', white: '白色', yellow: '黄色', gray: '灰色', orange: '橙色'};
const kinds = {cube: '方块', box: '长方体', sphere: '球体', cylinder: '圆柱', object: '物体'};
const processPhases = { wrapped_source: '等待拆包', table_released: '已放到桌面', unpack_pending: '拆包待核验',
  unpacked_confirmed: '黑件待抓取', black_held: '黑件已抓取', right_bin_released: '已投放 · 待核验', completed: '右盒投放已确认' };

function renderProcessInventory(container, document, progress) {
  const records = progress.process_records;
  const ordered = (progress.process_order ?? Object.keys(records)).map(id => records[id]).filter(Boolean);
  const pending = new Set(progress.pending_unpack_ids ?? []);
  const processing = new Set(progress.processing_ids ?? []);
  const completed = new Set(progress.completed_process_ids ?? []);
  const entityRows = catalog => Object.entries(catalog ?? {}).map(([id, item]) =>
    `${item.display_label ?? (item.display_id ? `#${item.display_id}` : '')} ${item.description
      ?? `${colors[item.color] ?? item.color ?? ''} ${kinds[item.kind] ?? item.kind ?? '物体'}`} · ID ${item.id ?? id}`
    + (item.kind === 'collection' ? ` · 已交付记录 ${item.members?.length ?? 0} 件（不逐件视觉匹配）` : ''));
  const pendingProducts = new Set(progress.pending_delivery_product_ids ?? []);
  const sections = [
    ['初始物体档案', entityRows(progress.initial_entities), true],
    ['当前场景实体', entityRows(progress.current_entities ?? progress.all_entities), true],
    ['待拆包', ordered.filter(item => pending.has(item.input_entity_id)).map(item =>
      `#${item.input_display_id ?? item.display_id} 粉色包装件 · ID ${item.input_entity_id} · ${item.to_table_arm ?? '—'}臂 · ${processPhases[item.phase] ?? item.phase}`)],
    ['已拆包 · 待投放', ordered.filter(item => pendingProducts.has(item.output_entity_id)).map(item =>
      `产物 ${item.output_entity_id} · ${item.to_right_bin_arm ?? '—'}臂 · ${processPhases[item.phase] ?? item.phase}`)],
    ['处理中', ordered.filter(item => processing.has(item.process_id ?? item.id)).map(item =>
      `工序 #${item.display_id} · ${processPhases[item.phase] ?? item.phase} · 输入 ${item.input_entity_id}`
      + (item.output_entity_id ? ` → 产物 ${item.output_entity_id} · ${item.to_right_bin_arm ?? '—'}臂` : ''))],
    ['已完成', ordered.filter(item => completed.has(item.process_id ?? item.id)).map(item =>
      `工序 #${item.display_id} · 输入 ${item.input_entity_id} → 产物 ${item.output_entity_id} · 右盒投放已确认`)],
  ];
  container.replaceChildren();
  container.hidden = false;
  for (const [title, rows, folded] of sections) {
    const section = document.createElement(folded ? 'details' : 'section');
    section.className = 'process-inventory-section';
    const heading = document.createElement(folded ? 'summary' : 'h3');
    heading.textContent = `${title}（${rows.length}）`;
    section.append(heading);
    if (rows.length) {
      const list = document.createElement('ul');
      for (const text of rows) {
        const item = document.createElement('li');
        item.textContent = text;
        list.append(item);
      }
      section.append(list);
    }
    container.append(section);
  }
}

export function objectRows(progress) {
  const objects = progress?.objects ?? [...(progress?.completed ?? []), ...(progress?.deferred ?? []), ...(progress?.unseen_objects ?? [])];
  if (!Array.isArray(objects)) return [];
  return [...new Map(objects.map(item => [item.id, item])).values()]
    .filter(item => !progress?.membership || progress.membership[item.id] === 'required')
    .sort((a, b) => (a.display_id ?? Infinity) - (b.display_id ?? Infinity))
    .map(item => ({id: item.id, number: item.display_id ?? '—',
      stage: (item.stage_index ?? 0) + 1,
      arm: progress?.stage_arms?.[String(item.stage_index ?? 0)] ?? progress?.requested_arm,
      regionId: (progress?.placement_regions ?? []).find(region => region.destination_id === item.destination_id)?.region_id,
      label: `${colors[item.color] ?? item.color ?? ''} ${kinds[item.kind] ?? item.kind ?? '物体'}`.trim(),
      status: ['released', 'awaiting_visual_confirmation'].includes(item.transport_status) ? '已释放 · 待视觉确认'
        : item.status === 'pending' && item.stage_index === 1 ? '桌面中转完成 · 待放入目标盒'
        : item.status === 'selected' && item.stage_index === 1 ? '当前目标 · 第二段'
        : statuses[item.status] ?? item.status ?? '未知',
      visibility: item.visibility === 'visible' ? '本帧可见' : '本帧未见',
      association: ['ambiguous', 'unsupported', 'unknown'].includes(item.association_status) ? '身份待确认' : '已关联',
      ...(item.destination_visual_evidence ? {destinationVisual:
        ['confirmed', 'at_destination'].includes(item.destination_visual_evidence.status) ? '去向已由视觉核对'
          : item.destination_visual_evidence.status === 'elsewhere' ? '视觉显示尚未到达目标区域' : '去向视觉待确认'} : {})}));
}

export function labelsForFrame(labels, frameId, generation, cameraId = '3') {
  return labels?.observation_id === frameId && labels.generation === generation && Array.isArray(labels.items)
    ? labels.items.filter(item => (item.camera_id ?? '3') === cameraId && typeof item.id === 'string'
      && Number.isFinite(item.x) && Number.isFinite(item.y)) : [];
}

export function createObjectList(container, {document: ownerDocument = globalThis.document} = {}) {
  return {render(progress) {
    if (progress?.process_records) {
      renderProcessInventory(container, ownerDocument, progress);
      return;
    }
    const rows = objectRows(progress);
    container.replaceChildren();
    container.hidden = !rows.length;
    if (!rows.length) return;
    const title = ownerDocument.createElement('h3');
    title.textContent = '目标清单';
    const list = ownerDocument.createElement('ol');
    for (const row of rows) {
      const item = ownerDocument.createElement('li');
      item.textContent = `#${row.number} ${row.label} · ${row.status} · ${row.visibility} · ${row.association}`;
      item.textContent += ` · ID ${row.id} · 第${row.stage}段${row.arm ? ` / ${row.arm}臂` : ''}`;
      if (row.regionId) item.textContent += ` · ${row.regionId}`;
      if (row.destinationVisual) item.textContent += ` · ${row.destinationVisual}`;
      item.title = row.id;
      list.append(item);
    }
    container.append(title, list);
  }};
}

const statuses = {observed: '待处理', selected: '当前目标', completed: '已完成', deferred: '未完成 · 暂缓', superseded: '身份未知 · 已重编号'};
const colors = {red: '红色', green: '绿色', blue: '蓝色', black: '黑色', white: '白色', yellow: '黄色', gray: '灰色', orange: '橙色'};
const kinds = {cube: '方块', box: '长方体', sphere: '球体', cylinder: '圆柱', object: '物体'};

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
      if (row.regionId) item.textContent += ` · ${row.regionId} / 区${row.number}`;
      if (row.destinationVisual) item.textContent += ` · ${row.destinationVisual}`;
      item.title = row.id;
      list.append(item);
    }
    container.append(title, list);
  }};
}

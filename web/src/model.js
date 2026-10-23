const ACTIVE_STATUSES = new Set(['running', 'starting', 'planning', 'executing', 'stopping']);

function finiteVector(values, label, positive = false) {
  if (!Array.isArray(values) || values.length !== 3) throw new Error(`${label}需要三个数值。`);
  const numbers = values.map(Number);
  if (numbers.some(value => !Number.isFinite(value) || (positive && value <= 0))) {
    throw new Error(`${label}必须为${positive ? '大于零的' : '有限'}数值。`);
  }
  return numbers;
}

export function buildScene(draft) {
  const ids = new Set();
  const markerColors = new Set();
  const table = {
    position: finiteVector(draft.table.position, '桌面位置'),
    size: finiteVector(draft.table.size, '桌面尺寸', true),
    color: String(draft.table.color ?? 'white').trim(),
  };
  const objects = draft.objects.map(object => {
    const id = String(object.id).trim();
    if (!id) throw new Error('物体 ID 不能为空。');
    if (ids.has(id)) throw new Error(`物体 ID 重复：${id}`);
    ids.add(id);
    const yaw = Number(object.yaw_deg);
    if (!Number.isFinite(yaw)) throw new Error(`${id} 的偏航角必须为有限数值。`);
    const normalized = {
      id,
      kind: String(object.kind).trim() || 'box',
      color: String(object.color).trim() || 'blue',
      position: finiteVector(object.position, `${id} 的位置`),
      size: finiteVector(object.size, `${id} 的尺寸`, true),
      yaw_deg: yaw,
    };
    if (object.fixed !== undefined) {
      if (typeof object.fixed !== 'boolean') throw new Error(`${id} 的固定属性必须为布尔值。`);
      normalized.fixed = object.fixed;
    }
    if (object.kind === 'cad') {
      if (typeof object.asset_id !== 'string' || !object.asset_id) throw new Error(`${id} 缺少 CAD 资产。`);
      normalized.asset_id = object.asset_id;
      if (object.pitch_deg !== undefined) {
        const pitch = Number(object.pitch_deg);
        if (!Number.isFinite(pitch)) throw new Error(`${id} 的俯仰角必须为有限数值。`);
        normalized.pitch_deg = pitch;
      }
      if (object.mass_kg !== undefined) {
        const mass = Number(object.mass_kg);
        if (!Number.isFinite(mass) || mass <= 0) throw new Error(`${id} 的 CAD 质量必须大于零。`);
        normalized.mass_kg = mass;
      }
    }
    if (object.visual_style === 'cup') normalized.visual_style = 'cup';
    if (object.kind === 'cylinder' && object.cross_section === 'ellipse') normalized.cross_section = 'ellipse';
    if (normalized.kind === 'marker') {
      if (!['red', 'orange', 'yellow', 'green', 'blue'].includes(normalized.color) || markerColors.has(normalized.color)) {
        throw new Error('每个桌面标记需使用不同的红、橙、黄、绿或蓝颜色。');
      }
      markerColors.add(normalized.color);
      const angle = yaw * Math.PI / 180;
      const c = Math.abs(Math.cos(angle)), s = Math.abs(Math.sin(angle));
      const [width, depth] = normalized.size;
      const extents = [(c * width + s * depth) / 2, (s * width + c * depth) / 2];
      if (extents.some((extent, axis) => Math.abs(normalized.position[axis] - table.position[axis]) + extent > table.size[axis] / 2)) {
        throw new Error(`${id} 的标记区域必须位于桌面内。`);
      }
      normalized.position[2] = table.position[2] + table.size[2] / 2 + .0001;
      normalized.size[2] = .0002;
    }
    if (normalized.kind === 'bin') {
      const thickness = Number(object.wall_thickness_m ?? .01);
      const mass = Number(object.mass_kg ?? .4);
      if (!Number.isFinite(thickness) || thickness <= 0 || normalized.size[0] <= 2 * thickness
          || normalized.size[1] <= 2 * thickness || normalized.size[2] <= thickness) {
        throw new Error(`${id} 的物料盒壁厚与外尺寸必须留下非零内腔。`);
      }
      if (!Number.isFinite(mass) || mass <= 0) throw new Error(`${id} 的物料盒质量必须为大于零的有限数值。`);
      normalized.wall_thickness_m = thickness;
      normalized.mass_kg = mass;
    }
    return normalized;
  });
  return {
    objects,
    table,
    ...(draft.support_tables ? { support_tables: draft.support_tables.map(item => ({
      id: item.id, position: finiteVector(item.position, '小桌位置'),
      size: finiteVector(item.size, '小桌尺寸', true),
    })) } : {}),
  };
}

export function sceneFromSnapshot(snapshot) {
  return {
    table: { position: [...snapshot.table.position], size: [...snapshot.table.size], color: snapshot.table.color ?? 'white' },
    ...(snapshot.support_tables ? { support_tables: snapshot.support_tables.map(item => ({
      id: item.id, position: [...item.position], size: [...item.size],
    })) } : {}),
    objects: snapshot.objects.map(object => {
      const [w, x, y, z] = object.quaternion ?? [1, 0, 0, 0];
      const item = {
        id: object.id,
        kind: object.kind,
        color: object.color,
        position: [...object.position],
        size: [...object.size],
        yaw_deg: Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) * 180 / Math.PI,
      };
      if (object.fixed !== undefined) item.fixed = object.fixed;
      if (object.kind === 'cad') {
        item.asset_id = object.asset_id;
        if (object.pitch_deg !== undefined) {
          item.pitch_deg = object.pitch_deg;
          // The local Y axis retains heading even at the flat pitch=90 pose.
          item.yaw_deg = Math.atan2(2 * (w * z - x * y), 1 - 2 * (x * x + z * z)) * 180 / Math.PI;
        }
        if (object.mass_kg !== undefined) item.mass_kg = object.mass_kg;
      }
      if (object.kind === 'bin') {
        item.wall_thickness_m = object.wall_thickness_m ?? .01;
        item.mass_kg = object.mass_kg ?? .4;
      }
      if (object.visual_style === 'cup') item.visual_style = 'cup';
      if (object.kind === 'cylinder' && object.cross_section === 'ellipse') item.cross_section = 'ellipse';
      return item;
    }),
  };
}

export function createCadObject(draft, asset) {
  if (!asset?.id) throw new Error('请选择 CAD 资产。');
  const size = finiteVector(asset.size_m, 'CAD 尺寸', true);
  let number = 1;
  while (draft.objects.some(object => object.id === `${asset.id}_${number}`)) number += 1;
  const table = draft.table;
  return { id: `${asset.id}_${number}`, kind: 'cad', asset_id: asset.id,
    color: asset.default_color, size, yaw_deg: 0,
    position: [Number(table.position[0]), Number(table.position[1]),
      Number(table.position[2]) + Number(table.size[2]) / 2 + size[2] / 2 + .02] };
}

export function controlsForState({ connected, busy, dirty, confirmed, status, instruction }) {
  const editDisabled = !connected || busy || ACTIVE_STATUSES.has(status);
  return {
    editDisabled,
    startDisabled: editDisabled || dirty || !confirmed || !instruction.trim(),
    resetDisabled: !connected,
    confirmDisabled: editDisabled || dirty,
  };
}

export function traceDetails(data = {}) {
  const confidence = data.confidence ?? data.output?.confidence ?? data.provider_details?.confidence;
  return {
    state: data.state ?? null,
    questions: data.questions ?? null,
    output: data.output ?? null,
    provider_details: data.provider_details ?? null,
    confidence: typeof confidence === 'number' || (confidence && typeof confidence === 'object') ? confidence : null,
    duration_s: typeof data.duration_s === 'number' ? data.duration_s : null,
  };
}

export function formatJson(value) {
  return value === null || value === undefined ? '未提供' : JSON.stringify(value, null, 2);
}

export function candidateCount(indices) {
  if (Array.isArray(indices)) return indices.length;
  return Number.isInteger(indices?.count) && indices.count >= 0 ? indices.count : null;
}

export function createEventTitleResolver(events = []) {
  const skills = new Map();
  const parents = new Map();
  const stages = { plan_place: 'place', plan_grasp: 'grasp', plan_grasp_and_place: 'grasp' };
  for (const event of events) {
    if (event.invocation_id && event.parent_invocation_id) parents.set(event.invocation_id, event.parent_invocation_id);
    if (event.invocation_id && ['skill_start', 'skill_complete', 'skill_error'].includes(event.kind)) {
      const stage = stages[event.data?.skill ?? event.title];
      if (stage) skills.set(event.invocation_id, stage);
    }
  }
  return function titleForEvent(event, title = event.title ?? event.kind ?? '事件') {
    const batch = ['planning_batch_start', 'planning_batch_complete'].includes(event.kind);
    const algorithm = ['algorithm_start', 'algorithm_complete', 'algorithm_error'].includes(event.kind);
    if (!batch && !algorithm) return title;
    let stage = ['grasp', 'place'].includes(event.data?.planning_stage) ? event.data.planning_stage
      : stages[event.data?.skill] ?? stages[event.data?.function];
    let id = event.invocation_id ?? event.parent_invocation_id;
    const visited = new Set();
    while (!stage && id && !visited.has(id)) {
      stage = skills.get(id);
      visited.add(id);
      id = parents.get(id);
    }
    // Older recordings identify placement through measured release metadata or their original title.
    if (!stage && batch && (event.data?.batch_scope === 'generated_positions' || event.data?.release_reference
      || event.title?.includes('放置') || event.title?.includes('plan_place_batch'))) stage = 'place';
    if (batch && stage) return `plan_${stage}_batch ${event.kind === 'planning_batch_start' ? '开始' : '完成'}`;
    if (algorithm && stage === 'place') return title.replace('plan_grasp_batch', 'plan_place_batch');
    return title;
  };
}

export function mergeEvents(current, incoming = [], mode = 'replace') {
  if (mode !== 'append') return incoming;
  if (!incoming.length) return current;
  const events = new Map(current.map(event => [event.id, event]));
  for (const event of incoming) events.set(event.id, event);
  return [...events.values()];
}

export function confidenceLabel(confidence) {
  if (typeof confidence === 'number') return `${(confidence * 100).toFixed(1)}%`;
  if (confidence && typeof confidence === 'object') {
    return Object.entries(confidence).map(([field, value]) => `${field} ${typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : JSON.stringify(value)}`).join(' · ');
  }
  return '未提供';
}

export function performanceSummary(metrics, events = []) {
  const calls = events.filter(event => event.kind === 'visual_decisions_call');
  const durations = calls.map(event => event.data?.elapsed_s).filter(Number.isFinite);
  const total = durations.reduce((sum, seconds) => sum + seconds, 0);
  return {
    count: String(calls.length),
    mean: durations.length ? `${(total / durations.length * 1000).toFixed(0)} ms` : '—',
    total: durations.length ? `${total.toFixed(2)} s` : '—',
    run: Number.isFinite(metrics.run_duration_s) ? `${metrics.run_duration_s.toFixed(2)} s` : '—',
  };
}

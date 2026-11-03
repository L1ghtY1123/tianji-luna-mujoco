import { isTaskInventory, inventoryEntities, inventoryTransformations } from './task-inventory.js';

// Display only summary fields; measured point arrays remain in their dedicated preview.
function pick(value, keys) {
  return Object.fromEntries(keys.filter(key => value?.[key] !== undefined).map(key => [key, value[key]]));
}

function scalars(value) {
  return Object.fromEntries(Object.entries(value ?? {}).filter(([, item]) =>
    item === null || ['boolean', 'number', 'string'].includes(typeof item)));
}

function verification(value) {
  if (!value) return null;
  return { ...pick(value, ['status', 'success', 'reason', 'source', 'verification_scope', 'open_confirmed', 'conflicting_evidence']),
    position: scalars(value.position), visual: scalars(value.visual),
    checks: scalars(value.checks), measurements: scalars(value.measurements) };
}

function progressCount(value) {
  return Array.isArray(value) ? value.length : Number.isInteger(value?.count) ? value.count : 0;
}

export function taskProgressSummary(progress) {
  if (!progress) return '';
  if (isTaskInventory(progress)) return `已完成 ${progressCount(progress.completed_input_ids)} / ${progressCount(progress.selected_input_ids)} 个目标 · 已登记产物 ${progressCount(progress.product_ids)} 个`;
  return '历史清单：请查看归档原始记录';
}

export function runEvidence(run) {
  const holding = run.holding;
  const prepared = holding?.attachment_evidence;
  const geometry = prepared?.attachment_geometry ?? prepared?.collision_evidence?.attachment_geometry;
  return {
    facts: run.facts ?? {},
    observation_budget: run.observation_budget ?? null,
    decision_budget: run.decision_budget ?? null,
    scene: run.scene_evidence ?? null,
    coordinate_reference: '放置固定两指前端TCP的世界XY；只搜索世界Z轴yaw。球底规划间隙=碰撞阈值+2 mm，实际不得低于碰撞阈值。',
    holding: holding ? {
      ...pick(holding, ['state', 'arm', 'verified', 'attached', 'reason', 'observation_id', 'attachment_detach_pending']),
      verification: verification(holding.verification),
      last_position_evidence: scalars(holding.last_position_evidence),
      ...pick(prepared, ['prepare_wall_ms']),
      collision_evidence: pick(prepared?.collision_evidence, ['geometry_coverage', 'full_object_coverage',
        'gripper_configuration_source', 'optimizer_attachment']),
      attachment_geometry: pick(geometry, ['sphere_count', 'sphere_budget', 'voxel_size_m',
        'effective_occupied_voxel_size_m', 'margin_m', 'measurement_uncertainty_m']),
    } : null,
    collision_model: run.collision_model ? pick(run.collision_model, ['mode', 'gripper', 'attached',
      'lift_carried_collision_checked', 'prepare_wall_ms']) : null,
    verification: verification(run.verification),
    metrics: run.metrics ?? null,
    task_inventory: run.task_inventory ?? null,
    task_progress: isTaskInventory(run.task_progress) ? {
      ...pick(run.task_progress, ['selected_input_ids', 'completed_input_ids', 'product_ids', 'retired_entity_ids']),
      completed_count: progressCount(run.task_progress.completed_input_ids),
      entities: inventoryEntities(run.task_progress).map(item => pick(item, ['id', 'display_id', 'kind', 'description', 'status', 'current_region', 'processing_history'])),
      objects: (run.task_progress.objects ?? []).map(item => ({
        ...pick(item, ['id', 'status', 'current_region', 'processing_history']),
        transfers: (item.transfers ?? []).map(transfer => ({ ...scalars(transfer), evidence: scalars(transfer.evidence) })),
      })),
      transformations: inventoryTransformations(run.task_progress).map(item => ({
        ...pick(item, ['id', 'skill', 'input_entity_id', 'output_entity_id']), evidence: scalars(item.evidence),
      })),
    } : null,
  };
}

export function runStatusSummary(run) {
  const evidence = runEvidence(run);
  const { holding, collision_model: model } = evidence;
  const parts = [];
  if (run.task_progress) parts.push(taskProgressSummary(run.task_progress));
  const table = run.scene_evidence?.table_cache;
  if (table?.table_revision) parts.push(`桌面：初始头部测量${table.cache_hit ? ' · 正在复用' : ''} · ${table.table_revision}`);
  if (run.scene_evidence?.excluded_instance_id) parts.push(`当前地图排除 ID ${run.scene_evidence.excluded_instance_id}`);
  if (holding) {
    const names = { held: '持物', release_pending: '释放待确认', provisional: '暂定持物', released: '已释放' };
    parts.push(`${holding.arm ?? '—'} 臂 ${names[holding.state] ?? holding.state ?? '持物状态未知'} · ${holding.verified ? '已验证' : '未确认'}`);
  }
  if (model) {
    const gripper = { fixed_open: '固定开爪', actual_feedback: '实测夹爪' }[model.gripper] ?? model.mode ?? '未提供';
    parts.push(`碰撞模型：${gripper} · ${model.attached ? '含持物球' : '无持物球'}`);
  }
  const result = evidence.verification ?? holding?.verification;
  if (result) parts.push(`执行验证：${result.status ?? (result.success ? '通过' : '未通过')}`);
  const preparation = model?.prepare_wall_ms ?? holding?.prepare_wall_ms;
  if (Number.isFinite(preparation)) parts.push(`几何准备 ${preparation.toFixed(1)} ms`);
  return parts.join('；');
}

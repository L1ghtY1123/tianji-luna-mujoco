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

function progressItems(value) {
  return Array.isArray(value) ? value : value?.first ?? [];
}

function progressCount(value) {
  return Array.isArray(value) ? value.length : Number.isInteger(value?.count) ? value.count : 0;
}

export function taskProgressSummary(progress) {
  if (!progress) return '';
  if (progress.process_records) return `已完成 ${progressCount(progress.completed_process_ids)} 道工序 · 待拆包 ${progressCount(progress.pending_unpack_ids)} 件 · 处理中 ${progressCount(progress.processing_ids)} 件`;
  const reasons = {
    grasp_verification_failed: '抓取未通过验证', grasp_verification_unknown: '抓取结果待确认',
    place_verification_failed: '放置未通过验证', place_verification_unknown: '放置结果待确认',
    no_reachable_arm: '当前机械臂不可达', plan_grasp_failed: '暂无可行抓取路径',
    plan_place_failed: '暂无可行放置路径', plan_grasp_timeout: '抓取规划超时',
    plan_place_timeout: '放置规划超时', task_progress_unknown: '剩余目标待确认',
  };
  const deferred = progressItems(progress.deferred);
  const labels = [...new Set(deferred.map(item => reasons[String(item.reason ?? '').split(':')[0]] ?? '需要重新评估'))];
  const reason = labels.length ? `（${labels.slice(0, 3).join('、')}${labels.length > 3 ? '等' : ''}）` : '';
  return `已完成 ${progressCount(progress.completed)} 个 · 暂缓 ${progressCount(progress.deferred)} 个${reason}`;
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
    task_progress: run.task_progress ? {
      ...pick(run.task_progress, ['scope', 'requested_arm', 'current_id', 'observation_id', 'stage_index',
        'stage_arms', 'inventory_revision', 'inventory_finalized', 'scene_entities']),
      placement_regions: (run.task_progress.placement_regions ?? []).map(region => ({
        ...pick(region, ['region_id', 'destination_id', 'layout_revision', 'stage_index', 'target_count']),
        ...pick(region.layout, ['source', 'observation_id', 'floor_z_m', 'section_count', 'measurement_time'])})),
      completed_count: progressCount(run.task_progress.process_records ? run.task_progress.completed_process_ids : run.task_progress.completed),
      ...(run.task_progress.process_records ? {
        process_records: Object.fromEntries(Object.entries(run.task_progress.process_records).map(([id, value]) => [id,
          pick(value, ['process_id', 'display_id', 'input_entity_id', 'output_entity_id', 'status', 'phase', 'to_table_arm', 'to_right_bin_arm'])])),
        ...pick(run.task_progress, ['pending_unpack_ids', 'processing_ids', 'completed_process_ids', 'product_ids']),
      } : {}),
      deferred_count: progressCount(run.task_progress.deferred),
      completed: progressItems(run.task_progress.completed).map(item => pick(item, ['id', 'kind', 'color', 'destination_id'])),
      deferred: progressItems(run.task_progress.deferred).map(item => pick(item, ['id', 'kind', 'color', 'reason'])),
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

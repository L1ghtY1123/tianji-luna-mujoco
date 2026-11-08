import {archivedEntityLabels, displayEntityName} from './replay-labels.js';
import {createCollisionColors} from './collision-colors.js';
import { createReplayCloud } from './replay-cloud.js';

export function collisionStages(events) {
  const starts = new Map();
  let node = null;
  const stages = [];
  for (const event of events) {
    if (event.kind === 'policy_node_started') node = event.data;
    if (event.kind === 'skill_start' && event.title === 'build_scene' && event.invocation_id) {
      starts.set(event.invocation_id, {...event.data.arguments, policy_node: node});
    }
    if (event.kind !== 'collision_scene_archived' || !event.data?.snapshot_artifact) continue;
    // Old recordings already contain exact build arguments; do not guess from colors or timing.
    const args = starts.get(event.invocation_id);
    const context = event.data.stage_context ?? (args ? {...args,
      excluded_instance_id: args.exclusion_state === 'selected' ? event.data.target_id : null} : {});
    stages.push({...event, data: {...event.data, stage_context: context}});
  }
  return stages;
}

export async function loadCollisionStage(event, load) {
  const snapshot = await load(event.data.snapshot_artifact);
  if (snapshot.status !== 'available') throw new Error(`该版本空间记忆不可用，${snapshot.reason ?? '未记录原因'}`);
  for (const key of ['scene_token', 'observation_id']) {
    if (snapshot[key] !== event.data[key]) throw new Error('空间记忆与事件版本不一致');
  }
  const assets = new Map();
  const layers = await Promise.all(Object.entries(snapshot.mesh_layers ?? {}).map(async ([name, layer]) => {
    if (!/^collision\/assets\/v1-[a-f0-9]{64}\.json$/.test(layer.mesh_artifact)) throw new Error('无效 CAD 归档路径');
    if (!assets.has(layer.mesh_artifact)) assets.set(layer.mesh_artifact, load(layer.mesh_artifact));
    const mesh = await assets.get(layer.mesh_artifact);
    if (!Array.isArray(mesh.vertices_local_m) || !Array.isArray(mesh.faces)) throw new Error('CAD 归档网格缺失');
    return [name, {...layer, vertices_local_m: mesh.vertices_local_m, faces: mesh.faces}];
  }));
  return {...snapshot, mesh_layers: Object.fromEntries(layers)};
}

export function createCollisionReplay(document, {load}) {
  const select = document.getElementById('collision-stage');
  const status = document.getElementById('collision-status');
  const container = document.getElementById('collision-preview');
  const entities = document.getElementById('collision-entities');
  let stages = [], viewer = null, version = 0, colorFor = createCollisionColors(), labels = new Map();
  const details = document.querySelector('#collision-details pre');
  async function show() {
    const current = ++version;
    viewer?.dispose(); viewer = null;
    entities.replaceChildren();
    if (details) details.textContent = '';
    const event = stages[Number(select.value)];
    if (!event) { status.textContent = '该实验缺少已归档的空间记忆。'; return; }
    status.textContent = '正在读取空间记忆…';
    try {
      const snapshot = await loadCollisionStage(event, load);
      if (current !== version) return;
      viewer = createReplayCloud(container, {collisionScene: snapshot, collisionColor: colorFor});
      const context = snapshot.stage_context ?? event.data.stage_context;
      const stageLabels = new Map(labels);
      for (const [id, label] of Object.entries(context.object_labels ?? {})) stageLabels.set(id, label);
      const name = (id, fallback) => displayEntityName(id, stageLabels, fallback);
      const scope = {global: '全局更新', local: '局部更新'}[context.scope] ?? '更新范围未归档';
      status.textContent = [
        `${event.t_s.toFixed(1)} 秒 · ${scope}`,
        context.excluded_instance_id ? `本阶段排除：${name(context.excluded_instance_id)}（抓取目标）。`
          : '本阶段没有指定排除目标。',
        '半透明面参与碰撞，虚线框表示已排除；颜色对应下方物体。',
      ].join('\n');
      if (details) details.textContent = JSON.stringify({
        scene_token: snapshot.scene_token, observation_id: snapshot.observation_id,
        generation: snapshot.generation, exclusion_state: snapshot.exclusion_state,
        objects: [...Object.values(snapshot.mesh_layers ?? {}), ...(snapshot.cuboid_layers ?? [])]
          .map(layer => ({name: name(layer.instance_id), id: layer.instance_id ?? layer.name,
            geometry: layer.geometry_source, enabled: layer.enabled})),
      }, null, 2);
      for (const layer of [...Object.values(snapshot.mesh_layers ?? {}), ...(snapshot.cuboid_layers ?? [])]) {
        const row = document.createElement('li');
        const type = layer.geometry_source === 'observed_obb' ? 'OBB' : layer.asset_id ? 'CAD' : 'CUBOID';
        const label = name(layer.instance_id, layer.role === 'measured_table' ? '桌面' : layer.asset_id ?? '未知物体');
        const exclusion = layer.instance_id === context.excluded_instance_id ? '抓取目标从环境排除' : '已排除（未归档具体原因）';
        row.textContent = `${label} · ${type} · ${layer.enabled ? '参与碰撞' : exclusion}`;
        const swatch = document.createElement('i'); swatch.className = 'collision-swatch'; swatch.style.color = colorFor(layer);
        if (layer.enabled) swatch.style.background = colorFor(layer); else swatch.style.borderStyle = 'dashed';
        row.prepend(swatch);
        entities.append(row);
      }
    } catch (error) { if (current === version) status.textContent = error.message; }
  }
  select.addEventListener('change', show);
  return {
    update(events) {
      stages = collisionStages(events);
      labels = archivedEntityLabels(events);
      colorFor = createCollisionColors();
      select.replaceChildren();
      for (const [index, event] of stages.entries()) {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `阶段 ${index + 1} · ${event.t_s.toFixed(2)} 秒 · ${({global: '全局更新', local: '局部更新'})[event.data.stage_context.scope] ?? '范围未归档'} · ${event.data.stage_context.excluded_instance_id ? displayEntityName(event.data.stage_context.excluded_instance_id, event.data.stage_context.object_labels, displayEntityName(event.data.stage_context.excluded_instance_id, labels)) : '空手观测／运动准备'}`;
        select.append(option);
      }
      select.disabled = !stages.length;
      select.value = '0';
      return show();
    },
    dispose() {
      version++;
      select.removeEventListener('change', show);
      viewer?.dispose(); viewer = null;
    },
  };
}

import { createReplayCloud } from './replay-cloud.js';

export function collisionStages(events) {
  return events.filter(event => event.kind === 'collision_scene_archived' && event.data?.snapshot_artifact);
}

export async function loadCollisionStage(event, load) {
  const snapshot = await load(event.data.snapshot_artifact);
  if (snapshot.status !== 'available') throw new Error(`该阶段碰撞场景不可用：${snapshot.reason ?? '未记录原因'}`);
  for (const key of ['scene_token', 'observation_id']) {
    if (snapshot[key] !== event.data[key]) throw new Error('阶段碰撞场景与事件版本不一致');
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
  let stages = [], viewer = null, version = 0;
  async function show() {
    const current = ++version;
    viewer?.dispose(); viewer = null;
    entities.replaceChildren();
    const event = stages[Number(select.value)];
    if (!event) { status.textContent = '该实验未归档阶段碰撞场景；无法用当前场景替代历史记录。'; return; }
    status.textContent = '正在读取该阶段的归档场景…';
    try {
      const snapshot = await loadCollisionStage(event, load);
      if (current !== version) return;
      viewer = createReplayCloud(container, {collisionScene: snapshot});
      status.textContent = `事件 #${event.id} · ${event.t_s.toFixed(2)} 秒 · ${snapshot.scene_token} · 观测 ${snapshot.observation_id}`;
      for (const layer of [...Object.values(snapshot.mesh_layers ?? {}), ...(snapshot.cuboid_layers ?? [])]) {
        const row = document.createElement('li');
        const type = layer.geometry_source === 'observed_obb' ? 'OBB' : layer.asset_id ? 'CAD' : 'CUBOID';
        row.textContent = `${type} · ${layer.instance_id ?? layer.name} · ${layer.enabled ? '参与碰撞' : '已排除'}${layer.asset_id ? ` · ${layer.asset_id}` : ''}`;
        entities.append(row);
      }
    } catch (error) { if (current === version) status.textContent = error.message; }
  }
  select.addEventListener('change', show);
  return {
    update(events) {
      stages = collisionStages(events);
      select.replaceChildren();
      for (const [index, event] of stages.entries()) {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `阶段 ${index + 1} · ${event.t_s.toFixed(2)} 秒 · ${event.data.exclusion_state} · ${event.data.target_id ?? '全体'}`;
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

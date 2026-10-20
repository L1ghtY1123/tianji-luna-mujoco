import './styles.css';
import './collision-spheres.css';
import './workspaces.css';
import { request } from './api.js';
import { createSceneView } from './scene-view.js';
import { createWorkspaceLayer } from './workspace-layer.js';

const element = id => document.getElementById(id);
const view = createSceneView(element('viewport'));
let layer = null;
let serverInstance = null;
let manifestRevision = null;
let following = true;
let loading = false;
let stopped = false;
let timer = null;
let snapshotTime = null;

function home() { view.home({ target: [0, 0, 1.1], offset: [2.45, -3.2, 1.6] }); }

function showStatus() {
  const time = snapshotTime === null ? '等待快照' : `机器人快照 ${snapshotTime.toFixed(2)} s`;
  element('workspace-status').textContent = `${following ? '实时跟随' : '已暂停'} · ${time}`;
}

function applyDisplay() {
  if (!layer) return;
  for (const arm of ['A', 'B']) layer.setVisible(arm, element(`show-${arm}`).checked);
  layer.setOpacity(Number(element('workspace-opacity').value) / 100);
  view.setOverlay(layer.object);
}

async function refresh() {
  if (loading || stopped) return;
  clearTimeout(timer);
  loading = true;
  element('refresh-workspaces').disabled = true;
  try {
    const state = await request('/api/workspace-scene');
    if (stopped) return;
    const snapshot = state.simulation;
    if (!layer || serverInstance !== state.server_instance_id) {
      const payload = await request('/api/workspaces');
      if (stopped) return;
      const nextLayer = createWorkspaceLayer(payload);
      layer?.dispose();
      layer = nextLayer;
      serverInstance = state.server_instance_id;
      manifestRevision = null;
      applyDisplay();
      element('workspace-count').textContent = payload.arms.map(arm =>
        `${arm.arm === 'A' ? '左臂' : '右臂'} ${arm.reachable_points.toLocaleString()} 个可达网格`).join(' · ');
    }
    if (manifestRevision !== snapshot.manifest_revision) {
      const manifest = await request('/api/manifest');
      if (manifest.revision !== snapshot.manifest_revision) throw new Error('场景刚更新，正在重新读取…');
      if (stopped) return;
      await view.loadManifest(manifest);
      manifestRevision = manifest.revision;
    }
    if (stopped) return;
    view.update(snapshot);
    view.setTable(snapshot.table);
    if (snapshotTime === null) home();
    snapshotTime = Number.isFinite(snapshot.time) ? snapshot.time : null;
    showStatus();
  } catch (error) {
    element('workspace-status').textContent = `读取失败：${error.message}`;
  } finally {
    element('workspace-loading').hidden = true;
    loading = false;
    element('refresh-workspaces').disabled = false;
    if (following && !stopped) timer = setTimeout(refresh, 500);
  }
}

element('home-view').addEventListener('click', home);
element('refresh-workspaces').addEventListener('click', refresh);
element('toggle-follow').addEventListener('click', () => {
  following = !following;
  element('toggle-follow').textContent = following ? '暂停跟随' : '继续跟随';
  element('toggle-follow').setAttribute('aria-pressed', String(!following));
  clearTimeout(timer);
  showStatus();
  if (following) refresh();
});
element('show-meshes').addEventListener('change', event => view.setGeometryVisible(event.target.checked));
for (const arm of ['A', 'B']) element(`show-${arm}`).addEventListener('change', applyDisplay);
element('workspace-opacity').addEventListener('input', event => {
  element('workspace-opacity').value = event.target.value;
  element('workspace-opacity-value').textContent = `${event.target.value}%`;
  applyDisplay();
});
window.addEventListener('pagehide', event => {
  stopped = true;
  clearTimeout(timer);
  if (!event.persisted) layer?.dispose();
});
window.addEventListener('pageshow', event => {
  if (!event.persisted) return;
  stopped = false;
  if (following) refresh();
});
refresh();

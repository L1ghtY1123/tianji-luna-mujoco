import './styles.css';
import './collision-spheres.css';
import { request } from './api.js';
import { createSceneView } from './scene-view.js';
import { createCollisionSphereLayer, validateCollisionSpheres } from './collision-sphere-layer.js';

const element = id => document.getElementById(id);
const view = createSceneView(element('viewport'));
const spheres = createCollisionSphereLayer();
let manifestRevision = null;
let following = true;
let loading = false;
let stopped = false;
let timer = null;
let snapshotTime = null;

function showStatus() {
  const time = snapshotTime === null ? '等待快照' : `仿真快照 ${snapshotTime.toFixed(2)} s`;
  element('sphere-status').textContent = `${following ? '实时跟随' : '已暂停'} · ${time}`;
}

async function refresh() {
  if (loading || stopped) return;
  clearTimeout(timer);
  loading = true;
  element('refresh-spheres').disabled = true;
  try {
    const payload = await request('/api/collision-spheres');
    validateCollisionSpheres(payload);
    const snapshot = payload.simulation;
    if (!snapshot || snapshot.manifest_revision == null) throw new Error('缺少开夹爪场景快照。');
    if (manifestRevision !== snapshot.manifest_revision) {
      const manifest = await request('/api/manifest');
      if (manifest.revision !== snapshot.manifest_revision) throw new Error('场景刚更新，正在重新读取…');
      if (stopped) return;
      await view.loadManifest(manifest);
      manifestRevision = manifest.revision;
    }
    if (stopped) return;
    spheres.update(payload);
    view.update(snapshot);
    view.setTable(snapshot.table);
    view.setOverlay(spheres.object);
    if (snapshotTime === null) view.home();
    snapshotTime = Number.isFinite(snapshot.time) ? snapshot.time : null;
    element('sphere-count').textContent = `${payload.link_count} 个 link · ${payload.sphere_count} 个碰撞球`;
    element('sphere-source').textContent = `来源：${payload.source}`;
    element('sphere-loading').hidden = true;
    showStatus();
  } catch (error) {
    element('sphere-status').textContent = `读取失败：${error.message}`;
    element('sphere-loading').hidden = true;
  } finally {
    loading = false;
    element('refresh-spheres').disabled = false;
    if (following && !stopped) timer = setTimeout(refresh, 500);
  }
}

element('home-view').addEventListener('click', () => view.home());
element('refresh-spheres').addEventListener('click', refresh);
element('toggle-follow').addEventListener('click', () => {
  following = !following;
  element('toggle-follow').textContent = following ? '暂停跟随' : '继续跟随';
  element('toggle-follow').setAttribute('aria-pressed', String(!following));
  clearTimeout(timer);
  showStatus();
  if (following) refresh();
});
element('show-meshes').addEventListener('change', event => view.setGeometryVisible(event.target.checked));
element('show-spheres').addEventListener('change', event => {
  spheres.object.visible = event.target.checked;
  view.setOverlay(spheres.object);
});
element('sphere-opacity').addEventListener('input', event => {
  const percent = Number(event.target.value);
  spheres.setOpacity(percent / 100);
  element('sphere-opacity-value').textContent = `${percent}%`;
  view.setOverlay(spheres.object);
});
window.addEventListener('pagehide', event => {
  stopped = true;
  clearTimeout(timer);
  if (!event.persisted) spheres.dispose();
});
window.addEventListener('pageshow', event => {
  if (!event.persisted) return;
  stopped = false;
  if (following) refresh();
});
refresh();

// Base owns the robot renderer; Luna supplies validated observation overlays only.
import { BufferGeometry } from 'three';

function serializeOverlay(overlay) {
  if (!overlay) return null;
  overlay.updateMatrixWorld(true);
  const result = overlay.toJSON();
  const geometries = new Map();
  overlay.traverse(object => {
    if (!object.geometry || geometries.has(object.geometry.uuid)) return;
    // EdgesGeometry and other derived geometries lack a portable fromJSON.
    // Send their actual buffers so the renderer does not reconstruct geometry.
    const geometry = new BufferGeometry().copy(object.geometry);
    geometry.uuid = object.geometry.uuid;
    geometries.set(geometry.uuid, geometry.toJSON());
    geometry.dispose();
  });
  result.geometries = [...geometries.values()];
  return result;
}

export function createBaseView(root, {externalScene = false} = {}) {
  const frame = document.createElement('iframe');
  frame.title = '实测机器人与 RGBD 感知场景';
  frame.className = 'base-twin-frame';
  root.replaceChildren(frame);
  let origin = null;
  let overlay = null;
  let generation = null;
  let scene = null;
  let active = true;
  let showRobot = true;
  const send = data => { if (origin) frame.contentWindow.postMessage(data, origin); };
  function sendOverlay() {
    const object = serializeOverlay(overlay);
    if (externalScene) {
      if (scene) send({type: 'robot-display-frame', scene, object, showRobot});
    } else send({type: 'measured-observation-overlay', object});
  }
  window.addEventListener('message', event => {
    if (event.source === frame.contentWindow && event.origin === origin && event.data?.type === 'robot-view-ready') {
      send({type: 'robot-view-active', active});
      sendOverlay();
    }
  });
  return {
    async loadManifest(manifest) {
      if (manifest.source !== 'tianji_base' || !manifest.viewer_url) throw new Error('Base 数字孪生接口不可用');
      const url = new URL(manifest.viewer_url);
      if (externalScene) url.searchParams.set('external_scene', '1');
      origin = url.origin;
      if (frame.src !== url.href) frame.src = url.href;
    },
    update(snapshot) {
      if (externalScene) throw new Error('use_display_frame_for_diagnostic_view');
      if (generation !== snapshot.generation) { overlay = null; sendOverlay(); }
      generation = snapshot.generation;
    },
    setOverlay(value) { overlay = value; sendOverlay(); },
    setDisplayFrame(snapshot, value) {
      if (!externalScene || !snapshot.base_scene?.ok) throw new Error('缺少 Base 机器人显示快照');
      scene = snapshot.base_scene;
      overlay = value;
      sendOverlay();
    },
    setGeometryVisible(value) { showRobot = Boolean(value); sendOverlay(); },
    setActive(value) { active = Boolean(value); send({type: 'robot-view-active', active}); },
    home() { send({type: 'robot-view-home'}); },
  };
}

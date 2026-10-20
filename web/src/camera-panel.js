// Preview RGB and cached task RGBD use separate endpoints and frame identities.
import { labelsForFrame } from './object-list.js';
import { regionsForFrame, appendPlacementOverlay } from './placement-overlay.js';
export function createCameraPanel(container, status, {
  document: ownerDocument = globalThis.document,
  request,
  initialMode = "preview",
  schedule = globalThis.setTimeout,
  cancel = globalThis.clearTimeout,
  isVisible = () => ownerDocument.visibilityState !== 'hidden',
} = {}) {
  let mode = initialMode;
  let running = false;
  let timer = null;
  let inFlight = null;
  let generation = 0;
  let observationId = null;
  let labelKey = '';
  let revision = 0;
  const cards = new Map();

  function element(tag, className, text) {
    const node = ownerDocument.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function next(delay = mode === "preview" ? 500 : 2000) {
    if (!running) return;
    if (timer !== null) cancel(timer);
    timer = schedule(() => { timer = null; void refresh(); }, delay);
  }

  function clear() {
    container.replaceChildren();
    cards.clear();
    observationId = null;
  }

  async function render(payload, requestedRevision) {
    if (!Number.isInteger(payload?.generation) || !Array.isArray(payload.cameras)) throw new Error('相机响应结构不完整。');
    if (payload.generation < generation) return;
    if (!payload.cameras.length) {
      clear();
      status.textContent = '等待三路 RGBD 相机采集。';
      return;
    }
    const identifier = mode === "preview" ? payload.preview_frame_id : payload.observation_id;
    const prefix = mode === "preview" ? "/api/camera-preview" : "/api/cameras";
    const channels = mode === "preview" ? [["rgb", "RGB"]] : [["rgb", "RGB"], ["depth", "深度伪彩"]];
    if (payload.cameras.length !== 3 || !identifier || !Number.isFinite(payload.time)) throw new Error('三路相机捕获信息不完整。');
    const ids = new Set();
    for (const camera of payload.cameras) {
      const id = String(camera.id);
      if (!['1', '2', '3'].includes(id) || ids.has(id) || typeof camera.frame_id !== 'string' || !camera.frame_id) throw new Error('相机帧标识无效。');
      ids.add(id);
      for (const [kind] of channels) {
        const path = `${prefix}/${id}/${kind}`;
        const bound = `${path}?frame_id=${encodeURIComponent(camera.frame_id)}`;
        if (camera[`${kind}_url`] !== path && camera[`${kind}_url`] !== bound) throw new Error('相机预览地址无效。');
      }
    }
    // The same captured image may receive labels after semantic processing finishes.
    const currentLabelKey = JSON.stringify(mode === 'task' ? payload.object_labels ?? null : null);
    if (identifier === observationId && payload.generation === generation && currentLabelKey === labelKey) return;
    const frames = ["2", "3", "1"].map(id => payload.cameras.find(camera => String(camera.id) === id)).map(camera => ({
      camera,
      images: channels.map(([kind, label]) => {
        const image = element('img');
        image.src = `${prefix}/${camera.id}/${kind}?frame_id=${encodeURIComponent(camera.frame_id)}`;
        image.alt = `${camera.label ?? `相机 ${camera.id}`} ${label}`;
        image.decoding = 'async';
        return image;
      }),
    }));
    // Decode off-DOM: retain the previous complete frame while any image is loading.
    await Promise.all(frames.flatMap(frame => frame.images.map(image => image.decode())));
    if (requestedRevision !== revision || payload.generation < generation) return;
    for (const { camera, images } of frames) {
      const id = String(camera.id);
      let card = cards.get(id);
      if (!card) {
        const node = element('article', 'camera-card');
        const heading = element('h3');
        const group = element('div', mode === 'preview' ? 'camera-images camera-rgb-only' : 'camera-images');
        const figures = channels.map(([, label]) => {
          const figure = element('figure');
          const caption = element('figcaption', '', label);
          group.append(figure);
          return { figure, caption };
        });
        const frameLabel = element('p', 'camera-frame');
        node.append(heading, group, frameLabel);
        container.append(node);
        card = { heading, figures, frameLabel };
        cards.set(id, card);
      }
      card.heading.textContent = camera.label ?? `相机 ${camera.id}`;
      images.forEach((image, index) => {
        const { figure, caption } = card.figures[index];
        const surface = element('div', 'camera-image-surface');
        surface.append(image);
        figure.replaceChildren(surface, caption);
        if (mode === 'task' && index === 0) {
          const overlay = element('div', 'object-camera-overlay');
          for (const item of labelsForFrame(payload.object_labels, identifier, payload.generation, id)) {
            const badge = element('span', `object-camera-label object-camera-${item.role ?? 'target'}`,
              item.label ?? `#${item.display_id} · ID ${item.id}`);
            badge.title = item.label ?? item.id;
            badge.style.cssText = `left:${Math.max(0, Math.min(.65, item.x)) * 100}%;top:${Math.max(0, Math.min(.85, item.y)) * 100}%;`;
            overlay.append(badge);
          }
          surface.append(overlay);
          const regions = regionsForFrame(payload.object_labels, identifier, payload.generation, id);
          appendPlacementOverlay(surface, regions, ownerDocument);
          caption.textContent = payload.object_labels?.analysis_status === 'analyzed'
            ? 'RGB · 本帧名称 / ID / 搬运状态' : 'RGB · 未分析：尚无本帧 ID 关联';
          if (regions.length) caption.textContent += ' · 固定分区 ' + [...new Set(regions.map(region =>
            `${region.region_id}（${region.source}，测量帧 ${region.source_observation_id}，t=${region.measurement_time ?? '—'}）`))].join('；');
        }
      });
      card.frameLabel.textContent = `帧 ${camera.frame_id}`;
    }
    generation = payload.generation;
    observationId = identifier;
    labelKey = currentLabelKey;
    status.textContent = `${mode === "preview" ? "实时 RGB 预览" : "最近任务观察"} ${observationId} · 场景 ${generation}`;
  }

  function refresh() {
    if (inFlight) return inFlight;
    if (timer !== null) { cancel(timer); timer = null; }
    if (!isVisible()) { next(); return Promise.resolve(); }
    inFlight = (async () => {
      const requestedRevision = revision;
      try {
        const payload = await request(mode === 'preview' ? '/api/camera-preview' : '/api/cameras');
        if (requestedRevision === revision) await render(payload, requestedRevision);
      }
      catch (error) {
        if (requestedRevision === revision) status.textContent = `相机预览读取失败：${error.message}`;
      }
      finally { inFlight = null; next(); }
    })();
    return inFlight;
  }

  return {
    refresh,
    setMode(value) {
      if (!['preview', 'task'].includes(value) || mode === value) return;
      mode = value;
      revision++;
      clear();
      status.textContent = value === 'preview' ? '等待实时 RGB 预览。' : '等待最近任务观察。';
      next(0);
    },
    start() { if (!running) { running = true; next(0); } },
    stop() { running = false; if (timer !== null) { cancel(timer); timer = null; } },
    setGeneration(value) {
      if (Number.isInteger(value) && value > generation) {
        generation = value;
        revision++;
        clear();
        status.textContent = '等待当前场景的三路 RGBD 相机采集。';
      }
    },
  };
}

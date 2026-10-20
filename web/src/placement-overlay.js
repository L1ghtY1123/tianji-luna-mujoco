// The server projects measured layouts into this capture; IDs stay frozen across frames.
export function regionsForFrame(labels, frameId, generation, cameraId) {
  if (labels?.observation_id !== frameId || labels.generation !== generation) return [];
  return (labels.regions ?? []).filter(item => item.camera_id === cameraId
    && Number.isInteger(item.slot_id) && Array.isArray(item.center) && item.center.length === 2
    && item.center.every(Number.isFinite) && Array.isArray(item.polygon) && item.polygon.length >= 3
    && item.polygon.every(point => point.length === 2 && point.every(Number.isFinite)));
}

export function appendPlacementOverlay(surface, regions, ownerDocument = globalThis.document) {
  if (!regions.length) return;
  const element = (tag, attributes = {}) => {
    const node = ownerDocument.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    return node;
  };
  const svg = element('svg', {viewBox: '0 0 100 100', preserveAspectRatio: 'none', class: 'placement-camera-overlay'});
  for (const region of regions) {
    const title = element('title');
    title.textContent = `${region.region_id} · 区${region.slot_id} · ${region.source} · 测量帧 ${region.source_observation_id}`;
    const polygon = element('polygon', {points: region.polygon.map(point => point.map(value => value * 100).join(',')).join(' '),
      fill: 'none', stroke: '#69e2ff', 'stroke-width': '1', 'vector-effect': 'non-scaling-stroke'});
    const text = element('text', {x: region.center[0] * 100, y: region.center[1] * 100,
      fill: '#fff', 'font-size': '4', 'text-anchor': 'middle', 'paint-order': 'stroke', stroke: '#123', 'stroke-width': '.5'});
    text.textContent = `区${region.slot_id}`;
    svg.append(title, polygon, text);
  }
  surface.append(svg);
}

import * as THREE from 'three';

export const obbStates = {
  current: {label: '本次观测', color: '#b7f56b'},
  retained: {label: '历史保留', color: '#b69cff'},
  stale: {label: '已失效，仅显示旧位置', color: '#ff7180'},
};

export function obbRows(payload, mode) {
  return payload.object_obbs?.[mode === 'collision' ? 'mapped' : 'observed'] ?? [];
}

export function validateObbPreview(payload) {
  if (payload.object_obbs === undefined) return;
  for (const key of ['observed', 'mapped']) {
    const rows = payload.object_obbs?.[key];
    if (!Array.isArray(rows)) throw new Error('OBB 列表无效。');
    const ids = new Set();
    for (const row of rows) {
      const obb = row?.obb;
      const matrix = obb?.T_frame_obb;
      const half = obb?.half_extents_m;
      if (!row || typeof row.instance_id !== 'string' || !row.instance_id || ids.has(row.instance_id) ||
          !Object.hasOwn(obbStates, row.status) || row.generation !== payload.generation ||
          typeof row.observation_id !== 'string' || !row.observation_id ||
          (row.status === 'current' && row.observation_id !== payload.observation_id) ||
          obb?.instance_id !== row.instance_id || obb.frame !== 'world' ||
          obb.source !== 'observed_points_obb_approximation' || obb.unobserved_surfaces_are_measured !== false ||
          obb.observation_provenance?.observation_id !== row.observation_id ||
          obb.observation_provenance?.generation !== row.generation ||
          !Array.isArray(half) || half.length !== 3 || half.some(value => !Number.isFinite(value) || value <= 0) ||
          !Array.isArray(matrix) || matrix.length !== 4 ||
          matrix.some(values => !Array.isArray(values) || values.length !== 4 || !values.every(Number.isFinite))) {
        throw new Error('OBB 的永久 ID、来源观测、尺寸或坐标无效。');
      }
      if (matrix[3].some((value, index) => Math.abs(value - (index === 3 ? 1 : 0)) > 1e-6)) {
        throw new Error('OBB 变换不是刚体位姿。');
      }
      for (let i = 0; i < 3; i += 1) {
        for (let j = 0; j < 3; j += 1) {
          const dot = matrix[i].slice(0, 3).reduce((sum, value, k) => sum + value * matrix[j][k], 0);
          if (Math.abs(dot - (i === j ? 1 : 0)) > 1e-5) throw new Error('OBB 旋转无效。');
        }
      }
      if (Math.abs(new THREE.Matrix4().set(...matrix.flat()).determinant() - 1) > 1e-5) {
        throw new Error('OBB 旋转无效。');
      }
      ids.add(row.instance_id);
    }
  }
}

function labelSprite(row, ownerDocument) {
  if (!ownerDocument) return null;
  const canvas = ownerDocument.createElement('canvas');
  // Headless DOM tests have no canvas; geometry remains independently testable.
  if (typeof canvas.getContext !== 'function') return null;
  canvas.width = Math.min(640, Math.max(160, row.instance_id.length * 24 + 24));
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (!context) return null;
  context.fillStyle = '#10202eee';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = obbStates[row.status].color;
  context.font = 'bold 40px sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(row.instance_id, canvas.width / 2, canvas.height / 2, canvas.width - 16);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(canvas), depthTest: false, depthWrite: false, sizeAttenuation: false}));
  sprite.name = `obb-label:${row.instance_id}`;
  sprite.scale.set(.045 * canvas.width / canvas.height, .045, 1);
  sprite.position.set(...row.obb.T_frame_obb.slice(0, 3).map(values => values[3]));
  sprite.position.z += row.obb.half_extents_m[2] + .018;
  sprite.renderOrder = 22;
  return sprite;
}

export function addObbLayers(object, rows, ownerDocument) {
  for (const row of rows) {
    const geometry = new THREE.BoxGeometry(...row.obb.half_extents_m.map(value => value * 2));
    const options = {color: obbStates[row.status].color, depthTest: false, depthWrite: false};
    const material = row.status === 'current' ? new THREE.LineBasicMaterial(options)
      : new THREE.LineDashedMaterial({...options, dashSize: .008, gapSize: .004});
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), material);
    geometry.dispose();
    box.name = `obb:${row.instance_id}`;
    box.userData = {instanceId: row.instance_id, status: row.status, observationId: row.observation_id};
    box.applyMatrix4(new THREE.Matrix4().set(...row.obb.T_frame_obb.flat()));
    box.computeLineDistances();
    box.renderOrder = 21;
    object.add(box);
    const label = labelSprite(row, ownerDocument);
    if (label) object.add(label);
  }
}

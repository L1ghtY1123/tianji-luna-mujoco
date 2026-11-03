import * as THREE from 'three';
import {addObbLayers, obbRows, obbStates, validateObbPreview} from './obb-layer.js';

const rawColor = new THREE.Color('#a7b3c4');
const modes = [['raw', '原始 RGBD 点云'], ['collision', '碰撞场景']];

function vector(value, count) {
  return Array.isArray(value) && value.length === count && value.every(Number.isFinite);
}

function rigidPose(value) {
  return vector(value, 7) && Math.abs(Math.hypot(...value.slice(3)) - 1) < 1e-5;
}

function nonempty(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function validCuboid(layer, payload) {
  if (!layer || !nonempty(layer.name) || typeof layer.enabled !== 'boolean' ||
      !rigidPose(layer.model?.pose) || !vector(layer.model?.dims, 3) ||
      layer.model.dims.some(value => value <= 0)) return false;
  if (layer.role === 'bin_shell_completion') {
    return layer.provenance === 'measured_planes_and_user_bin_dimensions' && nonempty(layer.instance_id) &&
      layer.observation_id === payload.observation_id && layer.generation === payload.generation;
  }
  if (layer.role === 'measured_table') {
    return (layer.name === 'semantic_table_forbidden' || /^semantic_table::.+$/.test(layer.name)) &&
      (layer.generation === undefined || layer.generation === payload.generation) &&
      (layer.observation_id === undefined || nonempty(layer.observation_id));
  }
  const observations = layer.geometry_observation_ids;
  return layer.role === 'semantic_instance' && layer.geometry_source === 'observed_obb' &&
    nonempty(layer.instance_id) && layer.name === `semantic_instance::${layer.instance_id}` &&
    layer.generation === payload.generation && nonempty(layer.observation_id) &&
    (observations === undefined || (Array.isArray(observations) && observations.every(nonempty) &&
      new Set(observations).size === observations.length && observations.includes(layer.observation_id)));
}

function validCadMesh(name, layer, payload) {
  return layer && name === layer.name && name === `semantic_instance::${layer.instance_id}` &&
    layer.role === 'semantic_instance' && layer.geometry_source === 'registered_cad' &&
    typeof layer.enabled === 'boolean' && nonempty(layer.instance_id) && nonempty(layer.asset_id) &&
    nonempty(layer.observation_id) && layer.generation === payload.generation &&
    [layer.source_sha256, layer.mesh_sha256].every(value => typeof value === 'string' && /^[a-f0-9]{64}$/i.test(value)) &&
    rigidPose(layer.pose_wxyz) && layer.sampling === 'none' &&
    Array.isArray(layer.vertices_local_m) && layer.vertices_local_m.length > 0 &&
    layer.vertices_local_m.every(point => vector(point, 3)) && layer.vertex_count === layer.vertices_local_m.length &&
    Array.isArray(layer.faces) && layer.faces.length > 0 && layer.face_count === layer.faces.length &&
    layer.faces.every(face => Array.isArray(face) && face.length === 3 &&
      face.every(index => Number.isSafeInteger(index) && index >= 0 && index < layer.vertex_count)) &&
    vector(layer.minimum_world_m, 3) && vector(layer.maximum_world_m, 3) &&
    layer.minimum_world_m.every((value, axis) => value <= layer.maximum_world_m[axis]);
}

function collisionDescription(scene) {
  const types = [];
  const colors = [];
  const evidence = [];
  if (Object.keys(scene?.mesh_layers ?? {}).length) {
    types.push('CAD');
    colors.push('青色：配准 CAD 网格');
    evidence.push('CAD 显示已发布的全部三角面，形状来自模型先验，位姿来自观测配准；隐藏表面不代表已测量。');
  }
  if (scene?.cuboid_layers?.some(layer => layer.geometry_source === 'observed_obb')) {
    types.push('OBB');
    colors.push('绿色：观测点 OBB 碰撞体');
    evidence.push('OBB 是观测点的实心包络近似，可能包含历史观测，原始来源见下方。');
  }
  if (scene?.voxel_layers?.length) {
    types.push('TSDF / ESDF');
    colors.push('青色：背景 TSDF 表面', '橙色：实例 TSDF 表面', '淡线框：ESDF 网格范围');
    evidence.push('当前显示 Mapper 的 TSDF 表面体素采样与推断桌体，未绘制完整 ESDF 距离场；原生 ESDF 距离样本可在下方展开查看。');
  }
  if (scene?.cuboid_layers?.some(layer => layer.role === 'bin_shell_completion')) colors.push('蓝色：盒壁补全');
  colors.push('灰色：已排除几何', '粉色：桌面碰撞体', '黄色：实测支撑范围');
  return {title: `${types.length ? types.join(' / ') : '桌面'} 碰撞场景`,
    legend: `${colors.join(' · ')}。`, diagnostic: evidence.join(' ') || '显示实测桌面对应的碰撞体；桌下延伸与边界扩展属于模型假设。'};
}

export function validatePointcloudPreview(payload) {
  validateObbPreview(payload);
  if (payload?.source !== 'registered_rgbd' || !Number.isInteger(payload.generation) || payload.generation < 0 ||
      typeof payload.scene_token !== 'string' || !payload.scene_token ||
      typeof payload.observation_id !== 'string' || !payload.observation_id) {
    throw new Error('点云缺少注册 RGBD 的观测与场景标识。');
  }
  const points = payload.points_world_m;
  const mask = payload.kept_mask;
  const labels = payload.cluster_labels;
  if (!Array.isArray(points) || points.length > 30000 ||
      points.some(point => !Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite)) ||
      !Array.isArray(mask) || mask.length !== points.length || mask.some(kept => typeof kept !== 'boolean')) {
    throw new Error('点云坐标、保留掩码或预览点数无效。');
  }
  const counts = payload.counts;
  if (!counts || !['input', 'kept', 'rejected'].every(name => Number.isSafeInteger(counts[name]) && counts[name] >= 0) ||
      counts.kept + counts.rejected !== counts.input || payload.total_point_count !== counts.input ||
      points.length > counts.input || !['uniform_original_indices', 'stratified_original_indices'].includes(payload.preview_sampling)) {
    throw new Error('点云总数或抽样来源不一致。');
  }
  const indices = payload.original_point_indices;
  if (indices !== undefined || payload.preview_sampling === 'stratified_original_indices') {
    if (!Array.isArray(indices) || indices.length !== points.length ||
        indices.some((index, row) => !Number.isSafeInteger(index) || index < 0 || index >= counts.input ||
          (row > 0 && index <= indices[row - 1]))) {
      throw new Error('点云原始索引必须逐行对应、递增且不重复。');
    }
  }
  const sampleKept = mask.filter(Boolean).length;
  if (sampleKept > counts.kept || points.length - sampleKept > counts.rejected) throw new Error('点云掩码与数量不一致。');
  if (labels !== null && (!Array.isArray(labels) || labels.length !== points.length ||
      labels.some(label => !Number.isSafeInteger(label) || label < -1))) {
    throw new Error('DBSCAN 点簇标签无效。');
  }
  if (payload.camera_ids !== undefined && (!Array.isArray(payload.camera_ids) || payload.camera_ids.length !== points.length ||
      payload.camera_ids.some(id => !['1', '2', '3'].includes(id)) || !Array.isArray(payload.pixels_rc) ||
      payload.pixels_rc.length !== points.length || payload.pixels_rc.some(pixel => !Array.isArray(pixel) || pixel.length !== 2 ||
        pixel.some(value => !Number.isSafeInteger(value) || value < 0)))) throw new Error('相机与像素来源未逐点对齐。');
  const scene = payload.collision_scene;
  if (scene?.status === 'available') {
    if (scene.frame !== 'world' || scene.units !== 'm' || ['generation', 'scene_token', 'observation_id'].some(key => scene[key] !== payload[key])) {
      throw new Error('碰撞场景与当前点云快照不一致。');
    }
    if (!Array.isArray(scene.voxel_layers) || scene.voxel_layers.some(layer =>
      !vector(layer.pose_wxyz, 7) || !vector(layer.dims_m, 3) || layer.dims_m.some(value => value <= 0) ||
      !Number.isFinite(layer.surface_voxel_size_m) || layer.surface_voxel_size_m <= 0 ||
      !Array.isArray(layer.surface_centers_world_m) || layer.surface_centers_world_m.length > 15000 ||
      layer.surface_centers_world_m.some(point => !vector(point, 3))) ||
      !vector(scene.table?.model?.pose, 7) || !vector(scene.table?.model?.dims, 3) ||
      scene.table.provenance !== 'inferred_ransac_support_completion') throw new Error('碰撞场景网格或推断桌体无效。');
    if (scene.cuboid_layers !== undefined && (!Array.isArray(scene.cuboid_layers) ||
        scene.cuboid_layers.some(layer => !validCuboid(layer, payload)))) {
      throw new Error('碰撞体或盒壁补全的几何、角色与观测来源无效。');
    }
    if (scene.mesh_layers !== undefined && (!scene.mesh_layers || typeof scene.mesh_layers !== 'object' ||
        Array.isArray(scene.mesh_layers) || Object.entries(scene.mesh_layers).some(([name, layer]) => !validCadMesh(name, layer, payload)))) {
      throw new Error('CAD 碰撞网格的身份、配准位姿、三角面或模型来源无效。');
    }
  }
}

function poseObject(object, pose) {
  object.position.set(...pose.slice(0, 3));
  object.quaternion.set(pose[4], pose[5], pose[6], pose[3]);
}

export function addCollisionLayers(object, scene) {
  for (const layer of scene.voxel_layers) {
    const color = layer.enabled ? (layer.role === 'background' ? '#4bdcca' : '#ffb05c') : '#8a92a3';
    const voxels = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: layer.enabled ? .72 : .22, depthWrite: false }),
      layer.surface_centers_world_m.length);
    voxels.name = layer.name;
    voxels.userData = { provenance: layer.provenance, enabled: layer.enabled, esdf: layer.esdf };
    const transform = new THREE.Matrix4();
    const scale = new THREE.Vector3().setScalar(layer.surface_voxel_size_m);
    for (const [row, point] of layer.surface_centers_world_m.entries()) {
      transform.compose(new THREE.Vector3(...point), new THREE.Quaternion(), scale);
      voxels.setMatrixAt(row, transform);
    }
    voxels.instanceMatrix.needsUpdate = true;
    object.add(voxels);
    const box = new THREE.BoxGeometry(...layer.dims_m);
    const bounds = new THREE.LineSegments(new THREE.EdgesGeometry(box),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: .2 }));
    box.dispose();
    bounds.name = `${layer.name}:esdf-bounds`;
    poseObject(bounds, layer.pose_wxyz);
    object.add(bounds);
  }
  for (const layer of Object.values(scene.mesh_layers ?? {})) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(layer.vertices_local_m.flat(), 3));
    geometry.setIndex(layer.faces.flat());
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: layer.enabled ? '#4bdcca' : '#8a92a3', transparent: true, opacity: layer.enabled ? .6 : .15,
      side: THREE.DoubleSide, depthWrite: false, toneMapped: false }));
    mesh.name = layer.name;
    mesh.userData = {geometrySource: layer.geometry_source, instanceId: layer.instance_id,
      assetId: layer.asset_id, observationId: layer.observation_id, generation: layer.generation,
      sourceSha256: layer.source_sha256, meshSha256: layer.mesh_sha256, enabled: layer.enabled};
    poseObject(mesh, layer.pose_wxyz);
    object.add(mesh);
  }
  for (const layer of scene.cuboid_layers ?? []) {
    const color = layer.enabled ? (layer.role === 'measured_table' ? '#f597ce' :
      layer.geometry_source === 'observed_obb' ? '#70de9a' : '#78b6ff') : '#8a92a3';
    const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({color, transparent: true,
        opacity: layer.enabled ? .35 : .1, depthWrite: false}));
    box.name = layer.name;
    box.userData = {provenance: layer.provenance, instanceId: layer.instance_id, enabled: layer.enabled,
      geometrySource: layer.geometry_source ?? layer.role, observationId: layer.observation_id,
      geometryObservationIds: layer.geometry_observation_ids, generation: layer.generation};
    poseObject(box, layer.model.pose);
    box.scale.set(...layer.model.dims);
    object.add(box);
  }
  const table = scene.table;
  const tableAlreadyRendered = scene.cuboid_layers?.some(layer => layer.role === 'measured_table' &&
    ['pose', 'dims'].every(key => layer.model[key].every((value, index) => value === table.model[key][index])));
  if (!tableAlreadyRendered) {
    const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ color: table.enabled ? '#f597ce' : '#8a92a3', transparent: true,
        opacity: table.enabled ? .25 : .1, depthWrite: false }));
    box.name = table.name;
    box.userData.provenance = table.provenance;
    poseObject(box, table.model.pose);
    box.scale.set(...table.model.dims);
    object.add(box);
  }
  if (Number.isFinite(table.height_m) && table.normal_xyz?.[2] > 0) {
    const [cx, cy] = table.model.pose;
    const [width, depth] = table.model.dims;
    const normal = table.normal_xyz;
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => {
      const x = cx + sx * width / 2;
      const y = cy + sy * depth / 2;
      return [x, y, table.height_m - (normal[0] * x + normal[1] * y) / normal[2]];
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(corners.flat(), 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    const fitted = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: '#ffd36b', transparent: true, opacity: .22, side: THREE.DoubleSide, depthWrite: false }));
    fitted.name = 'ransac-fitted-plane';
    fitted.userData.provenance = 'fitted_plane_displayed_within_completion_bounds';
    object.add(fitted);
  }
  const plane = table.support_plane;
  if (plane?.observed_polygon_xy_m?.length >= 3) {
    const normal = plane.normal_xyz;
    const points = plane.observed_polygon_xy_m.map(([x, y]) => new THREE.Vector3(x, y,
      -(normal[0] * x + normal[1] * y + plane.offset_m) / normal[2]));
    const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: '#ffe58c', depthTest: false }));
    outline.name = 'observed-support-polygon';
    object.add(outline);
  }
}

export function createPointcloudLayer(ownerDocument) {
  const object = new THREE.Group();
  object.name = 'registered-rgbd-preview';

  function clear() {
    for (const points of object.children) {
      points.geometry.dispose();
      points.material.map?.dispose();
      points.material.dispose();
    }
    object.clear();
  }

  return {
    object, clear, dispose: clear,
    update(payload, mode = 'raw', {showObbs = true} = {}) {
      validatePointcloudPreview(payload);
      if (!modes.some(([value]) => value === mode)) throw new Error('点云显示模式无效。');
      if (mode === 'collision') {
        if (payload.collision_scene?.status !== 'available') throw new Error('当前快照没有实际碰撞场景，请重新建图。');
        clear();
        addCollisionLayers(object, payload.collision_scene);
        if (showObbs) addObbLayers(object, obbRows(payload, mode), ownerDocument);
        return;
      }
      const positions = [];
      const pointColors = [];
      for (let index = 0; index < payload.points_world_m.length; index += 1) {
        positions.push(...payload.points_world_m[index]);
        pointColors.push(rawColor.r, rawColor.g, rawColor.b);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(pointColors, 3));
      geometry.computeBoundingSphere();
      const material = new THREE.PointsMaterial({ size: 3, sizeAttenuation: false, vertexColors: true,
        depthTest: false, depthWrite: false, transparent: true, opacity: .9, toneMapped: false });
      const points = new THREE.Points(geometry, material);
      points.renderOrder = 10;
      clear();
      object.add(points);
      if (showObbs) addObbLayers(object, obbRows(payload, mode), ownerDocument);
    },
  };
}

// This panel reads an already-built observation. Display controls never capture
// RGBD, run filtering, or write to the scene or planning settings.
export function createPointcloudPanel(container, status, {
  document: ownerDocument = globalThis.document, request, onOverlay, onEnvironmentVisible,
} = {}) {
  let open = false;
  let visible = false;
  let generation = null;
  let revision = 0;
  let inFlight = null;
  let payload = null;
  let environmentVisible = true;
  let showObbs = true;
  const layer = createPointcloudLayer(ownerDocument);

  function element(tag, className, text) {
    const node = ownerDocument.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const controls = element('div', 'pointcloud-controls');
  const modeLabel = element('label', 'pointcloud-mode');
  modeLabel.append(element('span', '', '叠层显示'));
  const mode = element('select');
  let collisionOption;
  for (const [value, title] of modes) {
    const option = element('option', '', title);
    option.value = value;
    if (value === 'collision') { collisionOption = option; option.disabled = true; }
    mode.append(option);
  }
  mode.value = 'raw';
  mode.disabled = true;
  modeLabel.append(mode);
  const refreshButton = element('button', 'button secondary', '读取最近建图快照');
  refreshButton.type = 'button';
  const visibilityButton = element('button', 'button secondary', '显示点云');
  visibilityButton.type = 'button';
  visibilityButton.disabled = true;
  visibilityButton.ariaPressed = 'false';
  const environmentButton = element('button', 'button secondary', '隐藏仿真物体');
  environmentButton.type = 'button';
  environmentButton.ariaPressed = 'false';
  const obbButton = element('button', 'button secondary', '隐藏 OBB 线框');
  obbButton.type = 'button';
  obbButton.ariaPressed = 'true';
  controls.append(modeLabel, refreshButton, visibilityButton, environmentButton, obbButton);
  const environmentNote = element('p', 'helper-text', '隐藏仿真物体后，仅保留机器人、相机与所选叠层；测量地图中的桌面碰撞体仍显示。');
  const counts = element('p', 'pointcloud-counts');
  const sample = element('p', 'helper-text');
  const legend = element('p', 'pointcloud-legend');
  const diagnostic = element('p', 'helper-text');
  const mapEvidence = element('div');
  const obbEvidence = element('div');
  const provenance = element('details');
  provenance.append(element('summary', '', '相机标定 / 像素来源 / 碰撞几何证据'));
  const provenanceText = element('pre');
  provenance.append(provenanceText);
  container.append(controls, environmentNote, counts, sample, legend, diagnostic, obbEvidence, mapEvidence, provenance);

  function clear() {
    payload = null;
    layer.clear();
    setVisible(false);
    counts.textContent = '';
    sample.textContent = '';
    legend.textContent = '';
    diagnostic.textContent = '';
    mapEvidence.replaceChildren();
    obbEvidence.replaceChildren();
    provenanceText.textContent = '';
    mode.disabled = true;
    visibilityButton.disabled = true;
    collisionOption.disabled = true;
    collisionOption.textContent = '碰撞场景';
  }

  function renderOverlay() {
    if (!payload) { onOverlay(null); return; }
    const description = collisionDescription(payload.collision_scene);
    legend.textContent = mode.value === 'collision' ? description.legend
      : '灰色：注册 RGBD 的过滤前可见点，包含保留与剔除样本；当前接口未提供逐点 RGB 颜色。';
    diagnostic.textContent = mode.value === 'collision'
      ? description.diagnostic
      : '原始点云是最近一次建图观测的实测采样，不代表最终碰撞障碍，也不随机器人当前姿态重新采集。';
    obbEvidence.replaceChildren();
    const boxes = obbRows(payload, mode.value);
    obbEvidence.append(element('p', 'helper-text', payload.object_obbs
      ? `${mode.value === 'collision' ? '地图关联' : '本次观测'} OBB 线框：${boxes.length} 个。绿实线：本次观测；紫虚线：历史保留；红虚线：已失效。线框用于查看观测来源，具体碰撞几何见碰撞场景；包络不证明隐藏表面已测量。`
      : '当前快照未导出独立 OBB 线框；碰撞场景显示已发布的几何。'));
    for (const row of boxes) {
      const dimensions = row.obb.half_extents_m.map(value => (value * 2000).toFixed(1)).join(' × ');
      obbEvidence.append(element('p', 'helper-text', `${row.instance_id} · ${obbStates[row.status].label} · ${dimensions} mm · 来源 ${row.observation_id}`));
    }
    if (!open || !visible) { onOverlay(null); return; }
    layer.update(payload, mode.value, {showObbs});
    onOverlay(layer.object);
  }

  function setVisible(value) {
    visible = Boolean(value) && open && Boolean(payload);
    visibilityButton.textContent = visible ? '隐藏点云' : '显示点云';
    visibilityButton.ariaPressed = String(visible);
    renderOverlay();
  }

  function setEnvironmentVisible(value) {
    environmentVisible = Boolean(value);
    environmentButton.textContent = environmentVisible ? '隐藏仿真物体' : '显示仿真物体';
    environmentButton.ariaPressed = String(!environmentVisible);
    onEnvironmentVisible?.(environmentVisible);
  }

  function refresh() {
    if (inFlight) return inFlight;
    if (!open) return Promise.resolve();
    const requestedRevision = revision;
    refreshButton.disabled = true;
    status.textContent = '正在读取最近一次已建图的 RGBD 快照…';
    inFlight = (async () => {
      try {
        const result = await request('/api/perception-preview');
        if (requestedRevision !== revision) return;
        validatePointcloudPreview(result);
        if (generation !== null && result.generation !== generation) throw new Error('点云属于其他场景，请等待当前场景的观察建图。');
        payload = result;
        mode.disabled = false;
        visibilityButton.disabled = false;
        const scene = payload.collision_scene;
        collisionOption.disabled = scene?.status !== 'available';
        collisionOption.textContent = collisionOption.disabled ? '碰撞场景' : collisionDescription(scene).title;
        if (mode.value === 'collision' && collisionOption.disabled) mode.value = 'raw';
        mapEvidence.replaceChildren();
        for (const [camera, evidence] of Object.entries(payload.cameras ?? {})) {
          const sensor = evidence.captured_sensor;
          const captured = sensor ? `采集有效 ${sensor.valid_depth_count} / ${sensor.valid_depth_count + sensor.invalid_depth_count}，无效 ${sensor.invalid_depth_count}；` : '旧快照无原始采集计数；';
          mapEvidence.append(element('p', 'helper-text', `相机 ${camera}：${captured}建图输入有效 ${evidence.valid_depth_count} / ${evidence.valid_depth_count + evidence.invalid_depth_count}（包含补边或禁用视图）；范围 ${evidence.depth_range_m?.map(value => value.toFixed(3)).join('–') ?? '无'} m。`));
        }
        if (scene?.status === 'available') {
          mapEvidence.append(element('p', 'helper-text', '碰撞检查使用下列已发布几何。遮挡 / 视野外空间未被独立阻止进入；当前没有未知空间运动门禁。碰撞模型不等于可见自由空间证明。'));
          for (const layer of scene.voxel_layers) {
            mapEvidence.append(element('p', 'helper-text', `${layer.enabled ? '参与碰撞' : '已排除'} · ${layer.name} · ${layer.position_source === 'action_prediction' ? '按释放反馈迁移的预测位置' : 'RGBD 建立'} · ${layer.voxel_size_m * 1000} mm 网格 · 显示 ${layer.displayed_surface_count} / Mapper 预览 ${layer.mapper_surface_preview_count} 个表面体素 · ESDF ${layer.esdf.cell_count} 单元。显示抽样不缩减规划网格。`));
          }
          for (const layer of Object.values(scene.mesh_layers ?? {})) {
            mapEvidence.append(element('p', 'helper-text', `${layer.enabled ? '参与碰撞' : '已排除'} · ${layer.instance_id} · CAD ${layer.asset_id} · 来源 ${layer.observation_id} · ${layer.vertex_count.toLocaleString()} 个顶点 / ${layer.face_count.toLocaleString()} 个三角面，完整网格、未抽样。模型与网格哈希见下方几何证据。`));
          }
          for (const layer of scene.cuboid_layers ?? []) {
            const dimensions = layer.model.dims.map(value => value.toFixed(3)).join(' × ');
            const detail = layer.role === 'measured_table' ? `实测桌面对应的碰撞体 · ${dimensions} m；桌下延伸与边界扩展属于模型假设。`
              : layer.geometry_source === 'observed_obb' ? `观测点 OBB 实心包络 · ${dimensions} m · 几何来源 ${(layer.geometry_observation_ids ?? [layer.observation_id]).join('、')}。`
              : `盒壁补全（实测平面 + 已知盒子尺寸） · ${dimensions} m；蓝色薄壳是推导几何，盒内空腔不填充。`;
            mapEvidence.append(element('p', 'helper-text', `${layer.enabled ? '参与碰撞' : '已排除'} · ${layer.name} · ${detail}`));
          }
          mapEvidence.append(element('p', 'helper-text', `推断桌体：RANSAC ${scene.table.ransac_inlier_count} 个内点，体积 ${scene.table.model.dims.map(value => value.toFixed(3)).join(' × ')} m；桌下延伸与边界扩展属于模型假设。`));
        } else {
          mapEvidence.append(element('p', 'helper-text', `实际碰撞地图暂不可用：${scene?.reason ?? '旧快照未导出地图，请等待下一次建图'}。原始点云不代表最终碰撞障碍。`));
        }
        provenanceText.textContent = JSON.stringify({
          frame: 'world', units: 'm', observation_id: payload.observation_id, scene_token: payload.scene_token,
          cameras: payload.cameras ?? null,
          point_source_examples: payload.camera_ids?.slice(0, 8).map((camera, index) => ({
            camera_id: camera, pixel_rc: payload.pixels_rc[index], point_world_m: payload.points_world_m[index] })),
          table: scene?.table, esdf_layers: scene?.voxel_layers?.length ? scene.voxel_layers.map(layer => ({ name: layer.name,
            pose_wxyz: layer.pose_wxyz, dims_m: layer.dims_m, esdf: layer.esdf })) : undefined,
          cuboid_layers: scene?.cuboid_layers,
          mesh_layers: Object.values(scene?.mesh_layers ?? {}).map(({vertices_local_m, faces, ...metadata}) => metadata),
          coverage: scene?.coverage,
          object_obbs: payload.object_obbs,
        }, null, 2);
        const total = payload.counts;
        counts.textContent = `全量点数：原始 ${total.input.toLocaleString()} · 保留 ${total.kept.toLocaleString()} · 剔除 ${total.rejected.toLocaleString()}`;
        const sampleKept = payload.kept_mask.filter(Boolean).length;
        const sampleRejected = payload.kept_mask.length - sampleKept;
        const samplingNote = payload.preview_sampling === 'stratified_original_indices'
          ? '按保留 / 剔除分层抽样，层内按原始索引均匀选点。两类预览点数不代表全量比例，请以上方全量数量为准。'
          : '按原始索引均匀抽样，可能未显示少量剔除点；以上数量统计全量数据。';
        sample.textContent = `预览 ${payload.points_world_m.length.toLocaleString()} / ${total.input.toLocaleString()} 点（预览中保留 ${sampleKept.toLocaleString()} · 剔除 ${sampleRejected.toLocaleString()}）；最多 30,000 点。${samplingNote}`;
        status.textContent = `观察节点 ${payload.checkpoint ?? '未提供'} · 观测 ${payload.observation_id} · 场景 ${payload.generation} · 建图 ${payload.scene_token}`;
        renderOverlay();
      } catch (error) {
        if (requestedRevision !== revision) return;
        clear();
        status.textContent = error.message.includes('perception_preview_unavailable')
          ? '尚无当前场景的点云建图快照。任务在观察节点完成建图后，可重新读取。'
          : `点云预览读取失败：${error.message}`;
      } finally {
        inFlight = null;
        refreshButton.disabled = false;
      }
    })();
    return inFlight;
  }

  mode.addEventListener('change', renderOverlay);
  refreshButton.addEventListener('click', () => void refresh());
  visibilityButton.addEventListener('click', () => setVisible(!visible));
  environmentButton.addEventListener('click', () => setEnvironmentVisible(!environmentVisible));
  obbButton.addEventListener('click', () => {
    showObbs = !showObbs;
    obbButton.textContent = showObbs ? '隐藏 OBB 线框' : '显示 OBB 线框';
    obbButton.ariaPressed = String(showObbs);
    renderOverlay();
  });

  return {
    refresh,
    setVisible,
    setOpen(value) {
      open = Boolean(value);
      if (open) return refresh();
      setVisible(false);
      setEnvironmentVisible(true);
      return Promise.resolve();
    },
    setGeneration(value) {
      if (!Number.isInteger(value) || value === generation) return;
      generation = value;
      revision += 1;
      clear();
      setEnvironmentVisible(true);
      status.textContent = '等待当前场景在观察节点完成点云建图。';
    },
    dispose() { open = false; revision += 1; clear(); setEnvironmentVisible(true); },
  };
}

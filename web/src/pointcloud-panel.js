import * as THREE from 'three';

const rawColor = new THREE.Color('#a7b3c4');
const modes = [['raw', '原始 RGBD 点云'], ['collision', 'TSDF / ESDF 碰撞场景（表面预览）']];

export function validatePointcloudPreview(payload) {
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
    const vector = (value, count) => Array.isArray(value) && value.length === count && value.every(Number.isFinite);
    if (!Array.isArray(scene.voxel_layers) || scene.voxel_layers.some(layer =>
      !vector(layer.pose_wxyz, 7) || !vector(layer.dims_m, 3) || layer.dims_m.some(value => value <= 0) ||
      !Number.isFinite(layer.surface_voxel_size_m) || layer.surface_voxel_size_m <= 0 ||
      !Array.isArray(layer.surface_centers_world_m) || layer.surface_centers_world_m.length > 15000 ||
      layer.surface_centers_world_m.some(point => !vector(point, 3))) ||
      !vector(scene.table?.model?.pose, 7) || !vector(scene.table?.model?.dims, 3) ||
      scene.table.provenance !== 'inferred_ransac_support_completion') throw new Error('碰撞场景网格或推断桌体无效。');
    if (scene.cuboid_layers !== undefined && (!Array.isArray(scene.cuboid_layers) || scene.cuboid_layers.some(layer =>
      layer.role !== 'bin_shell_completion' || layer.provenance !== 'measured_planes_and_user_bin_dimensions' ||
      typeof layer.enabled !== 'boolean' || typeof layer.instance_id !== 'string' || !layer.instance_id ||
      layer.observation_id !== payload.observation_id || layer.generation !== payload.generation ||
      !vector(layer.model?.pose, 7) || !vector(layer.model?.dims, 3) || layer.model.dims.some(value => value <= 0)))) {
      throw new Error('盒壁补全缺少有效尺寸先验来源，或不属于当前观测。');
    }
  }
}

function poseObject(object, pose) {
  object.position.set(...pose.slice(0, 3));
  object.quaternion.set(pose[4], pose[5], pose[6], pose[3]);
}

function addCollisionLayers(object, scene) {
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
  for (const layer of scene.cuboid_layers ?? []) {
    const wall = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({color: '#78b6ff', transparent: true,
        opacity: layer.enabled ? .35 : .1, depthWrite: false}));
    wall.name = layer.name;
    wall.userData = {provenance: layer.provenance, instanceId: layer.instance_id, enabled: layer.enabled};
    poseObject(wall, layer.model.pose);
    wall.scale.set(...layer.model.dims);
    object.add(wall);
  }
  const table = scene.table;
  const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ color: '#f597ce', transparent: true, opacity: .25, depthWrite: false }));
  box.name = table.name;
  box.userData.provenance = table.provenance;
  poseObject(box, table.model.pose);
  box.scale.set(...table.model.dims);
  object.add(box);
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

export function createPointcloudLayer() {
  const object = new THREE.Group();
  object.name = 'registered-rgbd-preview';

  function clear() {
    for (const points of object.children) {
      points.geometry.dispose();
      points.material.dispose();
    }
    object.clear();
  }

  return {
    object, clear, dispose: clear,
    update(payload, mode = 'raw') {
      validatePointcloudPreview(payload);
      if (!modes.some(([value]) => value === mode)) throw new Error('点云显示模式无效。');
      if (mode === 'collision') {
        if (payload.collision_scene?.status !== 'available') throw new Error('当前快照没有实际碰撞场景，请重新建图。');
        clear();
        addCollisionLayers(object, payload.collision_scene);
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
  const layer = createPointcloudLayer();

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
  controls.append(modeLabel, refreshButton, visibilityButton, environmentButton);
  const environmentNote = element('p', 'helper-text', '隐藏仿真物体后，仅保留机器人、相机与所选叠层；测量地图中的桌面碰撞体仍显示。');
  const counts = element('p', 'pointcloud-counts');
  const sample = element('p', 'helper-text');
  const legend = element('p', 'pointcloud-legend');
  const diagnostic = element('p', 'helper-text');
  const mapEvidence = element('div');
  const provenance = element('details');
  provenance.append(element('summary', '', '相机标定 / 像素来源 / 原生 ESDF 样本'));
  const provenanceText = element('pre');
  provenance.append(provenanceText);
  container.append(controls, environmentNote, counts, sample, legend, diagnostic, mapEvidence, provenance);

  function clear() {
    payload = null;
    layer.clear();
    setVisible(false);
    counts.textContent = '';
    sample.textContent = '';
    legend.textContent = '';
    diagnostic.textContent = '';
    mapEvidence.replaceChildren();
    provenanceText.textContent = '';
    mode.disabled = true;
    visibilityButton.disabled = true;
    collisionOption.disabled = true;
  }

  function renderOverlay() {
    if (!payload) { onOverlay(null); return; }
    const labels = {
      raw: '灰色：注册 RGBD 的过滤前可见点，包含保留与剔除样本；当前接口未提供逐点 RGB 颜色。',
      collision: '青色：背景 TSDF 表面 · 橙色：实例 TSDF 表面 · 灰色：已排除实例 · 粉色：推断桌面碰撞体 · 黄色：实测支撑范围。淡线框仅为 ESDF 网格范围。',
    };
    legend.textContent = labels[mode.value];
    diagnostic.textContent = mode.value === 'collision'
      ? '当前显示 Mapper 的 TSDF 表面体素采样与推断桌体，未绘制完整 ESDF 距离场；原生 ESDF 距离样本可在下方展开查看。'
      : '原始点云是最近一次建图观测的实测采样，不代表最终碰撞障碍，也不随机器人当前姿态重新采集。';
    if (!open || !visible) { onOverlay(null); return; }
    layer.update(payload, mode.value);
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
        if (mode.value === 'collision' && collisionOption.disabled) mode.value = 'raw';
        mapEvidence.replaceChildren();
        for (const [camera, evidence] of Object.entries(payload.cameras ?? {})) {
          const sensor = evidence.captured_sensor;
          const captured = sensor ? `采集有效 ${sensor.valid_depth_count} / ${sensor.valid_depth_count + sensor.invalid_depth_count}，无效 ${sensor.invalid_depth_count}；` : '旧快照无原始采集计数；';
          mapEvidence.append(element('p', 'helper-text', `相机 ${camera}：${captured}建图输入有效 ${evidence.valid_depth_count} / ${evidence.valid_depth_count + evidence.invalid_depth_count}（包含补边或禁用视图）；范围 ${evidence.depth_range_m?.map(value => value.toFixed(3)).join('–') ?? '无'} m。`));
        }
        if (scene?.status === 'available') {
          mapEvidence.append(element('p', 'helper-text', '碰撞检查覆盖建图范围内的实测障碍和推断支撑体。遮挡 / 视野外空间未被独立阻止进入；当前没有未知空间运动门禁。体素距离不等于可见自由空间证明。'));
          for (const layer of scene.voxel_layers) {
            mapEvidence.append(element('p', 'helper-text', `${layer.enabled ? '参与碰撞' : '已排除'} · ${layer.name} · ${layer.position_source === 'action_prediction' ? '按释放反馈迁移的预测位置' : 'RGBD 建立'} · ${layer.voxel_size_m * 1000} mm 网格 · 显示 ${layer.displayed_surface_count} / Mapper 预览 ${layer.mapper_surface_preview_count} 个表面体素 · ESDF ${layer.esdf.cell_count} 单元。显示抽样不缩减规划网格。`));
          }
          for (const layer of scene.cuboid_layers ?? []) {
            mapEvidence.append(element('p', 'helper-text', `${layer.enabled ? '参与碰撞' : '已排除'} · ${layer.name} · 盒壁补全（实测平面 + 已知盒子尺寸） · ${layer.model.dims.map(value => value.toFixed(3)).join(' × ')} m；蓝色薄壳是推导几何，盒内空腔不填充。`));
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
          table: scene?.table, esdf_layers: scene?.voxel_layers?.map(layer => ({ name: layer.name,
            pose_wxyz: layer.pose_wxyz, dims_m: layer.dims_m, esdf: layer.esdf })),
          derived_bin_walls: scene?.cuboid_layers,
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

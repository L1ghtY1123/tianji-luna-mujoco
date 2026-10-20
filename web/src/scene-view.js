import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createRenderSchedule } from './render-schedule.js';
import { createDragProjection, createObjectDrag, editableObjectPosition } from './object-drag.js';
import { applyScenePreview } from './scene-preview.js';
import { createTcpFrames } from './tcp-frames.js';

function tcpLabel(text) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  context.fillStyle = '#10202eee';
  context.fillRect(0, 0, 256, 64);
  context.fillStyle = '#ffffff';
  context.font = 'bold 36px sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, 128, 32);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
  sprite.scale.set(0.1, 0.025, 1);
  sprite.position.set(0, 0, -0.025);
  sprite.renderOrder = 22;
  return sprite;
}

function geometryFor(item, mesh) {
  const size = item.size ?? [0.05, 0.05, 0.05];
  if (item.type === 'mesh') {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(mesh.vertices.flat(), 3));
    geometry.setIndex(mesh.faces.flat());
    geometry.computeVertexNormals();
    return geometry;
  }
  if (item.type === 'sphere') return new THREE.SphereGeometry(size[0], 24, 16);
  if (item.type === 'ellipsoid') return new THREE.SphereGeometry(1, 24, 16).scale(...size);
  if (item.type === 'cylinder') return new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 32).rotateX(Math.PI / 2);
  if (item.type === 'capsule') return new THREE.CapsuleGeometry(size[0], size[1] * 2, 8, 24).rotateX(Math.PI / 2);
  if (item.type === 'plane') return new THREE.PlaneGeometry((size[0] || 3) * 2, (size[1] || 3) * 2);
  return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
}

export function createSceneView(container, { onSelect, onDragStart, onDragEnd, onDragCancel } = {}) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#101923');
  const camera = new THREE.PerspectiveCamera(39, 1, 0.01, 100);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  const renderSchedule = createRenderSchedule(30);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  container.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener('change', () => renderSchedule.invalidate());
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.25;
  controls.maxDistance = 8;
  controls.maxPolarAngle = Math.PI * 0.94;
  // Neutral diffuse lighting preserves the same authored colors as the RGBD scene.
  // Lambert's BRDF divides irradiance by pi.
  const ambient = new THREE.AmbientLight(0xffffff, Math.PI * 0.8);
  scene.add(ambient);
  const key = new THREE.DirectionalLight(0xffffff, Math.PI * 0.2);
  key.position.set(1.5, -2, 4);
  scene.add(key);
  const grid = new THREE.GridHelper(6, 60, 0x3b4e5d, 0x22313e);
  grid.rotateX(Math.PI / 2);
  grid.position.z = 0.001;
  scene.add(grid);
  const axes = new THREE.AxesHelper(0.22);
  axes.position.set(0, 0, 0.004);
  scene.add(axes);
  const tcpFrames = createTcpFrames(tcpLabel);
  scene.add(tcpFrames.root);
  const geometryRoot = new THREE.Group();
  scene.add(geometryRoot);
  let overlay = null;
  let environmentVisible = true;
  const objects = new Map();
  const meshCache = new Map();
  let manifestSequence = 0;
  let lastSnapshot = { geometry_poses: [], objects: [] };
  let homeTarget = new THREE.Vector3(0, 0, 0.6);
  let interactionEnabled = false;
  let selectedId = null;
  let hoveredId = null;
  let pointerDrag = null;
  let objectIds = new Set();
  let draft = null;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const canvas = renderer.domElement;
  const drag = createObjectDrag({ onDragEnd, onDragCancel, onChange: refreshDragView });
  controls.addEventListener('start', () => { hoveredId = null; updateHighlight(); });

  function home({ target = homeTarget.toArray(), offset = [1.65, -2.05, 1.45] } = {}) {
    drag.cancel();
    controls.target.fromArray(target);
    camera.position.copy(controls.target).add(new THREE.Vector3(...offset));
    controls.update();
  }

  function update(snapshot) {
    lastSnapshot = snapshot;
    tcpFrames.update(snapshot);
    if (tcpFrames.root.visible) renderSchedule.invalidate();
    objectIds = new Set((snapshot.objects ?? []).map(object => object.id));
    drag.updateSnapshot(snapshot);
    if (!objectIds.has(hoveredId)) hoveredId = null;
    applyLatestPoses();
    updateHighlight();
  }

  function applyLatestPoses() {
    for (const pose of lastSnapshot.geometry_poses ?? []) {
      const object = objects.get(String(pose.id));
      if (!object) continue;
      if (applyScenePreview(object, object.userData.definition, pose, draft, drag.active)) renderSchedule.invalidate();
    }
  }

  function releasePointer() {
    if (!pointerDrag) return;
    const pointerId = pointerDrag.pointerId;
    pointerDrag = null;
    if (canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
  }

  function refreshDragView() {
    if (!drag.active) releasePointer();
    controls.enabled = !drag.active;
    applyLatestPoses();
    updateHighlight();
  }

  function updateHighlight() {
    for (const object of objects.values()) {
      const id = object.userData.objectId;
      let color = 0;
      if (objectIds.has(id)) {
        if (id === drag.active?.id) color = 0x326d87;
        else if (id === hoveredId) color = 0x285d78;
        else if (id === selectedId) color = 0x203d50;
      }
      if (object.material.emissive.getHex() !== color) {
        object.material.emissive.setHex(color);
        renderSchedule.invalidate();
      }
    }
    canvas.style.cursor = drag.active ? 'grabbing' : interactionEnabled && hoveredId ? 'grab' : '';
  }

  function pointerRay(event) {
    const rect = canvas.getBoundingClientRect();
    pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    camera.updateMatrixWorld();
    raycaster.setFromCamera(pointer, camera);
    return raycaster.ray;
  }

  function pickObject(event) {
    if (!geometryRoot.visible) return null;
    pointerRay(event);
    geometryRoot.updateMatrixWorld(true);
    // Use the first visible geometry hit, so a robot link or table occludes
    // an object behind it instead of allowing a drag through the foreground.
    const hit = raycaster.intersectObjects(geometryRoot.children.filter(object => object.visible), false)[0];
    const id = hit?.object.userData.objectId;
    return objectIds.has(id) && (!draft || draft.objects.some(object => object.id === id)) ? hit.object : null;
  }

  function stopPointerEvent(event) {
    event.preventDefault();
    event.stopImmediatePropagation();
  }

  canvas.addEventListener('pointerdown', event => {
    if (!interactionEnabled || drag.active || event.button !== 0 || event.isPrimary === false) return;
    const object = pickObject(event);
    if (!object) return;
    const id = object.userData.objectId;
    const position = editableObjectPosition(id, object.position.toArray(), lastSnapshot, draft, drag.active);
    const depth = -new THREE.Vector3(...position).applyMatrix4(camera.matrixWorldInverse).z;
    const project = createDragProjection({
      position, ray: raycaster.ray, vertical: event.shiftKey && object.userData.definition.object_kind !== 'marker',
      screenY: event.clientY, metersPerPixel: 2 * depth * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / canvas.getBoundingClientRect().height,
    });
    if (!project) return;
    stopPointerEvent(event);
    selectedId = id;
    hoveredId = id;
    if (!drag.begin(id, position, lastSnapshot.generation)) return;
    pointerDrag = { pointerId: event.pointerId, project };
    canvas.setPointerCapture(event.pointerId);
    onSelect?.(id);
    onDragStart?.(id);
  }, { capture: true });

  canvas.addEventListener('pointermove', event => {
    if (pointerDrag && event.pointerId === pointerDrag.pointerId) {
      stopPointerEvent(event);
      const position = pointerDrag.project(pointerRay(event), event.clientY);
      if (position?.every(Number.isFinite)) drag.move(position);
      return;
    }
    if (!interactionEnabled || drag.active || event.buttons) return;
    const id = pickObject(event)?.userData.objectId ?? null;
    if (hoveredId !== id) { hoveredId = id; updateHighlight(); }
  }, { capture: true });

  canvas.addEventListener('pointerup', event => {
    if (!pointerDrag || event.pointerId !== pointerDrag.pointerId) return;
    stopPointerEvent(event);
    drag.finish();
  }, { capture: true });

  function cancelPointer(event) {
    if (!pointerDrag || event.pointerId !== pointerDrag.pointerId) return;
    stopPointerEvent(event);
    drag.cancel();
  }
  canvas.addEventListener('pointercancel', cancelPointer, { capture: true });
  canvas.addEventListener('lostpointercapture', cancelPointer);
  canvas.addEventListener('pointerleave', () => {
    if (!pointerDrag) { hoveredId = null; updateHighlight(); }
  });
  window.addEventListener('keydown', event => {
    if (event.key === 'Escape' && drag.active) {
      event.preventDefault();
      drag.cancel();
    }
  });
  window.addEventListener('blur', () => drag.cancel());

  function applyEnvironmentVisibility() {
    for (const object of objects.values()) {
      const definition = object.userData.definition;
      const environment = object.userData.objectId !== null || definition.name === 'table' || definition.name === 'ground';
      object.visible = environmentVisible || !environment;
    }
    grid.visible = environmentVisible;
    axes.visible = environmentVisible;
    renderSchedule.invalidate();
  }

  async function loadManifest(manifest) {
    const sequence = ++manifestSequence;
    const meshes = await Promise.all(manifest.geometries.map(async item => {
      let meshData = item.mesh;
      if (item.mesh_url) {
        if (!meshCache.has(item.mesh_url)) {
          const response = await fetch(item.mesh_url);
          if (!response.ok) throw new Error(`网格加载失败：${item.mesh_url}（${response.status}）`);
          meshCache.set(item.mesh_url, await response.json());
        }
        meshData = meshCache.get(item.mesh_url);
      }
      const geometry = geometryFor(item, meshData);
      const rgba = item.rgba ?? [0.62, 0.69, 0.77, 1];
      const color = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
      const material = new THREE.MeshLambertMaterial({ color, transparent: rgba[3] < 1, opacity: rgba[3], side: item.type === 'plane' ? THREE.DoubleSide : THREE.FrontSide });
      const object = new THREE.Mesh(geometry, material);
      object.name = String(item.id);
      object.userData.objectId = item.object_id ?? (item.name?.startsWith('object_geom_') ? item.name.slice('object_geom_'.length) : null);
      object.userData.definition = item;
      applyScenePreview(object, item, item, null);
      return object;
    }));
    if (sequence !== manifestSequence) {
      for (const object of meshes) { object.geometry.dispose(); object.material.dispose(); }
      return;
    }
    drag.cancel();
    for (const object of objects.values()) { object.geometry.dispose(); object.material.dispose(); }
    objects.clear();
    geometryRoot.clear();
    for (const object of meshes) { objects.set(object.name, object); geometryRoot.add(object); }
    applyEnvironmentVisibility();
    update(lastSnapshot);
    renderSchedule.invalidate();
  }

  const resize = new ResizeObserver(() => {
    const { width, height } = container.getBoundingClientRect();
    if (!width || !height) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    renderSchedule.invalidate();
  });
  resize.observe(container);
  home();
  renderer.setAnimationLoop(timeMs => {
    if (controls.enabled) controls.update();
    if (renderSchedule.shouldRender(timeMs)) renderer.render(scene, camera);
  });

  return {
    loadManifest,
    update,
    home,
    setTcpFramesVisible(value) {
      tcpFrames.setVisible(value);
      renderSchedule.invalidate();
    },
    setOverlay(value) {
      if (overlay !== value) {
        if (overlay) scene.remove(overlay);
        overlay = value;
        if (overlay) scene.add(overlay);
      }
      renderSchedule.invalidate();
    },
    setGeometryVisible(visible) {
      geometryRoot.visible = visible;
      renderSchedule.invalidate();
    },
    setEnvironmentVisible(visible) {
      environmentVisible = Boolean(visible);
      drag.cancel();
      hoveredId = null;
      applyEnvironmentVisibility();
      updateHighlight();
    },
    cancelInteraction() { drag.cancel(); },
    setDraft(value) {
      draft = value;
      applyLatestPoses();
    },
    setInteraction({ enabled, selectedId: id }) {
      interactionEnabled = enabled;
      selectedId = id;
      if (!enabled) hoveredId = null;
      drag.setEnabled(enabled);
      updateHighlight();
    },
    setTable(table) {
      if (table?.position) homeTarget = new THREE.Vector3(...table.position).add(new THREE.Vector3(0, 0, 0.15));
    },
  };
}

import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';
import * as THREE from 'three';
import { createRenderSchedule } from '../src/render-schedule.js';
import { createDragProjection, createObjectDrag, editableObjectPosition } from '../src/object-drag.js';
import { applyScenePreview } from '../src/scene-preview.js';
import { createTcpFrames } from '../src/tcp-frames.js';

// Keep real meshes, transforms and raycasting; replace only the browser/WebGL boundary.
const source = (await readFile(new URL('../src/scene-view.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\r?\n/gm, '').replace('export function createSceneView', 'function createSceneView');

function harness() {
  let scene;
  let draw;
  let time = 0;
  const listeners = {};
  const selected = [];
  const canvas = { style: {}, addEventListener: (name, callback) => { listeners[name] = callback; },
    setPointerCapture() {}, hasPointerCapture: () => false,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }) };
  class WebGLRenderer {
    domElement = canvas;
    setPixelRatio() {}
    setAnimationLoop(callback) { draw = callback; }
    render(value) { scene = value; }
  }
  class OrbitControls {
    target = new THREE.Vector3();
    enabled = true;
    constructor(camera) { this.camera = camera; }
    addEventListener() {}
    update() { this.camera.lookAt(this.target); }
  }
  const context = vm.createContext({ THREE: { ...THREE, WebGLRenderer }, OrbitControls, createRenderSchedule,
    createDragProjection, createObjectDrag, editableObjectPosition, applyScenePreview,
    createTcpFrames: () => createTcpFrames(),
    window: { devicePixelRatio: 1, addEventListener() {} }, ResizeObserver: class { observe() {} } });
  vm.runInContext(source, context);
  const view = context.createSceneView({ append() {} }, { onSelect: id => selected.push(id) });
  function render() { draw(time += 100); return scene; }
  return { view, render, listeners, selected };
}

function manifest() {
  const definitions = [
    { name: 'table' }, { name: 'ground', type: 'plane' },
    { name: 'object_geom_cube', object_id: 'cube', object_kind: 'cube' },
    { name: 'object_geom_bin__bottom', object_id: 'bin', object_kind: 'bin' },
    { name: 'object_geom_bin__wall_x_plus', object_id: 'bin', object_kind: 'bin' },
    { name: 'object_geom_marker', object_id: 'marker', object_kind: 'marker' },
    { name: 'object_geom_legacy' },
    { name: 'Link_Base' }, { name: 'arm_A_Link7' }, { name: 'arm_B_Link7' },
    { name: 'left_wrist_camera' }, { name: 'right_wrist_camera' }, { name: 'head_camera' },
  ];
  return { geometries: definitions.map((item, index) => ({
    id: String(index), type: 'box', size: [.02, .02, .02], position: [0, 0, 0], quaternion: [1, 0, 0, 0], ...item,
  })) };
}

test('isolating measured geometry hides configured environment and helpers while keeping robot, cameras and map', async () => {
  const ui = harness();
  await ui.view.loadManifest(manifest());
  const map = new THREE.Group();
  const measuredTable = new THREE.Mesh(new THREE.BoxGeometry(.8, .6, .3), new THREE.MeshBasicMaterial());
  measuredTable.name = 'semantic_table_forbidden';
  map.add(measuredTable);
  ui.view.setOverlay(map);
  ui.view.setEnvironmentVisible(false);
  const scene = ui.render();
  const root = scene.children.find(child => child.isGroup && child.children.some(mesh => mesh.userData.definition));
  assert.deepEqual([...root.children.map(mesh => mesh.visible)], [false, false, false, false, false, false, false, true, true, true, true, true, true]);
  assert.equal(scene.children.find(child => child.type === 'GridHelper').visible, false);
  assert.equal(scene.children.find(child => child.type === 'AxesHelper').visible, false);
  assert.equal(map.parent, scene);
  assert.equal(measuredTable.visible, true);
  ui.view.update({ generation: 1, objects: [], geometry_poses: [{ id: '7', position: [.2, 0, .4], quaternion: [1, 0, 0, 0] }] });
  assert.deepEqual(root.children[7].position.toArray(), [.2, 0, .4]);
  await ui.view.loadManifest(manifest());
  assert.equal(root.children[0].visible, false);
  assert.equal(root.children[10].visible, true);
  ui.view.setEnvironmentVisible(true);
  assert.equal(root.children.every(mesh => mesh.visible), true);
  assert.equal(scene.children.find(child => child.type === 'GridHelper').visible, true);
  measuredTable.geometry.dispose();
  measuredTable.material.dispose();
});

test('hidden simulation objects cannot be selected or dragged through the measured overlay', async () => {
  const ui = harness();
  const cube = manifest().geometries.find(item => item.object_id === 'cube');
  await ui.view.loadManifest({ geometries: [cube] });
  ui.view.update({ generation: 1, objects: [{ id: 'cube', position: [0, 0, 0] }], geometry_poses: [] });
  ui.view.home({ target: [0, 0, 0], offset: [0, -1, 0] });
  ui.view.setInteraction({ enabled: true, selectedId: null });
  ui.view.setEnvironmentVisible(false);
  const down = () => ui.listeners.pointerdown({ clientX: 400, clientY: 300, button: 0, pointerId: 1,
    preventDefault() {}, stopImmediatePropagation() {} });
  down();
  assert.deepEqual(ui.selected, []);
  ui.view.setEnvironmentVisible(true);
  down();
  assert.deepEqual(ui.selected, ['cube']);
  ui.view.cancelInteraction();
});

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = { passed: '#73dfb2', failed: '#ef8d88', unknown: '#ecc77d' };

function validMatrix(matrix) {
  return Array.isArray(matrix) && matrix.length === 4
    && matrix.every(row => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite));
}

export function candidateLineGeometry(candidates) {
  const positions = [], colors = [];
  // A compact tool-frame glyph: two fingers, a bridge and the +Z approach axis.
  const segments = [
    [[-.025, 0, -.035], [-.025, 0, 0]], [[.025, 0, -.035], [.025, 0, 0]],
    [[-.025, 0, -.035], [.025, 0, -.035]], [[0, 0, -.035], [0, 0, .025]],
  ];
  for (const candidate of candidates) {
    if (!validMatrix(candidate.matrix)) continue;
    const matrix = new THREE.Matrix4().set(...candidate.matrix.flat());
    const color = new THREE.Color(COLORS[candidate.status] ?? COLORS.unknown);
    for (const segment of segments) {
      for (const point of segment) {
        const world = new THREE.Vector3(...point).applyMatrix4(matrix);
        positions.push(world.x, world.y, world.z);
        colors.push(color.r, color.g, color.b);
      }
    }
    for (const [endpoint, axisColor] of [
      [[.035, 0, 0], '#ff5c61'], [[0, .035, 0], '#5cdd88'], [[0, 0, .035], '#5b9dff'],
    ]) {
      const tint = new THREE.Color(axisColor);
      for (const point of [[0, 0, 0], endpoint]) {
        const world = new THREE.Vector3(...point).applyMatrix4(matrix);
        positions.push(world.x, world.y, world.z);
        colors.push(tint.r, tint.g, tint.b);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return geometry;
}

export function replayBounds(cloud, lines) {
  return new THREE.Box3().setFromObject(cloud).union(new THREE.Box3().setFromObject(lines));
}

export function createReplayCloud(container, { points, candidates = [] }) {
  if (!Array.isArray(points) || points.some(point => !Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite))) {
    throw new Error('归档点云必须是有限的世界坐标 Nx3 数组');
  }
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#091722');
  const camera = new THREE.PerspectiveCamera(42, 1, .001, 100);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.minDistance = .02;
  controls.maxDistance = 20;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points.flat(), 3));
  const cloud = new THREE.Points(geometry, new THREE.PointsMaterial({ color: '#bac9d8', size: .002, sizeAttenuation: true }));
  scene.add(cloud);
  const lines = new THREE.LineSegments(candidateLineGeometry(candidates),
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: .85 }));
  scene.add(lines);
  let disposed = false;
  function render() { if (!disposed) renderer.render(scene, camera); }
  function resize() {
    const width = Math.max(1, container.clientWidth), height = Math.max(1, container.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    render();
  }
  function fit() {
    const box = replayBounds(cloud, lines);
    const center = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
    const radius = box.isEmpty() ? .1 : Math.max(.04, box.getSize(new THREE.Vector3()).length() / 2);
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(1, -1.4, .9).normalize().multiplyScalar(radius * 3.8));
    camera.near = Math.max(.0005, radius / 100);
    camera.far = Math.max(10, radius * 30);
    camera.updateProjectionMatrix();
    controls.update();
    render();
  }
  controls.addEventListener('change', render);
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize(); fit();
  return {
    fit,
    setCandidates(rows) {
      if (disposed) return;
      lines.geometry.dispose();
      lines.geometry = candidateLineGeometry(rows);
      render();
    },
    dispose() {
      disposed = true;
      observer.disconnect(); controls.dispose();
      cloud.geometry.dispose(); cloud.material.dispose();
      lines.geometry.dispose(); lines.material.dispose();
      renderer.dispose(); renderer.domElement.remove();
    },
  };
}

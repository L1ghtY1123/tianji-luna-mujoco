import * as THREE from 'three';

export function validateCollisionSpheres(payload) {
  if (payload?.gripper_state !== 'open' || payload.radius_policy !== 'original_yaml_plus_link_buffer') {
    throw new Error('查看器只接受固定开夹爪的原始规划碰撞球。');
  }
  if (!Array.isArray(payload.spheres) || !payload.spheres.length || payload.sphere_count !== payload.spheres.length) {
    throw new Error('碰撞球数据不完整。');
  }
  const links = new Set();
  for (const sphere of payload.spheres) {
    if (typeof sphere.link !== 'string' || !sphere.link ||
        !Array.isArray(sphere.center_m) || sphere.center_m.length !== 3 || !sphere.center_m.every(Number.isFinite) ||
        !Number.isFinite(sphere.radius_m) || sphere.radius_m <= 0) {
      throw new Error('碰撞球中心或半径无效。');
    }
    links.add(sphere.link);
  }
  if (payload.link_count !== links.size) throw new Error('碰撞球 link 数量不一致。');
}

function createSphereGeometry(minimumDetail, maximumRadiusM) {
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const normal = new THREE.Vector3();
  for (let detail = minimumDetail; ; detail += 1) {
    const geometry = new THREE.IcosahedronGeometry(1, detail);
    const positions = geometry.getAttribute('position');
    let maximumInset = 0;
    for (let index = 0; index < positions.count; index += 3) {
      a.fromBufferAttribute(positions, index);
      b.fromBufferAttribute(positions, index + 1).sub(a);
      c.fromBufferAttribute(positions, index + 2).sub(a);
      normal.crossVectors(b, c).normalize();
      maximumInset = Math.max(maximumInset, 1 - Math.abs(normal.dot(a)));
    }
    const supportedRadiusM = .0002 / maximumInset;
    if (maximumRadiusM <= supportedRadiusM) return { geometry, supportedRadiusM };
    geometry.dispose();
  }
}

export function createCollisionSphereLayer() {
  const object = new THREE.Group();
  object.name = 'original-open-gripper-collision-spheres';
  // Three shared meshes keep small spheres inexpensive without changing radii.
  // These radius/detail pairs keep facet inset below 0.2 mm, verified on creation.
  const groups = [
    { maximumRadiusM: .017, detail: 4, mesh: null, supportedRadiusM: 0 },
    { maximumRadiusM: .044, detail: 7, mesh: null, supportedRadiusM: 0 },
    { maximumRadiusM: Infinity, detail: 15, mesh: null, supportedRadiusM: 0 },
  ];
  let opacity = .65;
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const matrix = new THREE.Matrix4();

  function disposeMesh(group) {
    const mesh = group.mesh;
    if (!mesh) return;
    object.remove(mesh);
    mesh.dispose();
    mesh.geometry.dispose();
    mesh.material.dispose();
    group.mesh = null;
  }

  return {
    object,
    update(payload) {
      validateCollisionSpheres(payload);
      const spheresByGroup = groups.map(() => []);
      for (const sphere of payload.spheres) {
        const index = groups.findIndex(group => sphere.radius_m <= group.maximumRadiusM);
        spheresByGroup[index].push(sphere);
      }
      groups.forEach((group, groupIndex) => {
        const spheres = spheresByGroup[groupIndex];
        if (!spheres.length) {
          disposeMesh(group);
          return;
        }
        const maximumRadiusM = spheres.reduce((maximum, sphere) => Math.max(maximum, sphere.radius_m), 0);
        if (!group.mesh || group.mesh.count !== spheres.length || maximumRadiusM > group.supportedRadiusM) {
          // Large spheres beyond 175 mm increase this group's shared detail only.
          const { geometry, supportedRadiusM } = createSphereGeometry(group.detail, maximumRadiusM);
          disposeMesh(group);
          group.supportedRadiusM = supportedRadiusM;
          group.mesh = new THREE.InstancedMesh(
            geometry,
            new THREE.MeshStandardMaterial({ color: 0x5ac8ff, transparent: true, opacity, depthWrite: false, roughness: .56 }),
            spheres.length,
          );
          object.add(group.mesh);
        }
        const mesh = group.mesh;
        spheres.forEach((sphere, index) => {
          position.fromArray(sphere.center_m);
          // The backend already applies the original YAML link buffer exactly once.
          scale.setScalar(sphere.radius_m);
          mesh.setMatrixAt(index, matrix.compose(position, quaternion, scale));
        });
        mesh.instanceMatrix.needsUpdate = true;
        mesh.computeBoundingSphere();
      });
    },
    setOpacity(value) {
      opacity = value;
      for (const group of groups) if (group.mesh) group.mesh.material.opacity = value;
    },
    dispose() {
      for (const group of groups) disposeMesh(group);
    },
  };
}

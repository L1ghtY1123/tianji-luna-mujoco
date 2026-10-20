import * as THREE from 'three';

export function createWorkspaceLayer(payload) {
  if (payload?.source !== 'offline_single_arm_workspace' || payload.frame !== 'Link_Base'
      || payload.surface_policy !== 'exposed_voxel_faces' || payload.arms?.length !== 2) {
    throw new Error('工作空间来源或坐标系无效。');
  }
  for (const [index, arm] of payload.arms.entries()) {
    if (arm.arm !== ['A', 'B'][index] || arm.origin_m?.length !== 3 || !arm.origin_m.every(Number.isFinite)
        || !Number.isFinite(arm.resolution_m) || arm.resolution_m <= 0 || arm.shape?.length !== 3
        || !arm.shape.every(value => Number.isInteger(value) && value > 0) || arm.faces?.length !== 6) {
      throw new Error('工作空间网格参数无效。');
    }
    const cellCount = arm.shape.reduce((total, value) => total * value, 1);
    for (const [i, face] of arm.faces.entries()) {
      if (face.axis !== Math.floor(i / 2) || face.sign !== (i % 2 ? 1 : -1)
          || !Array.isArray(face.indices)
          || !face.indices.every(cell => Number.isInteger(cell) && cell >= 0 && cell < cellCount)) {
        throw new Error('工作空间表面索引无效。');
      }
    }
  }
  const object = new THREE.Group();
  object.name = 'offline-arm-workspaces';
  for (const arm of payload.arms) {
    const faceCount = arm.faces.reduce((total, face) => total + face.indices.length, 0);
    const vertices = new Float32Array(faceCount * 18);
    let offset = 0;
    for (const face of arm.faces) {
      const u = (face.axis + 1) % 3;
      const corners = [[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]];
      const order = face.sign === 1 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2];
      for (const cell of face.indices) {
        const coordinate = [Math.floor(cell / (arm.shape[1] * arm.shape[2])),
          Math.floor(cell / arm.shape[2]) % arm.shape[1], cell % arm.shape[2]];
        for (const corner of order) {
          for (let axis = 0; axis < 3; axis++) {
            const delta = axis === face.axis ? face.sign * .5 : corners[corner][axis === u ? 0 : 1];
            vertices[offset++] = arm.origin_m[axis] + (coordinate[axis] + delta) * arm.resolution_m;
          }
        }
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.computeVertexNormals();
    const material = new THREE.MeshLambertMaterial({ color: arm.arm === 'A' ? 0x36b7ff : 0xffa64d,
      transparent: true, opacity: .22, depthWrite: false, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = arm.arm;
    object.add(mesh);
  }
  return {
    object,
    setVisible(arm, visible) { object.getObjectByName(arm).visible = visible; },
    setOpacity(value) { for (const mesh of object.children) mesh.material.opacity = value; },
    dispose() { for (const mesh of object.children) { mesh.geometry.dispose(); mesh.material.dispose(); } },
  };
}

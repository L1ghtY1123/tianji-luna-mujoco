import * as THREE from 'three';

export function createTcpFrames(makeLabel) {
  const root = new THREE.Group();
  root.visible = false;
  const frames = new Map();
  for (const arm of ['A', 'B']) {
    const frame = new THREE.Group();
    frame.matrixAutoUpdate = false;
    const axes = new THREE.AxesHelper(0.08);
    axes.material.depthTest = false;
    axes.renderOrder = 20;
    frame.add(axes);
    const origin = new THREE.Mesh(new THREE.SphereGeometry(0.003), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
    origin.renderOrder = 21;
    frame.add(origin);
    if (makeLabel) frame.add(makeLabel(`${arm} · TCP`));
    frames.set(arm, frame);
    root.add(frame);
  }
  return {
    root,
    update(snapshot) {
      for (const [arm, frame] of frames) {
        // Backend uses the same calibration as placement; no second offset here.
        const pose = snapshot.arms?.[arm]?.tcp_front;
        frame.visible = Array.isArray(pose) && pose.length === 4 && pose.every(row => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite));
        if (frame.visible) frame.matrix.set(...pose.flat());
      }
    },
    setVisible(value) { root.visible = Boolean(value); },
  };
}

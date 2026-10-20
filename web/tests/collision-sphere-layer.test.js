import assert from 'node:assert/strict';
import test from 'node:test';
import { Matrix4, Quaternion, Vector3 } from 'three';
import { createCollisionSphereLayer } from '../src/collision-sphere-layer.js';

const payload = () => ({
  gripper_state: 'open', radius_policy: 'original_yaml_plus_link_buffer',
  sphere_count: 2, link_count: 2,
  spheres: [
    { id: 'left:0', link: 'left', center_m: [.2, -.3, .75], radius_m: .023 },
    { id: 'right:0', link: 'right', center_m: [.4, .3, .65], radius_m: .031 },
  ],
});

function transform(mesh, index) {
  const matrix = new Matrix4();
  const center = new Vector3();
  const scale = new Vector3();
  mesh.getMatrixAt(index, matrix);
  matrix.decompose(center, new Quaternion(), scale);
  return { center: center.toArray(), scale: scale.toArray() };
}

function near(actual, expected) {
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-7, `${actual} != ${expected}`));
}

function withRadii(radii) {
  return {
    gripper_state: 'open', radius_policy: 'original_yaml_plus_link_buffer',
    sphere_count: radii.length, link_count: 1,
    spheres: radii.map((radius_m, index) => ({ id: `link:${index}`, link: 'link', center_m: [index / 100, 0, 0], radius_m })),
  };
}

function maximumInset(geometry, radiusM) {
  const positions = geometry.getAttribute('position');
  const indices = geometry.getIndex();
  const count = indices?.count ?? positions.count;
  let minimumFaceRadius = 1;
  for (let index = 0; index < count; index += 3) {
    const vertices = [0, 1, 2].map(offset => new Vector3().fromBufferAttribute(
      positions, indices ? indices.getX(index + offset) : index + offset,
    ));
    const normal = vertices[1].clone().sub(vertices[0]).cross(vertices[2].clone().sub(vertices[0])).normalize();
    minimumFaceRadius = Math.min(minimumFaceRadius, Math.abs(normal.dot(vertices[0])));
    for (const vertex of vertices) assert.ok(Math.abs(vertex.length() - 1) < 1e-7, 'do not enlarge sphere vertices');
  }
  return radiusM * (1 - minimumFaceRadius);
}

function watchDisposal(mesh) {
  const released = [];
  for (const resource of [mesh, mesh.geometry, mesh.material]) {
    resource.addEventListener('dispose', () => released.push(resource));
  }
  return released;
}

test('renders every original world sphere at its exact center and supplied radius without extra padding', () => {
  const layer = createCollisionSphereLayer();
  const data = payload();
  const before = structuredClone(data);
  layer.update(data);
  const mesh = layer.object.children[0];
  assert.equal(mesh.isInstancedMesh, true);
  assert.equal(mesh.count, 2);
  data.spheres.forEach((sphere, index) => {
    const actual = transform(mesh, index);
    near(actual.center, sphere.center_m);
    near(actual.scale, Array(3).fill(sphere.radius_m));
  });
  assert.deepEqual(data, before);
  assert.ok(mesh.boundingSphere.radius > 0);
});

test('shared sphere geometry stays within 0.2 mm of the analytic surface at a 160 mm radius', () => {
  // The current planning model's maximum buffered radius is 150.866 mm.
  const maximumRadiusM = .160;
  const data = payload();
  data.spheres = Array.from({ length: 1096 }, (_, index) => ({
    id: `link:${index}`, link: 'link', center_m: [0, 0, 0], radius_m: maximumRadiusM,
  }));
  data.sphere_count = data.spheres.length;
  data.link_count = 1;
  const layer = createCollisionSphereLayer();
  layer.update(data);
  assert.equal(layer.object.children.length, 1);
  const mesh = layer.object.children[0];
  assert.equal(mesh.isInstancedMesh, true);
  assert.equal(mesh.count, data.sphere_count);
  near(transform(mesh, 0).scale, [maximumRadiusM, maximumRadiusM, maximumRadiusM]);
  const positions = mesh.geometry.getAttribute('position');
  const indices = mesh.geometry.getIndex();
  const count = indices?.count ?? positions.count;
  let minimumFaceRadius = 1;
  for (let index = 0; index < count; index += 3) {
    const vertices = [0, 1, 2].map(offset => new Vector3().fromBufferAttribute(
      positions, indices ? indices.getX(index + offset) : index + offset,
    ));
    const normal = vertices[1].clone().sub(vertices[0]).cross(vertices[2].clone().sub(vertices[0])).normalize();
    minimumFaceRadius = Math.min(minimumFaceRadius, Math.abs(normal.dot(vertices[0])));
    for (const vertex of vertices) assert.ok(Math.abs(vertex.length() - 1) < 1e-7, 'do not enlarge sphere vertices');
  }
  const maximumInsetM = maximumRadiusM * (1 - minimumFaceRadius);
  assert.ok(maximumInsetM <= .0002, `sphere facets inset by ${maximumInsetM * 1000} mm`);
  assert.ok(count / 3 <= 6000, 'keep a bounded shared geometry triangle budget');
  layer.dispose();
});

test('uses three shared geometries with sub-0.2 mm error without spending large-sphere triangles on small spheres', () => {
  // Current model counts, conservatively placed at the top of each radius group.
  const data = withRadii([...Array(787).fill(.017), ...Array(263).fill(.044), ...Array(46).fill(.175)]);
  const before = structuredClone(data);
  const layer = createCollisionSphereLayer();
  layer.update(data);
  assert.equal(layer.object.children.length, 3);
  let instanceCount = 0;
  let renderedTriangles = 0;
  for (const mesh of layer.object.children) {
    assert.equal(mesh.isInstancedMesh, true);
    instanceCount += mesh.count;
    const radiusM = transform(mesh, 0).scale[0];
    assert.ok(maximumInset(mesh.geometry, radiusM) <= .0002);
    renderedTriangles += mesh.count * (mesh.geometry.getIndex()?.count ?? mesh.geometry.getAttribute('position').count) / 3;
  }
  assert.equal(instanceCount, 1096);
  assert.ok(renderedTriangles < 1_000_000, `too many sphere triangles: ${renderedTriangles}`);
  assert.deepEqual(data, before);
  layer.dispose();
});

test('rebuilds only changed radius groups, preserving opacity and releasing resources when spheres move between groups', () => {
  const layer = createCollisionSphereLayer();
  layer.setOpacity(.42);
  layer.update(withRadii([.01, .03, .10]));
  assert.equal(layer.object.children.length, 3);
  const [small, medium, large] = layer.object.children;
  const smallReleased = watchDisposal(small);
  const mediumReleased = watchDisposal(medium);
  const largeReleased = watchDisposal(large);
  for (const mesh of layer.object.children) assert.equal(mesh.material.opacity, .42);
  layer.setOpacity(.27);
  for (const mesh of layer.object.children) assert.equal(mesh.material.opacity, .27);

  layer.update(withRadii([.025, .04, .12]));
  assert.equal(layer.object.children.length, 2);
  assert.deepEqual(smallReleased, [small, small.geometry, small.material]);
  assert.deepEqual(mediumReleased, [medium, medium.geometry, medium.material]);
  assert.deepEqual(largeReleased, []);
  assert.ok(layer.object.children.includes(large));
  const replacement = layer.object.children.find(mesh => mesh !== large);
  assert.equal(replacement.count, 2);
  near(transform(replacement, 0).scale, [.025, .025, .025]);
  near(transform(replacement, 1).center, [.01, 0, 0]);
  near(transform(large, 0).scale, [.12, .12, .12]);
  for (const mesh of layer.object.children) assert.equal(mesh.material.opacity, .27);

  const replacementReleased = watchDisposal(replacement);
  layer.dispose();
  assert.equal(layer.object.children.length, 0);
  assert.deepEqual(replacementReleased, [replacement, replacement.geometry, replacement.material]);
  assert.deepEqual(largeReleased, [large, large.geometry, large.material]);
  layer.dispose();
  assert.equal(replacementReleased.length, 3);
});

test('increases shared large-sphere detail above 175 mm without changing the supplied radius', () => {
  const layer = createCollisionSphereLayer();
  layer.update(withRadii([.17, .16]));
  const original = layer.object.children[0];
  const released = watchDisposal(original);
  const data = withRadii([.30, .20]);
  layer.update(data);
  assert.equal(layer.object.children.length, 1);
  const mesh = layer.object.children[0];
  assert.notEqual(mesh, original);
  assert.equal(mesh.count, 2);
  assert.ok(maximumInset(mesh.geometry, .30) <= .0002);
  near(transform(mesh, 0).scale, [.30, .30, .30]);
  near(transform(mesh, 1).scale, [.20, .20, .20]);
  assert.deepEqual(released, [original, original.geometry, original.material]);
  layer.update(data);
  assert.equal(layer.object.children[0], mesh, 'reuse the upgraded shared geometry on pose refresh');
  layer.dispose();
});

test('pose refresh reuses the sphere mesh and updates culling bounds; count changes release old resources', () => {
  const layer = createCollisionSphereLayer();
  const data = payload();
  layer.update(data);
  const mesh = layer.object.children[0];
  const oldBounds = mesh.boundingSphere.clone();
  data.spheres[0].center_m = [2, 3, 4];
  layer.update(data);
  assert.equal(layer.object.children[0], mesh);
  near(transform(mesh, 0).center, [2, 3, 4]);
  assert.equal(mesh.boundingSphere.equals(oldBounds), false);
  let disposed = 0;
  mesh.addEventListener('dispose', () => { disposed += 1; });
  mesh.geometry.addEventListener('dispose', () => { disposed += 1; });
  mesh.material.addEventListener('dispose', () => { disposed += 1; });
  data.spheres.pop(); data.sphere_count = 1; data.link_count = 1;
  layer.update(data);
  assert.equal(layer.object.children.length, 1);
  assert.equal(layer.object.children[0].count, 1);
  assert.equal(disposed, 3);
});

test('rejects closed grippers, closure envelopes, missing spheres and invalid geometry without replacing the last view', () => {
  const layer = createCollisionSphereLayer();
  layer.update(payload());
  const mesh = layer.object.children[0];
  const initial = Array.from(mesh.instanceMatrix.array);
  for (const change of [
    data => { data.gripper_state = 'closed'; },
    data => { data.radius_policy = 'closure_envelope'; },
    data => { data.sphere_count = 3; },
    data => { data.spheres[1].radius_m = -1; },
    data => { data.spheres[1].center_m[0] = NaN; },
  ]) {
    const invalid = payload(); change(invalid);
    assert.throws(() => layer.update(invalid));
    assert.equal(layer.object.children[0], mesh);
    assert.deepEqual(Array.from(mesh.instanceMatrix.array), initial);
  }
});

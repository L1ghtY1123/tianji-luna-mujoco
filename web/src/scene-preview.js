import { objectPreviewPosition } from './object-drag.js';
import { Quaternion, Vector3 } from 'three';

function finiteVector(values, positive = false) {
  if (!Array.isArray(values) || values.length !== 3) return null;
  if (values.some(value => value === '' || value === null)) return null;
  const numbers = values.map(Number);
  return numbers.every(value => Number.isFinite(value) && (!positive || value > 0)) ? numbers : null;
}

function fullDimensions(definition) {
  const size = definition.size;
  if (!size) return null;
  if (definition.type === 'box') return size.map(value => value * 2);
  if (definition.type === 'sphere') return [size[0] * 2, size[0] * 2, size[0] * 2];
  if (definition.type === 'cylinder') return [size[0] * 2, size[0] * 2, size[1] * 2];
  return null;
}

function differs(current, values) {
  return values.some((value, index) => Math.abs(value - current[index]) > 1e-7);
}

function binComponent(component, dimensions, thickness) {
  if (!dimensions || !Number.isFinite(thickness) || thickness <= 0) return null;
  const [width, depth, height] = dimensions;
  if (width <= 2 * thickness || depth <= 2 * thickness || height <= thickness) return null;
  const wallHeight = height - thickness;
  if (component === 'bottom') return { position: [0, 0, (-height + thickness) / 2], size: [width, depth, thickness] };
  if (component === 'wall_x_minus' || component === 'wall_x_plus') {
    return { position: [(component === 'wall_x_minus' ? -1 : 1) * (width - thickness) / 2, 0, thickness / 2], size: [thickness, depth, wallHeight] };
  }
  if (component === 'wall_y_minus' || component === 'wall_y_plus') {
    return { position: [0, (component === 'wall_y_minus' ? -1 : 1) * (depth - thickness) / 2, thickness / 2], size: [width - 2 * thickness, thickness, wallHeight] };
  }
  return null;
}

function rotatePosition(position, quaternion) {
  const [w, x, y, z] = quaternion;
  return new Vector3(...position).applyQuaternion(new Quaternion(x, y, z, w)).toArray();
}

// Draft lengths stay in meters. Only editable scene geometry receives preview
// transforms; robot poses always come from the live MuJoCo snapshot.
export function applyScenePreview(mesh, definition, pose, draft, activeDrag) {
  const id = definition.object_id ?? (definition.name?.startsWith('object_geom_') ? definition.name.slice('object_geom_'.length) : null);
  const edit = definition.name === 'table' ? draft?.table : draft?.objects.find(object => object.id === id);
  let position = objectPreviewPosition(id, pose.position, draft, activeDrag);
  let quaternion = pose.quaternion;
  let scale = [1, 1, 1];
  if (edit) {
    if (definition.name === 'table') position = finiteVector(edit.position) ?? pose.position;
    const yaw = Number(edit.yaw_deg);
    if (edit.yaw_deg !== '' && edit.yaw_deg !== null && Number.isFinite(yaw)) {
      const halfAngle = yaw * Math.PI / 360;
      quaternion = [Math.cos(halfAngle), 0, 0, Math.sin(halfAngle)];
    }
    const desired = finiteVector(edit.size, true);
    const original = fullDimensions(definition);
    if (definition.object_kind === 'bin' && definition.local_position_m) {
      const originalOffset = rotatePosition(definition.local_position_m, pose.quaternion);
      const center = pose.position.map((value, index) => value - originalOffset[index]);
      const desiredCenter = objectPreviewPosition(id, center, draft, activeDrag);
      const component = binComponent(definition.component, desired, Number(edit.wall_thickness_m));
      const offset = rotatePosition(component?.position ?? definition.local_position_m, quaternion);
      position = desiredCenter.map((value, index) => value + offset[index]);
      if (component && original?.every(value => value > 0)) scale = component.size.map((value, index) => value / original[index]);
    } else if (desired && original?.every(value => value > 0)) {
      scale = desired.map((value, index) => value / original[index]);
    }
  }
  if (!edit && definition.object_kind === 'bin' && definition.local_position_m && activeDrag?.id === id) {
    const offset = rotatePosition(definition.local_position_m, quaternion);
    position = activeDrag.position.map((value, index) => value + offset[index]);
  }
  if (definition.object_kind === 'marker') {
    const tablePosition = finiteVector(draft?.table?.position);
    const tableSize = finiteVector(draft?.table?.size, true);
    const z = tablePosition && tableSize ? tablePosition[2] + tableSize[2] / 2 + .0001 : pose.position[2];
    position = [position[0], position[1], z];
    scale[2] = 1;
  }
  const [w, x, y, z] = quaternion;
  const rotation = [x, y, z, w];
  const changed = differs(mesh.position.toArray(), position)
    || differs(mesh.quaternion.toArray(), rotation) || differs(mesh.scale.toArray(), scale);
  if (changed) {
    mesh.position.fromArray(position);
    mesh.quaternion.set(x, y, z, w);
    mesh.scale.fromArray(scale);
  }
  return changed;
}

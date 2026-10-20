import { Plane, Vector3 } from 'three';

// Horizontal movement uses the initial plane intersection, not the object's
// center, so grabbing an edge does not snap the object beneath the pointer.
export function createDragProjection({ position, ray, vertical = false, screenY, metersPerPixel }) {
  const start = [...position];
  if (vertical) return (nextRay, nextY) => [start[0], start[1], start[2] + (screenY - nextY) * metersPerPixel];
  const plane = new Plane(new Vector3(0, 0, 1), -start[2]);
  const first = ray.intersectPlane(plane, new Vector3());
  if (!first) return null;
  return nextRay => {
    const point = nextRay.intersectPlane(plane, new Vector3());
    return point ? [start[0] + point.x - first.x, start[1] + point.y - first.y, start[2]] : null;
  };
}

export function objectPreviewPosition(id, livePosition, draft, activeDrag) {
  if (id && id === activeDrag?.id) return activeDrag.position;
  const values = draft?.objects.find(object => object.id === id)?.position;
  const position = values?.map(Number);
  return position?.length === 3 && position.every(Number.isFinite) ? position : livePosition;
}

export function editableObjectPosition(id, geometryPosition, snapshot, draft, activeDrag) {
  const bodyPosition = snapshot.objects?.find(object => object.id === id)?.position ?? geometryPosition;
  return objectPreviewPosition(id, bodyPosition, draft, activeDrag);
}

export function createObjectDrag({ onDragEnd, onDragCancel, onChange = () => {} } = {}) {
  let enabled = false;
  let active = null;

  function cancel() {
    if (!active) return;
    const id = active.id;
    active = null;
    onChange();
    onDragCancel?.(id);
  }

  return {
    get active() { return active; },
    setEnabled(value) {
      enabled = value;
      if (!enabled) cancel();
    },
    begin(id, position, generation) {
      if (!enabled || active) return false;
      active = { id, position: [...position], startPosition: [...position], generation };
      onChange();
      return true;
    },
    move(position) {
      if (!active) return;
      active.position = [...position];
      onChange();
    },
    finish() {
      if (!active) return;
      const current = active;
      if (!current.position.some((value, index) => Math.abs(value - current.startPosition[index]) > 1e-6)) {
        cancel();
        return;
      }
      active = null;
      try {
        onDragEnd?.(current.id, [...current.position]);
      } finally {
        onChange();
      }
    },
    updateSnapshot(snapshot) {
      if (active && (snapshot.generation !== active.generation || !snapshot.objects?.some(object => object.id === active.id))) cancel();
    },
    cancel,
  };
}

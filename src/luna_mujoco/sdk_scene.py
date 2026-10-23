"""Separate project geometry provenance from public cuRobo constructor arguments."""

from copy import copy, deepcopy

import numpy as np


def sdk_voxel_grid(grid):
    """Represent integer cell counts through the public metric-dimension input.

    The pinned loader divides float32 dimensions by float32 voxel size and
    truncates for indexing. Choose the nearest representable dimension that
    cannot round one cell below the declared grid. ESDF samples are unchanged.
    """
    counts = np.rint(np.asarray(grid.dims, dtype=float) / grid.voxel_size).astype(int)
    size = np.float32(grid.voxel_size)
    dimensions = counts.astype(np.float32) * size
    for _ in range(3):
        below = dimensions / size < counts
        if not below.any():
            break
        dimensions[below] = np.nextafter(dimensions[below], np.float32(np.inf))
    if np.any((dimensions / size).astype(int) != counts):
        raise ValueError('voxel_dimension_cannot_preserve_integer_stride')
    result = copy(grid)
    result.dims = dimensions.astype(float).tolist()
    return result

PRIMITIVE_AUDIT_FIELDS = frozenset({"position_source", "settling_envelope_floor_z_m", "action_id", "observation_id"})


def sdk_scene_model(model, *, include_voxel_metadata=False):
    """Copy primitive geometry for the SDK, retaining provenance in the source.

    Native voxel grids are supplied separately. Their project metadata is kept
    only for the wrapper's named-layer handling during an upload. Remove only
    known audit fields: unknown geometry arguments must still fail SDK validation.
    """
    result = {}
    for kind, obstacles in (model or {}).items():
        if kind == "voxel":
            if include_voxel_metadata:
                result[kind] = obstacles
            continue
        result[kind] = {
            name: deepcopy({key: value for key, value in geometry.items()
                            if key not in PRIMITIVE_AUDIT_FIELDS})
            for name, geometry in obstacles.items()
        }
    return result

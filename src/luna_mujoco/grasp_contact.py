"""Pair-specific clearance for the intended target during open-hand descent.

The executing Link3 fingers and Link4 linkage may contact the selected target during descent and closure.
They keep the configured clearance from every other entity.
This policy never applies to approach transit, palms, cameras or the peer arm.
"""
from contextlib import contextmanager

import numpy as np


def gripping_pad_links(arm):
    if arm not in {'A', 'B'}:
        raise ValueError('grasp contact requires one selected arm')
    prefix = 'gripL' if arm == 'A' else 'gripR'
    return {prefix + '_Link3_L', prefix + '_Link3_R'}


def grasp_contact_links(arm):
    contacts = gripping_pad_links(arm)
    prefix = 'gripL' if arm == 'A' else 'gripR'
    return contacts | {prefix + '_Link4_L', prefix + '_Link4_R'}


def combine_grasp_costs(full, environment, sphere_links, arm):
    """Apply the selected-target exemption to the executing contact links."""
    arrays = [np.asarray(value) for value in (full, environment)]
    if (arrays[0].ndim != 3 or any(value.shape != arrays[0].shape for value in arrays)
            or any(not np.isfinite(value).all() for value in arrays)
            or set(sphere_links) != set(range(arrays[0].shape[-1]))):
        raise ValueError('grasp contact costs or sphere mapping are invalid')
    indices = [index for index, link in sphere_links.items() if link in grasp_contact_links(arm)]
    if not indices:
        raise ValueError('grasp contact geometry missing')
    result = arrays[0].copy()
    result[..., indices] = arrays[1][..., indices]
    return result


@contextmanager
def target_disabled_for_environment(runtime, planner):
    """Called with selected target enabled and the runtime GPU lock held.

Temporarily query other entities. The enclosing open-grasp scope owns the
original selected-layer state; restore enabled even when a GPU query fails.
"""
    scene = runtime.geometry_scene
    if scene.get('exclusion_state') != 'selected' or not scene.get('target_id'):
        raise ValueError('grasp contact requires selected target')
    prefix = 'semantic_instance::' + scene['target_id']
    layers = [(kind, name) for kind in ('cuboid', 'mesh', 'voxel')
              for name in runtime.geometry_model.get(kind, {})
              if name == prefix or name.startswith(prefix + '::')]
    if not layers:
        raise ValueError('grasp contact measured target layer missing')
    changed = []
    try:
        for kind, name in layers:
            data = getattr(planner.scene_collision_checker.data,
                           {'cuboid': 'cuboids', 'mesh': 'meshes', 'voxel': 'voxels'}[kind])
            changed.append((data, name))
            data.set_enabled(name, enabled=False, env_idx=0)
        yield
    finally:
        for data, name in reversed(changed):
            data.set_enabled(name, enabled=True, env_idx=0)

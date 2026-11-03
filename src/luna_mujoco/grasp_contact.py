"""Pair-specific clearance for the intended target during open-hand descent.

The two gripping pads may approach the target surface without the ordinary hard clearance.
They must still be nonpenetrating and keep the configured clearance from every other entity.
This policy never applies to approach transit, palms, cameras or the peer arm.
"""
from contextlib import contextmanager

import numpy as np


def gripping_pad_links(arm):
    if arm not in {'A', 'B'}:
        raise ValueError('grasp contact requires one selected arm')
    prefix = 'gripL' if arm == 'A' else 'gripR'
    return {prefix + '_Link3_L', prefix + '_Link3_R'}


def combine_grasp_costs(full, environment, nonpenetration, sphere_links, arm):
    """Keep full clearance except for pad/target pairs with nonpenetration proof."""
    arrays = [np.asarray(value) for value in (full, environment, nonpenetration)]
    if (arrays[0].ndim != 3 or any(value.shape != arrays[0].shape for value in arrays)
            or any(not np.isfinite(value).all() for value in arrays)
            or set(sphere_links) != set(range(arrays[0].shape[-1]))):
        raise ValueError('grasp contact costs or sphere mapping are invalid')
    indices = [index for index, link in sphere_links.items() if link in gripping_pad_links(arm)]
    if not indices:
        raise ValueError('grasp contact pad geometry missing')
    result = arrays[0].copy()
    result[..., indices] = np.maximum(arrays[1][..., indices], arrays[2][..., indices])
    return result


def combine_closure_costs(full, environment, sphere_links, arm):
    """Allow pad contact while retaining target clearance for every other link.

    Closing is an intentional contact motion. The static pregrasp target is
    not a rigid nonpenetration model for the moving/contacting pad pair.
    Environment, peer hand and nonpad target costs remain mandatory.
    """
    return combine_grasp_costs(full, environment, np.zeros_like(full), sphere_links, arm)


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

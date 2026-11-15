"""Current published TSDF geometry shared by grasp sampling and attachment."""

from itertools import product

import numpy as np


def current_tsdf_surface(observation, target):
    surface = target.get('tsdf_surface', {})
    if (surface.get('source') != 'curobo_tsdf_surface' or surface.get('frame') != 'world'
            or any(surface.get(key) != observation.get(key) for key in ('observation_id', 'generation'))):
        raise ValueError('grasp_requires_current_tsdf_surface')
    points = np.asarray(surface.get('points_world_m'), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('grasp_tsdf_surface_empty')
    return points


def tsdf_voxel_corners(observation, target):
    centers = current_tsdf_surface(observation, target)
    size = float(target['tsdf_surface'].get('tsdf_voxel_size_m', 0))
    if not np.isfinite(size) or size <= 0:
        raise ValueError('attachment_requires_tsdf_voxel_size')
    offsets = np.asarray(list(product((-.5, .5), repeat=3))) * size
    # Retain all eight corners of every observed surface voxel. The sphere fit
    # can cover complete cells without inventing a solid object bounding box.
    return (centers[:, None, :] + offsets).reshape(-1, 3)

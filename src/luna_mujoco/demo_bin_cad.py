"""Bounded upright CAD registration from current clean inventory depth.

Known shape is not known pose. A measured floor and two independent sloped
walls locate the shape; the complete point set must then agree with its surface.
"""
import hashlib
from functools import lru_cache
from itertools import combinations, permutations, product

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .cad_assets import _mesh_path, cad_asset

SOURCE = 'known_cad_measured_registration'
PRIOR_SOURCE = 'user_supplied_cad_shape'
SURFACE_TOLERANCE_M = .003


def _candidate_surface_spread(candidates, dimensions):
    """Bound pairwise whole-shape displacement, accounting for half-turn symmetry."""
    corners = np.array(list(product((-1., 1.), repeat=3))) * (np.asarray(dimensions) / 2)
    reference = candidates[0][1]
    bounds = []
    for _, rotation, translation, *_ in candidates:
        aligned = rotation.copy()
        if aligned[:2, 0] @ reference[:2, 0] < 0:
            aligned[:, :2] *= -1
        bounds.append(corners @ aligned.T + translation)
    return max((float(np.max(np.linalg.norm(first-second, axis=1)))
                for first, second in combinations(bounds, 2)), default=0.)


@lru_cache(maxsize=1)
def _geometry():
    asset = cad_asset('blue_box')
    visual = asset['visual']
    path = _mesh_path(visual['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != visual['sha256']:
        raise ValueError('demo_bin_cad_asset_hash_mismatch')
    surface = trimesh.load_mesh(path, process=True)
    if not surface.is_watertight or not surface.is_winding_consistent:
        raise ValueError('demo_bin_cad_requires_closed_oriented_surface')
    # cuRobo supports concave triangle meshes. The 49 convex pieces are for
    # MuJoCo contacts; uploading them as 49 world obstacles repeats BVH queries.
    meshes = {'cad_surface': {'vertices': surface.vertices.tolist(), 'faces': surface.faces.tolist()}}
    # Fixed surface samples are only a conservative nearest-surface residual:
    # every sampled point lies on the CAD, so sampling cannot invent agreement.
    samples, _ = trimesh.sample.sample_surface(surface, 300000, seed=0)
    tree = cKDTree(np.concatenate((surface.vertices, samples)))
    sections = asset['cavity_center_sections_m']
    slope, intercept = np.polyfit([row['z'] for row in sections],
                                 [row['half_size_xy'] for row in sections], 1)
    return asset, meshes, tree, slope, intercept


def _candidate_poses(points, planes, asset, slope, intercept):
    cosine = np.cos(np.deg2rad(3.))
    floors = [p for p in planes if abs(p['normal'][2]) >= cosine]
    walls = [p for p in planes if abs(p['normal'][2]) < np.sin(np.deg2rad(5.))
             and np.ptp(p['points'][:, 2]) >= .03]
    for floor in floors:
        tz = float(np.median(floor['points'][:, 2])-asset['cavity_floor_z_m'])
        for first, second in combinations(walls, 2):
            normals = []
            for wall in (first, second):
                normal = wall['normal'].copy()
                if normal[2] > 0:
                    normal = -normal
                normals.append(normal[:2]/np.linalg.norm(normal[:2]))
            if abs(normals[0] @ normals[1]) > np.sin(np.deg2rad(3.)):
                continue
            for axes in permutations((0, 1)):
                for first_sign in (-1, 1):
                    yaw = float(np.arctan2(normals[0][1], normals[0][0])
                                - axes[0]*np.pi/2 - (np.pi if first_sign < 0 else 0.))
                    # The CAD box has a half-turn symmetry. Canonicalize this
                    # shape-equivalent pose; never use a configured world yaw.
                    yaw = (yaw+np.pi/2) % np.pi-np.pi/2
                    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                                         [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
                    directions = [rotation[:2, axis] for axis in axes]
                    directions = [direction*np.sign(direction @ normal)
                                  for direction, normal in zip(directions, normals)]
                    if any(direction @ normal < cosine for direction, normal in zip(directions, normals)):
                        continue
                    offsets = [float(np.median(wall['points'][:, :2] @ direction
                                              - slope[axis]*wall['points'][:, 2])
                                     + slope[axis]*tz-intercept[axis])
                               for wall, direction, axis in zip((first, second), directions, axes)]
                    translation = np.r_[np.linalg.solve(np.array(directions), offsets), tz]
                    local = (points-translation) @ rotation
                    if np.mean(np.all(abs(local) <= np.array(asset['size_m'])/2+.003, axis=1)) < .85:
                        continue
                    yield rotation, translation, [p['point_count'] for p in (floor, first, second)]


def fit_cad_bin(points, *, original, planes, instance_id, observation_id, generation):
    asset, meshes, tree, slope, intercept = _geometry()
    candidates = []
    for rotation, translation, counts in _candidate_poses(points, planes, asset, slope, intercept):
        if any(np.linalg.norm(translation-old[2]) < .0001
               and np.max(abs(rotation-old[1])) < .0001 for old in candidates):
            continue
        residual = tree.query((points-translation) @ rotation, workers=1)[0]
        agreement = float(np.mean(residual <= SURFACE_TOLERANCE_M))
        if agreement >= .85:
            candidates.append((agreement, rotation, translation, counts, float(np.quantile(residual, .95))))
    if not candidates:
        raise ValueError('demo_bin_cad_plane_evidence_missing_or_inconsistent')
    candidates.sort(key=lambda row: (-row[0], row[4]))
    agreement, rotation, translation, counts, residual95 = candidates[0]
    # Each fit uses a 3 mm surface tolerance. Comparing two fits must allow
    # both measurement errors, including rotational displacement at the rim.
    # Require every pair to agree; do not merge a chain of distant candidates.
    spread = _candidate_surface_spread(candidates, asset['size_m'])
    equivalence_tolerance = 2 * SURFACE_TOLERANCE_M
    if spread > equivalence_tolerance:
        raise ValueError('demo_bin_cad_pose_ambiguous')
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    pose = [*translation.tolist(), np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    half = np.abs(rotation) @ (np.array(asset['size_m'])/2)
    return {'source': SOURCE, 'synthetic_geometry_is_measured': False,
        'instance_id': instance_id, 'observation_id': observation_id, 'generation': generation,
        'asset_id': 'blue_box', 'cad_sha256': asset['source']['sha256'],
        'cad_visual_sha256': asset['visual']['sha256'],
        'part_sha256': {'cad_surface': asset['visual']['sha256']},
        'prior_source': PRIOR_SOURCE, 'outer_dimensions_prior_m': asset['size_m'],
        'measurement_source': 'registered_inventory_points_after_robot_and_surface_filter',
        'measured_point_count': len(points), 'measured_points_sha256': hashlib.sha256(original.tobytes()).hexdigest(),
        'T_world_cad': transform.tolist(), 'floor_height_m': float(translation[2]+asset['cavity_floor_z_m']),
        'rim_height_m': float(translation[2]+asset['size_m'][2]/2),
        'registration': {'algorithm': 'bounded_floor_and_two_sloped_wall_correspondences',
            'degrees_of_freedom': 'upright_yaw_and_xyz', 'yaw_symmetry_rad': float(np.pi),
            'maximum_ransac_planes': 8, 'hypotheses_per_plane': 256, 'plane_sample_limit': 8000,
            'minimum_wall_vertical_span_m': .03, 'surface_sample_count': 300000,
            'surface_tolerance_m': SURFACE_TOLERANCE_M, 'minimum_model_agreement_fraction': .85,
            'candidate_count': len(candidates), 'pose_equivalence_tolerance_m': equivalence_tolerance,
            'maximum_candidate_surface_displacement_m': spread,
            'ambiguity_metric': 'pairwise_whole_shape_displacement_modulo_half_turn',
            'model_agreement_fraction': agreement, 'surface_residual_p95_m': residual95,
            'plane_inlier_counts': counts},
        'meshes': {name: {**model, 'pose': pose.copy()} for name, model in meshes.items()},
        'completed_bounds_world_m': [(translation-half).tolist(), (translation+half).tolist()]}

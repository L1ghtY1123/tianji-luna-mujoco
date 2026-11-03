"""Complete a measured support patch from a currently validated CAD plane."""
from copy import deepcopy

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from tianji_foundationpose.contracts import validate_pose


def complete_placement_reference(target, geometry, tolerance_m, *, support_spec=None):
    reference, cad = target.get('placement_reference'), target.get('cad')
    if not cad:
        return target
    if support_spec:
        if reference and (reference.get('observation_id') != cad.get('observation_id')
                          or reference.get('generation', cad.get('generation')) != cad.get('generation')):
            raise ValueError('cad_placement_reference_binding_mismatch')
        # A registered container's declared inner floor defines its region even
        # when an object hides part of that floor after release. A visible patch
        # is measurement evidence, not a new, smaller destination boundary.
        registered = catalog_placement_reference(target, geometry, support_spec)
        if registered.get('placement_geometry_issue'):
            return registered
        if registered is not target:
            if reference:
                registered['placement_reference'].update(
                    observed_floor_z_m=reference['floor_z_m'],
                    observed_region_polygon_world_xy_m=deepcopy(reference['region_polygon_world_xy_m']),
                    observed_region_measurement=deepcopy(reference.get('region_measurement')))
            return registered
    if not reference:
        return target
    if (cad.get('pose_validation', {}).get('accepted', cad.get('pose_validation', {}).get('status') == 'validated') is not True
            or cad.get('instance_id') != target['id']
            or reference.get('observation_id') != cad.get('observation_id')
            or reference.get('generation', cad.get('generation')) != cad.get('generation')
            or cad.get('mesh_sha256') != geometry.get('mesh_sha256')):
        raise ValueError('cad_placement_reference_binding_mismatch')
    floor = float(reference['floor_z_m'])
    anchor = np.asarray(reference['position_world_m'], dtype=float)
    observed = np.asarray(reference['region_polygon_world_xy_m'], dtype=float)
    if not np.isfinite([floor, tolerance_m]).all() or tolerance_m <= 0:
        raise ValueError('cad_placement_requires_finite_measured_height')
    pose = validate_pose(cad['T_world_object'])
    mesh = trimesh.Trimesh(vertices=geometry['vertices_m'], faces=geometry['faces'], process=True)
    world = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
    normals = mesh.facets_normal @ pose[:3, :3].T
    observed_area = ConvexHull(observed).volume
    candidates = []
    for index in np.flatnonzero(normals[:, 2] > .99):
        faces = mesh.facets[index]
        vertices = world[np.unique(mesh.faces[faces])]
        if np.max(np.abs(vertices[:, 2] - floor)) > tolerance_m:
            continue
        hull = ConvexHull(vertices[:, :2])
        # A convex hull must not bridge a hole, recess or disconnected plane.
        projected_area = float(mesh.facets_area[index] * normals[index, 2])
        if not np.isclose(hull.volume, projected_area, rtol=1e-5, atol=1e-9):
            continue
        if (hull.volume < observed_area
                or np.any(hull.equations[:, :2] @ anchor[:2] + hull.equations[:, 2] > 1e-8)):
            continue
        candidates.append((hull.volume, index, vertices[hull.vertices, :2]))
    if not candidates:
        return target
    area, index, polygon = max(candidates, key=lambda row: row[0])
    result = deepcopy(target)
    completed = result['placement_reference']
    completed['observed_region_polygon_world_xy_m'] = deepcopy(reference['region_polygon_world_xy_m'])
    completed['observed_region_measurement'] = deepcopy(reference.get('region_measurement'))
    completed['region_polygon_world_xy_m'] = polygon.tolist()
    completed['source'] = 'validated_cad_plane_with_measured_height'
    completed['region_measurement'] = {'method': 'validated_cad_convex_support_facet',
        'polygon_world_xy_m': polygon.tolist(), 'area_m2': float(area),
        'face_count': len(mesh.facets[index]), 'height_tolerance_m': float(tolerance_m),
        'asset_id': cad['asset_id'], 'mesh_sha256': cad['mesh_sha256'],
        'observation_id': cad['observation_id'], 'generation': cad['generation'],
        'T_world_object': deepcopy(cad['T_world_object']),
        'surface_source': 'registered_model_prior', 'floor_height_source': reference['source'],
        'collision_checked': False, 'unseen_area_observed': False}
    return result


def catalog_placement_reference(target, geometry, spec):
    """Use an asset's declared support facet, never a fabricated visible floor."""
    cad = target['cad']
    provenance = target.get('observation_provenance', {})
    observation_id = target.get('observation_id', provenance.get('observation_id'))
    generation = target.get('generation', provenance.get('generation'))
    if (cad.get('pose_validation', {}).get('accepted', cad.get('pose_validation', {}).get('status') == 'validated') is not True
            or cad.get('instance_id') != target['id']
            or not observation_id or generation is None
            or cad.get('observation_id') != observation_id
            or cad.get('generation') != generation
            or cad.get('mesh_sha256') != geometry.get('mesh_sha256')
            or spec.get('mesh_sha256') != geometry.get('mesh_sha256')):
        raise ValueError('cad_placement_reference_binding_mismatch')
    if spec.get('kind') != 'interior_floor' or target.get('kind') != 'bin':
        return target
    anchor = np.asarray(spec['point_canonical_m'], dtype=float)
    normal = np.asarray(spec['normal_canonical'], dtype=float)
    if (anchor.shape != (3,) or normal.shape != (3,) or not np.isfinite([anchor, normal]).all()
            or not np.isclose(np.linalg.norm(normal), 1., atol=1e-9, rtol=0)):
        raise ValueError('cad_catalog_support_plane_invalid')
    pose = validate_pose(cad['T_world_object'])
    normal_world = pose[:3, :3] @ normal
    if normal_world[2] <= .99:
        result = deepcopy(target)
        result.pop('placement_reference', None)
        result['placement_geometry_issue'] = {'reason': 'cad_support_not_horizontal',
            'normal_world': normal_world.tolist(), 'minimum_up_component': .99,
            'observation_id': cad['observation_id'], 'generation': cad['generation'],
            'meaning': 'The accepted CAD pose has no approximately horizontal declared floor. '
                'This is a placement capability limit, not a mask/depth validation rejection. '
                'Rebuilding this same pose cannot produce a horizontal floor.'}
        return result
    mesh = trimesh.Trimesh(vertices=geometry['vertices_m'], faces=geometry['faces'], process=True)
    world_anchor = anchor @ pose[:3, :3].T + pose[:3, 3]
    candidates = []
    for index in np.flatnonzero(mesh.facets_normal @ normal > 1. - 1e-6):
        faces = mesh.facets[index]
        vertices = mesh.vertices[np.unique(mesh.faces[faces])]
        if np.max(np.abs((vertices - anchor) @ normal)) > 1e-6:
            continue
        world = vertices @ pose[:3, :3].T + pose[:3, 3]
        hull = ConvexHull(world[:, :2])
        projected_area = mesh.facets_area[index] * (pose[:3, :3] @ mesh.facets_normal[index])[2]
        if (not np.isclose(hull.volume, projected_area, rtol=1e-5, atol=1e-9)
                or np.any(hull.equations[:, :2] @ world_anchor[:2] + hull.equations[:, 2] > 1e-8)):
            continue
        candidates.append((hull.volume, index, world[hull.vertices], len(faces)))
    if not candidates:
        return target
    area, index, boundary, face_count = max(candidates, key=lambda row: row[0])
    # The existing placement interface uses a horizontal Z bound; take the
    # highest point of the slightly tilted CAD floor, then retain collision IK.
    floor = float(boundary[:, 2].max())
    polygon = boundary[:, :2].tolist()
    result = deepcopy(target)
    result['placement_reference'] = {
        'source': 'validated_cad_support_plane', 'support_entity_id': target['id'],
        'observation_id': cad['observation_id'], 'generation': cad['generation'],
        'floor_z_m': floor, 'position_world_m': [*boundary[:, :2].mean(0).tolist(), floor],
        'region_polygon_world_xy_m': polygon,
        'region_measurement': {'method': 'registered_catalog_support_facet',
            'surface_source': 'registered_model_prior', 'floor_height_source': 'current_registered_cad',
            'height_rule': 'maximum_world_z_of_support_facet',
            'polygon_world_xy_m': polygon, 'area_m2': float(area), 'facet_index': int(index),
            'face_count': face_count, 'catalog_support': deepcopy(spec), 'asset_id': cad['asset_id'],
            'mesh_sha256': cad['mesh_sha256'], 'T_world_object': deepcopy(cad['T_world_object']),
            'observation_id': cad['observation_id'], 'generation': cad['generation'],
            'collision_checked': False, 'unseen_area_observed': False}}
    return result

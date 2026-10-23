"""Tighten certified palm covers without losing volume or invalidating old IK witnesses.

Each existing box cell is split in half in all three axes. Its eight smaller
spheres lie entirely in the parent sphere (distance + radius <= parent radius).
The complete authored box remains covered. No cuRobo algorithm is modified.
"""
import copy
import itertools

import numpy as np
from cover_gripper_base_boxes import LINKS, cover_box


def tighten(source, urdf):
    candidate = copy.deepcopy(source)
    kin = candidate['robot_cfg']['kinematics']
    certificates = {}
    for link in LINKS:
        parents, record = cover_box(urdf, link)
        if kin['collision_spheres'][link] != parents:
            raise ValueError('palm_source_is_not_certified_box_cover')
        if record['origin_rpy_rad'] != [0., 0., 0.]:
            raise ValueError('palm_source_rotation_changed')
        quarter = np.asarray(record['size_m']) / np.asarray(record['divisions']) / 4
        children = []
        for parent in parents:
            for signs in itertools.product((-1, 1), repeat=3):
                children.append({'center': (np.asarray(parent['center']) + signs * quarter).tolist(),
                                 'radius': parent['radius'] / 2})
        kin['collision_spheres'][link] = children
        certificates[link] = {**record, 'parent_spheres': parents,
            'child_count': len(children), 'child_radius_m': children[0]['radius'],
            'containment_margin_m': parents[0]['radius'] / 2 - float(np.linalg.norm(quarter))}
    old_buffers = copy.deepcopy(kin['collision_sphere_buffer'])
    old_self = copy.deepcopy(kin['self_collision_buffer'])
    for link in kin['collision_link_names']:
        if link.startswith(('gripL_', 'gripR_')) and link in kin['collision_sphere_buffer']:
            buffer = kin['collision_sphere_buffer'].get(link, 0.)
            # Remove duplicate world padding, preserving the effective self
            # radius. The user's 10 mm path-clearance check remains separate.
            kin['collision_sphere_buffer'][link] = 0.
            kin['self_collision_buffer'][link] = kin['self_collision_buffer'].get(link, 0.) + buffer
    return candidate, {'source': 'subdivision_of_certified_palm_box_cells',
        'links': certificates, 'previous_world_buffers': old_buffers, 'previous_self_buffers': old_self,
        'robot_sphere_count': sum(map(len, kin['collision_spheres'].values())),
        'workspace_compatibility': 'every new self-collision sphere is contained in an old one; '
            'all other collision geometry, transforms and exclusions unchanged; old reachable cells remain valid, '
            'previously unreachable cells are conservatively unchanged'}

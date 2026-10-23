import itertools
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parents[1]


def test_palm_envelope_is_symmetric_covers_cells_and_does_not_expand_old_self_geometry():
    kin = yaml.safe_load((ROOT/'resources/curobo/marvin_gui_dual.yml').read_bytes())['robot_cfg']['kinematics']
    path = ROOT/'resources/curobo/gripper-tight-cover-20261023.json'
    assert path.exists(), 'a reproducible coverage and containment certificate is required'
    certificate = json.loads(path.read_text(encoding='utf-8'))
    for link, record in certificate['links'].items():
        spheres = kin['collision_spheres'][link]
        centers = np.array([s['center'] for s in spheres])
        radii = np.array([s['radius'] for s in spheres])
        parents = record['parent_spheres']
        distances = np.linalg.norm(centers[:, None] - np.array([s['center'] for s in parents]), axis=-1)
        margins = np.array([s['radius'] for s in parents]) - distances - radii[:, None]
        assert (margins.max(axis=1) >= -1e-12).all()
        assert len(spheres) == 96 and radii.max() < .018
        size, origin = np.array(record['size_m']), np.array(record['origin_xyz_m'])
        divisions = np.array(record['divisions']) * 2
        step = size / divisions
        cell_centers = np.array(list(itertools.product(*(np.linspace(-s/2+d/2, s/2-d/2, n)
            for s, d, n in zip(size, step, divisions))))) + origin
        assert np.linalg.norm(cell_centers[:, None]-centers, axis=-1).min(axis=1).max() < 1e-10
        assert np.linalg.norm(step/2) <= radii.min()
        vertices = np.array(list(itertools.product(*(np.linspace(-s/2, s/2, n+1)
            for s, n in zip(size, divisions))))) + origin
        gaps = np.linalg.norm(vertices[:, None] - centers, axis=-1) - radii
        assert gaps.min(axis=1).max() <= 1e-10
        assert kin['collision_sphere_buffer'][link] == kin['self_collision_buffer'][link] == 0
        excess = np.maximum(origin-size/2-(centers-radii[:, None]).min(axis=0),
                            (centers+radii[:, None]).max(axis=0)-(origin+size/2))
        assert excess.max() < .01
    assert kin['collision_spheres']['gripL_Base'] == kin['collision_spheres']['gripR_Base']


def test_all_same_hand_contact_links_remain_allowed():
    from luna_mujoco.carrying import carrying_touch_links

    for arm, hand in [('A', 'L'), ('B', 'R')]:
        links = carrying_touch_links(arm)
        assert set(links) == {f'grip{hand}_Base', *[f'grip{hand}_Link{n}_{s}' for n in range(1,5) for s in 'LR']}

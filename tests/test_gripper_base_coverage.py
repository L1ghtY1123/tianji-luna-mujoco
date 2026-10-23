"""The planner must cover the actual authored palm primitive, not only its CAD."""
import copy
import hashlib
import itertools
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
CONFIG = ROOT/'resources/curobo/marvin_gui_dual.yml'
URDF = ROOT/'assets/marvin_urdf/marvin_with_grippers.urdf'
FIXTURE = json.loads((ROOT/'tests/fixtures/gripper_base_first_contact.json').read_bytes())


def box(link):
    node = ET.parse(URDF).getroot().find(f"link[@name='{link}']/collision")
    size = np.fromstring(node.find('geometry/box').get('size'), sep=' ')
    center = np.fromstring(node.find('origin').get('xyz'), sep=' ')
    rotation = Rotation.from_euler('xyz', np.fromstring(node.find('origin').get('rpy'), sep=' ')).as_matrix()
    return size, center, rotation


def margins(points, spheres):
    centers = np.asarray([row['center'] for row in spheres])
    radii = np.asarray([row['radius'] for row in spheres])
    return np.max(radii[None, :]-np.linalg.norm(points[:, None]-centers[None, :], axis=-1), axis=1)


@pytest.mark.parametrize('link', ['gripL_Base', 'gripR_Base'])
def test_raw_planning_spheres_cover_authored_primitive_surface_and_volume(link):
    kin = yaml.safe_load(CONFIG.read_bytes())['robot_cfg']['kinematics']
    size, center, rotation = box(link)
    points = np.array(list(itertools.product(*(np.linspace(-s/2, s/2, 25) for s in size))))
    points = points@rotation.T+center
    assert margins(points, kin['collision_spheres'][link]).min() >= -1e-9


@pytest.mark.parametrize('link', ['gripL_Base', 'gripR_Base'])
def test_frozen_contact_corner_is_covered_without_relying_on_world_buffer(link):
    point = np.asarray([FIXTURE['contact_base_local_m']])
    previous = FIXTURE['previous_base_spheres'][link]
    assert margins(point, previous)[0] < -.03
    current = yaml.safe_load(CONFIG.read_bytes())['robot_cfg']['kinematics']['collision_spheres'][link]
    assert margins(point, current)[0] >= 0


def test_only_two_base_lists_change_with_explicit_robot_budget_and_same_collision_policy():
    current = yaml.safe_load(CONFIG.read_bytes())
    other = copy.deepcopy(current)
    spheres = other['robot_cfg']['kinematics']['collision_spheres']
    assert FIXTURE['source_sphere_count'] == 349
    assert sum(map(len, spheres.values())) == 365
    for link in FIXTURE['previous_base_spheres']:
        assert len(spheres.pop(link)) == 12
    digest = hashlib.sha256(json.dumps(other, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == FIXTURE['unchanged_config_without_two_base_sphere_lists_sha256']


def test_generator_certifies_all_cell_vertices_and_reproduces_only_base_lists():
    from cover_gripper_base_boxes import build_candidate

    config = yaml.safe_load(CONFIG.read_bytes())
    previous = copy.deepcopy(config)
    previous['robot_cfg']['kinematics']['collision_spheres'].update(FIXTURE['previous_base_spheres'])
    actual, certificate = build_candidate(previous, URDF)
    assert actual == config
    assert certificate['robot_sphere_count'] == certificate['maximum_robot_spheres'] == 365
    for link, evidence in certificate['links'].items():
        size, center, rotation = box(link)
        assert evidence['divisions'] == [2, 2, 3]
        assert len(evidence['cells']) == 12
        volume = 0.
        for cell, sphere in zip(evidence['cells'], actual['robot_cfg']['kinematics']['collision_spheres'][link]):
            lower = np.asarray(cell['minimum_box_frame_m'])
            upper = np.asarray(cell['maximum_box_frame_m'])
            assert np.all(lower >= -size/2-1e-12) and np.all(upper <= size/2+1e-12)
            volume += np.prod(upper-lower)
            vertices = np.array(list(itertools.product(*zip(lower, upper)))) @ rotation.T + center
            assert margins(vertices, [sphere]).min() >= 0
        assert volume == pytest.approx(np.prod(size), abs=1e-14)


def test_generator_rejects_missing_primitive_and_unrelated_budget_growth(tmp_path):
    from cover_gripper_base_boxes import build_candidate, cover_box

    path = tmp_path/'invalid.urdf'
    path.write_text('<robot><link name="gripL_Base"><collision><geometry><sphere radius=".05"/>'
                    '</geometry></collision></link></robot>')
    with pytest.raises(ValueError, match='collision box'):
        cover_box(path, 'gripL_Base')
    config = yaml.safe_load(CONFIG.read_bytes())
    config['robot_cfg']['kinematics']['collision_spheres']['Link_Base'].append(
        {'center': [0., 0., 0.], 'radius': .01})
    with pytest.raises(ValueError, match='365-sphere'):
        build_candidate(config, URDF)

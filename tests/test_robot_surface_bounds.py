"""Broad-phase pruning must preserve first-return ownership and ambiguity."""
from copy import deepcopy

import mujoco
import numpy as np
import pytest
from test_own_robot_surface import ray_fixture

from luna_mujoco import robot_surface as surface


@pytest.mark.parametrize('rotation', [0, 37, 83])
@pytest.mark.parametrize('offset', [-.0025, 0, .0025, .01])
def test_rotated_visual_bounds_preserve_first_return_evidence(monkeypatch, rotation, offset):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
        <geom type="box" pos="0 0 1" euler="0 {rotation} 0" size=".15 .1 .03"/>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _, _, view = ray_fixture()
    geometry = np.full(1, -1, np.int32)
    hit = mujoco.mj_ray(model, data, np.zeros(3), np.array([0., 0., 1.]),
                        surface.RAY_GROUP, True, -1, geometry)
    view['depth_m'][:] = hit + offset
    pixels = np.column_stack(np.nonzero(view['valid_depth']))
    bounded = surface.first_surface_matches(model, data, view, pixels, visual_indices=[0])
    monkeypatch.setattr(surface, '_near_visual_bounds', lambda *args: np.ones(len(pixels), bool))
    reference = surface.first_surface_matches(model, data, view, pixels, visual_indices=[0])
    for key in ('matched', 'unresolved_near_robot', 'contact_surface_ambiguous'):
        np.testing.assert_array_equal(bounded[key], reference[key])
    for key in ('distance_m', 'geom_ids', 'pixel_offsets_rc'):
        np.testing.assert_array_equal(bounded[key][bounded['matched']], reference[key][reference['matched']])


def test_outside_padded_box_skips_rays_but_retains_original_ambiguity(monkeypatch):
    model, data, view = ray_fixture(front=False)
    view = deepcopy(view)
    view['depth_m'][:] = .99  # 10 mm before the box, but inside its enclosing sphere.
    def unexpected(*args):
        raise AssertionError('A point beyond every padded visual bound cannot match a mesh return')
    monkeypatch.setattr(mujoco, 'mj_multiRay', unexpected)
    result = surface.first_surface_matches(model, data, view, [[1, 1]], visual_indices=[0])
    assert result['matched'].tolist() == [False]
    assert result['unresolved_near_robot'].tolist() == [True]

"""Regression: a tilted flange must not be aimed at the finger-front target."""
import json
from pathlib import Path

import numpy as np

from luna_mujoco.demo_unpack_workflow import right_bin_transport_goal
from luna_mujoco.geometry import front_tcp_pose


def test_tilted_drop_targets_finger_front_and_keeps_lift_attitude():
    value = json.loads((Path(__file__).parent / 'fixtures/cad_drop_v17.json').read_text())
    # Actual rotation at the failed black grasp, 2026-10-23. Small held body
    # centred at the calibrated finger front, not at the wrist flange.
    rotation = np.array([[.475574035329721, -.610334435992852, .633499181658110],
                         [-.878803856163902, -.361689074652995, .311263225691887],
                         [.039155067509378, -.704750232009859, -.708374188667536]])
    value['lift_tcp'] = np.eye(4)
    value['lift_tcp'][:3, :3] = rotation
    value['holding']['arm'] = 'B'
    value['holding']['points_tool_m'] = [[x, y, z] for x in [-.02, .02]
                                        for y in [-.025, .025] for z in [.165, .205]]
    goal, evidence = right_bin_transport_goal(value['lift_tcp'], value['holding'],
        value['destination'], value['scene'], value['observation'])
    front = front_tcp_pose(goal, 'B')
    polygon = np.array(evidence['region_polygon_world_xy_m'])
    center = (polygon.min(0) + polygon.max(0)) / 2
    center[0] -= .25 * np.ptp(polygon[:, 0])
    np.testing.assert_allclose(front[:2, 3], center)
    np.testing.assert_array_equal(goal[:3, :3], rotation)
    assert 'carried_projection' not in evidence
    assert np.linalg.norm(goal[:2, 3] - front[:2, 3]) > .1

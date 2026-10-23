"""Reject a source grasp that cannot preserve its attitude above the bin."""
import json
from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.demo_grasp import filter_transport_reach, transport_tcp_height


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_loose_height_never_lowers_the_carried_bottom_below_bin_clearance(arm):
    assert transport_tcp_height(np.eye(3), -.2, 1.1, arm) == pytest.approx(1.4)


def test_v21_loose_transport_height_keeps_reachable_grasp_instead_of_fixed_z_rejection():
    frozen = json.loads((Path(__file__).parent / 'fixtures/v21_transport_height.json').read_text())
    accepted, evidence = filter_transport_reach([frozen['candidate']], frozen['points_world_m'], 'B', frozen)
    assert frozen['old_reach']['nominal_transport_wrist_distance_m'] > evidence['reach_screening_limit_m']
    assert accepted == [frozen['candidate']]
    row = evidence['candidate_evidence'][0]
    assert row['nominal_transport_wrist_distance_m'] == pytest.approx(row['horizontal_wrist_distance_m'])
    assert row['nominal_mount_world_m'][2] > frozen['old_reach']['nominal_mount_world_m'][2]


def test_v17_grasp_cannot_reach_bin_even_at_best_height():
    frozen = json.loads((Path(__file__).parent / 'fixtures/cad_drop_v17.json').read_text())
    pose = np.asarray(frozen['lift_tcp'])
    points = np.asarray(frozen['holding']['points_tool_m']) @ pose[:3, :3].T + pose[:3, 3]
    request = {'object_id': 'black', 'transport_region': {
        'center_world_xy_m': [.4809930434711731, -.6250034124693048],
        'occupied_top_z_m': .663499968, 'observation_id': 'current'}}
    candidate = {'index': 7, 'score': .9, 'T_world_tcp': pose.tolist()}
    accepted, evidence = filter_transport_reach([candidate], points, 'B', request)
    assert accepted == []
    assert evidence['candidate_evidence'][0]['horizontal_wrist_distance_m'] > evidence['reach_screening_limit_m'] + .01
    assert evidence['reach_screening_limit_m'] < .598


@pytest.mark.parametrize('arm,sign', [('A', 1), ('B', -1)])
def test_transport_screen_preserves_candidates_and_order(arm, sign):
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = [.30, sign * .25, .82]
    candidates = [{'index': i, 'score': i / 10, 'T_world_tcp': pose.tolist()} for i in (3, 2)]
    request = {'transport_region': {'center_world_xy_m': [.30, sign * .35],
        'occupied_top_z_m': .66, 'observation_id': 'current'}}
    accepted, evidence = filter_transport_reach(candidates, np.array([[.30, sign * .25, .61]]), arm, request)
    assert accepted == candidates
    assert evidence['accepted_count'] == 2
    assert evidence['screening_scope'] == 'nominal_transport_wrist_reach'

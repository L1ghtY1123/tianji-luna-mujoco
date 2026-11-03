from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.grasp_contact_prediction import bounded_contact_prediction


@pytest.mark.parametrize('label,expected', [('held', True), ('empty', False)])
def test_frozen_observed_contact_distinguishes_real_held_and_empty_trials(label, expected):
    # Observed RGBD and chosen TCP from actual trials 57/58, without object CAD or simulation truth.
    with np.load(Path(__file__).parent / 'fixtures/bounded_pad_contact_57_58.npz') as fixture:
        points, pose, limit = (fixture[label + suffix] for suffix in ('_points', '_pose', '_limit'))
        from luna_mujoco.geometry import front_tcp_pose
        result = bounded_contact_prediction(points, front_tcp_pose(pose, 'B'), 'B', float(limit), .02)
    assert result['opposing_contact_predicted'] is expected
    assert result['contact_probe_rad'] == pytest.approx(float(limit) - .016)
    assert result['is_holding_evidence'] is False
    if not expected:
        assert max(p['minimum_signed_gap_m'] for p in result['pads']) > .007


def test_points_outside_finite_finger_faces_cannot_prove_pinch_contact():
    with np.load(Path(__file__).parent / 'fixtures/bounded_pad_contact_57_58.npz') as fixture:
        points, pose = fixture['held_points'], fixture['held_pose']
        moved = points + pose[:3, 2] * .5
        result = bounded_contact_prediction(moved, pose, 'B', float(fixture['held_limit']), .02)
    assert result['opposing_contact_predicted'] is False
    assert all(row['observed_patch_point_count'] == 0 for row in result['pads'])

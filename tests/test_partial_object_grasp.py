"""Whole-object width is not the width of a local CAD grasp contact."""
from pathlib import Path

import numpy as np

from luna_mujoco.demo_gripper_quality import predict_gripper_compatibility


def test_reachable_cad_grasp_is_not_rejected_by_ungrasped_wide_base():
    with np.load(Path(__file__).parent / 'fixtures/v23_partial_grasp.npz') as data:
        result = predict_gripper_compatibility(data['points'], data['pose'],
            arm='A', uncertainty_m=.005, margin_m=.001)
    assert result['accepted']
    assert result['width_prediction_status'] == 'whole_cloud_width_outside_range'
    assert result['q_width_rad'] is None
    assert result['minimum_padded_noncontact_gap_m'] >= 0
    assert result['point_count'] == 4653

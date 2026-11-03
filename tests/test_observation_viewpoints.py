import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


def test_viewpoint_choices_are_distinct_joint_goals_not_recaptures():
    from luna_mujoco.observation_motion import observation_viewpoint_poses
    config = json.loads((Path(__file__).parents[1] / 'resources/demo_observation_poses.json').read_text())
    settings = {'observation': config['joints_rad']}
    before = deepcopy(settings)
    normal = observation_viewpoint_poses(settings, 'configured')
    positive = observation_viewpoint_poses(settings, 'wrist_tilt_positive')
    negative = observation_viewpoint_poses(settings, 'wrist_tilt_negative')
    for arm in ('A', 'B'):
        assert positive[arm][5] - normal[arm][5] == pytest.approx(np.deg2rad(12))
        assert negative[arm][5] - normal[arm][5] == pytest.approx(-np.deg2rad(12))
        assert positive[arm][:5] == normal[arm][:5]
        assert positive[arm][6] == normal[arm][6]
    assert settings == before
    with pytest.raises(ValueError, match='unknown_observation_viewpoint'):
        observation_viewpoint_poses(settings, 'invented')
    settings['observation']['A'][5] = -1.39
    with pytest.raises(ValueError, match='observation_joint_out_of_range'):
        observation_viewpoint_poses(settings, 'wrist_tilt_negative')


def test_nonconfigured_viewpoint_does_not_claim_required_return_pose(tmp_path):
    from test_unpack_final_return import controller

    from luna_mujoco.skills import argument_contract
    runner, _, _ = controller(tmp_path)
    contract = runner.registry.describe()['move_to_observe']
    regular = argument_contract(contract, {'viewpoint': 'configured'})
    alternative = argument_contract(contract, {'viewpoint': 'wrist_tilt_positive'})
    assert regular['planning_effects']['observation_pose_ready'] is True
    assert alternative['planning_effects']['observation_pose_ready'] is False
    assert alternative['requires']['scene_available'] is True
    assert alternative['requires']['grasp_held'] is False

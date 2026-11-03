from unittest.mock import Mock

import pytest

from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.public_planner import PublicPlanner


def test_contact_links_are_explicit_public_arguments_only_in_grasp_scope():
    sdk = Mock()
    planner = PublicPlanner(sdk, PlanningSettings(), batch=True)
    pads = ['gripL_Link3_L', 'gripL_Link3_R']
    with planner.grasp_contacts(pads):
        planner.plan_grasp('goal', 'start', disable_collision_links=[])
    assert sdk.plan_grasp.call_args.kwargs['disable_collision_links'] == pads
    planner.plan_grasp('place', 'start', disable_collision_links=[])
    assert sdk.plan_grasp.call_args.kwargs['disable_collision_links'] == []


def test_failed_grasp_restores_contact_scope():
    sdk = Mock()
    sdk.plan_grasp.side_effect = RuntimeError('GPU failure')
    planner = PublicPlanner(sdk, PlanningSettings())
    with (
        pytest.raises(RuntimeError, match='GPU failure'),
        planner.grasp_contacts(['gripR_Link3_L', 'gripR_Link3_R']),
    ):
        planner.plan_grasp(disable_collision_links=[])
    sdk.plan_grasp.side_effect = None
    planner.plan_grasp(disable_collision_links=[])
    assert sdk.plan_grasp.call_args.kwargs['disable_collision_links'] == []

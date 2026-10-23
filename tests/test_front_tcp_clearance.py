import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_region_placement import region_inputs

from luna_mujoco.placement import release_candidates, release_endpoint_evidence
from luna_mujoco.planning_settings import PlanningSettings


def tilted_goal():
    target, destination, holding, spheres = region_inputs()
    grasp = np.asarray(holding['grasp_support_reference']['T_world_tcp_at_grasp'])
    grasp[:3, :3] = Rotation.from_euler('xy', [25, 60], degrees=True).as_matrix()
    holding['grasp_support_reference']['T_world_tcp_at_grasp'] = grasp.tolist()
    goals = release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01)
    return holding, goals


def test_all_world_yaws_position_the_front_tcp_and_keep_hard_clearance():
    holding, goals = tilted_goal()
    assert len(goals) == 18
    for goal in goals:
        mount = np.asarray(goal['T_world_tcp'])
        front = mount @ [0, 0, .185, 1]
        np.testing.assert_allclose(front[:2], goal['release_reference']['target_reference_world_xy_m'], atol=1e-12)
        evidence = release_endpoint_evidence(goal, holding, mount)
        assert evidence['attachment_floor_gap_m'] == pytest.approx(.012)
        assert evidence['at_release_goal']


@pytest.mark.parametrize('gap,accepted', [(.00868, False), (.0099, False), (.0101, True), (.012, True), (.0139, True), (.0141, False)])
def test_actual_floor_gap_never_borrows_from_hard_clearance(gap, accepted):
    holding, goals = tilted_goal()
    goal = goals[0]
    actual = np.asarray(goal['T_world_tcp'])
    evidence = release_endpoint_evidence(goal, holding, actual)
    actual[2, 3] += gap - evidence['attachment_floor_gap_m']
    assert release_endpoint_evidence(goal, holding, actual)['at_release_goal'] is accepted


def test_release_setting_preserves_hard_clearance_without_extra_tracking_margin():
    assert PlanningSettings().place_release_height_m == pytest.approx(.01)
    assert PlanningSettings(trajectory_collision_clearance_m=.015).place_release_height_m == pytest.approx(.015)
    assert PlanningSettings(trajectory_collision_clearance_m=.5,
        optimizer_collision_activation_distance_m=.5).place_release_height_m == pytest.approx(.5)

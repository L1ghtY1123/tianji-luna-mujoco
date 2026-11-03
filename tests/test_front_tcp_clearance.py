import pytest

from luna_mujoco.planning_settings import PlanningSettings


def test_release_setting_preserves_hard_clearance_without_extra_tracking_margin():
    assert PlanningSettings().place_release_height_m == pytest.approx(0.)
    assert PlanningSettings(trajectory_collision_clearance_m=.015).place_release_height_m == pytest.approx(0.)
    assert PlanningSettings(trajectory_collision_clearance_m=.5,
        optimizer_collision_activation_distance_m=.5).place_release_height_m == pytest.approx(0.)

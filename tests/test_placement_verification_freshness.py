
import pytest
from test_placement_zones import intent, observation

from luna_mujoco.robot_skills.placement_zone import perceive_placement_zone


def test_center_requirement_cannot_degrade_to_whole_surface_preference():
    with pytest.raises(ValueError, match='placement_center_extent_required'):
        perceive_placement_zone(observation(), 'area', intent(zone='center'))



from unittest.mock import Mock, patch

import numpy as np

from luna_mujoco.demo_perception import DemoPerception
from luna_mujoco.perception import _remove_own_robot_surface


def test_cad_inventory_mode_does_not_construct_cpu_robot_ray_filter():
    with patch('luna_mujoco.perception.own_robot_surface_pixels', side_effect=AssertionError('CPU ray')):
        perception = DemoPerception(detector=Mock(), visual_inventory=Mock(), cad_resolver=Mock())
        assert perception.robot_surface_classifier is None
        groups = [[{'id': 'unchanged-sam-subject'}]]
        actual, evidence = _remove_own_robot_surface(groups, {}, {}, perception.robot_surface_classifier)
        assert actual is groups
        assert evidence['status'] == 'not_applied'
        mask = np.ones((4, 4), bool)
        table_evidence = perception.table_robot_surface_classifier({}, {'3': mask})
        assert table_evidence['status'] == 'not_applied'
        assert table_evidence['masks'] == {}
        assert mask.all()


def test_explicit_offline_robot_filter_still_uses_injected_classifier():
    classifier = Mock(return_value={'status': 'measured', 'masks': {}})
    perception = DemoPerception(detector=Mock(), visual_inventory=Mock(), cad_resolver=Mock(),
                                robot_surface_classifier=classifier)
    perception.table_robot_surface_classifier({}, {})
    classifier.assert_called_once_with({}, {})

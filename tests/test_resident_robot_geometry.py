from copy import deepcopy
from types import SimpleNamespace

import yaml

from luna_mujoco.resident_robot import ATTACHMENT_CAPACITY, reserve_attachment_links


def test_reserved_slots_are_fixed_for_both_arms_and_use_full_hand_exemptions():
    from luna_mujoco.backend_builder import ROOT
    from luna_mujoco.carrying import carrying_touch_links
    original = yaml.safe_load((ROOT/'resources/curobo/marvin_gui_dual.yml').read_text())
    before = deepcopy(original)
    result = reserve_attachment_links(original)
    assert original == before
    kin = result['robot_cfg']['kinematics']
    for arm in ('A', 'B'):
        link = 'jev_carried_object_' + arm
        assert len(kin['collision_spheres'][link]) == ATTACHMENT_CAPACITY == 31
        assert all(s['radius'] < 0 for s in kin['collision_spheres'][link])
        assert set(kin['self_collision_ignore'][link]) == set(carrying_touch_links(arm))
        assert 'TCP_Link_' + ('L' if arm == 'A' else 'R') not in kin['self_collision_ignore'][link]


def test_changing_measured_geometry_does_not_switch_planner_configuration(monkeypatch):
    import threading

    from luna_mujoco.geometry_service import GeometryRuntime
    runtime = object.__new__(GeometryRuntime)
    runtime._carrying_scope_lock = threading.RLock()
    runtime._lock = threading.Lock()
    runtime.carrying_binding = runtime.gripper_feedback = runtime.support_frame = None
    runtime.effective_config = {}
    selected = []
    runtime._select_planner_configuration_unlocked = lambda key, mode: selected.append((key, mode))

    for arm, radius, closure in [('A', .02, .3), ('B', .04, .4), ('A', .05, .5)]:
        with runtime.carrying_scope(SimpleNamespace(arm=arm, radius=radius, owner_id='test'),
                                    {'A':closure,'B':0.}):
            pass
    assert len(set(selected)) == 1
    assert selected[0][1] == 'open'


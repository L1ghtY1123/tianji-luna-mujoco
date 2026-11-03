from copy import deepcopy
from types import SimpleNamespace

import yaml
import pytest
import numpy as np

from luna_mujoco.resident_robot import ATTACHMENT_CAPACITY, reserve_attachment_links


@pytest.mark.parametrize('arm', ['A', 'B'])
@pytest.mark.parametrize('changed_peer', [False, True])
def test_feedback_update_rebuilds_peer_only_when_its_gripper_changed(monkeypatch, arm, changed_peer):
    import sys
    from luna_mujoco import resident_robot

    peer = 'B' if arm == 'A' else 'A'
    parsed = []
    class PeerRefreshReached(Exception):
        pass
    def parse(robot, **kwargs):
        parsed.append(robot)
        if len(parsed) > 1:
            raise PeerRefreshReached()
        return SimpleNamespace(kinematics_config=object())
    monkeypatch.setitem(sys.modules, 'curobo.kinematics', SimpleNamespace(
        Kinematics=object, KinematicsCfg=SimpleNamespace(from_data_dict=parse)))
    monkeypatch.setitem(sys.modules, 'curobo.types', SimpleNamespace(Pose=object))
    monkeypatch.setattr(resident_robot, 'configure_gripper_feedback', lambda robot, *args: robot)
    monkeypatch.setattr(resident_robot.ResidentRobotGeometry, '_evidence', staticmethod(lambda *args: None))
    robot = {'robot_cfg': {'kinematics': {'collision_link_names': [], 'collision_spheres': {}}}}
    resident = resident_robot.ResidentRobotGeometry(robot, 'fixture.urdf', peer_robot=robot, peer_links=['peer'])
    writes = []
    params = SimpleNamespace(update_link_spheres=lambda link, balls: writes.append(link))
    kinematics = SimpleNamespace(config=SimpleNamespace(kinematics_config=params),
                                 update_kinematics_config=lambda config: None)
    planner = SimpleNamespace(kinematics=kinematics, tool_frames=['tcp'],
                              device_cfg=SimpleNamespace(as_torch_dict=dict))
    backend = SimpleNamespace(arm=arm, ik_only=True, motion_planner=planner,
        _motion_clearance_checker=SimpleNamespace(kinematics=kinematics),
        _torch=SimpleNamespace(zeros=lambda shape, **kwargs: np.zeros(shape)))
    feedback = {arm: .4, peer: .3 if changed_peer else 0.}
    if changed_peer:
        with pytest.raises(PeerRefreshReached):
            resident.update(backend, feedback, None)
        assert len(parsed) == 2
    else:
        assert resident.update(backend, feedback, None)
        assert len(parsed) == 1
        assert writes == ['jev_carried_object_A', 'jev_carried_object_B']


def test_measured_p60_open_stop_noise_does_not_change_resident_geometry():
    from luna_mujoco.carrying import canonical_gripper_feedback

    assert canonical_gripper_feedback({'A': 2.5543970739738776e-11,
        'B': 2.3479546902520105e-11}) == {'A': 0., 'B': 0.}
    # Even a much smaller motion than a sensor can resolve remains exact
    # outside the documented open-stop numerical range.
    feedback = {'A': 1e-8, 'B': .100001}
    assert canonical_gripper_feedback(feedback) == feedback


def test_open_stop_reserve_covers_authored_gripper_kinematics():
    import numpy as np

    from luna_mujoco.demo_gripper_quality import gripper_width_model

    for arm in ('A', 'B'):
        model = gripper_width_model(arm)
        origin = model.transforms(0.)
        for value in (-1e-9, 1e-9):
            shifted = model.transforms(value)
            for name in origin:
                # Includes angular displacement of any point within a metre
                # of each gripper link, much larger than the authored hand.
                bound = (np.linalg.norm(shifted[name][:3, 3] - origin[name][:3, 3])
                         + np.linalg.norm(shifted[name][:3, :3] - origin[name][:3, :3], ord=2))
                assert bound < 1e-8


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


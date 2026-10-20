"""Independent own-robot FK coverage of the actual gripper interval bound."""

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from jev_mujoco.carrying import configure_gripper_feedback
from jev_mujoco.kinematics import robot_model

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf"


def configured_robot():
    robot = yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_bytes())
    return configure_gripper_feedback(robot, {"A": .4824312671843666, "B": .4824312671843666}, URDF)


def padding(robot, arm, step, urdf=URDF):
    from jev_mujoco.gripper_sweep import gripper_sweep_padding

    return gripper_sweep_padding(robot, arm, urdf, step)


@pytest.mark.parametrize("arm,prefix", [("A", "gripL_"), ("B", "gripR_")])
def test_dense_actual_urdf_fk_balls_are_covered_between_driver_samples(arm, prefix):
    robot = configured_robot()
    before = deepcopy(robot)
    spheres = robot["robot_cfg"]["kinematics"]["collision_spheres"]
    samples = np.linspace(0., .4824312671843666, 26)
    pads = padding(robot, arm, float(np.diff(samples).max()))
    model = robot_model()
    data = mujoco.MjData(model)
    joints = {joint.get("name"): joint for joint in ET.parse(URDF).getroot().findall("joint")}
    moving_links = [link for link in spheres if link.startswith(prefix) and link != prefix + "Base"]

    def centers(driver):
        for name, joint in joints.items():
            if not name.startswith(prefix) or joint.get("type") == "fixed":
                continue
            mimic = joint.find("mimic")
            value = driver if mimic is None else float(mimic.get("multiplier", "1")) * driver + float(mimic.get("offset", "0"))
            data.qpos[int(model.joint(name).qposadr[0])] = value
        mujoco.mj_forward(model, data)
        result = {}
        for link in moving_links:
            index = model.body(link).id
            local = np.asarray([sphere["center"] for sphere in spheres[link]])
            result[link] = local @ data.xmat[index].reshape(3, 3).T + data.xpos[index]
        return result

    sampled = [centers(value) for value in samples]
    for value in np.linspace(samples[0], samples[-1], 801):
        actual = centers(value)
        nearest = int(np.argmin(abs(samples - value)))
        for link in moving_links:
            # The same original radius occurs on both sides of containment.
            distance = np.linalg.norm(actual[link] - sampled[nearest][link], axis=1)
            assert np.all(distance <= pads[link] + 1e-10), (arm, link, value)
    assert robot == before
    assert set(pads) == set(robot["robot_cfg"]["kinematics"]["collision_link_names"])
    for link, values in pads.items():
        assert np.shape(values) == (len(spheres[link]),)
        assert np.isfinite(values).all() and np.all(values >= 0.)
        if link not in moving_links:
            np.testing.assert_array_equal(values, np.zeros(len(spheres[link])))
    assert any(np.any(pads[link] > 0) for link in moving_links)


def test_fixed_extra_link_padding_follows_parent_dependency_not_its_name():
    robot = configured_robot()
    kin = robot["robot_cfg"]["kinematics"]
    for name, parent in (("review_carried", "TCP_Link_L"), ("review_tip", "gripL_Link4_L")):
        kin.setdefault("extra_links", {})[name] = {"link_name": name, "parent_link_name": parent,
            "joint_name": name + "_fixed", "joint_type": "FIXED",
            "fixed_transform": [.02, .03, .04, 1., 0., 0., 0.]}
        kin["collision_link_names"].append(name)
        kin["collision_spheres"][name] = [{"center": [.01, .02, .03], "radius": .01}]
    before = deepcopy(robot)
    pads = padding(robot, "A", .02)
    np.testing.assert_array_equal(pads["review_carried"], [0.])
    assert pads["review_tip"][0] > 0
    assert robot == before
    zero = padding(robot, "A", 0.)
    assert all(np.all(values == 0.) for values in zero.values())


@pytest.mark.parametrize("change", ["zero_axis", "nan_axis", "nan_origin", "missing_mimic",
                                     "cyclic_mimic", "nan_multiplier", "invalid_limits", "dependent_prismatic"])
def test_malformed_own_motion_definition_is_rejected_instead_of_zero_padding(tmp_path, change):
    tree = ET.parse(URDF)
    joints = {joint.get("name"): joint for joint in tree.getroot().findall("joint")}
    joint = joints["gripL_Joint4_L"]
    if change in {"zero_axis", "nan_axis"}:
        joint.find("axis").set("xyz", "0 0 0" if change == "zero_axis" else "nan 0 1")
    elif change == "nan_origin":
        joint.find("origin").set("xyz", "nan 0 0")
    elif change in {"missing_mimic", "cyclic_mimic"}:
        joint.find("mimic").set("joint", "missing_driver" if change == "missing_mimic" else "gripL_Joint4_L")
    elif change == "nan_multiplier":
        joint.find("mimic").set("multiplier", "nan")
    elif change == "invalid_limits":
        joints["gripL_finger_joint"].find("limit").set("upper", "-1")
    else:
        joint.set("type", "prismatic")
    path = tmp_path / "malformed.urdf"
    tree.write(path)
    with pytest.raises(ValueError):
        padding(configured_robot(), "A", .02, path)


@pytest.mark.parametrize("step", [-.01, float("nan"), float("inf")])
def test_invalid_sampling_gap_cannot_become_a_continuous_certificate(step):
    with pytest.raises(ValueError):
        padding(configured_robot(), "A", step)

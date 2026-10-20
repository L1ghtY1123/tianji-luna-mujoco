"""Derive fixed-count sphere envelopes for the authored gripper closing motion."""

from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def _origin(joint):
    origin = joint.find("origin")
    values = {} if origin is None else origin.attrib
    xyz = np.fromstring(values.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(values.get("rpy", "0 0 0"), sep=" ")
    matrix = np.eye(4)
    matrix[:3, :3] = (
        _rotation([0, 0, 1], yaw)
        @ _rotation([0, 1, 0], pitch)
        @ _rotation([1, 0, 0], roll)
    )
    matrix[:3, 3] = xyz
    return matrix


def _joint_motion(joint, driver):
    if joint.get("type") == "fixed":
        return np.array([1., 0., 0.]), 0., 0.
    if joint.get("type") not in {"revolute", "continuous"}:
        raise ValueError("gripper closing envelopes require fixed or revolute joints")
    axis_element = joint.find("axis")
    axis = np.fromstring("1 0 0" if axis_element is None else axis_element.get("xyz"), sep=" ")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError("gripper joint axis must be a finite nonzero vector")
    if joint.get("name") == driver:
        return axis, 1., 0.
    mimic = joint.find("mimic")
    if mimic is None or mimic.get("joint") != driver:
        raise ValueError("gripper followers must mimic their authored driver directly")
    return axis, float(mimic.get("multiplier", 1)), float(mimic.get("offset", 0))


def create_gripper_closing_envelope(robot_config, arm, urdf_path):
    """Enclose each active gripper sphere's entire sweep in its locked link frame.

    Sampling is augmented by a continuous-motion bound. For a revolute ancestor,
    center speed is at most its mimic multiplier times the distance to the center.
    The sum of downstream translation lengths plus center norm bounds that
    distance at every configuration. Summing ancestor bounds gives a Lipschitz
    constant L; adding L * half the sampling interval covers every unsampled pose.
    """
    if arm not in {"A", "B"}:
        raise ValueError("gripper envelope arm must be A or B")
    prefix = "gripL_" if arm == "A" else "gripR_"
    base = prefix + "Base"
    driver = prefix + "finger_joint"
    document = ET.parse(urdf_path).getroot()
    joints = {joint.get("name"): joint for joint in document.findall("joint")}
    parents = {joint.find("child").get("link"): joint for joint in joints.values()}
    limit = joints[driver].find("limit")
    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("gripper driver must have finite increasing position limits")
    result = deepcopy(robot_config)
    kinematics = result.get("robot_cfg", result)["kinematics"]
    locked = float(kinematics["lock_joints"][driver])
    if not lower <= locked <= upper:
        raise ValueError("configured gripper lock is outside the authored limits")
    angles = np.linspace(lower, upper, 81)
    half_step = (upper - lower) / (2 * (len(angles) - 1))
    for link, spheres in kinematics["collision_spheres"].items():
        if not link.startswith(prefix) or link == base:
            continue
        chain = []
        ancestor = link
        while ancestor != base:
            joint = parents[ancestor]
            chain.append(joint)
            ancestor = joint.find("parent").get("link")
        chain.reverse()
        origins = [_origin(joint) for joint in chain]
        motions = [_joint_motion(joint, driver) for joint in chain]
        transforms = []
        for angle in [locked, *angles]:
            transform = np.eye(4)
            for origin, (axis, multiplier, offset) in zip(origins, motions):
                rotation = np.eye(4)
                rotation[:3, :3] = _rotation(axis, multiplier * angle + offset)
                transform = transform @ origin @ rotation
            transforms.append(transform)
        locked_inverse = np.linalg.inv(transforms[0])
        sampled = np.asarray([locked_inverse @ transform for transform in transforms[1:]])
        for sphere in spheres:
            center = np.asarray(sphere["center"], dtype=float)
            positions = sampled[:, :3, :3] @ center + sampled[:, :3, 3]
            enclosing_center = (positions.min(axis=0) + positions.max(axis=0)) / 2
            sampled_radius = np.linalg.norm(positions - enclosing_center, axis=1).max()
            speed_bound = 0.
            remaining_length = np.linalg.norm(center)
            for index in range(len(chain) - 1, -1, -1):
                speed_bound += abs(motions[index][1]) * remaining_length
                remaining_length += np.linalg.norm(origins[index][:3, 3])
            sphere["center"] = enclosing_center.tolist()
            sphere["radius"] += float(sampled_radius + speed_bound * half_step)
    return result

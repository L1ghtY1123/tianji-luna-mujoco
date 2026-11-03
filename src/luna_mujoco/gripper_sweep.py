"""Continuous world bounds for sampled motion of an authored gripper driver."""

import xml.etree.ElementTree as ET

import numpy as np


def safe_closure_prefix(angles, valid):
    """Only the contiguous collision-free prefix can authorize a close command."""
    failed = np.flatnonzero(~np.asarray(valid, dtype=bool))
    end = int(failed[0]) - 1 if len(failed) else len(angles) - 1
    return None if end <= 0 else float(angles[end])


def grasp_closure_samples(robot_config, arm, urdf_path, maximum_driver_rad=None):
    """Possible stopping positions before object contact, from authored limits."""
    if arm not in {'A', 'B'}:
        raise ValueError('closure preflight requires one arm')
    driver = 'gripL_finger_joint' if arm == 'A' else 'gripR_finger_joint'
    joint = next(joint for joint in ET.parse(urdf_path).getroot().findall('joint') if joint.get('name') == driver)
    limit = joint.find('limit')
    lower, upper = float(limit.get('lower')), float(limit.get('upper'))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError('closure preflight requires finite authored limits')
    if maximum_driver_rad is not None:
        if (type(maximum_driver_rad) not in (int, float) or not np.isfinite(maximum_driver_rad)
                or not lower < maximum_driver_rad <= upper):
            raise ValueError('closure limit must be inside authored limits')
        upper = float(maximum_driver_rad)
    angles = np.linspace(lower, upper, int(np.ceil((upper - lower) / .02)) + 1)
    padding = gripper_sweep_padding(robot_config, arm, urdf_path, float(np.diff(angles).max()))
    return angles, padding


def gripper_sweep_padding(robot_config, arm, urdf_path, max_step_rad):
    """Return per-sphere world-only padding covering the actual sample gaps.

    For each driver-dependent revolute ancestor, angular speed times a bound
    on the distance to the sphere center bounds its center speed. Downstream
    translation lengths plus the local center norm bound that distance for
    every angle. Summing ancestors gives L; L * half the largest sample gap
    covers the center between samples. The original sphere radius is unchanged
    in the robot model and in all self-collision queries.
    """
    if arm not in {"A", "B"} or not np.isfinite(max_step_rad) or max_step_rad < 0:
        raise ValueError("gripper sweep requires an arm and finite nonnegative sample gap")
    kin = robot_config["robot_cfg"]["kinematics"]
    document = ET.parse(urdf_path).getroot()
    joints = {joint.get("name"): joint for joint in document.findall("joint")}
    parents = {joint.find("child").get("link"): joint for joint in joints.values()}
    extras = kin.get("extra_links", {})
    driver = "gripL_finger_joint" if arm == "A" else "gripR_finger_joint"
    if driver not in joints or joints[driver].get("type") not in {"revolute", "continuous"}:
        raise ValueError("gripper sweep requires an authored angular driver")
    limit = joints[driver].find("limit")
    if limit is None:
        raise ValueError("gripper driver requires authored position bounds")
    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
    locked = float(kin["lock_joints"][driver])
    if not np.isfinite([lower, upper, locked]).all() or not lower <= locked <= upper or lower >= upper:
        raise ValueError("gripper driver position bounds or feedback are invalid")

    def vector(text):
        values = np.fromstring(text, sep=" ")
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("gripper sweep requires finite URDF origins and axes")
        return values

    multipliers = {}

    def multiplier(name, visiting=()):
        if name not in joints or name in visiting:
            raise ValueError("gripper sweep mimic target is missing or cyclic")
        if name in multipliers:
            return multipliers[name]
        joint = joints[name]
        kind = joint.get("type")
        if kind not in {"fixed", "revolute", "continuous"}:
            raise ValueError("gripper sweep supports fixed or angular URDF joints")
        axis = joint.find("axis")
        if kind != "fixed" and np.linalg.norm(vector("1 0 0" if axis is None else axis.get("xyz"))) < 1e-12:
            raise ValueError("gripper sweep joint axis must be nonzero")
        mimic = joint.find("mimic")
        if name == driver:
            if mimic is not None:
                raise ValueError("gripper driver cannot itself be a mimic")
            value = 1.
        elif mimic is not None:
            scale, offset = float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))
            if kind == "fixed" or not np.isfinite([scale, offset]).all():
                raise ValueError("gripper sweep mimic parameters are invalid")
            value = scale * multiplier(mimic.get("joint"), (*visiting, name))
        else:
            value = 0.
        if not np.isfinite(value):
            raise ValueError("gripper sweep mimic speed is invalid")
        multipliers[name] = value
        return value

    result = {}
    for link in kin["collision_link_names"]:
        centers = np.asarray([sphere["center"] for sphere in kin["collision_spheres"][link]], dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
            raise ValueError("gripper sweep requires finite original sphere centers")
        distance, speed = np.linalg.norm(centers, axis=1), np.zeros(len(centers))
        ancestor, visited = link, set()
        while ancestor != kin["base_link"]:
            if ancestor in visited:
                raise ValueError("gripper sweep link ancestry is cyclic")
            visited.add(ancestor)
            if ancestor in extras:
                extra = extras[ancestor]
                pose = np.asarray(extra["fixed_transform"], dtype=float)
                if (extra.get("joint_type") != "FIXED" or pose.shape != (7,) or not np.isfinite(pose).all()
                        or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6)):
                    raise ValueError("gripper sweep extra links must have rigid fixed transforms")
                distance += np.linalg.norm(pose[:3])
                ancestor = extra["parent_link_name"]
                continue
            joint = parents.get(ancestor)
            if joint is None:
                raise ValueError("gripper sweep link has no authored path to the robot base")
            speed += abs(multiplier(joint.get("name"))) * distance
            origin = joint.find("origin")
            if origin is not None:
                vector(origin.get("rpy", "0 0 0"))
                distance += np.linalg.norm(vector(origin.get("xyz", "0 0 0")))
            ancestor = joint.find("parent").get("link")
        result[link] = speed * float(max_step_rad) / 2
    return result

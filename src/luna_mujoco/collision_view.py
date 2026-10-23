"""Read-only display of the configured planning spheres with both grippers open."""

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "resources/curobo/marvin_gui_dual.yml"


@lru_cache(maxsize=1)
def collision_configuration():
    config = yaml.safe_load((ROOT / SOURCE).read_text(encoding="utf-8"))["robot_cfg"]["kinematics"]
    joints = dict(config["lock_joints"])
    urdf = ET.parse(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf").getroot()
    for joint in urdf.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            joints[joint.get("name")] = (joints[mimic.get("joint")] * float(mimic.get("multiplier", 1))
                                         + float(mimic.get("offset", 0)))
    return config, joints


def collision_sphere_snapshot(model, live_data, geometry_ids):
    config, open_joints = collision_configuration()
    data = mujoco.MjData(model)
    data.qpos[:] = live_data.qpos
    for name, position in open_joints.items():
        data.qpos[int(model.joint(name).qposadr[0])] = position
    mujoco.mj_fwdPosition(model, data)
    spheres = []
    buffers = config["collision_sphere_buffer"]
    for link in config["collision_link_names"]:
        body_id = model.body(link).id
        rotation = data.xmat[body_id].reshape(3, 3)
        buffer = buffers.get(link, 0.) if isinstance(buffers, dict) else float(buffers)
        for index, sphere in enumerate(config["collision_spheres"][link]):
            center = data.xpos[body_id] + rotation @ np.asarray(sphere["center"])
            spheres.append({"id": f"collision::{link}::{index}", "link": link,
                            "center_m": center.tolist(), "radius_m": float(sphere["radius"]) + buffer})
    poses = []
    for index in geometry_ids:
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.geom_xmat[index])
        poses.append({"id": str(index), "position": data.geom_xpos[index].tolist(),
                      "quaternion": quaternion.tolist()})
    return {"source": SOURCE, "gripper_state": "open", "radius_policy": "original_yaml_plus_link_buffer",
            "sphere_count": len(spheres), "link_count": len(config["collision_link_names"]),
            "spheres": spheres}, poses

"""Own robot model FK for hypothetical plans; never changes the physics instance."""

from functools import lru_cache

import mujoco
import numpy as np

from .model_builder import ARM_JOINTS, TCP_BODIES, ModelBuilder, normalize_scene


@lru_cache(maxsize=1)
def robot_model():
    builder = ModelBuilder()
    try:
        return builder.build(normalize_scene({"objects": []}))
    finally:
        builder.close()


def robot_tcp(arms, arm):
    model = robot_model()
    data = mujoco.MjData(model)
    for name, joints in ARM_JOINTS.items():
        values = np.asarray(arms[name]["joints_rad"], dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("FK requires seven measured or planned arm joints")
        for joint, value in zip(joints, values):
            data.qpos[int(model.joint(joint).qposadr[0])] = value
    mujoco.mj_forward(model, data)
    index = model.body(TCP_BODIES[arm]).id
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[index].reshape(3, 3)
    transform[:3, 3] = data.xpos[index]
    return transform



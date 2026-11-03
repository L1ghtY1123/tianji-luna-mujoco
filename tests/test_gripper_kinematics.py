"""Kinematics-only width screening must preserve the full-position FK numerics."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import mujoco
import numpy as np
import pytest

from luna_mujoco import demo_gripper_quality as quality


def full_position_state(model, q):
    data = mujoco.MjData(model.model)
    values = {**model.locks, model.driver: float(q)}
    def value(name):
        mimic = model.joints[name].find('mimic')
        if mimic is None:
            return values.get(name, 0.)
        return value(mimic.get('joint')) * float(mimic.get('multiplier', 1)) + float(mimic.get('offset', 0))
    for name, joint in model.joints.items():
        if joint.get('type') != 'fixed' and (name in values or joint.find('mimic') is not None):
            data.qpos[int(model.model.joint(name).qposadr[0])] = value(name)
    mujoco.mj_fwdPosition(model.model, data)
    return data


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_kinematics_matches_full_position_over_complete_gripper_range(arm):
    model = quality.gripper_width_model(arm)
    for q in np.unique(np.r_[np.linspace(model.lower, model.upper, 121), model.angles]):
        reference = full_position_state(model, q)
        transforms = model.transforms(q)
        tool = model.model.body(model.tool).id
        rotation = reference.xmat[tool].reshape(3, 3)
        for link, actual in transforms.items():
            body = model.model.body(link).id
            expected = np.eye(4)
            expected[:3, :3] = rotation.T @ reference.xmat[body].reshape(3, 3)
            expected[:3, 3] = rotation.T @ (reference.xpos[body] - reference.xpos[tool])
            np.testing.assert_array_equal(actual, expected)
        # Also compare every body, including all owning-hand touch links, not
        # only the retained noncontact TCP spheres returned by transforms().
        scratch = model._scratch.data
        np.testing.assert_array_equal(scratch.qpos, reference.qpos)
        np.testing.assert_array_equal(scratch.xpos, reference.xpos)
        np.testing.assert_array_equal(scratch.xmat, reference.xmat)
        np.testing.assert_array_equal(scratch.geom_xpos, reference.geom_xpos)
        np.testing.assert_array_equal(scratch.geom_xmat, reference.geom_xmat)


def test_transforms_reuses_thread_scratch_and_only_calls_public_kinematics(monkeypatch):
    model = quality.gripper_width_model('A')
    def forbidden(*args):
        pytest.fail('warm transform allocated data or ran unnecessary full dynamics')
    monkeypatch.setattr(mujoco, 'MjData', forbidden)
    monkeypatch.setattr(mujoco, 'mj_fwdPosition', forbidden)
    first = model.transforms(.1)
    saved = {key: value.copy() for key, value in first.items()}
    model.transforms(.7)
    for name in first:
        np.testing.assert_array_equal(first[name], saved[name])


def test_thread_scratch_isolated_with_interleaved_closures():
    model = quality.gripper_width_model('A')
    barrier = Barrier(2)
    expected = {q: model.transforms(q) for q in (.1, .7)}
    def worker(q):
        for _ in range(12):
            actual = model.transforms(q)
            barrier.wait()
            for name in expected[q]:
                np.testing.assert_array_equal(actual[name], expected[q][name])
        return model._scratch.data
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(worker, .1), pool.submit(worker, .7)
        one, two = first.result(), second.result()
    assert one is not two and one is not model._scratch.data

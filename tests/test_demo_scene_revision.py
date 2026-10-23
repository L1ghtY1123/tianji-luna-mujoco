"""The operator's revised layout and dynamic workpieces use real MuJoCo physics."""
import mujoco
import numpy as np
import pytest

from luna_mujoco.demo_scene import demo_observation_poses, demo_scene
from luna_mujoco.simulation import Simulation


def test_revised_world_coordinates_and_pink_yaw():
    scene = demo_scene(2)
    np.testing.assert_allclose(scene['table']['position'], [.572, -.026, .532])
    assert scene.get('support_tables', []) == []
    left, right = scene['objects'][:2]
    np.testing.assert_allclose(left['position'], [.481, .573, .571])
    np.testing.assert_allclose(right['position'], [.481, -.625, .571])
    assert (left['position'][1] + right['position'][1]) / 2 == pytest.approx(scene['table']['position'][1])
    assert all(item['yaw_deg'] == 180 for item in scene['objects'][2:])


def test_fixed_containers_and_dynamic_workpieces_keep_gravity():
    scene = demo_scene(2)
    scene['objects'][2]['position'][2] += .05
    sim = Simulation(scene, initial_joints=demo_observation_poses())
    try:
        model, data = sim._model, sim._data
        np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
        before = {}
        for item in scene['objects'][:2]:
            body = model.body('object_' + item['id'])
            assert model.body_jntnum[body.id] == 0
            before[body.name] = data.body(body.name).xpos.copy()
        pink = model.body('object_pink_cylinder_1')
        assert model.body_jntnum[pink.id] == 1 and model.body_gravcomp[pink.id] == 0
        for _ in range(40):
            sim._apply_motor_controls()
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        assert data.body(pink.name).xpos[2] < scene['objects'][2]['position'][2] - .005
        for name, position in before.items():
            np.testing.assert_array_equal(data.body(name).xpos, position)
    finally:
        sim.close()


def test_mock_black_is_horizontal_supported_and_remains_dynamic():
    scene = demo_scene(2)
    top = scene['table']['position'][2] + scene['table']['size'][2] / 2
    scene['objects'][2]['position'] = [.46, -.026, top + .055]
    sim = Simulation(scene, initial_joints=demo_observation_poses())
    sim.start()
    sim._condition.acquire()
    try:
        receipt = sim.mock_unpack({'center_world_xy_m': [.46, -.026], 'size_m': .32,
            'floor_z_m': top}, generation=sim._generation)
        body = sim._model.body('object_' + receipt['physical_object_id'])
        assert sim._model.body_jntnum[body.id] == 1 and sim._model.body_gravcomp[body.id] == 0
        assert abs(sim._data.body(body.name).xmat.reshape(3, 3)[2, 2]) < 1e-8
        assert receipt['after']['position'][2] == pytest.approx(top + .022 + .001)
        assert receipt['commanded_pose']['pitch_deg'] == 90
        for _ in range(3000):
            sim._apply_motor_controls()
            mujoco.mj_step(sim._model, sim._data)
        mujoco.mj_forward(sim._model, sim._data)
        result = sim._data.body(body.name)
        assert abs(result.xmat.reshape(3, 3)[2, 2]) < .03
        assert abs(result.xpos[2] - top - .022) < .003
        assert np.isfinite(sim._data.qpos).all()
    finally:
        sim._condition.release()
        sim.close()

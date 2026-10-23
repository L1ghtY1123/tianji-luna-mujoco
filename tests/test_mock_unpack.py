"""The authorized mock is a physical transaction, never visual evidence."""
from copy import deepcopy

import mujoco
import numpy as np
import pytest

from luna_mujoco.demo_scene import demo_observation_poses, demo_scene
from luna_mujoco.geometry import scene_object_rotation
from luna_mujoco.simulation import Simulation, StaleCommandError

AREA = {"center_world_xy_m": [.46, 0.], "size_m": .20, "floor_z_m": .572}


@pytest.fixture
def simulation():
    scene = demo_scene(2)
    scene["objects"][2]["position"] = [.46, 0., .572 + scene["objects"][2]["size"][2]/2]
    instance = Simulation(scene, initial_joints=demo_observation_poses())
    instance.start()
    try:
        yield instance
    finally:
        instance.close()


def integration_state(simulation):
    state = np.empty(mujoco.mj_stateSize(simulation._model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(simulation._model, simulation._data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


def test_mock_unpack_preserves_run_robot_controls_other_objects_and_baseline(simulation):
    baseline = simulation.confirm()
    with simulation._condition:
        sim = simulation
        old_model = sim._model
        sim._data.qvel[sim._arm_v["A"]] = np.linspace(.001, .007, 7)
        sim._data.qacc_warmstart[:] = np.linspace(0, .01, sim._model.nv)
        sim._data.qfrc_applied[:] = .001
        sim._data.xfrc_applied[:] = .002
        sim._desired_q["B"] += .0001
        sim._apply_motor_controls()
        desired = deepcopy((sim._desired_q, sim._desired_v, sim._desired_grip))
        before = {key: getattr(sim._data, key).copy() for key in
                  ("qpos", "qvel", "ctrl", "act", "qacc_warmstart", "qfrc_applied", "xfrc_applied")}
        before_time, revision = sim._data.time, sim._revision
        receipt = sim.mock_unpack(AREA, task_id="task", generation=sim._generation,
                                  target_id="canonical-1", owner_id="run-1")
        assert receipt["source"] == "mock_vla"
        assert receipt["is_visual_evidence"] is False
        assert receipt["target_id"] == "canonical-1" and receipt["owner_id"] == "run-1"
        assert receipt["completed"] is True and receipt["action_id"]
        assert receipt["completed_time"] == before_time
        assert receipt["completed_perf_counter_ns"] > 0 and receipt["completed_monotonic_ns"] > 0
        assert sim._generation == baseline["generation"] and sim._data.time == before_time
        assert sim._revision == revision + 1 and sim._confirmed
        assert sim._baseline["model"] is old_model
        physical_id = receipt["physical_object_id"]
        joint = sim._model.joint("free_" + physical_id)
        q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
        before["qpos"][q:q + 7] = sim._data.qpos[q:q + 7]
        before["qvel"][v:v + 6] = sim._data.qvel[v:v + 6]
        for key, expected in before.items():
            np.testing.assert_array_equal(getattr(sim._data, key), expected, err_msg=key)
        for old, new in zip(desired, (sim._desired_q, sim._desired_v, sim._desired_grip), strict=True):
            for arm in old:
                np.testing.assert_array_equal(new[arm], old[arm])
        item = next(obj for obj in sim.snapshot()["objects"] if obj["id"] == physical_id)
        assert item["kind"] == "cad" and item["asset_id"] == "black_core" and item["color"] == "black"
        assert item["size"] == [.044, .056, .0745]
        np.testing.assert_allclose(item["position"], [.56, -.22, .595], atol=1e-12)
        np.testing.assert_allclose(np.asarray(item["T_world_object"])[:3, :3],
                                   scene_object_rotation(receipt["after"]), atol=1e-12)
        assert sim._model.geom("object_visual_" + physical_id).type == mujoco.mjtGeom.mjGEOM_MESH
        assert receipt["commanded_size_m"] == item["size"]
        assert sim.latest_rgbd() is None and sim._latest_preview is None
    restored = simulation.reset()
    assert next(obj for obj in restored["objects"] if obj["id"] == physical_id)["color"] == "pink"


@pytest.mark.parametrize("count", [0, 2])
def test_mock_unpack_requires_one_supported_pink_entity_without_mutation(simulation, count):
    with simulation._condition:
        for index, object_id in enumerate(("pink_cylinder_1", "pink_cylinder_2")):
            joint = simulation._model.joint("free_" + object_id)
            q = int(joint.qposadr[0])
            simulation._data.qpos[q:q + 3] = [.43 + .06 * index, 0 if index < count else .6, .61]
        before, model, revision = integration_state(simulation), simulation._model, simulation._revision
        with pytest.raises(ValueError, match="unique.*pink"):
            simulation.mock_unpack(AREA, generation=simulation._generation)
        np.testing.assert_array_equal(integration_state(simulation), before)
        assert simulation._model is model and simulation._revision == revision


def test_mock_unpack_failed_compile_rolls_back_and_stale_generation_rejects(simulation, monkeypatch):
    with simulation._condition:
        before, model = integration_state(simulation), simulation._model
        with pytest.raises(StaleCommandError):
            simulation.mock_unpack(AREA, generation=simulation._generation - 1)
        def fail(_):
            raise RuntimeError("compile failed")
        monkeypatch.setattr(simulation._builder, "build", fail)
        with pytest.raises(RuntimeError, match="compile failed"):
            simulation.mock_unpack(AREA, generation=simulation._generation)
        np.testing.assert_array_equal(integration_state(simulation), before)
        assert simulation._model is model


@pytest.mark.parametrize("output", [
    {"position": [.36, .14, .59825], "yaw_deg": 30},
    {"position": [.36, -.14, .7], "yaw_deg": 30},
    {"position": [.9, -.14, .59825], "yaw_deg": 30},
    {"position": [.36, -.14, .59825], "yaw_deg": float("nan")},
])
def test_mock_unpack_rejects_unsupported_output_pose_without_mutation(simulation, output):
    with simulation._condition:
        before = integration_state(simulation)
        with pytest.raises(ValueError, match="output"):
            simulation.mock_unpack(AREA, output, generation=simulation._generation)
        np.testing.assert_array_equal(integration_state(simulation), before)


def test_recording_snapshot_binds_manifest_revision_to_actual_shape(simulation):
    from pathlib import Path

    with simulation._condition:
        first = simulation.recording_snapshot()
        old_urls = {row["mesh_url"] for row in first["geometry_update"]["geometry_manifest"]["geometries"]
                    if row.get("asset_id") == "pink_wrapped"}
        old_bytes = {url: Path(url.removeprefix("/")).read_bytes() for url in old_urls}
        receipt = simulation.mock_unpack(AREA, generation=simulation._generation)
        second = simulation.recording_snapshot(first["manifest_revision"])
        update = second["geometry_update"]
        assert update["geometry_manifest"]["revision"] == second["manifest_revision"]
        physical_id = receipt["physical_object_id"]
        assert next(obj for obj in update["scene"]["objects"] if obj["id"] == physical_id)["color"] == "black"
        new = [row for row in update["geometry_manifest"]["geometries"] if row.get("object_id") == physical_id]
        assert len(new) == 1 and new[0]["asset_id"] == "black_core"
        assert new[0]["mesh_url"] not in old_urls
        assert all(Path(url.removeprefix("/")).read_bytes() == value for url, value in old_bytes.items())
        assert "geometry_update" not in simulation.recording_snapshot(second["manifest_revision"])


def test_mock_unpack_rejects_active_gripper_or_arm_lock(simulation):
    with simulation._motion_locks["A"], simulation._condition:
        before = integration_state(simulation)
        with pytest.raises(RuntimeError, match="idle_arms"):
            simulation.mock_unpack(AREA, generation=simulation._generation)
        np.testing.assert_array_equal(integration_state(simulation), before)


def test_default_unpacked_cad_stays_in_head_observation_frame(simulation):
    """At least one camera must cover the part; the head also supports local mapping."""
    import trimesh

    from luna_mujoco.cad_assets import CAD_ROOT, cad_asset
    from luna_mujoco.cameras import camera_pose, load_calibration

    asset = cad_asset("black_core")
    vertices = np.asarray(trimesh.load_mesh(CAD_ROOT / asset["visual"]["path"]).vertices)
    calibration = load_calibration()["cameras"]
    with simulation._condition:
        target = simulation._unpack_target(AREA)
        output = simulation._unpack_output(target, None)
        mujoco.mj_fwdPosition(simulation._model, simulation._data)
        for camera_id in ("3",):
            transform = camera_pose(simulation._model, simulation._data, camera_id)
            intrinsic = calibration[camera_id]["intrinsics"]
            for yaw in range(0, 360, 15):
                rotation = scene_object_rotation({"yaw_deg": yaw, "pitch_deg": 90})
                world = vertices @ rotation.T + output["position"]
                optical = (world - transform[:3, 3]) @ transform[:3, :3]
                assert np.all(optical[:, 2] > 0)
                u = intrinsic["fx"] * optical[:, 0] / optical[:, 2] + intrinsic["cx"]
                v = intrinsic["fy"] * optical[:, 1] / optical[:, 2] + intrinsic["cy"]
                assert np.mean((u >= 0) & (u < intrinsic["width"])) > .99
                assert v.min() >= 0 and v.max() < intrinsic["height"]


@pytest.mark.parametrize("position", [[.46, 0, .70], [.55, 0, .61]])
def test_mock_unpack_does_not_select_floating_or_partially_outside_entity(simulation, position):
    with simulation._condition:
        q = int(simulation._model.joint("free_pink_cylinder_1").qposadr[0])
        simulation._data.qpos[q:q + 3] = position
        with pytest.raises(ValueError, match="unique.*pink"):
            simulation.mock_unpack(AREA, generation=simulation._generation)


def test_mock_unpack_rejects_output_collision_without_partial_swap(simulation):
    with simulation._condition:
        q = int(simulation._model.joint("free_pink_cylinder_2").qposadr[0])
        simulation._data.qpos[q:q + 3] = [.56, -.22, .61]
        before, model = integration_state(simulation), simulation._model
        with pytest.raises(ValueError, match="output_collides"):
            simulation.mock_unpack(AREA, generation=simulation._generation)
        assert simulation._model is model
        np.testing.assert_array_equal(integration_state(simulation), before)


def test_capture_from_previous_geometry_revision_cannot_publish_after_unpack(simulation, monkeypatch):
    def render_during_change(model, data, generation, arms):
        simulation.mock_unpack(AREA, generation=generation)
        return {"generation": generation, "observation_id": "old-shape"}
    monkeypatch.setattr(simulation, "_render_capture", render_during_change)
    with pytest.raises(StaleCommandError, match="geometry changed"):
        simulation.capture_rgbd()
    assert simulation.latest_rgbd() is None
    assert any(obj["color"] == "black" for obj in simulation.snapshot()["objects"])

def test_mock_output_clears_vla_hands_including_local_mapping_padding(simulation):
    from pathlib import Path

    import yaml

    from luna_mujoco.carrying import configure_gripper_feedback
    from luna_mujoco.kinematics import robot_model
    from luna_mujoco.model_builder import ARM_JOINTS

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'resources/curobo/marvin_gui_dual.yml').read_text())
    kin = configure_gripper_feedback(config, {'A': 0., 'B': 0.},
        root / 'assets/marvin_urdf/marvin_with_grippers.urdf')['robot_cfg']['kinematics']
    model, goals = robot_model(), {'A': [-86, 76, 88, -90, 2, 14, -24],
                                  'B': [106, 61, -113, -78, 27, 36, 37]}
    data = mujoco.MjData(model)
    for name, value in kin['lock_joints'].items():
        data.qpos[int(model.joint(name).qposadr[0])] = value
    for arm, names in ARM_JOINTS.items():
        for name, value in zip(names, np.deg2rad(goals[arm]), strict=True):
            data.qpos[int(model.joint(name).qposadr[0])] = value
    mujoco.mj_fwdPosition(model, data)
    with simulation._condition:
        output = simulation._unpack_output(simulation._unpack_target(AREA), None)
    # Covers every flat yaw, plus the measured obstacle's 2cm margin.
    half = np.asarray(output['size']) / 2
    xy_radius = np.linalg.norm(half[1:]) + .02
    extents = np.array([xy_radius, xy_radius, half[0] + .02])
    center = np.asarray(output['position'])
    for link in kin['collision_link_names']:
        body = model.body(link).id
        for sphere in kin['collision_spheres'][link]:
            world = data.xpos[body] + data.xmat[body].reshape(3, 3) @ sphere['center']
            radius = sphere['radius'] + kin['collision_sphere_buffer'].get(link, 0.)
            offset = np.abs(world - center) - extents
            gap = np.linalg.norm(np.maximum(offset, 0)) + min(max(offset), 0) - radius
            assert gap >= .01, (link, gap)


def test_mock_output_visible_to_head_at_required_observation_pose(simulation):
    """Scene-authoring check: actual occlusion, not only projection into the frame."""
    from luna_mujoco.cameras import CAMERA_NAMES
    from luna_mujoco.model_builder import ARM_JOINTS

    with simulation._condition:
        simulation.mock_unpack(AREA, generation=simulation._generation)
        model, data = simulation._model, simulation._data
        for arm, angles in demo_observation_poses().items():
            for name, value in zip(ARM_JOINTS[arm], angles, strict=True):
                data.qpos[int(model.joint(name).qposadr[0])] = value
        address = int(model.joint('free_pink_cylinder_1').qposadr[0])
        geometry = model.geom('object_visual_pink_cylinder_1').id
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        with mujoco.Renderer(model, height=360, width=640) as renderer:
            renderer.enable_segmentation_rendering()
            for yaw in range(0, 360, 30):
                rotation = scene_object_rotation({'yaw_deg': yaw, 'pitch_deg': 90.})
                mujoco.mju_mat2Quat(data.qpos[address+3:address+7], rotation.ravel())
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=CAMERA_NAMES['3'], scene_option=options)
                segmentation = renderer.render()
                assert np.sum(segmentation[:, :, 0] == geometry) >= 300, yaw

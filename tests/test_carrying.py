from copy import deepcopy

import numpy as np
import pytest


def binding(**updates):
    return {"arm": "A", "owner_id": "hold-1", "observation_id": "rgbd-2",
            "generation": 2, "scene_token": "scene-2", "instance_id": "sam-source",
            "frame": "tool", "state": "held", "points_tool_m": [[-.03, -.02, -.04], [.03, .02, .04]],
            "uncertainty_m": .004, "coverage": .6} | updates


def test_carrying_rejects_stale_owner_frame_or_observation():
    from jev_mujoco.carrying import CarryingBinding

    scene = {"scene_token": "scene-2", "observation_id": "rgbd-2", "generation": 2, "target_id": "sam-source"}
    valid = CarryingBinding.from_mapping(binding(), scene=scene, arm="A")
    assert valid.owner_id == "hold-1"
    for changes in ({"generation": 1}, {"arm": "B"}, {"scene_token": "old"},
                    {"observation_id": "old"}, {"frame": "world"}, {"owner_id": ""},
                    {"points_tool_m": [[float("nan"), 0, 0]]}):
        with pytest.raises(ValueError):
            CarryingBinding.from_mapping(binding(**changes), scene=scene, arm="A")


def test_measured_envelope_covers_observed_surface_and_uncertainty():
    from jev_mujoco.carrying import CarryingBinding, carrying_spheres

    value = CarryingBinding.from_mapping(binding(), scene={"scene_token": "scene-2", "observation_id": "rgbd-2",
                                                          "generation": 2, "target_id": "sam-source"}, arm="A")
    spheres = np.asarray(carrying_spheres(value))
    corners = np.asarray(value.points_tool_m)
    distance = np.linalg.norm(corners[:, None] - spheres[None, :, :3], axis=-1)
    assert np.all(np.min(distance - spheres[None, :, 3], axis=1) <= -.004 + 1e-12)
    assert 1 <= len(spheres) <= 256


def test_configured_carried_link_preserves_robot_and_only_ignores_hold_contacts():
    from jev_mujoco.carrying import CarryingBinding, create_carrying_robot_config

    robot = {"robot_cfg": {"kinematics": {"tool_frames": ["TCP_Link_L"],
             "collision_link_names": ["Link1_L", "TCP_Link_L", "gripL_Base", "gripR_Base"],
             "collision_spheres": {"Link1_L": [{"center": [0, 0, 0], "radius": .05}]},
             "self_collision_ignore": {"Link1_L": []}}}}
    original = deepcopy(robot)
    value = CarryingBinding.from_mapping(binding(), scene={"scene_token": "scene-2", "observation_id": "rgbd-2",
                                                          "generation": 2, "target_id": "sam-source"}, arm="A")
    configured, link = create_carrying_robot_config(robot, value)
    kin = configured["robot_cfg"]["kinematics"]
    assert robot == original
    assert kin["extra_links"][link]["parent_link_name"] == "TCP_Link_L"
    assert kin["collision_spheres"][link]
    assert link not in kin["self_collision_ignore"]["Link1_L"]
    assert link not in kin["self_collision_ignore"].get("gripR_Base", [])


def test_retreat_never_accepts_attachment_and_open_requires_driver_feedback():
    from jev_mujoco.carrying import validate_phase_request

    with pytest.raises(ValueError, match="retreat"):
        validate_phase_request({"phase": "retreat", "carrying_binding": binding()})
    with pytest.raises(ValueError, match="gripper"):
        validate_phase_request({"phase": "open", "carrying_binding": binding()})


def test_gripper_lock_tolerates_only_tiny_endpoint_feedback_error():
    from pathlib import Path

    import yaml

    from jev_mujoco.carrying import configure_gripper_feedback

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'resources/curobo/marvin_gui_dual.yml').read_bytes())
    urdf = root / 'assets/marvin_urdf/marvin_with_grippers.urdf'
    locked = configure_gripper_feedback(config, {'A': .800019, 'B': -.000019}, urdf)
    assert locked['robot_cfg']['kinematics']['lock_joints']['gripL_finger_joint'] == pytest.approx(.8)
    assert locked['robot_cfg']['kinematics']['lock_joints']['gripR_finger_joint'] == 0.
    with pytest.raises(ValueError, match='authored limits'):
        configure_gripper_feedback(config, {'A': .801, 'B': 0.}, urdf)


def test_derived_gripper_config_mirrors_only_mesh_verified_opposing_pad_and_linkage_pairs():
    from pathlib import Path

    import yaml

    from jev_mujoco.carrying import configure_gripper_feedback

    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root / "resources/curobo/marvin_gui_dual.yml").read_bytes())
    before = deepcopy(original)
    configured = configure_gripper_feedback(original, {"A": .55, "B": 0.}, root / "assets/marvin_urdf/marvin_with_grippers.urdf")
    ignores = configured["robot_cfg"]["kinematics"]["self_collision_ignore"]
    assert "gripL_Link4_R" in ignores["gripL_Link4_L"]
    assert "gripL_Link4_L" in ignores["gripL_Link4_R"]
    assert "gripL_Link3_R" in ignores["gripL_Link3_L"]
    assert "gripR_Link3_R" in ignores["gripR_Link3_L"]
    original_ignores = before["robot_cfg"]["kinematics"]["self_collision_ignore"]
    added = {(link, other) for link, values in ignores.items() for other in values if other not in original_ignores.get(link, [])}
    allowed = {(f"grip{arm}_Link{part}_{side}", f"grip{arm}_Link{part}_{other}")
               for arm in ("L", "R") for part in (3, 4) for side, other in (("L", "R"), ("R", "L"))}
    assert added <= allowed
    assert configured["robot_cfg"]["kinematics"]["collision_spheres"] == before["robot_cfg"]["kinematics"]["collision_spheres"]
    assert original == before


def test_target_contact_exemption_is_limited_to_final_tcp_zone():
    from jev_mujoco.carrying import target_contact_samples

    goal = np.eye(4)
    matrices = np.repeat(goal[None], 4, axis=0)
    matrices[1, 0, 3] = .016
    matrices[2, :3, :3] = [[1, 0, 0], [0, np.cos(.1), -np.sin(.1)], [0, np.sin(.1), np.cos(.1)]]
    matrices[3, 0, 3] = .014
    assert target_contact_samples(matrices, goal).tolist() == [True, False, False, True]






def test_top_surface_never_adds_unseen_bottom_or_accepts_prior_fields():
    from jev_mujoco.carrying import CarryingBinding, carrying_spheres
    scene = {"scene_token": "scene-2", "observation_id": "rgbd-2", "generation": 2, "target_id": "sam-source"}
    payload = binding(points_tool_m=[[-.03, -.02, .04], [.03, .02, .04]])
    value = CarryingBinding.from_mapping(payload, scene=scene, arm="A")
    spheres = np.asarray(carrying_spheres(value))
    bottom = np.array([0., 0., -.03])
    assert np.min(np.linalg.norm(spheres[:, :3] - bottom, axis=1) - spheres[:, 3]) > 0
    for field in ("estimated_bounds_tool_m", "shape_prior_provenance"):
        with pytest.raises(ValueError, match="fields"):
            CarryingBinding.from_mapping(payload | {field: {}}, scene=scene, arm="A")


def test_support_prefix_is_needed_only_until_full_soft_clearance():
    from jev_mujoco.carrying import CarryingBinding, support_prefix_distance

    scene = {"scene_token": "scene-2", "observation_id": "rgbd-2", "generation": 2, "target_id": "sam-source"}
    value = CarryingBinding.from_mapping(binding(), scene=scene, arm="A")
    plane = {"normal_xyz": [0, 0, 1], "replacement_cuboid": {"pose": [0, 0, .65, 1, 0, 0, 0], "dims": [1, 1, .1]}}
    pose = np.eye(4)
    pose[2, 3] = .74
    assert .03 < support_prefix_distance(value, plane, pose, .03) < .06
    pose[2, 3] = .84
    assert support_prefix_distance(value, plane, pose, .03) == 0


def test_support_contact_policy_is_bound_to_lift():
    from jev_mujoco.carrying import validate_phase_request

    with pytest.raises(ValueError, match="support"):
        validate_phase_request({"phase": "place", "carrying_binding": binding(), "support_contact_policy": "observed_plane_normal_departure"})


def test_actual_telemetry_must_match_the_first_named_command():
    from jev_mujoco.carrying import validate_phase_request

    payload = {"phase": "approach_grasp", "arm": "A", "positions_rad": [[0.] * 7],
               "joint_names": [f"Joint{i}_L" for i in range(1, 8)], "current_joints_rad": [0.] * 7}
    assert validate_phase_request(payload) == "approach_grasp"
    with pytest.raises(ValueError, match="current"):
        validate_phase_request(payload | {"current_joints_rad": [.02] * 7})










def test_world_z_lift_uses_tilted_support_frame_without_requiring_goal_collinearity():
    from jev_mujoco.carrying import support_departure_frame, validate_support_goal_projection

    tcp = np.eye(4)
    tcp[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    tcp[:3, 3] = [.4, .15, .8]
    normal = np.array([.0005, -.0003, 1.])
    normal /= np.linalg.norm(normal)
    frame = support_departure_frame("A", tcp, normal)
    assert np.allclose((tcp @ frame["T_tool_frame"])[:3, 2], normal, atol=1e-12)
    goal = tcp[:3, 3] + [.001, -.002, .1]  # original world-Z target plus small FK goal drift
    assert validate_support_goal_projection(tcp[:3, 3], goal, normal, .04) > .099
    assert np.array_equal(goal, tcp[:3, 3] + [.001, -.002, .1])
    with pytest.raises(ValueError, match="projection"):
        validate_support_goal_projection(tcp[:3, 3], tcp[:3, 3] + [.1, 0, 0], normal, .04)


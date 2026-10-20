"""SI contract and conservative observed-geometry cover for carried lift checks.

No SDK types cross this boundary. A measured cloud does not bound unseen geometry.
"""

import hashlib
import itertools
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import CuroboToolError

CARRIED_LINK = "gap_carried_object"
MAX_SAMPLES = 10_000
MAX_JOINT_STEP_RAD = math.radians(0.25)


def _fields(value, required, name):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"{name} requires exactly: {', '.join(sorted(required))}")


def _numbers(value, name):
    try:
        array = np.asarray(value, dtype=object)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) or not math.isfinite(x)
        for x in array.flat
    ):
        raise ValueError(f"{name} must contain finite numbers, not booleans")
    return np.asarray(value, dtype=np.float64)


def normalize_lift(request, scene_snapshot, attachment):
    policy = request.get("support_contact_policy") if isinstance(request, dict) else None
    if policy not in (None, "observed_plane_normal_departure"):
        raise ValueError("unsupported support contact policy")
    _fields(
        {key: value for key, value in request.items() if key != "support_contact_policy"}
        if isinstance(request, dict)
        else request,
        {
            "plan_id",
            "arm",
            "joint_names",
            "positions_rad",
            "times_s",
            "other_joints_rad",
            "target_id",
            "profile_revision",
            "gripper_joint_positions_rad",
            "gripper_configuration_source",
        },
        "request",
    )
    _fields(
        attachment,
        {
            "points_grasp_m",
            "T_grasp_ee_m",
            "T_tcp_ee_m",
            "contact_links",
            "clearance_m",
            "source_observation_id",
            "geometry_coverage",
        },
        "attachment",
    )
    _fields(scene_snapshot, {"collision_world_source", "scene_token"}, "scene_snapshot")
    if (
        scene_snapshot["collision_world_source"] != "esdf"
        or not isinstance(scene_snapshot["scene_token"], str)
        or not scene_snapshot["scene_token"].startswith("fixed-esdf:")
    ):
        raise ValueError("lift validation requires a published ESDF scene token")
    arm = request["arm"]
    _fields(request["gripper_joint_positions_rad"], {"A", "B"}, "gripper_joint_positions_rad")
    for value in request["gripper_joint_positions_rad"].values():
        if _numbers(value, "gripper joint position").shape != ():
            raise ValueError("each gripper driver position must be a scalar")
    if request["gripper_configuration_source"] not in {"profile_expected", "measured_feedback"}:
        raise ValueError("gripper configuration must be profile_expected or measured_feedback")
    if arm not in ("A", "B"):
        raise ValueError("arm must be A or B")
    suffix = "L" if arm == "A" else "R"
    names = [f"Joint{i}_{suffix}" for i in range(1, 8)]
    if request["joint_names"] != names:
        raise ValueError("joint_names must follow the selected arm's Joint1..Joint7 order")
    for name in ("plan_id", "target_id", "profile_revision"):
        if not isinstance(request[name], str) or not request[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if (
        not isinstance(attachment["source_observation_id"], str)
        or not attachment["source_observation_id"].strip()
    ):
        raise ValueError("source_observation_id is required")
    positions = _numbers(request["positions_rad"], "positions_rad")
    times = _numbers(request["times_s"], "times_s")
    peer = _numbers(request["other_joints_rad"], "other_joints_rad")
    if (
        positions.ndim != 2
        or positions.shape[1] != 7
        or not 2 <= len(positions) <= MAX_SAMPLES
        or peer.shape != (7,)
    ):
        raise ValueError("lift requires 2..10000 seven-joint samples and seven peer joints")
    if times.shape != (len(positions),) or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times_s must start at zero and strictly increase once per sample")
    points = _numbers(attachment["points_grasp_m"], "points_grasp_m")
    transform = _numbers(attachment["T_grasp_ee_m"], "T_grasp_ee_m")
    tcp_ee = _numbers(attachment["T_tcp_ee_m"], "T_tcp_ee_m")
    if (
        tcp_ee.shape != (4, 4)
        or not np.allclose(tcp_ee[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(tcp_ee[:3, :3].T @ tcp_ee[:3, :3], np.eye(3), atol=1e-5, rtol=0)
        or not math.isclose(np.linalg.det(tcp_ee[:3, :3]), 1.0, abs_tol=1e-5)
        or np.any(np.abs(tcp_ee[:3, 3]) > 1)
    ):
        raise ValueError("T_tcp_ee_m must be a rigid metre transform")
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not 1 <= len(points) <= 100_000
        or np.any(np.abs(points) > 2)
    ):
        raise ValueError("points_grasp_m requires 1..100000 xyz points within 2 metres")
    if transform.shape != (4, 4):
        raise ValueError("T_grasp_ee_m must be a rigid 4x4 metre transform")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
        or not math.isclose(np.linalg.det(rotation), 1.0, abs_tol=1e-5)
        or np.any(np.abs(transform[:3, 3]) > 1)
    ):
        raise ValueError("T_grasp_ee_m must be a rigid transform with translation within 1 metre")
    clearance = _numbers(attachment["clearance_m"], "clearance_m")
    if clearance.shape != () or not 0.001 <= float(clearance) <= 0.1:
        raise ValueError("clearance_m must be in [0.001, 0.1] metres")
    contacts = attachment["contact_links"]
    allowed = {f"grip{suffix}_Link{i}_{side}" for i in range(1, 5) for side in ("L", "R")}
    if (
        not isinstance(contacts, list)
        or not contacts
        or any(not isinstance(link, str) or link not in allowed for link in contacts)
        or len(set(contacts)) != len(contacts)
    ):
        raise ValueError("contact_links must be unique finger links of the selected gripper")
    if attachment["geometry_coverage"] != "observed_surface_only":
        raise ValueError("only observed_surface_only geometry is supported")
    return {
        "request": {
            **request,
            "positions_rad": positions.tolist(),
            "times_s": times.tolist(),
            "other_joints_rad": peer.tolist(),
            "gripper_joint_positions_rad": {
                arm: float(value) for arm, value in request["gripper_joint_positions_rad"].items()
            },
            "support_contact_policy": policy,
        },
        "scene_snapshot": dict(scene_snapshot),
        "attachment": {
            **attachment,
            "points_grasp_m": points.tolist(),
            "T_grasp_ee_m": transform.tolist(),
            "T_tcp_ee_m": tcp_ee.tolist(),
            "clearance_m": float(clearance),
        },
    }


def binding_digest(payload):
    """Bind every supplied command, time, frozen peer state and geometry to the check."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def points_in_ee_frame(attachment):
    # ^grasp T_ee maps EE -> grasp. Row vectors apply the inverse exactly once.
    transform = np.asarray(attachment["T_grasp_ee_m"], dtype=np.float64)
    points = np.asarray(attachment["points_grasp_m"], dtype=np.float64)
    return (points - transform[:3, 3]) @ transform[:3, :3]


def gripper_robot_config(robot_config, positions_rad, urdf_path):
    """Bind both authored driver joints; retain their URDF mimic chains and spheres."""
    _fields(positions_rad, {"A", "B"}, "gripper_joint_positions_rad")
    root = ET.parse(urdf_path).getroot()
    result = deepcopy(robot_config)
    locks = result.get("robot_cfg", result)["kinematics"]["lock_joints"]
    for arm, name in (("A", "gripL_finger_joint"), ("B", "gripR_finger_joint")):
        value = _numbers(positions_rad[arm], "gripper joint position")
        joint = root.find(f"joint[@name='{name}']")
        limit = None if joint is None else joint.find("limit")
        if limit is None or name not in locks or value.shape != ():
            raise ValueError("authored gripper driver joint and limits are required")
        if not float(limit.attrib["lower"]) <= float(value) <= float(limit.attrib["upper"]):
            raise ValueError("gripper position is outside authored URDF limits")
        locks[name] = float(value)
    return result


def attached_robot_config(robot_config, attachment, arm):
    """Append a fixed link using official configuration fields; retain original balls."""
    points = points_in_ee_frame(attachment)
    lower, upper = points.min(axis=0), points.max(axis=0)
    extent = upper - lower
    divisions = np.maximum(1, np.ceil(extent / 0.025).astype(int))
    if np.prod(divisions) > 4096:
        raise ValueError("observed object bounding box exceeds 4096 collision spheres")
    cell = extent / divisions
    radius = float(np.linalg.norm(cell) / 2 + attachment["clearance_m"])
    spheres = []
    for index in itertools.product(*(range(int(count)) for count in divisions)):
        center = lower + (np.asarray(index) + 0.5) * cell
        spheres.append({"center": center.tolist(), "radius": radius})
    result = deepcopy(robot_config)
    kinematics = result.get("robot_cfg", result)["kinematics"]
    if CARRIED_LINK in kinematics["collision_link_names"]:
        raise ValueError("robot configuration already contains a carried object")
    if any(link not in kinematics["collision_link_names"] for link in attachment["contact_links"]):
        raise ValueError("contact link is absent from the robot collision configuration")
    # Replace the list to avoid mutating YAML aliases such as mesh_link_names.
    kinematics["collision_link_names"] = [*kinematics["collision_link_names"], CARRIED_LINK]
    kinematics["collision_spheres"][CARRIED_LINK] = spheres
    kinematics["tool_frames"] = [*kinematics.get("tool_frames", []), CARRIED_LINK]
    tcp_ee = np.asarray(attachment["T_tcp_ee_m"])
    quaternion_xyzw = Rotation.from_matrix(tcp_ee[:3, :3]).as_quat()
    kinematics.setdefault("extra_links", {})[CARRIED_LINK] = {
        "link_name": CARRIED_LINK,
        "parent_link_name": "TCP_Link_L" if arm == "A" else "TCP_Link_R",
        "joint_name": "gap_carried_joint",
        "joint_type": "FIXED",
        "fixed_transform": [
            *tcp_ee[:3, 3].tolist(),
            float(quaternion_xyzw[3]),
            *quaternion_xyzw[:3].tolist(),
        ],
    }
    kinematics.setdefault("self_collision_ignore", {})[CARRIED_LINK] = list(
        attachment["contact_links"]
    )
    kinematics.setdefault("self_collision_buffer", {})[CARRIED_LINK] = 0.0
    return result, {
        "geometry_coverage": "observed_surface_only",
        "full_object_coverage": False,
        "geometry_model": "filled_observed_aabb_sphere_cover",
        "attachment_sphere_count": len(spheres),
        "clearance_m": attachment["clearance_m"],
        "observed_bounds_ee_m": [lower.tolist(), upper.tolist()],
        "coverage_limitation": "Unobserved object extent is not bounded by the measured cloud.",
    }


def require_scene_binding(payload, actual_token, metadata):
    if (
        payload["scene_snapshot"]["scene_token"] != actual_token
        or payload["request"]["target_id"] != metadata.get("selected_instance_id")
        or payload["attachment"]["source_observation_id"] != metadata.get("observation_set_id")
    ):
        raise CuroboToolError(
            "carried geometry does not match the published scene, target or observation",
            stage="scene",
            code="CUROBO_SCENE_CHANGED",
            status=409,
        )


def collision_samples(positions):
    """Additional collision samples only; no controller command or timestamp is changed."""
    original = np.asarray(positions, dtype=np.float64)
    samples = [original[0]]
    source_indices = [0]
    for index in range(1, len(original)):
        steps = max(
            1,
            int(
                np.ceil(np.max(np.abs(original[index] - original[index - 1])) / MAX_JOINT_STEP_RAD)
            ),
        )
        if len(samples) + steps > MAX_SAMPLES:
            raise ValueError("lift exceeds the 10000 collision-sample capacity")
        for step in range(1, steps + 1):
            samples.append(
                original[index - 1] + (original[index] - original[index - 1]) * step / steps
            )
            source_indices.append(index)
    return np.asarray(samples), source_indices


def checked_lift_result(result, payload):
    """Reject stale/incomplete service proof, including an unbound true boolean."""
    from .curobo_service_client import CuroboServiceError

    request = payload["request"]
    expected = {key: request[key] for key in ("plan_id", "target_id", "profile_revision")}
    expected["scene_token"] = payload["scene_snapshot"]["scene_token"]
    expected["binding_sha256"] = binding_digest(payload)
    expected["robot_config_sha256"] = hashlib.sha256(
        (Path(__file__).parent / "robot_config/marvin_gui_dual.yml").read_bytes()
    ).hexdigest()
    expected["validation_method"] = "public_RobotCollisionChecker_with_fixed_carried_link"
    expected["path_coverage"] = "sampled_linear_joint_segments"
    expected["source_observation_id"] = payload["attachment"]["source_observation_id"]
    expected["max_joint_check_step_rad"] = MAX_JOINT_STEP_RAD
    expected["gripper_joint_positions_rad"] = request["gripper_joint_positions_rad"]
    expected["gripper_configuration_source"] = request["gripper_configuration_source"]
    samples, _ = collision_samples(request["positions_rad"])
    count, invalid = result.get("checked_point_count"), result.get("first_invalid_index")
    support = result.get("support_contact")
    support_valid = support is None
    if request.get("support_contact_policy") is not None:
        from .support_contact import revision

        support_valid = (
            isinstance(support, dict)
            and support.get("policy") == request["support_contact_policy"]
            and isinstance(support.get("plane"), dict)
            and support["plane"].get("revision") == revision(support["plane"])
            and (not result.get("collision_free") or support.get("departure_complete") is True)
        )
    if (
        any(result.get(key) != value for key, value in expected.items())
        or not support_valid
        or type(result.get("collision_free")) is not bool
        or type(count) is not int
        or count != len(samples)
        or result.get("geometry_coverage") != "observed_surface_only"
        or result.get("full_object_coverage") is not False
        or (result["collision_free"] and invalid is not None)
        or (
            not result["collision_free"]
            and (type(invalid) is not int or not 0 <= invalid < len(request["positions_rad"]))
        )
    ):
        raise CuroboServiceError(
            "invalid or stale carried lift validation result",
            code="CUROBO_INVALID_RESULT",
            stage="validation",
        )
    return dict(result)

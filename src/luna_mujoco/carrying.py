"""Measured TCP-frame carrying geometry and scoped collision configuration."""

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from threading import Lock

import numpy as np

from .release_clearance import CLEARANCE_COST_EPSILON

# Keep the existing transport bound compatible with accepted payloads. New
# held models contain only the pregrasp capture's measured points;
# sphere budgeting below covers every return without truncating the surface.
MAX_CARRYING_POINTS = 2 * 3 * 640 * 480
DEFAULT_CARRYING_SPHERES = 24
MAX_CARRYING_SPHERES = 31
_GEOMETRY_CACHE_LIMIT = 32
_geometry_cache = OrderedDict()
_geometry_cache_lock = Lock()


@dataclass(frozen=True)
class CarryingBinding:
    arm: str
    owner_id: str
    observation_id: str
    generation: int
    scene_token: str
    instance_id: str
    frame: str
    state: str
    points_tool_m: tuple
    uncertainty_m: float
    coverage: float | None = None
    measurement_provenance: dict | None = None
    voxel_size_m: float = .01
    margin_m: float = .001
    sphere_budget: int = DEFAULT_CARRYING_SPHERES

    @cached_property
    def _geometry_key(self):
        # Validated bindings own immutable tuples. Hash the cloud once per
        # binding, not once for every placement yaw or collision check.
        points = np.asarray(self.points_tool_m, dtype="<f8")
        digest = hashlib.sha256(points.tobytes()).digest()
        measured_count = (self.measurement_provenance or {}).get("measured_point_count", len(points))
        return digest, self.voxel_size_m, self.margin_m, self.uncertainty_m, measured_count

    @classmethod
    def from_mapping(cls, payload, *, scene, arm):
        if not isinstance(payload, dict):
            raise TypeError("carrying_binding must contain measured holding evidence")
        # Old stored bindings use voxel_size_m for a spatial-split stop size.
        # It is not a voxel grid; accept the accurately named public parameter.
        if "minimum_partition_size_m" in payload:
            payload = dict(payload)
            partition = payload.pop("minimum_partition_size_m")
            if "voxel_size_m" in payload and payload["voxel_size_m"] != partition:
                raise ValueError("conflicting attachment partition sizes")
            payload["voxel_size_m"] = partition
        required = set(cls.__dataclass_fields__) - {"coverage", "measurement_provenance", "voxel_size_m", "margin_m", "sphere_budget"}
        if not required.issubset(payload) or set(payload) - set(cls.__dataclass_fields__):
            raise ValueError("carrying_binding fields are invalid")
        if arm not in ("A", "B") or payload["arm"] != arm:
            raise ValueError("carrying arm does not match request")
        for name in ("owner_id", "observation_id", "scene_token", "instance_id"):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError(f"carrying {name} must be nonempty")
        if type(payload["generation"]) is not int or payload["generation"] < 0:
            raise ValueError("carrying generation is invalid")
        for field in ("scene_token", "observation_id", "generation"):
            if payload[field] != scene.get(field):
                raise ValueError(f"carrying {field} is stale")
        if payload["instance_id"] != scene.get("target_id"):
            raise ValueError("carrying selected instance changed")
        if payload["frame"] != "tool" or payload["state"] not in ("held", "preflight"):
            raise ValueError("carrying must be held or preflight in the TCP tool frame")
        points = np.asarray(payload["points_tool_m"], dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or not 2 <= len(points) <= MAX_CARRYING_POINTS
                or not np.isfinite(points).all()):
            raise ValueError("carrying requires finite measured tool points")
        uncertainty = float(payload["uncertainty_m"])
        coverage = None if payload.get("coverage") is None else float(payload["coverage"])
        if not np.isfinite(uncertainty) or not 0 <= uncertainty <= .1 or (coverage is not None and (not np.isfinite(coverage) or not 0 < coverage <= 1)):
            raise ValueError("carrying uncertainty or coverage is invalid")
        if np.max(np.ptp(points, axis=0)) <= 0 or np.max(np.abs(points)) > 1:
            raise ValueError("carrying measured envelope is empty or out of bounds")
        provenance = payload.get("measurement_provenance")
        if provenance is not None:
            if (not isinstance(provenance, dict) or provenance.get("source") not in (
                    "registered_rgbd_visible_surface", "registered_rgbd_surface_with_explicit_height_completion")
                    or provenance.get("point_count") != len(points)):
                raise ValueError("carrying measurement provenance must match geometry points")
            if provenance["source"] == "registered_rgbd_surface_with_explicit_height_completion":
                measured = provenance.get("measured_point_count")
                completion = provenance.get("geometry_completion", {})
                reference = np.asarray(provenance.get("measured_reference_tool_m"), dtype=float)
                if (type(measured) is not int or not 16 <= measured < len(points)
                        or provenance.get("synthetic_point_count") != len(points) - measured
                        or completion.get("source") != "measured_surface_and_user_height_prior"
                        or completion.get("synthetic_points_are_measured") is not False
                        or reference.shape != (3,) or not np.isfinite(reference).all()
                        or not np.allclose(reference, points[:measured].mean(axis=0), atol=1e-7, rtol=0)):
                    raise ValueError("carrying completion provenance is invalid")
        voxel_size = float(payload.get("voxel_size_m", .01))
        margin = float(payload.get("margin_m", .001))
        budget = payload.get("sphere_budget", DEFAULT_CARRYING_SPHERES)
        if (not np.isfinite([voxel_size, margin]).all() or not .001 <= voxel_size <= .05
                or not 0 <= margin <= .03 or type(budget) is not int or not 1 <= budget <= MAX_CARRYING_SPHERES):
            raise ValueError("attachment partition size, margin or sphere budget is invalid")
        return cls(**(payload | {"points_tool_m": tuple(map(tuple, points.tolist())),
                                 "uncertainty_m": uncertainty, "coverage": coverage,
                                 "voxel_size_m": voxel_size, "margin_m": margin, "sphere_budget": budget}))


def carrying_spheres(binding, *, maximum_spheres=MAX_CARRYING_SPHERES):
    return carrying_geometry(binding, maximum_spheres=maximum_spheres)["spheres"]


def carrying_geometry(binding, *, maximum_spheres=MAX_CARRYING_SPHERES):
    """Cache a bounded visible-surface fit; callers receive independent results."""
    if type(maximum_spheres) is not int or maximum_spheres < 1:
        raise ValueError("maximum_spheres must be a positive integer")
    maximum_spheres = min(maximum_spheres, binding.sphere_budget, MAX_CARRYING_SPHERES)
    key = binding._geometry_key, maximum_spheres
    with _geometry_cache_lock:
        geometry = _geometry_cache.get(key)
        if geometry is None:
            geometry = _fit_carrying_geometry(binding, maximum_spheres)
            _geometry_cache[key] = geometry
            if len(_geometry_cache) > _GEOMETRY_CACHE_LIMIT:
                _geometry_cache.popitem(last=False)
        _geometry_cache.move_to_end(key)
        return deepcopy(geometry)


def _fit_carrying_geometry(binding, maximum_spheres):
    """At most 30 principal-direction splits, retaining every padded point."""
    points = np.asarray(binding.points_tool_m, dtype=float)
    voxel_size = binding.voxel_size_m
    groups = [(points, points.min(axis=0), points.max(axis=0), np.arange(len(points)))]
    while len(groups) < maximum_spheres:
        index = max(range(len(groups)), key=lambda i: round(float(np.max(groups[i][2] - groups[i][1])), 12))
        observed, lower, upper, indices = groups[index]
        if np.max(upper - lower) <= voxel_size:
            break
        # A slanted surface cut along a fixed TCP axis can leave a coarse
        # diagonal group whose covering ball intersects nearby gripper links.
        # Use this group's measured spread; no robot collision pair, point,
        # padding or sphere budget is changed to make the fit clear.
        # Picometre rounding affects partition decisions only. It makes tied
        # symmetric groups stable under harmless floating-point translations;
        # all original coordinates still determine the covering spheres.
        local = np.round(observed - observed[0], 12)
        centered = local - local.mean(axis=0)
        direction = np.linalg.svd(centered, full_matrices=False)[2][0]
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
        projected = centered @ direction
        left = projected < (projected.min() + projected.max()) * .5 - 1e-12
        children = [(observed[mask], indices[mask]) for mask in (left, ~left)]
        groups[index:index + 1] = [(child, child.min(axis=0), child.max(axis=0), child_indices)
                                  for child, child_indices in children]
    spheres = []
    source_counts = []
    measured_count = (binding.measurement_provenance or {}).get("measured_point_count", len(points))
    for observed, lower, upper, indices in groups:
        center = (lower + upper) * .5
        distances = np.linalg.norm(observed - center, axis=1)
        radius = float(distances.max())
        first = observed[int(np.argmax(np.round(distances, 12)))]
        opposite = observed[int(np.argmax(np.round(np.linalg.norm(observed - first, axis=1), 12)))]
        # Keep the original bounding-box sphere if neither candidate improves
        # it. Recheck every point, so tightening never trims measured surfaces.
        for candidate in (observed.mean(axis=0), (first + opposite) * .5):
            candidate_radius = float(np.linalg.norm(observed - candidate, axis=1).max())
            if candidate_radius < radius - 1e-12:
                center, radius = candidate, candidate_radius
        radius = max(.002, radius + binding.uncertainty_m + binding.margin_m)
        spheres.append([*center.tolist(), radius])
        measured = int(np.count_nonzero(indices < measured_count))
        source_counts.append({"measured": measured, "height_completion": len(indices) - measured})
    width = max(voxel_size, max(float(np.max(upper - lower)) for _, lower, upper, _ in groups))
    return {"spheres": spheres, "sphere_count": len(spheres), "sphere_budget": maximum_spheres,
            "minimum_partition_size_m": voxel_size, "maximum_group_extent_m": width,
            "fit_method": "bounded_principal_splits_tight_centers",
            "partition_direction_source": "point_group_principal_component",
            "source_counts_by_sphere": source_counts,
            "margin_m": binding.margin_m, "measurement_uncertainty_m": binding.uncertainty_m}


def carrying_touch_links(arm):
    """Reviewed owning hand assembly only; TCP also contains the camera bracket.

    These pairs allow grasp/release contact with the carried envelope. They do
    not remove any sphere, world check or ordinary robot self-collision pair.
    Explicit names prevent future cameras/accessories from inheriting contact.
    """
    if arm not in ("A", "B"):
        raise ValueError("carrying touch policy requires arm A or B")
    prefix = "gripL_" if arm == "A" else "gripR_"
    return (prefix + "Base", *(prefix + f"Link{part}_{side}"
                              for part in range(1, 5) for side in ("L", "R")))


def create_carrying_robot_config(robot_config, binding):
    """Derive the SDK's documented fixed extra-link configuration per request."""
    result = deepcopy(robot_config)
    kin = result["robot_cfg"]["kinematics"]
    link = f"jev_carried_object_{binding.arm}"
    tool = "TCP_Link_L" if binding.arm == "A" else "TCP_Link_R"
    kin.setdefault("extra_links", {})[link] = {
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_name": f"jev_carry_joint_{binding.arm}",
        "joint_type": "FIXED", "link_name": link, "parent_link_name": tool,
    }
    # YAML aliases may share collision_link_names with mesh_link_names.
    kin["collision_link_names"] = [*kin["collision_link_names"], link]
    kin["collision_spheres"][link] = [{"center": row[:3], "radius": row[3]} for row in carrying_spheres(binding)]
    buffers = kin.setdefault("collision_sphere_buffer", {})
    if isinstance(buffers, dict):
        buffers[link] = 0.
    kin.setdefault("self_collision_buffer", {})[link] = 0.
    ignored = kin.setdefault("self_collision_ignore", {})
    # An attached envelope is part of its owning hand's contact assembly.
    # TCP/camera, upstream/peer arms and all world geometry remain checked.
    contacts = [name for name in carrying_touch_links(binding.arm) if name in kin["collision_link_names"]]
    ignored[link] = contacts
    for contact in contacts:
        ignored.setdefault(contact, []).append(link)
    return result, link


def validate_phase_request(payload):
    phase = payload.get("phase")
    mode = payload.get("gripper_geometry_mode", "actual")
    if mode not in ("actual", "fixed_open"):
        raise ValueError("unknown gripper_geometry_mode")
    if mode == "fixed_open" and (payload.get("carrying_binding") is not None or phase not in ("open_grasp", "approach_grasp", "grasp", "close")):
        raise ValueError("fixed_open requires a detached grasp phase")
    if phase not in ("observation", "open_grasp", "approach_grasp", "grasp", "grasp_lift", "lift", "approach", "place", "transport", "close", "open", "retreat"):
        raise ValueError("unknown carrying validation phase")
    policy = payload.get("support_contact_policy")
    if policy is not None and (phase != "lift" or policy != "observed_plane_normal_departure"):
        raise ValueError("support contact policy applies only to observed normal lift departure")
    carrying = payload.get("carrying_binding")
    if phase == "observation" and carrying is not None:
        raise ValueError("observation requires detached geometry")
    if phase == "retreat" and carrying is not None:
        raise ValueError("retreat requires detached carrying geometry")
    if phase in ("open_grasp", "approach_grasp", "grasp", "grasp_lift") and carrying is not None:
        raise ValueError("grasp phases require detached geometry")
    if phase in ("lift", "approach", "place", "transport", "open") and carrying is None:
        raise ValueError(f"{phase} requires a measured carrying binding")
    if phase == "transport" and carrying.get("state") != "held":
        raise ValueError("transport requires a held carrying binding")
    if phase in ("close", "open", "open_grasp"):
        feedback = payload.get("gripper_joint_positions_rad")
        if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
            raise ValueError("gripper driver feedback is required")
        values = [*feedback.values(), payload.get("gripper_start_position_rad"), payload.get("gripper_end_position_rad")]
        try:
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise ValueError("gripper driver values must be finite")
        except (ValueError, TypeError) as error:
            raise ValueError("gripper driver values must be finite") from error
    if phase in ("grasp", "close"):
        target_contact_samples(np.eye(4)[None], payload.get("contact_goal_pose_world"))
    if "current_joints_rad" in payload:
        current = np.asarray(payload["current_joints_rad"], dtype=float)
        positions = np.asarray(payload.get("positions_rad"), dtype=float)
        suffix = "L" if payload.get("arm") == "A" else "R"
        expected = [f"Joint{i}_{suffix}" for i in range(1, 8)]
        names = payload.get("joint_names")
        if (current.shape != (7,) or not np.isfinite(current).all() or positions.ndim != 2
                or positions.shape[1] != 7 or not len(positions) or not isinstance(names, (list, tuple))
                or len(names) != 7 or set(names) != set(expected)
                or not np.allclose(positions[0, [names.index(name) for name in expected]], current, atol=1e-5, rtol=0)):
            raise ValueError("current_joints_rad must match the first named actual command")
    return phase


def support_gap_m(binding, plane, tool_matrix):
    balls = np.asarray(carrying_spheres(binding))
    matrix = np.asarray(tool_matrix)
    world = balls[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    table = plane["replacement_cuboid"]
    top = table["pose"][2] + table["dims"][2] / 2
    return float(np.min(world[:, 2] - balls[:, 3]) - top)


def support_prefix_distance(binding, plane, tool_matrix, clearance_m):
    gap = support_gap_m(binding, plane, tool_matrix)
    return 0. if gap > clearance_m else max(0., clearance_m + .003 - gap) / float(plane["normal_xyz"][2])


def support_departure_frame(arm, tool_matrix, normal):
    """Fixed virtual tool whose +Z is the observed world support normal."""
    from scipy.spatial.transform import Rotation
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1.) or normal[2] <= 0:
        raise ValueError("support normal must be a finite upward unit vector")
    tangent = np.array([1., 0, 0])
    tangent -= (tangent @ normal) * normal
    if np.linalg.norm(tangent) < .1:
        tangent = np.array([0., 1., 0]) - normal[1] * normal
    tangent /= np.linalg.norm(tangent)
    world_rotation = np.column_stack((tangent, np.cross(normal, tangent), normal))
    fixed = np.eye(4)
    fixed[:3, :3] = np.asarray(tool_matrix)[:3, :3].T @ world_rotation
    quaternion = Rotation.from_matrix(fixed[:3, :3]).as_quat()
    return {"arm": arm, "link_name": f"jev_support_axis_{arm}",
        "parent_link_name": "TCP_Link_L" if arm == "A" else "TCP_Link_R",
        "fixed_transform": [0., 0., 0., float(quaternion[3]), *quaternion[:3].tolist()],
        "T_tool_frame": fixed.tolist(), "normal_world": normal.tolist()}


def validate_support_goal_projection(start, goal, normal, distance):
    projection = float((np.asarray(goal) - np.asarray(start)) @ np.asarray(normal))
    if not np.isfinite(projection) or projection < distance - 1e-6:
        raise ValueError("lift goal upward support-normal projection does not clear the carried envelope")
    return projection


def canonical_gripper_feedback(feedback):
    """Ignore numerical noise at the authored open stop, not real closure."""
    if feedback is None:
        return None
    if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
        raise ValueError("gripper feedback must contain A and B driver radians")
    return {arm: 0. if abs(float(value)) <= 1e-12 else float(value)
            for arm, value in feedback.items()}


def configure_gripper_feedback(robot_config, feedback, urdf_path):
    """Lock authored drivers at measured radians; SDK resolves their mimics."""
    result = deepcopy(robot_config)
    # Conservative spheres overlap for these opposing meshes, whose actual
    # collision STLs remain separated over the authored driver range.
    kin = result["robot_cfg"]["kinematics"]
    # A nanometre covers the <1e-12 rad open-stop normalization for this
    # sub-metre gripper chain, including mimic joints, on both hands.
    for link, spheres in kin.get("collision_spheres", {}).items():
        if link.startswith(("gripL_", "gripR_")):
            for sphere in spheres:
                sphere['radius'] += 1e-9
    feedback = canonical_gripper_feedback(feedback)
    ignored = kin.setdefault("self_collision_ignore", {})
    for prefix in ("gripL_", "gripR_"):
        for part in ("Link3", "Link4"):
            for side, other in (("L", "R"), ("R", "L")):
                first, second = prefix + part + "_" + side, prefix + part + "_" + other
                if first in kin["collision_link_names"] and second in kin["collision_link_names"]:
                    values = ignored.setdefault(first, [])
                    if second not in values:
                        values.append(second)
    if feedback is None:
        return result
    if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
        raise ValueError("gripper feedback must contain A and B driver radians")
    joints = {joint.get("name"): joint for joint in ET.parse(urdf_path).getroot().findall("joint")}
    for arm, prefix in (("A", "gripL_"), ("B", "gripR_")):
        driver = prefix + "finger_joint"
        value = float(feedback[arm])
        limit = joints[driver].find("limit")
        lower, upper = float(limit.get('lower')), float(limit.get('upper'))
        # MuJoCo's soft limit can overshoot by a few 1e-5 radians at an empty close.
        # Preserve raw feedback upstream; only normalize the authored model lock.
        if not np.isfinite(value) or not lower - 1e-4 <= value <= upper + 1e-4:
            raise ValueError("gripper driver feedback exceeds authored limits")
        result["robot_cfg"]["kinematics"]["lock_joints"][driver] = float(np.clip(value, lower, upper))
    return result




def target_contact_samples(tool_matrices, goal_matrix):
    goal = np.asarray(goal_matrix, dtype=float)
    if (goal.shape != (4, 4) or not np.isfinite(goal).all()
            or not np.allclose(goal[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(goal[:3, :3].T @ goal[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(goal[:3, :3]), 1., atol=1e-5)):
        raise ValueError("grasp contact requires a finite rigid contact_goal_pose_world 4x4 matrix")
    matrices = np.asarray(tool_matrices, dtype=float).reshape(-1, 4, 4)
    distance = np.linalg.norm(matrices[:, :3, 3] - goal[:3, 3], axis=1)
    rotation = matrices[:, :3, :3] @ goal[:3, :3].T
    angle = np.arccos(np.clip((np.trace(rotation, axis1=1, axis2=2) - 1) / 2, -1., 1.))
    return (distance <= .015) & (angle <= np.deg2rad(5))
















def _opening_robot_configs(robot, detached_robot, payload, urdf_path):
    """Check only closed attached and fully open detached endpoint geometry."""
    arm = payload["arm"]
    feedback = payload["gripper_joint_positions_rad"]
    start, end = float(payload["gripper_start_position_rad"]), float(payload["gripper_end_position_rad"])
    if abs(start - float(feedback[arm])) > 1e-9 or end > start + 1e-9:
        raise ValueError("opening driver command differs from measured feedback or closes the hand")
    end_robot = configure_gripper_feedback(detached_robot, {**feedback, arm: end}, urdf_path)
    driver = "gripL_finger_joint" if arm == "A" else "gripR_finger_joint"
    start = robot["robot_cfg"]["kinematics"]["lock_joints"][driver]
    end = end_robot["robot_cfg"]["kinematics"]["lock_joints"][driver]
    proof = {"driver_samples_rad": [start, end], "scope": "closed_attached_and_open_detached_endpoints",
             "continuous_sweep_checked": False, "first_invalid_driver_rad": None,
             "checked_driver_sample_count": 0}
    return [(start, robot, None), (end, end_robot, None)], proof


def check_command_path(runtime, planner, payload, binding):
    """Check the actual arm command against measured obstacles and both robot arms.

    Grasp uses the fully open model; place approach/place use the frozen holding
    closure plus visible-surface spheres; retreat uses open detached geometry.
    """
    import torch
    import yaml
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from tianji_curobo.curobo_robot_config import ARM_CONFIG_SPECS, create_dual_arm_robot_config
    from tianji_curobo.grasp_lift import collision_samples
    from tianji_curobo.grasp_lift_compute import collision_verdict

    phase = validate_phase_request(payload)
    positions = np.asarray(payload.get("positions_rad"), dtype=float)
    names = payload.get("joint_names")
    all_names = ARM_CONFIG_SPECS["A"]["joint_names"] + ARM_CONFIG_SPECS["B"]["joint_names"]
    dual = payload["arm"] == "AB"
    if dual and (phase != "observation" or binding is not None):
        raise ValueError("dual command checking requires detached observation motion")
    expected_names = all_names if dual else ARM_CONFIG_SPECS[payload["arm"]]["joint_names"]
    if (positions.ndim != 2 or positions.shape[1] != len(expected_names) or not 1 <= len(positions) <= 10000
            or not np.isfinite(positions).all() or not isinstance(names, (list, tuple))
            or len(names) != len(expected_names) or set(names) != set(expected_names)):
        raise ValueError("actual command positions and named joint order are invalid")
    if not dual:
        peer = np.asarray(payload.get("other_joints_deg"), dtype=float)
        if peer.shape != (7,) or not np.isfinite(peer).all():
            raise ValueError("stationary arm telemetry is required")
    feedback = payload.get("gripper_joint_positions_rad")
    if feedback is None:
        raise ValueError("measured gripper feedback is required")
    fixed_open = payload.get("gripper_geometry_mode", "actual") == "fixed_open"
    original = yaml.safe_load(runtime._robot_config_path.read_bytes())
    robot = create_dual_arm_robot_config(original, urdf_path=runtime._urdf_path,
                                         asset_root_path=runtime._asset_root_path)
    modeled_feedback = {**feedback, payload['arm']: 0.} if fixed_open else feedback
    robot = configure_gripper_feedback(robot, modeled_feedback, runtime._urdf_path)
    detached_robot = deepcopy(robot)
    if binding is not None:
        robot, _ = create_carrying_robot_config(robot, binding)
    variants, sweep = ([(None, robot, None)], None)
    if phase == "open":
        variants, sweep = _opening_robot_configs(robot, detached_robot, payload, runtime._urdf_path)

    def make_checker(config, *, cache):
        key = json.dumps({"robot": config, "scene": payload["scene_token"],
            "scene_checker": id(planner.scene_collision_checker),
            "clearance": runtime.planning_settings.trajectory_collision_clearance_m}, sort_keys=True)
        checker = runtime._command_checkers.get(key) if cache else None
        if checker is None:
            checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
                robot_config=config, device_cfg=planner.device_cfg,
                scene_collision_checker=planner.scene_collision_checker,
                collision_activation_distance=runtime.planning_settings.trajectory_collision_clearance_m))
            if cache:
                runtime._command_checkers.clear()
                runtime._command_checkers[key] = checker
        return checker
    samples, source_indices = collision_samples(positions[:, [names.index(name) for name in expected_names]])
    both = samples
    if not dual:
        stationary = np.repeat([np.deg2rad(peer)], len(samples), axis=0)
        both = np.concatenate((samples, stationary) if payload["arm"] == "A" else (stationary, samples), axis=1)
    valid = np.ones(len(samples), dtype=bool)
    checked_count = 0
    for angle, exact, envelope in variants:
        checker = make_checker(exact, cache=True)
        world_checker = checker if envelope is None else make_checker(envelope, cache=False)
        if (set(checker.kinematics.joint_names) != set(all_names)
                or world_checker.kinematics.joint_names != checker.kinematics.joint_names):
            raise ValueError("collision checker must retain both robot arm chains")
        if checker.self_collision_cost is None or world_checker.collision_cost is None:
            raise ValueError("required command world/self collision terms are missing")
        ordered = both[:, [all_names.index(name) for name in checker.kinematics.joint_names]]
        with torch.no_grad():
            for offset in range(0, len(ordered), 128):
                chunk = ordered[offset:offset + 128]
                q = torch.as_tensor(chunk[None], **planner.device_cfg.as_torch_dict()).contiguous()
                state = checker.get_kinematics(q)
                world_state = state if envelope is None else world_checker.get_kinematics(q)
                world = world_checker.get_collision_distance(world_state).detach().cpu().numpy()
                world = np.where((world >= 0) & (world <= CLEARANCE_COST_EPSILON), 0., world)
                self_cost = checker.get_self_collision_distance(state.robot_spheres).detach().cpu().numpy()
                bounds = checker.get_bound(q).detach().cpu().numpy()
                valid[offset:offset + len(chunk)] &= collision_verdict(world, self_cost, bounds, len(chunk))
                checked_count += len(chunk)
        if sweep is not None:
            sweep["checked_driver_sample_count"] += 1
            if not valid.all():
                sweep["first_invalid_driver_rad"] = angle
                break
    invalid = next((index for index, free in enumerate(valid) if not free), None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    return {"ok": True, "collision_free": invalid is None, "phase": phase,
            **({"gripper_endpoints": sweep} if sweep is not None else {}),
            "motion_collision_checks_passed": invalid is None,
            "checked_point_count": checked_count, "command_point_count": len(positions),
            "first_invalid_index": None if invalid is None else source_indices[invalid],
            "scene_token": payload["scene_token"], "binding_sha256": digest,
            "attached": binding is not None, "carrying_owner_id": None if binding is None else binding.owner_id,
            "collision_scope": "measured_obstacles_and_both_robot_arms",
            "geometry_coverage": "visible_surface_only", "full_object_coverage": False,
            "gripper_joint_positions_rad": dict(feedback),
            "modeled_gripper_joint_positions_rad": dict(modeled_feedback),
            "gripper_configuration_source": "configured_open_gripper" if fixed_open else "holding_or_open_feedback",
            "failure_reason": "collision_or_joint_bounds" if invalid is not None else None,
            "max_joint_check_step_rad": np.deg2rad(.25),
            "validation_method": "public_RobotCollisionChecker_with_fixed_carried_link"}

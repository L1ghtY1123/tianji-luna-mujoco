"""Synthetic measured-plane support departure through pinned public cuRobo."""

import itertools
import json
import sys

import numpy as np


def main():
    from scipy.spatial.transform import Rotation
    from tianji_curobo.curobo_service import materialize_urdf
    from tianji_curobo.support_contact import SUPPORT_NAME, build_support_plane

    from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity

    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    scene = {"scene_token": "support-test", "observation_id": "synthetic-support", "generation": 1,
             "target_id": "surface", "disabled_obstacle_names": (), "exclusion_state": "held"}
    runtime.geometry_scene = scene
    runtime.geometry_grids = []
    runtime.geometry_model = {"cuboid": {}, "voxel": {}}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    with runtime.carrying_scope(None, {"A": .55, "B": 0.}), runtime.gpu_lock:
        runtime._prepare_scene_cache_unlocked()
        backend = runtime._compound_backend_unlocked("A", .01, .03)
        q = backend.motion_planner.default_joint_state.position.reshape(-1).detach().cpu().numpy()
        position, quaternion = backend._forward_tool_pose(backend._state(np.rad2deg(q)))
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([*quaternion[1:], quaternion[0]]).as_matrix()
    matrix[:3, 3] = position
    lower, upper = [.23, -.02, -.02], [.27, .02, .02]
    corners = np.array(list(itertools.product(*zip(lower, upper)))) @ matrix[:3, :3].T + matrix[:3, 3]
    center = corners.mean(axis=0)
    normal = np.array([.0005, -.0003, 1.]) if "--tilted" in sys.argv else np.array([0., 0., 1.])
    normal /= np.linalg.norm(normal)
    offset = -float(np.min(corners @ normal))
    returns = [[x, y, -(normal[0]*x + normal[1]*y + offset)/normal[2]] for x in np.linspace(center[0] - .065, center[0] + .065, 12)
               for y in np.linspace(center[1] - .065, center[1] + .065, 12)]
    height = max(point[2] for point in returns)
    table = {"pose": [center[0], center[1], height - .019999, 1, 0, 0, 0], "dims": [.14, .14, .04]}
    plane = build_support_plane(returns, normal, offset, table)
    assert plane is not None
    runtime.geometry_scene = scene | {"support_plane": plane}
    runtime.geometry_model = {"cuboid": {SUPPORT_NAME: table}, "voxel": {}}
    binding = {"arm": "A", "owner_id": "support-only", "observation_id": "synthetic-support", "generation": 1,
        "scene_token": "support-test", "instance_id": "surface", "frame": "tool", "state": "held",
        "points_tool_m": [lower, upper], "estimated_bounds_tool_m": {"min_m": lower, "max_m": upper},
        "shape_prior_provenance": {"source": "registered_rgbd_shape_estimate", "reference_observation_id": "synthetic-support",
                                   "instance_id": "surface", "assumptions": ["synthetic box resting on measured plane"]},
        "uncertainty_m": .002, "coverage": .5}
    goal = np.asarray(position).copy()
    goal[2] += .1
    if "--tilted" in sys.argv:
        goal[:2] += [.001, -.002]
    payload = {"arm": "A", "start_joints_deg": np.rad2deg(q).tolist(), "other_joints_deg": np.rad2deg(q).tolist(),
        "goal_position_m": goal.tolist(), "goal_quaternion_wxyz": quaternion.tolist(), "scene_token": "support-test",
        "planning_settings": runtime.planning_settings.model_dump(), "gripper_joint_positions_rad": {"A": .55, "B": 0.},
        "carrying_binding": binding, "support_contact_policy": "observed_plane_normal_departure"}
    try:
        result = run_operation(runtime, "/tools/plan_carried_pose", payload)
    except Exception as error:
        print(json.dumps({"sdk": sdk_identity()[0], "support_fixture_failed": True,
            "error": str(error), "details": getattr(error, "details", None)}, indent=2), flush=True)
        raise
    evidence = result["collision_evidence"]["support_departure_prefix"]
    assert evidence["optimizer_attachment"] is False and evidence["carried_body_guarded"]
    assert result["collision_evidence"]["optimizer_attachment"]
    checked = run_operation(runtime, "/carrying/validate", payload | {"phase": "lift",
        "positions_rad": result["trajectory"]["positions_rad"], "joint_names": result["trajectory"]["joint_names"]})
    assert checked["collision_free"] and checked["support_departure"]["departure_complete"], checked
    print(json.dumps({"sdk": sdk_identity()[0], "support_departure_prefix_optimizer_attachment": False,
        "support_departure_prefix_body_guarded": True, "continuation_optimizer_attachment": True,
        "support_prefix_end_gap_m": evidence["end_support_gap_m"],
        "required_optimizer_clearance_m": evidence["required_optimizer_clearance_m"],
        "tilted_fixture": "--tilted" in sys.argv,
        "support_frame": evidence["support_frame"],
        "user_goal_position_m": goal.tolist(),
        "user_goal_normal_projection_m": evidence["user_goal_normal_projection_m"],
        "native_point_count": len(result["trajectory"]["positions_rad"]),
        "support_proof": checked["support_departure"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Public manager coordinate witness; no task planning or robot execution."""

import json
from pathlib import Path

import numpy as np


def main():
    from scipy.spatial.transform import Rotation
    from tianji_curobo.curobo_service import materialize_urdf

    from luna_mujoco.carrying import CarryingBinding, carrying_spheres
    from luna_mujoco.geometry_service import ROOT, GeometryRuntime, sdk_identity

    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf",
        marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    source = json.loads(Path("/opt/jev/tests/artifacts/task3/attachment-contact-input.json").read_text())["A"]
    scene = {"scene_token": "coordinate-audit", "observation_id": "rgbd-audit", "generation": 1, "target_id": "source"}
    binding = CarryingBinding.from_mapping({"arm": "A", "owner_id": "audit", "observation_id": "rgbd-audit",
        "generation": 1, "scene_token": "coordinate-audit", "instance_id": "source", "frame": "tool", "state": "preflight",
        "points_tool_m": source["points_tool_m"], "estimated_bounds_tool_m": source["estimated_bounds_tool_m"],
        "shape_prior_provenance": {"source": "registered_rgbd_shape_estimate", "reference_observation_id": "rgbd-audit",
                                   "instance_id": "source", "assumptions": ["coordinate-only real RGBD local geometry audit"]},
        "uncertainty_m": .005, "coverage": .5}, scene=scene, arm="A")
    runtime.geometry_scene, runtime.geometry_grids, runtime.geometry_model = scene, [], {"cuboid": {}, "voxel": {}}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    with runtime.carrying_scope(binding, {"A": .55, "B": 0.}), runtime.gpu_lock:
        runtime._prepare_scene_cache_unlocked()
        backend = runtime._backend_unlocked("A")
        backend.attach_measured_carrying(binding, [0.] * 7)
        manager = backend.motion_planner.attachment_manager
        indices = manager.kinematics_params.get_sphere_index_from_link_name("jev_carried_object_A")
        local = manager.kinematics_params.link_spheres[0, indices].detach().cpu().numpy()
        expected = np.asarray(carrying_spheres(binding), dtype=np.float32)
        assert np.array_equal(local, expected), (local, expected)
        state = backend.motion_planner.compute_kinematics(backend._state([0.] * 7))
        pose = state.tool_poses.get_link_pose("TCP_Link_L")
        position = pose.position.detach().cpu().numpy().reshape(3)
        quaternion = pose.quaternion.detach().cpu().numpy().reshape(4)
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        expected_world = expected[:, :3] @ rotation.T + position
        actual_world = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)[indices.detach().cpu().tolist(), :3]
        error = float(np.max(np.abs(actual_world - expected_world)))
        assert error < 2e-6, error
    print(json.dumps({"sdk": sdk_identity()[0], "scope": "coordinate algebra only; not a grasp/holding proof",
        "input_frame": "physical_TCP_link_local", "configured_carried_link_transform": "identity_fixed_TCP_child",
        "world_objects_pose_offset": None, "manager_local_spheres_bitwise_equal_input": True,
        "maximum_world_fk_center_error_m": error, "sphere_count": len(expected)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Exercise held-source coverage using recorded real RGBD and pinned GPU FK."""
import json
from dataclasses import asdict

import numpy as np


def main():
    import torch
    from curobo.types import JointState
    from tianji_curobo.curobo_service import materialize_urdf

    from jev_mujoco.carrying import CarryingBinding
    from jev_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity

    directory = ROOT / "tests/artifacts/task3"
    payload = json.loads((directory / "real-rgbd-map-request.json").read_text())
    payload.pop("selected_object_estimate", None)
    observation = json.loads((directory / "real-rgbd-observation.json").read_text())
    measured = np.load(directory / "real-rgbd-source-points.npz")["points_world_m"]
    payload["selected_surface_world_m"] = measured.tolist()
    payload["exclusion_state"] = "held"
    urdf = materialize_urdf(ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf", marvin_meshes=ROOT / "assets/marvin_urdf/meshes", rag_meshes=ROOT / "assets/marvin_urdf/rag100_050/meshes")
    runtime = GeometryRuntime(urdf_path=urdf, asset_root_path=urdf.parent)
    from copy import deepcopy

    from tianji_curobo.semantic_esdf_transport import (
        decode_semantic_esdf_request,
        encode_semantic_esdf_request,
    )
    transport = {key: payload[key] for key in ("observation_set_id", "sam3_generation", "confirmed_instance_id", "selection_generation", "views")}
    head = decode_semantic_esdf_request(transport)
    for camera in ("1", "2"):
        head["views"][camera]["depth_left_m"][:] = 0
        head["views"][camera]["valid_left_mask"][:] = 0
    head_payload = deepcopy(payload) | encode_semantic_esdf_request(head) | {"exclusion_state": "observation"}
    head_scene = run_operation(runtime, "/perception/build", head_payload)
    assert list(head_scene["camera_metrics"]) == ["3"]
    assert head_scene["excluded_object_ids"] == []
    assert runtime.geometry_model["voxel"], "native background ESDF layer must remain"
    print(json.dumps({"head_only_actual_rgbd_map_built": True, "layers": head_scene["layer_count"], "native_camera_ids": list(head_scene["camera_metrics"])}), flush=True)
    scene = run_operation(runtime, "/perception/build", payload)
    initial = observation["arms"]["A"]["joints_rad"]
    with runtime.carrying_scope(None, payload["gripper_joint_positions_rad"]), runtime.gpu_lock:
        runtime._prepare_scene_cache_unlocked()
        backend = runtime._compound_backend_unlocked("A", .01, .03)
        model = backend.motion_planner
        state = JointState.from_position(torch.as_tensor([initial], **model.device_cfg.as_torch_dict()), joint_names=model.joint_names)
        pose = model.compute_kinematics(state).tool_poses.get_link_pose("TCP_Link_L").get_matrix()[0].detach().cpu().numpy()
    local = (measured - pose[:3, 3]) @ pose[:3, :3]
    binding = CarryingBinding.from_mapping({"arm": "A", "owner_id": "recorded-depth-regression", "observation_id": scene["observation_id"], "generation": scene["generation"], "scene_token": scene["scene_token"], "instance_id": scene["target_id"], "frame": "tool", "state": "held", "points_tool_m": local.tolist(), "uncertainty_m": .005}, scene=scene, arm="A")
    base = {"arm": "A", "phase": "place", "positions_rad": [initial], "current_joints_rad": initial, "joint_names": [f"Joint{i}_L" for i in range(1, 8)], "other_joints_deg": np.rad2deg(observation["arms"]["B"]["joints_rad"]).tolist(), "scene_token": scene["scene_token"], "planning_settings": payload["planning_settings"], "gripper_joint_positions_rad": payload["gripper_joint_positions_rad"], "carrying_binding": asdict(binding)}
    result = run_operation(runtime, "/carrying/validate", base)
    proofs = result["observation_coverage"]
    count = sum(proof["held_source_contact_sphere_count"] for proof in proofs)
    assert count > 0, result
    assert all(proof["held_source_binding_capture_valid"] for proof in proofs), result
    assert not any(proof["source_contact_intent_certifies_free_space"] for proof in proofs)
    assert result["attached"]
    report = {"sdk": sdk_identity()[0], "fixture": "recorded_actual_rgbd_sam3_surface_with_synthetic_logical_hold", "physical_success_claimed": False, "measured_points": len(measured), "held_source_contact_sphere_count": count, "collision_free": result["collision_free"], "failure_reason": result["failure_reason"], "observation_coverage": proofs}
    runtime.clear_carrying()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

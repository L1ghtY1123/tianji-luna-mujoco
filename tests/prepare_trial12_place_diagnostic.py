"""Create bounded diagnostic packets from recorded RGBD and actual grasp TCP."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.perception_mapping import decode_mapping_payload
from luna_mujoco.placement import carrying_binding, public_observation
from luna_mujoco.verification import verify_gripper_position


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data/visible-cloud-work/grasp-place-left-12"
    recorded = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    stage = json.loads((folder / "stage-1.json").read_text(encoding="utf-8"))
    data = np.load(root / "data/visible-cloud-work/post-grasp-association-12/temporal-inputs.npz", allow_pickle=True)
    before = data["before"].item()
    reconciled_path = root / "data/visible-cloud-work/post-grasp-association-12/runner-replay-observation.npz"
    current = np.load(reconciled_path, allow_pickle=True)["observation"].item()
    old = next(obj for obj in before["objects"] if obj["id"] == recorded["run"]["target_id"])
    source = next(obj for obj in current["objects"] if obj["kind"] == "cube" and obj["color"] == "red")
    bins = [obj for obj in current["objects"] if obj["kind"] == "bin" and obj["color"] == "blue"]
    head_bin = next(obj for obj in bins if "3" in obj["coverage"]["camera_ids"])
    reference = next(event["data"]["result"] for event in recorded["events"]
                     if event["title"] == "capture_reference" and event["kind"] == "execution_stage_complete")
    grasp, final = np.asarray(reference["tcp_before_lift"]), np.asarray(current["arms"]["A"]["tcp"])
    verification = verify_gripper_position(current["arms"]["A"])
    assert verification["status"] == "held"
    exact = json.loads((folder / "gpu-requests/008-perception-build.json").read_text(encoding="utf-8"))["body"]
    np.testing.assert_array_equal(old["points_world_m"], np.asarray(exact["selected_surface_world_m"], dtype=np.float32))
    settings = stage["run"]["planning_settings"]
    packets = {}
    algorithms = Algorithms.__new__(Algorithms)
    def capture(path, body):
        packets[path] = body
        return decode_mapping_payload(body)[1]
    algorithms.request = capture
    scene = algorithms.build_scene(current, source["id"], planning_settings=settings, exclusion_state="held")
    binding = carrying_binding(current, source["id"], "A", scene["scene_token"], owner_id="diagnostic-recorded-trial12")
    earlier = (np.asarray(old["points_world_m"]) - grasp[:3, 3]) @ grasp[:3, :3]
    after = (np.asarray(source["points_world_m"]) - final[:3, 3]) @ final[:3, :3]
    binding["points_tool_m"] = np.unique(np.vstack((earlier, after)), axis=0).tolist()
    binding.update(settings["attachment"])
    binding["measurement_provenance"].update(point_count=len(binding["points_tool_m"]),
        observation_ids=[before["observation_id"], current["observation_id"]],
        transport="actual_grasp_TCP_and_verified_rigid_hold", diagnostic_only=True)
    payload = Algorithms._motion_request(current, "A", scene, final, settings) | {"carrying_binding": binding}
    holding = deepcopy(binding) | {"verified": True, "attached": False, "verification": verification,
        "observation": public_observation(current) | {"arms": current["arms"]},
        "grasp_support_reference": {"T_world_tcp_at_grasp": grasp.tolist(),
            "table_height_m": before["table"]["surface_height_m"], "source": "recorded_RGBD_table_and_actual_grasp_TCP"}}
    association = json.loads((reconciled_path.parent / "reconciled-association.json").read_text(encoding="utf-8"))
    assumptions = {"diagnostic_only": True, "app_temporal_association_resolved": "mapping" in association,
        "source_current_id": source["id"], "source_prior_id": old["id"], "source_verification": verification,
        "destination_current_head_fragment_id": head_bin["id"], "separate_bin_fragment_ids": [obj["id"] for obj in bins],
        "bin_fragments_merged": True, "reconciled_unique_mapping": association.get("mapping"),
        "bin_identity_selected_by": "reviewed_current_masks_only_temporal_reconciliation",
        "goal_reference_source": "actual_current_merged_floor_reference_from_full_Runner_observe_replay",
        "source_pregrasp_cloud_byte_identical_to_recorded008": True, "oracle_used": False}
    output = root / "data/visible-cloud-work/trial12-place-diagnostic"
    (output / "gpu-requests").mkdir(parents=True, exist_ok=True)
    (output / "stage-1.json").write_text(json.dumps({"holding": holding, "identity_assumptions": assumptions}), encoding="utf-8")
    (output / "gpu-requests/017-perception-build.json").write_text(json.dumps({"path": "/perception/build", "body": packets["/perception/build"]}), encoding="utf-8")
    (output / "gpu-requests/018-carrying-prepare.json").write_text(json.dumps({"path": "/carrying/prepare", "body": payload}), encoding="utf-8")
    (output / "identity-proof.json").write_text(json.dumps(assumptions, indent=2), encoding="utf-8")
    print(json.dumps({"verification": verification, "union_point_count": len(binding["points_tool_m"]),
                      "current_head_bin_reference": head_bin["placement_reference"]["position_world_m"]}, indent=2))


if __name__ == "__main__":
    main()

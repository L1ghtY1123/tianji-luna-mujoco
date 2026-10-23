"""Evidence required to spend another frozen pool after IK or planner exhaustion."""
from copy import deepcopy

import numpy as np

from .algorithms import AlgorithmError, determinate_grasp_failure
from .geometry import TOOL_PROFILE

GRASP_CANDIDATE_COUNTS = (256, 512, 1000)
MAX_SAMPLING_ROUNDS = len(GRASP_CANDIDATE_COUNTS)


def grasp_candidate_count(sampling_round):
    if type(sampling_round) is not int or not 1 <= sampling_round <= MAX_SAMPLING_ROUNDS:
        raise ValueError("invalid_grasp_sampling_round")
    return GRASP_CANDIDATE_COUNTS[sampling_round - 1]


def validate_grasp_budget(grasps, sampling_round):
    count = grasps.get("requested_candidate_count")
    if type(count) is not int or count != grasp_candidate_count(sampling_round):
        raise AlgorithmError("grasp_sampling_candidate_budget_changed")


def grasp_input_identity(grasps):
    fields = ("generation", "observation_id", "arm", "model", "tool_profile_revision",
              "cloud", "T_world_cloud", "sampling_input_digest")
    if any(grasps.get(key) is None for key in fields):
        raise AlgorithmError("grasp_sampling_input_evidence_missing")
    # Budget is validated separately so only the scheduled count can vary.
    validate_grasp_budget(grasps, grasps.get("sampling_round"))
    return deepcopy({key: grasps[key] for key in fields})


def validate_sampling_evaluation(evaluation, snapshot, target, arm, scene, settings, destination, grasps):
    expected = {"arm": arm, "target_id": target, "generation": snapshot["generation"],
                "scene_token": scene["scene_token"], "planning_settings": settings,
                "tool_profile_revision": TOOL_PROFILE["revision"],
                "sampling_round": grasps["sampling_round"], "tested_count": len(grasps["candidates"])}
    if destination is not None:
        expected["destination_id"] = destination
    if any(evaluation.get(key) != value for key, value in expected.items()):
        raise AlgorithmError("IK evaluation is stale or incomplete", code="planning_stale_evaluation")
    for key, checked_arm in (("start_joints_deg", arm), ("other_joints_deg", "B" if arm == "A" else "A")):
        joints = np.asarray(evaluation.get(key), dtype=float)
        current = np.rad2deg(snapshot["arms"][checked_arm]["joints_rad"])
        if joints.shape != (7,) or not np.isfinite(joints).all() or np.max(np.abs(joints - current)) > 1.:
            raise AlgorithmError("IK evaluation start changed", code="planning_start_changed")


def validate_planning_exhaustion(error, evaluation):
    """An error name alone is never evidence that every candidate was exhausted."""
    if error.code != "planning_no_valid_candidate" or not isinstance(error.details, dict):
        return False
    proof = error.details.get("pool_exhaustion")
    if not isinstance(proof, dict) or proof.get("outcome") != "determinate_planner_failure":
        return False
    fields = ("arm", "target_id", "generation", "scene_token", "planning_settings",
              "tool_profile_revision", "sampling_round", "start_joints_deg", "other_joints_deg")
    if any(key not in evaluation or proof.get(key) != evaluation[key] for key in fields):
        return False
    indices = [candidate["candidate_index"] for candidate in evaluation["valid_candidates"]]
    planned = proof.get("candidate_indices")
    failures = error.details.get("failures")
    if (not indices or not isinstance(planned, list) or any(type(index) is not int for index in planned)
            or len(set(indices)) != len(indices) or len(planned) != len(indices) or set(planned) != set(indices)
            or type(proof.get("planned_candidate_count")) is not int
            or proof["planned_candidate_count"] != len(indices) or proof.get("unplanned_candidate_indices") != []
            or not isinstance(failures, list) or len(failures) != len(indices)
            or not all(determinate_grasp_failure(row) for row in failures)):
        return False
    return [row.get("candidate_index") for row in failures] == planned


def round_evidence(grasps, evaluation, sampling_round):
    return {"sampling_round": sampling_round, "requested_candidate_count": grasps["requested_candidate_count"],
            "raw_candidate_count": grasps.get("raw_candidate_count"),
            "generated_candidate_count": grasps.get("downward_filter", {}).get("input_count"),
            "direction_accepted_count": grasps.get("downward_filter", {}).get("accepted_count"),
            "tested_count": evaluation.get("tested_count", len(grasps["candidates"])),
            "valid_count": len(evaluation["valid_candidates"]),
            "batch_success_counts": [batch["success_count"] for batch in evaluation.get("batches", [])]}

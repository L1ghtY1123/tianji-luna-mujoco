"""Evidence required to spend another frozen pool after IK or planner exhaustion."""
from collections import Counter
from copy import deepcopy

import numpy as np

from .algorithms import AlgorithmError, determinate_grasp_failure
from .geometry import TCP_FRAMES, TOOL_PROFILE

GRASP_CANDIDATE_COUNTS = (256, 512, 1000)
MAX_SAMPLING_ROUNDS = len(GRASP_CANDIDATE_COUNTS)


def grasp_candidate_count(sampling_round):
    if type(sampling_round) is not int or not 1 <= sampling_round <= MAX_SAMPLING_ROUNDS:
        raise ValueError("invalid_grasp_sampling_round")
    return GRASP_CANDIDATE_COUNTS[sampling_round - 1]


def validate_grasp_budget(grasps, sampling_round):
    count = grasps.get("requested_candidate_count")
    if grasps.get('source') == 'cad_offline_library':
        if sampling_round != 1:
            raise AlgorithmError('cad_grasp_library_cannot_resample')
        generated = grasps.get('candidate_replay', {}).get('all_generated_candidates',
                                                        grasps.get('all_generated_candidates', []))
        if type(count) is not int or count <= 0 or count != len(generated):
            raise AlgorithmError('cad_grasp_library_candidate_budget_changed')
        return
    if type(count) is not int or count != grasp_candidate_count(sampling_round):
        raise AlgorithmError("grasp_sampling_candidate_budget_changed")


def empty_pool_filter(grasps):
    """Only a complete numeric rejection may consume another frozen pool."""
    direction = grasps.get('downward_filter', {})
    if direction.get('accepted_count') == 0:
        rows = direction.get('candidate_evidence', [])
        generated = grasps.get('candidate_replay', {}).get('all_generated_candidates')
        if (not isinstance(generated, list) or not generated
                or type(grasps.get('raw_candidate_count')) is not int
                or grasps['raw_candidate_count'] != len(generated)
                or any(not isinstance(row, dict) or type(row.get('index')) is not int for row in generated)
                or len({row['index'] for row in generated}) != len(generated)):
            raise AlgorithmError('empty grasp pool lacks original candidate identity evidence')
        if (direction.get('axis') != 'actual_tool_positive_z_in_world'
                or not isinstance(rows, list) or not rows
                or direction.get('input_count') != len(rows)
                or direction.get('rejected_count') != len(rows)
                or any(type(row.get('candidate_index')) is not int
                       or type(row.get('tilt_passed')) is not bool
                       or type(row.get('away_from_robot_passed')) is not bool
                       or (row['tilt_passed'] and row['away_from_robot_passed'])
                       or not all(type(row.get(key)) in (int, float) and np.isfinite(row[key])
                                  for key in ('downward_dot', 'horizontal_away_from_robot_dot'))
                       for row in rows)
                or len(rows) != len(generated)
                or {row['candidate_index'] for row in rows} != {row['index'] for row in generated}):
            raise AlgorithmError('empty grasp pool lacks complete direction filter evidence')
        return 'no_downward_candidate', 'grasp_no_downward_candidate', direction
    filtered = grasps.get('fixed_slot_reach_filter', {})
    rows = filtered.get('candidate_evidence', [])
    if (filtered.get('applied') is not True
            or filtered.get('screening_scope') != 'nominal_transport_wrist_reach'
            or not rows or filtered.get('input_count') != len(rows)
            or filtered.get('rejected_count') != len(rows) or filtered.get('accepted_count') != 0
            or any(row.get('reach_screening_passed') is not False for row in rows)):
        raise AlgorithmError('empty grasp pool lacks complete transport filter evidence')
    return 'no_transport_reachable_candidate', 'grasp_no_transport_reachable_candidate', filtered


def grasp_input_identity(grasps):
    fields = ("generation", "observation_id", "arm", "model", "tool_profile_revision", "tcp_frame",
              "cloud", "T_world_cloud", "sampling_input_digest")
    if grasps.get('source') == 'cad_offline_library':
        fields = ('source', 'generation', 'observation_id', 'arm', 'model', 'tool_profile_revision',
                  'tcp_frame', 'asset_id', 'source_sha256', 'library_sha256', 'T_world_source')
    if any(grasps.get(key) is None for key in fields):
        raise AlgorithmError("grasp_sampling_input_evidence_missing")
    # Budget is validated separately so only the scheduled count can vary.
    validate_grasp_budget(grasps, grasps.get("sampling_round"))
    return deepcopy({key: grasps[key] for key in fields})


def validate_sampling_evaluation(evaluation, snapshot, target, arm, scene, settings, destination, grasps):
    expected = {"arm": arm, "target_id": target, "generation": snapshot["generation"],
                "scene_token": scene["scene_token"], "planning_settings": settings,
                "tool_profile_revision": TOOL_PROFILE["revision"], "tcp_frame": TCP_FRAMES[arm],
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
    # Valid trajectories are checked in joint-travel order, which can differ
    # from their original pool order. Completeness is an ID bijection.
    failed_indices = [row.get('candidate_index') for row in failures]
    return (all(type(index) is int for index in failed_indices)
            and len(set(failed_indices)) == len(planned) and set(failed_indices) == set(planned))


def round_evidence(grasps, evaluation, sampling_round):
    result = {"sampling_round": sampling_round, "requested_candidate_count": grasps["requested_candidate_count"],
            "raw_candidate_count": grasps.get("raw_candidate_count"),
            "generated_candidate_count": grasps.get("downward_filter", {}).get("input_count"),
            "direction_accepted_count": grasps.get("downward_filter", {}).get("accepted_count"),
            "tested_count": evaluation.get("tested_count", len(grasps["candidates"])),
            "valid_count": len(evaluation["valid_candidates"]),
            "batch_success_counts": [batch["success_count"] for batch in evaluation.get("batches", [])]}
    screen = evaluation.get('destination_screening', {})
    if screen.get('applied') is True:
        result['destination_screening'] = {key: deepcopy(screen.get(key)) for key in (
            'scope', 'input_count', 'accepted_count', 'rejected_count', 'attached',
            'trajectory_checked', 'whole_destination_infeasible')}
        result['destination_screening']['rejection_reasons'] = dict(Counter(
            row['reason'] for row in screen.get('candidate_evidence', []) if not row['accepted']))
    return result

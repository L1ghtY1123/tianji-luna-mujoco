"""Measured motion operations; task compilation and graph policy live in runtime."""
import copy
import time
from pathlib import Path

import numpy as np

from ..algorithms import AlgorithmError
from ..demo_sampling import (
    MAX_SAMPLING_ROUNDS,
    empty_pool_filter,
    grasp_input_identity,
    round_evidence,
    validate_grasp_budget,
    validate_planning_exhaustion,
    validate_sampling_evaluation,
)
from ..demo_transfer_verification import (
    build_release_binding,
    freeze_release_execution,
    release_execution_context,
    released_target_map_is_current,
    validate_release_capture,
)
from ..observation_motion import observation_poses, source_view_evidence
from ..placement import observed_object
from ..runner import Runner
from ..skills import Registry
from .container_geometry import container_transport_region
from .evidence import table_release_reference
from .processing import ProcessingSkills
from .return_pose import prepare_empty_hand_motion_scene

DESTINATION_ID = "demo-destination"
ROOT = Path(__file__).resolve().parents[3]


def configured_destination(observation, region, *, identifier=DESTINATION_ID, support_id=None):
    """Bind operator XY intent to a currently measured support height."""
    center = np.asarray(region["center_world_xy_m"], dtype=float)
    polygon = np.asarray(region["region_polygon_world_xy_m"], dtype=float)
    floor = float(observation["table"]["surface_height_m"])
    if (center.shape != (2,) or polygon.shape != (4, 2)
            or not np.isfinite(center).all() or not np.isfinite(polygon).all() or not np.isfinite(floor)):
        raise ValueError("invalid_configured_destination")
    precise = {}
    if region.get('precise_xy'):
        tolerance = region.get('xy_tolerance_m')
        if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('configured_precise_region_requires_xy_tolerance')
        precise = {'precise_xy': True, 'xy_tolerance_m': tolerance}
    constraints = {}
    if region.get('required_robot_configurations'):
        constraints['required_robot_configurations'] = copy.deepcopy(region['required_robot_configurations'])
    return {"id": identifier, "kind": "marker", "color": "unmarked",
        "source": "configured_demo_region", "position_source": "configured_xy_measured_support",
        "position": [*center.tolist(), floor], "visible_centroid_world": [*center.tolist(), floor],
        "roles": ["target_region"], "visual_membership": "excluded",
        "coverage": {"camera_ids": [], "configured_region": True},
        "placement_reference": {**precise, **constraints, "floor_z_m": floor, "position_world_m": [*center.tolist(), floor],
            "region_polygon_world_xy_m": polygon.tolist(), "region_id": "region:" + identifier,
            "support_entity_id": support_id or "table", "observation_id": observation["observation_id"],
            "source": "configured_xy_measured_support"}}


class RobotOperations(ProcessingSkills, Runner):
    visual_plan_source = "responses_program"

    def __init__(self, *args, region=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = Registry(ROOT / "skill_catalog")
        if region is None:
            from ..demo_scene import demo_region
            region = demo_region()
        self.region = copy.deepcopy(region)

    def task_budgets(self, context):
        # Resource bounds must not encode the number of legs in a particular task.
        # The bounded graph controls when captures are requested.
        return {"normal": {"used": context.get("normal_capture_count", 0), "limit": 0},
            "recovery": {"used": context.get("recovery_capture_count", 0), "limit": 0},
            "decision_limit": 0}


    def is_observation_pose(self, context, observation=None):
        poses = observation_poses(context["planning_settings"])
        current = observation or self.simulation.robot_snapshot()
        return all(np.allclose(current["arms"][arm]["joints_rad"], poses[arm],
                               atol=np.deg2rad(1), rtol=0) for arm in ("A", "B"))

    def require_observation_pose(self, context, observation=None):
        if not self.is_observation_pose(context, observation):
            raise ValueError("mapping_requires_observation_pose")

    def require_grasp_view(self, context, observation=None):
        capture = observation if observation is not None else context.get('unpacked_grasp_capture', context.get('observation'))
        current = self.simulation.robot_snapshot()
        if (not capture or not capture.get('observation_id')
                or capture.get('generation') != self.state.get('generation')
                or current.get('generation') != self.state.get('generation')):
            raise ValueError('grasp_view_changed')
        for arm in ('A', 'B'):
            measured = np.asarray(capture.get('arms', {}).get(arm, {}).get('joints_rad'), float)
            live = np.asarray(current.get('arms', {}).get(arm, {}).get('joints_rad'), float)
            if (measured.shape != (7,) or live.shape != (7,) or not np.isfinite(measured).all()
                    or not np.isfinite(live).all() or not np.allclose(live, measured, atol=np.deg2rad(1), rtol=0)):
                raise ValueError('grasp_view_changed')
        product_capture = context.get('unpacked_grasp_capture')
        if product_capture is not None and (
                self.state.get('target_id') != context.get('unpacked_grasp_target_id')
                or product_capture.get('observation_id') != context.get('post_unpack_capture', {}).get('observation_id')
                or capture['observation_id'] != product_capture.get('observation_id')):
            raise ValueError('grasp_view_changed')

    async def build_scene(self, context, exclusion_state):
        if context["observation"].get("checkpoint") != "initial":
            raise ValueError("mapping_requires_observation_capture")
        self.require_grasp_view(context)
        self.require_grasp_view(context, context["observation"])
        return await super().build_scene(context, exclusion_state)

    async def plan_grasp(self, snapshot, target, arm, context):
        self.require_grasp_view(context)
        self.require_grasp_view(context, snapshot)
        capture_id = context.get('observation_pose_capture_id')
        if not capture_id or any(frame.get('observation_id') != capture_id
                for frame in (snapshot, context.get('observation', {}), context.get('scene', {}))):
            raise ValueError('grasp_requires_current_observation_scene')
        planning_snapshot = dict(snapshot, completed_target_ids=list(self.task_progress.completed_ids))
        result = await super().plan_grasp(planning_snapshot, target, arm, context)
        return {**result, "sampling_evidence": copy.deepcopy(context["sampling_evidence"])}

    async def plan_grasp_candidates(self, snapshot, target, arm, context):
        # IK and trajectory exhaustion share the same three bounded candidate pools.
        settings = copy.deepcopy(context["planning_settings"])
        placement = copy.deepcopy(context.get("grasp_placement_request"))
        token = context["scene"]["scene_token"]
        destination = self.state["destination_id"]
        initial_grasps = copy.deepcopy(context["grasps"])
        maximum_rounds = 1 if initial_grasps.get('source') == 'cad_offline_library' else MAX_SAMPLING_ROUNDS
        identity = None
        rounds = []
        trajectory_attempt_count = 0
        last_planning_error = None

        def check_current():
            self.check_generation()
            if (context["scene"]["scene_token"] != token or context["planning_settings"] != settings
                    or context.get("grasp_placement_request") != placement
                    or self.state["arm"] != arm or self.state["target_id"] != target
                    or self.state["destination_id"] != destination):
                raise AlgorithmError("grasp_sampling_input_changed", code="planning_stale_evaluation")

        for sampling_round in range(1, maximum_rounds + 1):
            check_current()
            if sampling_round > 1:
                context["grasps"] = await self.call_algorithm(self.algorithms.generate_grasps,
                    snapshot, target, arm=arm, placement_request=placement, planning_settings=settings,
                    sampling_round=sampling_round, report_progress=True)
                check_current()
            if (type(context["grasps"].get("sampling_round")) is not int
                    or context["grasps"]["sampling_round"] != sampling_round):
                raise AlgorithmError("grasp_sampling_round_changed", code="planning_stale_evaluation")
            validate_grasp_budget(context["grasps"], sampling_round)
            if sampling_round > 1 and grasp_input_identity(context["grasps"]) != identity:
                raise AlgorithmError("grasp_sampling_input_changed", code="planning_stale_evaluation")
            if not context['grasps']['candidates']:
                outcome, error_code, filtered = empty_pool_filter(context['grasps'])
                if identity is None:
                    identity = grasp_input_identity(initial_grasps)
                rounds.append({**round_evidence(context['grasps'], {'valid_candidates': [], 'tested_count': 0},
                                               sampling_round), 'outcome': outcome,
                               'empty_pool_filter': copy.deepcopy(filtered)})
                context['sampling_evidence'] = {'rounds': copy.deepcopy(rounds),
                    'maximum_sampling_rounds': maximum_rounds, 'arm': arm, 'target_id': target,
                    'scene_token': token,
                    'total_requested_candidate_count': sum(row['requested_candidate_count'] for row in rounds),
                    'total_tested_count': sum(row['tested_count'] for row in rounds),
                    'trajectory_attempt_count': trajectory_attempt_count}
                self.emit('grasp_pool_filtered_empty', '本池抓姿全部被几何筛选拒绝，检查剩余采样预算',
                          context['sampling_evidence'])
                if sampling_round == maximum_rounds:
                    self.state['facts']['grasps_ready'] = False
                    raise AlgorithmError('no eligible grasp in the bounded candidate pools',
                        code=error_code, details=context['sampling_evidence'])
                continue
            check_current()
            if sampling_round == 1:
                evaluations = await super().evaluate_grasp_candidates(snapshot, target, arm, context)
            else:
                evaluations = await self.call_algorithm(self.algorithms.evaluate_arms, snapshot, target,
                    context["grasps"], context["scene"], requested_arm=arm,
                    **({"destination_id": destination} if destination else {}),
                    planning_settings=settings, report_progress=True)
            check_current()
            evaluation = evaluations.get(arm, {})
            if not isinstance(evaluation.get("valid_candidates"), list):
                raise AlgorithmError("source IK outcome is unknown", code="invalid_source_ik_evaluation")
            validate_sampling_evaluation(evaluation, snapshot, target, arm, context["scene"],
                                         settings, destination, context["grasps"])
            rounds.append(round_evidence(context["grasps"], evaluation, sampling_round))
            evidence = {"rounds": copy.deepcopy(rounds), "maximum_sampling_rounds": maximum_rounds,
                        "total_requested_candidate_count": sum(row["requested_candidate_count"] for row in rounds),
                        "total_tested_count": sum(row["tested_count"] for row in rounds),
                        "arm": arm, "target_id": target, "scene_token": token}
            context["sampling_evidence"] = evidence
            if evaluation["valid_candidates"]:
                trajectory_attempt_count += 1
                rounds[-1]["outcome"] = "planning"
                evidence.update(trajectory_attempt_count=trajectory_attempt_count, rounds=copy.deepcopy(rounds))
                evaluation["sampling_evidence"] = evidence
                try:
                    result = await self.call_algorithm(self.algorithms.plan_grasp,
                        snapshot, target, arm, evaluations, context["scene"],
                        planning_settings=settings, report_progress=True,
                        **({"destination_id": destination} if destination else {}))
                except AlgorithmError as error:
                    check_current()
                    if not validate_planning_exhaustion(error, evaluation):
                        raise
                    last_planning_error = error
                    rounds[-1].update(outcome="determinate_planner_failure", planner_failure=copy.deepcopy(error.details))
                else:
                    check_current()
                    rounds[-1]["outcome"] = "planned"
                    evidence["rounds"] = copy.deepcopy(rounds)
                    return result
            else:
                screen = evaluation.get('destination_screening', {})
                rounds[-1]['outcome'] = ('no_sampled_destination_endpoint'
                    if screen.get('applied') is True and screen.get('input_count', 0) > 0
                    else 'no_eligible_source_ik')
            validate_sampling_evaluation(evaluation, snapshot, target, arm, context["scene"],
                                         settings, destination, context["grasps"])
            evidence.update(trajectory_attempt_count=trajectory_attempt_count, rounds=copy.deepcopy(rounds))
            if identity is None:
                identity = grasp_input_identity(initial_grasps)
                if placement is None or initial_grasps.get("sampling_round") != 1:
                    raise AlgorithmError("grasp_sampling_input_evidence_missing")
            if grasp_input_identity(context["grasps"]) != identity:
                raise AlgorithmError("grasp_sampling_input_changed", code="planning_stale_evaluation")
            if sampling_round == maximum_rounds:
                # Exhausted candidates remain archived, but cannot be retried as
                # a fresh first pool. The model selects the next recovery action.
                self.state['facts']['grasps_ready'] = False
                if last_planning_error is not None:
                    raise AlgorithmError("no valid three-phase grasp plan in the bounded candidate pools",
                        code="planning_no_valid_candidate", details={**last_planning_error.details,
                            "sampling_evidence": copy.deepcopy(evidence)}) from last_planning_error
                if any(row.get('outcome') == 'no_sampled_destination_endpoint' for row in rounds):
                    raise AlgorithmError('source IK succeeded but bounded destination endpoint screening was exhausted',
                        code='grasp_destination_samples_exhausted', details={**evidence,
                            'trajectory_attempt_count': 0, 'whole_destination_infeasible': False})
                raise AlgorithmError("no eligible source grasp target in the bounded candidate pools",
                    code="no_collision_free_source_ik", details={**evidence, "trajectory_attempt_count": 0})


    async def capture_scene_observation(self, context, arguments):
        capture = context.get('unpacked_grasp_capture')
        if capture is None:
            capture = context.pop('fine_capture', None)
        if arguments.get('capture_mode') == 'fresh':
            capture = None
        return await self.capture_observation(context, capture=capture)

    async def execute(self, skill, context, arguments=None):
        if skill == 'prepare_unpack':
            return await self._prepare_unpack(context)
        if skill == 'vla_unpack':
            return await self._vla_unpack(context)
        if skill == 'verify_unpack':
            return await self._verify_unpack(context)
        if skill in {"move_to_observe", "return_to_initial_pose"}:
            if skill == 'return_to_initial_pose':
                from .return_pose import return_to_initial_pose
                return await return_to_initial_pose(self, context)
            prepare_empty_hand_motion_scene(context)
            from ..observation_motion import observation_viewpoint_poses

            viewpoint = (arguments or {}).get('viewpoint', 'configured')
            if viewpoint == 'configured':
                return {**await self.move_to_observe(context), 'viewpoint': viewpoint}
            goals = observation_viewpoint_poses(context['planning_settings'], viewpoint)
            if viewpoint != 'configured':
                feedback = self.simulation.robot_snapshot()['arms']
                if all(np.max(np.abs(np.asarray(feedback[arm]['joints_rad']) - goals[arm]))
                       <= np.deg2rad(.05) for arm in ('A', 'B')):
                    raise ValueError('observation_viewpoint_unchanged')
            result = await self.move_to_observe(context, joint_goals=goals)
            return {**result, 'viewpoint': viewpoint, 'requested_joints_rad': goals}
        if skill == "observe_scene":
            capture = await self.capture_scene_observation(context, arguments or {})
            self.require_grasp_view(context, capture)
            result = await self.observe(context, checkpoint="initial", capture=capture)
            context["observation_pose_capture_id"] = result["observation_id"]
            context["observation_pose_reference"] = {
                key: copy.deepcopy(result[key]) for key in ("observation_id", "time", "arms")}
            self._check_source_view(result)
            self._require_visible_inventory(result)
            return result
        if skill == "build_scene":
            source = observed_object(context["observation"], self.state["target_id"])
            evidence = source_view_evidence(context["observation"], source)
            if not evidence["views_ready"]:
                raise ValueError("current_target_geometry_required")
            context["view_evidence"] = evidence
            self.state["facts"]["views_ready"] = True
            return await self.build_scene(context, "selected")
        if skill == "generate_grasps":
            self.require_grasp_view(context)
            if not self.state["facts"].get("views_ready"):
                raise ValueError("move_to_observe_required")
            observation = context["observation"]
            destination = observed_object(observation, self.state['destination_id'])
            if destination.get('kind') == 'bin':
                context['grasp_placement_request'] = {'object_id': self.state['target_id'],
                    'transport_region': container_transport_region(destination, context['scene'], observation)}
            else:
                context['grasp_placement_request'] = copy.deepcopy(self._bind_region(observation))
            context["grasps"] = await self.call_algorithm(self.algorithms.generate_grasps,
                observation, self.state["target_id"], arm=self.state["arm"],
                placement_request=context["grasp_placement_request"],
                planning_settings=context["planning_settings"], report_progress=True)
            context["grasps_observation_id"] = observation["observation_id"]
            return context["grasps"]
        if skill == "execute_place":
            result = await super().execute(skill, context, arguments)
            context["demo_place_execution"] = copy.deepcopy(result)
            context['release_execution_evidence'] = freeze_release_execution(self.state, context, self.run_id)
            self.holding_state = None
            context.pop('holding', None)
            return result
        if skill == "verify_place":
            target = self.state["target_id"]
            previous = context.get('release_verification_attempt')
            if previous and previous['target_id'] == target:
                carried = copy.deepcopy(previous['carried'])
                result = copy.deepcopy(previous['physical_release'])
            else:
                carried = copy.deepcopy(release_execution_context(self.state, context, self.run_id).get('holding'))
                result = await self.verify_place(context)
                if not result["success"]:
                    return result
                result["verified_perf_counter_ns"] = time.perf_counter_ns()
                self.task_progress.record_release(target, feedback_id=result["feedback_id"])
                context['release_verification_attempt'] = {
                    'target_id': target, 'carried': copy.deepcopy(carried),
                    'physical_release': copy.deepcopy(result)}
            # Rechecks retain the original physical release. A view change
            # cannot create a new release or erase the held/attachment proof.
            context['holding'] = carried
            binding = build_release_binding(self.state, context, self.task_progress, self.run_id, result)
            self.emit("release_verification_binding", "放置结果核验的执行链与槽位", binding)
            self.holding_state = None
            context.pop("holding", None)
            from .observation import reusable_release_capture

            reused = reusable_release_capture(context, self.simulation.robot_snapshot(),
                epoch=getattr(self, 'observation_epoch', None), binding=binding)
            capture = await self.capture_observation(context, capture=reused)
            if reused is not None:
                self.emit('observation_reused', '复用释放后的当前观测图像', {
                    'observation_id': capture['observation_id'], 'purpose': 'verify_transfer',
                    'reason': 'unchanged_post_release_capture', 'reused': True})
            validate_release_capture(capture, binding)
            context.update(post_place_capture=capture, latest_capture=capture)
            self.record_observation(capture, "post_place")
            self.sync_result_evidence(context, "post_place")
            after = await self.call_algorithm(self.algorithms.verify_release, capture, binding,
                context["visual_inventory_reference"], self.state["instruction"])
            if after.get('observation_assessment', {}).get('needs_observation'):
                self.emit('visual_transfer_result', 'Decisions：放置核验需要补充观测', after)
                context['use_retained_scene_for_return'] = True
                after['observation_assessment']['retry_owner'] = 'policy_graph'
            self.check_generation()
            current = build_release_binding(self.state, {**context, "holding": carried},
                                            self.task_progress, self.run_id, result)
            if current != binding or after.get("binding_id") != binding["binding_id"]:
                raise ValueError("release_verification_binding_changed")
            evidence = after["visual_evidence"]
            if (after.get("observation_id") != capture["observation_id"]
                    or evidence.get("observation_id") != capture["observation_id"]
                    or evidence.get("binding_id") != binding["binding_id"]
                    or evidence.get("release_feedback_id") != result["feedback_id"]
                    or evidence.get("target_id") != target
                    or evidence.get("destination_id") != binding["destination_id"]
                    or evidence.get("layout_revision") != binding["layout_revision"]
                    or evidence.get("identity_source") != "execution_chain"
                    or evidence.get("source") != "openai_decisions"):
                raise ValueError("visual_inventory_progress_frame_mismatch")
            confidence = evidence.get("confidence")
            if (type(confidence) not in (int, float) or not 0. <= confidence <= 1.
                    or evidence.get("status") == "at_destination" and confidence < .8):
                raise ValueError("release_verification_result_confidence")
            context["post_place_result"] = after
            context["visual_result"] = copy.deepcopy(evidence)
            self.state["visual_result"] = copy.deepcopy(evidence)
            context["post_place_verification_input"] = {
                "observation_id": capture["observation_id"],
                "visual_observation_id": evidence["observation_id"], "binding_id": binding["binding_id"]}
            self.sync_result_evidence(context, "post_place")
            self.emit("visual_transfer_result", "Decisions 放置结果核验", after)
            confirmed = self.task_progress.confirm_visual_transfer(target, evidence)
            if confirmed:
                from ..demo_completed_tracking import confirmed_capture_record

                self.task_progress.records[target]["completed_identity_track"] = {
                    "binding": copy.deepcopy(binding), "visual_evidence": copy.deepcopy(evidence),
                    "capture": confirmed_capture_record(capture)}
            outside = evidence.get('measured_object_position', {}).get('status') == 'outside'
            rejected = evidence.get('status') == 'rejected'
            result.update(success=confirmed, status="confirmed" if confirmed else "failure" if outside or rejected else "unknown",
                verification_scope="release_and_visual_destination", visual_evidence=evidence,
                reason=None if confirmed else "measured_object_outside_placement_region" if outside
                else "visual_placement_rejected" if rejected
                else "visual_destination_unconfirmed")
            result['observation_assessment'] = copy.deepcopy(after.get('observation_assessment', {}))
            if 'pending_transfer' in context:
                context['pending_transfer']['verified'] = confirmed
            context["place_verification"] = result
            context["demo_inventory_check"] = after.get("demo_inventory_check", {})
            if confirmed:
                context['post_place_verification_input'] = table_release_reference(capture, evidence, self.run_id)
                context['verified_release_reference'] = copy.deepcopy(
                    context['release_execution_evidence']['release_reference']
                    if 'release_execution_evidence' in context else context['place_plan']['goal']['release_reference'])
                self.state['facts']['table_release_verified'] = (
                    evidence.get('destination_id') == self.task_inventory.get('work_area_id'))
                self.sync_inventory(context)
                if not carried:
                    raise ValueError("release_geometry_missing")
                if not released_target_map_is_current(context, target):
                    if context['scene']['scene_token'] == binding['scene_token']:
                        context["scene"] = await self.call_algorithm(
                            self.algorithms.publish_released_object, context["scene"], target,
                            carried["grasp_support_reference"]["T_world_tcp_at_grasp"],
                            binding['release_tcp'], capture["observation_id"],
                            support_floor_z_m=context["destination"]["placement_reference"]["floor_z_m"])
                    else:
                        context['scene'] = await self.call_algorithm(self.algorithms.invalidate_scene_entity,
                            context['scene'], target, result['feedback_id'], source='verified_release_action')
                    context['scene_refresh_required'] = True
                    lifecycle = context.setdefault('mapping_lifecycle', {})
                    lifecycle['invalidated_entity_ids'] = sorted(
                        set(lifecycle.get('invalidated_entity_ids', ())) | {target})
                    self.state['facts'].update(scene_available=False, scene_ready=False, observation_map_ready=False)
                    self.emit("retained_scene_invalidated", "释放后几何失效，等待模型选择实测建图", {
                        "scene_token": context["scene"]["scene_token"], "target_id": target,
                        "source": "verified_release_action", "rgbd_remapped": False})
            return result
        return await super().execute(skill, context, arguments)



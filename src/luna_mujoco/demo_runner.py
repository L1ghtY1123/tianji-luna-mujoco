"""Execute a bounded Responses skill program against measured robot state."""
import asyncio
import copy
import time
from pathlib import Path

import numpy as np

from .algorithms import AlgorithmError
from .demo_lifecycle_perception import table_release_reference
from .demo_process import (
    ProcessTaskProgress,
    completed_input_ids,
    current_process_entity,
    initialize_processes,
)
from .demo_program import DemoProgram, ForEachTarget, IfResult, SkillCall, Stop
from .demo_sampling import (
    MAX_SAMPLING_ROUNDS,
    grasp_input_identity,
    round_evidence,
    validate_grasp_budget,
    validate_planning_exhaustion,
    validate_sampling_evaluation,
)
from .demo_transfer_verification import build_release_binding, validate_release_capture
from .demo_unpack_workflow import UnpackWorkflow, right_bin_transport_region
from .inventory_order import robot_distance_order
from .observation_motion import observation_poses, source_view_evidence
from .placement import observed_object
from .runner import Runner
from .skills import Registry
from .visual_assistance import VisualCallBudget

DESTINATION_ID = "demo-destination"
ROOT = Path(__file__).resolve().parents[2]


def configured_destination(observation, region):
    """Bind operator XY intent to a currently measured support height."""
    center = np.asarray(region["center_world_xy_m"], dtype=float)
    polygon = np.asarray(region["region_polygon_world_xy_m"], dtype=float)
    floor = float(observation["table"]["surface_height_m"])
    if (center.shape != (2,) or polygon.shape != (4, 2)
            or not np.isfinite(center).all() or not np.isfinite(polygon).all() or not np.isfinite(floor)):
        raise ValueError("invalid_configured_destination")
    return {"id": DESTINATION_ID, "kind": "marker", "color": "unmarked",
        "source": "configured_demo_region", "position_source": "configured_xy_measured_support",
        "position": [*center.tolist(), floor], "visible_centroid_world": [*center.tolist(), floor],
        "roles": ["target_region"], "visual_membership": "excluded",
        "coverage": {"camera_ids": [], "configured_region": True},
        "placement_reference": {"floor_z_m": floor, "position_world_m": [*center.tolist(), floor],
            "region_polygon_world_xy_m": polygon.tolist(), "region_id": "region:" + DESTINATION_ID,
            "support_entity_id": "table", "observation_id": observation["observation_id"],
            "source": "configured_xy_measured_support"}}


class DemoRunner(UnpackWorkflow, Runner):
    visual_plan_source = "responses_program"

    def __init__(self, *args, region=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = Registry(ROOT / "demo_skills")
        if region is None:
            from .demo_scene import demo_region
            region = demo_region()
        self.region = copy.deepcopy(region)
        self.demo_task = None
        self.demo_program = None
        self.program_steps = 0
        self.last_program_result = None

    def task_budgets(self, context):
        # The program can process any frozen count; observation cost scales with it.
        targets = len((self.demo_task or {}).get('transport_ids', []))
        captures_per_target = 8 if self.unpack_task else 4
        return {"normal": {"used": context.get("normal_capture_count", 0), "limit": 1 + captures_per_target * targets},
            "recovery": {"used": context.get("recovery_capture_count", 0), "limit": 0},
            "decision_limit": 0}

    async def observe(self, context, *, checkpoint=None, capture=None, camera_ids=None,
                      purpose=None, recovery=False):
        checkpoint = checkpoint or "initial"
        fixed = bool(self.task_progress.scene_inventory_observation_id)
        context["visual_plan"] = {checkpoint: ("verify_transfer" if checkpoint == "post_place" else
            "associate_inventory") if fixed else "initialize_inventory"}
        # Existing execution primitives own capture/version checks. Geometry purpose
        # makes the three-camera policy explicit even on the first inventory.
        observation = await super().observe(context, checkpoint=checkpoint, capture=capture,
            camera_ids=["1", "2", "3"], purpose="geometry", recovery=recovery)
        context["visual_inventory_reference"]["owner_id"] = self.run_id
        destination = configured_destination(observation, self.region)
        observation["objects"] = [obj for obj in observation["objects"] if obj["id"] != DESTINATION_ID]
        observation["objects"].append(destination)
        self.task_progress.scene_entities[DESTINATION_ID] = copy.deepcopy(destination)
        context["observation"] = observation
        context["inventory_observation"] = copy.deepcopy(observation)
        if not self.demo_task:
            self.demo_task = copy.deepcopy(observation.get("demo_task"))
            if not self.demo_task:
                raise ValueError("demo_task_missing_from_discovery")
            self.demo_program = DemoProgram.model_validate(observation.get("demo_program"))
            if self.demo_task.get('workflow') == 'unpack_transfer':
                from .demo_program import order_unpack_observation

                self.demo_program, changes = order_unpack_observation(self.demo_program)
                if changes:
                    self.emit('program_dependency_ordered', '按技能前提将黑件观测移到目标选择之前', {
                        'changes': changes, 'added_calls': [], 'removed_calls': []})
            self.state["demo_program"] = self.demo_program.model_dump()
            self.emit("skill_program", "Responses 生成的技能程序", self.state["demo_program"])
        return observation

    async def _step(self, name, context, arguments=None):
        self.check_generation()
        self.check_log()
        self.registry.validate_selection(name, self.state["facts"])
        self.begin_step(name)
        started = time.monotonic()
        self.emit("skill_start", name, {"arguments": arguments or {},
            "target_id": self.state.get("target_id"), "process_id": self.state.get('process_id'),
            "controller": "responses_program"})
        result = await self.execute(name, context, arguments or {})
        self.check_generation()
        self.check_log()
        for fact in self.registry.contracts[name].provides:
            self.state["facts"][fact] = bool(result.get("success", True)) if name.startswith("verify_") else True
        self.emit("skill_complete", name, {"result": result, "duration_s": time.monotonic() - started})
        self.state['task_progress'] = self.task_progress.snapshot()
        if (self.task_progress.inventory_initialized
                and name in {'bind_inventory', 'observe_scene', 'verify_place', 'verify_unpack',
                             'verify_drop', 'verify_grasp'}):
            calls = [event for event in self.events if event['kind'] == 'visual_decisions_call'
                     and event.get('invocation_id') == self.state['current_step']['invocation_id']]
            if not calls and name == 'bind_inventory':
                calls = [event for event in self.events if event['kind'] == 'visual_decisions_call'
                         and event['data'].get('observation_id') == context.get('observation', {}).get('observation_id')]
            observation_id = (calls[-1]['data'].get('observation_id') if calls else
                              context.get('observation', {}).get('observation_id'))
            self.emit('inventory_snapshot', '本节点清单', {'checkpoint': name,
                'observation_id': observation_id, 'model_event_ids': [event['id'] for event in calls],
                'evidence_source': 'gripper_feedback' if name == 'verify_grasp' else 'visual_checkpoint',
                'task_progress': self.state['task_progress']})
        return result

    def require_observation_pose(self, context, observation=None):
        poses = observation_poses(context["planning_settings"])
        current = observation or self.simulation.robot_snapshot()
        if not all(np.allclose(current["arms"][arm]["joints_rad"], poses[arm],
                               atol=np.deg2rad(1), rtol=0) for arm in ("A", "B")):
            raise ValueError("mapping_requires_observation_pose")

    def require_grasp_view(self, context, observation=None):
        self.require_observation_pose(context, observation)
        capture = context.get('unpacked_grasp_capture')
        if capture is None:
            return
        current = observation or self.simulation.robot_snapshot()
        if (not self.unpack_task or self.state.get('target_id') != context.get('unpacked_grasp_target_id')
                or self.current_lifecycle()['phase'] != 'unpacked_confirmed'
                or capture.get('generation') != self.state.get('generation')
                or capture.get('observation_id') != context.get('post_unpack_capture', {}).get('observation_id')
                or observation is not None and observation.get('observation_id') != capture.get('observation_id')
                or not all(np.allclose(current['arms'][arm]['joints_rad'], capture['arms'][arm]['joints_rad'],
                                       atol=np.deg2rad(1), rtol=0) for arm in ('A', 'B'))):
            raise ValueError('grasp_view_changed')

    async def build_scene(self, context, exclusion_state):
        self.require_grasp_view(context)
        self.require_grasp_view(context, context["observation"])
        if context["observation"].get("checkpoint") != "initial":
            raise ValueError("mapping_requires_observation_capture")
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

    async def plan_grasp_candidates(self, snapshot, target, arm, context, deadline):
        # IK and trajectory exhaustion spend the same three pools and deadline.
        settings = copy.deepcopy(context["planning_settings"])
        placement = copy.deepcopy(context.get("grasp_placement_request"))
        token = context["scene"]["scene_token"]
        destination = self.state["destination_id"]
        initial_grasps = copy.deepcopy(context["grasps"])
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
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f'plan_grasp_timeout:{settings["plan_timeout_s"]:g}s')
            return remaining

        for sampling_round in range(1, MAX_SAMPLING_ROUNDS + 1):
            remaining = check_current()
            if sampling_round > 1:
                context["grasps"] = await self.call_algorithm(self.algorithms.generate_grasps,
                    snapshot, target, arm=arm, placement_request=placement, planning_settings=settings,
                    sampling_round=sampling_round, report_progress=True, timeout_s=remaining)
                check_current()
            if (type(context["grasps"].get("sampling_round")) is not int
                    or context["grasps"]["sampling_round"] != sampling_round):
                raise AlgorithmError("grasp_sampling_round_changed", code="planning_stale_evaluation")
            validate_grasp_budget(context["grasps"], sampling_round)
            if sampling_round > 1 and grasp_input_identity(context["grasps"]) != identity:
                raise AlgorithmError("grasp_sampling_input_changed", code="planning_stale_evaluation")
            if not context['grasps']['candidates']:
                filtered = context['grasps'].get('fixed_slot_reach_filter', {})
                rows = filtered.get('candidate_evidence', [])
                if (filtered.get('applied') is not True
                        or filtered.get('screening_scope') != 'nominal_transport_wrist_reach'
                        or not rows or filtered.get('input_count') != len(rows)
                        or filtered.get('rejected_count') != len(rows) or filtered.get('accepted_count') != 0
                        or any(row.get('reach_screening_passed') is not False for row in rows)):
                    raise AlgorithmError('empty grasp pool lacks complete transport filter evidence')
                if identity is None:
                    identity = grasp_input_identity(initial_grasps)
                rounds.append({**round_evidence(context['grasps'], {'valid_candidates': [], 'tested_count': 0},
                                               sampling_round), 'outcome': 'no_transport_reachable_candidate',
                               'transport_reach_filter': copy.deepcopy(filtered)})
                context['sampling_evidence'] = {'rounds': copy.deepcopy(rounds),
                    'maximum_sampling_rounds': MAX_SAMPLING_ROUNDS, 'arm': arm, 'target_id': target,
                    'scene_token': token, 'shared_plan_timeout_s': settings['plan_timeout_s'],
                    'total_requested_candidate_count': sum(row['requested_candidate_count'] for row in rounds),
                    'total_tested_count': sum(row['tested_count'] for row in rounds),
                    'trajectory_attempt_count': trajectory_attempt_count}
                self.emit('grasp_pool_filtered_empty', '本池没有满足投放可达性的抓取，使用剩余采样预算',
                          context['sampling_evidence'])
                if sampling_round == MAX_SAMPLING_ROUNDS:
                    raise AlgorithmError('no transport-reachable grasp in the bounded candidate pools',
                        code='grasp_no_transport_reachable_candidate', details=context['sampling_evidence'])
                continue
            remaining = check_current()
            if sampling_round == 1:
                evaluations = await super().evaluate_grasp_candidates(snapshot, target, arm, context, deadline)
            else:
                evaluations = await self.call_algorithm(self.algorithms.evaluate_arms, snapshot, target,
                    context["grasps"], context["scene"], requested_arm=arm,
                    **({"destination_id": destination} if destination else {}),
                    planning_settings=settings, report_progress=True, timeout_s=remaining)
            check_current()
            evaluation = evaluations.get(arm, {})
            if not isinstance(evaluation.get("valid_candidates"), list):
                raise AlgorithmError("source IK outcome is unknown", code="invalid_source_ik_evaluation")
            validate_sampling_evaluation(evaluation, snapshot, target, arm, context["scene"],
                                         settings, destination, context["grasps"])
            rounds.append(round_evidence(context["grasps"], evaluation, sampling_round))
            evidence = {"rounds": copy.deepcopy(rounds), "maximum_sampling_rounds": MAX_SAMPLING_ROUNDS,
                        "total_requested_candidate_count": sum(row["requested_candidate_count"] for row in rounds),
                        "total_tested_count": sum(row["tested_count"] for row in rounds),
                        "arm": arm, "target_id": target, "scene_token": token,
                        "shared_plan_timeout_s": settings["plan_timeout_s"]}
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
                        **({"destination_id": destination} if destination else {}), timeout_s=check_current())
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
                rounds[-1]["outcome"] = "no_eligible_source_ik"
            validate_sampling_evaluation(evaluation, snapshot, target, arm, context["scene"],
                                         settings, destination, context["grasps"])
            evidence.update(trajectory_attempt_count=trajectory_attempt_count, rounds=copy.deepcopy(rounds))
            if identity is None:
                identity = grasp_input_identity(initial_grasps)
                if placement is None or initial_grasps.get("sampling_round") != 1:
                    raise AlgorithmError("grasp_sampling_input_evidence_missing")
            if grasp_input_identity(context["grasps"]) != identity:
                raise AlgorithmError("grasp_sampling_input_changed", code="planning_stale_evaluation")
            if sampling_round == MAX_SAMPLING_ROUNDS:
                if last_planning_error is not None:
                    raise AlgorithmError("no valid three-phase grasp plan in the bounded candidate pools",
                        code="planning_no_valid_candidate", details={**last_planning_error.details,
                            "sampling_evidence": copy.deepcopy(evidence)}) from last_planning_error
                raise AlgorithmError("no eligible source grasp target in the bounded candidate pools",
                    code="no_collision_free_source_ik", details={**evidence, "trajectory_attempt_count": 0})

    def _require_visible_inventory(self, observation):
        check = observation.get("demo_inventory_check", {})
        if self.unpack_task and check.get('status') == 'not_reassessed':
            visual = observation.get('visual_inventory', {})
            roles = visual.get('lifecycle_correspondence', {})
            coverage, evidence = visual.get('identity_coverage', {}), visual.get('evidence', {})
            required = {key for key, entity in observation.get('scene_entities', {}).items()
                        if entity.get('kind') not in {'table', 'marker'}}
            request = evidence.get('request_hash')
            threshold = coverage.get('minimum_confidence')
            if (not self.task_progress.inventory_finalized or not required or not request
                    or check.get('source') != 'frozen_initial_inventory'
                    or evidence.get('source') != 'openai_decisions'
                    or roles.get('source') != 'current_geometry_unique_lifecycle_role_proposal'
                    or roles.get('owner_id') != self.run_id
                    or roles.get('generation') != observation.get('generation')
                    or roles.get('reference_observation_id') != self.task_progress.scene_inventory_observation_id
                    or set(roles.get('bindings', {})) != {key for key in required
                        if observation['scene_entities'][key].get('kind') != 'bin'}
                    or coverage.get('purpose') != 'associate_inventory'
                    or set(coverage.get('required_entity_ids', [])) != required
                    or not required <= set(coverage.get('accepted_entity_ids', []))
                    or type(threshold) not in (int, float) or not .8 <= threshold <= 1.
                    or visual.get('missing_geometry_ids') != []
                    or any(row.get('observation_id') != observation.get('observation_id')
                           or row.get('request_hash') != request for row in (check, roles, coverage, evidence))):
                raise ValueError('source_inventory_unknown')
            return
        confidence = check.get("confidence")
        if (not observation.get("observation_id")
                or check.get("observation_id") != observation["observation_id"]
                or check.get("source") != "openai_decisions"
                or type(confidence) not in (int, float) or not .8 <= confidence <= 1.
                or check.get("status") != "none"):
            raise ValueError("unlisted_source_object" if check.get("status") == "extra"
                             else "source_inventory_unknown")

    def _resolve_source_inventory(self, observation):
        """Resolve visible completeness from this capture without rewriting Responses."""
        reference = observation.get("visual_reference", {})
        discovery = observation.get("visual_discovery") or {}
        visual = observation.get("visual_inventory", {})
        decisions = visual.get("evidence", {})
        coverage = visual.get("identity_coverage", {})
        check = observation["demo_inventory_check"]
        observation_id = observation["observation_id"]
        required = {identifier for identifier, entity in observation["scene_entities"].items()
                    if entity.get("kind") not in {"table", "marker"}}
        sources = set(self.demo_task["source_object_ids"])
        threshold = coverage.get("minimum_confidence")
        if (reference.get("observation_id") != observation_id
                or discovery.get("source") != "openai_responses"
                or discovery.get("observation_id") != observation_id
                or not discovery.get("request_hash")
                or decisions.get("source") != "openai_decisions"
                or decisions.get("observation_id") != observation_id
                or not decisions.get("request_hash")
                or coverage.get("source") != "openai_decisions"
                or coverage.get("observation_id") != observation_id
                or coverage.get("request_hash") != decisions["request_hash"]
                or check.get("request_hash") != decisions["request_hash"]
                or coverage.get("purpose") != "initialize_inventory"
                or type(threshold) not in (int, float) or not .8 <= threshold <= 1.
                or not required or set(coverage.get("required_entity_ids", [])) != required
                or not required <= set(coverage.get("accepted_entity_ids", []))
                or set(coverage.get("source_object_ids", [])) != sources
                or set(coverage.get("remaining_source_ids", [])) != sources
                or not sources <= set(coverage.get("accepted_left_entity_ids", []))):
            raise ValueError("source_inventory_unknown")
        self._check_source_view(observation)
        return {"status": "confirmed", "scope": "visible_source_inventory",
            "source": "same_capture_decisions_evidence", "observation_id": observation_id,
            "generation": observation.get("generation"),
            "responses_status": self.demo_task["source_inventory_complete"],
            "responses_request_hash": discovery["request_hash"],
            "decisions_request_hash": decisions["request_hash"],
            "required_entity_ids": sorted(required),
            "confirmed_entity_ids": sorted(coverage["accepted_entity_ids"]),
            "left_source_ids": sorted(sources), "demo_inventory_check": copy.deepcopy(check)}

    def _bind_inventory(self, context):
        task, observation = self.demo_task, context["observation"]
        targets = task["transport_ids"]
        count = task["requested_count"]
        if (not targets or len(set(targets)) != len(targets)
                or not set(targets) <= set(task["source_object_ids"])):
            raise ValueError("demo_transport_inventory_invalid")
        self._require_visible_inventory(observation)
        if task["scope"] == "count" and (type(count) is not int or len(targets) != count):
            raise ValueError("requested_count_exceeds_confirmed_inventory")
        if task["scope"] == "all" and set(targets) != set(task["source_object_ids"]):
            raise ValueError("incomplete_all_transport_inventory")
        resolution = self._resolve_source_inventory(observation) if task["scope"] == "all" else None
        distances = robot_distance_order([observed_object(observation, identifier)
                                          for identifier in task['source_object_ids']])
        targets = [row['id'] for row in distances]
        if count is not None:
            targets = targets[:count]
        task['transport_ids'] = targets
        task['execution_order'] = {'method': 'robot_base_xy_near_to_far',
            'source': 'measured_rgbd_centroids', 'robot_base_world_xy_m': [0., 0.],
            'observation_id': observation['observation_id'], 'generation': observation.get('generation'),
            'ranked_sources': distances, 'selected_ids': list(targets)}
        # Source identity was confirmed above. The count subset is a local task
        # policy, not a second visual identity judgment or an ID reassignment.
        for identifier in task['source_object_ids']:
            membership = 'required' if identifier in targets else 'excluded'
            # Initial visual identities are immutable and fingerprint container
            # anchors. Count selection belongs only to the live task ledger.
            for entities in (self.task_progress.records, self.task_progress.scene_entities):
                entities[identifier]['visual_membership'] = membership
        self.task_progress.requested_count = count
        self.task_progress.assess_membership({identifier: "required" if identifier in targets else "excluded"
                                             for identifier in self.task_progress.records})
        self.task_progress.freeze_inventory(observation)
        for rank, identifier in enumerate(targets, 1):
            self.task_progress.records[identifier]['execution_rank'] = rank
        self.task_progress.initial_head_reference = {
            key: copy.deepcopy(observation["views"]["2"][key])
            for key in ("intrinsics", "T_world_camera")}
        self.state["demo_task"] = copy.deepcopy(task)
        context['visual_inventory_reference']['demo_task'] = copy.deepcopy(task)
        self.emit('target_execution_order', '按实测距离由近到远执行，物体编号保持固定', task['execution_order'])
        if self.unpack_task:
            initialize_processes(task, self.task_progress.records)
            self.task_progress.process_task = task
            self.task_progress.route = 'box_table_box'
            self.sync_lifecycle(context)
        self.state["facts"].update(inventory_finalized=True, task_assessed=True)
        result = self.task_progress.snapshot()
        if resolution is not None:
            self.state["source_inventory_resolution"] = copy.deepcopy(resolution)
            result["source_inventory_resolution"] = resolution
        return result

    def _bind_region(self, observation):
        destination = observed_object(observation, DESTINATION_ID)
        if self.unpack_task:
            existing = self.task_progress.placement_request(DESTINATION_ID)
            if existing.get('layout', {}).get('method') != 'shared_work_area':
                return self.task_progress.bind_shared_work_area(destination, observation)
            return existing
        request = self.task_progress.placement_request(DESTINATION_ID)
        if "layout" not in request:
            self.task_progress.bind_placement_region(destination, observation)
        return self.task_progress.placement_request(DESTINATION_ID)

    def _check_source_view(self, observation):
        # The verified post-VLA view can use any camera for background pink
        # entities. Their identity/geometry still has to pass fused association.
        if self.unpack_task and observation.get('stage') == 'unpacked':
            return
        completed = set(self.task_progress.completed_ids)
        for identifier in self.demo_task["source_object_ids"]:
            lifecycle = self.demo_task.get('item_lifecycle', {}).get(identifier, {})
            if identifier in completed or lifecycle.get('phase', 'wrapped_source') != 'wrapped_source':
                continue
            source = observed_object(observation, identifier)
            if "2" not in source.get("coverage", {}).get("camera_ids", []):
                raise ValueError("left_observation_source_not_visible:" + identifier)

    async def execute(self, skill, context, arguments=None):
        if skill in self.unpack_skills:
            if not self.unpack_task:
                raise ValueError('unpack_skill_requires_unpack_task')
            return await self.execute_unpack(skill, context)
        if skill == "move_to_observe":
            if context.get('unpacked_grasp_capture'):
                raise ValueError('unpacked_grasp_view_already_confirmed')
            if (self.unpack_task and self.state.get('target_id')
                    and self.current_lifecycle()['phase'] == 'unpack_pending'):
                raise ValueError('unpack_verification_required_before_observation_motion')
            if not context.get("pending_transfer", {}).get("verified"):
                raise ValueError("return_requires_verified_release")
            context["use_retained_scene_for_return"] = True
            return await self.move_to_observe(context)
        if skill == "observe_scene":
            self.require_grasp_view(context)
            capture = context.get('unpacked_grasp_capture')
            if capture is None:
                capture = context.pop("fine_capture", None)
            result = await self.observe(context, checkpoint="initial", capture=capture)
            if self.unpack_task:
                self.set_observation_stage(result)
            context["observation_pose_capture_id"] = result["observation_id"]
            context["observation_pose_reference"] = {
                key: copy.deepcopy(result[key]) for key in ("observation_id", "time", "arms")}
            self._check_source_view(result)
            self._require_visible_inventory(result)
            return result
        if skill == "bind_inventory":
            return self._bind_inventory(context)
        if skill == "identify_destination":
            destination_id = self.demo_task['right_bin_id'] if self.unpack_task and self.current_lifecycle()['appearance'] == 'unpacked' else DESTINATION_ID
            requested = (arguments or {}).get('destination_id', destination_id)
            if requested != destination_id:
                raise ValueError('destination_differs_from_task_phase')
            destination = observed_object(context["observation"], destination_id)
            self.state["destination_id"] = destination_id
            record = self.task_progress.records[self.state["target_id"]]
            record.update(destination_id=destination_id, destination_kind=destination['kind'])
            if destination_id == DESTINATION_ID:
                self._bind_region(context["observation"])
            return destination
        if skill == "choose_arm":
            requested = (arguments or {}).get("arm", "auto")
            observation = context["observation"]
            evaluations = await self.call_algorithm(self.algorithms.evaluate_reachability, observation,
                self.state["target_id"], self.state['destination_id'], requested_arm=self.task_progress.bound_arm())
            source = observed_object(observation, self.state["target_id"])
            source_y = source["position"][1]
            preferred = "A" if source_y >= 0 else "B"
            available = [arm for arm in (preferred, "B" if preferred == "A" else "A")
                         if evaluations.get(arm, {}).get("reachable")]
            if requested != "auto":
                available = [arm for arm in available if arm == requested]
            if self.unpack_task:
                lifecycle = self.current_lifecycle()
                planned_arm = lifecycle['to_right_bin_arm' if lifecycle['appearance'] == 'unpacked' else 'to_table_arm']
                available = [arm for arm in available if arm == planned_arm]
            if not available:
                raise ValueError("no_reachable_arm")
            arm = available[0]
            self.task_progress.bind_arm(arm)
            self.state["arm"] = arm
            context["reachability_evaluations"] = evaluations
            self.state["facts"]["arms_evaluated"] = True
            return {"arm": arm, "evaluations": evaluations, "rule": "reachable_source_side"}
        if skill == "build_scene":
            context.pop("scene_refresh_required", None)
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
            if self.unpack_task and self.current_lifecycle()['appearance'] == 'unpacked':
                destination = observed_object(observation, self.demo_task['right_bin_id'])
                context['grasp_placement_request'] = {'object_id': self.state['target_id'],
                    'transport_region': right_bin_transport_region(destination, context['scene'], observation)}
            else:
                context['grasp_placement_request'] = copy.deepcopy(self._bind_region(observation))
            context["grasps"] = await self.call_algorithm(self.algorithms.generate_grasps,
                observation, self.state["target_id"], arm=self.state["arm"],
                placement_request=context["grasp_placement_request"],
                planning_settings=context["planning_settings"], report_progress=True)
            context["grasps_observation_id"] = observation["observation_id"]
            return context["grasps"]
        if skill == "plan_place":
            if self.unpack_task and self.current_lifecycle()['phase'] != 'wrapped_source':
                raise ValueError('table_place_requires_wrapped_source_stage')
            await self.verify_scene(context)
            observation = self.measured_state(context)
            await self.execution_stage(context, "prepare_attachment", self.prepare_attachment, context, observation)
            context["place_plan"] = await self.call_algorithm(self.algorithms.plan_place, observation,
                self.state["target_id"], self.state["arm"], context["scene"], context["holding"],
                DESTINATION_ID, planning_settings=context["planning_settings"],
                placement_request=self._bind_region(context["observation"]), report_progress=True,
                timeout_s=context["planning_settings"]["place_timeout_s"])
            context["place_plan_observation_id"] = context["observation"]["observation_id"]
            return context["place_plan"]
        if skill == "execute_place":
            result = await super().execute(skill, context, arguments)
            context["demo_place_execution"] = copy.deepcopy(result)
            return result
        if skill == "verify_place":
            carried = copy.deepcopy(context.get("holding"))
            result = await self.verify_place(context)
            if not result["success"]:
                return result
            # Freeze order after the physical feedback read, even if MuJoCo has
            # not advanced a simulation tick before the subsequent RGB snapshot.
            result["verified_perf_counter_ns"] = time.perf_counter_ns()
            target = self.state["target_id"]
            self.task_progress.record_release(target, feedback_id=result["feedback_id"])
            binding = build_release_binding(self.state, context, self.task_progress, self.run_id, result)
            self.emit("release_verification_binding", "放置结果核验的执行链与槽位", binding)
            self.holding_state = None
            context.pop("holding", None)
            capture = await self.capture_observation(context)
            validate_release_capture(capture, binding)
            context.update(post_place_capture=capture, latest_capture=capture)
            self.record_observation(capture, "post_place")
            self.sync_result_evidence(context, "post_place")
            after = await self.call_algorithm(self.algorithms.verify_release, capture, binding,
                context["visual_inventory_reference"], self.state["instruction"])
            if self.unpack_task and after.get('observation_assessment', {}).get('needs_observation'):
                self.emit('visual_transfer_result', 'Decisions：放置核验需要补充观测', after)
                # User order: lift, Decisions, then confirmed local obstacle.
                # An unresolved visual result must not publish pink geometry.
                capture = await self.supplement_result_observation(context, after['observation_assessment'], 'verify_place')
                validate_release_capture(capture, binding)
                context.update(post_place_capture=capture, latest_capture=capture)
                self.record_observation(capture, 'post_place')
                after = await self.call_algorithm(self.algorithms.verify_release, capture, binding,
                    context['visual_inventory_reference'], self.state['instruction'])
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
            if confirmed and not self.unpack_task:
                self.task_progress.records[target]["completed_identity_track"] = {
                    "binding": copy.deepcopy(binding), "visual_evidence": copy.deepcopy(evidence),
                    "capture": {key: capture[key] for key in
                        ("observation_id", "generation", "time", "captured_perf_counter_ns") if key in capture}}
            result.update(success=confirmed, status="confirmed" if confirmed else "unknown",
                verification_scope="release_and_visual_destination", visual_evidence=evidence,
                reason=None if confirmed else "visual_destination_unconfirmed")
            context["pending_transfer"]["verified"] = confirmed
            context["place_verification"] = result
            context["demo_inventory_check"] = after.get("demo_inventory_check", {})
            if confirmed and self.unpack_task:
                context['post_place_verification_input'] = table_release_reference(capture, evidence, self.run_id)
                lifecycle = self.current_lifecycle()
                lifecycle.update(phase='table_released', table_release_feedback_id=result['feedback_id'],
                    table_visual_evidence=copy.deepcopy(evidence))
                self.sync_lifecycle(context)
                self.state['facts']['table_release_verified'] = True
            if confirmed and (self.unpack_task or set(self.task_progress.completed_ids) != set(self.demo_task["transport_ids"])):
                if not carried:
                    raise ValueError("release_geometry_missing")
                if not self.unpack_task:
                    context["scene"] = await self.call_algorithm(
                        self.algorithms.publish_released_object, context["scene"], target,
                        carried["grasp_support_reference"]["T_world_tcp_at_grasp"],
                        context["release_tcp"], capture["observation_id"],
                        support_floor_z_m=context["observation"]["table"]["surface_height_m"])
                if self.unpack_task:
                    area = {**self.region, 'floor_z_m': context['observation']['table']['surface_height_m']}
                    local = await self.call_algorithm(self.algorithms.publish_local_object,
                        context['scene'], capture, target, area, 'wrapped')
                    context['scene'] = local['scene']
                    self.emit('local_object_obstacle', '头部RGBD更新桌面粉件障碍', local['measurement'])
                self.state["facts"]["scene_available"] = True
                self.emit("retained_scene_updated", "保留地图并更新已释放物体", {
                    "scene_token": context["scene"]["scene_token"], "target_id": target,
                    "source": "verified_release_rgbd_local_object" if self.unpack_task else
                        "release_feedback_rigid_prediction", "rgbd_remapped": False})
            return result
        return await super().execute(skill, context, arguments)

    async def _execute_program(self, nodes, context, target=None):
        """Interpret only typed nodes; each skill rechecks live contract facts."""
        for node in nodes:
            self.check_generation()
            if isinstance(node, Stop):
                raise ValueError("program_stopped:" + node.reason[:500])  # noqa: TRY004 — explicit stop node
            if isinstance(node, IfResult):
                if self.last_program_result is None:
                    raise ValueError("program_condition_requires_result")
                result = self.last_program_result
                status = ("unknown" if result.get("status") == "unknown" else
                          "failure" if result.get("success") is False else "success")
                await self._execute_program(
                    node.then_steps if status == node.condition else node.else_steps, context, target)
            elif isinstance(node, ForEachTarget):
                for index, identifier in enumerate(self.demo_task["transport_ids"]):
                    if self.unpack_task:
                        self.state['process_id'] = next(row['id'] for row in self.demo_task['process_records'].values()
                                                        if row['input_entity_id'] == identifier)
                    completed = completed_input_ids(self.demo_task) if self.unpack_task else set(self.task_progress.completed_ids)
                    if identifier in completed:
                        raise ValueError("program_target_already_completed")
                    if index:
                        # A binding lasts through one object's grasp and place.
                        # The next object must evaluate both workspaces afresh.
                        self.task_progress.stage_arms.clear()
                        self.reset_transfer(context)
                        self.state["facts"].update(observed=False, observation_pose_ready=False,
                            observation_map_ready=False, scene_available=bool(context.get("scene")),
                            arm_selected=False, task_assessed=True)
                        await self._execute_program(node.between_targets, context, identifier)
                    await self._execute_program(node.body, context, identifier)
                    completed = completed_input_ids(self.demo_task) if self.unpack_task else set(self.task_progress.completed_ids)
                    if identifier not in completed:
                        raise ValueError("program_target_not_verified:" + identifier)
            elif isinstance(node, SkillCall):
                self.program_steps += 1
                if self.program_steps > 128 * max(1, len(self.demo_task['transport_ids'])):
                    raise ValueError("program_execution_budget_exceeded")
                entity_target = current_process_entity(self.demo_task, target) if self.unpack_task and target else target
                bindings = {"$target": entity_target, "$destination": DESTINATION_ID, '$table_zone': DESTINATION_ID,
                            '$right_bin': self.demo_task.get('right_bin_id')}
                arguments = {arg.name: bindings.get(arg.value, arg.value) for arg in node.arguments}
                if node.skill == "identify_target":
                    if target is None:
                        raise ValueError("program_target_loop_required")
                    arguments.setdefault("target_id", entity_target)
                self.last_program_result = await self._step(node.skill, context, arguments)
                if node.skill.startswith("verify_"):
                    self.state["verification"] = self.last_program_result

    def _confirm_program_completion(self, context):
        completed = completed_input_ids(self.demo_task) if self.unpack_task else set(self.task_progress.completed_ids)
        if completed != set(self.demo_task["transport_ids"]):
            raise ValueError("program_finished_with_pending_targets")
        if self.demo_task["scope"] == "all" and not self.unpack_task:
            self._require_visible_inventory({
                "observation_id": context.get("post_place_capture", {}).get("observation_id"),
                "demo_inventory_check": context.get("demo_inventory_check", {})})
        self.state["status"] = "succeeded"

    async def finish_unpack_at_initial_pose(self, context):
        if completed_input_ids(self.demo_task) != set(self.demo_task['transport_ids']):
            raise ValueError('program_finished_with_pending_targets')
        try:
            self.require_observation_pose(context)
        except ValueError as error:
            if str(error) != 'mapping_requires_observation_pose':
                raise
            self.state['facts']['observation_pose_ready'] = False
            context['use_retained_scene_for_return'] = True
            await self._step('move_to_observe', context)
            self.state['facts']['observation_pose_ready'] = False
            self.require_observation_pose(context)
        self.state['facts']['observation_pose_ready'] = True

    async def _run(self, run_id, initial):
        context = {"initial": initial, "planning_settings": copy.deepcopy(self.state["planning_settings"]),
                   "visual_call_budget": VisualCallBudget(limit=0)}
        self.demo_task = None
        self.demo_program = None
        self.program_steps = 0
        self.last_program_result = None
        self.task_progress = ProcessTaskProgress(self.state["instruction"], "collection", None,
                                         route="direct", selection_order="inventory")
        self.state.update(task="grasp_place", route="direct", controller="responses_program", requested_arm=None,
                          attributes={"kind": "any", "color": "any", "position": "any"},
                          destination_attributes={"kind": "marker", "color": "any"})
        self.state["facts"].update(needs_place=True, collection_task=True, center_reachability_enabled=True)
        self.emit("run_start", "开始模型编排搬运流程", {"instruction": self.state["instruction"],
                                               "controller": "responses_program"})
        try:
            # Startup is already at the requested pose. No initial motion or raw map.
            self.require_observation_pose(context, initial)
            self.state["facts"]["observation_pose_ready"] = True
            self.state["facts"]["grasp_view_ready"] = True
            for name in ("observe_scene", "bind_inventory"):
                await self._step(name, context)
            await self._execute_program(self.demo_program.steps, context)
            if self.unpack_task:
                await self.finish_unpack_at_initial_pose(context)
            self._confirm_program_completion(context)
        except asyncio.CancelledError:
            self.state["status"] = "cancelled"
            self.emit("run_cancelled", "任务已取消", {})
            raise
        except Exception as error:  # noqa: BLE001 — owned task records terminal failure before returning
            message = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
            code = getattr(error, "code", None) or message
            self.state.update(status="needs_attention", error=message[:1000], error_code=code,
                              failed_step=copy.deepcopy(self.state.get("current_step")))
            from .run_conclusion import brief_error

            skill = (self.state.get('failed_step') or {}).get('skill')
            self.state['error_summary'] = brief_error(code, skill, message, getattr(error, 'details', None))
            self.emit("skill_error", self.state['error_summary'], {"code": code, 'skill': skill,
                      'summary': self.state['error_summary'], "error": message[:1000],
                      "details": getattr(error, "details", None)})
            if self.task_progress.inventory_initialized and skill in {
                    'observe_scene', 'verify_place', 'verify_unpack', 'verify_drop'}:
                calls = [event for event in self.events if event['kind'] == 'visual_decisions_call'
                         and event.get('invocation_id') == self.invocation_id]
                self.emit('inventory_snapshot', '核验未通过 · 保留已有清单', {
                    'checkpoint': skill, 'outcome': 'not_applied', 'error_code': code,
                    'observation_id': calls[-1]['data'].get('observation_id') if calls else None,
                    'model_event_ids': [event['id'] for event in calls],
                    'task_progress': self.task_progress.snapshot()})
        finally:
            self.state["task_progress"] = self.task_progress.snapshot()
            self.state["current_step"] = None
            self.state["metrics"] = {"run_duration_s": time.monotonic() - self.run_started_at,
                "responses_call_count": sum(event["kind"] == "visual_decisions_call"
                    and event["data"].get("endpoint") == "/v1/responses" for event in self.events)}
            self.emit("run_end", "任务结束", self.state)
            if self.experiment is not None:
                self.experiment.stop(self.state)

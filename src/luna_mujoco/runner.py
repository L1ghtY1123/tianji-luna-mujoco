"""Shared motion, sensing, and recording primitives for the Responses program runner."""
import asyncio
import copy
import json
import logging
import math
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .observation_motion import (
    initial_view_evidence,
    observation_poses,
    source_view_evidence,
    standalone_place_view_evidence,
)
from .perception import PerceptionError
from .placement import carrying_binding, grasp_support_reference, observed_object, public_observation
from .planning_settings import PlanningSettings
from .skills import Registry, SkillSelectionError
from .task_progress import original_source_bin
from .verification import (
    combine_holding_evidence,
    gripper_open_confirmed,
    verify_gripper_position,
)

LOGGER = logging.getLogger(__name__)


class TrajectoryStartError(ValueError):
    """Preserve the start guard while recording the measured joint mismatch."""

    code = "active_arm_moved_since_trajectory_start"

    def __init__(self, arm, actual, start, feedback):
        super().__init__(self.code)
        actual, start = np.asarray(actual), np.asarray(start)
        drift = np.abs(actual - start)
        joint = int(np.argmax(drift))
        self.details = {
            "arm": arm,
            "joint_name": f"Joint{joint + 1}_{'L' if arm == 'A' else 'R'}",
            "maximum_drift_deg": float(np.rad2deg(drift[joint])),
            "threshold_deg": 1.,
            "actual_joints_rad": actual.tolist(),
            "trajectory_start_joints_rad": start.tolist(),
            "generation": feedback.get("generation"),
            "feedback_time": feedback.get("time"),
        }


def public_summary(value):
    """Keep trace evidence usable without streaming thousands of cloud/trajectory entries."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): list(v) if k == "candidate_indices" and isinstance(v, (list, tuple)) else public_summary(v)
                for k, v in value.items() if not str(k).startswith("_")
                and k not in {"rgb", "depth_m", "valid_depth", "instance_masks", "masks", "mask",
                              "valid_depth_mask", "pixels_rc", "completed_geometry_points_world_m",
                              "points_world_m", "points_object_m", "points_tool_m", "point_sources", "views"}}
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return {"count": len(value), "first": [public_summary(v) for v in value[:3]]}
        return [public_summary(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Runner:
    def __init__(self, simulation, algorithms, decisions_factory, data_dir, *, plan_timeout_s=None,
                 planning_settings=None, visual_assistance=None, experiments=None):
        self.simulation = simulation
        self.algorithms = algorithms
        self.visual_assistance = visual_assistance
        self.data_dir = Path(data_dir)
        self.planning_settings = planning_settings or PlanningSettings()
        self.registry = Registry()
        self.worker = None
        self.events = []
        self.state = self.idle_state()
        self.run_id = None
        self.experiments = experiments
        self.experiment = None
        self.run_started_at = time.monotonic()
        self.invocation_id = None
        self.invocation_count = 0
        self.algorithm_call_count = 0
        self.recorded_observations = set()
        if plan_timeout_s is not None:
            if not math.isfinite(plan_timeout_s) or plan_timeout_s <= 0:
                raise ValueError("plan_timeout_must_be_positive_and_finite")
            self.plan_timeout_s = plan_timeout_s
        self.planning_worker = None
        self.planning_cancelled = None
        self.algorithm_cleanup_worker = None
        self.algorithm_cleanup_error = None
        self.algorithm_cleanup_revision = 0
        self.holding_state = None
        self.task_progress = None

    @property
    def visual_inventory(self):
        return getattr(getattr(self.algorithms, 'perception', None), 'visual_inventory', None)

    def visual_audit_sink(self):
        experiment, run_id = self.experiment, self.run_id
        done = experiment.retain_visual_call() if experiment is not None else threading.Event()

        def audit(evidence):
            self.record_visual_decision(evidence, experiment=experiment, run_id=run_id)
        return audit, done

    def record_visual_decision(self, evidence, *, experiment, run_id):
        """Keep actual image requests outside compact trace state."""
        artifact = experiment.visual_decision(evidence) if experiment is not None else None
        if run_id != self.run_id or self.state['status'] != 'running':
            return
        summary = {key: evidence.get(key) for key in
                   ('observation_id', 'request_hash', 'elapsed_s', 'http_status', 'error')}
        summary.update(artifact=artifact, question_count=len(evidence['request'].get('questions', [])),
                       purpose=evidence.get('purpose'), endpoint=evidence.get('endpoint', '/v1/decisions'),
                       response=evidence.get('response'))
        summary['image_selection'] = evidence.get('image_selection')
        summary['image_count'] = sum(part.get('type') == 'input_image'
            for message in evidence['request'].get('input', []) for part in message.get('content', []))
        self.emit('visual_decisions_call', 'Responses 初始 RGB 清单' if summary['endpoint'] == '/v1/responses'
                  else 'Decisions 图片判断', summary)

    @staticmethod
    def idle_state():
        return {"status": "idle", "instruction": "", "task": None, "target_id": None,
                "arm": None, "destination_id": None, "facts": {}, "error": None, "error_code": None,
                "current_step": None, "failed_step": None,
                "planning_attempts": []}

    @property
    def plan_timeout_s(self):
        return self.planning_settings.plan_timeout_s

    @plan_timeout_s.setter
    def plan_timeout_s(self, value):
        self.planning_settings = PlanningSettings.model_validate(
            {**self.planning_settings.model_dump(), "plan_timeout_s": value})

    @property
    def active(self):
        return self.worker is not None and not self.worker.done()

    @property
    def transfer_routes(self):
        return self.task_progress is not None and self.task_progress.route != 'direct'

    @property
    def algorithm_pending(self):
        return any(worker is not None and not worker.done()
                   for worker in (self.planning_worker, self.algorithm_cleanup_worker))

    def emit(self, kind, title, data):
        if self.task_progress is not None:
            self.state['task_progress'] = self.task_progress.snapshot()
        def encode_numpy(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"Cannot serialize {type(value).__name__} in run log")

        event = {"id": len(self.events) + 1, "time": time.time(), "kind": kind,
                 "title": title, "data": public_summary(data),
                 "t_s": time.monotonic() - self.run_started_at,
                 "target_id": self.state.get("target_id"),
                 "arm": self.state.get("arm"),
                 "invocation_id": self.invocation_id,
                 "parent_invocation_id": self.invocation_id if kind not in
                 {"skill_start", "skill_complete", "skill_error", "planning_attempt_failed"} else None}
        # Preserve model control inputs/outputs; summarize bulky algorithm payloads.
        if kind in {"jev", "skill_program"}:
            event["data"] = data
        if self.experiment is not None:
            self.experiment.event(event, data)
        self.events.append(event)
        if self.run_id and not self.state.get("log_error"):
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                with (self.data_dir / (self.run_id + ".jsonl")).open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event, ensure_ascii=False, default=encode_numpy) + "\n")
            except OSError as error:
                # Keep terminal events available over HTTP/WS even when disk logging fails.
                self.state["log_error"] = "run_log_write_failed:" + type(error).__name__
                LOGGER.exception("Run event could not be written")

    def check_log(self):
        if self.state.get("log_error"):
            raise RuntimeError(self.state["log_error"])

    def begin_step(self, skill):
        self.state["current_step"] = {"skill": skill, "started_at": time.time()}
        self.invocation_id = None
        if skill in self.registry.contracts:
            self.invocation_count += 1
            self.invocation_id = f"skill-{self.invocation_count:04d}"
            self.state["current_step"]["invocation_id"] = self.invocation_id
        if skill in {"plan_grasp", "execute_grasp"}:
            self.state["collision_model"] = {"gripper": "fixed_open", "attached": False,
                                              "lift_carried_collision_checked": False}
        elif skill in {"move_to_observe", "plan_place", "execute_place"}:
            self.state["collision_model"] = {"gripper": "actual_feedback",
                                              "attached": bool(self.holding_state and self.holding_state.get("attached"))}
        if skill == "plan_grasp":
            self.state["current_step"]["timeout_s"] = self.state["planning_settings"]["plan_timeout_s"]

    async def start(self, instruction, *, experiment=None):
        if self.active:
            raise ValueError("run_already_active")
        if self.algorithm_pending:
            raise ValueError("algorithm_request_pending")
        if self.algorithm_cleanup_error:
            raise ValueError("algorithm_cleanup_failed_reset_to_retry")
        if not 1 <= len(instruction.strip()) <= 200:
            raise ValueError("instruction_length_must_be_1_to_200")
        snapshot = self.simulation.robot_snapshot()
        if not snapshot["confirmed"]:
            raise ValueError("confirm_scene_first")
        self.run_id = uuid.uuid4().hex
        self.run_started_at = time.monotonic()
        self.invocation_count = 0
        self.algorithm_call_count = 0
        self.recorded_observations = set()
        self.experiment = None
        self.events = []
        self.state = {**self.idle_state(), "status": "running", "instruction": instruction.strip(),
                      "run_id": self.run_id, "generation": snapshot["generation"],
                      "planning_settings": self.planning_settings.model_dump(),
                      "settings_revision": self.planning_settings.revision()}
        # Missing facts already mean false in contracts. State them explicitly
        # so the decision model does not infer a previous run's scene or plans.
        self.state["facts"] = {fact: False for contract in self.registry.contracts.values()
                               for fact in (*contract.requires, *contract.provides)}
        if self.experiments is not None:
            from .experiments import ExperimentOptions
            try:
                self.experiment = self.experiments.start(self.run_id, self.state["instruction"],
                    experiment or ExperimentOptions(), {"planning_settings": self.state["planning_settings"],
                    "initial_robot_feedback": snapshot, "simulation_initial": self.simulation.snapshot()},
                    self.simulation)
                if self.experiment is not None:
                    self.run_started_at = self.experiment.started_at
                    self.state["experiment"] = self.experiment.summary()
            except Exception as error:
                self.state["experiment"] = {"error": "experiment_start_failed:" + type(error).__name__}
                LOGGER.exception("Experiment recording could not start")
        self.worker = asyncio.create_task(self._run(self.run_id, snapshot))
        return copy.deepcopy(self.state)

    async def cancel(self):
        # Reset invalidates physics generation; canceled algorithm threads cannot issue actions.
        if self.planning_cancelled is not None:
            self.planning_cancelled.set()
        if self.active:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
        self.worker = None

    def clear(self):
        self.holding_state = None
        self.run_id = None
        self.events = []
        self.state = self.idle_state()
        self.experiment = None
        self.invocation_id = None

    def clear_algorithm_state(self):
        """Queue Reset cleanup behind any old GPU request; new runs wait for both."""
        pending = (self.planning_worker, self.algorithm_cleanup_worker)
        generation = self.simulation.robot_snapshot()["generation"]
        self.algorithm_cleanup_revision += 1
        revision = self.algorithm_cleanup_revision
        self.algorithm_cleanup_error = None

        async def cleanup():
            for worker in pending:
                if worker is not None:
                    await asyncio.gather(asyncio.shield(worker), return_exceptions=True)
            # A later scene/reset schedules its own cleanup. Never clear its newer result.
            if (self.simulation.robot_snapshot()["generation"] != generation
                    or self.algorithm_cleanup_revision != revision):
                return
            try:
                result = await asyncio.to_thread(self.algorithms.clear_carrying)
                if result.get("ok") is not True or result.get("attached") is not False:
                    raise ValueError("algorithm_cleanup_not_confirmed")
                if self.algorithm_cleanup_revision == revision:
                    self.algorithm_cleanup_error = None
            except Exception as error:
                if self.algorithm_cleanup_revision == revision:
                    self.algorithm_cleanup_error = type(error).__name__ + ":" + str(error)[:300]
                LOGGER.exception("Reset could not clear GPU scene and attachment")

        self.algorithm_cleanup_worker = asyncio.create_task(cleanup())

    def snapshot_for_decision(self, context=None):
        snapshot = self.simulation.robot_snapshot()
        for feedback in snapshot["arms"].values():
            feedback.pop("gripper_position_samples", None)
        state = {"instruction": self.state["instruction"], "task": self.state["task"],
                "route": self.state.get('route', 'direct'),
                "attributes": self.state.get("attributes", {}), "target_id": self.state["target_id"],
                "destination_attributes": self.state.get("destination_attributes", {}),
                "destination_id": self.state["destination_id"],
                "execution_arm": self.state["arm"], "facts": dict(self.state["facts"]),
                "skill_rejection": copy.deepcopy((context or {}).get('skill_rejection')),
                "recovery_context": {"scene_available": bool((context or {}).get("scene")),
                    "scene_current": bool((context or {}).get('scene') and self.state['facts'].get('scene_ready')
                                          and not (context or {}).get('scene_refresh_required')),
                    "grasp_candidates_available": bool((context or {}).get('grasps')),
                    "bound_arm": self.task_progress.bound_arm() if self.task_progress else None,
                    "stage_arm_fixed": bool(self.task_progress and self.task_progress.bound_arm()),
                    "retry_plan_scope": "same_bound_arm"},
                "verification": copy.deepcopy(self.state.get("verification") or {}),
                "visual_result": copy.deepcopy(self.state.get("visual_result") or {}),
                "visual_assistance_available": bool(self.visual_assistance and self.visual_assistance.enabled),
                "visual_semantics": self.state.get('visual_semantics', 'optional'),
                "planning_settings": self.state.get("planning_settings", {}),
                "planning_attempts": self.state["planning_attempts"],
                "simulation": {key: snapshot[key] for key in ("generation", "time", "arms")
                               if key in snapshot},
                "recent_results": [event for event in self.events if event["kind"] == "skill_complete"][-3:]}
        # A retained holding capture is association history, not this run's observation.
        observation = context.get("observation") if context and state["facts"].get("observed") else None
        state["observation"] = public_observation(observation)
        if self.task_progress is not None:
            state['task_progress'] = self.task_progress.snapshot()
            state['failure'] = copy.deepcopy(self.task_progress.failure)
            state['requested_arm'] = self.task_progress.requested_arm
            if observation:
                state['target_distances'] = {arm: self.task_progress.rank_candidates(
                    observation | {'arms': snapshot['arms']}, arm=arm) for arm in ('A', 'B')}
        state['visual_scene'] = copy.deepcopy((context or {}).get('visual_scene', {}))
        evidence = context or {}
        current_id = (observation or {}).get('observation_id')
        scene = evidence.get('scene', {})
        result_phase = 'post_place' if state['facts'].get('released') else 'post_grasp'
        result_id = evidence.get(result_phase + '_capture', {}).get('observation_id')
        verification_input = evidence.get(result_phase + '_verification_input', {})
        visual_id = evidence.get('visual_result', {}).get('observation_id')
        state['perception_state'] = {
            'observation_id': current_id,
            'checkpoint': evidence.get('observation_checkpoint'),
            'scene_observation_id': scene.get('observation_id'),
            'scene_matches_observation': bool(current_id and scene.get('observation_id') == current_id),
            'scene_refresh_required': bool(evidence.get('scene_refresh_required')),
            'collision_policy': scene.get('collision_policy'),
            'source_views': public_summary(evidence.get('view_evidence', {})),
            'visual_scene_matches_observation': bool(current_id and
                evidence.get('visual_scene', {}).get('observation_id') == current_id),
            'result_capture_id': result_id,
            'visual_result_matches_capture': bool(result_id and
                visual_id == result_id),
            'verification_matches_capture': bool(result_id and
                verification_input.get('observation_id') == result_id),
            'new_visual_evidence_since_verification': bool(result_id and visual_id == result_id
                and verification_input.get('visual_observation_id') != visual_id),
            'geometry_source': 'rendered_rgbd_visible_returns_and_robot_calibration',
            'table_volume_source': 'inferred_from_rgbd_ransac_plane',
            'unobserved_space_collision_checked': False,
            'holding_verified': bool(evidence.get('holding', {}).get('verified')),
            'requires_observation_pose': bool(evidence.get('requires_observation_pose')),
            'release_endpoint_evidence': copy.deepcopy(evidence.get('release_endpoint_evidence', {})),
            'pending_transfer': public_summary(evidence.get('pending_transfer', {})),
            'association_evidence': copy.deepcopy(evidence.get('association_evidence', {})),
            'analysis_count': evidence.get('analysis_count', 0),
            'observation_pose_reached': bool(evidence.get('observation_pose_capture_id')),
        }
        state['observation_budget'] = self.task_budgets(context or {})
        budget = (context or {}).get('visual_call_budget')
        if budget is not None:
            state['visual_call_budget'] = {'limit': budget.limit, 'used': budget.used}
        if context and context.get("scene"):
            state["collision_scene"] = {key: context["scene"][key] for key in
                                        ("scene_token", "observation_id", "generation", "target_id")
                                        if key in context["scene"]}
        if context and "evaluations" in context:
            state["arm_evaluations"] = context["evaluations"]
        # Later attempts must not mutate a request already captured in the model log.
        return copy.deepcopy(state)

    def check_generation(self):
        if self.simulation.robot_snapshot()["generation"] != self.state["generation"]:
            raise ValueError("simulation_generation_changed")

    def record_failure(self, context, stage, code, evidence):
        if isinstance(evidence.get('details'), dict) and evidence['details'].get('status') == 'unknown':
            evidence = {**evidence, 'status': 'unknown'}
        self.task_progress.fail(self.state.get('target_id'), stage, code, public_summary(evidence))
        self.state['facts'].update(recovery_needed=True, plan_ready=False, place_plan_ready=False)
        for key in ('plan', 'place_plan'):
            context.pop(key, None)
        self.state['failure'] = copy.deepcopy(self.task_progress.failure)
        self.state['failed_step'] = copy.deepcopy(self.state.get('current_step'))
        self.emit('task_failure', '失败证据，技能程序已停止', self.state['failure'])

    def reset_transfer(self, context, *, retry_target=None):
        retained = {key: context[key] for key in ('initial', 'planning_settings', 'observation', 'inventory_observation',
                    'initial_capture', 'fine_capture', 'latest_capture', 'post_place_capture', 'post_unpack_capture',
                    'post_place_verification_input', 'visual_call_budget',
                    'head_observation', 'visual_inventory_reference', 'visual_plan', 'frozen_table', 'normal_capture_count', 'recovery_capture_count',
                    'observation_count', 'captured_observation_ids', 'analyzed_capture_ids',
                    'observation_pose_capture_id', 'observation_pose_reference',
                    'scene', 'scene_observation', 'pending_transfer',
                    'scene_validated_capture_id',
                    'visual_inputs_consumed', 'scene_inputs_consumed',
                    'requires_observation_pose', 'scene_refresh_required', 'association_evidence') if key in context}
        context.clear()
        context.update(retained)
        if retry_target is not None:
            context['retry_target_id'] = retry_target
            context['recovery_capture_mode'] = True
        self.state.update(target_id=None, destination_id=None, arm=None, verification=None,
                          visual_result=None, planning_attempts=[], failure=None)
        self.state['facts'] = {fact: False for contract in self.registry.contracts.values()
                               for fact in (*contract.requires, *contract.provides)}
        collection = (self.task_progress.scope == 'collection' or self.transfer_routes
                      or self.state['task'] != 'place' and self.task_progress.selection_order == 'inventory')
        self.state['facts'].update(needs_place=self.state['task'] != 'grasp',
            destination_ready=self.state['task'] == 'grasp', collection_task=collection,
            inventory_finalized=self.task_progress.inventory_finalized,
            visual_available=bool(self.visual_assistance and self.visual_assistance.enabled),
            task_assessed=not collection or retry_target is not None)
        self.state['facts']['center_reachability_enabled'] = True
        self.state['facts']['observed'] = bool(context.get('observation'))
        self.state['facts']['observation_map_ready'] = context.get('scene', {}).get('exclusion_state') == 'observation'
        self.state['arm'] = self.task_progress.bound_arm()
        self.state['facts']['arm_selected'] = self.state['arm'] is not None
        self.state['facts'].update(transfer_pending=bool(context.get('pending_transfer')),
                                  scene_available=bool(context.get('scene')))

    async def recover_task(self, context, action):
        """Apply the explicit recovery action. No failure code selects a recovery branch."""
        failure = self.task_progress.failure or {}

        def reject(code):
            raise SkillSelectionError(code, 'recover_task', {'action': action,
                'original_failure': failure.get('code'),
                'required': 'Correct the recovery choice using recovery_context; no action executed.'})

        if action == 'stop':
            progress = self.task_progress.snapshot()
            self.state['status'] = ('partial_success' if progress['completed'] else
                'needs_attention' if failure.get('evidence', {}).get('status') == 'unknown' else 'failed')
            self.state['error'] = failure.get('code', 'jev_selected_stop')
            self.state['error_code'] = failure.get('code')
            return {'action': action, 'status': self.state['status']}
        if action == 'reobserve':
            if context.get('recovery_capture_count', 0) >= 8:
                reject('recovery_observation_budget_exhausted')
            context['recovery_capture_mode'] = True
            target = self.state.get('target_id')
            self.invalidate_recovery_scene(context)
            result = await self.observe(context, purpose='geometry' if target else 'scene',
                                        camera_ids=None if context.get('holding') else ['3'])
            context['head_observation'] = copy.deepcopy(result)
            self.state['facts'].update(observed=True, verified=False, place_verification_completed=False,
                                       recovery_needed=False)
            if not context.get('holding'):
                # A failed map may have invalidated its server token, and a
                # selected world excludes the target. Publish a measured,
                # target-inclusive world before any recovery observation motion.
                await self.execution_stage(context, 'recovery_collision_scene', self.build_scene, context, 'observation')
            if self.task_progress.scope == 'collection' and not self.state.get('target_id'):
                self.state['facts']['task_assessed'] = False
            failure['latest_observation_id'] = result['observation_id']
            holding = context.get('holding')
            if holding:
                failure['latest_gripper_evidence'] = self.position_verdict(
                    self.simulation.robot_snapshot()['arms'][holding['arm']], context)
            return {'action': action, 'observation_id': result['observation_id']}
        if action in {'retry_grasp', 'retry_place', 'retry_plan'} and failure.get('attempt_count', 0) >= 3:
            reject('recovery_retry_budget_exhausted')
        if action == 'retry_plan':
            if context.get('holding') or self.holding_state:
                reject('retry_plan_requires_empty_hand')
            evaluations = context.get('reachability_evaluations') or context.get('evaluations')
            bound_arm = self.task_progress.bound_arm()
            if bound_arm is None or not context.get('scene') or not context.get('grasps'):
                reject('retry_plan_requires_bound_arm_scene_and_grasps')
            await self.verify_scene(context)
            if evaluations:
                context['evaluations'] = evaluations
            context['retry_plan_authorized_arm'] = bound_arm
            self.state['arm'] = bound_arm
            self.state['facts'].update(recovery_needed=False, arm_selected=True, plan_ready=False,
                                       arms_evaluated=bool(evaluations))
            self.task_progress.failure = None
            return {'action': action}
        if action == 'retry_place':
            holding = context.get('holding', {})
            if not holding.get('verified'):
                reject('retry_place_requires_verified_holding')
            evidence = self.position_verdict(self.simulation.robot_snapshot()['arms'][holding['arm']], context)
            if evidence['status'] != 'held':
                reject('retry_place_requires_current_holding_feedback')
            reuse_scene = bool(failure.get('stage') == 'plan_place' and context.get('scene')
                and self.state['facts'].get('scene_ready') and self.state['facts'].get('observed')
                and not context.get('scene_refresh_required'))
            if not reuse_scene:
                reject('retry_place_requires_saved_grasp_scene')
            await self.verify_scene(context)
            self.state['facts'].update(recovery_needed=False, place_plan_ready=False, scene_ready=True,
                                       observed=True, place_verification_completed=False, grasp_held=True,
                                       attached=bool(holding.get('attached')), verified=True)
            self.task_progress.failure = None
            return {'action': action, 'scene_reused': True, 'required_evidence': []}
        if action not in {'retry_grasp', 'defer'}:
            reject('invalid_recovery_choice')
        holding = context.get('holding') or self.holding_state
        recovery_arm = holding.get('arm') if holding else self.state.get('arm')
        if recovery_arm:
            feedback = self.simulation.robot_snapshot()['arms'][recovery_arm]
            if (not gripper_open_confirmed(feedback)
                    and self.position_verdict(feedback, context)['status'] != 'empty'):
                reject('recovery_requires_confirmed_empty_hand')
        if holding:
            feedback = self.simulation.robot_snapshot()['arms'][holding['arm']]
            empty = self.position_verdict(feedback, context)['status'] == 'empty'
            released = self.state['facts'].get('released') and gripper_open_confirmed(feedback)
            if not empty and not released:
                reject('recovery_requires_confirmed_empty_hand')
            if holding.get('attached'):
                detached = await self.call_algorithm(self.algorithms.detach_carrying)
                if detached.get('attached') is not False:
                    raise ValueError('recovery_detach_not_confirmed')
            context.pop('holding', None)
            self.holding_state = None
        target = self.state.get('target_id')
        if action == 'defer':
            if target is None:
                reject('defer_requires_selected_target')
            self.task_progress.defer(target, failure.get('code', 'unresolved'))
            self.refresh_observation_labels(context)
            if self.task_progress.scope == 'single':
                self.state.update(status='failed', error=failure.get('code'))
                return {'action': action, 'target_id': target}
            # Defer is already recorded. Clear its execution binding before any
            # awaited bootstrap so a failed refresh cannot plan the skipped ID.
            self.reset_transfer(context)
        else:
            self.task_progress.failure = None
        if action in {'retry_grasp', 'defer'}:
            await self.observe(context, purpose='geometry', camera_ids=['3'], recovery=True)
            await self.execution_stage(context, 'recovery_collision_scene', self.build_scene, context, 'observation')
            context['requires_observation_pose'] = True
            context.pop('observation_pose_capture_id', None)
            context.pop('observation_pose_reference', None)
        if action != 'defer':
            self.reset_transfer(context, retry_target=target if action == 'retry_grasp' else None)
        return {'action': action, 'target_id': target}

    def invalidate_recovery_scene(self, context):
        context['scene_refresh_required'] = True
        self.state['facts'].update(scene_ready=False, plan_ready=False, place_plan_ready=False,
                                   grasps_ready=False, arms_evaluated=False)
        if not context.get('holding'):
            self.state['facts']['arm_selected'] = False
            self.state['planning_attempts'] = []
        for key in ('plan', 'place_plan', 'grasps', 'evaluations'):
            context.pop(key, None)

    async def _run(self, run_id, initial):
        raise NotImplementedError("Use the Responses program runner")

    async def evaluate_grasp_candidates(self, snapshot, target, arm, context, deadline):
        evaluations = context.get('evaluations')
        if evaluations is None or self.state['facts'].get('center_reachability_enabled'):
            evaluations = await self.call_algorithm(self.algorithms.evaluate_arms, snapshot,
                target, context['grasps'], context['scene'], requested_arm=arm,
                **({'destination_id': self.state['destination_id']} if self.state['destination_id'] else {}),
                planning_settings=context['planning_settings'], report_progress=True,
                timeout_s=max(0., deadline - time.monotonic()))
        return evaluations

    async def plan_grasp(self, snapshot, target, arm, context):
        authorized_arm = context.pop('retry_plan_authorized_arm', None)
        if authorized_arm != arm and any(attempt["arm"] == arm for attempt in self.state["planning_attempts"]):
            raise ValueError("arm_already_attempted:" + arm)
        attempt = {"arm": arm, "status": "running"}
        timeout = context["planning_settings"]["plan_timeout_s"]
        attempt["effective_timeout_s"] = timeout
        self.state["planning_attempts"].append(attempt)
        started = time.monotonic()
        deadline = started + timeout
        try:
            result = await self.plan_grasp_candidates(snapshot, target, arm, context, deadline)
            attempt["status"] = "succeeded"
            return result
        except asyncio.CancelledError:
            attempt["status"] = "cancelled"
            raise
        except Exception as error:
            message = str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
            attempt.update(status="failed", error=message[:1000],
                           code=getattr(error, "code", type(error).__name__),
                           details=copy.deepcopy(getattr(error, "details", None)))
            if getattr(error, 'code', None) == 'planning_no_valid_candidate':
                self.emit('planning_attempt_failed', '规划失败，技能程序已停止', attempt)
            raise
        finally:
            attempt["duration_s"] = time.monotonic() - started

    async def plan_grasp_candidates(self, snapshot, target, arm, context, deadline):
        evaluations = await self.evaluate_grasp_candidates(snapshot, target, arm, context, deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f'plan_grasp_timeout:{context["planning_settings"]["plan_timeout_s"]:g}s')
        return await self.call_algorithm(
            self.algorithms.plan_grasp, snapshot, target, arm, evaluations, context["scene"],
            planning_settings=context["planning_settings"], report_progress=True,
            **({"destination_id": self.state["destination_id"]}
               if self.state["task"] == "grasp_place" else {}), timeout_s=remaining)

    async def call_algorithm(self, function, *args, report_progress=False, timeout_s=None, **kwargs):
        """Bound the task wait without pretending a remote GPU request can be killed."""
        run_id = self.run_id
        skill = (self.state.get("current_step") or {}).get("skill", function.__name__)
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        self.planning_cancelled = cancelled

        def record_progress(kind, title, data):
            if cancelled.is_set() or self.run_id != run_id or self.state["status"] != "running":
                return
            step = self.state.get("current_step")
            if not step or step["skill"] != skill:
                return
            if kind in ('ik_batch_start', 'planning_batch_start'):
                step.update(stage=kind.removesuffix('_start'), stage_started_at=time.time())
            for key in ("arm", "batch_index", "batch_count", "candidate_indices", "success_count"):
                if key in data:
                    step[key] = data[key]
            if "error" in data:
                step["last_error"] = data["error"]
            self.emit(kind, title, data)
            if self.state.get("log_error"):
                cancelled.set()

        def progress(kind, title, data):
            if not cancelled.is_set() and not loop.is_closed():
                loop.call_soon_threadsafe(record_progress, kind, title, data)

        archive_call = self.experiment is not None
        self.algorithm_call_count += 1
        call_id = f"algorithm-{self.algorithm_call_count:04d}"
        call_started = time.monotonic()
        if archive_call:
            self.emit("algorithm_start", function.__name__, {"call_id": call_id,
                      "function": function.__name__, "args": args, "kwargs": kwargs})
        if report_progress:
            kwargs.update(progress=progress, cancelled=cancelled)
        visual_done = None
        if self.visual_inventory is not None and function in (
                self.algorithms.observe, getattr(self.algorithms, "verify_release", None),
                getattr(self, "verify_unpacking", None), getattr(self, "verify_right_bin_drop", None)):
            self.visual_inventory.audit, visual_done = self.visual_audit_sink()
            discovery = getattr(self.algorithms.perception, 'visual_discovery', None)
            if discovery is not None:
                discovery.audit = self.visual_inventory.audit

        def invoke():
            try:
                return function(*args, **kwargs)
            finally:
                if visual_done is not None:
                    visual_done.set()

        worker = asyncio.create_task(asyncio.to_thread(invoke))
        self.planning_worker = worker

        def consume_result(task):
            # Reset/timeout may leave a remote request in flight; retrieve its eventual exception.
            if not task.cancelled():
                task.exception()

        worker.add_done_callback(consume_result)
        deadline = asyncio.timeout(timeout_s)
        try:
            async with deadline:
                result = await asyncio.shield(worker)
            if archive_call:
                self.emit("algorithm_complete", function.__name__, {"call_id": call_id, "result": result,
                          "duration_s": time.monotonic() - call_started})
            return result
        except TimeoutError as error:
            if archive_call:
                self.emit("algorithm_error", function.__name__, {"call_id": call_id, "error": "timeout",
                          "duration_s": time.monotonic() - call_started})
            if not deadline.expired():
                raise
            cancelled.set()
            raise RuntimeError(f"{skill}_timeout:{timeout_s:g}s") from error
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except Exception as error:
            if archive_call:
                self.emit("algorithm_error", function.__name__, {"call_id": call_id,
                          "error": str(error) if isinstance(error, (ValueError, RuntimeError)) else type(error).__name__,
                          "sam3_queries": getattr(error, "sam3_queries", None),
                          "details": getattr(error, "details", None),
                          "duration_s": time.monotonic() - call_started})
            raise

    def refresh_observation_labels(self, context):
        observation = context.get('observation', {})
        capture = context.get('latest_capture', {})
        if (self.task_progress is not None and observation.get('observation_id')
                and observation.get('observation_id') == capture.get('observation_id')
                and observation.get('generation') == capture.get('generation')):
            self.update_inventory_labels(observation)

    def update_inventory_labels(self, observation):
        from .observation_labels import make_object_labels

        self.task_progress.assign_display_ids(observation)
        labels = make_object_labels(observation, self.task_progress.records, self.task_progress.membership,
                                    holding=self.holding_state, destination_id=self.state.get('destination_id'),
                                    placement_regions=self.task_progress.placement_requests.values())
        if labels != self.state.get('object_labels'):
            self.state['object_labels'] = labels
            self.record_annotation(observation, labels)

    def record_annotation(self, capture, labels):
        if self.experiment is None:
            return
        # Each revision has its own immutable image and status snapshot.
        path = f"observations/labels-{len(self.events) + 1:05d}"
        if capture.get('camera_ids'):
            capture = {**capture, 'views': {key: value for key, value in capture['views'].items()
                                          if key in capture['camera_ids']}}
        images = self.experiment.observation_images(path, capture, labels)
        self.emit('observation_annotated', '观测 RGB：名称、ID、搬运状态', {
            'observation_id': capture['observation_id'], 'object_labels': labels, 'annotated_rgb': images})

    def task_budgets(self, context):
        progress = self.task_progress
        count = (sum(value == 'required' for value in progress.membership.values())
                 if progress is not None and progress.inventory_finalized else 1)
        stages = 2 if progress is not None and progress.route == 'box_table_box' else 1
        transfers = max(1, count) * stages
        final_visual_capture = int(bool(context.get('visual_inventory_reference')))
        return {'normal': {'used': context.get('normal_capture_count', 0), 'limit': 2 * transfers + final_visual_capture},
                'recovery': {'used': context.get('recovery_capture_count', 0), 'limit': 8},
                'decision_limit': max(400, 40 * transfers + 80)}

    async def capture_observation(self, context, *, capture=None, recovery=False):
        seen = context.setdefault('captured_observation_ids', set())
        if capture is not None and capture['observation_id'] in seen:
            return capture
        category = 'recovery' if recovery or context.get('recovery_capture_mode') else 'normal'
        budget = self.task_budgets(context)[category]
        if budget['used'] >= budget['limit']:
            raise ValueError(category + '_observation_budget_exhausted')
        if capture is None:
            capture = await asyncio.to_thread(self.simulation.capture_rgbd)
            self.check_generation()
        seen.add(capture['observation_id'])
        context[category + '_capture_count'] = budget['used'] + 1
        context['observation_count'] = context.get('observation_count', 0) + 1
        self.state['observation_budget'] = self.task_budgets(context)
        return capture

    async def observe(self, context, *, checkpoint=None, capture=None, camera_ids=None,
                      purpose=None, recovery=False):
        if purpose == 'scene':
            initialized = context.get('scene') or (self.task_progress and self.task_progress.scene_inventory_observation_id)
            camera_ids = ['1', '2', '3'] if initialized else ['3']
        elif purpose == 'geometry' and context.get('scene'):
            camera_ids = ['1', '2', '3']
        if checkpoint is None:
            if self.state["facts"].get("released"):
                checkpoint = "post_place"
            elif self.state["facts"].get("lifted") and self.state["task"] != "place":
                checkpoint = "post_grasp"
            else:
                checkpoint = "standalone_place" if self.state["task"] == "place" else "initial"
        if checkpoint not in {"initial", "fine", "post_grasp", "post_place", "standalone_place"}:
            raise ValueError("observation_checkpoint_not_declared")
        capture = await self.capture_observation(context, capture=capture, recovery=recovery)
        context.setdefault('analyzed_capture_ids', set()).add(capture['observation_id'])
        self.record_observation({**capture, 'camera_ids': camera_ids or ['1', '2', '3']}, checkpoint)
        context[checkpoint + "_capture"] = capture
        context['latest_capture'] = capture
        if checkpoint == "standalone_place":
            context["post_grasp_capture"] = capture
        if checkpoint in {'post_grasp', 'post_place', 'standalone_place'}:
            self.sync_result_evidence(context, 'post_grasp' if checkpoint == 'standalone_place' else checkpoint)
        self.check_generation()
        previous = context.get('inventory_observation', context.get('observation'))
        observation_options = {"instruction": self.state["instruction"], "checkpoint": checkpoint}
        hint = copy.deepcopy(self.state.get("attributes", {"kind": "any", "color": "any"}))
        hint['environment_queries'] = copy.deepcopy(self.state.get('environment_queries', []))
        fixed_scene = bool(self.task_progress and self.task_progress.scene_inventory_observation_id)
        if fixed_scene:
            hint['scene_queries'] = self.task_progress.scene_queries()
            hint['scene_entities'] = copy.deepcopy(self.task_progress.scene_entities)
            hint['visual_reference'] = context.get('visual_inventory_reference')
            hint['visual_progress'] = self.task_progress.visual_progress()
            hint['placement_regions'] = list(copy.deepcopy(self.task_progress.placement_requests).values())
        if self.state["task"] != "grasp":
            hint["destination"] = copy.deepcopy(self.state.get("destination_attributes", {"color": "any"}))
        if previous and previous.get('generation') == capture['generation']:
            hint["table"] = context.get('frozen_table', previous["table"])
        if camera_ids is not None:
            hint["camera_ids"] = camera_ids
        observation_options["source_hint"] = hint
        # Containers can be the source support even when the destination is a marker.
        hint['include_containers'] = True
        if previous and previous.get("source") == "rgbd" and checkpoint != "initial" and not self.state['facts'].get('collection_task'):
            source = next((obj for obj in previous["objects"]
                           if obj["id"] == self.state.get("target_id")), None)
            if source is not None:
                # Prior measured semantics refine a query, never supply a new
                # mask, point cloud or identity for the current capture.
                hint.update(kind=source["kind"], color=source["color"])
        cache_key = (capture['observation_id'], checkpoint, tuple(camera_ids or ('1', '2', '3')),
                     hint.get('kind'), hint.get('color'),
                     tuple(item['prompt'] for item in hint['environment_queries']),
                     tuple((item['kind'], item['color'], item['prompt'], tuple(item['entity_ids']))
                           for item in hint.get('scene_queries', [])),
                     json.dumps(hint.get('visual_progress', {}), sort_keys=True))
        cached = context.setdefault('analysis_cache', {})
        if cache_key not in cached:
            selected = context.get('visual_plan', {}).pop(checkpoint, None)
            expected = ('initialize_inventory' if not fixed_scene else
                        'verify_transfer' if checkpoint == 'post_place' else 'associate_inventory')
            if selected != expected:
                raise ValueError('visual_decisions_selection_required')
            hint['visual_decision'] = {'source': getattr(self, 'visual_plan_source', 'responses_program'), 'purpose': selected}
            try:
                cached[cache_key] = await self.call_algorithm(self.algorithms.observe, capture, **observation_options)
            except PerceptionError as error:
                initial = getattr(error, 'initial_inventory', None)
                if initial and self.task_progress is not None:
                    self.check_generation()
                    self.task_progress.initialize_scene_inventory({'objects': [],
                        'scene_entities': initial['entities'], 'observation_id': initial['observation_id']})
                    context.setdefault('visual_inventory_reference', copy.deepcopy(initial['reference']))
                    self.emit('visual_inventory', 'RGB清单已保留，当前几何尚未完成', {
                        'scene_entities': initial['entities'], 'evidence': initial['evidence'], 'geometry_ready': False})
                    self.update_inventory_labels({**capture, 'objects': [], 'camera_ids': camera_ids or ['3'],
                        'scene_entities': initial['entities'], 'visual_discovery': initial['evidence']})
                raise
            context['analysis_count'] = context.get('analysis_count', 0) + 1
        observation = copy.deepcopy(cached[cache_key])
        if not isinstance(observation.get('visual_inventory'), dict):
            raise PerceptionError('visual_inventory_required')
        if observation.get("source") != "rgbd" or observation["generation"] != self.state["generation"]:
            raise ValueError("invalid_rgbd_observation_generation")
        if checkpoint == "standalone_place" and observation.get("observation_id") != capture["observation_id"]:
            raise ValueError("invalid_rgbd_observation_identity")
        observation["checkpoint"] = checkpoint
        observation['camera_ids'] = list(camera_ids or ('1', '2', '3'))
        if fixed_scene:
            identifiers = [obj['id'] for obj in observation['objects']]
            if len(set(identifiers)) != len(identifiers) or not set(identifiers) <= set(self.task_progress.scene_entities):
                raise PerceptionError('visual_inventory_invalid_ids')
            observation['mapping_unresolved'] = [
                {'previous_id': identifier, 'status': 'unseen', 'candidates': [],
                 'reason': 'fixed_collision_entity_not_observed'}
                for identifier, entity in self.task_progress.scene_entities.items()
                if 'collision' in entity.get('roles', ()) and entity.get('kind') != 'table'
                and identifier not in identifiers]
            # Missing measurements do not erase a known obstacle. The map
            # publisher rejects this evidence; a final visual check can still
            # finish without requiring an otherwise unnecessary motion map.
            context['association_evidence'] = {'source': 'openai_decisions',
                'mapping': {identifier: identifier for identifier in identifiers},
                'unresolved': copy.deepcopy(observation['mapping_unresolved'])}
            observation['association_evidence'] = copy.deepcopy(context['association_evidence'])
        from .placement_layout import identify_placement_regions

        identify_placement_regions(observation)
        context["observation"] = observation
        if context.get('scene', {}).get('observation_id') != observation['observation_id']:
            self.state['facts']['observation_map_ready'] = False
        context['observation_purpose'] = purpose
        visual_scene = context.get('visual_scene')
        if visual_scene:
            from .visual_scene import capture_fingerprint

            if context.get('visual_scene_fingerprint') == capture_fingerprint(
                    observation, mask_ids=context.get('visual_scene_mask_ids')):
                semantics = {item['id']: item for item in visual_scene.get('objects', [])}
                for obj in observation['objects']:
                    if obj['id'] in semantics:
                        obj['visual_semantics'] = copy.deepcopy(semantics[obj['id']])
            else:
                context.pop('visual_scene', None)
                context.pop('visual_scene_fingerprint', None)
                context.pop('visual_scene_mask_ids', None)
        self.state['facts']['scene_inspected'] = bool(context.get('visual_scene'))
        if camera_ids is None or purpose in {'scene', 'geometry'}:
            if self.task_progress is not None:
                before_membership = self.task_progress.snapshot()['unresolved_membership']
                self.task_progress.initialize_scene_inventory(observation)
                self.task_progress.observe(observation['objects'], observation['observation_id'])
                visual_inventory = observation['visual_inventory']
                for identifier, visibility in visual_inventory.get('visibility', {}).items():
                    if identifier in self.task_progress.scene_entities:
                        self.task_progress.scene_entities[identifier]['visual_visibility'] = visibility
                context.setdefault('visual_inventory_reference', copy.deepcopy(observation.get('visual_reference', observation)))
                if not self.task_progress.inventory_finalized:
                    membership = {key: value for key, value in visual_inventory.get('membership', {}).items()
                                  if key in self.task_progress.records}
                    self.task_progress.assess_membership(membership)
                for identifier, evidence in visual_inventory.get('progress', {}).items():
                    if identifier in self.task_progress.records:
                        record = self.task_progress.records[identifier]
                        field = ('destination_tracking_evidence' if record.get('completed_identity_track')
                                 else 'destination_visual_evidence')
                        record[field] = copy.deepcopy(evidence)
                self.emit('visual_inventory', 'Decisions 视觉清单与身份关联', visual_inventory)
                self.update_inventory_labels(observation)
                after_membership = self.task_progress.snapshot()['unresolved_membership']
                if ((self.task_progress.scope == 'collection' or self.transfer_routes)
                        and after_membership and after_membership != before_membership):
                    self.state['facts']['task_assessed'] = False
            context['inventory_observation'] = copy.deepcopy(observation)
        context["observation_checkpoint"] = checkpoint
        if checkpoint == "post_grasp":
            context["post_grasp_processed"] = capture["observation_id"]
            context["observation_checkpoint"] = "post_grasp_semantic"
        if checkpoint == "post_place":
            context["post_place_processed"] = capture["observation_id"]
        if checkpoint in {"initial", "standalone_place"}:
            if purpose == 'scene':
                evidence = {'views_ready': False, 'purpose': 'inventory_and_changes'}
            elif checkpoint == 'initial' and self.state['facts'].get('collection_task'):
                source = [obj for obj in observation['objects'] if obj['kind'] not in {'bin', 'marker'}]
                source_views = [source_view_evidence(observation, obj) for obj in source]
                evidence = {'views_ready': bool(source) and any(item['views_ready'] for item in source_views),
                    'source_ids': [obj['id'] for obj in source], 'source_views': source_views}
            elif checkpoint == "initial":
                evidence = initial_view_evidence(observation, self.state["task"], self.state.get("attributes", {}),
                                                 self.state.get("destination_attributes", {}))
            else:
                evidence = standalone_place_view_evidence(observation, self.state.get("destination_attributes", {}))
            context["view_evidence"] = evidence
            self.state["facts"]["views_ready"] = evidence["views_ready"]
        holding = context.get("holding")
        if holding and holding.get("verified"):
            self.check_holding(observation, holding)
        return observation

    def position_verdict(self, feedback, context=None):
        settings = (context or {}).get("planning_settings", self.planning_settings.model_dump())
        return verify_gripper_position(feedback,
            tolerance_fraction=settings.get("holding_position_tolerance_fraction", .02),
            stable_window_s=settings.get("holding_stable_window_s", .2))

    def check_holding(self, observation, holding):
        evidence = self.position_verdict(self.simulation.robot_snapshot()["arms"][holding["arm"]])
        holding["last_position_evidence"] = evidence
        holding["observation"] = copy.deepcopy(observation)
        if evidence["status"] != "held":
            self.state["facts"].update(grasp_held=False, verified=False)
            self.state["verification"] = combine_holding_evidence(evidence)
            holding.update(verified=False, reason="holding_" + evidence["status"])
        self.holding_state = copy.deepcopy(holding)

    def sync_result_evidence(self, context, phase):
        """Expose consumed versus new evidence, without choosing a next action."""
        identifier = context.get(phase + '_capture', {}).get('observation_id')
        verified = context.get(phase + '_verification_input', {})
        visual_id = context.get('visual_result', {}).get('observation_id')
        visual_current = bool(identifier and visual_id == identifier)
        pending = bool(identifier and (verified.get('observation_id') != identifier or
            visual_current and verified.get('visual_observation_id') != visual_id))
        fact = 'place_evidence_pending' if phase == 'post_place' else 'grasp_evidence_pending'
        active = bool(self.state.get('target_id'))
        self.state['facts'].update({fact: active and pending, 'result_capture_ready': active and bool(identifier),
                                    'result_inspected': active and visual_current})

    def record_observation(self, capture, checkpoint):
        if self.experiment is None or capture["observation_id"] in self.recorded_observations:
            return
        self.recorded_observations.add(capture["observation_id"])
        path = f"observations/frame-{len(self.recorded_observations):04d}"
        reference = self.experiment.artifact(path, {"checkpoint": checkpoint, "capture": capture})
        self.emit("observation_recorded", "保存算法观察", {"checkpoint": checkpoint,
                  "observation_id": capture["observation_id"], "observation_artifact": reference,
                  "camera_ids": capture.get('camera_ids', list(capture.get('views', {})))})
        self.record_annotation(capture, {'observation_id': capture['observation_id'],
            'generation': capture.get('generation'), 'analysis_status': 'unprocessed', 'items': []})

    def measured_state(self, context):
        # The map owns its capture. Verification frames do not silently replace it.
        observation = copy.deepcopy(context.get("scene_observation", context["observation"]))
        feedback = self.simulation.robot_snapshot()
        self.check_generation()
        observation["arms"] = feedback["arms"]
        observation["feedback_time"] = feedback["time"]
        return observation

    async def verify_scene(self, context):
        # Observations occur at named checkpoints. Planning consumes the frozen snapshot.
        self.check_generation()
        if context.get('scene_refresh_required'):
            raise ValueError('recovery_requires_fresh_collision_scene')
        scene = context.get("scene")
        observation = context.get("scene_observation", context["observation"])
        if scene and scene["observation_id"] != observation["observation_id"]:
            raise ValueError("scene_observation_stale")
        if (scene and context.get('scene_validated_capture_id') != context['observation']['observation_id']
                and context.get("observation_checkpoint") not in {"post_grasp", "post_grasp_semantic", "post_place", "standalone_place"}
                and scene["observation_id"] != context["observation"]["observation_id"]):
            raise ValueError("scene_observation_stale")

    async def build_scene(self, context, exclusion_state):
        if context.get('holding') or self.holding_state is not None:
            raise ValueError('grasp_scene_requires_empty_hands')
        if self.transfer_routes and exclusion_state == 'selected' and not context.get('observation_pose_capture_id'):
            raise ValueError('observation_pose_required_before_manipulation_map')
        if context.get('requires_observation_pose') and exclusion_state != 'observation':
            raise ValueError('changed_environment_requires_observation_pose')
        observation = context["observation"]
        route_map = self.transfer_routes and exclusion_state == 'selected'
        if (route_map or exclusion_state == 'selected' and context.get('scene_refresh_required')
                and context.get('observation_pose_capture_id')
                and context.get('observation_pose_capture_id') != observation['observation_id']):
            reference = context.get('observation_pose_reference', context.get('fine_capture', {}))
            same_pose = (reference.get('observation_id') == context['observation_pose_capture_id']
                and observation.get('time', -1) >= reference.get('time', float('inf'))
                and all(arm in reference.get('arms', {}) and arm in observation.get('arms', {})
                    and np.allclose(observation['arms'][arm]['joints_rad'], reference['arms'][arm]['joints_rad'],
                                    atol=np.deg2rad(1), rtol=0) for arm in ('A', 'B')))
            if not same_pose:
                raise ValueError('observation_pose_analysis_required')
            if route_map:
                feedback = self.simulation.robot_snapshot()
                if not all(np.allclose(feedback['arms'][arm]['joints_rad'], reference['arms'][arm]['joints_rad'],
                                       atol=np.deg2rad(1), rtol=0) for arm in ('A', 'B')):
                    raise ValueError('observation_pose_analysis_required')
        mapping_observation = observation
        if self.transfer_routes:
            visible_ids = {obj['id'] for obj in observation['objects']}
            placed_ids = {record['id'] for record in self.task_progress.records.values()
                          if record['status'] == 'completed' or record.get('completed_stages')}
            if placed_ids - visible_ids:
                raise ValueError('placed_object_not_observed:' + ','.join(sorted(placed_ids - visible_ids)))
        replacing_scene = "scene" in context
        head_only = exclusion_state == 'observation' and not replacing_scene
        if head_only:
            mapping_observation = mapping_observation | {'views': {'3': observation['views']['3']}}
        mapping_observation = mapping_observation | {'observation_pose_verified': bool(route_map),
            'task_member_ids': [key for key, value in self.task_progress.membership.items()
                                if value == 'required'] if self.task_progress else []}
        context["scene"] = await self.call_algorithm(
            self.algorithms.build_scene, mapping_observation, None if exclusion_state == 'observation' else self.state["target_id"],
            planning_settings=context["planning_settings"], exclusion_state=exclusion_state)
        table_cache = context['scene'].get('mapping_diagnostics', {}).get('table_cache', {})
        self.state['scene_evidence'] = {
            'scene_token': context['scene'].get('scene_token'), 'exclusion_state': exclusion_state,
            'excluded_instance_id': self.state['target_id'] if exclusion_state == 'selected' else None,
            'camera_ids': ['3'] if head_only else ['1', '2', '3'],
            'observed_instance_ids': [item['id'] for item in observation['objects']],
            'table_cache': copy.deepcopy(table_cache),
            'association_unresolved': copy.deepcopy(observation.get('mapping_unresolved', []))}
        table_model = context['scene'].get('table')
        if observation.get("demo_tables") and table_cache.get("table_revision"):
            context["frozen_table"] = copy.deepcopy(observation["table"]) | {
                "table_revision": table_cache["table_revision"],
                "source_observation_id": table_cache["source_observation_id"],
                "source": "initial_inventory_ransac_table"}
            observation["table"] = copy.deepcopy(context["frozen_table"])
        elif table_model and table_cache.get('table_revision'):
            context['frozen_table'] = copy.deepcopy(observation['table']) | {
                'position': [*table_model['pose'][:2], table_cache['measured_height_m']],
                'size': [*table_model['dims'][:2], .006],
                'surface_height_m': table_cache['measured_height_m'],
                'table_revision': table_cache['table_revision'],
                'initial_head_observation_id': table_cache['initial_head_observation_id'],
                'source': 'initial_head_measured_ransac_plane'}
            observation['table'] = copy.deepcopy(context['frozen_table'])
        context["scene_observation"] = copy.deepcopy(observation)
        if exclusion_state != 'observation':
            context['scene_validated_capture_id'] = observation['observation_id']
            context.pop('pending_transfer', None)
            self.state['facts']['transfer_pending'] = False
        context.pop('scene_refresh_required', None)
        self.state['facts'].update(scene_available=True, observation_map_ready=exclusion_state == 'observation',
                                  scene_ready=exclusion_state != 'observation')
        for name in ("plan", "place_plan", "place_plan_observation_id"):
            context.pop(name, None)
        self.state["facts"].update(plan_ready=False, place_plan_ready=False)
        if replacing_scene and not context.get("holding"):
            for name in ("evaluations",):
                context.pop(name, None)
            if context.get('grasps_observation_id') != observation['observation_id']:
                context.pop('grasps', None)
                self.state['facts']['grasps_ready'] = False
            bound_arm = self.task_progress.bound_arm() if self.task_progress else None
            self.state["facts"].update(arms_evaluated=False, arm_selected=bound_arm is not None)
            self.state.update(arm=bound_arm, planning_attempts=[])
        return context["scene"]

    @staticmethod
    def offset_pose(pose, axis, distance, in_tool_frame):
        result = np.asarray(pose, dtype=float).copy()
        offset = np.eye(3)["xyz".index(axis)] * distance
        if in_tool_frame:
            offset = result[:3, :3] @ offset
        result[:3, 3] += offset
        return result.tolist()

    async def execution_stage(self, context, stage, function, *args, **kwargs):
        self.check_generation()
        skill = self.state["current_step"]["skill"]
        self.state["current_step"].update(stage=stage, stage_started_at=time.time())
        evidence = {"skill": skill, "stage": stage, "arm": self.state["arm"],
                    "generation": self.state["generation"]}
        self.emit("execution_stage_start", stage, evidence)
        self.check_log()
        started = time.monotonic()
        try:
            result = await function(*args, **kwargs)
            self.check_generation()
        except Exception as error:
            self.emit("execution_stage_error", stage,
                      {**evidence, "error": str(error)[:1000], "duration_s": time.monotonic() - started})
            raise
        self.emit("execution_stage_complete", stage,
                  {**evidence, "result": result, "duration_s": time.monotonic() - started})
        return result

    async def motor(self, function, *args):
        return await asyncio.to_thread(function, *args, self.state["generation"])

    async def validate_motion(self, context, phase, command, binding=None, *, measured_observation=None, **kwargs):
        observation = self.measured_state(context) if measured_observation is None else measured_observation
        if binding is not None:
            holding = context['holding']
            expected = {'arm': self.state['arm'], 'instance_id': self.state['target_id'],
                        'generation': self.state['generation'], 'scene_token': context['scene']['scene_token'],
                        'owner_id': holding['owner_id']}
            if any(binding.get(key) != value for key, value in expected.items()):
                raise ValueError('holding_binding_mismatch')
            evidence = self.position_verdict(observation['arms'][self.state['arm']], context)
            evidence.update(feedback_time=observation['feedback_time'], phase=phase,
                            feedback_id=f"feedback-{observation['generation']}-{observation['feedback_time']}")
            holding['last_position_evidence'] = evidence
            self.emit('holding_feedback_checked', '持物运动前检查实时夹爪反馈', evidence)
            if evidence['status'] != 'held':
                holding.update(verified=False, reason='holding_' + evidence['status'])
                self.state['facts'].update(grasp_held=False, verified=False)
                self.state['verification'] = combine_holding_evidence(evidence)
                self.holding_state = copy.deepcopy(holding)
                raise ValueError('holding_feedback_' + evidence['status'])
            self.holding_state = copy.deepcopy(holding)
        return await self.call_algorithm(
            self.algorithms.validate_execution, observation, self.state["arm"],
            context["scene"], phase, command, binding,
            planning_settings=context["planning_settings"], **kwargs)

    def prepare_command(self, command):
        positions = np.asarray(command["positions_rad"], dtype=float)
        feedback = self.simulation.robot_snapshot()
        current = np.asarray(feedback["arms"][self.state["arm"]]["joints_rad"])
        if positions.ndim != 2 or positions.shape[1] != 7 or not len(positions) or not np.isfinite(positions).all():
            raise ValueError("execution_requires_finite_joint_commands")
        drift = float(np.max(np.abs(current - positions[0])))
        if drift > np.deg2rad(1.):
            raise TrajectoryStartError(self.state["arm"], current, positions[0], feedback)
        result = copy.deepcopy(command)
        if drift > 1e-8:
            result["positions_rad"] = np.vstack((current, positions)).tolist()
            result["start_connector"] = {"source": "measured_joint_feedback", "maximum_drift_deg": float(np.rad2deg(drift)),
                                         "duration_s": result["dt"]}
        return result

    async def gripper(self, context, closed, *, binding=None, contact_goal=None):
        # Only two planning states: fully open and measured holding closure.
        try:
            result = await self.motor(self.simulation.set_gripper, self.state["arm"], closed)
        finally:
            feedback = self.simulation.robot_snapshot()
            if feedback["generation"] == self.state["generation"]:
                if closed:
                    # Vision may subsequently fail; a possibly loaded hand must block a new grasp.
                    self.holding_state = {"owner_id": self.run_id, "generation": feedback["generation"],
                                          "arm": self.state["arm"], "instance_id": self.state["target_id"],
                                          "state": "provisional", "attached": False,
                                          "verified": False, "reason": "closure_requires_position_verification"}
                elif not closed and binding is not None and gripper_open_confirmed(feedback["arms"][self.state["arm"]]):
                    pending = {**copy.deepcopy(context["holding"]), "state": "release_pending", "verified": False,
                               "release_tcp": copy.deepcopy(feedback["arms"][self.state["arm"]]["tcp"]),
                               "attachment_detach_pending": True}
                    context["holding"] = pending
                    self.holding_state = copy.deepcopy(pending)
                    self.state["facts"].update(released=True, grasp_held=False)
                    await self.call_algorithm(self.algorithms.detach_carrying)
                    pending.update(attached=False, attachment_detach_pending=False)
                    self.state["collision_model"] = {"gripper": "actual_feedback", "attached": False}
                    self.holding_state = copy.deepcopy(pending)
                    self.state["facts"]["attached"] = False
        return {"command": result}

    async def capture_holding_reference(self, context):
        arm, target = self.state["arm"], self.state["target_id"]
        observation = self.measured_state(context)
        feedback = observation["arms"][arm]
        binding = carrying_binding(observation, target, arm, context["scene"]["scene_token"], owner_id=self.run_id)
        binding.update(PlanningSettings.model_validate(context["planning_settings"]).attachment.model_dump())
        source = observed_object(observation, target)
        context["points_before_lift"] = np.asarray(source["points_world_m"]).copy()
        context["tcp_before_lift"] = copy.deepcopy(feedback["tcp"])
        context["holding"] = {**binding, "state": "provisional", "verified": False, "attached": False,
                              "source_attributes": {"kind": source["kind"], "color": source["color"], "position": "any"},
                              "gripper_joint_positions_rad": {
                                  side: state["gripper_position_rad"] for side, state in observation["arms"].items()},
                              "observation": copy.deepcopy(context["observation"]),
                              "grasp_support_reference": grasp_support_reference(
                                  observation, self.state['target_id'], feedback['tcp'])}
        if context.get('visual_inventory_reference'):
            context['holding'].update(
                visual_inventory_reference=copy.deepcopy(context['visual_inventory_reference']),
                scene_entities=copy.deepcopy(self.task_progress.scene_entities),
                scene_inventory_observation_id=self.task_progress.scene_inventory_observation_id)
        from .geometry import TOOL_PROFILE

        context['holding']['grasp_support_reference'].update(arm=arm,
            tool_profile_revision=TOOL_PROFILE['revision'], reference_phase='after_close_before_lift',
            feedback_id=f"feedback-{observation['generation']}-{observation['feedback_time']}",
            feedback_time=observation['feedback_time'])
        self.holding_state = copy.deepcopy(context["holding"])
        return {"source": "cached_rgbd_and_closure_feedback", "observation_id": observation["observation_id"],
                "tcp_before_lift": context["tcp_before_lift"], "holding_verified": False}

    async def verify_grasp(self, context):
        started = time.perf_counter()
        feedback_snapshot = self.simulation.robot_snapshot()
        self.check_generation()
        observation = context["observation"]
        arm = self.state["arm"]
        result = self.position_verdict(feedback_snapshot.get('arms', {}).get(arm, {}), context)
        result.update(feedback_id=f"feedback-{feedback_snapshot['generation']}-{feedback_snapshot['time']}",
                      feedback_time=feedback_snapshot['time'],
                      verification_wall_ms=(time.perf_counter() - started) * 1000)
        context["grasp_verification"] = result
        self.state['facts']['grasp_evidence_pending'] = False
        self.holding_state["verification"] = copy.deepcopy(result)
        if not result["success"]:
            context["holding"].update(verified=False, reason=result["status"])
            self.state["facts"].update(grasp_held=False, verified=False)
            self.holding_state.update(verified=False, reason=result["status"])
            return result
        holding = context["holding"]
        # Keep the unobstructed pregrasp surfaces in the measured grasp TCP frame.
        # This approximation assumes negligible closure displacement and no slip.
        holding.update(state="held", verified=True, verification=copy.deepcopy(result),
                       observation=copy.deepcopy(observation),
                       scene=copy.deepcopy(context["scene"]),
                       scene_observation=copy.deepcopy(context["scene_observation"]))
        self.holding_state = copy.deepcopy(holding)
        return result

    async def prepare_attachment(self, context, observation=None):
        """Prepare the carrying collision model after an independent grasp verdict."""
        holding = context['holding']
        if not holding.get('verified'):
            raise ValueError('attachment_requires_verified_holding')
        observation = self.measured_state(context) if observation is None else observation
        feedback = observation['arms'][self.state['arm']]
        self.check_generation()
        if self.position_verdict(feedback, context)['status'] != 'held':
            raise ValueError('attachment_requires_current_holding_feedback')
        if holding.get("attached"):
            return holding["attachment_evidence"]
        prepared = await self.call_algorithm(self.algorithms.prepare_carrying, observation,
                                             self.state['arm'], context["scene"], holding,
                                             planning_settings=context["planning_settings"])
        if prepared.get("attached") is not True:
            raise ValueError("attachment_preparation_failed")
        holding.update(attached=True, attachment_evidence=prepared, scene_token=context["scene"]["scene_token"])
        self.holding_state = copy.deepcopy(holding)
        self.state["facts"].update(attached=True, scene_ready=True)
        self.state["collision_model"] = {"gripper": "actual_feedback", "attached": True,
                                          "prepare_wall_ms": prepared.get("prepare_wall_ms")}
        return prepared

    async def execute_grasp(self, context):
        context["execution_started"] = True
        arm = self.state["arm"]
        plan, settings = context["plan"], context["planning_settings"]
        await self.verify_scene(context)
        results = {}
        results["open_gripper"] = await self.execution_stage(context, "open_gripper", self.gripper, context, False)
        for phase in ("approach", "grasp"):
            command = self.prepare_command(plan["phases"][phase])
            check_phase = "approach_grasp" if phase == "approach" else "grasp"
            await self.validate_motion(context, check_phase, command, contact_goal_pose_world=plan["T_world_tcp"],
                                       previous_check=plan.get('command_checks', {}).get(phase))
            results[phase] = await self.execution_stage(context, phase, self.motor,
                self.simulation.execute_trajectory, arm, command["positions_rad"], command["dt"])
            if phase == "approach":
                results["dwell_approach"] = await self.execution_stage(context, "dwell_approach", self.motor,
                    self.simulation.dwell, settings["dwell_approach_s"])
        results["close_gripper"] = await self.execution_stage(context, "close_gripper", self.gripper,
            context, True, contact_goal=plan["T_world_tcp"])
        results["dwell_grasp"] = await self.execution_stage(context, "dwell_grasp", self.motor,
            self.simulation.dwell, settings["dwell_grasp_s"])
        results["capture_reference"] = await self.execution_stage(context, "capture_reference",
            self.capture_holding_reference, context)
        command = self.prepare_command(plan["phases"]["lift"])
        # Closure changes finger geometry after the official open-hand plan.
        # Check that new model once; attachment still begins only after lift.
        await self.validate_motion(context, 'grasp_lift', command)
        results["lift"] = await self.execution_stage(context, "lift", self.motor,
            self.simulation.execute_trajectory, arm, command["positions_rad"], command["dt"])
        results["dwell_lift"] = await self.execution_stage(context, "dwell_lift", self.motor,
            self.simulation.dwell, settings["dwell_lift_s"])
        return {"arm": arm, "candidate_index": plan.get("candidate_index"), "stages": results,
                "lift_carried_collision_checked": False, "lift_closed_gripper_collision_checked": True}

    async def execute_place(self, context):
        plan, settings = context["place_plan"], context["planning_settings"]
        arm = self.state["arm"]
        if set(plan["phases"]) != {"approach", "place", "retreat"}:
            raise ValueError("place_requires_three_planned_phases")
        if context.get("place_plan_observation_id", context["observation"]["observation_id"]) != context["observation"]["observation_id"]:
            raise ValueError("scene_observation_stale")
        await self.verify_scene(context)
        holding = context["holding"]
        if not holding.get("attached") or not holding.get("verified"):
            raise ValueError("place_requires_attachment")
        current_points = np.asarray(holding["points_tool_m"])
        if "holding_reference" in plan and not np.array_equal(current_points, plan["holding_reference"]):
            raise ValueError("holding_geometry_changed_since_place_plan")
        binding = copy.deepcopy(plan["carrying_binding"])
        results = {}
        for phase in ("approach", "place"):
            command = self.prepare_command(plan["phases"][phase])
            await self.validate_motion(context, phase, command, binding,
                                       previous_check=plan.get('command_checks', {}).get(phase))
            results[phase] = await self.execution_stage(context, phase, self.motor,
                self.simulation.execute_trajectory, arm, command["positions_rad"], command["dt"])
            key = "dwell_place_approach_s" if phase == "approach" else "dwell_place_s"
            await self.execution_stage(context, key, self.motor, self.simulation.dwell, settings[key])
        release_feedback = self.measured_state(context)
        context["release_tcp"] = release_feedback["arms"][arm]["tcp"]
        from .placement import release_endpoint_evidence
        endpoint = release_endpoint_evidence(plan['goal'], holding, context['release_tcp'])
        context['release_endpoint_evidence'] = endpoint
        self.emit('release_endpoint_checked', '松爪前核对实际释放位置', endpoint)
        if not endpoint['at_release_goal']:
            raise ValueError('release_endpoint_not_reached')
        results['release_collision_check'] = await self.execution_stage(
            context, 'release_collision_check', self.validate_motion, context, 'place',
            {'positions_rad': [release_feedback['arms'][arm]['joints_rad']]}, binding,
            measured_observation=release_feedback)
        context['pending_transfer'] = {'target_id': self.state['target_id'], 'destination_id': self.state['destination_id'],
            'destination_kind': observed_object(context['observation'], self.state['destination_id'])['kind'],
            'scene_token': context['scene']['scene_token'], 'tcp_before_lift': copy.deepcopy(
                context['tcp_before_lift'] if 'tcp_before_lift' in context
                else holding['grasp_support_reference']['T_world_tcp_at_grasp']),
            'release_tcp': copy.deepcopy(context['release_tcp']),
            'release_height_m': plan.get('goal', {}).get('release_reference', {}).get(
                'nominal_object_floor_gap_m', settings['trajectory_collision_clearance_m'])}
        results["open_gripper"] = await self.execution_stage(context, "open_gripper", self.gripper,
            context, False, binding=binding)
        feedback = self.simulation.robot_snapshot()["arms"][arm]
        if not gripper_open_confirmed(feedback):
            raise ValueError("release_not_confirmed_by_gripper")
        self.state["facts"].update(released=True, grasp_held=False, attached=False)
        await self.execution_stage(context, "dwell_release", self.motor,
                                   self.simulation.dwell, settings["dwell_release_s"])
        command = self.prepare_command(plan["phases"]["retreat"])
        await self.validate_motion(context, "retreat", command,
                                   previous_check=plan.get('command_checks', {}).get('retreat'))
        results["retreat"] = await self.execution_stage(context, "retreat", self.motor,
            self.simulation.execute_trajectory, arm, command["positions_rad"], command["dt"])
        await self.execution_stage(context, "dwell_retreat", self.motor,
                                   self.simulation.dwell, settings["dwell_retreat_s"])
        return {"arm": arm, "released": True, "open_confirmed": True,
                "stages": results}

    async def verify_place(self, context):
        """Verify physical release; execute() subsequently checks visual destination evidence."""
        snapshot = self.simulation.robot_snapshot()
        self.check_generation()
        feedback = snapshot.get("arms", {}).get(self.state["arm"], {})
        completed = bool(self.state["facts"].get("place_execution_completed"))
        released = bool(self.state["facts"].get("released"))
        opened = feedback.get("gripper_closed_command") is False and gripper_open_confirmed(feedback)
        reason = None
        if not completed:
            reason = "place_execution_incomplete"
        elif not released:
            reason = "release_not_recorded"
        elif not opened:
            reason = "open_position_not_confirmed"
        result = {"success": reason is None, "status": "confirmed" if reason is None else "unknown",
                  "reason": reason, "source": "gripper_position", "verification_scope": "release_only",
                  "destination_pose_verified": False,
                  "feedback_id": f"feedback-{snapshot['generation']}-{snapshot['time']}",
                  "feedback_time": snapshot['time'],
                  "open_confirmed": opened, "released": released, "execution_completed": completed,
                  "position_rad": feedback.get("gripper_position_rad"),
                  "gripper_limits_rad": feedback.get("gripper_limits_rad")}
        context["place_verification"] = result
        self.state['facts']['place_evidence_pending'] = False
        return result

    async def prepare_grasp_scene(self, context):
        """One bounded move/capture/analyze/map operation; the caller owns any recovery."""
        if self.transfer_routes and not self.task_progress.route_destinations:
            raise ValueError('complete_route_destinations_required')
        if context.get('holding') or self.holding_state is not None:
            raise ValueError('grasp_scene_requires_empty_hands')
        self.state['facts'].update(scene_ready=False, grasps_ready=False, plan_ready=False)
        scene = context.get('scene', {})
        observation = context['observation']
        current_motion_map = (scene.get('exclusion_state') == 'observation'
                              and scene.get('observation_id') == observation['observation_id']
                              and not context.get('scene_refresh_required'))
        if not current_motion_map:
            if scene:
                # A selected map omits the last source. Measure its new static
                # location before ANY return motion, including the next route leg.
                observation = await self.execution_stage(context, 'current_pose_observation', self.observe,
                    context, checkpoint='initial', purpose='scene')
                context['head_observation'] = copy.deepcopy(observation)
            elif context.get('head_observation'):
                context['observation'] = copy.deepcopy(context['head_observation'])
            await self.execution_stage(context, 'observation_collision_scene', self.build_scene, context, 'observation')
        await self.execution_stage(context, 'observation_motion', self.move_to_observe, context)
        observation = await self.execution_stage(context, 'geometry_observation', self.observe, context,
            checkpoint='fine', capture=context['fine_capture'], purpose='geometry')
        source = observed_object(observation, self.state['target_id'])
        evidence = source_view_evidence(observation, source)
        self.state['facts']['views_ready'] = evidence['views_ready']
        context['view_evidence'] = evidence
        if not evidence['views_ready']:
            raise ValueError('current_target_geometry_required')
        return await self.execution_stage(context, 'instance_scene', self.build_scene, context, 'selected')

    async def move_to_observe(self, context):
        if context.get("holding") or self.holding_state is not None:
            raise ValueError("observation_motion_requires_empty_hands")
        poses = observation_poses(context["planning_settings"])
        original_arm = self.state["arm"]
        results = {}
        try:
            retained_return = context.pop("use_retained_scene_for_return", False)
            observation = context["scene_observation"] if retained_return else context["observation"]
            scene = context.get('scene', {})
            if (not retained_return and (scene.get('exclusion_state') != 'observation'
                    or scene.get('observation_id') != observation.get('observation_id')
                    or context.get('scene_refresh_required'))):
                raise ValueError('observation_motion_requires_current_map')
            if not scene:
                raise ValueError('observation_motion_requires_current_map')
            motion_scene = scene | {'return_after_place': retained_return}
            plan = await self.call_algorithm(self.algorithms.plan_dual_observation_motion,
                self.measured_state(context), motion_scene, poses,
                planning_settings=context["planning_settings"])
            feedback = self.simulation.robot_snapshot()
            current = feedback["arms"]
            paths = {arm: np.asarray(plan["paths"][arm], dtype=float) for arm in ("A", "B")}
            if len(paths["A"]) != len(paths["B"]):
                raise ValueError("dual_trajectory_sample_counts_differ")
            for arm in ("A", "B"):
                if np.max(np.abs(paths[arm][0] - current[arm]["joints_rad"])) > np.deg2rad(1):
                    raise TrajectoryStartError(arm, current[arm]["joints_rad"], paths[arm][0], feedback)
            # Both arms get the same measured-start connector and time axis.
            paths = {arm: np.vstack((current[arm]["joints_rad"], paths[arm])).tolist() for arm in ("A", "B")}
            await self.call_algorithm(self.algorithms.validate_dual_observation,
                self.measured_state(context), motion_scene, paths,
                planning_settings=context["planning_settings"])
            results = await self.execution_stage(context, "observe_pose_AB", self.motor,
                self.simulation.execute_dual_trajectory, paths, plan["dt"])
            # Keep the prior world solely for recovery motion until a complete
            # fresh scene replaces it. It must not authorize manipulation.
            context['scene_refresh_required'] = True
            capture = await self.capture_observation(context)
            self.check_generation()
            context['fine_capture'] = capture
            context['latest_capture'] = capture
            self.record_observation(capture, 'fine')
            context['observation_pose_capture_id'] = capture['observation_id']
            context['observation_pose_reference'] = {key: copy.deepcopy(capture[key])
                for key in ('observation_id', 'time', 'arms')}
            context.pop('requires_observation_pose', None)
        finally:
            self.state["arm"] = original_arm
        for key in ("evaluations", "grasps", "plan"):
            context.pop(key, None)
        context['view_evidence'] = {'views_ready': False, 'geometry_analysis_pending': True,
            'observation_id': capture['observation_id'], 'reason': 'new_pose_capture_not_analyzed'}
        context['observation_checkpoint'] = 'geometry_capture_pending'
        self.state['facts']['scene_inspected'] = False
        self.state["facts"].update(views_ready=False, scene_ready=False, grasps_ready=False, arms_evaluated=False,
                                  observation_map_ready=False, scene_available=True, geometry_capture_ready=True)
        return {'movements': results, 'observation_id': capture['observation_id'], 'semantic_processing': False}

    async def inspect_scene(self, context):
        observation = context['observation']
        if not self.visual_assistance or not self.visual_assistance.enabled:
            return {'success': False, 'status': 'unknown', 'reason': 'visual_assistance_not_configured'}
        result = await self.call_algorithm(self.visual_assistance.inspect_scene, observation,
            observation['objects'], self.state['instruction'], call_budget=context.get('visual_call_budget'))
        if result.get('observation_id') != observation['observation_id']:
            raise ValueError('visual_scene_observation_mismatch')
        semantics = {item['id']: item for item in result.get('objects', [])}
        for obj in observation['objects']:
            if obj['id'] in semantics:
                obj['visual_semantics'] = copy.deepcopy(semantics[obj['id']])
        context['visual_scene'] = result
        from .visual_scene import capture_fingerprint

        context['visual_scene_fingerprint'] = capture_fingerprint(observation)
        # Geometry may add wrist masks to this same capture. Keep semantics only
        # while every original ID/mask and all RGB pixels remain unchanged.
        context['visual_scene_mask_ids'] = {camera: list(view.get('instance_masks', {}))
                                            for camera, view in observation['views'].items()}
        if self.task_progress is not None:
            before_membership = self.task_progress.snapshot()['unresolved_membership']
            self.task_progress.observe(observation['objects'], observation['observation_id'])
            after_membership = self.task_progress.snapshot()['unresolved_membership']
            if ((self.task_progress.scope == 'collection' or self.transfer_routes)
                    and after_membership and after_membership != before_membership):
                self.state['facts']['task_assessed'] = False
        return result




    async def execute(self, skill, context, arguments=None):
        context['visual_plan'] = dict(arguments.get('visual_plan', {})) if isinstance(arguments, dict) else {}
        if isinstance(arguments, dict):
            arguments = {key: value for key, value in arguments.items() if key != 'visual_plan'}
        target, arm = self.state["target_id"], self.state["arm"]
        if skill == 'observe_scene':
            capture = context.get('post_place_capture') if context.get('pending_transfer') else None
            if (context.get('observation_pose_capture_id')
                    and context['observation_pose_capture_id'] == context.get('latest_capture', {}).get('observation_id')):
                capture = context['latest_capture']
            consumed = context.setdefault('scene_inputs_consumed', set())
            if capture is not None and capture['observation_id'] in consumed:
                raise SkillSelectionError('visual_input_already_analyzed', skill,
                    {'new_visual_input': {'expected': True, 'actual': False}})
            result = await self.observe(context, checkpoint='post_place' if context.get('pending_transfer') else 'initial',
                capture=capture, camera_ids=['3'], purpose='scene')
            consumed.add(result['observation_id'])
            context['head_observation'] = copy.deepcopy(result)
            self.state['facts'].update(observed=True, inventory_ready=True)
            return result
        if skill == 'recover_task':
            return await self.recover_task(context, arguments['action'])
        if skill == 'assess_task':
            judgment = arguments['judgment']
            self.task_progress.assess_membership(arguments.get('membership', {}))
            self.task_progress.assess_groups(arguments.get('group_membership', {}))
            self.task_progress.assess_unseen(arguments.get('unseen_relevance', {}))
            if judgment == 'continue' and not self.task_progress.inventory_finalized:
                try:
                    self.task_progress.freeze_inventory(context['observation'])
                except ValueError as error:
                    raise SkillSelectionError('inventory_not_ready', skill, {
                        'inventory_finalized': {'expected': True, 'actual': False, 'reason': str(error)}}) from error
                self.state['facts']['inventory_finalized'] = True
            if context.get('observation'):
                self.update_inventory_labels(context['observation'])
            if judgment in {'complete', 'partial'}:
                self.state['status'] = self.task_progress.terminal_status(judgment)
            elif judgment == 'unknown':
                self.record_failure(context, skill, 'task_progress_unknown', {'status': 'unknown'})
            return {'judgment': judgment, 'task_progress': self.task_progress.snapshot()}
        if skill == 'inspect_scene':
            from .visual_scene import capture_fingerprint

            key = capture_fingerprint(context['observation'])
            consumed = context.setdefault('visual_inputs_consumed', set())
            if key in consumed:
                raise SkillSelectionError('visual_input_already_analyzed', skill,
                    {'new_visual_input': {'expected': True, 'actual': False}})
            result = await self.inspect_scene(context)
            consumed.add(key)
            return result
        if (skill in {'plan_grasp', 'plan_place', 'execute_grasp', 'execute_place'}
                and self.task_progress is not None and self.task_progress.attempts.get(target, 0) >= 3):
            raise ValueError('recovery_retry_budget_exhausted')
        observation = context["observation"]
        if skill == "identify_target":
            target = arguments["target_id"]
            if target in ("missing", "ambiguous"):
                raise ValueError("target_" + target)
            source = observed_object(observation, target)
            if source["kind"] in {'bin', 'marker'}:
                raise ValueError("grasp_target_cannot_be_destination_bin")
            if context.get('retry_target_id') not in (None, target):
                raise ValueError('retry_requires_original_target')
            if self.task_progress is not None:
                if (self.task_progress.route == 'direct' and self.task_progress.scope == 'single'
                        and self.task_progress.selection_order == 'explicit' and not self.task_progress.inventory_finalized):
                    self.task_progress.assess_membership({identifier: 'required' if identifier == target else 'excluded'
                        for identifier in self.task_progress.records})
                    self.task_progress.freeze_inventory(observation)
                    self.state['facts']['inventory_finalized'] = True
                self.task_progress.select(source, self.state.get('destination_id'))
                self.update_inventory_labels(observation)
                if self.state.get('destination_id'):
                    destination = observed_object(observation, self.state['destination_id'])
                    self.task_progress.records[target]['destination_kind'] = destination['kind']
            self.state["target_id"] = target
            context['view_evidence'] = source_view_evidence(observation, source)
            if context.get('observation_purpose') == 'scene':
                context['view_evidence'].update(views_ready=False, geometry_analysis_pending=True)
            self.state['facts']['views_ready'] = context['view_evidence']['views_ready']
            if context.get('scene', {}).get('target_id') != target:
                self.state['facts']['scene_ready'] = False
            context["initial_points_world"] = np.asarray(source["points_world_m"]).copy()
            return {"target_id": target}
        if skill == "identify_destination":
            destination = arguments["destination_id"]
            if destination in ("missing", "ambiguous"):
                raise ValueError("destination_" + destination)
            if observed_object(observation, destination)["kind"] not in {'bin', 'marker'}:
                raise ValueError("destination_requires_observed_bin_or_marker")
            if self.transfer_routes:
                if not self.task_progress.route_destinations:
                    self.task_progress.confirm_route_destinations(arguments.get('route_destinations', {}), observation)
                expected = self.task_progress.route_destinations.get(target, {}).get(str(self.task_progress.stage_index()))
                if destination != expected:
                    raise ValueError('destination_differs_from_confirmed_route')
                to_table = (self.task_progress.route == 'box_table' or
                    self.task_progress.route == 'box_table_box' and self.task_progress.stage_index() == 0)
                expected_kind = 'marker' if to_table else 'bin'
                if observed_object(observation, destination)['kind'] != expected_kind:
                    raise ValueError('destination_does_not_match_route_stage')
                if self.task_progress.route == 'box_table_box' and self.task_progress.stage_index() == 1:
                    current_id = target or self.task_progress.pending_route_id()
                    if original_source_bin(self.task_progress.records[current_id]) == destination:
                        raise ValueError('final_destination_must_differ_from_source_bin')
            self.state["destination_id"] = destination
            if self.task_progress is not None and target in self.task_progress.records:
                self.task_progress.records[target]['destination_id'] = destination
                self.task_progress.records[target]['destination_kind'] = observed_object(observation, destination)['kind']
                self.refresh_observation_labels(context)
            if context.get("holding") and context.get("scene_observation"):
                mapped = context["scene_observation"]["objects"]
                mapped[:] = [obj for obj in mapped if obj["id"] != destination]
                mapped.append(copy.deepcopy(observed_object(observation, destination)))
            return {"destination_id": destination}
        if skill == "build_scene":
            return await self.prepare_grasp_scene(context)
        if skill == "generate_grasps":
            if not self.state["facts"].get("views_ready"):
                raise ValueError("move_to_observe_required")
            context["grasps"] = await self.call_algorithm(self.algorithms.generate_grasps, observation, target, arm=arm)
            context['grasps_observation_id'] = observation['observation_id']
            return context["grasps"]
        if skill == 'evaluate_reachability':
            context['evaluations'] = await self.call_algorithm(self.algorithms.evaluate_reachability,
                observation, target, self.state['destination_id'],
                requested_arm=self.task_progress.bound_arm())
            context['reachability_evaluations'] = copy.deepcopy(context['evaluations'])
            return context['evaluations']
        if skill == "choose_arm":
            arm = arguments["arm"]
            if self.state['facts'].get('center_reachability_enabled'):
                if self.task_progress.bound_arm() not in (None, arm):
                    raise ValueError('stage_arm_is_fixed')
                if not context['evaluations'].get(arm, {}).get('reachable'):
                    raise ValueError('no_reachable_arm')
                if any(attempt['arm'] == arm for attempt in self.state['planning_attempts']):
                    raise ValueError('arm_already_attempted:' + arm)
                self.task_progress.bind_arm(arm)
                self.state['arm'] = arm
                return {'arm': arm, 'stage_index': self.task_progress.stage_index(), 'binding': 'entire_stage'}
            if arm not in ("A", "B") or not context["evaluations"][arm].get("valid_candidates"):
                raise ValueError("no_reachable_arm")
            if any(attempt["arm"] == arm for attempt in self.state["planning_attempts"]):
                raise ValueError("arm_already_attempted:" + arm)
            self.state["arm"] = arm
            return {"arm": arm}
        if skill == "plan_grasp":
            await self.verify_scene(context)
            context["plan"] = await self.plan_grasp(observation, target, arm, context)
            return context["plan"]
        if skill == "execute_grasp":
            return await self.execute_grasp(context)
        if skill == "verify_grasp":
            result = await self.verify_grasp(context)
            self.refresh_observation_labels(context)
            if result["success"]:
                self.state["facts"].update(grasp_completed=True, grasp_held=True,
                                          attached=bool(context['holding'].get('attached')))
                for key in ("plan",):
                    context.pop(key, None)
            return result
        if skill == "plan_place":
            await self.verify_scene(context)
            if self.task_progress is not None:
                self.task_progress.bind_placement_region(observed_object(observation, self.state['destination_id']), observation)
                self.refresh_observation_labels(context)
            planning_observation = self.measured_state(context)
            await self.execution_stage(context, 'prepare_attachment', self.prepare_attachment,
                                       context, planning_observation)
            self.state['current_step'].update(stage='placement_candidates', stage_started_at=time.time())
            context["place_plan"] = await self.call_algorithm(self.algorithms.plan_place,
                planning_observation, target, arm, context["scene"], context["holding"],
                self.state["destination_id"], planning_settings=context["planning_settings"],
                placement_request=self.task_progress.placement_request(self.state['destination_id'])
                if self.task_progress is not None else None,
                report_progress=True, timeout_s=context["planning_settings"]["place_timeout_s"])
            context["place_plan_observation_id"] = context["observation"]["observation_id"]
            return context["place_plan"]
        if skill == "execute_place":
            return await self.execute_place(context)
        if skill == "verify_place":
            result = await self.verify_place(context)
            if result["success"]:
                self.holding_state = None
                context.pop("holding", None)
                release_action = context['pending_transfer'].setdefault('release_action_id', result['feedback_id'])
                self.task_progress.record_release(target, feedback_id=result['feedback_id'], action_id=release_action)
                if context['pending_transfer'].get('verified'):
                    # A failed map build cannot undo a confirmed release or
                    # spend another image request. Retry the map from this frame.
                    evidence = self.task_progress.records[target]['destination_visual_evidence']
                else:
                    after = await self.execution_stage(context, 'post_place_observation', self.observe,
                        context, checkpoint='post_place', purpose='scene')
                    evidence = after['visual_inventory'].get('progress', {}).get(target, {
                        'status': 'unknown', 'source': 'openai_decisions',
                        'observation_id': after['observation_id']})
                    if evidence.get('observation_id') != after['observation_id']:
                        raise ValueError('visual_inventory_progress_frame_mismatch')
                confirmed = self.task_progress.confirm_visual_transfer(target, evidence)
                result.update(success=confirmed, status='confirmed' if confirmed else 'unknown',
                    verification_scope='release_and_visual_destination',
                    destination_pose_verified=False, destination_region_verified=confirmed,
                    visual_evidence=evidence, reason=None if confirmed else 'visual_destination_unconfirmed')
                context['pending_transfer']['verified'] = confirmed
                context['place_verification'] = result
                context.update(requires_observation_pose=True, scene_refresh_required=True)
                remaining = any(record['status'] not in {'completed', 'deferred'}
                                and self.task_progress.membership.get(identifier, 'required') == 'required'
                                for identifier, record in self.task_progress.records.items())
                if confirmed and remaining:
                    await self.execution_stage(context, 'post_place_collision_scene', self.build_scene,
                        context, 'observation')
            return result
        raise ValueError("unknown_skill:" + skill)

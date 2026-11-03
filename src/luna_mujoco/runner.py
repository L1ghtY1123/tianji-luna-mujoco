"""Shared motion, sensing, and recording primitives for the Responses program runner."""
import asyncio
import copy
import json
import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .observation_motion import (
    initial_view_evidence,
    observation_poses,
    source_view_evidence,
    standalone_place_view_evidence,
)
from .perception import PerceptionError
from .placement import carrying_binding, grasp_support_reference, observed_object
from .planning_settings import PlanningSettings
from .skills import Registry
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
            "feedback_time": feedback.get("feedback_time", feedback.get("time")),
        }


def public_summary(value):
    """Keep trace evidence usable without streaming thousands of cloud/trajectory entries."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
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
    def __init__(self, simulation, algorithms, decisions_factory, data_dir, *,
                 planning_settings=None, visual_assistance=None, experiments=None):
        if decisions_factory is not None:
            raise ValueError("legacy_decisions_factory_not_supported")
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
                   ('observation_id', 'request_hash', 'elapsed_s', 'http_status', 'error',
                    'model_call', 'validation_errors', 'call_id', 'transport_attempt')}
        summary.update(artifact=artifact, question_count=len(evidence['request'].get('questions', [])),
                       purpose=evidence.get('purpose'), endpoint=evidence.get('endpoint', '/v1/decisions'),
                       response=evidence.get('response'))
        summary['image_selection'] = evidence.get('image_selection')
        from .visual_image_selection import decision_image_count

        summary['image_count'] = decision_image_count(evidence['request'])
        self.emit('visual_decisions_call', 'Responses 初始任务编排'
                  if summary['endpoint'] == '/v1/responses' else 'Decisions 图片判断', summary)

    @staticmethod
    def idle_state():
        return {"status": "idle", "instruction": "", "task": None, "target_id": None,
                "arm": None, "destination_id": None, "facts": {}, "error": None, "error_code": None,
                "current_step": None, "failed_step": None,
                "planning_attempts": []}

    @property
    def active(self):
        return self.worker is not None and not self.worker.done()

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
        if kind == "skill_program":
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
            self.state["collision_model"] = {"gripper": "phase_specific", "attached": False,
                                              "lift_carried_collision_checked": False}
        elif skill in {"move_to_observe", "plan_place", "execute_place"}:
            self.state["collision_model"] = {"gripper": "actual_feedback",
                                              "attached": bool(self.holding_state and self.holding_state.get("attached"))}

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

    def reset_transfer(self, context):
        retained = {key: context[key] for key in ('initial', 'planning_settings', 'configured_planning_settings', 'observation', 'inventory_observation',
                    'initial_capture', 'fine_capture', 'latest_capture', 'post_place_capture', 'post_unpack_capture',
                    'post_place_verification_input', 'visual_call_budget',
                    'head_observation', 'visual_inventory_reference', 'visual_plan', 'frozen_table', 'normal_capture_count', 'recovery_capture_count',
                    'observation_count', 'captured_observation_ids', 'analyzed_capture_ids', 'capture_epochs',
                    'mapping_lifecycle', 'task_binding_evidence',
                    'observation_pose_capture_id', 'observation_pose_reference',
                    'scene', 'scene_observation', 'pending_transfer',
                    'scene_validated_capture_id',
                    'visual_inputs_consumed', 'scene_inputs_consumed',
                    'requires_observation_pose', 'scene_refresh_required', 'association_evidence',
                    'initial_joint_reference') if key in context}
        context.clear()
        context.update(retained)
        self.state.update(target_id=None, destination_id=None, arm=None, verification=None,
                          visual_result=None, planning_attempts=[], failure=None)
        self.state['facts'] = {fact: False for contract in self.registry.contracts.values()
                               for fact in (*contract.requires, *contract.provides)}
        collection = (self.task_progress.scope == 'collection'
                      or self.state['task'] != 'place' and self.task_progress.selection_order == 'inventory')
        self.state['facts'].update(needs_place=self.state['task'] != 'grasp',
            destination_ready=self.state['task'] == 'grasp', collection_task=collection,
            inventory_finalized=self.task_progress.inventory_finalized,
            visual_available=bool(self.visual_assistance and self.visual_assistance.enabled),
            task_assessed=not collection)
        self.state['facts']['center_reachability_enabled'] = True
        self.state['facts']['observed'] = bool(context.get('observation'))
        map_current = bool(context.get('scene')) and not context.get('scene_refresh_required')
        self.state['facts']['observation_map_ready'] = (
            map_current and context['scene'].get('exclusion_state') == 'observation')
        self.state['arm'] = self.task_progress.bound_arm()
        self.state['facts']['arm_selected'] = self.state['arm'] is not None
        self.state['facts'].update(transfer_pending=bool(context.get('pending_transfer')),
                                  scene_available=map_current)

    async def _run(self, run_id, initial):
        raise NotImplementedError("Use the Responses program runner")

    async def evaluate_grasp_candidates(self, snapshot, target, arm, context):
        evaluations = context.get('evaluations')
        if evaluations is None or self.state['facts'].get('center_reachability_enabled'):
            evaluations = await self.call_algorithm(self.algorithms.evaluate_arms, snapshot,
                target, context['grasps'], context['scene'], requested_arm=arm,
                **({'destination_id': self.state['destination_id']} if self.state['destination_id'] else {}),
                planning_settings=context['planning_settings'], report_progress=True)
        return evaluations

    async def plan_grasp(self, snapshot, target, arm, context):
        attempt = {"arm": arm, "status": "running"}
        self.state["planning_attempts"].append(attempt)
        started = time.monotonic()
        try:
            result = await self.plan_grasp_candidates(snapshot, target, arm, context)
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

    async def plan_grasp_candidates(self, snapshot, target, arm, context):
        evaluations = await self.evaluate_grasp_candidates(snapshot, target, arm, context)
        return await self.call_algorithm(
            self.algorithms.plan_grasp, snapshot, target, arm, evaluations, context["scene"],
            planning_settings=context["planning_settings"], report_progress=True,
            **({"destination_id": self.state["destination_id"]}
               if self.state["task"] == "grasp_place" else {}))

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
                getattr(self, "verify_unpacking", None)):
            self.visual_inventory.audit, visual_done = self.visual_audit_sink()
            resolver = getattr(getattr(self.algorithms, 'perception', None), 'cad_resolver', None)
            if resolver is not None:
                resolver.matcher.audit = self.visual_inventory.audit

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
                          "details": getattr(error, "details", None) or getattr(error, "evidence", None),
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
        self.emit('observation_annotated', '观测 RGB：身份与几何来源（无后续任务状态）', {
            'observation_id': capture['observation_id'], 'object_labels': labels, 'annotated_rgb': images})

    def task_budgets(self, context):
        progress = self.task_progress
        count = (sum(value == 'required' for value in progress.membership.values())
                 if progress is not None and progress.inventory_finalized else 1)
        transfers = max(1, count)
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
        if budget['limit'] > 0 and budget['used'] >= budget['limit']:
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
        if checkpoint != 'initial' or 'initial_capture' not in context:
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
            # The compilation boxes belong to this immutable calibrated frame.
            # A fresh first observation must not relocate those boxes onto new pixels.
            hint['reference_capture'] = context.get('initial_capture')
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
            cached[cache_key] = await self.call_algorithm(self.algorithms.observe, capture, **observation_options)
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
                reference = context.setdefault('visual_inventory_reference',
                    copy.deepcopy(observation.get('visual_reference', observation)))
                tracks = observation.get('visual_reference', {}).get('container_identity_tracks')
                if tracks and 'container_identity_tracks' not in reference:
                    anchor_id = tracks.get('anchor_observation_id', tracks['reference_observation_id'])
                    if (tracks['reference_observation_id'] != reference.get('observation_id')
                            or anchor_id != observation['observation_id']
                            or tracks.get('generation') != observation.get('generation')
                            or tracks.get('owner_id') and tracks['owner_id'] != reference.get('owner_id')):
                        raise ValueError('container_tracking_reference_mismatch')
                    reference['container_identity_tracks'] = copy.deepcopy(tracks)
                if not self.task_progress.inventory_finalized:
                    membership = {key: value for key, value in visual_inventory.get('membership', {}).items()
                                  if key in self.task_progress.records}
                    self.task_progress.assess_observed_membership(membership)
                for identifier, evidence in visual_inventory.get('progress', {}).items():
                    if identifier in self.task_progress.records:
                        self.task_progress.record_visual_progress(identifier, evidence)
                self.emit('visual_inventory', 'Decisions 视觉清单与身份关联', visual_inventory)
                self.update_inventory_labels(observation)
                after_membership = self.task_progress.snapshot()['unresolved_membership']
                if (self.task_progress.scope == 'collection'
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
        if context.get('requires_observation_pose') and exclusion_state != 'observation':
            raise ValueError('changed_environment_requires_observation_pose')
        observation = context["observation"]
        observation_pose_verified = False
        if exclusion_state == 'selected' and context.get('observation_pose_capture_id'):
            reference = context.get('observation_pose_reference', context.get('fine_capture', {}))
            same_pose = (reference.get('observation_id') == context['observation_pose_capture_id']
                and observation.get('time', -1) >= reference.get('time', float('inf'))
                and all(arm in reference.get('arms', {}) and arm in observation.get('arms', {})
                    and np.allclose(observation['arms'][arm]['joints_rad'], reference['arms'][arm]['joints_rad'],
                                    atol=np.deg2rad(1), rtol=0) for arm in ('A', 'B')))
            if not same_pose:
                raise ValueError('observation_pose_analysis_required')
            feedback = self.simulation.robot_snapshot()
            if not all(np.allclose(feedback['arms'][arm]['joints_rad'], reference['arms'][arm]['joints_rad'],
                                   atol=np.deg2rad(1), rtol=0) for arm in ('A', 'B')):
                raise ValueError('observation_pose_analysis_required')
            observation_pose_verified = True
        mapping_observation = observation
        model_mapping = bool(context.get('mapping_lifecycle', {}).get('requested_scope'))
        if self.task_progress is not None and not model_mapping:
            visible_ids = {obj['id'] for obj in observation['objects']}
            placed_ids = {record['id'] for record in self.task_progress.records.values()
                          if record['status'] != 'consumed'
                          and (record['status'] == 'completed' or record.get('transfers'))}
            if placed_ids - visible_ids:
                raise ValueError('placed_object_not_observed:' + ','.join(sorted(placed_ids - visible_ids)))
        replacing_scene = "scene" in context
        head_only = exclusion_state == 'observation' and not replacing_scene and not model_mapping
        if head_only:
            mapping_observation = mapping_observation | {'views': {'3': observation['views']['3']}}
        mapping_observation = mapping_observation | {'observation_pose_verified': observation_pose_verified,
            'mapping_lifecycle': copy.deepcopy(context.get('mapping_lifecycle', {})),
            'task_member_ids': [key for key, value in self.task_progress.membership.items()
                                if value == 'required'] if self.task_progress else []}
        context["scene"] = await self.call_algorithm(
            self.algorithms.build_scene, mapping_observation, None if exclusion_state == 'observation' else self.state["target_id"],
            planning_settings=context["planning_settings"], exclusion_state=exclusion_state)
        snapshot = context['scene'].pop('collision_snapshot', None)
        if snapshot is not None and self.experiment is not None:
            if any(snapshot.get(key) != context['scene'].get(key)
                   for key in ('scene_token', 'observation_id', 'generation')):
                snapshot = {'status': 'unavailable', 'reason': 'collision_snapshot_binding_mismatch'}
            artifact = self.experiment.collision_scene(f'collision/stage-{len(self.events) + 1:05d}', snapshot)
            self.emit('collision_scene_archived', '阶段碰撞场景', {
                'snapshot_artifact': artifact, 'status': snapshot.get('status'),
                'scene_token': context['scene'].get('scene_token'),
                'observation_id': context['scene'].get('observation_id'),
                'exclusion_state': exclusion_state,
                'target_id': self.state.get('target_id')})
        table_cache = context['scene'].get('mapping_diagnostics', {}).get('table_cache', {})
        self.state['scene_evidence'] = {
            'scene_token': context['scene'].get('scene_token'), 'exclusion_state': exclusion_state,
            'excluded_instance_id': self.state['target_id'] if exclusion_state == 'selected' else None,
            'camera_ids': ['3'] if head_only else ['1', '2', '3'],
            'observed_instance_ids': [item['id'] for item in observation['objects']],
            'table_cache': copy.deepcopy(table_cache),
            'inventory_map': copy.deepcopy(context['scene'].get('inventory_map', {})),
            'coverage': copy.deepcopy(context['scene'].get('coverage', {})),
            'mapping_strategy': copy.deepcopy(context['scene'].get('mapping_strategy', {})),
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
        inventory_map = context['scene'].get('inventory_map', {})
        lifecycle = context.get('mapping_lifecycle', {})
        if inventory_map:
            records = inventory_map.get('entities', {})
            retired = set(lifecycle.get('retired_entity_ids', ()))
            lifecycle['invalidated_entity_ids'] = [identifier
                for identifier in lifecycle.get('invalidated_entity_ids', ())
                if identifier not in retired
                and records.get(identifier, {}).get('status') not in {'observed', 'retained'}]
        complete = context['scene'].get('coverage', {}).get('inventory_complete') is not False
        if complete:
            context.pop('scene_refresh_required', None)
        else:
            context['scene_refresh_required'] = True
        self.state['facts'].update(scene_available=complete,
                                  observation_map_ready=complete and exclusion_state == 'observation',
                                  scene_ready=complete and exclusion_state != 'observation')
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
        if not complete:
            coverage = context['scene'].get('coverage', {})
            missing = coverage.get('missing_entity_ids', [])
            stale = coverage.get('stale_entity_ids', [])
            return {**context['scene'], 'success': False, 'status': 'unknown',
                    'reason': 'inventory_map_incomplete', 'details': {
                        'component': 'perception_mapping', 'observation_id': observation['observation_id'],
                        'missing_entity_ids': missing, 'stale_entity_ids': stale,
                        'diagnostic_checks': [{'name': 'current_inventory_geometry',
                            'question': '所有未退役清单实体是否具有当前有效的实测碰撞层？',
                            'status': 'unknown', 'actual': {'missing': missing, 'stale': stale},
                            'expected': {'missing': [], 'stale': []}}],
                        'sam3_queries': [{key: query[key] for key in
                            ('camera_id', 'prompt', 'candidate_count', 'candidate_scores', 'status') if key in query}
                            for query in observation.get('sam3_queries', [])]}}
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

    async def motor(self, function, *args, **kwargs):
        if function in (self.simulation.execute_trajectory, self.simulation.set_gripper):
            kwargs['execution_settings'] = self.state.get('planning_settings', self.planning_settings.model_dump())
        return await asyncio.to_thread(function, *args, self.state["generation"], **kwargs)

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

    async def validate_during_dwell(self, context, phase, command, binding=None, *, duration_s, **kwargs):
        """Count validation inside the wall-clock hold, never append another full hold."""
        started = time.monotonic()
        check = await self.validate_motion(context, phase, command, binding, **kwargs)
        validation_s = time.monotonic() - started
        await self.motor(self.simulation.dwell, max(0., duration_s - validation_s))
        return {**check, 'requested_duration_s': duration_s,
                'validation_s': validation_s, 'duration_s': time.monotonic() - started,
                'overrun_s': max(0., validation_s - duration_s)}

    def prepare_command(self, command, *, feedback=None):
        positions = np.asarray(command["positions_rad"], dtype=float)
        feedback = self.simulation.robot_snapshot() if feedback is None else feedback
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
        close_limit = None
        if closed:
            proof = context.get('plan', {}).get('command_checks', {}).get('closure', {})
            if proof.get('full_range_rejection'):
                close_limit = proof['grasp_closure_preflight']['safe_driver_limit_rad']
            feedback = self.measured_state(context)
            await self.validate_motion(context, 'grasp_closure',
                {'positions_rad': [feedback['arms'][self.state['arm']]['joints_rad']], 'dt': .02},
                measured_observation=feedback, closure_driver_limit_rad=close_limit)
        try:
            options = {} if close_limit is None else {'maximum_position_rad': close_limit}
            result = await self.motor(self.simulation.set_gripper, self.state["arm"], closed, **options)
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
                                  observation, self.state['target_id'], feedback['tcp'],
                                  source_reference=context.get('plan', {}).get('lift_clearance'),
                                  scene_token=context['scene']['scene_token'],
                                  source_record=self.task_progress.records.get(target) if self.task_progress else None)}
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
                                             planning_settings=context["planning_settings"],
                                             placement_request=context.get('grasp_placement_request'))
        if prepared.get("attached") is not True:
            raise ValueError("attachment_preparation_failed")
        holding.update(attached=True, attachment_evidence=prepared, scene_token=context["scene"]["scene_token"])
        self.holding_state = copy.deepcopy(holding)
        self.state["facts"].update(attached=True, scene_ready=True)
        self.state["collision_model"] = {"gripper": "actual_feedback", "attached": True,
                                          "prepare_wall_ms": prepared.get("prepare_wall_ms")}
        return prepared

    async def execute_grasp(self, context):
        arm = self.state["arm"]
        plan, settings = context["plan"], context["planning_settings"]
        await self.verify_scene(context)
        # Source geometry is already known before any motor command. Check it
        # now; closure still records the actual measured TCP independently.
        grasp_support_reference(self.measured_state(context), self.state['target_id'], plan['T_world_tcp'],
                                source_reference=plan.get('lift_clearance'),
                                scene_token=context['scene']['scene_token'],
                                source_record=self.task_progress.records.get(self.state['target_id'])
                                if self.task_progress else None)
        context["execution_started"] = True
        results = {}
        results["open_gripper"] = await self.execution_stage(context, "open_gripper", self.gripper, context, False)
        for phase in ("approach", "grasp"):
            feedback = self.measured_state(context)
            command = self.prepare_command(plan["phases"][phase], feedback=feedback)
            check_phase = "approach_grasp" if phase == "approach" else "grasp"
            await self.validate_motion(context, check_phase, command, contact_goal_pose_world=plan["T_world_tcp"],
                                       measured_observation=feedback,
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
        closed = self.simulation.robot_snapshot()['arms'][arm]
        closure_evidence = self.position_verdict(closed, context)
        if closure_evidence['status'] != 'held':
            raise ValueError('grasp_closure_limit_reached_without_holding')
        results["capture_reference"] = await self.execution_stage(context, "capture_reference",
            self.capture_holding_reference, context)
        feedback = self.measured_state(context)
        command = self.prepare_command(plan["phases"]["lift"], feedback=feedback)
        # The initial lift checks the actual closed robot without an object
        # attachment. Its result does not certify the object's swept volume.
        results['lift_collision_check'] = await self.execution_stage(context, 'validate_lift',
            self.validate_motion, context, 'grasp_lift', command, measured_observation=feedback)
        results["lift"] = await self.execution_stage(context, "lift", self.motor,
            self.simulation.execute_trajectory, arm, command["positions_rad"], command["dt"])
        results["dwell_lift"] = await self.execution_stage(context, "dwell_lift", self.motor,
            self.simulation.dwell, settings["dwell_lift_s"])
        results['verify_holding_after_lift'] = await self.execution_stage(context, 'verify_holding_after_lift',
            self.verify_grasp, context)
        if not results['verify_holding_after_lift']['success']:
            raise ValueError('grasp_not_held_after_lift')
        results['prepare_attachment'] = await self.execution_stage(context, 'prepare_attachment',
            self.prepare_attachment, context)
        carried_scope = 'observed_surface_only'
        if context['holding'].get('cad_tool') is not None:
            carried_scope = 'registered_cad_volume'
        elif context['holding'].get('obb_tool') is not None:
            carried_scope = 'observed_obb_volume'
        return {"arm": arm, "candidate_index": plan.get("candidate_index"), "stages": results,
                "lift_carried_collision_checked": False, "lift_closed_gripper_collision_checked": True,
                "carried_geometry_scope": carried_scope}

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
            feedback = self.measured_state(context)
            command = self.prepare_command(plan["phases"][phase], feedback=feedback)
            if phase == 'place':
                results['dwell_place_approach_s'] = await self.execution_stage(
                    context, 'dwell_place_approach_s', self.validate_during_dwell,
                    context, phase, command, binding, duration_s=settings['dwell_place_approach_s'],
                    measured_observation=feedback,
                    previous_check=plan.get('command_checks', {}).get(phase))
            else:
                await self.validate_motion(context, phase, command, binding,
                                           measured_observation=feedback,
                                           previous_check=plan.get('command_checks', {}).get(phase))
            results[phase] = await self.execution_stage(context, phase, self.motor,
                self.simulation.execute_trajectory, arm, command["positions_rad"], command["dt"])
        release_feedback = self.measured_state(context)
        context["release_tcp"] = release_feedback["arms"][arm]["tcp"]
        from .demo_placement import release_endpoint_evidence
        endpoint = release_endpoint_evidence(plan['goal'], holding, context['release_tcp'])
        context['release_endpoint_evidence'] = endpoint
        self.emit('release_endpoint_checked', '松爪前核对实际释放位置', endpoint)
        if not endpoint['at_release_goal']:
            raise ValueError('release_endpoint_not_reached')
        results['release_collision_check'] = await self.execution_stage(
            context, 'release_collision_check', self.validate_during_dwell, context, 'place',
            {'positions_rad': [release_feedback['arms'][arm]['joints_rad']]}, binding,
            measured_observation=release_feedback, duration_s=settings['dwell_place_s'])
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
        # Release changes object geometry even when later visual verification is
        # inconclusive. The already checked retreat belongs to this skill;
        # subsequent skills require a model-selected measured map update.
        context['scene_refresh_required'] = True
        lifecycle = context.setdefault('mapping_lifecycle', {})
        lifecycle['invalidated_entity_ids'] = sorted(
            set(lifecycle.get('invalidated_entity_ids', ())) | {self.state['target_id']})
        self.state['facts'].update(scene_available=False, scene_ready=False, observation_map_ready=False)
        feedback = self.measured_state(context)
        command = self.prepare_command(plan["phases"]["retreat"], feedback=feedback)
        results['dwell_release'] = await self.execution_stage(context, 'dwell_release',
            self.validate_during_dwell, context, 'retreat', command,
            duration_s=settings['dwell_release_s'],
            measured_observation=feedback,
            previous_check=plan.get('command_checks', {}).get('retreat'))
        results["retreat"] = await self.execution_stage(context, "retreat", self.motor,
            self.simulation.execute_trajectory, arm, command["positions_rad"], command["dt"])
        await self.execution_stage(context, "dwell_retreat", self.motor,
                                   self.simulation.dwell, settings["dwell_retreat_s"])
        completed_feedback = self.simulation.robot_snapshot()
        self.check_generation()
        return {"arm": arm, "released": True, "open_confirmed": True,
                "stages": results, 'completed_feedback': {
                    key: completed_feedback[key] for key in ('generation', 'time')}}

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

    async def move_to_observe(self, context, *, joint_goals=None):
        if context.get("holding") or self.holding_state is not None:
            raise ValueError("observation_motion_requires_empty_hands")
        poses = observation_poses(context["planning_settings"]) if joint_goals is None else joint_goals
        original_arm = self.state["arm"]
        results = {}
        try:
            retained_return = context.pop("use_retained_scene_for_return", False)
            observation = context["scene_observation"] if retained_return else context["observation"]
            scene = context.get('scene', {})
            if (context.get('scene_refresh_required') or not retained_return and (
                    scene.get('exclusion_state') != 'observation'
                    or scene.get('observation_id') != observation.get('observation_id'))):
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
                self.simulation.execute_dual_trajectory, paths, plan["dt"],
                execution_settings=context['planning_settings'])
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

    async def execute(self, skill, context, arguments=None):
        """Execute one shared primitive; task policy belongs to the registered graph."""
        if skill not in {'identify_target', 'plan_grasp', 'execute_grasp', 'verify_grasp', 'execute_place'}:
            raise ValueError("unknown_skill:" + skill)
        target, arm = self.state["target_id"], self.state["arm"]
        observation = context["observation"]
        if skill == "identify_target":
            target = arguments["target_id"]
            if target in ("missing", "ambiguous"):
                raise ValueError("target_" + target)
            source = observed_object(observation, target)
            if source["kind"] in {'bin', 'marker'}:
                raise ValueError("grasp_target_cannot_be_destination_bin")
            if self.task_progress is not None:
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
        if skill == "execute_place":
            return await self.execute_place(context)

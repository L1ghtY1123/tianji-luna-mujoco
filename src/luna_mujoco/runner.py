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
from .workspace import WorkspaceAtlas

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
    from dataclasses import fields, is_dataclass
    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): list(v) if k == "candidate_indices" and isinstance(v, (list, tuple)) else public_summary(v)
                for k, v in value.items() if not str(k).startswith("_")
                and k not in {"rgb", "depth_m", "valid_depth", "instance_masks", "masks", "mask",
                              "valid_depth_mask", "pixels_rc", "completed_geometry_points_world_m",
                              "points_world_m", "points_object_m", "points_tool_m", "point_sources", "views",
                              "silhouette_mask", "raw_identity_mask", "points", "pixels", "view"}}
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return {"count": len(value), "first": [public_summary(v) for v in value[:3]]}
        return [public_summary(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Runner:
    def __init__(self, simulation, algorithms, decisions_factory, data_dir, *,
                 planning_settings=None, experiments=None):
        if decisions_factory is not None:
            raise ValueError("legacy_decisions_factory_not_supported")
        self.simulation = simulation
        self.algorithms = algorithms
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
        from .object_memory import ObjectMemory
        from .spatial_memory import SpatialMemory
        self.object_memory = ObjectMemory()
        self.spatial_memory = SpatialMemory()
        self.geometry_measurements = {}
        self.planning_regions = {}
        self.workspace_atlas = WorkspaceAtlas()

    @property
    def visual_identity(self):
        return getattr(getattr(self.algorithms, 'perception', None), 'visual_identity', None)

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
        title = 'Decisions 图片判断'
        if summary['endpoint'] == '/v1/responses':
            revision = (summary.get('model_call') or {}).get('attempt', 0)
            title = ('Responses 物体记忆更新' if summary['purpose'] == 'update_object_memory' else
                     f'Responses 图校验修订 {revision}' if revision else 'Responses 初始任务编排')
        self.emit('visual_decisions_call', title, summary)

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
            from .runtime.memory import execution_snapshot
            self.state['task_progress'] = execution_snapshot(
                self.task_progress, self.object_memory, self.geometry_measurements, self.planning_regions)
            self.state['temporal_memory'] = self.object_memory.snapshot()
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
        if kind in {"skill_program", "terminal_recovery_started"}:
            event["data"] = data
        if self.experiment is not None:
            self.experiment.event(event, data)
        self.events.append(event)
        if self.run_id and self.experiment is None and not self.state.get("log_error"):
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                with (self.data_dir / (self.run_id + ".jsonl")).open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event, ensure_ascii=False, default=encode_numpy) + "\n")
            except OSError as error:
                # Keep terminal events available over HTTP/WS even when disk logging fails.
                self.state["log_error"] = "run_log_write_failed:" + type(error).__name__
                LOGGER.exception("Run event could not be written")

    def check_log(self):
        from .core.failure_policy import TERMINAL_SKILLS

        if (self.state.get("log_error")
                and (self.state.get('current_step') or {}).get('skill') not in TERMINAL_SKILLS):
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
        elif skill in {"move_to_observe", "plan_pose", "execute_place"}:
            self.state["collision_model"] = {"gripper": "actual_feedback",
                                              "attached": bool(self.holding_state and self.holding_state.get("attached"))}

    def open_experiment_memory(self, run_id):
        """Policy runners bind their persistent instance store before publishing a new run."""

    async def start(self, instruction, *, experiment=None):
        if self.active:
            raise ValueError("run_already_active")
        if self.algorithm_pending:
            raise ValueError("algorithm_request_pending")
        if self.algorithm_cleanup_error:
            raise ValueError("algorithm_cleanup_failed_reset_to_retry")
        if not 1 <= len(instruction.strip()) <= 4000:
            raise ValueError("instruction_length_must_be_1_to_4000")
        from .core.scene_scope import scene_sections

        scene_sections(instruction)
        snapshot = self.simulation.robot_snapshot()
        if not snapshot["confirmed"]:
            raise ValueError("confirm_scene_first")
        run_id = uuid.uuid4().hex
        self.open_experiment_memory(run_id)
        self.run_id = run_id
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
        self.motion_epoch = self.simulation.motion_epoch()
        self.worker = asyncio.create_task(self._run(self.run_id, snapshot))
        return copy.deepcopy(self.state)

    async def cancel(self):
        # Stop invalidates motor epochs before cancelling async work.
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
        self.geometry_measurements = {}
        self.planning_regions = {}
        self.run_id = None
        self.events = []
        self.state = self.idle_state()
        self.experiment = None
        self.invocation_id = None

    def clear_algorithm_state(self):
        """Queue Reset cleanup behind any old GPU request; new runs wait for both."""
        pending = (self.planning_worker, self.algorithm_cleanup_worker)
        generation = self.simulation.snapshot()["generation"]
        self.algorithm_cleanup_revision += 1
        revision = self.algorithm_cleanup_revision
        self.algorithm_cleanup_error = None

        async def cleanup():
            for worker in pending:
                if worker is not None:
                    await asyncio.gather(asyncio.shield(worker), return_exceptions=True)
            # A later scene/reset schedules its own cleanup. Never clear its newer result.
            if (self.simulation.snapshot()["generation"] != generation
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
        if self.simulation.robot_snapshot(include_tcp=False)["generation"] != self.state["generation"]:
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
        retained = {key: context[key] for key in ('initial', 'planning_settings', 'observation', 'objects_observation',
                    'initial_capture', 'fine_capture', 'latest_capture', 'post_place_capture', 'post_unpack_capture',
                    'post_place_verification_input',
                    'head_observation', 'identity_reference', 'frozen_table', 'normal_capture_count', 'recovery_capture_count',
                    'observation_count', 'captured_observation_ids', 'analyzed_capture_ids', 'capture_epochs',
                    'scene_observation', 'registered_capture', 'placement_semantics', 'grasp_sampling_budget',
                    'observation_pose_capture_id', 'observation_pose_reference',
                    'pending_transfer',
                    'visual_inputs_consumed', 'scene_inputs_consumed',
                    'requires_observation_pose', 'association_evidence',
                    'initial_joint_reference', 'objects_updates', '_observation_analysis', 'initial_plan',
                    'grounded_refs', 'grounded_observation', '_grounding_attempt') if key in context}
        grounded = self.state['facts'].get('task_grounded', False)
        capture_ready = self.state['facts'].get('capture_ready', False)
        context.clear()
        context.update(retained)
        self.state.update(target_id=None, destination_id=None, arm=None, verification=None,
                          visual_result=None, planning_attempts=[], failure=None)
        self.state['facts'] = {fact: False for contract in self.registry.contracts.values()
                               for fact in (*contract.requires, *contract.provides)}
        collection = (self.task_progress.scope == 'collection'
                      or self.state['task'] != 'place' and self.task_progress.selection_order == 'visual')
        self.state['facts'].update(needs_place=self.state['task'] != 'grasp',
            destination_ready=self.state['task'] == 'grasp', collection_task=collection,
            task_grounded=grounded, capture_ready=capture_ready)
        self.state['facts']['center_reachability_enabled'] = True
        self.state['facts']['observed'] = bool(context.get('observation'))
        map_current = bool(self.spatial_memory.scene) and not self.spatial_memory.refresh_required
        self.state['facts']['observation_map_ready'] = (
            map_current and self.spatial_memory.scene.get('exclusion_state') == 'observation')
        self.state['arm'] = self.task_progress.bound_arm()
        self.state['facts']['arm_selected'] = self.state['arm'] is not None
        self.state['facts'].update(transfer_pending=bool(context.get('pending_transfer')),
                                  collision_map_initialized=self.spatial_memory.initialized,
                                  scene_available=map_current)

    async def _run(self, run_id, initial):
        raise NotImplementedError("Use the Responses program runner")

    async def evaluate_grasp_candidates(self, snapshot, target, arm, context):
        evaluations = context.get('evaluations')
        if evaluations is None or self.state['facts'].get('center_reachability_enabled'):
            evaluations = await self.call_algorithm(self.algorithms.evaluate_arms, snapshot,
                target, context['grasps'], self.spatial_memory.scene, requested_arm=arm,
                placement_request=context['grasp_placement_request'],
                **({'destination_id': self.state['destination_id']} if self.state['destination_id'] else {}),
                planning_settings=context['planning_settings'], report_progress=True)
        return evaluations

    async def plan_grasp(self, snapshot, target, arm, context):
        binding = self.motion_plan_binding(context, observation=snapshot)
        attempt = {"arm": arm, "status": "running"}
        self.state["planning_attempts"].append(attempt)
        started = time.monotonic()
        try:
            result = await self.plan_grasp_candidates(snapshot, target, arm, context)
            binding.require_current(self.motion_plan_binding(context))
            result['input_binding'] = binding.model_dump()
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
            self.algorithms.plan_grasp, snapshot, target, arm, evaluations, self.spatial_memory.scene,
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
            if kind == 'observation_stage':
                step.update(stage=title, stage_started_at=time.time())
            for key in ("arm", "batch_index", "batch_count", "candidate_indices", "success_count",
                        "request_index", "candidate_group_index", "candidate_group_count",
                        "solver_success_count", "command_validated_candidate_count"):
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
        if self.visual_identity is not None and function in (
                self.algorithms.observe, getattr(self.algorithms, "verify_release", None),
                getattr(self, "verify_unpacking", None)):
            self.visual_identity.audit, visual_done = self.visual_audit_sink()

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
            self.update_objects_labels(observation)

    def update_objects_labels(self, observation):
        from .observation_labels import make_object_labels

        labels = make_object_labels(observation, self.task_progress.records, self.task_progress.membership,
                                    holding=self.holding_state, destination_id=self.state.get('destination_id'),
                                    placement_regions=self.task_progress.placement_requests.values())
        if labels != self.state.get('object_labels'):
            self.state['object_labels'] = labels
            self.record_annotation(observation, labels)

    def record_annotation(self, capture, labels):
        self.state['object_labels'] = copy.deepcopy(labels)
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
                 if progress is not None else 1)
        transfers = max(1, count)
        final_visual_capture = int(bool(context.get('identity_reference')))
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
            initialized = self.spatial_memory.scene or (self.task_progress and self.object_memory.observation_id)
            camera_ids = ['1', '2', '3'] if initialized else ['3']
        elif purpose == 'geometry' and self.spatial_memory.scene:
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
        previous = context.get('objects_observation', context.get('observation'))
        observation_options = {"checkpoint": checkpoint, "instruction": self.state['instruction']}
        from .runtime.memory import perception_motion_context

        hint = {"kind": "any", "color": "any",
                'cad_motion_context': perception_motion_context(self.object_memory, self.spatial_memory)}
        from .observation_motion import collision_mapping_scope

        history = self.spatial_memory.geometry_history(capture['generation'])
        if history:
            hint['published_geometry'] = history
            strategy, _ = collision_mapping_scope(capture, capture, context['planning_settings'],
                                                  initialized=self.spatial_memory.initialized)
            hint['local_geometry_only'] = strategy['requested_scope'] == 'local'
        hint['grounded_observation'] = context.get('grounded_observation')
        has_instance_memory = bool(self.object_memory.observation_id)
        if has_instance_memory:
            hint['scene_queries'] = self.object_memory.scene_queries()
            hint['scene_entities'] = copy.deepcopy(self.object_memory.entities())
            hint['visual_reference'] = context.get('identity_reference')
            hint['association_scope'] = copy.deepcopy(context.get('association_scope', {}))
            # The compilation boxes belong to this immutable calibrated frame.
            # A fresh first observation must not relocate those boxes onto new pixels.
            hint['reference_capture'] = context.get('initial_capture')
        if previous and previous.get('generation') == capture['generation']:
            hint["table"] = context.get('frozen_table', previous["table"])
        if camera_ids is not None:
            hint["camera_ids"] = camera_ids
        observation_options["source_hint"] = hint
        # Containers can be the source support even when the destination is a marker.
        hint['include_containers'] = True
        # The execution adapter owns reuse of the full capture + identity-history inputs.
        # A second, appearance-only cache would hide changed history from perception.
        observation = copy.deepcopy(await self.call_algorithm(self.algorithms.observe, capture,
                                                              report_progress=True, **observation_options))
        context['analysis_count'] = context.get('analysis_count', 0) + 1
        if not isinstance(observation.get('visual_identity'), dict):
            raise PerceptionError('visual_identity_required')
        if observation.get("source") != "rgbd" or observation["generation"] != self.state["generation"]:
            raise ValueError("invalid_rgbd_observation_generation")
        if observation.get("observation_id") != capture["observation_id"]:
            raise ValueError("invalid_rgbd_observation_identity")
        observation["checkpoint"] = checkpoint
        observation['full_collision_survey'] = bool(capture.get('full_collision_survey'))
        observation['mapping_world'] = {key: capture.get(key) for key in
                                        ('session_id', 'bridge_instance_id', 'geometry_revision')}
        observation['mapping_world']['run_id'] = self.run_id
        observation['camera_ids'] = list(camera_ids or ('1', '2', '3'))
        if has_instance_memory:
            identifiers = [obj['id'] for obj in observation['objects']]
            declared = set(self.object_memory.entities()) | set(observation.get('scene_entities', {}))
            if len(set(identifiers)) != len(identifiers) or not set(identifiers) <= declared:
                raise PerceptionError('visual_identity_invalid_ids')
            observation['mapping_unresolved'] = [
                {'previous_id': identifier, 'status': 'unseen', 'candidates': [],
                 'reason': 'known_collision_entity_not_observed'}
                for identifier, entity in self.object_memory.entities().items()
                if 'collision' in entity.get('roles', ()) and entity.get('kind') != 'table'
                and identifier not in identifiers
                and identifier not in context.get('association_scope', {}).get('deferred_entity_ids', [])]
            # Missing measurements do not erase a known obstacle. The map
            # publisher rejects this evidence; a final visual check can still
            # finish without requiring an otherwise unnecessary motion map.
            context['association_evidence'] = {'source': 'openai_decisions',
                'mapping': {identifier: identifier for identifier in identifiers},
                'unresolved': copy.deepcopy(observation['mapping_unresolved'])}
            observation['association_evidence'] = copy.deepcopy(context['association_evidence'])
        from .placement_layout import identify_placement_regions

        identify_placement_regions(observation)
        if (camera_ids is None or purpose in {'scene', 'geometry'}) and self.task_progress is not None:
            self.record_object_observation(context, observation)
        context["observation"] = observation
        self.state['facts']['source_support_current'] = False
        self.state['facts']['scene_ready'] = False
        if self.spatial_memory.scene.get('observation_id') != observation['observation_id']:
            self.state['facts']['observation_map_ready'] = False
        context['observation_purpose'] = purpose
        if camera_ids is None or purpose in {'scene', 'geometry'}:
            if self.task_progress is not None:
                self.task_progress.register_entities(self.object_memory.entities())
                self.task_progress.observe(observation['objects'], observation['observation_id'])
                visual_identity = observation['visual_identity']
                context.setdefault('identity_reference',
                    copy.deepcopy(observation.get('visual_reference', observation)))
                membership = {key: value for key, value in visual_identity.get('membership', {}).items()
                              if key in self.task_progress.records}
                self.task_progress.assess_observed_membership(membership)
                for identifier, evidence in visual_identity.get('progress', {}).items():
                    if identifier in self.task_progress.records:
                        self.task_progress.record_visual_progress(identifier, evidence)
                self.emit('visual_identity', 'Decisions 视觉物体记录与身份关联', visual_identity)
                self.update_objects_labels(observation)
            context['objects_observation'] = copy.deepcopy(observation)
        context["observation_checkpoint"] = checkpoint
        if checkpoint == "post_grasp":
            context["post_grasp_processed"] = capture["observation_id"]
            context["observation_checkpoint"] = "post_grasp_semantic"
        if checkpoint == "post_place":
            context["post_place_processed"] = capture["observation_id"]
        if checkpoint in {"initial", "standalone_place"}:
            if purpose == 'scene':
                evidence = {'views_ready': False, 'purpose': 'objects_and_changes'}
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

    def check_holding(self, observation, holding):
        holding['observation'] = copy.deepcopy(observation)
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
        from .runtime.memory import planning_observation

        # The map owns its capture. Verification frames do not silently replace it.
        observation = planning_observation(
            context.get('scene_observation') or context['observation'],
            self.object_memory, self.planning_regions)
        feedback = self.simulation.robot_snapshot(expected_generation=self.state['generation'])
        self.check_generation()
        observation["arms"] = feedback["arms"]
        observation["feedback_time"] = feedback["time"]
        return observation

    async def verify_scene(self, context):
        # Observations occur at named checkpoints. Planning consumes the frozen snapshot.
        self.check_generation()
        if self.spatial_memory.refresh_required:
            raise ValueError('recovery_requires_fresh_collision_scene')
        scene = self.spatial_memory.scene
        observation = (context.get('scene_observation') or context['observation'])
        if scene and scene["observation_id"] != observation["observation_id"]:
            raise ValueError("scene_observation_stale")
        if (scene and self.spatial_memory.validated_capture_id != context['observation']['observation_id']
                and context.get("observation_checkpoint") not in {"post_grasp", "post_grasp_semantic", "post_place", "standalone_place"}
                and scene["observation_id"] != context["observation"]["observation_id"]):
            raise ValueError("scene_observation_stale")

    async def build_scene(self, context, exclusion_state):
        if context.get('holding') or self.holding_state is not None:
            raise ValueError('grasp_scene_requires_empty_hands')
        if context.get('requires_observation_pose') and exclusion_state != 'observation':
            raise ValueError('changed_environment_requires_observation_pose')
        observation = context["observation"]
        from .memory_contracts import CollisionUpdate
        from .observation_motion import collision_mapping_scope

        self.check_generation()
        if observation.get('generation') != self.state['generation']:
            raise ValueError('collision_mapping_capture_generation_changed')
        scene = self.spatial_memory.scene
        lifecycle = self.spatial_memory.lifecycle()
        required = set(lifecycle.get('required_entity_ids', ()))
        covered = set((scene or {}).get('object_layers', {}).get('entities', {}))
        if (scene and not self.spatial_memory.refresh_required and not observation.get('full_collision_survey')
                and scene.get('generation') == observation['generation']
                and scene.get('observation_id') == observation['observation_id']
                and scene.get('database_revision') == self.spatial_memory.database_revision
                and required <= covered and scene.get('exclusion_state') == exclusion_state
                and (exclusion_state == 'observation' or scene.get('target_id') == self.state['target_id'])):
            self.state['facts'].update(scene_available=True, collision_map_initialized=self.spatial_memory.initialized,
                                       scene_ready=exclusion_state == 'selected', observation_map_ready=exclusion_state == 'observation')
            return scene
        strategy, pose = collision_mapping_scope(observation, observation, context['planning_settings'],
                                                 initialized=self.spatial_memory.initialized)
        if strategy['requested_scope'] == 'global':
            feedback = await asyncio.to_thread(self.simulation.joint_feedback, self.state['generation'])
            strategy, pose = collision_mapping_scope(observation, feedback, context['planning_settings'])
        self.spatial_memory.request_update(CollisionUpdate.model_validate(strategy))
        self.state['mapping_strategy'] = {**strategy, 'pose_evidence': pose}
        replacing_scene = bool(self.spatial_memory.scene)
        mapping_observation = {**observation, 'observation_pose_verified': strategy['requested_scope'] == 'global',
                               'mapping_lifecycle': self.spatial_memory.lifecycle()}
        if observation.get('mapping_world'):
            mapping_observation['mapping_lifecycle']['mapping_world'] = observation['mapping_world']
        if replacing_scene and self.spatial_memory.scene['generation'] != observation['generation']:
            parent = context.get('stopped_map_parent')
            if not parent or parent['world'] != observation.get('mapping_world'):
                raise ValueError('collision_memory_world_continuity_unconfirmed')
            mapping_observation['mapping_lifecycle']['stopped_parent'] = parent
        if context.get('rejected_global_token'):
            mapping_observation['mapping_lifecycle']['rejected_global_token'] = context['rejected_global_token']
        if replacing_scene:
            lifecycle = mapping_observation['mapping_lifecycle']
            lifecycle['previous_scene_token'] = self.spatial_memory.scene['scene_token']
            if strategy['requested_scope'] == 'local':
                from .spatial_memory import local_geometry_change

                history = self.spatial_memory.geometry_history(observation['generation'])
                retained = {}
                for obj in observation['objects']:
                    anchor = history.get(obj['id'])
                    if not anchor or obj.get('local_geometry_change'):
                        continue
                    if bool(obj.get('cad')) != bool(anchor.get('cad')):
                        continue
                    if obj.get('cad') and obj['cad']['mesh_sha256'] != anchor['cad']['mesh_sha256']:
                        continue
                    if local_geometry_change(anchor, obj.get('points_world_m'), observation['views']) is None:
                        retained[obj['id']] = anchor['observation_id']
                lifecycle['retained_visible_sources'] = retained
                mapping_observation['objects'] = [
                    {**obj, 'cad': history[obj['id']]['cad']} if obj['id'] in retained and obj.get('cad')
                    else obj for obj in observation['objects']]
        database_revision = self.spatial_memory.database_revision
        published_scene = await self.call_algorithm(
            self.algorithms.build_scene, mapping_observation, None if exclusion_state == 'observation' else self.state["target_id"],
            planning_settings=context["planning_settings"], exclusion_state=exclusion_state)
        if strategy['requested_scope'] == 'global':
            try:
                feedback = await asyncio.to_thread(self.simulation.joint_feedback, self.state['generation'])
                current, _ = collision_mapping_scope(observation, feedback, context['planning_settings'])
                if current['requested_scope'] != 'global':
                    raise ValueError('global_collision_map_observation_pose_changed_during_publication')
            except (ValueError, RuntimeError, OSError):
                context['rejected_global_token'] = published_scene['scene_token']
                self.spatial_memory.invalidate([])
                raise
        self.check_generation()
        context.pop('rejected_global_token', None)
        snapshot = published_scene.pop('collision_snapshot', None)
        surfaces = published_scene.pop('tsdf_surfaces', {})
        for measured in observation['objects']:
            if measured['id'] in surfaces:
                measured['tsdf_surface'] = surfaces[measured['id']]
        from .runtime.memory import freeze_scene_observation

        frozen_observation = freeze_scene_observation(observation)
        self.spatial_memory.accept_publication(published_scene, observation, exclusion_state=exclusion_state,
                                               database_revision=database_revision)
        context['scene_observation'] = frozen_observation
        context.pop('stopped_map_parent', None)
        if snapshot is not None and self.experiment is not None:
            if any(snapshot.get(key) != self.spatial_memory.scene.get(key)
                   for key in ('scene_token', 'observation_id', 'generation')):
                snapshot = {'status': 'unavailable', 'reason': 'collision_snapshot_binding_mismatch'}
            lifecycle = self.spatial_memory.lifecycle()
            stage_context = {
                'skill': 'build_scene', 'policy_node': copy.deepcopy(self.state.get('active_policy_node', {})),
                'scope': lifecycle['requested_scope'], 'reason': lifecycle['scope_reason'], 'pose_evidence': pose,
                'parent_scene_token': mapping_observation['mapping_lifecycle'].get('previous_scene_token'),
                'entity_ids': copy.deepcopy(lifecycle.get('required_entity_ids', [])),
                'retired_entity_ids': copy.deepcopy(lifecycle.get('retired_entity_ids', [])),
                'exclusion_state': exclusion_state,
                'excluded_instance_id': self.state.get('target_id') if exclusion_state == 'selected' else None,
                'world_semantics': 'measured_updates_with_spatial_memory' if replacing_scene else 'initial_workspace_measurement',
                'object_labels': {key: value.get('description') or value.get('kind') or key
                                  for key, value in self.object_memory.entities().items()}
                                  if self.task_progress else {}}
            snapshot['stage_context'] = stage_context
            artifact = self.experiment.collision_scene(f'collision/stage-{len(self.events) + 1:05d}', snapshot)
            self.emit('collision_scene_archived', '空间记忆快照', {
                'snapshot_artifact': artifact, 'status': snapshot.get('status'),
                'scene_token': self.spatial_memory.scene.get('scene_token'),
                'observation_id': self.spatial_memory.scene.get('observation_id'),
                'exclusion_state': exclusion_state, 'stage_context': stage_context,
                'target_id': self.state.get('target_id')})
        table_cache = self.spatial_memory.scene.get('mapping_diagnostics', {}).get('table_cache', {})
        self.state['scene_evidence'] = {
            'scene_token': self.spatial_memory.scene.get('scene_token'), 'exclusion_state': exclusion_state,
            'excluded_instance_id': self.state['target_id'] if exclusion_state == 'selected' else None,
            'camera_ids': sorted(observation['views']),
            'observed_instance_ids': [item['id'] for item in observation['objects']],
            'table_cache': copy.deepcopy(table_cache),
            'object_layers': copy.deepcopy(self.spatial_memory.scene.get('object_layers', {})),
            'coverage': copy.deepcopy(self.spatial_memory.scene.get('coverage', {})),
            'mapping_strategy': copy.deepcopy(self.spatial_memory.scene.get('mapping_strategy', {})),
            'association_unresolved': copy.deepcopy(observation.get('mapping_unresolved', []))}
        table_model = self.spatial_memory.scene.get('table')
        if observation.get("demo_tables") and table_cache.get("table_revision"):
            context["frozen_table"] = copy.deepcopy(observation["table"]) | {
                "table_revision": table_cache["table_revision"],
                "source_observation_id": table_cache["source_observation_id"],
                "source": "initial_objects_ransac_table"}
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
        if exclusion_state != 'observation':
            context.pop('pending_transfer', None)
            self.state['facts']['transfer_pending'] = False
        complete = not self.spatial_memory.refresh_required
        self.state['facts'].update(scene_available=complete,
                                  collision_map_initialized=self.spatial_memory.initialized,
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
            coverage = self.spatial_memory.scene.get('coverage', {})
            missing = coverage.get('missing_entity_ids', [])
            stale = coverage.get('stale_entity_ids', [])
            return {**self.spatial_memory.scene, 'success': False, 'status': 'unknown',
                    'reason': 'collision_objects_incomplete', 'details': {
                        'component': 'perception_mapping', 'observation_id': observation['observation_id'],
                        'missing_entity_ids': missing, 'stale_entity_ids': stale,
                        'diagnostic_checks': [{'name': 'current_objects_geometry',
                            'question': '所有未退役物体记录实体是否具有当前有效的实测碰撞层？',
                            'status': 'unknown', 'actual': {'missing': missing, 'stale': stale},
                            'expected': {'missing': [], 'stale': []}}],
                        'sam3_queries': [{key: query[key] for key in
                            ('camera_id', 'prompt', 'candidate_count', 'candidate_scores', 'status') if key in query}
                            for query in observation.get('sam3_queries', [])]}}
        return self.spatial_memory.scene

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
                      {**evidence, "error": str(error)[:1000], "details": getattr(error, "details", {}),
                         "duration_s": time.monotonic() - started})
            raise
        self.emit("execution_stage_complete", stage,
                  {**evidence, "result": result, "duration_s": time.monotonic() - started})
        return result

    async def motor(self, function, *args, **kwargs):
        if function in (self.simulation.execute_trajectory, self.simulation.set_gripper):
            kwargs['execution_settings'] = self.state.get('planning_settings', self.planning_settings.model_dump())
        return await asyncio.to_thread(self.simulation.run_motor, self.motion_epoch,
            function, *args, self.state['generation'], **kwargs)

    async def validate_motion(self, context, phase, command, binding=None, *, measured_observation=None, **kwargs):
        observation = self.measured_state(context) if measured_observation is None else measured_observation
        if binding is not None:
            holding = context['holding']
            expected = {'arm': self.state['arm'], 'instance_id': self.state['target_id'],
                        'generation': self.state['generation'], 'scene_token': self.spatial_memory.scene['scene_token'],
                        'owner_id': holding['owner_id']}
            if any(binding.get(key) != value for key, value in expected.items()):
                raise ValueError('holding_binding_mismatch')
            if not holding.get('verified'):
                raise ValueError('carrying_requires_visual_grasp_verification')
        return await self.call_algorithm(
            self.algorithms.validate_execution, observation, self.state["arm"],
            self.spatial_memory.scene, phase, command, binding,
            planning_settings=context["planning_settings"], **kwargs)

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

    async def open_gripper(self, context, **kwargs):
        return await self.motor(self.simulation.set_gripper, self.state['arm'], False, **kwargs)

    async def _dispatch_transfer(self, context, kind, motion_id, phases, binding, checks):
        from .motion_sequence import build_sequence
        from .robot_errors import CHECKED_START_TOLERANCE_DEG, CheckedStartChanged
        arm = self.state['arm']
        settings = context['planning_settings']
        order = ('approach', 'grasp', 'lift') if kind == 'grasp' else ('approach', 'place', 'retreat')
        planned = context['plan' if kind == 'grasp' else 'place_plan']
        for attempt in range(2):
            self.require_plan_binding(context, planned)
            await self.verify_scene(context)
            fresh = self.measured_state(context)
            checked_start = np.asarray(phases['approach']['positions_rad'][0])
            measured = np.asarray(fresh['arms'][arm]['joints_rad'])
            if np.max(abs(measured - checked_start)) > np.deg2rad(CHECKED_START_TOLERANCE_DEG):
                command = self.prepare_command(planned['phases']['approach'], feedback=fresh)
                options = {'contact_goal_pose_world': planned['T_world_tcp']} if kind == 'grasp' else {}
                checks['approach'] = await self.validate_motion(context,
                    'approach_grasp' if kind == 'grasp' else 'approach', command, binding,
                    measured_observation=fresh, **options)
                phases['approach'] = command
            prefix_only = kind == 'place'
            sequence = build_sequence(phases, order[:2] if prefix_only else order,
                gripper_events={} if prefix_only else {kind: {'closed': True}},
                gripper_transition_s=settings['post_gripper_hold_s'])
            try:
                executed = await self.execution_stage(context, kind + '_sequence', self.motor,
                    self.simulation.execute_trajectory, arm, sequence['positions_rad'], phases['approach']['dt'],
                    times_s=sequence['times_s'], gripper_events=sequence['events'], motion_id=motion_id)
            except CheckedStartChanged as error:
                self.emit('trajectory_start_changed', '派发前实测关节起点变化', {
                    **error.details, 'motion_id': motion_id, 'attempt': attempt + 1,
                    'checked_feedback_time': fresh['feedback_time'],
                    'planner_start_rad': planned['phases']['approach']['positions_rad'][0]})
                if attempt or error.details.get('dispatched') is not False:
                    raise
                continue
            except Exception as error:
                if kind == 'grasp' and (getattr(error, 'details', {}) or {}).get('dispatched') is not False:
                    if self.simulation.robot_snapshot()['generation'] == self.state['generation']:
                        self.record_motion_receipt(context, 'lift_started', {
                            'source': 'grasp_dispatch_unconfirmed', 'reason': str(error)[:500],
                            'meaning': 'The previous support must be observed again.'})
                raise

            if prefix_only:
                return await self._release_and_retreat(context, motion_id, phases['retreat'], binding, checks, executed)
            return executed

    async def _release_and_retreat(self, context, motion_id, planned_retreat, binding, checks, transport):
        from .demo_placement import release_endpoint_evidence
        from .robot_errors import CheckedStartChanged

        arm, settings = self.state['arm'], context['planning_settings']
        for attempt in range(2):
            self.require_plan_binding(context, context['place_plan'])
            await self.verify_scene(context)
            measured = self.measured_state(context)
            current = measured['arms'][arm]
            endpoint = release_endpoint_evidence(context['place_plan']['goal'], context['holding'], current['tcp'])
            if not endpoint['at_release_goal']:
                raise ValueError('release_endpoint_not_reached')
            opened = current['gripper_limits_rad'][0]
            checks['release_collision_check'] = await self.validate_motion(context, 'open',
                {'positions_rad': [current['joints_rad']], 'dt': planned_retreat['dt']},
                binding, measured_observation=measured, gripper_end_position_rad=opened)
            retreat = self.prepare_command(planned_retreat, feedback=measured)
            current.update(gripper_position_rad=opened, gripper_closed_fraction=0.)
            checks['retreat'] = await self.validate_motion(context, 'retreat', retreat, measured_observation=measured)
            positions = list(retreat['positions_rad'])
            times = (np.arange(len(positions)) * retreat['dt']).tolist()
            hold = settings['post_gripper_hold_s']
            if hold > 0.:
                positions.insert(0, positions[0])
                times = [0., *(t+hold for t in times)]
            try:
                result = await self.execution_stage(context, 'place_open_and_retreat', self.motor,
                    self.simulation.execute_trajectory, arm, positions, retreat['dt'], times_s=times,
                    gripper_events=[{'phase': 'place', 'closed': False, 'time_s': 0.}], motion_id=motion_id)
            except CheckedStartChanged as error:
                if attempt or error.details.get('dispatched') is not False:
                    raise
                self.emit('trajectory_start_changed', '释放前重新核对实测姿态', error.details)
                continue
            result['transport_before_release'] = transport
            result['duration_s'] += transport['duration_s']
            return result

    async def capture_holding_reference(self, context, grasp_feedback, source_observation):
        arm, target = self.state["arm"], self.state["target_id"]
        # Lift invalidates the current support relation. Keep the dated source
        # measurement that authorized this sequence, then bind its actual grasp TCP.
        observation = copy.deepcopy(source_observation)
        observation['arms'] = copy.deepcopy(grasp_feedback['arms'])
        observation['feedback_time'] = grasp_feedback['time']
        feedback = observation["arms"][arm]
        binding = carrying_binding(observation, target, arm, self.spatial_memory.scene["scene_token"], owner_id=self.run_id)
        binding.update(PlanningSettings.model_validate(context["planning_settings"]).attachment.model_dump())
        source = observed_object(observation, target)
        context["points_before_lift"] = np.asarray(source["points_world_m"]).copy()
        context["tcp_before_lift"] = copy.deepcopy(feedback["tcp"])
        context["holding"] = {**binding, "state": "provisional", "verified": False, "attached": False,
                              "source_attributes": {"kind": source["kind"], "color": source["color"], "position": "any"},
                              "gripper_joint_positions_rad": {
                                  side: state["gripper_position_rad"] for side, state in observation["arms"].items()},
                              "observation": copy.deepcopy(source_observation),
                              "grasp_support_reference": grasp_support_reference(
                                  observation, self.state['target_id'], feedback['tcp'],
                                  source_reference=context.get('plan', {}).get('lift_clearance'),
                                  scene_token=self.spatial_memory.scene['scene_token'])}
        if context.get('identity_reference'):
            context['holding'].update(
                identity_reference=copy.deepcopy(context['identity_reference']),
                scene_entities=copy.deepcopy(self.object_memory.entities()),
                scene_objects_observation_id=self.object_memory.observation_id)
        from .geometry import TOOL_PROFILE

        context['holding']['grasp_support_reference'].update(arm=arm,
            tool_profile_revision=TOOL_PROFILE['revision'], reference_phase='scheduled_close_before_lift',
            feedback_id=f"feedback-{observation['generation']}-{observation['feedback_time']}",
            feedback_time=observation['feedback_time'])
        self.holding_state = copy.deepcopy(context["holding"])
        return {"source": "pregrasp_rgbd_and_close_waypoint_feedback", "observation_id": observation["observation_id"],
                "tcp_before_lift": context["tcp_before_lift"], "holding_verified": False}

    async def sample_gripper(self, context, arm, after_time, checkpoint):
        from .robot_skills.grasp_verification import gripper_feedback_state

        epoch = self.observation_epoch
        capture = await asyncio.to_thread(self.simulation.gripper_verification_sample,
                                          arm, self.state['generation'], after_time)
        self.check_generation()
        if (epoch != self.observation_epoch or capture['generation'] != self.state['generation']
                or capture.get('arm') != arm):
            raise ValueError('gripper_verification_binding_changed')
        self.emit('gripper_verification_feedback', checkpoint, {
            **gripper_feedback_state(capture, arm), 'feedback_id': capture['feedback_id'],
            'source': capture['source'], 'camera_ids': list(capture.get('views', {})),
            'wrist_error': capture.get('wrist_error')})
        if capture.get('views'):
            self.record_observation(capture, checkpoint)
        return capture

    async def verify_grasp(self, context):
        from .robot_skills.grasp_verification import (
            grasp_visual_context,
            holding_from_feedback,
            inherit_grasp_identity,
            object_held_question,
        )
        from .visual_api import VisualDecisionError

        if context.get('grasp_execution', {}).get('completed') is not True:
            raise ValueError('visual_grasp_requires_completed_sequence')
        target, arm = self.state['target_id'], self.state['arm']
        capture = await self.sample_gripper(context, arm, context['grasp_execution']['completed_feedback_time_s'],
                                            'verify_grasp')
        context['grasp_verification_sample'] = capture
        measured = holding_from_feedback(capture, arm)
        holding_judgment = {'judgments': {'object_held': {'status': measured['status']}},
            'evidence': {'source': measured['source'], 'observation_id': capture['observation_id'],
                         'generation': capture['generation'], 'gripper_feedback': [measured['feedback']],
                         'wrist_error': capture.get('wrist_error'), 'model_called': False}}
        if measured['requires_visual'] and capture.get('views'):
            presence_context = grasp_visual_context(capture, context['holding'])
            presence_context.pop('target_id')
            presence_context['visual_camera_ids'] = [presence_context['primary_wrist_camera_id']]
            try:
                holding_judgment = await self._judge(context, [object_held_question(arm)],
                    include_task=False, extra=presence_context, reference={}, verification_capture=capture)
            except VisualDecisionError as error:
                holding_judgment = {
                    'judgments': {'object_held': {'status': 'unknown', 'reason': 'visual_request_failed'}},
                    'evidence': {**copy.deepcopy(error.evidence), 'error': str(error)}}
        presence = holding_judgment['judgments']['object_held']['status']
        presence_evidence = {'status': presence, 'rule': 'measured_position_then_closed_gripper_vision',
            'source': holding_judgment.get('evidence', {}).get('source', 'openai_decisions'),
            'observation_id': capture['observation_id']}
        identity_binding = None
        state = 'empty' if presence == 'no' else 'unknown'
        if presence == 'yes':
            try:
                identity_binding = inherit_grasp_identity(context['holding'], context['grasp_execution'], capture,
                    target_id=target, arm=arm, owner_id=self.run_id, motion_epoch=self.motion_epoch,
                    action_epoch=self.observation_epoch)
            except ValueError as error:
                identity_binding = {'source': 'continuous_grasp_execution', 'status': 'unknown',
                                    'reason': str(error), 'outcome': 'unknown', 'visual_identity_verified': False}
            else:
                state = 'held'
        answer = {'held': 'yes', 'empty': 'no'}.get(state, 'unknown')
        result = {'success': answer == 'yes', 'status': 'held' if answer == 'yes' else answer,
            'source': 'gripper_holding_and_execution_identity',
            'visual_evidence': {'source': presence_evidence['source'], 'observation_id': capture['observation_id'],
                'generation': capture['generation'], 'image_input': 'executing_wrist_rgb' if capture.get('views') else None,
                'grasp_execution': copy.deepcopy(context['grasp_execution'])},
            'reason': None if answer == 'yes' else (identity_binding or {}).get('reason', 'visual_grasp_' + state),
            'observation_id': capture['observation_id'],
            'feedback_id': f"feedback-{capture['generation']}-{capture['time']}",
            'feedback_time': capture['time'], 'target_id': target, 'arm': arm,
            'generation': capture['generation'], 'action_epoch': self.observation_epoch,
            'invocation_id': self.state['current_step']['invocation_id']}
        result['holding_judgment'] = holding_judgment
        result['identity_binding'] = identity_binding
        result['target_identity_outcome'] = identity_binding['outcome'] if identity_binding else 'not_checked'
        result['verification_steps'] = {'holding_presence': presence, 'target_identity': result['target_identity_outcome']}
        result['grasp_state'] = state
        result['holding_presence'] = presence_evidence
        context['grasp_verification'] = result
        self.state['facts']['grasp_evidence_pending'] = False
        holding = context['holding']
        holding.update(state=state, verified=result['success'], verification=copy.deepcopy(result))
        if result['success']:
            holding.update(state='held', observation=copy.deepcopy(context['observation']),
                scene=copy.deepcopy(self.spatial_memory.scene),
                scene_observation=copy.deepcopy(context.get('scene_observation', {})))
            self.state['facts'].update(grasp_held=True, recovery_needed=False)
            self.holding_state = copy.deepcopy(holding)
            result['attachment'] = await self.prepare_attachment(context)
        else:
            holding['reason'] = 'visual_grasp_' + answer
            self.state['facts'].update(grasp_held=False, verified=False)
        self.holding_state = copy.deepcopy(holding)
        return result

    async def prepare_attachment(self, context, observation=None):
        """Prepare the carrying collision model after an independent grasp verdict."""
        holding = context['holding']
        if any(holding.get(key) != value for key, value in {
                'arm': self.state['arm'], 'instance_id': self.state['target_id'],
                'owner_id': self.run_id, 'generation': self.state['generation']}.items()):
            raise ValueError('attachment_holding_binding_mismatch')
        if not holding.get('verified'):
            raise ValueError('attachment_requires_verified_holding')
        observation = self.measured_state(context) if observation is None else observation
        self.check_generation()
        scene = self.spatial_memory.scene
        prepared = holding.get('attachment_evidence', {})
        expected = {'arm': self.state['arm'], 'instance_id': self.state['target_id'],
                    'carrying_owner_id': self.run_id, 'generation': scene['generation'],
                    'observation_id': scene['observation_id'], 'scene_token': scene['scene_token']}
        if holding.get('attached') and all(prepared.get(key) == value for key, value in expected.items()):
            return prepared
        holding['attached'] = False
        self.holding_state = copy.deepcopy(holding)
        self.state['facts']['attached'] = False
        prepared = await self.call_algorithm(self.algorithms.prepare_carrying, observation,
                                             self.state['arm'], scene, holding,
                                             planning_settings=context["planning_settings"],
                                             placement_request=context.get('grasp_placement_request'))
        if prepared.get("attached") is not True:
            raise ValueError("attachment_preparation_failed")
        holding.update(attached=True, attachment_evidence=prepared, scene_token=self.spatial_memory.scene["scene_token"])
        from .memory_contracts import ObjectOperation

        self.record_object_operation(context, ObjectOperation(entity_id=self.state['target_id'],
            event_key=self.run_id + ':' + holding['verification']['feedback_id'] + ':grasp_verified',
            kind='grasp_verified', current_region=None, observed_support=None,
            transport_status='held', evidence={'source': 'visual_holding_execution_identity_and_attachment',
                'identity_binding': copy.deepcopy(holding['verification']['identity_binding']),
                'holding_presence': copy.deepcopy(holding['verification']['holding_presence']),
                'arm': self.state['arm'], 'feedback_id': holding['verification']['feedback_id'],
                'observation_id': holding['verification']['observation_id'], 'scene_token': prepared['scene_token']}))
        self.spatial_memory.accept_attachment(self.state['target_id'], prepared)
        self.holding_state = copy.deepcopy(holding)
        self.state["facts"].update(attached=True, scene_ready=not self.spatial_memory.refresh_required)
        self.state["collision_model"] = {"gripper": "actual_feedback", "attached": True,
                                          "prepare_wall_ms": prepared.get("prepare_wall_ms")}
        return prepared

    def motion_plan_binding(self, context, *, observation=None):
        from .core.manipulation import MotionPlanBinding

        return MotionPlanBinding.capture(target_id=self.state['target_id'],
            destination_id=self.state['destination_id'], arm=self.state['arm'],
            observation=context['observation'] if observation is None else observation,
            scene=self.spatial_memory.scene, planning_settings=context['planning_settings'])

    def require_plan_binding(self, context, plan):
        from .core.manipulation import MotionPlanBinding

        MotionPlanBinding.model_validate(plan['input_binding']).require_current(self.motion_plan_binding(context))

    def record_object_operation(self, context, operation):
        """Accept a typed physical receipt; the controller adds durable storage."""
        self.object_memory.record_operation(operation)

    def record_motion_receipt(self, context, kind, evidence):
        from .memory_contracts import ObjectMotionReceipt, ObjectOperation

        feedback = self.simulation.robot_snapshot()
        self.check_generation()
        receipt = ObjectMotionReceipt(entity_id=self.state['target_id'], kind=kind,
            arm=self.state['arm'], generation=feedback['generation'],
            feedback_time_s=feedback['time'], evidence=evidence)
        # A planned lift makes the old position unconfirmed. Gripper feedback
        # alone says nothing about whether this object moved, is held or was released.
        changes = ({'current_region': None, 'confirmed_observation_id': None,
                    'observed_support': None, 'transport_status': 'position_unconfirmed'}
                   if kind == 'lift_started' else {})
        operation = ObjectOperation(entity_id=receipt.entity_id,
            event_key=self.run_id + ':' + self.state['current_step']['invocation_id'] + ':' + kind,
            kind=kind, **changes, last_motion_receipt=receipt,
            evidence=receipt.model_dump(mode='json'))
        self.record_object_operation(context, operation)

    async def execute_grasp(self, context):
        arm, plan = self.state['arm'], context['plan']
        self.require_plan_binding(context, plan)
        await self.verify_scene(context)
        grasp_support_reference(self.measured_state(context), self.state['target_id'], plan['T_world_tcp'],
            source_reference=plan.get('lift_clearance'), scene_token=self.spatial_memory.scene['scene_token'])
        context['execution_started'] = True
        async with self.simulation.motion_scope([arm], 'grasp', self.state['generation'], self.motion_epoch) as motion_id:
            await self.execution_stage(context, 'open_gripper', self.open_gripper, context, motion_id=motion_id)
            feedback = self.measured_state(context)
            phases = copy.deepcopy(plan['phases'])
            phases['approach'] = self.prepare_command(phases['approach'], feedback=feedback)
            checks = {}
            checking = copy.deepcopy(feedback)
            for phase, check_phase in (('approach', 'approach_grasp'), ('grasp', 'grasp')):
                checks[phase] = await self.validate_motion(context, check_phase, phases[phase],
                    measured_observation=checking, contact_goal_pose_world=plan['T_world_tcp'],
                    previous_check=plan.get('command_checks', {}).get(phase))
                checking['arms'][arm]['joints_rad'] = phases[phase]['positions_rad'][-1]
            checks['closure'] = await self.validate_motion(context, 'grasp_closure',
                {'positions_rad': [checking['arms'][arm]['joints_rad']], 'dt': .02},
                measured_observation=checking)
            checks['lift'] = await self.validate_motion(context, 'grasp_lift', phases['lift'],
                measured_observation=checking,
                previous_check=plan.get('command_checks', {}).get('lift'))
            self.holding_state = {'owner_id': self.run_id, 'generation': self.state['generation'],
                'arm': arm, 'instance_id': self.state['target_id'], 'state': 'provisional',
                'verified': False, 'attached': False, 'reason': 'continuous_grasp_in_progress'}
            executed = await self._dispatch_transfer(context, 'grasp', motion_id, phases, None, checks)
            self.record_motion_receipt(context, 'lift_started', {
                'source': 'completed_grasp_sequence', 'meaning': 'Lift executed. Confirm holding before attaching.'})
        closed_at = next(event['feedback'] for event in executed['gripper_events'] if event['closed'])
        await self.capture_holding_reference(context, closed_at, feedback)
        result = {'arm': arm, 'candidate_index': plan.get('candidate_index'),
            'stages': {'continuous_sequence': executed}, 'command_checks': checks,
            'lift_carried_collision_checked': checks['lift'].get('lift_carried_collision_checked', False),
            'lift_gripper_geometry': checks['lift']['gripper_configuration_source'],
            'attachment_stage': 'after_lift_and_holding_verification',
            'holding_verification_required': True}
        completed = self.simulation.robot_snapshot()
        context['grasp_execution'] = {'arm': arm, 'completed': True, 'target_id': self.state['target_id'],
            'owner_id': self.run_id, 'motion_epoch': self.motion_epoch, 'action_epoch': self.observation_epoch,
            'observation_id': context['holding']['observation_id'],
            'scene_token': context['holding']['scene_token'],
            'completed_feedback_time_s': completed['time'],
            'generation': executed['generation'], 'duration_s': executed['duration_s']}
        return result

    async def execute_place(self, context):
        from .container_release import release_mode
        from .demo_placement import release_endpoint_evidence

        plan = context['place_plan']
        arm = self.state['arm']
        self.require_plan_binding(context, plan)
        mode = release_mode(plan['placement_request'])
        if mode != plan['goal']['release_reference']['release_mode']:
            raise ValueError('placement_release_mode_changed')
        if set(plan['phases']) != {'approach', 'place', 'retreat'}:
            raise ValueError('place_requires_three_phases')
        await self.verify_scene(context)
        holding = context['holding']
        if not holding.get('attached') or not holding.get('verified'):
            raise ValueError('place_requires_visual_grasp_and_attachment')
        if not np.array_equal(holding['points_tool_m'], plan['holding_reference']):
            raise ValueError('holding_geometry_changed_since_place_plan')
        binding = copy.deepcopy(plan['carrying_binding'])
        checking = self.measured_state(context)
        phases = copy.deepcopy(plan['phases'])
        phases['approach'] = self.prepare_command(phases['approach'], feedback=checking)
        checks = {}
        for phase in ('approach', 'place'):
            checks[phase] = await self.validate_motion(context, phase, phases[phase], binding,
                measured_observation=checking, previous_check=plan.get('command_checks', {}).get(phase))
            checking['arms'][arm]['joints_rad'] = phases[phase]['positions_rad'][-1]
        open_position = checking['arms'][arm]['gripper_limits_rad'][0]
        checks['release_collision_check'] = await self.validate_motion(context, 'open',
            {'positions_rad': [checking['arms'][arm]['joints_rad']], 'dt': phases['place']['dt']},
            binding, measured_observation=checking, gripper_end_position_rad=open_position)
        checking['arms'][arm].update(gripper_position_rad=open_position, gripper_closed_fraction=0.)
        checks['retreat'] = await self.validate_motion(context, 'retreat', phases['retreat'],
            measured_observation=checking, previous_check=plan.get('command_checks', {}).get('retreat'))
        async with self.simulation.motion_scope([arm], 'place', self.state['generation'], self.motion_epoch) as motion_id:
            executed = await self._dispatch_transfer(context, 'place', motion_id, phases, binding, checks)
        release_feedback = next(event['feedback'] for event in executed['gripper_events'] if not event['closed'])
        context['release_tcp'] = release_feedback['arms'][arm]['tcp']
        endpoint = release_endpoint_evidence(plan['goal'], holding, context['release_tcp'])
        context['release_endpoint_evidence'] = endpoint
        context['pending_transfer'] = {'release_mode': mode, 'target_id': self.state['target_id'],
            'destination_id': self.state['destination_id'],
            'destination_kind': observed_object(context['observation'], self.state['destination_id'])['kind'],
            'scene_token': self.spatial_memory.scene['scene_token'],
            'tcp_before_lift': copy.deepcopy(holding['grasp_support_reference']['T_world_tcp_at_grasp']),
            'release_tcp': copy.deepcopy(context['release_tcp']),
            'release_height_m': plan['goal']['release_reference']['configured_bottom_clearance_m']}
        self.record_motion_receipt(context, 'release_commanded', {
            'source': 'continuous_place_command', 'release_feedback_time': release_feedback['time'],
            'carried_geometry_at_open': {
                'source': 'visually_verified_holding_and_actual_open_waypoint',
                'destination_id': self.state['destination_id'],
                'source_observation_id': holding['observation_id'],
                'grasp_verification_observation_id': holding['verification']['observation_id'],
                'grasp_verification_source': holding['verification']['source'],
                'object_reference_world_m': copy.deepcopy(endpoint['actual_reference_world_m']),
                'T_world_tcp': copy.deepcopy(context['release_tcp']),
                'scope': 'Rigid carrying prediction at the recorded opening command. '
                    'Current RGBD supplies the subsequent landing and identity evidence.'},
            'meaning': 'Opening command dispatched at release waypoint. Visual placement confirmation is pending.'})
        await self.call_algorithm(self.algorithms.detach_carrying)
        holding.update(state='release_pending', attached=False)
        self.state['facts'].update(released=True, grasp_held=False, attached=False)
        self.spatial_memory.invalidate([self.state['target_id']])
        self.state['facts'].update(scene_available=False, scene_ready=False, observation_map_ready=False)
        if not endpoint['at_release_goal']:
            raise ValueError('release_endpoint_not_reached')
        completed = self.simulation.robot_snapshot()
        self.check_generation()
        return {'arm': arm, 'released': True, 'release_commanded': True,
            'stages': {**checks, 'continuous_sequence': executed},
            'completed_feedback': {key: completed[key] for key in ('generation', 'time')}}


    async def verify_place(self, context):
        """Validate the execution receipt before measuring the released gripper."""
        snapshot = self.simulation.robot_snapshot(include_tcp=False)
        self.check_generation()
        completed = bool(self.state['facts'].get('place_execution_completed'))
        commanded = context.get('demo_place_execution', {}).get('release_commanded') is True
        result = {'success': completed and commanded, 'status': 'confirmed' if completed and commanded else 'unknown',
            'source': 'continuous_place_execution', 'verification_scope': 'execution_only',
            'destination_pose_verified': False, 'release_commanded': commanded,
            'released': commanded, 'execution_completed': completed,
            'feedback_id': f"feedback-{snapshot['generation']}-{snapshot['time']}",
            'feedback_time': snapshot['time'],
            'reason': None if completed and commanded else 'place_execution_incomplete'}
        context['place_verification'] = result
        self.state['facts']['place_evidence_pending'] = False
        return result

    async def dispatch_observation_path(self, context, scene, plan, stage, **validation):
        from .robot_errors import CheckedStartChanged

        paths = {arm: np.asarray(plan['paths'][arm], dtype=float) for arm in ('A', 'B')}
        if len(paths['A']) != len(paths['B']):
            raise ValueError('dual_trajectory_sample_counts_differ')
        async with self.simulation.motion_scope(['A', 'B'], 'observe', self.state['generation'],
                                                self.motion_epoch) as motion_id:
            for attempt in range(2):
                measured = self.measured_state(context)
                checked = {}
                for arm, path in paths.items():
                    start = measured['arms'][arm]['joints_rad']
                    if np.max(np.abs(path[0] - start)) > np.deg2rad(1.):
                        raise TrajectoryStartError(arm, start, path[0], measured)
                    checked[arm] = np.vstack((start, path)).tolist()
                await self.call_algorithm(self.algorithms.validate_dual_observation, measured, scene, checked,
                                          planning_settings=context['planning_settings'], **validation)
                await self.verify_scene(context)
                try:
                    return await self.execution_stage(context, stage, self.motor,
                        self.simulation.execute_dual_trajectory, checked, plan['dt'],
                        execution_settings=context['planning_settings'], motion_id=motion_id)
                except CheckedStartChanged as error:
                    self.emit('trajectory_start_changed', '双臂派发前实测起点变化', {
                        **error.details, 'motion_id': motion_id, 'attempt': attempt + 1})
                    if attempt or error.details.get('dispatched') is not False:
                        raise

    async def move_to_observe(self, context, *, joint_goals=None):
        if context.get("holding") or self.holding_state is not None:
            raise ValueError("observation_motion_requires_empty_hands")
        poses = observation_poses(context["planning_settings"]) if joint_goals is None else joint_goals
        original_arm = self.state["arm"]
        results = {}
        try:
            retained_return = context.pop("use_retained_scene_for_return", False)
            observation = context['scene_observation'] if retained_return else context["observation"]
            scene = self.spatial_memory.scene
            if (self.spatial_memory.refresh_required or not retained_return and (
                    scene.get('exclusion_state') != 'observation'
                    or scene.get('observation_id') != observation.get('observation_id'))):
                raise ValueError('observation_motion_requires_current_map')
            if not scene:
                raise ValueError('observation_motion_requires_current_map')
            motion_scene = scene | {'return_after_place': retained_return}
            plan = await self.call_algorithm(self.algorithms.plan_dual_observation_motion,
                self.measured_state(context), motion_scene, poses,
                planning_settings=context["planning_settings"])
            results = await self.dispatch_observation_path(context, motion_scene, plan, 'observe_pose_AB')
            # Keep the prior world solely for recovery motion until a complete
            # fresh scene replaces it. It must not authorize manipulation.
            self.spatial_memory.invalidate([])
            context.pop('requires_observation_pose', None)
        finally:
            self.state["arm"] = original_arm
        for key in ("evaluations", "grasps", "plan"):
            context.pop(key, None)
        context['view_evidence'] = {'views_ready': False, 'geometry_analysis_pending': True,
            'reason': 'graph_observe_scene_required_after_movement'}
        context['observation_checkpoint'] = 'geometry_capture_pending'
        self.state["facts"].update(views_ready=False, scene_ready=False, grasps_ready=False, arms_evaluated=False,
                                  observation_map_ready=False, scene_available=True, geometry_capture_ready=False)
        return {'movements': results, 'capture_required': True, 'semantic_processing': False}

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
                self.update_objects_labels(observation)
                if self.state.get('destination_id'):
                    destination = observed_object(observation, self.state['destination_id'])
                    self.task_progress.records[target]['destination_kind'] = destination['kind']
            self.state["target_id"] = target
            context['view_evidence'] = source_view_evidence(observation, source)
            if context.get('observation_purpose') == 'scene':
                context['view_evidence'].update(views_ready=False, geometry_analysis_pending=True)
            self.state['facts']['views_ready'] = context['view_evidence']['views_ready']
            if self.spatial_memory.scene.get('target_id') != target:
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

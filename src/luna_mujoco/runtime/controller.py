"""Compose model planning, evidence judgments and registered robot operations."""
import asyncio
import copy
import json
import os
import time
from functools import partial
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agent.decisions import DecisionsJudgments
from ..agent.registration import ObjectRegistrar
from ..agent.responses import ResponsesBudget, ResponsesCompiler
from ..agent.schema import (
    Graph,
    TaskSpec,
    skill_catalog,
)
from ..algorithms import AlgorithmError
from ..compilation_memory import CompilationMemory
from ..core.decisions import ChoiceRequest, GraphDecisionRequest, GraphDecisionResult, JudgmentRequest
from ..memory_reference import bound_identity_coverage_source, current_memory_entities
from ..object_database import ObjectDatabase
from ..perception import PerceptionError
from ..placement import observed_object
from ..robot_skills.container_geometry import container_transport_region
from ..robot_skills.operations import DESTINATION_ID, RobotOperations, configured_destination
from ..robot_skills.placement_region import measured_table_region
from ..robot_skills.placement_zone import measured_object_region
from ..robot_skills.unpack import VLAUnpack
from ..runner import Runner, public_summary
from ..skills import NoArguments
from ..visual_api import VisualAPI
from ..vla_configuration import vla_unpack_configuration
from .diagnostics import DiagnosticError, failure_record, judgment_details, skill_error_result
from .execution_state import bind_failure_context, result_status
from .memory import execution_snapshot
from .native_graph import NativeGraphExecutor
from .placement import (
    invalidate_placement,
    perceive_for_controller,
    prepare_fixed_placement,
)
from .support import support_context, support_memory
from .task_ledger import PolicyTaskProgress


class TargetArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    target_id: str


class DestinationArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    destination_id: str
    destination_kind: Literal['support_region', 'container']
    release_mode: Literal['surface_setdown', 'container_release'] = Field(description=(
        'Use surface_setdown for a configured support_region. Use container_release for a container, '
        'with release TCP above the measured rim by the operator-configured offset. '
        'Gentle placement on a container floor is unsupported.'))
    placement_mode: Literal['configured_xy'] = Field(default='configured_xy', description=(
        'Use the operator-configured square center XY. TCP Z is measured support height plus '
        'the configured table offset, or measured container top plus the configured container offset.'))
    placement_preference: Literal['clearance', 'reach'] = Field(default='clearance', description=(
        'Support-region candidate ranking: clearance first seeks separation from measured occupied '
        'areas up to the configured soft-clearance distance, then favors arm reach; reach favors '
        'shorter shoulder-to-wrist distance. Both retain workspace, IK and carried collision checks. '
        'Exact configured XY is immutable. Decisions approves the entire region or numbered candidates; '
        'this numerical preference ranks only candidates that satisfy the original requirement.'))
    orientation_mode: Literal['preserve'] = Field(default='preserve', description=(
        'All placements preserve the full measured grasp rotation. No automatic tilt or yaw search.'))


    @model_validator(mode='after')
    def coherent_release_mode(self):
        if self.release_mode == 'container_release' and self.destination_kind != 'container':
            raise ValueError('container_release_requires_container')
        if self.destination_kind == 'container' and self.release_mode != 'container_release':
            raise ValueError('container_requires_container_release')
        return self


class ArmArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    arm: Literal['auto', 'A', 'B'] = 'auto'


class PrepareUnpackArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class VerifyPlaceArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    release_mode: Literal['bound', 'surface_setdown', 'container_release'] = Field(default='bound', description=(
        'Declare the same mode as identify_destination and the executed transfer. surface_setdown '
        'requires current mapping before later movement. Ordinary container_release can publish '
        'operator-authorized contents occupancy. bound resolves an existing direct caller binding.'))


class ObserveArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    capture_mode: Literal['reuse_current', 'fresh'] = Field(default='reuse_current', description=(
        'Use the latest three-camera capture when its action/scene/command versions and measured '
        'arm/gripper feedback still match. fresh always acquires a new capture. With reuse_current, age alone and a '
        'failed decision do not force another capture. A changed viewpoint or scene requires acquisition. '
        'An identical failed analysis is not repeated. Reuse preserves the original observation ID/time.'))


class SceneObserveArguments(ObserveArguments):
    full_collision_survey: bool = Field(default=False, description=(
        'Request new measurements for a full collision map at the configured dual-arm observation pose. '
        'Ordinary observations update the existing collision memory locally.'))


class ObservationMotionArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    viewpoint: Literal['configured', 'wrist_tilt_positive', 'wrist_tilt_negative'] = Field(
        default='configured', description=(
            'configured returns to the operator observation pose and can satisfy the required final '
            'return. The two wrist_tilt alternatives offset both sixth joints by +12 or -12 degrees from '
            'that pose to obtain different wrist-camera views. They are movement proposals, not certified '
            'visible viewpoints; current-map cuRoboV2-Fast planning and execution checks are mandatory. '
            'After the selected movement, observe_scene analyzes the new capture.'))


class BuildSceneArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    reason: str = ''
    exclusion_state: Literal['selected', 'observation'] = 'selected'


class PolicyController(RobotOperations):
    """The composition root owns model calls; skills receive only bound state."""

    def __init__(self, *args, compiler=None, judgments=None, known_assets=None,
                 environment_assumptions=None, object_database=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment_assumptions = copy.deepcopy(environment_assumptions or {})
        schemas = {'identify_target': TargetArguments, 'identify_destination': DestinationArguments,
                   'choose_arm': ArmArguments, 'build_scene': BuildSceneArguments,
                   'prepare_unpack': PrepareUnpackArguments, 'observe_scene': SceneObserveArguments, 'capture_scene': ObserveArguments,
                   'verify_place': VerifyPlaceArguments,
                   'move_to_observe': ObservationMotionArguments}
        names = ('ground_task', 'capture_scene', 'move_to_observe', 'return_to_initial_pose', 'observe_scene', 'select_target', 'identify_target', 'identify_destination',
                 'choose_arm', 'build_scene', 'generate_grasps', 'plan_grasp', 'execute_grasp', 'recover_failed_grasp',
                 'verify_grasp', 'perceive_placement_zone',
                 'plan_place', 'execute_place', 'verify_place', 'prepare_unpack',
                 'vla_unpack', 'verify_unpack',
                 'verify_task_completion', 'prepare_failure_return', 'observe_after_failure',
                 'build_failure_scene', 'return_after_failure')
        for name in names:
            self.registry.register(name, schemas.get(name, NoArguments), partial(self._dispatch, name))
        config = VisualAPI.from_environment()
        self.judgments = judgments or DecisionsJudgments(api_key=config.api_key, timeout_s=config.timeout_s)
        self.instance_directory = self.data_dir / 'instances'
        self.object_database = None
        if object_database is not None:
            self._bind_instance_database(object_database)
        self.compiler = compiler or ResponsesCompiler(api_key=config.api_key,
            budget=ResponsesBudget(),
            compilation_memory=CompilationMemory(self.data_dir / 'compilation-memory.sqlite3'),
            timeout_s=float(os.environ.get('LUNA_RESPONSES_TIMEOUT_S', '600')),
            registry=self.registry, known_assets=known_assets, configured_regions={DESTINATION_ID: {
                'description': 'configured unmarked work area on the measured main table',
                'region': self.region}})
        self.unpack_effect = VLAUnpack(self.simulation)
        self.task_spec = None
        self.active_object_task = None
        self.object_tasks = {}
        self.graph_executor = None
        self.observation_epoch = 0

    def clear(self):
        from ..object_memory import ObjectMemory
        from ..spatial_memory import SpatialMemory

        super().clear()
        self.graph_executor = None
        self.object_database = self.object_registrar = self.memory_writer = None
        self.algorithms.perception.visual_identity.load_identity_archive = None
        self.object_memory = ObjectMemory()
        self.spatial_memory = SpatialMemory()

    def _bind_instance_database(self, database):
        from .physical_memory import PhysicalMemoryWriter

        self.object_database = database
        self.memory_writer = PhysicalMemoryWriter(database)
        self.object_registrar = ObjectRegistrar(database, self.judgments)
        database.subscribe(self._database_committed)
        self.algorithms.perception.visual_identity.load_identity_archive = database.identity_archive

    def open_experiment_memory(self, run_id):
        # Bootstrap can fail before it builds this experiment’s graph.
        self.graph_executor = None
        # The run UUID also scopes reads of RGBD references, operation history and product lineage.
        database = ObjectDatabase(self.instance_directory / run_id / 'object-memory.sqlite3')
        self._bind_instance_database(database)

    async def _dispatch(self, name, context, arguments):
        return await self.execute(name, context, arguments.model_dump())

    async def start(self, instruction, *, experiment=None):
        if (self.state.get('partial_execution_failure')
                and self.simulation.robot_snapshot()['generation'] == self.state.get('generation')):
            raise ValueError('partial_execution_requires_explicit_recovery')
        if self.holding_state is not None:
            raise ValueError('holding_state_requires_explicit_recovery')
        return await super().start(instruction, experiment=experiment)

    async def motor(self, function, *args, **kwargs):
        # Every dispatched actuator change invalidates previously captured evidence,
        # including a partially executed action that later fails.
        self._require_execution_recovery()
        step = self.state.get('current_step')
        if step is not None:
            step.setdefault('motor_commands', []).append(function.__name__)
        self.state['facts'].update(observation_pose_ready=False, grasp_view_ready=False, at_initial_pose=False)
        self.state['facts']['task_verified'] = False
        self.observation_epoch += 1
        return await super().motor(function, *args, **kwargs)

    def _require_execution_recovery(self):
        if not self.state.get('partial_execution_failure'):
            return
        self.state['facts']['recovery_needed'] = True
        name = (self.state.get('current_step') or {}).get('skill')
        if name == 'return_after_failure' and self.state['facts'].get('terminal_motion_stopped') is True:
            return
        contract = self.registry.contracts.get(name)
        if contract is None or contract.requires.get('recovery_needed') is not True:
            raise ValueError('partial_execution_requires_explicit_recovery')

    def _record_partial_execution_failure(self, context, result):
        step = self.state.get('current_step') or {}
        if (not step.get('motor_commands') and not step.get('effect_commands')
                or self.state.get('partial_execution_failure')):
            return
        failure = {**copy.deepcopy(result), 'partial_execution': copy.deepcopy(step)}
        self.state['partial_execution_failure'] = failure
        if self.task_progress is not None:
            self.record_failure(context, step.get('stage', step['skill']),
                                result.get('reason', 'partial_execution_unconfirmed'), failure)
        else:
            self.state['facts'].update(recovery_needed=True, plan_ready=False, place_plan_ready=False)
            self.state['failed_step'] = copy.deepcopy(step)

    async def capture_observation(self, context, *, capture=None, recovery=False, bootstrap=False):
        authority = context.get('capture_authority')
        if capture is None and not bootstrap and not authority:
            raise ValueError('observation_requires_graph_capture_contract')
        epochs = context.setdefault('capture_epochs', {})
        if capture is not None and epochs.get(capture['observation_id'], 0) != self.observation_epoch:
            raise ValueError('fresh_observation_required_after_action')
        result = await super().capture_observation(context, capture=capture, recovery=recovery)
        if capture is None:
            result['capture_authority'] = 'task_bootstrap' if bootstrap else authority
        epochs[result['observation_id']] = self.observation_epoch
        context['latest_capture'] = result
        return result

    async def capture_scene_observation(self, context, arguments, *, purpose='associate_objects'):
        from ..robot_skills.observation import capture_reuse_rejection

        mode = SceneObserveArguments.model_validate(arguments).capture_mode
        capture = context.get('latest_capture')
        identifier = (capture or {}).get('observation_id')
        initial_reference = bool(identifier and not context.get('analyzed_capture_ids')
            and identifier == context.get('initial_capture', {}).get('observation_id')
            == context.get('identity_reference', {}).get('observation_id'))
        reason = 'model_requested_fresh_capture'
        if capture is not None:
            reason = capture_reuse_rejection(capture, self.simulation.robot_snapshot(
                include_tcp=False, expected_generation=self.state['generation']),
                epoch_matches=context.get('capture_epochs', {}).get(identifier) == self.observation_epoch,
                now=time.monotonic(), max_age_s=None)
        if initial_reference and mode == 'reuse_current' and reason is not None:
            raise DiagnosticError('initial_reference_capture_invalidated', {
                'component': 'observation_provenance', 'observation_id': identifier,
                'reason': reason, 'scope': 'first_analysis_of_compilation_reference'})
        if mode == 'fresh':
            reason = 'model_requested_fresh_capture'
        result = await self.capture_observation(context, capture=capture if reason is None else None)
        context.pop('fine_capture', None)
        product_capture = context.get('unpacked_grasp_capture')
        if product_capture and product_capture['observation_id'] != result['observation_id']:
            # Drop only a stale sensing constraint; the product's verified RGB
            # references and lineage stay available to the new identity judgment.
            context.pop('unpacked_grasp_capture', None)
            context.pop('unpacked_grasp_target_id', None)
        usage = {'observation_id': result['observation_id'], 'capture_mode': mode,
                 'reused': reason is None, 'previous_observation_id': identifier,
                 'reason': reason or 'current_capture_versions_and_feedback_match',
                 'captured_monotonic_s': result.get('captured_monotonic_s'),
                 'reuse_scope': 'initial_reference_analysis' if initial_reference and reason is None else 'recent_current_capture',
                 'purpose': purpose, 'other_purpose_verdict_reused': False}
        context['observation_acquisition'] = usage
        self.emit('observation_reused' if reason is None else 'observation_acquired',
                  '复用当前观测图像' if reason is None else '采集新的观测图像', usage)
        return result

    async def _step(self, name, context, arguments=None):
        from ..core.failure_policy import TERMINAL_SKILLS
        from ..robot_skills.return_pose import observation_motion_scene_ready

        self.check_generation()
        terminal = name in TERMINAL_SKILLS
        if not terminal:
            self.check_log()
        self.state['facts']['observation_motion_scene_ready'] = observation_motion_scene_ready(
            context, collision=self.spatial_memory.collision_snapshot())
        if name == 'move_to_observe':
            configured = (arguments or {}).get('viewpoint', 'configured') == 'configured'
            # The authored recovery may return to the configured observation pose
            # without another semantic permission check. Physical prerequisites remain.
            if not configured:
                await self._require_visibility_recovery(context)
        if terminal and name != 'prepare_failure_return' and not context.get('terminal_motion_stopped'):
            raise ValueError('terminal_motion_not_stopped')
        if name == 'prepare_failure_return':
            context.setdefault('terminal_failed_step', copy.deepcopy(self.state.get('current_step')))
            context.setdefault('terminal_original_failure', copy.deepcopy(
                getattr(getattr(self, 'graph_executor', None), 'last_failure', None)
                or self.state.get('failure_report') or self.state.get('current_step')))
        self.begin_step(name)
        if self.state.get('partial_execution_failure'):
            self.state['facts']['recovery_needed'] = True
            if self.registry.contracts[name].invalidates_observation and not terminal:
                self._require_execution_recovery()
        started = time.monotonic()
        self.emit('skill_start', name, {'arguments': arguments or {}, 'target_id': self.state.get('target_id')})
        effects = self.registry.contracts[name].provides
        if self.registry.contracts[name].invalidates_observation:
            self.observation_epoch += 1
        try:
            if self.registry.contracts[name].captures_observation:
                context['capture_authority'] = name
            result = await self.registry.execute(name, self.state['facts'], context, arguments or {})
        except (Exception, asyncio.CancelledError) as error:
            for fact in effects:
                self.state['facts'][fact] = False
            for fact, value in self.registry.contracts[name].failure_effects.items():
                if value is None:
                    self.state['facts'].pop(fact, None)
                else:
                    self.state['facts'][fact] = value
            failure = skill_error_result(error, self.state, arguments or {})
            self._record_partial_execution_failure(context, failure)
            if self.task_progress is not None and not terminal:
                self._persist_memory(context, name, failure)
            raise
        finally:
            context.pop('capture_authority', None)
            self.state['facts']['observation_motion_scene_ready'] = observation_motion_scene_ready(
                context, collision=self.spatial_memory.collision_snapshot())
            if self.registry.contracts[name].invalidates_observation:
                # A new raw result photograph is not an ID-associated objects
                # observation. Successful provides below may explicitly restore it.
                self.state['facts']['observed'] = False
                self.state['facts']['capture_ready'] = False
                self.state['facts']['task_verified'] = False
        self.check_generation()
        status = result_status(result)
        for fact in effects:
            self.state['facts'][fact] = status == 'success'
        if status in {'failure', 'unknown'}:
            for fact, value in self.registry.contracts[name].failure_effects.items():
                if value is None:
                    self.state['facts'].pop(fact, None)
                else:
                    self.state['facts'][fact] = value
            self._record_partial_execution_failure(context, result)
        elif (self.registry.contracts[name].requires.get('recovery_needed') is True
              and not self.state['facts'].get('recovery_needed')):
            self.state.pop('partial_execution_failure', None)
        self.emit('skill_complete', name, {'result': result, 'duration_s': time.monotonic() - started})
        if self.task_progress is not None and not terminal:
            if name != 'observe_scene' or status != 'success':
                self._persist_memory(context, name, result)
            self.sync_memory(context)
            if name.startswith('verify_') or name == 'observe_scene':
                calls = [event for event in self.events if event['kind'] == 'visual_decisions_call'
                         and event.get('invocation_id') == self.state['current_step']['invocation_id']]
                self.emit('objects_snapshot', '时间记忆数据库快照', {
                    'checkpoint': name, 'task_progress': self.state['task_progress'],
                    'temporal_memory': self.object_memory.snapshot(),
                    'model_event_ids': [event['id'] for event in calls],
                    'evidence_source': (result.get('container_entry', {}).get('source') or
                        result.get('holding_presence', {}).get('source') or
                        result.get('visual_evidence', {}).get('source') or 'visual_checkpoint')})
        return result

    def record_object_operation(self, context, operation):
        """Commit at the physical checkpoint, before any later stage can fail."""
        self.object_memory = self.memory_writer.operation(self.object_memory, operation)
        if self.task_progress is not None:
            self.sync_memory(context)

    def _database_committed(self, changes):
        self.spatial_memory.synchronize_database(changes, world_key=self.object_registrar.world_key)
        if changes and self.object_registrar.world_key is not None:
            from ..object_memory import ObjectMemory, restored_object_record

            refreshed = ObjectMemory()
            active_ids = set(self.object_memory.entities()) if self.task_spec is not None else None
            refreshed.restore(tuple(restored_object_record(row) for row in
                                    self.object_database.entities(world_key=self.object_registrar.world_key)
                                    if active_ids is None or row['id'] in active_ids))
            self.object_memory = refreshed
        if self.spatial_memory.refresh_required:
            self.state['facts'].update(scene_available=False, scene_ready=False,
                                       observation_map_ready=False, observation_motion_scene_ready=False)

    async def verify_scene(self, context):
        self._database_committed(self.object_database.changes_since(self.spatial_memory.database_revision))
        return await super().verify_scene(context)

    def record_object_observation(self, context, observation):
        """Persist accepted measurements before exposing either memory's new version."""
        self.object_memory, proposed_geometry = self.memory_writer.observation(
            self.object_memory, self.geometry_measurements, capture=context['latest_capture'],
            observation=observation, run_id=self.run_id, step=self.state['current_step'])
        if proposed_geometry != self.geometry_measurements:
            self.spatial_memory.invalidate([])
        self.geometry_measurements = proposed_geometry

    def _persist_memory(self, context, skill, result, *, source_memory=None):
        """Record dated evidence without deciding graph control or task completion."""
        self.object_memory = self.memory_writer.checkpoint(
            self.object_memory if source_memory is None else source_memory,
            skill=skill, result=result, status=result_status(result),
            capture=context.get('latest_capture', {}), observation=context.get('observation', {}),
            target_id=self.state.get('target_id'), run_id=self.run_id, step=self.state['current_step'],
            release_execution=context.get('release_execution_evidence'),
            release_attempt=context.get('release_verification_attempt'),
            container_receipt=context.get('container_entry_receipt') if skill == 'verify_place' else None)
        if skill == 'verify_place' and result.get('success') is True and result.get('container_entry'):
            # The committed transfer changes physical state and invalidates geometry.
            # Keep the exact receipt bound to the next observed workspace rebuild.
            scene = self.spatial_memory.scene
            self.spatial_memory.accept_container_occupancy(scene, result['container_entry'],
                previous_scene_token=scene['scene_token'])

    async def _bootstrap(self, context):
        self.object_registrar.world_key = self.simulation.world_id
        if self.object_registrar.world_key is None:
            raise DiagnosticError('robot_world_identity_unavailable', {})
        self._database_committed(self.object_database.changes_since(0))
        self.compiler.scene_context['grasp_parameter_defaults'] = {
            key: context['planning_settings'][key] for key in
            ('cad_grasp_insertion_m', 'rag_grasp_insertion_m')}
        self.compiler.scene_context['container_release_region'] = {key: context['planning_settings'][key]
            for key in ('container_release_fixed_xy', 'container_release_x_m', 'container_release_y_m',
                        'container_release_region_size_m')}
        self.compiler.scene_context['environment_assumptions'] = copy.deepcopy(self.environment_assumptions)
        self.compiler.scene_context['completion_requirements'] = copy.deepcopy(
            self.environment_assumptions.get('completion_requirements', {}))
        if context.get('initial_joint_reference'):
            self.compiler.scene_context['initial_joint_reference'] = copy.deepcopy(context['initial_joint_reference'])
        capability = vla_unpack_configuration(context['planning_settings'])
        self.compiler.scene_context.setdefault('capabilities', {})['vla_unpack'] = copy.deepcopy(capability)
        self.compiler.configured_regions[capability['input_region_id']] = {
            'description': ('Unmarked VLA input area on the measured table. Its polygon describes the '
                'processing capability acceptance area, not permission to change a requested placement XY. '
                'When the user requests the designated/configured region, identify_destination uses '
                'placement_mode=configured_xy and fixes the supplied center_world_xy_m. '
                'The capability region precise_xy flag describes processing tolerance only.'),
            'capability': 'vla_unpack', 'region': copy.deepcopy(capability['input_region'])}
        if self.compiler.compilation_memory is not None:
            self.compiler.compilation_memory.bind_context({
                'planning_settings': context['planning_settings'],
                'environment_assumptions': self.environment_assumptions,
                'known_assets': self.compiler.known_assets,
                'configured_regions': self.compiler.configured_regions,
                'skill_detail': self.compiler.skill_detail})
        capture = await self.capture_observation(context, bootstrap=True)
        at_observation_pose = self.is_observation_pose(context, capture)
        self.state['facts']['observation_pose_ready'] = at_observation_pose
        self.compiler.initial_facts = {
            **getattr(self.compiler, 'initial_facts', {}), 'observation_pose_ready': at_observation_pose,
            'collision_map_initialized': self.spatial_memory.initialized, 'at_initial_pose': True}
        self.begin_step('compile_task')
        self.record_observation(capture, 'initial')
        audit, done = self.visual_audit_sink()
        self.compiler.audit = audit
        try:
            self.compiler.memory_reference = self.object_registrar.reference()
            plan = await self.call_algorithm(self.compiler.compile, capture, instruction=self.state['instruction'])
            if not plan.task.supported:
                raise ValueError('task_capability_missing:' + plan.task.reason)
            from ..observation_labels import proposal_box_labels

            self.record_annotation(capture, proposal_box_labels(capture, plan.scene_inventory))
            self.emit('responses_localization', 'Responses 已定位声明物体并生成任务选择器', {
                'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'declared_objects': [row.model_dump(mode='json') for row in plan.declared_objects],
                'scene_inventory': [row.model_dump(mode='json') for row in plan.scene_inventory],
                'task_selectors': [row.model_dump(mode='json') for row in plan.task_selectors],
                'destination_regions': [row.model_dump(mode='json') for row in plan.destination_regions]})
        finally:
            done.set()
        # Symbolic refs stay in the graph. No permanent identity is allocated during compilation.
        from ..object_memory import ObjectMemory
        from .failure_return import wrap_failure_exits
        self.task_spec = plan.task
        self.task_bindings = {'entity_ref_bindings': {}}
        self.policy_graph = Graph.model_validate(wrap_failure_exits(plan.graph.model_dump(mode='json')))
        self.object_tasks = {}
        self.active_object_task = None
        self.object_memory = ObjectMemory()
        context.update(initial_plan=plan, initial_capture=capture, latest_capture=capture, grounded_refs={},
                       identity_reference={'entities': {}, 'identity_references': {}})
        self.state['facts'].update(capture_ready=True, task_grounded=False)
        self.task_progress.configure_task(self.task_spec, [])
        self.state.update(task_spec=self.task_spec.model_dump(mode='json'),
                          policy_graph=self.policy_graph.model_dump(mode='json'))
        self.emit('skill_catalog', '本次 graph 使用的 skills 契约', {'skills': skill_catalog(self.registry)})
        self.emit('skill_program', 'Responses 已校验执行 graph', self.state['policy_graph'])

    async def observe(self, context, *, checkpoint=None, capture=None, camera_ids=None,
                      purpose=None, recovery=False):
        checkpoint = checkpoint or 'initial'
        from .grounding import fixed_targets_delivered, prepare_grounding

        scene_only = recovery and context.get('terminal_motion_stopped', False)
        # A map refresh after final delivery still registers what the cameras can see.
        # The completed transfer already binds its source, target and destination.
        skip_role_binding = scene_only or (bool(context.get('grounded_refs'))
            and fixed_targets_delivered(self.task_spec, self.task_progress))
        capture = await self.capture_observation(context, capture=capture, recovery=recovery)
        from ..robot_skills.placement_resolution import digest
        from .memory import perception_motion_context
        from .observation_inputs import observation_identity_scope, observation_inputs

        context['association_scope'] = observation_identity_scope(
            self.object_memory, self.spatial_memory, self.task_progress, self.task_spec,
            target_id=self.state.get('target_id'))

        def analysis_key():
            inputs = observation_inputs(context)
            return digest({'capture_id': capture['observation_id'], 'checkpoint': checkpoint, 'scene_only': skip_role_binding, 'inputs': inputs,
                'full_collision_survey': capture.get('full_collision_survey', False),
                'cad_motion_context': perception_motion_context(self.object_memory, self.spatial_memory)})

        key = analysis_key()
        previous = context.get('_observation_analysis', {})
        if previous.get('key') == key:
            if previous.get('failure'):
                raise DiagnosticError('observation_unchanged_since_failed_analysis', {
                    'previous_failure': previous['failure'], 'observation_id': capture['observation_id'],
                    'required_evidence': 'Change viewpoint or identity inputs; identical pixels are not new evidence.'})
            if context.get('observation', {}).get('observation_id') == capture['observation_id']:
                self.emit('observation_analysis_reused', '复用未变化的场景观测结果',
                    {'observation_id': capture['observation_id'], 'action_epoch': self.observation_epoch})
                return context['observation']
        try:
            await prepare_grounding(self, context, capture, scene_only=skip_role_binding)
            observation = await Runner.observe(self, context, checkpoint=checkpoint, capture=capture,
                camera_ids=['1', '2', '3'], purpose='geometry', recovery=recovery)
        except (ValueError, RuntimeError) as error:
            if any(word in str(error) for word in ('unknown', 'uncertain', 'identity_unresolved')):
                context['_observation_analysis'] = {'key': analysis_key(), 'failure': str(error)}
            raise
        # A configured work area is an explicit prior, not a visually invented marker.
        regions = {} if scene_only else self.compiler.configured_regions
        for identifier, configuration in regions.items():
            region = configured_destination(observation, configuration.get('region', self.region),
                identifier=identifier, support_id=self.task_bindings.get('support_id'))
            observation['objects'] = [obj for obj in observation['objects'] if obj['id'] != identifier]
            observation['objects'].append(region)
            self.planning_regions[identifier] = copy.deepcopy(region)
        context.update(observation=observation, objects_observation=copy.deepcopy(observation))
        self.sync_memory(context)
        context['_observation_analysis'] = {'key': analysis_key()}
        return observation

    async def _refresh_observed_support(self, context, identifier):
        entities = self.object_memory.entities()
        entity = entities.get(identifier, {})
        previous = entity.get('observed_support')
        capture = context['latest_capture']
        record = self.task_progress.records.get(identifier, {})
        if (isinstance(previous, dict)
                and previous.get('observation_id') == capture['observation_id']
                and previous.get('source') in {'openai_decisions', 'measured_rgbd_geometry'} and previous.get('request_hash')
                and previous.get('support_entity_id')):
            return
        candidates = {key: {field: value[field] for field in ('id', 'kind', 'color', 'description') if field in value}
                      for key, value in entities.items() if value.get('kind') in {'table', 'bin'}}
        if not candidates:
            raise ValueError('observed_support_candidates_missing')
        measured = support_context(capture, context.get('observation', {}), entity, candidates)
        measured['support_memory'] = support_memory(entity)
        proposal = entity.get('current_identity_proposal', {})
        # A dated relation can focus a CURRENT question without establishing its answer.
        remembered = measured['support_memory'].get('previous_support_assessment') or {}
        if (proposal.get('observation_id') == capture['observation_id']
                and proposal.get('parent_id') in candidates):
            proposed_support = proposal['parent_id']
            hypothesis_source = proposal
        elif remembered.get('support_entity_id') in candidates:
            proposed_support = remembered['support_entity_id']
            hypothesis_source = remembered
        else:
            proposed_support = proposal.get('parent_id')
            hypothesis_source = proposal
        if proposed_support in candidates:
            measured['support_hypothesis'] = {'support_entity_id': proposed_support,
                'source': hypothesis_source.get('source'),
                'observation_id': hypothesis_source.get('observation_id'),
                'role': 'Dated hypothesis to check in CURRENT images, not an accepted current relation. '
                        'Select the support established by CURRENT evidence from the offered support choices.'}
        from ..agent.support import support_question
        requests = [support_question(entity, candidates, proposed_support)]
        # Dated verified relations are memory, while old free-text descriptions
        # and requested destinations are not current support evidence.
        reference = copy.deepcopy(context.get('identity_reference', {}))
        reference.setdefault('entities', {identifier: measured['subject'], **measured['support_candidates']})
        for reference_entity in reference['entities'].values():
            reference_entity.pop('description', None)
            reference_entity.pop('initial_parent_id', None)
        import hashlib

        from .support import geometric_support

        support_id = geometric_support(measured)
        support_source = 'measured_rgbd_geometry' if support_id is not None else 'openai_decisions'
        if support_id is not None:
            geometry = measured['current_measured_geometry']
            digest = hashlib.sha256(json.dumps(geometry, sort_keys=True, allow_nan=False).encode()).hexdigest()
            result = {'evidence': {'request_hash': digest}, 'judgments': {
                'current_support': {'status': 'known', 'value': support_id, 'source': support_source}}}
            judgment = result['judgments']['current_support']
            score = {'geometry_evidence': copy.deepcopy(geometry)}
        else:
            result = await self._judge(context, requests, extra=measured, include_task=False, reference=reference)
            judgment = result['judgments']['current_support']
            support_id = judgment.get('value') if judgment['status'] == 'known' else None
            score = {'confidence': judgment.get('confidence')}
        if support_id not in candidates:
            rejected = judgment['status'] == 'no' or judgment.get('value') == 'none_of_listed'
            raise DiagnosticError('observed_support_rejected' if rejected else 'observed_support_unknown',
                                  judgment_details(result, requests))
        proof = {'support_entity_id': support_id, 'observation_id': capture['observation_id'],
                 'generation': capture.get('generation'), 'source': support_source,
                 'request_hash': result['evidence']['request_hash'],
                 'assessment_basis': 'current_evidence_and_verified_memory',
                 'support_memory': copy.deepcopy(measured['support_memory']),
                 **score, 'judgments': copy.deepcopy(result['judgments'])}
        from ..memory_contracts import ObjectOperation

        self.record_object_operation(context, ObjectOperation(entity_id=identifier,
            event_key=self.run_id + ':' + self.state['current_step']['invocation_id'] + ':support',
            kind='observed_support', observation_id=capture['observation_id'],
            observed_support=copy.deepcopy(proof), evidence=copy.deepcopy(proof)))
        if identifier in self.task_progress.records:
            self.task_progress.records[identifier]['observed_support'] = copy.deepcopy(proof)
            if not record.get('transfers') and not record.get('input_entity_id'):
                initial = record.setdefault('initial_source', {})
                initial.update(measurement_observation_id=capture['observation_id'],
                    task_source_evidence={**copy.deepcopy(proof), 'status': 'yes',
                        'target_id': identifier, 'source_id': support_id,
                        'source_kind': entities[support_id]['kind']})
        self.task_progress.ledger_revision += 1
        for snapshot in (context.get('observation', {}), context.get('objects_observation', {})):
            if snapshot.get('observation_id') != capture['observation_id']:
                continue
            for records in (snapshot.get('scene_entities', {}), snapshot.get('object_memory_snapshot', {}).get('entities', {})):
                if identifier in records:
                    records[identifier]['observed_support'] = copy.deepcopy(proof)
        self.sync_memory(context)
        self.emit('observed_support_refreshed', 'Decisions 确认当前支撑关系', {'target_id': identifier, **proof})

    def sync_memory(self, context):
        objects = self.object_memory.snapshot()
        self.state['task_bindings'] = copy.deepcopy(self.task_bindings)
        self.state['task_progress'] = execution_snapshot(
            self.task_progress, self.object_memory, self.geometry_measurements, self.planning_regions)
        reference = context.get('identity_reference')
        if reference is not None:
            reference['object_memory_snapshot'] = objects
            reference['entities'] = current_memory_entities(reference)
            stored = self.object_database.reference(reference['entities'])
            reference['identity_references'] = stored['identity_references']
            reference['previous_observation_id'] = stored['previous_observation_id']

    def _check_source_view(self, observation):
        """Validate only the identities attached to current measured geometry."""
        visual = observation.get('visual_identity', {})
        coverage = visual.get('identity_coverage', {})
        evidence = visual.get('evidence', {})
        measured = {item['id'] for item in observation['objects']
                    if item.get('kind') not in {'table', 'marker'}}
        if (not bound_identity_coverage_source(observation)
                or coverage.get('observation_id') != observation['observation_id']
                or (coverage.get('source') != 'rgbd_instance_association' and (
                    not coverage.get('request_hash') or coverage['request_hash'] != evidence.get('request_hash')
                    or coverage.get('selection_rule') != 'probability_argmax'))
                or not measured.issubset(coverage.get('accepted_entity_ids', []))):
            raise DiagnosticError('source_objects_unknown', {
                'component': 'current_identity_evidence',
                'observation_id': observation['observation_id'],
                'identity_coverage': copy.deepcopy(coverage),
                'measured_entity_ids': sorted(measured)})

    async def _classify_task_inputs(self, context):
        """Ask only for newly observed instances whose task membership is unresolved."""
        observation = context['observation']
        unresolved = []
        for obj in observation['objects']:
            identifier = obj['id']
            record = self.task_progress.records.get(identifier)
            if record is None or self.task_progress.membership.get(identifier) != 'unknown':
                continue
            if self.task_spec.scope == 'ids':
                self.task_progress.membership[identifier] = (
                    'required' if identifier in self.task_spec.target_refs else 'excluded')
                continue
            cached = record.get('task_membership_evidence', {})
            if (not cached.get('request_hash') or (cached.get('answer', {}).get('status') == 'unknown'
                    and cached.get('observation_id') != observation['observation_id'])):
                result = await self._judge(context, [JudgmentRequest(name='input_matches_task',
                    instructions=f'The CURRENT {obj["kind"]} instance marked ID{identifier} itself matches the input selector '
                    + json.dumps(self.task_spec.target_selector, ensure_ascii=False)
                    + ' and satisfies the source restriction in task.source_ref when present.')],
                    extra={'verification_scope': 'input_membership',
                           'subject_history': {key: value for key, value in self.object_memory.entities()[identifier].items()
                               if key in {'id', 'observed_support', 'processing_history', 'input_entity_id',
                                          'output_entity_id', 'creation_evidence', 'transfers', 'last_position_measurement'}},
                           'subject': {key: copy.deepcopy(obj[key]) for key in
                               ('id', 'kind', 'color', 'position', 'measured_relationships') if key in obj}})
                answer = result['judgments']['input_matches_task']
                cached = {'observation_id': observation['observation_id'], 'answer': copy.deepcopy(answer),
                          'request_hash': result['evidence']['request_hash']}
                record['task_membership_evidence'] = cached
            status = cached['answer']['status']
            self.task_progress.membership[identifier] = (
                'required' if status == 'yes' else 'excluded' if status == 'no' else 'unknown')
            if status == 'unknown':
                unresolved.append({'entity_id': identifier, **copy.deepcopy(cached)})
        return unresolved

    async def _select_target(self, context):
        observation = context['observation']
        completed = self.task_progress.completed_input_ids()
        pending = set(self.task_progress.input_ids) - completed
        if pending:
            return {'status': 'failure', 'success': False, 'reason': 'selected_input_unfinished',
                    'input_ids': sorted(pending)}
        pose_facts = {key: self.state['facts'].get(key, False)
                      for key in ('observation_pose_ready', 'grasp_view_ready')}
        self.reset_transfer(context)
        self.state['facts'].update(pose_facts)
        if self.task_spec.quantity is not None and len(completed) >= self.task_spec.quantity:
            return {'status': 'empty', 'success': False, 'reason': 'requested_quantity_selected'}
        unresolved_membership = await self._classify_task_inputs(context)
        candidates = [obj for obj in observation['objects']
            if self.task_progress.membership.get(obj['id']) == 'required'
            and obj['id'] not in completed
            and not self.task_progress.records.get(obj['id'], {}).get('input_entity_id')
            and self.task_progress.records.get(obj['id'], {}).get('status') not in {'consumed', 'completed', 'deferred'}
            and (self.task_spec.scope != 'ids' or obj['id'] in self.task_spec.target_refs)]
        # Object membership comes from committed memory; the current observation
        # supplies measurement and visibility, never a second independent queue.
        remembered = {row['id']: row for row in self.object_database.entities(
            world_key=self.object_registrar.world_key)}
        candidates = [obj for obj in candidates if obj['id'] in remembered
                      and remembered[obj['id']].get('status', 'active') == 'active']
        if candidates:
            selected, evidence = candidates[0], []
            if self.task_spec.selection_order == 'explicit':
                pending_order = [i for i in self.task_spec.target_refs if i not in completed]
                selected = next((obj for obj in candidates if pending_order and obj['id'] == pending_order[0]), None)
                if selected is None:
                    return {'status': 'unknown', 'success': False, 'reason': 'requested_next_id_not_visible'}
                evidence.append({'source': 'explicit_user_order'})
            elif len(candidates) > 1 or self.task_spec.selection_order in {'left_to_right', 'right_to_left'}:
                order = self.task_spec.selection_order
                if order in {'left_to_right', 'right_to_left'}:
                    required = {i for i, value in self.task_progress.membership.items() if value == 'required'}
                    absent = required - completed - {obj['id'] for obj in candidates}
                    if absent or unresolved_membership or observation.get('visual_identity', {}).get('unassociated_candidates'):
                        return {'status': 'unknown', 'success': False, 'reason': 'ordered_comparison_set_incomplete'}
                subjects = []
                for obj in candidates:
                    row = {key: obj[key] for key in ('id', 'kind', 'color', 'description') if key in obj}
                    points = np.asarray(obj.get('points_world_m', []))
                    if points.ndim == 2 and points.shape[1] == 3 and len(points) and np.isfinite(points).all():
                        row['center_world_m'] = np.median(points, axis=0).tolist()
                        row['visible_bounds_world_m'] = np.quantile(points, [.02, .98], axis=0).tolist()
                    subjects.append(row)
                choices = tuple(obj['id'] for obj in candidates) + ('unknown',)
                criterion = {
                    'left_to_right': '本轮取机器人最左边的合格对象，即基座 Y 最大的对象。',
                    'right_to_left': '本轮取机器人最右边的合格对象，即基座 Y 最小的对象。',
                    'visual': '本轮优先选择夹持面露出、周边障碍少的合格对象。'}[order]
                descriptions = {}
                for subject in subjects:
                    identifier = subject['id']
                    center = subject.get('center_world_m')
                    position = f'当前基座 Y={center[1]:+.3f} 米' if center else '当前基座位置未知'
                    descriptions[identifier] = f"ID {identifier}, {remembered[identifier].get('description', '')}, {position}"
                judged = await self._judge(context, [ChoiceRequest(name='next_target', choices=choices,
                    descriptions=descriptions, instructions=criterion +
                    '仅在当前列出的未完成 ID 中选择。基座 +Y 是机器人左侧，图像左右随相机变化。'
                    '结合当前图和测量，有遮挡或顺序无法区分时选 unknown。')], include_task=False, reference={}, extra={
                            'verification_scope': 'target_selection', 'candidate_objects': subjects,
                            'instruction': self.state['instruction'], 'selection_order': order,
                            'visual_modalities': ['rgb'],
                            'visual_masks': {camera: view.get('identity_masks', view.get('instance_masks', {}))
                                             for camera, view in observation.get('views', {}).items()}})
                answer = judged['judgments']['next_target']
                if answer.get('value') not in choices[:-1]:
                    return {'status': 'unknown', 'success': False, 'reason': 'next_target_visually_unresolved',
                            'details': judged}
                selected = next(obj for obj in candidates if obj['id'] == answer['value'])
                evidence.append(judged)
            else:
                evidence.append({'source': 'sole_current_eligible_object'})
            return {'status': 'success', 'success': True, 'selected_id': selected['id'],
                    'observation_id': observation['observation_id'], 'selection_evidence': evidence}
        unresolved = observation.get('visual_identity', {}).get('unassociated_candidates', [])
        details = {'unassociated_candidates': copy.deepcopy(unresolved),
                   'unresolved_membership': unresolved_membership}
        missing_requested = set(self.task_spec.target_refs) - completed if self.task_spec.scope == 'ids' else set()
        if missing_requested:
            return {'status': 'unknown', 'success': False, 'reason': 'target_identity_unresolved',
                    'details': {**details, 'missing_entity_ids': sorted(missing_requested)}}
        if self.task_spec.scope == 'all':
            from ..agent.completion import source_presence_request

            requests = [source_presence_request()]
            presence = await self._judge(context, requests, extra={'verification_scope': 'task_completion'})
            context['source_presence_evidence'] = {
                'observation_id': observation['observation_id'], 'epoch': self.observation_epoch,
                'result': copy.deepcopy(presence)}
            answer = presence['judgments']['source_has_remaining_input']['status']
            if answer != 'no':
                return {'status': 'unknown', 'success': False,
                    'reason': 'remaining_input_unlocalized' if answer == 'yes' else 'source_objects_unknown',
                    'details': {**details, **judgment_details(presence, requests)}}
            return {'status': 'empty', 'success': False, 'reason': 'source_confirmed_empty',
                    'observation_id': observation['observation_id'], 'source_presence': presence}
        return {'status': 'empty', 'success': False, 'reason': 'no_current_eligible_input',
                'observation_id': observation['observation_id'], 'details': details}

    async def build_scene(self, context, exclusion_state):
        # Only a verified transformation can retire an input identity.
        if self.task_progress is None:
            return await super().build_scene(context, exclusion_state)
        from .memory import measurement_reference

        records = self.object_memory.entities()
        for measured in context['observation']['objects']:
            stored = records.get(measured['id'])
            if stored is not None and stored.get('geometry_reference') != measurement_reference(
                    measured, context['observation']):
                raise ValueError('collision_geometry_not_committed_to_database:' + measured['id'])
        retired = self.object_memory.snapshot()['retired_entity_ids']
        self.spatial_memory.require_entities(self._collision_entity_ids(context), retired)
        return await super().build_scene(context, exclusion_state)

    def _collision_entity_ids(self, context):
        """Keep active physical identities in scope through partial observations."""
        entities = self.object_memory.entities()
        return {identifier for identifier, row in entities.items()
                if row.get('kind') not in {'table', 'marker'} and row.get('status', 'active') == 'active'}

    def _bind_region(self, observation):
        identifier = self.state.get('destination_id') or self.task_bindings.get('destination_id')
        if identifier.startswith(('table-region:', 'surface-region:')):
            destination = (measured_table_region(observation, identifier.removeprefix('table-region:'))
                if identifier.startswith('table-region:') else
                measured_object_region(observation, identifier.removeprefix('surface-region:')))
            observation['objects'] = [obj for obj in observation['objects'] if obj['id'] != identifier]
            observation['objects'].append(destination)
            observation.setdefault('scene_entities', {})[identifier] = copy.deepcopy(destination)
            self.planning_regions[identifier] = copy.deepcopy(destination)
            self.task_progress.bind_shared_work_area(destination, observation)
            request = self.task_progress.placement_requests[(self.task_progress.current_id, self.task_progress.transfer_index(), identifier)]
            request['release_mode'] = self.task_progress.records[self.task_progress.current_id]['release_mode']
            request['source'] = destination['source']
            request['free_placement'] = True
            request['placement_preference'] = self.task_progress.records[self.task_progress.current_id].get(
                'placement_preference', 'clearance')
            return self.task_progress.placement_request(identifier)
        destination = observed_object(observation, identifier)
        if (destination['kind'] == 'bin' and self.task_progress.records[self.task_progress.current_id]['release_mode']
                == 'container_release'):
            from ..container_release import configure_release_region

            configure_release_region(destination, self.state['planning_settings'])
        self.task_progress.bind_shared_work_area(destination, observation)
        key = (self.task_progress.current_id, self.task_progress.transfer_index(), identifier)
        request = self.task_progress.placement_requests[key]
        request['release_mode'] = self.task_progress.records[self.task_progress.current_id]['release_mode']
        reference = destination['placement_reference']
        request['free_placement'] = not (reference.get('precise_xy') or reference.get('fixed_xy'))
        request.pop('fixed_xy', None)
        request['placement_preference'] = self.task_progress.records[self.task_progress.current_id].get(
            'placement_preference', 'clearance')
        if reference.get('precise_xy'):
            request.update(selected_reference_world_xy_m=copy.deepcopy(reference['position_world_m'][:2]),
                           xy_tolerance_m=reference['xy_tolerance_m'], precise_xy=True)
        else:
            request.pop('precise_xy', None)
            if reference.get('fixed_xy'):
                request.update(fixed_xy=True,
                    selected_reference_world_xy_m=copy.deepcopy(reference['position_world_m'][:2]))
        return self.task_progress.placement_request(identifier)

    async def execute(self, skill, context, arguments=None):
        if skill not in self.registry.handlers:
            raise ValueError('unregistered_skill:' + skill)
        arguments = arguments or {}
        if skill in {'identify_target', 'identify_destination', 'observe_scene', 'build_scene', 'choose_arm',
                     'generate_grasps'}:
            invalidate_placement(self, context, zone=skill in {'identify_target', 'identify_destination', 'observe_scene'})
        if skill == 'capture_scene':
            capture = await self.capture_scene_observation(context, arguments)
            return {'success': True, 'observation_id': capture['observation_id'], 'source': 'base_rgbd'}
        if skill == 'ground_task':
            from .grounding import prepare_grounding
            return await prepare_grounding(self, context, context['latest_capture'],
                                           initial=not self.state['facts'].get('task_grounded'))
        if skill == 'perceive_placement_zone':
            invalidate_placement(self, context, zone=True)
            if self.state['destination_id'].startswith(('table-region:', 'surface-region:')):
                self._bind_region(context['observation'])
            return await perceive_for_controller(self, context)
        if skill == 'identify_target' and self.object_tasks:
            identifier = arguments['target_id']
            record = self.task_progress.records.get(identifier, {})
            root = record.get('root_input_id', identifier)
            if self.active_object_task is not None and root != self.active_object_task.target_refs[0]:
                raise ValueError('target_outside_active_object_task:' + identifier)
        if skill == 'select_target':
            return await self._select_target(context)
        if skill == 'verify_task_completion':
            await self._complete(context)
            self.state['status'] = 'running'
            context['completion_verification_epoch'] = self.observation_epoch
            context['completion_verification_ledger_revision'] = self.task_progress.ledger_revision
            return {'success': True, 'evidence': copy.deepcopy(self.state['completion_evidence'])}
        if skill == 'build_scene':
            mapping = BuildSceneArguments.model_validate(arguments)
            if mapping.exclusion_state == 'selected' and not (
                    self.state['facts'].get('target_selected') and self.state.get('target_id')):
                raise ValueError('selected_mapping_requires_current_target')
            if mapping.exclusion_state == 'observation':
                self.state['facts'].update(scene_ready=False, views_ready=False)
                return await self.build_scene(context, 'observation')
        if skill == 'identify_target' and self.state['facts'].get('target_selected'):
            if context.get('holding') or self.state.get('partial_execution_failure'):
                raise ValueError('target_change_requires_verified_release')
            previous = self.task_progress.records.get(self.state.get('target_id'), {})
            if (self.state['facts'].get('grasp_completed') and not previous.get('confirmed_observation_id')
                    and not previous.get('container_entry')):
                raise ValueError('target_change_requires_verified_release')
            pose_facts = {key: self.state['facts'].get(key, False)
                          for key in ('observation_pose_ready', 'grasp_view_ready')}
            self.reset_transfer(context)
            self.state['facts'].update(pose_facts)
        if skill == 'identify_target':
            await self._refresh_observed_support(context, arguments['target_id'])
        if skill == 'vla_unpack':
            return await self._vla_unpack(context)
        if skill == 'move_to_observe':
            result = await super().execute(skill, context, arguments)
            if result_status(result) == 'success':
                if arguments.get('viewpoint', 'configured') == 'configured':
                    self.require_observation_pose(context)
                context.pop('unpacked_grasp_capture', None)
                context.pop('unpacked_grasp_target_id', None)
            return result
        if skill == 'generate_grasps':
            # Candidate replacement revokes any plan built for the previous set,
            # including when the new sampling attempt fails.
            self.state['facts'].update(plan_ready=False, grasps_ready=False)
            for key in ('plan', 'plan_binding', 'plan_observation_id', 'grasps', 'grasps_observation_id'):
                context.pop(key, None)
            self.emit('skill_configuration', '本次抓取使用工作台配置', {
                'source': 'operator_configuration',
                'parameters': {key: context['planning_settings'][key] for key in (
                    'cad_grasp_insertion_m', 'rag_grasp_insertion_m')},
                'target_id': self.state['target_id'], 'arm': self.state['arm']})
            observation = context['observation']
            destination = observed_object(observation, self.state['destination_id'])
            if destination['kind'] == 'bin':
                self._bind_region(observation)
                self.require_grasp_view(context)
                if not self.state['facts'].get('views_ready'):
                    raise ValueError('current_target_geometry_required')
                context['grasp_placement_request'] = {
                    'object_id': self.state['target_id'],
                    'transport_region': container_transport_region(destination, self.spatial_memory.scene, observation),
                    'orientation_mode': 'preserve',
                    'release_mode': self.task_progress.records[self.state['target_id']]['release_mode'],
                    'free_placement': not destination['placement_reference'].get('fixed_xy', False)}
                context['grasps'] = await self.generate_bound_grasps(context, observation)
                context['grasps_observation_id'] = observation['observation_id']
                return context['grasps']
        if skill == 'identify_destination':
            if arguments['release_mode'] == 'container_release' and arguments['destination_kind'] != 'container':
                raise ValueError('container_release_requires_container')
            identifier = arguments['destination_id']
            observation = context['observation']
            entity = observation.get('scene_entities', {}).get(identifier, {})
            if entity.get('kind') == 'table':
                destination = measured_table_region(observation, identifier)
                identifier = destination['id']
            else:
                destination = observed_object(observation, identifier)
                if destination['kind'] not in {'marker', 'bin'} and arguments['destination_kind'] == 'support_region':
                    destination = measured_object_region(observation, identifier)
                    identifier = destination['id']
            if destination['kind'] not in {'marker', 'bin'}:
                raise ValueError('destination_requires_measured_region_or_container')
            measured_kind = 'container' if destination['kind'] == 'bin' else 'support_region'
            if arguments['destination_kind'] != measured_kind:
                raise DiagnosticError('destination_kind_mismatch', {
                    'destination_id': identifier, 'declared_kind': arguments['destination_kind'],
                    'measured_kind': measured_kind, 'entity_kind': destination['kind'],
                    'observation_id': observation.get('observation_id')})
            if destination['kind'] == 'bin' and arguments['release_mode'] == 'container_release':
                from ..container_release import configure_release_region

                configure_release_region(destination, self.state['planning_settings'])
            if (arguments.get('placement_mode') == 'configured_xy'
                    and destination.get('position_source') != 'configured_xy_measured_support'
                    and not destination.get('placement_reference', {}).get('configured_release_region')):
                raise ValueError('fixed_xy_requires_configured_region')
            local_task = self.active_object_task or self.task_spec
            if local_task is not None and arguments['destination_id'] not in {
                    local_task.destination_ref, local_task.work_area_ref, *local_task.allowed_intermediate_refs}:
                return {'success': False, 'status': 'failure', 'reason': 'intermediate_destination_not_compiled'}
            if destination.get('position_source') == 'configured_xy_measured_support':
                if arguments.get('placement_mode') == 'configured_xy':
                    destination['placement_reference']['fixed_xy'] = True
                else:
                    destination['placement_reference'].pop('fixed_xy', None)
            for key in ('grasps', 'plan', 'place_plan', 'place_plan_binding', 'place_plan_observation_id',
                        'grasp_placement_request', 'reachability_evaluations'):
                context.pop(key, None)
            for fact in ('grasps_ready', 'plan_ready', 'place_plan_ready',
                         'arm_selected', 'arms_evaluated',
                         'destination_is_container'):
                self.state['facts'][fact] = False
            self.state['destination_id'] = identifier
            destination = {**destination, 'reference_id': arguments['destination_id']}
            context['destination'] = destination
            context['destination_orientation_mode'] = 'preserve'
            self.task_progress.records[self.state['target_id']].update(
                destination_id=identifier, destination_ref=arguments['destination_id'],
                destination_kind=destination['kind'], release_mode=arguments['release_mode'],
                placement_preference=arguments.get('placement_preference', 'clearance'))
            if destination['kind'] == 'marker':
                self._bind_region(context['observation'])
            self.state['facts']['destination_is_container'] = measured_kind == 'container'
            self.state['facts']['container_release_mode'] = arguments['release_mode'] == 'container_release'
            return {'destination_id': identifier, 'success': True, 'release_mode': arguments['release_mode'],
                    'placement_preference': arguments.get('placement_preference', 'clearance'),
                    'selection_scope': 'bound_region_not_yet_validated_motion_goal'}
        if skill == 'choose_arm':
            target = observed_object(context['observation'], self.state['target_id'])
            requested = arguments.get('arm', 'auto')
            locked = self.task_progress.bound_arm()
            if locked is not None:
                if requested not in ('auto', locked):
                    raise ValueError('single_arm_transfer_cannot_switch_arm')
                requested = locked
            evaluations = await self.call_algorithm(self.algorithms.evaluate_reachability,
                context['observation'], self.state['target_id'], self.state['destination_id'],
                requested_arm=None if requested == 'auto' else requested,
                scene=self.spatial_memory.scene,
                release_mode=self.task_progress.records[self.state['target_id']]['release_mode'],
                planning_settings=context['planning_settings'])
            context['reachability_evaluations'] = evaluations
            available = [arm for arm in ('A', 'B') if evaluations.get(arm, {}).get('reachable')
                         and requested in ('auto', arm)]
            if context.get('holding'):
                available = [arm for arm in available if arm == context['holding']['arm']]
            if not available:
                return {'success': False, 'reason': 'no_common_arm_for_source_and_fixed_destination', 'evaluations': evaluations}
            failure = public_summary(getattr(self.graph_executor, 'last_failure', None)) or {}
            failed_arm = failure.get('arm') if failure.get('target_id') == self.state['target_id'] else None
            target_position = np.asarray(target['points_world_m'], float).mean(axis=0)
            arm_costs = {}
            for candidate_arm in available:
                tcp = np.asarray(context['observation']['arms'][candidate_arm]['tcp'], float)
                arm_costs[candidate_arm] = float(np.linalg.norm(target_position - tcp[:3, 3]))
            arm = min(available, key=lambda value: (value == failed_arm, arm_costs[value], value))
            context['arm_selection'] = {'source': 'numerical_feasibility_and_tcp_distance',
                'eligible_arms': available, 'source_distance_m': arm_costs, 'previous_failed_arm': failed_arm,
                'selected_arm': arm}
            self.task_progress.bind_arm(arm)
            self.state['arm'] = arm
            return {'arm': arm, 'evaluations': evaluations, 'success': True}
        if skill == 'plan_place':
            for key in ('place_plan', 'place_plan_binding', 'place_plan_observation_id'):
                context.pop(key, None)
            self.state['facts']['place_plan_ready'] = False
            await self.verify_scene(context)
            observation = self.measured_state(context)
            await self.execution_stage(context, 'prepare_attachment', self.prepare_attachment, context, observation)
            request = self._bind_region(context['observation'])
            selection = await prepare_fixed_placement(self, context, observation, request)
            if selection.get('status') != 'success':
                return selection
            binding = self.motion_plan_binding(context, observation=observation)
            context['place_plan'] = await self.call_algorithm(self.algorithms.plan_place, observation,
                self.state['target_id'], self.state['arm'], self.spatial_memory.scene, context['holding'],
                self.state['destination_id'], planning_settings=context['planning_settings'],
                placement_request=self._bind_region(context['observation']),
                placement_resolution=context['placement_resolution'], report_progress=True)
            binding.require_current(self.motion_plan_binding(context))
            context['place_plan']['input_binding'] = binding.model_dump()
            context['place_plan_observation_id'] = context['observation']['observation_id']
            context['place_plan_binding'] = {key: self.state[key] for key in ('target_id', 'destination_id', 'arm')}
            return context['place_plan']
        if skill == 'execute_place':
            expected = {key: self.state[key] for key in ('target_id', 'destination_id', 'arm')}
            if context.get('place_plan_binding') != expected:
                raise ValueError('place_plan_binding_changed')
        if skill == 'verify_unpack':
            result = await super().execute(skill, context, arguments)
            if result.get('success'):
                result['product_id'] = result['output_entity_id']
            return result
        return await super().execute(skill, context, arguments)

    async def _require_visibility_recovery(self, context):
        from ..agent.visibility_recovery import (
            visibility_allows_motion,
            visibility_assessment,
            visibility_requests,
        )

        capture_id = context.get('latest_capture', {}).get('observation_id')
        failure = public_summary(getattr(getattr(self, 'graph_executor', None), 'last_failure', None))
        arguments = (failure or {}).get('arguments', {})
        subject_id = (arguments.get('target_id') or arguments.get('destination_id')
                      or (failure or {}).get('target_id') or (failure or {}).get('destination_id'))
        subject = context.get('identity_reference', {}).get('entities', {}).get(subject_id, {})
        subject = {key: subject[key] for key in ('id', 'color', 'kind') if key in subject}
        if subject_id:
            subject['id'] = subject_id
        failed_checks = (failure or {}).get('details', {}).get('diagnostic_checks', [])
        required_question = next((check['question'] for check in failed_checks
            if check.get('status') in {'no', 'unknown'} and isinstance(check.get('question'), str)), None)
        proof = context.get('visibility_recovery', {})
        if (proof.get('observation_id') != capture_id or proof.get('epoch') != self.observation_epoch
                or proof.get('failed_result') != failure):
            result = await self._judge(context, visibility_requests(subject,
                failed_skill=(failure or {}).get('skill'), required_judgment=required_question), extra={
                'verification_scope': 'graph_recovery',
                'subject': subject,
                'required_judgment': required_question,
                'failed_result': failure,
                'failed_node': (failure or {}).get('node_id')})
            proof = {'observation_id': context['latest_capture']['observation_id'],
                'epoch': self.observation_epoch, 'failed_result': failure,
                'answer': visibility_assessment(result['judgments'])}
            context['visibility_recovery'] = proof
        self.emit('visibility_recovery_checked', '补观测移动的视觉原因', proof)
        if not visibility_allows_motion(proof.get('answer', {})):
            raise ValueError('viewpoint_requires_visual_limitation')
        context.pop('visibility_recovery', None)

    async def _judge(self, context, requests, *, extra=None, include_task=True, reference=None, verification_capture=None):
        capture = context['latest_capture'] if verification_capture is None else verification_capture
        if verification_capture is not None:
            if (capture.get('generation') != self.state['generation'] or capture.get('source') != 'gripper_wrist_rgbd'
                    or len(requests) != 1 or requests[0].name != 'object_held'):
                raise ValueError('invalid_wrist_judgment_scope')
        elif context.get('capture_epochs', {}).get(capture['observation_id']) != self.observation_epoch:
            raise ValueError('fresh_observation_required_before_visual_judgment')
        audit, done = self.visual_audit_sink()
        self.judgments.audit = audit
        try:
            judgment_context = ({'task': self.task_spec.model_dump(mode='json'),
                         'object_task': self.active_object_task.model_dump(mode='json') if self.active_object_task else None,
                         'environment_assumptions': copy.deepcopy(self.environment_assumptions),
                         'target_id': self.state.get('target_id'),
                         'destination_id': self.state.get('destination_id')} if include_task else {})
            judgment_context = {**judgment_context, **(extra or {})}
            observation = context.get('observation', {})
            scope = judgment_context.get('verification_scope')
            if scope in {'current_support', 'input_membership'} and observation.get('observation_id') == capture['observation_id']:
                identifiers = {judgment_context.get('subject', {}).get('id'),
                               *judgment_context.get('support_candidates', {})}
                masks = {camera: {'ID' + identifier: mask for identifier, mask in view.get('identity_masks', view.get('instance_masks', {})).items()
                                  if identifier in identifiers}
                         for camera, view in observation.get('views', {}).items()}
                judgment_context.update(visual_modalities=['rgb'], visual_masks=masks)
            elif (scope in {'graph_recovery', 'policy_condition', 'task_completion'}
                    and observation.get('observation_id') == capture['observation_id']):
                judgment_context.setdefault('visual_masks', {camera: view.get('identity_masks', view.get('instance_masks', {}))
                    for camera, view in observation.get('views', {}).items()})
            return await self.call_algorithm(self.judgments.judge, capture, requests=requests,
                context=judgment_context,
                reference=context['identity_reference'] if reference is None else reference)
        finally:
            done.set()

    async def _execute_graph(self, context):
        active_input = None

        async def enter_object_task(task):
            self.active_object_task = TaskSpec.model_validate(task) if task is not None else None
            self.state['active_object_task'] = copy.deepcopy(task)


        def confirmed_targets():
            completed = self.task_progress.completed_input_ids()
            return [identifier for identifier in self.task_spec.target_refs if identifier not in completed]

        async def before_target(identifier, index):
            nonlocal active_input
            if active_input is not None and active_input != identifier:
                shared_evidence = {key: self.state['facts'].get(key, False)
                                   for key in ('observed', 'observation_pose_ready', 'grasp_view_ready')}
                self.reset_transfer(context)
                self.state['facts'].update(shared_evidence)
            active_input = identifier

        async def after_target(identifier):
            completed = self.task_progress.completed_input_ids()
            if identifier not in completed:
                return {'status': 'unknown', 'success': False,
                        'reason': 'policy_target_not_verified:' + identifier}

        async def invoke(name, args):
            return await self._invoke_policy_skill(name, args, context)

        async def judge(request: GraphDecisionRequest):
            name, bindings = request.condition, request.bindings
            choices, branches = request.choices, request.branches
            declared = {(node.condition, tuple(node.choices), node.rule) for scope in [graph, *graph.subgraphs]
                        for node in scope.nodes if node.type == 'router'}
            if (name, tuple(choices), request.rule) not in declared:
                raise ValueError('undeclared_policy_judgment:' + name)
            from .decision_inputs import available_branch_choices
            available_choices = available_branch_choices(choices, branches) if choices else ()
            if choices and not any(choice != 'unknown' for choice in available_choices):
                return GraphDecisionResult(status='unknown', value='unknown', reason='no_available_declared_branch',
                    details={'producer': 'runtime_branch_availability', 'declared_branches': list(branches)})
            if request.rule == 'failure_return':
                return self._failure_return_judgment(context, available_choices)
            from .program_decisions import observation_evidence, observation_recovery
            if request.rule != 'visual':
                assessment = observation_evidence(context, self.state)
                at_pose = self.is_observation_pose(context)
                value = observation_recovery(available_choices, branches,
                    needs_viewpoint_change=assessment['needs_viewpoint_change'],
                    empty_hand=assessment['empty_hand'], at_observation_pose=at_pose)
                return GraphDecisionResult(status='unknown' if value == 'unknown' else 'proposed', value=value,
                    selection_mode='proposal' if value != 'unknown' else None,
                    reason='program_' + request.rule, details={'source': 'program', 'model_called': False,
                        'observation_assessment': assessment, 'already_at_observation_pose': at_pose})
            requests = [ChoiceRequest(name='condition', instructions=name, choices=available_choices,
                                      selection_mode='proposal') if choices else
                        JudgmentRequest(name='condition', instructions=name)]
            result = await self._judge(context, requests, extra={
                'verification_scope': 'policy_condition', 'policy_bindings': copy.deepcopy(bindings)})
            # The executor consumes this exact typed result. A bool/None return
            # discards the probability, abstention reason and failed question.
            return GraphDecisionResult(**result['judgments']['condition'], details=judgment_details(result, requests))

        graph = self.policy_graph
        completed = self.task_progress.completed_input_ids()
        pending = [identifier for identifier in self.task_progress.input_ids if identifier not in completed]
        repeatable = {name for name, contract in self.registry.contracts.items() if contract.retry_safe}

        def emit_graph(kind, title, data):
            if kind == 'policy_node_started':
                self.state['active_policy_node'] = copy.deepcopy(data)
            self.emit(kind, title, data)

        self.graph_executor = NativeGraphExecutor(invoke, target_ids=pending,
            resolve_target_ids=confirmed_targets,
            object_task_context=enter_object_task,
            bindings={'$source': self.task_bindings.get('source_id'),
                      '$destination': self.task_bindings.get('destination_id'),
                      '$work_area': self.task_bindings.get('work_area_id')},
            judge=judge, before_target=before_target, after_target=after_target,
            result_bindings={name: contract.result_bindings
                             for name, contract in self.registry.contracts.items()},
            is_target_pending=lambda identifier: identifier not in (
                self.task_progress.completed_input_ids()),
            check_cancelled=self.check_generation, emit=emit_graph, repeatable_skills=repeatable,
            step_limit=131072)
        await self.graph_executor.run(graph)

    async def _invoke_policy_skill(self, name, args, context):
        from ..core.outcome_evidence import outcome_state
        from ..core.outcomes import GraphSkillResult
        try:
            result = await self._step(name, context, args)
        except (ValueError, AlgorithmError, PerceptionError, TimeoutError) as error:
            result = skill_error_result(error, self.state, args)
        result = bind_failure_context(result, skill=name, arguments=args, state=self.state)
        outputs = {field: result[field] for field in self.registry.contracts[name].result_bindings.values()
                   if field in result}
        return GraphSkillResult(status=result_status(result), outputs=public_summary(outputs),
                                evidence=outcome_state(result))

    @staticmethod
    def _failure_return_judgment(context, choices):
        value = context.get('failure_return_mode', 'unknown')
        value = value if value in choices else 'unknown'
        return GraphDecisionResult(status='unknown' if value == 'unknown' else 'proposed', value=value,
            selection_mode=None if value == 'unknown' else 'proposal', reason='program_failure_return',
            details={'source': 'program', 'model_called': False})

    async def _execute_terminal_return(self, context):
        from .failure_return import terminal_graph

        graph = terminal_graph()
        self.state['terminal_recovery_graph'] = graph
        self.emit('terminal_recovery_started', '任务失败，执行统一回位收尾', {'graph': graph})
        async def invoke(name, arguments):
            return await self._invoke_policy_skill(name, arguments, context)
        async def judge(request):
            return self._failure_return_judgment(context, request.choices)
        executor = NativeGraphExecutor(invoke, target_ids=[], bindings={}, judge=judge,
            check_cancelled=self.check_generation, emit=self.emit,
            result_bindings={name: contract.result_bindings for name, contract in self.registry.contracts.items()})
        await executor.run(graph)

    async def _complete(self, context):
        from ..agent.completion import (
            completion_expected_status,
            completion_failures,
            completion_obligations,
            source_presence_request,
        )

        completed = self.task_progress.completed_input_ids()
        if (set(completed) != set(self.task_progress.input_ids)
                or self.task_spec.scope == 'ids' and set(completed) != set(self.task_spec.target_refs)
                or self.task_spec.quantity is not None and len(completed) != self.task_spec.quantity):
            raise ValueError('policy_finished_with_pending_targets')
        pose_proof = None
        if self.task_spec.completion_pose == 'observation':
            feedback = self.simulation.joint_feedback(self.state['generation'])
            if not self.is_observation_pose(context, feedback):
                raise DiagnosticError('completion_pose_unconfirmed', {
                    'component': 'robot_feedback', 'expected_pose': 'observation',
                    'actual_feedback': feedback})
            pose_proof = {'success': True, 'expected_pose': 'observation',
                          'source': 'actual_joint_feedback', 'actual_feedback': feedback}
        if self.task_spec.completion_pose == 'initial':
            from ..robot_skills.return_pose import initial_pose_verdict
            if context.get('holding') or self.holding_state is not None:
                raise DiagnosticError('completion_pose_unconfirmed', {'reason': 'empty_hands_unconfirmed'})
            pose_proof = initial_pose_verdict(context['initial_joint_reference'], self.simulation.joint_feedback(self.state['generation']))
            self.state['completion_pose_evidence'] = pose_proof
            self.state['facts']['at_initial_pose'] = bool(pose_proof['success'])
            if not pose_proof['success']:
                raise DiagnosticError('completion_pose_unconfirmed', pose_proof)
        self.state['completion_pose_evidence'] = copy.deepcopy(pose_proof)
        previous = self.state.get('completion_evidence', {}).get('judgments', {})
        mode = 'fresh' if completion_failures(previous) else 'reuse_current'
        if self.task_spec.scope == 'all' or self.task_spec.visual_completion_conditions or any(
                task.visual_completion_conditions for task in self.object_tasks.values()):
            capture = await self.capture_scene_observation(context, {'capture_mode': mode}, purpose='task_completion')
        else:
            capture = context.get('latest_capture', context['observation'])
        completion_key = (capture['observation_id'], self.observation_epoch, self.task_progress.ledger_revision)
        if (completion_failures(previous)
                and context.get('completion_request_key') == completion_key):
            raise ValueError('task_completion_unchanged_evidence')
        self.record_observation(capture, 'task_completion')
        requests = [JudgmentRequest(name=f'visual_completion_{index}', instructions=condition)
                    for index, condition in enumerate(self.task_spec.visual_completion_conditions)]
        condition_subjects = {}
        for item_index, identifier in enumerate(self.task_progress.input_ids):
            item_task = self.object_tasks.get(identifier)
            if item_task:
                condition_subjects.update({f'object_{item_index}_visual_{index}': {
                    'target_id': identifier, 'object_task': item_task.model_dump(mode='json')}
                    for index in range(len(item_task.visual_completion_conditions))})
                requests.extend(JudgmentRequest(name=f'object_{item_index}_visual_{index}', instructions=condition)
                    for index, condition in enumerate(item_task.visual_completion_conditions))
        presence = context.get('source_presence_evidence', {})
        reuse_presence = (self.task_spec.scope == 'all' and presence.get('observation_id') == capture['observation_id']
                         and presence.get('epoch') == self.observation_epoch)
        if self.task_spec.scope == 'all' and not reuse_presence:
            requests.append(source_presence_request())
        if requests:
            result = await self._judge(context, requests, extra={'verification_scope': 'task_completion',
                'verified_robot_pose': pose_proof, 'condition_subjects': condition_subjects})
        else:
            result = {'judgments': {}, 'evidence': {'source': 'verified_task_ledger_and_joint_feedback',
                      'model_called': False}, 'verified_obligations': completion_obligations(
                          self.task_progress.snapshot(), self.object_memory.snapshot())}
        if reuse_presence:
            requests.append(source_presence_request())
            result['judgments']['source_has_remaining_input'] = copy.deepcopy(
                presence['result']['judgments']['source_has_remaining_input'])
            result['reused_source_presence'] = copy.deepcopy(presence)
        self.state['completion_evidence'] = result
        context['completion_request_key'] = completion_key
        result['expected_status'] = {request.name: completion_expected_status(request.name) for request in requests}
        failed = completion_failures(result['judgments'])
        if failed:
            details = judgment_details(result, requests, failed_names=failed)
            for check in details['diagnostic_checks']:
                check['expected_status'] = completion_expected_status(check['name'])
            raise DiagnosticError('task_completion_unconfirmed', details)
        self.state['status'] = 'succeeded'

    async def _run(self, run_id, initial):
        from ..robot_skills.return_pose import initial_joint_reference
        context = {'initial': initial, 'planning_settings': copy.deepcopy(self.state['planning_settings']),
                   'initial_joint_reference': initial_joint_reference(initial)}
        from ..object_memory import ObjectMemory
        from ..spatial_memory import SpatialMemory
        self.object_memory = ObjectMemory()
        self.spatial_memory = SpatialMemory()
        self.geometry_measurements = {}
        self.planning_regions = {}
        self.task_spec = self.task_bindings = self.policy_graph = None
        self.object_tasks = {}
        self.active_object_task = None
        self.observation_epoch = 0
        self.state.pop('partial_execution_failure', None)
        for key in ('failure_report', 'policy_failure', 'error_details', 'error_code', 'failed_step', 'failure_return'):
            self.state.pop(key, None)
        self.compiler.reset()
        self.task_progress = PolicyTaskProgress(self.state['instruction'], 'collection', None,
                                                selection_order='visual')
        self.state.update(task='grasp_place', controller='responses_policy_graph',
                          requested_arm=None, attributes={'kind': 'any', 'color': 'any'},
                          destination_attributes={'kind': 'any', 'color': 'any'})
        self.state['facts'].update(needs_place=True, collection_task=True, center_reachability_enabled=True,
                                   objects_ready=False, collision_map_initialized=False,
                                   observed=False, task_verified=False, terminal_motion_stopped=False)
        self.emit('run_start', '执行在线策略图', {'instruction': self.state['instruction']})
        try:
            self.begin_step('initialize_task_geometry')
            self.check_generation()
            cleared = await self.call_algorithm(self.algorithms.clear_carrying)
            self.check_generation()
            if cleared.get('ok') is not True or cleared.get('attached') is not False:
                raise DiagnosticError('task_geometry_reset_unconfirmed', {
                    'stage': 'initialize_task_geometry',
                    'diagnostic_checks': [{'name': 'clear_previous_task_geometry', 'actual': cleared,
                                          'expected': {'ok': True, 'attached': False}}]})
            self.state['facts'].update(observation_pose_ready=self.is_observation_pose(context, initial),
                                       grasp_view_ready=False)
            self.begin_step('compile_task')
            await self._bootstrap(context)
            await self._execute_graph(context)
            if (not self.state['facts'].get('task_verified')
                    or context.get('completion_verification_epoch') != self.observation_epoch
                    or context.get('completion_verification_ledger_revision') != self.task_progress.ledger_revision):
                raise ValueError('policy_finished_without_current_completion_verification')
            self.state['status'] = 'succeeded'
        except asyncio.CancelledError:
            self.state['status'] = 'cancelled'
            raise
        except Exception as error:  # noqa: BLE001 -- task boundary records unexpected failures and stops motion
            record = failure_record(error, self.state)
            primary = next(iter(getattr(getattr(self, 'graph_executor', None), 'unresolved_failures', {}).values()), None)
            primary = primary or context.get('terminal_original_failure')
            if primary and primary.get('error_code'):
                from .diagnostics import build_failure_report, result_details
                details = result_details(primary)
                cause = primary['error_code']
                if (cause != record['error_code'] or primary.get('skill') != record['failure_report'].get('skill')
                        or (details.get('message') and details['message'] != record['error_details'].get('message'))):
                    followup = record['failure_report']
                    record.update(error_code=cause, error_details=details, failure_report=build_failure_report(
                        cause, details, {**self.state, 'current_step': {'skill': primary.get('skill')}}))
                    record['failure_report']['subsequent_failure'] = followup
                record['failure_report']['original_task_failure'] = copy.deepcopy(primary)
            elif primary:
                record['failure_report']['original_task_failure'] = copy.deepcopy(primary)
            failed_step = copy.deepcopy((primary or {}).get('partial_execution') or context.get('terminal_failed_step')
                                        or self.state.get('current_step'))
            if primary and primary.get('skill') and (failed_step or {}).get('skill') != primary['skill']:
                failed_step = {k: primary[k] for k in ('skill', 'invocation_id', 'node_id') if primary.get(k)}
            report = record['failure_report']
            message = record['error_details'].get('message')
            validation = record['error_details'].get('validation_errors') or []
            if not message and validation:
                first = validation[0]
                message = first.get('message') if isinstance(first, dict) else str(first)
            parts = [report.get('skill') or report.get('stage'), record['error_code'], message]
            display_error = ' · '.join(dict.fromkeys(str(part) for part in parts if part))
            self.state.update(status='needs_attention', error=display_error[:1000], **record,
                              failed_step=failed_step)
            if not context.get('failure_return_attempted'):
                context['terminal_original_failure'] = copy.deepcopy(record['failure_report'])
                try:
                    await self._execute_terminal_return(context)
                except Exception as return_error:  # noqa: BLE001 -- terminal recovery cannot replace the task failure
                    self.state['failure_return'] = {**self.state.get('failure_return', {}),
                        'status': 'blocked', 'success': False, 'task_succeeded': False,
                        'reason': str(return_error)[:1000], 'original_failure': context['terminal_original_failure']}
                    await asyncio.to_thread(self.simulation.stop_motion)
            elif self.state.get('failure_return', {}).get('status') not in {'returned', 'already_at_observation'}:
                self.state['failure_return'] = {**self.state.get('failure_return', {}),
                    'status': 'blocked', 'success': False, 'task_succeeded': False,
                    'reason': str(error)[:1000]}
                await asyncio.to_thread(self.simulation.stop_motion)
            self.state.update(status='needs_attention', failed_step=failed_step, **record)
            memory = getattr(self.compiler, 'compilation_memory', None)
            if memory is not None:
                try:
                    memory.record_execution_failure(catalog=skill_catalog(self.registry), run_id=self.run_id,
                                                    report=record['failure_report'])
                except Exception as memory_error:  # noqa: BLE001 -- preserve the original task failure and its evidence
                    self.emit('compilation_memory_error', '执行经验写入失败', {'error': str(memory_error)})
            self.emit('skill_error', '策略执行需要处理', {'code': record['error_code'],
                      'details': record['error_details'], 'failure_report': record['failure_report']})
        finally:
            self.state['task_progress'] = execution_snapshot(
                self.task_progress, self.object_memory, self.geometry_measurements, self.planning_regions)
            self.state['current_step'] = None
            memory_calls = sum(event['kind'] == 'visual_decisions_call'
                and event.get('data', {}).get('purpose') == 'update_object_memory' for event in self.events)
            self.state['metrics'] = {'run_duration_s': time.monotonic() - self.run_started_at,
                'responses_call_count': self.compiler.budget.used + memory_calls,
                'compile_responses_call_count': self.compiler.budget.used,
                'objects_responses_call_count': memory_calls}
            self.emit('run_end', '任务结束', self.state)
            if self.experiment is not None:
                self.experiment.stop(self.state)

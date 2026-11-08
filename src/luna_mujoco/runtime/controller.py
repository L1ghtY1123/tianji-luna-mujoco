"""Compose model planning, evidence judgments and registered robot operations."""
import asyncio
import copy
import json
import os
import time
from functools import partial
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agent.decisions import DecisionsJudgments
from ..agent.inventory import InventoryRegistrar
from ..agent.responses import ResponsesBudget, ResponsesCompiler
from ..agent.schema import (
    Graph,
    TaskSpec,
    object_task_specs,
    skill_catalog,
    validate_graph,
)
from ..algorithms import AlgorithmError
from ..compilation_memory import CompilationMemory
from ..core.decisions import ChoiceRequest, GraphDecisionRequest, GraphDecisionResult, JudgmentRequest
from ..core.placement import ResolvePlacementArguments
from ..inventory_order import robot_distance_order
from ..inventory_reference import bound_identity_coverage_source, current_inventory_entities
from ..perception import PerceptionError
from ..placement import observed_object, public_observation
from ..robot_skills.container_geometry import container_transport_region
from ..robot_skills.operations import DESTINATION_ID, RobotOperations, configured_destination
from ..robot_skills.placement_region import measured_table_region
from ..robot_skills.placement_zone import measured_object_region
from ..robot_skills.unpack import VLAUnpack
from ..runner import Runner, public_summary
from ..skills import NoArguments
from ..visual_api import VisualAPI
from ..vla_configuration import vla_unpack_configuration
from ..world_inventory import WorldInventory
from .decision_inputs import destination_reference_evidence
from .diagnostics import DiagnosticError, failure_record, judgment_details, skill_error_result
from .execution_state import bind_failure_context, result_status
from .inventory import PolicyTaskProgress
from .memory import execution_snapshot
from .native_graph import NativeGraphExecutor
from .placement import (
    candidate_capture_rejection,
    capture_candidate_view,
    invalidate_placement,
    perceive_for_controller,
    resolve_xy_then_z,
)
from .support import support_context, support_memory


class TargetArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    target_id: str


class DestinationArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    destination_id: str
    destination_kind: Literal['support_region', 'container']
    release_mode: Literal['surface_setdown', 'container_release'] = Field(description=(
        'Choose surface_setdown for gentle placement near a support, including a container floor. '
        'Choose container_release only when the task permits dropping into an open container from '
        'just above its rim. This semantic choice is bound before grasp screening; numerical height '
        'comes from operator configuration. Never use container_release for a support_region.'))
    placement_mode: Literal['free_region', 'configured_xy'] = Field(default='free_region', description=(
        'Use configured_xy when the user names the workbench designated/set/configured area: '
        'use its configured XY exactly, with no XY search. Use free_region for an ordinary '
        'surface/container area without a specified point. No numerical coordinates are generated here.'))
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


class ObservationMotionArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    viewpoint: Literal['configured', 'wrist_tilt_positive', 'wrist_tilt_negative'] = Field(
        default='configured', description=(
            'configured returns to the operator observation pose and can satisfy the required final '
            'return. The two wrist_tilt alternatives offset both sixth joints by +12 or -12 degrees from '
            'that pose to obtain different wrist-camera views. They are movement proposals, not certified '
            'visible viewpoints; current-map cuRobo planning and execution checks are mandatory. '
            'After the selected movement, observe_scene analyzes the new capture.'))


class BuildSceneArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    scope: Literal['decide', 'global', 'local'] = 'decide'
    reason: str = ''
    entity_ids: list[str] = Field(default_factory=list, description=(
        'Local CAD/OBB object IDs to refresh. Declared table IDs may accompany these objects; '
        'measured table cuboids use the separate support input and are always included. '
        'A nonempty local selection must contain at least one CAD/OBB object. '
        'An empty list with scope=local uses the currently selected target_id. '
        'Global uses an empty list. Region names and marker IDs are not collision entities.'))
    exclusion_state: Literal['selected', 'observation'] = 'selected'

    @model_validator(mode='after')
    def coherent_scope(self):
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError('duplicate_mapping_entity')
        if self.scope == 'global' and self.entity_ids:
            raise ValueError('global_mapping_cannot_restrict_entities')
        return self


class PolicyController(RobotOperations):
    """The composition root owns model calls; skills receive only bound state."""

    def __init__(self, *args, compiler=None, judgments=None, known_assets=None,
                 environment_assumptions=None, world_inventory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment_assumptions = copy.deepcopy(environment_assumptions or {})
        schemas = {'identify_target': TargetArguments, 'identify_destination': DestinationArguments,
                   'choose_arm': ArmArguments, 'build_scene': BuildSceneArguments,
                   'prepare_unpack': PrepareUnpackArguments, 'observe_scene': ObserveArguments,
                   'verify_place': VerifyPlaceArguments,
                   'move_to_observe': ObservationMotionArguments,
                   'resolve_placement': ResolvePlacementArguments}
        names = ('move_to_observe', 'return_to_initial_pose', 'observe_scene', 'select_target', 'identify_target', 'identify_destination',
                 'choose_arm', 'build_scene', 'generate_grasps', 'plan_grasp', 'execute_grasp', 'recover_failed_grasp',
                 'verify_grasp', 'perceive_placement_zone', 'resolve_placement',
                 'plan_place', 'execute_place', 'verify_place', 'prepare_unpack',
                 'vla_unpack', 'verify_unpack',
                 'verify_task_completion')
        for name in names:
            self.registry.register(name, schemas.get(name, NoArguments), partial(self._dispatch, name))
        config = VisualAPI.from_environment()
        self.judgments = judgments or DecisionsJudgments(api_key=config.api_key, timeout_s=config.timeout_s)
        self.world_inventory = world_inventory or WorldInventory(self.data_dir.parent / 'world-inventory.sqlite3')
        self.world_namespace = self.simulation.world_id
        self.inventory_registrar = InventoryRegistrar(self.world_inventory, self.judgments)
        self.algorithms.perception.visual_inventory.register_instance = self.inventory_registrar.register_observed
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

    async def capture_observation(self, context, *, capture=None, recovery=False):
        epochs = context.setdefault('capture_epochs', {})
        if capture is not None and epochs.get(capture['observation_id'], 0) != self.observation_epoch:
            raise ValueError('fresh_observation_required_after_action')
        result = await super().capture_observation(context, capture=capture, recovery=recovery)
        epochs[result['observation_id']] = self.observation_epoch
        context['latest_capture'] = result
        return result

    async def capture_scene_observation(self, context, arguments, *, purpose='associate_inventory'):
        from ..robot_skills.observation import capture_reuse_rejection

        mode = ObserveArguments.model_validate(arguments).capture_mode
        capture = context.get('latest_capture')
        identifier = (capture or {}).get('observation_id')
        initial_reference = bool(identifier and not context.get('analyzed_capture_ids')
            and identifier == context.get('initial_capture', {}).get('observation_id')
            == context.get('visual_inventory_reference', {}).get('observation_id'))
        reason = 'model_requested_fresh_capture'
        if capture is not None:
            reason = capture_reuse_rejection(capture, self.simulation.robot_snapshot(),
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
        from ..robot_skills.return_pose import observation_motion_scene_ready

        self.check_generation()
        self.check_log()
        self.state['facts']['observation_motion_scene_ready'] = observation_motion_scene_ready(
            context, collision=self.spatial_memory.collision_snapshot())
        if name == 'move_to_observe':
            configured = (arguments or {}).get('viewpoint', 'configured') == 'configured'
            # The authored recovery may return to the configured observation pose
            # without another semantic permission check. Physical prerequisites remain.
            if not configured:
                await self._require_visibility_recovery(context)
        self.begin_step(name)
        if self.state.get('partial_execution_failure'):
            self.state['facts']['recovery_needed'] = True
            if self.registry.contracts[name].invalidates_observation:
                self._require_execution_recovery()
        started = time.monotonic()
        self.emit('skill_start', name, {'arguments': arguments or {}, 'target_id': self.state.get('target_id')})
        effects = self.registry.contracts[name].provides
        if self.registry.contracts[name].invalidates_observation:
            self.observation_epoch += 1
        try:
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
            if self.task_progress is not None:
                self._persist_inventory(context, name, failure)
            raise
        finally:
            self.state['facts']['observation_motion_scene_ready'] = observation_motion_scene_ready(
                context, collision=self.spatial_memory.collision_snapshot())
            if self.registry.contracts[name].invalidates_observation:
                # A new raw result photograph is not an ID-associated inventory
                # observation. Successful provides below may explicitly restore it.
                self.state['facts']['observed'] = False
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
        if self.task_progress is not None:
            if name != 'observe_scene' or status != 'success':
                self._persist_inventory(context, name, result)
            self.sync_inventory(context)
            if name.startswith('verify_') or name == 'observe_scene':
                calls = [event for event in self.events if event['kind'] == 'visual_decisions_call'
                         and event.get('invocation_id') == self.state['current_step']['invocation_id']]
                self.emit('inventory_snapshot', '时间记忆数据库快照', {
                    'checkpoint': name, 'task_progress': self.state['task_progress'],
                    'temporal_memory': self.object_memory.snapshot(),
                    'model_event_ids': [event['id'] for event in calls],
                    'evidence_source': result.get('container_entry', {}).get('source') or (
                        'robot_feedback' if name == 'verify_grasp' else 'visual_checkpoint')})
        return result

    def record_object_operation(self, context, operation):
        """Commit at the physical checkpoint, before any later stage can fail."""
        proposed = copy.deepcopy(self.object_memory)
        proposed.record_operation(operation)
        from ..memory_contracts import ObjectRecord

        record = ObjectRecord.model_validate(proposed.entities()[operation.entity_id])
        self.world_inventory.record(operation.entity_id, operation.event_key, operation.kind,
            record.model_dump(mode='json', exclude={'id'}), operation.evidence)
        self.object_memory = proposed
        if self.task_progress is not None:
            self.sync_inventory(context)

    def record_object_observation(self, context, observation):
        """Persist accepted measurements before exposing either memory's new version."""
        from .memory import geometry_update, merge_geometry_measurements, observation_update

        capture = context['latest_capture']
        if any(observation.get(key) != capture.get(key) for key in ('observation_id', 'generation')):
            raise ValueError('observation_commit_capture_mismatch')
        proposed_memory = copy.deepcopy(self.object_memory)
        proposed_memory.observe(observation_update(observation, proposed_memory.entities()))
        proposed_geometry = merge_geometry_measurements(self.geometry_measurements, geometry_update(observation))
        self._persist_inventory({**context, 'observation': observation}, 'observe_scene',
            {'status': 'success'}, source_memory=proposed_memory)
        if proposed_geometry != self.geometry_measurements:
            self.spatial_memory.invalidate([])
        self.geometry_measurements = proposed_geometry

    def _persist_inventory(self, context, skill, result, *, source_memory=None):
        """Record dated evidence without deciding graph control or task completion."""
        import numpy as np

        from ..visual_inventory import mask_box

        source_memory = self.object_memory if source_memory is None else source_memory
        memory = source_memory.entities()
        observation = context.get('observation', {})
        capture = context.get('latest_capture', {})
        operation_observation_id = ((result.get('container_entry') or result.get('visual_evidence', {})).get('observation_id')
            if skill == 'verify_place' else None) or observation.get('observation_id')
        observed = skill == 'observe_scene' and result_status(result) == 'success'
        identifiers = {key for key, entity in memory.items() if entity['status'] == 'active'} if observed else {
            self.state.get('target_id'), result.get('selected_id'), result.get('input_entity_id'), result.get('output_entity_id')}
        if skill == 'select_target':
            identifiers = {result.get('selected_id')}
        if observed:
            self.world_inventory.store_capture(capture)
            self.world_inventory.store_instance_masks(capture, observation)
        proposed_memory = copy.deepcopy(source_memory)
        events = []
        for identifier in identifiers & memory.keys():
            entity = proposed_memory.entities()[identifier]
            if observed:
                masks = [(int(np.count_nonzero(mask)), camera, mask)
                    for camera, view in observation.get('views', {}).items()
                    if (mask := view.get('instance_masks', {}).get(identifier)) is not None and np.any(mask)]
                if masks:
                    _, camera, mask = max(masks, key=lambda row: row[0])
                    entity.update(reference_observation_id=observation['observation_id'],
                        reference_camera_id=camera, bbox=mask_box(mask), identity_source='openai_decisions',
                        identity_request_hash=observation['visual_inventory']['evidence']['request_hash'])
            from ..memory_contracts import ObjectOperation, ObjectRecord
            proposed_memory.update(ObjectRecord.model_validate(entity))
            event = self.run_id + ':' + self.state['current_step']['invocation_id']
            if observed:
                event += ':observation'
            evidence = {
                'run_id': self.run_id, 'observation_id': operation_observation_id,
                'status': result_status(result), 'reason': result.get('reason'),
                'step': copy.deepcopy(self.state['current_step'])}
            from .memory import release_memory_changes

            memory_result = result
            if skill == 'verify_place' and context.get('container_entry_receipt'):
                memory_result = {**result, 'container_entry': context['container_entry_receipt']}
            changes = (release_memory_changes(entity, context.get('release_execution_evidence', {}),
                context.get('release_verification_attempt', {}), memory_result) if skill == 'verify_place' else {})
            operation = ObjectOperation(
                entity_id=identifier, event_key=event, kind=skill,
                observation_id=operation_observation_id, evidence=evidence,
                **changes)
            proposed_memory.record_operation(operation)
            current = ObjectRecord.model_validate(proposed_memory.entities()[identifier]).model_dump(mode='json')
            events.append((identifier, operation.event_key, operation.kind,
                           {key: value for key, value in current.items() if key != 'id'}, operation.evidence))
        self.world_inventory.record_many(events)
        self.object_memory = proposed_memory

    async def _bootstrap(self, context):
        self.inventory_registrar.world_key = self.world_namespace + ':' + str(self.state['generation'])
        self.compiler.scene_context['grasp_parameter_defaults'] = {
            key: context['planning_settings'][key] for key in
            ('grasp_insertion_m', 'grasp_maximum_tilt_deg', 'grasp_require_away_from_robot')}
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
        capture = await self.capture_observation(context)
        at_observation_pose = self.is_observation_pose(context, capture)
        self.state['facts']['observation_pose_ready'] = at_observation_pose
        self.compiler.initial_facts = {
            **getattr(self.compiler, 'initial_facts', {}), 'observation_pose_ready': at_observation_pose,
            'collision_map_initialized': self.spatial_memory.initialized, 'at_initial_pose': True}
        self.record_observation(capture, 'initial')
        audit, done = self.visual_audit_sink()
        self.compiler.audit = audit
        try:
            self.compiler.memory_reference = self.inventory_registrar.reference()
            plan = await self.call_algorithm(self.compiler.compile, capture, instruction=self.state['instruction'])
            if not plan.task.supported:
                raise ValueError('task_capability_missing:' + plan.task.reason)
            proposal_observations = await self.call_algorithm(
                self.algorithms.perception.prepare_proposal_observations,
                capture, [entity.model_dump(mode='json') for entity in plan.entities])
            self.inventory_registrar.audit = audit
            registration = await self.call_algorithm(self.inventory_registrar.bind_proposals,
                plan.entities, capture, instruction=self.state['instruction'],
                proposal_observations=proposal_observations)
            from .compiled_task import bind_compiled_task
            proposal = bind_compiled_task(plan, registration, self.inventory_registrar.entities())
        finally:
            done.set()
        self.task_spec = TaskSpec.model_validate(proposal['task_spec'])
        self.task_inventory = copy.deepcopy(proposal['task_inventory'])
        self.policy_graph = Graph.model_validate(proposal['policy_graph'])
        self.object_tasks = object_task_specs(self.policy_graph)
        self.active_object_task = None
        validate_graph(self.policy_graph, self.registry)
        from ..object_memory import restored_object_record
        stored = self.world_inventory.entities(world_key=self.inventory_registrar.world_key)
        self.object_memory.restore(tuple(restored_object_record(entity) for entity in stored))
        self.object_memory.restore(tuple(restored_object_record(entity) for entity in proposal['entities'].values()))
        reference = {**self.world_inventory.reference(self.object_memory.entities()),
            'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
            'views': {camera: {'rgb': view['rgb']} for camera, view in capture['views'].items()},
            'unresolved_proposals': copy.deepcopy(proposal['unresolved_proposals']),
            'inventory_state': self.object_memory.snapshot()}
        reference['owner_id'] = self.run_id
        reference['identity_references'].update(registration.get('identity_references', {}))
        from ..observation_labels import responses_reference_labels

        self.record_annotation(capture, responses_reference_labels(capture, reference))
        context.update(initial_capture=capture, latest_capture=capture, fine_capture=capture,
                       visual_inventory_reference=reference)
        self.object_memory.observation_id = capture['observation_id']
        self.task_progress.register_entities(self.object_memory.entities(),
            {key: row.get('visual_membership', 'unknown') for key, row in proposal['entities'].items()})
        self.task_progress.configure_task(self.task_spec, [], object_tasks=self.object_tasks)
        self.state.update(task_spec=self.task_spec.model_dump(mode='json'),
                          policy_graph=self.policy_graph.model_dump(mode='json'))
        self.emit('skill_catalog', '本次 graph 使用的 skills 契约', {'skills': skill_catalog(self.registry)})
        self.emit('skill_program', 'Responses 已校验执行 graph', self.state['policy_graph'])

    async def observe(self, context, *, checkpoint=None, capture=None, camera_ids=None,
                      purpose=None, recovery=False):
        checkpoint = checkpoint or 'initial'
        capture = await self.capture_observation(context, capture=capture, recovery=recovery)
        from ..robot_skills.placement_resolution import digest
        from .memory import perception_motion_context
        from .observation_inputs import observation_inputs, observation_identity_scope

        context['association_scope'] = observation_identity_scope(
            self.object_memory, self.spatial_memory, self.task_progress, self.task_spec,
            target_id=self.state.get('target_id'))

        def analysis_key():
            inputs = observation_inputs(context)
            return digest({'capture_id': capture['observation_id'], 'checkpoint': checkpoint, 'inputs': inputs,
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
            observation = await Runner.observe(self, context, checkpoint=checkpoint, capture=capture,
                camera_ids=['1', '2', '3'], purpose='geometry', recovery=recovery)
        except (ValueError, RuntimeError) as error:
            if any(word in str(error) for word in ('unknown', 'uncertain', 'identity_unresolved')):
                context['_observation_analysis'] = {'key': analysis_key(), 'failure': str(error)}
            raise
        # A configured work area is an explicit prior, not a visually invented marker.
        for identifier in self.compiler.configured_regions:
            configuration = self.compiler.configured_regions[identifier]
            region = configured_destination(observation, configuration.get('region', self.region),
                identifier=identifier, support_id=self.task_inventory.get('support_id'))
            observation['objects'] = [obj for obj in observation['objects'] if obj['id'] != identifier]
            observation['objects'].append(region)
            self.planning_regions[identifier] = copy.deepcopy(region)
        context.update(observation=observation, inventory_observation=copy.deepcopy(observation))
        self.sync_inventory(context)
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
                and previous.get('source') == 'openai_decisions' and previous.get('request_hash')
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
                        'If CURRENT evidence shows a different support, answer no to this hypothesis.'}
        from ..agent.support import support_question
        requests = [support_question(entity, candidates, proposed_support)]
        # Dated verified relations are memory, while old free-text descriptions
        # and requested destinations are not current support evidence.
        reference = copy.deepcopy(context.get('visual_inventory_reference', {}))
        reference.setdefault('entities', {identifier: measured['subject'], **measured['support_candidates']})
        for reference_entity in reference['entities'].values():
            reference_entity.pop('description', None)
            reference_entity.pop('initial_parent_id', None)
        result = await self._judge(context, requests, extra=measured, include_task=False, reference=reference)
        judgment = result['judgments']['current_support']
        if proposed_support in candidates and judgment['status'] == 'no':
            # A false hypothesis rules out that support, not all other supports.
            # Ask one different selection question; never retry this predicate until yes.
            remaining = {key: value for key, value in candidates.items() if key != proposed_support}
            measured['rejected_support_hypothesis'] = {
                'support_entity_id': proposed_support, 'observation_id': capture['observation_id'],
                'request_hash': result['evidence']['request_hash'], 'probability': judgment.get('probability')}
            measured.pop('support_hypothesis', None)
            measured['support_candidates'] = {key: value for key, value in measured['support_candidates'].items()
                                              if key in remaining}
            proposed_support = None
            requests = [support_question(entity, remaining)]
            result = await self._judge(context, requests, extra=measured, include_task=False, reference=reference)
            judgment = result['judgments']['current_support']
        if proposed_support in candidates:
            support_id = proposed_support if judgment['status'] == 'yes' else None
            score = {'probability': judgment.get('probability')}
        else:
            support_id = judgment.get('value') if judgment['status'] == 'known' else None
            score = {'confidence': judgment.get('confidence')}
        if support_id not in candidates:
            rejected = judgment['status'] == 'no' or judgment.get('value') == 'none_of_listed'
            raise DiagnosticError('observed_support_rejected' if rejected else 'observed_support_unknown',
                                  judgment_details(result, requests))
        proof = {'support_entity_id': support_id, 'observation_id': capture['observation_id'],
                 'generation': capture.get('generation'), 'source': 'openai_decisions',
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
        for snapshot in (context.get('observation', {}), context.get('inventory_observation', {})):
            if snapshot.get('observation_id') != capture['observation_id']:
                continue
            for records in (snapshot.get('scene_entities', {}), snapshot.get('inventory_state', {}).get('entities', {})):
                if identifier in records:
                    records[identifier]['observed_support'] = copy.deepcopy(proof)
        self.sync_inventory(context)
        self.emit('observed_support_refreshed', 'Decisions 确认当前支撑关系', {'target_id': identifier, **proof})

    def sync_inventory(self, context):
        inventory = self.object_memory.snapshot()
        self.task_inventory['selected_ids'] = list(self.task_progress.input_ids)
        self.state['task_inventory'] = copy.deepcopy(self.task_inventory)
        self.state['task_progress'] = execution_snapshot(
            self.task_progress, self.object_memory, self.geometry_measurements, self.planning_regions)
        reference = context.get('visual_inventory_reference')
        if reference is not None:
            reference['inventory_state'] = inventory
            reference['entities'] = current_inventory_entities(reference)
            reference['identity_references'] = self.world_inventory.reference(reference['entities'])['identity_references']

    def _check_source_view(self, observation):
        """Validate only the identities attached to current measured geometry."""
        visual = observation.get('visual_inventory', {})
        coverage = visual.get('identity_coverage', {})
        evidence = visual.get('evidence', {})
        measured = {item['id'] for item in observation['objects']
                    if item.get('kind') not in {'table', 'marker'}}
        threshold = coverage.get('minimum_confidence')
        if (not bound_identity_coverage_source(observation)
                or coverage.get('observation_id') != observation['observation_id']
                or not coverage.get('request_hash')
                or coverage['request_hash'] != evidence.get('request_hash')
                or type(threshold) not in (int, float) or not .8 <= threshold <= 1.
                or not measured.issubset(coverage.get('accepted_entity_ids', []))):
            raise DiagnosticError('source_inventory_unknown', {
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
            if cached.get('observation_id') != observation['observation_id']:
                result = await self._judge(context, [JudgmentRequest(name='input_matches_task',
                    instructions='The current identified physical instance subject.id is an eligible input '
                    'requested by task, including its input description and source restriction. '
                    'Judge present task membership only. Use current appearance, source relation and '
                    'recorded processing history. Task completion and future feasibility are separate.')],
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
        if self.task_spec.selection_order == 'nearest':
            ordered = robot_distance_order(candidates)
            by_id = {obj['id']: obj for obj in candidates}
            candidates = [by_id[row['id']] for row in ordered]
        elif self.task_spec.scope == 'ids':
            ranks = {identifier: index for index, identifier in enumerate(self.task_spec.target_refs)}
            candidates.sort(key=lambda obj: ranks[obj['id']])
        if candidates:
            selected, evidence = candidates[0], []
            if len(candidates) > 1 and self.task_spec.selection_order == 'explicit' and self.task_spec.scope != 'ids':
                measured = {obj['id']: obj for obj in public_observation(observation)['objects']}
                # Keep every eligible candidate in consideration within the single-question choice limit.
                for offset in range(0, len(candidates), 62):
                    batch = candidates[offset:offset + 62]
                    if offset:
                        batch = [selected, *batch]
                    options = {obj['id']: obj for obj in batch}
                    requests = [ChoiceRequest(name='next_input', selection_mode='proposal',
                        choices=(*options, 'unknown'), instructions=(
                            'Choose the next eligible input to handle from selection_candidates. Honor the user task '
                            'and any stated ordering. Otherwise prefer an exposed grasping surface, less occlusion '
                            'and more clearance from neighbors or container walls in CURRENT RGB and measured geometry. '
                            'Compare candidate positions with the supplied support/container region polygons. '
                            'For otherwise similar inputs, favor room for the gripper around the object. '
                            'Use the permanent ID labels to identify candidates. The array order has no priority. '
                            'This is an operation proposal. Numerical grasp, collision and trajectory checks follow. '
                            'Choose unknown when the required ordering or candidate binding cannot be resolved.'))]
                    judgment = await self._judge(context, requests, extra={
                        'verification_scope': 'next_input_selection',
                        'selection_support_geometry': [obj for obj in measured.values()
                                                       if obj.get('kind') in {'bin', 'table'}],
                        'selection_candidates': [measured[identifier] for identifier in options]})
                    identifier = judgment['judgments']['next_input'].get('value')
                    if identifier not in options:
                        return {'status': 'unknown', 'success': False, 'reason': 'target_selection_unknown',
                                'details': judgment_details(judgment, requests)}
                    selected = options[identifier]
                    evidence.append({'request_hash': judgment['evidence']['request_hash'],
                                     'choice': copy.deepcopy(judgment['judgments']['next_input'])})
            return {'status': 'success', 'success': True, 'selected_id': selected['id'],
                    'observation_id': observation['observation_id'], 'selection_evidence': evidence}
        unresolved = observation.get('visual_inventory', {}).get('unassociated_candidates', [])
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
                    'reason': 'remaining_input_unlocalized' if answer == 'yes' else 'source_inventory_unknown',
                    'details': {**details, **judgment_details(presence, requests)}}
            return {'status': 'empty', 'success': False, 'reason': 'source_confirmed_empty',
                    'observation_id': observation['observation_id'], 'source_presence': presence}
        return {'status': 'empty', 'success': False, 'reason': 'no_current_eligible_input',
                'observation_id': observation['observation_id'], 'details': details}

    async def build_scene(self, context, exclusion_state):
        # Only a verified transformation can retire an input identity.
        if self.task_progress is None:
            return await super().build_scene(context, exclusion_state)
        retired = self.object_memory.snapshot()['retired_entity_ids']
        self.spatial_memory.require_entities(self._collision_entity_ids(context), retired)
        return await super().build_scene(context, exclusion_state)

    def _collision_entity_ids(self, context):
        """Collect measured, retained and action-affected collision bodies."""
        entities = self.object_memory.entities()
        identifiers = {obj['id'] for obj in context.get('observation', {}).get('objects', [])}
        identifiers.update((self.spatial_memory.scene or {}).get('inventory_map', {}).get('entities', {}))
        identifiers.update(self.spatial_memory.lifecycle().get('invalidated_entity_ids', []))
        identifiers.update((self.state.get('target_id'), self.state.get('destination_id')))
        return {identifier for identifier in identifiers if identifier in entities
                and entities[identifier].get('kind') not in {'table', 'marker'}}

    async def _mapping_strategy(self, context, arguments):
        request = BuildSceneArguments.model_validate(arguments)
        inventory = self.object_memory.entities()
        retired = set(self.object_memory.snapshot()['retired_entity_ids'])
        known = {key: value for key, value in inventory.items()
                 if key not in retired and value.get('kind') not in {'marker', 'table'}}
        tables = {key for key, value in inventory.items()
                  if key not in retired and value.get('kind') == 'table'}
        observed_ids = {item['id'] for item in context['observation']['objects']}
        unavailable = set(request.entity_ids) - set(known) - tables
        if unavailable:
            raise DiagnosticError('mapping_entity_not_in_inventory', {
                'requested_entity_ids': request.entity_ids,
                'unavailable_entity_ids': sorted(unavailable),
                'retired_entity_ids': sorted(unavailable & retired),
                'target_id': self.state.get('target_id')})
        # Table cuboids are already carried by the measured support input; the
        # per-instance layer selector addresses only CAD/OBB objects. Preserve
        # the declared support choice in the audit reason, never as an object mask.
        selected_tables = sorted(set(request.entity_ids) & tables)
        entities = [identifier for identifier in request.entity_ids if identifier not in tables]
        if request.scope == 'local' and selected_tables and not entities:
            raise ValueError('local_mapping_requires_object_entities')
        scope, source = request.scope, self.state.get('active_policy_node', {}).get('author', 'responses')
        if scope == 'decide' and not self.spatial_memory.initialized:
            # Before a base exists, global is the sole permitted scope.
            scope, source = 'global', 'sole_permitted_scope'
        if scope == 'local' and not entities:
            target = self.state.get('target_id')
            if target not in known:
                raise DiagnosticError('mapping_entity_not_in_inventory', {
                    'target_id': target, 'reason': 'local_mapping_requires_current_target_or_explicit_entities'})
            entities = [target]
        decision = None
        if scope == 'decide':
            previous_scene = self.spatial_memory.scene or {}
            retained_map_available = bool(previous_scene.get('scene_token'))
            mapped_inventory = previous_scene.get('inventory_map', {})
            layer_records = mapped_inventory.get('entities')
            lifecycle = self.spatial_memory.lifecycle()
            required_ids = sorted(self._collision_entity_ids(context) - retired)
            local_candidates = sorted(set(known) & observed_ids)
            layer_state = {}
            for identifier in required_ids:
                record = (layer_records or {}).get(identifier, {}) if retained_map_available else {}
                names = record.get('layer_names', [])
                has_layer = bool(names) if not retained_map_available or layer_records is not None else None
                layer_state[identifier] = {
                    'currently_observed': identifier in observed_ids, 'has_retained_layer': has_layer,
                    'layer_status': record.get('status', 'missing' if has_layer is False else 'unknown'),
                    'layer_names': copy.deepcopy(names), 'last_observation_id': record.get('last_observation_id'),
                    'invalidated': identifier in lifecycle.get('invalidated_entity_ids', [])
                                   or record.get('status') == 'stale'}
            scope_effects = {
                'global': {'update_candidate_ids': local_candidates,
                    'effect': 'Create missing layers and update existing layers for all currently observed inventory entities; '
                              'retain unseen known layers and their existing freshness.',
                    'proves_workspace_coverage': False},
                'local': {'update_candidate_ids': local_candidates,
                    'effect': 'Create or update only model-selected currently observed entities; retain other existing layers. '
                              'Unselected missing or stale required layers remain unresolved. Requires an initialized global base.',
                    'proves_workspace_coverage': False}}
            requests = []
            if scope == 'decide':
                requests.append(ChoiceRequest(name='mapping_scope', choices=('global', 'local', 'unknown'),
                    selection_mode='proposal',
                    instructions=('Choose the mapping update scope for the task, current three-camera evidence, '
                                  'actual retained layers and changed entities. Distinguish creating missing collision '
                                  'layers from refreshing existing layers: measured inventory is not yet a collision map. '
                                  'Use required_collision_entity_ids, entity_layer_state and scope_effects. Global creates '
                                  'or updates all currently observed inventory; it does not mean all space is visible. '
                                  'Local creates or updates a selected subset while retaining other existing layers; '
                                  'unselected required missing/stale layers remain unresolved. The initial global base '
                                  'has already been published. Neither choice proves safe motion. Choose unknown '
                                  'when the strategy cannot be justified. Consider the task objects, source, destination, '
                                  'current observation extent and changed geometry. This selects inventory geometry updates, '
                                  'not robot visibility or motion feasibility. Do not infer unseen geometry.')))
            names = {f'update_entity_{index}': key
                     for index, key in enumerate(local_candidates)} if not entities else {}
            requests.extend(JudgmentRequest(name=name, instructions=(
                'If a local mapping update is selected, does inventory entity ' + identifier +
                ' need its current measured geometry integrated to create a missing collision layer or refresh '
                'an existing layer for this task? Use entity_layer_state, not geometry_status alone. '
                'No movement does not imply no mapping work: an observed object may have no collision layer yet. '
                'Consider its collision role even if it is not the grasp target. A fresh unchanged retained layer '
                'may need no update. Updating does not prove movement or authorize clearing old surfaces. '
                'Unseen does not mean absent or safe; keep uncertain evidence uncertain.'))
                for name, identifier in names.items())
            decision = await self._judge(context, requests, extra={'verification_scope': 'mapping_update', 'mapping_evidence': {
                'requested_scope': scope, 'observation': public_observation(context['observation']),
                'previous_observation': public_observation(context.get('scene_observation')),
                'inventory_entities': {key: {field: value[field] for field in (
                    'id', 'kind', 'geometry_status', 'observation_id') if field in value}
                    for key, value in known.items()},
                'previous_scene': public_summary(previous_scene),
                'retained_map_available': retained_map_available,
                'required_collision_entity_ids': required_ids,
                'entity_layer_state': layer_state, 'scope_effects': scope_effects,
                'sensor_coverage_contract': {
                    'role': 'task_mapping_evidence_not_whole_robot_visibility_requirement',
                    'unknown_space_remains_unknown': True,
                    'full_environment_coverage_claimed': False,
                    'mandatory_numerical_checks': ['inventory_geometry_freshness',
                        'scene_and_command_bindings', 'full_robot_world_collision',
                        'full_robot_self_collision', 'joint_bounds']},
                'local_update_candidates': local_candidates,
                'lifecycle': copy.deepcopy(lifecycle)}})
            if scope == 'decide':
                scope = decision['judgments']['mapping_scope'].get('value', 'unknown')
            if scope == 'local' and not entities:
                if any(decision['judgments'][name]['status'] == 'unknown' for name in names):
                    scope = 'unknown'
                else:
                    entities = [key for name, key in names.items()
                                if decision['judgments'][name]['status'] == 'yes']
            context['mapping_strategy_evidence'] = decision
            self.emit('mapping_strategy_decision', 'Decisions 选择建图范围', decision)
            if scope not in {'global', 'local'} or (scope == 'local' and not entities):
                return {'status': 'unknown', 'reason': 'mapping_strategy_unknown',
                        'details': judgment_details(decision, requests)}
            source = 'decisions'
        reason = request.reason.strip()
        if not reason:
            reason = ('Decisions mapping choice; request_hash=' + decision['evidence']['request_hash']
                      if decision is not None else 'Initial global base required; local scope is not yet permitted'
                      if source == 'sole_permitted_scope' else 'Explicit scope in ' + source + '-authored graph node')
        if selected_tables:
            reason += '; declared support cuboids use measured table input: ' + ','.join(selected_tables)
        return {'requested_scope': scope, 'scope_reason': reason,
                'mapping_entity_ids': entities if scope == 'local' else [], 'strategy_source': source}

    def _bind_region(self, observation):
        identifier = self.state.get('destination_id') or self.task_inventory.get('destination_id')
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
                     'resolve_placement', 'generate_grasps'}:
            invalidate_placement(self, context, zone=skill in {'identify_target', 'identify_destination', 'observe_scene'})
        if skill == 'perceive_placement_zone':
            invalidate_placement(self, context, zone=True)
            if self.state['destination_id'].startswith(('table-region:', 'surface-region:')):
                self._bind_region(context['observation'])
            return await perceive_for_controller(self, context)
        if skill == 'resolve_placement':
            if 'placement_zone' not in context:
                raise ValueError('placement_zone_required')
            fresh_candidates = (arguments.get('capture_mode') == 'fresh'
                and context['placement_zone'].get('semantic_scope') == 'candidate_selection')
            if fresh_candidates:
                await self.verify_scene(context)
                result = await capture_candidate_view(self, context, context['placement_zone'])
                if result['status'] != 'success':
                    return result
            rejection = candidate_capture_rejection(self, context, context['placement_zone'])
            if rejection is not None:
                return rejection
            if not fresh_candidates:
                await self.verify_scene(context)
            observation = self.measured_state(context)
            await self.execution_stage(context, 'prepare_attachment', self.prepare_attachment, context, observation)
            request = self._bind_region(context['observation'])
            preference = arguments.get('preference', 'decide')
            if preference != 'decide':
                self.task_progress.records[self.state['target_id']]['placement_preference'] = preference
                request = self._bind_region(context['observation'])
            return await resolve_xy_then_z(self, context, observation, request)
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
            strategy = await self._mapping_strategy(context, arguments)
            if strategy.get('status') == 'unknown':
                return strategy
            from ..memory_contracts import CollisionUpdate
            self.spatial_memory.request_update(CollisionUpdate.model_validate(strategy))
            self.state['mapping_strategy'] = copy.deepcopy(strategy)
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
                    'grasp_insertion_m', 'grasp_maximum_tilt_deg', 'grasp_require_away_from_robot')},
                'target_id': self.state['target_id'], 'arm': self.state['arm']})
            observation = context['observation']
            destination = observed_object(observation, self.state['destination_id'])
            if destination['kind'] == 'bin':
                self.require_grasp_view(context)
                if not self.state['facts'].get('views_ready'):
                    raise ValueError('current_target_geometry_required')
                context['grasp_placement_request'] = {
                    'object_id': self.state['target_id'],
                    'transport_region': container_transport_region(destination, self.spatial_memory.scene, observation),
                    'orientation_mode': 'preserve',
                    'release_mode': self.task_progress.records[self.state['target_id']]['release_mode'],
                    'free_placement': True}
                context['grasps'] = await self.call_algorithm(self.algorithms.generate_grasps,
                    observation, self.state['target_id'], arm=self.state['arm'],
                    placement_request=context['grasp_placement_request'],
                    planning_settings=context['planning_settings'], report_progress=True)
                context['grasps_observation_id'] = observation['observation_id']
                return context['grasps']
        if skill == 'identify_destination':
            if arguments['release_mode'] == 'container_release' and arguments['destination_kind'] != 'container':
                raise ValueError('container_release_requires_container')
            context.pop('destination_permission', None)
            permission = None
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
            if (arguments.get('placement_mode') == 'configured_xy'
                    and destination.get('position_source') != 'configured_xy_measured_support'):
                raise ValueError('fixed_xy_requires_configured_region')
            local_task = self.active_object_task or self.task_spec
            if (local_task is not None and (local_task.constraints or self.task_spec.constraints)
                    and arguments['destination_id'] not in {
                        local_task.destination_ref, local_task.work_area_ref}):
                requests = [JudgmentRequest(name='intermediate_destination_allowed', instructions=(
                    'Does placing the current target temporarily at the selected intermediate destination '
                    'respect the original user handling and location constraints? An intermediate transfer '
                    'does not satisfy the final destination. Judge semantic permission, not geometric safety.'))]
                judgment = await self._judge(context, requests, extra={
                    'verification_scope': 'intermediate_destination',
                    'intermediate_destination_id': arguments['destination_id'],
                    'destination_id': arguments['destination_id'],
                    'destination_evidence': {key: copy.deepcopy(destination[key]) for key in
                        ('id', 'kind', 'color', 'position', 'position_source', 'placement_reference') if key in destination},
                    'destination_kind': destination['kind'],
                    'intermediate_placement_mode': arguments.get('placement_mode'),
                    'declared_destination_references': destination_reference_evidence(
                        observation, self.compiler.configured_regions, local_task),
                    'final_destination_id': local_task.destination_ref})
                verdict = judgment['judgments']['intermediate_destination_allowed']['status']
                if verdict != 'yes':
                    return {'success': False, 'status': 'failure' if verdict == 'no' else 'unknown',
                            'reason': ('intermediate_destination_rejected' if verdict == 'no'
                                       else 'intermediate_destination_constraint_unconfirmed'),
                            'details': judgment_details(judgment, requests)}
                permission = judgment
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
            if permission is not None:
                from ..robot_skills.placement_resolution import digest

                context['destination_permission'] = {
                    'target_id': self.state['target_id'], 'destination_id': identifier,
                    'reference_id': arguments['destination_id'],
                    'task_sha256': digest(local_task.model_dump(mode='json')),
                    'observation_id': observation['observation_id'],
                    'region_sha256': digest(destination.get('placement_reference', {})),
                    'evidence': copy.deepcopy(permission)}
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
            observed_object(context['observation'], self.state['target_id'])
            if self.state['destination_id'] is not None:
                observed_object(context['observation'], self.state['destination_id'])
            requested = arguments.get('arm', 'auto')
            evaluations = await self.call_algorithm(self.algorithms.evaluate_reachability,
                context['observation'], self.state['target_id'], self.state['destination_id'],
                requested_arm=None if requested == 'auto' else requested)
            context['reachability_evaluations'] = evaluations
            available = [arm for arm in ('A', 'B') if evaluations.get(arm, {}).get('reachable')
                         and requested in ('auto', arm)]
            if context.get('holding'):
                available = [arm for arm in available if arm == context['holding']['arm']]
            if not available:
                return {'success': False, 'reason': 'no_reachable_arm', 'evaluations': evaluations}
            if len(available) == 1:
                arm = available[0]
            else:
                from ..core.decisions import ChoiceRequest
                requests = [ChoiceRequest(name='execution_arm',
                    instructions='Choose for the bound CURRENT target_id and destination_id, not the original '
                                 'processing input. Choose one eligible arm as a planning proposal using the '
                                 'current target and destination, their measured positions and the supplied '
                                 'workspace evaluations. A is the robot left arm, B its right arm. This selection '
                                 'does not certify a grasp pose, placement pose, or collision-free trajectory; '
                                 'subsequent numerical planners must validate those. A pending above-container '
                                 'release pose is expected at this stage and does not by itself prevent selecting '
                                 'an arm. Preserve explicit user arm constraints. Choose unknown only if the '
                                 'supplied task constraints or entity bindings cannot be resolved. '
                                 'Use previous_same_target_failure to compare the earlier arm and numerical '
                                 'rejection with current evidence. Consider another eligible arm when the failed '
                                 'inputs have not improved. A new capture ID alone is not improved evidence. '
                                 'The previous failure does not establish that either arm will now succeed.',
                    choices=(*available, 'unknown'), selection_mode='proposal')]
                entities = public_observation(context['observation'])['objects']
                failure = public_summary(getattr(self.graph_executor, 'last_failure', None))
                if (failure and (failure.get('target_id') != self.state['target_id']
                        or failure.get('skill') not in {'plan_grasp', 'execute_grasp', 'plan_place', 'execute_place'})):
                    failure = None
                judgment = await self._judge(context, requests, extra={
                    'verification_scope': 'execution_arm',
                    'previous_same_target_failure': failure,
                    'feasible_arms': available, 'workspace_evaluations': evaluations,
                    'robot_arm_names': {'A': 'robot left arm', 'B': 'robot right arm'},
                    'current_measured_entities': [entity for entity in entities
                        if entity['id'] in {self.state['target_id'], self.state['destination_id']}]})
                arm = judgment['judgments']['execution_arm'].get('value')
                if arm not in available:
                    return {'status': 'unknown', 'reason': 'execution_arm_unknown',
                            'details': judgment_details(judgment, requests)}
            self.task_progress.bind_arm(arm)
            self.state['arm'] = arm
            return {'arm': arm, 'evaluations': evaluations, 'success': True}
        if skill == 'plan_place':
            for key in ('place_plan', 'place_plan_binding', 'place_plan_observation_id'):
                context.pop(key, None)
            self.state['facts']['place_plan_ready'] = False
            if 'placement_resolution' not in context:
                raise ValueError('placement_resolution_required')
            await self.verify_scene(context)
            observation = self.measured_state(context)
            await self.execution_stage(context, 'prepare_attachment', self.prepare_attachment, context, observation)
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
        subject = context.get('visual_inventory_reference', {}).get('entities', {}).get(subject_id, {})
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

    async def _judge(self, context, requests, *, extra=None, include_task=True, reference=None):
        capture = context['latest_capture']
        if context.get('capture_epochs', {}).get(capture['observation_id']) != self.observation_epoch:
            xy_proposal = ((extra or {}).get('verification_scope') == 'placement_xy_proposal'
                and (extra or {}).get('scene_token') == self.spatial_memory.scene.get('scene_token')
                and bool((extra or {}).get('scene_token'))
                and all(isinstance(item, ChoiceRequest) and item.selection_mode == 'proposal' for item in requests))
            if xy_proposal:
                # This selects a location in an already verified region. Old RGB
                # is explicitly historical context, never a current visual fact.
                extra = {**extra, 'image_evidence_scope': 'historical_scene_for_xy_proposal'}
            elif (extra or {}).get('verification_scope') not in {'graph_recovery', 'policy_condition'}:
                raise ValueError('fresh_observation_required_before_judgment')
            else:
                # Recovery has current RGB, without an implicit full observation.
                capture = await self.capture_observation(context, recovery=True)
                self.record_observation(capture, 'recovery_decision')
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
            if (observation.get('observation_id') == capture['observation_id']
                    and observation.get('generation') == capture.get('generation')
                    and 'current_instance_locations' not in judgment_context):
                from ..task_binding_images import current_instance_locations
                judgment_context['current_instance_locations'] = current_instance_locations(observation)
            return await self.call_algorithm(self.judgments.judge, capture, requests=requests,
                context=judgment_context,
                reference=context['visual_inventory_reference'] if reference is None else reference)
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
            try:
                result = await self._step(name, context, args)
            except (ValueError, AlgorithmError, PerceptionError, TimeoutError) as error:
                result = skill_error_result(error, self.state, args)
            from ..core.outcome_evidence import outcome_state
            from ..core.outcomes import GraphSkillResult
            result = bind_failure_context(result, skill=name, arguments=args, state=self.state)
            outputs = {field: result[field] for field in self.registry.contracts[name].result_bindings.values()
                       if field in result}
            return GraphSkillResult(status=result_status(result), outputs=public_summary(outputs),
                                    evidence=outcome_state(result))

        async def judge(request: GraphDecisionRequest):
            from ..agent.completion import completion_obligations

            name, bindings = request.condition, request.bindings
            choices, branches = request.choices, request.branches
            declared = {(node.condition, tuple(node.choices)) for scope in [graph, *graph.subgraphs]
                        for node in scope.nodes if node.type == 'router'}
            if (name, tuple(choices)) not in declared:
                raise ValueError('undeclared_policy_judgment:' + name)
            from .decision_inputs import available_branch_choices, branch_evidence
            available_choices = available_branch_choices(choices, branches) if choices else ()
            if choices and not any(choice != 'unknown' for choice in available_choices):
                return GraphDecisionResult(status='unknown', value='unknown', reason='no_available_declared_branch',
                    details={'producer': 'runtime_branch_availability', 'declared_branches': list(branches)})
            requests = [ChoiceRequest(name='condition', instructions=name, choices=available_choices,
                                      selection_mode='proposal') if choices else
                        JudgmentRequest(name='condition', instructions=name)]
            available_operations = branch_evidence(graph, branches, skill_catalog(self.registry),
                request.last_failure) if choices else {'skill_contracts': skill_catalog(self.registry)}
            result = await self._judge(context, requests, extra={
                'verification_scope': 'policy_condition', 'policy_bindings': copy.deepcopy(bindings),
                'last_result': copy.deepcopy(request.last_result),
                'last_failure': copy.deepcopy(request.last_failure),
                'declared_branches': copy.deepcopy(list(branches)),
                'current_facts': copy.deepcopy(self.state['facts']),
                **available_operations,
                'selected_arm': self.state.get('arm'),
                'arm_evaluations': context.get('reachability_evaluations', {}),
                'edge_selection_only': True,
                'verified_obligations': completion_obligations(self.task_progress.snapshot(), self.object_memory.snapshot())})
            # The executor consumes this exact typed result. A bool/None return
            # discards the probability, abstention reason and failed question.
            return GraphDecisionResult(**result['judgments']['condition'], details=judgment_details(result, requests))

        graph = self.policy_graph
        completed = self.task_progress.completed_input_ids()
        pending = [identifier for identifier in self.task_inventory.get('selected_ids', []) if identifier not in completed]
        repeatable = {name for name, contract in self.registry.contracts.items() if contract.retry_safe}

        def emit_graph(kind, title, data):
            if kind == 'policy_node_started':
                self.state['active_policy_node'] = copy.deepcopy(data)
            self.emit(kind, title, data)

        self.graph_executor = NativeGraphExecutor(invoke, target_ids=pending,
            resolve_target_ids=confirmed_targets,
            object_task_context=enter_object_task,
            bindings={'$source': self.task_inventory.get('source_id'),
                      '$destination': self.task_inventory.get('destination_id'),
                      '$work_area': self.task_inventory.get('work_area_id')},
            judge=judge, before_target=before_target, after_target=after_target,
            result_bindings={name: contract.result_bindings
                             for name, contract in self.registry.contracts.items()},
            is_target_pending=lambda identifier: identifier not in (
                self.task_progress.completed_input_ids()),
            check_cancelled=self.check_generation, emit=emit_graph, repeatable_skills=repeatable,
            step_limit=131072)
        await self.graph_executor.run(graph)

    async def _complete(self, context):
        from ..agent.completion import (
            COMPLETION_MILESTONE_RULE,
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
            feedback = self.simulation.robot_snapshot()
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
            pose_proof = initial_pose_verdict(context['initial_joint_reference'], self.simulation.robot_snapshot())
            self.state['completion_pose_evidence'] = pose_proof
            self.state['facts']['at_initial_pose'] = bool(pose_proof['success'])
            if not pose_proof['success']:
                raise DiagnosticError('completion_pose_unconfirmed', pose_proof)
        self.state['completion_pose_evidence'] = copy.deepcopy(pose_proof)
        previous = self.state.get('completion_evidence', {}).get('judgments', {})
        mode = 'fresh' if completion_failures(previous) else 'reuse_current'
        capture = await self.capture_scene_observation(context, {'capture_mode': mode}, purpose='task_completion')
        completion_key = (capture['observation_id'], self.observation_epoch, self.task_progress.ledger_revision)
        if (completion_failures(previous)
                and context.get('completion_request_key') == completion_key):
            raise ValueError('task_completion_unchanged_evidence')
        self.record_observation(capture, 'task_completion')
        lineage = ('For original input IDs ' + json.dumps(self.task_inventory['selected_ids'])
                   + ' and their independently bound processing output lineages, ')
        # Evaluate each declared requirement. Asking the same conjunction again
        # as goal_satisfied adds no evidence and can contradict its own terms.
        requests = []
        for index, condition in enumerate(self.task_spec.completion_conditions):
            requests.append(JudgmentRequest(name=f'completion_condition_{index}', instructions=(
                lineage + 'Do the CURRENT images and recorded per-object verified evidence support this single '
                'user completion condition? ' + condition + ' Do not infer success from an empty work queue. '
                'For requested robot posture use verified_robot_pose numerical feedback, not visual joint estimation.'
                + COMPLETION_MILESTONE_RULE)))
        for item_index, identifier in enumerate(self.task_inventory['selected_ids']):
            item_task = self.object_tasks.get(identifier)
            if item_task is None:
                continue
            for condition_index, condition in enumerate(item_task.completion_conditions):
                requests.append(JudgmentRequest(name=f'object_{item_index}_condition_{condition_index}', instructions=(
                    'For original input ID ' + identifier + ' and its independently bound output lineage only, '
                    'do CURRENT images and verified ledger evidence support this item requirement: ' + condition
                    + '? Another item completing its own goal is not evidence for this one.'
                    + COMPLETION_MILESTONE_RULE)))
        presence = context.get('source_presence_evidence', {})
        reuse_presence = (self.task_spec.scope == 'all' and presence.get('observation_id') == capture['observation_id']
                         and presence.get('epoch') == self.observation_epoch)
        if self.task_spec.scope == 'all' and not reuse_presence:
            requests.append(source_presence_request())
        result = await self._judge(context, requests, extra={
            'verified_obligations': completion_obligations(self.task_progress.snapshot(), self.object_memory.snapshot()),
            'verified_robot_pose': pose_proof, 'verification_scope': 'task_completion'})
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
        self.task_spec = self.task_inventory = self.policy_graph = None
        self.object_tasks = {}
        self.active_object_task = None
        self.observation_epoch = 0
        self.state.pop('partial_execution_failure', None)
        for key in ('failure_report', 'policy_failure', 'error_details', 'error_code', 'failed_step'):
            self.state.pop(key, None)
        self.compiler.reset()
        self.task_progress = PolicyTaskProgress(self.state['instruction'], 'collection', None,
                                                selection_order='inventory')
        self.state.update(task='grasp_place', controller='responses_policy_graph',
                          requested_arm=None, attributes={'kind': 'any', 'color': 'any'},
                          destination_attributes={'kind': 'any', 'color': 'any'})
        self.state['facts'].update(needs_place=True, collection_task=True, center_reachability_enabled=True,
                                   inventory_ready=False, collision_map_initialized=False,
                                   observed=False, task_verified=False)
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
            self.state.update(status='needs_attention', error=str(error)[:1000], **record,
                              failed_step=copy.deepcopy(self.state.get('current_step')))
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
                'inventory_responses_call_count': memory_calls}
            self.emit('run_end', '任务结束', self.state)
            if self.experiment is not None:
                self.experiment.stop(self.state)

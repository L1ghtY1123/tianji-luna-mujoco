"""Compose model planning, evidence judgments and registered robot operations."""
import asyncio
import copy
import json
import os
import time
from functools import partial
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agent.decisions import ChoiceRequest, DecisionsJudgments, JudgmentRequest
from ..agent.responses import ResponsesBudget, ResponsesCompiler
from ..agent.schema import (
    Graph,
    TaskSpec,
    object_task_specs,
    skill_catalog,
    validate_graph,
)
from ..agent.task_binding import (
    bind_destination_choice,
    destination_candidates,
    destination_question,
    target_membership_question,
)
from ..algorithms import AlgorithmError
from ..core.placement import ResolvePlacementArguments
from ..inventory_order import robot_distance_order
from ..inventory_reference import bound_identity_coverage_source
from ..perception import PerceptionError
from ..placement import observed_object, public_observation
from ..robot_skills.container_geometry import container_transport_region
from ..robot_skills.operations import DESTINATION_ID, RobotOperations, configured_destination
from ..robot_skills.placement_region import measured_table_region
from ..robot_skills.placement_zone import measured_object_region
from ..robot_skills.unpack import VLAUnpack
from ..runner import Runner, public_summary
from ..skills import NoArguments
from ..task_progress import current_table_measurements
from ..visual_api import VisualAPI
from ..visual_assistance import VisualCallBudget
from ..vla_configuration import vla_unpack_configuration
from .diagnostics import DiagnosticError, exception_details, failure_record, judgment_details
from .execution_state import result_status
from .inventory import PolicyTaskProgress
from .native_graph import NativeGraphExecutor
from .parameters import GraspArguments, apply_grasp_parameters
from .placement import (
    candidate_capture_rejection,
    capture_candidate_view,
    invalidate_placement,
    perceive_for_controller,
    resolve_xy_then_z,
)
from .placement_retry import placement_attempt, placement_retry_block, record_placement_failure
from .support import support_context, support_memory


class TargetArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    target_id: str


class DestinationArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    destination_id: str
    destination_kind: Literal['support_region', 'container']
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


class ArmArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    arm: Literal['auto', 'A', 'B'] = 'auto'


class PrepareUnpackArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class ObserveArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    capture_mode: Literal['reuse_current', 'fresh'] = Field(default='reuse_current', description=(
        'Use the latest three-camera capture when its action/scene/command versions and measured '
        'arm/gripper feedback still match. Both modes reuse unchanged evidence; age alone and a '
        'failed decision do not force another capture. A changed viewpoint or scene requires acquisition. '
        'An identical failed analysis is not repeated. Reuse preserves the original observation ID/time.'))


class ObservationMotionArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    viewpoint: Literal['configured', 'wrist_tilt_positive', 'wrist_tilt_negative'] = Field(
        default='configured', description=(
            'configured returns to the operator observation pose and can satisfy the requested post-delivery '
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
                 environment_assumptions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment_assumptions = copy.deepcopy(environment_assumptions or {})
        schemas = {'identify_target': TargetArguments, 'identify_destination': DestinationArguments,
                   'choose_arm': ArmArguments, 'build_scene': BuildSceneArguments,
                   'prepare_unpack': PrepareUnpackArguments, 'observe_scene': ObserveArguments,
                   'move_to_observe': ObservationMotionArguments,
                   'generate_grasps': GraspArguments, 'resolve_placement': ResolvePlacementArguments}
        names = ('move_to_observe', 'return_to_initial_pose', 'observe_scene', 'confirm_inventory', 'identify_target', 'identify_destination',
                 'choose_arm', 'build_scene', 'generate_grasps', 'plan_grasp', 'execute_grasp',
                 'verify_grasp', 'perceive_placement_zone', 'resolve_placement',
                 'plan_place', 'execute_place', 'verify_place', 'prepare_unpack',
                 'vla_unpack', 'verify_unpack',
                 'verify_task_completion')
        for name in names:
            self.registry.register(name, schemas.get(name, NoArguments), partial(self._dispatch, name))
        config = VisualAPI.from_environment()
        self.compiler = compiler or ResponsesCompiler(api_key=config.api_key,
            budget=ResponsesBudget(),
            timeout_s=float(os.environ.get('LUNA_RESPONSES_TIMEOUT_S', '180')),
            registry=self.registry, known_assets=known_assets, configured_regions={DESTINATION_ID: {
                'description': 'configured unmarked work area on the measured main table',
                'region': self.region}})
        self.judgments = judgments or DecisionsJudgments(api_key=config.api_key, timeout_s=config.timeout_s)
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
        from ..agent.completion import has_confirmed_delivery, required_observation_return
        from ..robot_skills.return_pose import observation_motion_scene_ready

        self.check_generation()
        self.check_log()
        self.state['facts']['observation_motion_scene_ready'] = observation_motion_scene_ready(context)
        previous = getattr(getattr(self, 'graph_executor', None), 'last_result', None) or {}
        if name == 'move_to_observe':
            configured = (arguments or {}).get('viewpoint', 'configured') == 'configured'
            required_return = configured and (context.pop('pending_required_return', False)
                or required_observation_return(self.environment_assumptions.get('completion_requirements', {}),
                                               previous.get('reason')))
            visibility_motion = context.pop('pending_visibility_motion', False)
            if not required_return and (not configured or visibility_motion
                    or previous.get('status') in {'failure', 'unknown'}):
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
            self._record_partial_execution_failure(context, {
                'status': 'failure', 'success': False,
                'reason': getattr(error, 'code', None) or str(error) or type(error).__name__})
            raise
        finally:
            self.state['facts']['observation_motion_scene_ready'] = observation_motion_scene_ready(context)
            if self.registry.contracts[name].invalidates_observation:
                # A new raw result photograph is not an ID-associated inventory
                # observation. Successful provides below may explicitly restore it.
                self.state['facts']['observed'] = False
                self.state['facts']['task_verified'] = False
        self.check_generation()
        status = result_status(result)
        for fact in effects:
            self.state['facts'][fact] = status == 'success'
        # Persist history only after the observation skill and all of its checks
        # succeeded and its declared observed effect has been applied.
        if (status == 'success' and 'observed' in effects
                and self.environment_assumptions.get('completion_requirements', {}).get('after_each_object')):
            feedback = self.simulation.robot_snapshot()
            if self.is_observation_pose(context, feedback):
                delivered = set(self.task_progress.completed_input_ids())
                for identifier in (self.task_inventory or {}).get('selected_ids', []):
                    current = identifier
                    while self.task_progress.records.get(current, {}).get('output_entity_id'):
                        current = self.task_progress.records[current]['output_entity_id']
                    record = self.task_progress.records.get(current, {})
                    if has_confirmed_delivery(record):
                        delivered.add(identifier)
                for identifier in sorted(delivered):
                    self._record_post_delivery_observation(context, identifier)
        if status != 'success':
            self._record_partial_execution_failure(context, result)
        elif (self.registry.contracts[name].requires.get('recovery_needed') is True
              and not self.state['facts'].get('recovery_needed')):
            self.state.pop('partial_execution_failure', None)
        self.emit('skill_complete', name, {'result': result, 'duration_s': time.monotonic() - started})
        if self.task_progress is not None:
            self.sync_inventory(context)
            if name.startswith('verify_') or name == 'observe_scene':
                calls = [event for event in self.events if event['kind'] == 'visual_decisions_call'
                         and event.get('invocation_id') == self.state['current_step']['invocation_id']]
                self.emit('inventory_snapshot', '当前物体清单', {
                    'checkpoint': name, 'task_progress': self.state['task_progress'],
                    'model_event_ids': [event['id'] for event in calls],
                    'evidence_source': 'gripper_feedback' if name == 'verify_grasp' else 'visual_checkpoint'})
        return result

    async def _bootstrap(self, context):
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
            'at_initial_pose': True}
        self.record_observation(capture, 'initial')
        audit, done = self.visual_audit_sink()
        self.compiler.audit = audit
        try:
            proposal = await self.call_algorithm(self.compiler.discover, capture,
                                                 instruction=self.state['instruction'])
        finally:
            done.set()
        self.task_spec = TaskSpec.model_validate(proposal['task_spec'])
        if not self.task_spec.supported:
            raise ValueError('task_capability_missing:' + self.task_spec.reason)
        self.task_inventory = copy.deepcopy(proposal['task_inventory'])
        self.policy_graph = Graph.model_validate(proposal['policy_graph'])
        self.object_tasks = object_task_specs(self.policy_graph)
        self.active_object_task = None
        validate_graph(self.policy_graph, self.registry)
        reference = copy.deepcopy(proposal['reference'])
        reference['owner_id'] = self.run_id
        from ..observation_labels import responses_reference_labels

        self.record_annotation(capture, responses_reference_labels(capture, reference))
        context.update(initial_capture=capture, latest_capture=capture, fine_capture=capture,
                       visual_inventory_reference=reference)
        self.task_progress.initialize_scene_inventory({'objects': [], 'scene_entities': proposal['entities'],
                                                       'observation_id': capture['observation_id']})
        self.state.update(task_spec=self.task_spec.model_dump(mode='json'),
                          policy_graph=self.policy_graph.model_dump(mode='json'))
        self.emit('skill_catalog', '本次 graph 使用的 skills 契约', {'skills': skill_catalog(self.registry)})
        self.emit('skill_program', 'Responses 初始 graph', self.state['policy_graph'])
        diagnostics = proposal.get('evidence', {}).get('graph_preflight_diagnostics', [])
        if diagnostics:
            self.emit('policy_preflight_diagnostics', '初始 graph 中需运行时恢复的前提缺口', {'diagnostics': diagnostics})

    async def observe(self, context, *, checkpoint=None, capture=None, camera_ids=None,
                      purpose=None, recovery=False):
        if self.task_spec is None:
            raise ValueError('task_compilation_required_before_observation_skill')
        checkpoint = checkpoint or 'initial'
        capture = await self.capture_observation(context, capture=capture, recovery=recovery)
        from ..robot_skills.placement_resolution import digest
        from .recovery_evidence import observation_inputs

        def analysis_key():
            inputs = observation_inputs(context, self.state['facts'])
            inputs['physical_facts'].pop('recovery_needed', None)
            return digest({'capture_id': capture['observation_id'], 'checkpoint': checkpoint, 'inputs': inputs})

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
        context['visual_plan'] = {checkpoint: 'verify_transfer' if checkpoint == 'post_place' else 'associate_inventory'}
        try:
            observation = await Runner.observe(self, context, checkpoint=checkpoint, capture=capture,
                camera_ids=['1', '2', '3'], purpose='geometry', recovery=recovery)
        except (ValueError, RuntimeError) as error:
            if any(word in str(error) for word in ('unknown', 'uncertain', 'identity_unresolved')):
                context['_observation_analysis'] = {'key': analysis_key(), 'failure': str(error)}
            raise
        observation['task_inventory'] = self.task_inventory
        # A configured work area is an explicit prior, not a visually invented marker.
        for identifier in self.compiler.configured_regions:
            configuration = self.compiler.configured_regions[identifier]
            region = configured_destination(observation, configuration.get('region', self.region),
                identifier=identifier, support_id=self.task_inventory.get('support_id'))
            observation['objects'] = [obj for obj in observation['objects'] if obj['id'] != identifier]
            observation['objects'].append(region)
            self.task_progress.scene_entities[identifier] = copy.deepcopy(region)
        context.update(observation=observation, inventory_observation=copy.deepcopy(observation))
        context['_observation_analysis'] = {'key': analysis_key()}
        return observation

    async def _refresh_observed_support(self, context, identifier):
        entities = self.task_progress.scene_entities
        entity = entities.get(identifier, {})
        previous = entity.get('observed_support')
        capture = context['latest_capture']
        source_id = entity.get('current_region') or entity.get('initial_parent_id')
        source_missing = (entities.get(source_id, {}).get('kind') == 'bin'
                          and not any(obj['id'] == source_id for obj in context.get('observation', {}).get('objects', [])))
        if (previous is None and source_id and not source_missing) or (isinstance(previous, dict)
                and previous.get('observation_id') == capture['observation_id']
                and previous.get('source') == 'openai_decisions' and previous.get('request_hash')):
            return
        candidates = {key: {field: value[field] for field in ('id', 'kind', 'color', 'description') if field in value}
                      for key, value in entities.items() if value.get('kind') in {'table', 'bin'}}
        if not candidates:
            raise ValueError('observed_support_candidates_missing')
        measured = support_context(capture, context.get('observation', {}), entity, candidates)
        measured['support_memory'] = support_memory(entity, self.task_progress.records.get(identifier, {}),
            getattr(self.task_progress, 'attempts', {}).get(identifier, 0))
        requests = [JudgmentRequest(name='support_' + str(index), instructions=(
            'The CURRENT direct support or containing receptacle of the ' + entity.get('color', '') + ' '
            + entity.get('kind', 'object') + ' (ID ' + identifier + ') is the '
            + support.get('color', '') + ' ' + support['kind'] + ' (ID ' + key + '). '
            'Assess the current relationship from CURRENT RGB, measured geometry and dated verified '
            'support_memory. A previously verified location remains useful when partially occluded; '
            'do not discard it only because a later camera view is incomplete. Check recorded transfers, '
            'attempts and contradictory current evidence before retaining a relation. Reference labels '
            'and requested destinations alone are not facts. For an object inside a box on a table, '
            'the box is the direct container, not the table underneath. Use ordinary resting-object '
            'support from the spatial relationship; seeing the exact contact pixels or hidden box floor '
            'is not required. This asks which support contains/carries the object, not whether its pose '
            'or contact force was measured exactly. Judge only this support relation, independently '
            'of CAD retrieval, task completion and whether a future grasp is feasible. Keep uncertain '
            'when the combined evidence is insufficient or contradictory. This answer does not select '
            'a recovery action; a separate visibility assessment determines whether another view is needed.'))
            for index, (key, support) in enumerate(candidates.items())]
        # Dated verified relations are memory, while old free-text descriptions
        # and requested destinations are not current support evidence.
        reference = copy.deepcopy(context.get('visual_inventory_reference', {}))
        reference.setdefault('entities', {identifier: measured['subject'], **measured['support_candidates']})
        for record in reference['entities'].values():
            record.pop('description', None)
            record.pop('initial_parent_id', None)
        result = await self._judge(context, requests, extra=measured, include_task=False, reference=reference)
        judgments = [result['judgments'][request.name] for request in requests]
        accepted = [index for index, decision in enumerate(judgments) if decision['status'] == 'yes']
        if len(accepted) != 1 or any(decision['status'] != 'no' for index, decision in enumerate(judgments)
                                     if index not in accepted):
            raise DiagnosticError('observed_support_unknown', judgment_details(result, requests))
        index = accepted[0]
        proof = {'support_entity_id': list(candidates)[index], 'observation_id': capture['observation_id'],
                 'generation': capture.get('generation'), 'source': 'openai_decisions',
                 'request_hash': result['evidence']['request_hash'],
                 'assessment_basis': 'current_evidence_and_verified_memory',
                 'support_memory': copy.deepcopy(measured['support_memory']),
                 'confidence': judgments[index]['probability'], 'judgments': copy.deepcopy(result['judgments'])}
        entity['observed_support'] = copy.deepcopy(proof)
        if identifier in self.task_progress.records:
            self.task_progress.records[identifier]['observed_support'] = copy.deepcopy(proof)
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
        inventory = self.task_progress.inventory_state()
        self.state['task_inventory'] = copy.deepcopy(self.task_inventory)
        self.state['task_progress'] = self.task_progress.snapshot()
        reference = context.get('visual_inventory_reference')
        if reference is not None:
            reference['task_inventory'] = copy.deepcopy(self.task_inventory)
            reference['inventory_state'] = inventory
            reference['entities'] = {key: copy.deepcopy(value) for key, value in inventory['entities'].items()
                                     if key not in inventory['retired_entity_ids']}

    def _required_visible_ids(self, observation):
        current = {item['id'] for item in observation['objects']
                   if item.get('kind') not in {'table', 'marker'}}
        if not self.task_progress.inventory_finalized:
            return set(self.task_inventory['source_object_ids']) | current
        # Established IDs survive partial views. Only newly consumed measurements
        # need current identity proof; missing geometry remains in the map ledger.
        return current

    def _require_visible_inventory(self, observation):
        # Missing background coverage belongs to the collision guard, not task identity.
        required = self._required_visible_ids(observation)
        visible = {obj['id'] for obj in observation['objects']}
        missing = required - visible
        if missing:
            raise DiagnosticError('task_target_geometry_unknown:' + ','.join(sorted(missing)), {
                'component': 'perception_geometry', 'observation_id': observation['observation_id'],
                'diagnostic_checks': [{'name': 'required_visible_ids', 'actual': sorted(visible),
                                       'expected': sorted(required), 'missing': sorted(missing)}],
                'unknowns': ['缺少几何不等于 SAM3 低分；需要检查对应相机的查询、身份关联和有效深度记录。']})

    def _check_source_view(self, observation):
        # Any calibrated camera may support a target. Full-space coverage is checked separately.
        self._require_visible_inventory(observation)
        self._validate_inventory_evidence(observation)

    async def build_scene(self, context, exclusion_state):
        # Only a verified transformation can retire an input identity.
        if self.task_progress is None:
            return await super().build_scene(context, exclusion_state)
        retired = self.task_progress.inventory_state()['retired_entity_ids']
        metadata = context.setdefault('mapping_lifecycle', {})
        metadata['retired_entity_ids'] = retired
        metadata['required_entity_ids'] = [identifier
            for identifier, record in self.task_progress.scene_entities.items()
            if identifier not in retired and record.get('kind') not in {'table', 'marker'}]
        return await super().build_scene(context, exclusion_state)

    async def _mapping_strategy(self, context, arguments):
        request = BuildSceneArguments.model_validate(arguments)
        inventory = self.task_progress.scene_entities
        retired = set(self.task_progress.inventory_state()['retired_entity_ids'])
        known = {key: value for key, value in inventory.items()
                 if key not in retired and value.get('kind') not in {'marker', 'table'}}
        tables = {key for key, value in inventory.items()
                  if key not in retired and value.get('kind') == 'table'}
        observed_ids = {item['id'] for item in context['observation']['objects']}
        if set(request.entity_ids) - set(known) - tables:
            raise ValueError('mapping_entity_not_in_inventory')
        # Table cuboids are already carried by the measured support input; the
        # per-instance layer selector addresses only CAD/OBB objects. Preserve
        # the declared support choice in the audit reason, never as an object mask.
        selected_tables = sorted(set(request.entity_ids) & tables)
        entities = [identifier for identifier in request.entity_ids if identifier not in tables]
        if request.scope == 'local' and selected_tables and not entities:
            raise ValueError('local_mapping_requires_object_entities')
        scope, source = request.scope, self.state.get('active_policy_node', {}).get('author', 'responses')
        decision = None
        if scope == 'decide' or (scope == 'local' and not entities):
            previous_scene = context.get('scene') or {}
            retained_map_available = bool(previous_scene.get('scene_token'))
            mapped_inventory = previous_scene.get('inventory_map', {})
            layer_records = mapped_inventory.get('entities')
            lifecycle = context.get('mapping_lifecycle', {})
            required_ids = sorted((set(known) | set(mapped_inventory.get('required_entity_ids', []))
                                   | set(lifecycle.get('required_entity_ids', []))) - retired)
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
                              'Unselected missing or stale required layers remain unresolved, including on the first map.',
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
                                  'unselected required missing/stale layers remain unresolved. Either scope may be chosen '
                                  'for the first map when justified. Neither choice proves safe motion. Choose unknown '
                                  'when the strategy cannot be justified. Consider the task objects, source, destination, '
                                  'current observation extent and changed geometry. Head/wrist cameras need not see '
                                  'the back of the shoulder or every surface of their own tools; absence of a whole-tool '
                                  'free-ray certificate is a coverage limitation, not proof that task geometry is missing '
                                  'or that a collision occurred. Do not infer unseen geometry.')))
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
            decision = await self._judge(context, requests, extra={'mapping_evidence': {
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
                      if decision is not None else 'Explicit scope in ' + source + '-authored graph node')
        if selected_tables:
            reason += '; declared support cuboids use measured table input: ' + ','.join(selected_tables)
        return {'requested_scope': scope, 'scope_reason': reason,
                'mapping_entity_ids': entities if scope == 'local' else [], 'strategy_source': source}

    def _validate_inventory_evidence(self, observation):
        visual = observation.get('visual_inventory', {})
        coverage = visual.get('identity_coverage', {})
        evidence = visual.get('evidence', {})
        required = self._required_visible_ids(observation)
        threshold = coverage.get('minimum_confidence')
        check = observation.get('demo_inventory_check', {})
        confidence = check.get('confidence')
        if (not bound_identity_coverage_source(observation)
                or coverage.get('observation_id') != observation['observation_id']
                or not coverage.get('request_hash')
                or coverage['request_hash'] != evidence.get('request_hash')
                or type(threshold) not in (int, float) or not .8 <= threshold <= 1.
                or not required.issubset(coverage.get('accepted_entity_ids', []))
                or check.get('source') != 'openai_decisions'
                or check.get('observation_id') != observation['observation_id']
                or check.get('request_hash') != coverage['request_hash']
                or type(confidence) not in (int, float) or not 0. <= confidence <= 1.
                or check.get('status') not in {'none', 'not_requested', 'unknown'}
                or check.get('status') != 'unknown' and confidence < .8):
            raise DiagnosticError(
                'unlisted_source_object' if check.get('status') in {'extra', 'unlisted'}
                else 'source_inventory_unknown', {
                    **copy.deepcopy(evidence), 'component': 'inventory_evidence_validation',
                    'observation_id': observation['observation_id'],
                    'diagnostic_checks': [
                        {'name': 'identity_coverage', 'actual': copy.deepcopy(coverage),
                         'expected': {'source': 'openai_decisions or bound mixed_decisions_measured_container',
                                      'minimum_confidence': .8,
                                      'accepted_entity_ids_include': sorted(required),
                                      'observation_id': observation['observation_id'],
                                      'request_hash': evidence.get('request_hash')}},
                        {'name': 'source_inventory_check', 'actual': copy.deepcopy(check),
                         'expected': 'current matching observation/hash; none/not_requested with confidence >= .8, '
                                     'or unknown with pending closure'}]})
        # Enumeration uncertainty does not negate separately confirmed identities.
        # Retain it as unknown; only fresh final source evidence can close an all task.
        self.state['visible_inventory_resolution'] = copy.deepcopy(check)

    async def _confirm_task_binding(self, context, task_spec=None, target_ids=None):
        """Confirm proposed task roles with current Decisions evidence before any motion."""
        if task_spec is None and self.object_tasks:
            proofs = {}
            for identifier, item_task in self.object_tasks.items():
                proofs[identifier] = await self._confirm_task_binding(context, item_task, [identifier])
            if self.task_spec.scope == 'all':
                proofs['aggregate'] = await self._confirm_task_binding(context, self.task_spec)
            self.state['object_task_binding_evidence'] = proofs
            return {'success': True, 'object_tasks': proofs}
        task_spec = task_spec or self.task_spec
        targets = list(target_ids if target_ids is not None else self.task_inventory['source_object_ids'])
        current_entities = {obj['id']: {
            **{key: copy.deepcopy(obj[key]) for key in (
                'id', 'kind', 'color', 'description', 'coverage', 'position',
                'visible_centroid_world', 'source', 'position_source', 'observation_provenance') if key in obj},
            'entity_type': ('configured_region' if obj['id'] in self.compiler.configured_regions
                            else 'observed_entity')}
            for obj in context['observation']['objects']}
        for identifier, measured in current_table_measurements(context['observation']).items():
            current_entities[identifier] = {**current_entities.get(identifier, {}), **measured}
        from ..task_binding_images import current_instance_locations

        locations = current_instance_locations(context['observation'])
        choices = destination_candidates(list(current_entities.values()), locations)
        observed_destination = (task_spec.destination_ref is not None
                                and task_spec.destination_ref not in self.compiler.configured_regions)
        requests = []
        required = []
        if task_spec.source_ref is not None:
            source = context['observation'].get('scene_entities', {}).get(task_spec.source_ref, {})
            description = json.dumps({key: source[key] for key in ('kind', 'color', 'description')
                                      if key in source}, ensure_ascii=False)
            requests.append(JudgmentRequest(name='source_identity', instructions=(
                'Is entity ' + task_spec.source_ref + ' (' + description + ') the ORIGINAL supporting surface, '
                'containing receptacle or source region from which input objects ' + json.dumps(targets)
                + ' matching ' + repr(task_spec.target_selector) + ' are to be taken in the original instruction? '
                'Source means the support/container/region, NOT the input material or the object to be moved. '
                'An input object cannot be its own source. For an object inside a box, assess the box as '
                'the source, not the contents or the table under that box. '
                'Check only this origin relationship in CURRENT RGB; do not judge final delivery, processing or CAD. '
                'Use reference boxes to locate the proposed entity; labels, descriptions and a measured plane '
                'alone do not establish the visual answer. Do not judge metric containment or whole-surface visibility.')))
            required.append('source_identity')
        configured_bindings = {}
        for role, identifier in (('destination', task_spec.destination_ref),
                                 ('work_area', task_spec.work_area_ref)):
            if identifier is None:
                continue
            if identifier in self.compiler.configured_regions:
                region = observed_object(context['observation'], identifier)
                support = region.get('placement_reference', {})
                if support.get('source') != 'configured_xy_measured_support':
                    raise ValueError('configured_region_requires_measured_support:' + identifier)
                support_id = self.task_inventory.get('support_id') or support['support_entity_id']
                name = role + '_support_visible'
                configured_bindings[identifier] = {
                    'source': 'operator_configuration_and_measured_support', 'support_entity_id': support_id,
                    'observation_id': context['observation']['observation_id'],
                    'placement_reference': copy.deepcopy(support)}
                support_entity = context['observation'].get('scene_entities', {}).get(support_id, {})
                color = support_entity.get('color')
                surface = ((color + ' ') if color and color != 'unknown' else '') + 'tabletop'
                description = support_entity.get('description', 'supporting surface')
                instructions = (
                    f'Is any part of the {surface} ({support_id}, {description}) '
                    'visible in at least one CURRENT RGB camera? A visible part of its supporting surface is sufficient. '
                    'This question is only surface visibility. The configured coordinate region does not need a visible '
                    'marker or boundary. Do not judge the region position, free placement space, robot reach or task completion.')
            else:
                name = role + '_identity'
                entity = context['observation'].get('scene_entities', {}).get(identifier, {})
                description = json.dumps({key: entity[key] for key in ('kind', 'color', 'description')
                                          if key in entity}, ensure_ascii=False)
                instructions = ('For the active object described by context.object_task.target_selector, '
                    'does its requested ' + role + ' refer to visible entity ' + identifier
                    + ' (' + description + ')? Use context.object_task.' + role + '_ref and instruction '
                    'with the current RGB and identity reference. Different objects may have different destinations. '
                    'The question is ' + role + ' identity, not whether the object is already placed or the motion is safe. '
                    'Do not require an empty surface or whole-surface visibility.')
            if role != 'destination' or identifier in self.compiler.configured_regions:
                requests.append(JudgmentRequest(name=name, instructions=instructions))
                required.append(name)
            if role == 'destination':
                destination_type = ('configured_region' if identifier in self.compiler.configured_regions
                                    else 'observed_entity')
                if destination_type == 'observed_entity':
                    destination_request = destination_question(
                        self.task_spec.instruction, task_spec.target_selector, choices,
                        processing=task_spec.processing)
                else:
                    destination_instructions = (
                        'Original user instruction: ' + json.dumps(self.task_spec.instruction, ensure_ascii=False)
                        + '. For object IDs ' + json.dumps(targets) + ' matching '
                        + repr(task_spec.target_selector) + ', is destination ' + identifier
                        + ' the support, container or configured region that instruction requests? '
                        'Compare current RGB, identity references and current_entities with measured_location_rule, '
                        'including distinctions such as left/right. '
                        + 'This is a configured region; require authorization in the original instruction or '
                           'required processing. An unspecified free table area does not authorize a processing region. '
                        + 'Judge destination selection only. The proposal in object_task is not user authorization. '
                        'The unchanged task still constrains free space, subregion, relative position and pose '
                        'within this destination; perceive_placement_zone and resolve_placement check these before execution.')
                    destination_request = JudgmentRequest(
                        name='destination_requirement_matches', instructions=destination_instructions)
                requests.append(destination_request)
                required.append('destination_requirement_matches')
        for index, identifier in enumerate(targets):
            name = 'target_' + str(index)
            requests.append(target_membership_question(name, identifier, task_spec.target_selector,
                self.task_spec.instruction, processing=task_spec.processing))
            required.append(name)
        if task_spec.scope == 'all':
            requests.append(JudgmentRequest(name='source_complete', instructions=(
                'Is the current source scope sufficiently visible to confirm that the supplied candidate IDs '
                'include every independently distinguishable object matching the original target_selector? '
                'Additional matching objects or unresolved occluded relevant areas prevent a yes. '
                'Do not treat the proposed inventory size as evidence that the source is complete.')))
        if not requests:
            raise ValueError('task_binding_has_no_verifiable_roles')
        result = await self._judge(context, requests, extra={
            'candidate_ids': targets, 'configured_regions': self.compiler.configured_regions,
            'object_task': task_spec.model_dump(mode='json'),
            'proposed_task_binding': {
                'status': 'proposed_not_yet_confirmed',
                'original_instruction': self.task_spec.model_dump(mode='json').get('instruction'),
                'target_ids': targets, 'destination_id': task_spec.destination_ref,
                'destination_type': ('configured_region'
                    if task_spec.destination_ref in self.compiler.configured_regions else 'observed_entity')
                    if task_spec.destination_ref is not None else None},
            'measured_location_rule': (
                'Positions are metres in robot world: +X is forward, +Y is robot LEFT, -Y is robot RIGHT, '
                '+Z is up. Camera image left/right depends on viewpoint and is not generally robot left/right; '
                'respect any perspective explicitly named in the original instruction. visible_centroid_world '
                'is the measured visible surface centroid, not the full object centre or a free placement. '
                'Use observation_provenance to distinguish current and historical measurements. '
                'Configured region positions are supplied coordinates, not observed physical markers. '
                'Locations help interpret the requested side; they do not alone prove identity or safety.'
                ' Default task convention: unqualified left/right mean ROBOT/world left/right, '
                'not the left/right edge of a camera image. An explicitly named camera/image/user '
                'viewpoint in the instruction overrides this default. For robot-relative side, '
                'use the current measured world Y of the already identity-associated entity; '
                'camera images still establish which physical object each ID denotes.'),
            'configured_bindings': configured_bindings,
            'current_entities': list(current_entities.values()),
            'current_instance_locations': locations,
            **({'destination_candidates': choices} if observed_destination else {})})
        if observed_destination:
            bind_destination_choice(result, choices, task_spec.destination_ref)
        if all(result['judgments'][name]['status'] == 'yes' for name in required):
            from ..robot_skills.placement_resolution import digest

            result['confirmed_task_scope'] = {
                'owner_id': self.run_id, 'generation': context['observation']['generation'],
                'task_sha256': digest(task_spec.model_dump(mode='json')), 'input_ids': targets,
                'destination_ref': task_spec.destination_ref,
                'request_hash': result['evidence'].get('request_hash')}
        context['task_binding_evidence'] = copy.deepcopy(result)
        self.state['task_binding_evidence'] = copy.deepcopy(result)
        self.emit('task_binding', 'Decisions 确认任务对象与区域', result)
        judgments = result['judgments']
        for index, identifier in enumerate(targets):
            self.task_progress.records[identifier]['task_membership_evidence'] = {
                **judgments['target_' + str(index)], **result['evidence']}
        if 'source_complete' in judgments:
            self.task_inventory['source_scope_evidence'] = copy.deepcopy(result)
            self.state['source_inventory_resolution'] = {
                'status': 'confirmed' if judgments['source_complete']['status'] == 'yes' else 'unknown',
                'responses_status': task_spec.source_inventory_complete,
                'scope': 'task_source_scope', 'accepted_ids': targets,
                'evidence': copy.deepcopy(result['evidence'])}
        for name in required:
            if judgments[name]['status'] != 'yes':
                raise DiagnosticError('task_binding_unconfirmed:' + name,
                    judgment_details(result, requests, failed_names=[name]))
        source = self.task_progress.scene_entities.get(task_spec.source_ref)
        if source is not None and source.get('kind'):
            # Tables and configured regions can live in the scene inventory
            # rather than the per-object grasp geometry list. Their identity
            # has just been confirmed by Decisions; this grants no clearance.
            for identifier in targets:
                initial = self.task_progress.records[identifier].setdefault('initial_source', {})
                initial['measurement_observation_id'] = context['observation']['observation_id']
                initial['task_source_evidence'] = {
                    **copy.deepcopy(result['evidence']), 'status': 'yes', 'target_id': identifier,
                    'source_id': source['id'], 'source_kind': source['kind']}
        return result

    def _bind_inventory(self, context):
        task, observation = self.task_inventory, context['observation']
        self._require_visible_inventory(observation)
        self._validate_inventory_evidence(observation)
        targets = list(task['source_object_ids'])
        ordered = []
        if self.task_spec.selection_order == 'nearest':
            if self.state['visible_inventory_resolution']['status'] == 'unknown':
                raise DiagnosticError('inventory_order_unconfirmed', {
                    'component': 'inventory_evidence_validation', 'stage': 'bind_inventory',
                    'observation_id': observation['observation_id'],
                    'diagnostic_checks': [{'name': 'selection_order', 'actual': 'nearest',
                        'expected': 'complete source enumeration before global nearest selection'},
                        {'name': 'source_inventory_check',
                         'actual': copy.deepcopy(self.state['visible_inventory_resolution'])}],
                    'known_facts': ['目标身份已独立确认，但当前来源枚举仍为 unknown。'],
                    'unknowns': ['未确认是否还有遗漏的匹配物体，因此不能证明已选物体是全局最近者。'],
                    'requested_order': self.task_spec.selection_order})
            ordered = robot_distance_order([observed_object(observation, identifier) for identifier in targets])
            targets = [row['id'] for row in ordered]
        if self.task_spec.quantity is not None:
            if len(targets) < self.task_spec.quantity:
                raise ValueError('requested_count_exceeds_confirmed_inventory')
            targets = targets[:self.task_spec.quantity]
        if not targets or len(targets) != len(set(targets)):
            raise ValueError('invalid_target_inventory')
        task['selected_ids'] = targets
        task['execution_order'] = {'source': ('measured_rgbd_centroids' if ordered else 'confirmed_inventory_order'),
                                  'selection_order': self.task_spec.selection_order, 'selected_ids': targets,
                                  'candidates': ordered, 'observation_id': observation['observation_id']}
        self.emit('target_execution_order', '按任务约束确定执行顺序', task['execution_order'])
        self.task_progress.requested_count = self.task_spec.quantity
        for identifier, record in self.task_progress.records.items():
            # Visual selector membership and the requested count are separate facts.
            # A matched candidate outside the selected count remains a scene obstacle.
            record['candidate_membership'] = record.get('visual_membership', 'unknown')
            if identifier in task['source_object_ids']:
                record['visual_membership'] = 'required' if identifier in targets else 'excluded'
        self.task_progress.assess_membership({identifier: 'required' if identifier in targets else 'excluded'
                                             for identifier in self.task_progress.records})
        self.task_progress.freeze_inventory(observation)
        for rank, identifier in enumerate(targets, 1):
            self.task_progress.records[identifier]['execution_rank'] = rank
        self.task_progress.configure_task(self.task_spec, targets, object_tasks=self.object_tasks)
        self.sync_inventory(context)
        self.state['facts'].update(inventory_finalized=True, task_assessed=True)
        self.state['task_inventory'] = copy.deepcopy(task)
        context['visual_inventory_reference']['task_inventory'] = copy.deepcopy(task)
        self.emit('inventory_snapshot', '固定任务物体清单', {'checkpoint': 'bind_inventory',
                  'task_progress': self.task_progress.snapshot(), 'model_event_ids': [],
                  'evidence_source': 'decisions_task_binding'})
        return self.task_progress.snapshot()

    def _bind_region(self, observation):
        identifier = self.state.get('destination_id') or self.task_inventory.get('destination_id')
        if identifier.startswith(('table-region:', 'surface-region:')):
            destination = (measured_table_region(observation, identifier.removeprefix('table-region:'))
                if identifier.startswith('table-region:') else
                measured_object_region(observation, identifier.removeprefix('surface-region:')))
            observation['objects'] = [obj for obj in observation['objects'] if obj['id'] != identifier]
            observation['objects'].append(destination)
            observation.setdefault('scene_entities', {})[identifier] = copy.deepcopy(destination)
            self.task_progress.scene_entities[identifier] = copy.deepcopy(destination)
            self.task_progress.bind_shared_work_area(destination, observation)
            request = self.task_progress.placement_requests[(self.task_progress.transfer_index(), identifier)]
            request['source'] = destination['source']
            request['free_placement'] = True
            request['placement_preference'] = self.task_progress.records[self.task_progress.current_id].get(
                'placement_preference', 'clearance')
            return self.task_progress.placement_request(identifier)
        destination = observed_object(observation, identifier)
        self.task_progress.bind_shared_work_area(destination, observation)
        key = (self.task_progress.transfer_index(), identifier)
        request = self.task_progress.placement_requests[key]
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
            if self.active_object_task is None or root != self.active_object_task.target_refs[0]:
                raise ValueError('target_outside_active_object_task:' + identifier)
        if skill == 'confirm_inventory':
            if self.task_progress.inventory_finalized:
                return {'success': True, 'inventory': self.task_progress.snapshot(),
                        'confirmation_reused': True, 'scope': 'immutable_initial_task_selection',
                        'current_scene_reverified': False}
            await self._confirm_task_binding(context)
            snapshot = self._bind_inventory(context)
            return {'success': True, 'inventory': snapshot}
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
            context.setdefault('mapping_lifecycle', {}).update(strategy)
            self.state['mapping_strategy'] = copy.deepcopy(strategy)
            if mapping.exclusion_state == 'observation':
                self.state['facts'].update(scene_ready=False, views_ready=False)
                return await self.build_scene(context, 'observation')
        if skill == 'identify_target' and self.state['facts'].get('target_selected'):
            if context.get('holding') or self.state.get('partial_execution_failure'):
                raise ValueError('target_change_requires_verified_release')
            previous = self.task_progress.records.get(self.state.get('target_id'), {})
            if self.state['facts'].get('grasp_completed') and not previous.get('confirmed_observation_id'):
                raise ValueError('target_change_requires_verified_release')
            pose_facts = {key: self.state['facts'].get(key, False)
                          for key in ('observation_pose_ready', 'grasp_view_ready')}
            self.reset_transfer(context)
            self.state['facts'].update(pose_facts)
            self.state['facts']['task_assessed'] = True
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
            choice = apply_grasp_parameters(context, GraspArguments.model_validate(arguments))
            self.emit('skill_parameters_selected', '本次抓取的模型参数与配置来源', {
                **choice, 'target_id': self.state['target_id'], 'arm': self.state['arm']})
            observation = context['observation']
            destination = observed_object(observation, self.state['destination_id'])
            if destination['kind'] == 'bin':
                self.require_grasp_view(context)
                if not self.state['facts'].get('views_ready'):
                    raise ValueError('current_target_geometry_required')
                context['grasp_placement_request'] = {
                    'object_id': self.state['target_id'],
                    'transport_region': container_transport_region(destination, context['scene'], observation),
                    'orientation_mode': 'preserve',
                    'free_placement': True}
                context['grasps'] = await self.call_algorithm(self.algorithms.generate_grasps,
                    observation, self.state['target_id'], arm=self.state['arm'],
                    placement_request=context['grasp_placement_request'],
                    planning_settings=context['planning_settings'], report_progress=True)
                context['grasps_observation_id'] = observation['observation_id']
                return context['grasps']
        if skill == 'identify_destination':
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
                    'intermediate_destination_id': arguments['destination_id'],
                    'destination_kind': destination['kind'],
                    'final_destination_id': local_task.destination_ref})
                verdict = judgment['judgments']['intermediate_destination_allowed']['status']
                if verdict != 'yes':
                    return {'success': False, 'status': 'failure' if verdict == 'no' else 'unknown',
                            'reason': ('intermediate_destination_rejected' if verdict == 'no'
                                       else 'intermediate_destination_constraint_unconfirmed'),
                            'details': judgment_details(judgment, requests)}
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
            context['destination'] = destination
            context['destination_orientation_mode'] = 'preserve'
            self.task_progress.records[self.state['target_id']].update(
                destination_id=identifier, destination_ref=arguments['destination_id'],
                destination_kind=destination['kind'],
                placement_preference=arguments.get('placement_preference', 'clearance'))
            if destination['kind'] == 'marker':
                self._bind_region(context['observation'])
            self.state['facts']['destination_is_container'] = measured_kind == 'container'
            return {'destination_id': identifier, 'success': True,
                    'placement_preference': arguments.get('placement_preference', 'clearance'),
                    'selection_scope': 'bound_region_not_yet_validated_motion_goal'}
        if skill == 'choose_arm':
            requested = arguments.get('arm', 'auto')
            evaluations = await self.call_algorithm(self.algorithms.evaluate_reachability,
                context['observation'], self.state['target_id'], self.state['destination_id'],
                requested_arm=None if requested == 'auto' else requested)
            available = [arm for arm in ('A', 'B') if evaluations.get(arm, {}).get('reachable')
                         and requested in ('auto', arm)]
            if context.get('holding'):
                available = [arm for arm in available if arm == context['holding']['arm']]
            if not available:
                return {'success': False, 'reason': 'no_reachable_arm', 'evaluations': evaluations}
            if len(available) == 1:
                arm = available[0]
            else:
                from ..agent.decisions import ChoiceRequest
                requests = [ChoiceRequest(name='execution_arm',
                    instructions='Choose for the bound CURRENT target_id and destination_id, not the original '
                                 'processing input. Choose one eligible arm as a planning proposal using the '
                                 'current target and destination, their measured positions and the supplied '
                                 'workspace evaluations. A is the robot left arm, B its right arm. This selection '
                                 'does not certify a grasp pose, placement pose, or collision-free trajectory; '
                                 'subsequent numerical planners must validate those. A pending above-container '
                                 'release pose is expected at this stage and does not by itself prevent selecting '
                                 'an arm. Preserve explicit user arm constraints. Choose unknown only if the '
                                 'supplied task constraints or entity bindings cannot be resolved.',
                    choices=(*available, 'unknown'), selection_mode='proposal')]
                entities = public_observation(context['observation'])['objects']
                judgment = await self._judge(context, requests, extra={
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
            context['reachability_evaluations'] = evaluations
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
            attempt = placement_attempt(context, self.state, observation,
                                        self.registry.contracts['plan_place'].requires)
            blocked = placement_retry_block(context, attempt)
            if blocked:
                raise AlgorithmError('Placement inputs already failed; new numerical evidence required',
                    code='place_no_valid_candidate', details=blocked['placement_evidence'])
            try:
                context['place_plan'] = await self.call_algorithm(self.algorithms.plan_place, observation,
                    self.state['target_id'], self.state['arm'], context['scene'], context['holding'],
                    self.state['destination_id'], planning_settings=context['planning_settings'],
                    placement_request=self._bind_region(context['observation']),
                    placement_resolution=context['placement_resolution'], report_progress=True)
            except AlgorithmError as error:
                if error.code == 'place_no_valid_candidate':
                    record_placement_failure(context, attempt, error.details)
                raise
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
        proof = context.get('visibility_recovery', {})
        if proof.get('observation_id') != capture_id or proof.get('epoch') != self.observation_epoch:
            result = await self._judge(context, visibility_requests(), extra={
                'verification_scope': 'graph_recovery',
                'failed_result': getattr(getattr(self, 'graph_executor', None), 'last_result', None),
                'failed_node': self.state.get('current_step')})
            proof = {'observation_id': context['latest_capture']['observation_id'],
                'epoch': self.observation_epoch, 'answer': visibility_assessment(result['judgments'])}
            context['visibility_recovery'] = proof
        self.emit('visibility_recovery_checked', '补观测移动的视觉原因', proof)
        if not visibility_allows_motion(proof.get('answer', {})):
            raise ValueError('viewpoint_requires_visual_limitation')
        context.pop('visibility_recovery', None)

    async def _request_recovery_judgments(self, context, requests, extra):
        from ..agent.visibility_recovery import VISIBILITY_FIELDS, visibility_assessment

        requested = any(request.name in VISIBILITY_FIELDS for request in requests)
        failed = extra.get('failed_node', {})
        failure = (failed.get('id'), failed.get('skill'), extra.get('failed_result', {}).get('reason'))
        capture_id = context.get('latest_capture', {}).get('observation_id')
        proof = context.get('visibility_recovery', {})
        reuse = (requested and proof.get('observation_id') == capture_id
                 and proof.get('epoch') == self.observation_epoch and proof.get('failure') == failure
                 and 'judgments' in proof)
        if reuse:
            requests = [request for request in requests if request.name not in VISIBILITY_FIELDS]
            extra = {**extra, 'current_visibility_assessment': proof['answer']}
        decision = await self._judge(context, requests, extra=extra) if requests else {'judgments': {}}
        if reuse:
            decision['judgments'].update(proof['judgments'])
            self.emit('visibility_recovery_reused', '复用同一画面的视觉缺口判断', {
                'observation_id': capture_id, 'failure': failure, 'answer': proof['answer']})
        elif requested:
            judgments = {name: decision['judgments'][name] for name in VISIBILITY_FIELDS}
            context['visibility_recovery'] = {
                'observation_id': context['latest_capture']['observation_id'],
                'epoch': self.observation_epoch, 'failure': failure,
                'answer': visibility_assessment(judgments), 'judgments': judgments}
        return decision

    async def _judge(self, context, requests, *, extra=None, include_task=True, reference=None):
        capture = context['latest_capture']
        if context.get('capture_epochs', {}).get(capture['observation_id']) != self.observation_epoch:
            xy_proposal = ((extra or {}).get('verification_scope') == 'placement_xy_proposal'
                and (extra or {}).get('scene_token') == context.get('scene', {}).get('scene_token')
                and bool((extra or {}).get('scene_token'))
                and all(isinstance(item, ChoiceRequest) and item.selection_mode == 'proposal' for item in requests))
            if xy_proposal:
                # This selects a location in an already verified region. Old RGB
                # is explicitly historical context, never a current visual fact.
                extra = {**extra, 'image_evidence_scope': 'historical_scene_for_xy_proposal'}
            elif (extra or {}).get('verification_scope') not in {'graph_recovery', 'graph_recovery_continuation'}:
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
            if judgment_context.get('verification_scope') in {'graph_recovery', 'graph_recovery_continuation'}:
                from ..recovery_context_compaction import compact_recovery_context

                judgment_context = compact_recovery_context(judgment_context)
            return await self.call_algorithm(self.judgments.judge, capture, requests=requests,
                context=judgment_context,
                reference=context['visual_inventory_reference'] if reference is None else reference)
        finally:
            done.set()

    def _record_post_delivery_observation(self, context, identifier):
        from ..agent.completion import has_confirmed_delivery

        requirement = self.environment_assumptions.get('completion_requirements', {}).get('after_each_object', {})
        if not requirement:
            return
        feedback = self.simulation.robot_snapshot()
        current = identifier
        while self.task_progress.records[current].get('output_entity_id'):
            current = self.task_progress.records[current]['output_entity_id']
        record = self.task_progress.records[current]
        if (identifier not in self.task_progress.completed_input_ids()
                and not has_confirmed_delivery(record)):
            raise ValueError('policy_target_not_verified:' + identifier)
        delivery = {'entity_id': current, 'confirmed_observation_id': record.get('confirmed_observation_id'),
                    'current_region': record.get('current_region'), 'transfer_index': record.get('transfer_index')}
        previous = self.task_progress.records[identifier].get('post_delivery_observation', {})
        if (previous.get('status') == 'verified' and previous.get('generation') == feedback.get('generation')
                and previous.get('delivery') == delivery):
            return  # Later robot motion does not undo this verified historical requirement.
        capture = context.get('latest_capture', {})
        observation = context.get('observation', {})
        observation_id = observation.get('observation_id')
        if requirement.get('robot_pose') == 'observation' and (
                not self.state['facts'].get('observation_pose_ready')
                or not self.is_observation_pose(context, feedback)):
            raise ValueError('object_observation_pose_not_reached:' + identifier)
        if requirement.get('observe_inventory') and (
                not self.state['facts'].get('observed') or not observation_id
                or capture.get('observation_id') != observation_id
                or capture.get('generation') != feedback.get('generation')
                or context.get('capture_epochs', {}).get(observation_id) != self.observation_epoch
                or requirement.get('robot_pose') == 'observation' and not self.is_observation_pose(context, capture)):
            raise ValueError('object_post_delivery_observation_missing:' + identifier)
        evidence = {'status': 'verified', 'input_id': identifier,
            'delivery': delivery,
            'source': 'actual_joint_feedback_and_completed_inventory_observation',
            'observation_id': observation_id, 'generation': feedback.get('generation'),
            'action_epoch': self.observation_epoch, 'required_pose': requirement.get('robot_pose'),
            'actual_joints_rad': {arm: copy.deepcopy(feedback['arms'][arm]['joints_rad']) for arm in ('A', 'B')},
            'inventory_observed': bool(requirement.get('observe_inventory'))}
        self.task_progress.records[identifier]['post_delivery_observation'] = evidence
        record['post_delivery_observation'] = evidence
        if record.get('transfers'):
            record['transfers'][-1]['post_delivery_observation'] = copy.deepcopy(evidence)
        self.task_progress.ledger_revision += 1
        self.sync_inventory(context)
        self.emit('object_post_delivery_observation', '物体交付后的回位与观测证据', evidence)

    async def _execute_graph(self, context):
        active_input = None

        async def propose_recovery(node, result, history, bindings):
            from ..agent.recovery import RecoveryPlanner, recovery_measurements
            from .recovery_evidence import (
                blocked_operations,
                operation_evidence,
                operation_signature,
                operation_signatures,
                reusable_observation_result,
            )

            if self.state.get('partial_execution_failure'):
                return {'action': 'stop', 'reason': 'unresolved_physical_side_effect', 'evidence': result}
            async def request(requests, extra):
                return await self._request_recovery_judgments(context, requests, extra)
            planner = RecoveryPlanner(request, skill_catalog(self.registry), self.registry.validate_arguments)
            def placement_block():
                attempt = placement_attempt(context, self.state, self.measured_state(context),
                    self.registry.contracts['plan_place'].requires, node_id=node['id'])
                return placement_retry_block(context, attempt)

            rejected = []
            seen = context.setdefault('graph_recovery_signatures', set())
            for _ in range(3):
                evidence = operation_evidence(context, node['id'], result.get('reason'), self.state['facts'],
                    self.task_progress.completed_input_ids(), failed_skill=node.get('skill'), target_id=self.state.get('target_id'))
                capture_id = context.get('latest_capture', {}).get('observation_id')
                capture_current = context.get('capture_epochs', {}).get(capture_id) == self.observation_epoch
                unavailable = blocked_operations(seen, evidence, capture_current=capture_current)
                if node.get('skill') == 'plan_place':
                    # Pixels/facts alone cannot establish whether numerical goals changed.
                    unavailable = [row for row in unavailable if row.get('action') != 'retry']
                    blocked = placement_block()
                    if blocked:
                        unavailable.append(blocked)
                proposal = await planner.propose(node, public_summary(result), public_summary(history[-8:]), bindings,
                    facts=copy.deepcopy(self.state['facts']),
                    inventory=list(context.get('visual_inventory_reference', {}).get('entities', {})) +
                              list(self.compiler.configured_regions),
                    context={'ledger': public_summary(self.task_progress.snapshot()),
                        'completion_requirements': copy.deepcopy(self.environment_assumptions.get(
                            'completion_requirements', {})),
                        'active_object_task': self.active_object_task.model_dump(mode='json') if self.active_object_task else None,
                        'original_task': self.task_spec.model_dump(mode='json'),
                        'rejected_proposals': unavailable + rejected,
                        'current_measurements': public_summary(recovery_measurements(context)),
                        'reusable_recovery_result': reusable_observation_result(
                            context, history, self.observation_epoch, self.registry.contracts) or {},
                        'current_observation_id': context.get('latest_capture', {}).get('observation_id')})
                if proposal['action'] == 'use_recovery_result':
                    current = reusable_observation_result(context, history, self.observation_epoch,
                                                          self.registry.contracts)
                    if not current or current['node_id'] != proposal.get('source_node_id'):
                        proposal = {'action': 'rejected', 'reason': 'recovery_result_no_longer_current'}
                if proposal['action'] == 'skill':
                    try:
                        self.registry.validate_arguments(proposal['skill'], proposal['arguments'])
                    except ValueError as error:
                        proposal.update(action='rejected', reason=str(error))
                if proposal['action'] not in {'retry', 'skill', 'rejected'}:
                    self.emit('policy_recovery_decision', 'Decisions 生成恢复 node/edge', {
                        'node_id': node['id'], 'proposal': proposal, 'history': public_summary(history)})
                    return proposal
                evidence = operation_evidence(context, node['id'], result.get('reason'), self.state['facts'],
                    self.task_progress.completed_input_ids(), failed_skill=node.get('skill'), target_id=self.state.get('target_id'))
                signature = operation_signature(evidence, proposal)
                signatures = operation_signatures(evidence, proposal)
                if node.get('skill') == 'plan_place' and proposal['action'] == 'retry':
                    blocked = placement_block()
                    if blocked:
                        proposal = {**proposal, **blocked, 'action': 'rejected'}
                elif signature in seen:
                    proposal = {**proposal, 'action': 'rejected', 'reason': 'identical_evidence_and_operation_no_progress'}
                seen.update(signatures)
                self.emit('policy_recovery_decision', 'Decisions 生成恢复 node/edge', {
                    'node_id': node['id'], 'proposal': proposal, 'history': public_summary(history)})
                if proposal['action'] != 'rejected':
                    if proposal.get('skill') == 'move_to_observe':
                        from ..agent.completion import required_observation_return

                        required_return = required_observation_return(
                            self.environment_assumptions.get('completion_requirements', {}), result.get('reason'))
                        context['pending_required_return'] = required_return
                        context['pending_visibility_motion'] = not required_return
                    return proposal
                rejected.append({key: proposal[key] for key in ('action', 'skill', 'arguments', 'reason') if key in proposal})
            return {'action': 'stop', 'reason': 'decisions_recovery_repeated_invalid_proposals', 'evidence': rejected}

        async def enter_object_task(task):
            self.active_object_task = TaskSpec.model_validate(task) if task is not None else None
            self.state['active_object_task'] = copy.deepcopy(task)


        def confirmed_targets():
            if not self.task_progress.inventory_finalized:
                raise ValueError('task_inventory_not_confirmed')
            completed = self.task_progress.completed_input_ids()
            return [identifier for identifier in self.task_inventory['selected_ids'] if identifier not in completed]

        async def before_target(identifier, index):
            nonlocal active_input
            if active_input is not None and active_input != identifier:
                shared_evidence = {key: self.state['facts'].get(key, False)
                                   for key in ('observed', 'observation_pose_ready', 'grasp_view_ready')}
                self.reset_transfer(context)
                self.state['facts'].update(**shared_evidence, task_assessed=True)
            active_input = identifier

        async def after_target(identifier):
            completed = self.task_progress.completed_input_ids()
            if identifier not in completed:
                return {'status': 'unknown', 'success': False,
                        'reason': 'policy_target_not_verified:' + identifier}
            try:
                self._record_post_delivery_observation(context, identifier)
            except ValueError as error:
                if not str(error).startswith(('object_post_delivery_observation_missing:',
                                              'object_observation_pose_not_reached:')):
                    raise
                return {'status': 'unknown', 'success': False, 'reason': str(error),
                        'required_obligation': copy.deepcopy(self.environment_assumptions.get(
                            'completion_requirements', {}).get('after_each_object', {}))}

        async def invoke(name, args):
            try:
                return await self._step(name, context, args)
            except (ValueError, AlgorithmError, PerceptionError) as error:
                code = getattr(error, 'code', None) or str(error)
                unknown = code.startswith(('visual_identity_unknown', 'task_binding_unconfirmed',
                    'task_target_geometry_unknown', 'source_inventory_unknown', 'observed_support_unknown',
                    'observed_support_current_geometry_required', 'observed_instance_missing_or_ambiguous',
                    'task_completion_unconfirmed'))
                details = exception_details(error)
                if (name == 'resolve_placement' and code == 'placement_no_candidate'
                        and 'effective_arguments' in details and self.state.get('active_policy_node')):
                    from .recovery_evidence import operation_evidence, record_failed_operation

                    evidence = operation_evidence(context, self.state['active_policy_node']['node_id'], code,
                        self.state['facts'], self.task_progress.completed_input_ids())
                    record_failed_operation(context.setdefault('graph_recovery_signatures', set()), evidence, name,
                        [details['effective_arguments'], {'preference': 'decide'}, {}])
                if code.startswith(('task_binding_unconfirmed', 'task_completion_unconfirmed')) and any(
                        check.get('status') == 'no' for check in details.get('diagnostic_checks', [])
                        if check.get('required') is True):
                    unknown = False  # A negative visual answer is not missing evidence.
                return {'success': False, 'status': 'unknown' if unknown else 'failure',
                        'reason': getattr(error, 'code', None) or str(error),
                        'error_code': getattr(error, 'code', None),
                        'message': str(error)[:1000], 'details': details,
                        'partial_execution': copy.deepcopy(self.state.get('current_step'))}

        async def judge(name, bindings):
            from ..agent.completion import completion_obligations

            declared = {node.condition for scope in [graph, *graph.subgraphs]
                        for node in scope.nodes if node.type == 'router'}
            if name not in declared:
                raise ValueError('undeclared_policy_judgment:' + name)
            requests = [JudgmentRequest(name='condition', instructions=name)]
            result = await self._judge(context, requests, extra={
                'verification_scope': 'policy_condition', 'policy_bindings': copy.deepcopy(bindings),
                'verified_obligations': completion_obligations(self.task_progress.snapshot())})
            status = result['judgments']['condition']['status']
            if status not in {'yes', 'no'}:
                if status == 'unknown':
                    # Preserve the router's third outcome so its explicit unknown edge can run.
                    return None
                raise DiagnosticError('policy_judgment_unknown:judgment:' + name,
                                      judgment_details(result, requests))
            return status == 'yes'

        graph = self.policy_graph
        completed = self.task_progress.completed_input_ids()
        pending = [identifier for identifier in self.task_inventory.get('selected_ids', []) if identifier not in completed]
        native_options = {'propose_recovery': propose_recovery,
            'reusable_result_skills': {name for name, contract in self.registry.contracts.items()
                if contract.recovery_result_reusable and not contract.invalidates_observation},
            'reusable_result_ignored_arguments': {name: contract.recovery_result_ignored_arguments
                for name, contract in self.registry.contracts.items() if contract.recovery_result_reusable},
            'repeatable_skills': {name for name, contract in self.registry.contracts.items() if contract.recovery_retry_safe}}

        def emit_graph(kind, title, data):
            if kind == 'policy_graph_patch':
                self.state['policy_graph'] = copy.deepcopy(data['policy_graph'])
            elif kind == 'policy_node_started':
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
            check_cancelled=self.check_generation, emit=emit_graph, **native_options,
            step_limit=max(4096, min(131072, len(pending) * 128 * 3 + 128)))
        await self.graph_executor.run(graph)

    async def _complete(self, context):
        from ..agent.completion import COMPLETION_MILESTONE_RULE, completion_obligations

        completed = self.task_progress.completed_input_ids()
        if set(completed) != set(self.task_inventory['selected_ids']):
            raise ValueError('policy_finished_with_pending_targets')
        pose_proof = None
        if self.task_spec.completion_pose == 'observation':
            feedback = self.simulation.robot_snapshot()
            if not self.is_observation_pose(context, feedback):
                raise DiagnosticError('completion_pose_unconfirmed', {
                    'component': 'robot_feedback', 'expected_pose': 'observation',
                    'actual_feedback': feedback})
            pose_proof = {'success': True, 'expected_pose': 'observation', 'source': 'actual_joint_feedback'}
        if self.task_spec.completion_pose == 'initial':
            from ..robot_skills.return_pose import initial_pose_verdict
            if context.get('holding') or self.holding_state is not None:
                raise DiagnosticError('completion_pose_unconfirmed', {'reason': 'empty_hands_unconfirmed'})
            pose_proof = initial_pose_verdict(context['initial_joint_reference'], self.simulation.robot_snapshot())
            self.state['completion_pose_evidence'] = pose_proof
            self.state['facts']['at_initial_pose'] = bool(pose_proof['success'])
            if not pose_proof['success']:
                raise DiagnosticError('completion_pose_unconfirmed', pose_proof)
        previous = self.state.get('completion_evidence', {}).get('judgments', {})
        mode = 'fresh' if any(row.get('status') != 'yes' for row in previous.values()) else 'reuse_current'
        capture = await self.capture_scene_observation(context, {'capture_mode': mode}, purpose='task_completion')
        completion_key = (capture['observation_id'], self.observation_epoch, self.task_progress.ledger_revision)
        if (previous and any(row.get('status') != 'yes' for row in previous.values())
                and context.get('completion_request_key') == completion_key):
            raise ValueError('task_completion_unchanged_evidence')
        self.record_observation(capture, 'task_completion')
        requirements = '; '.join(self.task_spec.completion_conditions)
        lineage = ('For original input IDs ' + json.dumps(self.task_inventory['selected_ids'])
                   + ' and their independently bound processing output lineages, ')
        requests = [JudgmentRequest(name='goal_satisfied', instructions=(
            lineage + 'Do the CURRENT images and recorded per-object verified evidence support the user completion '
            'conditions? ' + requirements + ' Do not infer success from an empty work queue. '
            'For requested robot posture use verified_robot_pose numerical feedback, not visual joint estimation.'
            + COMPLETION_MILESTONE_RULE))]
        # Preserve the model's conditions verbatim, but expose which obligation
        # failed instead of reporting only one opaque conjunction probability.
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
        if self.task_spec.scope == 'all':
            requests.append(JudgmentRequest(name='source_complete', instructions=(
                'Is the source scope sufficiently observed to confirm no unprocessed objects matching '
                'target_selector remain, including initially missed targets? Occluded relevant areas '
                'or ambiguous membership are uncertain, not empty. Non-target objects may remain.')))
        result = await self._judge(context, requests, extra={
            'verified_obligations': completion_obligations(self.task_progress.snapshot()),
            'verified_robot_pose': pose_proof, 'verification_scope': 'task_completion'})
        self.state['completion_evidence'] = result
        context['completion_request_key'] = completion_key
        if any(row['status'] != 'yes' for row in result['judgments'].values()):
            raise DiagnosticError('task_completion_unconfirmed', judgment_details(result, requests))
        self.state['status'] = 'succeeded'

    async def _run(self, run_id, initial):
        from ..robot_skills.return_pose import initial_joint_reference
        context = {'initial': initial, 'planning_settings': copy.deepcopy(self.state['planning_settings']),
                   'initial_joint_reference': initial_joint_reference(initial),
                   'visual_call_budget': VisualCallBudget(limit=0)}
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
                                   inventory_finalized=False, inventory_ready=False, task_assessed=False,
                                   observed=False, task_verified=False)
        self.emit('run_start', '执行物体清单策略图', {'instruction': self.state['instruction']})
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
            self.emit('skill_error', '策略执行需要处理', {'code': record['error_code'],
                      'details': record['error_details'], 'failure_report': record['failure_report']})
        finally:
            self.state['task_progress'] = self.task_progress.snapshot()
            self.state['current_step'] = None
            self.state['metrics'] = {'run_duration_s': time.monotonic() - self.run_started_at,
                                     'responses_call_count': self.compiler.budget.used}
            self.emit('run_end', '任务结束', self.state)
            if self.experiment is not None:
                self.experiment.stop(self.state)

"""Compile and validate a model-authored graph before any task skill executes."""
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock

from pydantic import ValidationError

from ..visual_api import VisualAPI, VisualInventoryError
from ..visual_inventory import image_part, reference_content
from .schema import (
    GraphRevision,
    InitialPlan,
    object_task_specs,
    response_schema,
    skill_catalog,
    validate_graph,
)
from .sensor_context import rgbd_context

BOOTSTRAP_FACTS = {
    'observed': False, 'inventory_ready': False,
    'observation_pose_ready': True, 'grasp_view_ready': False, 'needs_place': True,
    'target_selected': False, 'destination_selected': False, 'arm_selected': False,
    'source_support_current': False,
    'scene_ready': False, 'collision_map_initialized': False,
    'grasp_held': False, 'recovery_needed': False, 'task_verified': False,
    'placement_zone_ready': False, 'placement_resolved': False,
}

MAX_PLAN_REVISIONS = 2


@dataclass
class ResponsesBudget:
    """Reserve one initial proposal and at most two validation-driven revisions."""
    used: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def reserve(self, *, revision=False):
        with self._lock:
            allowed = 0 < self.used <= MAX_PLAN_REVISIONS if revision else self.used == 0
            if not allowed:
                raise VisualInventoryError('responses_budget_exhausted')
            self.used += 1


class ResponsesCompiler(VisualAPI):
    endpoint = '/v1/responses'
    source = 'openai_responses_proposal'
    error_prefix = 'responses'
    maximum_timeout_s = 600.

    def __init__(self, *, registry=None, configured_regions=None, known_assets=None,
                 initial_facts=None, scene_context=None, budget=None, skill_detail='compact', memory_reference=None,
                 compilation_memory=None, **kwargs):
        super().__init__(**kwargs)
        if registry is None:
            from ..skills import Registry
            registry = Registry()
        self.registry = registry
        self.memory_reference = deepcopy(memory_reference or {})
        self.compilation_memory = compilation_memory
        self.skill_detail = skill_detail
        self.configured_regions = deepcopy(configured_regions or {})
        self.scene_context = deepcopy(scene_context or {})
        self.known_assets = {}
        for identifier, asset in (known_assets or {}).items():
            size = asset.get('size_m')
            if (not isinstance(identifier, str) or not identifier or not isinstance(asset.get('label'), str)
                    or not isinstance(size, (list, tuple)) or len(size) != 3
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                           for value in size)):
                raise ValueError('invalid_configured_asset')
            self.known_assets[identifier] = {'label': asset['label'], 'size_m': list(size)}
        self.initial_facts = dict(BOOTSTRAP_FACTS if initial_facts is None else initial_facts)
        self.budget = budget if budget is not None else ResponsesBudget()
        self.original_task = None
        self._initial_key = None
        self._initial_plan = None
        self.last_evidence = {}
        self._graph_proposal = None

    def reset(self):
        """Start a new task without carrying its budget, identities, goal or cached responses."""
        self.budget = ResponsesBudget()
        self.original_task = None
        self._initial_key = self._initial_plan = None
        self.last_evidence = {}
        self._graph_proposal = None
        self.cache.clear()

    def request(self, payload, capture, purpose, parse, *, model_call=None):
        audit = self.audit
        self._graph_proposal = None

        def record_compilation(event):
            if self._graph_proposal is not None:
                event['graph_proposal'] = deepcopy(self._graph_proposal)
            if audit is not None:
                audit(event)

        self.audit = record_compilation
        try:
            return super().request(payload, capture, purpose, parse, model_call=model_call)
        finally:
            self.audit = audit

    def compile(self, capture, *, instruction, configured_regions=None):
        key = (capture['observation_id'], capture.get('generation'), instruction)
        if self._initial_plan is not None and key == self._initial_key:
            return self._initial_plan.model_copy(deep=True)
        catalog = skill_catalog(self.registry)
        regions = self.configured_regions if configured_regions is None else configured_regions
        context = {'instruction': instruction, 'skill_catalog': catalog,
            'skill_instructions': self._skill_instructions(catalog), 'configured_regions': regions,
            'known_assets': self.known_assets, 'initial_facts': self.initial_facts,
            'scene_context': self.scene_context,
            'rgbd_context': rgbd_context(capture),
            'object_memory': {identifier: {key: value for key, value in entity.items() if key in
                {'id', 'kind', 'color', 'description', 'reference_observation_id', 'reference_camera_id',
                 'bbox', 'current_region', 'visibility', 'input_entity_id', 'output_entity_id', 'processing_history'}}
                for identifier, entity in self.memory_reference.get('entities', {}).items()}}
        plan = self._generate(capture, context)
        if plan.task.instruction != instruction:
            raise VisualInventoryError('immutable_task_instruction_changed')
        self.original_task = plan.task
        self._initial_plan, self._initial_key = plan.model_copy(deep=True), key
        return plan


    def _skill_instructions(self, catalog):
        documents = getattr(self.registry, 'documents', {})
        return {name: documents[name] for name in catalog if name in documents}

    def _generate(self, capture, context):
        if set(capture.get('views', {})) != {'1', '2', '3'}:
            raise VisualInventoryError('responses_requires_three_rgb_views')
        catalog = skill_catalog(self.registry)
        from .graph_prompt import FINAL_GRAPH_CHECK, graph_context
        from .graph_authoring import expand_graph_routes, expand_scope_routes

        model = InitialPlan
        context = graph_context(context, skill_detail=self.skill_detail)
        if self.compilation_memory is not None:
            context['compilation_memory'] = self.compilation_memory.snapshot(catalog)
        schema = response_schema(model, catalog)
        root_task = deepcopy(schema['$defs']['TaskSpec'])
        root_task['properties']['instruction']['enum'] = [context['instruction']]
        schema['$defs']['InitialTaskSpec'] = root_task
        schema['properties']['task'] = {'$ref': '#/$defs/InitialTaskSpec'}
        schema['$defs']['EntityProposal']['properties']['asset_id'] = (
            {'anyOf': [{'type': 'string', 'enum': list(self.known_assets)}, {'type': 'null'}]}
            if self.known_assets else {'type': 'null'})
        known_ids = list(context.get('object_memory', {}))
        schema['$defs']['EntityProposal']['properties']['existing_id'] = (
            {'anyOf': [{'type': 'string', 'enum': known_ids}, {'type': 'null'}]}
            if known_ids else {'type': 'null'})
        self.budget.reserve()
        content = [{'type': 'input_text', 'text': json.dumps({
            **context, 'observation_id': capture['observation_id'], 'generation': capture.get('generation')}, ensure_ascii=False)}]
        content.extend(reference_content(self.memory_reference))
        for camera in ('2', '3', '1'):
            rgb = capture['views'][camera]['rgb']
            height, width = rgb.shape[:2]
            content.extend([{'type': 'input_text', 'text': json.dumps({
                'camera_id': camera, 'width_px': width, 'height_px': height})}, image_part(rgb)])
        content.append({'type': 'input_text', 'text': FINAL_GRAPH_CHECK})
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}],
                   'reasoning': {'effort': 'high'}, 'max_output_tokens': 64000,
                   'text': {'format': {'type': 'json_schema', 'name': 'robot_policy',
                                      'strict': True, 'schema': schema}}}
        frozen_declaration = None
        revision_base_graph = None
        rejected_text = ''

        def validation_details(error):
            if isinstance(error, ValidationError):
                return [{'location': item['loc'], 'message': message}
                        for item in error.errors(include_input=False, include_url=False)
                        for message in getattr(item.get('ctx', {}).get('error'), 'issues', [item['msg']])]
            return getattr(error, 'issues', [str(error)])

        def validate_control_flow(result):
            from .schema import validate_batch_policy

            validate_batch_policy(result.graph, result.task)
            validation_catalog = deepcopy(catalog)
            final_facts = {}
            if result.task.supported:
                if result.task.completion_pose:
                    pose_fact = {'initial': 'at_initial_pose',
                                 'observation': 'observation_pose_ready'}[result.task.completion_pose]
                    final_facts[pose_fact] = True
                if 'verify_task_completion' in validation_catalog:
                    validation_catalog['verify_task_completion']['requires'].update(final_facts)
                    final_facts['task_verified'] = True
            validate_graph(result.graph, validation_catalog,
                           initial_facts=context.get('initial_facts', {}),
                           require_completion=result.task.supported,
                           required_final_facts=final_facts)

        def parse(body, evidence):
            nonlocal rejected_text
            if body.get('status') != 'completed':
                raise VisualInventoryError('responses_incomplete', evidence={
                    'response_status': body.get('status'),
                    'incomplete_details': body.get('incomplete_details'), 'usage': body.get('usage', {})})
            texts = [part['text'] for item in body.get('output', []) if item.get('type') == 'message'
                     for part in item.get('content', []) if part.get('type') == 'output_text']
            rejected_text = ''.join(texts)
            revision_allowed = False
            parsing_revision = frozen_declaration is not None
            rejected_revision = None
            try:
                if frozen_declaration is not None:
                    edits = json.loads(rejected_text)
                    if edits.get('root') is not None:
                        edits['root'] = expand_scope_routes(edits['root'], catalog=catalog)
                    edits['subgraphs'] = [expand_scope_routes(scope, catalog=catalog) for scope in edits.get('subgraphs', [])]
                    revision = GraphRevision.model_validate(edits)
                    self._graph_proposal = {**frozen_declaration, 'graph': revision.apply_to(revision_base_graph)}
                    parsing_revision = False
                else:
                    self._graph_proposal = json.loads(rejected_text)
                revision_allowed = True
                self._graph_proposal['graph'] = expand_graph_routes(self._graph_proposal['graph'], catalog=catalog)
                result = model.model_validate(self._graph_proposal)
                revision_allowed = False
                if result.task.instruction != context['instruction']:
                    raise ValueError('immutable_task_instruction_changed')
                item_tasks = object_task_specs(result.graph)
                for name in (*result.task.processing, *(skill for task in item_tasks.values()
                                                        for skill in task.processing)):
                    if name not in catalog:
                        raise ValueError('unknown_processing_skill:' + name)
                    if not catalog[name].get('processing_effect'):
                        raise ValueError('skill_is_not_processing_effect:' + name)
                required_pose = context.get('scene_context', {}).get('completion_requirements', {}).get('robot_pose')
                if result.task.supported and required_pose and result.task.completion_pose != required_pose:
                    raise ValueError('requested_completion_pose_missing:' + required_pose)
                revision_allowed = True
                validate_control_flow(result)
                revision_allowed = False
                for entity in result.entities:
                    if (entity.geometry_policy == 'unknown_obb'
                            and entity.geometry_policy_instruction not in context['instruction']):
                        raise ValueError('geometry_policy_instruction_not_supplied:' + entity.ref)
                    if entity.ref in context['configured_regions']:
                        raise ValueError('configured_region_is_not_visual_entity:' + entity.ref)
                    if entity.asset_id is not None and entity.asset_id not in self.known_assets:
                        raise ValueError('unknown_configured_asset:' + entity.asset_id)
                known = {entity.ref for entity in result.entities} | set(context['configured_regions'])
                for task in (result.task, *item_tasks.values()):
                    for reference in (task.source_ref, task.destination_ref, task.work_area_ref):
                        if reference is not None and reference not in known:
                            raise ValueError('unknown_task_reference:' + reference)
            except ValueError as error:
                details = validation_details(error)
                paths = list(getattr(error, 'counterexamples', []))
                if parsing_revision:
                    revision_allowed = True
                    self._graph_proposal = {**frozen_declaration, 'graph': deepcopy(revision_base_graph)}
                    rejected_revision = rejected_text
                elif isinstance(error, ValidationError):
                    revision_allowed = all(item['location'] and item['location'][0] == 'graph' for item in details)
                rejected = self._graph_proposal or {'invalid_json_text': rejected_text}
                if (isinstance(error, ValidationError) and revision_allowed and all(
                        item['message'].removeprefix('Value error, ').split(':', 1)[0]
                        in {'uncalled_subgraph', 'graph_unreachable_nodes'} for item in details)):
                    from ..core.topology import reachable_graph_for_diagnostics

                    try:
                        diagnostic = {**rejected, 'graph': reachable_graph_for_diagnostics(rejected['graph'])}
                        validate_control_flow(model.model_validate(diagnostic))
                    except ValueError as diagnostic_error:
                        additional = validation_details(diagnostic_error)
                        details.extend(additional)
                        paths.extend(getattr(diagnostic_error, 'counterexamples', []))
                        if isinstance(diagnostic_error, ValidationError):
                            revision_allowed = all(item['location'] and item['location'][0] == 'graph'
                                                   for item in additional)
                if self.compilation_memory is not None and revision_allowed:
                    self.compilation_memory.record_failure(catalog=catalog,
                        request_hash=evidence['request_hash'], observation_id=capture['observation_id'],
                        issues=details, graph=rejected.get('graph', {}),
                        counterexamples=paths)
                raise VisualInventoryError('responses_invalid_plan',
                    evidence={'validation_errors': details, 'rejected_proposal': rejected,
                              'validation_paths': paths,
                              'rejected_revision': rejected_revision,
                              'graph_revision_allowed': revision_allowed}) from None
            self.last_evidence = {**evidence, 'responses_calls_used': self.budget.used,
                                  'responses_call_limit': 1 + MAX_PLAN_REVISIONS}
            return result

        initial_payload = deepcopy(payload)
        model_call = {'kind': 'initial_compile', 'attempt': 0}
        validation_history = []
        for revision in range(MAX_PLAN_REVISIONS + 1):
            try:
                return self.request(payload, capture, 'compile_task_revision' if revision else 'compile_task',
                                    parse, model_call=model_call)
            except VisualInventoryError as error:
                if (str(error) != 'responses_invalid_plan' or not error.evidence.get('graph_revision_allowed')
                        or revision == MAX_PLAN_REVISIONS):
                    raise
                rejected = error.evidence['rejected_proposal']
                if frozen_declaration is None:
                    frozen_declaration = {name: deepcopy(rejected[name]) for name in ('entities', 'task')}
                revision_base_graph = deepcopy(rejected['graph'])
                validation_history.append({'attempt': revision, 'request_hash': error.evidence['request_hash'],
                                           'validation_errors': error.evidence['validation_errors']})
                feedback = {
                    'revision': revision + 1,
                    'previous_request_hash': error.evidence['request_hash'],
                    'validation_errors': error.evidence['validation_errors'],
                    'validation_paths': error.evidence.get('validation_paths', []),
                    'previous_validation_attempts': deepcopy(validation_history),
                    'rejected_proposal': rejected,
                    'rejected_revision': error.evidence.get('rejected_revision'),
                    'instruction': ('Return scope edits using root, subgraphs and remove_subgraph_ids. '
                        'root=null keeps the current root. Otherwise supply its complete replacement. '
                        'subgraphs contains complete bodies only for scopes to replace or add. '
                        'Every omitted subgraph is preserved. Remove a definition only by explicitly naming '
                        'it in remove_subgraph_ids and repairing its callers. The application applies only '
                        'these declared edits and validates the resulting complete graph. '
                        'Entities and the root task are frozen application inputs; do not output or rewrite them. '
                        'Repair graph nodes, edges and subgraph contracts to fulfill those requirements. '
                        'Use validation_paths.fact_dependencies to locate the last fact invalidation and '
                        'its registered producers; producers still need their own prerequisites. '
                        'Repair the earliest broken dependency in the continuation. Do not skip a required skill. '
                        'When changing a route, replace its existing (source, on) edge rather than adding '
                        'a second edge with the same pair. Remove obsolete disconnected nodes. '
                        'Address all reported errors together and inspect every related call site and path. '
                        'Preserve corrections for every previous_validation_attempts entry. '
                        'A blocked path may hide downstream errors, so run the full final graph check. '
                        'All images, user requirements, initial facts and skill contracts remain those '
                        'in the original input. Task execution has not started.'),
                }
                self.budget.reserve(revision=True)
                payload = deepcopy(initial_payload)
                payload['text']['format']['schema'] = response_schema(GraphRevision, catalog)
                payload['text']['format']['name'] = 'robot_graph_revision'
                payload['input'][0]['content'].append({'type': 'input_text', 'text': json.dumps(feedback, ensure_ascii=False)})
                model_call = {'kind': 'validation_revision', 'attempt': revision + 1,
                              'revision_format': 'scope_replacements',
                              'previous_request_hash': feedback['previous_request_hash'],
                              'validation_errors': feedback['validation_errors']}

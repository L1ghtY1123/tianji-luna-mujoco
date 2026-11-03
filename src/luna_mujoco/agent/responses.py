"""One initial inventory/task/graph proposal; runtime recovery belongs to Decisions."""
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

from pydantic import ValidationError

from ..visual_api import VisualAPI, VisualInventoryError
from ..visual_inventory import image_part
from .schema import (
    InitialPlan,
    TaskSpec,
    object_task_specs,
    response_schema,
    skill_catalog,
    validate_graph,
)

BOOTSTRAP_FACTS = {
    'observed': False, 'inventory_ready': False, 'inventory_finalized': False, 'task_assessed': False,
    'observation_pose_ready': True, 'grasp_view_ready': False, 'needs_place': True,
    'target_selected': False, 'destination_selected': False, 'arm_selected': False,
    'scene_ready': False, 'grasp_held': False, 'recovery_needed': False, 'task_verified': False,
    'placement_zone_ready': False, 'placement_resolved': False,
}


@dataclass
class ResponsesBudget:
    """Reserve exactly one model compilation per task, including concurrent callers."""
    used: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def reserve(self):
        with self._lock:
            if self.used:
                raise VisualInventoryError('responses_budget_exhausted')
            self.used = 1


class ResponsesCompiler(VisualAPI):
    endpoint = '/v1/responses'
    source = 'openai_responses_proposal'
    error_prefix = 'responses'

    def __init__(self, *, registry=None, configured_regions=None, known_assets=None,
                 initial_facts=None, scene_context=None, budget=None, **kwargs):
        super().__init__(**kwargs)
        if registry is None:
            from ..skills import Registry
            registry = Registry()
        self.registry = registry
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
        self._discovery = None
        self.last_evidence = {}

    def reset(self):
        """Start a new task without carrying its budget, identities, goal or cached responses."""
        self.budget = ResponsesBudget()
        self.original_task = None
        self._initial_key = self._initial_plan = self._discovery = None
        self.last_evidence = {}
        self.cache.clear()

    def compile(self, capture, *, instruction, configured_regions=None):
        key = (capture['observation_id'], capture.get('generation'), instruction)
        if self._initial_plan is not None and key == self._initial_key:
            return self._initial_plan.model_copy(deep=True)
        catalog = skill_catalog(self.registry)
        regions = self.configured_regions if configured_regions is None else configured_regions
        context = {'instruction': instruction, 'skill_catalog': catalog,
            'skill_instructions': self._skill_instructions(catalog), 'configured_regions': regions,
            'known_assets': self.known_assets, 'initial_facts': self.initial_facts,
            'scene_context': self.scene_context}
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
        # A compact view of the same contracts, not a recommended action sequence.
        # Include provides and explicit invalidations together so a visual success
        # cannot be mistaken for renewed observation or collision-map evidence.
        context = {**context, 'planning_state_transitions': {
            name: {'requires': contract.get('requires', {}),
                   'on_success': {**dict.fromkeys(contract.get('provides', []), True),
                                  **contract.get('planning_effects', {})},
                   'argument_rules': contract.get('argument_rules', [])}
            for name, contract in catalog.items()},
            'execution_failure_context': {
                'observed_service_errors': ['decisions_judgment_http_503', 'visual_decisions_http_503'],
                'meaning': 'Decision service temporarily unavailable; no visual verdict was returned. '
                           'This is not low confidence, collision, or evidence that the task changed.',
                'recovery_owner': 'model_authored_graph',
                'transport_policy': 'Decisions retries an identical frozen request up to three times only '
                    'on transient transport/HTTP failures without a verdict. Every attempt is archived. '
                    'This never repeats a robot skill, changes pixels, or retries a valid unknown/no answer.'},
            'known_capability_outcomes': {
                name: contract.get('errors', []) for name, contract in catalog.items()},
            'preflight': (
                'Before returning your graph, trace every continuing path from the supplied initial facts. '
                'At each node check requires, apply provides and planning_effects (including false and null), '
                'and account for argument_rules. The compact planning_state_transitions lists these same '
                'contracts, not a prescribed sequence. An action or verification success restores only its '
                'declared facts: confirming an object location does not by itself refresh inventory observation '
                'or publish a collision map. Facts invalidated by a scene change stay invalid until an '
                'explicitly chosen capability establishes them. Check subsequent motion prerequisites and '
                'the final completion requirements, including after the last object. Choose the needed '
                'capabilities, ordering and recovery branches yourself. Do not output a plan that you know '
                'will need a second call to repair these dependencies.')}
        from .graph_prompt import FINAL_GRAPH_CHECK, graph_context

        model = InitialPlan
        context = graph_context(context)
        schema = response_schema(model, catalog)
        root_task = deepcopy(schema['$defs']['TaskSpec'])
        root_task['properties']['instruction']['enum'] = [context['instruction']]
        schema['$defs']['InitialTaskSpec'] = root_task
        schema['properties']['task'] = {'$ref': '#/$defs/InitialTaskSpec'}
        schema['$defs']['EntityProposal']['properties']['asset_id'] = (
            {'anyOf': [{'type': 'string', 'enum': list(self.known_assets)}, {'type': 'null'}]}
            if self.known_assets else {'type': 'null'})
        self.budget.reserve()
        content = [{'type': 'input_text', 'text': json.dumps({
            **context, 'observation_id': capture['observation_id'], 'generation': capture.get('generation')}, ensure_ascii=False)}]
        for camera in ('2', '3', '1'):
            rgb = capture['views'][camera]['rgb']
            height, width = rgb.shape[:2]
            content.extend([{'type': 'input_text', 'text': json.dumps({
                'camera_id': camera, 'width_px': width, 'height_px': height})}, image_part(rgb)])
        content.append({'type': 'input_text', 'text': FINAL_GRAPH_CHECK})
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}],
                   'reasoning': {'effort': 'medium'}, 'max_output_tokens': 24000,
                   'text': {'format': {'type': 'json_schema', 'name': 'robot_policy',
                                      'strict': True, 'schema': schema}}}

        def parse(body, evidence):
            if body.get('status') != 'completed':
                raise VisualInventoryError('responses_incomplete', evidence={
                    'response_status': body.get('status'),
                    'incomplete_details': body.get('incomplete_details'), 'usage': body.get('usage', {})})
            texts = [part['text'] for item in body.get('output', []) if item.get('type') == 'message'
                     for part in item.get('content', []) if part.get('type') == 'output_text']
            try:
                result = model.model_validate_json(''.join(texts))
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
                validate_graph(result.graph, catalog, judgments=set(result.task.completion_conditions).union(
                                   *(set(task.completion_conditions) for task in item_tasks.values())),
                               initial_facts=context.get('initial_facts', {}),
                               require_completion=result.task.supported,
                               required_final_facts=({
                                   **({'at_initial_pose': True} if result.task.completion_pose == 'initial'
                                      and 'verify_task_completion' not in catalog else {}),
                                   **({'task_verified': True} if 'verify_task_completion' in catalog else {})}
                                   if result.task.supported else None))
                if (result.task.supported and 'verify_task_completion' in catalog
                        and set(item_tasks) != set(result.task.target_refs)):
                    raise ValueError('every_candidate_requires_independent_object_task')
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
                details = ([{'location': item['loc'], 'message': item['msg']}
                            for item in error.errors(include_input=False, include_url=False)[:8]]
                           if isinstance(error, ValidationError) else str(error)[:1000])
                try:
                    rejected = json.loads(''.join(texts))
                except ValueError:
                    rejected = {'invalid_json_text': ''.join(texts)}
                raise VisualInventoryError('responses_invalid_plan',
                    evidence={'validation_errors': details, 'rejected_proposal': rejected}) from None
            self.last_evidence = {**evidence, 'responses_calls_used': self.budget.used,
                                  'responses_call_limit': 1}
            from .graph_validation import validate_capabilities
            self.last_evidence['graph_preflight_diagnostics'] = validate_capabilities(
                result.graph, catalog, initial_facts=context.get('initial_facts', {}))
            return result

        return self.request(payload, capture, 'compile_task', parse,
            model_call={'kind': 'initial_compile', 'attempt': 0, 'event_key': None, 'trigger': None})

    def discover(self, capture, *, instruction):
        """Bridge the existing perception inventory while retaining generic task/graph fields."""
        plan = self.compile(capture, instruction=instruction)
        if self._discovery is not None:
            return deepcopy(self._discovery)
        if not plan.task.supported:
            raise VisualInventoryError('task_unsupported', evidence={'reason': plan.task.reason})
        identifiers = {entity.ref: 'instance-' + uuid4().hex[:16] for entity in plan.entities}
        entities = {}
        target_refs = set(plan.task.target_refs)
        for entity in plan.entities:
            identifier = identifiers[entity.ref]
            roles = (['table', 'collision'] if entity.kind == 'table' else ['target_region']
                     if entity.kind == 'marker' else ['container', 'collision']
                     if entity.kind == 'bin' else ['collision'])
            item = {'id': identifier, 'kind': entity.kind, 'color': entity.color,
                    'description': entity.description, 'bbox': entity.bbox,
                    'appearance_keywords': list(entity.appearance_keywords),
                    'reference_camera_id': entity.reference_camera_id,
                    'initial_parent_id': identifiers.get(entity.parent_ref), 'identity_source': self.source,
                    'visual_membership': entity.membership if entity.ref in target_refs else 'excluded',
                    'roles': roles}
            if entity.kind not in {'table', 'marker'}:
                item['segmentation_query'] = {'kind': entity.kind, 'color': entity.color,
                                              'prompt': entity.segmentation_prompt,
                                              'aliases': list(entity.segmentation_aliases)}
            if entity.asset_id is not None:
                item.update(asset_id=entity.asset_id, asset_source='configured_catalog_responses_proposal')
            if entity.geometry_policy == 'unknown_obb':
                item['geometry_policy'] = {'mode': 'unknown_obb', 'instance_id': identifier,
                    'source': 'user_instruction_interpreted_by_responses',
                    'instruction_quote': entity.geometry_policy_instruction}
            entities[identifier] = item
        target_ids = [identifiers[ref] for ref in plan.task.target_refs]
        for number, identifier in enumerate(target_ids, 1):
            entities[identifier]['display_id'] = number
        task_data = plan.task.model_dump(mode='json')
        task_data['target_refs'] = target_ids
        for name in ('source_ref', 'destination_ref', 'work_area_ref'):
            task_data[name] = identifiers.get(task_data[name], task_data[name])
        task = TaskSpec.model_validate(task_data)
        self.original_task = task

        def bind_literals(value):
            if isinstance(value, str):
                return identifiers.get(value, value)
            if isinstance(value, list):
                return [bind_literals(item) for item in value]
            if isinstance(value, dict):
                return {key: bind_literals(item) for key, item in value.items()}
            return value

        graph = plan.graph.model_dump(mode='json')

        def bind_nodes(nodes):
            for node in nodes:
                if node['type'] == 'skill':
                    node['arguments'] = bind_literals(node['arguments'])
        bind_nodes(graph['nodes'])
        for subgraph in graph['subgraphs']:
            if subgraph['task'] is not None:
                subgraph['input_id'] = bind_literals(subgraph['input_id'])
                for field in ('target_refs', 'source_ref', 'destination_ref', 'work_area_ref'):
                    subgraph['task'][field] = bind_literals(subgraph['task'][field])
            bind_nodes(subgraph['nodes'])
        task_data = task.model_dump(mode='json')
        selected = target_ids if task.quantity is None else target_ids[:task.quantity]
        support_ref = plan.task.work_area_ref or plan.task.destination_ref
        proposals = {entity.ref: entity for entity in plan.entities}
        visited = set()
        while support_ref in proposals and support_ref not in visited:
            support = proposals[support_ref]
            if support.kind == 'table':
                break
            visited.add(support_ref)
            support_ref = support.parent_ref
        if support_ref not in proposals or proposals[support_ref].kind != 'table':
            support_ref = next((entity.ref for entity in plan.entities if entity.kind == 'table'), None)
        inventory = {'source_id': task.source_ref, 'support_id': identifiers.get(support_ref),
                     'destination_id': task.destination_ref, 'work_area_id': task.work_area_ref,
                     'source_object_ids': target_ids, 'selected_ids': selected,
                     'object_tasks': {key: value.model_dump(mode='json')
                                      for key, value in object_task_specs(graph).items()}}
        if graph['version'] == 2:
            inventory['subgraph_by_input_id'] = {row['input_id']: row['id'] for row in graph['subgraphs']
                                                if row['input_id'] is not None}
        reference = {'observation_id': capture['observation_id'], 'entities': deepcopy(entities),
                     'views': {camera: {'rgb': capture['views'][camera]['rgb']} for camera in ('1', '2', '3')},
                     'task_spec': task_data, 'policy_graph': graph,
                     'task_inventory': inventory,
                     'inventory_state': {'entities': deepcopy(entities), 'retired_entity_ids': [], 'transformations': {}}}
        self._discovery = {'entities': entities, 'reference': reference, 'evidence': self.last_evidence,
                           'task_spec': task_data, 'policy_graph': graph,
                           'task_inventory': inventory}
        return deepcopy(self._discovery)

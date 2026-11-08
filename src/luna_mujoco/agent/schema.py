"""Data-only task and bounded policy graph contracts shared by model and runtime."""
from copy import deepcopy
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from ..core.graph import Edge, Graph, Identifier, Node, StrictModel, Subgraph, TaskSpec


class EntityProposal(StrictModel):
    ref: str = Field(min_length=1)
    existing_id: str | None = Field(default=None, description=(
        'Existing permanent instance ID from object_memory when the current subject matches its dated '
        'reference. Null for a proposed new instance. Never invent an ID or reuse another same-kind instance.'))
    kind: Literal['table', 'bin', 'marker', 'cube', 'sphere', 'cylinder', 'object']
    color: str = Field(description=(
        'Short ENGLISH color adjective, e.g. pink, blue, black or unknown. '
        'Used literally in SAM3 appearance fallback prompts; never localize this field. '
        'Human-facing description may use the user language.'))
    description: str = Field(description=(
        'Describe visible color and shape, plus the location that distinguishes this individual in '
        'reference_camera_id. Name that camera when using image-left/right/upper/lower. '
        'For similar objects, specify their separate positions relative to visible neighbors. '
        'Do not use only one object/another object, task role, or a future destination as identification.'))
    bbox: list[float] = Field(min_length=4, max_length=4, description=(
        'Visible subject box [x_min, y_min, x_max, y_max], normalized to [0,1] in the named camera. '
        'Both extents must be positive. Configured unmarked regions are referenced directly, not entities.'))
    reference_camera_id: Literal['1', '2', '3'] = Field(description=(
        'Choose the initial camera that shows this individual largest and most clearly separated from '
        'similar neighbors. Prefer a complete view over a clipped or occluded fragment when available. '
        'Use this same camera for bbox and the appearance description.'))
    parent_ref: str | None
    membership: Literal['required', 'excluded', 'unknown']
    segmentation_prompt: str = Field(description=(
        'Short English visual noun phrase for SAM3, even when the user task is Chinese. '
        'For example green block or blue box. Human descriptions may remain Chinese.'))
    appearance_keywords: tuple[str, ...] = Field(default=(), max_length=8, description=(
        'Visible appearance descriptors: color, shape, texture, pattern or material when observable. '
        'Proposals, not verified identity; do not invent hidden properties or task roles.'))
    segmentation_aliases: tuple[str, ...] = Field(default=(), max_length=2, description=(
        'Up to two distinct short English visual noun phrases, e.g. pink rectangular block, pink package. '
        'Complement segmentation_prompt; omit location, task relations and unsupported properties.'))
    asset_id: str | None = Field(default=None, description=(
        'Optional model identity from the configured known_assets catalog. Null when unrecognized. '
        'An asset identifies declared dimensions, never current position or orientation.'))
    geometry_policy: Literal['auto', 'unknown_obb'] = Field(default='auto', description=(
        'unknown_obb only when the USER explicitly declares this subject unknown/unmodeled/no CAD. '
        'Otherwise auto uses runtime Decisions CAD retrieval. An unrecognized appearance alone is not such a declaration.'))
    geometry_policy_instruction: str | None = Field(default=None, description=(
        'Exact nonempty quote from the user instruction supporting unknown_obb; null for auto. '
        'This is a representation requirement, not visual identity or a CAD match.'))

    @model_validator(mode='after')
    def valid_box(self):
        if self.geometry_policy == 'unknown_obb' and (
                not self.geometry_policy_instruction or not self.geometry_policy_instruction.strip()
                or self.asset_id is not None):
            raise ValueError('unknown_geometry_requires_instruction_and_no_asset')
        x1, y1, x2, y2 = self.bbox
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError('invalid_rgb_box')
        if self.kind not in {'table', 'marker'} and not self.segmentation_prompt.strip():
            raise ValueError('missing_segmentation_prompt')
        if any(not value.strip() or len(value) > 256
               for value in (*self.appearance_keywords, *self.segmentation_aliases)):
            raise ValueError('invalid_appearance_description')
        return self


def object_task_specs(graph):
    """Collect the immutable per-input task from each owned subgraph."""
    data = graph.model_dump() if isinstance(graph, Graph) else Graph.model_validate(graph).model_dump()
    return {scope['input_id']: TaskSpec.model_validate(scope['task'])
            for scope in data['subgraphs'] if scope['task'] is not None}


def validate_batch_policy(graph, task):
    """Collection inputs are selected from current sensing through a shared policy."""
    if not task.supported or task.scope == 'ids':
        return
    from ..runtime.validation import GraphValidationError

    data = graph.model_dump()
    issues = []
    if any(scope['input_id'] is not None for scope in data['subgraphs']):
        issues.append('batch_requires_shared_subgraphs:use_runtime_selected_input')
    skills = [node for scope in [data, *data['subgraphs']] for node in scope['nodes']
              if node['type'] == 'skill']
    if not any(node['skill'] == 'select_target' for node in skills):
        issues.append('batch_requires_select_target:initial_candidates_do_not_close_membership')
    for node in skills:
        if (node['skill'] == 'identify_target'
                and node['arguments'].get('target_id') in task.target_refs):
            issues.append('batch_target_must_use_runtime_binding:' + node['id'])
    if issues:
        raise GraphValidationError(issues)


class InitialPlan(StrictModel):
    entities: list[EntityProposal] = Field(min_length=1, max_length=1024)
    task: TaskSpec
    graph: Graph

    @field_validator('graph')
    @classmethod
    def graph_owners_match_task(cls, graph: Graph, info: ValidationInfo):
        task = info.data.get('task')
        if task is not None and not set(object_task_specs(graph)) <= set(task.target_refs):
            raise ValueError('object_task_outside_candidate_inventory')
        return graph

    @model_validator(mode='after')
    def coherent_entities(self):
        refs = {entity.ref for entity in self.entities}
        if len(refs) != len(self.entities):
            raise ValueError('duplicate_scene_reference')
        for entity in self.entities:
            if entity.parent_ref is not None and (entity.parent_ref not in refs or entity.parent_ref == entity.ref):
                raise ValueError('invalid_parent_reference')
        if not set(self.task.target_refs) <= refs:
            raise ValueError('unknown_task_target')
        if any(entity.ref in self.task.target_refs and entity.membership == 'excluded'
               for entity in self.entities):
            raise ValueError('task_target_membership_conflict')
        return self


class RootGraphRevision(StrictModel):
    id: Identifier
    nodes: list[Node] = Field(min_length=1, max_length=512)
    entry: Identifier
    edges: list[Edge] = Field(max_length=1536)


class GraphRevision(StrictModel):
    """Explicit scope replacements preserve all unedited model-authored control flow."""
    root: RootGraphRevision | None = Field(description=(
        'Null preserves the root scope. Otherwise supply its complete id, nodes, entry and edges. '
        'Subgraph bodies are edited separately.'))
    subgraphs: list[Subgraph] = Field(max_length=1024, description=(
        'Complete bodies of only the subgraphs to replace or add, matched by id. '
        'Every omitted subgraph keeps its current definition. Include all nodes and edges of each edited body.'))
    remove_subgraph_ids: tuple[Identifier, ...] = Field(max_length=1024, description=(
        'Explicitly remove these existing subgraph definitions. Update every affected caller. '
        'Omitting a body from subgraphs never deletes it.'))

    @model_validator(mode='after')
    def distinct_edits(self):
        identifiers = [scope.id for scope in self.subgraphs]
        if (len(set(identifiers)) != len(identifiers)
                or len(set(self.remove_subgraph_ids)) != len(self.remove_subgraph_ids)
                or set(identifiers) & set(self.remove_subgraph_ids)):
            raise ValueError('graph_revision_conflicting_scope_edits')
        return self

    def apply_to(self, graph):
        result = deepcopy(graph)
        existing = {scope['id'] for scope in result['subgraphs']}
        if not set(self.remove_subgraph_ids) <= existing:
            raise ValueError('graph_revision_remove_unknown_scope')
        if self.root is not None:
            result.update(self.root.model_dump(mode='json'))
        replacements = {scope.id: scope.model_dump(mode='json') for scope in self.subgraphs}
        updated, scopes = set(), []
        for scope in result['subgraphs']:
            identifier = scope['id']
            if identifier in self.remove_subgraph_ids:
                continue
            if identifier in replacements:
                if identifier not in updated:
                    scopes.append(replacements[identifier])
                    updated.add(identifier)
            else:
                scopes.append(scope)
        scopes.extend(scope for identifier, scope in replacements.items() if identifier not in updated)
        result['subgraphs'] = scopes
        return result


def skill_catalog(registry):
    return registry.describe() if hasattr(registry, 'describe') else deepcopy(registry)


def argument_schema(contract):
    return contract.get('arguments', contract.get('argument_schema', {
        'type': 'object', 'properties': {}, 'additionalProperties': False}))


def validate_graph(graph, catalog, *, initial_bindings=(),
                   initial_facts=None, require_completion=False, required_final_facts=None):
    """Validate the graph and skill contracts without generating or rewriting nodes."""
    from ..runtime.validation import validate_capabilities

    graph = graph if isinstance(graph, Graph) else Graph.model_validate(graph)
    validate_capabilities(graph, skill_catalog(catalog), initial_bindings=initial_bindings,
                          initial_facts=initial_facts, require_completion=require_completion,
                          required_final_facts=required_final_facts)
    return graph


def response_schema(model, catalog):
    """Use a closed argument object for each registered skill in strict Responses JSON."""
    from .graph_authoring import RouteGroup, ScopeFailureRoute

    schema = model.model_json_schema()
    definitions = schema.get('$defs', {})
    definitions['Edge'] = RouteGroup.model_json_schema()
    definitions['Edge']['description'] = (
        'One explicit route group. The compiler creates an edge for each sources × outcomes pair, '
        'with the same target and an independent max_traversals budget. Edge IDs are generated by code. '
        'All source nodes must support all listed outcomes. Scope defaults apply only when explicitly declared.')
    # Declare root call interfaces first, then fill their bodies. Local node
    # declarations precede entry/edge references in every scope.
    for name, fields in (
            ('Graph', ('version', 'id', 'nodes', 'subgraphs', 'entry', 'edges')),
            ('Subgraph', ('id', 'input_id', 'task', 'nodes', 'entry', 'edges')),
            ('RootGraphRevision', ('id', 'nodes', 'entry', 'edges'))):
        if name in definitions:
            properties = definitions[name]['properties']
            definitions[name]['properties'] = {key: properties[key] for key in fields}
            properties = definitions[name]['properties']
            for outcome in ('failure', 'unknown'):
                properties['on_' + outcome] = {'anyOf': [ScopeFailureRoute.model_json_schema(), {'type': 'null'}],
                    'description': 'Explicit common ' + outcome + ' route for skill and subgraph-call nodes '
                        'in this scope. An explicit route for a source/outcome overrides this declaration. '
                        'Null declares no default. Routers and ends always need their own explicit routes. '
                        'Choose a declared local node. Prefer a shared failure exit or compatible recovery entry.'}
            properties['edges']['description'] = (
                'Explicit local route groups for normal control and exceptions to on_failure/on_unknown. '
                'Omit routes already covered by the declared scope defaults. Group sources with identical '
                'outcomes, target and traversal limit. Targets stay in this scope.')
            # Declare common continuations before the model emits their exceptions.
            properties['edges'] = properties.pop('edges')
    processing = [name for name, contract in skill_catalog(catalog).items() if contract.get('processing_effect')]
    field = definitions['TaskSpec']['properties']['processing']
    field['items'] = {'type': 'string', 'enum': processing} if processing else {'type': 'string'}
    if not processing:
        field['maxItems'] = 0
    # Express ownership in the output schema, not only a post-hoc validator.
    # Shared subgraphs have neither an item owner nor an item TaskSpec.
    owned = deepcopy(definitions['Subgraph'])
    shared = deepcopy(definitions['Subgraph'])
    owned['properties']['input_id'] = {'type': 'string', 'minLength': 1, 'description': (
        'A literal candidate ref from the frozen root task.target_refs, never $selected, $current or $product. '
        'For a runtime-selected input or product, use a shared scope with input_id=null and task=null; '
        'its skill arguments may use the declared result bindings.')}
    owned['properties']['task'] = {'$ref': '#/$defs/TaskSpec'}
    owned['description'] = 'Optional instance-specific task scope. Shared scopes can process runtime-selected instances.'
    shared['properties']['input_id'] = {'type': 'null'}
    shared['properties']['task'] = {'type': 'null'}
    shared['description'] = (
        'Shared scope without an object task. Its skills retain their own declared prerequisites; '
        'it may establish shared facts before an object-owned scope is called.')
    definitions['Subgraph'] = {'anyOf': [owned, shared]}
    definitions['SubgraphNode']['properties']['subgraph_id']['description'] = (
        'Every call receives CURRENT caller facts and returns its changed facts. The next object does '
        'not restart initial_facts. A preceding object may have invalidated observed or scene_ready; '
        'establish any required evidence explicitly before the next skill that consumes it.')
    skill_base = definitions['SkillCall']
    variants = []
    for name, contract in skill_catalog(catalog).items():
        variant = deepcopy(skill_base)
        variant['properties']['skill'] = {'type': 'string', 'const': name}
        arguments = deepcopy(argument_schema(contract))
        argument_definitions = arguments.pop('$defs', {})

        def inline(node, local_definitions=argument_definitions):
            if isinstance(node, dict):
                if '$ref' in node:
                    reference = node['$ref'].removeprefix('#/$defs/')
                    return inline(deepcopy(local_definitions[reference]))
                return {key: inline(value) for key, value in node.items()}
            if isinstance(node, list):
                return [inline(value) for value in node]
            return node
        variant['properties']['arguments'] = inline(arguments)
        if name == 'verify_place' and 'release_mode' in variant['properties']['arguments'].get('properties', {}):
            mode = variant['properties']['arguments']['properties']['release_mode']
            mode['enum'] = ['surface_setdown', 'container_release']
            mode.pop('default', None)
        variants.append(variant)
    if not variants:
        raise ValueError('empty_skill_catalog')
    definitions['SkillCall'] = {'anyOf': variants}
    definitions.pop('JsonValue', None)

    def normalize(node):
        if isinstance(node, dict):
            node.pop('discriminator', None)
            if 'oneOf' in node:
                node['anyOf'] = node.pop('oneOf')
            if node.get('type') == 'object':
                node['additionalProperties'] = False
                node['required'] = list(node.get('properties', {}))
            for value in node.values():
                normalize(value)
        elif isinstance(node, list):
            for value in node:
                normalize(value)
    normalize(schema)
    return schema

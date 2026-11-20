"""Data-only task and bounded policy graph contracts shared by model and runtime."""
from copy import deepcopy
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, ValidationInfo, field_validator, model_validator

from ..core.graph import Graph, StrictModel, TaskSpec
from ..core.grounding import SelectionRule, relation_refs


def normalize_color(value):
    if not isinstance(value, str):
        return value
    names = {'黑': 'black', '白': 'white', '蓝': 'blue', '粉': 'pink', '粉红': 'pink',
             '红': 'red', '绿': 'green', '黄': 'yellow', '橙': 'orange', '灰': 'gray',
             '棕': 'brown', '褐': 'brown', '紫': 'purple', 'grey': 'gray'}
    value = value.strip().casefold()
    return names.get(value.removesuffix('色'), value)


ColorName = Annotated[str, BeforeValidator(normalize_color)]


class CameraBox(StrictModel):
    camera_id: Literal['1', '2', '3']
    bbox: list[float] = Field(min_length=4, max_length=4, description=(
        'Visible subject bounds [left, top, right, bottom], normalized to [0,1] in this camera. '
        'Origin is the image top-left; x increases rightward and y downward.'))

    @model_validator(mode='after')
    def valid_box(self):
        x1, y1, x2, y2 = self.bbox
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError('invalid_rgb_box')
        return self


class DeclaredObject(StrictModel):
    scene_quote: str = Field(min_length=1, description='Exact distinguishing noun phrase from the explicit scene paragraph including its attributes.')
    kind: Literal['table', 'bin', 'marker', 'object']
    color: ColorName = Field(description='English color name explicitly stated in scene_quote, else unknown. Keep the original wording in color_quote.')
    color_quote: str | None = Field(default=None, description='Exact color word from scene_quote, or null if unspecified.')
    declared_count: int | None = Field(default=None, ge=1, description='Explicit object count in scene_quote, else null.')

    segmentation_prompt: str = Field(min_length=1, max_length=256, description='Short natural English object noun with declared visual qualifiers. CAD is metadata: use black mechanical part for a black CAD object. Omit articles and inferred colors.')
    asset_id: str | None = Field(default=None, description='Exact CAD repository code explicitly named for this subject, else null.')
    initially_on_table: bool = Field(default=False, description='True only when the scene explicitly says this item rests on a table initially.')

    @model_validator(mode='after')
    def explicit_color(self):
        if self.color == 'unknown' and self.color_quote is None:
            return self
        if self.color == 'unknown' or not self.color_quote or self.color_quote not in self.scene_quote:
            raise ValueError('declared_color_requires_scene_quote')
        return self


class SceneProposal(StrictModel):
    """One camera's unverified physical surface; never a task selector or permanent ID."""
    scene_quote: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    camera_id: Literal['1', '2', '3']
    bbox: list[float] | None = Field(default=None, description='Normalized xyxy in this camera, or null when uncertain.')
    description: str = Field(description='Observed appearance in this camera.')
    visibility: Literal['visible', 'partial', 'uncertain'] = 'visible'

    @model_validator(mode='after')
    def valid_box(self):
        if self.bbox is not None:
            CameraBox(camera_id=self.camera_id, bbox=self.bbox)
        return self


class TaskSelector(StrictModel):
    """User requirements independent of the scene's observed appearance hypotheses."""
    scene_quote: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    selection: SelectionRule
    requested_attributes: str = Field(description=(
        'A self-contained description of only appearance/category constraints requested by the user. '
        'Use container/receptacle for a box used to hold objects; include the noun, not just its color. '
        'Never add model-inferred shape. '
        'A required supporting table may use table surface.'))
    instruction_quote: str = Field(default='', description='Exact user phrase for this role, including relative, ordinal and other qualifiers. Empty only for an implicit support/reference.')
    kind: Literal['table', 'bin', 'marker', 'object'] = Field(description=(
        'User-requested broad semantic type: bin for a container/receptacle, table for a support surface, '
        'marker for a marker. Use object when the user does not require one of these types. '
        'Observed shape is not a required type; put only explicitly requested shape in requested_attributes.'))
    color: ColorName = Field(default='unknown', description=(
        'English name of the color required by the USER, or unknown for unrestricted color. Never copy an observed color '
        'when the instruction does not require it.'))
    appearance_constraints: tuple[str, ...] = Field(default=(), description=(
        'Additional USER-requested appearance tests beyond kind and color, e.g. printed text, material '
        'or explicitly requested shape. Empty for just a box/container, a table, or a pink object. '
        'Position, containment, count and rank belong in selection, not here.'))
    segmentation_prompt: str = Field(min_length=1, max_length=256)
    segmentation_aliases: tuple[str, ...] = Field(default=(), max_length=2)
    proposal_refs: tuple[str, ...] = Field(default=(), description='Optional inventory hints, not proven identities.')
    geometry_policy: Literal['auto', 'unknown_tsdf'] = 'auto'
    geometry_policy_instruction: str | None = None


class DestinationRegion(StrictModel):
    """Spatial references are not physical objects."""
    ref: str
    source: Literal['configured', 'visual']
    support_ref: str
    boxes: list[CameraBox] = Field(default_factory=list)


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
    declared_objects: list[DeclaredObject] = Field(min_length=1, max_length=128)
    scene_inventory: list[SceneProposal] = Field(min_length=0, max_length=1024,
        description='Visible instances of declared_objects only. Skip undeclared or unseen objects.')
    task_selectors: list[TaskSelector] = Field(min_length=1, max_length=1024)
    destination_regions: list[DestinationRegion] = Field(default_factory=list)
    support_table_refs: list[str] = Field(description=(
        'Refs of the visible TABLE entities needed for workspace collision geometry and requested '
        'source/destination support measurements. Declare each as kind=table in task_selectors, including '
        'when destination_ref is a configured unmarked region. Supported manipulation requires at '
        'least one measured table. A configured region ID cannot substitute for a table entity.'))
    task: TaskSpec
    graph: Graph

    @field_validator('graph')
    @classmethod
    def graph_owners_match_task(cls, graph: Graph, info: ValidationInfo):
        task = info.data.get('task')
        if task is not None and not set(object_task_specs(graph)) <= set(task.target_refs):
            raise ValueError('object_task_outside_candidate_objects')
        return graph

    @model_validator(mode='after')
    def coherent_entities(self):
        from ..core.scene_scope import scene_sections

        scene, _ = scene_sections(self.task.instruction)
        declared = {d.scene_quote: d for d in self.declared_objects}
        if len(declared) != len(self.declared_objects) or any(q not in scene for q in declared):
            raise ValueError('scene_class_requires_unique_exact_scene_quote')
        for d in self.declared_objects:
            if d.asset_id and d.asset_id not in scene:
                raise ValueError('declared_cad_id_requires_scene_quote')
        by_id = {p.proposal_id: p for p in self.scene_inventory}
        if any(p.scene_quote not in declared for p in self.scene_inventory):
            raise ValueError('proposal_outside_declared_scene')
        for row in self.task_selectors:
            scope = declared.get(row.scene_quote)
            kind = row.kind if row.kind in {'table', 'bin', 'marker'} else 'object'
            if scope is None or scope.kind != kind:
                raise ValueError('物体或选择器不在声明的场景中，请补充场景段。')
            if scope.color != 'unknown' and row.color not in {'unknown', scope.color}:
                raise ValueError('scene_color_conflicts_with_declaration')
        for selector in self.task_selectors:
            if any(p in by_id and by_id[p].scene_quote != selector.scene_quote for p in selector.proposal_refs):
                raise ValueError('proposal_hint_outside_selector_scope')
        refs = {entity.ref for entity in self.task_selectors}
        if len(refs) != len(self.task_selectors):
            raise ValueError('duplicate_scene_reference')
        dependencies = {entity.ref: relation_refs(entity.selection.predicate) for entity in self.task_selectors}
        if any(not values <= refs or key in values for key, values in dependencies.items()):
            raise ValueError('invalid_selection_reference')
        resolved = set()
        while dependencies:
            ready = {key for key, values in dependencies.items() if values <= resolved}
            if not ready:
                raise ValueError('cyclic_selection_relations')
            resolved.update(ready)
            dependencies = {key: value for key, value in dependencies.items() if key not in ready}
        tables = {entity.ref for entity in self.task_selectors if entity.kind == 'table'}
        if (self.task.supported and not self.support_table_refs
                or len(set(self.support_table_refs)) != len(self.support_table_refs)
                or not set(self.support_table_refs) <= tables):
            raise ValueError('declared_support_table_required')
        proposals = {p.proposal_id for p in self.scene_inventory}
        if len(proposals) != len(self.scene_inventory):
            raise ValueError('duplicate_scene_proposal')
        source_tables = [s.ref for s in self.task_selectors if s.kind == 'table' and s.selection.role == 'source']
        for selector in self.task_selectors:
            if not selector.instruction_quote and source_tables == [selector.ref]:
                def explicitly_on(predicate):
                    return (predicate.op == 'on' and predicate.ref == selector.ref
                            or predicate.op == 'and' and any(explicitly_on(t) for t in predicate.terms))
                for target in self.task_selectors:
                    if target.selection.role == 'target' and explicitly_on(target.selection.predicate):
                        phrase = next((s for s in ('桌面上', '桌子上', 'on the tabletop', 'on the table')
                                       if s in target.instruction_quote and s in self.task.instruction), None)
                        if phrase:
                            selector.instruction_quote = phrase
                            break
            if not set(selector.proposal_refs) <= proposals:
                raise ValueError('unknown_scene_proposal_hint')
            if selector.instruction_quote and selector.instruction_quote not in self.task.instruction:
                raise ValueError('selector_quote_not_in_user_instruction')
            if not selector.instruction_quote and selector.selection.role not in {'support', 'reference'}:
                raise ValueError(f'requested_selector_requires_user_quote:{selector.ref}:{selector.selection.role}')
            if selector.geometry_policy == 'unknown_tsdf' and (
                    not selector.geometry_policy_instruction
                    or selector.geometry_policy_instruction not in self.task.instruction):
                raise ValueError('geometry_policy_quote_not_in_user_instruction')
        if not set(self.task.target_refs) <= refs:
            raise ValueError('unknown_task_target')
        for region in self.destination_regions:
            if region.support_ref not in refs:
                raise ValueError('unknown_region_support')
        return self


def skill_catalog(registry):
    from ..core.failure_policy import TERMINAL_SKILLS

    catalog = registry.describe() if hasattr(registry, 'describe') else deepcopy(registry)
    return {name: contract for name, contract in catalog.items() if name not in TERMINAL_SKILLS}


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
    definitions['RouterNode']['properties']['rule']['enum'] = ['visual', 'observation_recovery']
    definitions['Edge'] = RouteGroup.model_json_schema()
    definitions['Edge']['description'] = (
        'One explicit route group. The compiler creates an edge for each sources × outcomes pair, '
        'with the same target and an independent max_traversals budget. Edge IDs are generated by code. '
        'All source nodes must support all listed outcomes. Scope defaults apply only when explicitly declared.')
    # Declare root call interfaces first, then fill their bodies. Local node
    # declarations precede entry/edge references in every scope.
    for name, fields in (
            ('Graph', ('version', 'id', 'nodes', 'subgraphs', 'entry', 'edges')),
            ('Subgraph', ('id', 'input_id', 'task', 'nodes', 'entry', 'edges'))):
        if name in definitions:
            properties = definitions[name]['properties']
            definitions[name]['properties'] = {key: properties[key] for key in fields}
            properties = definitions[name]['properties']
            for outcome in ('failure', 'unknown'):
                properties['on_' + outcome] = {'anyOf': [ScopeFailureRoute.model_json_schema(), {'type': 'null'}],
                    'description': 'Explicit common ' + outcome + ' route for skill and subgraph-call nodes '
                        'in this scope. An explicit route for a source/outcome overrides this declaration. '
                        'Null declares no default. Routers need explicit routes for their declared outcomes. '
                        'Failure/unknown end nodes have no outgoing routes. A success end may have only '
                        'an optional unknown receipt-failure route. Never add terminal self-loops. '
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
    from .subgraph_library import template_schemas
    definitions['Subgraph']['anyOf'].extend(template_schemas(skill_catalog(catalog)))
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
        if name == 'identify_destination':
            destination_variants = []
            for kind, mode in (('support_region', 'surface_setdown'), ('container', 'container_release')):
                destination_arguments = deepcopy(variant['properties']['arguments'])
                destination_arguments['properties']['destination_kind'] = {'type': 'string', 'const': kind}
                destination_arguments['properties']['release_mode'] = {'type': 'string', 'const': mode}
                destination_variants.append(destination_arguments)
            variant['properties']['arguments'] = {'anyOf': destination_variants}
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

"""One initial RGB inventory, independent of segmentation and depth."""
import json
from copy import deepcopy
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .demo_program import DemoProgram
from .visual_api import VisualAPI, VisualInventoryError
from .visual_inventory import image_part


class DiscoveredEntity(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ref: str
    kind: Literal['table', 'bin', 'marker', 'cube', 'sphere', 'cylinder', 'object']
    color: str
    description: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    membership: Literal['required', 'excluded', 'unknown']
    segmentation_prompt: str
    parent_ref: str | None

    @model_validator(mode='after')
    def valid_box(self):
        x1, y1, x2, y2 = self.bbox
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError('invalid_rgb_box')
        if self.kind not in {'table', 'marker'} and not self.segmentation_prompt.strip():
            raise ValueError('missing_segmentation_prompt')
        return self


class DiscoveryOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    entities: list[DiscoveredEntity] = Field(min_length=1, max_length=64)

    @model_validator(mode='after')
    def coherent_scene(self):
        refs = {item.ref: item for item in self.entities}
        if len(refs) != len(self.entities) or sum(item.kind == 'table' for item in self.entities) != 1:
            raise ValueError('invalid_scene_inventory')
        for item in self.entities:
            if item.parent_ref is not None and (item.parent_ref == item.ref or
                    item.parent_ref not in refs or refs[item.parent_ref].kind not in {'table', 'bin'}):
                raise ValueError('invalid_source_reference')
        # Marker geometry currently uses one region per colour.
        colors = [item.color for item in self.entities if item.kind == 'marker']
        if len(colors) != len(set(colors)):
            raise ValueError('ambiguous_marker_regions')
        return self


class VisualDiscovery(VisualAPI):
    endpoint = '/v1/responses'
    source = 'openai_responses'
    error_prefix = 'visual_discovery'

    def discover(self, capture, *, instruction):
        content = [{'type': 'input_text', 'text': json.dumps({
            'observation_id': capture['observation_id'], 'generation': capture['generation'],
            'task': instruction,
            'instructions': 'Inventory every visible entity ON the robot working table, including the table, '
            'open containers, their visible contents, obstacles and flat coloured destination regions (marker). '
            'Include objects not mentioned by the task. Exclude the robot and everything outside this table. '
            'Use RGB only. Enumerate physically distinct items, including identical ones, separately. '
            'ref is a temporary unique reference. bbox is [left,top,right,bottom] normalized to the head RGB. '
            'Describe appearance and initial relative location to distinguish identical objects. '
            'parent_ref identifies the table or containing bin, null only when unresolved or for the table. '
            'membership is required only when the instruction identifies that item as transport work; '
            'containers, table, regions and obstacles are excluded. Unknown for an ambiguous quota/subset. '
            'segmentation_prompt is a concise English visual phrase for SAM3 (e.g. red cube, blue open box), '
            'not an ID. Identical instances may share the same prompt. Do not invent hidden objects.'}, ensure_ascii=False)},
            image_part(capture['views']['3']['rgb'])]
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}],
                   'reasoning': {'effort': 'low'}, 'max_output_tokens': 6000,
                   'text': {'format': {'type': 'json_schema', 'name': 'tabletop_inventory',
                                      'strict': True, 'schema': DiscoveryOutput.model_json_schema()}}}
        def parse(body, evidence):
            if body.get('status') != 'completed':
                raise VisualInventoryError('visual_discovery_incomplete')
            texts = [part['text'] for item in body.get('output', []) if item.get('type') == 'message'
                     for part in item.get('content', []) if part.get('type') == 'output_text']
            output = DiscoveryOutput.model_validate_json(''.join(texts))
            ids = {item.ref: ('table' if item.kind == 'table' else 'marker-' + item.color
                             if item.kind == 'marker' else 'instance-' + uuid4().hex[:16])
                   for item in output.entities}
            entities = {}
            for item in output.entities:
                identifier = ids[item.ref]
                entities[identifier] = {'id': identifier, 'kind': item.kind, 'color': item.color,
                    'description': item.description, 'bbox': item.bbox, 'reference_camera_id': '3',
                    'visual_membership': item.membership if item.kind not in {'table', 'bin', 'marker'} else 'excluded',
                    'initial_parent_id': ids.get(item.parent_ref), 'identity_source': self.source,
                    'roles': ['table', 'collision'] if item.kind == 'table' else
                             ['target_region'] if item.kind == 'marker' else
                             ['container', 'collision'] if item.kind == 'bin' else ['collision']}
                if item.kind not in {'table', 'marker'}:
                    entities[identifier]['segmentation_query'] = {
                        'kind': item.kind, 'color': item.color, 'prompt': item.segmentation_prompt}
            return {'entities': entities, 'evidence': evidence,
                    'reference': {'observation_id': capture['observation_id'], 'entities': entities,
                                  'views': {'3': {'rgb': capture['views']['3']['rgb']}}}}
        return self.request(payload, capture, 'initialize_inventory', parse)


class DemoDiscoveredEntity(DiscoveredEntity):
    color: str = Field(description="Observed color as a lowercase English label, e.g. blue, pink, gray; "
                      "use unknown when uncertain. Do not infer color from object kind or task.")
    reference_camera_id: Literal['1', '2', '3']

    @field_validator('color')
    @classmethod
    def canonical_color(cls, value: str) -> str:
        normalized = value.strip().lower()
        return {'蓝': 'blue', '蓝色': 'blue', '藍': 'blue', '藍色': 'blue',
                '粉色': 'pink', '粉红色': 'pink', '粉紅色': 'pink',
                '灰': 'gray', '灰色': 'gray', 'grey': 'gray'}.get(normalized, normalized)


class DemoTransportArms(BaseModel):
    model_config = ConfigDict(extra='forbid')
    object_ref: str
    to_table_arm: Literal['A', 'B']
    to_right_bin_arm: Literal['A', 'B']


class DemoDiscoveredTask(BaseModel):
    model_config = ConfigDict(extra='forbid')
    supported: bool = Field(description="Whether the requested route/count/subset is expressible by the supplied skills; not visual completeness.")
    reason: str
    scope: Literal['all', 'count']
    requested_count: int | None = Field(ge=1)
    source_bin_ref: str | None
    destination_table_ref: str | None
    workflow: Literal['transfer', 'unpack_transfer'] = 'transfer'
    right_bin_ref: str | None = None
    transport_arms: list[DemoTransportArms] = Field(default_factory=list)
    source_inventory_complete: Literal['confirmed', 'unknown'] = Field(
        description="Whether every independently distinguishable pink source object in primary camera 2 is enumerated, "
        "without auxiliary evidence of an extra independent object or count contradiction. An auxiliary clipped fragment "
        "does not need an individual ID match merely to confirm the primary inventory; not a claim about hidden contents.")

    @model_validator(mode='after')
    def coherent_count(self):
        if (self.scope == 'all') != (self.requested_count is None):
            raise ValueError('demo_task_count_scope_conflict')
        return self


class DemoDiscoveryOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    entities: list[DemoDiscoveredEntity] = Field(min_length=1)
    task: DemoDiscoveredTask
    program: DemoProgram

    @model_validator(mode='after')
    def coherent_scene(self):
        refs = {item.ref: item for item in self.entities}
        if len(refs) != len(self.entities):
            raise ValueError('duplicate_scene_reference')
        for item in self.entities:
            if item.parent_ref is not None and (item.parent_ref == item.ref or
                    item.parent_ref not in refs or refs[item.parent_ref].kind not in {'table', 'bin'}):
                raise ValueError('invalid_source_reference')
        return self


def demo_discovery_schema():
    """Pydantic discriminators become the supported strict anyOf union syntax."""
    schema = DemoDiscoveryOutput.model_json_schema()

    def normalize(node):
        if isinstance(node, dict):
            node.pop('discriminator', None)
            if 'oneOf' in node:
                node['anyOf'] = node.pop('oneOf')
            if node.get('type') == 'object' and 'properties' in node:
                node['required'] = list(node['properties'])
            for value in node.values():
                normalize(value)
        elif isinstance(node, list):
            for value in node:
                normalize(value)
    normalize(schema)
    return schema


class DemoVisualDiscovery(VisualAPI):
    """One cached RGB-only discovery and skill program, then permanent IDs."""
    endpoint = '/v1/responses'
    source = 'openai_responses'
    error_prefix = 'visual_discovery'

    def discover(self, capture, *, instruction):
        from .demo_program import PROGRAM_ARGUMENTS
        from .inventory_order import head_rgb_box_order
        from .skills import Registry

        if set(capture['views']) != {'1', '2', '3'}:
            raise VisualInventoryError('demo_discovery_requires_three_rgb_views')
        registry = Registry(Path(__file__).resolve().parents[2] / 'demo_skills')
        content = [{'type': 'input_text', 'text': json.dumps({
            'observation_id': capture['observation_id'], 'generation': capture['generation'],
            'task': instruction, 'skill_catalog': registry.describe(),
            'skill_instructions': registry.documents,
            'allowed_program_arguments': {name: PROGRAM_ARGUMENTS.get(name, {})
                                          for name in registry.contracts if name != 'bind_inventory'},
            'program_initial_facts': {
                'observed': True, 'inventory_ready': True, 'inventory_finalized': True, 'task_assessed': True,
                'observation_pose_ready': True, 'needs_place': True, 'target_selected': False,
                'destination_selected': False, 'arm_selected': False, 'scene_ready': False,
                'grasp_held': False, 'recovery_needed': False},
            'instructions': 'Inventory the large working table, visibly identifiable portions of adjacent '
            'support tables, both blue boxes and the source box contents. Include obstacles and all '
            'pink source objects, even for a requested subset. Exclude the robot. Use only RGB appearance. '
            'There can be two blue boxes: the left box containing pink objects is the source; '
            'the initially empty right box is the final destination after unpacking and a collision entity. '
            'Three cameras show the same scene: camera 2 is the physical LEFT wrist and primary source view; '
            'camera 1 is the right wrist; camera 3 is head. Give one ref per physical entity across views. '
            'Prefer camera 2 for the source box and its pink contents; prefer camera 1 for the separate right box, '
            'when that entity is visibly identifiable in the respective wrist image. Choose the reference '
            'with the clearest visible subject and useful box extent, rather than a small clipped head-view '
            'fragment when the wrist offers a clearer view. Camera 3 supplies the large-table view and '
            'supplementary cross-view context. Do not require camera 3 to see all source objects. '
            'Camera roles do not prove an entity is present or establish identity; inspect the actual RGB. '
            'Each bbox is normalized [left,top,right,bottom] in reference_camera_id: x is divided by '
            'that image WIDTH and y by that image HEIGHT. Select reference_camera_id from the image actually used for this bbox, never from which arm owns that side of the scene. A right-side box visible in the head image has camera 3, not camera 1. Cameras can have DIFFERENT image heights; '
            'never normalize a wrist coordinate by the head image dimensions. Every pink source '
            'object must have its bbox in camera 2. If not distinguishable there, source_inventory_complete '
            'must be unknown. Other entities can use whichever camera shows them. '
            'source_inventory_complete=confirmed means every independently distinguishable pink source object in '
            'the primary camera 2 view is enumerated, with no unresolved evidence of an additional independent '
            'source object or contradictory count in the other views. Other cameras need not show all source objects. '
            'This does not require assigning every auxiliary fragment to an individual source ID: an auxiliary '
            'clipped view alone does not create a new entity or force unknown merely because its individual ID '
            'cannot be matched. Count physical objects, not image fragments; never invent an extra entity just '
            'to name a partial view of the same source-bin contents. '
            'Occluded bare floor or box-rim pixels alone do not imply unknown when no visible unexplained '
            'pink subject remains. This does not certify that hidden contents are absent. Keep '
            'source_inventory_complete=unknown if a camera-2 source object\'s identity, count or source membership '
            'is uncertain, or an auxiliary view gives credible evidence of an additional independent object or a count '
            'contradiction that the available views cannot reconcile. Do not omit a genuinely possible additional '
            'source object to satisfy the primary-view rule; retain its uncertainty when the images cannot resolve it. '
            'Source completeness is separate from each listed object identity: missing rim pixels or '
            'robot-arm occlusion alone do not make a clearly visible listed object ambiguous. '
            'An explicit count can still be supported when enough individual visible source objects are '
            'listed; the executor additionally requires current Decisions identity and visible-unlisted '
            'checks before motion. Do not change unknown to confirmed to satisfy a count request. Include '
            'all visible source objects even when only a subset is requested. parent_ref identifies '
            'the visibly identified support table or containing bin, otherwise null. Include a table only '
            'when actual tabletop pixels identify it; never infer an entirely hidden support beneath a box. '
            'The large destination table must be visible. segmentation_prompt is '
            'a short visible description, such as pink object, pink foam wrapped object, elliptical pink object '
            'or blue box. Describe the observed shape; do not infer that a pink object is a cup. '
            'The executor tries four bounded pink-object descriptions and associates their proposals using '
            'Decisions evidence. Tables and bins are excluded from transport. '
            'User prior: each pink object is pink foam packaging around ONE black workpiece. VLA unwraps '
            'the foam. A process links its pink input to a separately identified black output; never invent currently '
            'hidden black cores as observed entities. For an unpack request set workflow=unpack_transfer, '
            'right_bin_ref to the distinct robot-right blue box. Each selected item must complete left bin '
            'to table, VLA, then black core to right bin before starting another. transport_arms lists every '
            'source object_ref with to_table_arm and to_right_bin_arm: A is robot left, B robot right; '
            'choose A for source pickup and B for the right-workspace unpacked core, subject to executor '
            'workspace checks. For simple transfer-only requests set workflow=transfer. '
            'Parse all or a positive explicit count. All source pink objects have membership required; local code selects '
            'the nearest requested_count using measured RGBD centers and horizontal distance to the robot base. '
            'Initial camera-2 row order only assigns permanent display labels, never pickup priority. '
            'All selected objects execute nearest to farthest, independent of color or shape. '
            'Do not estimate metric distances from RGB; the executor binds the order after same-capture RGBD. '
            'Never guess an ambiguous request, '
            'hidden contents, identity or source completeness. supported describes task/skill expressibility only. '
            'For an expressible task with incomplete visual evidence, retain supported=true and '
            'source_inventory_complete=unknown; the existing evidence gates will stop unresolved observations. '
            'Set supported=false with reason only for a '
            'different route, unsupported count or specifically named subset that this template cannot express. '
            'Generate program using the supplied skill catalog and typed nodes, not arbitrary code or poses. '
            'Prelude observe_scene, bind_inventory has already run at the observation pose by the '
            'time your program executes. Use one for_each_target over the fixed selected IDs in measured '
            'near-to-far execution order, symbolic $target '
            'and $destination (table), $table_zone, $right_bin. choose_arm auto respects the frozen per-leg '
            'Responses arm assignment; it does not silently choose a different arm. The allowed_program_arguments '
            'map is exhaustive: each listed key permits only its listed symbolic values. All other skills '
            'MUST use arguments:[], including plan_place. Contract inputs are executor state, not program '
            'arguments. identify_destination binds the destination for later plan_place. '
            'Respect all contract requires/provides and program_initial_facts. In particular, '
            'identify_destination requires target_selected=true, so it follows identify_target inside '
            'the target loop, not before that loop. Between targets, target/destination/arm/scene and '
            'grasp/place facts reset; inventory_finalized/task_assessed/needs_place persist. '
            'verify_grasp success is required before placement; verify_place success before advancing. '
            'For unpack_transfer, after verify_place: prepare_unpack moves BOTH empty arms above the table; '
            'mock_unpack simulates VLA (no real VLA exists). verify_unpack first checks all three CURRENT RGB '
            'views at the VLA pose in ONE Decisions request. A clear tabletop black core in ANY view is enough. '
            'After this first result, ALWAYS return BOTH arms to the observation pose. If the black core is '
            'visible, first add its measured local RGBD obstacle to the retained scene; otherwise omit only '
            'the active unpacked item and obsolete pink layer for this return. At the observation pose, '
            'verify_unpack captures fresh three-camera RGBD and confirms the tabletop black core again. '
            'Unknown/failure stops. The mandatory return is INTERNAL to verify_unpack; do NOT duplicate it '
            'in the program. Then observe_scene analyzes that observation-pose capture, including all current '
            'inventory entities, and build_scene rebuilds the GLOBAL three-view TSDF/ESDF before black grasp. '
            'Every pink or black grasp requires both arms at the observation pose and a fresh global map. '
            'Decisions is not called for individual GraspGen candidates. After verify_unpack, $target refers to the newly '
            'confirmed black product entity of this process, distinct from the original pink input. '
            'The exact continuation after the success guard is observe_scene FIRST, then identify_target, '
            'identify_destination with $right_bin, choose_arm, build_scene, generate_grasps, plan_grasp, '
            'execute_grasp, verify_grasp. Then plan_transport uses attached ordinary plan_pose above right-bin '
            'robot-near half with the finger-front XY and lift orientation; execute_transport, drop, verify_drop. No table plan_place in '
            'the black stage. Only verify_drop completes the item. Every visual verification must succeed '
            'before dependent actions; unknown/failure stop. The table is one shared region with no slots. '
            'After the last verified drop the executor always returns both arms to their initial observation pose '
            'under the retained collision scene; it does not rebuild the map. Do not add a duplicate final return. '
            'Unknown/failure stop with '
            'a flat if_result success guard (then_steps:[], else_steps:[stop]) immediately after each '
            'verification. Keep the continuation after the guard, not nested inside it, to stay within '
            'the four-level program bound. '
            'evidence. Before another target, move_to_observe returns under the retained collision scene, '
            'then observe_scene associates the same fixed inventory. Maps use only listed entities and reuse '
            'the first measured table cuboids. verify_place updates only the local pink obstacle from head '
            'RGBD before VLA movement. verify_unpack may use head or wrist RGBD to add the visible black '
            'obstacle BEFORE its mandatory return; this does not remap the background. After the return, '
            'observe_scene and build_scene refresh all inventory geometry globally. No movement is allowed '
            'between that observation-pose capture and grasp planning. No raw simulator object pose is '
            'visual evidence. Responses is never rerun. '
            'The configured unmarked destination is named demo-destination by the executor; do not invent '
            'a visible marker. Program must be generated from the request; no hidden default plan exists.'
        }, ensure_ascii=False)}]
        for camera in ('2', '3', '1'):
            height, width = capture['views'][camera]['rgb'].shape[:2]
            content.extend([{'type': 'input_text', 'text': json.dumps({
                                'image_role': 'RGB camera ' + camera, 'camera_id': camera,
                                'width_px': width, 'height_px': height,
                                'bbox_coordinates': 'x / width_px; y / height_px, independently for this image'})},
                            image_part(capture['views'][camera]['rgb'])])
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}],
            'reasoning': {'effort': 'low'}, 'max_output_tokens': 10000,
            'text': {'format': {'type': 'json_schema', 'name': 'demo_scene_and_skill_program',
                               'strict': True, 'schema': demo_discovery_schema()}}}

        def parse(body, evidence):
            if body.get('status') != 'completed':
                raise VisualInventoryError('visual_discovery_incomplete')
            texts = [part['text'] for item in body.get('output', []) if item.get('type') == 'message'
                     for part in item.get('content', []) if part.get('type') == 'output_text']
            output = DemoDiscoveryOutput.model_validate_json(''.join(texts))
            task = output.task
            if not task.supported:
                raise VisualInventoryError('demo_task_unsupported', evidence={'reason': task.reason})
            refs = {item.ref: item for item in output.entities}
            source = refs.get(task.source_bin_ref)
            destination = refs.get(task.destination_table_ref)
            if source is None or source.kind != 'bin' or source.color.lower() != 'blue':
                raise VisualInventoryError('demo_blue_source_bin_unconfirmed')
            if destination is None or destination.kind != 'table':
                raise VisualInventoryError('demo_destination_table_unconfirmed')
            right_bin = refs.get(task.right_bin_ref)
            if task.workflow == 'unpack_transfer' and (right_bin is None or right_bin.kind != 'bin'
                    or right_bin.color != 'blue' or right_bin.ref == source.ref):
                raise VisualInventoryError('demo_right_bin_unconfirmed')
            sources = [item for item in output.entities
                       if item.parent_ref == source.ref and item.color.lower() == 'pink'
                       and item.kind not in {'table', 'bin', 'marker'}]
            if not sources or any(item.membership == 'unknown' for item in sources):
                raise VisualInventoryError('source_inventory_unknown')
            if any(item.reference_camera_id != '2' for item in sources):
                raise VisualInventoryError('demo_source_left_reference_required')
            if task.requested_count is not None and task.requested_count > len(sources):
                raise VisualInventoryError('requested_count_exceeds_confirmed_inventory')
            ids = {item.ref: ('table' if item.ref == destination.ref else
                             'instance-' + uuid4().hex[:16]) for item in output.entities}
            entities = {}
            for item in output.entities:
                identifier = ids[item.ref]
                entities[identifier] = {
                    'id': identifier, 'kind': item.kind, 'color': item.color, 'description': item.description,
                    'bbox': item.bbox, 'reference_camera_id': item.reference_camera_id,
                    'initial_parent_id': ids.get(item.parent_ref), 'identity_source': self.source,
                    'visual_membership': 'excluded',
                    'roles': ['table', 'collision'] if item.kind == 'table' else
                             ['target_region'] if item.kind == 'marker' else
                             ['container', 'collision'] if item.kind == 'bin' else ['collision']}
                if item.kind not in {'table', 'marker'}:
                    entities[identifier]['segmentation_query'] = {
                        'kind': item.kind, 'color': item.color, 'prompt': item.segmentation_prompt}
            ordered = head_rgb_box_order([entities[ids[item.ref]] for item in sources])
            selected = ordered if task.requested_count is None else ordered[:task.requested_count]
            for number, identifier in enumerate(ordered, 1):
                entities[identifier].update(display_id=number)
                if identifier in selected:
                    entities[identifier]['visual_membership'] = 'required'
            demo_task = {'scope': task.scope, 'requested_count': task.requested_count,
                'source_bin_id': ids[source.ref], 'destination_table_id': ids[destination.ref],
                'source_object_ids': ordered, 'transport_ids': selected, 'ordering_camera_id': '2',
                'source_inventory_complete': task.source_inventory_complete}
            if task.workflow == 'unpack_transfer':
                arms = {item.object_ref: item for item in task.transport_arms}
                if len(arms) != len(task.transport_arms) or set(arms) != {item.ref for item in sources}:
                    raise VisualInventoryError('demo_item_arm_assignments_required')
                demo_task.update(workflow='unpack_transfer', right_bin_id=ids[right_bin.ref],
                    prior='pink foam wraps one black core; unpack process links distinct input and output entities',
                    item_lifecycle={ids[ref]: {'phase': 'wrapped_source', 'appearance': 'wrapped',
                        'appearance_revision': 0, 'to_table_arm': item.to_table_arm,
                        'to_right_bin_arm': item.to_right_bin_arm} for ref, item in arms.items()})
            # Canonical SAM3 vocabulary applies only to validated source IDs/types;
            # identity still requires current Decisions evidence.
            entities[demo_task['source_bin_id']]['segmentation_query']['prompt'] = 'blue box'
            for identifier in demo_task['source_object_ids']:
                entity = entities[identifier]
                entity['segmentation_query']['prompt'] = 'pink object'
                entity['segmentation_prompts'] = [
                    'pink object', 'pink foam wrapped object',
                    'object inside the left blue box', 'elliptical pink object']
            program = output.program.model_dump()
            reference = {'observation_id': capture['observation_id'], 'entities': entities,
                'views': {camera: {'rgb': capture['views'][camera]['rgb']} for camera in ('2', '3', '1')},
                'demo_task': deepcopy(demo_task), 'demo_program': deepcopy(program)}
            return {'entities': entities, 'reference': reference, 'evidence': evidence,
                    'demo_task': demo_task, 'demo_program': program}
        return self.request(payload, capture, 'initialize_inventory', parse)

"""One initial RGB inventory, independent of segmentation and depth."""
import json
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

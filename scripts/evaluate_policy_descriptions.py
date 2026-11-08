"""Compare skill prose against one archived initial request, without moving the robot."""
import argparse
import base64
import io
import json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from luna_mujoco.agent.graph_prompt import FINAL_GRAPH_CHECK, GRAPH_RULES
from luna_mujoco.agent.responses import ResponsesCompiler
from luna_mujoco.visual_api import VisualAPI, VisualInventoryError


class FrozenImagesCompiler(ResponsesCompiler):
    def __init__(self, *, frozen_images, frozen_schema, **kwargs):
        super().__init__(**kwargs)
        self.frozen_images = frozen_images
        self.frozen_schema = frozen_schema

    def request(self, payload, *args, **kwargs):
        if payload['text'] != self.frozen_schema:
            raise ValueError('ablation_baseline_schema_changed: use a run compiled by the current schema')
        # Preserve the exact original JPEG bytes, not a decode/re-encode variant.
        images = iter(self.frozen_images)
        for part in payload['input'][0]['content']:
            if part['type'] == 'input_image':
                part['image_url'] = next(images)
        return super().request(payload, *args, **kwargs)


def summary(event):
    request = event['request']
    text = ''.join(part.get('text', '') for part in request['input'][0]['content'])
    return {'elapsed_s': event.get('elapsed_s'), 'text_characters': len(text),
            'usage': (event.get('response') or {}).get('usage', {}), 'error': event.get('error')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8350')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--detail', choices=('compact', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, trust_env=False, timeout=30) as client:
        prefix = f'/api/experiments/{args.run_id}'

        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response

        events = get(prefix + '/events').json()['events']
        entry = next(event for event in events if event['kind'] == 'visual_decisions_call'
                     and event.get('data', {}).get('endpoint', '').endswith('/responses'))
        record = get(prefix + '/files/' + entry['artifact']).json()
        original = get(prefix + '/files/' + record['artifact']).json()
        parts = original['request']['input'][0]['content']
        context = json.loads(parts[0]['text'])
        if context.get('rules') != GRAPH_RULES or parts[-1].get('text') != FINAL_GRAPH_CHECK:
            raise ValueError('ablation_baseline_prompt_changed: use a run compiled by the current prompt')
        capture = {'observation_id': original['observation_id'],
                   'generation': context.get('generation'), 'views': {}}
        frozen_images = []
        camera = None
        for part in parts[1:]:
            if part['type'] == 'input_text':
                try:
                    camera = json.loads(part['text']).get('camera_id')
                except json.JSONDecodeError:
                    continue
            elif part['type'] == 'input_image':
                raw = get(prefix + '/files/' + part['image_url']['path']).content
                frozen_images.append('data:image/jpeg;base64,' + base64.b64encode(raw).decode())
                capture['views'][camera] = {'rgb': np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))}
    if set(capture['views']) != {'1', '2', '3'}:
        raise ValueError('archive_requires_three_camera_images')
    root = Path.cwd() / 'skill_catalog'
    catalog = context['skill_catalog']
    documents = {}
    for name, contract in catalog.items():
        current = json.loads((root / name / 'contract.json').read_text(encoding='utf-8'))
        # Only restore the explanatory fields omitted by compact presentation.
        contract.update(id=name, capability=current['capability'], description=current['description'])
        documents[name] = (root / name / 'SKILL.md').read_text(encoding='utf-8')
    context['skill_instructions'] = documents
    config = VisualAPI.from_environment()
    calls = []
    compiler = FrozenImagesCompiler(frozen_images=frozen_images, frozen_schema=original['request']['text'],
        api_key=config.api_key,
        timeout_s=180, audit=calls.append, registry=catalog, skill_detail=args.detail,
        configured_regions=context['configured_regions'], known_assets=context['known_assets'],
        initial_facts=context['initial_facts'], scene_context=context['scene_context'])
    result = {'source_run': args.run_id, 'baseline_detail': context.get('skill_description_detail'),
              'baseline': summary(original), 'variant_detail': args.detail,
              'limitation': 'One frozen-input comparison; no execution or success-rate estimate.'}
    try:
        plan = compiler._generate(capture, context)
        result.update(valid=True, plan=plan.model_dump(mode='json'))
    except VisualInventoryError as error:
        result.update(valid=False, reason=str(error), evidence=error.evidence)
    if calls:
        result['variant'] = summary(calls[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in {'plan', 'evidence'}},
                     ensure_ascii=False), flush=True)
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

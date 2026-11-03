"""Side-by-side locating aids for verified same-capture comparison questions."""
import json
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw

from .demo_identity_presentation import FRAME_COLOR, _decode_image
from .observation_labels import raster_visual_annotations
from .visual_inventory import image_part


def scope_comparison_questions(payload):
    """Bind verified comparison panels to their predicate at the request boundary."""
    messages = payload.get('input')
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return payload
    source = messages[0].get('content', [])
    if not isinstance(source, list) or not any(isinstance(part, dict)
            and 'same_capture_identity_comparison' in part.get('text', '') for part in source):
        return payload
    result = deepcopy(payload)
    questions = {q['name']: q for q in result['questions'] if q['type'] == 'predicate'}
    content = result['input'][0]['content']
    kept, index = [], 0
    while index < len(content):
        part = content[index]
        try:
            meta = json.loads(part.get('text', 'null'))
        except ValueError:
            meta = None
        question = questions.get(meta.get('question')) if isinstance(meta, dict) else None
        if (question is not None and meta.get('image_role') == 'same_capture_identity_comparison'
                and isinstance(question['instructions'], str) and index + 1 < len(content)
                and content[index + 1]['type'] == 'input_image'):
            text = (f'Do the labels {meta["reference"]} in the left panel and {meta["candidate"]} in the right panel '
                'refer to the same physical object? These panels come from the same camera and frozen capture. '
                'Compare the indicated object and its location among its neighbors. '
                'A box and an object inside it are separate objects. Partial visibility or different rectangle '
                'boundaries do not by themselves imply different objects. Answer false for different subjects; '
                'remain uncertain only when the indicated subjects cannot be distinguished.')
            question['instructions'] = [{'type': 'input_text', 'text': text}, part, content[index + 1]]
            index += 2
        else:
            kept.append(part)
            index += 1
    result['input'][0]['content'] = kept
    return result


def same_capture_comparison_images(payload, proposals):
    result = deepcopy(payload)
    content = result['input'][0]['content']
    context = json.loads(content[0]['text'])
    proof = context.get('same_capture_reference', {})
    current = context.get('observation_id')
    if (not proposals or not current or proof.get('initial_reference_observation_id') != current
            or proof.get('current_observation_id') != current
            or proof.get('generation') != context.get('generation')):
        return result
    annotated, raw, pending, raw_camera, existing = {}, {}, None, None, set()
    for part in content:
        if part['type'] == 'input_text':
            raw_camera = (part['text'].rsplit(' ', 1)[-1]
                          if part['text'].startswith('CURRENT RGB camera ') else None)
            try:
                pending = json.loads(part['text'])
            except ValueError:
                pending = None
            if isinstance(pending, dict) and pending.get('image_role') == 'same_capture_identity_comparison':
                existing.add(pending['question'])
        elif part['type'] == 'input_image':
            if raw_camera is not None:
                raw[raw_camera] = part
                raw_camera = None
            if isinstance(pending, dict) and pending.get('image_role') in {
                    'initial_reference_annotations', 'current_annotations'}:
                annotated[pending['image_role'], pending['camera']] = (pending, part)
            pending = None
    predicates = {q['name'] for q in result['questions'] if q['type'] == 'predicate'}
    for row in context.get('same_capture_box_correspondence_proposals', []):
        camera, candidate = row['camera'], row['candidate']
        name = 'identity_' + candidate.removeprefix('candidate-')
        if name not in proposals or name not in predicates or name in existing:
            continue
        if camera not in proof.get('verified_camera_ids', []):
            continue
        reference = annotated.get(('initial_reference_annotations', camera))
        observed = annotated.get(('current_annotations', camera))
        if reference is None or observed is None:
            continue
        refmeta, _ = reference
        curmeta, _ = observed
        if (not refmeta.get('source_image_sha256')
                or refmeta['source_image_sha256'] != curmeta.get('source_image_sha256')
                or refmeta.get('source_rgb_size_px') != curmeta.get('source_rgb_size_px')):
            continue
        ref = next((x for x in refmeta['items'] if x['id'] == proposals[name]), None)
        cur = next((x for x in curmeta['items'] if x['id'] == candidate), None)
        if ref is None or cur is None:
            continue
        if camera not in raw:
            continue
        rgb, image_hash = _decode_image(raw[camera])
        if (image_hash != refmeta['source_image_sha256']
                or [rgb.shape[1], rgb.shape[0]] != refmeta['source_rgb_size_px']):
            continue
        # Show just the subject named by each question. Other physical subjects
        # remain in raw RGB, but unrelated labels/boxes cannot cross this pair.
        refimage, curimage = [Image.fromarray(raster_visual_annotations(
            rgb, [item], frame_color=FRAME_COLOR, label_layout='near_box')) for item in (ref, cur)]
        a, b = ref['box'], cur['box']
        w, h = refimage.size
        bounds = (max(0, int(min(a[0], b[0])*w)-16), max(0, int(min(a[1], b[1])*h)-32),
                  min(w, int(max(a[2], b[2])*w)+16), min(h, int(max(a[3], b[3])*h)+16))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        crops = [im.crop(bounds) for im in (refimage, curimage)]
        scale = min(3., 480/crops[0].height, 384/crops[0].width)
        crops = [im.resize((round(im.width*scale), round(im.height*scale)), Image.Resampling.NEAREST)
                 for im in crops]
        panel = Image.new('RGB', (2*crops[0].width+8, crops[0].height+32), 'white')
        draw = ImageDraw.Draw(panel)
        for index, (crop, label) in enumerate(zip(crops, [ref['image_label'], cur['image_label']], strict=True)):
            x = index*(crop.width+8)
            panel.paste(crop, (x, 32))
            draw.text((x+5, 8), ('REFERENCE ' if index == 0 else 'CURRENT ') + label, fill='black')
        content.extend([{'type': 'input_text', 'text': json.dumps({
            'image_role': 'same_capture_identity_comparison', 'question': name,
            'reference': ref['image_label'], 'candidate': cur['image_label'], 'camera_id': camera,
            'observation_id': current, 'generation': context['generation'],
            'source_image_sha256': refmeta['source_image_sha256'], 'crop_bounds_xyxy_px': bounds,
            'meaning': 'Left is the reference subject, right is the current candidate. Both panels use the same '
                'crop coordinates in the same verified frozen capture. Labels locate subjects, not an answer. '
                'Judge whether the two indicated subjects are the same physical object. A container remains '
                'the subject even though its box also encloses separate contents; reject a contents mask '
                'when the reference subject is the container. Keep unknown for genuinely ambiguous subjects. '
                'Full original views remain above.'})}, image_part(np.asarray(panel))])
        existing.add(name)
    return result

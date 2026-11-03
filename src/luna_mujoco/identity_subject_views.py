"""Append capture-bound SAM subject pixels to an existing identity clarification."""
import base64
import json
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .multiview_identity import grouped_candidates, validate_group_proof
from .visual_api import VisualInventoryError
from .visual_inventory import array_stamp

SUBJECT_VIEW_RULE = (
    'The following additional views show every candidate questioned in this same frozen clarification. '
    'Each view uses that candidate camera\'s original CURRENT RGB and the already archived SAM silhouette. '
    'Pixels inside the silhouette are unchanged. Pixels outside it are replaced by neutral gray [128,128,128]. '
    'Gray is excluded image area, not the subject material, a new observation, or inferred hidden geometry. '
    'The original full scenes and annotations above remain available as context. '
    'C labels and cameras identify the same candidates as the original questions. For each identity predicate, '
    'judge the proposed subject itself, not another object inside its surrounding bounding rectangle. '
    'SAM may be imperfect and its mask is not an identity answer. Reference IDs, uncertainty, '
    'distinct-subject rules, shared instructions and all questions remain unchanged. '
    'No view is singled out according to an expected answer.')


def _metadata(part):
    if part.get('type') != 'input_text':
        return {}
    try:
        value = json.loads(part['text'])
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _bound_subject(candidate, row, subject, capture, context):
    provenance = row.get('candidate_provenance', {})
    camera = row.get('camera')
    view = capture.get('views', {}).get(camera, {})
    rgb, mask = view.get('rgb'), subject.get('silhouette_mask')
    valid = (
        provenance == subject.get('candidate_provenance')
        and provenance.get('candidate') == candidate and provenance.get('camera') == camera
        and provenance.get('stage') == 'segmentation_before_inventory_surface_assignment'
        and bool(capture.get('observation_id')) and type(capture.get('generation')) is int
        and all(provenance.get(key) == capture[key] == context.get(key) for key in ('observation_id', 'generation'))
        and isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        and isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == rgb.shape[:2]
        and bool(np.any(mask)) and array_stamp(mask) == provenance.get('silhouette_mask')
        and 'rgb' in provenance.get('view', {})
        and all(key in view and array_stamp(view[key]) == stamp for key, stamp in provenance.get('view', {}).items())
        and provenance.get('calibration') == {key: view[key] for key in
            ('calibration_id', 'intrinsics', 'frame_id', 'capture_time') if key in view})
    if not valid:
        raise VisualInventoryError('visual_identity_subject_binding_invalid', evidence={
            'candidate_id': candidate, 'camera_id': camera,
            'observation_id': capture.get('observation_id'), 'generation': capture.get('generation')})
    return rgb, mask, provenance


def _subject_image(rgb, mask, candidate, camera):
    pixels = np.full((rgb.shape[0] + 32, rgb.shape[1], 3), 255, dtype=np.uint8)
    pixels[32:] = np.where(mask[:, :, None], rgb, 128)
    image = Image.fromarray(pixels)
    try:
        font = ImageFont.truetype('Arial.ttf', 18)
    except OSError:
        font = ImageFont.load_default(size=18)
    ImageDraw.Draw(image).text((8, 6), 'C' + candidate.removeprefix('candidate-')
        + ' - CURRENT SAM SUBJECT - camera ' + camera, fill=(0, 0, 0), font=font)
    stream = BytesIO()
    image.save(stream, format='PNG')
    return {'type': 'input_image', 'image_url': 'data:image/png;base64,'
            + base64.b64encode(stream.getvalue()).decode('ascii')}


def append_identity_subject_views(payload, capture, subjects):
    """Preserve every original part/question; uniformly show only questioned candidates."""
    numbers = sorted({int(question['name'].split('_')[1]) for question in payload['questions']
                      if question['type'] == 'predicate' and question['name'].startswith(('match_', 'identity_'))})
    if not numbers:
        return payload
    content = payload['input'][0]['content']
    context = _metadata(content[0])
    validate_group_proof(context.get('current_measured_view_groups', {}), capture)
    numbers = sorted({int(member.removeprefix('candidate-')) for number in numbers
                      for member in grouped_candidates(context, 'candidate-' + str(number))})
    image_count = sum(part['type'] == 'input_image' for message in payload['input'] for part in message['content'])
    if image_count + len(numbers) > 128:
        raise VisualInventoryError('visual_inventory_input_limit')
    rows = {_metadata(part)['candidate']: _metadata(part) for part in content if 'candidate' in _metadata(part)}
    additions = []
    sizes = set()
    for number in numbers:
        candidate = 'candidate-' + str(number)
        row = rows.get(candidate, {})
        rgb, mask, provenance = _bound_subject(candidate, row, subjects.get(candidate, {}), capture, context)
        camera = provenance['camera']
        sizes.add((rgb.shape[1], rgb.shape[0]))
        description = {'image_role': 'current_candidate_sam_subject', 'candidate': candidate,
            'camera': camera, 'observation_id': capture['observation_id'], 'generation': capture['generation'],
            'source_rgb_sha256': provenance['view']['rgb']['sha256'],
            'source_silhouette_mask_sha256': provenance['silhouette_mask']['sha256'],
            'source_provenance': 'original request candidate_provenance for this same Candidate and camera',
            'source_rgb_size_px': [rgb.shape[1], rgb.shape[0]], 'display_size_px': [rgb.shape[1], rgb.shape[0] + 32],
            'original_image_offset_xy_px': [0, 32], 'outside_mask_rgb': [128, 128, 128],
            'resampling': 'none', 'encoding': 'lossless PNG', 'subject_pixels_unchanged': True}
        additions.extend([{'type': 'input_text', 'text': json.dumps(description, ensure_ascii=False)},
                          _subject_image(rgb, mask, candidate, camera)])
    dimensions = str(next(iter(sizes))[0]) + 'x' + str(next(iter(sizes))[1]) + ' ' if len(sizes) == 1 else ''
    rule = {'current_sam_subject_views': {'applies_to': ['candidate-' + str(number) for number in numbers],
        'rule': SUBJECT_VIEW_RULE, 'layout': 'One ' + dimensions + 'original image per candidate, no crop or resize; '
            'a separate 32px header names C and camera.'}}
    result = deepcopy(payload)
    result['input'][0]['content'].extend([{'type': 'input_text', 'text': json.dumps(rule, ensure_ascii=False)}, *additions])
    return result

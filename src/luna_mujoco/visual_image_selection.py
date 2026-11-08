"""Remove duplicate display evidence without dropping cameras or identity questions."""
import hashlib
import re
from copy import deepcopy


def decision_image_count(payload):
    shared = sum(part.get('type') == 'input_image'
        for message in payload.get('input', []) for part in message.get('content', []))
    scoped = sum(part.get('type') == 'input_image'
        for question in payload.get('questions', []) if isinstance(question.get('instructions'), list)
        for part in question['instructions'])
    return shared + scoped


def compact_decision_images(payload):
    original = decision_image_count(payload)
    payload = deepcopy(payload)
    images, selected = {}, {}
    for message_index, message in enumerate(payload.get('input', [])):
        current_raw = False
        for part_index, part in enumerate(message.get('content', [])):
            if part.get('type') == 'input_text':
                # Only standalone camera labels establish a role, never prompt prose.
                current_raw = re.fullmatch(
                    r'CURRENT(?: RGB camera [123]| camera [123] — original RGB)', part['text']) is not None
            elif part.get('type') == 'input_image':
                digest = hashlib.sha256(part['image_url'].encode()).hexdigest()
                position = (message_index, part_index)
                images[position] = {'digest': digest, 'current_raw': current_raw}
                previous = selected.get(digest)
                if previous is None or current_raw and not images[previous]['current_raw']:
                    selected[digest] = position
                current_raw = False
            else:
                current_raw = False
    # Number the images in their final order across messages, including forward aliases.
    numbers = {position: number for number, position in enumerate(sorted(selected.values()), 1)}
    for position, entry in images.items():
        message_index, part_index = position
        retained_position = selected[entry['digest']]
        if position == retained_position:
            continue
        direction = 'above' if retained_position < position else 'below'
        text = (f'Identical image bytes to image {numbers[retained_position]} {direction}; '
                'the preceding role also refers to that image. This is image reuse, '
                'not a new observation or an identity judgment.')
        payload['input'][message_index]['content'][part_index] = {'type': 'input_text', 'text': text}
    # Question images are local evidence: an alias to another question would
    # change its meaning. Count them, but never deduplicate across questions.
    retained = decision_image_count(payload)
    return payload, {'original_image_count': original, 'image_count': retained,
                     'removed_image_count': original - retained}

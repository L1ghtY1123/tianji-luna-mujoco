"""Remove duplicate display evidence without dropping cameras or identity questions."""
import hashlib
import json
from copy import deepcopy


def compact_decision_images(payload):
    payload = deepcopy(payload)
    seen, original, retained = {}, 0, 0
    for message in payload.get('input', []):
        content, empty_annotation = [], False
        for part in message.get('content', []):
            if part.get('type') == 'input_text':
                empty_annotation = False
                try:
                    metadata = json.loads(part['text'])
                except (ValueError, TypeError):
                    metadata = None
                if isinstance(metadata, dict) and metadata.get('image_role') in {
                        'initial_reference_annotations', 'current_annotations'}:
                    empty_annotation = not metadata.get('items') and not metadata.get('regions')
                content.append(part)
            elif part.get('type') == 'input_image':
                original += 1
                digest = hashlib.sha256(part['image_url'].encode()).hexdigest()
                if empty_annotation:
                    content.append({'type': 'input_text', 'text':
                        'No locating boxes or regions in this annotation; use the full RGB above.'})
                elif digest in seen:
                    content.append({'type': 'input_text', 'text':
                        f'Identical image bytes to image {seen[digest]} above; the preceding role also refers '
                        'to that image. This is image reuse, not a new observation or an identity judgment.'})
                else:
                    retained += 1
                    seen[digest] = retained
                    content.append(part)
                empty_annotation = False
            else:
                content.append(part)
        message['content'] = content
    return payload, {'original_image_count': original, 'image_count': retained,
                     'removed_image_count': original - retained}

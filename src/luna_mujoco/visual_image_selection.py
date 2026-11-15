"""Count authored multimodal evidence without changing image roles or order."""
import json
from copy import deepcopy

from .evidence_versions import DECISION_INPUT_VERSION


def bind_decision_images(payload, capture):
    """Number the actual transmitted images and validate their adjacent provenance."""
    payload = deepcopy(payload)
    contents = [message.get('content', []) for message in payload.get('input', [])]
    contents.extend(question['instructions'] for question in payload.get('questions', [])
                    if isinstance(question.get('instructions'), list))
    image_index = 0
    for content in contents:
        for index, part in enumerate(content):
            if part.get('type') != 'input_image':
                continue
            if index == 0 or content[index - 1].get('type') != 'input_text':
                raise ValueError('decision_image_metadata_missing')
            metadata = json.loads(content[index - 1]['text'])
            if not isinstance(metadata, dict) or not metadata.get('modality') or not metadata.get('image_size_px'):
                raise ValueError('decision_image_metadata_invalid')
            if metadata.get('role') in {'current', 'same_capture_reference'} and (
                    metadata.get('observation_id') != capture.get('observation_id')
                    or metadata.get('generation') != capture.get('generation')
                    or metadata.get('camera_id') not in capture.get('views', {})):
                raise ValueError('decision_current_image_capture_mismatch')
            if metadata.get('role') not in {'current', 'same_capture_reference', None} and (
                    not metadata.get('observation_id') or metadata.get('generation') is None):
                raise ValueError('decision_dated_image_provenance_missing')
            image_index += 1
            metadata['evidence_input_version'] = DECISION_INPUT_VERSION
            metadata['image_id'] = f'image_{image_index:03d}'
            metadata['image_index'] = image_index
            content[index - 1]['text'] = json.dumps(metadata, ensure_ascii=False)
    return payload


def decision_image_count(payload):
    shared = sum(part.get('type') == 'input_image'
        for message in payload.get('input', []) for part in message.get('content', []))
    scoped = sum(part.get('type') == 'input_image'
        for question in payload.get('questions', []) if isinstance(question.get('instructions'), list)
        for part in question['instructions'])
    return shared + scoped

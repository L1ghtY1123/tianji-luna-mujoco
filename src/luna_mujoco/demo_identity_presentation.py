"""Label complete camera views without adding crops or duplicate subject images."""
import base64
import hashlib
import json
import re
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image

from .observation_labels import raster_visual_annotations
from .visual_inventory import image_part

FRAME_COLOR = (235, 235, 235)


def _image_label(label):
    """Compact pixel labels retain their full names and canonical IDs in context."""
    anchor = label.startswith('Current anchor · ')
    label = label.removeprefix('Current anchor · ')
    label = re.sub(r'\bReference (\d+)', r'R\1', label)
    return label + ('*' if anchor else '')


def release_verification_context(progress, purpose):
    """Expose recorded pending releases as context, never as visual confirmation."""
    if purpose != "verify_transfer":
        return None
    targets = []
    for identifier, record in (progress or {}).items():
        if not record.get("release_feedback_id") or record.get("status") == "completed":
            continue
        targets.append({"entity_id": identifier, **{
            key: deepcopy(record[key]) for key in (
                "release_feedback_id", "destination_id", "region_id", "transfer_index", "placement_region")
            if key in record}})
    if not targets:
        return None
    return {"source": "controller_recorded_release", "targets": targets,
        "rule": "These are recorded controller actions for the selected permanent IDs, not proof of current "
                "visual identity, support, release or success. The IDs were selected from the initial inventory; "
                "their recorded release feedback and assigned destinations provide temporal context for a possible "
                "move. A selected status can persist while visual verification is pending. "
                "Independently match current RGB to the permanent subject, and check visible support in its assigned "
                "region and separation from the gripper. Destination-region geometry is intention, not a measured object "
                "pose. Keep unknown if identity or placement remains ambiguous; do not assign an ID or success "
                "solely from this action record."}


def decode_image_part(part):
    encoded = part["image_url"]
    if not isinstance(encoded, str) or not encoded.startswith("data:image/jpeg;base64,"):
        raise ValueError("demo_identity_presentation_requires_embedded_jpeg")
    data = base64.b64decode(encoded.split(",", 1)[1], validate=True)
    return np.asarray(Image.open(BytesIO(data)).convert("RGB")), hashlib.sha256(data).hexdigest()


def box_overlap(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def overlapping_reference_view(candidate, references):
    return any(box_overlap(candidate['box'], reference['box']) >= .5 for reference in references)


def improve_identity_presentation(payload):
    """Label each complete camera view once, with stable reference and candidate names."""
    payload = deepcopy(payload)
    content = payload["input"][0]["content"]
    context = json.loads(content[0]["text"])
    entities = context["memory_entities"]
    labels = {identifier: 'Reference ' + identifier for identifier in entities}
    legend = {label: {"entity_id": identifier, "image_label": _image_label(label),
                      "description": entities[identifier].get("description", "")}
              for identifier, label in labels.items()}
    raw, annotations, records = {}, {}, []
    pending = None
    for index, part in enumerate(content):
        if part["type"] == "input_text":
            text = part["text"]
            if text.startswith("INITIAL REFERENCE RGB camera "):
                pending = ("initial", text.rsplit(" ", 1)[-1])
            elif text.startswith("CURRENT RGB camera "):
                pending = ("current", text.rsplit(" ", 1)[-1])
            else:
                try:
                    metadata = json.loads(text)
                except ValueError:
                    continue
                role = {"initial_reference_annotations": "initial", "current_annotations": "current"}.get(
                    metadata.get("image_role"))
                if (metadata.get('image_role') == 'instance_reference'
                        and metadata.get('pixels_identical_to_current_same_camera') is True):
                    role = 'initial'
                    metadata['image_role'] = 'initial_reference_annotations'
                if role:
                    key = (role, metadata["camera"])
                    annotations[key] = metadata
                    records.append((index, key, metadata))
        elif part["type"] == "input_image" and pending is not None:
            raw[pending] = part
            pending = None
    for (role, camera), metadata in annotations.items():
        if (role == 'initial' and metadata.get('pixels_identical_to_current_same_camera') is True
                and ('current', camera) in raw):
            raw['initial', camera] = raw['current', camera]
    identical_cameras = [camera for role, camera in raw if role == 'current'
                         and ('initial', camera) in raw
                         and raw[('initial', camera)]['image_url'] == raw[('current', camera)]['image_url']]
    context['identical_reference_image_cameras'] = identical_cameras
    # Choose a representative view only. A model verdict still establishes its ID.
    context['reference_view_questions'] = [
        'identity_' + item['id'].removeprefix('candidate-')
        for camera in identical_cameras for item in annotations.get(('current', camera), {}).get('items', [])
        if item['id'].startswith('candidate-')
        and overlapping_reference_view(item, annotations.get(('initial', camera), {}).get('items', []))]
    for entity in entities.values():
        proposal = entity.get('current_identity_proposal', {})
        if proposal.get('observation_id') != context['observation_id']:
            continue
        for item in annotations.get(('current', proposal['camera_id']), {}).get('items', []):
            if item['id'].startswith('candidate-') and box_overlap(item['box'], proposal['bbox']) >= .5:
                context['reference_view_questions'].append(
                    'identity_' + item['id'].removeprefix('candidate-'))
    for question in payload["questions"]:
        # Deepcopy of the whole payload preserves aliases between questions.
        # Own each option here too, including historical archived requests.
        if 'choices' in question:
            question["choices"] = deepcopy(question["choices"])
        if question["name"].startswith("identity_"):
            number = question["name"].removeprefix("identity_")
            labels["candidate-" + number] = "C" + number
        for option in question.get('choices', []):
            if option["value"] in entities:
                description = re.sub(r"^(?:(?:R|Reference )\d+ · )+", "", option.get("description", ""))
                option["description"] = labels[option["value"]] + " · " + description
    context["identity_display_legend"] = legend
    context["candidate_display_legend"] = {label: identifier for identifier, label in labels.items()
                                          if identifier.startswith("candidate-")}
    context['image_label_legend'] = {_image_label(label): {
        'display_label': label,
        'candidate' if identifier.startswith('candidate-') else 'entity_id': identifier}
        for identifier, label in labels.items()}
    for anchor in context.get('current_confirmed_container_anchors', []):
        anchor['display_label'] = 'Current anchor · ' + labels[anchor['entity_id']]
        anchor['image_label'] = _image_label(anchor['display_label'])
        context['image_label_legend'][anchor['image_label']] = {
            'display_label': anchor['display_label'], 'entity_id': anchor['entity_id'],
            'annotation_role': 'current_confirmed_container_anchor'}
    context["display_label_rule"] = (
        "C labels locate current candidates. R labels locate permanent references. "
        "R* labels locate confirmed current anchors. Labels match image_label_legend exactly.")
    content[0]["text"] = json.dumps(context, ensure_ascii=False)
    for index, key, metadata in records:
        if key not in raw:
            continue
        rgb, image_hash = decode_image_part(raw[key])
        located_items = list(metadata['items'])
        reference_key = ('initial', key[1])
        if key[0] == 'current' and key[1] in identical_cameras:
            located_items.extend(annotations.get(reference_key, {}).get('items', []))
            metadata['includes_initial_references'] = True
        items = [{**item, "display_label": labels.get(item["id"], item["id"])}
                 for item in located_items]
        for item in items:
            if item.get('annotation_role') == 'current_confirmed_container_anchor':
                item['display_label'] = 'Current anchor · ' + item['display_label']
            item['image_label'] = _image_label(item['display_label'])
        metadata.update(items=items, frame_color=list(FRAME_COLOR),
                        source_rgb_size_px=[rgb.shape[1], rgb.shape[0]], source_image_sha256=image_hash,
                        annotation_layout='near_box', derived_image=True,
                        coordinates='boxes normalized to full source RGB; short labels adjacent to their boxes')
        content[index]["text"] = json.dumps(metadata)
        content[index + 1] = image_part(raster_visual_annotations(
            rgb, items, regions=metadata.get("regions", []), frame_color=FRAME_COLOR, label_layout='near_box'))
    return payload

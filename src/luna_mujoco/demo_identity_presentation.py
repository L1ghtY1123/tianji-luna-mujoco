"""Render identity locating aids without assigning or merging physical identities."""
import base64
import hashlib
import json
import re
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image

from .multiview_identity import grouped_view_instruction
from .observation_labels import raster_visual_annotations
from .visual_inventory import image_part

FRAME_COLOR = (235, 235, 235)


def _image_label(label):
    """Compact pixel labels retain their full names and canonical IDs in context."""
    anchor = label.startswith('Current anchor · ')
    label = label.removeprefix('Current anchor · ')
    label = re.sub(r'\bReference (\d+)', r'R\1', label)
    label = re.sub(r'\bCandidate (\d+)', r'C\1', label)
    return label + ('*' if anchor else '')


def ground_identity_locations(payload):
    """Put existing calibrated measurements next to the instance question they support."""
    result = deepcopy(payload)
    context = json.loads(result['input'][0]['content'][0]['text'])
    locations = {row['candidate']: row for row in context.get('candidate_geometric_location_hints', [])}
    marker = ' Current measured candidate surface: '
    for question in result['questions']:
        if not question['name'].startswith('identity_'):
            continue
        candidate = 'candidate-' + question['name'].removeprefix('identity_')
        if candidate not in locations:
            continue
        question['instructions'] = question['instructions'].split(marker, 1)[0] + marker + json.dumps(
            locations[candidate]) + (
                '. Coordinates are metres in one calibrated robot frame: +Y is robot LEFT, -Y is RIGHT, '
                '+X is forward. Compare the same physical instance, not merely the same category, color '
                'or CAD model. Check other CURRENT candidates and reference-located subjects together: '
                'simultaneously separate objects cannot share one ID. Different cameras can see different '
                'partial surfaces of one object, so disjoint point bounds alone are not an identity verdict. '
                'Use RGB, reference context and these measurements jointly; reject an outside-inventory '
                'subject and keep unknown for ambiguous correspondence.')
    return result


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
                "release_feedback_id", "destination_id", "region_id", "slot_id", "transfer_index", "assigned_slot")
            if key in record}})
    if not targets:
        return None
    return {"source": "controller_recorded_release", "targets": targets,
        "rule": "These are recorded controller actions for the selected permanent IDs, not proof of current "
                "visual identity, support, release or success. The IDs were selected from the initial inventory; "
                "their recorded release feedback and assigned destinations provide temporal context for a possible "
                "move. A selected status can persist while visual verification is pending. "
                "Independently match current RGB to the permanent subject, and check visible support in its assigned "
                "region and separation from the gripper. Assigned-slot geometry is intention, not a measured object "
                "pose. Keep unknown if identity or placement remains ambiguous; do not assign an ID or success "
                "solely from this action record."}


def _decode_image(part):
    encoded = part["image_url"]
    if not isinstance(encoded, str) or not encoded.startswith("data:image/jpeg;base64,"):
        raise ValueError("demo_identity_presentation_requires_embedded_jpeg")
    data = base64.b64decode(encoded.split(",", 1)[1], validate=True)
    return np.asarray(Image.open(BytesIO(data)).convert("RGB")), hashlib.sha256(data).hexdigest()


def _source_zoom(context, raw, annotations, labels):
    for camera in sorted(context.get('same_capture_reference', {}).get('verified_camera_ids', [])):
        parts = _source_camera_zoom(context, raw, annotations, labels, camera)
        if parts:
            return parts
    return []


def _source_camera_zoom(context, raw, annotations, labels, camera):
    proof = context.get("same_capture_reference", {})
    if (camera not in proof.get("verified_camera_ids", [])
            or not context.get("observation_id")
            or context["observation_id"] != proof.get("initial_reference_observation_id")
            or context["observation_id"] != proof.get("current_observation_id")):
        return []
    initial, current = raw.get(("initial", camera)), raw.get(("current", camera))
    if initial is None or current is None or initial["image_url"] != current["image_url"]:
        return []
    source_ids = context.get("source_inventory", {}).get("source_object_ids", [])
    references = annotations.get(("initial", camera), {}).get("items", [])
    by_id = {item["id"]: item for item in references}
    if not 1 <= len(source_ids) <= 4 or any(identifier not in by_id for identifier in source_ids):
        return []
    source_items = [by_id[identifier] for identifier in source_ids]
    boxes = np.asarray([item["box"] for item in source_items], dtype=float)
    if (boxes.shape != (len(source_ids), 4) or not np.isfinite(boxes).all()
            or np.any(boxes < 0) or np.any(boxes > 1)
            or np.any(boxes[:, 2:] <= boxes[:, :2])):
        return []
    rgb, image_hash = _decode_image(initial)
    height, width = rgb.shape[:2]
    lower = np.floor(boxes[:, :2].min(axis=0) * [width, height]).astype(int)
    upper = np.ceil(boxes[:, 2:].max(axis=0) * [width, height]).astype(int)
    padding = max(4, int(np.ceil(.15 * max(upper - lower))))
    lower = np.maximum(lower - padding, 0)
    upper = np.minimum(upper + padding, [width, height])
    crop_width, crop_height = upper - lower
    crop = rgb[lower[1]:upper[1], lower[0]:upper[0]]
    scale = min(4., 520. / crop_width, 600. / crop_height)
    display_size = np.maximum(np.rint([crop_width * scale, crop_height * scale]).astype(int), 1)
    enlarged = np.asarray(Image.fromarray(crop).resize(tuple(display_size), Image.Resampling.NEAREST))
    parts = []
    for role, items in (("initial", source_items),
                        ("current", annotations.get(("current", camera), {}).get("items", []))):
        local_items = []
        for item in items:
            box = np.asarray(item["box"]) * [width, height, width, height]
            # Broad container proposals remain in the full frame; this view
            # displays only boxes contained in the complete source-group crop.
            if np.any(box[:2] < lower) or np.any(box[2:] > upper):
                continue
            local_box = (box - np.tile(lower, 2)) / np.tile([crop_width, crop_height], 2)
            local_items.append({**item, "box": local_box.tolist(),
                                "display_label": item.get("display_label", labels.get(item["id"], item["id"]))})
        metadata = {
            "image_role": "same_capture_source_group_zoom", "annotation_role": role,
            "source_camera_id": camera, "source_object_ids": source_ids,
            "observation_id": context["observation_id"],
            "source_image_sha256": image_hash, "source_rgb_size_px": [width, height],
            "crop_bounds_xyxy_px": [*lower.tolist(), *upper.tolist()],
            "display_size_px": display_size.tolist(),
            "scale_xy": (display_size / [crop_width, crop_height]).tolist(),
            "resampling": "nearest; display magnification only",
            "items": local_items, "frame_color": list(FRAME_COLOR),
            "coordinates": "item boxes normalized to crop; short labels are adjacent to their boxes",
            "annotation_layout": "near_box", "derived_image": True,
            "meaning": "Same frozen RGB source group, not new evidence or established correspondence. "
                       "All source reference neighbors are retained. Broad candidate boxes are shown "
                       "in full views, not this crop. Reference labels are permanent references; Candidate labels are "
                       "unassociated proposals. Return unknown if identity remains ambiguous.",
        }
        parts.extend([{"type": "input_text", "text": json.dumps(metadata)},
                      image_part(raster_visual_annotations(enlarged, local_items, frame_color=FRAME_COLOR,
                                                           label_layout='near_box'))])
    return parts


def _current_candidate_zooms(context, raw, annotations, labels, images_per_camera):
    """Magnify current proposal pixels only; neither masks nor identity assignments."""
    if context.get('purpose') != 'associate_inventory':
        return []
    parts = []
    for (role, camera), metadata in annotations.items():
        if role != 'current' or (role, camera) not in raw:
            continue
        rgb, image_hash = _decode_image(raw[(role, camera)])
        height, width = rgb.shape[:2]
        camera_images = 0
        groups = []
        for item in metadata['items']:
            if item.get('annotation_role') == 'current_confirmed_container_anchor':
                continue  # Already located in the full current view; no new proposal/detail request.
            box = np.asarray(item['box'], float) * [width, height, width, height]
            if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[:2] < 0)
                    or np.any(box[2:] > [width, height]) or np.any(box[2:] <= box[:2])):
                continue
            # Large bins are already clear in the full frame. Reserve each
            # camera's detail images for small proposals.
            if np.prod(box[2:] - box[:2]) > .08 * width * height:
                continue
            # Near-identical rectangles share an enlarged image, not an ID or question.
            group = next((group for group in groups if np.max(np.abs(group[0][1] - box)) <= 2.), None)
            if group is None:
                groups.append([(item, box)])
            else:
                group.append((item, box))
        for group in groups:
            boxes = np.asarray([box for _, box in group])
            lower, upper = np.floor(boxes[:, :2].min(0)), np.ceil(boxes[:, 2:].max(0))
            padding = max(16, int(np.ceil(.5 * max(upper - lower))))
            lower = np.maximum(lower.astype(int) - padding, 0)
            upper = np.minimum(upper.astype(int) + padding, [width, height])
            crop_size = upper - lower
            scale = min(4., 512. / max(crop_size))
            if scale <= 1. or camera_images >= images_per_camera:
                continue  # Large proposals already occupy much of the full image.
            display_size = np.rint(crop_size * scale).astype(int)
            crop = rgb[lower[1]:upper[1], lower[0]:upper[0]]
            enlarged = np.asarray(Image.fromarray(crop).resize(tuple(display_size), Image.Resampling.NEAREST))
            candidate_labels = [labels.get(item['id'], item['id']) for item, _ in group]
            local_box = (np.concatenate([boxes[:, :2].min(0), boxes[:, 2:].max(0)])
                         - np.tile(lower, 2)) / np.tile(crop_size, 2)
            display_item = {'id': group[0][0]['id'], 'box': local_box.tolist(),
                            'display_label': ' / '.join(candidate_labels),
                            'image_label': ' / '.join(_image_label(label) for label in candidate_labels)}
            zoom = {'image_role': 'current_candidate_detail_zoom', 'source_camera_id': camera,
                'observation_id': context['observation_id'], 'generation': context.get('generation'),
                'source_image_sha256': image_hash, 'source_rgb_size_px': [width, height],
                'crop_bounds_xyxy_px': [*lower.tolist(), *upper.tolist()],
                'display_size_px': display_size.tolist(), 'candidate_labels': candidate_labels,
                'image_labels': [_image_label(label) for label in candidate_labels],
                'annotation_layout': 'near_box', 'derived_image': True,
                'original_candidates': [deepcopy(item) for item, _ in group],
                'resampling': 'nearest; at most 4x display magnification of existing pixels',
                'meaning': 'CURRENT RGB proposal and surrounding pixels, not a mask, new observation or '
                    'established identity. A shared crop does not merge candidates or their questions. '
                    'This crop does not prove inventory completeness; inspect all full CURRENT images '
                    'for other subjects and keep unknown when the proposal cannot be matched.'}
            parts.extend([{'type': 'input_text', 'text': json.dumps(zoom)},
                          image_part(raster_visual_annotations(enlarged, [display_item], frame_color=FRAME_COLOR,
                                                               label_layout='near_box'))])
            camera_images += 1
    return parts


def add_visible_inventory_guide(payload):
    """Locate frozen source references for the global check without matching them."""
    payload = deepcopy(payload)
    questions = [item for item in payload["questions"] if item["name"] == "scene_inventory_closure"]
    if not questions:
        return payload
    content = payload["input"][0]["content"]
    context = json.loads(content[0]["text"])
    labels = {item["entity_id"]: label for label, item in context.get("identity_display_legend", {}).items()}
    boxes = {}
    for part in content:
        if part["type"] != "input_text":
            continue
        try:
            metadata = json.loads(part["text"])
        except ValueError:
            continue
        if metadata.get("image_role") == "initial_reference_annotations":
            for item in metadata["items"]:
                boxes.setdefault(item["id"], []).append({"camera_id": metadata["camera"], "box": item["box"]})
    references = []
    completed = {row["entity_id"]: row for row in context.get("completed_source_correspondences", [])}
    state = context.get('inventory_state', {})
    retired = set(state.get('retired_entity_ids', ()))
    for identifier in context["source_inventory"]["source_object_ids"]:
        if identifier in retired:
            references.append({'entity_id': identifier, 'reference_kind': 'retired_input',
                'output_entity_ids': [record['output_entity_id'] for record in state.get('transformations', {}).values()
                                      if record.get('input_entity_id') == identifier and record.get('output_entity_id')],
                'reference_rule': 'Historical consumed input; its independent verified outputs are not extra source candidates.'})
            continue
        if identifier in completed:
            references.append({**completed[identifier], "reference_kind": "same_capture_completed_correspondence",
                               "reference_rule": "Use the supplied CURRENT locations; historical source-box position "
                                   "is not current. Only status confirmed is accepted correspondence."})
            continue
        entity = context["fixed_entities"].get(identifier, {})
        references.append({"entity_id": identifier, "reference_display_label": labels.get(identifier),
                           "display_id": entity.get("display_id"), "description": entity.get("description", ""),
                           "reference_boxes": boxes.get(identifier, [])})
    guide = {"scope": "initial_reference_inventory_not_a_current_count_or_matching_verdict",
             "listed_reference_count": len(references), "references": references}
    questions[0]["instructions"] = questions[0]["instructions"].split(
        " Compare the listed references below", 1)[0] + (
        " Compare the listed references below with all three full CURRENT RGB images, including outside any zoom crop. "
        "When a listed object has a separate same-capture completed correspondence, use its CURRENT location "
        "instead of its historical source position; unknown correspondence does not establish a match. "
        "Count each physical object once across cameras: multiple views or visibly connected faces of the same "
        "identifiable object are not additional objects. An unresolved SAM candidate in an auxiliary view "
        "does not by itself establish another physical object; inspect the full RGB and the clearest available view. "
        "Do not merge subjects merely because they share a color or are nearby. If a possibly distinct visible task "
        "subject remains unresolved, answer unknown. The listed reference count does not establish the current number "
        "or prove there are no extra subjects; cropped reference views cannot establish absence outside their bounds."
        " Visible-source reference guide: " + json.dumps(guide, ensure_ascii=False))
    return payload


def improve_identity_presentation(payload):
    """Keep request questions/choices and full RGB, replacing only locating aids."""
    payload = deepcopy(payload)
    # Generated zooms are replaced, not accumulated when a request is presented again.
    content, skip_image = [], False
    for part in payload["input"][0]["content"]:
        if skip_image:
            skip_image = False
            if part["type"] == "input_image":
                continue
        if part["type"] == "input_text":
            try:
                metadata = json.loads(part["text"])
            except ValueError:
                metadata = {}
            if metadata.get("image_role") in {"same_capture_source_group_zoom", "current_candidate_detail_zoom"}:
                skip_image = True
                continue
        content.append(part)
    payload["input"][0]["content"] = content
    context = json.loads(content[0]["text"])
    entities = context["fixed_entities"]
    labels = {identifier: context.get("fixed_reference_labels", {}).get(identifier, f"Reference {index + 1}")
              for index, identifier in enumerate(sorted(entities))}
    legend = {label: {"entity_id": identifier, "image_label": _image_label(label),
                      "description": entities[identifier].get("description", ""),
                      "display_id": entities[identifier].get("display_id")}
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
                if role:
                    key = (role, metadata["camera"])
                    annotations[key] = metadata
                    records.append((index, key, metadata))
        elif part["type"] == "input_image" and pending is not None:
            raw[pending] = part
            pending = None
    for question in payload["questions"]:
        # Deepcopy of the whole payload preserves aliases between questions.
        # Own each option here too, including historical archived requests.
        if 'choices' in question:
            question["choices"] = deepcopy(question["choices"])
        if question["name"].startswith("identity_"):
            number = question["name"].removeprefix("identity_")
            labels["candidate-" + number] = "Candidate " + number
        for option in question.get('choices', []):
            if option["value"] in entities:
                description = re.sub(r"^(?:(?:R|Reference )\d+ · )+", "", option.get("description", ""))
                option["description"] = labels[option["value"]] + " · " + description
    context["identity_display_legend"] = legend
    for group in context.get('current_measured_view_groups', {}).get('groups', []):
        for candidate in group['members']:
            labels[candidate] = 'Candidate ' + candidate.removeprefix('candidate-')
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
        "Reference labels locate permanent references and Candidate labels locate unassociated candidates. "
        "Current anchor labels locate independently tracked containers in this CURRENT capture; "
        "their omitted identity questions do not remove them from the scene. "
        "Derived annotation images place short R/C labels next to their own boxes: R5 means Reference 5, "
        "C1 means Candidate 1; image_label_legend maps each to the full ID. R5* marks a CURRENT tracked anchor. "
        "These overlays are locating aids, not raw images; unchanged full raw RGB images are supplied separately. "
        "Neutral frame colors encode no identity. Labels never replace canonical choice values.")
    content[0]["text"] = json.dumps(context, ensure_ascii=False)
    for index, key, metadata in records:
        if key not in raw:
            continue
        rgb, image_hash = _decode_image(raw[key])
        items = [{**item, "display_label": labels.get(item["id"], item["id"])}
                 for item in metadata["items"]]
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
    content.extend(_source_zoom(context, raw, annotations, labels))
    # Full views remain primary. Give every camera the same bounded detail
    # allowance so earlier cameras cannot consume a later wrist view's slots.
    content.extend(_current_candidate_zooms(context, raw, annotations, labels, 2))
    # Overlap tables and their limitations remain in shared context. Repeating
    # the table and the same warning in every question obscures the identity task.
    for question in payload["questions"]:
        if not question["name"].startswith("identity_"):
            continue
        candidate = "candidate-" + question["name"].removeprefix("identity_")
        question["instructions"] = question["instructions"].split(" Its display label is ", 1)[0]
        question["instructions"] += f" Its display label is {labels[candidate]}; return a canonical permanent ID."
        question['instructions'] += grouped_view_instruction(context, candidate)
    return add_visible_inventory_guide(payload)

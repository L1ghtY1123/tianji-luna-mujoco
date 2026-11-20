"""Present scoped RGBD and measured subject boxes without assigning identities."""
import base64
import hashlib
import json
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERA_ROLES = {'1': 'right wrist, arm B', '2': 'left wrist, arm A', '3': 'head'}


def subject_description(label, camera, box):
    """Describe the exact image annotation without asserting physical identity."""
    coordinates = ', '.join(f'{value:.3f}' for value in box)
    return (f'Label {label} in camera {camera} ({CAMERA_ROLES[camera]}). '
            f'Image box [left, top, right, bottom] = [{coordinates}] in normalized image coordinates.')


def image_part(array):
    stream = BytesIO()
    Image.fromarray(np.asarray(array)).save(stream, format='PNG')
    return {'type': 'input_image', 'image_url': 'data:image/png;base64,'
            + base64.b64encode(stream.getvalue()).decode('ascii'), 'detail': 'high'}


def draw_subject_annotations(rgb, masks, boxes):
    """Anchor each label on its own segmented surface, including hollow containers."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    palette = ((255, 220, 0), (0, 240, 240), (255, 100, 60), (220, 110, 255), (180, 255, 80))
    colors = {label: palette[int(hashlib.sha256(str(label).encode()).hexdigest()[:8], 16) % len(palette)]
              for label in boxes}
    result = np.asarray(rgb).copy()
    for label, value in masks.items():
        mask = np.asarray(value, dtype=bool)
        if np.any(mask):
            result[mask & ~binary_erosion(mask)] = colors[label]
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    width, height = image.size
    for label, box in boxes.items():
        x1, y1, x2, y2 = box
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError('invalid_visual_subject_box')
        color = colors[label]
        draw.rectangle((int(x1 * width), int(y1 * height), min(width - 1, int(x2 * width)),
                        min(height - 1, int(y2 * height))), outline=color, width=2)
        bounds = draw.textbbox((0, 0), str(label), font=font)
        label_width, label_height = bounds[2] - bounds[0] + 6, bounds[3] - bounds[1] + 6
        mask = masks.get(label)
        leader = None
        if mask is not None and np.any(mask):
            # A box may enclose contents. Place its label in the widest patch
            # of the actual subject, never at the box centre or on a hole.
            distances = distance_transform_edt(mask)
            y, x = np.unravel_index(np.argmax(distances), np.asarray(mask).shape)
            left, top = int(x) - label_width // 2, int(y) - label_height // 2
            if (max(label_width, label_height) / 2 > distances[y, x]
                    or label_width > (x2 - x1) * width * .6):
                # Long archive IDs must not hide a small object's appearance.
                leader = (int(x), int(y))
                left, top = int(x1 * width), int(y1 * height) - label_height
        else:
            left, top = int(x1 * width), int(y1 * height) - label_height
        left = max(0, min(left, width - label_width))
        top = max(0, min(top, height - label_height))
        if leader is not None:
            draw.line((leader, (left + label_width // 2, top + label_height)), fill=color, width=1)
        draw.rectangle((left, top, left + label_width, top + label_height), fill=(0, 0, 0), outline=color, width=1)
        draw.text((left + 3 - bounds[0], top + 3 - bounds[1]), str(label), font=font, fill=color)
    return np.asarray(image)


def image_content(array, **metadata):
    height, width = np.asarray(array).shape[:2]
    metadata['image_size_px'] = [width, height]
    if 'role' in metadata:
        role = metadata['role']
        prefix = 'current_' if role == 'current' else 'reference_'
        metadata['image_role'] = prefix + metadata['modality']
        metadata['frame_role'] = {
            'current': 'CURRENT', 'same_capture_reference': 'SAME_CAPTURE',
            'before_release': 'BEFORE', 'before_unpacking': 'BEFORE',
        }.get(role, 'DATED_REFERENCE')
    elif metadata.get('modality') == 'catalog_reference':
        metadata.update(image_role='catalog_reference', frame_role='STATIC_REFERENCE')
    return [{'type': 'input_text', 'text': json.dumps(metadata, ensure_ascii=False)}, image_part(array)]


def subject_detail_content(capture, camera, label, mask, *, role='current'):
    """Matched RGB/mask crops keep small subjects legible without resizing pixels."""
    from .visual_identity import mask_box

    view = capture['views'][camera]
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    source_box = mask_box(mask)
    left, top, right, bottom = source_box
    x1, y1 = max(0, int(left * width) - 8), max(0, int(top * height) - 8)
    x2, y2 = min(width, int(np.ceil(right * width)) + 8), min(height, int(np.ceil(bottom * height)) + 8)
    detail_mask = mask[y1:y2, x1:x2]
    boxes = {label: mask_box(detail_mask)}
    # A text badge can cover an entire small/thin subject. The adjacent metadata
    # names this single subject and its mask supplies its exact pixels.
    rgb = view['rgb'][y1:y2, x1:x2]
    common = {'role': role, 'camera_id': camera, 'camera_role': CAMERA_ROLES[camera],
        'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'time_s': capture.get('time'), 'subject': label, 'subject_boxes': boxes,
        'subject_mask_labels': [label], 'crop_pixels': [x1, y1, x2, y2],
        'source_image_size_px': [width, height], 'source_subject_box': source_box,
        'box_coordinates': 'normalized [left, top, right, bottom] in THIS cropped image',
        'input_scope': view.get('scope_source', 'full_camera_frame'),
        'annotation': 'Unscaled, unpainted RGB detail. Subject label is in metadata; the binary mask uses identical coordinates.'}
    return [*image_content(rgb, modality='rgb', excluded_pixels='unobserved', **common),
        *image_content(detail_mask.astype(np.uint8) * 255, modality='mask',
            white='subject', black='other_pixels', **common)]


def gripper_detail_content(capture, arm):
    """Camera-calibrated close view of the hand; no object identity is assumed."""
    camera = {'A': '2', 'B': '1'}[arm]
    view = capture.get('views', {}).get(camera, {})
    tcp = capture.get('arms', {}).get(arm, {}).get('tcp')
    if tcp is None or any(key not in view for key in ('rgb', 'T_world_camera', 'intrinsics')):
        return []
    point = np.linalg.inv(np.asarray(view['T_world_camera'])) @ np.r_[np.asarray(tcp)[:3, 3], 1.]
    if not np.isfinite(point).all() or point[2] <= 0:
        return []
    k = view['intrinsics']
    u, v = k['fx'] * point[0] / point[2] + k['cx'], k['fy'] * point[1] / point[2] + k['cy']
    # 11 cm spans the finger contact area without resizing away thin material.
    dx, dy = k['fx'] * .055 / point[2], k['fy'] * .055 / point[2]
    height, width = view['rgb'].shape[:2]
    x1, x2 = max(0, int(u - dx)), min(width, int(u + dx))
    y1, y2 = max(0, int(v - dy)), min(height, int(v + dy))
    if x2 <= x1 or y2 <= y1:
        return []
    return image_content(view['rgb'][y1:y2, x1:x2], modality='rgb', role='current', camera_id=camera,
        arm=arm, observation_id=capture['observation_id'],
        generation=capture.get('generation'), time_s=capture.get('time'),
        crop_pixels=[x1, y1, x2, y2], source_image_size_px=[width, height],
        annotation='Unscaled close view of the finger contact area around the calibrated front TCP.')


def capture_content(capture, *, modalities=('rgb',), masks=None, boxes=None, role='current', camera_order=(),
                    subject_crops=False, regions=None):
    """Depth has a fixed display scale because Decisions accepts images, not float arrays.

    Encoding is explicit and identical across captures. No per-frame normalization,
    geometric statistics, filtering, registration or identity assignment occurs here.
    """
    if not set(modalities) <= {'rgb', 'mask', 'depth'}:
        raise ValueError('unknown_visual_modality')
    views = capture.get('views', {})
    cameras = sorted(views, key=lambda key: (key not in camera_order,
        camera_order.index(key) if key in camera_order else key))
    content = []
    for camera in cameras:
        view = views[camera]
        common = {'role': role, 'camera_id': camera, 'observation_id': capture.get('observation_id'),
                  'generation': capture.get('generation'), 'time_s': capture.get('time'),
                  'camera_role': CAMERA_ROLES.get(camera)}
        subject_boxes = dict((boxes or {}).get(camera, {}))
        for subject, mask in (masks or {}).get(camera, {}).items():
            if np.any(mask):
                from .visual_identity import mask_box
                subject_boxes[subject] = mask_box(mask)
        common.update(input_scope=view.get('scope_source', 'full_camera_frame'),
                      excluded_pixels='unobserved', subject_boxes=subject_boxes,
                      box_coordinates='normalized [left, top, right, bottom], origin at image top-left, x right, y down',
                      image_direction='Image left/right is local to this camera view. Robot left/right is named in camera_role.',
                      subject_mask_labels=[label for label, mask in (masks or {}).get(camera, {}).items() if np.any(mask)])
        if 'rgb' in modalities:
            rgb = draw_subject_annotations(view['rgb'], (masks or {}).get(camera, {}), subject_boxes)
            projected = (regions or {}).get(camera, [])
            if projected:
                from .observation_labels import raster_visual_annotations

                rgb = raster_visual_annotations(rgb, [], regions=projected, label_layout='near_box')
            region_metadata = {'projected_regions': projected} if projected else {}
            content.extend(image_content(rgb, modality='rgb', **common, **region_metadata))
            if subject_crops:
                height, width = view['rgb'].shape[:2]
                for subject, box in subject_boxes.items():
                    left, top, right, bottom = box
                    x1, y1 = max(0, int(left * width) - 8), max(0, int(top * height) - 8)
                    x2, y2 = min(width, int(np.ceil(right * width)) + 8), min(height, int(np.ceil(bottom * height)) + 8)
                    crop_mask = (masks or {}).get(camera, {}).get(subject)
                    crop = view['rgb'][y1:y2, x1:x2]
                    crop_masks = {subject: crop_mask[y1:y2, x1:x2]} if crop_mask is not None else {}
                    crop_boxes = {subject: [(left * width - x1) / (x2 - x1),
                        (top * height - y1) / (y2 - y1), (right * width - x1) / (x2 - x1),
                        (bottom * height - y1) / (y2 - y1)]}
                    crop = draw_subject_annotations(crop, crop_masks, crop_boxes)
                    content.extend(image_content(crop, modality='rgb', subject=subject,
                        crop_pixels=[x1, y1, x2, y2], source_image_size_px=[width, height],
                        **{**common, 'subject_boxes': crop_boxes,
                           'subject_mask_labels': [subject] if crop_mask is not None else [],
                           'annotation': 'labelled_subject_boundary'}))
        if 'mask' in modalities:
            for subject, mask in (masks or {}).get(camera, {}).items():
                if not np.any(mask):
                    continue
                content.extend(image_content(np.asarray(mask, dtype=bool).astype(np.uint8) * 255,
                    modality='mask', subject=subject, white='subject', black='other_pixels',
                    **{**common, 'subject_boxes': {subject: subject_boxes[subject]}, 'subject_mask_labels': [subject],
                       'excluded_pixels': 'mask black encodes other pixels'}))
        if 'depth' in modalities and 'depth_m' in view:
            depth = np.asarray(view['depth_m'], float)
            valid = np.isfinite(depth) & (depth > 0) & np.asarray(view.get('valid_depth', True), bool)
            encoded = np.zeros(depth.shape, dtype=np.uint8)
            encoded[valid] = np.clip(np.rint(depth[valid] / 3. * 254) + 1, 1, 255).astype(np.uint8)
            content.extend(image_content(encoded, modality='depth', encoding='fixed_linear_grayscale',
                zero='invalid_depth', depth_m_per_step=3. / 254, depth_m_at_value_1=0.,
                maximum_display_depth_m=3., **common))
    return content


def reference_content(reference, *, modalities=('rgb',), current_capture=None, include_initial=False,
                      subject_crops=False, labelled_only=False):
    """Show one labelled reference frame per capture/camera/pixel scope."""
    from .visual_identity import mask_box

    frames = {}
    initial = reference if include_initial and (reference or {}).get('views') else None
    if initial is not None:
        for camera, view in initial['views'].items():
            stamp = hashlib.sha256(np.asarray(view['rgb']).tobytes()).hexdigest()
            key = (initial.get('observation_id'), initial.get('generation'), initial.get('time'), camera, stamp)
            frames[key] = {'capture': {**initial, 'views': {camera: view}},
                'camera': camera, 'boxes': {}, 'masks': {}, 'instances': []}
    for identifier, stored in (reference or {}).get('identity_references', {}).items():
        primary_camera = stored['camera_id']
        views = stored.get('views') or {primary_camera: {'rgb': stored['rgb']}}
        for camera, view in views.items():
            same_initial = (initial is not None and camera in initial['views']
                and stored.get('observation_id') == initial.get('observation_id')
                and stored.get('generation') == initial.get('generation'))
            frame_capture = stored
            if same_initial:
                frame_capture = initial
                view = {**view, **initial['views'][camera]}
            mask = view.get('mask')
            box = mask_box(mask) if mask is not None and np.any(mask) else view.get('bbox',
                stored.get('bbox') if camera == primary_camera else None)
            if labelled_only and box is None:
                continue
            stamp = hashlib.sha256(np.asarray(view['rgb']).tobytes()).hexdigest()
            key = (frame_capture.get('observation_id'), frame_capture.get('generation'), frame_capture.get('time'), camera, stamp)
            if key not in frames:
                frames[key] = {'capture': {**frame_capture, 'views': {camera: view}},
                               'camera': camera, 'boxes': {}, 'masks': {}, 'instances': []}
            frame = frames[key]
            if box is not None:
                frame['boxes'][identifier] = box
                frame['instances'].append(identifier)
            if mask is not None:
                frame['masks'][identifier] = mask
    content = []
    for frame in frames.values():
        camera = frame['camera']
        stored = frame['capture']
        same_capture = (current_capture is not None and stored.get('observation_id') is not None
            and stored.get('observation_id') == current_capture.get('observation_id')
            and stored.get('generation') == current_capture.get('generation'))
        role = 'same_capture_reference' if same_capture else 'dated_reference'
        content.append({'type': 'input_text', 'text': json.dumps({
            'reference_instance_ids': frame['instances'], 'reference_camera_id': camera,
            'scope': 'Each labelled box identifies a reference instance.', 'temporal_relation': role})})
        content.extend(capture_content(frame['capture'], modalities=modalities,
            boxes={camera: frame['boxes']}, masks={camera: frame['masks']}, role=role,
            subject_crops=subject_crops))
    if not frames:
        content.extend(capture_content(reference or {}, modalities=('rgb',), role='dated_reference'))
    return content

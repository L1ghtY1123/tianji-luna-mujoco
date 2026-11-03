"""Capture-bound identity overlays; task state belongs to its own event timeline."""
import base64
from io import BytesIO
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {'red': '红色', 'green': '绿色', 'blue': '蓝色', 'black': '黑色', 'white': '白色',
          'yellow': '黄色', 'gray': '灰色', 'orange': '橙色', 'pink': '粉色'}
KINDS = {'cube': '方块', 'box': '长方体', 'sphere': '球体', 'cylinder': '圆柱',
         'bin': '盒子', 'marker': '放置区域', 'object': '物体', 'table': '桌面'}
IDENTITY_NAMES = {'openai_decisions': 'Decisions 已关联 ID',
                  'measured_container_correspondence': 'RGBD 对应沿用 ID',
                  'mixed_decisions_measured_container': 'Decisions 与 RGBD 对应关联 ID'}


def responses_reference_labels(capture, reference):
    """Draw the original model boxes on their input RGB, before any SAM result."""
    items = []
    for identifier, entity in reference.get('entities', {}).items():
        camera, box = entity.get('reference_camera_id'), entity.get('bbox')
        if camera not in capture['views'] or box is None:
            continue
        left, top, right, bottom = box
        name = COLORS.get(entity.get('color'), '') + KINDS.get(entity.get('kind'), '物体')
        items.append({'id': identifier, 'camera_id': camera, 'name': name,
            'role': 'proposal', 'status': 'unverified', 'box_source': 'responses_reference_proposal',
            'identity_source': 'responses_proposal_not_verified', 'geometry_status': 'not_measured',
            'label': f'{name} · Responses 提议\nID {identifier}',
            'x': left, 'y': top, 'width': right - left, 'height': bottom - top})
    return {'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
        'analysis_status': 'responses_proposal', 'overlay_scope': 'initial_responses_reference',
        'image_source': 'captured_rgb', 'overlay_source': 'responses_reference_proposal',
        'used_as_model_input': False, 'items': items}


def make_object_labels(observation, records, membership, *, holding=None, destination_id=None, placement_regions=()):
    # Keep the compatibility argument, but never put later holding feedback on an old capture.
    labels = {'observation_id': observation['observation_id'], 'generation': observation.get('generation'),
              'analysis_status': 'analyzed', 'overlay_scope': 'capture_identity_geometry',
              'image_source': 'captured_rgb', 'overlay_source': 'program_identity_summary',
              'used_as_model_input': False, 'items': []}
    objects = list(observation['objects'])
    measured_ids = {obj['id'] for obj in objects}
    if observation.get('visual_discovery'):
        objects.extend(entity for identifier, entity in observation.get('scene_entities', {}).items()
                       if identifier not in measured_ids and entity.get('bbox'))
    for obj in objects:
        identifier = obj['id']
        record = records.get(identifier, {})
        member = membership.get(identifier, 'excluded' if identifier not in records else 'unknown')
        role = 'target' if member == 'required' else 'obstacle' if member == 'excluded' else 'unknown'
        if obj.get('kind') in {'bin', 'marker'}:
            role = 'container' if obj['kind'] == 'bin' else 'region'
        if identifier == destination_id:
            role = 'destination'
        identity_source = obj.get('identity_source', 'unrecorded')
        status = 'associated' if identity_source in IDENTITY_NAMES else 'unverified'
        number = record.get('display_id', obj.get('display_id'))
        color = obj.get('color')
        name_source = 'measured_attributes'
        if color not in COLORS:
            color = observation.get('scene_entities', {}).get(identifier, {}).get('color')
            name_source = 'inventory_description' if color in COLORS else 'generic_kind'
        name = COLORS.get(color, '') + KINDS.get(obj.get('kind'), '物体')
        title = (f'#{number} ' if number is not None else '') + name
        identity_text = IDENTITY_NAMES.get(identity_source, '身份依据未记录')
        text = f'{title} · {identity_text}\nID {identifier}'
        provenance = {'name_source': name_source, 'identity_source': identity_source,
                      'measured_color': obj.get('color'),
                      'point_count': obj.get('coverage', {}).get('point_count'),
                      'collision_status': 'see_build_scene_event'}
        for camera, view in observation.get('views', {}).items():
            mask = view.get('instance_masks', {}).get(identifier)
            if mask is None or not np.any(mask):
                if (observation.get('visual_discovery') and obj.get('bbox')
                        and camera == obj.get('reference_camera_id', '3')):
                    left, top, right, bottom = obj['bbox']
                    cached_table = identifier in observation.get('demo_tables', {}).get('tables', {})
                    labels['items'].append({'id': identifier, 'display_id': number, 'camera_id': camera,
                        'name': name, 'role': role, 'status': 'unverified', **provenance,
                        'box_source': 'responses_reference_proposal',
                        'geometry_status': 'cached_cuboid' if cached_table else 'missing',
                        'label': text + (' · 已缓存桌面几何' if cached_table else ' · 缺少当前几何'),
                        'x': left, 'y': top,
                        'width': right - left, 'height': bottom - top})
                continue
            rows, columns = np.nonzero(mask)
            height, width = mask.shape
            labels['items'].append({'id': identifier, 'display_id': number, 'camera_id': camera,
                'name': name, 'role': role, 'status': status, 'label': text, **provenance,
                'box_source': 'associated_instance_mask',
                'geometry_status': 'registered_depth' if obj.get('coverage', {}).get('point_count') else 'mask_only',
                'x': float(columns.min() / width), 'y': float(rows.min() / height),
                'width': float((columns.max() - columns.min() + 1) / width),
                'height': float((rows.max() - rows.min() + 1) / height)})
    labels['regions'] = project_placement_regions(observation, placement_regions)
    return labels


def project_placement_regions(observation, regions, *, require_visible_support=True):
    """Project frozen measured layouts; never use this display data for motion."""
    result = []
    for region in regions:
        layout = region.get('layout', {})
        support = layout.get('support_entity_id')
        for camera_id, view in observation.get('views', {}).items():
            mask = view.get('instance_masks', {}).get(support)
            if (require_visible_support and (mask is None or not np.any(mask))
                    or 'T_world_camera' not in view or 'intrinsics' not in view):
                continue
            height, width = mask.shape if mask is not None else view['rgb'].shape[:2]
            transform, intrinsics = np.asarray(view['T_world_camera']), view['intrinsics']
            for cell in layout.get('cells', []):
                xy = np.asarray([cell['center_world_xy_m'], *cell['polygon_world_xy_m']])
                world = np.column_stack((xy, np.full(len(xy), layout['floor_z_m'])))
                camera = (world - transform[:3, 3]) @ transform[:3, :3]
                if np.any(camera[:, 2] <= 0) or not np.isfinite(camera).all():
                    continue
                pixels = camera[:, :2] / camera[:, 2, None] * [intrinsics['fx'], intrinsics['fy']]
                normalized = (pixels + [intrinsics['cx'], intrinsics['cy']]) / [width, height]
                result.append({'region_id': region['region_id'], 'slot_id': cell['slot_id'],
                    'camera_id': camera_id, 'support_entity_id': support,
                    'source': layout.get('source'), 'source_observation_id': layout.get('observation_id'),
                    'measurement_time': layout.get('measurement_time'),
                    'layout_revision': region.get('layout_revision'),
                    'center': normalized[0].tolist(), 'polygon': normalized[1:].tolist()})
    return result


def group_candidate_boxes(items):
    """One outline for near-identical proposals; retain every candidate alias."""
    grouped = []
    for item in items:
        match = None
        if str(item.get('id', '')).startswith('candidate-') and 'box' in item:
            box = np.asarray(item['box'], float)
            for other in grouped:
                if not str(other.get('id', '')).startswith('candidate-') or 'box' not in other:
                    continue
                prior = np.asarray(other['box'], float)
                intersection = np.prod(np.maximum(np.minimum(box[2:], prior[2:]) - np.maximum(box[:2], prior[:2]), 0))
                union = np.prod(box[2:] - box[:2]) + np.prod(prior[2:] - prior[:2]) - intersection
                if union > 0 and intersection / union >= .94:
                    match = other
                    break
        if match is None:
            grouped.append(dict(item))
        else:
            match['display_label'] = (match.get('display_label', match['id']) + ' / '
                                      + item.get('display_label', item['id']))
            if 'image_label' in match and 'image_label' in item:
                match['image_label'] += ' / ' + item['image_label']
    return grouped


def _near_box_annotations(rgb, items, regions, table_polygon, frame_color):
    """Identity-only derived overlay: compact badges adjacent to their own boxes."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).copy()
    width, height = image.size
    drawing = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, min(18, round(width / 36))))
    color = frame_color if frame_color is not None else (235, 235, 235)
    boxes, labels = [], []
    for item in items:
        if 'box' not in item:
            raise ValueError('near_box_annotations_require_boxes')
        box = tuple(np.asarray(item['box']) * [width, height, width, height])
        boxes.append(box)
        # Preserve every proposal outline, including near-identical candidates.
        drawing.rectangle(box, outline='black', width=4)
        drawing.rectangle(box, outline=color, width=2)
    for item in group_candidate_boxes(items):
        box = tuple(np.asarray(item['box']) * [width, height, width, height])
        labels.append((item.get('image_label', item.get('display_label', item['id'])), box))
    for region in regions:
        polygon = [(x * width, y * height) for x, y in region['polygon']]
        drawing.line(polygon + polygon[:1], fill='#69e2ff', width=2)
        points = np.asarray(polygon)
        box = (*points.min(0), *points.max(0))
        boxes.append(box)
        labels.append((f"{region['region_id']} slot {region['slot_id']}", box))
    if table_polygon:
        polygon = [(x * width, y * height) for x, y in table_polygon]
        drawing.line(polygon + polygon[:1], fill=color, width=2)
        points = np.asarray(polygon)
        box = (*points.min(0), *points.max(0))
        boxes.append(box)
        labels.append(('TABLE', box))

    def overlap(first, second):
        return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
            0, min(first[3], second[3]) - max(first[1], second[1]))

    placed = []
    for label, box in labels:
        left, top, right, bottom = box
        text_box = drawing.textbbox((0, 0), label, font=font)
        label_width = min(width, text_box[2] - text_box[0] + 8)
        label_height = min(height, text_box[3] - text_box[1] + 8)
        positions = [
            (left, top - label_height - 3), (right - label_width, top - label_height - 3),
            (left, bottom + 3), (right - label_width, bottom + 3),
            (left - label_width - 3, top), (right + 3, top),
            (left + 2, top + 2), (right - label_width - 2, top + 2),
            (left + 2, bottom - label_height - 2), (right - label_width - 2, bottom - label_height - 2)]
        candidates = [(x, y, x + label_width, y + label_height) for x, y in positions
                      if x >= 0 and y >= 0 and x + label_width <= width and y + label_height <= height]
        if not candidates:
            x = max(0, min(left, width - label_width))
            y = max(0, min(top, height - label_height))
            candidates = [(x, y, x + label_width, y + label_height)]
        # Avoid other labels first, then subjects; ties favor the adjacent upper edge.
        badge = min(candidates, key=lambda candidate: (
            sum(overlap(candidate, previous) for previous in placed),
            sum(overlap(candidate, subject) for subject in boxes), overlap(candidate, box)))
        placed.append(badge)
        drawing.rectangle(badge, fill='#101010', outline='white', width=1)
        drawing.text((badge[0] + 4 - text_box[0], badge[1] + 4 - text_box[1]),
                     label, font=font, fill='white')
    return np.asarray(image)


def raster_visual_annotations(rgb, items, *, regions=(), table_polygon=(), frame_color=None,
                              label_layout='margin'):
    """Default margin layout; identity callers may opt into adjacent short badges."""
    if label_layout == 'near_box':
        return _near_box_annotations(rgb, items, regions, table_polygon, frame_color)
    if label_layout != 'margin':
        raise ValueError('invalid_visual_annotation_layout')
    items = group_candidate_boxes(items)
    height, width = rgb.shape[:2]
    font = ImageFont.load_default(size=14)
    labels = [item.get('display_label', item['id']) for item in items]
    labels += [f"{region['region_id']} slot {region['slot_id']}" for region in regions]
    if table_polygon:
        labels.append('TABLE SCOPE (measured)')
    margin = max(220, int(max((font.getlength(label) for label in labels), default=0)) + 24)
    row_height = 26
    image = Image.new('RGB', (width + margin, max(height, (len(labels) + 1) * row_height)), '#10212d')
    image.paste(Image.fromarray(np.asarray(rgb, dtype=np.uint8)), (0, 0))
    pixels = np.asarray(image).copy()
    colors = [(255, 90, 220), (255, 210, 70), (100, 220, 255), (130, 255, 160)]
    anchors = []
    for index, item in enumerate(items):
        if 'box' in item:
            continue
        mask = np.asarray(item['mask'], bool)
        rows, columns = np.nonzero(mask)
        if not len(rows):
            continue
        interior = mask.copy()
        interior[1:, :] &= mask[:-1, :]
        interior[:-1, :] &= mask[1:, :]
        interior[:, 1:] &= mask[:, :-1]
        interior[:, :-1] &= mask[:, 1:]
        color = frame_color if frame_color is not None else colors[index % len(colors)]
        pixels[:height, :width][mask & ~interior] = color
        anchors.append((item.get('display_label', item['id']), color,
                        (int(columns.max()), int(np.median(rows)))))
    image = Image.fromarray(pixels)
    drawing = ImageDraw.Draw(image)
    for index, item in enumerate(items):
        if 'box' not in item:
            continue
        left, top, right, bottom = np.asarray(item['box']) * [width, height, width, height]
        color = frame_color if frame_color is not None else colors[index % len(colors)]
        drawing.rectangle((left, top, right, bottom), outline=color, width=2)
        anchors.append((item.get('display_label', item['id']), color, (right, (top + bottom) / 2)))
    for region in regions:
        polygon = [(x * width, y * height) for x, y in region['polygon']]
        drawing.line(polygon + polygon[:1], fill='#69e2ff', width=2)
        center = tuple(np.asarray(region['center']) * [width, height])
        anchors.append((f"{region['region_id']} slot {region['slot_id']}", '#69e2ff', center))
    if table_polygon:
        polygon = [(x * width, y * height) for x, y in table_polygon]
        drawing.line(polygon + polygon[:1], fill='#ffffff', width=2)
        anchors.append(('TABLE SCOPE (measured)', '#ffffff', max(polygon)))
    # Draw no label backgrounds over objects. Connectors terminate at the margin.
    for index, (label, color, anchor) in enumerate(anchors):
        y = 8 + row_height * index
        drawing.line([anchor, (width + 6, y + 7)], fill=color, width=1)
        drawing.text((width + 10, y), label, fill=color, font=font)
    return np.asarray(image)


def annotated_rgb_svg(rgb, labels, camera_id):
    """Embed original RGB with escaped labels; never alter perception input pixels."""
    height, width = rgb.shape[:2]
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    root = Element('svg', xmlns='http://www.w3.org/2000/svg', width=str(width), height=str(height + 28),
                   viewBox=f'0 0 {width} {height + 28}')
    SubElement(root, 'rect', width=str(width), height='28', fill='#10212d')
    state = '已分析' if labels.get('analysis_status') == 'analyzed' else '未分析：尚无本帧掩码与 ID 关联'
    SubElement(root, 'text', x='8', y='19', fill='white', **{'font-size': '12'}).text = (
        f"相机 {camera_id} · {labels['observation_id']} · {state}")
    SubElement(root, 'image', x='0', y='28', width=str(width), height=str(height),
               href='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii'))
    for region in labels.get('regions', []):
        if region['camera_id'] != camera_id:
            continue
        points = ' '.join(f'{x * width},{y * height + 28}' for x, y in region['polygon'])
        SubElement(root, 'polygon', points=points, fill='none', stroke='#69e2ff', **{'stroke-width': '2'})
        x, y = region['center']
        SubElement(root, 'text', x=str(x * width), y=str(y * height + 28), fill='#69e2ff',
                   **{'font-size': '14'}).text = f"{region['region_id']} 区{region['slot_id']}"
        SubElement(root, 'desc').text = (f"测量来源 {region['source']} · 帧 {region['source_observation_id']}"
                                        f" · t={region.get('measurement_time')} · {region['layout_revision']}")
    placed = []
    for item in labels['items']:
        if item['camera_id'] != camera_id:
            continue
        color = '#7bffa5' if item['status'] == 'completed' else '#ffc85c' if item['role'] == 'target' else '#91ccff'
        x, y = item['x'] * width, item['y'] * height + 28
        box_width, box_height = item['width'] * width, item['height'] * height
        SubElement(root, 'rect', x=str(x), y=str(y), width=str(box_width), height=str(box_height),
                   fill='none', stroke=color, **{'stroke-width': '2'})
        label_width = min(width, max(215, len(item['label'].split('\n')[0]) * 13))
        left, top = max(0, min(x, width - label_width)), max(28, y - 40)
        while any(abs(top - prior_y) < 40 and left < prior_x + prior_width and left + label_width > prior_x
                  for prior_x, prior_y, prior_width in placed) and top + 80 <= height + 28:
            top += 40
        placed.append((left, top, label_width))
        SubElement(root, 'line', x1=str(x), y1=str(y), x2=str(left), y2=str(top + 38), stroke=color)
        SubElement(root, 'rect', x=str(left), y=str(top), width=str(label_width), height='38',
                   fill='#10212d', **{'fill-opacity': '.88'})
        for line, text in enumerate(item['label'].split('\n')):
            SubElement(root, 'text', x=str(left + 4), y=str(top + 15 + 16 * line), fill=color,
                       **{'font-size': '12', 'font-family': 'sans-serif'}).text = text
    return tostring(root, encoding='unicode')

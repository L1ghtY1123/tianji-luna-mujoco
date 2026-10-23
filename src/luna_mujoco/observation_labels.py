"""Display-only labels from current measured masks and the separate task ledger."""
import base64
from io import BytesIO
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {'red': '红色', 'green': '绿色', 'blue': '蓝色', 'black': '黑色', 'white': '白色',
          'yellow': '黄色', 'gray': '灰色', 'orange': '橙色', 'pink': '粉色'}
KINDS = {'cube': '方块', 'box': '长方体', 'sphere': '球体', 'cylinder': '圆柱',
         'bin': '盒子', 'marker': '放置区域', 'object': '物体', 'table': '桌面'}
STATUSES = {'observed': '未搬运', 'selected': '当前目标', 'held': '夹持中',
            'completed': '已搬运·释放已确认', 'deferred': '未完成·暂缓',
            'pending': '中转完成·待下一段', 'unknown': '待确认任务归属',
            'obstacle': '障碍物·不搬运', 'destination': '目的区域', 'container': '盒子·碰撞障碍',
            'region': '放置区域'}


def make_object_labels(observation, records, membership, *, holding=None, destination_id=None, placement_regions=()):
    labels = {'observation_id': observation['observation_id'], 'generation': observation.get('generation'),
              'analysis_status': 'analyzed', 'items': []}
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
        status = record.get('status', 'observed') if role == 'target' else role
        if holding and holding.get('instance_id') == identifier and holding.get('verified'):
            status = 'held'
        number = record.get('display_id', obj.get('display_id'))
        name = COLORS.get(obj.get('color'), obj.get('color', '')) + KINDS.get(obj.get('kind'), '物体')
        title = (f'#{number} ' if number is not None else '') + name
        text = f'{title} · {STATUSES.get(status, status)}\nID {identifier}'
        for camera, view in observation.get('views', {}).items():
            mask = view.get('instance_masks', {}).get(identifier)
            if mask is None or not np.any(mask):
                if (observation.get('visual_discovery') and obj.get('bbox')
                        and camera == obj.get('reference_camera_id', '3')):
                    left, top, right, bottom = obj['bbox']
                    cached_table = identifier in observation.get('demo_tables', {}).get('tables', {})
                    labels['items'].append({'id': identifier, 'display_id': number, 'camera_id': camera,
                        'name': name, 'role': role, 'status': status,
                        'geometry_status': 'cached_cuboid' if cached_table else 'missing',
                        'label': text + (' · 已缓存桌面几何' if cached_table else ' · 缺少当前几何'),
                        'x': left, 'y': top,
                        'width': right - left, 'height': bottom - top})
                continue
            rows, columns = np.nonzero(mask)
            height, width = mask.shape
            labels['items'].append({'id': identifier, 'display_id': number, 'camera_id': camera,
                'name': name, 'role': role, 'status': status, 'label': text,
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
    return grouped


def raster_visual_annotations(rgb, items, *, regions=(), table_polygon=(), frame_color=None):
    """Keep all text in an added margin; only contours cross measured RGB."""
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

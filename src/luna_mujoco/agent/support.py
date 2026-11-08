"""Ask one current support relation, independently of task membership."""
from ..core.decisions import ChoiceRequest, JudgmentRequest


def support_camera_order(cameras, locations, subject_id, candidates, *, table_regions=()):
    """Show the view that locates both sides of this relation first; retain every view."""
    def relevance(camera):
        identifiers = {item['id'] for item in locations.get('views', {}).get(camera, {}).get('items', [])}
        identifiers.update(region['support_entity_id'] for region in table_regions
                           if region['camera_id'] == camera and region.get('label_anchor') is not None)
        subject_visible = subject_id in identifiers
        supports_visible = len(identifiers.intersection(candidates))
        return (subject_visible and supports_visible > 0, subject_visible, supports_visible)
    return sorted(cameras, key=relevance, reverse=True)


def support_question(subject, candidates, proposed_support=None):
    identifier = subject['id']
    if proposed_support not in candidates:
        return ChoiceRequest(name='current_support', choices=tuple(candidates) + ('none_of_listed', 'unknown'),
            instructions=(
                'Which listed entity currently supports or contains object ID' + identifier + '? '
                'For an object among a bin\'s contents, select that bin even if the object extends above its rim. '
                'Use CURRENT labelled views, current_measured_geometry and dated support_memory. '
                'Hidden contact pixels need not be visible. Choose none_of_listed if a different support '
                'is evident, or unknown if evidence cannot determine the support.'))
    support = candidates[proposed_support]
    relation = 'inside' if support['kind'] == 'bin' else 'directly supported by'
    return JudgmentRequest(name='current_support', instructions=(
        'Is the ' + subject.get('color', '') + ' object ID' + identifier
        + ' currently ' + relation + ' the ' + support.get('color', '') + ' '
        + support['kind'] + ' ID' + proposed_support + '? '
        + ('Inside means the object is among the bin contents; it may extend above the rim. '
           if support['kind'] == 'bin' else 'Directly supported means this is the surface it rests on. ')
        + 'Use the CURRENT labelled views, current_measured_geometry and dated support_memory. '
          'Hidden contact pixels need not be visible. A clearly different support is false; '
          'insufficient evidence is uncertain.'))


def scope_support_evidence(context, reference, request):
    """Project detached evidence onto the subject and the support(s) being queried."""
    supports = context.get('support_candidates', {})
    proposed = context.get('support_hypothesis', {}).get('support_entity_id')
    if not isinstance(request, ChoiceRequest) and proposed in supports:
        supports = {proposed: supports[proposed]}
    identifiers = {context['subject']['id'], *supports}
    aliases = {}
    if not isinstance(request, ChoiceRequest) and proposed in supports:
        aliases = {context['subject']['id']: 'A', proposed: 'B'}
        context['subject']['image_label'] = 'A'
        supports[proposed]['image_label'] = 'B'
    context['support_candidates'] = supports
    geometry = context.get('current_measured_geometry', {})
    for name in ('measured_supports', 'measured_tables'):
        if name in geometry:
            geometry[name] = {key: value for key, value in geometry[name].items() if key in supports}
    current_views = context.get('current_instance_locations', {}).get('views', {})
    other_ids = sorted({item['id'] for view in current_views.values() for item in view['items']} - identifiers)
    if aliases:
        aliases.update({identifier: 'C' + str(index + 1) for index, identifier in enumerate(other_ids)})
    for view in current_views.values():
        for item in view['items']:
            if item['id'] in aliases:
                item['image_label'] = aliases[item['id']]
    if reference is not None:
        reference['entities'] = {key: value for key, value in reference.get('entities', {}).items()
                                 if key in identifiers}
        for identifier, entity in reference['entities'].items():
            if identifier in aliases:
                entity['image_label'] = aliases[identifier]


def support_table_regions(capture, context):
    """Locate dated measured table bounds in the current calibrated views."""
    import numpy as np

    from ..observation_labels import project_placement_regions

    regions = []
    tables = context.get('current_measured_geometry', {}).get('measured_tables', {})
    for identifier, table in tables.items():
        model = table.get('model', {})
        pose, dims = np.asarray(model.get('pose', [])), np.asarray(model.get('dims', []))
        floor = table.get('surface_height_m')
        if (pose.shape != (7,) or dims.shape != (3,) or not np.isfinite(pose).all()
                or not np.isfinite(dims).all() or np.any(dims <= 0)
                or not np.allclose(np.abs(pose[3:]), [1, 0, 0, 0], atol=1e-9)
                or type(floor) not in (int, float) or not np.isfinite(floor)
                or not table.get('source_observation_id')):
            continue
        lower, upper = pose[:2] - dims[:2] / 2, pose[:2] + dims[:2] / 2
        label = context.get('support_candidates', {}).get(identifier, {}).get('image_label', 'ID' + identifier)
        regions.append({'region_id': label + ' table', 'layout': {
            'support_entity_id': identifier, 'center_world_xy_m': pose[:2].tolist(),
            'region_polygon_world_xy_m': [[lower[0], lower[1]], [upper[0], lower[1]],
                                         [upper[0], upper[1]], [lower[0], upper[1]]],
            'floor_z_m': floor, 'source': 'retained_measured_table_projection',
            'observation_id': table['source_observation_id']}})
    projected = project_placement_regions(capture, regions, require_visible_support=False)
    visible = []
    for region in projected:
        polygon = np.asarray(region['polygon'])
        if np.any(polygon.max(axis=0) <= 0) or np.any(polygon.min(axis=0) >= 1):
            continue
        center = np.asarray(region['center'])
        boxes = context.get('current_instance_locations', {}).get('views', {}).get(
            region['camera_id'], {}).get('items', [])
        anchors = [center, *((polygon + center) / 2)]
        region['label_anchor'] = next((point.tolist() for point in anchors
            if np.all((point > .05) & (point < .95))
            and not any(np.all(point >= item['box'][:2]) and np.all(point <= item['box'][2:])
                        for item in boxes)), None)
        visible.append(region)
    return visible

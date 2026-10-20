"""Equal-area radial partitions of a measured convex support polygon."""
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


def identify_placement_regions(observation):
    """Region identity follows the stable support entity, never detection order."""
    for obj in observation['objects']:
        reference = obj.get('placement_reference')
        if reference is None:
            continue
        reference.update(region_id='region:' + obj['id'], support_entity_id=obj['id'],
                         observation_id=observation.get('observation_id'),
                         plane={'normal_world': [0., 0., 1.], 'height_m': reference['floor_z_m']})


def area_and_center(polygon):
    points = np.asarray(polygon, dtype=float)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    twice_area = float(cross.sum())
    if twice_area <= 1e-12:
        raise ValueError('placement_region_has_no_area')
    center = ((points + following) * cross[:, None]).sum(axis=0) / (3 * twice_area)
    return twice_area / 2, center


def _center_symmetric_region(points):
    """Intersect a measured hull with its reflection, retaining an inner region.

    This makes opposite equal-area rays full center lines even when partial
    boundary sampling is asymmetric; no area outside the observed hull is added.
    """
    _, center = area_and_center(points)
    hull = ConvexHull(points)
    result = points.copy()
    for equation in hull.equations:
        normal, offset = -equation[:2], equation[2] + 2 * equation[:2] @ center
        clipped = []
        for first, second in zip(result, np.roll(result, -1, axis=0)):
            a, b = first @ normal + offset, second @ normal + offset
            if a <= 1e-12:
                clipped.append(first)
            if (a <= 1e-12) != (b <= 1e-12):
                clipped.append(first + a / (a - b) * (second - first))
        result = np.asarray(clipped)
    result = result[ConvexHull(result).vertices]
    return result


def partition_region(polygon, target_count):
    if type(target_count) is not int or target_count < 1:
        raise ValueError('placement_requires_positive_target_count')
    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('placement_requires_measured_region_polygon')
    try:
        hull = ConvexHull(points)
    except QhullError as error:
        raise ValueError('placement_region_has_no_area') from error
    points = _center_symmetric_region(points[hull.vertices])
    area, center = area_and_center(points)
    sections = 2 * ((target_count + 1) // 2)
    # Start at the +X ray, then walk the measured hull counterclockwise.
    # A triangle fan makes enclosed area linear along each boundary edge.
    intersections = []
    for first, second in zip(points, np.roll(points, -1, axis=0)):
        dy = second[1] - first[1]
        if abs(dy) > 1e-12:
            fraction = (center[1] - first[1]) / dy
            if 0 <= fraction <= 1:
                intersections.append(first + fraction * (second - first))
    start = max(intersections, key=lambda point: point[0])
    angles = np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), 2 * np.pi)
    vertices = points[np.argsort(angles)]
    vertices = vertices[np.linalg.norm(vertices - start, axis=1) > 1e-10]
    boundary = np.vstack((start, vertices, start))
    left, right = boundary[:-1] - center, boundary[1:] - center
    areas = (left[:, 0] * right[:, 1] - right[:, 0] * left[:, 1]) / 2
    cumulative = np.concatenate(([0.], np.cumsum(areas)))

    def boundary_at(value):
        index = min(int(np.searchsorted(cumulative, value, side='right')) - 1, len(areas) - 1)
        fraction = (value - cumulative[index]) / areas[index]
        return boundary[index] + fraction * (boundary[index + 1] - boundary[index])

    cells = []
    for index in range(sections):
        low, high = cumulative[-1] * index / sections, cumulative[-1] * (index + 1) / sections
        inside = boundary[(cumulative > low + 1e-12) & (cumulative < high - 1e-12)]
        cell = np.vstack((center, boundary_at(low), inside, boundary_at(high)))
        cell_area, cell_center = area_and_center(cell)
        cells.append({'index': index, 'area_m2': cell_area, 'center_world_xy_m': cell_center.tolist(),
                      'polygon_world_xy_m': cell.tolist()})
    return {'target_count': target_count, 'section_count': sections, 'area_m2': area,
            'center_world_xy_m': center.tolist(), 'region_polygon_world_xy_m': points.tolist(),
            'method': 'equal_area_radial_partitions', 'cells': cells}


def number_layout_in_head_image(layout, head_view, floor_z):
    """Freeze slot labels using the initial calibrated head image orientation."""
    transform = np.asarray(head_view['T_world_camera'], dtype=float)
    intrinsics = head_view['intrinsics']

    def project(points):
        world = np.column_stack((points, np.full(len(points), floor_z)))
        camera = (world - transform[:3, 3]) @ transform[:3, :3]
        if np.any(camera[:, 2] <= 0):
            raise ValueError('placement_region_outside_initial_head_view')
        return camera[:, :2] / camera[:, 2, None] * [intrinsics['fx'], intrinsics['fy']] + [intrinsics['cx'], intrinsics['cy']]

    centers = project([cell['center_world_xy_m'] for cell in layout['cells']])
    heights = [np.ptp(project(cell['polygon_world_xy_m'])[:, 1]) for cell in layout['cells']]
    tolerance = float(np.median(heights)) / 2
    rows, row, anchor = [], [], None
    for index in sorted(range(len(centers)), key=lambda i: (centers[i, 1], centers[i, 0], i)):
        if anchor is not None and centers[index, 1] - anchor > tolerance + 1e-8:
            rows.extend(sorted(row, key=lambda i: centers[i, 0]))
            row, anchor = [], None
        if anchor is None:
            anchor = centers[index, 1]
        row.append(index)
    rows.extend(sorted(row, key=lambda i: centers[i, 0]))
    layout['cells'] = [layout['cells'][index] | {'slot_id': number,
        'initial_head_center_pixel_xy': centers[index].tolist()} for number, index in enumerate(rows, 1)]
    layout['numbering_source'] = 'initial_head_image_rows'
    return layout


def partition_in_head_frame(polygon, target_count, head_view, floor_z):
    direction = np.asarray(head_view['T_world_camera'], dtype=float)[:2, 0]
    if np.linalg.norm(direction) < 1e-8:
        raise ValueError('placement_head_horizontal_direction_unobservable')
    direction = direction / np.linalg.norm(direction)
    rotation = np.column_stack((direction, [-direction[1], direction[0]]))
    layout = partition_region(np.asarray(polygon) @ rotation, target_count)
    for cell in layout['cells']:
        cell['center_world_xy_m'] = (np.asarray(cell['center_world_xy_m']) @ rotation.T).tolist()
        cell['polygon_world_xy_m'] = (np.asarray(cell['polygon_world_xy_m']) @ rotation.T).tolist()
    layout['center_world_xy_m'] = (np.asarray(layout['center_world_xy_m']) @ rotation.T).tolist()
    layout['region_polygon_world_xy_m'] = (np.asarray(layout['region_polygon_world_xy_m']) @ rotation.T).tolist()
    layout['partition_direction_world_xy'] = direction.tolist()
    return number_layout_in_head_image(layout, head_view, floor_z)


def measured_support_walls(destination):
    """Keep independent bin-wall samples, never the visibility-dependent floor patch."""
    if destination.get('kind') != 'bin':
        return np.empty((0, 3))
    points = np.asarray(destination.get('points_world_m', []), dtype=float)
    reference = destination.get('placement_reference', {})
    if points.ndim != 2 or points.shape[1] != 3 or 'floor_z_m' not in reference:
        return np.empty((0, 3))
    return points[np.isfinite(points).all(axis=1) & (points[:, 2] > reference['floor_z_m'] + .015)]


def _wall_translation_observable(points):
    """Require nonparallel measured wall normals, not sliding along one wall."""
    sampled = points[::max(1, int(np.ceil(len(points) / 2000)))]
    _, neighbours = cKDTree(points).query(sampled, k=18)
    local = points[neighbours]
    centered = local - local.mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centered, centered))
    normals = vectors[:, :, 0]
    planar = (values[:, 1] > 1e-8) & (values[:, 0] < .05 * values[:, 1]) & (np.abs(normals[:, 2]) < .2)
    normals = normals[planar, :2]
    if len(normals) < 30:
        return False
    directions = np.linalg.eigvalsh(normals.T @ normals / len(normals))
    return directions[0] > .1 * directions[1]


def measured_region_translation(layout, reference, previous_walls=(), current_walls=()):
    """Require independent support motion in addition to congruent floor patches.

    A smaller visible patch cannot relocate or shrink an established region.
    Rotation/shape changes without a unique translation remain unresolved.
    """
    current = np.asarray(reference.get('region_polygon_world_xy_m', []), dtype=float)
    previous = np.asarray(layout['region_polygon_world_xy_m'], dtype=float)
    if current.shape != previous.shape or current.ndim != 2:
        return None
    delta = current.mean(axis=0) - previous.mean(axis=0)
    aligned = previous + delta
    if (np.max(cKDTree(aligned).query(current)[0]) > .002
            or np.max(cKDTree(current).query(aligned)[0]) > .002):
        return None
    dz = reference.get('floor_z_m', layout['floor_z_m']) - layout['floor_z_m']
    if np.linalg.norm(delta) <= .002 and abs(dz) <= .002:
        return None
    translation = np.array([*delta, dz])
    before, after = np.asarray(previous_walls), np.asarray(current_walls)
    for walls in (before, after):
        if (walls.ndim != 2 or walls.shape[1] != 3 or len(walls) < 30
                or np.any(np.ptp(walls[:, :2], axis=0) < .9 * np.ptp(previous, axis=0))
                or np.ptp(walls[:, 2]) < .01):
            return None
        if not _wall_translation_observable(walls):
            return None
    if all(np.mean(cKDTree(target).query(source)[0] <= .002) >= .95
           for source, target in ((before, after), (after, before))):
        return None  # Independent measurements still support a stationary box.
    # Multiple observed walls must support the same 3D translation. A changed
    # free floor patch, stationary walls, one wall, or unknown marker pose cannot.
    for source, target in ((before + translation, after), (after - translation, before)):
        if np.mean(cKDTree(target).query(source)[0] <= .002) < .95:
            return None
    return translation

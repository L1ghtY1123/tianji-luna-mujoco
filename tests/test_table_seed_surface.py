import json
from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.cameras import deproject_view
from luna_mujoco.demo_geometry import fit_inventory_table
from luna_mujoco.perception import PerceptionError


def frozen_table_inputs():
    path = Path(__file__).parent / 'fixtures/black_b4_table_seed'
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    arrays = np.load(path.with_suffix('.npz'))
    views, excluded = {}, {}
    for camera, calibration in metadata['views'].items():
        views[camera] = {**calibration, 'depth_m': arrays[camera + '_depth_m'],
                        'valid_depth': arrays[camera + '_valid_depth']}
        rows, columns = np.indices(views[camera]['depth_m'].shape)
        height, width = rows.shape
        mask = np.zeros(rows.shape, bool)
        for entity in metadata['entities'].values():
            if entity.get('kind') != 'table' and entity.get('reference_camera_id') == camera:
                x1, y1, x2, y2 = entity['bbox']
                mask |= ((columns >= x1 * width) & (columns < x2 * width)
                         & (rows >= y1 * height) & (rows < y2 * height))
        excluded[camera] = mask
    table = next(value for value in metadata['entities'].values() if value['kind'] == 'table')
    camera = table['reference_camera_id']
    return views[camera], table['bbox'], excluded[camera], {
        key: deproject_view(view, ~excluded[key]) for key, view in views.items() if key != camera}


def test_frozen_b4_vertical_bin_wall_cannot_expand_table_seed_bounds():
    view, bbox, excluded, supplemental = frozen_table_inputs()
    depth_before = view['depth_m'].copy()
    table = fit_inventory_table(view, bbox, excluded, supplemental_points=supplemental)
    center = np.asarray(table['model']['pose'][:3])
    dims = np.asarray(table['model']['dims'])
    lower, upper = center - dims / 2, center + dims / 2
    # Frozen pixel (218,586) is a vertical surface at y=-.560602486,
    # with a measured local normal 87.32 degrees from the fitted plane.
    assert lower[1] > -.46
    assert .75 < dims[1] < .85
    assert dims[0] > .55
    # Independently inspected table pixels must remain supported and unchanged.
    anchors = np.zeros_like(excluded)
    anchors[[175, 250, 350], [320, 220, 450]] = True
    points = deproject_view(view, anchors)
    assert np.all((points[:, :2] >= lower[:2]) & (points[:, :2] <= upper[:2]))
    np.testing.assert_allclose(points[:, 2], table['surface_height_m'], atol=1e-6)
    np.testing.assert_array_equal(view['depth_m'], depth_before)


def test_deferred_supplemental_region_preserves_frozen_table_and_inlier_evidence():
    view, bbox, excluded, supplemental = frozen_table_inputs()
    expected = fit_inventory_table(view, bbox, excluded, supplemental_points=supplemental)
    def restricted(seed, normal, offset):
        lower, upper = seed[:, :2].min(0) - .02, seed[:, :2].max(0) + .02
        return {camera: points[np.all((points[:, :2] >= lower) & (points[:, :2] <= upper), axis=1)
                               & (np.abs(points @ normal + offset) <= .03)]
                for camera, points in supplemental.items()}
    actual = fit_inventory_table(view, bbox, excluded, supplemental_points=restricted)
    assert actual == expected


def flat_table_view(size, focal):
    return {'depth_m': np.full((size, size), .6), 'valid_depth': np.ones((size, size), bool),
        'intrinsics': {'fx': focal, 'fy': focal, 'cx': (size - 1) / 2, 'cy': (size - 1) / 2},
        'T_world_camera': [[1., 0., 0., .55], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]}


def test_dense_horizontal_table_preserves_all_measured_seed_extents():
    view = flat_table_view(50, 200.)
    table = fit_inventory_table(view, [0., 0., 1., 1.], np.zeros((50, 50), bool))
    assert table['model']['dims'] == pytest.approx([.147, .147, .3])
    assert table['model']['pose'][:3] == pytest.approx([.55, 0., .45])
    assert table['inlier_count'] == 2500


def test_sparse_coplanar_seed_is_unknown_not_a_filled_support_cuboid():
    # More than the old 100-point minimum, but 24mm spacing gives no
    # locally supported 16-neighbour plane within the existing 20mm radius.
    view = flat_table_view(12, 25.)
    with pytest.raises(PerceptionError, match='demo_inventory_table_unobservable'):
        fit_inventory_table(view, [0., 0., 1., 1.], np.zeros((12, 12), bool))

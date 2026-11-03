"""Image, action binding and measured-only product geometry regressions."""
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from processing_fixtures import before_kwargs, capture_and_action, client, release_evidence, work_area

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.placement import measured_object_geometry
from luna_mujoco.robot_skills import processing_evidence as processing
from luna_mujoco.visual_api import VisualInventoryError


def test_actual_79639_unpack_reply_still_stops_at_point77_despite_point85_choice_probability():
    from pathlib import Path

    body = json.loads((Path(__file__).parent/'fixtures/unpack-79639-response.json').read_text(encoding='utf-8'))
    visual = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body))))
    capture, action = capture_and_action()
    release = release_evidence()
    before = deepcopy(release)
    with pytest.raises(VisualInventoryError, match='invalid_response'):
        processing.verify_unpacking(visual, capture, target_id='pink', release_evidence=release,
            action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert release == before
    assert body['answers'][1]['confidence'] == .77  # Historical two-question answer is never remapped.



def test_independent_pairs_keep_original_pixels_and_ignore_mock_pose():
    from luna_mujoco.visual_inventory import image_part

    capture, action = capture_and_action()
    rgb = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)
    capture['views']['3'].update(rgb=rgb, intrinsics={'fx': 100., 'fy': 100., 'cx': 80., 'cy': 60.})
    action.update(commanded_pose={'position': [999, 999, 999]}, simulation_truth={'id': 'do-not-use'})
    visual, calls = client()
    result = processing.verify_unpacking(visual, capture, target_id='pink', release_evidence=release_evidence(),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    for call in calls:
        content = call['input'][0]['content']
        camera = json.loads(content[0]['text'])['camera_id']
        assert image_part(capture['views'][camera]['rgb']) in content
        assert 'commanded_pose' not in json.dumps(call) and 'do-not-use' not in json.dumps(call)
        assert 'current_work_area_zoom' not in json.dumps(call)
    assert result['visual_evidence']['work_area_zoom'] is None
    np.testing.assert_array_equal(capture['views']['3']['rgb'], rgb)


@pytest.mark.parametrize('field,value', [('status', 'unknown'), ('target_id', 'other'),
    ('release_feedback_id', 'other'), ('confidence', .79), ('identity_source', 'appearance_only')])
def test_unpack_visibility_cannot_bind_id_without_verified_table_release(field, value):
    capture, action = capture_and_action()
    release = release_evidence()
    release[field] = value
    visual, calls = client()
    with pytest.raises(VisualInventoryError, match='table_release'):
        processing.verify_unpacking(visual, capture, target_id='pink', release_evidence=release,
            action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert not calls



@pytest.mark.parametrize('case', ['unknown_source', 'nonfinite_floor'])
def test_invalid_work_area_does_not_fabricate_a_crop_or_call_model(case):
    capture, action = capture_and_action()
    area = work_area()
    if case == 'unknown_source':
        area['source'] = 'simulation_truth'
    else:
        area['floor_z_m'] = float('nan')
    visual, calls = client()
    with pytest.raises(VisualInventoryError, match='work_area'):
        processing.verify_unpacking(visual, capture, target_id='pink', release_evidence=release_evidence(),
            action_evidence=action, owner_id='run', work_area=area, **before_kwargs())
    assert not calls



@pytest.mark.parametrize('case', ['offscreen', 'no_calibration', 'behind_camera'])
def test_unavailable_head_crop_preserves_three_views_without_new_visibility_gate(case):
    capture, action = capture_and_action()
    if case == 'offscreen':
        capture['views']['3']['T_world_camera'][0][3] = -100
    elif case == 'behind_camera':
        capture['views']['3']['T_world_camera'][2][3] = 0.
    else:
        capture['views']['3'].pop('intrinsics')
    visual, calls = client()
    result = processing.verify_unpacking(visual, capture, target_id='pink', release_evidence=release_evidence(),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert result['success']
    assert result['visual_evidence']['work_area_zoom'] is None
    assert sum(part['type'] == 'input_image' for part in calls[0]['input'][0]['content']) == 1



@pytest.mark.parametrize('field,value', [('target_id', 'other'), ('owner_id', 'other'),
    ('generation', 2), ('completed', False), ('source', 'simulation_truth'),
    ('completed_time', 3.), ('completed_perf_counter_ns', 201)])
def test_bad_unpack_action_stops_before_network(field, value):
    module = processing
    capture, action = capture_and_action()
    action[field] = value
    visual, calls = client()
    with pytest.raises(VisualInventoryError, match='result_evidence_'):
        module.verify_unpacking(visual, capture, target_id='pink', release_evidence=release_evidence(),
                                action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert not calls



def surface(z):
    return np.array([[.3, -.1, z], [.34, -.1, z], [.34, -.07, z], [.3, -.07, z]])



@pytest.mark.parametrize('levels', [(.62,), (.56, .62), (.66,)])
def test_product_geometry_preserves_measured_points_without_completing_or_clipping_height(levels):
    points = np.vstack([surface(z) for z in levels])
    target = {'id': 'product', 'asset_id': 'black_core', 'points_world_m': points.copy()}
    measured = measured_object_geometry(target)
    np.testing.assert_array_equal(measured, points)
    np.testing.assert_array_equal(target['points_world_m'], points)
    assert set(target) == {'id', 'asset_id', 'points_world_m'}


@pytest.mark.parametrize('obsolete', [
    {'completed_geometry_points_world_m': []},
    {'completed_geometry_points_world_m': [[.3, -.1, .56]]},
    {'dimension_prior': {}},
    {'geometry_completion': {'source': 'measured_surface_and_user_height_prior'}},
    {'geometry_completion': {'synthetic_point_count': 1}},
])
def test_old_product_completion_requires_reobservation_instead_of_silent_reuse(obsolete):
    points = surface(.62)
    target = {'id': 'product', 'points_world_m': points.copy(), **deepcopy(obsolete)}
    with pytest.raises(ValueError, match='^synthetic_scene_geometry_unsupported:reobserve_required$'):
        measured_object_geometry(target)
    np.testing.assert_array_equal(target['points_world_m'], points)
    for key, value in obsolete.items():
        assert target[key] == value


@pytest.mark.parametrize('points', [[], [[.3, -.1]], [[.3, -.1, float('nan')]]])
def test_asset_metadata_cannot_replace_missing_or_invalid_observed_product_depth(points):
    with pytest.raises(ValueError, match='^measured_object_geometry_required$'):
        measured_object_geometry({'asset_id': 'black_core', 'points_world_m': points})

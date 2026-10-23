"""Current appearance changes never mint task IDs or promote uncertain evidence."""
import importlib
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from decision_fixtures import decision_answer
from test_demo_partial_identity import inputs

from luna_mujoco.demo_geometry import complete_source_height
from luna_mujoco.demo_perception import DemoPerception, DemoVisualInventory
from luna_mujoco.perception import PerceptionError
from luna_mujoco.visual_api import VisualInventoryError


def lifecycle_module():
    spec = importlib.util.find_spec('luna_mujoco.demo_lifecycle_perception')
    assert spec is not None, 'phase-specific perception boundary is missing'
    return importlib.import_module(spec.name)


def record(phase='unpacked_confirmed'):
    result = {'phase': phase, 'appearance': 'wrapped' if phase in {
        'wrapped_source', 'table_released', 'unpack_pending'} else 'unpacked',
        'appearance_revision': 0 if phase == 'unpack_pending' else 1,
        'to_table_arm': 'A', 'to_right_bin_arm': 'B'}
    if phase == 'unpack_pending':
        result.update(table_release_feedback_id='table-release', table_visual_evidence={
            'target_id': 'pink', 'status': 'at_destination', 'confidence': .95,
            'identity_source': 'execution_chain', 'release_feedback_id': 'table-release',
            'binding_id': 'table-binding', 'request_hash': 'table-request',
            'observation_id': 'table-before', 'source': 'openai_decisions'})
    return result


def work_area():
    return {'center_world_xy_m': [.36, 0.],
            'region_polygon_world_xy_m': [[.18, -.18], [.54, -.18], [.54, .18], [.18, .18]],
            'floor_z_m': .56, 'source': 'configured_xy_measured_support'}


def capture_and_action(source='mock_vla'):
    capture, _, _, _ = inputs()
    capture.update(observation_id='after', generation=3, time=2., captured_perf_counter_ns=200)
    capture['views']['3'].update(T_world_camera=[[1, 0, 0, .36], [0, -1, 0, 0],
        [0, 0, -1, 1.56], [0, 0, 0, 1]], intrinsics={'fx': 30., 'fy': 30., 'cx': 16., 'cy': 12.})
    action = {'source': source, 'owner_id': 'run', 'target_id': 'pink', 'generation': 3,
              'action_id': 'action-1', 'completed': True, 'completed_time': 1.,
              'completed_perf_counter_ns': 100, 'table_release_feedback_id': 'table-release'}
    return capture, action



def before_kwargs(capture=None, lifecycle=None, *, owner='run', action=None):
    current, default_action = capture_and_action()
    capture, action = capture or current, action or default_action
    lifecycle = lifecycle or record('unpack_pending')
    before = deepcopy(capture)
    before.update(observation_id='table-before', time=min(capture['time'], action['completed_time'])-.5,
                  captured_perf_counter_ns=action['completed_perf_counter_ns']-1)
    return {'active_target_ids': ['pink'], 'table_release_capture': before, 'table_release_reference': lifecycle_module().table_release_reference(
        before, lifecycle['table_visual_evidence'], owner)}

def client(answers=None):
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        rows = []
        for question in payload['questions']:
            negative = question['name'].endswith('_conflict') or question['name'] == 'workpiece_still_held'
            value, confidence = (answers or {}).get(question['name'], ('no' if negative else 'yes', .95))
            rows.append(decision_answer(question, value, confidence))
        return httpx.Response(200, json={'model': 'gpt-6-luna', 'answers': rows})

    return DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond))), calls


def test_current_black_queries_keep_initial_pink_reference_and_same_id():
    module = lifecycle_module()
    _, _, entities, reference = inputs()
    original = deepcopy(entities)
    task = {'item_lifecycle': {'pink': record()}}
    current = module.lifecycle_entities(entities, task)
    assert set(current) == {'pink'}
    assert current['pink']['color'] == 'black'
    assert entities == original == reference['entities']
    queries = DemoPerception._segmentation_queries(current)
    assert queries and all(query['color'] == 'black' and 'pink' not in query['prompt'] for query in queries)
    assert module.lifecycle_entities(current, task) == current


def test_unpacking_checks_three_rgb_without_mutating_phase_or_geometry():
    module = lifecycle_module()
    capture, action = capture_and_action()
    lifecycle = record('unpack_pending')
    before = deepcopy(lifecycle)
    visual, calls = client()
    result = module.verify_unpacking(visual, capture, target_id='pink', lifecycle=lifecycle,
                                    action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert result['success'] and result['status'] == 'confirmed'
    assert result['target_id'] == 'pink' and result['appearance_revision'] == 1
    assert result['geometry_updated'] is False and lifecycle == before
    assert len(calls) == 1
    assert sum(part['type'] == 'input_image' for part in calls[0]['input'][0]['content']) == 2
    assert result['visual_evidence']['request_hash'] == result['request_hash']


def test_unpack_prompt_accepts_one_visible_current_view_without_rgb_workspace_judgment():
    module = lifecycle_module()
    capture, action = capture_and_action()
    visual, calls = client()
    module.verify_unpacking(visual, capture, target_id='pink', lifecycle=record('unpack_pending'),
                            action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    payload = calls[0]
    content = payload['input'][0]['content']
    context = json.loads(content[0]['text'])
    questions = {question['name']: question for question in payload['questions']}
    assert set(questions) == {'unpack_result', 'wrapper_removed', 'observation_sufficient'}
    assert questions['unpack_result']['type'] == 'predicate'
    assert 'A clear view in one camera is enough' in context['views']
    assert 'right workspace' not in questions['unpack_result']['instructions']
    assert 'three CURRENT camera views together' in context['views']
    assert 'historical references' in context['views']
    assert 'alone does not establish success' in context['stage']
    assert 'black inner part' in context['task']
    texts = [part['text'] for part in content if part['type'] == 'input_text']
    assert [text for text in texts if text.startswith('CURRENT RGB camera')] == [
        'CURRENT RGB camera 2', 'CURRENT RGB camera 1', 'CURRENT RGB camera 3']
    from luna_mujoco.visual_inventory import image_part
    actual = {part['image_url'] for part in content if part['type'] == 'input_image'}
    assert {image_part(view['rgb'])['image_url'] for view in capture['views'].values()} <= actual


def test_actual_79639_unpack_reply_still_stops_at_point77_despite_point85_choice_probability():
    from pathlib import Path

    body = json.loads((Path(__file__).parent/'fixtures/unpack-79639-response.json').read_text(encoding='utf-8'))
    visual = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body))))
    capture, action = capture_and_action()
    lifecycle = record('unpack_pending')
    before = deepcopy(lifecycle)
    with pytest.raises(VisualInventoryError, match='invalid_response'):
        lifecycle_module().verify_unpacking(visual, capture, target_id='pink', lifecycle=lifecycle,
            action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert lifecycle == before
    assert body['answers'][1]['confidence'] == .77  # Historical two-question answer is never remapped.


@pytest.mark.parametrize('black,same', [(('no', .6), ('unknown', .94)), (('yes', .72), ('unknown', .13))])
def test_frozen_unpack_unknown_answers_are_not_promoted_by_prompt_clarification(black, same):
    module = lifecycle_module()
    capture, action = capture_and_action()
    visual, _ = client({'unpack_result': black})
    result = module.verify_unpacking(visual, capture, target_id='pink', lifecycle=record('unpack_pending'),
                                    action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert result['success'] is False and result['status'] == 'unknown'
    assert result['appearance'] == 'wrapped' and result['appearance_revision'] == 0
    assert result['visual_evidence']['judgments']['unpack_result'] != 'yes'
    assert result['visual_evidence']['answers'][0]['probability'] == (black[1] if black[0] == 'yes' else 1-black[1])


def test_head_zoom_is_only_fixed_region_pixels_and_ignores_mock_pose():
    import base64
    import hashlib
    from io import BytesIO

    from PIL import Image

    from luna_mujoco.visual_inventory import image_part

    capture, action = capture_and_action()
    rgb = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)
    capture['views']['3'].update(rgb=rgb, intrinsics={'fx': 100., 'fy': 100., 'cx': 80., 'cy': 60.})
    action.update(commanded_pose={'position': [999, 999, 999]}, simulation_truth={'id': 'do-not-use'})
    visual, calls = client()
    result = lifecycle_module().verify_unpacking(visual, capture, target_id='pink', lifecycle=record('unpack_pending'),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    content = calls[0]['input'][0]['content']
    metadata = json.loads(content[-2]['text'])
    assert metadata['image_role'] == 'current_work_area_zoom'
    assert metadata['source_camera_id'] == '3' and metadata['observation_id'] == capture['observation_id']
    assert metadata['crop_bounds_xyxy_px'] == [50, 30, 110, 90]
    assert metadata['display_size_px'] == [240, 240]
    assert 'not an object mask' in metadata['meaning'] and 'not proof' in metadata['meaning']
    original = image_part(rgb)
    assert metadata['source_image_sha256'] == hashlib.sha256(
        base64.b64decode(original['image_url'].split(',', 1)[1])).hexdigest()
    expected = np.asarray(Image.fromarray(rgb[30:90, 50:110]).resize((240, 240), Image.Resampling.NEAREST))
    assert content[-1] == image_part(expected)
    image = Image.open(BytesIO(base64.b64decode(content[-1]['image_url'].split(',', 1)[1])))
    assert image.size == (240, 240)
    assert result['visual_evidence']['work_area_zoom'] == metadata
    assert 'commanded_pose' not in json.dumps(calls[0]) and 'do-not-use' not in json.dumps(calls[0])
    np.testing.assert_array_equal(capture['views']['3']['rgb'], rgb)


@pytest.mark.parametrize('field,value', [('status', 'unknown'), ('target_id', 'other'),
    ('release_feedback_id', 'other'), ('confidence', .79), ('identity_source', 'appearance_only')])
def test_unpack_visibility_cannot_bind_id_without_verified_table_release(field, value):
    capture, action = capture_and_action()
    lifecycle = record('unpack_pending')
    lifecycle['table_visual_evidence'][field] = value
    visual, calls = client()
    with pytest.raises(VisualInventoryError, match='table_release'):
        lifecycle_module().verify_unpacking(visual, capture, target_id='pink', lifecycle=lifecycle,
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
        lifecycle_module().verify_unpacking(visual, capture, target_id='pink', lifecycle=record('unpack_pending'),
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
    result = lifecycle_module().verify_unpacking(visual, capture, target_id='pink', lifecycle=record('unpack_pending'),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert result['success']
    assert result['visual_evidence']['work_area_zoom'] is None
    assert sum(part['type'] == 'input_image' for part in calls[0]['input'][0]['content']) == 1


@pytest.mark.parametrize('answer,confidence', [('yes', .79), ('unknown', .99), ('no', .99)])
def test_unpacking_uncertainty_does_not_authorize_black_geometry(answer, confidence):
    module = lifecycle_module()
    capture, action = capture_and_action()
    visual, _ = client({'unpack_result': (answer, confidence)})
    result = module.verify_unpacking(visual, capture, target_id='pink', lifecycle=record('unpack_pending'),
                                    action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert result['success'] is False and result['status'] == 'unknown'
    assert result['geometry_updated'] is False
    assert result['appearance'] == 'wrapped' and result['appearance_revision'] == 0


@pytest.mark.parametrize('field,value', [('target_id', 'other'), ('owner_id', 'other'),
    ('generation', 2), ('completed', False), ('source', 'simulation_truth'),
    ('completed_time', 3.), ('completed_perf_counter_ns', 201)])
def test_bad_unpack_action_stops_before_network(field, value):
    module = lifecycle_module()
    capture, action = capture_and_action()
    action[field] = value
    visual, calls = client()
    with pytest.raises(VisualInventoryError, match='lifecycle_'):
        module.verify_unpacking(visual, capture, target_id='pink', lifecycle=record('unpack_pending'),
                                action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    assert not calls


def surface(z):
    return np.array([[.3, -.1, z], [.34, -.1, z], [.34, -.07, z], [.3, -.07, z]])


def test_black_height_prior_is_60mm_and_keeps_measured_points():
    target = {'points_world_m': surface(.62)}
    before = target['points_world_m'].copy()
    complete_source_height(target, support_z_m=.56, support_source='cached_measured_table',
                           observation_id='after', height_prior_m=.06)
    np.testing.assert_array_equal(target['points_world_m'], before)
    assert target['geometry_completion']['height_prior_m'] == .06
    assert target['geometry_completion']['height_prior_source'] == 'user_specified_60mm'
    assert np.min(target['completed_geometry_points_world_m'][:, 2]) == pytest.approx(.56)


def test_measured_full_height_is_not_augmented_or_called_synthetic():
    target = {'points_world_m': np.vstack((surface(.56), surface(.62)))}
    complete_source_height(target, support_z_m=.56, support_source='cached_measured_table',
                           observation_id='after', height_prior_m=.06)
    assert 'completed_geometry_points_world_m' not in target
    assert target['geometry_completion']['status'] == 'measured_height_covered'
    assert target['geometry_completion']['synthetic_point_count'] == 0


def test_inconsistent_black_height_is_not_forced_to_user_prior():
    target = {'points_world_m': surface(.66)}
    with pytest.raises(PerceptionError, match='demo_height_prior_inconsistent'):
        complete_source_height(target, support_z_m=.56, support_source='cached_measured_table',
                               observation_id='after', height_prior_m=.06)


def test_current_black_does_not_require_camera2_or_completed_table_slots():
    capture, detections, entities, reference = inputs(('1', '3'))
    reference['demo_task']['item_lifecycle'] = {'pink': record()}
    visual, calls = client({'identity_0': ('pink', .95), 'identity_1': ('unknown', .6),
                           'visibility_pink': ('visible', .99), 'demo_unlisted_pink': ('none', .99)})
    result = visual.resolve(capture, detections, entities=entities, reference=reference,
                            instruction='unwrap then move to right bin', purpose='associate_inventory')
    assert result['identity_coverage']['remaining_source_ids'] == []
    assert result['missing_geometry_ids'] == []
    assert len(calls) == 1
    assert 'black' in calls[0]['questions'][0]['choices'][0]['description']


def test_model_lifecycle_context_excludes_simulator_truth_and_action_coordinates():
    capture, detections, entities, reference = inputs(('1',))
    lifecycle = record() | {'mock_action': {'physical_truth_after': 'FORBIDDEN_SIMULATOR_TRUTH',
        'commanded_pose': {'position': [123.456, 0, 0]}},
        'unpacking_evidence': {'response': {'unused_private_context': 'FORBIDDEN_OLD_RESPONSE'}}}
    reference['demo_task']['item_lifecycle'] = {'pink': lifecycle}
    visual, calls = client({'identity_0': ('pink', .95), 'visibility_pink': ('visible', .99),
                           'demo_unlisted_pink': ('none', .99)})
    visual.resolve(capture, detections, entities=entities, reference=reference,
                   instruction='unwrap', purpose='associate_inventory')
    encoded = json.dumps(calls[0])
    assert 'FORBIDDEN_' not in encoded and '123.456' not in encoded
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])
    assert context['item_lifecycle']['pink'] == record()
    assert 'initially empty but may now contain completed black' in context['geometric_location_rule']
    assert reference['demo_task']['item_lifecycle']['pink'] == lifecycle


@pytest.mark.parametrize('bin_confidence', [.4, .95])
def test_lifecycle_changes_only_item_appearance_and_preserves_stable_bin_reference(bin_confidence):
    capture, detections, entities, reference = inputs(('3', '1'))
    entities['pink'].update(reference_camera_id='3', bbox=[.1, .1, .4, .4])
    entities['right-bin'] = {'id': 'right-bin', 'kind': 'bin', 'color': 'blue',
        'description': 'initially empty right blue box', 'roles': ['collision'],
        'reference_camera_id': '1', 'bbox': [.1, .1, .8, .8]}
    reference.update(observation_id='initial', views=deepcopy(capture['views']))
    reference['demo_task'].update(right_bin_id='right-bin', item_lifecycle={'pink': record()})
    before = deepcopy(reference)
    visual, calls = client({'identity_0': ('pink', .95), 'identity_1': ('right-bin', bin_confidence),
        'visibility_pink': ('visible', .99), 'visibility_right-bin': ('visible', .99),
        'demo_unlisted_pink': ('none', .99)})
    if bin_confidence < .8:
        with pytest.raises(VisualInventoryError, match='visual_identity_unknown'):
            visual.resolve(capture, detections, entities=entities, reference=reference,
                           instruction='unwrap', purpose='associate_inventory')
    else:
        result = visual.resolve(capture, detections, entities=entities, reference=reference,
                                instruction='unwrap', purpose='associate_inventory')
        assert result['missing_geometry_ids'] == []
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])
    questions = {question['name']: question for question in calls[0]['questions']}
    assert 'Containers retain their INITIAL visual references' in context['rules']
    assert 'empty or occupied does not change the container ID' in context['rules']
    assert 'Task roles and camera numbers do not assign identity' in context['rules']
    assert 'compare the full RGB images with the initial reference' in questions['identity_1']['instructions']
    assert 'Still-wrapped pink inputs keep their original IDs' in questions['identity_1']['instructions']
    assert 'Is initial entity right-bin visible' in questions['visibility_right-bin']['instructions']
    assert 'current lifecycle appearance of entity pink' in questions['visibility_pink']['instructions']
    assert context['identity_display_legend']['R2']['entity_id'] == 'right-bin'
    assert [option['value'] for option in questions['identity_1']['choices']] == [
        'pink', 'right-bin', 'reject', 'unknown']
    assert reference['entities'] == before['entities']
    assert reference['demo_task'] == before['demo_task']


def drop_inputs():
    capture, action = capture_and_action('validated_held_attachment_pose_drop')
    action.update(release_feedback_id='released', destination_id='right-bin', holding_verified=True,
                  attachment_verified=True, open_confirmed=True, before_observation_id='before')
    reference = {'observation_id': 'before', 'generation': 3, 'time': .5,
                 'views': deepcopy(capture['views']),
                 'demo_task': {'right_bin_id': 'right-bin', 'item_lifecycle': {
                     'pink': record('right_bin_released'), 'previous': record('completed')}}}
    return capture, action, reference


def test_drop_verification_uses_real_before_and_current_images_and_execution_binding():
    module = lifecycle_module()
    assert hasattr(module, 'verify_right_bin_drop'), 'drop result verification is missing'
    capture, action, reference = drop_inputs()
    visual, calls = client()
    result = module.verify_right_bin_drop(visual, capture, target_id='pink', lifecycle=record('right_bin_released'),
        action_evidence=action, reference=reference, owner_id='run')
    assert result['success'] is True and result['visual_evidence']['status'] == 'at_destination'
    assert result['visual_evidence']['destination_id'] == 'right-bin'
    assert result['visual_evidence']['release_feedback_id'] == 'released'
    assert result['geometry_updated'] is False
    assert len(calls) == 1
    assert sum(part['type'] == 'input_image' for part in calls[0]['input'][0]['content']) == 1
    assert result['visual_evidence']['before_observation_id'] == 'before'


@pytest.mark.parametrize('value,confidence', [('unknown', .99), ('no', .99), ('yes', .79)])
def test_seeing_only_previous_bin_objects_cannot_complete_current_item(value, confidence):
    module = lifecycle_module()
    assert hasattr(module, 'verify_right_bin_drop'), 'drop result verification is missing'
    capture, action, reference = drop_inputs()
    visual, _ = client({'current_item_added': (value, confidence)})
    result = module.verify_right_bin_drop(visual, capture, target_id='pink', lifecycle=record('right_bin_released'),
        action_evidence=action, reference=reference, owner_id='run')
    assert result['success'] is False and result['visual_evidence']['status'] == 'unknown'


@pytest.mark.parametrize('field,value', [('holding_verified', False), ('attachment_verified', False),
    ('open_confirmed', False), ('destination_id', 'source-bin'), ('release_feedback_id', ''),
    ('before_observation_id', 'initial-empty-bin')])
def test_drop_bad_chain_or_stale_before_frame_stops_before_network(field, value):
    module = lifecycle_module()
    assert hasattr(module, 'verify_right_bin_drop'), 'drop result verification is missing'
    capture, action, reference = drop_inputs()
    action[field] = value
    visual, calls = client()
    with pytest.raises(VisualInventoryError, match='lifecycle_'):
        module.verify_right_bin_drop(visual, capture, target_id='pink', lifecycle=record('right_bin_released'),
            action_evidence=action, reference=reference, owner_id='run')
    assert not calls


def test_unpack_pending_stops_before_sam_or_geometric_estimation():
    capture, _, entities, reference = inputs()
    reference['demo_task']['item_lifecycle'] = {'pink': record('unpack_pending')}
    with pytest.raises(PerceptionError, match='unpacking_visual_confirmation_required_before_geometry'):
        DemoPerception()._observe_entities(capture, '', None, {}, entities, reference, None)


def test_completed_black_remains_required_current_geometry_without_old_table_track():
    capture, detections, entities, reference = inputs(('1',))
    reference['demo_task']['item_lifecycle'] = {'pink': record('completed')}
    visual, calls = client({'identity_0': ('pink', .95), 'visibility_pink': ('visible', .99),
                           'progress_pink': ('unknown', .99), 'demo_unlisted_pink': ('none', .99)})
    result = visual.resolve(capture, detections, entities=entities, reference=reference,
        progress={'pink': {'status': 'completed', 'completed_identity_track': {'obsolete': True}}},
        instruction='unwrap', purpose='associate_inventory')
    assert len(calls) == 1
    assert result['identity_coverage']['required_entity_ids'] == ['pink']
    assert result['missing_geometry_ids'] == []
    assert not any(question['name'].startswith('progress_') for question in calls[0]['questions'])
    np.testing.assert_array_equal(result['groups'][0][0]['points'], detections[0]['points'])


@pytest.mark.parametrize('phase,height', [('wrapped_source', .11), ('unpacked_confirmed', .044)])
def test_observe_applies_current_phase_height_and_keeps_current_depth(monkeypatch, phase, height):
    from luna_mujoco.perception import Perception

    capture, _, entities, reference = inputs()
    reference.update(observation_id='initial', demo_program={'version': 1, 'steps': []})
    reference['demo_task'].update(source_bin_id='source-bin', item_lifecycle={'pink': record(phase)})
    measured = surface(.56 + height)
    view = {'instance_masks': {'pink': np.ones((2, 2), bool)}, 'depth_m': np.ones((2, 2)),
            'valid_depth': np.ones((2, 2), bool)}
    observation = {'visual_reference': reference, 'visual_inventory': {},
                   'objects': [{'id': 'pink', 'color': 'measured_color', 'points_world_m': measured},
                               {'id': 'source-bin', 'placement_reference': {'floor_z_m': .56}}],
                   'scene_entities': entities, 'views': {'2': view}}
    monkeypatch.setattr(Perception, 'observe', lambda *args, **kwargs: deepcopy(observation))
    engine = DemoPerception()
    engine.demo_tables = {'primary_table_id': 'table', 'tables': {'table': {'surface_height_m': .56}}}
    result = engine.observe(capture)
    target = result['objects'][0]
    assert target['geometry_completion']['height_prior_m'] == height
    assert target['color'] == ('black' if phase == 'unpacked_confirmed' else 'pink')
    np.testing.assert_array_equal(target['points_world_m'], measured)
    np.testing.assert_array_equal(result['views']['2']['depth_m'], view['depth_m'])


def test_completed_bin_geometry_is_retained_without_upright_completion(monkeypatch):
    from luna_mujoco.perception import Perception

    capture, _, entities, reference = inputs()
    reference.update(observation_id='initial', demo_program={'version': 1, 'steps': []})
    reference['demo_task'].update(source_bin_id='source-bin', item_lifecycle={'pink': record('completed')})
    # A dropped core may be sideways or piled on previous contents.
    measured = surface(.43)
    observation = {'visual_reference': reference, 'visual_inventory': {},
                   'objects': [{'id': 'pink', 'color': 'black', 'points_world_m': measured}],
                   'scene_entities': entities, 'views': {}}
    monkeypatch.setattr(Perception, 'observe', lambda *args, **kwargs: deepcopy(observation))
    engine = DemoPerception()
    engine.demo_tables = {'primary_table_id': 'table', 'tables': {'table': {'surface_height_m': .56}}}
    result = engine.observe(capture)
    assert result['objects'][0]['id'] == 'pink'
    np.testing.assert_array_equal(result['objects'][0]['points_world_m'], measured)
    assert 'completed_geometry_points_world_m' not in result['objects'][0]
    assert result['objects'][0]['geometry_completion']['synthetic_point_count'] == 0

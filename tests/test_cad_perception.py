"""CAD matching must not authorize mismatched or unsupported geometry."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import trimesh


def api():
    import luna_mujoco.cad_perception as module
    return module


def cad_resolver(*args, **kwargs):
    return api().CadGeometryResolver(*args, **kwargs)


def fixture():
    mask = np.zeros((32, 32), dtype=bool)
    mask[6:27, 6:27] = True
    depth = np.full(mask.shape, 2., dtype=np.float32)
    depth[mask] = .95
    rows, cols = np.nonzero(mask)
    points = np.column_stack(((cols - 16) * .95 / 100, (rows - 16) * .95 / 100,
                              np.full(len(rows), .95)))
    pose = np.eye(4)
    pose[2, 3] = 1.
    view = {'rgb': np.zeros((32, 32, 3), np.uint8), 'depth_m': depth,
            'valid_depth': np.ones(mask.shape, bool), 'intrinsics': {
                'fx': 100., 'fy': 100., 'cx': 16., 'cy': 16.},
            'T_world_camera': np.eye(4), 'frame_id': 'observation', 'camera_id': '1',
            'calibration_id': 'calibration', 'capture_time': 1., 'simulation_generation': 2}
    capture = {'observation_id': 'observation', 'generation': 2, 'time': 1., 'views': {'1': view}}
    group = [{'entity_id': 'object-1', 'camera_id': '1', 'mask': mask,
              'silhouette_mask': mask, 'score': .9, 'points': points}]
    target = {'id': 'object-1', 'points_world_m': points, 'obb': {'source': 'measured'},
              'observation_provenance': {'observation_id': 'observation', 'generation': 2}}
    return capture, group, target, pose


class Match:
    def __init__(self, asset_id):
        self.asset_id = asset_id

    def match(self, capture, group, target, catalog, *, instruction):
        status = self.asset_id if self.asset_id in {'unknown', 'no_catalog_candidate'} else 'candidate'
        return {'status': status, 'asset_id': self.asset_id if status == 'candidate' else None,
                'evidence': {'source': 'openai_decisions', 'role': 'cad_candidate_proposal',
                'observation_id': capture['observation_id'], 'request_hash': 'match-hash'}}


def test_explicit_unknown_geometry_uses_current_obb_without_cad_retrieval():
    from unittest.mock import Mock
    capture, group, target, _ = fixture()
    target['geometry_policy'] = {'mode': 'unknown_obb', 'instance_id': target['id'],
        'source': 'user_instruction_interpreted_by_responses', 'instruction_quote': '粉色物体没有CAD'}
    matcher, estimator = Mock(), Mock()
    result = cad_resolver(BoxCatalog(), matcher, estimator).resolve(
        capture, group, target, instruction='粉色物体没有CAD，放到桌面')
    assert result['obb'] == target['obb']
    assert result['cad_match']['geometry_policy'] == 'explicit_unknown_obb'
    matcher.match.assert_not_called()
    estimator.estimate_instance.assert_not_called()


@pytest.mark.parametrize('change', ['id', 'quote'])
def test_explicit_unknown_policy_requires_bound_id_and_original_instruction(change):
    capture, group, target, _ = fixture()
    target['geometry_policy'] = {'mode': 'unknown_obb', 'instance_id': target['id'],
        'source': 'user_instruction_interpreted_by_responses', 'instruction_quote': '没有CAD'}
    target['geometry_policy']['instance_id' if change == 'id' else 'instruction_quote'] = 'invented'
    with pytest.raises(ValueError, match='geometry_policy_binding_invalid'):
        cad_resolver(BoxCatalog(), Match('box')).resolve(
            capture, group, target, instruction='这个物体没有CAD')


class Pose:
    def __init__(self, pose, source_hash):
        self.pose, self.source_hash = pose, source_hash

    def estimate_instance(self, view, mask, score, **kwargs):
        assert kwargs['mask_capture_id'] == view['frame_id']
        assert kwargs['mask_camera_id'] == view['camera_id']
        return {'T_world_object': self.pose.tolist(), 'T_world_source': self.pose.tolist(),
                'T_source_canonical_m': np.eye(4).tolist(), 'source_sha256': self.source_hash,
                'model_id': 'box', 'usage': 'pose_hypothesis_only', 'capture_id': view['frame_id'],
                'camera_id': view['camera_id'], 'calibration_id': view['calibration_id'],
                'simulation_generation': view['simulation_generation'], 'request_sha256': 'pose-hash'}


class BoxCatalog:
    def __init__(self):
        self.records = {'box': {'asset_id': 'box', 'model_id': 'box', 'mesh_key': 'box',
                              'source_sha256': 'source-hash', 'description': 'box', 'display_name': 'Box'}}

    def record(self, key):
        return deepcopy(self.records[key])

    def geometry(self, key):
        mesh = trimesh.creation.box(extents=[.2, .2, .1])
        return {'vertices_m': mesh.vertices, 'faces': mesh.faces, 'T_source_canonical_m': np.eye(4),
                'source_sha256': 'source-hash', 'mesh_sha256': 'mesh-hash', 'asset_id': 'box',
                'path': '/checked/box.stl'}


def test_actual_catalog_mesh_is_metric_and_original_source_frame_preserved():
    catalog = api().CadCatalog.from_file()
    geometry = catalog.geometry('black_core')
    assert geometry['asset_id'] == 'black_workpiece'
    np.testing.assert_allclose(np.ptp(geometry['vertices_m'], axis=0), [.044, .056, .0745], atol=1e-7)
    np.testing.assert_allclose(geometry['T_source_canonical_m'][:3, 3], [0, .006, -.12525])
    mesh = trimesh.Trimesh(vertices=geometry['vertices_m'], faces=geometry['faces'], process=False)
    points, _ = trimesh.sample.sample_surface(mesh, 64, seed=0)
    assert np.isfinite(points).all()


def test_catalog_names_keep_original_library_identity_as_explicit_alias():
    catalog = api().CadCatalog.from_file()
    assert set(catalog.records) == {'blue_box', 'black_workpiece'}
    assert catalog.record('blue_box')['display_name'] == '蓝色盒子'
    assert catalog.record('THREADED_VISUAL_SAMPLE_MM')['asset_id'] == 'black_workpiece'
    assert catalog.record('black_workpiece')['source_asset_id'] == 'THREADED_VISUAL_SAMPLE_MM'
    assert 'blue container' in catalog.record('blue_box')['segmentation_aliases']


def test_catalog_rejects_source_hash_mismatch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / 'resources/cad_catalog.json').read_text(encoding='utf-8'))
    payload['assets'][0]['source_sha256'] = '0' * 64
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='cad_source_hash'):
        api().CadCatalog(payload['assets'], resource_root=api().CadCatalog.from_file().resource_root).geometry('blue_box')


def test_no_catalog_candidate_keeps_observed_obb_without_calling_pose():
    capture, group, target, _ = fixture()
    resolver = cad_resolver(BoxCatalog(), Match('no_catalog_candidate'), pose_estimator=None)
    result = resolver.resolve(capture, group, target, instruction='move object')
    assert result['cad_match']['status'] == 'no_catalog_candidate'
    assert result['obb'] == target['obb']
    assert 'cad' not in result


def test_unknown_retrieval_fails_explicitly_without_pose_or_confirmed_cache():
    capture, group, target, _ = fixture()
    resolver = cad_resolver(BoxCatalog(), Match('unknown'), pose_estimator=None)
    with pytest.raises(ValueError, match='cad_retrieval_uncertain') as caught:
        resolver.resolve(capture, group, target, instruction='move object')
    assert caught.value.details['status'] == 'unknown'
    assert caught.value.details['instance_id'] == target['id']
    assert resolver.confirmed_models == {}


def test_known_cad_uses_current_pose_and_replaces_obb_without_mutating_target():
    capture, group, target, pose = fixture()
    target['cad_match'] = {'status': 'unknown', 'evidence': {'observation_id': 'older'}}
    resolver = cad_resolver(BoxCatalog(), Match('box'), Pose(pose, 'source-hash'))
    result = resolver.resolve(capture, group, target, instruction='move object')
    assert result['cad']['pose_validation']['status'] == 'accepted_by_policy'
    assert result['cad']['observation_id'] == 'observation'
    assert result['cad']['generation'] == 2
    assert result['cad']['instance_id'] == 'object-1'
    assert result['cad']['pose_validation']['accepted'] is True
    assert result['cad']['pose_validation']['post_registration_geometry_check'] is False
    shape = result['cad']['shape_bounds']
    np.testing.assert_allclose(shape['minimum_world_m'], [-.1, -.1, .95])
    np.testing.assert_allclose(shape['maximum_world_m'], [.1, .1, 1.05])
    np.testing.assert_array_equal(shape['T_world_object'], result['cad']['T_world_object'])
    assert shape['mesh_sha256'] == result['cad']['mesh_sha256']
    assert 'obb' not in result and 'obb' in target
    assert 'cad_match' not in result and 'cad_match' in target


@pytest.mark.parametrize('failure', ['wrong_source', 'missing_depth', 'stale_capture'])
def test_selected_cad_failure_never_silently_falls_back_to_obb(failure):
    capture, group, target, pose = fixture()
    digest = 'source-hash'
    if failure == 'wrong_source':
        digest = 'other-source'
    elif failure == 'missing_depth':
        capture['views']['1']['valid_depth'][:] = False
    else:
        capture['views']['1']['frame_id'] = 'older'
    resolver = cad_resolver(BoxCatalog(), Match('box'), Pose(pose, digest))
    with pytest.raises(ValueError, match='cad_'):
        resolver.resolve(capture, group, target, instruction='move object')


def test_pose_fit_uses_identity_silhouette_and_records_geometry_mask_difference():
    capture, group, target, pose = fixture()
    silhouette = group[0]['mask'].copy()
    group[0]['silhouette_mask'] = silhouette
    group[0]['mask'][6:10, 6:27] = False
    target['points_world_m'] = target['points_world_m'][84:]

    class MaskCheckedPose(Pose):
        def estimate_instance(self, view, mask, score, **options):
            np.testing.assert_array_equal(mask, silhouette)
            return super().estimate_instance(view, mask, score, **options)

    result = cad_resolver(BoxCatalog(), Match('box'),
        MaskCheckedPose(pose, 'source-hash')).resolve(capture, group, target, instruction='move object')
    evidence = result['cad']['mask_evidence'][0]
    assert evidence['source'] == 'identity_bound_silhouette'
    assert evidence['geometry_filtered_pixels'] == 84
    assert evidence['silhouette_pixels'] == int(silhouette.sum())
    assert result['cad']['pose_validation']['surface']['full_point_count'] == len(target['points_world_m'])


def test_explicit_low_confidence_candidate_is_only_a_proposal_not_confirmed_cad():
    module = api()
    capture, group, target, _ = fixture()

    class Response:
        status_code = 200
        def json(self):
            return {'answers': [{'name': 'cad_model', 'type': 'choice', 'choice': 'R1', 'confidence': .7}]}

    class Client:
        def post(self, url, **kwargs):
            text = json.dumps(kwargs['json'])
            assert 'CURRENT RGB' in text and 'MASKED SUBJECT' in text and 'CAD CATALOG' in text
            return Response()

    matcher = module.CadModelMatcher(api_key='test-only', client=Client())
    result = matcher.match(capture, group, target, BoxCatalog(), instruction='move object')
    assert result['asset_id'] == 'box' and result['status'] == 'candidate'
    assert result['evidence']['role'] == 'cad_candidate_proposal'
    assert result['evidence']['cad_confirmed'] is False
    assert result['evidence']['answers'][0]['confidence'] == .7


def test_matcher_binds_labeled_sheet_choice_to_asset_without_task_distraction():
    import base64
    from io import BytesIO

    from PIL import Image

    module = api()
    capture, group, target, _ = fixture()
    capture['views']['2'] = deepcopy(capture['views']['1'])
    group[0].update(prompt='blue container', prompt_proposals=['blue container', 'blue bin'])
    target.update(kind='bin', color='blue', description='Open rectangular storage box',
                  appearance_keywords=['open rim'], description_provenance={'source': 'responses_entity_proposal'})

    class Response:
        status_code = 200
        def json(self):
            return {'answers': [{'name': 'cad_model', 'type': 'choice', 'choice': 'R1', 'confidence': .97}]}

    class Client:
        def post(self, url, **kwargs):
            payload = kwargs['json']
            content = payload['input'][0]['content']
            text = json.dumps(content)
            assert 'grasp the pink object' not in text
            assert 'CURRENT MASKED SUBJECT' in text and 'CAD CATALOG' in text
            assert '"kind": "bin"' in '\n'.join(item.get('text', '') for item in content)
            context = json.loads(content[0]['text'])
            assert context['observed_appearance']['description'] == target['description']
            assert context['measured_geometry']['point_count'] == len(target['points_world_m'])
            assert context['measured_geometry']['world_axis_extent_m'] == pytest.approx([.19, .19, 0])
            assert context['catalog_references'][0]['canonical_extent_m'] == pytest.approx([.2, .2, .1])
            assert context['sam_query_evidence'] == [{'camera_id': '1', 'queries': ['blue container', 'blue bin']}]
            assert context['catalog_references'][0]['asset_id'] == 'box'
            images = [item for item in content if item['type'] == 'input_image']
            assert len(images) == 1
            with Image.open(BytesIO(base64.b64decode(images[0]['image_url'].split(',', 1)[1]))) as sheet:
                assert sheet.size == (768, 514)
            assert [row['value'] for row in payload['questions'][0]['choices']] == ['R1', 'no_catalog_candidate', 'unknown']
            return Response()

    result = module.CadModelMatcher(api_key='test-only', client=Client()).match(
        capture, group, target, BoxCatalog(), instruction='grasp the pink object')
    assert result['asset_id'] == 'box'
    assert result['evidence']['catalog_choices']['R1']['asset_id'] == 'box'
    assert result['evidence']['presentation_version'] == 'subject_reference_sheet_v2'
    assert result['evidence']['camera_ids'] == ['1']


def test_display_exposure_uses_only_uniform_gain_without_changing_original_pixels():
    rgb = np.full((10, 10, 3), 240, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    rgb[mask] = [8, 10, 12]
    rgb[4:6, 4:6] = [16, 20, 24]
    original = rgb.copy()
    displayed, gain = api()._display_exposure(rgb, mask)
    assert gain == pytest.approx(128 / 24)
    np.testing.assert_array_equal(displayed[2, 2], [43, 53, 64])
    np.testing.assert_array_equal(displayed[4, 4], [85, 107, 128])
    np.testing.assert_array_equal(displayed[0, 0], [96, 96, 96])
    np.testing.assert_array_equal(rgb, original)


def test_explicit_unknown_does_not_promote_runner_up_catalog_probability():
    capture, group, target, _ = fixture()

    class Response:
        status_code = 200
        def json(self):
            return {'answers': [{'name': 'cad_model', 'type': 'choice', 'choice': 'unknown',
                                 'confidence': .6, 'probabilities': {'unknown': .51, 'R1': .49}}]}

    class Client:
        def post(self, *args, **kwargs):
            return Response()

    matcher = api().CadModelMatcher(api_key='test', client=Client())
    result = matcher.match(capture, group, target, BoxCatalog(), instruction='move object')
    assert result['status'] == 'unknown' and result['asset_id'] is None
    assert result['evidence']['answers'][0]['choice'] == 'unknown'
    with pytest.raises(ValueError, match='cad_retrieval_uncertain'):
        cad_resolver(BoxCatalog(), matcher).resolve(capture, group, target, instruction='move object')


def test_small_catalog_validation_tolerance_is_scale_bounded():
    geometry = api().CadCatalog.from_file().geometry('black_core')
    policy = api().cad_tolerance_policy(geometry, .015)
    assert policy['tolerance_m'] == pytest.approx(.0022, abs=1e-8)
    assert policy['configured_cap_m'] == .015
    assert policy['shortest_extent_fraction'] == .05
    assert api().cad_tolerance_policy(geometry, .001)['tolerance_m'] == .001


def test_perception_passes_bound_entity_description_without_simulator_asset_hint():
    from geometry_perception_fixture import GeometryPerception
    from test_perception import FrozenSegmenter, capture_fixture

    capture, masks = capture_fixture()
    entities = {identifier: {'id': identifier, 'kind': 'cube', 'color': 'any', 'roles': ['collision'],
        'description': 'Measured candidate ' + identifier, 'appearance_keywords': ['flat top'],
        'identity_source': 'responses_program', 'asset_id': 'UNTRUSTED_ASSET',
        'segmentation_query': {'kind': 'cube', 'color': 'any', 'prompt': 'cube'}}
        for identifier in ('first', 'second')}
    captured = []

    class Resolver:
        def identity_constraints(self, generation):
            return {}

        def resolve(self, current_capture, group, target, *, instruction):
            captured.append(deepcopy(target))
            return target

    result = GeometryPerception(FrozenSegmenter(masks), cad_resolver=Resolver()).observe(
        capture, source_hint={'scene_entities': entities})
    assert len(result['objects']) == len(captured) == 2
    for target in captured:
        assert target['description'] == entities[target['id']]['description']
        assert target['appearance_keywords'] == ['flat top']
        assert target['description_provenance']['source'] == 'responses_entity_proposal'
        assert target['description_provenance']['binding_source'] == 'openai_decisions'
        assert target['description_provenance']['observation_id'] == capture['observation_id']
        assert target['description_provenance']['cad_evidence'] is False
        assert 'asset_id' not in target and 'SIM_SECRET' not in repr(target)


def test_catalog_rejects_source_paths_outside_asset_root(tmp_path):
    record = {'asset_id': 'box', 'mesh_key': 'box', 'geometry': 'rigid_reference',
              'source_sha256': 'hash'}
    root = tmp_path / 'cad'
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'units': 'm', 'assets': {'box': {
        'source': {'units': 'mm', 'sha256': 'hash', 'path': '../../outside.stl'}}}}))
    with pytest.raises(ValueError, match='cad_asset_path_invalid'):
        api().CadCatalog([record], resource_root=tmp_path).geometry('box')


def test_validated_model_match_reused_but_each_observation_requires_new_pose():
    capture, group, target, pose = fixture()

    class OnceMatcher(Match):
        def match(self, *args, **kwargs):
            if hasattr(self, 'used'):
                raise AssertionError('Stable model must not trigger another semantic choice')
            self.used = True
            return super().match(*args, **kwargs)

    estimator = Pose(pose, 'source-hash')
    resolver = cad_resolver(BoxCatalog(), OnceMatcher('box'), estimator)
    first = resolver.resolve(capture, group, target, instruction='move object')
    capture['observation_id'] = 'later'
    capture['views']['1']['frame_id'] = 'later'
    target['observation_provenance']['observation_id'] = 'later'
    second = resolver.resolve(capture, group, target, instruction='move object')
    assert first['cad']['match_evidence']['observation_id'] == 'observation'
    evidence = second['cad']['match_evidence']
    assert evidence['match_source_observation_id'] == 'observation'
    assert evidence['current_identity_binding']['observation_id'] == 'later'
    assert evidence['historical_match_reused'] is True
    assert second['cad']['pose_evidence']['capture_id'] == 'later'
    estimator.pose = pose.copy()
    estimator.pose[0, 3] = .08
    updated = resolver.resolve(capture, group, target, instruction='move object')
    assert updated['cad']['T_world_object'][0][3] == pytest.approx(.08)


def test_model_match_cache_isolated_by_generation_and_permanent_instance():
    capture, group, target, pose = fixture()
    matcher = Match('box')
    resolver = cad_resolver(BoxCatalog(), matcher, Pose(pose, 'source-hash'))
    resolver.resolve(capture, group, target, instruction='move object')
    matcher.asset_id = 'no_catalog_candidate'
    other = deepcopy(target)
    other['id'] = 'object-2'
    other_group = [{**group[0], 'entity_id': 'object-2'}]
    assert 'cad' not in resolver.resolve(capture, other_group, other, instruction='move object')
    capture['generation'] = 3
    capture['views']['1']['simulation_generation'] = 3
    assert 'cad' not in resolver.resolve(capture, group, target, instruction='move object')


def test_unknown_or_failed_pose_never_populates_model_match_cache():
    capture, group, target, pose = fixture()
    matcher = Match('no_catalog_candidate')
    estimator = Pose(pose.copy(), 'source-hash')
    resolver = cad_resolver(BoxCatalog(), matcher, estimator)
    assert 'cad' not in resolver.resolve(capture, group, target, instruction='move object')
    matcher.asset_id = 'box'
    estimator.source_hash = 'wrong-source'
    with pytest.raises(ValueError, match='cad_pose_provenance_mismatch'):
        resolver.resolve(capture, group, target, instruction='move object')
    matcher.asset_id = 'no_catalog_candidate'
    assert 'cad' not in resolver.resolve(capture, group, target, instruction='move object')


@pytest.mark.parametrize('confidence,expected_calls', [(.88, 1), (.79, 2)])
def test_high_confidence_no_catalog_binding_reuses_identity_not_old_obb(confidence, expected_calls):
    capture, group, target, _ = fixture()
    class NoCatalog(Match):
        calls = 0
        def match(self, *args, **kwargs):
            self.calls += 1
            answer = super().match(*args, **kwargs)
            answer['evidence']['answers'] = [{'type': 'choice', 'name': 'cad_model',
                'choice': 'no_catalog_candidate', 'confidence': confidence,
                'probabilities': [{'value': 'no_catalog_candidate', 'probability': .91}]}]
            return answer
    matcher = NoCatalog('no_catalog_candidate')
    resolver = cad_resolver(BoxCatalog(), matcher)
    first = resolver.resolve(capture, group, target, instruction='move object')
    capture['observation_id'] = 'later'
    capture['views']['1']['frame_id'] = 'later'
    target['observation_provenance']['observation_id'] = 'later'
    target['obb'] = {'source': 'fresh_measured_obb', 'observation_id': 'later'}
    second = resolver.resolve(capture, group, target, instruction='move object')
    assert matcher.calls == expected_calls
    assert second['obb'] == target['obb'] and second['obb'] != first['obb']
    if expected_calls == 1:
        assert second['cad_match']['evidence']['observation_id'] == 'observation'
        assert second['cad_match']['current_identity_binding']['observation_id'] == 'later'
        assert second['cad_match']['historical_match_reused'] is True
    # A changed catalog must permit another model retrieval for this same instance.
    resolver.catalog.records['box']['source_sha256'] = 'new-cad-revision'
    resolver.resolve(capture, group, target, instruction='move object')
    assert matcher.calls == expected_calls + 1
    other = deepcopy(target)
    other['id'] = 'object-2'
    resolver.resolve(capture, [{**group[0], 'entity_id': 'object-2'}], other, instruction='move object')
    assert matcher.calls == expected_calls + 2
    capture['generation'] = 3
    capture['views']['1']['simulation_generation'] = 3
    resolver.resolve(capture, group, target, instruction='move object')
    assert matcher.calls == expected_calls + 3

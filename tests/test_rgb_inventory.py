"""RGB discovery owns the ledger; SAM3 owns only current geometry."""
import importlib.util
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from test_visual_inventory import client_with_answers, fixture

from jev_mujoco.task_progress import TaskProgress
from jev_mujoco.visual_inventory import VisualInventory


def entities():
    return {'red': {'id': 'red', 'kind': 'cube', 'color': 'red', 'description': 'red cube on left',
                    'visual_membership': 'required', 'bbox': [.1, .2, .4, .7]},
            'other': {'id': 'other', 'kind': 'cube', 'color': 'red', 'description': 'red cube on right',
                      'visual_membership': 'excluded', 'bbox': [.6, .2, .9, .7]}}


def test_discovery_uses_responses_rgb_only_and_allocates_stable_ids():
    assert importlib.util.find_spec('jev_mujoco.visual_discovery') is not None, 'RGB discovery is missing'
    from jev_mujoco.visual_discovery import VisualDiscovery
    capture, _ = fixture()
    calls, audits = [], []
    discovered = {'entities': [
        {'ref': 'table', 'kind': 'table', 'color': 'brown', 'description': 'working table',
         'bbox': [0., 0., 1., 1.], 'membership': 'excluded', 'segmentation_prompt': '', 'parent_ref': None},
        {'ref': 'a', 'kind': 'cube', 'color': 'red', 'description': 'red cube on left',
         'bbox': [.1, .1, .4, .6], 'membership': 'required', 'segmentation_prompt': 'red cube', 'parent_ref': 'table'},
        {'ref': 'b', 'kind': 'cube', 'color': 'red', 'description': 'red cube on right',
         'bbox': [.6, .1, .9, .6], 'membership': 'required', 'segmentation_prompt': 'red cube', 'parent_ref': 'table'}]}
    def respond(request):
        calls.append(json.loads(request.content))
        assert str(request.url) == 'https://api.openai.com/v1/responses'
        return httpx.Response(200, json={'status': 'completed', 'model': 'gpt-6-luna', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(discovered)}]}]})
    discovery = VisualDiscovery(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)), audit=audits.append)
    first = discovery.discover(capture, instruction='move both red cubes')
    # Geometry changes must not affect a request built solely from the same head RGB.
    capture['views']['3']['depth_m'] *= 7
    capture['views']['3']['valid_depth'][:] = False
    capture['views']['3']['instance_masks'] = {'unused': np.ones((24, 32), bool)}
    second = discovery.discover(capture, instruction='move both red cubes')
    assert first['entities'] == second['entities']
    assert len(first['entities']) == 3 and len(calls) == len(audits) == 1
    assert sum(e['visual_membership'] == 'required' for e in first['entities'].values()) == 2
    assert calls[0]['text']['format']['type'] == 'json_schema'
    assert [part['type'] for part in calls[0]['input'][0]['content']] == ['input_text', 'input_image']
    request_text = json.dumps(calls[0])
    assert 'depth_m' not in request_text and 'points_world' not in request_text
    assert audits[0]['endpoint'] == '/v1/responses'


def test_decisions_binds_masks_without_receiving_world_geometry_or_mask_contours():
    capture, detection = fixture()
    fixed = entities()
    fixed['red']['position'] = [12345., 12345., 12345.]
    calls = []
    result = VisualInventory(api_key='test', client=client_with_answers(
        {'identity_0': 'red', 'visibility_red': 'visible', 'visibility_other': 'visible'}, calls)).resolve(
            capture, [detection], instruction='move left red cube', entities=fixed)
    assert result['groups'][0][0]['entity_id'] == 'red'
    text = '\n'.join(p['text'] for p in calls[0]['input'][0]['content'] if p['type'] == 'input_text')
    assert '12345' not in text and 'visible_centroid_world_m' not in text
    assert 'purple contour' not in json.dumps(calls[0])
    assert result['missing_geometry_ids'] == ['other']
    assert result['membership'] == {'red': 'required', 'other': 'excluded'}


def test_zero_sam_candidates_still_allows_visual_transfer_judgment():
    capture, _ = fixture()
    result = VisualInventory(api_key='test', client=client_with_answers(
        {'visibility_red': 'visible', 'visibility_other': 'unseen', 'progress_red': 'at_destination'}, [])).resolve(
            capture, [], instruction='move red', entities=entities(),
            progress={'red': {'destination_id': 'other'}})
    assert result['progress']['red']['status'] == 'at_destination'
    assert result['groups'] == []
    assert set(result['missing_geometry_ids']) == {'red', 'other'}


def test_semantic_inventory_keeps_object_without_sam_geometry():
    progress = TaskProgress('move red', scope='collection')
    semantic = entities()
    observation = {'observation_id': 'initial', 'objects': [], 'table': {},
                   'scene_entities': semantic, 'visual_inventory': {'membership': {'red': 'required', 'other': 'excluded'}}}
    progress.initialize_scene_inventory(observation)
    progress.observe([], 'initial')
    assert set(progress.scene_entities) == {'red', 'other'}
    assert 'red' in progress.records and progress.membership['red'] == 'required'
    assert progress.scene_entities['red']['geometry_status'] == 'missing'
    assert 'position' not in progress.records['red']


def test_same_kind_distinct_entities_keep_distinct_mask_ownership():
    capture, detection = fixture()
    second = deepcopy(detection)
    mask = np.roll(second['mask'], 16, axis=1)
    second.update(mask=mask, silhouette_mask=mask, pixels_rc=np.column_stack(np.nonzero(mask)))
    result = VisualInventory(api_key='test', client=client_with_answers(
        {'identity_0': 'red', 'identity_1': 'other', 'visibility_red': 'visible', 'visibility_other': 'visible'}, [])).resolve(
            capture, [detection, second], instruction='move left red cube', entities=entities())
    assert [group[0]['entity_id'] for group in result['groups']] == ['red', 'other']


def test_initial_rgb_boxes_are_displayed_when_sam_misses_but_never_reused_as_current_boxes():
    from jev_mujoco.observation_labels import make_object_labels
    capture, _ = fixture()
    capture.update(objects=[], scene_entities=entities(), visual_discovery={'source': 'openai_responses'})
    initial = make_object_labels(capture, {}, {'red': 'required'})
    assert {item['id'] for item in initial['items']} == {'red', 'other'}
    assert all(item['geometry_status'] == 'missing' for item in initial['items'])
    capture['observation_id'] = 'later'
    capture['visual_discovery'] = None
    assert make_object_labels(capture, {}, {'red': 'required'})['items'] == []


def test_real_perception_flow_discovers_before_sam_and_retains_missing_geometry():
    from geometry_perception_fixture import ScriptedDiscovery
    from test_perception import capture_fixture

    from jev_mujoco.perception import Perception
    capture, _ = capture_fixture()
    order = []
    class Discovery(ScriptedDiscovery):
        def discover(self, capture, *, instruction):
            order.append('responses')
            return super().discover(capture, instruction=instruction)
    class Detector:
        def segment(self, rgb, text, timestamp_us):
            order.append(('sam3', text))
            return []
    calls = []
    inventory = VisualInventory(api_key='test', client=client_with_answers(
        {'visibility_first': 'visible', 'visibility_second': 'visible'}, calls))
    result = Perception(Detector(), visual_discovery=Discovery(), visual_inventory=inventory).observe(
        capture, source_hint={'camera_ids': ['3'], 'visual_decision': {'source': 'jev', 'purpose': 'initialize_inventory'}})
    assert order == ['responses', ('sam3', 'cube')]
    assert set(result['scene_entities']) == {'first', 'second'}
    assert set(result['visual_inventory']['missing_geometry_ids']) == {'first', 'second'}
    assert len(result['mapping_unresolved']) == 2 and result['objects'] == []
    progress = TaskProgress('move both', scope='collection')
    progress.initialize_scene_inventory(result)
    progress.observe(result['objects'], result['observation_id'])
    assert set(progress.membership) == {'first', 'second'}


@pytest.mark.parametrize('stage', ['sam3', 'decisions'])
def test_discovered_inventory_survives_a_later_failure_and_is_not_rediscovered(stage):
    from geometry_perception_fixture import ScriptedDiscovery
    from test_perception import capture_fixture

    from jev_mujoco.perception import Perception, PerceptionError
    from jev_mujoco.visual_inventory import VisualInventoryError
    capture, _ = capture_fixture()
    class Discovery(ScriptedDiscovery):
        calls = 0
        def discover(self, capture, *, instruction):
            self.calls += 1
            return super().discover(capture, instruction=instruction)
    class Detector:
        fail = stage == 'sam3'
        def segment(self, *args):
            if self.fail:
                raise PerceptionError('sam3_service_unavailable')
            return []
    class Inventory:
        fail = True
        def resolve(self, capture, detections, **kwargs):
            if self.fail:
                raise VisualInventoryError('visual_decisions_http_429')
            return {'groups': [], 'membership': {}, 'evidence': {}}
    discovery, detector, inventory = Discovery(), Detector(), Inventory()
    engine = Perception(detector, visual_discovery=discovery, visual_inventory=inventory)
    with pytest.raises(PerceptionError) as failure:
        engine.observe(capture, source_hint={'camera_ids': ['3'],
            'visual_decision': {'source': 'jev', 'purpose': 'initialize_inventory'}})
    preserved = getattr(failure.value, 'initial_inventory', {})
    assert set(preserved.get('entities', {})) == {'first', 'second'}
    assert preserved['observation_id'] == capture['observation_id']
    detector.fail = inventory.fail = False
    capture['observation_id'] = 'new-capture'
    for view in capture['views'].values():
        view['frame_id'] = 'new-capture'
    result = engine.observe(capture, source_hint={'scene_entities': preserved['entities'],
        'visual_reference': preserved['reference'], 'camera_ids': ['1', '2', '3'],
        'visual_decision': {'source': 'jev', 'purpose': 'associate_inventory'}})
    assert discovery.calls == 1 and set(result['scene_entities']) == {'first', 'second'}


async def test_runner_preserves_initial_rgb_identity_on_failure_and_updates_with_three_cameras(tmp_path):
    from test_jev_visual_policy import visual_runner

    from jev_mujoco.perception import PerceptionError
    runner = visual_runner(tmp_path)
    captured = []
    def failed(capture, **options):
        captured.append(options['source_hint'])
        error = PerceptionError('visual_decisions_http_429')
        error.initial_inventory = {'entities': entities(), 'reference': {'entities': entities(),
            'observation_id': capture['observation_id'], 'views': {}},
            'observation_id': capture['observation_id'], 'evidence': {'source': 'openai_responses'}}
        raise error
    runner.algorithms.observe = failed
    context = {'visual_plan': {'initial': 'initialize_inventory'}}
    with pytest.raises(PerceptionError):
        await runner.observe(context, purpose='scene', recovery=True)
    assert set(runner.task_progress.scene_entities) == {'red', 'other'}
    assert context['visual_inventory_reference']['observation_id']
    assert {item['id'] for item in runner.state.get('object_labels', {}).get('items', [])} == {'red', 'other'}
    context['visual_plan'] = {'initial': 'associate_inventory'}
    with pytest.raises(PerceptionError):
        await runner.observe(context, purpose='scene', recovery=True)
    assert captured[-1]['camera_ids'] == ['1', '2', '3']
    assert set(captured[-1]['scene_entities']) == {'red', 'other'}


def test_rgb_inventory_numbering_preserves_row_grouping_without_sam_masks():
    semantic = entities()
    semantic['red']['bbox'] = [.1, .30, .3, .50]
    semantic['other']['bbox'] = [.6, .29, .8, .51]
    semantic['other']['visual_membership'] = 'required'
    observation = {'observation_id': 'initial', 'objects': [], 'scene_entities': semantic}
    progress = TaskProgress('move both', scope='collection')
    progress.initialize_scene_inventory(observation)
    result = progress.freeze_inventory(observation)
    assert {item['id']: item['display_id'] for item in result['objects']} == {'red': 1, 'other': 2}

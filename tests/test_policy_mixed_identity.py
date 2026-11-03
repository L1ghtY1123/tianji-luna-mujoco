"""The runtime accepts separately bound measured identity, not a source label alone."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_demo_container_tracking import module, tracked_case
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.runtime.controller import PolicyController


def case(tmp_path):
    _, _, capture, detections, entities, reference = tracked_case()
    proof = module().container_correspondence(capture, detections, reference, entities)
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.task_progress = SimpleNamespace(inventory_finalized=True)
    coverage = {'source': 'mixed_decisions_measured_container', 'observation_id': capture['observation_id'],
        'request_hash': 'current-decisions', 'minimum_confidence': .8, 'accepted_entity_ids': ['right'],
        'measured_container_entity_ids': ['right'], 'container_correspondence_binding_hash': proof['binding_hash']}
    observation = {'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'objects': [{'id': 'right', 'kind': 'bin'}], 'scene_entities': entities,
        'visual_inventory': {'identity_coverage': coverage, 'container_correspondence': deepcopy(proof),
            'evidence': {'request_hash': 'current-decisions', 'container_correspondence': deepcopy(proof)}},
        'demo_inventory_check': {'source': 'openai_decisions', 'observation_id': capture['observation_id'],
            'request_hash': 'current-decisions', 'confidence': .67, 'status': 'unknown'}}
    return runner, observation


def test_runtime_accepts_current_bound_container_identity_without_claiming_source_completeness(tmp_path):
    runner, observation = case(tmp_path)
    runner._check_source_view(observation)
    assert runner.state['visible_inventory_resolution']['status'] == 'unknown'


@pytest.mark.parametrize('bad', ['hash', 'generation', 'capture', 'source', 'identity', 'kind', 'missing_proof'])
def test_runtime_rejects_unbound_mixed_identity(tmp_path, bad):
    runner, observation = case(tmp_path)
    visual = observation['visual_inventory']
    proof = visual['container_correspondence']
    if bad == 'hash': visual['identity_coverage']['container_correspondence_binding_hash'] = 'other'
    elif bad == 'generation': observation['generation'] += 1
    elif bad == 'capture': proof['observation_id'] = 'stale'
    elif bad == 'source': proof['source'] = 'simulation_truth'
    elif bad == 'identity': visual['identity_coverage']['measured_container_entity_ids'] = ['another']
    elif bad == 'kind': observation['scene_entities']['right']['kind'] = 'object'
    elif bad == 'missing_proof': visual.pop('container_correspondence')
    with pytest.raises(ValueError, match='source_inventory_unknown'):
        runner._check_source_view(observation)

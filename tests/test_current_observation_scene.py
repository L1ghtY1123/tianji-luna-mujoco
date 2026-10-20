from copy import deepcopy

import pytest
from test_geometry_scene_reuse import mapped_runtime
from test_perception_mapping import wire
from test_route_observation import route_context

from jev_mujoco.perception_mapping import decode_mapping_payload


@pytest.mark.parametrize('legacy', [
    {'retained_bin_ids': ['placed']},
    {'completed_transfer': {'target_id': 'placed'}},
])
def test_fresh_map_rejects_hidden_voxel_reuse_requests(legacy):
    with pytest.raises(ValueError, match='predicted_scene_reuse_disabled'):
        decode_mapping_payload(wire() | legacy)


@pytest.mark.parametrize('kind', ['bin', 'marker'])
async def test_completed_obstacle_requires_current_evidence_for_either_destination(tmp_path, kind):
    runner, state, algorithms = route_context(tmp_path)
    state['observation_pose_capture_id'] = state['observation']['observation_id']
    state['observation_pose_reference'] = deepcopy(state['observation'])
    runner.task_progress.records['placed'] = {'id': 'placed', 'status': 'completed', 'destination_kind': kind}
    with pytest.raises(ValueError, match='placed_object_not_observed'):
        await runner.build_scene(state, 'selected')
    assert algorithms.map_calls == []


def test_fresh_map_does_not_keep_old_object_layers(monkeypatch):
    runtime, payload = mapped_runtime(monkeypatch)
    runtime.geometry_model['voxel']['semantic_instance::old-placed'] = {'position_source': 'action_prediction'}
    result = runtime.build_perception(payload)
    assert 'semantic_instance::old-placed' not in runtime.geometry_model['voxel']
    assert 'old-placed' not in result['collision_objects']
    assert not hasattr(runtime, 'predicted_layers')


async def test_completed_object_visible_only_to_wrist_allows_fresh_map(tmp_path, monkeypatch):
    runner, state, algorithms = route_context(tmp_path)
    state['observation_pose_capture_id'] = state['observation']['observation_id']
    state['observation_pose_reference'] = deepcopy(state['observation'])
    runner.task_progress.records['placed'] = {'id': 'placed', 'status': 'completed', 'destination_kind': 'bin'}
    state['observation']['objects'].append({'id': 'placed', 'kind': 'cube',
        'measurement_sources': {'camera_ids': ['1'], 'source': 'registered_rgbd'}})
    build = algorithms.build_scene

    def measured_only(observation, *args, **kwargs):
        assert 'retained_bin_ids' not in observation
        assert 'completed_transfer' not in observation
        return build(observation, *args, **kwargs)

    monkeypatch.setattr(algorithms, 'build_scene', measured_only)
    await runner.build_scene(state, 'selected')
    assert algorithms.map_calls

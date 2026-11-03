import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_resolve_placement import setup

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.placement_candidates import subdivision_cell_centers
from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.runtime.placement import resolve_xy_then_z, xy_proposal_context


def test_authorized_xy_question_uses_constraints_without_rejudging_scene_visibility():
    from luna_mujoco.agent.decisions import ChoiceRequest, DecisionsJudgments

    class Judge(DecisionsJudgments):
        def request(self, payload, capture, purpose, parse):
            assert capture['views'] == {}
            content = payload['input'][0]['content']
            assert all(item['type'] == 'input_text' for item in content)
            introduction = json.loads(content[0]['text'])
            assert 'Do not rejudge the whole task or current visibility' in introduction['rules']
            return parse({'answers': [{'type': 'choice', 'name': 'placement_xy',
                'choice': 'p2', 'confidence': .3}]}, {})

    context = {'verification_scope': 'placement_xy_proposal',
               'placement_zone': {'semantic_scope': 'entire_region'},
               'xy_candidates': [{'candidate_id': 'p2', 'travel_xy_m': .2}]}
    request = ChoiceRequest(name='placement_xy', instructions='Choose a proposed XY',
                            choices=('p2', 'unknown'), selection_mode='proposal')
    result = Judge().judge({'observation_id': 'old-pixels-current-region',
        'views': {'1': {'rgb': np.zeros((4, 4, 3), np.uint8)}}}, requests=[request], context=context)
    assert result['judgments']['placement_xy']['status'] == 'proposed'
    context['placement_zone']['semantic_scope'] = 'candidate_selection'
    with pytest.raises(ValueError, match='placement_xy_requires_authorized_region'):
        Judge().judge({'observation_id': 'o', 'views': {}}, requests=[request], context=context)


def test_xy_choice_preserves_constraints_without_per_sphere_height_logs():
    zone = {'destination_id': 'box', 'intent': {'relation': 'inside'},
            'polygon_world_xy_m': [[0, 0], [1, 0], [1, 1], [0, 1]],
            'semantic_evidence': {'large_previous_model_payload': 'x' * 30000}}
    detail = {'initial_goal_position_m': [.2, .3, .4], 'required_clearance_m': .005,
              'world_clear_sample_count': 31, 'held_ik_sample_count': 31,
              'tested_heights': [{'world_collision_costs': [0.] * 332} for _ in range(31)]}
    attempts = [{'subdivision_cells': 4, 'candidate_indices': [0], 'reason': 'placement_no_candidate',
        'details': {'candidate_failures': [{'candidate_index': 0,
            'code': 'release_height_no_held_ik_goal', 'details': detail}]}}]
    region, failures = xy_proposal_context(zone, attempts)
    assert region['polygon_world_xy_m'] == zone['polygon_world_xy_m']
    assert failures[0]['candidates'][0]['reason'] == 'release_height_no_held_ik_goal'
    assert failures[0]['candidates'][0]['world_clear_sample_count'] == 31
    assert len(json.dumps([region, failures])) < 1500
    assert len(detail['tested_heights']) == 31  # Original audit evidence remains complete.


def test_complete_subdivision_levels_have_no_root_or_parent_centers():
    polygon = [[0, 0], [4, 0], [4, 2], [0, 2]]
    offset = [[.1, -.1], [.3, .1]]
    four = subdivision_cell_centers(polygon, offsets_world_xy_m=offset, cell_count=4)
    np.testing.assert_allclose(four, [[.8, .5], [.8, 1.5], [2.8, .5], [2.8, 1.5]])
    for count in (8, 16):
        rows = subdivision_cell_centers(polygon, offsets_world_xy_m=offset, cell_count=count)
        assert len({tuple(row) for row in rows}) == count


@pytest.mark.parametrize('reject_first_level', [False, True])
async def test_decisions_selects_xy_before_numeric_height_and_levels_are_bounded(monkeypatch, reject_first_level):
    algorithm, _, observation, scene, holding, request, zone = setup(monkeypatch)
    zone.update(semantic_scope='entire_region', semantic_evidence={}, original_requirement='put on table')
    events = []
    async def judge(context, requests, *, extra, include_task):
        assert include_task is False
        choice = requests[0].choices[0]
        events.append(('decision', extra['subdivision_cells'], choice))
        return {'judgments': {'placement_xy': {'status': 'proposed', 'value': choice, 'confidence': .6}}}
    async def call(function, *args, **kwargs):
        kwargs.pop('report_progress')
        level = kwargs['subdivision_cells']
        index = kwargs['candidate_indices'][0]
        assert events[-1] == ('decision', level, 'placement-' + str(index))
        events.append(('numeric', level, index))
        if reject_first_level and level == 4:
            raise AlgorithmError('occupied', code='placement_no_candidate')
        return function(*args, **kwargs)
    controller = SimpleNamespace(state={'target_id': 'seen-red', 'arm': 'A', 'destination_id': 'demo-destination'},
        algorithms=algorithm, _judge=judge, call_algorithm=call)
    context = {'placement_zone': zone, 'holding': holding, 'scene': scene,
        'planning_settings': PlanningSettings().model_dump()}
    result = await resolve_xy_then_z(controller, context, observation, request)
    assert result['status'] == 'success'
    levels = [row[1] for row in events if row[0] == 'numeric']
    assert levels[-1] == (8 if reject_first_level else 4)
    assert len(result['candidates']) == 1
    assert context['placement_resolution'] == result
    assert result['selection_evidence']['search_order'] == 'decisions_xy_then_numeric_z'

"""Live JEV semantics probe over explicit test fixtures; no robot or GPU calls."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

from jev_mujoco.decisions import DecisionError, Decisions
from jev_mujoco.task_progress import TaskProgress

BIND_CASES = [
    ('Move a red object and a green object into a blue box',
     {'requested_count': 2, 'color': 'any', 'destination_color': 'blue', 'selection_order': 'inventory'}, []),
    ('Move an orange object and a green object into the box',
     {'requested_count': 2, 'color': 'any', 'selection_order': 'inventory'}, []),
    ('将单个红色物体和单个绿色物体放进盒子',
     {'requested_count': 2, 'color': 'any', 'selection_order': 'inventory'}, []),
    ('桌面上有两个红色方块和三个黑色圆柱，把盒内绿色物体搬到黄色区域',
     {'requested_count': None, 'kind': 'any', 'color': 'green', 'route': 'box_table',
      'destination_color': 'yellow'}, ['black', 'red']),
    ('把红色盒子里的物体搬到桌面黄色区域',
     {'requested_count': None, 'color': 'any', 'route': 'box_table', 'destination_color': 'yellow'}, []),
    ('用右臂把左边蓝盒中的红色方块搬到桌面黄色区域',
     {'requested_count': None, 'kind': 'cube', 'color': 'red', 'position': 'any',
      'arm': 'B', 'route': 'box_table', 'destination_kind': 'marker', 'destination_color': 'yellow'}, []),
    ('桌面有三个红色障碍物，将盒内绿色和白色物体放到黄色区域',
     {'requested_count': None, 'color': 'any', 'route': 'box_table', 'selection_order': 'inventory'}, ['red']),
    ('用左臂把两个红色方块和三个绿色圆柱放进蓝色盒子',
     {'requested_count': 5, 'kind': 'any', 'color': 'any', 'arm': 'A',
      'route': 'table_box', 'selection_order': 'inventory'}, []),
    ('把红色和绿色物体放进白色盒子',
     {'requested_count': None, 'color': 'any', 'route': 'table_box', 'selection_order': 'inventory'}, []),
    ('先搬绿色物体，再搬红色物体，都放进白色盒子',
     {'requested_count': None, 'color': 'any', 'route': 'table_box', 'selection_order': 'explicit'}, []),
    ('把同一个绿色物体从红色盒子搬到桌面黄色区域，再搬进蓝色盒子',
     {'requested_count': 1, 'color': 'green', 'route': 'box_table_box', 'selection_order': 'inventory'}, []),
]


def membership_state(instruction, objects, route):
    progress = TaskProgress(instruction, route=route, scope='collection', selection_order='inventory')
    progress.observe(objects, 'synthetic-semantics-fixture')
    return {'instruction': instruction, 'route': route, 'task': 'grasp_place',
            'task_progress': progress.snapshot(), 'observation': {'objects': objects},
            'fixture_notice': 'Synthetic measured-evidence fixture for semantic testing, not a robot observation.'}


def box_object(identifier, color, bin_id):
    return {'id': identifier, 'kind': 'cube', 'color': color, 'position': [.4, 0., .65],
            'measured_relationships': [{'relation': 'above_observed_bin_floor', 'object_id': bin_id,
                                       'visible_xy_fraction': .99, 'source': 'synthetic_test_fixture'}]}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=Path(__file__).resolve().parents[1] / '.env')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first', type=int, default=0)
    args = parser.parse_args()
    load_dotenv(args.env_file)
    model = TypeSafeModel(os.getenv('JEV_MODEL', 'jev-1.13.0'), provider=TypeSafeProvider(
        api_key=os.environ['TYPESAFE_API_KEY'], base_url=os.getenv('TYPESAFE_BASE_URL', 'https://api.typesafe.ai')))
    records, results = [], []
    decisions = Decisions(model, lambda kind, title, data: records.append(data),
                          minimum_confidence=float(os.getenv('JEV_MINIMUM_CONFIDENCE', '.6')))

    async def check(label, call, expected, project):
        start = len(records)
        row = {'case': label, 'expected': expected}
        try:
            answer = await asyncio.wait_for(call, timeout=60)
            actual = project(answer)
            row.update(actual=actual, passed=actual == expected)
        except (DecisionError, TimeoutError) as error:
            row.update(error=str(error), passed=False)
        row['decisions'] = records[start:]
        results.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != 'decisions'},
                         ensure_ascii=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    cases = BIND_CASES[:args.first] if args.first else BIND_CASES
    for instruction, expected, environment in cases:
        expected = {**expected, 'environment_colors': environment}
        await check(instruction, decisions.bind_request(instruction), expected,
                    lambda answer, keys=tuple(expected): {
                        key: ([item['color'] for item in answer['environment_queries']]
                              if key == 'environment_colors' else answer[key]) for key in keys})
    if not args.first:
        bins = [{'id': 'source-blue', 'kind': 'bin', 'color': 'blue', 'position': [.4, .2, .6]},
                {'id': 'source-white', 'kind': 'bin', 'color': 'white', 'position': [.4, -.2, .6]}]
        objects = [box_object('red-blue', 'red', 'source-blue'),
                   box_object('red-white', 'red', 'source-white'),
                   box_object('green-blue', 'green', 'source-blue'), *bins]
        state = membership_state('把蓝色盒子里的红色物体放到桌面黄色区域', objects, 'box_table')
        await check('source container and target color are separate', decisions.assess_task(state),
                    {'red-blue': 'required', 'red-white': 'excluded', 'green-blue': 'excluded'},
                    lambda answer: answer['membership'])
        state = membership_state('桌面有红色物体和黑色物体，把盒子里的绿色物体放到黄色区域',
                                 [objects[2], {'id': 'red-table', 'kind': 'cube', 'color': 'red',
                                  'position': [.2, .3, .65]},
                                  {'id': 'black-table', 'kind': 'cylinder', 'color': 'black',
                                   'position': [.2, -.3, .65]}, *bins], 'box_table')
        await check('environment colors are not task members', decisions.assess_task(state),
                    {'green-blue': 'required', 'red-table': 'excluded', 'black-table': 'excluded'},
                    lambda answer: answer['membership'])
        instruction = '把两个红色物体和一个绿色物体放进白盒'
        objects = [{'id': name, 'kind': 'cube', 'color': color, 'position': [distance, 0., .6]}
                   for name, color, distance in [('red-near', 'red', .3), ('red-middle', 'red', .4),
                                                  ('red-far', 'red', .5), ('green-far', 'green', .8)]]
        progress = TaskProgress(instruction, route='table_box', scope='collection',
                                requested_count=3, selection_order='inventory')
        progress.observe(objects, 'synthetic-group-quota-fixture')
        state = {'instruction': instruction, 'route': 'table_box', 'task': 'grasp_place',
                 'task_progress': progress.snapshot(), 'observation': {'objects': objects}}

        def selected_group_counts(answer):
            progress.assess_membership(answer['membership'])
            progress.assign_display_ids(state['observation'])
            members = progress.snapshot()['objects']
            return {color: sum(obj['color'] == color for obj in members) for color in ('red', 'green')}

        await check('separate group quotas survive inventory limiting', decisions.assess_task(state),
                    {'red': 2, 'green': 1}, selected_group_counts)
        destinations = [{'id': 'white-destination', 'kind': 'bin', 'color': 'white'},
                        {'id': 'blue-destination', 'kind': 'bin', 'color': 'blue'}]
        for color, expected in [('green', 'white-destination'), ('red', 'blue-destination')]:
            state = {'instruction': '把红色物体放进蓝色盒子，绿色物体放进白色盒子',
                     'route': 'table_box', 'task_progress': {'stage_index': 0,
                     'current': {'id': 'current-' + color, 'kind': 'cube', 'color': color}}}
            await check('destination binding for ' + color, decisions.choose_destination(state, destinations),
                        expected, lambda answer: answer)
        state = {'instruction': '把同一个绿色物体从蓝盒搬到黄色桌面区，再放进白盒',
                 'route': 'box_table_box', 'task_progress': {'stage_index': 1,
                 'current': {'id': 'same-green', 'color': 'green', 'initial_source': {
                     'measured_relationships': [{'relation': 'above_observed_bin_floor',
                     'object_id': 'blue-destination', 'visible_xy_fraction': .99}]}}}}
        await check('second stage keeps the object and excludes the source box',
                    decisions.choose_destination(state, destinations + [{'id': 'yellow-area', 'kind': 'marker'}]),
                    'white-destination', lambda answer: answer)
    if not all(row['passed'] for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())

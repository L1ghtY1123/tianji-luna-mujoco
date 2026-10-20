"""Actual JEV/RGBD/GPU route acceptance; physical truth is recorded only for scoring."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from rgbd_pick_place_smoke import ROOT, RecordedAlgorithms, ScoredSimulation

from jev_mujoco.decisions import Decisions
from jev_mujoco.planning_settings import PlanningSettings
from jev_mujoco.runner import Runner, public_summary


def fixture(route, count, identical):
    objects = [{'id': 'marker', 'kind': 'marker', 'color': 'yellow',
                'position': [.38, 0., .6001], 'size': [.20, .20, .0002]}]
    objects.append({'id': 'left-bin', 'kind': 'bin', 'color': 'blue',
                    'position': [.45, .23, .65], 'size': [.24, .22, .10], 'wall_thickness_m': .01})
    if route == 'box_table_box':
        objects.append({'id': 'right-bin', 'kind': 'bin', 'color': 'blue',
                        'position': [.45, -.23, .65], 'size': [.24, .22, .10], 'wall_thickness_m': .01})
    for index in range(count):
        x = .43 + index * .07 if count <= 2 else .405 + (index % 2) * .09
        y = (.23 if route != 'table_box' else 0.) + (0. if count <= 2 else -.045 + (index // 2) * .09)
        objects.append({'id': f'source-{index}', 'kind': 'cube',
                        'color': 'red' if identical or index == 0 else 'green',
                        'position': [x, y,
                                     .635 if route != 'table_box' else .625], 'size': [.04] * 3})
    targets = '红色方块' if count == 1 else '所有方块'
    instruction = {'box_table': f'把蓝色盒子里的{targets}放到黄色桌面区域',
                   'table_box': f'把桌面上的{targets}放进蓝色盒子',
                   'box_table_box': f'把左边蓝色盒子里的{targets}逐个先放到黄色桌面区域，再把同一物体放进右边蓝色盒子'}[route]
    return objects, instruction


async def main(args):
    load_dotenv(ROOT / '.env', override=False)
    args.output.mkdir(parents=True, exist_ok=True)
    model = TypeSafeModel(os.environ.get('JEV_MODEL', 'jev-1.13.0'), provider=TypeSafeProvider(
        api_key=os.environ['TYPESAFE_API_KEY'], base_url=os.environ.get('TYPESAFE_BASE_URL', 'https://api.typesafe.ai')))
    simulation = ScoredSimulation()
    simulation.start()
    runner = None
    try:
        objects, instruction = fixture(args.route, args.count, args.identical)
        simulation.configure({'objects': objects})
        await asyncio.to_thread(simulation.confirm)
        before = simulation.snapshot()
        algorithms = RecordedAlgorithms(args.output)
        await asyncio.to_thread(algorithms.clear_carrying)
        settings = PlanningSettings.model_validate(json.loads(args.settings_file.read_text(encoding='utf8'))
                    ['planning_settings']['settings']) if args.settings_file else PlanningSettings()
        runner = Runner(simulation, algorithms, lambda record: Decisions(model, record), args.output / 'runs',
                        planning_settings=settings, visual_assistance=None)
        await runner.start(instruction)
        try:
            await asyncio.wait_for(asyncio.shield(runner.worker), args.timeout_s)
        except TimeoutError:
            await asyncio.to_thread(simulation.reset)
            await runner.cancel()
        (args.output / 'result.json').write_text(json.dumps({'instruction': instruction, 'fixture_inputs': objects,
            'run': public_summary(runner.state), 'events': runner.events}, ensure_ascii=False, indent=2), encoding='utf8')
        (args.output / 'task-progress.json').write_text(json.dumps(runner.task_progress.snapshot(),
            ensure_ascii=False, indent=2), encoding='utf8')
        (args.output / 'independent-oracle.json').write_text(json.dumps({'before': before,
            'after': simulation.snapshot(), 'gripper_commands': simulation.gripper_oracle}, ensure_ascii=False,
            indent=2), encoding='utf8')
        print(json.dumps({'route': args.route, 'status': runner.state['status'], 'error': runner.state['error']},
                         ensure_ascii=False), flush=True)
        return int(runner.state['status'] != 'succeeded')
    finally:
        if runner and runner.active:
            await asyncio.to_thread(simulation.reset)
            await runner.cancel()
        await asyncio.to_thread(simulation.close)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=['box_table', 'table_box', 'box_table_box'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument('--identical', action='store_true')
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--settings-file', type=Path)
    raise SystemExit(asyncio.run(main(parser.parse_args())))

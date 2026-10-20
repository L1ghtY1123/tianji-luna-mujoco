"""Isolated acceptance: real JEV and sensors; scene truth is scorer-only."""
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
from jev_mujoco.visual_assistance import VisualAssistance


def fixture(case, *, bin_source_xy_m=(.43, .15), bin_source_size_m=(.18, .18, .10),
            marker_xy_m=(.66, .15)):
    if case == 'bin_to_marker':
        return [
            {'id': 'source', 'kind': 'cube', 'color': 'red', 'position': [*bin_source_xy_m, .635], 'size': [.05]*3},
            {'id': 'bin', 'kind': 'bin', 'color': 'blue', 'position': [*bin_source_xy_m, .6 + bin_source_size_m[2]/2],
             'size': list(bin_source_size_m), 'wall_thickness_m': .01},
            {'id': 'marker', 'kind': 'marker', 'color': 'yellow', 'position': [*marker_xy_m, .6001],
             'size': [.20, .18, .0002]}], '用左臂把蓝色盒子里的红色方块放到黄色标记区'
    return [
        {'id': 'source-red', 'kind': 'cube', 'color': 'red', 'position': [.455, .009, .625], 'size': [.05]*3},
        {'id': 'source-green', 'kind': 'cube', 'color': 'green', 'position': [.56, -.06, .625], 'size': [.05]*3},
        {'id': 'bin', 'kind': 'bin', 'color': 'blue', 'position': [.508, .324, .65],
         'size': [.22, .24, .10], 'wall_thickness_m': .01}], (
             '用左臂把桌面上所有物体放进蓝色盒子里' if case == 'all' else
             '用左臂依次把桌上的红色方块和绿色方块放进蓝色盒子里' if case == 'collection' else
             '用左臂把桌上的红色方块放进蓝色盒子，绿色方块不要动')


async def main(case, output, timeout, *, settings_file=None, offset_xy=(0., 0.), **layout):
    load_dotenv(ROOT / '.env', override=False)
    output.mkdir(parents=True, exist_ok=True)
    model = TypeSafeModel(os.environ.get('JEV_MODEL', 'jev-1.13.0'), provider=TypeSafeProvider(
        api_key=os.environ['TYPESAFE_API_KEY'], base_url=os.environ.get('TYPESAFE_BASE_URL', 'https://api.typesafe.ai')))
    simulation = ScoredSimulation()
    simulation.start()
    runner = None
    try:
        objects, instruction = fixture(case, **layout)
        for obj in objects:
            obj['position'][0] += offset_xy[0]
            obj['position'][1] += offset_xy[1]
        simulation.configure({'objects': objects})
        await asyncio.to_thread(simulation.confirm)
        before = simulation.snapshot()
        algorithms = RecordedAlgorithms(output)
        await asyncio.to_thread(algorithms.clear_carrying)
        settings = (PlanningSettings.model_validate(json.loads(settings_file.read_text(encoding='utf8'))
                    ['planning_settings']['settings']) if settings_file else PlanningSettings())
        runner = Runner(simulation, algorithms, lambda record: Decisions(model, record), output / 'runs',
                        planning_settings=settings, visual_assistance=VisualAssistance.from_environment())
        await runner.start(instruction)
        try:
            await asyncio.wait_for(asyncio.shield(runner.worker), timeout)
        except TimeoutError:
            await asyncio.to_thread(simulation.reset)
            await runner.cancel()
        result = {'case': case, 'fixture_inputs': objects, 'instruction': instruction, 'run': public_summary(runner.state),
                  'events': runner.events}
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
        (output / 'independent-oracle.json').write_text(json.dumps(
            {'before': before, 'after': simulation.snapshot(), 'gripper_commands': simulation.gripper_oracle},
            ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({'status': runner.state['status'], 'error': runner.state['error'],
                          'case': case, 'output': str(output)}, ensure_ascii=False), flush=True)
        return int(runner.state['status'] != 'succeeded')
    finally:
        if runner and runner.active:
            await asyncio.to_thread(simulation.reset)
            await runner.cancel()
        await asyncio.to_thread(simulation.close)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['collection', 'all', 'subset', 'bin_to_marker'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=420)
    parser.add_argument('--settings-file', type=Path)
    parser.add_argument('--offset-xy', type=float, nargs=2, default=(0., 0.))
    parser.add_argument('--bin-source-xy-m', type=float, nargs=2, default=(.43, .15))
    parser.add_argument('--bin-source-size-m', type=float, nargs=3, default=(.18, .18, .10))
    parser.add_argument('--marker-xy-m', type=float, nargs=2, default=(.66, .15))
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(
        args.case, args.output, args.timeout_s, settings_file=args.settings_file, offset_xy=args.offset_xy,
        bin_source_xy_m=args.bin_source_xy_m,
        bin_source_size_m=args.bin_source_size_m, marker_xy_m=args.marker_xy_m)))

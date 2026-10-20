"""Live JEV request/selection replay; no perception, planning, or robot commands.

Pass an isolated-run result.json containing the recorded target decision. Each
case reuses its confirmed membership and measured objects; completion below is
only a local ledger transition to exercise the next target choice.
"""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

from jev_mujoco.decisions import DecisionError, Decisions
from jev_mujoco.task_progress import TaskProgress

CASES = [
    ('抓取盒子中的红色和绿色物体，放在桌面的黄色区域', 'inventory', [1, 2]),
    ('抓取盒子中的绿色和红色物体，放在桌面的黄色区域', 'inventory', [1, 2]),
    ('先搬盒子中的红色物体，再搬绿色物体，都放在桌面的黄色区域', 'explicit', [2, 1]),
    ('先搬盒子中的绿色物体，再搬红色物体，都放在桌面的黄色区域', 'explicit', [1, 2]),
    ('先搬盒子中远处的物体，再搬近处的物体，都放在桌面的黄色区域', 'explicit', [2, 1]),
]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first', type=int, default=0)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    replay = json.loads(args.replay.read_text(encoding='utf-8'))
    original = next(event['data']['state'] for event in replay['events']
                    if event['kind'] == 'jev' and event['title'] == '选择目标物体')
    model = TypeSafeModel(os.getenv('JEV_MODEL', 'jev-1.13.0'), provider=TypeSafeProvider(
        api_key=os.environ['TYPESAFE_API_KEY'], base_url=os.getenv('TYPESAFE_BASE_URL', 'https://api.typesafe.ai')))
    records, results = [], []
    decisions = Decisions(model, lambda kind, title, data: records.append(data),
                          minimum_confidence=float(os.getenv('JEV_MINIMUM_CONFIDENCE', '.6')))
    cases = CASES[:args.first] if args.first else CASES
    for instruction, order, expected_ids in cases:
        start_record = len(records)
        row = {'instruction': instruction, 'expected_order': order, 'expected_display_ids': expected_ids}
        try:
            request = await asyncio.wait_for(decisions.bind_request(instruction), timeout=60)
            objects = copy.deepcopy(original['observed_objects'])
            progress = TaskProgress(instruction, scope=request['scope'], route=request['route'],
                                    requested_count=request['requested_count'],
                                    selection_order=request['selection_order'])
            progress.observe(objects, original['task_progress']['observation_id'])
            progress.assess_membership({obj['id']: 'required' for obj in objects})
            progress.assign_display_ids({'objects': objects})
            state = {**copy.deepcopy(original), 'instruction': instruction,
                     'attributes': {key: request[key] for key in ('kind', 'color', 'position')}}
            selected = []
            for index in range(len(expected_ids)):
                state['task_progress'] = progress.snapshot()
                target = await asyncio.wait_for(decisions.choose_target(state, objects), timeout=60)
                obj = next((obj for obj in objects if obj['id'] == target), None)
                if obj is None:
                    selected.append(target)
                    break
                selected.append(obj['display_id'])
                progress.select(obj, 'replay-yellow-marker')
                progress.complete(target, observation_id=f'local-ledger-only-{index}')
            row.update(actual_order=request['selection_order'], requested_count=request['requested_count'],
                       selected_display_ids=selected,
                       passed=request['selection_order'] == order and request['requested_count'] is None
                       and selected == expected_ids)
        except (DecisionError, ValueError, TimeoutError) as error:
            row.update(error=str(error), passed=False)
        row['decisions'] = records[start_record:]
        results.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != 'decisions'},
                         ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    if not all(row['passed'] for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())

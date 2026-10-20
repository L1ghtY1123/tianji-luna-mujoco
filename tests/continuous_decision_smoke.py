"""Live JEV recovery decisions replayed from measured acceptance evidence."""
import asyncio
import copy
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

from jev_mujoco.decisions import Decisions


async def main():
    load_dotenv('/opt/jev/.env')
    logs = list(Path('/acceptance/explicit-left-v6/runs').glob('*.jsonl'))
    events = [json.loads(line) for line in logs[0].read_text().splitlines()]
    measured = next(event['data']['state'] for event in events if event['kind'] == 'jev'
                    and (event['data'].get('state', {}).get('failure') or {}).get('code') == 'recovery_retry_budget_exhausted')
    model = TypeSafeModel(os.environ.get('JEV_MODEL', 'jev-1.13.0'), provider=TypeSafeProvider(
        api_key=os.environ['TYPESAFE_API_KEY'], base_url=os.environ.get('TYPESAFE_BASE_URL', 'https://api.typesafe.ai')))
    results = []
    for name in ('held_exhausted', 'empty_collection_exhausted', 'held_first_failure'):
        state = copy.deepcopy(measured)
        if name == 'empty_collection_exhausted':
            state['facts'].update(grasp_held=False, attached=False, collection_task=True)
            state['verification'] = {'status': 'empty', 'position': {'status': 'empty'}}
            state['instruction'] = '把桌面所有物体放入盒中，难以完成的先跳过，继续其他物体，最后报告未完成项'
            state['task_progress']['scope'] = 'collection'
            state['simulation'] = {}
        elif name == 'held_first_failure':
            state['failure'].update(code='place_no_valid_candidate', attempt_count=1)
            state['task_progress']['attempts'] = {state['target_id']: 1}
        records = []
        choice = await Decisions(model, lambda *args, records=records: records.append(args[2])).choose_recovery(state)
        results.append({'case': name, 'choice': choice, 'record': records[-1]})
        print(json.dumps({'case': name, 'choice': choice}), flush=True)
    Path('/acceptance/recovery-live.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())

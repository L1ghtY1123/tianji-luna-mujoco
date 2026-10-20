"""Explicit live JEV smoke, not collected by pytest; never commands the robot."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

from jev_mujoco.decisions import DecisionError, Decisions

CASES = [
    ('抓取盒子中绿色物体放到桌面黄色区域',
     {'requested_count': None, 'selection_order': 'inventory', 'route': 'box_table',
      'color': 'green', 'destination_color': 'yellow'}, []),
    ('把盒子里的红色物体放到桌面指定区域',
     {'requested_count': None, 'selection_order': 'inventory', 'route': 'box_table'}, []),
    ('抓取盒子中一个绿色物体放到桌面黄色区域',
     {'requested_count': 1, 'selection_order': 'inventory', 'route': 'box_table'}, []),
    ('桌面有一个红色和一个黑色物体，把盒子里的一个物体放到桌面黄色区域',
     {'requested_count': 1, 'selection_order': 'inventory', 'route': 'box_table'}, ['black', 'red']),
    ('用左臂把两个红色和三个绿色物体放进蓝盒',
     {'requested_count': 5, 'selection_order': 'inventory', 'arm': 'A'}, []),
    ('把桌面上所有物体放进蓝盒',
     {'requested_count': None, 'selection_order': 'inventory', 'route': 'table_box'}, []),
    ('把盒子里的物体放到桌面黄色区域',
     {'requested_count': None, 'selection_order': 'inventory', 'route': 'box_table'}, []),
    ('把蓝色盒子里的所有方块放到黄色桌面区域',
     {'requested_count': None, 'selection_order': 'inventory', 'route': 'box_table'}, []),
    ('桌面上有三个红色障碍物，把盒子中绿色物体放到桌面黄色区域',
     {'requested_count': None, 'selection_order': 'inventory', 'route': 'box_table'}, ['red']),
    ('把同一个绿色物体从一个盒子搬到桌面，再搬进另一个盒子',
     {'requested_count': 1, 'selection_order': 'inventory', 'route': 'box_table_box'}, []),
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--first', type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')
    model = TypeSafeModel(os.getenv('JEV_MODEL', 'jev-1.13.0'), provider=TypeSafeProvider(
        api_key=os.environ['TYPESAFE_API_KEY'], base_url=os.getenv('TYPESAFE_BASE_URL', 'https://api.typesafe.ai')))
    records = []
    decisions = Decisions(model, lambda kind, title, data: records.append(data),
                          minimum_confidence=float(os.getenv('JEV_MINIMUM_CONFIDENCE', '.6')))
    results = []
    cases = CASES[:args.first] if args.first else CASES
    for instruction, expected, environment in cases:
        try:
            result = await asyncio.wait_for(decisions.bind_request(instruction), timeout=60)
        except DecisionError as error:
            row = {'instruction': instruction, 'error': str(error), 'passed': False,
                   'output': records[-1].get('output'),
                   'confidence': records[-1].get('provider_details', {}).get('confidence'),
                   'probabilities': records[-1].get('provider_details', {}).get('probabilities', {}).get('quantity')}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        actual = {key: result[key] for key in expected}
        actual_environment = [item['color'] for item in result['environment_queries']]
        passed = actual == expected and actual_environment == environment
        row = {'instruction': instruction, 'expected': expected, 'actual': actual,
               'environment_colors': actual_environment, 'passed': passed,
               'confidence': records[-1].get('provider_details', {}).get('confidence')}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    path = args.output or root / 'data/request-scope-20261016/request-scope-jev.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    if not all(item['passed'] for item in results):
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())

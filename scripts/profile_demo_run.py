"""Report saved run wall times; nested algorithm/stage totals are not additive."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def grouped(events, kind, field='duration_s'):
    groups = defaultdict(list)
    for event in events:
        if event['kind'] == kind and isinstance(event.get('data', {}).get(field), (int, float)):
            groups[event['title']].append(event['data'][field])
    return sorted([{'node': name, 'calls': len(values), 'total_s': round(sum(values), 3),
                        'maximum_s': round(max(values), 3)} for name, values in groups.items()],
                  key=lambda row: -row['total_s'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('state', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding='utf-8'))
    run, events = state['run'], state['events']
    duration = run.get('metrics', {}).get('run_duration_s')
    report = {'run_id': run['run_id'], 'status': run['status'], 'error': run.get('error'),
        'total_wall_s': duration, 'goal_wall_s': 180,
        'within_goal': run['status'] == 'succeeded' and duration is not None and duration <= 180,
        'timing_semantics': 'Skill rows are sequential. Algorithm and execution rows are nested detail; do not add tables.',
        'settings': {k: run['planning_settings'][k] for k in
                  ('time_scale', 'trajectory_collision_clearance_m', 'plan_timeout_s')},
        'skills': grouped(events, 'skill_complete'), 'algorithms': grouped(events, 'algorithm_complete'),
        'execution': grouped(events, 'execution_stage_complete'),
        'vision_api': grouped(events, 'visual_decisions_call', 'elapsed_s'),
        'perception': [event['data']['result']['perception_timings_s'] for event in events
                    if event['kind'] == 'algorithm_complete' and event['title'] == 'observe'
                    and 'perception_timings_s' in event['data'].get('result', {})]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

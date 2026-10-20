"""Offline physical scoring. This module is never imported by robot execution."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull


def inside_polygon(xy, polygon, tolerance=.002):
    equations = ConvexHull(np.asarray(polygon)).equations
    return bool(np.all(equations[:, :2] @ xy + equations[:, 2] <= tolerance))


def score(oracle, progress, expected_count, route):
    failures, checks = [], []
    records = progress.get('objects', [])
    initial = [obj for obj in oracle['before']['objects'] if obj['id'].startswith('source-')]
    final = {obj['id']: obj for obj in oracle['after']['objects']}
    if len(records) != expected_count or len(initial) != expected_count:
        return {'passed': False, 'failures': ['inventory_count_mismatch'], 'checks': []}
    cost = np.array([[np.linalg.norm(np.asarray(record['initial_source']['position']) - obj['position'])
                      for obj in initial] for record in records])
    rows, columns = linear_sum_assignment(cost)
    matched = {}
    for row, column in zip(rows, columns):
        if cost[row, column] > .04:
            failures.append('initial_identity_not_supported:' + records[row]['id'])
        if len(initial) > 1 and np.sort(cost[row])[1] - cost[row, column] < .01:
            failures.append('initial_identity_ambiguous:' + records[row]['id'])
        matched[records[row]['id']] = initial[column]['id']
    layouts = {(r['stage_index'], r['destination_id']): r['layout']
               for r in progress.get('placement_regions', []) if 'layout' in r}
    expected_stages = 2 if route == 'box_table_box' else 1
    for record in records:
        stages = record.get('completed_stages', [])
        if record['status'] != 'completed' or len(stages) != expected_stages:
            failures.append('incomplete_route:' + record['id'])
            continue
        last = stages[-1]
        layout = layouts.get((last['stage_index'], last['destination_id']))
        if layout is None:
            failures.append('missing_frozen_layout:' + record['id'])
            continue
        cell = next(cell for cell in layout['cells'] if cell['slot_id'] == record['display_id'])
        actual = final[matched[record['id']]]
        transform = np.asarray(actual['T_world_object'])
        half = np.asarray(actual['size']) / 2
        corners = np.array([[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1])
                            for z in (-half[2], half[2])]) @ transform[:3, :3].T + transform[:3, 3]
        in_slot = inside_polygon(np.asarray(actual['position'][:2]), cell['polygon_world_xy_m'])
        in_region = all(inside_polygon(corner[:2], layout['region_polygon_world_xy_m']) for corner in corners)
        gap = float(corners[:, 2].min() - layout['floor_z_m'])
        settled = -.003 <= gap <= .005
        checks.append({'instance_id': record['id'], 'physical_id': actual['id'], 'slot_id': record['display_id'],
            'stage_arms': [stage['arm'] for stage in stages], 'center_in_assigned_slot': in_slot,
            'footprint_in_region': in_region, 'physical_bottom_gap_m': gap, 'settled_on_floor': settled})
        if not settled:
            failures.append('physical_object_not_settled_on_floor:' + record['id'])
    return {'passed': not failures, 'failures': failures, 'checks': checks,
            'xy_checks_are_diagnostic_only': True,
            'scope': 'independent route completion and final settling; TCP slot is checked by release endpoint audit'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--count', required=True, type=int)
    parser.add_argument('--route', required=True)
    args = parser.parse_args()
    result = score(json.loads((args.directory / 'independent-oracle.json').read_text()),
                   json.loads((args.directory / 'task-progress.json').read_text()), args.count, args.route)
    (args.directory / 'physical-score.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    raise SystemExit(int(not result['passed']))

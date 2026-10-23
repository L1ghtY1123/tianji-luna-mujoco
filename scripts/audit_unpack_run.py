"""Read-only simulation acceptance; never imported by the controller or perception."""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from luna_mujoco.cad_assets import CAD_ROOT, cad_asset


def points_inside_polygon(points, polygon, tolerance=.002):
    """Closed-section containment, including rounded corners; QA tolerance only."""
    points, polygon = np.asarray(points), np.asarray(polygon)
    inside = np.zeros(len(points), dtype=bool)
    distance_squared = np.full(len(points), np.inf)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        length_squared = float(edge @ edge)
        if length_squared == 0:
            continue
        fraction = np.clip((points - start) @ edge / length_squared, 0, 1)
        closest = start + fraction[:, None] * edge
        distance_squared = np.minimum(distance_squared, np.sum((points - closest) ** 2, axis=1))
        if edge[1] != 0:
            crossing = ((start[1] > points[:, 1]) != (end[1] > points[:, 1]))
            x_intersection = start[0] + (points[:, 1] - start[1]) * edge[0] / edge[1]
            inside ^= crossing & (points[:, 0] < x_intersection)
    return inside | (distance_squared <= tolerance ** 2)


@lru_cache(maxsize=3)
def cad_vertices(asset_id):
    return np.asarray(trimesh.load_mesh(CAD_ROOT / cad_asset(asset_id)['visual']['path']).vertices)


def cad_objects(state, config):
    box = next(obj for obj in state['simulation']['objects'] if obj['id'] == config['right_bin']['id'])
    shape = cad_asset(box['asset_id'])
    box_inverse = np.linalg.inv(np.asarray(box['T_world_object']))
    objects = []
    for obj in state['simulation']['objects']:
        if obj.get('asset_id') not in {'pink_wrapped', 'black_core'}:
            continue
        transform = np.asarray(obj['T_world_object'])
        relative = box_inverse @ transform
        vertices = cad_vertices(obj['asset_id']) @ relative[:3, :3].T + relative[:3, 3]
        inside = (obj['asset_id'] == 'black_core'
            and np.all(points_inside_polygon(vertices[:, :2], shape['cavity_bottom_section']['polygon_xy_m']))
            and vertices[:, 2].min() >= shape['cavity_floor_z_m'] - .002
            and vertices[:, 2].max() <= shape['size_m'][2] / 2 + .002)
        objects.append({'physical_id': obj['id'], 'asset_id': obj['asset_id'], 'color': obj['color'],
            'size_m': obj['size'], 'center_world_m': transform[:3, 3].tolist(),
            'inside_right_bin': bool(inside), 'containment_basis': 'actual_CAD_vertices_in_conservative_bottom_section',
            'cad_source_sha256': cad_asset(obj['asset_id'])['source']['sha256']})
    return objects


def primitive_objects(state, config):
    box = config['right_bin']
    center, half = np.asarray(box['position']), np.asarray(box['size']) / 2
    inner = half[:2] - box['wall_thickness_m']
    floor, top = center[2] - half[2] + box['wall_thickness_m'], center[2] + half[2]
    objects = []
    for obj in state['simulation']['objects']:
        if obj.get('color') not in {'pink', 'black'}:
            continue
        transform = np.asarray(obj['T_world_object'])
        extent = np.abs(transform[:3, :3]) @ (np.asarray(obj['size']) / 2)
        lower, upper = transform[:3, 3] - extent, transform[:3, 3] + extent
        inside = (obj['color'] == 'black' and np.all(lower[:2] >= center[:2] - inner - .002)
                  and np.all(upper[:2] <= center[:2] + inner + .002)
                  and lower[2] >= floor - .002 and upper[2] <= top + .002)
        objects.append({'physical_id': obj['id'], 'color': obj['color'], 'size_m': obj['size'],
            'center_world_m': transform[:3, 3].tolist(), 'inside_right_bin': bool(inside)})
    return objects


def completed_processes(ledger):
    processes = ledger['process_records']
    identifiers = ledger.get('completed_process_ids', [])
    records = {row['id']: row for row in ledger.get('objects', [])}
    inputs, outputs = set(), set()
    valid = bool(identifiers) and len(set(identifiers)) == len(identifiers)
    for identifier in identifiers:
        process = processes.get(identifier, {})
        input_id, output_id = process.get('input_entity_id'), process.get('output_entity_id')
        source, product = records.get(input_id, {}), records.get(output_id, {})
        creation = process.get('creation_evidence', {})
        valid = valid and bool(input_id and output_id and input_id != output_id
            and input_id not in inputs and output_id not in outputs
            and process.get('status') == 'completed' and process.get('phase') == 'completed'
            and source.get('status') == 'consumed' and product.get('status') == 'completed'
            and creation.get('observation_id') and creation.get('request_hash'))
        for record, stage_index in ((source, 0), (product, 1)):
            stages = record.get('completed_stages', [])
            valid = valid and len(stages) == 1 and stages[0].get('stage_index') == stage_index
            valid = valid and bool(stages[0].get('observation_id'))
        inputs.add(input_id)
        outputs.add(output_id)
    return len(identifiers), bool(valid and not inputs.intersection(outputs))


def audit(state, expected, config):
    is_cad = config['right_bin'].get('asset_id') == 'blue_box'
    objects = cad_objects(state, config) if is_cad else primitive_objects(state, config)
    run = state['run']
    ledger = run.get('task_progress', {})
    completed = ledger.get('completed', [])
    ordered = all([stage['stage_index'] for stage in item.get('completed_stages', [])] == [0, 1]
                  and item.get('lifecycle', {}).get('phase') == 'completed' for item in completed)
    completed_count = len(completed)
    if 'process_records' in ledger:
        completed_count, ordered = completed_processes(ledger)
    events = state.get('events', [])
    mapping = [event for event in events if event['kind'] == 'skill_start' and event['title'] == 'build_scene']
    result = {'run_id': run.get('run_id'), 'runtime_status': run['status'],
        'evidence_source': 'independent MuJoCo end-state, not provided to model or runtime control',
        'expected_completed': expected, 'physical_black_inside_right_bin': sum(item['inside_right_bin'] for item in objects),
        'ledger_completed': completed_count, 'both_stages_confirmed_in_order': ordered,
        'responses_calls': run.get('metrics', {}).get('responses_call_count'),
        'build_scene_calls': len(mapping), 'holding': bool(run.get('holding')), 'objects': objects}
    result['passed'] = (run['status'] == 'succeeded' and not result['holding']
        and result['physical_black_inside_right_bin'] == completed_count == expected and ordered
        and result['responses_calls'] == 1 and len(mapping) == expected * 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('state', type=Path)
    parser.add_argument('--expected', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = audit(json.loads(args.state.read_text(encoding='utf-8')), args.expected,
                   json.loads((root / 'resources/demo-config.json').read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

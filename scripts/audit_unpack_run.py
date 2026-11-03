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


def verified_transfers(record):
    transfers = record.get('transfers', [])
    observations = [item.get('observation_id') for item in transfers]
    return bool(transfers and len(set(observations)) == len(observations)
        and all(item.get('observation_id') and item.get('destination_id') and item.get('arm') in {'A', 'B'}
                and item.get('transfer_index') == index for index, item in enumerate(transfers)))


def completed_inputs(ledger, destination_id):
    """Check observed input/output chains without prescribing a motion sequence."""
    identifiers = ledger.get('completed_input_ids', [])
    records = {row['id']: row for row in ledger.get('objects', [])}
    transformations = ledger.get('transformations', {})
    transformations = list(transformations.values()) if isinstance(transformations, dict) else transformations
    by_input, outputs = {}, set()
    valid = bool(identifiers and destination_id and len(set(identifiers)) == len(identifiers)
        and len(records) == len(ledger.get('objects', []))
        and set(identifiers).issubset(ledger.get('selected_input_ids', [])))
    for transformation in transformations:
        source, output = transformation.get('input_entity_id'), transformation.get('output_entity_id')
        if not source or not output or source in by_input or output in outputs or source == output:
            valid = False
        by_input[source] = transformation
        outputs.add(output)
    terminal_ids = set()
    for identifier in identifiers:
        current, visited, skills = identifier, set(), []
        valid = valid and verified_transfers(records.get(identifier, {}))
        while current in by_input:
            if current in visited:
                valid = False
                break
            visited.add(current)
            transformation = by_input[current]
            output = transformation.get('output_entity_id')
            product = records.get(output, {})
            creation, evidence = product.get('creation_evidence', {}), transformation.get('evidence', {})
            source = records.get(current, {})
            valid = valid and bool(source.get('status') == 'consumed' and source.get('output_entity_id') == output
                and product.get('input_entity_id') == current and transformation.get('skill')
                and product.get('processing_skill') == transformation['skill']
                and all(creation.get(key) and creation[key] == evidence.get(key)
                        for key in ('observation_id', 'request_hash', 'action_id')))
            skills.append(transformation.get('skill'))
            current = output
        product = records.get(current, {})
        transfers = product.get('transfers', [])
        last = transfers[-1] if transfers else {}
        valid = valid and bool(current != identifier and current not in terminal_ids
            and 'vla_unpack' in skills and product.get('status') == 'completed'
            and verified_transfers(product) and product.get('current_region') == destination_id
            and last.get('destination_ref', last.get('destination_id')) == destination_id)
        terminal_ids.add(current)
    return len(identifiers), bool(valid)


def audit(state, expected, config):
    is_cad = config['right_bin'].get('asset_id') == 'blue_box'
    objects = cad_objects(state, config) if is_cad else primitive_objects(state, config)
    run = state['run']
    ledger = run.get('task_progress', {})
    destination = run.get('task_spec', {}).get('destination_ref') or run.get('task_inventory', {}).get('destination_id')
    completed_count, verified_lineage = completed_inputs(ledger, destination)
    events = state.get('events', [])
    mapping = [event for event in events if event['kind'] == 'skill_start' and event['title'] == 'build_scene']
    result = {'run_id': run.get('run_id'), 'runtime_status': run['status'],
        'evidence_source': 'independent MuJoCo end-state, not provided to model or runtime control',
        'expected_completed': expected, 'physical_black_inside_right_bin': sum(item['inside_right_bin'] for item in objects),
        'ledger_completed': completed_count, 'verified_entity_lineage': verified_lineage,
        'responses_calls': run.get('metrics', {}).get('responses_call_count'),
        'build_scene_calls': len(mapping), 'holding': bool(run.get('holding')), 'objects': objects}
    result['passed'] = (run['status'] == 'succeeded' and not result['holding']
        and result['physical_black_inside_right_bin'] == completed_count == expected and verified_lineage
        and result['responses_calls'] == 1)
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

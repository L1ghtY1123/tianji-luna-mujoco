"""CPU-only containment certificate. Reads a candidate; never fits or edits it."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml


def certify_cells(cells, spheres, max_depth=20):
    """Prove triangles/tetrahedra are in the union using convexity of each ball.

    A cell is accepted only if all its vertices belong to the same sphere.
    Longest-edge bisection partitions any unresolved cell without removing area
    or volume. An uncovered vertex is a concrete counterexample; exhausting the
    depth budget is an inconclusive failure, never a coverage claim.
    """
    original_count = len(cells)
    vertex_count = cells.shape[1]
    pairs = np.array([(i, j) for i in range(vertex_count) for j in range(i + 1, vertex_count)])
    checked = 0
    for depth in range(max_depth + 1):
        checked += len(cells)
        distances = (np.linalg.norm(cells[:, :, None, :] - spheres[None, None, :, :3], axis=-1)
                     - spheres[None, None, :, 3])
        uncovered = distances.min(axis=2) > 1e-11
        if uncovered.any():
            cell, vertex = np.argwhere(uncovered)[0]
            return {'complete': False, 'uncovered_point_m': cells[cell, vertex].tolist(),
                    'gap_m': float(distances[cell, vertex].min()), 'depth': depth}
        covered = (distances.max(axis=1) <= 0).any(axis=1)
        cells = cells[~covered]
        if not len(cells):
            return {'complete': True, 'original_cells': original_count,
                    'checked_cells': checked, 'depth': depth, 'unproved_cells': 0}
        if depth == max_depth:
            return {'complete': False, 'unproved_cells': len(cells), 'depth': depth}
        edges = cells[:, pairs[:, 0], :] - cells[:, pairs[:, 1], :]
        split = pairs[(edges * edges).sum(axis=2).argmax(axis=1)]
        rows = np.arange(len(cells))
        midpoint = (cells[rows, split[:, 0]] + cells[rows, split[:, 1]]) / 2
        left, right = cells.copy(), cells.copy()
        left[rows, split[:, 0]] = midpoint
        right[rows, split[:, 1]] = midpoint
        cells = np.concatenate([left, right])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.candidate.read_bytes())['robot_cfg']['kinematics']
    spheres = {link: np.array([row['center'] + [row['radius']] for row in rows])
               for link, rows in config['collision_spheres'].items()}
    report = {'candidate_sha256': hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
              'total_spheres': sum(map(len, spheres.values())), 'symmetry': {}, 'links': {}}
    rules = json.loads(args.rules.read_text())['rules']
    for rule in rules:
        if not (rule['target_link'].startswith('grip') and 'Link4' in rule['target_link']):
            continue
        part = rule['parts'][0]
        source, target = spheres[part['source_link']], spheres[rule['target_link']]
        transform = np.array(part['transform'])
        error = source[:, :3] @ transform[:3, :3].T + transform[:3, 3] - target[:, :3]
        report['symmetry'][rule['target_link']] = {
            'center_error_m': float(abs(error).max()),
            'radius_error_m': float(abs(source[:, 3] - target[:, 3]).max())}
    for link in ['gripL_Link4_L', 'gripL_Link4_R']:
        mesh_root = args.project / 'assets/marvin_urdf/meshes'
        collision_path, visual_path = mesh_root / 'collision' / f'{link}.stl', mesh_root / 'visual' / f'{link}.obj'
        collision = trimesh.load_mesh(collision_path, process=True)
        visual = trimesh.load_mesh(visual_path, process=True)
        anchor = collision.center_mass
        assert collision.is_convex and collision.is_watertight and collision.is_winding_consistent
        assert np.max(np.sum((anchor - collision.triangles_center) * collision.face_normals, axis=1)) <= 0
        # For this closed convex mesh, joining each boundary triangle to an
        # interior point partitions the entire solid into tetrahedra.
        tetrahedra = np.concatenate([np.broadcast_to(anchor, (len(collision.faces), 1, 3)),
                                     collision.triangles], axis=1)
        report['links'][link] = {
            'collision_sha256': hashlib.sha256(collision_path.read_bytes()).hexdigest(),
            'visual_sha256': hashlib.sha256(visual_path.read_bytes()).hexdigest(),
            'count': len(spheres[link]), 'max_radius_m': float(spheres[link][:, 3].max()),
            'collision_surface': certify_cells(collision.triangles, spheres[link]),
            'collision_volume': certify_cells(tetrahedra, spheres[link]),
            'visual_surface': certify_cells(visual.triangles, spheres[link])}
    report['passed'] = (all(entry['center_error_m'] < 1e-12 and entry['radius_error_m'] < 1e-12
                            for entry in report['symmetry'].values())
                        and all(entry['max_radius_m'] <= .020
                                and all(entry[name]['complete'] for name in
                                        ('collision_surface', 'collision_volume', 'visual_surface'))
                                for entry in report['links'].values()))
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

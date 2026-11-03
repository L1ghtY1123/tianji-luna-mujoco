"""Stage a frame-preserving low-poly CAD publication without modifying live assets."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh

from build_cad_assets import bin_hulls, cavity_section, object_hulls


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def stage(project, library, candidates, output):
    project, library, output = project.resolve(), library.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(library) or library.is_relative_to(output):
        raise ValueError('stage_requires_new_directory_outside_library')
    output.mkdir(parents=True)
    resources = output / 'resources'
    for name in ('cad', 'cad_catalog', 'registration_meshes'):
        shutil.copytree(project / 'resources' / name, resources / name)
    catalog = json.loads((project / 'resources/cad_catalog.json').read_text(encoding='utf-8'))
    manifest = json.loads((resources / 'cad/manifest.json').read_text(encoding='utf-8'))
    staged_library = output / 'CAD'
    shutil.copytree(library, staged_library, ignore=shutil.ignore_patterns('__pycache__', '.git', 'runtime'))
    library_index = json.loads((staged_library / 'cad/index.json').read_text(encoding='utf-8'))
    by_source = {row['source']['sha256']: row for row in library_index['assets']}
    old_to_new = {}
    registration = {'version': 1, 'models': {}}
    audit = json.loads((candidates / 'audit.json').read_text(encoding='utf-8'))
    for asset_id, record in manifest['assets'].items():
        candidate = next(row for row in audit['candidates']
                         if row['asset_id'] == asset_id and row['requested_faces'] == 1000)
        path = candidates / candidate['path']
        if digest(path) != candidate['sha256'] or candidate['source_sha256'] != record['visual']['sha256']:
            raise ValueError('candidate_source_hash_mismatch')
        mesh = trimesh.load(path, force='mesh', process=True)
        if not 0 < len(mesh.faces) <= 1000 or not np.isfinite(mesh.vertices).all():
            raise ValueError('invalid_lowpoly_mesh')
        np.testing.assert_allclose(mesh.extents, record['size_m'], atol=1e-6, rtol=0)
        directory = resources / 'cad' / asset_id
        visual = resources / 'cad' / record['visual']['path']
        mesh.export(visual)
        # Read back the STL, so source and registration triangles derive from
        # precisely the published float32 geometry, not an in-memory variant.
        mesh = trimesh.load(visual, force='mesh', process=False)
        center = np.array(record.get('preprocessing', record['source'])['aabb_center_mm'])
        canonical_source = mesh.copy()
        canonical_source.vertices = mesh.vertices * 1000 + center
        if asset_id == 'pink_wrapped':
            canonical_source.export(resources / 'cad' / record['preprocessing']['trimmed_source']['path'])
            record['preprocessing']['trimmed_source']['sha256'] = digest(
                resources / 'cad' / record['preprocessing']['trimmed_source']['path'])
            record['preprocessing']['original_source_preserved'] = False
            source_mesh = trimesh.load(candidates / 'raw-pink-1000.ply', force='mesh')
        else:
            source_mesh = canonical_source
        original_source = deepcopy(record['source'])
        old_path = resources / 'cad' / original_source['path']
        source_path = old_path.with_suffix('.stl')
        source_mesh.export(source_path)
        if old_path != source_path:
            old_path.unlink()
        source_hash = digest(source_path)
        old_to_new[original_source['sha256']] = source_hash
        record['source'].update(path=source_path.relative_to(resources / 'cad').as_posix(),
            sha256=source_hash, original_sha256=original_source['sha256'],
            unit_evidence='Original source millimetres and axes retained during offline simplification')
        # Preserve the declared canonical origin and dimensions. Never refit
        # the CAD origin to an observation, a grasp or a simplified centroid.
        np.testing.assert_allclose(source_mesh.bounds, original_source['bounds_mm'], atol=.001, rtol=0)
        record['visual'].update(sha256=digest(visual), triangles=len(mesh.faces))
        record['reduction'] = {**candidate, 'original_source_sha256': original_source['sha256'],
            'coordinate_transform': np.eye(4).tolist(), 'grasp_positions_changed': False}
        for old in record['collision']:
            (resources / 'cad' / old['path']).unlink()
        welded = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
        pieces = bin_hulls(welded) if asset_id == 'blue_box' else object_hulls(welded)
        record['collision'] = []
        for index, piece in enumerate(pieces):
            if not piece.is_watertight or piece.volume <= 0:
                raise ValueError('invalid_lowpoly_collision_hull')
            path = directory / f'collision-{index:02d}.stl'
            piece.export(path)
            record['collision'].append({'path': path.relative_to(resources / 'cad').as_posix(),
                'sha256': digest(path), 'volume_m3': float(piece.volume), 'vertices': len(piece.vertices)})
        if asset_id == 'blue_box':
            record['cavity_bottom_section'] = cavity_section(welded, -.0675 + .000001)
            record['cavity_floor_plus_1mm_section'] = cavity_section(welded, -.0665)
        library_record = by_source[original_source['sha256']]
        old_library_path = staged_library / 'cad' / library_record['source']['path']
        library_path = old_library_path.with_suffix('.stl')
        shutil.copy2(source_path, library_path)
        if old_library_path != library_path:
            old_library_path.unlink()
        library_record['source'].update(path=library_path.relative_to(staged_library / 'cad').as_posix(),
            format='stl', sha256=source_hash, original_sha256=original_source['sha256'],
            face_count=len(source_mesh.faces),
            simplification='1000-face target; original axes, origin and units retained')
        if asset_id != 'pink_wrapped':
            row = next(row for row in catalog['assets'] if row['mesh_key'] == asset_id)
            row['source_sha256'] = source_hash
            if 'placement_support' in row:
                row['placement_support']['mesh_sha256'] = digest(visual)
            registration_mesh = mesh.copy()
            registration_mesh.apply_scale(1000.)
            registration_path = resources / 'registration_meshes' / (asset_id + '.ply')
            registration_mesh.export(registration_path)
            registration['models'][asset_id] = {
                'path': registration_path.name, 'source_sha256': source_hash,
                'original_mesh_sha256': 'pending_public_registry_registration',
                'mesh_sha256': digest(registration_path), 'units': 'mm',
                'frame': 'original_registry_canonical', 'frame_transform': np.eye(4).tolist(),
                'face_count': len(mesh.faces), 'source': 'published_lowpoly_surface'}

    manifest['reduction_generator'] = {'script': 'scripts/stage_lowpoly_cad.py',
        'script_sha256': digest(__file__), 'target_faces': 1000}
    write_json(resources / 'cad/manifest.json', manifest)
    write_json(resources / 'cad_catalog.json', catalog)
    write_json(resources / 'registration_meshes/manifest.json', registration)
    write_json(staged_library / 'cad/index.json', library_index)
    grasp_root = staged_library / 'grasps/THREADED_VISUAL_SAMPLE_MM/tianji'
    grasp_index = json.loads((grasp_root / 'index.json').read_text(encoding='utf-8'))
    poses_path = grasp_root / grasp_index['grasps']['path']
    poses = json.loads(poses_path.read_text(encoding='utf-8'))
    rows_digest = hashlib.sha256(json.dumps(poses['grasps'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    original_hash = poses['cad_sha256']
    poses['cad_sha256'] = old_to_new[original_hash]
    poses['geometry_update'] = {'original_cad_sha256': original_hash,
        'source_frame_transform': np.eye(4).tolist(), 'grasp_rows_sha256': rows_digest,
        'poses_changed': False, 'current_geometry_requires_runtime_collision_checks': True}
    write_json(poses_path, poses)
    grasp_index['cad_source']['sha256'] = poses['cad_sha256']
    grasp_index['grasps']['sha256'] = digest(poses_path)
    write_json(grasp_root / 'index.json', grasp_index)
    viewer = grasp_root / grasp_index['viewer']
    html = viewer.read_text(encoding='utf-8')
    start = html.index('const D=') + len('const D=')
    data, end = json.JSONDecoder().raw_decode(html[start:])
    data['registry'] = poses
    source = manifest['assets']['black_core']['source']
    black = trimesh.load(resources / 'cad' / source['path'], force='mesh')
    data['cad'] = {'v': (black.vertices * .001).tolist(), 'f': black.faces.tolist()}
    html = html[:start] + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + html[start + end:]
    viewer.write_text(html, encoding='utf-8')
    file_hashes = json.loads((grasp_root / 'file_hashes.json').read_text(encoding='utf-8'))
    for name in file_hashes:
        file_hashes[name] = digest(grasp_root / name)
    write_json(grasp_root / 'file_hashes.json', file_hashes)
    # A whole-file hash includes the deliberately changed CAD binding. Keep a
    # stronger invariant on the actual 512 grasp records instead.
    layout_test = staged_library / 'tests/test_repository_layout.py'
    text = layout_test.read_text(encoding='utf-8')
    old = "self.assertEqual(sha256(poses), '95208cf9a17a71a3c49b31bffd76fd521e963173c2fa4ae914693a04e8ca7c9c')"
    if old not in text:
        raise ValueError('unexpected_grasp_layout_test')
    text = text.replace(old, "self.assertEqual(hashlib.sha256(json.dumps(read_json(poses)['grasps'], "
        "sort_keys=True, separators=(',', ':')).encode()).hexdigest(), '" + rows_digest + "')")
    layout_test.write_text(text, encoding='utf-8')
    for document in staged_library.rglob('*.md'):
        text = document.read_text(encoding='utf-8')
        document.write_text(text.replace('blue_box_410x310x145.step', 'blue_box_410x310x145.stl'), encoding='utf-8')
    runtime = staged_library / 'runtime'
    shutil.copytree(resources, runtime)
    runtime_manifest = deepcopy(manifest)
    for record in runtime_manifest['assets'].values():
        source = next(row['source'] for row in library_index['assets']
                      if row['source']['sha256'] == record['source']['sha256'])
        record['source']['library_path'] = 'cad/' + source['path']
        record['source'].pop('path')
    write_json(runtime / 'cad/manifest.json', runtime_manifest)
    # The independent source library is authoritative; no duplicate source mesh
    # belongs in runtime/cad/source (all files here were created in this stage).
    for path in (runtime / 'cad/source').iterdir():
        if path.is_file():
            path.unlink()
    (runtime / 'cad/source').rmdir()
    if (library / 'runtime/README.md').is_file():
        shutil.copy2(library / 'runtime/README.md', runtime / 'README.md')
    write_json(output / 'migration.json', {'source_hash_updates': old_to_new,
        'grasp_rows_sha256': rows_digest, 'grasp_count': len(poses['grasps']),
        'active_geometry': 'one_lowpoly_surface_for_pose_planning_and_validation',
        'pink_perception': 'unknown_object_OBB_GraspGen_unchanged'})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(stage(args.project, args.library, args.candidates, args.output))

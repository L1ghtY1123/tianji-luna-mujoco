"""Convex CAD volume partitioning must cover cells, not just sampled surfaces."""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from luna_mujoco.cad_attachment import convex_volume_spheres, partition_convex_volume


def box_vertices():
    return np.array(list(product((-.022, .022), (-.028, .028), (-.03725, .03725))))


def test_partition_preserves_complete_volume_and_every_cell_has_a_covering_sphere():
    vertices = box_vertices()
    cells = partition_convex_volume(vertices, maximum_cells=24, minimum_size_m=.02)
    assert 1 < len(cells) <= 24
    assert sum(ConvexHull(cell).volume for cell in cells) == pytest.approx(ConvexHull(vertices).volume)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [57, -33, 11], degrees=True).as_matrix()
    pose[:3, 3] = [.02, -.01, .04]
    spheres, evidence = convex_volume_spheres(vertices, maximum_spheres=24, minimum_size_m=.02,
        padding_m=.006, transform=pose)
    assert len(spheres) == len(cells)
    for cell, sphere in zip(cells, spheres, strict=True):
        transformed = cell @ pose[:3, :3].T + pose[:3, 3]
        assert np.max(np.linalg.norm(transformed - sphere[:3], axis=1)) <= sphere[3] - .006 + 1e-12
    assert evidence['covers_model_volume'] is True
    assert evidence['partition_count'] == len(cells)


def test_separate_components_and_large_faces_are_covered_with_unchanged_padding():
    vertices = np.vstack([box_vertices() + [-.05, 0, 0], box_vertices() + [.05, .03, .01]])
    spheres, _ = convex_volume_spheres(vertices, maximum_spheres=31, minimum_size_m=.005,
        padding_m=.009, transform=np.eye(4))
    assert len(spheres) <= 31
    rng = np.random.default_rng(59)
    hull = ConvexHull(vertices)
    # Original triangles may cross many partition planes; their whole convex
    # volume remains covered, including the interiors between components.
    barycentric = rng.dirichlet(np.ones(4), size=2000)
    triangles = vertices[hull.simplices[rng.integers(len(hull.simplices), size=2000)]]
    tetrahedra = np.concatenate([triangles, np.broadcast_to(vertices.mean(0), (2000, 1, 3))], axis=1)
    points = np.vstack([vertices, np.einsum('ni,nij->nj', barycentric, tetrahedra)])
    balls = np.asarray(spheres)
    gaps = np.linalg.norm(points[:, None] - balls[None, :, :3], axis=2) - balls[None, :, 3]
    assert np.max(gaps.min(axis=1)) <= -.009 + 1e-12


def test_near_rigid_float32_transform_cannot_shrink_the_cover():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [11, 22, 33], degrees=True).as_matrix().astype(np.float32)
    pose[:3, :3] *= 1.000001
    vertices = box_vertices()
    balls, _ = convex_volume_spheres(vertices, maximum_spheres=24, minimum_size_m=.02,
        padding_m=.006, transform=pose)
    cells = partition_convex_volume(vertices, maximum_cells=24, minimum_size_m=.02)
    for cell, ball in zip(cells, balls, strict=True):
        world = cell @ pose[:3, :3].T + pose[:3, 3]
        assert np.linalg.norm(world - ball[:3], axis=1).max() <= ball[3] - .006 + 1e-12


def test_repeated_poses_reuse_canonical_fit_but_recheck_transformed_radius(monkeypatch):
    from luna_mujoco import cad_attachment

    calls = []
    original = cad_attachment.trimesh.nsphere.minimum_nsphere

    def counted(points):
        calls.append(len(points))
        return original(points)

    monkeypatch.setattr(cad_attachment.trimesh.nsphere, 'minimum_nsphere', counted)
    vertices = box_vertices() * 1.12345
    arguments = {'maximum_spheres': 19, 'minimum_size_m': .019, 'padding_m': .006}
    first, proof = convex_volume_spheres(vertices, transform=np.eye(4), **arguments)
    count = len(calls)
    assert count == proof['partition_count']
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [13, -29, 47], degrees=True).as_matrix() * 1.000001
    pose[:3, 3] = [.3, -.2, .4]
    second, _ = convex_volume_spheres(vertices.copy(), transform=pose, **arguments)
    assert len(calls) == count
    cells = partition_convex_volume(vertices, maximum_cells=19, minimum_size_m=.019)
    for cell, ball in zip(cells, second, strict=True):
        transformed = cell @ pose[:3, :3].T + pose[:3, 3]
        assert np.linalg.norm(transformed - ball[:3], axis=1).max() <= ball[3] - .006 + 1e-12
    first[0][3] = 999
    restored, _ = convex_volume_spheres(vertices, transform=np.eye(4), **arguments)
    assert restored[0][3] < 1
    convex_volume_spheres(vertices * 1.01, transform=pose, **arguments)
    assert len(calls) > count



@pytest.mark.parametrize('count', [0, -1, 32, True, 1.5])
def test_partition_rejects_invalid_sphere_budget(count):
    with pytest.raises(ValueError, match='cad_partition'):
        partition_convex_volume(box_vertices(), maximum_cells=count, minimum_size_m=.02)


def test_frozen_b1_attachment_reduces_anchor_protrusion_without_reducing_padding():
    from luna_mujoco.carrying import CarryingBinding, carrying_geometry, held_collision_binding
    from luna_mujoco.object_obb import held_geometry_points

    path = Path(__file__).parents[1] / 'data/cad-obb-iteration-20261030/black-B1-placement/event-00149.json'
    if not path.exists():
        pytest.skip('local frozen B1 archive is unavailable')
    record = json.loads(path.read_text(encoding='utf-8'))
    _observation, target, arm, scene, holding, _ = record['args']
    # Reuse the measured grasp with the newly published mesh in the same frame.
    # The frozen archive remains unchanged; production rejects stale bindings.
    from luna_mujoco.cad_perception import CadCatalog
    geometry = CadCatalog.from_file().geometry(holding['cad_tool']['mesh_key'])
    holding['cad_tool'].update({key: geometry[key] for key in ('source_sha256', 'mesh_sha256')})
    binding = CarryingBinding.from_mapping(held_collision_binding(holding, scene, arm, target), scene=scene, arm=arm)
    fitted = carrying_geometry(binding)
    assert fitted['fit_method'] == 'cad_convex_volume_partition_cover'
    assert fitted['sphere_count'] <= holding['sphere_budget'] == 24
    assert fitted['geometry_padding_m'] == pytest.approx(.006)
    rotation = np.asarray(holding['grasp_support_reference']['T_world_tcp_at_grasp'])[:3, :3]
    bottom = (held_geometry_points(holding) @ rotation.T)[:, 2].min()
    old = np.asarray(holding['attachment_evidence']['attachment_spheres_tool_m'])
    new = np.asarray(fitted['spheres'])
    old_excess = bottom - ((old[:, :3] @ rotation.T)[:, 2] - old[:, 3]).min()
    new_excess = bottom - ((new[:, :3] @ rotation.T)[:, 2] - new[:, 3]).min()
    assert old_excess > .026
    assert new_excess < .02
    assert fitted['covers_model_volume'] is True

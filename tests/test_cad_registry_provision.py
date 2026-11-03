"""Provisioning may write only verified, correctly framed model snapshots."""
import importlib.util
from pathlib import Path

import pytest


def script():
    path = Path(__file__).resolve().parents[1] / 'scripts/provision_foundationpose_registry.py'
    spec = importlib.util.spec_from_file_location('cad_registry_provision', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registration_rejects_wrong_canonical_origin():
    record = {'model_id': 'black_core', 'source_sha256': 'digest', 'units': 'mm',
              'geometry': 'rigid_reference', 'sdk_commit': 'a1b694b83e633c2cb6115b9063d940a687759392',
              'T_source_canonical_m': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
              'extents_mm': [44, 56, 74.5]}
    expected = {'model_id': 'black_core', 'source_sha256': 'digest',
                'center_m': [0, .006, -.12525], 'extents_mm': [44, 56, 74.5]}
    with pytest.raises(ValueError, match='canonical'):
        script().verify_registration(record, expected)


def test_registration_accepts_exact_source_frame():
    record = {'model_id': 'blue_box', 'source_sha256': 'digest', 'units': 'mm',
              'geometry': 'rigid_reference', 'sdk_commit': 'a1b694b83e633c2cb6115b9063d940a687759392',
              'T_source_canonical_m': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, .0725], [0, 0, 0, 1]],
              'extents_mm': [410, 310, 145]}
    expected = {'model_id': 'blue_box', 'source_sha256': 'digest',
                'center_m': [0, 0, .0725], 'extents_mm': [410, 310, 145]}
    assert script().verify_registration(record, expected) is None


def test_source_path_cannot_escape_read_only_cad_root(tmp_path):
    outside = tmp_path / 'outside.stl'
    outside.write_bytes(b'source')
    root = tmp_path / 'cad'
    root.mkdir()
    with pytest.raises(ValueError, match='outside'):
        script().scoped_file(root, '../outside.stl')

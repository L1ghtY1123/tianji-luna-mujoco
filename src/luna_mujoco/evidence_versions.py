"""Reproducible versions for model evidence, independent of task identities."""
import hashlib
import json
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

DECISION_INPUT_VERSION = 'scene-association-v4-declared-scene'


def evidence_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def implementation_manifest():
    """Freeze the running process's source/configuration version, never its secrets."""
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    paths = [*package.rglob('*.py'), *root.joinpath('resources/policy').rglob('*.json')]
    paths.extend(root.joinpath('skill_catalog').rglob('SKILL.md'))
    paths.extend(root / path for path in (
        'resources/cad_subject_policy.json', 'resources/workcell_assumptions.json',
        'resources/robot/marvin_gui_dual.yml'))
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(paths) if path.is_file()}
    for name in ('tianji_curobo', 'tianji_graspgen', 'sam3_service', 'tianji_foundationpose'):
        spec = find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            files['algorithm-services/' + name] = 'unavailable'
            continue
        service = Path(next(iter(spec.submodule_search_locations)))
        files.update({'algorithm-services/' + name + '/' + path.relative_to(service).as_posix():
                      hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(service.rglob('*.py'))})
    return {'implementation_sha256': evidence_digest(files), 'files': files,
            'decision_input_version': DECISION_INPUT_VERSION}

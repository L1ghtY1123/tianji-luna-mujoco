"""Reversible exact-JSON references for Decisions recovery context only."""
import hashlib
import json
from copy import deepcopy

FORMAT = 'exact_json_pointer_v1'
MINIMUM_CHARACTERS = 600


def _serialized(value):
    # Preserve key order and numeric representation: 1 and 1.0 do not merge.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_serialized(value).encode('utf-8')).hexdigest()


def _has_reference_key(value):
    if isinstance(value, dict):
        return '$context_ref' in value or any(_has_reference_key(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_reference_key(child) for child in value)
    return False


def compact_recovery_context(context):
    """Keep first complete values and refer only to byte-identical later JSON values."""
    if (context.get('verification_scope') not in {'graph_recovery', 'graph_recovery_continuation'}
            or 'context_compaction' in context or _has_reference_key(context)):
        return deepcopy(context)
    seen = {}
    reference_count = 0

    def encode(value, path):
        nonlocal reference_count
        if isinstance(value, (dict, list)):
            serialized = _serialized(value)
            if len(serialized) >= MINIMUM_CHARACTERS:
                digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
                previous = seen.get(digest)
                if previous is not None and previous[1] == serialized:
                    reference_count += 1
                    return {'$context_ref': previous[0], 'sha256': digest}
                seen[digest] = (path, serialized)
        if isinstance(value, dict):
            return {key: encode(child, path + '/' + key.replace('~', '~0').replace('/', '~1'))
                    for key, child in value.items()}
        if isinstance(value, list):
            return [encode(child, path + '/' + str(index)) for index, child in enumerate(value)]
        return value

    # candidate_projection is consumed by the image builder before the model
    # sees this context. Leave that optional input directly usable by its caller.
    compacted = {key: deepcopy(value) if key == 'candidate_projection' else
                 encode(value, '#/' + key.replace('~', '~0').replace('/', '~1'))
                 for key, value in context.items()}
    if not reference_count:
        return deepcopy(context)
    compacted['context_compaction'] = {
        'format': FORMAT, 'reference_count': reference_count,
        'expanded_sha256': _digest(context),
        'rule': 'Within this context only, an object containing exactly $context_ref and sha256 '
                'means the complete value at that JSON Pointer in this same context. The first '
                'complete value is retained. Decode ~1 as / and ~0 as ~ in path keys. References '
                'are exact duplicates, not new evidence, changed values, or omitted safety facts. '
                'Read the referenced full value wherever it appears. SHA256 binds compact UTF-8 '
                'JSON with original key order; all original context data can be reconstructed.'}
    if _serialized(expand_recovery_context(compacted)) != _serialized(context):
        raise ValueError('recovery_context_roundtrip_mismatch')
    return compacted


def expand_recovery_context(context):
    """Restore an archived context and verify both each reference and the full hash."""
    metadata = context.get('context_compaction')
    if metadata is None:
        return deepcopy(context)
    if metadata.get('format') != FORMAT:
        raise ValueError('recovery_context_reference_format_invalid')
    root = {key: value for key, value in context.items() if key != 'context_compaction'}

    def decode(value, active):
        if isinstance(value, dict) and set(value) == {'$context_ref', 'sha256'}:
            path = value['$context_ref']
            if not isinstance(path, str) or not path.startswith('#/') or path in active:
                raise ValueError('recovery_context_reference_path_invalid')
            target = root
            try:
                for part in path[2:].split('/'):
                    key = part.replace('~1', '/').replace('~0', '~')
                    target = target[int(key)] if isinstance(target, list) else target[key]
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ValueError('recovery_context_reference_path_invalid') from error
            restored = decode(target, active | {path})
            if _digest(restored) != value['sha256']:
                raise ValueError('recovery_context_reference_hash_mismatch')
            return restored
        if isinstance(value, dict):
            return {key: decode(child, active) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child, active) for child in value]
        return value

    expanded = decode(root, set())
    if _digest(expanded) != metadata['expanded_sha256']:
        raise ValueError('recovery_context_expanded_hash_mismatch')
    return expanded

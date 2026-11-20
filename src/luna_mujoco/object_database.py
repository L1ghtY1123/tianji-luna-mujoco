"""Permanent instance identities and dated memory, independent of task control."""
import json
import sqlite3
from contextlib import closing, contextmanager
from io import BytesIO
from pathlib import Path
from threading import local

import numpy as np

from .memory_contracts import ObjectRecord
from .object_memory import ObjectMemory, restored_object_record


class ObjectDatabase:
    def __init__(self, path):
        self.path = Path(path)
        self._subscribers = []
        self._transactions = local()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction(write=True) as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creation_key TEXT NOT NULL UNIQUE,
                initial_json TEXT NOT NULL,
                state_json TEXT NOT NULL
            )''')
            connection.execute('''CREATE TABLE IF NOT EXISTS history (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL REFERENCES entities(id),
                event_key TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity_id, event_key)
            )''')
            connection.execute('''CREATE TABLE IF NOT EXISTS captures (
                observation_id TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                rgb_npz BLOB NOT NULL
            )''')
            connection.execute('''CREATE TABLE IF NOT EXISTS instance_masks (
                observation_id TEXT NOT NULL REFERENCES captures(observation_id),
                entity_id INTEGER NOT NULL REFERENCES entities(id),
                masks_npz BLOB NOT NULL,
                PRIMARY KEY (observation_id, entity_id)
            )''')
            connection.execute('''CREATE TABLE IF NOT EXISTS object_changes (
                revision INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL REFERENCES entities(id),
                previous_state_json TEXT,
                state_json TEXT NOT NULL
            )''')
            connection.execute('''CREATE TRIGGER IF NOT EXISTS object_created
                AFTER INSERT ON entities BEGIN
                INSERT INTO object_changes(entity_id, state_json) VALUES (NEW.id, NEW.state_json);
                END''')
            connection.execute('''CREATE TRIGGER IF NOT EXISTS object_updated
                AFTER UPDATE OF state_json ON entities WHEN OLD.state_json != NEW.state_json BEGIN
                INSERT INTO object_changes(entity_id, previous_state_json, state_json)
                VALUES (NEW.id, OLD.state_json, NEW.state_json);
                END''')

    @property
    def revision(self):
        with self._transaction() as connection:
            return connection.execute('SELECT COALESCE(MAX(revision), 0) FROM object_changes').fetchone()[0]

    def subscribe(self, callback):
        """Notify consumers only after a successful physical-state transaction commits."""
        self._subscribers.append(callback)

    def grounding_ids(self, group_ids):
        """Find committed physical groups without persisting task reference bindings."""
        with self._transaction() as connection:
            result = {}
            for group_id in group_ids:
                rows = connection.execute('SELECT entity_id FROM history WHERE event_key=? AND kind=?',
                    ('rgbd-grounding:' + group_id, 'instance_rgbd_grounded')).fetchall()
                if len(rows) > 1:
                    raise ValueError('duplicate_committed_grounding_group')
                if rows:
                    result[group_id] = str(rows[0]['entity_id'])
            return result

    def changes_since(self, revision):
        with self._transaction() as connection:
            return [{'revision': row['revision'], 'entity_id': str(row['entity_id']),
                     'previous': json.loads(row['previous_state_json']) if row['previous_state_json'] else None,
                     'state': json.loads(row['state_json'])}
                    for row in connection.execute('SELECT * FROM object_changes WHERE revision>? ORDER BY revision',
                                                  (revision,))]

    def store_capture(self, capture):
        """Store immutable RGBD and calibration, independently of task lifetime."""
        metadata = {key: capture[key] for key in ('observation_id', 'generation', 'time') if key in capture}
        views = capture.get('raw_views', capture['views'])
        arrays = {camera: view['rgb'] for camera, view in views.items()}
        metadata['cameras'] = {}
        for camera, view in views.items():
            if not all(key in view for key in ('depth_m', 'valid_depth', 'intrinsics', 'T_world_camera')):
                continue
            arrays['depth__' + camera] = view['depth_m']
            arrays['valid__' + camera] = view['valid_depth']
            metadata['cameras'][camera] = {'intrinsics': view['intrinsics'],
                'T_world_camera': np.asarray(view['T_world_camera']).tolist()}
        encoded = BytesIO()
        np.savez_compressed(encoded, **arrays)
        with self._transaction(write=True) as connection:
            connection.execute('INSERT OR IGNORE INTO captures VALUES (?, ?, ?)',
                (capture['observation_id'], json.dumps(metadata), encoded.getvalue()))

    def store_instance_masks(self, capture, observation):
        """Persist accepted per-camera masks against their exact frozen RGBD capture."""
        if any(observation.get(key) != capture.get(key) for key in ('observation_id', 'generation')):
            raise ValueError('instance_mask_capture_mismatch')
        masks = {}
        for camera, view in observation.get('views', {}).items():
            for identifier, mask in view.get('identity_masks', view.get('instance_masks', {})).items():
                if not identifier.isascii() or not identifier.isdecimal():
                    continue
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != capture['views'][camera]['depth_m'].shape:
                    raise ValueError('instance_mask_shape_mismatch')
                if np.any(mask):
                    masks.setdefault(identifier, {})[camera] = mask
        with self._transaction(write=True) as connection:
            for identifier, views in masks.items():
                previous = connection.execute(
                    'SELECT masks_npz FROM instance_masks WHERE observation_id=? AND entity_id=?',
                    (capture['observation_id'], identifier)).fetchone()
                if previous is not None:
                    with np.load(BytesIO(previous['masks_npz']), allow_pickle=False) as archive:
                        saved = {camera: archive[camera].copy() for camera in archive.files}
                    for camera, mask in views.items():
                        if camera in saved and not np.array_equal(saved[camera], mask):
                            raise ValueError('immutable_instance_mask_changed')
                    views = {**saved, **views}
                encoded = BytesIO()
                np.savez_compressed(encoded, **views)
                connection.execute('INSERT INTO instance_masks VALUES (?, ?, ?) '
                    'ON CONFLICT(observation_id, entity_id) DO UPDATE SET masks_npz=excluded.masks_npz',
                    (capture['observation_id'], identifier, encoded.getvalue()))

    def reference(self, entities):
        """Load dated RGB with confirmed visual boxes or measured silhouette masks."""
        captures, references, previous_observation_id = {}, {}, None
        with self._transaction() as connection:
            if entities:
                placeholders = ','.join('?' for _ in entities)
                for event in connection.execute(
                    'SELECT kind, payload_json FROM history WHERE entity_id IN (' + placeholders + ') '
                    "AND kind IN ('observe_scene', 'instance_registered') ORDER BY sequence DESC", tuple(entities)):
                    payload = json.loads(event['payload_json'])
                    evidence = payload.get('evidence', {})
                    if event['kind'] == 'instance_registered' or evidence.get('status') == 'success':
                        previous_observation_id = evidence.get('observation_id')
                        if previous_observation_id:
                            break
            for identifier, entity in entities.items():
                capture_id = entity.get('reference_observation_id')
                camera = entity.get('reference_camera_id')
                if not capture_id or not camera or not entity.get('bbox'):
                    continue
                if capture_id not in captures:
                    stored = connection.execute('SELECT * FROM captures WHERE observation_id = ?',
                                                (capture_id,)).fetchone()
                    if stored is None:
                        raise ValueError('memory_reference_capture_missing:' + capture_id)
                    with np.load(BytesIO(stored['rgb_npz']), allow_pickle=False) as archive:
                        captures[capture_id] = (json.loads(stored['metadata_json']),
                            {key: archive[key].copy() for key in archive.files})
                metadata, images = captures[capture_id]
                references[identifier] = {**metadata, 'camera_id': camera, 'rgb': images[camera],
                    'bbox': list(entity['bbox']), 'source': entity.get('identity_source'),
                    'registration_basis': entity.get('registration_basis'), 'entity_id': identifier,
                    'request_hash': entity.get('identity_request_hash')}
                stored_masks = connection.execute(
                    'SELECT masks_npz FROM instance_masks WHERE observation_id=? AND entity_id=?',
                    (capture_id, identifier)).fetchone()
                views = {}
                if stored_masks is not None:
                    with np.load(BytesIO(stored_masks['masks_npz']), allow_pickle=False) as masks:
                        for view_id in masks.files:
                            calibration = metadata.get('cameras', {}).get(view_id)
                            if calibration is None:
                                continue  # Older RGB-only captures contain no geometric evidence.
                            views[view_id] = {**calibration, 'rgb': images[view_id],
                                'depth_m': images['depth__' + view_id],
                                'valid_depth': images['valid__' + view_id], 'mask': masks[view_id].copy()}
                for view_id in ('1', '2', '3'):
                    if view_id not in images or view_id in views:
                        continue
                    calibration = metadata.get('cameras', {}).get(view_id, {})
                    views[view_id] = {**calibration, 'rgb': images[view_id]}
                    if 'depth__' + view_id in images:
                        views[view_id].update(depth_m=images['depth__' + view_id], valid_depth=images['valid__' + view_id])
                proposal = entity.get('current_identity_proposal') or {}
                if (proposal.get('observation_id') == capture_id
                        and proposal.get('generation') == metadata.get('generation')):
                    for view_id, box in proposal.get('boxes', {}).items():
                        if view_id in views:
                            views[view_id]['bbox'] = list(box)
                            views[view_id]['box_source'] = 'confirmed_instance_mask_bounds'
                for view in views.values():
                    view['scope_source'] = 'full_stored_camera_frame'
                references[identifier]['views'] = views
                # Preserve the original pixel coordinates of older wrist-primary records.
                views[camera].setdefault('bbox', list(entity['bbox']))
                if camera != '3':
                    from .visual_identity import mask_box
                    head = views.get('3', {})
                    mask = head.get('mask')
                    box = mask_box(mask) if mask is not None and np.any(mask) else head.get('bbox')
                    references[identifier].update(camera_id='3' if head else None, rgb=head.get('rgb'), bbox=box)
                else:
                    references[identifier]['rgb'] = views[camera]['rgb']
        return {'entities': entities, 'identity_references': references,
                'previous_observation_id': previous_observation_id}

    @contextmanager
    def _transaction(self, *, write=False):
        active = getattr(self._transactions, 'connection', None)
        if active is not None:
            yield active
            return
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA foreign_keys = ON')
            has_changes = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='object_changes'").fetchone()
            before = (connection.execute('SELECT COALESCE(MAX(revision),0) FROM object_changes').fetchone()[0]
                      if write and has_changes else 0)
            with connection:
                connection.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
                self._transactions.connection = connection
                try:
                    yield connection
                finally:
                    self._transactions.connection = None
            if write and self._subscribers:
                changes = self.changes_since(before)
                if changes:
                    for callback in self._subscribers:
                        callback(changes)

    @contextmanager
    def transaction(self):
        """Group existing database operations into one commit and one notification."""
        with self._transaction(write=True):
            yield

    def commit_observation(self, capture, observation, events):
        with self.transaction():
            self.store_capture(capture)
            self.store_instance_masks(capture, observation)
            self.record_many(events)

    def identity_archive(self, entities):
        """Load older confirmed references, keeping each frame bound to its original identity."""
        from .memory_reference import IDENTITY_ARCHIVE_LIMIT, select_identity_history

        references, owners, selection = {}, {}, {}
        for identifier, entity in entities.items():
            record = self.get(identifier)
            history = record.get('history', [])
            original = entity.get('initial_reference')
            if original:
                history = [{'kind': 'scene_instance_confirmed', 'state': original}, *history]
            states, total = select_identity_history(history, entity)
            selection[identifier] = {'eligible_count': total, 'selected_observation_ids': [
                state['reference_observation_id'] for state in states]}
            for index, state in enumerate(states):
                observation_id = state['reference_observation_id']
                historical = self.reference({identifier: state}).get('identity_references', {}).get(identifier)
                if historical is not None:
                    label = identifier + '_archive_' + str(index + 1)
                    references[label] = historical
                    owners[label] = {'entity_id': identifier, 'observation_id': observation_id}
        return {'identity_references': references, 'archive_owners': owners,
                'archive_selection': {'policy': 'recent_alternate_camera_original_v1',
                    'limit_per_entity': IDENTITY_ARCHIVE_LIMIT, 'entities': selection}}

    def allocate(self, creation_key, entity):
        """Allocate once for an independently identified instance creation event."""
        if not isinstance(creation_key, str) or not creation_key.strip():
            raise ValueError('objects_creation_key_required')
        if not isinstance(entity, dict) or 'id' in entity or 'history' in entity:
            raise ValueError('objects_entity_attributes_required')
        state = ObjectRecord.model_validate({'id': '1', **entity}).model_dump(
            mode='json', exclude={'id'}, exclude_none=True)
        encoded = json.dumps(entity, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self._transaction(write=True) as connection:
            existing = connection.execute(
                'SELECT id, initial_json FROM entities WHERE creation_key = ?', (creation_key,)
            ).fetchone()
            if existing is not None:
                if existing['initial_json'] != encoded:
                    raise ValueError('objects_creation_key_conflict')
                return str(existing['id'])
            inserted = connection.execute(
                'INSERT INTO entities (creation_key, initial_json, state_json) VALUES (?, ?, ?)',
                (creation_key, encoded, json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)),
            )
            return str(inserted.lastrowid)

    def record(self, identifier, event_key, kind, state, evidence):
        self.record_many([(identifier, event_key, kind, state, evidence)])

    def register_product(self, creation_key, attributes, *, reference_masks=None):
        """Atomically register the output, retire its input, and append both histories."""
        if not isinstance(creation_key, str) or not creation_key.strip():
            raise ValueError('objects_creation_key_required')
        if not isinstance(attributes, dict) or {'id', 'history'} & attributes.keys():
            raise ValueError('objects_entity_attributes_required')
        ObjectRecord.model_validate({'id': '1', **attributes})
        encoded = json.dumps(attributes, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if attributes.get('creation_evidence', {}).get('localization') and reference_masks is None:
            raise ValueError('product_reference_mask_required')
        with self._transaction(write=True) as connection:
            encoded_masks = None
            if reference_masks is not None:
                capture_id = attributes.get('reference_observation_id')
                capture = connection.execute('SELECT * FROM captures WHERE observation_id=?', (capture_id,)).fetchone()
                if capture is None:
                    raise ValueError('product_reference_capture_missing')
                metadata = json.loads(capture['metadata_json'])
                if (metadata.get('generation') != attributes['creation_evidence']['generation']
                        or capture_id != attributes['creation_evidence']['observation_id']
                        or set(reference_masks) != {attributes.get('reference_camera_id')}):
                    raise ValueError('product_reference_measurement_mismatch')
                from .visual_identity import array_stamp

                with np.load(BytesIO(capture['rgb_npz']), allow_pickle=False) as images:
                    for camera, mask in reference_masks.items():
                        if (camera not in metadata['cameras'] or camera not in images
                                or mask.dtype != np.bool_ or mask.shape != images[camera].shape[:2]
                                or array_stamp(mask) != attributes['creation_evidence']['localization']['mask']):
                            raise ValueError('product_reference_mask_conflict')
                stream = BytesIO()
                np.savez_compressed(stream, **reference_masks)
                encoded_masks = stream.getvalue()
            source_id = attributes['input_entity_id']
            source = self._entity(connection, source_id)
            source_record = restored_object_record({**json.loads(source['state_json']), 'id': source_id})
            if attributes.get('world_key') != source_record.world_key:
                raise ValueError('product_scene_scope_mismatch')
            existing = connection.execute('SELECT id,initial_json FROM entities WHERE creation_key=?',
                                          (creation_key,)).fetchone()
            if existing is not None:
                if existing['initial_json'] != encoded or source_record.output_entity_id != str(existing['id']):
                    raise ValueError('objects_creation_key_conflict')
                stored = self._entity(connection, str(existing['id']))
                return {'id': str(existing['id']), **json.loads(stored['state_json'])}
            inserted = connection.execute(
                'INSERT INTO entities (creation_key,initial_json,state_json) VALUES (?,?,?)',
                (creation_key, encoded, encoded))
            identifier = str(inserted.lastrowid)
            if encoded_masks is not None:
                connection.execute('INSERT INTO instance_masks VALUES (?, ?, ?)',
                    (attributes['reference_observation_id'], identifier, encoded_masks))
            product = ObjectRecord.model_validate({'id': identifier, **attributes,
                'creation_evidence': {**attributes['creation_evidence'], 'output_entity_id': identifier}})
            memory = ObjectMemory()
            memory.restore((source_record,))
            memory.register_product(product)
            events = []
            for entity_id, entity in memory.entities().items():
                state = ObjectRecord.model_validate(entity).model_dump(mode='json', exclude={'id'})
                events.append((entity_id, creation_key, 'processing_result', state, product.creation_evidence))
            self._record_events(connection, events)
            return memory.entities()[identifier]

    def record_many(self, events):
        """Commit related instance histories together, including input/output transitions."""
        with self._transaction(write=True) as connection:
            self._record_events(connection, events)

    def _record_events(self, connection, events):
        for identifier, event_key, kind, state, evidence in events:
            if not isinstance(event_key, str) or not event_key.strip() or not isinstance(kind, str) or not kind.strip():
                raise ValueError('objects_event_key_and_kind_required')
            if not isinstance(state, dict) or {'id', 'history'} & state.keys() or not isinstance(evidence, dict):
                raise ValueError('objects_event_state_and_evidence_required')
            if set(state) - ObjectRecord.model_fields.keys():
                raise ValueError('object_memory_snapshot_contains_nonphysical_fields')
            payload = json.dumps({'state': state, 'evidence': evidence},
                                 ensure_ascii=False, sort_keys=True, allow_nan=False)
            entity = self._entity(connection, identifier)
            origin = json.loads(entity['initial_json'])
            if 'initial_reference' in state and state['initial_reference'] != origin.get('initial_reference'):
                raise ValueError('objects_initial_reference_is_immutable')
            if origin.get('world_key') is not None and 'world_key' in state and state['world_key'] != origin['world_key']:
                raise ValueError('objects_instance_scene_scope_changed')
            existing = connection.execute(
                'SELECT kind, payload_json FROM history WHERE entity_id = ? AND event_key = ?',
                (entity['id'], event_key),
            ).fetchone()
            if existing is not None:
                if existing['kind'] != kind or existing['payload_json'] != payload:
                    raise ValueError('objects_event_conflict')
                continue
            previous = restored_object_record({**json.loads(entity['state_json']), 'id': str(entity['id'])})
            merged = ObjectRecord.model_validate(previous.model_dump() | state).model_dump(
                mode='json', exclude={'id'}, exclude_none=True)
            connection.execute('UPDATE entities SET state_json = ? WHERE id = ?',
                (json.dumps(merged, ensure_ascii=False, sort_keys=True, allow_nan=False), entity['id']))
            connection.execute(
                'INSERT INTO history (entity_id, event_key, kind, payload_json) VALUES (?, ?, ?, ?)',
                (entity['id'], event_key, kind, payload),
            )

    def get(self, identifier):
        """Read one instance and all recorded observations and operations."""
        with self._transaction() as connection:
            entity = self._entity(connection, identifier)
            events = connection.execute('SELECT * FROM history WHERE entity_id = ? ORDER BY sequence',
                                        (entity['id'],)).fetchall()
            return {**json.loads(entity['state_json']), 'id': str(entity['id']),
                'history': [{'event_key': event['event_key'], 'kind': event['kind'],
                             'recorded_at': event['recorded_at'], **json.loads(event['payload_json'])}
                            for event in events]}

    def entities(self, *, world_key=None):
        """Return current recorded state, including unseen and retired instances."""
        with self._transaction() as connection:
            rows = [{**json.loads(row['state_json']), 'id': str(row['id'])}
                    for row in connection.execute('SELECT id, state_json FROM entities ORDER BY id')]
            return [row for row in rows if world_key is None or row.get('world_key') == world_key]

    @staticmethod
    def _entity(connection, identifier):
        if not isinstance(identifier, str) or not identifier.isascii() or not identifier.isdecimal():
            raise ValueError('invalid_permanent_entity_id')
        if identifier != str(int(identifier)) or int(identifier) < 1:
            raise ValueError('invalid_permanent_entity_id')
        entity = connection.execute('SELECT * FROM entities WHERE id = ?', (identifier,)).fetchone()
        if entity is None:
            raise ValueError('unknown_permanent_entity_id:' + identifier)
        return entity

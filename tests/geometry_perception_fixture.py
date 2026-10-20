"""Explicit visual decisions for measured-geometry tests; never infer identity."""

from jev_mujoco.perception import Perception
from jev_mujoco.visual_inventory import VisualInventory


class ScriptedVisualInventory:
    def __init__(self, identities, kinds=None):
        self.identities = identities
        self.kinds = kinds or {}

    def resolve(self, capture, detections, **kwargs):
        by_camera = {}
        entities, judgments = {}, {}
        for index, detection in enumerate(detections):
            camera = detection['camera_id']
            offset = by_camera.get(camera, 0)
            labels = self.identities[camera] if isinstance(self.identities, dict) else self.identities
            assert offset < len(labels), f'Unscripted visual candidate {camera}:{offset}'
            identifier = labels[offset]
            by_camera[camera] = offset + 1
            judgments['identity_' + str(index)] = identifier
            if identifier != 'reject':
                entities.setdefault(identifier, {'kind': self.kinds.get(identifier, detection['kind'])})
        for camera, count in by_camera.items():
            labels = self.identities[camera] if isinstance(self.identities, dict) else self.identities
            assert count == len(labels), f'Missing scripted visual candidates in camera {camera}'
        evidence = {'source': 'openai_decisions', 'observation_id': capture['observation_id'],
                    'generation': capture['generation']}
        return VisualInventory(api_key='unused')._groups(
            detections, [str(index) for index in range(len(detections))], entities, judgments, evidence)


class GeometryPerception(Perception):
    """Use fixed fixture decisions while exercising the real geometry boundary."""

    def __init__(self, detector, *, identities=('first', 'second'), kinds=None, prompts=None, **kwargs):
        self.prompts = prompts or {}
        super().__init__(detector, visual_inventory=ScriptedVisualInventory(identities, kinds), **kwargs)

    def observe(self, capture, *, source_hint=None, **kwargs):
        hint = dict(source_hint or {'kind': 'cube', 'color': 'any'})
        hint['visual_decision'] = {'source': 'jev', 'purpose': 'initialize_inventory'}
        inventory = self.visual_inventory
        identities = {key for values in inventory.identities.values() for key in values} if isinstance(
            inventory.identities, dict) else set(inventory.identities)
        kind = hint.get('kind', 'cube')
        kind = 'cube' if kind in {'any', 'object'} else kind
        color = hint.get('color', 'any')
        entities = {}
        if hint.get('scene_queries') is not None:
            for index, query in enumerate(hint['scene_queries']):
                for identifier in query.get('entity_ids', ['query-' + str(index)]):
                    entities[identifier] = {'id': identifier, 'kind': query['kind'], 'color': query['color'],
                        'roles': ['collision'], 'segmentation_query': query}
        else:
            for identifier in sorted(identities - {'reject'}):
                item_kind = inventory.kinds.get(identifier, kind)
                prompt = 'open box' if item_kind == 'bin' else item_kind
                prompt = prompt if color == 'any' else color + ' ' + prompt
                entities[identifier] = {'id': identifier, 'kind': item_kind, 'color': color,
                    'roles': ['collision'], 'segmentation_query': {
                        'kind': item_kind, 'color': color, 'prompt': self.prompts.get(identifier, prompt)}}
        # These fixtures declare marker identities explicitly, independently of depth.
        for marker_color in ('yellow', 'red', 'green', 'blue'):
            identifier = 'marker-' + marker_color
            entities[identifier] = {'id': identifier, 'kind': 'marker', 'color': marker_color, 'roles': ['target_region']}
        hint.setdefault('scene_entities', entities)
        return super().observe(capture, source_hint=hint, **kwargs)


class ScriptedDiscovery:
    def __init__(self, identifiers=('first', 'second')):
        self.identifiers = identifiers

    def discover(self, capture, *, instruction):
        entities = {identifier: {'id': identifier, 'kind': 'cube', 'color': 'red',
            'roles': ['collision'], 'visual_membership': 'required',
            'segmentation_query': {'kind': 'cube', 'color': 'red', 'prompt': 'cube'}}
            for identifier in self.identifiers}
        return {'entities': entities, 'reference': {'entities': entities, 'views': {'3': {'rgb': capture['views']['3']['rgb']}}},
                'evidence': {'source': 'scripted_test_discovery'}}

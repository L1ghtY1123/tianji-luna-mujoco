"""Expand reviewed, parameterized policy scopes before ordinary graph validation.

No runtime actions or model calls occur here. Expanded bodies remain ordinary
nodes and edges, including every recovery action and finite traversal budget.
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import Field

from ..core.graph import StrictModel

LIBRARY_PATH = Path(__file__).resolve().parents[3] / 'resources/policy/subgraphs.json'


class TemplateDeclaration(StrictModel):
    id: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]{0,95}$')
    template: str
    arguments: dict


def _library():
    return json.loads(LIBRARY_PATH.read_text(encoding='utf-8'))['templates']


def library_catalog(catalog=None):
    """Publish interfaces and allowed skills; exclude unavailable capabilities."""
    templates = _library()
    fingerprint = hashlib.sha256(json.dumps(templates, sort_keys=True).encode()).hexdigest()

    def skills(name):
        template = templates[name]
        result = {node['skill'] for node in template['body']['nodes'] if node['type'] == 'skill'}
        for dependency in template['dependencies']:
            result |= skills(dependency['template'])
        return result

    return {name: {key: deepcopy(template[key]) for key in (
                'description', 'arguments', 'requires', 'success', 'failure', 'unknown')}
            | {'allowed_skills': sorted(skills(name)),
               'fingerprint': fingerprint}
            for name, template in templates.items() if catalog is None or skills(name) <= catalog.keys()}


def template_schemas(catalog):
    return [{'type': 'object', 'properties': {
        'id': {'type': 'string', 'pattern': r'^[A-Za-z][A-Za-z0-9_-]{0,39}$'},
        'template': {'type': 'string', 'const': name},
        'arguments': deepcopy(template['arguments'])},
        'required': ['id', 'template', 'arguments'], 'additionalProperties': False}
        for name, template in library_catalog(catalog).items()]


def expand_templates(scopes, *, catalog=None):
    templates = _library()
    available = library_catalog(catalog)

    def instantiate(declaration):
        declaration = TemplateDeclaration.model_validate(declaration)
        if declaration.template not in available:
            raise ValueError('unknown_or_unavailable_subgraph_template:' + declaration.template)
        template = templates[declaration.template]
        errors = list(Draft202012Validator(template['arguments']).iter_errors(declaration.arguments))
        if errors:
            raise ValueError('invalid_subgraph_inputs:' + declaration.id + ':' + errors[0].message)
        arguments = declaration.arguments
        if (arguments.get('release_mode') == 'container_release'
                and arguments.get('destination_kind') != 'container'):
            raise ValueError('container_release_requires_container')

        def bind(value):
            if isinstance(value, dict):
                if set(value) == {'$parameter'}:
                    return deepcopy(arguments[value['$parameter']])
                return {key: bind(item) for key, item in value.items()}
            if isinstance(value, list):
                return [bind(item) for item in value]
            return value

        scope = bind(template['body'])
        prefix = declaration.id + '__'
        ids = {node['id']: prefix + node['id'] for node in scope['nodes']}
        dependency_ids = {row['id']: declaration.id + '_' + row['id'] for row in template['dependencies']}
        scope['id'], scope['entry'] = declaration.id, ids[scope['entry']]
        for node in scope['nodes']:
            node['id'] = ids[node['id']]
            if node['type'] == 'subgraph':
                node['subgraph_id'] = dependency_ids[node['subgraph_id']]
        for edge in scope['edges']:
            if 'sources' in edge:
                edge['sources'] = [ids[source] for source in edge['sources']]
            else:
                edge['source'] = ids[edge['source']]
            edge['target'] = ids[edge['target']]
        for outcome in ('failure', 'unknown'):
            if scope.get('on_' + outcome):
                scope['on_' + outcome]['target'] = ids[scope['on_' + outcome]['target']]
        result = [scope]
        for row in template['dependencies']:
            result.extend(instantiate({**bind(row), 'id': dependency_ids[row['id']]}))
        return result

    result = []
    for scope in scopes:
        if isinstance(scope, dict) and 'template' in scope and len(str(scope.get('id', ''))) > 40:
            raise ValueError('subgraph_template_id_too_long')
        result.extend(instantiate(scope) if isinstance(scope, dict) and 'template' in scope else [deepcopy(scope)])
    return result

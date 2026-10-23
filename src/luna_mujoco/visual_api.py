"""Dedicated official OpenAI image calls, bounded caching and replay evidence."""
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import httpx


class VisualInventoryError(ValueError):
    def __init__(self, reason, *, evidence=None):
        super().__init__(reason)
        self.evidence = evidence or {}


def audit_response(body, secret, *, bounded=True):
    if isinstance(body, dict):
        selected = {key: body[key] for key in ('model', 'answers', 'usage', 'output', 'status',
                                              'incomplete_details') if key in body}
        if isinstance(body.get('error'), dict):
            selected['error'] = {key: body['error'][key] for key in ('type', 'code') if key in body['error']}
    else:
        selected = {'malformed_body': body}
    encoded = json.dumps(selected, ensure_ascii=False)
    if secret:
        encoded = encoded.replace(secret, '[redacted]')
    if bounded and len(encoded) > 32768:
        return {'truncated_response': encoded[:16384], 'size_chars': len(encoded)}
    return json.loads(encoded)


class VisualAPI:
    endpoint = ''
    source = ''
    error_prefix = ''

    @classmethod
    def from_environment(cls, *, audit=None):
        path = Path(os.environ.get('LUNA_DECISIONS_CONFIG_FILE',
                                  str(Path.home() / '.config/openai-client/decisions.json')))
        config = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        # OPENAI_API_KEY/jojo.json belong to a different provider in this app.
        key = os.environ.get('LUNA_DECISIONS_API_KEY') or config.get('api_key', '')
        return cls(api_key=key, timeout_s=float(os.environ.get('LUNA_DECISIONS_TIMEOUT_S', '60')), audit=audit)

    def __init__(self, *, api_key='', client=None, timeout_s=60., audit=None):
        if not math.isfinite(timeout_s) or not 1. <= timeout_s <= 180.:
            raise ValueError('visual_decisions_invalid_timeout')
        self.api_key, self.client, self.timeout_s, self.audit = api_key, client, timeout_s, audit
        self.cache = OrderedDict()

    def request(self, payload, capture, purpose, parse):
        image_selection = None
        if self.endpoint == '/v1/decisions':
            from .visual_image_selection import compact_decision_images

            payload, image_selection = compact_decision_images(payload)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            result = self.cache[key]
            if isinstance(result, VisualInventoryError):
                raise result
            return deepcopy(result)
        event = None
        evidence = {'source': self.source, 'observation_id': capture['observation_id'], 'request_hash': key}
        try:
            if not self.api_key:
                raise VisualInventoryError(self.error_prefix + '_not_configured')
            started = time.monotonic()
            event = {'endpoint': self.endpoint, 'request': deepcopy(payload), 'response': None,
                     'observation_id': capture['observation_id'], 'request_hash': key, 'purpose': purpose,
                     'elapsed_s': 0., 'http_status': None, 'error': None}
            if image_selection is not None:
                event['image_selection'] = image_selection
            response = (self.client or httpx).post('https://api.openai.com' + self.endpoint,
                headers={'Authorization': 'Bearer ' + self.api_key}, json=payload,
                timeout=self.timeout_s, follow_redirects=False)
            event['http_status'] = response.status_code
            event['elapsed_s'] = round(time.monotonic() - started, 3)
            evidence.update(http_status=response.status_code, elapsed_s=event['elapsed_s'])
            try:
                body = response.json()
                event['response'] = audit_response(body, self.api_key)
            except ValueError:
                body = None
                event['response'] = {'invalid_body': 'non_json', 'size_bytes': len(response.content),
                                     'sha256': hashlib.sha256(response.content).hexdigest()}
            if response.status_code != 200:
                raise VisualInventoryError(self.error_prefix + '_http_' + str(response.status_code))
            if not isinstance(body, dict):
                raise VisualInventoryError(self.error_prefix + '_invalid_response')
            evidence.update(model=body.get('model'), usage=body.get('usage', {}))
            result = parse(body, evidence)
            event['response'] = audit_response(body, self.api_key, bounded=False)
            self.cache[key] = deepcopy(result)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as error:
            error = error if isinstance(error, VisualInventoryError) else VisualInventoryError(
                self.error_prefix + ('_transport_error' if isinstance(error, httpx.HTTPError) else '_invalid_response'))
            error.evidence = {**evidence, **error.evidence}
            if event is not None:
                event['error'] = str(error)
                event['elapsed_s'] = round(time.monotonic() - started, 3)
            self.cache[key] = error
            raise error from None
        finally:
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
            if event is not None and self.audit is not None:
                try:
                    self.audit(event)
                except (OSError, ValueError, TypeError, RuntimeError):
                    error = VisualInventoryError('visual_decisions_audit_failed', evidence=evidence)
                    self.cache[key] = error
                    raise error from None
        return result

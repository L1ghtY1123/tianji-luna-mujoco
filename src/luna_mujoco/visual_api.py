"""Dedicated official OpenAI image calls, bounded caching and replay evidence."""
import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import httpx


class VisualInventoryError(ValueError):
    def __init__(self, reason, *, evidence=None):
        super().__init__(reason)
        self.evidence = evidence or {}


def decision_diagnostic_checks(questions, answers, minimum_confidence):
    """Describe the existing decision gate; never supply a missing model reason."""
    by_name = {answer.get('name'): answer for answer in answers if isinstance(answer, dict)}
    checks = []
    for question in questions:
        answer = by_name.get(question['name'], {})
        predicate = question.get('type') == 'predicate'
        actual = answer.get('probability' if predicate else 'confidence')
        valid = (type(actual) in {int, float} and math.isfinite(actual) and 0 <= actual <= 1
                 and answer.get('type') == question.get('type'))
        status, reason = 'unknown', 'invalid_or_missing_answer'
        if valid and not predicate and answer.get('choice') not in [
                item['value'] for item in question.get('choices', [])]:
            valid, reason = False, 'invalid_answer_choice'
        if valid and minimum_confidence is not None:
            if predicate:
                status = ('yes' if actual >= minimum_confidence else 'no'
                          if actual <= round(1 - minimum_confidence, 12) else 'unknown')
                reason = 'predicate_probability_gate'
            else:
                status = 'known' if actual >= minimum_confidence and answer.get('choice') != 'unknown' else 'unknown'
                reason = 'explicit_unknown' if answer.get('choice') == 'unknown' else 'confidence_gate'
        checks.append({'name': question['name'], 'component': 'decisions',
            'question': question.get('instructions'), 'answer': deepcopy(answer),
            'metric': 'predicate_probability' if predicate else 'decisions_confidence',
            'actual': actual, 'threshold': minimum_confidence, 'comparison': '>=',
            **({'negative_threshold': round(1 - minimum_confidence, 12), 'negative_comparison': '<='}
               if predicate and minimum_confidence is not None else {}),
            'status': status, 'reason': reason})
    return checks


def diagnostic_evidence(value, secret=''):
    """Keep error evidence text-only and redact credentials even in malformed answers."""
    if isinstance(value, dict):
        return {key: diagnostic_evidence(item, secret) for key, item in value.items()
                if key.lower() not in {'api_key', 'authorization', 'image_url', 'rgb_jpeg'}}
    if isinstance(value, (list, tuple)):
        return [diagnostic_evidence(item, secret) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, str):
        value = value.replace(secret, '[redacted]') if secret else value
        return re.sub(r'data:[^,\s]*,[^\s]*', '[data URI omitted]', value, flags=re.IGNORECASE)[:8192]
    return value


def audit_response(body, secret, *, bounded=True):
    if isinstance(body, dict):
        selected = {key: body[key] for key in ('model', 'answers', 'usage', 'output', 'status',
                                              'incomplete_details') if key in body}
        if isinstance(body.get('error'), dict):
            selected['error'] = {key: body['error'][key] for key in ('type', 'code') if key in body['error']}
            if body['error'].get('code') == 'invalid_json_schema':
                selected['error']['message'] = diagnostic_evidence(body['error'].get('message', ''), secret)
    else:
        selected = {'malformed_body': body}
    encoded = json.dumps(selected, ensure_ascii=False)
    if secret:
        encoded = encoded.replace(secret, '[redacted]')
    if bounded and len(encoded) > 32768:
        safe = json.loads(encoded)
        return {**{key: safe[key] for key in ('model', 'usage', 'status', 'incomplete_details', 'error') if key in safe},
                'truncated_response': encoded[:16384], 'size_chars': len(encoded)}
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

    def request(self, payload, capture, purpose, parse, *, model_call=None):
        # Transport retries repeat only a frozen, non-actuating Decisions request.
        # A valid unknown/no answer is final; this never retries a robot skill.
        attempts = 3 if self.endpoint == '/v1/decisions' else 1
        for attempt in range(1, attempts + 1):
            try:
                return self._request_once(payload, capture, purpose, parse,
                                          model_call=model_call, transport_attempt=attempt)
            except VisualInventoryError as error:
                transient = (error.evidence.get('http_status') in {429, 502, 503, 504}
                             or str(error) == self.error_prefix + '_transport_error')
                if not transient or attempt == attempts or str(error) == 'visual_decisions_audit_failed':
                    raise
                time.sleep(.5 * attempt)

    def _request_once(self, payload, capture, purpose, parse, *, model_call=None, transport_attempt=1):
        from uuid import uuid4

        image_selection = None
        if self.endpoint == '/v1/decisions':
            from .identity_comparison_images import scope_comparison_questions
            from .visual_image_selection import compact_decision_images

            payload = scope_comparison_questions(payload)
            payload, image_selection = compact_decision_images(payload)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            result = self.cache[key]
            if isinstance(result, VisualInventoryError):
                raise result
            return deepcopy(result)
        event = None
        evidence = {'source': self.source, 'observation_id': capture['observation_id'], 'request_hash': key,
                    'component': 'decisions' if self.endpoint == '/v1/decisions' else self.source,
                    'stage': purpose, 'purpose': purpose, 'camera_ids': list(capture.get('views', {}))}
        if self.endpoint == '/v1/decisions':
            evidence.update(questions=diagnostic_evidence(payload.get('questions', []), self.api_key),
                            known_facts=['decision_gate_uses_decisions_confidence_or_predicate_probability_not_sam3_score'],
                            unknowns=['model_internal_reason_not_returned'])
        try:
            if not self.api_key:
                raise VisualInventoryError(self.error_prefix + '_not_configured')
            started = time.monotonic()
            event = {'endpoint': self.endpoint, 'request': deepcopy(payload), 'response': None,
                     'observation_id': capture['observation_id'], 'request_hash': key, 'purpose': purpose,
                     'elapsed_s': 0., 'http_status': None, 'error': None,
                     'call_id': uuid4().hex, 'transport_attempt': transport_attempt}
            if model_call is not None:
                event['model_call'] = diagnostic_evidence(model_call, self.api_key)
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
            if self.endpoint == '/v1/decisions':
                answers = body.get('answers')
                evidence['diagnostic_checks'] = diagnostic_evidence(decision_diagnostic_checks(
                    payload.get('questions', []), answers if isinstance(answers, list) else [],
                    getattr(self, 'minimum_confidence', None)), self.api_key)
            evidence = diagnostic_evidence(evidence, self.api_key)
            result = parse(body, evidence)
            if isinstance(result, dict):
                # Parsers attach raw answers after interpreting them. Sanitize only
                # evidence fields, leaving judgments, geometry and model plans intact.
                for field in ('evidence', 'visual_evidence', 'response'):
                    if field in result:
                        result[field] = diagnostic_evidence(result[field], self.api_key)
            event['response'] = audit_response(body, self.api_key, bounded=False)
            self.cache[key] = deepcopy(result)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as error:
            if isinstance(error, httpx.HTTPError):
                evidence.update(transport_error_type=type(error).__name__, configured_timeout_s=self.timeout_s)
                if event is not None:
                    event.update(transport_error_type=type(error).__name__, configured_timeout_s=self.timeout_s)
            error = error if isinstance(error, VisualInventoryError) else VisualInventoryError(
                self.error_prefix + ('_transport_error' if isinstance(error, httpx.HTTPError) else '_invalid_response'))
            error.evidence = diagnostic_evidence({**evidence, **error.evidence}, self.api_key)
            if event is not None:
                event['error'] = str(error)
                if 'validation_errors' in error.evidence:
                    event['validation_errors'] = error.evidence['validation_errors']
                event['elapsed_s'] = round(time.monotonic() - started, 3)
            # An unavailable service supplies no verdict to cache. Successful
            # judgments (including unknown), semantic errors and audit failures do.
            if not (self.endpoint == '/v1/decisions' and (
                    evidence.get('http_status') in {429, 502, 503, 504}
                    or str(error) == self.error_prefix + '_transport_error')):
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

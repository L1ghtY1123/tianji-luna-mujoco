"""Project tracking route around the unchanged SDK HTTP/array contracts."""
import hashlib
import json
import logging
import threading
import time

from tianji_foundationpose.service import make_server as sdk_server

from tianji_foundationpose.contracts import (
    MAX_BODY_BYTES,
    SDK_COMMIT,
    PoseError,
    decode_request,
    validate_pose,
)


def make_server(host, port, backend, registry):
    server = sdk_server(host, port, backend, registry)
    inference_lock = threading.Lock()

    class Handler(server.RequestHandlerClass):
        def do_POST(self):
            # One lock covers BOTH routes, including all per-model mutable state.
            if not inference_lock.acquire(blocking=False):
                self.send_json(503, {'ok': False, 'error': 'busy'})
                return
            try:
                if self.path == '/v1/track':
                    self.track()
                else:
                    super().do_POST()
            finally:
                inference_lock.release()

        def track(self):
            try:
                if self.headers.get('Content-Type') != 'application/json':
                    raise PoseError('invalid_request')
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= MAX_BODY_BYTES:
                        raise ValueError('body size')
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError('incomplete body')
                    payload = json.loads(body)
                    if not isinstance(payload, dict) or set(payload) != {'frame', 'seed'}:
                        raise ValueError('tracking envelope')
                    request = decode_request(payload['frame'])
                    seed = payload['seed']
                    if (not isinstance(seed, dict) or set(seed) != {
                            'T_camera_object', 'instance_id', 'source_sha256', 'observation_id'}
                            or any(not isinstance(seed[key], str) or not seed[key] for key in (
                                'instance_id', 'source_sha256', 'observation_id'))
                            or seed['observation_id'] == request['capture_id']):
                        raise ValueError('tracking seed')
                    validate_pose(seed['T_camera_object'])
                except (ValueError, TypeError, UnicodeError) as error:
                    raise PoseError('invalid_request') from error
                started = time.monotonic()
                result = backend.estimate(request, tracking_seed=seed)
                result.update(ok=True, status='estimated', version=1, sdk_commit=SDK_COMMIT,
                    capture_id=request['capture_id'], camera_id=request['camera_id'], model_id=request['model_id'],
                    translation_unit='m', camera_axes='opencv', algorithm='foundationpose',
                    request_sha256=hashlib.sha256(body).hexdigest(), elapsed_s=time.monotonic() - started)
                self.send_json(200, result)
            except PoseError as error:
                self.send_json(422, {'ok': False, 'error': error.code, 'message': str(error)})
            except Exception:
                logging.getLogger(__name__).exception('FoundationPose tracking failed')
                self.send_json(500, {'ok': False, 'error': 'inference_failed'})

    server.RequestHandlerClass = Handler
    return server

"""Explicit instance seeds for official FoundationPose tracking; no SDK edits."""
import hashlib
import json
import urllib.error
import urllib.request

from tianji_foundationpose.client import FoundationPoseClient
from tianji_foundationpose.contracts import SDK_COMMIT, PoseError, encode_request, validate_pose


class FoundationPoseTrackingClient(FoundationPoseClient):
    def track(self, rgb, depth_m, mask, K, *, model_id, capture_id, camera_id, score, seed):
        payload = {'frame': encode_request(rgb, depth_m, mask, K, model_id, capture_id, camera_id, score),
                   'seed': seed}
        body = json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.url + '/v1/track', data=body,
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read(1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            raw = error.read(1024 * 1024 + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise PoseError('service_unavailable') from error
        try:
            if len(raw) > 1024 * 1024:
                raise PoseError('invalid_response')
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise PoseError('invalid_response')
            if result.get('ok') is not True:
                raise PoseError(result.get('error', 'invalid_response'))
            expected = {'version': 1, 'algorithm': 'foundationpose', 'sdk_commit': SDK_COMMIT,
                'capture_id': capture_id, 'camera_id': camera_id, 'model_id': model_id,
                'translation_unit': 'm', 'camera_axes': 'opencv', 'status': 'estimated',
                'estimation_mode': 'track_one', 'tracking_seed': seed,
                'source_sha256': seed['source_sha256'], 'request_sha256': hashlib.sha256(body).hexdigest()}
            if any(result.get(key) != value for key, value in expected.items()):
                raise PoseError('provenance_mismatch')
            validate_pose(result['T_camera_object'])
            validate_pose(result['T_source_canonical_m'])
            return result
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, PoseError):
                raise
            raise PoseError('invalid_response') from error

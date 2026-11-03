"""Small Python client for registered SAM3 instance masks and RGBD frames."""
import hashlib
import json
import os
import urllib.error
import urllib.request

from .contracts import SDK_COMMIT, PoseError, encode_request, validate_pose


class FoundationPoseClient:
    def __init__(self,url=None,timeout_s=180):
        self.url = (url or os.getenv("LUNA_FOUNDATIONPOSE_URL","http://127.0.0.1:8365")).rstrip("/")
        self.timeout_s = timeout_s

    def estimate(self,rgb,depth_m,mask,K,*,model_id,capture_id,camera_id,score):
        payload = encode_request(rgb,depth_m,mask,K,model_id,capture_id,camera_id,score)
        body = json.dumps(payload,allow_nan=False).encode()
        request = urllib.request.Request(self.url + "/v1/estimate",data=body,
                                         headers={"Content-Type":"application/json"},method="POST")
        try:
            with urllib.request.urlopen(request,timeout=self.timeout_s) as response:
                raw = response.read(1024*1024+1)
        except urllib.error.HTTPError as error:
            raw = error.read(1024*1024+1)
        except (urllib.error.URLError,TimeoutError,OSError) as error:
            raise PoseError("service_unavailable") from error
        try:
            if len(raw) > 1024*1024:
                raise PoseError("invalid_response")
            result = json.loads(raw)
            if not isinstance(result,dict):
                raise PoseError("invalid_response")
            if result.get("ok") is not True:
                raise PoseError(result.get("error","invalid_response"))
            expected = {"version":1,"algorithm":"foundationpose","sdk_commit":SDK_COMMIT,"capture_id":capture_id,
                        "camera_id":camera_id,"model_id":model_id,"translation_unit":"m",
                        "camera_axes":"opencv","status":"estimated",
                        "request_sha256":hashlib.sha256(body).hexdigest()}
            if any(result.get(k) != v for k,v in expected.items()):
                raise PoseError("provenance_mismatch")
            validate_pose(result["T_camera_object"])
            validate_pose(result["T_source_canonical_m"])
            return result
        except (KeyError,TypeError,ValueError) as error:
            if isinstance(error,PoseError):
                raise
            raise PoseError("invalid_response") from error

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luna_mujoco.pose_estimation import CadPoseEstimator
from tianji_foundationpose.contracts import PoseError


class PoseClient:
    def __init__(self):
        self.calls = 0
    def estimate(self, *args, **kwargs):
        self.calls += 1
        return {"T_camera_object":np.eye(4).tolist(),"T_source_canonical_m":np.eye(4).tolist(),
                "capture_id":kwargs["capture_id"],"camera_id":kwargs["camera_id"]}

class LunaTest(unittest.TestCase):
    def view(self):
        transform = np.eye(4)
        transform[0,3] = 2
        return {"rgb":np.zeros((10,10,3),np.uint8),"depth_m":np.ones((10,10),np.float32),
                "valid_depth":np.ones((10,10),bool),"intrinsics":{"fx":100,"fy":100,"cx":5,"cy":5},
                "frame_id":"f1","camera_id":"1","capture_time":1.0,"simulation_generation":2,
                "calibration_id":"cal1","T_world_camera":transform.tolist()}

    def test_world_pose_uses_optical_camera_calibration(self):
        client = PoseClient()
        result = CadPoseEstimator(pose_client=client).estimate_instance(self.view(),
            np.ones((10,10),bool),.9,model_id="blue_box",mask_capture_id="f1",mask_camera_id="1")
        self.assertEqual(result["T_world_object"][0][3],2)
        self.assertEqual(result["usage"],"pose_hypothesis_only")
        self.assertEqual(client.calls,1)

    def test_foreign_capture_rejected_before_network(self):
        client = PoseClient()
        with self.assertRaises(PoseError):
            CadPoseEstimator(pose_client=client).estimate_instance(self.view(),
                np.ones((10,10),bool),.9,model_id="blue_box",mask_capture_id="f2",mask_camera_id="1")
        self.assertEqual(client.calls,0)

    def test_source_origin_is_transformed_through_predicted_rotation(self):
        class RotatedClient:
            def estimate(self,*args,**kwargs):
                pose = [[0,-1,0,.1],[1,0,0,.2],[0,0,1,.3],[0,0,0,1]]
                origin = [[1,0,0,.05],[0,1,0,.02],[0,0,1,0],[0,0,0,1]]
                return {"T_camera_object":pose,"T_source_canonical_m":origin,
                        "capture_id":kwargs["capture_id"],"camera_id":kwargs["camera_id"]}
        result = CadPoseEstimator(pose_client=RotatedClient()).estimate_instance(self.view(),
            np.ones((10,10),bool),.9,model_id="blue_box",mask_capture_id="f1",mask_camera_id="1")
        np.testing.assert_allclose(np.asarray(result["T_world_source"])[:3,3],[2.12,.15,.3])

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from luna_mujoco.pose_estimation import CadPoseEstimator
from tianji_foundationpose.client import FoundationPoseClient


class SelectionTest(unittest.TestCase):
    def test_default_backend_is_foundationpose(self):
        with patch.dict("os.environ",{},clear=True):
            self.assertIsInstance(CadPoseEstimator().pose_client, FoundationPoseClient)

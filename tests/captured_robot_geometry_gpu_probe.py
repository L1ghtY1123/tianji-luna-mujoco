import json
from pathlib import Path

import numpy as np
import torch
import yaml
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import DeviceCfg
from tianji_curobo.curobo_robot_config import create_dual_arm_robot_config
from tianji_curobo.curobo_service import materialize_urdf

from luna_mujoco.carrying import configure_gripper_feedback
from luna_mujoco.geometry_service import ROOT, sdk_identity
from luna_mujoco.perception_mapping import create_feedback_segmenter

arms=json.loads(Path('/tmp/run4-capture-feedback.json').read_text())
feedback={arm:arms[arm]['gripper_position_rad'] for arm in ('A','B')}
urdf=materialize_urdf(ROOT/'assets/marvin_urdf/marvin_with_grippers.urdf',marvin_meshes=ROOT/'assets/marvin_urdf/meshes',rag_meshes=ROOT/'assets/marvin_urdf/rag100_050/meshes')
segmenter=create_feedback_segmenter(robot_config_path=ROOT/'resources/curobo/marvin_gui_dual.yml',urdf_path=urdf,asset_root_path=urdf.parent,feedback=feedback)
q={arm:np.rad2deg(arms[arm]['joints_rad']).tolist() for arm in ('A','B')}
captured=np.asarray(segmenter.sdk_boundary.robot_spheres(q))
robot=create_dual_arm_robot_config(yaml.safe_load((ROOT/'resources/curobo/marvin_gui_dual.yml').read_bytes()),urdf_path=urdf,asset_root_path=urdf.parent)
robot=configure_gripper_feedback(robot,None,urdf)
checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=robot,device_cfg=DeviceCfg(),scene_model={'cuboid':{}},collision_activation_distance=.01))
both={f'Joint{i}_'+('L' if arm=='A' else 'R'):v for arm in ('A','B') for i,v in enumerate(arms[arm]['joints_rad'],1)}
position=torch.as_tensor([[[both[name] for name in checker.kinematics.joint_names]]],**checker.device_cfg.as_torch_dict())
actual=checker.get_kinematics(position).robot_spheres[0,0].detach().cpu().numpy()
report={'sdk':sdk_identity()[0],'feedback':feedback,'sphere_counts':[len(captured),len(actual)],'links':{}}
for link in ['gripL_Link3_L','gripL_Link3_R','gripL_Link4_L','gripL_Link4_R','Link1_L']:
 ids=checker.kinematics.kinematics_config.get_sphere_index_from_link_name(link).detach().cpu().numpy()
 center=np.linalg.norm(actual[ids,:3]-captured[ids,:3],axis=1)
 radius=np.abs(actual[ids,3]-captured[ids,3])
 report['links'][link]={'indices':ids.tolist(),'max_center_delta_m':float(center.max()),'max_radius_delta_m':float(radius.max()),'first_captured_sphere':captured[ids[0]].tolist(),'first_fixed_open_sphere':actual[ids[0]].tolist()}
print(json.dumps(report,indent=2),flush=True)

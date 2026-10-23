import json
import time

from tianji_curobo.curobo_service import materialize_urdf

from luna_mujoco.geometry_service import ROOT, GeometryRuntime, run_operation, sdk_identity

folder=ROOT/'data/visible-cloud-work/run5-replay'
mapping=json.loads((folder/'007-perception-build.json').read_text())
validation=json.loads((folder/'014-carrying-validate.json').read_text())
urdf=materialize_urdf(ROOT/'assets/marvin_urdf/marvin_with_grippers.urdf',marvin_meshes=ROOT/'assets/marvin_urdf/meshes',rag_meshes=ROOT/'assets/marvin_urdf/rag100_050/meshes')
runtime=GeometryRuntime(urdf_path=urdf,asset_root_path=urdf.parent)
scene=run_operation(runtime,mapping['path'],mapping['body'])
assert scene['scene_token']==validation['body']['scene_token']
started=time.perf_counter()
result=run_operation(runtime,validation['path'],validation['body'])
import numpy as np

from luna_mujoco.perception_mapping import decode_mapping_payload

views=decode_mapping_payload(mapping['body'])[0]['views']
table=scene['table']
center,half=np.asarray(table['pose'][:3]),np.asarray(table['dims'])/2
returns=[]
for camera,view in views.items():
 depth=np.asarray(runtime.perception_mapper.capture_segmenter.filtered_depths[camera])
 rows,cols=np.where((np.asarray(view['valid_left_mask'])!=0)&(depth>0))
 k=np.asarray(view['left_intrinsics']); z=depth[rows,cols]
 local=np.column_stack(((cols-k[0,2])*z/k[0,0],(rows-k[1,2])*z/k[1,1],z))
 transform=np.asarray(view['T_base_left_camera'])
 world=local@transform[:3,:3].T+transform[:3,3]
 keep=(world[:,0]>=center[0]-half[0])&(world[:,0]<=center[0]+half[0])&(world[:,1]>=center[1]-half[1])&(world[:,1]<=center[1]+half[1])&(world[:,2]>=center[2]-half[2])
 returns.extend(world[keep,2].tolist())
report={'table_cuboid':table,'table_bounds_min_m':(center-half).tolist(),'table_bounds_max_m':(center+half).tolist(),'native_voxel_models':runtime.geometry_model['voxel'],'nonrobot_tabletask_return_count':len(returns),'nonrobot_tabletask_return_height_quantiles_m':np.quantile(returns,[0,.5,.9,.99,1]).tolist() if returns else [],'sdk':sdk_identity()[0],'scene_token':scene['scene_token'],'phase':result['phase'],'checked_points':result['checked_point_count'],'collision_free':result['collision_free'],'failure_reason':result['failure_reason'],'wall_s':time.perf_counter()-started,'proofs':result['observation_coverage']}
runtime.clear_carrying()
print(json.dumps(report,indent=2),flush=True)

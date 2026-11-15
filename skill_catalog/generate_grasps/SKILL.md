# generate_grasps

For CAD use the execution-arm offline library with current FoundationPose and zero additional insertion by default. For unknown objects feed the current published TSDF surface point cloud to GraspGen and request one native 100-candidate pool per round. GraspGen model contact TCP is converted to the actual executing-arm finger-front TCP, with no additional insertion. Left arm A and right arm B have explicit distinct orientation transforms; the service model frame and 195 mm base-to-contact offset are checked. Robot mount-to-front remains 185 mm. The graph may request one additional round after confirmed exhaustion. Remain bound to the same qualified execution arm, target, capture and scene.

参数、前提、效果、绑定和结果以同目录 contract.json 为准。所有跨技能恢复显式出现在 graph 中。

GraspGen 候选先按真实末端 +Z 在世界坐标中的朝向筛选，允许世界 +X 前方至 −Z 下方的扇区，偏离世界 XZ 平面的左右角度默认最多 30°，控制台可调范围 0–45°。CAD 使用离线抓姿。规划完成后对实际抓取端点 FK 再核对相同朝向条件。抓取 Lift 显式指定世界 +Z 150 mm，并用公开 FK 检查每个规划插值点的横向偏移和提升高度，采用已有 IK 位置容差 5 mm。方向筛选证据、逐候选通过结果和 Lift 位移核验随规划归档。Lift 完成并通过持物核验后建立 attachment。

所有 GraspGen 抓取按执行臂当前指端 TCP 选择较近的平行夹爪半转等价姿态，随后进入 IK 和完整碰撞检查。CAD 离线抓姿按库中 A/B 归属读取。

GraspGen model contact TCP is aligned with the actual gripper_front_A/B, with distinct left/right orientation transforms and zero extra insertion. Check the declared Robotiq model and base-to-contact 195 mm conversion; actual mount-to-front offset is 185 mm. Old grasp_insertion_m settings are retired. Keep forward/downward +Z sector and all actual-gripper geometry/collision checks.

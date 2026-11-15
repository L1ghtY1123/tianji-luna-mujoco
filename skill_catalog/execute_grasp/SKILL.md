# execute_grasp

Execute the bound approach, grasp, closure and Lift sequence with the detached grasp-planning robot and open-gripper model. Command full close at grasp. Validate the actual sequence, full closure range, selected-target contact links, environment, peer arm and self-collision. Retain provisional holding evidence after Lift. verify_grasp confirms holding, inherits the target ID from the bound continuous execution, and creates attachment for transport and placement.

参数、前提、效果、绑定和结果以同目录 contract.json 为准。所有跨技能恢复显式出现在 graph 中。

GraspGen 候选先按真实末端 +Z 在世界坐标中的朝向筛选，允许世界 +X 前方至 −Z 下方的扇区，偏离世界 XZ 平面的左右角度默认最多 30°，控制台可调范围 0–45°。CAD 使用离线抓姿。规划完成后对实际抓取端点 FK 再核对相同朝向条件。抓取 Lift 显式指定世界 +Z 150 mm，并用公开 FK 检查每个规划插值点的横向偏移和提升高度，采用已有 IK 位置容差 5 mm。方向筛选证据、逐候选通过结果和 Lift 位移核验随规划归档。Lift 完成并通过持物核验后建立 attachment。

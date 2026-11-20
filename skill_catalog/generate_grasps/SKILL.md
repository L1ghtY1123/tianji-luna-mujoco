# generate_grasps

For CAD use the execution-arm offline library with current FoundationPose and zero additional insertion by default. For unknown objects feed the current published TSDF surface point cloud to GraspGen and request one native 100-candidate pool per round. GraspGen model contact TCP is converted to the actual executing-arm finger-front TCP, then advanced along its local +Z by the configured Tianji insertion (default 10 mm, range 0–20 mm). Screen the actual RAG opening and bilateral pad support against the same TSDF before IK. Prefer top entry over a measured bin floor. Try the narrower projected jaw openings first within each approach group. Left arm A and right arm B have explicit distinct orientation transforms; the service model frame and 195 mm base-to-contact offset are checked. Robot mount-to-front remains 185 mm. Each graph generation node requests one pool. The graph may request one additional pool after confirmed exhaustion on the same input. plan_grasp only evaluates its supplied pool. Remain bound to the same qualified execution arm, target, capture and scene.

参数、前提、效果、绑定和结果以同目录 contract.json 为准。所有跨技能恢复显式出现在 graph 中。

GraspGen 候选先转换到实际 Tianji TCP。保留末端 +Z 从机器人正前到正下的整个 90° 扇区，允许相对前下平面的左右偏角至 45°。夹爪绕自身 +Z 的半转不改变接近方向。CAD 使用离线抓姿。可行性由 cuRobo 的 IK、机器人自身及环境碰撞、实际 FK 与目标位姿误差检查决定。抓取 Lift 仍沿世界 +Z 150 mm，并用公开 FK 检查插值点的横向偏移和提升高度。Lift 完成并通过持物核验后建立 attachment。

所有 GraspGen 抓取按执行臂当前指端 TCP 选择较近的平行夹爪半转等价姿态，随后进入 IK 和完整碰撞检查。CAD 离线抓姿按库中 A/B 归属读取。

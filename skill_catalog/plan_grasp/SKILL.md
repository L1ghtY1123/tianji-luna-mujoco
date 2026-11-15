# plan_grasp

Use official cuRoboV2-Fast plan_grasp with at most the configured grasp_goalset_size ranked candidates per request. Approach, grasp and Lift use the detached grasp-planning robot and open-gripper model. Check the full authored closure range separately. Commands only request full open or full close; reject colliding candidates without a bounded partial-close fallback. Validate the robot, selected-target contact links, environment, peer arm and self-collision. Remove numerically rejected candidates and continue within the frozen pool. Stop at the first fully validated grasp. Attachment is created by verify_grasp after Lift and target-held confirmation. Command checking stops at a proven invalid chunk and requires complete checking for success.

参数、前提、效果、绑定和结果以同目录 contract.json 为准。所有跨技能恢复显式出现在 graph 中。

Report success_count only after command validation. Record solver_success_count separately. request_index counts requests and candidate_group_index preserves the original configured candidate grouping across rejection retries.

GraspGen 候选先按真实末端 +Z 在世界坐标中的朝向筛选，允许世界 +X 前方至 −Z 下方的扇区，偏离世界 XZ 平面的左右角度默认最多 30°，控制台可调范围 0–45°。CAD 使用离线抓姿。规划完成后对实际抓取端点 FK 再核对相同朝向条件。抓取 Lift 显式指定世界 +Z 150 mm，并用公开 FK 检查每个规划插值点的横向偏移和提升高度，采用已有 IK 位置容差 5 mm。方向筛选证据、逐候选通过结果和 Lift 位移核验随规划归档。Lift 完成并通过持物核验后建立 attachment。

Unknown-object candidates already use the selected arm front TCP with zero additional insertion. Preserve endpoint +Z forward/downward sector and actual robot contact, closure, environment and self-collision checks.

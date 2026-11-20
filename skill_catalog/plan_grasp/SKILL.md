# plan_grasp

抓取从目标沿末端 −Z 后退 150 mm 处接近，沿末端 +Z 进给，Lift 沿固定世界 +Z 提升 150 mm。

Use official cuRoboV2-Fast plan_grasp with at most the configured grasp_goalset_size ranked candidates per request. Approach, grasp and Lift use the detached grasp-planning robot and open-gripper model. Check the full authored closure range separately. Commands only request full open or full close; reject colliding candidates without a bounded partial-close fallback. Validate the robot, selected-target contact links, environment, peer arm and self-collision. Remove numerically rejected candidates and continue within the frozen pool. Stop at the first fully validated grasp. Attachment is created by verify_grasp after Lift and target-held confirmation. Command checking stops at a proven invalid chunk and requires complete checking for success.

参数、前提、效果、绑定和结果以同目录 contract.json 为准。所有跨技能恢复显式出现在 graph 中。

Report success_count only after command validation. Record solver_success_count separately. request_index counts requests and candidate_group_index preserves the original configured candidate grouping across rejection retries.

GraspGen 候选先转换到实际 Tianji TCP。保留末端 +Z 从机器人正前到正下的整个 90° 扇区，允许相对前下平面的左右偏角至 45°。夹爪绕自身 +Z 的半转不改变接近方向。CAD 使用离线抓姿。可行性由 cuRobo 的 IK、机器人自身及环境碰撞、实际 FK 与目标位姿误差检查决定。抓取 Lift 仍沿世界 +Z 150 mm，并用公开 FK 检查插值点的横向偏移和提升高度。Lift 完成并通过持物核验后建立 attachment。

Plan the frozen Tianji TCP poses supplied by generate_grasps. Preserve the applied insertion and orientation. Validate the selected target against all robot links except the executing left/right Link3 and Link4 links during permitted contact phases, and check both fingers against the rest of the scene.

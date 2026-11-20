# execute_grasp

Enable the selected arm once for pre-opening, approach, closure and Lift. Disable after the complete sequence or failure. Execute with the detached grasp-planning robot and open-gripper model. Command full close at grasp. Validate the actual sequence, full closure range, selected-target contact links, environment, peer arm and self-collision. Retain provisional holding evidence after Lift. verify_grasp confirms holding, inherits the target ID from the bound continuous execution, and creates attachment for transport and placement.

参数、前提、效果、绑定和结果以同目录 contract.json 为准。所有跨技能恢复显式出现在 graph 中。

GraspGen 候选先转换到实际 Tianji TCP。保留末端 +Z 朝机器人前方或下方的姿态，排除向后、向上和以横向为主的姿态。夹爪绕自身 +Z 的半转不改变接近方向。CAD 使用离线抓姿。可行性由 cuRobo 的 IK、机器人自身及环境碰撞、实际 FK 与目标位姿误差检查决定。抓取 Lift 仍沿世界 +Z 150 mm，并用公开 FK 检查插值点的横向偏移和提升高度。Lift 完成并通过持物核验后建立 attachment。

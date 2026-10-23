# plan_place

粉件保持抓取旋转，夹爪前端对准固定共享区域中心XY=[0.50160,0.11646]m，无分区。SDK目标用标定前端偏移转成法兰；不以点云中心对准、不要求整个附件投影包含。Z根据attachment与当前夹爪及实际世界碰撞选取尽量靠近10mm间隙的高度。直接官方plan_grasp派生plan_place，不使用plan_place_batch。前两段闭爪attachment，开爪后detach撤离；仅当前闭合/全开端点检查，无开爪扫掠。遵循高级设置，版本不符、无解、碰撞或反馈不满足时停止。

前提：needs_place=true, grasp_held=true, target_selected=true, destination_selected=true, scene_ready=true, place_plan_ready=false, released=false, recovery_needed=false。

输出事实：place_plan_ready。完整原始输入、视觉请求、版本及失败证据保存回放；unknown不作为成功。

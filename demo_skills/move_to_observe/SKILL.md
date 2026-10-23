# move_to_observe

双臂沿最近保留场景回固定初始观测位姿，遵循10mm碰撞检查和高级时间设置，到位采三RGBD，不自动建图。用于每个粉件前、最后黑件入盒后的收尾；verify_unpack内部也必须回位。最后成功回位才报告任务完成。已绑定抓取视图后不得重复运动使视图失效。

前提：{"scene_available": true, "grasp_held": false}。

输出：observation_pose_ready, grasp_view_ready。

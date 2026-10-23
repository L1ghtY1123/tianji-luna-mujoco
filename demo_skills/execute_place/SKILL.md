# execute_place

执行已验证的带物接近、放下轨迹。松爪前检查实际关节、夹爪、attachment、目标区域与10mm碰撞阈值；开爪确认、detach、阶段等待后沿设定轴lift。这里不采结果图、不分割或建立粉件局部障碍。lift完成后由verify_place调用Decisions；只有放置已确认才用同次RGBD建立粉件局部障碍，之后才能移动到VLA位姿。顺序不可提前。

前提：needs_place=true、grasp_held=true、place_plan_ready=true、released=false、recovery_needed=false。

输出：place_execution_completed。

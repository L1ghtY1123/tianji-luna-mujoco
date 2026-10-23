# execute_place

执行已验证的带物接近、放下轨迹。松爪前检查实际关节、夹爪、attachment、目标区域与10mm碰撞阈值；开爪确认、detach、阶段等待后沿设定轴lift。这里不采结果图、不分割或建立粉件局部障碍。lift完成后由verify_place调用Decisions；只有放置已确认才用同次RGBD建立粉件局部障碍，之后才能移动到VLA位姿。顺序不可提前。

前提：needs_place=true、grasp_held=true、place_plan_ready=true、released=false、recovery_needed=false。

输出：place_execution_completed。

放置阶段的1秒按墙钟计时，检查包含在停留预算内，不乘时间伸长倍率；检查超时则记录overrun_s并等待检查完成，绝不跳过碰撞检查。检查器按几何结构缓存，将两夹爪驱动关节保留为官方FK的活动输入，每次传入实际角度；不反复重建锁定关节模型。附件、场景或净距改变仍区分模型。执行检查只上传所需世界，不刷新未参与检查的IK/规划器几何。

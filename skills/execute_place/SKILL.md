# execute_place

**目的与调用时机：**执行持物 approach/place、核对释放端点、开爪 detach 后空手 retreat。

**输入与前提：**已验证 holding/attachment、place_plan_ready、有效场景且未释放；计划必须完整包含三段并与原持物几何、场景/帧绑定一致。

**固定算法：**持物 approach/place → 停留 → 实测前端 TCP/球底到位检查 → 实测完整构型碰撞检查 → 开爪 → 确认张开 → detach → 复核并执行空爪 retreat。目标底面保留完整 hard clearance；默认最小10 mm、规划12 mm、实际允许10–14 mm。2 mm为已有到位容差，不是降低安全间距。前端 TCP 三维位置及 XY 误差≤2 mm、姿态误差≤2°、实际球底与目标间距差≤2 mm。松爪前使用同一次实测反馈，对当前双臂关节、两侧实际夹爪和原 attachment 进行完整三维碰撞复核；失败保持夹持，不开爪、不detach。无需 attachment XY 投影包含约束，不增加盒沿/体素补偿。

**输出与状态：**place_execution_completed、released 与阶段记录；不采结果 RGBD，不生成视觉落点证据。

**失败含义与边界：**端点未达标保留夹持返回错误；开爪、detach 或撤离失败说明具体阶段，不能把部分执行当完整成功。实际释放后可能已有状态变化，恢复须使用当前反馈。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。

持物执行的approach入口和place阶段边界分别读取robot_snapshot，以当前夹爪稳定窗口确认held；空夹爪或未知反馈立即停止，并保存feedback_id、时刻、阶段和原始判定，attachment保留。碰撞复核使用两臂当前实测夹爪角度，不用抓取时角度覆盖。绑定的arm、目标ID、generation、scene token和owner必须一致。张开反馈确认后才detach；detach失败保留已释放与detach待完成状态，不撤离；未确认张开不detach，撤离失败也不撤销已经释放的事实。
最终指令的 world cost 使用与既有轨迹检查一致的1e-9数值容差；自碰撞和关节边界检查不变。

**停留计时：**停留参数按实际单调时钟计时，不按仿真时间计时，也不乘 time_scale；设置1 s表示实际等待约1 s。返回 duration_s 为实际等待秒数，simulation_duration_s 单独记录期间仿真推进量。轨迹仍按原仿真时间积分执行，time_scale=3不变；停留前后的夹爪、到位和碰撞检查耗时不包含在停留值中。

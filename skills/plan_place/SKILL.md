# plan_place

**目的与调用时机：**确认持物后准备 attachment，以指定分区中心作为 TCP 世界 XY，搜索完整放置三段。

**输入与前提：**verified holding、scene、destination_id、planning_settings，以及冻结的全局搬运总数 N、object_id/object_display_id、region_id/slot_id、inventory_revision/layout_revision。

**固定算法：**prepare_attachment → 读取冻结区域和指定 slot 的实测底面 → 生成 18 个世界 Z 轴 yaw → 共享 batched IK100 → 每批16个候选规划持物 approach/place → 按偏航变化、关节路程、候选编号检查实际指令及张开无 attachment 的 retreat → 返回首个完整可行方案。规划不移动机器人或松爪。

抓取姿态基准 Rg 取闭爪完成、lift 之前的实际 TCP，记录 arm、tool_profile_revision、feedback_id、feedback_time 及 reference_phase=after_close_before_lift。中转重新抓取时覆盖为该次基准。每个候选的旋转 R=Rz(yaw) @ Rg；yaw 为0°至340°、步长20°，不额外估计物体抓前六自由度姿态。

TCP 是两指前端中心，沿模型安装端 TCP_Link 的局部 Z 偏移185 mm。应用目标先固定前端 TCP 世界 XY=slot 中心，再换算安装端目标：p_mount.xy=slot.xy-(R*[0,0,0.185]).xy。内部兼容字段 T_world_tcp、attachment 球坐标及 SDK 求解仍使用安装端参考系；不把两者混称为同一原点。R=Rz(yaw)@R_grasp，yaw绕世界Z，18个方向。安装端 Z=floor_z+(hard_clearance+0.002)-min_i((R*c_i).z-r_i)，前端 Z 由同一刚体变换确定。 目标底面保留完整 hard clearance；默认最小10 mm、规划12 mm、实际允许10–14 mm。2 mm为已有到位容差，不是降低安全间距。前端 TCP 三维位置及 XY 误差≤2 mm、姿态误差≤2°、实际球底与目标间距差≤2 mm。松爪前使用同一次实测反馈，对当前双臂关节、两侧实际夹爪和原 attachment 进行完整三维碰撞复核；失败保持夹持，不开爪、不detach。无需 attachment XY 投影包含约束，不增加盒沿/体素补偿。

区域和分区只确定 TCP XY；不要求 attachment 球体的 XY 投影处于区域或分区内，不做二维 footprint 占用淘汰。盒壁、已放物体及其他障碍保留在真实三维碰撞场景中，由 attachment/机器人与场景碰撞检查判断可行性，不抬高释放位置绕过碰撞。

N 为冻结清单的全局 required 总数，不计障碍、盒子或区域；多目的地仍保留同一总数和对象编号。K=2×ceil(N/2)，N=1也分两区。区域中心成对直线等面积划分；非对称边界只取其内部对称区域。方向对齐初始头部相机水平轴的平面投影，各分区按初始头部投影行列编号，物体i只用slot i。首次 identify_destination 冻结区域及 layout；完成、暂缓或遮挡不重新划分。

保存 region_id、support_entity_id、原始观测、平面和布局版本。区域边界与独立盒壁点云共同确认平移（双向95%在2 mm内，盒壁覆盖两轴及高度，并排除静止假设）才更新位置。局部底面切换、仅见平行墙、未测 marker 位姿或旋转/形变证据不足均不推断运动。

盒底无需显式 marker；由水平底面及更高盒壁实测点确定底板平面，取实测平面凸包内的内接矩形，采样间距≤15 mm、每轴网格≤256、最小边长30 mm。采样密度变化和局部遮挡不切碎底板，障碍物由三维场景检查。首次 plan_place 使用三相机几何划分并冻结布局，按初始头部视角编号；初始头部观测不提前冻结部分可见区域。区域冻结后，用冻结区域内的当前底面点检查指定 slot 支撑；新视角选出的另一块矩形不裁掉这些点。缺少当前底面测量返回 placement_slot_support_unobserved，不换区。

**输出与状态：**place_plan 三段、18个候选的固定 TCP XY/yaw、球底目标间距及选择证据；实际 attachment 不改变。candidate_count 统计真实生成候选，计算补齐项不参与计数或选优。每批至多16个真实候选，不足时复用末项补齐；IK通过后才创建持物轨迹配置。IK通过不代表整条路径可行，接近/放置通过也必须检查空爪撤离。

**失败含义与边界：**区域缺失、slot底面未测、attachment准备失败、IK无解、轨迹或碰撞失败分别保留原阶段证据；18个yaw耗尽返回失败，不换区。算法无解不证明区域超出工作空间。服务异常立即返回。目标底面保留完整 hard clearance；默认最小10 mm、规划12 mm、实际允许10–14 mm。2 mm为已有到位容差，不是降低安全间距。前端 TCP 三维位置及 XY 误差≤2 mm、姿态误差≤2°、实际球底与目标间距差≤2 mm。松爪前使用同一次实测反馈，对当前双臂关节、两侧实际夹爪和原 attachment 进行完整三维碰撞复核；失败保持夹持，不开爪、不detach。无需 attachment XY 投影包含约束，不增加盒沿/体素补偿。名义间距、请求间距与实际反馈分开记录。

本技能及持物恢复不建立场景。JEV 在 skill 外选择步骤和恢复，技能内部不再请求 JEV。共享批次规则见 src/jev_mujoco/planning_batches.py；完整定义见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。
释放端点的 world cost 使用与既有轨迹检查相同的 1e-9 数值容差，消除浮点舍入残差；不改变以米计的 hard clearance 或目标 Z。
项目侧IK独立hard checker使用公开RobotCollisionChecker接口，将非负且≤1e-9的activation cost视为数值零，与轨迹检查一致；该值不是米制间距补偿。SDK位姿收敛和可行性判断保持原配置。

2026-10-19 放置准备去重：一次静止的 plan_place 调用在准备 attachment 前读取一份实测快照，持物状态判定、prepare_carrying、后续 IK 和放置轨迹规划共用该快照，避免夹爪反馈微小变化立即废弃刚初始化的持物模型。执行前和松爪前仍独立读取最新反馈并进行原有碰撞检查；不量化夹爪角度，不复用旧碰撞结论，不改变 batched IK 或 plan_place_batch 算法。

# TSDF/ESDF、放置 Lift 与拆包核验审计

本次修改项目封装，未修改 cuRobo SDK 或 vendor。不能把“使用官方算法”写成“项目所有逻辑与官方示例完全一致”。本轮实测通过粉件放置、拆包补充观测及确认；完整两件任务仍未通过，停止于第一件黑件闭爪后的净距检查。

## 官方版本与调用边界

锁定 cuRobo `8e734f3ced1df898990bcd92de40abce475907db`，运行服务 health 返回同一版本。参照该版本的 [Mapper 官方示例](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/examples/getting_started/volumetric_mapping.py) 及只读的 MotionPlanner、ToolPoseCriteria、SeedIK 源码。沿用之前的 SDK 文件校验，当前 diff 未涉及 SDK/vendor。

| 环节 | 使用方式与项目边界 |
|---|---|
| TSDF | 官方 Mapper.integrate(CameraObservation)，三相机标定、语义 mask 和坐标变换为项目输入 |
| ESDF | 官方公开高级接口 Mapper.integrator.compute_esdf/get_voxel_grid；未重写距离传播 |
| 地图分辨率 | 主流程原已是 TSDF=0.02m、ESDF=0.02m；本轮核验构造、缓存及实际结果，没有把原本2cm声称为新改动 |
| 桌子/盒子/局部物体 | 首次实测桌面 cuboid 复用；盒壁先验补全和局部物体包围体是明确的项目几何输入，不冒称官方TSDF产物 |
| plan_place 接近/放下 | 官方 MotionPlanner.plan_grasp(plan_grasp_to_lift=False)，使用实际闭爪和attachment |
| plan_place Lift | 换为全开、无attachment的模型，使用官方 plan_grasp 的公开 linear_motion + plan_pose 组合，并保留放下末端原生 JointState 信息 |
| 最终命令检查 | 官方 RobotCollisionChecker，检查实际闭爪/attachment/双臂及五次插值后的命令；10mm净距不变 |

Demo 传入 table_model=None，不执行历史兼容路径的 crop_tsdf_to_table。语义删除发生在深度输入侧；桌面不重新体素化。旧兼容代码仍在只读 vendor 中，不能泛称仓库每条历史路径都没有项目后处理。以前报告过的原生建图退出139，本次未复现，也未声称已修复。

## 放置轨迹缺陷

旧封装只传放下末端关节位置，重新构造 JointState 时丢失 dt。官方 plan_grasp 的 Lift 会保留优化后的末端 JointState；SeedIK 仅在 dt 存在时启用速度相关的种子限制。丢失 dt 改变了求解条件，可选到远处关节分支。

修复跨接口传递优化轨迹末端的 dt、velocity、acceleration，正常 pose 调用不受影响；异常分支也清理临时状态。dt 不使用控制周期替代、不乘时间伸长。原失败记录的75mm Lift 最大关节变化约128.84°。新实测 Lift：

- TCP 上升74.99984mm，轨迹中最大横向偏移0.7963mm，最大姿态变化0.08294°。
- 最大关节范围8.5499°，仿真执行约0.9秒。
- 这是一次实测结果，不能保证所有起点的冗余机械臂都不换分支；原失败实验未保存完整原生轨迹和输入图，无法原样重放。

旧实现把检查耗时与完整1秒等待串联。现检查计入对应停留的墙钟预算；若超过预算仍等待检查完成并记录 overrun_s，绝不为了计时绕过检查。检查器将两夹爪驱动角直接作为官方FK输入，按附件/场景/配置缓存，消除了细微反馈变化导致的反复模型构建。验证执行命令时也不更新无关的IK/规划器几何。

新实测接近后停留1.003秒、放下后开爪前1.002秒、释放后1.108秒（含调度开销）。注意现有高级设置分别配置“开爪前停留”和“释放后等待”，二者之间还有真实开爪动作1.441秒；不能把从放下结束直到Lift开始的整个区间声称为1秒。时间伸长只作用于运动轨迹。

## 原拆包报错节点

原用户实验 `37845fa4f0d34f37b1a371abfe743bcf` 第二件在回观测位姿后的 verify_unpack 停止：unpack_result=1.00，wrapper_removed=0.94，observation_sufficient=0.23。后者属于 unknown，并非API报错。旧充分性问题额外要求判断“全部条件”、身份及历史证据限制，与节点允许的 BEFORE/CURRENT 比较范围存在冲突。

现在充分性问题逐节点列出条件，不自我判断，不增加跨外观身份、完整轮廓或全相机可见要求；保留≥0.8/≤0.2阈值和一次补充观测上限。三相机仍合并一个请求，不能仅因结果概率高就覆盖充分性 unknown。

新实测当前VLA位置证据不足；按规则回观测位姿后，三个概率分别为1.00、1.00、0.95，核验通过，再重建全局2cm场景。尚未完成第二件验收，不能声称旧失败的原图已重新验证。

## 验证及剩余问题

新增 `scripts/audit_mapping_and_checker.py` 在Ubuntu GPU离线运行：固定三相机合成深度下，封装与直接官方调用的64,000个ESDF值完全相同；A/B两臂七阶段×两种反馈，共28组动态FK与新建锁定关节检查器对比通过，最大球坐标/半径误差4.47e-7m。该测试不证明真实遮挡区域的几何完整性。

本轮33个相关测试文件最终386项通过（23.75秒），涵盖建图配置/缓存、原生末端状态、停留预算、夹爪/附件、放置和补充观测。Ruff检查全部21个新增/修改Python文件通过，git diff --check通过。两个旧融合请求测试漏列既有observation_sufficient条件，按实际契约修正，并新增明确作用范围的断言。未重跑全库；上一轮记录的其他基线失败不因此视为已修复。

新实测归档 `7b2684b63974475aa5f3058ac94605bb`，名称 `audit-1023-native-lift-state-two-items`。总时长196.802秒，停在第一件黑件 execute_grasp / grasp_lift 起点。独立MuJoCo FK确认，指端球到实测桌面净距：开爪25.40mm、实际闭爪8.81mm，低于10mm。官方抓取是在开爪模型下规划；闭爪后模型改变，当前候选因此被执行检查拒绝。没有删检查、降低净距或擅自取消用户要求的深入1cm。该问题仍待解决，不能称为全流程完成或三分钟目标达成。

| 实测阶段 | 秒 |
|---|---:|
| 初始 observe（含Responses与语义处理） | 50.46 |
| 首次/第二次 build_scene | 4.98 / 1.50 |
| 粉件 GraspGen / batched IK / grasp规划 | 12.59 / 2.42 / 14.70 |
| attachment / place规划 | 3.85 / 4.31 |
| 两次拆包判断 | 1.33 / 1.39 |
| 黑件 observe / GraspGen / batched IK / grasp规划 | 6.93 / 0.62 / 3.54 / 20.48 |

测试过程和原始报告位于 `data/mapping-place-unpack-audit-20261023/`，未上传实验图片、深度或运行数据。CPU配额保持web=2、cuRobo=1、SAM3/GraspGen各0.5。服务使用挂载代码，已重启cuRobo并完成13配置启动预热（111.716秒），没有重建镜像。

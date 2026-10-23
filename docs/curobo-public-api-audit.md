# cuRobo 调用审计（2026-10-23）

## 本轮发现与边界

本轮287项回归、56组GPU几何对照、13配置启动预热及未验收边界，见[2026-10-23验证记录](verification/2026-10-23-planner-call-and-geometry.md)。后文2026-10-22的时间/实验结果保留其历史范围。

先前“所有功能完全遵循官方”表述不准确。官方SDK源码一致、使用公开API、项目调用流程和官方示例一致，是三个不同结论。下面列出的项目后处理不是官方算法内部修改，但也不能描述成只有高级参数与插值。

- 抓取IK和批量规划请求曾漏传夹爪反馈；`fixed_open`检查还把两侧都当成全开。现在必须提交A/B实测开度，只把执行臂设为张开，保持对侧实测状态。
- 官方抓取三段共享规划时的几何。执行闭爪后，原lift轨迹以实际闭合模型检查；此时无attachment，符合用户要求的lift后才附物。没有重写官方三段式算法。
- 删除单臂原始规划轨迹的额外整段检查，保留最终五次插值命令的10mm硬净距验证。执行阶段只复用完整请求哈希一致的结果；新增起点连接、对侧关节、夹爪、附件或场景改变都会重新检查。不能承诺不再发生执行前检查。
- `plan_place`正常无解曾销毁已预热compound规划器；已改为保留。SDK异常仍清理异常实例。vendor保留的单目标抓取方法仍有旧的失败清缓存行为，但本服务未注册`/tools/plan_grasp`，当前demo使用`plan_grasp_batch`。它不是已经修复或开放的功能。
- 双臂移动仍有服务端和客户端的最终命令验证；支撑法向撤离兼容路径使用公开criterion参数及分段规划，当前demo运输不进入此分支。这些都不能描述为官方示例原样执行。SDK默认配置与项目配置亦不相同。

| 阶段 | 执行手夹爪几何 | attachment | 对侧手 |
|---|---|---|---|
| 抓取IK、接近、下降 | 完全张开 | 无 | 实测开度及关节 |
| 抓取lift | 实测闭合，检查原lift命令 | 无；核验后才附物 | 实测开度及关节 |
| place接近、下降 | 实测闭合 | 有 | 实测开度及关节 |
| place释放后lift | 完全张开 | 无 | 实测开度及关节 |
| 黑件运输plan_pose | 实测闭合 | 有 | 实测开度及关节 |
| 松爪后空手运动 | 实测张开 | 无 | 实测开度及关节 |

附件默认24球、最多31球，允许与同手全部9个夹爪link接触；TCP、相机、前臂、另一臂和世界碰撞继续检查。不做开爪扫掠。数值对照脚本`scripts/audit_planning_geometry.py`先生成MuJoCo FK参考，再在cuRobo GPU环境核对IK、优化器、图搜索及检查器；它不执行机器人运动。测试结果在本轮完成后单独记录，不以脚本存在代替验证通过。

锁定版本：`8e734f3ced1df898990bcd92de40abce475907db`。审计范围为本项目生产服务、兼容规划入口及离线工作空间生成；不是只检查 SDK 文件是否改变。

## 已纠正的越界方式

1. 原 vendor `GraspMotionPlanner/GraspBatchMotionPlanner` 继承官方类，覆盖 `plan_pose` 清空起始状态 dt、克隆结果；原项目 mixin 还覆盖内部重试/finetune 参数。生产构造器已全部绕开这些类，直接实例化官方类。项目 `PublicPlanner` 是组合调用；官方 `plan_grasp` 的内部 self 调用不会进入项目方法。
2. 移除项目 `ClearanceCostTolerance` 检查器子类，使用原始 `RobotCollisionChecker`。应用层可以解释返回的误差、拒绝轨迹，但不改官方检查函数。
3. 移除对 `scene_collision_checker.data.voxels.params` 的直接 round 写入。改为公开 `VoxelGrid.dims` 输入转换，保持整数格数和原始 ESDF 样本；GPU 实测 120 格不再被截断为 119 格。
4. 不再以每次测得的夹爪角度和附件球数组创建、销毁整套求解器。双侧各预留 31 球，经公开几何 API 原位更新。卸载不销毁规划器。

历史 vendor 文件保持只读；其中的旧覆盖类不再进入生产构造链。项目继承的是传输/验证适配器 `_NativeCuroboBackend`，不是官方求解器类。

## 全部算法入口

| 项目用途 | 官方调用 | 项目边界处理 |
|---|---|---|
| batched IK | `InverseKinematics.solve_pose` | 独立位姿批、公开 seed 配置；返回后按10mm世界距离筛选解并记录FK |
| 单目标 IK 兼容入口 | `planner.ik_solver.solve_pose` | 类型、关节顺序、场景版本和结果有效性检查 |
| `plan_grasp_batch` | `BatchMotionPlanner.plan_grasp` | 官方全部阶段；挑选成功候选、保留官方导数，最终插值命令检查碰撞 |
| `plan_grasp` | `MotionPlanner.plan_grasp` | vendor历史方法，不注册为本服务HTTP入口；当前单目标抓放求解器用于下行place调用 |
| `plan_place` | `MotionPlanner.plan_grasp`，公开 `plan_grasp_to_lift=False` | 前两段保持attachment；松爪detach后另调公开规划撤离 |
| `plan_place_batch` 兼容入口 | `BatchMotionPlanner.plan_grasp`，同上两阶段模式 | 当前demo不调用；保留输入/释放位置/结果检查 |
| 普通持物或空爪位姿运动 | `MotionPlanner.plan_pose` | 公开 max_attempts；原始轨迹返回后五次插值与执行时间伸长 |
| 双臂观测/VLA移动 | `MotionPlanner.plan_cspace` 或 `plan_pose` | 14关节/双TCP目标、官方求解、10mm验证 |
| 轴向移动/支撑法向撤离 | `ToolPoseCriteria.linear_motion` + `plan_pose` | 当前place Lift采用官方默认scale=1。旧支撑兼容入口仍有有界参数尝试，但已删除额外100000倍放大；该入口不由当前demo调用，不冒充官方plan_grasp |
| 固定对侧手臂、夹爪反馈、附件 | `compute_kinematics`、`update_link_spheres`、`update_kinematics_config`、`AttachmentManager.update/detach` | 测得关节/球转换为公开输入；几何更新后公开 `graph_planner.reset_buffer` 清除过期路径 |
| 世界地图上传 | `Scene.create`、`update_world`、公开 `VoxelData.load_batch` | 清理已替换mesh缓存、一次加载全部命名体素层、显式目标层开关 |
| 碰撞与执行复核 | `RobotCollisionChecker`、`get_kinematics/get_collision_distance/get_self_collision_distance/get_bound`；图搜索可行性API | 关节采样、返回值判定、10mm约束，保留世界/自碰撞检查 |
| 三视角 TSDF/ESDF | `RobotSegmenter`、`Mapper.integrate`、`integrator.compute_esdf/get_voxel_grid` | 清单/SAM掩膜、输入坐标转换、导出/导入测量块裁剪，不替换官方融合或距离变换 |
| 选臂位置可达性 | `InverseKinematics.solve_pose` + `ToolPoseCriteria.track_position` | 明确只判断位置可达性，不冒充碰撞可行性；左右实例启动预热 |
| 离线工作空间生成 | 同一公开位置IK | 独立离线命令、批量采样与结果保存，不在任务中生成 |

构造时使用公开 `InverseKinematicsCfg/MotionPlannerCfg/RobotCollisionCheckerCfg`；机器人、seed、阈值、缓存容量属于配置。IK配置在构造前复用同一解析模型，使IK、轨迹优化、图搜索、独立检查器共享几何参数；没有替换求解器内部模块、梯度、代价函数实现、优化循环或CUDA/Warp内核。

## 官方接口限制必须如实显示

锁定版本 `plan_grasp` 不公开 max_attempts/finetune 参数，因此不得靠覆盖内部 `plan_pose` 注入：单目标内部默认 max_attempts=5，批量内部默认=1。高级设置已修改说明，effective_config明确记录未覆盖。普通 `plan_pose/plan_cspace` 仍按其公开参数传入设置。不能把“官方原算法”说成“所有配置与官方示例相同”。

## 预热与缓存

启动完成13个常驻配置后才ready：A/B各IK100、批4/16、普通单臂、单目标抓放，以及AB双臂、A/B位置可达性。实测最终启动51.33秒。用户任务期间换物、换臂、松爪和重置地图不销毁兼容规划器。重置依然清除旧场景证据，旧地图不能执行。

测得新的夹爪开度需要重新生成锁定关节几何，再通过公开API复制到原GPU缓冲；这是几何更新，不是重新预热。只换附件的GPU实测约2–3ms，开度改变约1.4–1.8s。真实prepare还包括场景上传、对侧手臂和附件检查，不能把2–3ms写成整个attachment技能耗时。

显式改变求解器设置、增加工具拓扑或场景超过预留容量仍须重建相应配置。任意尺寸RGBD地图的分配不属于固定运动求解器预热，未宣称所有任意输入均无首次开销。

## 验证证据

- `data/resident-geometry-final-gpu.log`：IK/轨迹优化/图搜索/硬碰撞检查的球与独立新建FK一致，GPU参数地址保持；公开体素输入得到正确整数索引。
- `data/resident-public-api-run.json`：完整单件流程成功，194.214秒；`data/resident-public-api-timings.json`保留分阶段统计。运行期间无规划器重建；此轮在最终体素输入修订前，最终版本另行验收。
- `data/strict-sdk-source-audit.json`：本轮重新下载官方commit源码归档，对容器788个文件逐一比对，changed为空，涵盖Python/CUDA/C++/配置/资产；只缺不参与算法的examples/README.rst。GitHub API超时/限流后改用官方codeload归档，未使用第三方源码。启动另对锁定清单中的361个SDK Python文件逐一核验，发现不一致即拒绝启动。
- CPU回归覆盖组合调用不拦截内部方法、几何缓存生命周期、附件/松爪、输入转换、清单状态和固定ID。完整代码库中仍有历史fixture问题，不能以针对性回归冒充整个仓库测试全绿。

最终代码针对性回归219项通过、Ruff通过、清单前端8项通过。扩大检查暴露旧fixture与当前场景/封装不符（旧小桌回放、缺少局部障碍方法的视觉替身、1nm夹爪余量断言等）；完整前端317/318通过，另一个CAD编辑器fixture失败，不是清单测试。未宣称完整仓库全绿。

最终体素输入/13配置预热版本实验`6316357da2bc42d3b4b67b27e931b0e4`在85.147秒因粉件空夹停止。实际反馈0.8rad，视觉模型没有参与该抓取判定。`data/6316357-physical-diagnostic.json`仅用于离线回放诊断：下降抓取段粉件Y由0.49437移动到0.47480m并倾斜，发生在close_gripper开始前；抬升后物体回落盒内。未将模拟真值送入感知/规划，也未取消用户指定的沿工具Z深入1cm。当前方向/非接触球预筛不能证明目标与手指形成可靠夹持；此物理抓取稳定性问题未由预热修订解决。前一轮成功不代表最终版本已稳定或达到180秒。


2026-10-22维护记录：按用户要求清理Ubuntu data中的历史测试产物。上文resident-public-api-run.json、resident-final-run.json等原始测试导出在Ubuntu已清除；小型官方源码校验结果、官方文件哈希清单、常驻几何最终GPU验证日志保留。Windows历史诊断未清除。本次重新对运行容器788个官方文件核验，changed仍为空。清理明细见data-cleanup-2026-10-22.json；算法及运行配置未修改。

# 可见点云、桌面去飞点与单臂抓放执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. 用户已确认执行。状态概览更新至 2026-10-10 集成检查点 20；各项所引旧轮次证据保持其原适用范围，勾选不代表完整物理抓放通过。

**Goal:** 用三个 RGBD 可见点云完成水平启动和同臂抓放；grasp 固定张开模型并保留现有 approach/lift 设置，结束阶段更新实际夹爪并建立持久 attachment；place 直接使用该状态，默认保持物体初始姿态、离盒底 1 cm 释放。仿真 RGBD 不加噪声，去噪优先评估 Open3D。

**Architecture:** 统一观察合同；点云保留真实来源，可选 Open3D 过滤在 ESDF 前执行；共享规划器消费表面球 attachment 和测得的放置参照点。Runner 管理跨 grasp/place 的持物绑定；缓存管理与 attachment 内容更新分离。

**Tech Stack:** Python 3.12、MuJoCo 3.10.0、SAM3、GraspGen、cuRobo `8e734f3ced1df898990bcd92de40abce475907db`、现有 TypeSafe/Pydantic AI 和 Vite/Three.js。

**Spec:** [本次审核后的设计](../specs/2026-10-09-visible-cloud-grasp-place-design.md)。取代同日上一版 revision 计划中冲突的工作项，不继续实现其形状补全和箱腔模型。

## 本次实施的完整流程

输入颜色/类别及目标盒 → 水平双臂下的 initial 三相机快照 → 头部 SAM3 检查所需目标 → 腕视角不足时用头部 RGBD 建图并避障移动到观察位姿 → fine 三相机快照及建图 → GraspGen + plan_grasp → 闭爪和按高级设置 lift → post_grasp 观察确认持物 → 更新实际闭爪球并建立 attachment → plan_place → 离盒底 1 cm、保持初始姿态释放 → detach、实测开爪模型 → release 观察和建图 → 重新规划短撤离 → post_place 观察及场景更新。

独立 grasp 在建立 attachment 后结束；独立 place 从已有持物绑定和 standalone_place 新观察开始，不重做空手初始观察运动；本轮建图成功后才允许 plan_place。已有三相机、物料盒编辑器、grasp/place 模块沿用并修正，不另起一套实现。

**当前证据界限：** 最近完整 CPU 检查点为 [893 passed、56 skipped](../../../data/visible-cloud-work/root-integration-suite-20.txt)，前端 [172 passed](../../../data/visible-cloud-work/root-frontend-suite-4.txt)，Ruff/Vite 构建通过。A/B 两臂均已实际完成独立 grasp/lift、RGBD 验证及持久 attachment。后续 place 仍出现保守持物包络冲突、当前可见片段关联不足及技能合同拒绝；这些失败保留，不以放宽碰撞或自动替换选择绕过。当前 attachment 已修为仅使用 post_grasp 当前全部目标点及同次实际 TCP，旧点只作验证参考。A/B 完整 grasp→standalone place 和单次 grasp_and_place、默认 1 cm 释放独立评分仍未通过，不得以单元测试或只读回放替代。逐轮原始证据见[实施记录](2026-10-09-visible-cloud-progress.md)。

## Global Constraints

- 环境只用实际 RGBD；自身反馈、标定和模型可用；仿真真值只供渲染、编辑、Reset 与独立评分。
- 不补物体形状，不填充整个物体 AABB，不拟合盒尺寸/箱腔，不检查是否装得下；盒子保留 RGBD 识别、放置点和可见点云碰撞。
- 当前双臂水平启动；头部优先检查，先建图后观察运动；只在显式节点采集。
- 全部机器人碰撞球保留；grasp 运动固定张开模型、不 attach；grasp 结束建立闭爪模型与持久 attachment，place 入口已具有该状态。
- approach/lift 保留现有坐标系、轴和偏移高级设置，不强制反向路径。按用户要求假设 lift 可行，不新增携物 lift 检查、反向轨迹功能或接触离桌实现。
- 仿真 RGBD 禁止加噪声；取消严格向下连通过滤。Open3D 候选过滤不得只因点簇小或不连桌面而删除障碍，也不得制造已知自由空间。
- SDK/vendor 只读；项目适配层实现过滤和缓存管理。当前项目无 Git 元数据，沿用备份及哈希，不创建仓库。
- 本地验证后再同步 Ubuntu，仅操作本项目服务。只读 GPU API 核查和性能基准不能被描述为物理抓放验收。

## Review Focus

- Open3D 过滤对真实小物体/薄壁/稀疏表面的误删，以及密集浮岛未被识别：任务 2。
- 旧形状尺寸经 holding/public state/verification 隐性回流，或 256 球上限静默截断点云：任务 1、4。
- 意外改动 approach/lift 设置、闭爪立即 attach、独立 grasp 返回后绑定被 scope 清掉：任务 4。
- 模型热更新只更新末端检查器而未更新 IK/优化器，或缓存复用另一只手的附件：任务 5。
- 无盒拟合后仍要求 cavity/size、将点云最低点或 TCP 当真实底面、释放被遮挡误判成功：任务 6。

## Task 1：收敛感知数据与观察合同

**Files:** `resources/cameras/calibration.json`，`src/luna_mujoco/cameras.py`、`perception.py`、`placement.py`、`runner.py`、`algorithms.py`、`verification.py` 和 `skills/observe/*`；实际合同回归位于 `tests/test_perception.py`、`test_visible_perception.py`、`test_visible_verification.py`、`test_rgbd_application.py`、`test_cameras.py`、`test_camera_api.py`，不另建原计划名 `test_visible_cloud_contract.py`。

**Interfaces:** 实例包含身份、语义、颜色、可见点及来源和观察 ID；`points_world_m` 只包含实测点。`Runner.observe(context, *, checkpoint)` 使用 initial/fine/post_grasp/release/post_place/standalone_place 显式节点。

- [x] 实现并用可见点合同回归验证：控制请求不要求规则体尺寸、CAD pose、shape_prior_provenance 或 cavity；冻结点集只有实测点和坐标变换，不补全不可见形状。
- [x] 核对三路 RGB/公制深度、内参、固定安装外参和坐标约定；腕外参使用采集时自身关节反馈与 FK，同一快照绑定同一机器人状态。标定及实际 MuJoCo 渲染测试通过。
- [x] 显式采集节点、固定传感器输入元数据毒化、公共状态脱敏回归通过；相机页面轮询只读已缓存快照，不新增观察。仿真对象 pose/geom/segmentation ID/接触对象 ID 不进入感知决策。
- [x] 删除形状/完整体积依赖；关联使用语义、颜色、实测点双向重叠及深度重投影。第 6 轮较大盒不同面中位数相距 127.3 mm 的误拒已修为点集分离必要条件，唯一性阈值未放宽；相邻同色分离对象负例保留。见[记录 RGBD 回放](../../../data/visible-cloud-work/post-grasp-association-6/review.txt)。
- [x] 持物验证结合实际夹爪反馈、抓前可见点、post_grasp RGBD 和自身 TCP；冲突/观测不足返回 unknown。第 8/9 轮 A 臂实测持物验证已通过；不要求连续追踪，不以真值补证。
- [x] 上述实际测试文件进入完整 CPU 回归；真实 SAM3 initial/fine/post_grasp 记录已回放。局部盒上沿不能冒充盒底的修复保留实测盒底约 0.61105 m，见[盒底复核](../../../data/visible-cloud-work/floor-reference-fix-report.txt)。这不是硬件 RGBD 验收。

## Task 2：Open3D 可选去噪与点簇预览

**Files:** 新增 `src/luna_mujoco/pointcloud_filter.py`、`tests/test_pointcloud_filter.py`；修改 `src/luna_mujoco/perception_mapping.py`、`geometry_service.py` 及地图调试显示。

**Interfaces:** `filter_observed_points(points_world, point_sources, *, settings)` 返回原点索引对应的 keep_mask、rejected_mask、可选 cluster_labels 和 counts。映射保存相机/像素来源与实例归属；降采样保留来源。项目适配层在目标排除及 TSDF/ESDF 建图前应用过滤，禁止改 vendor。仿真默认 OFF，包含关闭原生路径中隐含的统计/半径/小连通簇过滤。

- [x] 保存 registered raw depth、实测点、掩码及建图请求；独立 GPU 重放核对 OFF 下 input=kept，移除旧隐含小簇过滤，unsupported 目标掩码像素留在背景。可查看原始/过滤/剔除点与体素层；未把无噪声仿真中的显示差异当作已定位硬件飞点成因。
- [x] Open3D 0.19 statistical/radius 可选实现及有界离线评估完成：4 份无额外噪声 MuJoCo RGBD/SAM3 记录，每份最多且实际 20,000 个原始采样点，共 48 次调用。评估小源物体、盒底、壁面和上沿均保留，部分桌面/其他盒面被删；抽稀影响密度，不能外推全量误删率。默认仍 OFF，见[滤波报告](../../../data/visible-cloud-work/filter-assessment-report.txt)。
- [x] DBSCAN 仅诊断；包括 -1 在内的所有标签不改变 keep mask。没有最大簇、小簇或“不连桌面即删除”规则，也不声称能识别密集浮岛。
- [x] CPU 回归验证小结构/稀疏点、目标/held 来源及拒绝深度不雕刻自由射线；记录 RGBD 已对源/盒底/壁/上沿逐项比较。`test_pointcloud_filter.py`、`test_perception_mapping.py` 和 GPU 合同测试通过。
- [x] 不加仿真噪声；显式观察节点提供原始/保留/剔除掩码和数量。有界样本滤波耗时、5 ms 采样进程 RSS 和记录输入 GPU 建图耗时已报告，未放入伺服循环；不是完整链路性能或精确算法峰值。
- [ ] 真实硬件 RGBD、标注飞点与有效薄/远处稀疏结构的评估，以及未抽稀全量输入的误删率、完整建图延迟和 DBSCAN 内存上限仍待验证；据此决定真实机器人是否启用/调参。

## Task 3：保持审核通过的初始观察流程

**Files:** 新增 `src/luna_mujoco/observation_motion.py`、`resources/observation_poses.json`、`skills/move_to_observe/{SKILL.md,contract.json}`、`tests/test_observation_motion.py`；修改 `runner.py`、`algorithms.py` 和场景适配。

**Interfaces:** `Algorithms.plan_observation_motion(observation, scene, arm, goal_joints_rad, *, planning_settings)` 输出版本绑定的关节轨迹；`Runner.move_to_observe(context)` 顺序处理 A/B 实际状态。

- [x] CPU 回归覆盖缺源/缺盒、头部单视角建图、非空手拒绝、目标保留、A 成功 B 失败事实及旧计划失效。另有两例真实 JEV+SAM3+MuJoCo RGBD 初始缺目标验收，完整日志确认零 cuRobo/GraspGen/电机/轨迹/夹爪调用，见[负例证据](../../../data/visible-cloud-work/real-initial-negative-acceptance/report.txt)。
- [x] 固定 A/B 观察姿态由模型/FK、标定及实际渲染 RGBD 验证；双臂水平启动保留，源/盒/桌在到位三视角可见，见[观察姿态报告](../../../data/visible-cloud-work/observation-poses-report.md)。没有编辑器坐标或 READY 控制输入。
- [x] A 到位后 fine 观察更新 B 运动证据，B 到位后再次 fine。覆盖判定使用注册深度的首个返回前自由段、真实视锥及实测任务高度范围；范围外明确 unknown，全部机器人球的 world/self/peer 检查保留。
- [x] 观察运动/相机 CPU 测试、独立 GPU 检查、参考场景实际 A/B 两次避障观察运动和到位三相机记录通过；第 5 轮精确请求 GPU 重放 561 个轨迹样本也通过。这只证明这些参考输入，不是所有场景或 A/B 完整抓放验收。

## Task 4：grasp 结束建立持久表面球 attachment

**Files:** `src/luna_mujoco/carrying.py`、`placement.py`、`geometry_service.py`、`backend_builder.py`、`runner.py`；实际回归在 `tests/test_carrying.py`、`test_visible_workflow.py`、`test_rgbd_workflow.py`、`test_visible_surface_gpu_contract.py` 等，不另建原计划中的两个测试文件名。

**Interfaces:** `CarryingBinding` 消费 points_tool_m、observation_id、generation、instance_id、arm、state 和表面采样/余量配置，不再强制 estimated_bounds_tool_m 或 shape_prior_provenance。`carrying_spheres` 只覆盖实测表面体素。grasp 结束时在 post_grasp 持物确认后更新闭爪球并建立持久绑定；成功返回时 place-ready。运动规划仍使用现有 approach/lift 参数，不增加 reverse_grasp_segment。

- [x] 表面球/来源回归通过：分离可见面之间不填整个 AABB；测得点按离散精度覆盖，超球预算合并采样分辨率而不截掉尾点；未见体积仍 unknown。
- [x] grasp 固定张开、无 attachment；保留原 approach/lift 坐标系、轴和偏移，不改成反向路径，也不调用闭爪后携物 lift 规划。相关 CPU 回归与实际 A 臂执行通过。
- [x] 观察地图保留目标；grasp 请求仅排除所选 RGBD 实例的世界碰撞层。桌/盒/其他对象及全部机器人球保留；GPU 记录确认 267 个原机器人球仍在。
- [x] CPU 生命周期回归及第 8/9 轮独立 grasp 证明：post_grasp 确认后更新实测闭爪模型、建图并准备 attachment，成功返回仍持久有效。准备失败保留 provisional/实际持物事实，不自动松爪或伪报 place-ready。
- [x] standalone place 使用原 binding；起点/路径碰撞、旧目标不复活、取消/跨请求状态由 CPU 和独立 GPU 见证验证。物理 place 全流程仍归任务 7 未完成项。
- [x] 接触豁免限定为当前持物臂必要指垫与附件的关系，不删机器人球或豁免整手；carrying/workflow/GPU 合同测试通过。旧支持离桌实验只作历史。

## Task 5：夹爪状态更新与规划器复用

**Files:** `src/luna_mujoco/geometry_service.py`、`backend_builder.py`、`carrying.py`；实际回归在 `tests/test_geometry_service.py`、`test_backend_builder.py`、`test_carrying.py`、`test_visible_surface_gpu_contract.py`，独立 GPU 脚本/结果在 `data/visible-cloud-work/`。

**Interfaces:** 锁定 SDK 的 attachment 更新只影响一个 model，不能假定会自动传播所有 solver。现实现采用有界、精确键的配置复用，键包含实际 A/B 夹爪反馈与 binding；需要时重建相关后端，不量化反馈求缓存命中。场景版本与持物绑定分别管理；scope 不销毁跨 grasp/place 的绑定。数值等价校验只用于采集状态核对，不改变 FK、缓存键或碰撞球。

- [x] 已只读核对锁定 SDK public kinematics/attachment、CUDA graph 和容量行为；没有套用旧版文档或修改 SDK。选择精确配置复用的依据和局限见[GPU 报告](../../../data/visible-cloud-work/gpu-report.md)。
- [x] A/B 独立、open/实测 closed、mimic FK、attach/detach、下一请求无旧附件、IK/优化/终检一致的 CPU 回归通过；原机器人碰撞球均保留。
- [x] 固定附件槽与有界精确配置键已实现，反馈不量化；复用同一配置，变化时更新相关后端。闭爪后不承诺零重建成本。
- [x] 默认槽为 inactive；detach 后清除正半径附件。checker-only 后端创建后再 detach 的缺陷已修；独立 GPU 验证 IK/TrajOpt/graph/peer/终检及后续 native/checker 复用，不继承旧附件。
- [x] 已记录部分独立 GPU 时延和复用见证：一次 prepare 189.83 ms、同配置重复 7.35 ms、native place 3282.2 ms、观察规划 4053.83 ms。prepare 计时不含此前后端构建，不能解释为完整冷启动；其他输入/后续版本以报告原始结果为准。
- [x] 上述实际测试文件进入完整 CPU 回归，锁定 SDK GPU 状态/坐标/碰撞见证通过后接入当前流程；第 8/9 轮实际 grasp 已完成模型切换和持久 attachment。
- [ ] 完整冷启动/预热、任意夹爪连续开闭、attach/detach、首次/重复规划的统一性能基准及 GPU 内存峰值仍待补齐；不能声称已证明所有切换都不拖慢。

## Task 6：简化 place 与三种 skill

**Files:** `src/luna_mujoco/placement.py`、`algorithms.py`、`runner.py`、`verification.py`、`decisions.py`、`skills/plan_grasp_and_place/*` 及相关合同；实际回归在 `tests/test_visible_placement.py`、`test_visible_workflow.py`、`test_rgbd_workflow.py`、`test_release_association.py` 等。

**Interfaces:** `release_goal(visible_points_tool, destination_reference_world, tool_rotation_world, *, release_height_m, offset_world_m, grasp_support_reference)` 返回 TCP 放置目标和实测参照来源。视觉提供目标区域世界 XY；界面 Z 对应 release_height_m，定义为物体底部离盒内底面的距离，默认 0.01。grasp_support_reference 保存实测桌面高度和闭爪时实际 TCP 位姿；默认恢复该朝向以保持物体初始姿态，在水平支撑、无滑动条件下应用底面高度关系。机器人绝对 Z 仍需有效盒底高度参考，不从仿真模型取值。不接收完整尺寸或 box cavity。

- [x] place 使用盒语义及有效 RGBD XY/盒底参考，不拟合 cavity、尺寸或检查是否装得下；缺深度拒绝，无仿真中心输入。实测盒底要求同实例的支持证据，局部上沿不能独立冒充底面。
- [x] CPU 验证默认 `z_tcp_release=z_tcp_at_grasp+z_box_floor-z_table+0.01`。姿态/支撑前提不足时不声称真实底面距离，不用可见点最低点代替物底；正的 hard-clearance 冲突明确报错，独立 GPU 证明零代价继续、真实正代价拒绝。默认高度未暗加球半径或安全余量；显式 40 mm 重试须另记，不能算默认 10 mm 成功。
- [x] 实现携物 place、实际开爪检查、detach、release 显式观察/建图及无附件撤离重规划，箱壁/底面继续避障。CPU 和独立 GPU 规划见证通过；完整实际释放/撤离仍待任务 7。
- [x] 三种规划 skill 同臂衔接；抓前只评估抓姿 IK，正式 place 使用既有 attachment/实测闭爪。右侧组合试验发现张开夹爪的目的地 IK 被错误地作为硬门槛，现已移除该可选推测筛选；放置可行性明确推迟到持物后。独立 grasp 不要求盒；独立 place 要 held+attached，且新 observe/new build_scene；失败/取消保留持物，已释放则保留 release_pending，观察不足 unknown，不做完整体积验证。
- [x] 遮挡、参照变化、旧轨迹、Reset/取消迟到和清理失败回归通过。release 关联仅加已知释放位姿/高度预测，保留静止/跟手假设和唯一性；10/40/150 mm、同色歧义及实际 TCP ±20 mm 偏差共 17 项复核通过，见[关联报告](../../../data/visible-cloud-work/release-association-review/independent-review.txt)。
- [x] 新运行显式初始化全部合同 facts=false，只在动作完成后置真；当前观察与场景就绪分开。新鲜度/目录独立回归和三状态真实 JEV 只读回放通过：初始选 observe，观察后及已选盒但未建图时选 build_scene；完整 15 skills+stop，零物理/GPU调用。见[复核](../../../data/visible-cloud-work/standalone-place-freshness-review.txt)及[回放结果](../../../data/visible-cloud-work/place-stage-selection-replay/result.json)。此结果仅覆盖这三个 standalone place 状态，不证明全流程所有选择可靠。

## Task 7：设置、展示、部署与完整验收

**Files:** 修改 `src/luna_mujoco/planning_settings.py`、`web/src/main.js`、`web/src/scene-preview.js`、`web/src/planning-settings.js`、`web/src/camera-panel.js`、`web/src/scene-view.js`、`README.md` 和 `tests/rgbd_pick_place_smoke.py`；沿用 `tests/test_bins.py` 与 `web/tests/bin-editor.test.js` 并补充所需回归。

- [x] 开口物料盒编辑器及底/四壁物理碰撞实现，bins Python/前端测试通过；颜色/位置/尺寸只用于仿真编辑，规划盒壁/底来自 RGBD。
- [x] 观察 A/B 位姿、可选 Open3D/DBSCAN、attachment 采样/余量/预算、原 approach/lift、place 高度/姿态已接入高级设置，运行使用冻结设置。旧容量/箱腔/接触离桌设置移除；持物已有 binding 不被中途设置改写，见[设置实现](../../../data/visible-cloud-work/settings-ui-completion-report.md)及[独立复核](../../../data/visible-cloud-work/settings-endtoend-review.txt)。
- [x] 实现 cached raw/保留/剔除显示、观察节点、provisional/held/release_pending、实测持物证据、grasp 固定张开/无附件与结束后实测模型/附件及阶段耗时。API/DOM 测试通过，见[点云展示报告](../../../data/visible-cloud-work/pointcloud-ui-report.md)。
- [x] 检查点 23 全量 Python 953 passed/56 skipped、前端 172 passed、Ruff/Vite 构建通过；按清单备份同步 Ubuntu。逐轮清单与备份记录见 progress；只更新本项目服务。
- [x] 三种规划与执行/观察/验证技能合同、文档同步给 JEV；完整目录与 stop 保留，不筛选 eligible、不替模型选下一步。未持物 place 拒绝、独立 grasp 无盒、同源同目的同臂组合及本轮观察/场景 facts 回归通过；真实模型选择证据见任务 6。
- [x] 已完成的单项执行验收：水平启动后 A/B 避障观察运动及腕视角改善；两个缺目标真实初始负例零运动；A/B 两臂多轮实际 grasp/lift、RGBD 持物确认和持久 attachment。MuJoCo RGBD 未加噪声，原 approach/lift 设置保留；真值只存独立评分文件。current-only attachment 已另在 A19/B9 实际抓取成功，与早期旧绑定轮次区分。
- [x] 第 10 轮重复技能选择已有记录回放修复证据：静态 requires 直接进入 criterion；去除全局 prompt 中重复的 place/build_scene 流程说明后，11 个记录状态真实 JEV 回放全部合同合法，原失败 event24→evaluate_arms、event39→plan_grasp。完整 16 选项保留，未计算 eligibility 或替换选择。见[首次失败](../../../data/visible-cloud-work/catalog-contract-replay-trial10/result.json)及[第二版通过](../../../data/visible-cloud-work/catalog-contract-replay-trial10-v2/result.json)。这是一组各调用一次的记录回放，不保证任意后续调用，也不是新物理执行。
- [x] 右侧 B9 实际放置与开爪通过，独立评分确认入箱（显式 50 mm 释放/1 mm attachment 体素）；释放后语义修复及新地图/空手撤离/最终命令通过原记录 GPU 重放。后者未执行物理撤离，不能据此勾选完整闭环。
- [ ] 真实 SAM3、GraspGen、cuRobo、MuJoCo 完整 grasp→standalone place，以及单次 grasp_and_place，分别覆盖 A/B；不因 A 臂独立 grasp 已成功而勾选。
- [ ] 默认 10 mm 离底释放、保持初始姿态、实际开爪/detach/release 观察/撤离/post_place 成功与独立物理评分仍待完整闭环；若 40 mm 显式重试成功，只算该设置通过。
- [ ] 浏览器实际视觉/交互 QA 尚未通过（工具调用超时，已有 DOM/build 证据）；完整切换性能及硬件 RGBD 过滤边界分别保留在任务 5、2。最终交付前根据闭环结果再次同步验证状态，不得报告全部完成。

## 执行次序

1 → 3 → 4 → 5 → 6 → 7；任务 2 作为真实 RGBD 数据的并行评估/预览工作，不阻塞无噪声仿真的抓放主流程。先用小规模真实数据与 GPU 见证锁定 grasp 结束后的持久 attachment 和热更新行为，再跑完整抓放。

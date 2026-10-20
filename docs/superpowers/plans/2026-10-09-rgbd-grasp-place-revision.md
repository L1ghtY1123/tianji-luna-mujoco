# 三相机无特权抓放修订执行计划

本版已被 [可见点云执行计划](2026-10-09-visible-cloud-grasp-place.md) 取代。以下任务保留作历史记录，以用户第二次审核后的新计划为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax. 本次用户请求交付执行方案；本文件中的未完成步骤不表示已经实现或验收。

**Goal:** 从当前水平双臂姿态出发，仅用三个 RGBD 的环境观测完成单臂抓取、独立放置及组合抓放，并明确管理 attachment 和离散观察证据。

**Architecture:** 传感器输出冻结观察；感知输出实测点云、估计几何与覆盖；共享规划适配器负责场景和三种规划入口；Runner 只根据实际反馈管理执行状态。恢复显式观察运动分支，去掉执行阶段的隐式采图依赖。保留已有可验证模块，修正与新规格不一致的部分，不整体照搬 base。

**Tech Stack:** Python 3.12、MuJoCo 3.10.0、现有 SAM3/GraspGen、cuRobo `8e734f3ced1df898990bcd92de40abce475907db`、TypeSafe/Pydantic AI、Three.js/Vite。

**Spec:** [修订设计](../specs/2026-10-09-rgbd-grasp-place-revised-design.md)。此前同日计划作为历史记录，冲突时以本修订为准。

## Global Constraints

- 环境真值只供渲染、编辑、Reset 和独立评分；不能进入 JEV、规划、执行判断或成功判定。
- 水平双臂启动；三路真实渲染的注册 RGBD；只在显式观察节点采集。
- 单臂完成抓放；双臂只在空手补观察分支按顺序运动；不实现换手。
- 全部机器人碰撞球保留，夹爪状态与反馈一致；容器四壁和底板参与碰撞。
- 无物体 CAD 依赖；实测表面、推断形状、未知范围分别表达。
- attachment 只用于规划，不焊接或瞬移物体；provisional 不等于已抓取，release_pending 不等于已释放。
- 官方 SDK/vendor 只读；仅修改项目适配层。先本地验证，再同步 Ubuntu，仅操作本项目服务，保留回滚归档和哈希。
- 当前工作目录无 Git 元数据；不新增仓库、不制造提交记录，用现有备份及文件清单保存变更。

## Review Focus

- 对称物体、缺面点云不能因 PCA 跳变误判身份或凭少量膨胀声称完整几何：任务 2。
- 肩部在相机外不等于桌面被遮挡，碰撞代理球不能代替新环境扫掠空间证据：任务 3。
- 观察位姿移动时源物体仍是障碍，A 已移动/B 失败时状态不得回滚：任务 3。
- release_pending 时视觉遮挡不能形成“先验证才能撤离、先撤离才能验证”的死循环：任务 4、5。
- 新观察、取消、Reset 或服务异常不得让旧轨迹/暂存 attachment 在后续动作生效：任务 1、4、5。

## Task 1：信息边界与观察合同

**Files:** 修改 `src/jev_mujoco/cameras.py`、`simulation.py`、`perception.py`、`runner.py`、`verification.py`，`skills/observe/{SKILL.md,contract.json}`；新增 `tests/test_observation_checkpoints.py`，扩展现有 `test_cameras.py` 与无真值测试。

**Interfaces:** `Runner.observe(context, *, checkpoint)` 接收 `initial/fine/post_grasp/post_place/standalone_place`；每份结果记录 observation_id、generation、capture_time、checkpoint、views、robot_feedback。算法和验证消费该观察，不能自行读取 Simulation 环境状态。

- [ ] 写回归：固定传感器输入而更改/毒化环境元数据，JEV 输入、规划请求和验证结论不变；initial 无目标零运动；计划和执行不会出现合同外的 capture 调用。
- [ ] 集中初始与阶段观察入口；三路同帧/外参随捕获关节冻结；头部先检索任务所需源/盒，故障与无匹配区分。
- [ ] 新观察更新版本并使相关派生数据失效；执行不能依靠旧 observed 布尔值接受过期图。
- [ ] 运行 `python -m pytest tests/test_cameras.py tests/test_camera_api.py tests/test_observation_checkpoints.py`；所有新增断言通过，并检查实际观察日志的捕获次数。

## Task 2：局部点云与可用于携带的几何范围

**Files:** 修改 `src/jev_mujoco/perception.py`、`placement.py`、`verification.py`、`algorithms.py`、`carrying.py`；扩展 `tests/test_perception.py`，新增 `tests/test_estimated_geometry.py`。

**Interfaces:** 感知实例保留 measured_points、estimated_shape（frame、bounds、prior、uncertainty）、coverage 和感知生成的 instance_id；保持兼容读取时必须显式 estimated，禁止旧 T_world_object 被解释为 CAD 真值。`carrying.py` 的共享 `geometry_for_carrying(instance, T_world_tcp)` 输出 TCP 局部点云、包络及来源。

- [ ] 写回归：缺面规则物体有明确先验/覆盖记录；局部薄片、被遮挡箱底或突出部分不能凭少量膨胀输出“完整可靠”；外包盒空角不作为实占/自由空间证据；对称朝向变化不造成身份交换；点云变换往返正确。
- [ ] 分开实测点云与形状拟合，抓姿只使用实测点云；实现规则刚体的可审计拟合与容器内腔估计。
- [ ] 持物包络从上述估计生成，不直接读取编辑器尺寸；候选放置检查物体完整估计范围、腔体和开爪空间。
- [ ] 运行 `python -m pytest tests/test_perception.py tests/test_estimated_geometry.py tests/test_bins.py`；用现有真实 RGBD 记录复核点云、几何先验与盒内可用范围。

## Task 3：水平启动到观察位姿

**Files:** 新增 `src/jev_mujoco/observation_motion.py`、`resources/observation_poses.json`、`skills/move_to_observe/{SKILL.md,contract.json}`、`tests/test_observation_motion.py`；修改 `runner.py`、`algorithms.py`、`perception_mapping.py`、`geometry_service.py`。

**Interfaces:** `Algorithms.plan_observation_motion(observation, scene, arm, goal_joints_rad, *, planning_settings)` 返回与普通关节轨迹相同的关节、时间和场景版本结构；`Runner.move_to_observe(context)` 顺序执行 A/B 并记录实际完成情况。scene purpose 为 observation_motion，目标层必须 included。

- [ ] 写回归：缺有效图/缺姿态/非空手/剔除目标场景零运动；另一臂改变时重规划或拒绝；A 完成 B 失败保留 A 反馈；A 到位若使 B 路径证据失效，在此离散节点更新观察；移动后必须 fine 观察。
- [ ] 在当前机器人模型、标定和 RGBD 中生成并验证观察姿态，保存 A/B 七关节、模型/标定版本与验证记录；不得把 READY 名称当有效证明。
- [ ] 接通头部单视角建图与 joint planning，严格使用实际头部视图，不复制成三份；保留机器人自碰撞和另一臂碰撞。
- [ ] 重新验证未知空间检查：定位新扫掠区域及其传感证据，去除仅凭全球投影可见性判断的错误门槛；未观测空间仍不得默认为空。无符合条件的路线返回覆盖不足。
- [ ] 运行 `python -m pytest tests/test_observation_motion.py tests/test_perception_mapping.py`；真实 GPU 验证水平启动、A→B 路径、目标碰撞保留和到位后视角改善。

## Task 4：attachment 和释放状态机

**Files:** 修改 `src/jev_mujoco/carrying.py`、`geometry_service.py`、`backend_builder.py`、`runner.py`、`verification.py`；扩展 `tests/test_carrying.py`，新增 `tests/test_hold_release_states.py` 和对应 GPU 场景。

**Interfaces:** holding state 使用 `empty/provisional/held/release_pending/unknown`，记录 arm、绑定实例、TCP 局部几何、最近观察及反馈时间。规划假设与实际状态分离。`Algorithms.validate_release_retreat(observation, arm, scene, trajectory, holding, release_volume, *, planning_settings)` 分别检查静态预计物体与意外随手两种假设，只有全部通过返回可执行；它复用 GPU 阶段碰撞检查，release_volume 为明确推断的可能落下范围。

- [ ] 写回归：机器人未撞而携物撞箱壁必须拒绝；只允许指垫必要接触；全开/全闭及途中扫掠受检；请求结束、异常和 Reset 无残留 attachment。
- [ ] 收窄当前整个 owning gripper 的持物碰撞豁免；保留实际指垫接触。用几何证据解决球包络误报，不增大整手豁免。
- [ ] 闭爪合理反馈后建立 provisional，同一有限提升同时检查已抓住/静态源仍在原处两种假设；由 post_grasp 证据确认 held，不调用 MuJoCo 接触对象 ID 判断持物。
- [ ] 支撑面接触只在离桌/下放过渡允许；保留桌面、箱底、箱壁，检查放置后开爪完整扫掠。
- [ ] 开爪到位进入 release_pending，实际 attachment 与规划假设分离；静态包络覆盖最后持物位置到支持面的可能下落范围，结合携带假设双重检查一次有限低速撤离，随后 post_place 确认。范围不可约束或反馈异常则保留事实并停止，不能自动松爪或假装空手。
- [ ] 运行 `python -m pytest tests/test_carrying.py tests/test_hold_release_states.py tests/test_place_service.py`；补真实 GPU 的箱壁碰撞、指垫接触、释放两假设、清理及坐标验证。

## Task 5：三种规划 skill 与执行衔接

**Files:** 修改 `src/jev_mujoco/algorithms.py`、`runner.py`、`decisions.py`、`placement.py`；新增 `skills/plan_grasp_and_place/{SKILL.md,contract.json}`，调整现有抓取/放置执行与验证合同；新增 `tests/test_plan_grasp_and_place.py`，扩展运行器回归。

**Interfaces:** `Algorithms.plan_grasp_and_place(observation, source_id, destination_id, arm, evaluations, scene, *, planning_settings, progress=None, cancelled=None)` 返回同一候选及分阶段预检结果；独立 `plan_grasp` 不再暗含放置语义，独立 `plan_place` 只接受实际 held。

- [ ] 写回归：只抓取不要求盒子；组合抓姿能抓不能放时继续下一真实候选；都无解则零抓取动作；空手独立 place 拒绝；重观察改变持物/盒后旧剩余轨迹拒绝。
- [ ] 提取共享抓取候选、携物提升和 native place 规划，保持候选分数及来源，不修改用户参数来绕过无解。
- [ ] 合同连接抓取执行、显式 post_grasp 观察/验证、剩余放置检查、释放撤离及显式 post_place 验证；所有返回事实有实际证据。
- [ ] 组合预检选择可检查持物和可在放置后观察的位置；不采用隐式无限观察/重试，不在失败时换手。
- [ ] 运行 `python -m pytest tests/test_plan_grasp_and_place.py tests/test_runner.py tests/test_runner_failures.py tests/test_planning_recovery.py tests/test_decisions_place.py`，并核对完整技能目录及 stop 仍由 JEV 选择。

## Task 6：高级设置、界面和实际验收

**Files:** 修改 `src/jev_mujoco/planning_settings.py`、`app.py`、`web/src/planning-settings.js`、`camera-panel.js` 及现有运行进度模块；扩展设置/前端测试、`tests/rgbd_pick_place_smoke.py`，更新 README 和部署记录。

- [ ] 在现有设置上增加观察 A/B 姿态与阶段等待，统一共享/抓取/放置设置；展示三个规划入口和观察节点、几何估计与持物状态。保持设置在单次运行内冻结。
- [ ] 通过相关 Python/前端测试及前端构建；用真实界面检查单位、错误说明、阶段图像和状态，不只看数据接口。
- [ ] 本地通过后按文件清单与哈希同步 Ubuntu，保留部署前归档，仅重启本项目所需服务，不修改共享 base 镜像或密钥。
- [ ] 真实 SAM3、GraspGen、cuRobo 与 MuJoCo 接触执行：独立 grasp 后 standalone place、组合 grasp_and_place，分别覆盖 A/B；全程无真值控制。
- [ ] 运行缺源/缺盒、腕盲区、重复颜色、估计不足、盒过小、携物/开爪碰撞、释放未知、取消及 Reset 的关键负例。单元测试通过不能代替物理抓放验收。
- [ ] 保存三路观察、规划场景和候选、attachment 生命周期、实际反馈、失败原因以及独立 oracle 评分。完整任务未成功之前不得报告完成。

## 执行顺序与交付

任务 1 → 2 → 3 → 4 → 5 → 6。任务 2 的几何处理与任务 3 的观测运动可在观察接口固定后分工并行；Runner 和公开接口由单一集成者负责。每项先补针对实际风险的回归，再做最小实现、针对性验证和审查；不重复跑与变更无关的长 GPU 流程。

最终交付为可运行本地/Ubuntu 源码、观察姿态配置、高级设置、三种规划 skill、实际端到端记录及可恢复部署备份。现有相机、盒子和部分 GPU 适配可以保留，但必须按上述新合同重新验收，不能因此前单项测试通过直接勾选本计划。

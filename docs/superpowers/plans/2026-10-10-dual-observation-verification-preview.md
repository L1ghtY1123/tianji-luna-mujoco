# 双臂观察、位置持物验证与三相机预览修改方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. 本文是待实施方案，不是实施完成记录。用户本轮要求方案，不执行产品修改。

**Goal:** 修正双臂观察运动，使用夹爪位置判断持物，明确各阶段视觉职责，保持 JEV 自主选 skill，并提供左腕—头部—右腕低频实时 RGB 预览。

**Architecture:** JEV 选择任务动作；skill 内部执行固定算法并返回事实。任务 RGBD、冻结碰撞场景和网页 RGB 预览分别保存。执行 skill 负责动作及末次采集，验证 skill 真正计算结果，避免重复采集和空壳验证。

**Tech Stack:** 当前 Python/MuJoCo、锁定 cuRobo、SAM3、GraspGen、TypeSafe/JEV、现有 Vite/Three.js 网页；OpenAI 仅作为可选视觉辅助。

**Spec:** 本文“设计约束”及本轮用户要求。此前方案与本文冲突时，以本文为准。

## 设计约束

- 控制链只使用 RGBD、标定和机器人自身反馈。仿真不添加 RGBD 噪声；环境物体真值和接触物体 ID 不进入决策。
- 夹持判断只使用夹爪位置，不增加力、力矩、电流或驱动状态码。TCP/关节反馈仍正常服务运动规划。
- 持物位置判据是一项工程假设，不能独立识别夹住的是哪个对象，也不能区分全部机械卡滞；日志明确证据来源。
- 抓后点云跟随不作为持物门槛，本轮删除该强制路径，不建立新的 ICP、外观追踪或遮挡重建系统。
- 初始观察运动必须是双臂 14 关节联合规划、同一时间轴同步执行，无串行降级。
- grasp 固定 approach/grasp/lift；固定开爪碰撞模型，无 attachment。Lift 按用户约定不增加持物避障。
- 抓取动作结束后 verify_grasp 根据位置结果建立 attachment；plan_place 必须等待该实际结果。
- attachment 使用抓前精细三相机可见点云，在闭合后的实际 TCP 下变换一次。默认闭合位移可忽略、刚性持物无滑移。
- place 前两段使用固定持物闭合模型和 attachment；实际开爪位置确认后 detach；第三段使用预先规划好的开爪无 attachment 模型。
- grasp/place 默认共享一次场景；JEV 决定显式重建。场景体素保持 10 mm；目标静态层排除，桌面、盒子及其他实测障碍保留。
- 外层由 JEV 每次选择 skill；内部不请求 JEV 选择下一子步骤。缺少证据返回 unknown，不等同 false。
- 网页按用户已选方案提供低频实时 RGB 预览，左腕—头部—右腕固定一排。显示与任务观察独立。
- 本次不修改 tianji_base、vendor 或官方 SDK，不增加自动换手、主动寻找视角或无限重试。

## 各阶段视觉职责

| 节点 | 采集与处理 | 输出用途 |
|---|---|---|
| 网页预览 | 低频 RGB，默认目标 2 fps；不生成深度或调用模型 | 人查看当前机器人视角；不更新任务状态 |
| 初始 observe | 同步三路 RGBD；头部查源/目的，双腕查源 | 确认所需对象存在及双腕覆盖；头部缺失按原规则报错 |
| 初始 move_to_observe | 使用已有头部 RGBD 建临时含目标地图，联合移动后采集 fine RGBD | 得到供抓姿、attachment、建图使用的可见点云 |
| build_scene | 读取选定任务快照，不重新拍照或运行 SAM3 | 建立并冻结碰撞场景 |
| generate_grasps | 三相机融合目标点云及实测场景点云 | GraspGen 候选，不补全隐藏面 |
| post_grasp | lift 与停留结束后采集一次 RGBD；默认不调用 SAM3 | 保存执行结果图像，供显示、人工及可选视觉辅助；不替换 attachment 点云 |
| plan_place | 已有场景、目标参考、持物点云与自身反馈 | 规划三段；不自动观察 |
| 松爪时 | 只读夹爪位置 | 确认开爪并 detach，不采集 RGBD |
| post_place | retreat 与停留结束后采集一次 RGBD | verify_place 的可见落点证据 |
| standalone place observe | 确认持物位置，观察目的容器；不要求重新分割手中物体 | 更新目的参考；是否建图仍由 JEV 选择 |

初始抓放 SAM3 为 4 次图像/文本请求；fine 为 6 次；post_grasp 为 0 次。verify_place 固定按头部、左腕、右腕检查已有 post_place 帧，每个视图最多源/盒子两次查询；可判定时结束，否则继续，最多 6 次。不扫描与任务无关的类别。缓存同一 capture 的处理结果，重复验证不重复推理。

## 验证与 attachment 的唯一职责

位置判断使用实际夹爪位置与空夹闭合基准的差值；必须已下发完整闭合指令，不能把“命令本身只要求闭一半”误判成持物。空夹基准来自夹爪配置/空夹标定，不在夹持中自动重标定。稳定窗口默认 0.2 s；初始位置容差取有效行程的 2%，是待样本验收的工程默认值，不是硬件精度承诺。窗口位置极差不超过该容差才视为稳定；稳定位置与空夹基准相差大于容差为 held，否则 empty。只有“位置容差”和“稳定窗口”两项验证参数，不增加力、电流或目标尺寸参数。实现时对现有空夹与已夹物样本验证这两个默认值。

- 稳定且明显未到空夹闭合位置：held，证据是 position。
- 稳定且到达空夹闭合位置：empty。
- 尚未稳定或位置不可用：unknown。
- 在已有闭合/提升/停留期间记录位置，不新增动作。提升后再次确认；位置显示空夹时不能继续 place。
- 开爪比较实际位置与完全张开基准。未到位则保持 attachment，不执行 retreat。
- 已张开只表示 open_confirmed；不能自动令 placement_confirmed 为真。

| skill | 修改后的固定职责 |
|---|---|
| execute_grasp | 张开 → approach → grasp → 闭合及停留 → 保存实际抓取 TCP、抓前点云和位置记录 → lift/停留 → 原始 post_grasp RGBD；返回执行记录，不作 SAM3 持物判断 |
| verify_grasp | 计算位置持物结果 → held 时构造固定闭合模型和抓前云 attachment → 发布持物事实；unknown 交回 JEV，无重复拍照 |
| plan_grasp / plan_grasp_and_place | 保留现有用户要求的两个任务入口，共用抓取规划算法；组合入口明确放置在实际持物后规划 |
| plan_place | 原场景中规划 attached approach/place，再以假想 open/detached 状态规划 retreat；不改变真实 attachment |
| execute_place | approach → place → 开爪位置确认 → detach → 释放停留 → 既有 retreat/停留 → 原始 post_place RGBD；返回执行记录 |
| verify_place | 处理上述固定快照，头/左/右顺序判断可见落点；发布 confirmed / contradicted / unknown，不再采集 |
| observe | 保留初始语义观察；允许 JEV 显式请求当前阶段适用的新任务观察，标明 purpose。新增观察不隐式更新地图 |
| identify_target / identify_destination / choose_arm | JEV 在外层选择 target_id/destination_id/arm 作为调用参数；这些 skill 内部只做固定校验和绑定，不再内部请求 JEV |

放置可见判据沿用实测目标点、盒子可见区域与盒底参考，不做完整容积检查。明确观察到目标在盒内才确认；明确在盒外才反证；被挡住、只见盒沿、缺少足够深度支持或身份歧义均 unknown。单次观察不声称长期稳定。完全不可见时不以原规划目标代替结果。

unknown 由 JEV 选择继续适用的观察、调用可选视觉辅助或结束为“待确认”。不自动把 unknown 改成失败，也不强制接上下一 skill。可选辅助不会从隐藏图像推断确定事实。物理执行错误保留实际已执行阶段和持物状态，仍按错误报告。

显式持物重建场景时，用当前 TCP 下的既有持物几何剔除重复的持物表面，不把该预测几何作为新的视觉观测。不能为重建强制恢复 post_grasp SAM3 依赖。

## OpenAI 可选辅助

增加一个按需的 inspect_visual_result skill，不进入默认固定步骤。JEV 选择后才调用，默认关闭；启用使用 Ubuntu 现有服务端配置，不向网页或日志输出密钥。

输入为明确的待判断问题、任务观察帧（优先相关相机，最多 4 张含必要前后对照）和简短位置/动作事实；不接收实时预览帧、不发送全历史和点云。调用视觉模型的 Responses 图像输入接口，返回 verdict、visibility、简短 evidence。模型由配置明确选择并验证支持图像，不在代码中猜测当前可用型号。

同一问题/同一观察结果缓存，最多一次推理；无证据、超时、无效输出均 unknown，不自动重试或执行机器人动作。辅助结果属于语义证据，不能更改位置持物记录或碰撞场景。官方接口参考：https://developers.openai.com/api/docs/guides/images-vision 。

## JEV 输入规模

用唯一一份紧凑 skill 目录替代完整 SKILL.md、重复 contract 和选项描述三份传输。每项含名称、用途、必要输入/前提、输出及配置可用性；所有当前技能仍可供 JEV 判断，程序不指定调用路线。

按问题投影状态：选 skill 只看任务、事实、候选摘要及最近必要结果；选臂才看双臂证据；选对象才看对象列表。图像、点云、轨迹、逐帧反馈、完整日志留在服务端，文本以 ID 引用，不靠让 JEV 猜测未提供的信息。

请求前保守估算 state+question token，工程软预算 24k，为官方 32k 单问题限制留余量；记录提供商实际 input_tokens。无官方精确 tokenizer 时标注估算，不伪称精确计数。只裁减重复/非当前决策所需证据，不能静默删目标或技能；仍超限则返回明确预算错误。远期目录超过 Choice 容量时再引入 JEV 两级选择，本轮不搭建大型检索系统。

官方当前限制为请求总计 64k，state+最长问题 32k：https://docs.typesafe.ai/models 。

## 网页预览

固定顺序左腕 camera 2、头部 camera 3、右腕 camera 1。三卡等宽、不镜像、不拉伸；窄窗口维持横排并横向滚动。默认显示 RGB，不把深度并排挤占主画面。保留“最近任务观察”切换，用来查看验证实际使用的 RGB/深度。

预览默认目标 2 fps，每组从同一机器人/场景快照渲染三路 RGB，独立 preview_frame_id 和时间。任务 capture_id、最新任务观察、SAM3 结果、场景 token 均不能被覆盖。只保留最新预览；单个渲染任务在途，忙时跳帧、不积压。当前一组完成后优先服务任务采集。页面隐藏/断开或无人查看时停止预览需求；多个浏览器共享最新预览，不各自触发全套渲染。

预览源按需从仿真物理快照复制后渲染，不长时间持有物理锁，不在物理步进线程运行渲染。场景 Reset/代次变化清除旧帧。UI 标明“实时 RGB 预览”或“任务观察”，并显示帧时间；处理中的图像不会混用。

## Review Focus

1. 单次联合规划不能在执行层重新变成先 A 后 B；两臂起点时间、采样间隔及轨迹长度一致。
2. 完全遮挡时，夹爪位置确认仍可建立 attachment；不能因语义实例缺失提前抛异常。
3. 放置末次图像不可见时，保留开爪事实但 placement 为 unknown，JEV 仍能接收结果。
4. 网页连续刷新与任务采集竞争时，不覆盖捕获身份，不延迟成积压队列，不触发 SAM3。
5. 紧凑 JEV 输入保留决定下一 skill 所需的真实证据；不能通过暗中删除技能变成固定路径。

## 实施任务与验收

### Task 1 — 双臂观察运动

Files: `src/jev_mujoco/algorithms.py`, `geometry_service.py`, `runner.py`, `simulation.py`, `skills/move_to_observe/*`；扩展现有 observation/dual 测试。

Interface: `plan_dual_observation_motion(snapshot, scene, goal_joints, planning_settings)` 返回统一 dt、A/B 轨迹、场景身份与联合碰撞证据；`execute_dual_trajectory(paths, dt, generation)` 原子提交双臂同一起始时刻。

- [ ] 先以测试复现当前两次单臂调用及不同步执行。
- [ ] 复用 vendored 原有双臂位姿接口；两个配置观察关节位姿先经自身 FK 得到双 TCP 目标，保持现有观察参数语义。
- [ ] 完成联合轨迹后处理、全 14 关节命令碰撞检查和原子执行；不并行调用两个单臂规划器。
- [ ] CPU 检查接口、起止、取消/Reset；真实 GPU 验证两臂互碰障碍，MuJoCo 日志核对同一起始时刻。

### Task 2 — 位置验证和视觉职责

Files: `verification.py`, `runner.py`, `perception.py`, `algorithms.py`, `planning_settings.py`, `skills/observe/*`, `skills/execute_*/*`, `skills/verify_*/*`；`tests/test_fixed_skill_flow.py`、visible workflow/verification 与新位置判据测试。

Interface: 位置判定返回 held/empty/unknown 与位置证据；capture 与语义处理分离；放置判定返回 confirmed/contradicted/unknown。执行记录保存真实末次 capture、闭合后 TCP、位置窗口和动作结果。

- [ ] 测试空夹、正常持物、提升中变为空夹、位置未稳定，以及 RGBD 全遮挡。
- [ ] 从 execute 中移出重复判定到 verify，使后者真正计算结果；每次动作结束最多采集一次任务 RGBD。
- [ ] verify_grasp 成功后一次建立抓前云 attachment；post_grasp 无 SAM3、无点云跟随门槛。
- [ ] verify_place 使用同一 post_place 快照执行固定头/左/右算法；松爪不采图、无 release 重建或撤离重规划。
- [ ] 测试开爪未到位不能 detach/retreat，开爪到位且目标遮挡必须 unknown，明确盒外不能成功。
- [ ] 删除只有旧强制跟随/抓后识别路径使用的助手函数、错误文案和测试；保留三相机抓前融合回归。

### Task 3 — JEV 调度和输入预算

Files: `decisions.py`, `skills.py`, `runner.py`, `tests/test_decisions.py`, `tests/test_runner.py`, `web/src/main.js`、`run-evidence.js` 及相关测试。

Interface: 精简 skill 目录；JEV 在调度层选择 skill 及必要的 target_id/destination_id/arm，执行入口消费已绑定参数，内部不持有 Decisions。沿用目前分开的有限选项判断，不新增自由文本参数生成。实际位置、可见落点、待确认分别作为事实。unknown 不生成未经验证的成功事实，任务可停留或结束为 needs_attention。

- [ ] 测试相同阶段 JEV 可选择不同合法 skill；仍向其提供当前完整技能目录。
- [ ] 实现按决策种类的状态投影，去除提示词中的旧抓后建图与写死下一 skill 指令。
- [ ] 实现预算估算与服务端真实 token 日志，添加扩大目录/候选/历史的边界测试。
- [ ] 让 unknown 回到 JEV；完成条件由真实结果检查，不能只凭模型宣告成功。

### Task 4 — 可选视觉辅助

Files: 新建 `src/jev_mujoco/visual_assistance.py`、`skills/inspect_visual_result/{SKILL.md,contract.json}`；修改 `runner.py`、配置与相应测试。

- [ ] 测试未启用/无配置、图像上限、一次调用、结果缓存、超时和不可见返回 unknown。
- [ ] 使用已有服务端 OpenAI 配置创建独立图像判断接口；不自动选择/切换型号，不导出凭据。
- [ ] 以录制的清晰、部分遮挡和完全遮挡任务图像验证结果与耗时，常规任务 OpenAI 调用应为零。

### Task 5 — 三路实时 RGB 网页

Files: `simulation.py`, `cameras.py`, `app.py`, `web/src/camera-panel.js`, `web/src/styles.css`, `web/index.html`, `web/tests/camera-panel.test.js`、相机 API 测试。

Interface: 独立 preview 元数据/图像接口，不改变 `/api/cameras` 的任务观察语义；每组统一 preview_frame_id，页面顺序固定 2/3/1。

- [ ] 测试预览刷新次数增加但任务 capture/SAM3/build_scene 调用数不增加。
- [ ] 实现限频、最新帧缓存、忙时丢帧及无人查看停用；串行复用可用渲染资源。
- [ ] 实现左/头/右横排、模式与时间提示；测试乱序响应、Reset 后旧帧与任务帧隔离。
- [ ] 桌面和窄窗口实际浏览器验收，确认顺序、横排、无镜像以及机械臂运动中画面持续更新。

### Task 6 — 完整验收与同步

- [ ] 本地 Python 测试、Ruff；网页 `npm test` 和 `npm run build`。
- [ ] Ubuntu 真实 GPU 检查双臂联合观察、attached place 两段和 detached retreat；记录 SDK 身份。
- [ ] MuJoCo 完整运行单独抓取、抓取并放置、已持物独立放置；加空夹及放后遮挡案例。仿真真值仅在独立验收报告中对比，不进入控制输入。
- [ ] 核对完整任务的 SAM3/OpenAI 次数、建图次数、预览刷新与 token 用量；不以单元测试替代物理闭环结果。
- [ ] 更新各 skill 的实际输入输出及流程说明，保存证据；确认 Windows/Ubuntu 变更文件哈希一致。只管理本项目服务与验证容器。

## 本轮产物

仅本修改方案。尚未修改产品源码、启动服务或调用付费视觉 API。没有 Git 仓库时不创建虚构提交；实施时采用现有备份与双端哈希同步方式。

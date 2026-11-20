# 场景理解与实例关联重构设计

日期：2026-11-17；2026-11-18 用户确认按推荐方案实施。状态：实施中。

## 1. 用户目标与范围

用户明确授权抛弃现有实例关联设计。Responses 应理解完整工作空间环境，同时表达任务目标和目标放置位置；实例关联应利用 SAM3、标定 RGBD、FoundationPose、IoU，并由 Decisions 判断跨相机对应、新实例、跨时刻身份。

本设计中的完整场景指当前工作空间内可观测的所有物理对象、容器和支撑表面，包括非任务物体。遮挡或未观测范围必须明确表示，不把三张图或一次检索等同于全场景覆盖。配置的释放区域与物理容器/桌面分开表示。

重构范围包含编译输入输出、候选感知、关联证据、身份数据库提交、任务绑定、相关回放和技能合同。机器人仍只通过 Base；不读取 Simulation 物体真值，不更换 SDK，不顺带改运动控制或规划求解器。

## 2. 已核实的问题

证据基线：`data/acceptance/transfer-ten-cuboid/report.md` 及十次 `events.json`。完整任务 0/10，五次首要失败为 `task_grounding_unknown`；另五次包含子图绑定、执行和路径规划问题，不把它们全部算成关联错误。

| 事实 | 直接证据 | 设计影响 |
|---|---|---|
| 非任务场景提案会被删除 | `runtime/compiled_task.py:task_entity_proposals`；只读探针输入 target + visible_other_object，输出仅 target | 场景清单不得以任务成员资格过滤 |
| 候选描述被作为任务外观条件 | 第 10 次 Responses 把用户要求的黑色物体描述成 roughly rectangular solid；外观题最终 no=0.52，支撑题 yes=0.55 | 用户明确要求与模型观测猜测分开存储，后者不能变成额外任务要求 |
| 当前相机对应可能留下重复物理组 | 第 10 次头部右盒和右腕右盒保持分离，两个组均满足 robot_right，导致 unique_task_subject_unconfirmed | 需增加实际可比较的几何证据与全局占用检查，而非只比较文字和轮廓 |
| 一处不确定会取消整个身份提交 | `ground_observation` 汇总 missing 后返回 unknown，`prepare_grounding` 不提交；已确认桌面也无法登记 | 场景身份更新与任务可执行性拆开 |
| FoundationPose 不能参与首次身份判断 | `CadGeometryResolver.resolve` 要求 group 中 entity_id 已等于 target.id，调用位于身份确认之后 | 位姿假设应允许以临时候选为输入 |
| 初始身份参考事件名不匹配 | 登记写 instance_rgbd_grounded；`select_identity_history` 接受 instance_registered/processing_result/成功 observe_scene；只读探针得到新事件 0 条、旧事件 1 条 | 初始参考需要明确持久化入口，并修复档案选择 |
| SDK 能力未通过服务暴露 | 锁定 SAM3 提供 add_geometric_prompt；现有服务仅 segment_text | 通过算法服务适配器开放官方框提示，使 Responses 框真正有用途 |

以上是代码与日志诊断。重复右盒是否完全由缺少几何证据引起、框提示能提高多少成功率，仍需冻结观测对比验证。

## 3. 方案选择

两个可行方案：

1. 保留 Responses 框，输出完整场景清单；SAM3 全图文本检索提供召回，框提示提供补充候选；统一几何证据后交 Decisions 关联。优点是保留环境理解的定位信息，并能修正漏分割；代价是扩展自有 SAM3 服务协议。
2. Responses 只输出场景类别和任务语义，SAM3 文本检索加深度残余区域发现物体。接口改动较少，但放弃已有定位信息，对相似物体和漏检补救更依赖额外检索。

推荐方案 1。用户已把是否保留框交由实现方判断；用户已于 2026-11-18 确认实施。不会保留两个并行主框架。框的意义始终是候选空间提示，不是实例身份证明。

## 4. 总体数据流

```text
Base 冻结三路 RGBD、反馈、标定和工作空间范围
    → Responses 一次输出场景清单 + 任务选择器 + 目的地区域 + graph
    → SAM3 文本/框提示候选 + 深度残余范围检查
    → 单视角 mask 整理、三维测量、候选 CAD 位姿
    → 同次跨相机证据与 Decisions 关联
    → 当前场景物体组
    → 历史证据与 Decisions 身份关联
    → 独立提交已确认实例和当前观测
    → 在场景实例上绑定任务目标与目的地
    → 几何建图、规划、抓放
```

后续 observe_scene 走相同场景更新链。缺少某个任务目标，不得阻止无冲突的场景事实落库；未知身份的有效可见深度继续保留为匿名碰撞几何。运动仍必须通过自己的观测、地图和碰撞前提。

## 5. Responses 的新合同

`InitialPlan` 分开表达以下数据，不能让一个 EntityProposal 同时兼任它们：

- `scene_inventory`：工作空间内当前可见物理对象的逐相机提案，每条含本地 proposal_id、camera_id、可选 bbox、可见描述、检索短语、可选 CAD 类别假设和 visibility。各相机提案默认独立，暂不拥有永久 ID。
- `task_selectors`：目标、来源、目的地、支撑及参考关系。`requested_attributes` 仅来自用户要求，保留原文依据；模型猜测的颜色以外形状、包装或纹理只进入 observed_description 和检索提示。
- `destination_regions`：区分视觉可见位置与配置区域。对可见放置面可给图像提案；无标记配置区域保留既有 region ID，由程序根据实测支撑面、配置坐标和当次标定投影显示。框本身不生成三维放置点。
- 原有 TaskSpec 和显式 graph：保留一次编译、有限恢复、采集和动作权限；绑定对象改为 scene inventory 上的任务引用。

所有物体都进入场景感知，不再调用 task_entity_proposals 删除环境对象。Responses 的跨相机对应建议若保留，只作为可审计假设。盒子与内容物、桌子与桌面上的工件有独立物理身份。

## 6. SAM3 候选与场景覆盖

算法库保留旧 segment_text 合同以服务实际既有调用者，新增独立的框提示操作，调用锁定官方 `Sam3Processor.add_geometric_prompt`。Responses 的归一化 xyxy 在协议边界转为官方要求的 cxcywh；每个独立提示重置 prompt 状态，避免相邻对象的提示累积。

候选来自完整场景的文本查询与本次有效框提示。保留全图文本检索结果，框偏移或遗漏不会直接排除候选。官方框提示仍可能产生多个相关 mask，需要同图 IoU、深度和 Decisions 选择，不能把返回第一项当成框内目标。

每个候选保存原始 mask、工作空间内 mask、有效深度、查询/框来源以及采集和标定版本。可选的已知机器人深度过滤使用现有公开机器人几何接口；机器人表面及混合掩膜必须可标为无效，不能凭含有效深度就当物体。

同视角完全相同 mask 合并来源；高 IoU、包含或分裂关系作为显式分割歧义，不能一律 NMS 合并。两个接触物体不能只因点云相连就获得一个 ID。

对未被候选解释的当前有效深度，保留残余范围和匿名几何；用有限的通用物体/容器检索与由残余区域提出的框提示补检索。残余点云组件只是区域提案，不能直接分配 ID。后续新物体可使用初始类别、CAD 目录类别和通用 object 类型进入同一路径，不依赖再次调用 Responses；Decisions 判断其物理有效性和任务归属，不重写任务语言。

输出 coverage 记录各可见提案是否得到 mask、残余范围、遮挡与未观测范围。不能保证找到完全遮挡的物体，也不把 SAM3 空结果当成场景为空。

## 7. 在 ID 分配之前构造统一几何证据

`CandidateObservation` 是单相机观测，不带永久 ID；新增 `PoseHypothesis` 绑定 candidate_id、asset_id、当前 capture、mask 哈希、CAD 哈希和标定。

无 CAD：使用实际可见点云、稳健范围、点间距离和双向深度投影，不拟造完整实体体积。粉色物体仍按未知几何处理，不将近似粉色 CAD 用于精确配准。

有 CAD：根据用户声明目录及当前候选类别形成模型假设，在永久身份确认前调用现有 `CadPoseEstimator.estimate_instance`。初次用 register，不需要 instance_id；一般先处理有效深度最充分的视角，只在跨相机证据不足时增加第二视角配准。同一 CAD 允许多个当前实例，不能把 asset_id 当 instance_id。

FoundationPose 结果是当前位姿假设，不自带永久身份，也不具有官方 API 未暴露的可信度分数。其投影和几何一致性进入 Decisions 证据；不新增 ICP，不修改官方估计器。缺少 CAD、失败或对称姿态歧义必须保持显式状态。

几何指标按比较条件使用：

| 指标 | 合法比较 | 解释 |
|---|---|---|
| proposal-box/mask IoU | 同相机同次采集 | 分割与提案是否对应，不是身份结论 |
| mask-mask IoU | 同一像素坐标系 | 检测重复、包含、混合候选 |
| 跨相机投影 IoU/coverage | 世界点云或 CAD 先投到目标相机 | 必须 z-buffer，并用目标深度处理遮挡；只比较有效共同可见范围 |
| 可见点云范围 IoU、距离 | 同次统一机器人坐标 | 仅为部分表面证据，缺少重叠不能直接否定对应 |
| CAD 位姿/包围体 IoU | 同次统一世界坐标中的同模型假设 | 支持两视角同物体假设；对称姿态不能仅按欧拉角差排除 |
| 跨时刻投影/体积 IoU | 明确写出静止假设或动作预测 | 零重叠不证明新物体；相机运动必须补偿 |

共同可见支持为空时返回 unavailable，不返回假零 IoU。数值误差参数由当前传感器/标定配置和回放标定取得，记录其来源，不把任意经验阈值称为测量证明。

## 8. Decisions 的关联职责

Decisions 只裁决实例与必要视觉事实，程序负责计算几何、组装候选、检查唯一性和提交。每个 HTTP 请求仍为一个问题，原图、轮廓、投影和历史图的角色明确标注。

跨相机问题：针对当前主体，在同一份联合证据中显示相关相机的所有竞争候选和数值比较；选项为可行的跨相机组合，包括仅单视角可见、无有效分割和 unknown。单视角物体是合法结果，不强制三路都有 mask。组合只按明确几何不可能性剪枝，不凭类别短语不相同彻底排除。

先独立处理同视角重复/分裂关系，再形成“一相机一个已整理观测”的组。所有组合选择都经过候选独占及两两几何一致性检查；同一 mask 不允许被两个物理组占用。同一冲突组件最多一次联合复核，不沿循环反复改变独立答案。组合规模超出预算时明确返回 evidence_budget_exceeded；不能截去可行组合后宣称唯一。

跨时刻问题：当前物理组与所有相关历史实例竞争，提供当前三维/CAD、历史外观与 mask、初始参考、最近参考及实际操作。候选选择为某个 existing_id、new_instance、invalid_segmentation、unknown。新实例判断必须同时显示可能对应的丢失/遮挡旧实例，不能把 no_match 自动改成 new。

数据库为空时不做虚假的历史匹配；Decisions 在同次分组物理有效性裁决中确认新实例。数据库非空时，已确认的当前对应同样进入竞争约束，防止另一组给同一物体重复注册。

概率解释沿用当前合同，confidence 仅归档；不靠调低门槛减少 unknown。联合互斥、缺失证据、物理矛盾是程序独立约束，不能被模型高概率越过。

## 9. 时间记忆与状态

实例状态至少区分当前可见、遮挡/未定位、执行推定携带、退役。未看见只更新可见性，不删除永久身份、不生成替代 ID、不把旧位姿写成当前测量。

持久保存不可覆盖的首次确认参考和可更新的最近确认参考。所有登记、观测、加工事件使用统一证据结构；旧事件记录只在存储读取边界兼容，修复 instance_rgbd_grounded 档案遗漏。保留现有数据库、ID、历史及 world_key，不清空数据库来制造成功率。

场景更新返回 confirmed_instances、unresolved_candidates、missing_tracks、coverage；任务绑定单独返回 bindings 与 unresolved_selectors。同一事务提交本轮无冲突的确认结果；有冲突的关联组件整体不提交，独立组件可以提交。任务 unknown 不回滚独立物理事实。

提交前后仍校验采集、机器人动作代次、任务取消、数据库 revision。幂等键依赖物理观测/确认事件，不能用任务 ref 充当实例创建键。

抓取执行链和放置回执只作为有来源的运动证据：保持当前 explicit `visual_identity_verified=false` 含义，不能重构时悄悄标为独立视觉确认。重新可见后再走统一关联。没有连续观测时，两个完全相同物体交换位置可能无法消歧，应保留 unknown。

## 10. 任务选择与几何消费

任务选择器在已建立的当前场景实例上运行。颜色/类别等只使用用户明确条件与当前核验事实；Responses 自己猜的“矩形”“包装”等不能导致用户只要求的“黑色物体”被否定。

机器人左右、相对位置、数量和序号保留明确坐标定义。once 绑定锁定已确认永久 ID，移动后继续找该 ID；each_selection 用于集合下一轮。配置区域保留空间参考，不登记为伪物体。

绑定表需成为 graph 参数、subgraph input_id、object_tasks、task ledger 的唯一数据源，消除已识别 ID 与活动子图归属不一致。第 2 次 target_outside_active_object_task 作为接口回归验收，不把它混进视觉准确率。

几何建图消费本次确认的实例几何和未归属的有效深度；FoundationPose 当前结果复用到建图，不因分配 ID 再跑一次。历史不可见物体可以保留身份和过期几何来源，规划前提独立验证，禁止用部分场景识别成功直接许可运动。

## 11. 模块边界与替换清单

- `agent/schema.py`、`graph_prompt.py`、`responses.py`：完整场景清单与任务选择器分离。
- `candidate_observation.py`、`perception.py`：全场景候选、框提示与覆盖；SAM3 HTTP 客户端仍归算法库，Luna 仅依赖其公开接口。
- `association_depth.py`、`observation_overlap.py`、`pose_estimation.py`：统一候选几何、可见性投影、CAD 假设。
- `current_instances.py`、`temporal_identity.py`：用统一关联证据替换现有多轮局部对应与补丁链。
- `task_grounding.py`：只保留任务选择/绑定职责，停止拥有整个场景身份的成功开关。
- `agent/registration.py`、`object_database.py`、`memory_reference.py`：场景事实提交、不可变初始参考与当前参考。
- `runtime/grounding.py`、`compiled_task.py`、`native_graph.py`：编排、结果消费与唯一绑定表。
- 算法库 `services/sam3_service`：框提示协议/适配/契约验证；FoundationPose 复用固定版本公开接口。
- 当前 AGENTS、README、项目技能、ground_task/observe_scene 合同、回放同步修改，删除“只识别任务物体”等已被用户新要求替代的指令。历史实验保留。

不增加平行项目、worktree 或 sub-agent；不修改官方 SDK。现有具体名称可在实现中简化，但不能同时留两条可执行关联主链。

## 12. 验收与交付证据

先固定输入诊断，再现场实验；不以减少 unknown 或健康检查代替正确关联。

1. 把现有十次原始采集作为冻结回放集，人工核对可见对象/mask 对应；只评估证据中可区分的身份。仿真配置如用于评分，必须隔离在评估端，不能传给关联算法。
2. 保留本次只读探针的两个可复现结果：环境对象被过滤、首次参考新事件被遗漏。新实现必须通过相反的验收条件。
3. 检查框偏移/缺框、文本漏检、重复/分裂 mask、左右同 CAD 盒子、只单视角可见、缺深度/遮挡，输出具体阶段原因。
4. 检查普通物体首次登记、重复提交、跨相机不重复 ID、物体移位保留 ID、非目标物体移位、新物体出现、遮挡重现、同形物体不可区分时 unknown。
5. 检查明确任务失败但独立场景事实可提交；无关物体身份不确定不能被误当成整个地图无效，也不能被用来绕过地图安全检查。
6. 检查 FoundationPose 候选前置、每实例跟踪状态隔离、CAD 对称性以及同图/投影/三维 IoU 的输入来源。无共同可见像素时不产生伪失败。
7. 按仓库约束在 data/acceptance 保存一次性诊断/回放结果，不在 Luna 新建 tests 目录；算法库运行既有 unittest 和框提示合同检查。静态检查、web build、实际入口分开记录。
8. 正式验证用同样的 80×60×110 mm 粉色长方体和黑件各五次任务；同时报告完整任务结果、关联错误、ID 切换/重复、漏检、各阶段 unknown、请求数、耗时与其他执行失败，不隐藏失败试次。

只有通过对应验证后才声称改善。若原有轨迹起点、下发或路径规划问题仍阻止交付，明确列为独立缺陷，不伪称本次关联重构已经解决整个任务可靠性。

## 13. SDK 依据与实施约束

- SAM3 固定提交 `5dd401d1c5c1d5c3eedff06d41b77af824517619`，已核对官方 `set_text_prompt`、`add_geometric_prompt`、`reset_all_prompts`：https://github.com/facebookresearch/sam3/blob/5dd401d1c5c1d5c3eedff06d41b77af824517619/sam3/model/sam3_image_processor.py
- FoundationPose 固定提交 `a1b694b83e633c2cb6115b9063d940a687759392`，已核对官方 register/track_one 示例及本地适配：https://github.com/NVlabs/FoundationPose/blob/a1b694b83e633c2cb6115b9063d940a687759392/run_demo.py
- 设计基线的 SAM3 服务协议不含框提示；本次实施已扩展服务和 CPU 客户端，具体部署及验证证据见实施计划与验收目录。
- SDK 源码保持只读；服务端若因协议扩展需要替换镜像，先完成兼容验证，Luna 空闲时只替换本用户相关服务，保留其他调用者的 segment_text 行为。

## 14. 自审

按仓库禁止 sub-agent 的要求自行复核。设计已区分物理实例与任务角色、CAD 类别与实例身份、实测与动作推定、未知可见性与不存在、部分场景更新与运动许可。最主要的待实测风险是框提示收益、模型对同形物体的消歧能力、增加 CAD 前置计算的延迟和遮挡下几何指标的有效支持。

本设计已于 2026-11-18 获用户确认并进入实施。当前实现说明见 `docs/scene-association-implementation-20261118.md`；验证结论以验收目录中的具体版本和试次为准。

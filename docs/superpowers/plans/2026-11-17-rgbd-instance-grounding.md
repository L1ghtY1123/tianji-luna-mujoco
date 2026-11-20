# RGBD 与 IoU 实例关联及延后身份登记 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository instructions prohibit sub-agents and parallel worktrees.

**Goal:** 以机器人坐标系解释任务，用同次多视角 RGBD 完成定位与身份核验后才提交永久 ID，统一首次和后续观测路径。

**Architecture:** Responses 编译不可变语义选择器和显式图，SAM3 根据任务相关检索词生成无永久 ID 的候选。当前分组、任务定位、历史匹配各自返回证据，由 Runtime 统一检查后经数据库事务登记。模糊候选和无任务身份的可见障碍分别保存在观测证据和匿名碰撞地图中。

**Tech Stack:** Python/Pydantic、NumPy、现有 SAM3 segment_text、Decisions、SQLite、现有 cuRoboV2-Fast v0.9.0、现有 Luna 单页工作台。

**Spec:** [机器人坐标系下的 RGBD 实例定位与身份登记](../specs/2026-11-17-rgbd-instance-grounding-design.md)。开始实施前完整阅读此设计和仓库当前指令。

**执行状态：** 核心实现已部署；本文件保留原始验收清单。实际完成证据见 [执行记录](2026-11-17-grounding-progress.md)，未完成的全场景测试和 A/B/C 对照不得视为通过。两种搬运任务的完整验收仍未通过。

## Global Constraints

- 自然语言中的左右、前后、上下及排序统一以机器人固定坐标系 Link_Base 为准：+X 向前，+Y 向左，+Z 向上。
- 三路 RGBD 同步采集；头部相机提供整体上下文，腕部相机补充细节和盲区。
- 每任务仍只有一次 Responses；无框相机仍参与任务相关候选检索。
- 未完成 RGBD 关联和任务/历史核验，不得 allocate 永久 ID，不得执行物理动作。
- `proposal_ref`、`candidate_id`、`group_id`、`entity_id` 是不同引用，不能靠字符串外形混用。
- 不更改共享 SAM3 wire contract，不修改 SDK，不新建算法镜像，不访问仿真真值。
- 不引入 DINO/DINOv2 或其他新增外观特征模型；IoU 只在同一次采集的同一相机图像坐标系计算，不直接决定身份。
- 保留三个项目独立启动、Base 唯一机器人接口和 Luna 单 HTML 工作台。
- 依仓库规则不创建 tests/、web/tests/ 或工作树；验证驱动放在 `%TEMP%/luna-grounding-validation`，结果归档 `data/acceptance/instance-grounding`。
- 当前工作区有大量已有改动，实施时按文件检查差异，不覆盖、不批量提交其他工作。

## Review Focus

1. 相机画面反向/裁剪和同色同类查询合并造成选错目标：任务 1、2、4 必须有等价变换与多查询验证。
2. 无框、漏检和深度无效被误当成目标不存在：任务 2、4、7 必须有未知结果验证。
3. 同一物体多视角重复计数或两个相同物体误合并：任务 3、4 必须有分组置换、冲突和遮挡验证。
4. 历史错误登记、运动后位置变化和事务中断造成 ID 污染：任务 5、6 必须检查数据库差异和幂等性。
5. 定位尚未成功时恢复路径绕过 graph 或删除背景障碍：任务 6、7 必须拒绝越界动作并保留匿名碰撞。

## 接口和文件职责

新增以下 Luna 应用模块，不复制算法服务：

- `src/luna_mujoco/core/grounding.py`：冻结的 SelectionSpec、ViewHint、CandidateObservation、CurrentGroup、HistoryMatch、GroundingResult 数据合同。图像数组仅留在执行层；模型上下文使用显式摘要。
- `src/luna_mujoco/candidate_observation.py`：任务相关检索、同图框/mask 证据、候选深度和来源；复用现有 SAM3 客户端及反投影函数。
- `src/luna_mujoco/observation_overlap.py`：纯 NumPy 的框/掩膜重叠指标，输入有效性和确定的像素边界规则；不读数据库、不调用模型。
- `src/luna_mujoco/task_grounding.py`：机器人坐标系谓词、三值组合、排序/数量及观测覆盖检查，组织必要的视觉判断。

复用并改造 `current_instances.py`（当前分组）、`temporal_identity.py`（历史核验）、`agent/registration.py`（仅经验证结果提交）、`runtime/controller.py`（编排）。不得同时保留旧的 RGB-only 身份登记入口。

下文 Python 相对路径均相对于 `src/luna_mujoco/`，`skill_catalog/`、`web/`、`docs/` 相对于仓库根目录。

共享数据约定：

```text
SelectionSpec(ref, role, queries, scope_ref, relations, quantifier, binding_mode, view_hints)
CandidateObservation(observation_id, generation, candidates, query_evidence, coverage_evidence,
                     proposal_overlap_evidence, mask_pair_evidence)
CurrentGroup(group_id, observation_id, candidate_ids, geometry_evidence, correspondence_evidence)
HistoryMatch(group_id, outcome=existing|new|unknown, entity_id?, evidence)
GroundingResult(status=success|unknown|failure, observation_id, generation,
                selected_groups_by_ref, current_groups, history_matches,
                missing_evidence, evidence)
```

group_id 和 candidate_id 均携带该次观测范围；同相机 mask 内容哈希生成稳定候选键。Query 去重键为 observation/camera/规范化文字，不以 kind/color 代替文字。不可变结果只能引用同一次采集，除明确标为历史的 HistoryMatch 证据。

## Task 1: 语义合同和 Responses 只读提案

**Files:** 新建 `core/grounding.py`；修改 `core/graph.py`、`agent/schema.py`、`agent/responses.py`、`agent/graph_prompt.py`；删除替代后的 `agent/initial_identity.py`。

**Interfaces:** `compile` 产出原策略图及 `SelectionSpec`；`view_hints` 支持每视图多个候选/未见/歧义；永久 ID 提案只能出现在建议字段中。

- [ ] 修改运行代码前，先用临时验证驱动采集设计对照 A 的基线，归档固定输入、原始回答、模型/源码版本、初始数据库快照及指标。后续比较使用该归档，不为保留 A 建平行项目、工作树或旧运行入口。
- [ ] 在临时验证驱动创建反例：机器人 Y 大的一件出现在头部画面右侧；确认当前描述参照不固定的实现不能满足新语义合同。
- [ ] 定义冻结合同、机器人坐标系谓词白名单和三值结果。保留任务原文；把 selection order 与抓取执行顺序分开。
- [ ] 修改 Responses 图像顺序为 3、2、1；取消参照相机决定语言方向；区分清晰视图和语言空间坐标系。
- [ ] 初次 Decisions 不再确认永久身份。删除原 `confirm_proposal -> allocate` 的可执行调用，不能只新增旁路后保留旧路径。
- [ ] 验证空/错框、多个同视图候选、未支持谓词及机器人左右/最左区别；编译期间数据库行数必须不变。

## Task 2: 从任务选择器提取候选，不依赖永久实例

**Files:** 新建 `candidate_observation.py`、`observation_overlap.py`；修改 `perception.py`、`demo_perception.py`、`association_depth.py`、`planning_settings.py`；共享服务保持不变。

**Interfaces:** `collect_candidates(capture, selections, detector, checkpoint=None) -> CandidateObservation`。输入 detector 使用现有 segment(rgb,prompt,timestamp_us)；输出包含所有必要类别候选及原始失败原因。

重叠纯函数：`box_mask_metrics(box_xyxy_normalized, mask) -> dict` 返回 box_iou/mask_coverage/box_fill；`mask_pair_metrics(left_mask, right_mask) -> dict` 返回 mask_iou、两个方向的交集占比和 exact_alias。输入非空二维 bool mask，尺寸一致；无效输入抛明确验证错误，由调用层记录 invalid。调用层先校验 observation/generation/camera/size，补入来源字段；无框直接生成 hint_status=absent、指标 null，不调用计算函数。

- [ ] 先验证现有查询按 kind/color 合并导致不同形状检索丢失的反例；无框相机中存在目标也必须被检索。
- [ ] 将现有分割和有效深度提取拆成可复用前半段，后半段几何估计不重复调用 SAM3。保持同 capture 的缓存语义。
- [ ] 按完整规范化查询去重；收集目标/参照/排序范围所需候选。SAM3 模型返回的 selected_index 不作为任务选择结果。
- [ ] 在同图坐标计算 proposal-mask 关系作为证据，不裁 mask、不按 IoU 单独确定身份、不删除所有框外候选。
- [ ] 实现设计中的半开框、像素中心覆盖及 mask 外接框规则。矩形框与外接框 IoU 为连续面积，轮廓覆盖为像素计数；使用原始轮廓而非有效深度子集。
- [ ] 给每个框保存所有兼容候选的指标矩阵；不按 top-1 丢掉候选。完全相同 mask 合并时保留查询来源；mask_iou>=0.90 仅产生 suspected_duplicate 诊断，不执行 NMS 或合并。
- [ ] 新增数值验收：相同矩形 IoU=1、分离矩形=0、相等 mask 的 mask_iou=1；松框/紧框/空心轮廓、一个框含两个物体、包含型掩膜必须保持正确指标和候选数量。无框为 null；空 mask/非法框/不同尺寸和混合观测显式拒绝。
- [ ] 深度不足保留 RGB 候选和原因；测量质量门槛沿用 30 点/0.5 有效率。保留原始 RGBD，工作空间外数据不能当空场景证据。
- [ ] 验证多框、无框、错框、框跨物体、类别别名、相同 mask 去重、缺深度；提取阶段不得写 ObjectDatabase。

## Task 3: 当前跨相机分组不读取历史 ID

**Files:** 修改 `current_instances.py`、`association_depth.py`、`identity_association.py`、`subject_assignment.py`、`raw_visual_input.py`。

**Interfaces:** `group_current_candidates(client, capture, candidates: CandidateObservation) -> tuple[list[CurrentGroup], list[dict]]`；它不接受 entities/history。复用现有 current_instance_groups 的无 seed 路径和冲突检查，禁止永久 ID 产生的种子。

- [ ] 用同物体三视图、两个近邻同形物体、一个视图缺失和重复掩膜构造当前分组验证；先固定预期失败案例。
- [ ] 保留每个候选独立三维证据，统一发送到 Decisions；保留原始模型选择与独立程序验收结果。
- [ ] 不跨相机比较二维 IoU，不把高同图 IoU 当跨相机合并依据；相机运动前后的历史图不走 IoU 路径。
- [ ] 实施同相机唯一、一对一占用、传递一致性、空间矛盾否决；not_separated 仅表示未排除，不算证明。
- [ ] 未确认视图不合入组；未解决对应若可能改变计数或目标，显式记录 unresolved correspondence。单视图组可存在，但不能默认为与其他未知组不同。
- [ ] 验证输入候选/相机顺序置换不改变可证明的分组；故意注入违反三维约束的模型选择必须被拒绝。

## Task 4: 通用任务条件和观测覆盖

**Files:** 新建 `task_grounding.py`；修改 `task_subjects.py`、`runtime/support.py`、`observation_geometry.py`、`agent/decision_state.py`、`agent/decisions.py`。

**Interfaces:** `resolve_task(capture, selections, groups, history_matches, judgments) -> GroundingResult`。只接受测得 group 和明确历史匹配，不写数据库、不移动机器人。

- [ ] 为 robot_left、left_of、ordinal、count/all、inside/on、历史加工状态与三值逻辑建立参数化验证；不以颜色/盒子名字写专用分支。
- [ ] 在固定世界坐标计算方向和相对轴顺序。排序采用完整有效点范围和可配置 0.02m 初始误差扩张；可见表面与物体完整中心区分记录。
- [ ] 组织关系依赖；容器和内容物分开分组，二维框重叠不能证明 inside。程序提供几何证据和矛盾，Decisions 判断剩余视觉关系。
- [ ] 为需要全体比较的选择器产生覆盖结果；未知遮挡、漏检或未归组候选可能改变答案时 unknown。SAM3 无结果不得证明不存在。
- [ ] 模型与数值不一致返回带原因的 unknown/failure；不能重问同题直到得到某个答案。
- [ ] 将提案编号、候选编号、同图重叠矩阵和原 RGB 标注作为定位证据；高 IoU 但违反机器人空间条件不得通过，低 IoU 但提案错误时允许由完整证据纠正，记录理由。
- [ ] 验证镜像画面、不同机器人侧别、任一路独有目标、多层关系、近邻排序区间重叠、遮挡改变数量，以及错误 Responses 框可被证据纠正。

## Task 5: 历史匹配与事务登记

**Files:** 修改 `temporal_identity.py`、`memory_reference.py`、`memory_contracts.py`、`object_database.py`、`agent/registration.py`、`runtime/physical_memory.py`。

**Interfaces:** `match_history(capture, groups, memory_reference, client) -> dict[str, HistoryMatch]`；`ObjectRegistrar.commit_grounding(capture, grounding, *, world_key, expected_revision) -> dict`。返回原有 identifiers/entities/reference 格式，供现有下游复用。

- [ ] 先验证错误初始 RGB 参考、搬运后旧位置、两件同物体遮挡交换以及事务中断不能得到不实 ID。
- [ ] 历史模块只返回候选身份结论；已存在 ID 未解决时不能退化为自动 new。保持带日期参考及现有最多 3 个更早参考策略。
- [ ] 对当前任务选中组及必要参照核验身份；比较所需但未选中的候选不必登记永久 ID。
- [ ] 删除感知路径内 register_instance/register_observed 自行提交的回调及其注册事务；all/count 新发现输入也必须返回 GroundingResult，经统一提交入口登记。
- [ ] 在同一事务保存冻结 capture、确认 masks、登记/更新历史与绑定依据；提交前复查 capture/generation/command revision 与数据库版本。失败回滚，重复成功调用幂等。
- [ ] 为新证据保存 registration_basis；保留旧记录及原始事件，旧 RGB-only 参考不自动取得 RGBD 验证状态。
- [ ] 验证成功后 ID 跨时刻保留、无关候选不增加 entities 行、并发版本变化/取消无半套登记、历史图使用历史标定。

## Task 6: Runtime 延迟绑定和显式恢复

**Files:** 修改 `runtime/controller.py`、`runtime/compiled_task.py`、`runtime/native_graph.py`、`runtime/observation_inputs.py`、`robot_skills/observation.py`、`skills.py`、`core/recovery_coverage.py`；新建 `skill_catalog/ground_task/contract.json` 和 `skill_catalog/ground_task/SKILL.md`；修改 `skill_catalog/observe_scene/contract.json`、`skill_catalog/select_target/contract.json`、`skill_catalog/execute_grasp/contract.json`、`skill_catalog/execute_place/contract.json` 及其说明。

**Interfaces:** 新注册 `ground_task` 无参数技能，使用已冻结 capture 和编译选择器，返回 GraphSkillResult；success 才发布 `task_grounded` 与永久绑定。已有 `observe_scene` 复用同一候选/分组/历史路径。

- [ ] 在 graph 验证中加入未满足 task_grounded 就调用依赖物体 ID 的物理技能的拒绝案例。
- [ ] 初始任务图保留符号引用；取消 compile 后立即 bind/register 的流程，ground_task 成功后调用新的绑定入口。禁止把临时引用传给抓取/放置。
- [ ] 对 success/unknown/failure 定义证据输出；所有需要的新采集和运动由图声明，ground_task 内不得隐藏执行。
- [ ] once 绑定在成功登记后冻结，后续观测只追踪原身份；each_selection 仅在图选择新输入时求值。验证已搬离左侧或序号变化的目标仍保持原 ID，抓放中不换目标。
- [ ] 提交期间使用 Runtime 同步机制检查命令/代次；提交后若事实变化，使当前绑定/几何失效并禁止执行，而非回退为旧图可用。
- [ ] 验证首次无 ID 的补观测流程、禁止隐藏腕部运动、相同输入缓存、取消和代次变化；新观测仍只使用最初一次 Responses 编译出的语义要求。

## Task 7: 匿名障碍与同页证据回放

**Files:** 修改 `semantic_builder.py`、`workspace_mapping.py`、`collision_mapping.py`、`planning_settings.py`、`runtime/diagnostics.py`、`web/src/replay-evidence.js`、`web/src/replay-labels.js`、`web/src/run-evidence.js`；不创建新 HTML。

**Interfaces:** 地图输入同时接收确认实例和未归属的有效深度；回放记录 `proposal -> candidate -> group -> entity`、空间约束、覆盖结论、历史比较和提交/拒绝原因。

- [ ] 先复现“任务只抓一件，但另一件可见障碍被 semantic_only 丢弃”的失败验收。
- [ ] 清理 demo_mapping 强制省略背景的分支。复用现有匿名 TSDF/ESDF 和 RobotSegmenter 边界；保留被过滤机器人表面的原始测量 provenance。
- [ ] 不把临时比较对象登记为永久实例；不把它们从避障场景删掉。缺少当前任务身份时仍可构建供恢复使用的匿名场景。
- [ ] 在同页回放区分候选框、最终目标框与永久 ID，显示三路缺失/歧义原因；失败尝试可回放但不能变为已确认历史身份。
- [ ] 回放显示 overlap 数值的比较对象和图像来源，区分 0、null、invalid；保存重复候选诊断及未合并原因，不把最高重叠显示成“已确认”。
- [ ] 验证未知障碍碰撞、单视图目标、多视图不同有效范围、遮挡仍未知，以及 UI 没有新开页面。

## Task 8: 端到端验收和旧路径清理

**Files:** 同步 `AGENTS.md`、`README.md`、`.agents/skills/luna-policy-isolation/SKILL.md`、`docs/raw-decisions-20261111.md`、`docs/history-timing-validation-20261107.md` 及当前技能合同。删除被替换的初始 RGB-only 登记专属函数和 prompt，不删除历史实验。

- [ ] 临时驱动采用 unittest，路径为 `%TEMP%/luna-grounding-validation/check_grounding.py`，覆盖设计验收矩阵；先用冻结输入验证新旧行为差异。
- [ ] 串行执行该驱动，预期所有正例成立、负例 unknown/failure 且数据库无错误增量；编译修改的 Python 文件，检查 diff 和无旧入口引用。
- [ ] 使用三项目独立启动的仿真，通过 Base 接口完成参数化位置/颜色/数量/视角/遮挡场景；不能让测试任务算法读物体真值。独立验收端记录误选、ID 切换、unknown 与成功率及每阶段延迟。
- [ ] 明确区分正常可判定场景成功和不可判定场景正确停止。此前两个搬运任务也要复测，但它们不是全部验收范围。
- [ ] 在 `data/acceptance/instance-grounding` 保存帧、框、掩膜、分组、约束、模型回答、独立否决、数据库提交和动作结果；不得只以容器健康作为通过。
- [ ] 按设计执行 A 当前程序 / B 新架构无 IoU 数值 / C 新架构加 IoU 的对照。B/C 共用固定提案、掩膜和深度，A/C 使用相同原始任务和观测序列，数据库独立，所有模型结果完整归档。
- [ ] 同时报告可判定关联成功率、错误提交率、正常 unknown 比例、ID 切换/重复登记、可见视图对应准确率和召回率及延迟。原始计数和配对差异支持结论；只提高拒绝率不得宣称成功率提升。
- [ ] 自审全部修改与现有工作区差异，验证无新镜像/无联合启动/无新页面/无 SDK 修改；仅部署本任务文件。没有完成端到端验收前不得宣布已解决全部误关联问题。

## 验证命令约定

Windows 本地验证运行明确路径，避免依赖当前失效目录；实现阶段应创建临时驱动后再执行：

```powershell
$env:PYTHONPATH='C:/project/tianji-luna-mujoco/src;C:/project/tianji-algorithms/services'
& C:/project/tianji-luna-mujoco/.venv/Scripts/python.exe "$env:TEMP/luna-grounding-validation/check_grounding.py"
```

预期：验收驱动退出码 0，所有必需案例通过；每个 unknown 案例同时断言没有错误永久 ID 和物理动作。实际仿真按现有三个仓库各自 `bash start.sh` 手工启动，保持当前镜像名与服务边界。具体运行配置、测量误差和验收结果随证据归档，不能用隐藏阈值代替未解决的身份歧义。

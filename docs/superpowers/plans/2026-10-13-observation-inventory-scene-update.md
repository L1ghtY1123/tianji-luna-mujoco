# 连续抓放的观测、物体清单与场景更新执行方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本方案建议在当前任务中顺序集成；用户已确认执行。当前实施记录见 data/observation-inventory-20261013/progress.md；未完成的验收与部署不视为已完成。

**Goal:** 在盒内到桌面标记、桌面单物体到盒内、桌面指定多个物体到盒内三类任务中，维护可理解且稳定的物体清单，减少正常抓放后的重复感知/建图，并让 JEV 依据明确证据选择观测、更新、重新建图和恢复。

**Architecture:** 沿用现有 JEV、Registry、Runner、TaskProgress 和算法适配器。清单、观测证据、碰撞场景分别维护版本；同帧跨相机关联与跨时刻 ID 关联分开；新增固定算法能力，不引入自动恢复状态机。正常转移优先更新目标层，其他对象/环境变化按用户规则要求回观测位姿重建。

**Tech Stack:** Python 3.12、Pydantic AI/JEV、NumPy/SciPy、SAM3、GraspGen、cuRobo、MuJoCo 3.10.0、现有 OpenAI 视觉适配器、原生 JavaScript/Three.js/Vite。

**Spec:** [累计讨论与最新约束](../../skill-specs/2026-10-13-observation/README.md)、[观测共用约定](../../skill-specs/2026-10-13-observation/common.md)、[JEV skill 规范](../../skill-standards/jev-skill-standard/SKILL.md)。讨论记录的“后续用户明确修订”与“后续补充”优先于更早建议；本文把它们收敛成拟实施行为。

## 实施状态（2026-10-13）

任务 1–6 已接入本地及 Ubuntu 隔离副本，包含稳定清单、两类关联、明确观测职责、原体素变换、JEV 选择和工作台展示。下方保留原始实施步骤；原先拟定的测试名称不代表实际测试名称。实际行为、指标和限制以 [实现说明](../../observation-inventory.md) 及逐次记录 `data/observation-inventory-20261013/progress.md` 为准。

- [x] 非 Git 基线备份、组件回归、实际 GPU 层变换探针。
- [x] 时序关联 50 段序列 / 550 帧采集，40 段留出集评分；补充逐相机独立标签评分。
- [x] 一次完整独立审查及其问题修复；前端 235 项测试与构建通过。
- [x] 单个指定物体到盒内成功两次；盒内到标记三布局成功；指定两个物体连续抓放成功；全部入盒修复重复检测后正常收尾。失败记录与修订差异保留，不混称最终版本成功率。
- [x] 完整回归、工作台清单与回放图浏览器核对。实际报告为 `data/observation-inventory-20261013/acceptance-report.md`。
- [ ] 全部矩阵在同一最终版本上的重复验收、配对性能目标。当前存在放置无可行路径、头部变化未知导致重建；不能宣称普遍成功或达到 30% 提速。
- [x] Ubuntu 生产部署、104 文件哈希核对、服务健康、原布局及规划参数恢复；回退包保存于项目 `data/observation-inventory-20261013/deployment-backup`。执行报告见同目录 `acceptance-report.md`、`deployment-result.json`。

OpenAI 图像服务在隔离验收中返回过 HTTP 502/503。此结果保留为未知，不构成视觉辅助已验证。当前不得宣称所有流程通过或已经部署。

## Global Constraints

- JEV 在 skill 外选择步骤、目标、观测目的、恢复、暂缓和终止；skill 内只执行稳定、有界的算法。程序可以拒绝不满足前提的动作，不自动改调其他 skill。
- 用户指定执行臂必须保留；其他相机的观测可供该臂使用。三个相机同时看见、双腕同时看见都不是通用就绪门槛。
- 全部/部分/数量/组合/排除/明确顺序沿用自然语言任务判断；禁止颜色专用流程和固定“先红后绿”。无明确顺序时向 JEV 提供执行臂当前 TCP 最近优先证据。
- 物体清单与碰撞场景分离。首次头部可见目标按实测距离由近到远编号，从 1 开始；完成后不重排编号、不清空历史、不重复计数。
- 其他物体、盒子或环境发生变化，必须回配置的观测位姿后重建场景。仅本轮搬运目标发生预期变化时优先目标层更新；未知不能当作没变。
- 环境信息只来自真实可见 RGBD；自身 FK、关节、夹爪反馈、标定可用。预测体素、实测表面和未知空间明确区别，仿真物体真值只允许用于独立验收评分。
- 盒内不要求精确摆放或码垛；保留盒壁、已有物体、机械臂、自碰撞和关节限位约束。
- OpenAI 只提供图像语义/变化/结果辅助，不输出控制动作或度量位姿。保留当前全任务最多 8 次视觉请求；同帧对同目的缓存复用，超时/错误不自动重试。
- 保留完整精简技能目录、任务原文、执行臂约束、未完成/暂缓/完成清单和关键未知项；不靠删除关键约束压缩上下文。
- cuRobo 固定 commit 为 `8e734f3ced1df898990bcd92de40abce475907db`；SDK、vendor、URDF 和网格保持只读。当前 313 个对称碰撞球保持不变，350 上限保留。
- 沿用现有回放删除、执行图、点云/体素/桌面 cuboid 展示，新增证据接入这些入口；不重新开发已完成能力。
- 本地 `C:/project/tianji-jev-mujoco`；部署目标 `/home/guangyu/projects/tianji-jev-mujoco`。当前目录无 Git，实施前使用文件清单、SHA-256 和压缩快照代替提交基线。生产任务运行时不写入挂载源码。

## Review Focus

1. 头部遮挡或分割波动造成假变化/假不变；未分割障碍、盒壁和已完成对象不能被忽略。归属任务 3。
2. 相距 5 cm 的相同工件、视角缺失或视野变化导致 ID 交换、合并、拆分。归属任务 2。
3. 释放后物体滚动/倾倒而预测层仍在释放位置；不得把预测或历史自由空间当成当前实测证据。归属任务 4。
4. Reset、迟到请求、换目标、换接触过滤和重建时，场景与清单版本混用导致重复完成或执行旧计划。归属任务 1、4、5。
5. 自然语言只选部分实例、数量约束、明确顺序与默认最近优先冲突；上下文压缩不得丢失这些条件。归属任务 5。

## 收敛后的数据与行为

### 三类状态分别保存

| 状态 | 内容 | 更新不会隐式触发什么 |
|---|---|---|
| 清单 | 内部稳定 ID、显示编号、特征、位置及来源、任务状态、可见性、身份关联状态 | 不隐式建图或移动 |
| 观测证据 | capture ID、动作 ID、标定/会话版本、各相机实际像素/深度、检测/关联/变化结果 | 不直接宣布任务完成 |
| 碰撞场景 | scene token、各图层、测量/预测来源、接触排除策略、当前深度证据、失效范围 | 不重置清单或重编号 |

沿用 `TaskProgress.records` 的内部 ID 和 `status`，补 `display_id`、`visibility`、`association_status`、`last_seen_observation_id`、`position_source`；`observed` 显示为“待抓取”，原有 `selected/completed/deferred` 保持兼容。不可见、关联未知与任务完成分开保存。`ledger_revision` 仅在清单内容改变时增长。

首次头部可见任务候选按其可见深度参考点到头部相机原点的距离编号；近似等距时按画面从左到右。此距离是可见表面参考，不宣称真实物体中心。编号仅分配一次；后来新发现目标追加未用编号。头部没有看到的区域不因此视为空，也不宣称已经枚举所有对象。初始编号次序不覆盖执行臂最近优先或用户指定顺序。

### 两种关联独立输出证据

- `cross_camera`：同一 capture 中不同相机的 SAM3 掩码，经注册深度、共同视域投影和深度冲突检查，得到当前物体实例。无需每个实例都出现在三个相机中。
- `temporal`：当前实例对历史清单做一对一匹配，延续 ID。类别/颜色、固定头部掩码、当前实测空间/表面以及本轮抓放角色构成证据；不能仅凭最近或唯一候选就接受。
- 每类均输出 `matched/ambiguous/unseen`、候选、像素/点支持量、比较指标、竞争候选差距和具体原因；匹配分数不展示成准确概率。
- 已移除旧 12 cm 搜索假定；采用头部 IoU 0.50/竞争差 0.15、缺头部时 5 mm/20% 可见表面支持、明确唯一外观的顺序规则。阈值不等于准确概率。
- 只有受影响的相同工件无法恢复对应关系时，由 JEV 明确选择重新枚举这些未完成记录；保留旧历史及未知对应，不清空完成数量，不把用户点名的历史实例换成其他实例。

### 每次抓放后的两条路径

动作结束并撤离后的同步 RGBD 可以复用执行 skill 已采集的帧。正常情况下先分析头部语义，其他相机的原始深度仍可供碰撞证据检查，不必每轮三路全量 SAM3。

| 证据 | JEV 应遵守的策略 |
|---|---|
| 只有本轮目标预期变化，其他相关区域有有效比较证据 | 选择目标层更新，清单标记完成，继续选择剩余目标 |
| 其他目标、已完成对象、盒子或环境变化 | 选择观测运动准备、回配置观测位姿、全量重建；清单 ID 尽量保留 |
| 算法不确定，相关图像可解释 | 在预算内选择一次前后图像辅助分析；同输入不重复付费 |
| 辅助后仍不确定，或相关区域不可见 | 旧图不据此获得有效性；JEV 选择补充观测/恢复，不能以 unknown 放行普通抓放 |

这是一组任务约束和技能前提，不是 Runner 中的自动跳转表。验证成功只更新完成证据；下一动作仍是 JEV 输出。

### 预测目标层的具体边界

抓取时保存目标层和 tool-relative 参考。释放时用实测 TCP 变换得到释放瞬间预测；落稳位置只可由已声明支撑/下落假设估计，不能将 TCP 当成真实落稳位姿。迁移源对象层，恢复其作为静态障碍的状态，保留其他有效层和桌面模型；不把对象周围的历史自由空间随层迁移。

**执行修订（用户最新确认优先）：** 盒内目标直接复用原体素并变换位姿，不添加局部重建、额外包络或新的未知空间运动门槛。当前帧只用于变化比较和已有验证；位置明确标为预测，不声称真实落稳。保持原始自由空间证据原位、版本校验及既有碰撞检查。其他环境变化或辅助仍未知而 JEV 选择重建时，先保存本次已搬运层，再做观测运动地图与完整重建。

SDK 若允许公开更新独立层 pose，使用该接口；否则以保留的测量数据在项目侧重新构造单层并发布完整场景集合，仍保留未变化层。两种方式均需 GPU 验证。不得修改 SDK、只改显示坐标或悄悄退化为每轮全量三相机语义建图。

其他环境变化后，回观测位姿前先使用原位当前深度形成该运动所需的碰撞证据；到位后再全量重建。这解决“旧图已失效却先用旧图返回”的问题。

## Skill 边界

| skill | 固定职责 | 不负责 |
|---|---|---|
| `observe_scene` | 按指定 capture（新采集/已有帧）分析任务范围，输出当前实例、清单关联和可见范围；仅在 JEV 显式选择身份策略时重枚举指定歧义集合 | 运动、建图、选择恢复 |
| `observe_geometry` | 分析指定物体/区域的当前分割与实测几何 | 全任务重新枚举、补全隐藏表面 |
| `observe_result` | 获取指定动作后的 RGBD/反馈；已有有效结果帧直接复用 | 宣布成功、决定恢复 |
| `check_scene_changes` | 比较指定前后帧，区分目标预期变化、其他变化、未知 | 自动移动或调用 OpenAI |
| `inspect_scene` | 有限 OpenAI 调用；明确目的为语义/身份辅助或前后变化解释 | 数值位姿、深度、控制命令 |
| `inspect_visual_result` | 辅助解释动作结果图像 | 取代夹爪反馈或保证完整碰撞几何 |
| `build_scene` | 从指定有效 RGBD 建立场景；观测运动模式允许尚无目标 | 移动、刷新任务清单 |
| `update_scene` | 对 JEV 指定且有版本绑定的目标层执行迁移/更新，并发布新 scene token | 自动决定需要全量重建 |
| `move_to_observe` | 消费有效运动碰撞证据，到指定配置观测位姿，并返回新 capture | 自行决定为何移动、隐式建图、声称已找到最佳视角 |

`identify_target/assess_task/recover_task` 继续承担 JEV 的目标选择、任务判断和恢复输出；沿用抓姿、规划、执行及验证技能。两类关联是观测中的固定算法模块，不另外增加两个 JEV 调用步骤。新增运行时 skill 仅在 contract、文档、参数选择、执行分支、结果摘要和测试同时具备时注册；`observe` 完成迁移后退出实时目录，回放仍识别旧名称。

`scene_ready` 若保留，仅为“当前选中目标/接触策略/动作所需证据”的派生兼容状态；删除按相机数量以及集合所有对象决定它的逻辑。契约不合法返回 `skill_contract_not_satisfied`，交回 JEV，不自动换 skill。

## 文件职责与实施任务

新增算法文件限于 `association.py`（两类匹配）、`scene_changes.py`（变化检测）、`scene_update.py`（目标层更新请求与证据）。其余沿用既有模块，不做全项目重构。下述函数名为拟新增接口，参数使用现有字典/数组契约；每个结果都必须含 capture/generation 或 scene token 绑定。

执行顺序：完成 Task 1 的基线保存后，提前执行 Task 4 的 SDK 能力与独立探针步骤，尽早确认体素更新可行；随后按 Task 1–7 顺序集成。探针结论不等于正式场景更新功能已完成。

### Task 1：稳定清单、显示编号与基线

**Files:** 修改 `src/luna_mujoco/task_progress.py`、`tests/test_task_progress.py`；记录 `data/observation-inventory-20261013/baseline-manifest.json` 和同目录 `baseline.zip`。
**Interfaces:** 新增 `TaskProgress.assign_display_ids(observation: dict) -> dict[str, int]`、`TaskProgress.rank_candidates(observation: dict, *, arm: str) -> list[dict]`；`snapshot()` 补充上述清单字段。距离排序只提供证据，不改变当前目标。

- [ ] 记录本地文件基线、远端版本/空闲状态和现有参数。读取文档与源码差异，以实际部署状态为准；不保存或输出凭据。
- [ ] 先写 `test_head_distance_numbering_is_stable_after_completion`：初始排序 ID 为 `[1,2,3]`，完成 1 后其余仍为 `[2,3]`；`test_unseen_completed_object_stays_completed`；`test_new_detection_does_not_reuse_completed_number`；`test_tcp_ranking_does_not_change_display_ids`。
- [ ] 运行 `.venv/Scripts/python.exe -m pytest tests/test_task_progress.py -q`，确认新增行为缺失导致失败；实现上述方法和字段，保留旧接口兼容，再运行至全部通过。
- [ ] 验证同一完成事件重复到达不重复计数；保存本任务变更文件哈希及结果，不调用 JEV 或机器人。

### Task 2：拆分两类关联并建立准确度数据集

**Files:** 新增 `src/luna_mujoco/association.py`、`tests/test_instance_association.py`、`tests/association_replay.py`；修改 `perception.py`、`runner.py`、`tests/test_perception.py`、`tests/test_task_perception.py`。
**Interfaces:** `associate_camera_detections(detections: list[dict], views: dict) -> dict` 返回 `groups/evidence/unresolved`；`associate_inventory(previous: dict, current: dict, *, action: dict | None, parameters: dict) -> dict` 返回 `mapping/evidence/unresolved`；`Perception.associate` 保留兼容入口直到调用方完成迁移。

- [ ] 先写测试：`test_two_parts_five_cm_apart_keep_two_ids`、`test_missing_wrist_does_not_create_new_identity`、`test_single_candidate_without_support_is_not_accepted`、`test_two_identical_parts_swapped_unobservably_are_ambiguous`、`test_placed_object_and_table_object_do_not_merge`。断言唯一对应/保持两个 ID/无法判定时不强配；保留原 FOV 与前景深度冲突回归。
- [ ] 运行 `.venv/Scripts/python.exe -m pytest tests/test_instance_association.py tests/test_perception.py tests/test_task_perception.py -q`，确认新用例先失败。
- [x] 将现有跨相机几何检查封装到第一接口；第二接口采用固定头部掩码、无可比头部时的可见表面、全局唯一明确外观的顺序规则，并要求相互唯一对应。实现不使用加权代价优化。固定算法不询问 JEV；参数在留出评估前冻结，匹配指标不映射成概率。
- [ ] 准备不同物体、相同工件、遮挡、移动、释放后视野改变的数据；至少 50 段独立序列、500 次跨时刻对应机会，另统计跨相机对应机会。由离线标注标明“视觉可判定/不可判定”，仿真真值只给评分器。校准/留出按序列分离，不按相邻帧随机拆分。
- [ ] 重跑针对测试；新增回放命令 `.venv/Scripts/python.exe tests/association_replay.py --manifest data/observation-inventory-20261013/association/manifest.json --split test --output data/observation-inventory-20261013/association/report.json`。报告两类准确率、覆盖率、未知率、ID 切换、误合并/拆分、样本数及按序列统计的不确定区间。目标：留出集错误延续 ID 为 0，可判定样例正确关联覆盖率至少 95%；这是验收目标，不是现有能力或总体正确率保证。

### Task 3：明确观测目的与固定头部变化检查

**Files:** 新增 `src/luna_mujoco/scene_changes.py`、`tests/test_scene_changes.py`；修改 `runner.py`、`perception.py`、`observation_motion.py`、`visual_assistance.py`、`visual_scene.py`、`tests/test_observation_motion.py`、`tests/test_visual_scene_assistance.py`。
**Interfaces:** `compare_scene_observations(before: dict, after: dict, *, transferred_id: str, parameters: dict) -> dict` 返回 `status=expected_only|unexpected|unknown`、`changed_ids/regions/evidence/before_capture_id/after_capture_id`。扩展 `VisualAssistance.inspect_scene(..., reference=None, call_budget=None, purpose="semantics")`，新增 `purpose="scene_change"` 并保留原调用兼容。

- [ ] 先写 `test_only_transferred_object_is_expected_change`、`test_bin_wall_motion_requires_reobservation`、`test_existing_bin_object_displaced_is_unexpected`、`test_robot_occlusion_is_unknown_not_motion`、`test_unsegmented_depth_obstacle_is_detected`、`test_changed_calibration_rejects_mask_comparison`、`test_duplicate_frame_pair_reuses_visual_result`。
- [ ] 运行 `.venv/Scripts/python.exe -m pytest tests/test_scene_changes.py tests/test_observation_motion.py tests/test_visual_scene_assistance.py -q` 确认新行为失败。用同标定头部掩码 IoU、轮廓位移和有效深度差比较；分离盒壁与搬运目标引起的内容变化，不忽略整个盒子。容差由静止数据/深度误差及当前规划尺度校准，不引入“固定几个像素就一定没变”的结论。
- [ ] 实现已有帧分析、新采集、移动采集的独立计数和缓存键；默认复用 post_place capture。初次头部语义发现，其他原始同步深度保留；逐目标缺什么才分析什么。移除双腕/三路计数门槛和“所有源物体都 ready 才能抓当前物体”。
- [ ] 前后图像辅助一次覆盖全部可疑区域，显式标注时间、候选 ID 与问题；不要求模型输出米制坐标。预算耗尽、超时、图像不足返回 unknown；同帧同目的不自动重试。针对测试通过后保存三种状态的可视化证据。

### Task 4：目标层更新与碰撞证据版本

**Files:** 新增 `src/luna_mujoco/scene_update.py`、`tests/test_scene_update.py`、`tests/scene_update_gpu_smoke.py`；修改 `algorithms.py`、`geometry_service.py`、`geometry.py`、`perception_mapping.py`、`collision_mapping.py`、`placement.py`、`runner.py` 及 `tests/test_geometry_scene_reuse.py`。
**Interfaces:** `make_scene_update(scene: dict, holding: dict, capture: dict, changes: dict) -> dict` 生成目标 ID、旧 token、新 capture、预测变换和来源证据；`Algorithms.update_scene(request: dict) -> dict` 经既有 GPU 服务通道返回新 `scene_token/layer_evidence/collision_evidence`。清单不由此函数修改。

- [ ] 先在锁定 SDK 的公开源码/示例中核对层变换与场景发布接口，记录使用的公开入口及版本；运行独立最小 GPU 探针，检查目标移动前后旧位置/新位置的碰撞查询改变、其他层不变。若没有原位 pose 更新入口，验证项目侧单层重构再发布路径；不修改 SDK。无法提供实际碰撞更新时明确报告该能力未完成，不只移动前端显示。
- [ ] 写失败测试：`test_scene_update_moves_only_selected_layer`、`test_scene_update_does_not_move_free_space_evidence`、`test_current_depth_blocks_prediction_that_misses_obstacle`、`test_hidden_landing_stays_predicted`、`test_scene_update_preserves_other_layers`、`test_stale_scene_update_after_reset_is_rejected`、`test_next_target_exclusion_is_recomputed`。
- [ ] 运行 `.venv/Scripts/python.exe -m pytest tests/test_scene_update.py tests/test_geometry_scene_reuse.py -q`，确认新行为失败；实现源目标层撤销/变换/恢复、当前深度证据绑定及原子发布。只有成功发布才替换 scene token；旧计划、接触排除和 attachment 证据同时失效或重新绑定。移除旧占据不表示已经证明原区域空闲。
- [ ] 释放后预测不要求完整局部重建；可见时修正，遮挡时明确推导属性。全量场景比较与预测层不一致，必须保留差异证据；不能用松弛碰撞参数掩盖差异。返回观测位姿的地图允许无目标，使用当前原位深度而非已知过期图。
- [ ] 针对测试通过后，在 Ubuntu 隔离环境运行 `python tests/scene_update_gpu_smoke.py`。保存层哈希、位姿、场景 token、实际 cuRobo 查询与相机深度证据。相同已测区域的碰撞判断与全量建图对照；未知区不得伪装成等价的空闲区。通过后才接入正常快路径。

### Task 5：JEV、契约与连续任务闭环接入

**Files:** 修改 `runner.py`、`decisions.py`、`skills.py`、`task_progress.py`；新建 `skills/{observe_scene,observe_geometry,observe_result,check_scene_changes,update_scene}/SKILL.md` 与对应 `contract.json`；修改 `skills/{observe,move_to_observe,build_scene,inspect_scene,identify_target,assess_task,recover_task}` 的文档和契约；测试 `test_collection_runner.py`、`test_transfer_decisions.py`、`test_jev_perception_policy.py`、`test_recovery_scene_budget.py`，新增 `test_observation_contracts.py`。
**Interfaces:** 由 Runner 执行以上算法结果并更新版本；JEV 输入增加精简 `inventory/change_evidence/scene_validity/association_evidence`。新 `update_scene` 参数由 JEV 选择目标和已存在的 release/capture 引用，不能自由生成数值位姿。`observe_scene` 增加 `identity_policy=preserve|reenumerate_ambiguous` 和 `affected_ids`；默认 preserve，只有 JEV 显式选择重枚举且指定对象确有无法恢复的关联证据时才执行，保留旧历史。`identify_target` 仍只负责绑定 JEV 选定的目标，不混入身份重建。

- [ ] 先写测试覆盖三个基本任务、自然语言子集/数量/排除/明确顺序、默认 TCP 最近、指定臂不重新选择、另一腕提供几何、目标已完成不再抓；同一失败证据下不同 JEV 恢复输出产生相应不同动作，程序无自动重试。
- [ ] 写 `test_unexpected_change_requires_observation_pose_before_full_rebuild`、`test_observation_motion_uses_current_depth_scene`、`test_invalid_skill_returns_evidence_without_substitution`、`test_context_compaction_preserves_ids_constraints_and_unknowns`、`test_reenumeration_does_not_substitute_explicit_historical_targets`，确认现状失败。
- [ ] 在同一集成变更中接入新 skill 的 contract、参数选择、执行分支、摘要及状态失效。逐项写明固定步骤、未知/失败和预算；观测运动消费地图、返回 capture，不隐式建图。初始目标未发现时仍允许观测运动地图，但缺少安全运动证据不能凭空放行。
- [ ] 正常完成一物体后清空旧轨迹和该目标的派生规划，保留清单、有效 capture、地图层及预算；不再无条件清除所有环境层。确认其他变化后使旧普通抓放前提失效，JEV 必须选择观测运动流程。重新建立清单只影响歧义的未完成集合；任务完成仍依据原指令和已验证记录。
- [ ] 运行 `.venv/Scripts/python.exe -m pytest tests/test_collection_runner.py tests/test_transfer_decisions.py tests/test_jev_perception_policy.py tests/test_recovery_scene_budget.py tests/test_observation_contracts.py -q`。所有运行时技能均有实现、契约和输出；记录新的精简目录与输入大小，不用删除技能解决上下文限制。

### Task 6：工作台和回放展示

**Files:** 新增 `web/src/object-list.js`、`web/tests/object-list.test.js`；修改 `web/src/main.js`、`camera-panel.js`、`pointcloud-panel.js`、`replay.js`、`execution-graph.js`、`styles.css`、`web/index.html`、`src/luna_mujoco/app.py`、`experiments.py`。
**Interfaces:** `createObjectList(container)` 返回 `render(taskProgress)`；相机 API 的当前 task capture 增加帧绑定的 `object_labels`（display_id、边界/锚点、状态）；展示只能使用同一帧标签。预测/测量图层由碰撞预览证据提供，禁止从编辑器真值生成任务标签。

- [ ] 写前端测试：编号完成后不跳号、任务状态与可见性分开、旧帧标签不叠到新实时画面、清单更新不触发建图请求、历史记录没有新字段仍可回放。
- [ ] 运行 `pnpm --dir web test` 确认新行为失败；增加清单、头部编号叠加、关联证据摘要、预测层标记和全量/局部更新耗时。实时预览没有对应观测时隐藏过期标签；不将推导位置画成当前检测掩码。
- [ ] 接入现有执行图节点和回放详情，保留删除及原始日志；所有可见文字至少 20 px，沿用工作台布局。
- [ ] 运行 `pnpm --dir web test`、`pnpm --dir web build`；使用浏览器核对正常、部分完成、未知、回放四种状态，保存截图。打开清单、点云或回放不引发新观测/建图。

### Task 7：端到端验收、性能与部署

**Files:** 复用并扩展 `tests/continuous_transfer_smoke.py`；产物 `data/observation-inventory-20261013/{acceptance,performance,deployment}/`；同步更新 `README.md`、观测定义和项目规范中本次实际实现的描述，安装的规范副本保持一致。
**Interfaces:** 验收程序以自然语言指令调用真实 JEV/SAM3/GraspGen/cuRobo/MuJoCo；独立评分器读取物理真值仅生成结果，不向控制链传递信息。

- [ ] 执行本地完整 Python 测试及 lint：`.venv/Scripts/python.exe -m pytest -q`、`.venv/Scripts/python.exe -m ruff check src tests`；执行前端 test/build。根据变更做独立代码审查，修复发现的问题后再进入远端闭环。
- [ ] 在独立仿真与 GPU 服务实例完成下方矩阵；常规三类任务各至少 3 个布局，其他关键情形至少各 1 个。保留失败样本，不用只挑成功记录的方式出报告。
- [ ] 分别统计 RGBD、SAM3、关联、OpenAI、全量建图、目标层更新、IK、规划、执行和 JEV 时间与调用数。相同布局/指令做基线与新版配对测试；单列首次与后续物体、冷启动与热启动、正常复用与环境变化路径。工程目标为正常复用路径“感知+场景准备”中位耗时比基线降低至少 30%，不承诺整轮固定秒数；未达到继续定位瓶颈，不以关碰撞或少测关键证据换速度。
- [ ] 发布准确率与性能报告，列明样本数、失败/未知和未验证项。无 privileged 信息泄漏测试通过；旧 `instance association ambiguous` 记录可回放且无同帧无信息重复请求。
- [ ] 在实际开始实施和验收后，按既有授权范围准备部署：先确认任务空闲、保存布局/参数/历史与待改文件备份，哈希检查避免覆盖用户新改动，分批同步本次变更到 Ubuntu，仅重载本用户相关服务。用户现已授权执行；部署须以实际验证和空闲状态为依据。
- [ ] 部署后核对三类工作流入口、清单/标签/地图版本、回放、服务健康和本地远端哈希；保留可恢复基线。只清理确认属于本轮且不再使用的临时资源。

## 端到端验收矩阵

| 场景 | 必须观察到的结果 |
|---|---|
| 盒内物体到桌面标记 | 指定对象、指定臂、实测标记；完成记录和障碍更新一致 |
| 桌面单物体到盒内 | 正常验证成功，预测/测量属性明确，无无意义重新观测 |
| 红绿等指定多个到盒内 | 子集正确、未选对象不动、默认排序可解释、ID 稳定 |
| 桌面所有目标到盒内 | 不遗漏当前任务要求，不把暂时遮挡当作全部完成 |
| 相同工件相距 5 cm | 分割、跨相机和跨时刻分别评分；不交换 ID，不误合并 |
| 左臂放完后只右腕能看见下一目标 | 不因缺左腕掩码强制回姿态；相关证据足够可继续 |
| 盒内已放目标仅一腕可见或完全遮挡 | 可复用预测层但保留不确定；不得伪造完整形状/最终位姿或空闲空间 |
| 盒子或其他物体被推移 | JEV 选择回观测位姿并重建；返回运动不使用已知失效图 |
| 相同工件不可观察地互换 | 不能保证物理序号；可见证据有冲突时保持未知，JEV 可重枚举受影响集合，历史和完成量不丢失 |
| 抓取失败或放置验证未知 | JEV 选择恢复/暂缓/结束，无固定自动恢复；未完成项最终可见 |
| Reset/服务超时/迟到结果 | 旧 ID/图/轨迹不能污染新会话；任务可恢复到一致状态 |

## 交付与自审

- [ ] 交付实现文件、技能契约、清单和关联报告、GPU 场景更新证据、真实闭环报告、分阶段耗时、部署/回退清单。
- [x] 方案已覆盖三个任务、清单编号、两类关联、头部变化检查、OpenAI 边界、目标层预测、其他变化必重观测、JEV 恢复、上下文和既有 UI 回归。
- [x] 方案区分现状、拟新增接口与验收目标；未将 SDK 单层变换能力、预测最终位姿或准确率描述为已经验证。
- [x] 已将五项 Review Focus 分配到具体测试；没有为每种颜色、物体数量或固定顺序增加专用 skill。
- [x] 用户已确认实施；隔离仿真实验和视觉 API 已调用，生产源码已更新并核对健康及哈希。未满足的性能/重复验收目标在实施状态中单独列出。

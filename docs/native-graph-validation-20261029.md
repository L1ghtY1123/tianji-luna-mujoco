# 显式 graph 与回放验证记录

## 本次追加：图例、执行顺序、router 恢复

本次没有新增物理抓放实验。读取既有归档、验证界面，不把归档成功算作新实验成功。

- 图例独立解释来源、node 结果、edge 条件与线型。默认 Responses 编排保持清晰；恢复扩展明确折叠，实际访问逐次展示，不合并重试、不把可能 edge 当执行记录。
- `a150686f6a4f47a48bba9c158b970c34`、`41eb6015ac6a4f378994f6e89fb1dded`、`949fe60a2481470782aa2ae4820ba991` 的浏览器验证通过：实际调用分别 24/63/40 次，与事件一致；node 数量满足初始显示 + 运行时折叠 = 完整归档；文字无越界、页面无异常。密集恢复实验不再产生数十条跨图回环。证据 `data/graph-clarity-browser.json`。
- 最新成功归档的 SVG/PNG/完整 JSON 导出及真实 JPG、MP4 预览通过：`data/replay-redesign-browser-result.json`。SVG/PNG 是明确标注的折叠视图，JSON 保留全部 node/edge。
- 复现 controller 对 router unknown 提前抛异常后修复，使已有 unknown edge 可执行。另复现未声明 unknown/no edge 的 router 不能进入 Decisions 恢复，现允许模型选择能力、归档扩展、重新判断；保留原始 no/unknown 结果，不转成 success。
- 第一次 adapter 修复后全后端测试 3511 passed / 58 skipped；随后新增 router 恢复变更的相关测试最终 120 passed，包括不允许重播不可重复物理动作。完整前端 380 passed，Vite build 成功，仍有既有 bundle >500kB 提示；修改 Python 文件 Ruff 通过。
- 运行服务空闲、视频编码结束后同步代码；87 个 manifest 文件 SHA256 一致。未更改碰撞阈值、抓姿、采样预算或物理执行限值。

架构与稳定性剩余问题见 `gap-source-audit-20261029.md` 的追加核查：大 controller/Runner 继承耦合、共享 context、历史配置别名仍在；这些没有因本次 UI 改动自动解决。轨迹跟踪失败案例及过往感知/恢复失败仍是有效实验记录，不能删除后声称高成功率。

日期：2026-10-29。源码对照和整体结构见 [GaP 源码对照](gap-source-audit-20261029.md)。本文区分代码检查、浏览器检查和真实 MuJoCo 实验；三者不能互相替代。

## 代码与界面

- 后端最近完整套件：3508 passed、58 skipped；日志 `data/native-final-verified-tests.txt`。随后最终位姿证明规则定向检查 159 passed（`data/native-final-pose-proof-tests.txt`），覆盖实际位姿不正确仍失败及不强制插入回位动作。
- 最新恢复参数校验、建图来源、感知提示、初始上下文与 controller 定向回归：75 passed；`data/native-latest-targeted-tests.txt`。容器分割回退另有 86 项相关检查通过，`data/native-container-fallback-tests.txt`。
- 前端完整套件：376 passed；`data/native-replay-final-tests.txt`。
- Vite 构建通过；仍有原有大于 500 kB 的 bundle 提示，`data/native-replay-final-build.txt`。
- 当前 core、controller、native executor、恢复与感知文件 Ruff 检查通过。
- 部署校验清单见 `data/native-graph-deploy-files.json`；最终逐文件 SHA-256 比对结果见 `data/native-deployment-verification.json`。

真实 Edge 浏览器读取实验 `a150686f6a4f47a48bba9c158b970c34`，验证 root graph 与两个物体 subgraph、重复调用图已移除、node 文字边界没有溢出、SVG/PNG/JSON 导出、归档 JPG 与 MP4 预览、无页面异常。脚本和结果：`data/replay-redesign-browser.cjs`、`data/replay-redesign-browser-result.json`。

graph 默认按原始字号横向滚动，可缩放、完整查看和全宽打开。完整查看会缩小长 subgraph，阅读长图应使用 100% 与横向滚动；不是声称任意规模 graph 都能在一屏保持可读。非相邻 edge 使用曲线和不同端口；复杂恢复 graph 仍可能存在交叉，不能声称解决了任意图无交叉布局。

## 真实机器人实验

以下使用真实 Responses / Decisions / SAM3 / GraspGen / cuRobo 与 MuJoCo 服务。每次任务只调用一次初始 Responses，执行中无额外 Responses、无 sub-agent。保留用户调整的场景布局和现行碰撞阈值。

| 实验 ID | 验证到的行为 | 失败或限制 | 后续改动 |
|---|---|---|---|
| `b469762935fd4777bac50be7e0d295c5` | 粉色实际抓取、放置与核验成功 | 切换绿色时局部身份不确定使全部观测失败 | 冻结清单后允许部分观测；缺失 ID 保留历史、明确标为未测得 |
| `257ec5edbde34c3c93952434fd57f23b` | 粉色搬运成功；出现 Decisions 恢复选择 | 不断选择无进展的同类采集 | 排除当前证据下已拒绝的精确参数组合，不替模型选择下一步 |
| `a150686f6a4f47a48bba9c158b970c34` | 粉色搬运成功，并显式执行双臂回观测位姿与观测；进入绿色 subgraph | 恢复选择 build_scene 后出现局部 `decision` 未赋值异常 | 修正作者来源与实际局部模型调用的区别，增加回归 |
| `61d036d52f964bcfa66b06ad0ed3922a` | 归档清楚地将绿色目的地设为 `table_main` | 初始 graph 未建立最终 `at_initial_pose`，契约拒绝 | 明确“每物体观测位姿”和“全任务初始位姿”是两个完成条件；未硬插动作 |
| `b54234f402cc4c73a92d8e1efc6cbca2` | 粉色、绿色都已得到当前掩码和测量 | 中文 SAM3 提示未检测到源盒，碰撞层缺失；未进入实际抓取 | 提示契约要求英文视觉短语；容器增加有界颜色同义词回退，仍须 Decisions 关联 |
| `e95004229f5a47878af3e26d25d34344` | Responses 正确指出输入图没有绿色物体 | 测试恢复场景与热重载竞争，实际退回单粉色默认场景；此轮不能用于评价双物体视觉成功率 | 测试启动器核对服务实例、实际对象集合；先完成重载再恢复场景 |
| `41eb6015ac6a4f378994f6e89fb1dded` | 粉色完成；绿色暂未形成当前身份绑定，Decisions 实际追加了 22 个 graph revision | 737.07 秒后仍在绿色绑定阶段停止；一次 Responses、71 次 Decisions。绿色 SAM3 在头部/左腕最高 0.953，深度有效；身份分类置信度仅 0.51/0.59 | 部分观测返回绕过了既有身份澄清；现已接入同一冻结 RGB 的一次成对澄清，非重复采集或降低阈值 |

双物体重测 `e46161b43e624677aae706c0020fa329` **完整成功**，耗时 332.77 秒：一次初始 Responses、20 次 Decisions。粉色与绿色分别完成抓取、抬升、放置与放置核验；每个物体完成后双臂回观测位姿并重新观测；最后回任务开始位姿并完成全任务核验。初始 graph 包含 root 与两个独立物体 subgraph，此轮没有运行时 graph patch。

最终实际头部 RGB 归档为 `decisions/call-bd7c2dcafcf34bca95c665f751e4eebb-image-007.jpg`，本地核对副本 `data/native-run14-final-7.jpg`：两物体均在桌面，源盒为空。此图是最终 Decisions 请求的 **CURRENT RGB camera 3**，不是历史参考图、另画的展示图或重演。`verify_grasp` 本身使用夹爪位置反馈；抓取成功不能仅靠这一项推断，此轮同时核对了后续实际搬运、落定图像和完整放置核验。

一次成功不能证明稳定成功率。继续在相同用户布局、相同代码和阈值下重复实验；新实验结果另行追加，不以成功轮覆盖前面的失败记录。

失败轮的冻结证据对照：`data/native-run15-clarification-result.json`。一次成对澄清对左腕绿色候选返回 green=0.99、container=0；头部模糊候选仍为 unknown。保持肯定阈值 0.8、竞争身份否定阈值 0.2。已加入唯一匹配、持续不确定、多个身份同时肯定的回归，相关 54 项通过（`data/native-clarification-tests.txt`）。澄清不移动机器人、不重拍、不改已有任务；仍未知则返回部分观测，由 graph 决定恢复。API 不返回内部推理，因此这个对照只能证明问题表述影响该冻结样本，不能声称已解释全部低置信度或所有新场景都将成功。

部署这一修正时旧 websocket 阻塞了开发服务的优雅热重载。确认上一实验结束后，仅重启本项目 web 容器，再核对用户场景后开始下一轮；此维护过程不计作机器人实验成功，也没有修改其他项目服务或 CPU 配额。

实验 `3acc37e84aad4d08b0868f6b69920ff3` 在初始身份确认 unknown 后，Decisions 选择的新观测 node 三次成功，但一直选择新观测，最终 stop，没有接回原 graph。恢复续接现在单独交给 Decisions 判断，原 node 仍需重新执行并通过检查。新增选择继续、停止和 unknown 的回归；连同身份与 SAM3 回退相关检查共 70 项通过（`data/native-continuation-tests.txt`）。本轮完整后端检查最初为 3489 passed、2 failed、58 skipped；两个失败是旧测试仍假定空容器不使用同义词回退，已更新为有界回退契约并在这 70 项中复验，未改动数值安全条件。

成功的 skill 可以直接返回观测数据而没有 `success` 字段，执行器仍会归一化为 success。恢复历史现在显式携带这个执行器判定的 status，避免续接判断把成功的数据结果误当作没有成功；原始结果不变。冻结的失败上下文对照中，聚焦续接问题选择了 `retry_failed_node`（选项概率 0.60，choice confidence 0.47）。这是操作提议，不是已经成功或安全的证据；随后仍须执行并核验原 node。

实验 `f31d66033aaf4d0dbf48210fdc3b5b9b` 的初始 Responses 生成了两个正确物体 subgraph，并额外重复了这两个 subgraph ID；重复条目带非空 input_id、却带 null task，被 `subgraph_task_owner_required` 拒绝。这不是两个有效共享 subgraph。输出 JSON schema 已将归属两字段联动约束为“input_id 非空且 task 非空”或“两者都 null”；不限制 subgraph 数量、技能顺序或 edge 选择。85 项 graph/schema/controller 定向检查通过（`data/native-final-schema-tests.txt`）。

实验 `64ccca6acae84a009beaefcdc7c9e16e` 中，粉色完成，绿色再次实际抓起，但其桌面放置失败。420 个允许 XY 中只尝试了肩部距离最短的 12 个，均集中在桌边，12 次均未找到 approach 路径；随后恢复判断被 `fresh_observation_required_before_judgment` 阻断。修正为保留首个可达排序点，再按最远点覆盖选择至多 12 个分散候选；不扩大规划预算、不放宽碰撞约束。恢复判断在动作后采集并归档当前 RGBD 作为模型输入，不执行 observe_scene、不关联 ID、不发布地图、不设置 observed=true。普通判断的旧帧拒绝仍保留。49 项定向检查通过（`data/native-placement-recovery-tests.txt`）。

## 本轮修正的边界

生产执行的是 `core/graph.py` 的 graph/subgraph/node/edge，`runtime/native_graph.py` 沿 edge 调度；与旧 steps 遍历器分开。运行时 Decisions 可产生新的恢复 node/edge，先校验、归档再执行。参数选择同时受注册 schema、交叉参数校验、当前事实与已拒绝组合约束。

每物体回观测要求来自用户，写入 `completion_requirements.after_each_object`，由初始 Responses 明确编排。runtime 只验证，skill 不暗中补动作。不可见不等于不存在，历史测量也不能冒充当前可用于运动的几何。这里没有通过降低视觉门槛、忽略未知碰撞层或关闭机器人碰撞来使实验通过。

实验 `8140ed0157154950a89a45c5618bdbd0` 已包含 `verify_task_completion`，却因其后的 router 判断句未逐字出现在 completion_conditions 中，被 `undeclared_policy_judgment` 拒绝。Native router 现在可以声明自己的证据问题；判断只选择 edge，不产生任何 skill 效果或完成事实。根 graph 成功路径仍须建立 task_verified/at_initial_pose。运行时仍只接受 graph 中已声明的判断。92 项相关检查通过（`data/native-router-validation-tests.txt`）。旧 steps 库的判断合同不变。

未通过初始校验的 graph 也从错误归档的完整 rejected_proposal 展示、导出，标注“校验未通过，未执行”；不会把模型提议当成实际执行。无效 subgraph 归属单列，不冒充有效物体或共享 subgraph。

实验 `964754d19a274029855e3c0ff24dcb95` 两物体均真实抓取、放置成功；各自回双臂观测位姿并观测，304.67 秒完成两物体 subgraph。随后最终完成条件“每次回位并观测”判断为 0.64，恢复后为 0.17，未通过。原因是运行器只检查出口事实，却没有向 Decisions 提供各物体的历史回位观测凭据。持续恢复不能修复这个输入缺口，Codex 在 546.70 秒主动取消该实验，不记为全任务成功。

新增每个输入 ID 的 post_delivery_observation 凭据：交付已核验、实测双臂观测姿态、已完成身份观测的 observation_id、generation、action_epoch。拒绝旧帧、只有原始采集、错误姿态或未交付的情况。凭据进入清单和最终 Decisions 输入；仍由 Decisions 核验所有条件，不覆盖其判断、不降低 0.8 门槛。

该轮 Responses 为绿色选择了配置区域 demo-destination。当前该区域与 vla-unpack-input 的数值范围相同，均由现有界面参数形成；不是该轮在整张桌面上自由选点。其他实验也有选择测得桌面实体的 graph。Responses 决定目的地引用，skill 在该引用允许的范围内求具体安全落点；不要将其表述为每次都由 Responses 输出最终 XY。

实验 `ad09ee64985d4fe1b0e92e8b906ad8b1` 两物体实际搬运、回位观测与永久 ID 历史凭据均成功；292.61 秒结束两物体 subgraph。最终回位 node 被 scene_refresh_required 拒绝，尽管机器人实际已在初始位姿；共享 wrapper 在 no-op 判定前检查地图，且原能力未声明 retry-safe，导致 Decisions 建图后仍不能续接。现在仅对实际运动要求有效地图，纯反馈 no-op 仍检查代际、空手和释放核验，不伪造地图刷新。回位能力声明可重新规划的幂等重试；未处理的物理副作用仍由 runtime 拒绝恢复。此轮最终 needs_attention，不计为全任务成功。

实验 `1a14764623394814a81c1b31e0d85514` 在绿色 approach 执行时触发 trajectory_tracking_error：A 臂 joint 5 实测误差 0.030088 rad 超过 0.030 rad。未闭爪、未将本轮绿色记作抓取成功，保留部分执行保护；未提高阈值。此前完整代码检查中新增共享检查函数使 6 个直接调用基础 Runner 的回归失败，已将仅依赖 context 的检查改成独立函数，原有地图、空手和覆盖检查继续执行。

实验 `2cf379aec0644fe5b8172e205eaa7fcf` 的 graph 已有最终 verify_task_completion，却缺单独 return node，被静态 at_initial_pose 事实规则拒绝。此处限制过强：最终 verifier 已独立读取初始关节参考和实际反馈。现要求成功路径必须建立 task_verified；有此最终 verifier 时，是否另需回位动作由 Responses 决定，实际位姿不正确仍在运行时失败并交给 Decisions。不含最终 verifier 的兼容目录仍须建立显式最终位姿事实。159 项定向检查通过；没有给图插入回位 node。

## 最新完整验收

实验 `949fe60a2481470782aa2ae4820ba991` 完整成功，318.92 秒，1 次初始 Responses、23 次 Decisions，无额外 Responses、无运行时 graph patch。粉色与绿色分别实际抓取和放置，完成每物体双臂回观测位姿、清单观测及历史凭据记录；最后实测初始关节位姿和整项任务核验均通过。最终 Decisions 输入中包含两个不同输入 ID 的 post_delivery_observation，action_epoch 分别为 20、40；goal_satisfied=0.99，所有单项条件均通过既有 0.8 门槛。

真实当前头部输入图为 `decisions/call-136ae9d29b0f4de895539698a43685b2-image-007.jpg`；视觉复核两物体均在桌面，左盒为空。完整证据摘录：`data/native-final-949fe60a2481470782aa2ae4820ba991-evidence.json`。本轮绿色目的地为 Responses 选择的 demo-destination 配置桌面区域，最终 XY 由放置 skill 在该范围内求解。

这证明本轮任务实际完成，不能抹去前述身份、恢复、回位和一次轨迹跟踪超限失败，亦不能证明任意 grasp-place 的稳定成功率。代码分层与显式 graph 已落地；泛化能力仍受技能能力范围、观测和真实规划/执行结果限制。

回放浏览器最终检查使用成功实验 e46161b43e624677aae706c0020fa329：root + 2 个物体 subgraph、SVG/PNG/JSON 导出、文字边界、实际 JPG 与 MP4 预览全部通过，无页面异常。最新成功实验的 3D/RGB 视频在后台编码，当前 graph 与实际观测图已可读取。20/21 轮因维护重启中断的录像已启动离线恢复（`data/native-video-repair.txt`），仅使用原始归档，保留原任务 cancelled/needs_attention 状态；此处不声称编码已完成。用户初始场景已恢复，最后部署核对 86 个文件无差异。

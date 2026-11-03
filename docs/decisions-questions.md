# Decisions 问题与接受规则（2026-11-03）

2026-11-03：加工任务的最终目的地选择明确区分 input selector、来源、中间加工区和产物最终交付。问题内完整保留原指令和当前候选 ID／世界位置；未指定视角时，较大世界 Y 为左、较小 Y 为右，不能被历史参考图的“图像左右”描述覆盖。不判断加工是否已完成、未来是否可达或 CAD 选择；仍需原0.8置信度及与 graph 提案一致。支撑问题独立于是否换视角，unknown 不自行触发运动。详见[本轮审计](batch-and-graph-audit-20261103.md)。

本页只描述当前协议；历史实验以其归档请求为准。一次初始 Responses 编写 graph，运行时 Decisions 返回有类型判断/选项，不输出自然语言坐标。数值碰撞、IK、工作空间和轨迹由代码与算法检查。

## 问题分工

`source_identity` 的“来源”仅指输入原先所在的支撑／容器／来源区域，不是输入材料或输入物体本身。输入物体不能是自己的来源；盒内物体的直接来源是盒子，不是盒下桌面。该问题独立于输入成员资格、加工、CAD 和最终交付。

| 问题组 | 只回答什么 | 不要求什么 |
|---|---|---|
| `identity_N`、必要时 `match_*` | 当前 SAM3 主体对应哪个永久物体 ID | 不判断 CAD 型号、抓姿、任务完成或掩码质量 |
| `visibility_*` | 任一当前相机是否能辨认该物体 | 不要求三个相机都看见，不可见不等于不存在 |
| `cad_model` | 当前已关联主体最适合哪个 CAD 型号参考 | SAM 别名无需等于 CAD 文件名；不估计位姿，不再用事实 confidence 门槛拒绝明确型号提议 |
| `target_*`、`source_identity`、目的地绑定 | 物体/来源/目的地是否符合原始指令 | 不提前要求加工产物存在，不把中间加工区当最终目的地 |
| `support_*` | 目标的当前直接支撑或直接容器 | 盒内物体的直接容器是盒子；无需在 RGB 看见接触点或盒底；带日期的已验证支撑记忆有效 |
| `placement_destination_matches` | 本次转移的支撑/容器或配置区域是否符合语言 | 不问是否已经放好，不要求配置区域有可见标记 |
| `placement_additional_spatial_constraint` | 原句是否另有子区域、相对位置或姿态要求 | 只判语言含义；“右侧盒子”的右侧用于选择盒子，不自动成为盒内右侧要求 |
| `placement_xy` | 在已授权整区中选择一个候选 ID | 操作提议；不证明 Z、碰撞、IK，也不要求当前目标仍在原地可见 |
| `placement_candidate_*` | 候选足迹是否满足本次语言位置/姿态要求 | 不夹带 CAD、加工、回观测位姿、最终回位或整体任务完成 |
| `place_result` | 目标是否已在本次指定支撑面/容器内，视觉关系大致正确 | 允许位置误差；不要求释放后 XYZ、SAM3、FoundationPose、OBB、深度或不可见区域标记 |
| `placement_near_target` | 目标底部是否大致在投影标出的指定区域内或附近 | 不要求中心重合或整个物体投影位于框内；标注不是物体检测结果 |
| `workpiece_still_held` | 当前物体是否仍被夹爪实际保留/支撑 | 二维投影位于手指之间不等于夹持；已张开回执本身也不证明放置成功 |
| `unpack_result`、`wrapper_removed` | 当前黑色产物在桌面、粉色输入已被处理 | 不重新判断未来右盒交付，产物几何另由当前 RGBD 建立 |
| `scene_inventory_closure`、`source_complete` | 可见任务对象是否列全／all 来源范围是否充分 | 可见范围完整不代表隐藏处为空 |
| `mapping_scope`、`execution_arm`、恢复能力/参数 | 选择一个合法操作或参数 | 提议不是事实和安全证明；多个可用选项不要求每个都低概率 |
| `goal_satisfied`、`completion_condition_*` | 原始任务的最终要求与各物体完成账本是否满足 | 历史已确认动作按证据保留，不要求把历史动作在当前 RGB 重演 |
| 图内 `judgment:` | Responses 明确写出的当前条件 | 技能 success 已含其契约后置条件，重复条件会增加调用，不能替模型隐删 |

## 概率和缓存

事实 predicate：概率 >=0.8 为 yes，<=0.2 为 no，中间为 unknown。事实 choice：合法非 unknown 选项且 confidence >=0.8 才确认。没有把低置信度强改为成功。

操作 proposal（CAD 检索、XY、执行臂、建图范围、恢复能力/参数）保留所选值和原始 confidence，不用事实阈值重复否决；明确 unknown 仍返回 unknown。提议之后仍检查真实前提。已确认同实例型号和同输入判断可复用，但不得把旧位姿或旧 RGB 标为当前证据。

普通视觉问题使用冻结的当前三相机 RGB。INITIAL/PRODUCT 只是历史外观参考；主图、主体裁剪与 CAD 参考图明确标注。`verify_place` 一次请求同时询问放置关系、是否仍夹持，以及需要时的区域附近关系，保留模型原始回答，不平均/拼接多个概率。全部必要条件满足才确认；明确否定返回 `failure / visual_placement_rejected`，没有确定结论才返回 unknown。

已由 Decisions 授权整区的 `entire_region` 放置核验，历史目标外观留在共享 input，当前三相机图像放进各 question 的 instructions。支撑问题写明目标与支撑的实际 color/kind；区域问题使用同次 RGB 的区域标注；夹持反馈只提供给夹持问题。不再把完整 TaskSpec、后续加工和回位要求重复塞进这些视觉问题。`candidate_selection` 仍保留原始语言位置/姿态约束，不能借整区简化丢掉杯子左侧、朝向等要求。两种路径均无新增 SAM3、FoundationPose 或几何观测。无历史参考图时只引用绑定的外观描述，不声称有 REFERENCE 图片。图片日志同时计入公共 input 与 question.instructions；同一冻结相机图在多个问题中复用时分别计数。verify_place 不追加全局清单判断，全局完成由 graph 中独立 verify_task_completion 调用。

原位重复相同像素不能提高证据质量。有效 no/unknown 不自动重试；只有没有有效回答的临时网络/429/502/503/504 错误按既有上限重发相同冻结请求。不同视角由 graph 中显式运动取得，且必须已有可用于规划的地图。

夹持问题明确执行臂：A 为左臂／腕部相机 2，B 为右臂／腕部相机 1，头部为相机 3。当前三视角均保留，夹持问题最后给出执行臂腕部图；不能把另一臂空手当成本臂释放的证据。原始反馈只说明夹爪状态，不能替代物体视觉结果。

恢复问题写明原失败 node、skill 和错误，目标是修复该节点的前提，不是提前执行后续任务。是否需要新视角与是否有可用恢复操作是独立问题；没有视觉缺口不意味着地图修复也应停止。`move_to_observe` 可在前提修复后由 Decisions 显式选择重新规划，未解决的实际 motor/effect 部分执行仍阻止重试。`build_scene` 的参数效果明确 `observation_map_ready`，selected 与 observation 不混淆。

错误报告分别列出具体 question、原回答/概率、门槛、采集和对象 ID。选对 CAD 后的 `observed_support_unknown` 是支撑问题，`visual_identity_unknown` 是实例对应问题，不能统称 CAD confidence 失败。当前差异和真实耗时见 [诊断记录](runtime-diagnosis-20261102.md)。
回位恢复的 `retry_precondition_conflicts` 包含 `observation_motion_scene_ready`：selected 地图没有保留场景授权时不能提供 retry 选项。该事实由已有地图条件计算，不能由视觉概率或模型选择覆盖。修复动作及参数仍由 Decisions 选择。
同采集身份 predicate：将该问题专属 R/C 对比面板放在 instructions 中，直接问标注的两个主体是否为同一个物体。明确“盒子与内容物不同”“裁剪或框边界差异不等于换物体”。仅已核验相机/代次/源像素的面板适用；其他对应逻辑、共享完整场景及视觉否定/unknown 均保留。
`intermediate_destination_allowed` 的明确 no 与不确定 unknown 分开返回。若模型任务字段和完成条件互相冲突，不能通过重问同一张图修复，也不能把否定改为通过。初始 Responses 字段说明明确要求 object subgraph 的最终目的地覆盖加工产物谱系。
配置区域的 `destination_support_visible` / `work_area_support_visible` 直接描述支撑的颜色、外观和 ID，询问任一当前相机中的部分表面可见性。该问题不要求可见坐标边界、空闲落点或运动安全证明，不增加感知调用。
最终汇总与单物体条件采用一致的谱系及时间范围：`goal_satisfied` 和 `completion_condition_*` 明确选中输入 ID，历史阶段依赖对应验证回执，不能把消耗物体在最终 RGB 中不可见当作过去阶段失败，也不能仅凭 completed 状态通过。当前矛盾与缺失回执仍必须报告。

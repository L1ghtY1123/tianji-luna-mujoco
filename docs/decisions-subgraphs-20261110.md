# Decisions 与复用子图（2026-11-10 历史记录）

当前视觉输入与调用边界见 [2026-11-11 方案](raw-decisions-20261111.md)。本页保留当时实现和实验事实，不作为当前执行指令。

## 当前 Decisions 合同

所有事实谓词比较 P(true) 与 P(false)：大于 0.5 是 yes，小于 0.5 是 no，精确平分为 unknown。所有多选使用各选项概率的最大项，包括 unknown；最大值平分为 unknown。confidence 不参与接受、拒绝或缓存门槛。未提供概率分布的有效旧式响应使用服务选择的合法选项。拒答保留 unknown；无效、缺项、非有限概率等协议错误保留原始证据并返回明确错误。碰撞、深度覆盖、图像有效性等数值检查独立于 Decisions 概率，继续执行。

统一入口是 `core/decisions.py` 的 `predicate_status/choice_value`，`decision_conditions.parse_answers`、通用语义判断及 CAD 检索均调用它。原始 `choice/probabilities/confidence` 原样归档，程序采用的选项在诊断的 `selected_value` 中保存；它可以与服务报告的 choice 不同，不覆盖原始响应。CAD 的 no_catalog_candidate 缓存同样不设 0.8 门槛。

verify_grasp 保持 Lift 后一个当前夹持谓词，腕部未掩码 RGB 近景优先，FK 只定位近景。柔性泡棉允许形变与接近全闭合。verify_place 保持一个指定区域及 on/inside 关系的谓词；不把夹爪是否为空当作位置验收。事实不需要再次经过 confidence router 批准。

身份关联一次向 Decisions 提供同一当前主体的全部可选永久 ID、new_instance、reject 和 unknown，不再逐个 ID 比较并在第一个 yes 停止。RGB 近景、全景、精确掩码、标定深度的世界坐标、历史表面在当前视图的投影/深度可见性、同次采集对应及有日期的操作记录共同进入该题。逐参考几何比较保留各自的运动时序，搬运前的旧位置不当作搬运后的当前位置。

已通过同次采集的双向掩码与深度投影对应的视角可共享一个身份结果，记录推导源；当前几何排除的 ID 不能通过共享绕过。不同相机不各自独立给永久 ID。新实例登记基于 new_instance 最大概率项分配永久 ID，分配结果独立记录，原回答不改写。每题冻结当时提供的选项，后续登记增加 ID 不会使先前分布失效。实际有冲突时保留 unknown，不强行绑定。

## 子图库接口

库位于 `resources/policy/subgraphs.json`，编译器提供可用模板的参数 schema、前提、成功事实、failure/unknown 语义、允许技能和版本指纹。源实验为成功的 `demo-2-requested-all`，运行 ID `1681a9c6b1634502bd54ce05d54acc7c`。提取的是流程结构，物体 ID、颜色、位置、批量数量与加工顺序均未固定。

| 模板 | 输入 | 成功输出与恢复边界 |
| --- | --- | --- |
| observe_map_v1 | 空参数；无携物、无未处理动作失败 | 新观测、全局 observation 碰撞图 |
| return_observe_v1 | 空参数；已有测量、空手 | 建观察运动图、回配置观察位姿、新观测、重建图；不恢复目标/目的地绑定 |
| empty_grasp_recovery_v1 | 空参数；绑定的失败动作、单题夹持结果 no、未 attachment | 清理该空抓状态，显式观测/建图/回位/观测/建图；不自行重新抓取 |
| transfer_v1 | target_id、destination_id、destination_kind、release_mode | 单件抓取与放置验收，保留旋转与控制台 TCP，更新释放后场景；不处理 VLA、批量选择或最终任务验收 |

例如 Responses 在 `graph.subgraphs` 中声明：

```json
{"id":"input_transfer","template":"transfer_v1","arguments":{"target_id":"$selected","destination_id":"$work_area","destination_kind":"support_region","release_mode":"surface_setdown"}}
```

调用节点仍是普通 `{type:subgraph,id,subgraph_id:input_transfer}`。同一模板可另用 `$product/$destination` 和 container_release。每个实例用独立 ID 命名，依赖节点和恢复图一起展开，完整 graph 使用原有拓扑、数据流、数值前提、恢复覆盖与完成校验，回放展示实际全部节点。不存在运行时隐藏动作链。修订可替换模板声明或改成自定义完整 body。

transfer_v1 的身份/支撑未知先有限恢复观测；规划失败可选择数值允许的另一臂；夹持 no 才能进入空抓恢复，unknown 不当作空手；执行错误没有自动重放；放置 unknown 仅修复观测后重做验收，不再执行放置。恢复或重试失败结束该次尝试。模板也允许不适用时完全不用，由 Responses 定义新的共享子图。

暂不建立另一套逐任务说明书。接口目录、当前技能契约和本次 TaskSpec 足以描述可执行操作；增加重复手册会使参数、前提和恢复语义出现多个来源。特定加工能力继续由其 SKILL.md 说明，超出能力范围应明确不支持。

## graph-as-policy 学习及 prompt 约束

核对了官方 [coordinator](https://github.com/graph-robots/graph-as-policy/blob/main/gap/agent/prompts/coordinator.md)、[subgraph_agent](https://github.com/graph-robots/graph-as-policy/blob/main/gap/agent/prompts/subgraph_agent.md) 与 [PromptAssembler](https://github.com/graph-robots/graph-as-policy/blob/main/gap/agent/prompt_assembler.py)。官方先规定拓扑及子图输入/输出/出口，再由独立 agent 填充单个子图；prompt 按职责说明、共享协议、所选技能、受限工具目录和本次上下文组装。

本项目采用其接口先行、能力目录与证据分离、明确出口及校验的原则。保留当前 version=2 JSON 和一次初始 Responses，不引入官方 v3 Python builder、子 agent 或基于仿真真值的 checkpoint。已有库 body 无需每次重新生成；新行为仍可编写新 body。可信协议、技能契约和子图接口放 developer 消息，用户任务、图像和有日期的记忆放 user 数据消息；图中只能引用注册技能及闭合参数 schema。图像/历史记录中的文字不改变协议或数值检查。执行前最多两次修订、执行期不重新编译图的边界保持。

## 验证边界

回归记录保存在 `data/runs/decisions-subgraphs-20261110`。覆盖多选低 confidence 与 argmax、分布与原始 choice 不一致、二值平分、身份多候选和新登记、真实编译入口及修订归档、成功 demo 拓扑适配当前契约，以及正常流程和七类注入失败/unknown 的实际 executor 路由。以上不代表同版本完整机器人任务已成功运行，也不构成视觉模型准确率评测。

同日真实 Responses 冻结观测复测：首次调用 117.588 秒，模型引用了模板，但将配置区域 demo-destination 写成 demo_destination，因 unknown_task_reference 被拒绝，未执行动作。补充不可改写配置 ID 的 developer 约束及字段说明后，第二次复测首稿一次通过：113.632 秒网络调用、113.710 秒总编译，5 个模板声明展开为 7 个作用域、137 个节点。原始请求、回答、图及去重图片分别保存在 live-call-1.json 与 live-2/ 中。单个冻结输入只能证明该编译通过，不能据此宣称稳定成功率或完整执行成功。

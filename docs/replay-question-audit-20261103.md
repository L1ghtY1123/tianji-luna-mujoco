# Graph 回放、问题与 unknown 审计

本轮保留一次初始 Responses 编排、Decisions 选择已声明 edge、固定技能实现。没有降低概率门槛，没有隐藏真实执行失败，没有改动 cuRobo、碰撞计算或 CAD。

## 已定位的缺陷

- 原 graph 渲染把所有节点排成横向单行，恢复边绕过主线，未执行 failure/unknown 也醒目着色。页面仍有“Runtime 追加节点”的旧说明。现在默认显示正常分支和实际走过的全部分支，纵向分层、直角连线；完整初始图可展开，JSON 始终完整。实际失败不会被过滤。
- 配置区域问题曾把“绑定明确冲突”也描述为 uncertain；同帧身份 predicate 曾把“不同主体”和“无法区分”混写为 reject。已明确否定与缺证据的不同含义。
- 放置语义候选全部可见且全部 no 时，程序仍返回 unknown。现在返回 failure / placement_candidates_rejected；有未观测或未决候选时仍保留 unknown，任一候选 yes 则可继续。
- 同帧身份转换未接受合法 refusal，误报格式错误。现在保留拒答及无概率的原始结果；普通身份和清单闭合复用严格解析器，不再把任意错误类型吞成 unknown，也不为拒答补 0 分。
- 碰撞 CAD 使用统一半透明颜色，且只展示时间/token。现在实例跨阶段同色，桌面灰色，实心参与碰撞、透视虚线表示环境层已排除；展示建图节点、用途、全局/局部更新、排除目标及实际层状态。没有修改 enabled 或碰撞算法。

## unknown 实际由哪里产生

| 生产方 | 输入与转换 | 含义 |
|---|---|---|
| agent/decisions.py；decision_conditions.py | predicate p≥0.8 为 yes，p≤0.2 为 no，其间 unknown；事实 choice confidence<0.8 或显式 unknown；typed refusal | 模型原始答案经过应用规则变成状态。predicate 原始输出是概率，不是 unknown 字符串 |
| identity_proposals.py | 同帧身份提议的 predicate 未通过时，提议不能绑定永久 ID | p 很低是明确拒绝该配对，但其他身份尚未确定；派生 identity unknown 不等于模型给了中间概率。保留 derivation_reason 和原始答案 |
| identity_correspondence.py | 必须恰有一个肯定配对，其余全部否定 | 多个肯定、无肯定或未排除的替代配对产生未确定身份；派生分数不是 API 的分类 confidence |
| runtime/placement.py | 几何缺失、支撑/候选图像版本过期、空间语义未确定 | 部分 unknown 在调用 Decisions 之前由程序产生；不是模型慢或模型拒答 |
| runner.py、verification.py | 地图缺失/过期；夹爪反馈不足或释放回执未确认 | 程序根据地图或机器人反馈生成 unknown，不经过视觉概率 |
| runtime/controller.py | 对身份、支撑、任务绑定、来源完整性、验收缺证据等结果分类 | 需要结合具体 reason 与 diagnostic_checks；明确违反必要条件不能伪装成缺证据 |
| runtime/native_graph.py | 子图结果传播；Responses 声明的 unknown end；物体成功端缺完成回执 | 终点和父子图重复显示同一结果，不代表多次独立模型失败 |
| robot_skills/processing.py 等技能证据 | 加工后产物深度/身份或所需回执未确认 | 技能缺必要证据；不一定存在一次低概率 Decisions 回答 |

SAM3 score 用于候选分割，不能直接解释 graph unknown。回放失败节点新增 outcome_evidence：记录 skill/router/子图返回/声明终点/完成回执检查的来源、reason 和原始诊断证据。旧实验不补造不存在的来源。

明确问题能修复提示词导致的错误，不能保证遮挡、身份不可区分、拒答、服务错误或证据缺失永不发生。不能删除 unknown 出口或把它转换为成功来提高表面成功率。

## 提问范围核对

静态清单位于 data/runs/prompt-question-inventory-20261103.json：36 个构造位置，包含动态问题族。另检查实际发送前的同帧/配对、多视角、主体图和释放问题改写，不能只审查初始字符串。

| 问题族 / 文件 | 核对边界 |
|---|---|
| graph_prompt.py、responses.py、schema.py、21 个技能契约 | 一次编排；主线简短；恢复需具体可恢复结果及已满足前提；复用后续节点。router 单一谓词；yes/no 含义分别成立，未知不等于可移动相机 |
| cad_perception.py：cad_model | 当前掩码主体与目录模型提议；不估计位姿，不判断永久身份/目的地；unknown 与 no_catalog_candidate 区分 |
| visual_inventory.py：identity/visibility/progress | 永久实例、可见性、当前位置分开；container 与 contents 分开；历史参考不当作当前位置 |
| identity_proposals/correspondence、identity_comparison_images、identity_subject_views、multiview_identity、demo_identity_presentation | 同帧提议、跨帧配对、实测跨视角分组有各自证据；原始 predicate 与派生身份结果分开归档 |
| demo_completed_tracking.py：tracked_*；demo_perception.py：scene_inventory_closure | 可移动主体与清单遗漏分开；清单闭合不等于隐藏空间为空 |
| task_binding.py；controller.py：target/source/destination/work_area | 目标成员、来源、最终目的地、中间配置区域分开；配置区域不要求 RGB 中有标记；加工产物不替代原输入选择器 |
| controller.py：support_* | 直接支撑/容纳关系，不要求看见接触像素或隐藏盒底；日期明确的历史证据可佐证连续性 |
| controller.py：mapping_scope/update_entity_*、execution_arm | 操作提议，不是空间覆盖或运动安全证明；删除与建图范围无关的全机器人 free-ray 文字 |
| agent/placement.py；runtime/placement.py | 区域身份、额外语言限制、候选语义、XY 操作提议分开；数值 Z/碰撞/IK 后续验证 |
| demo_transfer_verification.py；release_decision_input.py | 一次当前三相机 RGB 分别判断支撑、区域附近、仍夹持；明确反例给低概率，不能混同缺证据 |
| robot_skills/processing_evidence.py；observation_requirements.py | 加工技能固定结果与包装消失分别判断；观测充分是能确认或排除，不要求动作成功 |
| completion.py；controller.py：completion/object/source_has_remaining_input | 历史里程碑看对应回执，当前来源残留看当前 RGB；来源残留需要 no，其他必要条件需要 yes；机器人位姿用反馈 |
| visibility_recovery.py；controller.py：condition | 看不到、裁剪、遮挡独立判断；只选择 Responses 已声明恢复边；不从低分推断遮挡 |

question 类型、名称、选项和答案长度由当前协议检查；错误类型、非法分数、缺失/乱序答案仍报协议错误，不能变成正常 unknown。

## 碰撞快照解释

新快照 stage_context 归档 build_scene 节点、scope/reason、局部 entity_ids、退役 ID、排除目标和物体描述。旧实验只在同一 invocation_id 的 skill_start 存在时恢复其真实建图参数，不按颜色或前后时间猜测。

global 是更新全部当前观测清单实体；local 是更新指定实体并保留其他已存在层。两者都不等于全空间可见，也不意味着每层都刚刚测量。selected 的抓取目标从环境层排除；此时还未生成 lift 后的 attachment。每层 enabled 是该快照中的实际值，虚线只改变显示，不改变碰撞设置。

## 验证

已通过：Python 全源 AST 与 Ruff F；21 个技能契约；否定/未决/不可见候选组合、拒答与非法返回、身份派生原因、嵌套诊断证据；实际历史图的投影/布局/环/真实 failure 保留/不可变性；浏览器真实页面及完整图切换；48 模块前端构建。

完整仿真：

| 实验 | 总耗时 | Responses | 其余耗时 | 结果 |
|---|---:|---:|---:|---|
| 48a664a8e79042f0865df5be9af489d9 | 197.246s | 54.488s | 142.758s | 物理任务成功；精简 prompt 后全部故障直接结束，未保留有效恢复，因此继续修正提示词 |
| 0aede4f653444b47be4922e73f735973 | 216.105s | 69.959s | 146.146s | 50 nodes、1 router，保留换臂恢复；任务成功，未执行 failure/unknown |

最终实验：选中输入等于完成输入，产物谱系和两次释放确认，实际双臂 observation 反馈通过；只在结束调用一次 move_to_observe。25 次 Decisions 请求共 70 个问题，名称/类型/选项/答案长度及数值范围符合实际协议，无 refusal、显式 unknown 或中间 predicate 概率。逐次答案报告：data/runs/replay-decisions-audit-20261103.json。

对最终模型原图注入规划失败的控制流探针：A→B、B→A 均经过显式换臂与抓姿重生成后接回主线；替代臂不可用时在执行抓取前停止；原 graph 不变。此探针验证路由，不声称已经实测物理失败后的恢复成功率。

附加回归覆盖：清单闭合 refusal 的 null 概率保留为待最终核对的 unknown；伪装成普通 choice 的空分数、错误来源/版本绑定继续拒绝。删除已经没有调用方的旧 current-item-drop 核验/裁剪辅助函数及专属导入，避免废弃 unknown 路径继续混在源码中。

证据：data/runs/replay-audit-final-20261103.json、replay-audit-final-summary-20261103.json、replay-recovery-probe-20261103.json。本轮两次完整任务成功，不据此宣称所有任务或恢复分支都达到稳定成功率。

最终浏览器检查：默认视图 41 nodes，完整编排 50 nodes；四次建图切换后永久 ID 颜色保持一致，无 pageerror。前端构建 48 modules 通过，保留既有 Three.js 大于 500 kB 的 bundle 提示。应用源码和文档共 28 文件与 Ubuntu SHA256 一致；SDK/算法库未改动。

---
name: luna-policy-isolation
description: Use when modifying, debugging, extending, or reviewing tianji-luna-mujoco task planning, robot skills, visual decisions, inventory mapping, placement, or cuRobo integration. Preserves the ownership boundary between Responses, Decisions, and physical skills.
---

# Luna 任务策略隔离

本 skill 指导 Codex 开发，不是机器人运行时能力。先阅读项目 [AGENTS.md](../../../AGENTS.md) 和 README；AGENTS 是当前要求的唯一合同，本文不累积历史覆盖条款。不使用 JEV runtime 或 sub-agent。

| 层 | 职责 | 边界 |
|---|---|---|
| Responses | 一次初始清单、任务要求及 graph/subgraph/node/edge | 不加载开发者写死的拆包流程，不二次修订 |
| Decisions | 视觉/语言事实、原图分支、运行时恢复能力及参数选择 | 有类型输出，不把动作提议当事实，不替代数值安全检查 |
| Runtime | 按 edge 执行、永久 ID 与参数绑定、事实版本、账本和 graph patch 审计 | 不暗中移动、观测、换臂、建图或重播成功动作 |
| Robot skills | 选中能力内部算法、传感器与机器人接口 | 可固定内部步骤及真实前提，不串接隐藏任务流程 |

## 修改检查

1. 追踪实际入口、contract、handler、模型输入、结果消费方与回放。只描述真实能力和前提，不用命名把旧 steps 包装成 graph。
2. graph 只接受当前 version=2；每个输入永久 ID 有独立任务 subgraph。node ID 是控制节点，CAD asset ID 是模型，均不替代永久实例 ID。
3. 修改 questions 时分别核对 CAD 选择、实例身份、可见性、支撑、仍夹持、目标区域与全局完成。没有参考图片就不能声称图片存在；三相机证据可在公共输入或 question.instructions 内，日志需完整计数。
4. verify_place 是一次融合三相机 RGB 的定性判断，不重建几何，不追加隐藏清单核验。verify_task_completion 由 graph 显式调用。
5. 每件交付后的双臂 observation 回位和观测属于用户要求，必须由 Responses 显式编排。VLA 的固定起始姿态属于技能前提，不能成为所有任务的硬编码流程。
6. CAD/OBB/cuboid 绑定永久 ID；SAM3 生成候选，Decisions 匹配目录，FoundationPose 估计位姿。当前位姿来源和坐标有效即直接使用，不恢复 ICP 或吻合门槛。仿真真值不进入感知或规划。
7. 普通 XY 分区选择、配置 XY 直用、Z 靠近碰撞阈值；模型判断不覆盖碰撞、工作空间、IK、证据版本失败。未知物体 GraspGen、已知 CAD 分臂离线抓姿；attachment 在 lift 后建立。
8. unknown 保留为真实 graph 结果，交由已有 edge 或 Decisions 选择恢复，不自动转 yes。仅视角缺证据才考虑换观测位姿；相同输入不重复观测。恢复采集原始 RGBD 不等于 observe_scene，也不暗中发布地图。
9. 删除替代实现时一并清理专属测试、依赖、prompt、前端和部署文件；迁移当前有效回归，不靠兼容分支保留旧框架。第三方 SDK 保持只读。
10. 按 AGENTS 的 CPU 限额完成针对性测试、全量必要检查与真实仿真；报告成功与未覆盖的边界，不以健康接口替代任务验证。

模型 prompt、技能合同和观测恢复边界参见 `docs/native-graph-implementation.md`、`docs/luna-skills-guide.md`。历史报告仅是当时的证据，不作为当前实现要求。

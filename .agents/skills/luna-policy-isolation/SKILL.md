---
name: luna-policy-isolation
description: Use when modifying, debugging, extending, or reviewing tianji-luna-mujoco task planning, robot skills, visual decisions, inventory mapping, placement, or cuRobo integration. Preserves the ownership boundary between Responses, Decisions, and physical skills.
---

# Luna 任务策略隔离

本 skill 指导 Codex 开发，不是机器人运行时能力。先阅读项目 [AGENTS.md](../../../AGENTS.md) 和 README；AGENTS 是当前要求的唯一合同，本文不累积历史覆盖条款。不使用 JEV runtime 或 sub-agent。

| 层 | 职责 | 边界 |
|---|---|---|
| Responses | 初始实例、任务要求及完整图，显式路由组由程序展开，优先开局一次 Responses，执行前按图校验错误最多修订两次 | 保留原清单和根任务要求，逐版归档，首次校验通过即采用 |
| Decisions | 视觉/语言事实与已声明分支选择 | 有类型输出，不把动作提议当事实，不替代数值安全检查 |
| Memory | 时间记忆数据库保存实例和历史，空间记忆保存碰撞场景及其版本和有效性 | 原始观测、测量缓存及配置区域归执行层；时间记忆拒绝任务进度和当前碰撞有效性字段 |
| Runtime | 按 edge 执行、永久 ID 与参数绑定、事实版本、账本和已声明路径审计 | 不暗中移动、观测、换臂、建图或重播成功动作 |
| Robot skills | 选中能力内部算法、传感器与机器人接口 | 可固定内部步骤及真实前提，不串接隐藏任务流程 |

## 修改检查

1. 追踪实际入口、contract、handler、模型输入、结果消费方与回放。只描述真实能力和前提，不用命名把旧 steps 包装成 graph。
2. graph 只接受当前 version=2；共享 subgraph 可循环处理运行时选中的输入，清单保存身份与历史。独立任务 subgraph 为可选表达。node ID 是控制节点，CAD asset ID 是模型，均不替代永久实例 ID。
3. 修改 questions 时分别核对 CAD 选择、实例身份、可见性、支撑、目标区域与全局完成。没有参考图片就不能声称图片存在；三相机证据可在公共输入或 question.instructions 内，日志需完整计数。
4. verify_place 对普通 container_release 校验执行绑定、实际盒上释放与开爪，按用户规则建立入盒回执。该分支无需 Decisions。surface_setdown 和额外落地关系使用同次三相机 RGB 逐题核验。空间记忆允许以已测容器包围范围覆盖已入盒且无法逐件定位的内容物，并记录规则和几何来源。
5. 批处理每件交付后由 Responses graph 显式回配置 observation 位姿并重新观测，再选择当前输入。all/count 使用共享子图，初始数量不限制运行时成员。all 以最新观测确认来源盒已空结束本批，count 按请求数量验收。最终双臂回位由实际关节反馈验收。VLA 起始姿态仍属于加工技能前提。
6. CAD/OBB/cuboid 绑定永久 ID；SAM3 生成候选，Decisions 匹配目录，FoundationPose 估计位姿。当前位姿来源和坐标有效即直接使用，不恢复 ICP 或吻合门槛。仿真真值不进入感知或规划。
7. 普通 XY 分区选择、配置 XY 直用、Z 靠近碰撞阈值；模型判断不覆盖碰撞、工作空间、IK、证据版本失败。未知物体 GraspGen、已知 CAD 分臂离线抓姿；attachment 在 lift 后建立。
8. unknown 保留为真实 graph 结果，沿已有 edge 或由 Decisions 在已有恢复分支中选择，不自动转 yes。恢复优先回配置观测位姿后重观测，配置回位无需额外视觉许可题，保留空手、地图和运动检查；额外腕部视角才需要视觉缺失判断。相同输入不重复观测。恢复采集原始 RGBD 不等于 observe_scene，也不暗中发布地图。
9. 删除替代实现时一并清理专属测试、依赖、prompt、前端和部署文件；迁移当前有效回归，不靠兼容分支保留旧框架。第三方 SDK 保持只读。
10. 按 AGENTS 的 CPU 限额完成针对性测试、全量必要检查与真实仿真；报告成功与未覆盖的边界，不以健康接口替代任务验证。

模型 prompt、技能合同和观测恢复边界参见 `docs/native-graph-implementation.md`、`docs/luna-skills-guide.md`。历史报告仅是当时的证据，不作为当前实现要求。

# 旧入口清理与失败证据

用户要求彻底移除旧任务模式，并具体说明 Decisions、SAM3 与运动规划各自在哪个检查失败。本次不修改算法 SDK、不降低置信或碰撞阈值、不增加模型调用。

## 清理范围

- Runner 删除 `recover_task`、`assess_task`、`inspect_scene` 分派、`snapshot_for_decision`、隐藏补观测/建图编排及旧重试授权旁路。保留当前控制器调用的五个物理原语。
- `demo_mode` 参数不再接受；非空旧 `decisions_factory` 在构造时拒绝。PolicyController 直接执行也要求注册 handler，仅注入旧 contract 不能复活旧能力。
- 去掉旧视觉发现审计接线和异常触发初始清单初始化。
- 当前结果谓词不兼容旧 choice 回答；历史档案只读展示，不进入实时证据解析。
- Web 删除旧 process_records/stage_index、固定两段路线、旧 program 转图、旧技能别名及专用完成计数。旧归档可以查看原始内容，不能由前端重新编出一条现代执行流程。

部署挂载路径 `/opt/jev`、历史档案、底层 MuJoCo 瞬时拆包效果名、地图实体的生存状态管理不是旧任务控制器。它们不提供旧任务模式开关。新增技能继续通过 contract、参数 schema 与显式 handler 扩展；显式重试由策略图预算控制。

## 错误报告

`runtime/diagnostics.py` 将本次错误证据转换为 `failure_report`，保存于状态、`skill_error` 和 `run_end`。底层 `error_code/error_details` 与图层 `policy_failure` 分开保存。

报告保留当前问题、原始回答、实际概率/置信度、验收规则、相机/候选/实体、观测 ID 和请求哈希。Decisions 采用自身置信度或谓词概率，SAM3 采用独立分割分数；有效深度像素数是 RGBD 过滤。零候选时不推断内部过滤原因。对服务没有返回的内部理由明确标为未知。诊断文本净化凭据和图像数据，非有限数保留为字符串以免错误报告破坏状态 API。

处理技能在返回值顶层携带的证据同样保留。视觉判断通过后发生深度/工作空间失败，归类到几何检查；不把先前视觉分数当成后续失败根因。图的 `judgment:` 条件返回 unknown 时也保留问题和概率。

## 真实归档核对

运行 `ac92904bbee146b78e98a7e1013d3664` 的身份关联为 0.59，低于 Decisions 的 0.8 门槛。这不是 SAM3 分割分数。参考框错误及同帧修正复放的证据见 `no-lifecycle-validation.md`；模型内部低分原因没有独立解释，不把假设冒充确定原因。

运行 `1d0ed2dfda8a42349ddf04ae9476e0e5` 的事件 82 是 `planning_no_valid_candidate`。读取完整原始事件得到 32 条 `command_validation/execution_observation_coverage_insufficient` 和 3 条 planner 阶段记录；它们是失败记录数，不是独立候选总数。首个缺口在采样 0、球 2，球为 `[-0.000458054, 0.161345229, 1.121313810, 0.058698270]` 米（XYZ 和半径）；包含差值约 `1.184088119e-9` 米，当前容差为 `1e-9` 米。后续采样还有更大的包含差值，不能仅凭首点判定整条路径已安全。

本次将原始事件离线转换成诊断报告用于展示核对，没有改写旧实验，也没有调用 Responses/Decisions 或执行机器人动作。完整物理拆包尚未通过，覆盖检查的数值边界与真实空间缺口需继续独立验证。

## 验证记录

- 完整 Python 检查：2690 passed、63 skipped、33 failed，349.63 秒，输出 `data/legacy-cleanup-suite.txt`。其中 26 个失败名与上一轮相同；另外 6 项为闭包未知仍采用旧清单门槛的测试，1 项为新增诊断强取未记录 SAM 分数造成原错误被覆盖。
- 6 项旧预期迁移并增加身份/时效/漏项/最终闭包保护，文件内 72 passed。缺失 SAM 分数记录 null 和 unknown，保留原身份错误；相关文件 16 passed。
- 最终集中复核：312 passed，17.03 秒，包含策略图、错误报告、旧入口拒绝、模型层、当前清单与感知、结果谓词、归档等，输出 `data/cleanup-final-focused.txt`。没有以此宣称 26 项原有全量失败已修。
- 前端串行 349 passed；Vite 构建通过。修改的后端模块及测试 Ruff 通过，`git diff --check` 通过。构建仍有已有的大 chunk 提示。
- 本轮不重复发起付费模型请求或物理任务；真实完整拆包仍以先前未完成状态记录。
- 部署：261 个文件逐字节一致，累计 60 个已移除旧文件在服务器均不存在。应用前备份为 `data/before-no-lifecycle-20261026T052639.tar.gz`。本项目四容器配额仍为 2/1/0.5/0.5 核；cuRobo 与 GraspGen 就绪，运行状态 idle。
- 浏览器：真实旧实验的完整策略图与原始错误可读；将同一实验事件 82 离线生成的报告仅在浏览器测试中注入后，具体失败次数、坐标与差值均显示。零脚本异常、零写请求，未改写实验归档。证据为 `data/cleanup-browser.json`、`data/cleanup-diagnostics-browser.png`；后续新运行直接归档结构化报告。
- 展示复查后将对象测量值改为逐字段中文说明并保留单位，单项组件测试 9 passed。

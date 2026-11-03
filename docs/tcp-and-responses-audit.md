# 夹爪前端坐标与 Responses 重复调用核查

2026-10-27。修改已部署；最终版本实验 103 完整通过，实际仅调用 1 次 Responses。以下同时保留中间实验的失败与限制。

## 两次调用的实际原因

已读取实验 `bec3e4d4764c4cedab9d99f9c5a30001`（98）和
`d7e1b825e18d45ef829235aaae32a21e`（99）的原始请求、模型公开 JSON 与校验错误。
两次追加调用均发生在初始编排阶段，尚未执行机器人动作。HTTP 均为 200；第一次
返回的图被项目契约校验拒绝，第二次由模型修正自己的提案：

| 实验 | 初始图错误 | 缺少的事实 |
|---|---|---|
| 98 | `planning_precondition_not_satisfied:map_after_final_drop:observed` | drop 使 observed=false；verify_drop 不重新建立完整清单观测，图却随后直接 build_scene |
| 99 | `planning_precondition_not_satisfied:prepare_vla:scene_available` | 放下后旧地图失效；verify_place 仅确认放置，没有发布当前碰撞地图，图却直接 prepare_unpack |

这些前提原已包含在技能契约中，不能将问题归因于没给模型信息。现在首次输入增加由注册契约动态生成的紧凑状态转移表，将 provides、planning_effects、参数条件和前提放在一起。模型需逐条追踪失效并检查自己的完整图。表中没有指定任务步骤或拆包模板，解释器不补节点。

回放调用记录区分 initial_compile、compile_repair、event_revision、revision_repair，保存尝试编号、事件键与具体校验错误。格式/契约修正仍与事件修订共享最多两次额外调用，执行中新事件仍须先经 Decisions 判断。

实验 101 暴露了另一种追加调用：建图判断返回 `decisions_judgment_http_503`，入盒后观测返回 `visual_decisions_http_503`。初始图没有处理这些结果，Decisions 两次选择 revise，因此实际为 3 次 Responses；物理任务成功也不能算一次编排通过。首次输入现增加这两项真实服务故障的含义与恢复注意事项，要求模型考虑可重复、无动作能力的有界恢复；具体重试节点仍完全由模型选择，代码不添加节点、不改变图或阈值。执行事件修订的审计 trigger 同时记录原始事件及原因。

复核发现原 HTTP 缓存也会阻碍模型选择的恢复：相同请求的 503 被保存，下一次调用只重抛旧错误。现仅对 `/v1/decisions` HTTP503 不缓存错误；第一次仍抛出错误，不进行自动重发。成功结果、语义错误、配置错误和审计失败的缓存保持。是否再次调用、次数和顺序仍由模型图决定。

实验 102 说明仅增加提示还不稳定：初始图再次缺少 `prepare_vla` 的 scene_available；修正后物体成功入盒，但最终观测中右腕相机的 blue bin 候选身份未知（SAM3 score=0.76171875，派生身份 choice=unknown、confidence=0，门槛仍为0.8）。其后修订生成不支持的 `map_for_return.ok` 条件且预算耗尽，任务未完成最终回位。没有放宽身份或碰撞检查来通过。

因此首次编排推理强度由 low 改为 medium，事件修订保持 low。实验 103 首次响应 64.453s，一次通过契约校验；模型自行生成 8 个窄范围有界重试节点，分别用于其选择的观测、建图和核验。首次调用更长，不能据单次总耗时声称全面提速。语义未知、真实环境变化和模型图错误仍可能需要修订；没有保证任何任务永远只调用一次。

## 坐标合同

过去应用中的 `tcp` 和 `T_world_tcp` 指 `TCP_Link_L/R` 安装端，前端另有 `tcp_front`。
这来自项目的工具适配与 cuRobo 配置，并非 SDK 要求 attachment 使用安装端。
GraspGen 的模型帧、机器人的安装端、校准夹爪前端是不同的定义，需显式转换。

本轮将应用反馈、单/双臂笛卡尔目标、IK/FK 证据及 attachment 统一为
`gripper_front_A/B`。沿安装端局部 Z 的固定距离为 0.185 m：

```
T_world_front = T_world_mount @ Translation(0, 0, 0.185)
p_front = p_mount - [0, 0, 0.185]
```

项目通过 cuRobo 的固定 extra_links 定义前端工具；附件链接在同一个前端位置，输入球为前端局部坐标。相机安装外参、URDF 碰撞球及完整夹爪碰撞子树保持其物理链接定义。内部夹爪接触模型仍用安装端局部几何，由明确适配器转换输入；这不改变公开轨迹目标。

旧的 `frame=tool` 持物和无匹配 tcp_frame 的绑定候选不自动猜测或重标。历史实验不改写，测试加载器显式转换。工作空间原本已经以同一前端位置采样，其物理标定未变，不能仅因接口改名重写 atlas 哈希。

夹爪前端的位置误差才是新 IK 验收位置误差。历史候选 826 的前端误差 5.54 mm，超过未改变的 5 mm IK 位置容差，因此不再接受；该位置容差与普通障碍的 5 mm 碰撞净距是独立参数。

官方依据为锁定 cuRobo commit `8e734f3ced1df898990bcd92de40abce475907db` 的
[固定链接配置](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/robot/loader/kinematics_loader_cfg.py)
和 [AttachmentManager.update](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/collision/attachment_manager.py)。未传 world_objects_pose_offset 时，update 使用附件链接局部球，不会自动替项目转换安装端与前端。

## 验证记录

- 全量回归：3338 passed、58 skipped（439.45 s）。后续仅整理测试导入及一个等价表达式，受影响 45 项再次通过；本轮修改范围 Ruff 通过。
- 独立 GPU 几何审计：双臂 × 4 种常驻规划配置 × 7 个阶段，共 56 组。比较 MuJoCo 与 cuRobo 的 planner、checker、IK/graph 中机器人及附件球，最大位置差 1.4712e-6 m；前端 FK 矩阵最大逐元素差 1.3639e-6。此审计没有执行动作，不能代替物理任务。
- 真实实验记录如下；独立验收核对最终物理入盒及输入/产物 ID 关联。CAD 只用于此离线验收，没有提供给运行时规划。
- 最后两处修改（初始推理强度、503缓存）后，相关回归 107 passed、Ruff 通过。未修改运行时重试、调用预算或技能内部动作。
- 将实验 103 原始模型图中的最终观测/建图 retry 节点原样取出，做离线 HTTP 503→200 故障注入：两者均在相同请求的第二次调用成功，无额外 Responses；此测试没有执行物理动作。另有集成测试确认成功后仍缓存、首次503不自动重试。

| 实验 | 完整任务 | 总耗时 | Responses次数 | 独立验收 |
|---|---|---:|---:|---|
| [100](http://192.168.3.4:8350/replay?run=660caa873558461f9b8a3eeb5b43f70d&view=graph) | 成功 | 273.94s | 1 | 通过 |
| [101](http://192.168.3.4:8350/replay?run=a4c7b6cf19504e898c7ac4cfca9e127a&view=graph) | 成功 | 424.59s | 3 | 物理/ID通过；单次调用指标不通过 |
| [102](http://192.168.3.4:8350/replay?run=ad81f6dcf6fc49e8b07ba08e93d89609&view=graph) | 未完成最终回位 | 233.82s | 3 | 物理/ID通过；完整任务不通过 |
| [103](http://192.168.3.4:8350/replay?run=aec148bb36bf47ea8170ee1193c38f4f&view=graph) | 成功，最终版本 | 250.15s | 1 | 通过 |

完成的三轮均包含双臂抓取、lift 后附件绑定、桌面放置、双臂 VLA 起始、入盒与双臂最终回位。最终版本只有一轮完整实验，不能以此推断任意任务成功率。实验 103 两臂最终关节误差约 0.000004°，硬净距仍为5mm。

证据位于 `data/tcp-response-tests-final.txt`、`data/tcp-response-lint.txt`、`data/tcp-response-geometry-audit.json`、`data/tcp-response-validation.json`、各轮原始状态与独立 audit JSON，以及 `data/tcp-response-model-retry-audit.json`。中间实验没有删除或改写。

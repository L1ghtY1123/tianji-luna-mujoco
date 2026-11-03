# 框架、模型输入与 CAD 清理核对（2026-11-03）

## 清理原因与范围

此前只清理了主执行入口，仍因旧测试和历史回放保留兼容分支；基础技能注册表也没有和正式控制器一起收敛。这是清理不完整，不能把正式入口不调用当成代码已经删除。

本次以 timing-89-report 为远端基线，对实际入口、注册表、ID 绑定、模型请求、前端、测试、依赖及部署文件逐项核对。个人 writing-clear-code/SKILL.md 增加“替换实现时同步清理”，要求同轮删除替代实现、专属测试/配置/依赖，迁移有效回归，远端传播删除；SDK 与历史实验数据不在删除范围。

## 当前框架

- 一次初始 Responses，输出候选清单、独立任务与 version=2 graph/subgraph/node/edge。
- Runtime 按 edge 执行；Decisions 进行有类型的视觉/语义判断、选择分支与恢复能力，新的 graph patch 校验归档后执行。
- 永久实例 ID、CAD asset ID、graph node ID 分别表示物体、模型、控制节点；每个输入实例独立 subgraph 与完成账本。
- 正式控制器原已过滤四个旧运输技能；本次继续删除它们在基础注册表、RobotOperations 和专属实现中的残留。现在 21 个契约与正式能力一致。
- GaP 借鉴的是技能图与显式控制流形式。本项目 version=2 协议不是官方 version=3 workflow 的直接兼容实现。官方对照：[固定版本 README](https://github.com/graph-robots/graph-as-policy/blob/71549e1584839337eb355c280e7b1e50c805b8c7/README.md)、[grocery packing workflow](https://github.com/graph-robots/graph-as-policy/blob/71549e1584839337eb355c280e7b1e50c805b8c7/examples/grocery_packing/packing_graph/workflow.json)。

## 删除与保留

删除旧 steps ID 绑定分支、前端 steps 解释器、额外 Responses 修订元数据；删除 plan_transport / execute_transport / drop / verify_drop 的契约、执行分派、运输/投放实现及不再调用的 plan_attached_pose 适配方法。保留容器当前几何计算，集中在 robot_skills/container_geometry.py。

删除没有生产调用方的 mesh_queries、release_measurement、Python run_conclusion、task_request，以及 libigl 依赖。前端现行 run-conclusion.js 继续负责结果显示。demo_app 是控制台入口，experiment_reencode 是维护 CLI，均有明确用途；没有把它们误判为孤立模块。

删除释放后深度测量、单相机放置核验及最后物体隐式追加全局清单判断。迁移容器几何、当前 graph、清单与归档的有效测试，删除退役功能专属测试。AGENTS、开发 skill 与当前技能指南改为一致的当前合同，去掉堆叠的历史覆盖指令。

## 模型输入

完整读取 P89 的一次 Responses 和 24 次 Decisions 归档，并再次核对 P90 的实际请求，包含实际 prompt、公共输入、question.instructions 与回答，见 data/framework-audit-20261103/model-inputs.json。图片只保留归档路径，未泄露密钥或 base64。

确认初始 Responses 收到三相机 RGB、原始语言、技能契约和任务绑定。CAD 选择、实例身份、可见性、支撑/目的关系、加工与交付核验分开；工作台配置区域不要求在 RGB 里出现实体标记。已授权整个区域时，XY 数值候选选择可以仅用结构化区域证据，不重复视觉判断。

修复两个实证问题：无参考图的黑色产物不再称为“REFERENCE 中的物体”；图片计数包含各 question.instructions，而非只统计公共 input。每个放置问题均能获得三相机当前 RGB。verify_place 不进行释放后 SAM3/FP/OBB/深度测量，也不隐藏全局完成调用。概率仍表达不确定性，不把 unknown 强行改成 yes。

## CAD

独立仓库递归检查 74 个网格，全部不超过 1000 面：蓝盒主网格及配准网格 1000 面，黑件 999 面。项目副本与独立仓库按实际映射比较 74 个网格，哈希全部一致。清单见 data/framework-audit-20261103/cad-face-counts.json 与 cad-sync.json。

粉色网格属于 MuJoCo 场景效果资产，不在识别 CAD 目录中；感知仍将粉色包装作为未知 OBB。目录中的识别资产为 blue_box 与 black_workpiece。

## 验证

- 后端最终全量：4158 passed / 58 skipped；另有 2 项仍注入旧运输字段的回归失败。将其迁移到当前 place_plan / plan / grasps 失效约束后，所在文件 7 项全部通过；没有改动安全检查。日志为 full-tests.txt 与 rebinding-tests.txt。
- 前端 391 项全部通过，Vite 构建成功；Ruff F 检查通过；两份开发 skill 校验通过。项目 editable 安装成功，依赖元数据已刷新。
- 相对本轮开始快照，后端 Python 净减少 1105 行，web/src 净减少 95 行；专属退役测试同步删除，当前行为测试保留或迁移。统计见 cleanup-diff.json。
- timing-90 同步核对：项目 1078 文件、独立 CAD 112 文件全部哈希一致；62 个删除目标均不存在，其中 39 个为过期网页构建文件。
- cuRobo 官方 SDK 361 文件哈希未变；五容器 CPU 总额仍为 4 核，未修改其他用户服务。
- 完整仿真 P90 成功，run_id=87f7a0eec53b4fcb84420719f20d007b。总计 231.087 秒，Responses 72.271 秒，其余 158.816 秒；一次 Responses、24 次 Decisions，中位耗时 1.414 秒，最大 2.582 秒。两次 verify_place 分别 1.486/1.335 秒，每题均有当前三相机 RGB。
- 本轮粉色输入确为 observed_obb_mesh_surface_samples，A 臂使用 GraspGen；黑件为 black_workpiece / cad_offline_library，B 臂使用离线库。实际技能目录 21 项，无四个退役运输技能。根 graph 为 5 node / 4 edge，独立物体 subgraph 为 39 node / 38 edge，没有 steps。
- 无 skill_error / algorithm_error，交付后观测记录 2 次。本轮未触发恢复 graph patch；一次成功不能证明所有恢复分支或长期成功率。证据见 P90-result.json、P90-model-inputs.json 与 data/performance-unpack-20261101/P90-audit。

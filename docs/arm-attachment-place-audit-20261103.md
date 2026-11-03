# CAD/OBB、选臂、回位与放置耗时核查（2026-11-03）

## 模型表示的来源

当前实验指令是 `data/performance-unpack-20261101/unpack-instruction-observation.txt`，其中明确写了“粉色包裹无CAD，用当前OBB和GraspGen；蓝盒和黑工件用CAD”。初始 Responses 据此给粉色实例输出 `geometry_policy=unknown_obb`，引用原话并绑定永久 ID。`cad_perception.py` 核对原指令引用、实例 ID 和 generation 后才使用这条显式策略。这不是程序从颜色自动推断“粉色一定没有 CAD”。

生产识别目录 `resources/cad_catalog.json` 登记蓝盒和黑色工件。独立 CAD 库中粉色仿真外观文件的存在，不等于它被登记为可识别资产。没有用户明确指定 `unknown_obb` 时，默认走自动 CAD 检索：Decisions 读取当前 SAM3 主体图、RGB、资产描述和参考图，选择 CAD 或明确的 `no_catalog_candidate`；已确认 CAD 再交 FoundationPose 求当前位姿。目录无匹配时使用实测 OBB；证据不确定不能伪装成“无 CAD”。

已确认的 CAD 模型对应可按永久 ID、generation 和源哈希缓存；新的几何观测仍更新当前位姿。FoundationPose 位姿按现有用户策略直接采用，不增加点云/轮廓吻合验证。

## 为什么选左臂或右臂

P85/P86 的 Responses graph 都显式调用 `choose_arm(arm="auto")`。运行时先查询离线工作空间包络；只有一臂通过时直接选该臂，多臂通过时由 Decisions 的 `execution_arm` 问题选择。

- 粉色目标当前位置只有 A（左臂）通过位置包络检查。
- 黑色目标当前位置 A/B 均通过；Decisions 综合当前目标、右侧目的地和双臂工作空间信息选 B。此次右盒的 64 个区域采样中 A 命中 0、B 命中 40；这只是有限采样证据，不是整个区域不可达的数学证明。
- 仿真配置 `resources/vla_unpack.json` 把产物生成区放在右侧，并有 `required_workspace_arm="B"`。这会影响场景分布；不能把实验中黑工件总由右臂取说成完全没有场景先验。

这里还没有对两臂完整抓放轨迹分别求解再比较最优值。选臂之后，仍须通过该臂的 IK、碰撞和完整轨迹检查。运行时没有“粉色→左臂，黑色→右臂”的颜色映射。

## 为什么交付粉色物体后回观测位姿

当前任务指令及完成要求明确包含“每次交付后双臂回配置观测位姿并观测”。P85/P86 的 Responses graph 为两次交付分别生成 `move_to_observe(viewpoint="configured")` 和 `observe_scene`；运行时核对实际双臂关节反馈与完成观测的回执。`plan_place`、`execute_place` 不暗中追加回位。

回位是此任务的完成要求，不是完成一次放置在物理上必然需要的步骤。其他移动仍须在模型 graph 中显式出现；技能内保留其必要的数值算法步骤。

## attachment 的实际瓶颈与修改

P85/P86 的 `prepare_carrying` 算法耗时约 2.69–2.79 秒。其中 `robot_geometry_update` 约 2.5 秒，球数据上传约 0.00013 秒。慢的是闭爪反馈变化后重新解析机器人运动学与自碰撞对，不是把几十个 attachment 球传到 GPU。

旧实现即使只有执行臂手指变化，也重建主机器人、另一臂模型和坐标变换用的参考模型。修改 `ResidentRobotGeometry.update`：

1. 主机器人按实际闭爪反馈更新，碰撞球及自碰撞检查保留。
2. 只有另一臂的夹爪反馈变化时，才更新另一臂模型。
3. 临时参考模型只计算链接坐标变换，因此不构造它不会使用的碰撞对；实际规划器、另一臂和最终检查器仍保留完整碰撞几何。

P87 两次 `prepare_carrying` 为 **0.997 / 0.969 秒**，其中机器人几何更新约 0.88–0.90 秒。仍有一次官方 CPU 解析，不能描述为完全没有 CPU 开销。attachment 仍只在 lift 成功且确认持物后建立，时机、球覆盖和硬净距均未放宽。

第二轮 P88 为 **1.042 / 1.060 秒**。

## plan_place 实际没有走与 plan_grasp 相同的规划器

生产固定落点路径是 `/tools/plan_place`：`resolve_placement` 已经确定唯一 XYZ/姿态。旧的 `compound_planner=True` 标志没有传到 `build_geometry_backend`，导致构造了普通 `MotionPlanner`。这与多抓姿使用 `BatchMotionPlanner` 不同。

现已透传该标志，固定落点采用官方 `BatchMotionPlanner`、batch=1，调用公开 `plan_grasp` 的 goalset→approach→place 阶段。API 名称沿用官方接口，不改变用户技能 `plan_place` 的含义。释放后空夹爪撤离仍单独规划和检查；不改变解析出的 XYZ、姿态、attachment 或碰撞阈值。

完整技能耗时（包括检查和撤离规划，不仅是 GPU 求解）：

| 实验 | 粉色 plan_place | 黑工件 plan_place |
|---|---:|---:|
| P85，修改前 | 6.185 s | 8.070 s |
| P86，修改前 | 6.552 s | 8.369 s |
| P87，修改后 | 4.064 s | 4.667 s |
| P88，修改后 | 5.451 s | 4.356 s |

另有冻结黑工件输入的独立单目标 BatchMotionPlanner 诊断，3 次均通过；热运行服务片段约 1.01–1.04 秒。旧 MotionPlanner 在该独立诊断中末端 FK 检查失败，因此该对照不能作为成功轨迹之间的提速倍率。失败原始记录保存在 `data/performance-place-20261103/original-B-report.json`。

## batch 数量与调用次数的对照

使用 P86 原始抓姿池和场景，通过真实 cuRobo 服务及应用检查，分别测 2、4、8 个候选一批。每组 3 次，首轮单独保留，不计入下表热运行均值；每臂每档热样本数为 2。

| 抓姿批容量 | 左臂热均值 | 右臂热均值 |
|---|---:|---:|
| 2 | 2.626 s | 2.665 s |
| 4 | 4.524 s | 5.636 s |
| 8，旧逻辑补齐到16 | 12.406 s | 17.940 s |
| 8，本轮真实容量8 | 7.078 s | 9.481 s |

这些样本均第一批成功，更大 batch 没有减少调用次数，反而计算、检查了更多候选。生产默认保留 2；失败继续原抓姿池，不削减候选覆盖。服务容量档补齐为 2/4/8/16，修复 5–8 个候选被补到16的问题。唯一已解析放置目标用 batch=1，没有理由复制成多个相同位置。

这是当前两个抓姿池的测量，不能声称 batch=2 对所有场景都最优。测量包含公开 SDK、传输及应用检查；不能与官方纯 IK 的毫秒基准直接比较。

P87 抓取技能仍为 11.265/11.085 秒。诊断前依次创建多档 batch 导致容量2缓存被淘汰；服务日志证实任务中重新构造 A/B 容量2分别花 5.304/5.055秒。后续完整实验在正常启动预热后进行，避免把诊断缓存状态当成正常热运行。

P88 在正常启动预热后，两次完整 `plan_grasp` 技能为 **6.389 / 6.183 秒**；这包含当前任务的预筛查与各项检查，不等于表中冻结抓姿池的热基准。

## 证据与验证范围

- 锁定官方 cuRobo commit：`8e734f3ced1df898990bcd92de40abce475907db`。只修改本项目组合适配器，未修改官方 SDK 或 vendor。
- 新增回归覆盖执行臂与另一臂夹爪变化的区分、复合规划标志传递，以及 2/4/8/16 容量边界。
- 第一轮 P87 完整拆包成功，仅一次 Responses；总计232.020秒，扣除 Responses 为165.676秒。全部25份模型请求及158次图片引用通过原始传输哈希复核。
- 第二轮 P88 完整拆包成功，仅一次 Responses、24次 Decisions；总计239.438秒，扣除 Responses 为156.790秒。全部25份模型请求及153次图片引用通过原始传输哈希复核。两轮均无失败/unknown节点，且每次交付后的双臂配置观测位姿及完成观测回执均通过核对。两次成功不代表已测得长期成功率。
- 最终全量测试 **4539通过、58跳过**，332.82秒；原始输出 `data/performance-place-20261103/full-tests-timing88.txt`。未用跳过的 GPU 测试代替真实服务与物理任务验收。
- 本轮未修改模型 questions/prompt、概率阈值、自然语言目标、工作空间包络或 CAD 数据。
- 所有真实服务诊断原始数据保存在 `data/performance-place-20261103/`；完整任务归档在 `data/performance-unpack-20261101/P87-audit/` 等目录。

# plan_grasp_batch GPU 性能诊断：2026-09-30

## 结论

当前十几秒延迟的主要计算负载是轨迹优化中的碰撞检查。一次预热后的 Nsight Systems 记录中，自碰撞占累计 GPU kernel 时间的 53.2%，机器人对场景的扫掠碰撞占 30.0%，普通场景碰撞另占 1.8%。初始化、内部 IK、JEV、HTTP 和 Hermite 不能解释这次已预热请求的主要延迟。

已经用单变量 A/B/A 对照确认一个适配问题：JEV 实际加载 4 层 ESDF，却沿用 base 的容量估算分配 8 层，空层增加 GPU 工作。只去掉空槽位后，扫掠场景碰撞累计 GPU 时间从 3.638 s 降到 2.212 s；恢复原配置后回到 3.591 s。全部测试均为 16/16 成功。

初始诊断时这项优化尚未部署；用户随后授权按官方接口实施，部署结果见本文末尾。它不能消除占比最大的 self collision，也不能据此承诺实时性能。共享 GPU 负载变化明显，不应把某一次 wall time 的差值当作稳定的整体加速比例。

## 实验范围与边界

- 记录来源：`f1b9dc84546a4bb5a6e480def57b386e`，网页实际 batch 14.227642 s。
- 输入日志只保留前三个完整 A 臂目标变换；本次把这三个真实目标循环重复到 16 项。因此是代表性诊断 batch，不是原始 16 个不同目标的完整回放。
- 保持原场景、起始双臂关节、目标排除规则、1362 个原始碰撞球、10 mm hard clearance、LM 32 / IK 16 / TrajOpt 2，以及全部运动约束。
- 设置 revision：`1188f72ea6ff234d`；Lift 为 tool Z -0.1 m。
- 使用临时独立容器，镜像仍为现有 `jodell-curobo:0.2.0-guangyu`，项目只读挂载。没有调用生产服务的写接口，没有执行机器人运动。
- 未修改生产源码、skills、NVIDIA SDK、机器人模型或算法镜像；未重启生产容器。临时容器退出自动删除。
- GPU：RTX PRO 6000 Blackwell Workstation Edition；driver 580.173.02，PyTorch 2.11.0+cu128，CUDA 12.8。
- SDK commit：`8e734f3ced1df898990bcd92de40abce475907db`。
- Nsight Systems 2026.1.3，CUDA profiler capture range，CUDA graph node tracing；Python 3.12 sys.monitoring 标注 NVTX，不替换 SDK 函数。

## 直接测得的 GPU 开销

下表为 kernel 累计时间，不是端到端 wall time。比例只描述该次 profile 的 GPU 工作分布。

| 类别 | 原配置 A / s | 原配置占比 | 仅压缩空 ESDF 槽 B / s | 恢复原配置 A / s |
| --- | ---: | ---: | ---: | ---: |
| self collision | 6.439695 | 53.17% | 6.617246 | 6.447595 |
| swept world collision | 3.638374 | 30.04% | 2.211680 | 3.590533 |
| 普通 world collision | 0.219403 | 1.81% | 0.139283 | 0.199928 |
| kinematics | 0.411575 | 3.40% | 0.413656 | 0.378719 |
| dynamics / RNEA | 0.339968 | 2.81% | 0.347536 | 0.325066 |
| 其余 kernel | 1.062350 | 8.77% | 1.097771 | 1.046105 |
| 合计 | 12.111365 | 100% | 10.827171 | 11.987946 |

原配置最重的 self-collision kernel 主要形状为 grid 41472、block 512，共 808 次，累计 5.566 s。对应的计算维度是 16 个目标 × 2 个 TrajOpt seeds × 4 个 line-search 试探 × 81 个 rollout 采样点，再乘每个状态的自碰撞检查。4 个 line-search 试探不是 4 个新的抓取目标。

主要 voxel swept kernel 也执行 808 次。原配置 grid 882576、block 128；压缩空槽后 grid 441288、block 128，线程规模恰好减半。全部真实 ESDF 层，包括被禁用的目标层，都保留，未删障碍物或改变距离数据。

## 为什么原模型的 self collision 很重

原始模型共 1362 个球、38 个 collision links。冻结对臂后的 A 模型有 507540 对检查，B 模型有 508106 对检查。球数不是检查次数，self collision 的负载取决于未被 ignore 的球对数。

A 模型中：

- 运动—运动球对：19712。
- 运动—静止球对：360074。
- 静止—静止球对：127754，占 25.2%。

静止球对目前也会随候选、轨迹采样点与优化迭代重复计算。此项是潜在优化对象，但不能直接删除：必须先证明当前冻结姿态下不碰撞，并保证规划期间保持不动。当前最终检查器继承 planner 的 ignore 配置，若调整 ignore，还必须保留完整原模型的独立校验。

四份模型已逐字节核对：本地 base、本地 JEV、Ubuntu base、Ubuntu JEV 的 YAML SHA256 都是 `d7e627475865271a66bc785258dd59177080eb3a6e106cce26cb4465971421ae`。球、半径 buffer、ignore、夹爪锁定配置一致。相关冻结对臂、ignore、碰撞更新和 hard checker 函数也做了 AST 对照。未发现 JEV 适配丢失 ignore 或额外增加碰撞球。

这只能证明模型负载一致；没有运行 base 的同输入基准，因此不声称 base 的端到端耗时必然相同。

## ESDF 缓存的具体适配问题

`vendor/tianji_curobo/curobo_scene.py:collision_cache_for_scene` 沿用 base 的约定：voxel metadata 列出物体，另外还要为背景保留一层，容量按 `1 << len(voxels).bit_length()` 分配。

JEV 的 `GeometryRuntime.build_geometry` 已经把桌面放进 metadata。当前 metadata 是桌面 + 三个物体，共 4 层，再套上述公式会分配 8 层。

诊断子类仅把 `collision_cache['voxel']['layers']` 设为实际 `len(geometry_grids)=4`，保留 mesh=1、cuboid=5、voxel dims=[2.4, 2.4, 2.0]、voxel_size=0.01。没有修改 SDK 或生产代码。

官方 swept kernel 的 launch 规模含 `data.max_n`，它是容量；未启用的槽位也会分配线程，线程判断后返回。因此空容量不是完全免费的。

正式修正应位于 JEV 的场景适配层，并正确保留场景增长时扩容、缓冲区更新和 planner 重用逻辑；不能只为这个四层场景硬编码 4。

## 三段抓取为什么有四次规划

固定版本的官方 `BatchMotionPlanner.plan_grasp` 顺序为：

1. 原始起点 → grasp goalset：完整 IK + TrajOpt，取得目标可达性和 goalset index。
2. 原始起点 → pregrasp：生成 approach 轨迹。
3. approach 终点 → grasp：带直线约束，生成 grasp 轨迹。
4. grasp 终点 → lift：带 lift 约束，生成 lift 轨迹。

第 1 条轨迹不执行，也没有作为后三段的 seed 复用。当前每个 batch 项的 goalset=1，因此第 1 次没有候选内选择作用，但仍增加“原始起点到 grasp 独立规划成功”的预筛门槛。它不是 GraspGen score 排序，也不是单独的 Batched IK。

当前每次 plan_pose 有初始优化和一次 finetune，所以函数级计时记录 4 次 TrajOpt solve、8 次 LBFGS optimize。前一轮同输入诊断中 TrajOpt 累计约 13.1 s，内部 IK 四次累计约 0.04 s，最终原始轨迹复核约 0.11–0.17 s。没有 Hermite 后碰撞复核。

官方没有只跳过第 1 步的公开参数。应用层可以研究用公开 batch plan_pose 组合三段，但这属于策略变化，必须审核，不能把它描述成原生 plan_grasp 的普通参数调整。它还会改变随机数消耗顺序与最终成功判定，不能保证新旧候选结果完全一致。

## Wall time 与共享 GPU 的限制

| 变体 | cold / s | warm before / s | traced warm / s | warm after / s |
| --- | ---: | ---: | ---: | ---: |
| 原配置 A | 41.672 | 12.429 | 12.599 | 13.536 |
| 仅去空槽 B | 39.330 | 10.731 | 11.251 | 9.269 |
| 恢复原配置 A | 41.009 | 12.904 | 12.477 | 8.675 |

每一项均 16/16 成功。A/B/A 的 kernel 结果支持空容量造成额外计算；但原配置的无 trace warm time 也在 8.675–13.536 s 波动，不能据此给出固定端到端提速百分比。

实验中同卡存在其他用户的 Python、SAM3、Isaac Sim 等进程；本诊断退出后 GPU 利用率仍约 45%。没有停止这些任务。尚未获得独占 GPU 的稳定延迟分布，因此没有量化共享负载究竟增加多少时间。

另做了一次开启官方公开 torch_jit / torch_compile 开关的独立实验，在 cold 阶段因显存不足失败：申请 666 MiB 时仅剩约 171 MiB。它没有产生 warmed 数据，不能据此判断 compile 的性能。上述主要碰撞计算是自定义 CUDA / Warp kernel，开启外围 PyTorch compile 也不能直接消除这些几何计算。

## 已确认的改进顺序与边界

1. 首先修正 JEV 的 ESDF 缓存估算；这是已用 A/B/A 确认、可以保留全部碰撞语义的适配问题。还需覆盖场景增删、Reset、缓存扩容与连续请求，不能直接部署本次诊断子类。
2. 若要更大幅度降低延迟，需审核四次完整规划中的预筛，以及静止—静止球对的重复计算。这些影响规划或验证策略，尚未修改。
3. 当前不足 16 候选也会补齐到 16，是另一项已确认的额外负载；它不解释本次本来就有 16 项的主要开销。调整 batch 大小会改变已审批的执行策略，不能直接换掉。
4. 不以减少原碰撞球、放大球包络、降低 10 mm clearance、关闭 self collision、减少 sweep 覆盖或任意降低 seeds 来宣称解决。

## 证据与源码

- 汇总和逐 kernel 数据：`data/planning-kernel-audit-2026-09-30.json`。
- 前一轮函数级数据：`data/planning-speed-monitored.json`。
- 原始 Nsight reports、CSV、日志与 probe：Ubuntu `/tmp/jev-kernel-audit-guangyu-40ixvrwu`。
- 官方抓取流程：https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/motion/motion_planner_batch.py
- 官方 swept launch：https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/geom/collision/wp_autograd.py
- 官方 collision storage：https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/geom/data/data_scene.py

结束检查：网页仍是同一 run_id、succeeded 状态和设置 revision。性能原因已进一步定位到 kernel，并确认了一个可消除的适配开销；实时性能尚未实现。

## 用户授权后的实施与部署

2026-09-30，按“尽量按照 cuRobo 官方操作”的要求，只在项目 `GeometryRuntime` 修正容量估算，通过现有官方 `MotionPlannerCfg.create(collision_cache=...)` 和 IK config 传入。原生 `plan_grasp`、球模型、self collision ignore、seeds、10 mm hard clearance、目标排除及三段原始轨迹复核全部保留。没有修改 SDK、vendor、算法镜像或 skills。

默认 4 层实际 ESDF 使用 4 层缓存；5 层扩为 8 层；缩小场景时复用已有容量和 planner。扩容会清理所有相关 solver 引用和快照，避免继续使用旧 CUDA graph；同一个 backend 只销毁一次，单项清理失败仍清理其他项并报告错误。

验证记录：

- 先运行新增回归测试，4 个测试都在旧代码上按预期失败；修正后相关模块 34 个测试通过。
- Windows 全量复跑：`python -X utf8 -m pytest -q -ra`，229 passed、51 skipped。跳过项均为 Linux Bash / 容器进程测试，本轮没有改部署脚本。
- 首次全量运行中，`test_planning_timeout_is_terminal_and_late_result_cannot_move_robot` 的 50 ms 计时断言记录 42.3 ms 而失败；单独及全量复跑均通过。未修改该测试或 runner，此偶发计时问题未在本轮修复。
- `ruff check src/jev_mujoco/geometry_service.py tests/test_geometry_service.py` 通过。
- 独立 GPU 测试 4 层 → 5 层 → 恢复 4 层全部 16/16 成功；实际观察到容量 4 → 8 → 8，并验证扩容重建、缩小后复用同一 backend。
- 算法镜像没有 pytest，因此未安装依赖或修改镜像。按固定版本官方 ESDF 测试的构建方式，用公开 API 运行 standalone smoke，额外断言实际规划成功，验证通过。

在本轮较低 GPU 竞争条件下，相邻旧/新代码测试的预热 wall time 为：

| 配置 | warm before / s | 监测函数调用的 warm / s | warm after / s |
| --- | ---: | ---: | ---: |
| 修正前，8 层容量 | 7.855 | 7.931 | 8.221 |
| 修正后，4 层容量 | 6.929 | 6.987 | 7.141 |

同组两次普通 warm 的平均耗时降低约 1.00 s（约 12.5%）。样本少且服务器仍共享，这不是稳定性能保证；不能把此前 12–14 s 到现在约 7 s 的全部差值算成代码收益。仍使用原先三个真实候选循环组成 16 项，未执行机器人运动。

源文件与测试已同步到 Ubuntu。重新加载 `jev-mujoco-curobo-guangyu` 后，实际 HTTP 服务验证 cold 30.867 s、warm 7.004 s，均 16/16 成功，返回完整 approach/grasp/lift。首次构建 planner 的冷启动仍明显更慢。

三容器镜像 ID 未改变，网页与 GraspGen 容器启动时间未改变；仅 JEV cuRobo 容器重启。源码同步触发网页后端自动重载，内存场景已重置，网页状态为 idle；保存的用户参数 revision 仍为 `1188f72ea6ff234d`。历史运行文件保留。

静止球对去重需要新增专用 checker 及等价性验证；跳过 goalset 预规划没有官方开关。按本轮官方流程优先的边界，这两项保留原行为，主要的 self collision 瓶颈仍然存在。

实施证据：`data/planning-cache-fix-2026-09-30.json`；临时探针和日志在 Ubuntu `/tmp/jev-cache-fix-guangyu-O4rdCRVW`。

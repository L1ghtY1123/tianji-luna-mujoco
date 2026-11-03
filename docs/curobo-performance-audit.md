# cuRobo 性能与部署方式核查（2026-10-26）

当前生产状态（run09–15）：默认4个TrajOpt种子、开放夹爪规划保留抓取目标、10mm完整命令检查；启动预热11个profile，batch16按需构建。run11已实际完成一次源盒到桌面抓放，三个抓取批次6.900/6.219/4.994s，整轮110.134s且Responses1次。下文2.6–3.1s及16候选9.305s属于此前固定2种子、不同目标碰撞范围的容量单变量对照，不能当成当前生产完整抓取性能或完整拆包成功。VLA全流程仍未通过，稳定性未达标。

容量审计结论：已定位并修复碰撞缓存过度预分配。当时真实地图与同四候选，保持默认 16/2 seeds、CPU1、全部障碍及碰撞判据，仅将 mesh/cuboid/voxel 槽从 **128/128/8** 缩为 **2/5/4**，隔离纯规划热态公开 `plan_grasp` 从 **13.733–14.629 秒降至 2.642–3.071 秒**，约 4.5–5.5 倍。两配置所有三次调用均为 4/4 SDK 成功。19:18 香港时间已部署常驻服务，同一实测输入 HTTP 三次 **3.133 / 2.545 / 3.027 秒**，均 4/4 SDK 成功；这仍是纯规划，完整运动任务另行验收。SDK、净距、seeds、项目 batch4 和 CPU 配额均不变。

## 常驻 HTTP 复验（19:18 部署后）

上述缓存性能对照固定使用当时的 16/2 IK/TrajOpt seeds。后续真实下探干涉核查增加了开放手爪对目标的碰撞要求，并将项目 TrajOpt 默认改为 4，显式保存的用户配置仍保留。相同首批 `[0,38,48,74]`、固定 random_seed=123、两个独立冷进程的第一次规划调用：保留目标时 2 seeds 为 9.310 秒，没有通过最终 10mm 检查的候选；4 seeds 为 11.251 秒，候选 48 的 native 与 Hermite 接近/下探均通过完整检查。该时长含构造/预热，不是热态性能对照，也不能与上面的目标排除请求混称同一输入。记录在 `data/gripper-run07-audit/first-batch-target-{2,4}-report.json`，还需真实运动验收。

13 个 profile 启动预热 63.264 秒。通过当前网页所用的 8352 服务发布历史实测 f580 地图并请求相同四候选，三次 SDK 时间分别为 2.626 / 2.509 / 2.994 秒；没有 backend 构建。首次上传地图 0.428 秒，后两次复用；响应明确记录实际容量 2/5/4。测试没有执行运动。源输入 SHA256 与隔离对照相同，完整结果为 `data/acceptance-repair-live-performance.json`，脚本 `data/verify_live_compact_capacity.py`。

19:41 HKT 对此前 batch16 比较中的同一 16 个目标再次调用该常驻服务：SDK 从 65.870 秒降至 **9.305 秒**（约 7.1 倍），HTTP 总时间 **9.572 秒**，16/16 成功；包含 0.118 秒地图上传、无后端构建。该诊断没有改变生产默认 batch4，没有执行运动。原始四批来源 event62/65/68/71，结果 `data/curobo-compact-batch16-comparison.json`，完整响应 `data/curobo-compact-batch16-response.json`，脚本 `data/compare_compact_curobo_batch16.py`。保留旧结果以供同输入核对。

## 根因、容量对照与本轮修复

生产 `warm_start()` 原先强制 128 mesh、128 cuboid、8 voxel；此前较快的隔离 runtime 没有调用这段预热，缓存按实际 scene 分配。锁定 SDK 的 `_src/geom/collision/wp_sweep_collision_kernel.py` 与 `wp_collision_kernel.py` 明确采用 `num_spheres × max_n_obstacles` 二维并行，预留空槽也增加 GPU launch 的工作域。因此大缓存不仅占显存，还让每轮碰撞优化付费；此前仅看 `effective_config` 的 seeds 等字段相同，漏掉了这个关键配置差异。

复现输入为 f580 的 event41 实测建图和 event62 首四个抓姿，保留 1 cuboid、2 CAD mesh、3 measured voxel 层。离线编码、独立 Python runtime 在已有 cuRobo 容器中串行运行，始终 `cpu.max=100000 100000`，不改服务地图、不重启、不执行运动。容量实验只替换项目 runtime 的已准备 cache；未改 SDK 实例、算法或场景内容。

| mesh/cuboid/voxel slots | 首次 SDK 秒 | 热态 SDK 秒 | profiler 热态 SDK 秒 | 每次 SDK 成功 |
|---|---:|---:|---:|---:|
| 128/128/8 | 17.980 | 14.629 | 13.733 | 4/4 |
| 4/8/8 | 6.865 | 3.425 | 3.826 | 4/4 |
| 2/5/4 | 6.171 | 2.642 | 3.071 | 4/4 |

2/5/4 比 4/8/8 再省约 20–23%；选取通用的小容量下限，不按物体身份、臂或任务硬编码。SDK 成功不是覆盖证书或可执行成功。此前 Run2 的同输入容量对照也得到 128/128/8 热态 12.902/12.406 秒、4/5/8 为 2.956/3.262 秒，热态均4/4。

`geometry_collision_cache()` 继续调用现有公开配置入口的容量计算，保留 voxel 每层尺寸与分辨率校验；mesh/cuboid/voxel 数量按实际数量向上分桶。启动2/5/4，实际增长超桶时先销毁旧图并重建，所有障碍完整上传；容量变小时仍保留已构建后端。4→5 voxel 将一次扩至8，随后5→8无需重建。不存在截断障碍、禁用碰撞或复用旧尺寸 CUDA buffer。响应 `effective_config.collision_cache_capacity` 暴露三类实际容量，方便未来诊断。对于真实大场景，增长后的容量成本仍存在；本轮没有复杂的多容量 LRU。

cProfile 的128槽热态有48次 CUDA event同步，共11.729秒；这代表等待 GPU，不是网络或纯CPU计算。本轮没有重新做逐kernel GPU trace，不把全部11.729秒都归给某一kernel；容量单变量对照和锁定源码共同支持上述根因。更少 kernel 空槽是本次速度收益，不能把允许更多显存误解为应盲目加大碰撞数组。

逐进程 allocated/reserved 与全设备 free 快照见 `data/curobo-capacity-f580-{128,4-8-8,2-5-4}.json`。全设备量同时受其他进程影响，reserved 也不是实测峰值；本次只构建一个 batch4 后端，不能外推到全部13个常驻profile。原始日志与脚本为 `data/curobo-capacity-*.log`、`data/profile_collision_capacity.py`、`data/capture_f580_performance_replay.py`。下面保留本修复前的生产与历史隔离证据，避免把旧性能数字当部署后验证。

| 容量 | 本进程 allocated bytes | 本进程 reserved bytes | 全设备 free 前→后 bytes |
|---|---:|---:|---:|
|128/128/8|1,513,428,992|7,981,760,512|8,646,295,552→7,723,548,672|
|4/8/8|1,513,418,240|7,981,760,512|8,950,382,592→7,503,347,712|
|2/5/4|1,421,143,552|7,889,485,824|7,991,984,128→7,595,622,400|

## 部署后真实运行与受控纯规划（追加核查）

真实运行 `f580821fdd4a43deaeb919d19c308840` 由 Decisions 选择 global 建图，共 30 个抓姿、8 批（前七批各 4，末批 2，后端容量仍为 4）。八个 `planning_batch_complete` 合计 **144.028 秒**；其中 `public_plan_grasp` 合计 **136.489 秒**，占 94.8%。SDK 返回 30/30 成功，最终命令仍因运动覆盖不足被拒绝，没有电机执行。这既不是完整任务成功，也不是 30 条轨迹已证明安全。

| 批次事件 | 实际候选 | 公开 plan_grasp 秒 | operation 秒 | HTTP 往返秒 |
|---|---:|---:|---:|---:|
| 62 | 4 | 19.344 | 19.508 | 19.515 |
| 65 | 4 | 18.378 | 18.408 | 18.417 |
| 68 | 4 | 14.512 | 14.541 | 14.549 |
| 71 | 4 | 13.936 | 13.966 | 13.972 |
| 74 | 4 | 13.405 | 13.433 | 13.440 |
| 77 | 4 | 12.185 | 12.213 | 12.220 |
| 80 | 4 | 23.820 | 23.854 | 23.861 |
| 83 | 2 | 20.909 | 20.984 | 20.990 |

operation lock 排队仅 1.2–2.5 微秒；没有 backend 构建。首批场景上传 124 毫秒，此后没有上传；热态对臂更新约 3.2–6.7 毫秒。HTTP 往返比 operation 多约 6–9 毫秒，不能用网络或缓存构建解释这十几秒。末批两候选仍按四槽后端补齐，因此不应按二分之一工作量理解。响应旧字段 `planning_time_s=0.0` 没有实际求解时间信息，以新增 wall timing 为准。

新旧 `effective_config` **逐字段完全相同**：同机器人配置 SHA、16/2 seeds、CUDA graph、自碰撞、10mm 硬净距、30mm activation、同四槽后端。起始 A/B 关节完全相同；抬升 offset 只差约 0.18 微米。世界均为 1 cuboid、2 mesh、3 voxel，三个 grid cells 仍为 41,328 / 44,280 / 14,976，合计 100,584。当前库存 voxel tensor 报告 201,168 bytes；历史报告的 0.384 MiB 是按 float32 估算，不能混作当前真实 tensor 字节。相同规模不代表两次测量的占据值、目标姿态或 SDK 随机状态完全相同。

运行结束后，在保持当前已发布地图、配置、起始关节及热服务不变的条件下，仅替换目标抓姿，进行了授权的两次纯规划：

| 当前地图下的抓姿 | 公开 SDK 秒 | HTTP 秒 | SDK 成功 | cgroup throttle 增量秒 |
|---|---:|---:|---:|---:|
| 旧 Run2 首四抓姿 | 15.981 | 16.232 | 4/4 | 0.296 |
| 新运行首四抓姿 | 17.567 | 17.654 | 4/4 | 0.361 |

这不是旧完整场景的重放，而是**当前同一地图上的抓姿对照**。旧抓姿在当前条件下同样慢，不能把历史约六倍差异归因于新抓姿。设备利用率在四次调用前后快照为 100/100/100/98%，全设备空闲显存约 16,331→16,323 MiB；在此前本项目 cuRobo 已空闲时，整设备也有 97% 利用率。因此共享 GPU 负载是明确存在的混杂因素，但没有逐 kernel 或进程 GPU 时间归因，尚不能定量判定它解释了多少差异。CPU usage 接近墙钟时间，throttle 约 2%；CUDA 同步可能忙等，不能据此断言纯 CPU 算术瓶颈。随后 batch16 前设备已空闲，也说明负载随时间变化，不能把全部慢请求直接归咎共享 GPU。

### batch4 与 batch16 的实测边界

batch4 来自项目 `DemoAlgorithms.grasp_trajectory_batch_capacity=4`，基类容量为 16；`geometry_service` 对超过四个目标自动选择已有的十六槽后端，启动已预热 4/16。公开 `BatchMotionPlanner` 支持 `MotionPlannerCfg.create(max_batch_size=16)`，不是 SDK 只支持四个候选。锁定源码的 `plan_grasp` 依次执行 goalset、approach、grasp、lift 四次内部 `plan_pose`；每阶段都求解完整 B，graph seed 查询展开为 B×seeds。增大批量减少外部调用，但不消除这些求解工作。

另一次授权纯规划将新运行前四批的 **同十六个真实目标**合为 batch16，保持当前地图、起点、16/2 seeds 与全部成功判据。结果 SDK **65.870 秒**、HTTP **66.200 秒**、16/16 SDK 成功，无 backend 构建；首次给该已预热 profile 上传当前 scene 为 121 毫秒。相同十六目标在真实运行中分四批的 SDK 总时间为 **66.169 秒**：分别约 0.243/0.242 候选每秒，名义差仅约 0.45%，且负载时间不同，不能认定提速。单批等待却从十几秒增长至约一分钟。

batch16 前后全设备空闲显存为 16,334→15,564 MiB，快照减少约 770 MiB；进程 PyTorch reserved、瞬时峰值没有从现服务取到，**不能把这 770 MiB 当成进程峰值**。CPU usage 增加 65.812 秒，throttle 增加 2.049 秒，配额始终 1 core。没有 OOM，但仅凭仍有显存不能推断更大批量会加速。因此本轮不把生产 batch4 改为 16，也不改 seeds。下一项有判别力的诊断是受控负载下的 SDK 内部 graph/IK/TrajOpt 与 CUDA 等待 profile；目前不能安全承诺某项参数修改能解决全部耗时。

证据：`data/curobo-live-f580821-performance.json`、`data/curobo-live-controlled-comparison.json`、`data/curobo-live-batch16-comparison.json`；原始事件/响应及两份探针脚本保存在同目录 `curobo-live-*`、`compare_live_curobo_*.py`。追加共三次纯规划，没有发布地图、更换配置、重启服务或执行运动。窗口完成后已交还根代理开展下一实验。

## 范围与证据

- 官方 SDK 锁定 `8e734f3ced1df898990bcd92de40abce475907db`，隔离进程启动调用现有 `sdk_identity()` 核对来源。
- 复用真实 unpack Run2 `a35ec939b72c40a1939d95f8418bc274` 的事件 38（RGBD 建图）和 59（首批完整抓取请求/响应）。只读恢复归档数组，重新编码原始测量数据；没有读取仿真对象真值。
- 在现有 `tianji-luna-curobo-guangyu` 容器中运行额外、有限的 Python 进程，仍受 `cpu.max = 100000 100000` 限制。未改变正在服务的 runtime、未重启、未下发运动。
- GPU：RTX PRO 6000 Blackwell Workstation Edition。最近一次比较前全设备可用 30,641,684,480 bytes，比较后 20,426,784,768 bytes；本进程最后 allocated 2,459,898,880 / reserved 9,221,177,344 bytes。设备可用量也会受其他既有进程影响，不是本项目独占显存。
- 完整结构化结果：`data/curobo-performance-audit.json`；原始记录：`data/curobo-performance-seeds-output.txt`。另有第一次原始记录 `data/curobo-performance-replay-output.txt`。探针源码为 `data/profile_curobo_performance_audit.py`、`data/capture_performance_replay.py`，导出的缓存方法快照为 `data/performance-peer-method.py`。

## 实测拆分

同一批 4 个真实候选，默认 16/2。最近一次进程结果如下，单位秒：

| 场景 | 全部请求内计算 | backend 构建 | 场景上传/准备 | 对臂更新 | 公开 plan_grasp |
|---|---:|---:|---:|---:|---:|
| 新建隔离 runtime 首次 | 9.686 | 3.473 | 0.295 | 0.0415 | 5.855 |
| 已使用后端再次规划 | 2.458 | 0 | 0.000027 | 0.00259 | 2.428 |

前两次相同输入实验的热态为 3.91–4.13 秒，说明单次时间会受机器/GPU 当前负载和正常求解随机性影响。冷态曾为 15.08–15.33 秒，其中构建 6.52–7.03 秒、公开 `plan_grasp` 7.88–8.14 秒。

这些冷态是刻意新建、未调用 `warm_start()` 的隔离 runtime，不能解释为生产服务漏预热。现服务已预热 A/B 的 IK100、batch4/16、普通 MotionPlanner、compound grasp/place，以及 AB 和两臂位置 workspace，共 13 个 profile。相同 scene/backend 的重复上传已被现有缓存消除；本次热态没有重复构建。设置、拓扑或地图缓存容量确实变化时仍须重建。

默认热态额外 cProfile 记录 2.903 秒，其中 48 次 CUDA event synchronize 合计 2.368 秒，PRM 的 `torch._unique2` 约 0.211 秒。事件同步是主机等待 GPU 完成的时间，不能等同 CPU 算术时间，也不是网络等待；本次没有做逐 kernel GPU trace，不能再把这 2.368 秒精确分配到每一种碰撞 kernel。

机器人有 595 个 sphere slots，其中 533 个启用，62 个为两臂预留且禁用的 attachment slots。世界为 1 cuboid、2 mesh、3 voxel grids，分别 44,280 / 41,328 / 14,976 cells，总计 100,584 cells（单 float32 距离值约 0.384 MiB，不含 GPU cache、梯度或 solver workspace）。采用实际双臂碰撞几何与对臂约束，不能与不同机器人的单目标官方基准直接比较。

同一条历史最终命令的独立校验：首次 0.846 秒，热态 0.105 秒，均检查 517 个样本并拒绝未知空间。5 次覆盖证书检查分别合计 3.88 / 2.05 毫秒。因此该案例的首次 checker 创建/碰撞管线比覆盖证书更重，重复请求确实复用了 checker；没有发现每个 grasp batch 都无故重建。先前一次首次校验为 1.70 秒，同样 coverage 仅 3.84 毫秒。这里没有逐项拆出 checker 构建、YAML 解析、FK 与硬碰撞各自成本。

旧 Run2 的 14 个 `planning_batch_complete` 合计 190.419 秒，事件时长包括规划、端点检查和最终命令校验，不能当作 190 秒纯 GPU 求解。历史 100 点 IK 档案 `data/ik-frozen-profile.json` 为 solve_time_s 14.9–15.3 毫秒、整调用 91.9–361.4 毫秒；这是历史数据，并非本次重测。旧 mesh ablation 仅作诊断，禁用碰撞几何的速度不能当安全性能结果。

## 同输入公开参数比较

保持 measured scene、候选、硬净距 10mm、soft activation 30mm、自碰撞和最终校验代码；仅改变公开 IK / TrajOpt seed 数。正常随机种子状态随调用推进。

| IK / TrajOpt seeds | 热态全部调用（秒） | 每次 SDK 完成候选 |
|---|---:|---:|
| 16 / 2，生产默认 | 2.46–2.59 | 4 / 4 |
| 8 / 1，诊断配置 | 1.46–1.51 | 4 / 4 |
| 32 / 4，诊断配置 | 5.83–6.09 | 4 / 4 |

8/1 的结果说明减少搜索计算量能改善这个样本的延迟，不证明更难场景的成功率；生产不改成 8/1。SDK 成功也不表示该命令能执行：本次实测命令仍因覆盖不足被拒绝。

锁定源码 `MotionPlannerCfg.create` 的 batch 默认就是 16/2，single 默认是 32/4；本项目没有把 batch 默认 seed 意外放大。更多显存可保留兼容的 warmed solver，但不会自动减少求解工作。32/4 更慢；扩大 GPU/RAM 或 planner LRU 没有证据能解决本次主瓶颈。CPU 配额始终为 1 core；比较中各请求 cpu usage 接近 wall time，但 CUDA 同步可能忙等，不能据此断言纯 CPU 瓶颈；记录了每次 cgroup throttle 增量，没有改变 CPU 配额。

## 本次代码改进及边界

`GeometryRuntime._refresh_peer_collision_unlocked` 原先每次都重新做对臂 FK、逐 link sphere 上传并重置图缓存。现在仅在同一后端、完全一致的 active/peer joints、同一 resident geometry、且调用允许缓存时复用；模型更新、场景重置、换后端、失败或临时 `cache_snapshot=False` 均失效。缓存最多保留每臂一个后端引用。当前 attachment 上传始终执行，不因对臂命中而跳过。

微基准 30 次固定快照：第一次实测原始总 0.775 秒、中位 21.99 毫秒；缓存总 0.0379 秒、命中中位 4.47 微秒。完整记录中的另一轮有相同量级收益。这是重复 peer upload 的局部改善，不能称为整个任务加速。

**复合 grasp/place 的 pinned vendor 调用显式要求 `cache_snapshot=False`，本次保留该语义，因此此缓存不加速它们。** 普通允许缓存的 pose/linear 路径可以受益。隔离探针的历史 `cached_warm` 行名仅表示启用了候选实现，行内仍有 `peer_collision_update` 就代表未命中；不可把默认 warm 波动归功于缓存。导出方法快照使用 backend-id cache slot；最终代码改成有界的每臂 slot，复用条件相同且换 backend 强制更新；最终差别经过本地负回归，未重新声称 GPU 端到端加速。

新增 `planning_timing.py` 用 ContextVar 按请求收集 inclusive wall timing，不同步 CUDA、不更改 SDK 对象或内部调用。服务记录请求读入/JSON 解码、全局 operation lock 排队、operation 总时间、后端创建/预热、场景/模型/attachment 上传、公开 `plan_pose` / `plan_grasp`、IK+硬检查+FK、最终命令检查和 coverage。公开 grasp 内部的 plan_pose 不被拦截，也不伪造内部阶段计时。客户端单列建立连接/编码/HTTP 往返与响应 JSON 解码；响应编码时间经 `Server-Timing` header 传出。嵌套 inclusive 数字不能直接相加。异常路径也保留已收集阶段并日志记录总 wall time。

新增字段只承载观测，不用于规划、通过判据或重试。现有失败仍失败，不删除碰撞或覆盖检查。HTTP 往返减服务时间至多给出编码、传输、调度等混合残差，不能称为纯网络时间。

## 官方 host / cloud 判断

cuRobo 官方提供安装在 NVIDIA GPU 主机上的 CUDA/PyTorch/Warp 库；其教程是本机安装包并构造 Python planner，公开 `MotionPlanner` / `BatchMotionPlanner` 接口没有托管云 host 参数。[官方安装](https://nvlabs.github.io/curobo/latest/getting-started/installation.html)、[公开 MotionPlanner](https://nvlabs.github.io/curobo/latest/api/curobo.motion_planner.html)、[公开 BatchMotionPlanner](https://nvlabs.github.io/curobo/latest/api/curobo.batch_motion_planner.html)。

应用可以自己把 planner 包装为 RPC/HTTP 服务，部署到另一台 GPU 主机或云虚拟机。本项目就是这种自托管方式：web 使用 `LUNA_CUROBO_URL=http://127.0.0.1:8352`，映射到自己 cuRobo 容器的 8310；8350 是 web UI。NVlabs 的 ENPIRE 也明确说明用 Portal RPC 连接自行启动的 cuRobo server（8611），这不是 NVIDIA 托管推理端点。[NVlabs ENPIRE 部署文档](https://github.com/NVlabs/ENPIRE/blob/main/enpire/env/docs/CUROBO_SETUP.md)。

截至本次核查，在上述官方 cuRobo 安装、公开 API、仓库及 NVIDIA 文档搜索中，**未找到 cuRobo 官方 managed cloud endpoint、服务认证或托管请求契约**；不能把自建 host 地址称作官方云服务。NVIDIA 有单独的 cuOpt managed service，但它不是 cuRobo。[cuOpt 官方 managed service 文档](https://docs.nvidia.com/cuopt/service/latest/quickstart.html)。

官方最新基准在 RTX 6000 Ada、指定 2600 个问题上列出平均 plan time 0.038 秒，并另列 solver time；这些条件与本项目双臂几何、真实 TSDF/ESDF、四候选、多阶段 grasp、硬检查和 CPU 限额不同，不能承诺本项目也达该数。[官方基准与 profiling](https://nvlabs.github.io/curobo/latest/reference/benchmarks.html)。网页为 latest，实际适配仍以锁定提交源码为准。

## 验证与未测部分

本地针对性能边界、几何服务、resident geometry、覆盖与夹爪校验的回归通过；并发请求计时隔离、SDK 内部调用不拦截、失败计时、客户端元数据、peer 缓存换模型/关节/后端/临时覆盖/附件均有负回归。具体命令和最终计数见交付消息。

初次隔离审计时尚未部署；随后根代理已统一部署新计时，并取得上文真实八批、两次四候选对照及一次十六候选纯规划证据。随后容量审计才定位并修复本文件开头的预分配问题。本性能审计没有执行运动。仍未重测全部旧14批、完整plan_place或普通plan_pose，也未证明完整任务成功率。seed、batch容量、LRU和资源配额未改；只修改碰撞缓存启动/增长容量，没有逐kernel GPU trace或常驻服务进程峰值显存证据。

## 后续真实运动：多次观测超过4层的冷启动回归
实验41抓取总9.169秒中原生plan_grasp约1.712秒、重新构建/预热5.441秒；plan_place总16.393秒，原生调用约.347秒，但放置与撤离后端分别重建4.906/4.955秒。日志显示此前盒子多次实测表面保留后体素层从4到5，容量变成8并销毁全部求解器。
启动体素槽位现为16（mesh2/cuboid5不变），仍按真实增长扩容并保留全部几何；物体TSDF/ESDF仍20mm。完整预热43.65秒，11个profile，最后allocated12.241GiB、reserved13.729GiB、设备剩余38.826GiB。实验43总plan_grasp4.903秒、plan_place7.397秒、双臂VLA准备1.185秒；没有发生地图容量重建。这是实际任务运行记录，不是同输入同GPU负载下的严格加速比测试。

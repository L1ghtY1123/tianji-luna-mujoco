# cuRobo batch 与网页参数

本项目锁定 NVIDIA cuRobo commit `8e734f3ced1df898990bcd92de40abce475907db`。参数定义集中在 `src/luna_mujoco/planning_settings.py`，网页通过该 Pydantic schema 生成控件，没有另一套手工维护的 YAML/JSON 参数目录。

## 当前 TCP 与放置间隙

TCP 是两指前端中心，沿模型安装端 TCP_Link 的局部 Z 偏移185 mm。应用目标先固定前端 TCP 世界 XY=slot 中心，再换算安装端目标：p_mount.xy=slot.xy-(R*[0,0,0.185]).xy。内部兼容字段 T_world_tcp、attachment 球坐标及 SDK 求解仍使用安装端参考系；不把两者混称为同一原点。R=Rz(yaw)@R_grasp，yaw绕世界Z，18个方向。安装端 Z=floor_z+(hard_clearance+0.002)-min_i((R*c_i).z-r_i)，前端 Z 由同一刚体变换确定。 目标底面保留完整 hard clearance；默认最小10 mm、规划12 mm、实际允许10–14 mm。2 mm为已有到位容差，不是降低安全间距。前端 TCP 三维位置及 XY 误差≤2 mm、姿态误差≤2°、实际球底与目标间距差≤2 mm。松爪前使用同一次实测反馈，对当前双臂关节、两侧实际夹爪和原 attachment 进行完整三维碰撞复核；失败保持夹持，不开爪、不detach。无需 attachment XY 投影包含约束，不增加盒沿/体素补偿。

## 实际调用链

JEV选择执行臂 → 原有batched IK → 原有每批最多16个plan_grasp_batch → 对当前批成功返回的三段原始轨迹计算Σ|Δq| → 按运动量升序选择并走原有执行准备/检查 → 有可执行解立即返回；本批全失败才进入下一批。score仅保留原调度顺序及同运动量时的稳定决胜。不增加规划调用，不为选优扫描后续批次，不改变批量服务、碰撞参数或执行速度。

独立 batched IK 每臂每批最多 100 个候选。IK 通过只证明目标姿态可达，不证明三段运动可行。两阶段都保留候选索引、臂与 TCP 变换对应关系。后续规划复用评估时的起始关节、另一臂关节和目标矩阵；任一关节漂移超过 1° 时停止，不偷偷更换起点。

`plan_grasp` 与 `execute_grasp` 仍各为一个 skill。JEV 在每个 skill 前重新判断；batch 内部的数值筛选由 cuRobo 和确定的选优规则完成。

## 两处 IK 必须相同的配置

“两处 IK”指独立 batched IK 与 `plan_grasp_batch` 内部 IK。以下参数由同一个配置函数构造，在实例化 solver 前注入。

| 网页参数 / API 字段 | 默认 | 可设范围 | 含义 |
|---|---:|---:|---|
| IK 优化 seeds / `ik_num_seeds` | 16 | 整数 1–64 | 每个目标的 IK 优化起点数 |
| IK 初始解 seeds / `ik_seed_solver_num_seeds` | 32 | 整数 1–128，且 ≥ IK 优化 seeds | LM 初始解生成阶段搜索数，不等于上一项 |
| IK 位置容差 / `ik_position_threshold_m` | 5 mm | 0.1–20 mm | TCP 位置成功阈值；API m |
| IK 方向容差 / `ik_rotation_threshold_rad` | 0.05 rad ≈ 2.865° | 0.001–0.35 rad | TCP 方向成功阈值；网页 °，API rad |
| `random_seed` | 123 | 0–2147483647 | 随机数种子；不承诺 GPU 逐位复现 |
| `self_collision_check` | true | true / false | 自碰撞检查；也传入 TrajOpt |
| hard clearance / `trajectory_collision_clearance_m` | 10 mm | 10–500 mm | 两处 IK 的碰撞代价激活距离，以及独立 world hard checker 的要求 |

固定项也需要一致：IK 优化器 `ik/lbfgs_ik.yml`（LBFGS）、`success_requires_convergence=true`、`use_cuda_graph=true`、`max_goalset=1`、`multi_env=false`。固定项没有网页切换控件。

默认 seed 数采用锁定版本官方 **batch MotionPlanner** 配置：IK=16、LM=32、TrajOpt=2。独立 `InverseKinematicsCfg.create` 自身默认 IK=32，本项目显式传 16 来与 batch 内部 IK 对齐。单任务 MotionPlanner 的 IK=32 / TrajOpt=4，以及原 base 自定义的 TrajOpt=16，都不是当前 batch 默认。三种 seeds 是不同阶段的搜索数量；`random_seed=123` 则只是随机数发生器的起始编号，不是第四种搜索数量。依据：[官方 batch 默认](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/motion/motion_planner_cfg.py#L165-L168)、[官方 LM 默认](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/solver/solver_ik_cfg.py#L160)。

`TrajOpt seeds ≤ IK seeds ≤ LM seeds` 是本项目增加的参数校验，不是 cuRobo 官方普遍适用的硬性输入要求。锁定 SDK 在 batch 内部要求 IK 返回 TrajOpt 所需数量的解；当 `return_seeds` 大于配置的 IK seeds 时，SDK 会上调本次 IK seeds。单 tool 的 LM 初始化配置不足时，也会自动扩大实际搜索数。本项目当前选择拒绝这类组合，以免两处 IK 的实际计算预算与网页值不同；不能把这个项目策略描述为官方算法定律。

机器人模型、碰撞球及自碰撞忽略表相同；抓取的 IK、TrajOpt、hard world checker 使用固定张开夹爪；抓取完成后 attachment 和放置使用实测闭合 driver/mimic 球，保留原有机器人球与 link buffer；同一次运行的 scene token、target 排除、另一臂状态、TCP 变换相同。MuJoCo 的物体接触保持启用。实例 ESDF 单侧最小留白为 max(机器人最大球半径,目标 attachment 半径上界)+max(实际 IK 激活距离,轨迹激活距离,硬间距)+2 cm，再向外取整到各实例体素尺寸；不再使用 50 cm 参数上限。2 cm 是计算边界余量，不是碰撞阈值或物体膨胀。

**IK 与轨迹容量分别固定为 100 和 16。** grasp/place 共用 `src/luna_mujoco/planning_batches.py` 的轨迹分批与补齐规则，每批最多 16 个真实候选。不足部分重复最后一个目标，返回后只处理真实项，避免 5、9 等候选数变化各自触发新的 CUDA 初始化。grasp 服务裁掉补齐项，`batch_size` 报告真实候选数；place 服务接收并返回 16 项，调用方在排序、失败与事件统计前排除补齐项。`batch_capacity` 报告实际容量。独立 batched IK 仍使用容量 100，和每个目标内部的 seeds 数量不同。共享的是公开 batch planner、求解器配置与分批规则；grasp 保留张开无 attachment 的三段，place 保留持物前两段和单独规划的张开空爪撤离，不新增或合并 JEV skill。

## 轨迹规划参数

| 网页参数 / API 字段 | 默认 | 可设范围 | 生效位置 |
|---|---:|---:|---|
| soft activation / `optimizer_collision_activation_distance_m` | 30 mm | 30–500 mm，且 ≥ hard clearance | TrajOpt 避障代价开始起作用的距离，不是成功判定间距 |
| TrajOpt seeds / `trajopt_num_seeds` | 2 | 1–8，且 ≤ IK 优化 seeds | 每目标的初始轨迹数量；不能与 IK seeds 混为同一参数 |
| 最大规划 attempts / `plan_max_attempts` | 1 | 1–10 | 每批每个阶段的公开 `plan_pose(max_attempts)` |
| `finetune_attempts` | 1 | 1–10 | 轨迹时间细调尝试次数 |
| `finetune_dt_scale` | 0.55 | >0 且 ≤1 | 内部细调 dt，缩放后限制在 0.002–0.2 s；不是执行时间伸长 |
| SDK 采样频率 / `freq_hz` | 50 Hz | 整数 20–100 Hz | 实际设置 `interpolation_dt=1/freq_hz`，不是 MuJoCo 积分步频 |
| 规划总时限 / `plan_timeout_s` | 180 s | >0 且 ≤1800 s | 整个 plan_grasp skill 的等待上限，包括仅零源IK时的最多两轮追加采样、全部 batch、初始化和复核；每轮不重置；增加 seeds/attempts 后可相应提高 |

TrajOpt 的位置/方向阈值、自碰撞和 random seed 也显式使用共享参数。关节限位、速度/加速度/jerk 约束、碰撞球等继续来自机器人配置；未开放网页修改。其他优化器权重、迭代数、graph planner 设置继续使用锁定 SDK / Tianji wrapper 的配置，不能声称网页已经覆盖全部 cuRobo 参数。

成功候选的三段会串接后送入现有轨迹复核器，按关节最大步幅 0.25°加密检查。复核检查 active/peer 的原始碰撞球（夹爪固定张开）。它是离散路径检查，不是连续碰撞的数学证明。每个候选的检查可以依次提交，但三段轨迹求解使用真正的 batch planner。

桌面作为 `semantic_table_forbidden` cuboid 保留；测量高度为上表面，裁去桌面以下点和体素，不保留重复桌面体素，目标排除只禁用对应的 `semantic_instance::<target_id>`。10 mm 是原始规划路径被检查构型中，开爪碰撞球相对 ESDF 的要求间距。当前普通实例 ESDF 网格为 10 mm；有实测底板的盒子为 2 mm、网格 Z 采样中心对齐底板。距离值使用 float16；它不等同于实际 mesh、闭合夹爪及控制跟踪全过程的严格 10 mm 保证。接近/抓取的最终 Hermite 指令和实际起点连接段接受固定开爪检查；lift 按本阶段约定直接执行原有提升，不引入 attachment。

## Approach / Lift 与执行

| 参数 | 默认 | 可设范围 / 含义 |
|---|---|---|
| `approach_axis` | z | x/y/z |
| `approach_offset_m` | −100 mm | −500…+500 mm，不能为 0 |
| `approach_in_tool_frame` | true | true=tool，false=world |
| `lift_axis` | z | x/y/z |
| `lift_offset_m` | +100 mm | −500…+500 mm，不能为 0 |
| `lift_in_tool_frame` | false | 仿真默认 world +Z 提升；原 base 默认 tool −Z，不能盲目照搬 |
| `time_scale` | 3 | 1–10 倍，三段共用 |
| `dwell_approach_s` | 1 s | 0–30 s，Approach 后 |
| `dwell_grasp_s` | 2 s | 0–30 s，夹爪闭合后，记录参考位姿前 |
| `dwell_lift_s` | 1 s | 0–30 s，Lift 后，验证前 |

抓取先在当前批成功的原始完整轨迹中按七关节Σ|Δq|选优，再仅对选中候选用官方输出的 q/v/a/dt 做五次 Hermite 补点；`执行 dt = SDK dt`，默认 `time_scale=3`。例如原始 101 点、dt=0.02 s，补成 301 点，仍按 0.02 s 的点间隔执行，时长从 2 s 变为 6 s。非整数倍率的区间数向上取整，实际倍率可能有很小偏差。停留、夹爪动作和求解耗时不乘倍率。日志分别保存原始与执行点数、`raw_dt`、`dt`、`raw_duration_s`、`command_duration_s`。

补点复用来自 `tianji_base` 的同一五次 Hermite 函数。“五次”是多项式阶数，补点倍率默认 3，二者没有数值上的对应关系。base 抓取调用的 50 Hz / 20 ms 只是名义点间隔，不能据此断言真机固件按固定 50 Hz 执行。本项目的轨迹采样间隔由 `freq_hz` 明确决定。

项目适配器读取并保留 cuRobo 公开结果的原始 q/v/a/dt；plan_grasp_batch 仅对原始轨迹复核 hard clearance。网页后端先按返回的原始轨迹采样点计算关节运动量，只对选中候选按相同关节顺序和有效长度补点，再走原有接近/抓取命令检查；执行前仍按最新反馈检查。三段均遵守起点连接约束，lift 按本阶段用户约定无碰撞。缺失或无效的导数会明确失败，不使用零导数或数值差分替代。MuJoCo 每 1 ms 仿真时间更新控制参考与力矩，在相邻执行点之间线性取值；这是 1 kHz 仿真控制步长，不等于墙钟必定实时达到 1 kHz，也不同于 SDK 轨迹采样频率。原 base 的 native `velocity_ratio` / `acceleration_ratio` 是真机原生运动设置，当前网页不提供不生效的真机参数。

## 放置与点云滤波

组合任务依次调用 plan_grasp、execute_grasp、verify_grasp、plan_place：抓取提升后凭夹爪反馈验证持物，plan_place 开始建立 attachment，不增加抓后RGBD。完整放置路径不在抓取前承诺可行。

放置只使用两个固定碰撞模型：抓取后的实测闭合状态加 attachment，以及完全张开且无 attachment。plan_place 在执行前规划三个阶段；前两段使用持物模型，第三段使用空爪模型。执行松爪后 detach，再执行已有撤离，不做中途观察或重规划。两种配置分别缓存。实际轨迹检查 world/self/peer 和关节边界，不再做开合角度扫掠或全空间可见性证明。

球底规划间隙 h=hard_clearance+2 mm；实际须在[hard_clearance, hard_clearance+4 mm]内，默认10–14 mm。2 mm到位容差不降低安全下限；松爪前用同一次关节与夹爪反馈复核前端TCP、球底间距及完整构型碰撞。名义物体底面间距另记，不读取隐藏几何。

放置按冻结清单的全局搬运总数N划分区域，K=2×ceil(N/2)，N=1也分两区；对象编号绑定slot。固定两指前端TCP世界XY为该分区中心，仅搜索18个世界Z轴yaw。显式转换成SDK安装端目标，共享batched IK100筛选、轨迹16规划；前两段带实测闭合夹爪及attachment，撤离为空爪。指定slot耗尽即返回失败，不换分区；不施加attachment二维投影约束。

| API 字段 | 默认 | 可设范围 / 含义 |
|---|---|---|
| `place_approach_axis` | z | x/y/z，生成放置接近位姿的轴 |
| `place_approach_offset_m` | +100 mm | −500…+500 mm，不能为 0；相对释放目标偏移 |
| `place_approach_in_tool_frame` | false | true=tool，false=world |
| `place_retreat_axis` | z | x/y/z，释放后撤离的轴 |
| `place_retreat_offset_m` | +100 mm | −500…+500 mm，不能为 0；相对实际释放 TCP 偏移 |
| `place_retreat_in_tool_frame` | false | true=tool，false=world |
| `place_release_height_m` | hard clearance +2 mm（默认12 mm） | 兼容只读的球底规划间隙；实际下限为hard clearance、上限为hard clearance+4 mm |
| `place_timeout_s` | 180 s | >0 且 ≤1800 s，持物后的放置规划及碰撞复核时限 |
| `dwell_place_approach_s` | 1 s | 0–30 s，放置接近后停留 |
| `dwell_place_s` | 2 s | 0–30 s，到释放位姿后、开爪前停留 |
| `dwell_release_s` | 1 s | 0–30 s，开爪并 detach 后、撤离前等待 |
| `dwell_retreat_s` | 1 s | 0–30 s，撤离后、post_place 验证前停留 |

上述距离 API 使用 m，网页显示 mm。`time_scale` 与抓取共用，停留不乘倍率。组合任务的抓取阶段仍独立使用 `plan_timeout_s`，正式放置阶段使用 `place_timeout_s`。

新版流程的验证输出保存在 `data/simplification-20261010/`，历史物理试验不作为新版验收。

持物球使用抓前精细三相机可见点云，在闭合后的实际抓取 TCP 下转换一次；抓后观察只验证持物。默认最小分组尺寸 10 mm、附加余量 1 mm。球预算允许1–128，旧版默认16，demo无保存设置时默认128；合法保存值原样保留，当前持物仍使用绑定时冻结的预算。按最长轴最多切分127次（预算16时最多15次）；每组比较包围盒中心、均值中心、近似最远点对中点三种快速估计，选择覆盖半径最小者。达到尺寸阈值或球数上限即停止，结果按有效预算缓存复用，不保证各碰撞对间距随球数单调改善。每个球覆盖对应实测样本及测量不确定度，不额外叠加半体素空壳，不补隐藏表面。普通实例 TSDF/ESDF 为 10 mm；盒底测量存在时，该盒子使用 2 mm 并对齐底板采样中心，消除 10 mm 体素把底板上偏约5 mm的误差。仅配置官方 MapperCfg，不平移实测几何、不改写距离值、不增加释放高度。

官方 attachment 的接口和语义见 [AttachmentManager 源码](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/collision/attachment_manager.py) 和 [Mapper 文档](https://nvlabs.github.io/curobo/latest/api/curobo.perception.html)。SDK 源码未修改。


2026-10-10 持物验证：高级参数 `holding_position_tolerance_fraction=0.02`（有效行程比例）和 `holding_stable_window_s=0.2` 秒。完整闭合命令后，稳定位置明显早于空夹闭合基准为 held，达到基准为 empty，其余为 unknown。不用力、电流或驱动状态码。位置明确时不调用视觉；位置 unknown 直接调用 OpenAI VLM，不再运行 SAM3 或点云匹配。VLM 仍看不清则保留 unknown；若 JEV 已显式取得同一快照的相反视觉证据，也保留 unknown。放置先用 SAM3/RGBD，未知或分割服务不可用时自动调用 VLM。位置不能区分机械卡滞或辨认夹住对象的身份。

固定请求高度与碰撞检查冲突时报告 `release_clearance_conflict`；不通过改高度消除冲突。以下旧实验的较高释放位置仅为历史记录，不是当前验收依据。

2026-10-11 新版放置验收：真实 RGBD/JEV/SAM3/GraspGen/cuRobo 仿真抓放完成，选择世界偏航 +45°，16 个持物球，实际名义释放高度 28.919 mm（含 10 mm 地图补偿）。明确测试条件为左臂指令和 `plan_max_attempts=2`；不是默认自动选臂或 attempts=1 的成功保证。RGBD 确认放入盒区域，独立物理检查确认物体完整落在盒内。此前不可达、起点间距不足及单次优化未收敛的失败记录均保留在 `data/place-yaw-20261011/`，没有通过禁用碰撞绕过。

2026-10-11 后续故障定位：运行 `804d175f20f448ee99d113662e8f6259` 的八个释放候选全部被背景 ESDF 误报碰撞。锁定 SDK 的体素加载器用 float32 尺寸除以体素大小，将背景网格的 120 个 Y 单元写成 119.999992；碰撞内核转为整数时截断成 119，导致距离场索引错位。项目在每次上传场景后，把公开 `VoxelData.params` 的三个单元数量原位取整，保持距离数据、物理尺寸、位姿、体素大小与碰撞阈值不变；不修改 SDK 或 vendor 源码。独立同布局 RGBD 重建测试使用日志恢复的原始 16 球和八个世界释放位姿，修复后均无端点冲突；盒底和盒壁接触对照仍报碰撞。证据保存于 `data/place-diagnosis-20261011/`；端点检查本身不等于完整轨迹或物理执行成功。

部署后恢复该次已确认场景，复跑原指令“抓取红色物体到蓝色盒子里”：运行 `c987324e6135438a8672f2c515bc54e1` 成功，JEV 自动选择 A 臂，保留 attempts=1、八个世界 yaw 候选与原碰撞参数。选中 yaw 增量 0°，plan_place 用时 15.22 s，名义释放高度 32.559 mm；实际释放、detach、撤离和末次 RGBD 验证全部完成。独立物理评分确认物体八角位于盒内且底部落到盒底，最近侧壁距离 89.46 mm。完整本地测试 901 通过、56 跳过。

历史搜索版本曾按一个实测空余位置的 72 个 yaw 搜索，最多 5 个位置；当前实现使用上述按任务数量固定的区域分区与每处 18 个 yaw。找到完整可行的放置及撤离后仍提前返回。普通实例和盒子的 TSDF/ESDF 碰撞地图统一为 2 cm，盒子保留实测底板 Z 对齐；同一场景版本和排除集合复用已上传的 GPU 数据，关节反馈和 attachment 检查仍使用当前值。缓存容量预留 8 层不代表场景增加障碍物。

2026-10-15 标记边界与碰撞网格余量分开：实测 marker 多边形使用上传持物球 + hard clearance；它不是 ESDF 网格边界，不增加一个体素。当时障碍占用筛选和释放高度保留了 hard clearance + map_resolution 补偿；当前已取消释放高度补偿和二维占用筛选。硬间距仍为配置值且最小 10 mm，未缩小持物球、未放宽 cuRobo 检查。

2026-10-17 数值一致性：释放端点和最终指令的 world cost 使用与既有轨迹检查相同的 1e-9 代价容差，仅处理 float16/float32 舍入残差；它不是以米计的间距调整，hard clearance 仍为原配置。
2026-10-17 IK后独立hard checker与轨迹/释放检查统一采用1e-9 activation-cost数值容差；仅项目侧公开RobotCollisionChecker子类处理浮点零附近结果，不降低米制hard clearance，不改SDK收敛条件。


2026-10-19 执行计时：停留参数按实际单调时钟计时，不按仿真时间计时，也不乘 time_scale；设置1 s表示实际等待约1 s。返回 duration_s 为实际等待秒数，simulation_duration_s 单独记录期间仿真推进量。轨迹仍按原仿真时间积分执行，time_scale=3不变；停留前后的夹爪、到位和碰撞检查耗时不包含在停留值中。

2026-10-19 放置准备去重：一次静止的 plan_place 调用在准备 attachment 前读取一份实测快照，持物状态判定、prepare_carrying、后续 IK 和放置轨迹规划共用该快照，避免夹爪反馈微小变化立即废弃刚初始化的持物模型。执行前和松爪前仍独立读取最新反馈并进行原有碰撞检查；不量化夹爪角度，不复用旧碰撞结论，不改变 batched IK 或 plan_place_batch 算法。

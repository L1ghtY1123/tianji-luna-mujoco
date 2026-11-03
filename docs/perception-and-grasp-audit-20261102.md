# 感知边界、回位归属与抓取耗时

用户再次确认：每个物体交付后，双臂回配置观测位姿并观测，包含最后一件。保留这个外部完成条件；禁止 runtime 自动插入跨技能流程。技能内部可实现固定算法步骤。

| 当前需要 | 实际输入与算法 | 不会因此自动执行的动作 |
|---|---|---|
| 当前场景实例、永久 ID 和几何 | observe_scene：SAM3 当前掩码；Decisions 实例身份与 CAD 模型匹配分别判断 | 不移动，不发布碰撞地图 |
| 已知 CAD 的新位姿 | 已绑定 CAD + 当前实例掩码、RGBD、内外参；FoundationPose 选有效深度视角，结果变换到 world；必要时尝试另一有效相机 | 不把 CAD 类别当永久实例 ID；不对未知粉色包裹调用 FP |
| 未知物体几何 | 当前已绑定掩码的标定深度拟合 OBB；保留永久 ID，更新几何版本 | 不凭几何变化创建新身份 |
| 发布碰撞场景 | build_scene 使用已观测的 CAD、OBB、桌面 cuboid；全局／局部范围来自 graph | 不运行 Responses，不补观测移动 |
| 放置结果 | verify_place：一批三相机 RGB Decisions，支撑／入容器、指定区域附近、仍夹持分别判断；末端反馈独立检查 | 不调用 SAM3、FP、OBB 或释放后深度测量，不回位 |
| 视觉缺口、语义分支、整体完成 | 当前三相机 RGB + 对应的问题与账本／动作证据 | 不因调用 Decisions 就建立完整几何场景 |
| 拆包后新物体 | verify_unpack 的结果判断使用当前 RGB；其内部另调用 SAM3 + 深度测新产物位置供工作空间检查；完整 CAD 位姿由之后 graph 选中的 observe_scene 建立 | 不强制拆包后回观测位姿 |

verify_place 的执行来自 Responses 生成的显式 node；execute_place 只执行计划、释放与撤离，不调用 verify_place。runtime 要求有交付证据才确认完成，但不会替模型补一个核验节点。核验技能内部的图像采集、单次 Decisions 与反馈检查属于技能算法。

move_to_observe 也必须有显式 node。正常交付后的配置回位来自 resources/workcell_assumptions.json 中用户要求；失败恢复只在当前 Decisions 明确判断看不到、看不完整或遮挡时允许换视角，并且仍需合法地图与碰撞规划。CAD 不匹配、API 失败、概率不确定本身不授权回位。prepare_unpack 的固定双臂姿态是 VLA 技能前提，不是观测位姿。

历史 P60 实际图只有两次 move_to_observe，分别在粉色交付、黑工件交付之后。拆包后直接使用当前位置观测。末尾 return_to_initial_pose 来自历史最终初始位姿条件，但该实验初始位姿等于观测位姿，因此实际没有再移动。初始与配置观测位姿不同时，两种完成条件有区别，不能混称。本轮依照用户最新“实验结束返回观测位姿”将外部最终要求明确为 observation；最后一次交付回位已满足时，不需要额外运动。

## 性能证据与修改

P60 黑工件 plan_grasp 技能 30.985 秒：前置 IK／目标预筛查约 6.75 秒；算法内部约 24.18 秒。批次 HTTP 16.58 秒，其中官方 public_plan_grasp 13.95 秒、机器人几何更新 2.49 秒、场景上传 0.10 秒。HTTP 编解码只有毫秒级，不能归咎于 Decisions 或网络。各区间包含关系不同，不重复相加。

冻结 P60 的 CPU profile 进一步定位：36 个候选的预抓取 lift 包络重复对同一 CAD 做 36 次分割、864 次球心拟合，热态 3.84 秒。现在按完整 CAD 顶点字节、球预算与分割尺寸复用固定坐标系的分割和球心；每个候选仍变换所有分割顶点、重新算覆盖半径与 padding。不是跳过碰撞检查，也不提前安装 attachment。

同一实验空手夹爪反馈约 2.3–2.6e-11 rad，旧的 1e-12 开口零位归一化范围过窄，导致多个常驻求解器重新解析相同机器人模型。范围改为 1e-9 rad，仅作用于规划模型；夹爪球保守扩张 1e-8 m，测试覆盖实际双手连杆运动及一米半径旋转误差上界。原始反馈不变，大于该范围的闭合仍逐值更新。

官方依据固定为 cuRobo commit `8e734f3ced1df898990bcd92de40abce475907db`：公开 BatchMotionPlanner.plan_grasp 内部包含 goalset、approach、grasp、lift 四次阶段求解，所有 batch 元素保持相同形状以使用 CUDA graph。项目实际 batch=4、CUDA graph=true、轨迹 seeds=4、IK seeds=16；没有改 SDK、内部 plan_pose 或缩减安全检查。官方示例与源码：

- https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/motion/motion_planner_batch.py
- https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/examples/getting_started/motion_planning.py

## 验证

先复现两项失败测试，再实现修复。附件覆盖、双手反馈、lift 包络、抓取批次与夹爪运动学针对性测试 100 项通过。Ubuntu 冻结 P60 输入复测：source_clearance_lift 热态从 3.842 秒降至 0.111 秒，lift_offset_m 仍为 0.1。

| 实验 | 总时长 | 初始 Responses | 其余总时长 | 粉色 / 黑件 plan_grasp | Decisions 次数 / 中位耗时 |
|---|---:|---:|---:|---:|---:|
| P61 `f590bfaa34b94e04923ad63cc58c0bf2` | 316.73s | 96.831s | 219.90s | 9.91 / 16.12s | 28 / 1.221s |
| P62 `9c5e70e0eb4845f69b40a52256abbcca` | 311.77s | 89.697s | 222.08s | 16.70 / 22.86s | 24 / 1.243s |

两轮均真实完成粉色 OBB + GraspGen 抓取、工作台配置区域放置、拆包、黑件 CAD + 对应臂离线抓姿、右盒入盒。每轮只有一次 Responses，无运行时 Responses；四次 verify_place 均确认，分别为 4.28/5.07/4.41/5.27 秒。每轮有两个配置 move_to_observe，均在交付后；无拆包后额外回位，无末尾 return_to_initial_pose。最终 task.completion_pose 为 observation，数值反馈已核验。

P61 无失败技能，但 resolve_placement 内部有四次候选拒绝（两个高度/碰撞无解、一个携物 IK 无解、一个离线工作空间不满足），继续搜索后成功。它们被归档为 algorithm_error，不能写成“没有任何算法错误”。P62 无技能或算法错误。黑件原生 GPU plan_grasp 分别为 11.18/12.82 秒；机器人几何更新分别仅 0.000020/0.000031 秒，原先 P60 为 2.49 秒。候选与物体随机位置不同，不能把整体耗时差全部视作同输入优化收益。

部署 timing-65 核对 1028 个项目文件、独立 CAD 库 79 个文件，无哈希不一致；CPU 总配额仍为 4 核。未改 SDK/vendor、独立 CAD 模型或 GPU 算法配置。静态检查与 diff 空白检查通过。完整回归首次 4475 passed、58 skipped，11 个旧半径预留断言失败；更新为本次保守预留量后，相关 61 项复验通过，最终 `python -m pytest tests -q --disable-warnings --tb=short`：4486 passed、58 skipped、1 warning，484.85 秒；无失败。报告与更新后的测试再同步为 timing-65-report，不重启或重新执行任务。

此前 P58/P59 的纯视觉 unknown 仍是已知稳定性缺口；本轮两次成功不能证明普遍成功率，也未实际触发遮挡换视角恢复。不得以性能优化、降低概率门槛或仿真真值输入宣称视觉问题已经解决。SDK 原生求解仍占黑件抓取规划的大部分时间。

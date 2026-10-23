# 三相机感知与抓取放置设计

2026 年 10 月 9 日。状态：用户已审核并授权执行；实现及实际GPU验收进行中。下表“当前行为”记录修改前基线。

目标是在现有单页 MuJoCo 工作台中输入“抓取红色方块到蓝色盒子里”，完成相机观测、物体及容器选择、抓取、运输、放置、释放和验证。JEV 与规划器只能使用三个 RGBD 相机的环境观测，以及机器人关节、夹爪反馈和标定。场景编辑器与离线评分可以读取仿真真值，执行链不能读取物体的真值身份、颜色、尺寸或位姿。

推荐第一阶段实现三相机感知及同臂完整抓放。抓取前检查所选抓姿是否允许该臂把物体放入目标盒；没有完整可行方案时不开始抓取。桌面中转换手和空中直接交接分别作为后续能力，第一阶段不自动执行。

## 当前实现与改动依据

本地项目为 `C:/project/tianji-jev-mujoco`，Ubuntu 项目为 `/home/guangyu/projects/tianji-jev-mujoco`。2026 年 10 月 9 日核对的 `runner.py`、`simulation.py`、`algorithms.py` 和 `resources/curobo/marvin_gui_dual.yml` 在两处 SHA256 一致。现有规划锁定 cuRobo commit `8e734f3ced1df898990bcd92de40abce475907db`，MuJoCo 为 3.10.0。

| 当前代码位置 | 当前行为 | 本次变化 |
| --- | --- | --- |
| `runner.py:154` | 每轮 JEV state 包含仿真物体与桌面真值，观测前也可见 | 改为相机观测及机器人反馈的白名单投影 |
| `algorithms.py:91`、`geometry.py:79` | 元数据识别与完整几何表面点云 | 使用 RGB 分割及对应深度的可见表面点云 |
| `geometry.py:116`、`geometry_service.py:121` | 从真值形状、尺寸、位姿生成 ESDF | 从三路深度与分割融合观测碰撞场景 |
| `runner.py:345`、`runner.py:388`、`runner.py:453` | 用真实物体位姿检查漂移与抓取成功 | 用相机追踪及关节反馈；真值评分单独保存 |
| `runner.py:186` | 仅允许 grasp 任务 | 新增 grasp_place，place 仅用于已验证持物状态 |
| `geometry_service.py:189`、`backend_builder.py:184` | 只公开几何建图、IK batch、grasp batch，后端要求 batch_size | 增加类型化 plan_place 路由及单任务 compound 后端 |
| `model_builder.py:88` | box 是实心长方体 | 新增独立 bin 类型，保留实心 box |

项目 README 的“技能边界”规定：后续 skill 增删、契约、语义或执行流程调整先经用户审核。本设计列出这些变化，不把新增 SKILL.md 等同于算法已经实现。

## 相机与标定

三路相机对应为 1 右腕 B、2 左腕 A、3 头部。外参来源是 `C:/project/tianji_base/extensions/cameras/camera-extrinsics/defaults/camera_extrinsics.json`。其中平移单位为 mm，导入时只转换平移至 m，旋转不变。

腕部使用 `T_world_rgb = T_world_arm_mount @ FK(q_capture) @ T_flange_rgb`。头部默认矩阵已经位于共享 base_link/world 系，按现有 `ffs_scene_input.py:101` 和 `capture_bundle.py:715` 的规则直接使用，不再乘 B 臂安装变换。`TCP_Link_L/R` 是 URDF 法兰参考，不等同于夹持中心；沿用项目已有 grasp EE 与工具中心变换。

外参描述的是 RGB 光学坐标系，不能使用相机外壳网格的旋转。光学坐标为 X 向右、Y 向下、Z 向前；MuJoCo 相机坐标到该坐标的固定轴变换须用投影回归验证。[MuJoCo 3.10 相机定义](https://mujoco.readthedocs.io/en/3.10.0/XMLreference.html#body-camera)

找到的非零内参来自 Ubuntu 历史冻结采集，作为第一阶段仿真标定版本，不声称是当前实时设备配置：

| 相机 | RGB 分辨率 | fx | fy | cx | cy |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 右腕 | 640 × 480 | 374.025974 | 374.025974 | 317.941558 | 235.863636 |
| 2 左腕 | 640 × 480 | 369.230769 | 369.230769 | 318.987179 | 243.089744 |
| 3 头部 | 640 × 360 | 291.891892 | 291.891892 | 320.040541 | 194.635135 |

腕部原始文件是 `/home/guangyu/.cache/tianji-codex/stereo-20260910/configured-live/1-calibration.json` 与 `2-calibration.json`；头部为 `/home/guangyu/.cache/tianji-codex/grasp-repair-20260906/multiview-depth/3/camera-calibration.json`。实现时导入原始浮点精度、设备身份及文件哈希，表中六位值仅用于阅读。这些 RGB VirtualCamera 已去畸变，畸变为零；IR/左深度的内参不同，不能混用。

仿真传感器先输出 RGB 对齐的米制深度，RGB 与深度共用 RGB K、光学外参和分辨率。每帧携带 camera_id、frame_id、simulation_generation、capture_time、K、T_world_rgb、捕获时机器人反馈和有效深度掩码。若复用原有语义建图中名为 left 的字段，明确设置 left_intrinsics=RGB K、depth_left_m=对齐深度、T_base_left_camera=T_world_rgb、left_to_rgb=单位矩阵；这是注册 RGBD 的适配合同。

原始三路保留各自分辨率。需要统一尺寸的 batch Mapper 入口采用 640 × 480：头部上下各补 60 行无效深度，同时 cy 加 60，分割与有效掩码做相同变换。补边不构造观测点，不通过仅改 fovy 近似内参。

渲染线程拥有独立 MuJoCo Renderer 与 GL 上下文；短时复制同一物理快照后渲染三路，不长时间锁住物理线程。模型更新与 Reset 使旧 frame_id 和观测结果失效。第一阶段保留网页容器的 CPU 物理，增加 OSMesa 离屏渲染依赖并验证实际耗时；现有网页镜像缺少该依赖，实施时需要重建网页镜像。若实际吞吐无法满足任务观测，再单独评估 EGL，不能宣称未测试的实时帧率。

## 环境观测与信息边界

新增项目侧相机与感知适配模块，复用现有 `jodell-sam3:0.2.0-guangyu` 和 GraspGen 镜像，以独立 JEV 容器与端口运行。SAM3 消费实际 RGB；禁止使用 MuJoCo segmentation ID、对象名称、编辑器颜色或几何类型生成掩码。颜色来自掩码区域 RGB，形状来自图像与可见点云的识别或拟合，并附置信度与推断来源。

观测实例使用感知生成的 track_id，而不是仿真 object_id。每个实例给出类别、颜色、位置范围、可见点云引用、观测来源、时间与不确定性。JEV 不接收图片数组、点云数组或真值实例映射。JEV 当前仅支持文本 state，视觉模型先形成可核查的结构化证据。[TypeSafe state 合同](https://docs.typesafe.ai/concepts/state)

头部负责整体识别；腕部用于补充遮挡表面、持物追踪与放置确认。三路融合时只使用同一捕获快照的标定和关节反馈。主相机、时间或标定不一致时报告感知错误；不退回真值链。

GraspGen 使用实际分割目标的可见点云，声明坐标系为 camera 或估计 object frame。世界抓姿由相机外参与已有工具变换合成，不再乘 MuJoCo 的 T_world_object。保留原始 score、source_batch/source_index 和最多两次推理的规则。

碰撞地图包括观测背景、拟合桌面和所有实例。原 plan_grasp 的求解仍沿用选定目标层排除规则，以允许末端接触；项目阶段复核分别处理目标：闭爪前把目标视为静态物体，仅在最终接触区域允许选定手指接触，其他连杆碰目标仍拒绝；闭爪后从静态环境排除该目标并使用随手包络。目标盒的底部、四壁和其他物体一直保留。盒子内腔的自由空间须有深度射线或可靠结构拟合证据；不可把整个盒子掩码当作实心体，也不可禁用整个盒子来获得路径。

项目当前 vendor 是已冻结的包装快照，tianji_base 的部分历史 Mapper 描述针对另一个 SDK commit。当前运行时以锁定的 8e734f3 commit 为准。实现复用前必须核对该版本的公开 Mapper、VoxelGrid 与层装载 API，使用项目侧适配，不直接复制当前 base 文件替换冻结快照或 SDK。SDK 与 vendor 快照保持只读。

相机未看见的空间与物体背面不视为已知安全。路径限制在已观测自由空间；物体包络可以结合观测支持的形状先验，但须记录估计范围与覆盖限制。无法确定盒内腔或持物范围时停止并给出具体观测不足原因，不能把局部点云称为完整物体。

机器人关节、夹爪位置、机器人 FK、标定和模型自碰撞信息属于可获得的机器人信息。物体位置、桌面模型、物体形状和颜色则必须来自感知。Runner 的 context、recent_results、错误详情、JEV 日志及规划 payload 一并检查，防止通过辅助字段泄漏真值。网页场景编辑、三维显示、Reset 基线及独立 oracle 评分继续使用仿真内部状态。

## 指令和物料盒

新增 bin 类型：一个刚体包含底板和四面矩形壁，顶部开放。外长宽高和壁厚须合法，内腔尺寸大于零；物料盒默认受重力，放在固定桌面上，经稳定和确认后作为场景基线。编辑器支持颜色、位置、yaw、外尺寸与壁厚，拖动、预览、应用、稳定、确认和 Reset 沿用现有流程，文字仍至少 20 px。

实心 box 仍是可抓取长方体。容器实例必须由图像与内腔几何证据识别为 bin，避免“红色长方体”误当“红色盒子”。物理模型按底板和四壁碰撞；感知与规划不会直接读取编辑器填入的尺寸。

JEV 分别绑定 source 的颜色、形状与位置，以及 destination 的颜色、容器类别与位置。它们是独立 Choice 问题，避免源物体与盒子的颜色串位；后续从当前感知实例中分别选唯一实例。无匹配或多重匹配明确失败。保留全部 skills + stop，以及 A/B/none 选项，由 JEV 作答后执行契约校验。

grasp 继续只抓取。grasp_place 需要源与容器都识别成功。standalone place 只有同一后端会话仍有已验证、可追踪的持物状态才允许；空闲 Reset 后执行“放到蓝色盒子里”报告没有已持物体。Runner 增加独立于单轮 run 的 holding_state，保存 arm、感知 track_id、观测点云/范围、T_tcp_object、机器人反馈、generation 与最近验证时间；不保存仿真 object_id 或真值 pose。成功 grasp 后保留该状态，下一轮 place 先重新观察并验证仍持有同一实例，再将实际持物事实导入本轮，旧 scene/trajectory 不继承。释放、Reset、场景重建或检测到丢失使它失效；后端重启也不从磁盘恢复持物事实。

## 放置算法与持物检查

直接使用项目已复制的 tianji_base PlanPlaceRequest 和 plan_place_native。现有接口的三段为 approach、place、retreat，内部复用官方 plan_grasp 的三段求解。接口不会开夹爪；释放由执行 skill 完成。

PlanPlaceRequest 没有 attached_object 字段。复用 plan_place 不等于完成持物避障。锁定 SDK 有公开 AttachmentManager 及 RobotCollisionChecker，但原三段 planner 没有放置后的释放回调。第一阶段通过项目侧按阶段的碰撞检查补齐：抓取后的 lift、放置 approach/place 使用观测持物包络；release 后的 retreat 删除随手包络，并把已放物体加入静态碰撞层。原 plan_place 返回的 retreat 是释放前预测世界中的暂定轨迹，不能保证释放后仍有效。复用 vendor 的纯配置辅助函数须核对版本和输入，不修改它或 SDK。

持物包络由观测点云和观测支持的形状估计产生，以固定 carried link/碰撞球配置交给公开检查器；仅允许与持物手的明确接触 link 接触，另一臂及环境继续检查。采用填充观测包围盒时记录 observed_surface_only 与未观测范围限制，不承诺完整几何覆盖。所有实际执行补点也必须通过阶段对应检查，不能仅复核原始规划点后把 Hermite 新点直接下发。

原机器人规划器继续保留开爪球配置，独立阶段复核则按实际夹爪 driver 与 mimic 关节更新球位姿。复核抓取闭爪过程、持物闭爪状态，以及盒内 release 的完整开爪运动；手指与被持物的预期接触仅限明确接触 link，盒壁与其他障碍不豁免。开爪扫掠与盒壁相撞时不得释放，即使原 plan_place 与持物包络均通过。

设捕获物体估计位姿为 T_world_object，实际持物关系为 `T_tcp_object = inverse(T_world_tcp) @ T_world_object`。放置目标从盒内可用区域、底面高度、物体范围和误差余量确定，TCP 目标为 `T_world_object_goal @ inverse(T_tcp_object)`，不把盒中心直接当作 TCP 目标。

release 的物理底面间隙默认计算为 hard clearance 加持物碰撞包络在底部的外凸量、感知高度不确定度和 5 mm 额外余量，同时检查盒沿与四壁余量。实际计算值写入规划证据，额外余量放在高级设置；开爪后由重力落到盒底。这样避免把被抓物体与盒底的计划接触错误套用普通 hard clearance。狭窄容器、未观测盒底、物体范围或夹爪路径不满足要求时拒绝；第一阶段不放宽碰撞阈值或禁用底板。额外余量不能替代包络复核，过大的释放高度也必须通过盒内 containment 与稳定性检查。

grasp_place 的抓姿选择必须绑定同一候选。evaluate_arms 先提供各候选源抓姿与对应放置 TCP 的双臂 IK 证据，明确标为 pair_ik_only，不冒充完整路径证明。JEV 选臂后，plan_grasp 从该臂候选的实际 lift 末端关节、估计持物变换调用 plan_place 做完整预检。当前批次高 score 候选后续不可放时，检查下一可行候选；不能只证明某个 grasp 与另一个无关 place pose 分别有 IK。该臂有限候选无完整解时，在运动前返回真实失败证据给 JEV，允许选择尚未尝试的另一臂或停止；尚未做完整预检的臂保持 unknown。每臂一次规划尝试、冻结起点、共享超时与 score 顺序沿用现有机制。

预检仅过滤明显无法完成的抓姿。实际抓起后重新观察，测量持物关系，更新碰撞世界，使用当前关节重新调用 plan_place。旧计划不得跨 observation_id、scene_token、generation 或持物绑定执行。抓起后规划失败保持持物状态并报告失败；不能未经技能授权自动张开夹爪、切臂或继续旧轨迹。

## Skill 目录与状态变化

现有九个 skills 继续存在，语义变化包括 observe 改为 RGBD 感知、build_scene 改为观测建图、generate_grasps 改用相机点云、evaluate_arms 增加同候选放置证据、plan_grasp 增加 grasp_place 的后续预检、verify_grasp 改用传感器证据。choose_arm 和 identify_target 保留 JEV 选择后检查机制；execute_grasp 仍一次执行完整抓取，但闭爪停留后、lift 运动前必须内部采集 RGBD，估计实际物体与 TCP 关系并检查 lift 的实际执行点。该内部阶段不调用 JEV；估计不足、发生滑移或碰撞时停止并保持夹爪，不用仿真真值补齐。

新增四个 skills：

| Skill | 主要 requires | 成功产物 |
| --- | --- | --- |
| identify_destination | 当前 observed；needs_place；destination 尚未选择 | destination_selected 与目标盒 track_id |
| plan_place | grasp_held；destination_selected；当前 scene_ready；尚无 place_plan | place_plan_ready 与三段轨迹、持物绑定、碰撞证据 |
| execute_place | grasp_held；place_plan_ready；尚未 released | 连续接近、放置、开爪、撤离；released 与 place_execution_completed |
| verify_place | released；place_execution_completed；尚未 place_verification_completed | place_verification_completed；成功时另写 place_verified |

任务布尔 needs_place 在 bind_task 返回后计算为 task 属于 grasp_place/place，避免用 conjunctive requires 错误表达 OR。verify_grasp 的 verified 表示检查已经执行，额外 grasp_held 仅在验证成功时写 true；verify_place 的 place_verification_completed 表示检查完成，place_verified 只表示成功。Runner 用明确的验证分支写这些实际条件效果，不引入通用工作流框架，也不把“成功返回就把所有 provides 置 true”用于 success=false。release 完成时立即更新 released=true、grasp_held=false；即使随后的观测或撤离失败，也保留这个实际物理状态，不能等整个 execute_place 返回后才写入。只有撤离成功才写 place_execution_completed。

终止规则按任务明确区分：grasp 在 verify_grasp 完成后按 success 结束；grasp_place 的 verify_grasp 失败立即结束，成功继续；grasp_place/place 只在 verify_place 完成后按 success 结束。四个新增 skills 所需任务事实在 bind_task 返回后写入。新增 facts 与当前 requires 的正反条件逐项检查，保证重复 observe/build_scene 不依赖旧 true 值而跳过新观测。

抓取验证成功后，观测和 scene_ready 失效，保留可重关联的源与盒 track_id；JEV 必须再次选择 observe/build_scene，plan_place 只消费新结果。execute_place 释放后内部重新采集 RGBD并更新静态地图，持物关系失效。撤离前检查原 retreat 的实际起点和新世界碰撞；若无效，使用当前关节、新地图和原 retreat 末端目标，通过项目侧暴露的既有 PlanToPoseRequest 与 plan_to_pose_native 重新规划撤离，并检查实际执行点。该数值重规划属于 execute_place 的内部阶段，不增加 JEV 调用或隐藏 skill。无法建立可靠新地图、确认放置状态或获得有效撤离时停止，保留 released=true，但不写 place_execution_completed 或宣称任务成功。Reset 仍可恢复完整仿真基线。

| 阶段变化 | 必须失效的事实与产物 | 可保留的实际证据 |
| --- | --- | --- |
| 抓取验证成功 | observed、scene_ready、grasps_ready、arms_evaluated、arm_selected、plan_ready、target_selected、destination_selected 置 false；删除旧点云绑定、IK 与执行轨迹 | grasp_completed=true；grasp_held=true；独立 carrying_arm、holding_state；源与盒旧 track_id 仅作重关联参考 |
| 抓取后 observe | 仅唯一重关联成功时重新写 target_selected/destination_selected；始终重新建立 scene | 新 observation_id 下的观测、持物范围与盒身份；歧义或缺失不能继承旧选择有效性 |
| 场景或持物绑定变化 | 失效 place_plan_ready 与三段轨迹；需要重新 build_scene/plan_place | 确认仍持物的实际反馈与 carrying_arm |
| 确认 release | grasp_held=false、holding_state 清除；place 计划的 retreat 转为待新世界复核 | released=true；捕获时反馈与释放事件，后续失败也保留 |
| Reset、场景重建、后端重启 | 所有单轮产物和会话持物证据失效；旧请求不得发布 | 独立历史日志 |

build_scene 新增 observed=true 门禁，并验证 observation_id。抓取阶段的 identify_target、generate_grasps、evaluate_arms、choose_arm、plan_grasp、execute_grasp 加 grasp_completed=false；place-only 初始状态设 grasp_completed=true。这些技能仍在 JEV 的完整选项中，但选后不满足条件就拒绝，不能持物期间重新调用抓取流程。holding_state 的 carrying_arm 独立于旧 arm_selected/IK 证据；新 place 只使用这只实际持物手。

旧 track_id 不等于新观测已经识别成功。抓取后 observe 必须以点云、RGB 特征、机器人反馈与空间一致性唯一重关联源与盒，成功才激活选择事实；失败不通过仿真对象身份补齐，也不静默替换另一只同色盒子。首次 identify_destination 仍由 JEV 选择。

典型顺序是 observe → identify_target/identify_destination → build_scene/generate_grasps → evaluate_arms → choose_arm → plan_grasp → execute_grasp → verify_grasp → observe → build_scene → plan_place → execute_place → verify_place。允许依赖满足的准备技能互换顺序；这不是 Python 代替 JEV 选择技能。

verify_grasp 使用夹爪反馈及跨帧目标点云追踪证明物体随手运动。verify_place 使用开爪反馈、目标与盒内腔的关联、物体位于盒内并稳定、以及手撤离后物体不再随手运动。遮挡或对称物体无法唯一配准时报告 unobservable/uncertain；不从 oracle 补齐执行证据。oracle 可以另存实际物体位置用于对比感知误判，不能进入 skill success、后续 JEV state 或规划。

第一阶段继续只允许规划开始前的可恢复无解返回 JEV；服务异常、契约错误、执行错误和超时仍终止。新增 preflight/place 明确独立错误分类，保留等待上限、pending、迟到结果丢弃和 generation 门禁。有限搜索失败不描述为数学不可达。

## 高级设置

继续由 planning_settings.py 的单一 Pydantic schema 生成网页，保存 revision，Start 冻结参数，默认锁定高级参数，Reset 保留已应用参数。现有 IK、seeds、hard/soft clearance 与规划预算仍共享，旧设置文件使用新增字段默认值迁移。

新增 Place 分组，分别暴露 approach 轴/偏移/tool-frame、retreat 轴/偏移/tool-frame、放置候选预算、release 底面额外余量和放置超时。第一阶段推荐 approach/retreat 用 world Z 向上 100 mm，便于从盒子上方进入和退出；这是项目容器场景默认，区别于上游工具坐标默认。当前 PlanPlaceRequest 不公开各阶段约束权重，不增设无法生效的控件。

| Place 控件 | 实际字段或用途 | 推荐初始值 |
| --- | --- | --- |
| Approach 轴、偏移、坐标系 | approach_axis、approach_offset_m、approach_in_tool_frame | z、+0.10 m、false |
| Retreat 轴、偏移、坐标系 | 原接口 lift_axis、lift_offset_m、lift_in_tool_frame | z、+0.10 m、false |
| hard/soft clearance | trajectory_collision_clearance_m、optimizer_collision_activation_distance_m | 共用现有 0.01/0.03 m |
| attempts、频率 | max_attempts、freq_hz | 共用现有 1、50 Hz |
| release 额外余量 | 项目计算放置目标，不是 SDK 请求字段 | 0.005 m |
| place/preflight 预算 | 项目等待与候选搜索控制，不是 SDK 请求字段 | 180 s，候选总数不超过本轮真实 GraspGen 产物 |

GPU 请求使用 goal_position_m 和 goal_quaternion_wxyz；start_joints_deg、other_joints_deg 仍按现有包装合同使用度，响应 positions_rad 按关节名字映射后执行。不会把 base Capability 的毫米变换字段混入 GPU 请求。time_scale 在项目侧保留 q/v/a 后补点，不重复请求服务 Hermite 伸长；只有明确选用服务 trajectory_point_scale 路径时才由服务执行该过程。

执行分组增加放置接近停留、释放后稳定等待、撤离后停留，并沿用项目时间伸长策略。所有显示长度为 mm、角度为度，算法 API 为 m/rad。不能通过调整 Lift/Place 参数偷偷修正用户已应用的坐标系。

## 换手能力的范围

| 方案 | 建议阶段 | 条件与取舍 |
| --- | --- | --- |
| 同臂完成整条抓放 | 第一阶段 | 同候选预检并在抓起后重观察；失败前尽量避免无后续路径的抓取 |
| 桌面中转再抓取 | 第二阶段 | A 放到双方可达且稳定的支撑区域，释放验证，A 撤离，新观测后 B 重新 GraspGen，再到最终盒 |
| 空中直接交接 | 第三阶段 | 配对双方抓姿、供体闭爪几何、持物范围、受体接近与闭合、碰撞和持有权转移 |

桌面中转复用普通抓取问题，但必须检查 A 放置、A 撤离、B 抓取及 B 最终放置都可行，且中转姿态稳定。单点双臂 IK 不能证明完整中转任务可行。

GraspGen 能产生受体抓姿候选，但当前服务只把目标物体点云送入模型，没有供体手的占用几何。抓姿评分不能证明交接双方兼容。直接交接还需受体持物验证成功后才允许供体释放；失败时供体继续持物。[GraspGen 官方实现与场景过滤示例](https://github.com/NVlabs/GraspGen/blob/main/scripts/demo_collision_free_grasps.py)

## 文件边界与验收

预计修改 src/luna_mujoco 的 runner、decisions、algorithms、geometry_service、backend_builder、planning_settings、model_builder、simulation、verification 与 app；新增相机、感知和放置几何模块。修改相关现有 skills 并新增四个 skill；修改 web 的场景编辑、参数分组、相机观测显示与任务状态。部署增加独立 SAM3 配置及网页 OSMesa 依赖。项目不运行时导入 tianji_base，不修改官方 SDK、vendor 快照或共享算法镜像。

验收必须覆盖以下可重复行为：

1. 三相机 RGB/深度同帧、深度为 m；标定反投影与腕部运动在统一 world 系一致，头部不重复变换。
2. 固定 RGBD 时改变仿真元数据、真值 object_id、尺寸或隐藏物体位置不改变 JEV 输入或规划 payload；完全遮挡的目标不能从清单“被发现”。
3. RGB 的源颜色与盒颜色不会互串；同色多个实例歧义、缺失目标和非容器均明确失败。
4. bin 内腔可容纳物体，四壁/底板保留碰撞；过窄、遮挡内腔及盒沿干涉拒绝。
5. 高分 grasp 无法 place、低分 grasp 可 place 时选完整可行候选；单臂无法完成时让 JEV 用真实证据选择其他臂或停止。
6. 机器人球不碰撞但持物体碰盒壁时拒绝；释放后的物体仍阻挡撤离；Hermite 补点也被检查。
7. post-grasp 配准或环境变化使 preflight 失效，重新规划；持物丢失、感知不足和错误不得伪造 success。
8. execute_place 的内部三段与释放均记录实际事件，阶段之间不调用 JEV；verify_place 成功前整轮不能标成功。
9. 在接近、持物、释放和撤离阶段 Reset，机器人、盒子、物体与 pending 状态按基线恢复，迟到计划不能覆盖。
10. 旧 grasp 指令及用户参数仍可使用；成功 grasp 后新 Start(place) 通过新观测继承合法持物状态；无持物状态、过期绑定、Reset 后状态均拒绝。verify_grasp 与 verify_place 的 success=false 不写成功事实，grasp_place 不能在前者成功后提前结束。
11. Ubuntu 实际运行相机离屏渲染、SAM3、GraspGen、cuRobo 单任务 plan_place；网页实际验证编辑、相机图像、参数锁定、追踪与 reset。记录初始化和预热耗时，不预先承诺实时性能。

当前仅完成源码/标定调查和基线验证：`.venv/Scripts/python.exe -m pytest -q tests/test_decisions.py tests/test_runner.py tests/test_runner_failures.py tests/test_verification.py`，58 passed。新感知、放置、持物碰撞和换手尚未实施，以上验收尚未运行。

审核对象为本设计的第一阶段：三相机感知、四个新增放置相关 skills、现有 skill 的信息及验证语义变化、同臂完整抓放预检和高级设置。第二、三阶段不包含在本次实施范围。审核后再编写详细实施计划及执行方式。

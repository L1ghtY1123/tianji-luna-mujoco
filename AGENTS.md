# Luna 开发约束

本文仅描述当前合同；历史实验不是开发指令。修改前阅读 README 与 `.agents/skills/luna-policy-isolation/SKILL.md`。用户最新明确要求优先。

## 工作边界

- 只在 `C:/project/tianji-luna-mujoco` 和 `/home/guangyu/projects/tianji-luna-mujoco` 工作；独立 CAD 库 `/home/guangyu/projects/CAD` 与统一算法库 `/home/guangyu/projects/tianji-algorithms` 已获本任务维护授权。不要创建平行项目或工作树，不使用 sub-agent，不修改其他用户服务。
- 目录边界：算法客户端、协议、服务适配、SDK 清单和注册数据只在 tianji-algorithms 维护，Luna 不保留 vendor/services/算法客户端副本。Luna 的 src 放应用和仿真逻辑，resources/robot 放机器人配置，assets 放静态模型，scripts 放应用维护工具。Luna data 只保留实验、运行日志、回放缓存和应用规划参数；正式入口不得加载其中脚本。
- 官方 SDK、安装包与参考工程只读；算法只通过匹配固定版本的公开接口调用。cuRobo 源码基线见 `../tianji-algorithms/manifests/curobo-sdk-files.json`。不能修改 SDK 或省略碰撞检查来提速。
- 五容器合计最多 4 逻辑核：web 1.5、cuRobo 1、SAM3/GraspGen/FoundationPose 各 0.5。OMP/MKL/OpenBLAS 单线程，web 软件渲染最多 2 线程。本地重检查串行。可增加本项目 RAM/VRAM，以实测收益与不 OOM 为准。
- 替换功能时同步删除旧入口、实现、专属配置、无效依赖和专属测试；当前仓库按用户要求不保留 `tests/`、`web/tests/` 和测试专属依赖；用静态检查、构建、实际入口及按需完整仿真验证。检查动态注册、前端、部署副本，远端同步包括删除。历史实验留在 data，不能靠历史兼容分支延长旧框架寿命。

## 模型与 graph

- 时间记忆数据库保存实例集合、永久 ID、历史观测、操作记录和加工谱系，通过 `ObjectRecord/ObjectObservation/ObjectOperation` 更新。空间记忆就是碰撞场景，保存实际几何及其来源、版本、启用和有效性状态，通过 `CollisionUpdate/CollisionSnapshot` 管理。原始观测、`GeometryMeasurement/GeometryObservation` 测量缓存及配置区域归执行层。`PolicyTaskProgress` 只持有本次任务责任和回执。Graph 接收 `GraphSkillResult`，Decisions 接收单题 `DecisionQuery`。回放使用“时间记忆数据库”和“空间记忆”，时间记忆显示独立提交快照。禁止将任务状态写回实例数据库。

- 入口为初始 GPT-6 Luna Responses，输入自然语言、三相机初始 RGB、配置区域、CAD 目录、技能契约及 prompt，输出候选实例、独立任务要求及 `version=2` 显式策略图。边采用 `sources/outcomes/target/max_traversals` 路由组声明，程序按源与结果的笛卡尔积展开并生成唯一边 ID。模型通过 on_failure/on_unknown 显式声明作用域公共失败去向，单独路由覆盖相应默认，router/end 继续显式声明。模型明确决定全部路由和次数，完整图继续校验。开局优先一次 Responses。任务技能执行前，图校验错误允许触发最多两次 Responses 修订，首次通过即采用。修订使用同一冻结输入，保留原候选清单和根任务要求，逐版归档完整输出与错误。运行时执行最终通过校验的图。Responses 只在开局编译和执行前图校验修订阶段调用。observe_scene 用 SAM3 与 Decisions 关联实例并更新记忆。新实例确认后由独立登记器分配永久 ID。
- 编排失败记忆由独立 CompilationMemory 保存到 data/runs/compilation-memory.sqlite3，按请求哈希去重并按当前可执行契约指纹提供历史约束与反例。新任务和任务重置保留记录。真实终止技能失败独立存入同库 execution_failures，按运行去重、按可执行契约筛选，提供有日期的失败证据。Responses 修订保留此前各轮错误，数据流与恢复覆盖的独立错误同轮报告。实例记忆继续独立保存物理身份和操作历史。
- 图修订使用 root、subgraphs、remove_subgraph_ids 声明作用域替换、添加和删除。省略的作用域保留，程序只应用模型明确提供的编辑，再校验完整图。原始 Responses 修订输出与合成后的完整 graph_proposal 同时归档，回放展示每版完整图。
- 实例集合属于可更新、可查询的时间记忆数据库。graph/subgraph 负责逐物体流程和有限恢复，允许共享子图使用 $selected 循环处理当前目标。独立 subgraph.task 为可选表达，执行账本在选中输入时登记处理责任。graph node ID 表示控制节点，不能当作物体 ID；CAD asset ID 表示可被多个物体实例共享的模型。
- Responses 编排正常主线及所需恢复分支。谓词 router 使用空 choices，恢复选择器声明 edge.on 对应的 choices 及 unknown。Decisions 在可用的恢复分支中选择，可经子图返回调用方选择器。无合法恢复时明确结束。Decisions 输出有类型的视觉/语言判断、选项和概率，只选择已经声明的 edge；不能生成 node/edge、选择图外技能或补参数。Runtime 只解释 edge、维护证据与账本，未处理的结果明确停止。
- 跨 skill 的观测、回位、换臂、建图、核验和恢复必须出现在模型 graph。skill 内允许固定算法步骤和真实前提。禁止隐藏流程、重播已完成物理效果、用空队列宣称完成。每个 subgraph 显式到达自己的 end。
- 批处理每件交付后由 Responses graph 显式回配置 observation 位姿并重新观测，再选择当前输入。all/count 使用共享子图，初始数量不限制运行时成员。最新观测确认来源盒已空即结束本批。最终双臂回位由实际关节反馈验收。VLA 起始姿态仍属于加工技能前提。
- 初始候选不是已确认事实；observe_scene、select_target、verify_task_completion 由 graph 安排。完成证明绑定动作 epoch 与清单版本。产物独立永久 ID，保留输入/产物谱系及加工回执。

## Decisions 与感知

- 每轮实例关联使用执行层投影的 association_scope。当前输入及其产物优先，已交付实例保留物理历史并按需比较。单个身份谓词的历史证据限定到被比较实例，当前已接受主体继续保留标注与几何，帮助区分相似物体。新实例登记前仍检查全部可用历史身份。已交付内容物只有在入盒回执、有效空间占据和当前容器 RGBD 对应一致，且通过已有容器位置稳定性检查，且候选完整有效表面位于占据范围内时，才记录为 deferred_candidates。记录包含采集、场景版本、容器、回执、测量范围和占据范围，identity_assigned=false。来源盒、工作区及未完成物体的已知支撑容器继续完整关联。占据失效、容器对应缺失、候选伸出范围或后续任务重新操作该容器时，恢复通常关联。空间记忆继续保留交付实例的占据；单件位姿保持未测量。association_scope 参与观测复用键。

- 每个 Decisions HTTP 请求只能包含一个 question。复用冻结采集，逐题保存原始回答、请求来源和转换原因。CAD 模型选择、实例身份、需要恢复时的可见性、支撑关系及任务结果按需分别判断。当前已接受身份直接提供可见事实，全清单完整性不作为执行前提。单个新实例提议 unknown 独立保存反馈，已确认实例继续处理。问题只描述本次需要的事实；历史身份/位置明确标记为历史，不与当前像素或任务目标混为一谈。
- SAM3 提供当前候选掩码；Decisions 关联永久实例、选择目录 CAD，模型匹配可缓存。不能仅凭分割词把候选当作已确认 CAD。颜色不能硬编码执行臂或模型。
- 初始登记复用感知层 SAM3 缓存和注册深度，提供当前掩码、表面摘要及参考表面投影。参考/当前证据绑定采集与代次，缺失几何如实标记。确认后的掩码参考传给首个 observe_scene，完整观测按原记忆事务持久化。
- 看不到、看不完整、遮挡与语义否定分开。恢复由 Decisions 结合逐轮数值失败和当前证据选择图中有限分支；观测恢复允许配置回位或满足前提的腕部换视角，已在配置位姿时判断该分支的实际证据增益；配置回位不再额外要求遮挡判断，仍要求空手、合法地图和数值运动检查。额外腕部视角才要求独立视觉缺失判断。原位重拍不改善相同输入的置信度；相同采集复用缓存，必要采集没有任意调用次数上限。
- 有效 no/unknown 不自动重试到 yes，不修改事实阈值来制造成功。仅对无有效回答的 HTTP 429/502/503/504 或网络故障，最多三次重发同一冻结请求并独立归档，不重放技能。
- verify_place 对普通 container_release 校验执行绑定、实际盒上释放与开爪，按用户规则建立入盒回执。该分支无需 Decisions。surface_setdown 和额外落地关系使用同次三相机 RGB 逐题核验。空间记忆允许以已测容器包围范围覆盖已入盒且无法逐件定位的内容物，并记录规则和几何来源。

## 几何、抓取与放置

- 生产 `LUNA_GEOMETRY_MODE=cad_obb`：实测桌面 cuboid、未知物体当前观测 OBB、已知物体登记 CAD。粉色包装是未知 OBB；仿真中的粉色网格不是识别 CAD。未知 OBB 用 GraspGen；已知 CAD 按执行臂用离线抓姿，失败不能暗中改走 GraspGen。
- `LUNA_CAD_ROOT` 指向独立 CAD 库；项目 resources/cad 是运行副本。蓝盒 1000 面、黑件 999 面，配准和碰撞使用相同表面，禁止恢复高面数双模型分支。保持坐标、A/B 离线抓姿及其哈希绑定一致。
- FoundationPose 输入当前 SAM3 掩码、RGBD、CAD 和对应标定；选有效深度最充分的相机，输出转换到世界坐标。有效且来源/坐标绑定一致的位姿直接采用，不做 ICP 或额外几何吻合检查。新观测更新位姿，不能把历史位姿当新测量。
- 清单永久 ID 绑定 CAD/OBB。初次必须成功发布全局基础场景；局部更新要求 collision_map_initialized，只刷新选中实体，保留其他历史实体及未知状态。基础场景的存在与当前几何是否新鲜是两个事实；动作影响几何必须失效、预测更新或重新观测并如实标记。全局更新不等于全空间可见，只有明确退役才删除实体。
- 碰撞世界只包含已观测清单实体及用户声明的工作台条件，不从仿真真值或清单外物体补地图。射线可见性仅是证据诊断，不要求全机被相机覆盖；全机环境/自碰撞、关节及场景新鲜度检查保留。
- TCP 使用校准 `gripper_front_A/B`，安装端局部 +Z 185mm；GraspGen 输入帧只在接口转换一次。三相机标定遵循只读参考工程，内外参设备序列号匹配；仿真是理想针孔，原生畸变只作来源记录。
- 三段抓取使用锁定官方 plan_grasp 的接触链接参数，按执行臂排除两侧 Link3 夹指和 Link4 连杆。SDK 在 goalset/grasp/lift 求解时临时排除，approach 时恢复；适配器负责异常后的恢复。执行前逐段核对环境、另一臂、非接触部件与目标接触条件。MotionPlanBinding 绑定目标、目的地、臂、采集、场景和参数，变化后重新规划。
- 抓取第二段及闭合的接触许可限定为当前目标与执行臂两侧 Link3、Link4 的配对，共 34 个配置球。这些链接与其他环境物体、另一臂、其余部件与目标及自碰撞继续使用原数值检查。未知物体端点按完整实测点云检查张开接触垫网格凸包间隙，计入配置的测量不确定性和碰撞余量。
- attachment 在抬升成功、机械反馈通过后作为执行层携物碰撞模型建立。不调用视觉夹持核验，不向时间记忆写入夹持/未夹持状态。首段 lift 不要求尚未建立的 attachment；建立后绑定执行臂、抓取变换和场景版本。普通盒上释放结合完整执行绑定按用户规则登记入盒，单独开合事件只记录机械反馈。
- 工作空间使用已有离线包络；位置可达不代表姿态 IK 或路径无碰撞。模型在数值允许且符合任务要求的臂中选择，只有一臂满足时直接解析。通用任务不按颜色固定左右臂。
- 放置保持抓取姿态为默认。工作台“指定/设定区域”使用配置 XY；普通区域按 4→8→16 分区提出 XY，Decisions 按原始语言选候选 ID，再求 Z。surface_setdown 的 Z 从碰撞阈值附近求最近无碰撞高度。container_release 的完整物体最低点名义高于已发布盒沿 10mm，控制台可调；不把 TCP 高度当物体间隙，不自动抬高重试。
- `perceive_placement_zone` 给区域，`resolve_placement` 求落点，`identify_destination` 显式绑定 release_mode；`plan_place` 对 surface_setdown 用 cuRobo 规划接近、放下及释放后撤离，对 container_release 调用既有携物 plan_pose，仅移动到释放位姿后停稳开爪，无隐藏撤离。规划含夹爪和持物碰撞、工作空间、IK 与轨迹检查。精确 XY 不得为成功率漂移，中间工作区不等于最终目的地。
- vla_unpack 的配置 XY、固定姿态和双臂起始状态是该 skill 前提，不是通用编排模板；prepare_unpack 显式执行并核验。仿真加工效果及产物随机位置只用于效果内部，视觉确认产物后再进入地图。

## 验证与文档

影响技能用途、前提、算法、结果、恢复或状态时，同步 contract/SKILL.md、docs/luna-skills-guide.md 与 README。报告真实实测，区分静态/构建检查、健康检查和完整仿真；保留失败证据，不把一次成功说成稳定成功率。性能目标：扣除初始 Responses 后 120–240 秒，按实际阶段耗时定位。

完全相同的路由声明按 source、outcome、target、max_traversals 合并，次数保持原声明值。不同目标或不同次数仍返回冲突。verify_place 显式声明释放模式，工作区轻放的地图失效前提继续传递，普通盒上释放采用独立的条件性占据发布结果。

关联范围同时覆盖当前运动所需的目的地几何。当前存在未完成输入或产物时，目标盒内容物恢复逐件关联，优先在加工准备和携物遮挡前取得当前 CAD/OBB；直接搬运任务持续保留这项目的地几何需求。逐件交付后尚未选定下一输入的观测可使用有效容器占据。空间占据保持参与全部数值碰撞检查。

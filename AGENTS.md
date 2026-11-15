# Luna 开发约束

本文仅描述当前合同；历史实验不是开发指令。修改前阅读 README 与 `.agents/skills/luna-policy-isolation/SKILL.md`。用户最新明确要求优先。

## 工作边界

- 只在 `C:/project/tianji-luna-mujoco` 和 `/home/guangyu/projects/tianji-luna-mujoco` 工作；独立 CAD 库 `/home/guangyu/projects/CAD` 与统一算法库 `/home/guangyu/projects/tianji-algorithms` 已获本任务维护授权。不要创建平行项目或工作树，不使用 sub-agent，不修改其他用户服务。
- 目录边界：算法客户端、协议、服务适配、SDK 物体记录和注册数据只在 tianji-algorithms 维护，Luna 不保留 vendor/services/算法客户端副本。Luna 的 src 放应用和仿真逻辑，resources/robot 放机器人配置，assets 放静态模型，scripts 放应用维护工具。Luna data 只保留实验、运行日志、回放缓存和应用规划参数；正式入口不得加载其中脚本。
- 官方 SDK、安装包与参考工程只读；算法只通过匹配固定版本的公开接口调用。所有涉及 cuRobo 的技能统一使用 cuRoboV2-Fast v0.9.0，规划工厂启用 官方默认求解器配置。源码基线见 `../tianji-algorithms/manifests/curobo-sdk-files.json`。不能修改 SDK 或省略碰撞检查来提速。
- 五容器合计最多 4 逻辑核：web 1.5、cuRobo 1、SAM3/GraspGen/FoundationPose 各 0.5。OMP/MKL/OpenBLAS 单线程，web 软件渲染最多 2 线程。本地重检查串行。可增加本项目 RAM/VRAM，以实测收益与不 OOM 为准。
- 替换功能时同步删除旧入口、实现、专属配置、无效依赖和专属测试；当前仓库按用户要求不保留 `tests/`、`web/tests/` 和测试专属依赖；用静态检查、构建、实际入口及按需完整仿真验证。检查动态注册、前端、部署副本，远端同步包括删除。历史实验留在 data，不能靠历史兼容分支延长旧框架寿命。

## 模型与 graph

- 时间记忆数据库保存实例集合、永久 ID、历史观测、操作记录和加工谱系，通过 `ObjectRecord/ObjectObservation/ObjectOperation` 更新。空间记忆就是碰撞场景，保存实际几何及其来源、版本、启用和有效性状态，通过 `CollisionUpdate/CollisionSnapshot` 管理。原始观测、`GeometryMeasurement/GeometryObservation` 测量缓存及配置区域归执行层。`PolicyTaskProgress` 只持有本次任务责任和回执。Graph 接收 `GraphSkillResult`，Decisions 接收单题 `DecisionQuery`。回放使用“时间记忆数据库”和“空间记忆”，时间记忆显示独立提交快照。禁止将任务状态写回实例数据库。

- 入口为初始 GPT-6 Luna Responses，输入自然语言、三相机初始 RGB、配置区域、CAD 目录、技能契约及 prompt，输出候选实例、独立任务要求及 `version=2` 显式策略图。边采用 `sources/outcomes/target/max_traversals` 路由组声明，程序按源与结果的笛卡尔积展开并生成唯一边 ID。模型通过 on_failure/on_unknown 显式声明作用域公共失败去向，单独路由覆盖相应默认，router/end 继续显式声明。模型决定任务组合、自定义路由和次数；可显式引用 resources/policy/subgraphs.json 的固定模板并绑定输入，程序展开模板及依赖，再对全部节点和边执行相同完整校验。模板不得隐藏物理动作或替代最终任务验收。开局只调用一次 Responses。完整输出与校验错误归档。校验通过后执行，失败立即结束编译。observe_scene 用 SAM3 与 Decisions 关联实例并更新记忆。新实例确认后由独立登记器分配永久 ID。
- 编排失败记忆由独立 CompilationMemory 保存到 data/runs/compilation-memory.sqlite3，按请求哈希去重并按当前可执行契约指纹提供历史约束与反例。新任务和任务重置保留记录。真实终止技能失败独立存入同库 execution_failures，按运行去重、按可执行契约筛选，提供有日期的失败证据。Responses 修订保留此前各轮错误，数据流与恢复覆盖的独立错误同轮报告。实例记忆继续独立保存物理身份和操作历史。
- 实例集合属于可更新、可查询的时间记忆数据库。graph/subgraph 负责逐物体流程和有限恢复，允许共享子图使用 $selected 循环处理当前目标。独立 subgraph.task 为可选表达，执行账本在选中输入时登记处理责任。graph node ID 表示控制节点，不能当作物体 ID；CAD asset ID 表示可被多个物体实例共享的模型。
- Responses 编排正常主线及所需恢复分支。谓词 router 使用空 choices，恢复选择器声明 edge.on 对应的 choices 及 unknown。程序规则选择已知恢复；只有未解决的当前视觉事实才调用 Decisions。无合法恢复时明确结束。Decisions 输出有类型的视觉判断、选项和概率，只选择已经声明的 edge；不能生成 node/edge、选择图外技能或补参数。Runtime 只解释 edge、维护证据与账本，未处理的结果明确停止。
- 跨 skill 的观测、回位、换臂、建图、核验和恢复必须出现在模型 graph。skill 内允许固定算法步骤和真实前提。禁止隐藏流程、重播已完成物理效果、用空队列宣称完成。每个 subgraph 显式到达自己的 end。
- 批处理每件交付后优先在当前位置重新观测并选择当前输入；只有程序确认必要信息缺失时才沿 graph 回观测位姿补观测。all/count 使用共享子图，初始数量不限制运行时成员。最新观测确认来源盒已空即结束本批。最终双臂回位由实际关节反馈验收。VLA 起始姿态仍属于加工技能前提。
- 初始候选不是已确认事实；observe_scene、select_target、verify_task_completion 由 graph 安排。完成证明绑定动作 epoch 与物体记录版本。产物独立永久 ID，保留输入/产物谱系及加工回执。

## Decisions 与感知

- Decisions 仅处理必要视觉事实：实例身份、夹持、落点、加工结果、来源余物及未被当前数值证据证明的支撑关系。真正新出现且任务归属未知的物体可按需做一次视觉归属。每轮多候选选物体交 Decisions 比较过滤图像；选臂、建图范围、明确错误恢复、放置许可和标准任务回执验收交程序；语言要求由初始 Responses 编译到 TaskSpec。
- 所有当前 RGB、深度、SAM3、Decisions 和建图输入严格使用同次工作空间掩码。实例掩码与工作空间取交集，永久屏蔽工作空间外的物体。原始 RGBD 单独归档。
- Responses 从初始三 RGB 理解场景并仅提出任务涉及实例，mentioned_in_task 区分任务与场景描述，reference_camera_id 是描述锚点。SAM3 对每类主词在三相机分割。每个候选同时保留实际分割短语和原检索词/别名；备用短语与同掩码合并不能丢失原查询归属。首次登记与历史选主体都按该来源召回候选，再交 Decisions 确认身份。
实例关联只覆盖自然语言任务中的目标、来源、目的地、加工区域及其必要支撑。场景描述用于定位和消歧。初始 Responses 提出任务实例，Decisions 先在描述参考相机的当前带标签 RGB 中选择一个掩码，再仅为这些选中主体关联其他相机掩码。其他同类掩码可作为选择项和画面上下文，额外可见物体不自动登记。后续观测使用任务实例的已确认带日期参考选择当前掩码，再关联其余视角并继承永久 ID。近期参考不足时使用更早的确认参考。all/count 任务按目标描述发现新增任务输入，确认新身份后登记。程序保留同相机唯一性、跨相机关联一致性和可靠几何冲突检查。跨相机几何检查使用双向投影，并纳入本次已确认的同物体其他视角。每个方向独立报告有效深度对应、遮挡和冲突，任一可靠冲突排除该候选。 各主体先独立选择暂定掩码；跨相机物理分组后，使用当前多视角及日期参考做一次主体复核，允许选择其他掩码或明确不匹配；改变物理组时重新归组一次。复核不把初选当作已确认身份；同一物体存在多个归属提案时，用竞争描述和日期参考共同裁决，败选主体最多重选一次。已裁定掩码保留所属主体标签并从重选项排除，未解决冲突不提交。剩余候选仍需视觉确认，容器与内容物按各自掩码区分。历史选主体只发送包含该主体标注的参考视图。Decisions 的当前三相机 RGB 同时标明已确认视角，state 保存对应关系及来源。历史记录仅登记任务涉及的物体。碰撞场景覆盖工作空间内的可见几何，已匹配 CAD 使用 CAD，桌面使用实测 cuboid，其余使用 TSDF/ESDF。额外物体通过匿名深度建图，无需 Responses 命名或 SAM3 分割。已完成入盒物体保留入盒历史，后续由当前匿名 TSDF 表示可测表面，个体位姿保持未测量。每次 build_scene 都从当前观测重建整个工作空间：当前 CAD 位姿来自 SAM3 与 FoundationPose，其他可见几何来自当前 RGBD 的 TSDF/ESDF，桌面 cuboid 保留实测缓存。不选择局部范围，不合并历史物体碰撞几何或旧 TSDF 体素。历史身份与操作记录独立保留，未观测空间不因此被证明为空。测试任务不限定程序能力，也不产生任务专属建图规则。
- 历史关联使用数据库最近已确认的多相机参考、初始三相机参考、当前实测位置及操作历史。当前归组只使用同次标定与深度；历史位置作为有日期证据，移动后不把旧位置投影当当前匹配。成功 observe_scene 沿现有数据库事务更新最近参考和所有已确认视角掩码。confidence 仅归档。
- 数据库保存永久 ID、原始参考及操作历史。新实例登记仍要求工作空间内有效深度，已有实例同样受严格工作空间掩码约束。身份接受后才进入数值机器人/桌面过滤、CAD/FoundationPose 或 TSDF 建图；数值结果不改写为模型回答。已交付内容物保留有来源的碰撞占据，但不靠容器跟踪自动赋予当前身份。
- 每个 Decisions HTTP 请求一个问题，复用同一冻结采集，保留原始请求、回答和转换原因。二值/多选按概率最大项，confidence 只归档，不设 0.8 等接受或缓存门槛。平分、拒答、明确 unknown 保留未知。有效 no/unknown 不通过重复同一问题强行变成 yes；新观测可沿显式有限恢复重新判断。
- 恢复 router 的 rule=observation_recovery/failure_return 是程序选择，只有 rule=visual 才调用模型看当前图像。单一合法路径用直接 edge。所有回位、观测、换臂与恢复动作仍在最终展开 graph 中，保留有限边预算、空手/地图/碰撞/关节前提。没有当前有效图像不能通过旧图声称当前视觉事实。
verify_grasp 在 Lift 后只调用一次 Decisions 判断是否持物，输入执行臂当前夹持接触区原始分辨率 RGB 和同次夹爪反馈，不做夹爪状态或运算，也不调用 SAM3。明确未持物允许空抓恢复；请求异常、拒答或证据不足保留 unknown。确认持物后，程序核对连续抓取记录的目标、执行臂、运行、代次、来源观测、场景及动作版本，继承抓取前已确认目标的永久 ID，再建立 attachment。identity_binding 明确标记 continuous_grasp_execution 和 visual_identity_verified=false；这属于执行过程推定身份。执行记录不一致时保留持物结果与身份未确定状态，不建立 attachment。历史参考保留用于抓取前关联与放置核验，正常主线不再单独做抓后身份题。
- verify_place 复用 object_held 正向题，只发送当前夹持接触区原始分辨率 RGB 与同次夹爪反馈。明确 no 取反为空爪成立，yes/unknown 不允许确认释放。之后 place_result 在 direct_success、inferred_success、failure、unknown 中选择。入盒任务以进入指定盒子为成功，控制台小方框只约束机器人释放位置，不约束盒内最终落点；桌面区域任务仍验收指定桌面区域。推测成功要求已确认持有目标、实际到达配置释放 XYZ（现有位置容差 5 mm）并释放、当前空爪，且没有交付失败的可见反证，不要求一定被遮挡。命令发出不等于实际到达；缺失绑定证据为 unknown，明确落在目的地外为 failure。当前三相机 RGB、释放前 RGB 与日期身份参考用于落点题；仅桌面任务投影 PLACEMENT_REGION。保留原始回答、执行推定身份、placement_outcome 与回执；推测入盒不伪造个体当前位姿，confidence 仅归档。
- TaskSpec.allowed_intermediate_refs、unsupported_placement_constraints、visual_completion_conditions 由初始 Responses 明确编译。固定目标能力之外的放置条款声明不支持；运行时不改 XY、重新解释许可或让模型放宽要求。标准交付、加工、数量/顺序与最终姿态使用已核验回执、谱系和实际关节反馈，只有额外视觉要求再次看图。
- 观测权限仍由 captures_observation 声明。只有 observe_scene、verify_grasp、verify_place、verify_unpack、verify_task_completion 和初始 bootstrap 可采集；规划、运动、router 不隐藏采集。仅对无有效回答的 HTTP 429/502/503/504 或网络故障最多三次重发同一冻结请求并独立归档，不重放技能。

## 几何、抓取与放置

- 所有当前 RGB、深度、SAM3、Decisions 和建图输入严格使用同次工作空间掩码。实例掩码与工作空间取交集，永久屏蔽工作空间外的物体。原始 RGBD 单独归档。
execute_grasp 和 execute_place 均下发三段时间序列。放置在 place 段末尾松爪，随后执行张开、已 detach 的 retreat。

- 生产 `LUNA_GEOMETRY_MODE=cad_esdf`：实测桌面 cuboid、未知物体当前掩码 RGBD 的 ESDF、已知物体登记 CAD。未知物体用同次 TSDF 表面输入 GraspGen，模型 TCP 转换到 A/B 真实末端，不再沿 Z 额外伸入；已知 CAD 用 FoundationPose 和离线抓姿，默认额外伸入 0mm。未知物体的几何与 attachment 统一使用 TSDF。
- `LUNA_CAD_ROOT` 指向独立 CAD 库；项目 resources/cad 是运行副本。蓝盒 1000 面、黑件 999 面，配准和碰撞使用相同表面，禁止恢复高面数双模型分支。保持坐标、A/B 离线抓姿及其哈希绑定一致。
- FoundationPose 输入当前 SAM3 掩码、RGBD、CAD 和对应标定；选有效深度最充分的相机，输出转换到世界坐标。有效且来源/坐标绑定一致的位姿直接采用，不做 ICP 或额外几何吻合检查。新观测更新位姿，不能把历史位姿当新测量。
- 物体记录永久 ID 绑定 CAD/TSDF。每次建图均发布当前观测的全工作空间碰撞场景，替换上一轮物体几何；历史档案不随地图替换删除。动作影响几何后，由 graph 显式观测和全局重建。全局重建不等于全空间可见。
TCP 统一使用 gripper_front_A/B，实际机器人定义为安装端局部 +Z 185 mm。GraspGen 服务在边界将 Robotiq 模型的 base_frame 沿 +Z 195 mm 转为模型 TCP，应用与可视化消费末端位姿。训练模型的夹爪尺寸差异需保留在来源记录中。
- 三段抓取使用锁定官方 plan_grasp 的接触链接参数，按执行臂排除两侧 Link3 夹指和 Link4 连杆。SDK 在 goalset/grasp/lift 求解时临时排除，approach 时恢复；适配器负责异常后的恢复。执行前逐段核对环境、另一臂、非接触部件与目标接触条件。MotionPlanBinding 绑定目标、目的地、臂、采集、场景和参数，变化后重新规划。
- 抓取第二段及闭合的接触许可限定为当前目标与执行臂两侧 Link3、Link4 的配对，球索引从当前机器人配置读取。这些链接与其他环境物体、另一臂、其余部件与目标及自碰撞继续使用原数值检查。未知物体端点按完整实测点云检查张开接触垫网格凸包间隙，计入配置的测量不确定性和碰撞余量。
- 工作空间使用已有离线包络；位置可达不代表姿态 IK 或路径无碰撞。程序在数值允许且符合任务要求的臂中按同目标失败记录和 TCP 距离排序。通用任务不按颜色固定左右臂。
放置 XY 使用控制台正方形中心，旋转保持实际抓取姿态。末端 Z = 实测桌面或盒口最高点 Z + 物体高度 + 控制台高度余量。物体高度来自同次 CAD/TSDF 几何竖直范围及来源支撑，独立于碰撞球。默认余量 20 mm，含为保守附着包络增加的 10 mm。若抓取 TCP 高于物体顶部，使用较大的 TCP 到来源支撑面偏移避免低估。实际物体底面另按抓取偏移校验，假设搬运中物体不滑移。碰撞几何独立检查目标及轨迹，不搜索目标高度。选臂时用可见高度估计，抓取后用已绑定 CAD/TSDF 再检查同一臂。
- vla_unpack 的配置 XY、固定姿态和双臂起始状态是该 skill 前提，不是通用编排模板；prepare_unpack 显式执行并核验。仿真加工效果及产物随机位置只用于效果内部，视觉确认产物后再进入地图。

## 验证与文档

影响技能用途、前提、算法、结果、恢复或状态时，同步 contract/SKILL.md、docs/luna-skills-guide.md 与 README。报告真实实测，区分静态/构建检查、健康检查和完整仿真；保留失败证据，不把一次成功说成稳定成功率。性能目标：扣除初始 Responses 后 120–240 秒，按实际阶段耗时定位。

完全相同的路由声明按 source、outcome、target、max_traversals 合并，次数保持原声明值。不同目标或不同次数仍返回冲突。verify_place 显式声明释放模式，工作区轻放的地图失效前提继续传递，普通盒上释放采用独立的条件性占据发布结果。

完全相同的完整子图定义合并为一个作用域，同 ID 不同定义继续拒绝。身份/支撑恢复覆盖限定于契约明确要求空手的技能。其他节点允许作者声明有限恢复或明确终止。恢复继续由 Responses 显式编排，保留已完成的物理操作与加工谱系。

编排失败记忆按技能和具体前提保留不同错误实例，不能让同一错误码的两条示例覆盖其他前提失败。提示中的历史证据保留事实依赖，完整符号路径在原始实验中归档。



实验归档仅保留 graph、视频、观测图、时间记忆数据库、空间记忆及实际模型/算法原始输入输出。复用相同数组和图像，以引用连接记录；视频编码完成后清理中间文件。

二值事实判断按较大概率输出结果，P(true)>0.5 为 true，P(true)<0.5 为 false。多选取选项概率的最大项；有效响应未包含分布时使用服务合法选择。confidence 仅作审计，任何 Decisions 接受/缓存逻辑不得追加 0.8 等 confidence 门槛。精确平分、模型 unknown、拒答保留 unknown；协议错误明确记录。


当前统一失败收尾遵循 [终止失败回位](docs/terminal-failure-return-20261111.md)：具体恢复耗尽或未处理异常均执行显式固定收尾图，从实际关节规划回观测位姿并停止，保留原任务失败；不安全或无法确认的回位报告 blocked。

GraspGen uses 100 candidates per round with one graph-controlled additional round. plan_grasp submits the configured grasp_goalset_size ranked candidates per request and retains all numerical validation. Confirmed surface release invalidates voxel geometry pending explicit RGBD remapping.

Attachment follows completed Lift and Decisions holding presence and continuous execution identity inheritance. Planning and executing the entire grasp sequence use the detached robot and the official grasp-planning open-gripper model, with a separate closure check. The executing Link3/Link4 links retain the selected-target contact policy during grasp and Lift. Other robot/environment and self-collision checks remain active. Successful verification prepares carrying geometry for later transport and placement. The earlier pre-Lift virtual carrying and opening-range path sweep implementation has been removed.

Responses 每个任务只调用一次，解析或图校验失败立即结束编译。固定模板由程序展开，逐物体子图的 completion_pose 在输出 schema 中限定为空，最终回位由根任务声明。执行采用三倍时间伸长和五次 Hermite 插值。MuJoCo 关节目标以 50 Hz 更新，物理积分与跟踪误差检查按原物理步运行。

实验回放按 graph、视频、decisions、数据库、碰撞场景、原始数据排列。decisions 页只展示实际请求图片，每排四张，按发送顺序读取归档，同时提供 state、图片元数据、questions 与回答。初始描述绑定携带描述相机与采集锚点。新实例确认记录实际比较是否执行，P 标签与永久 ID 显式映射。每张模型图片记录实际尺寸，CAD 预览使用独立静态参考角色。

未知物体的 attachment 使用抓取前已发布的同次多相机 TSDF 表面体素，将每个体素的八个角点按闭爪时实际 TCP 转到夹爪坐标，再生成覆盖全部体素的碰撞球。放置足迹沿用同一组体素角点。来源绑定采集、代次、永久 ID 和体素尺寸，缺失或过期时报告失败。已知 CAD attachment 使用已配准 CAD。Lift 完成、视觉持物成立且连续执行绑定通过后建立 attachment，身份按执行记录继承。

GraspGen 候选先按真实末端 +Z 在世界坐标中的朝向筛选，允许世界 +X 前方至 −Z 下方的扇区，偏离世界 XZ 平面的左右角度默认最多 30°，控制台可调范围 0–45°。CAD 使用离线抓姿。规划完成后对实际抓取端点 FK 再核对相同朝向条件。抓取 Lift 显式指定世界 +Z 150 mm，并用公开 FK 检查每个规划插值点的横向偏移和提升高度，采用已有 IK 位置容差 5 mm。方向筛选证据、逐候选通过结果和 Lift 位移核验随规划归档。Lift 完成并通过持物核验后建立 attachment。

Decisions 输入以实际发送图片的顺序编号，每张图标明相机角色、观测 ID、代次、时刻、CURRENT/历史/静态目录角色及归一化框坐标约定。当前图片与请求采集绑定检查通过后才发送。跨相机选项描述给出 C 标签、相机和该图中的框坐标，相邻时刻 P 选项给出历史采集及对应图像位置。初始分组和 CAD 选项分别描述当前标签与目录标签。裁剪和单主体掩码元数据只列出本图主体。支撑关系问题与实际发送的 RGB 轮廓一致，放置核验提供目标和目的地的带日期参考，逐物体完成条件绑定对应目标及子任务。回放同时显示选项代码、完整选项概率分布及单独的 confidence 字段。

固定 single_arm_transfer_v1 在入口显式执行 observe_scene，capture_mode 为 reuse_current，由现有采集版本和关节反馈检查决定复用或新采集，然后进入 identify_target。模板入口要求空手、无需恢复且仍需交付。observed 与 grasp_view_ready 由入口观测建立，全部节点和失败边在展开 graph 中可见。

持物判断中，实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。state 说明接近闭合时仍可持物、背景与遮挡的区别、深色材料及历史图片角色。verify_grasp、verify_place 和失败收尾共用视觉定义。抓后身份从连续执行绑定继承，明确记录推定来源，不另发身份题。

所有 GraspGen 抓取按执行臂当前指端 TCP 选择较近的平行夹爪半转等价姿态，随后进入 IK 和完整碰撞检查。CAD 离线抓姿按库中 A/B 归属读取。

CAD 模型绑定采用 resources/cad_subject_policy.json 中的用户场景声明。已确认身份的 black/object 绑定黑色工件 CAD，blue/bin 绑定蓝盒 CAD，其余身份使用 TSDF。程序按该声明绑定模型并保留来源，FoundationPose 使用当前 SAM3 掩码和 RGBD 估计位姿。

持物和空爪题使用同次 TCP 与相机标定生成的 11 cm 夹持接触区 RGB，保留原始像素分辨率，不加入历史整幅腕图。当前图保持工作空间过滤、不依赖 SAM3、不带物体标签；投影不可用时退回完整当前腕图。失败收尾对两腕使用相同规则。历史参考仍用于实例身份与放置落点核验。

持物与空爪题的 state 同时提供当前采集的夹爪反馈，实测归一化闭合程度与全闭目标命令分别标注。0 表示机械全开，1 表示机械全闭，该值不是夹持力。实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。张开或闭合命令仅表示动作意图。实测归一化位置距机械端点 0.02 以内归为对应端点，其余归为 partially_closed。最终由 Decisions 综合图像与反馈判断，程序不做夹爪状态或运算。缺失或无效反馈明确标为不可用。夹爪反馈仅用于持物和空爪题；抓后身份记录为连续执行推定。

初始及历史任务描述绑定只从该描述声明或历史记录保存的 SAM3 主词及别名检索候选，候选是否对应实物仍由 Decisions 判断；没有候选的相机转查其他视角。每次建图重新输出所有当前未知任务实例的 TSDF 表面，并绑定同次观测与代次。持物、空爪及失败收尾使用按同次机器人指端 TCP 和相机标定生成的当前局部 RGB，不依赖目标检测结果；持物题不发送历史整图；当前局部投影不可用时才退回完整腕图。桌面区域放置验收投影执行绑定的指定区域；入盒验收以指定容器整体为目的地，控制台区域只约束机器人释放位置。

初始编排必须声明 support_table_refs，引用 entities 中实际可见且参与支撑测量的 table；配置区域不是桌子实体。程序通过当前或缓存实测桌面多边形，唯一绑定控制台区域及支撑高度。用户明确指定的机器人左右通过 robot_relative_side 单独编译；初始候选以同帧 RGBD 标定坐标核对（+X 向前、+Y 机器人左侧、-Y 右侧），再交 Decisions 选择。此约束只用于初始定位，不限制已确认永久 ID 之后的位置。

当前有限任务边界与证据实现见 [可靠性修正](docs/scoped-reliability-20261115.md)。未知物体 GraspGen 不再额外沿 Z 伸入；模型接触 TCP 显式转换到 A/B 指端，旧 grasp_insertion_m 保存值退役。当前只要求同臂覆盖来源与固定目的地、固定 CAD。物理记忆提交使用独立 PhysicalMemoryWriter，历史失败按实现/合同/子图/配置版本筛选，较早身份参考每实体最多 3 次采集。

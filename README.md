# Tianji Luna MuJoCo

当前验收范围为同臂覆盖来源与固定目的地、固定 CAD 和放置配置。GraspGen 已取消额外 30 mm 伸入，按 A/B 夹爪前端 TCP 分别转换。实现、证据版本和关联修正见 [有限任务可靠性](docs/scoped-reliability-20261115.md)。

当前 Decisions 只处理必要视觉事实，使用有效区域过滤 RGB 和按需深度图，以 SAM3 框和实例参考框标明主体。多候选的本轮目标由 Decisions 看过滤图选择；数值选臂、建图、明确恢复与标准任务回执验收由程序处理；任务许可与额外视觉要求开局编译一次。当前接口及验证边界见 [工作空间过滤与视觉判断](docs/raw-decisions-20261111.md)。



观测时序修正：接受的实例更新先提交数据库，再更新内存与任务账本；几何变化使碰撞场景待发布。当前支撑证据与永久 ID 分开，重新观测后的再次抓取需由在线 graph 显式恢复 `source_support_current`，不插入隐藏技能。历史失败核对与验证见 [历史故障和观测时序](docs/history-timing-validation-20261107.md)。

恢复动作由 Responses 编排、程序依据已知失败和数值结果选择有限分支。配置回位和满足前提的腕部换视角均可用于观测恢复；已在配置位姿时，应判断所选分支能否改变缺失证据。配置回位无需额外遮挡许可题，空手、有效地图和实际运动检查保留。编译校验反馈包含失败路径上最后一次前提变更及注册的事实产生技能；校验失败即终止；运行中不修订图，不改写已编译根任务。

正常主线的空手身份或支撑 unknown 校验观测恢复覆盖。单一路径可直接连接恢复技能，已知恢复按程序规则选择。数值抓取失败依据具体拒绝证据进入有界替代路径或结束。

精简编排输入中的技能 contract 同样包含观测恢复要求；生成前的末尾检查区分原样重规划与观测恢复。候选池 `pool_expansion_not_permitted` 仅限制技能内部扩展冻结候选池，不表示整个 graph 不可恢复。静态图合法与恢复覆盖充分仍需分别验收。

开局只调用一次 Responses，按作用域输出显式路由组 `sources/outcomes/target/max_traversals`。作用域公共失败去向由 `on_failure/on_unknown` 明确声明。程序逐项展开并生成边 ID，完整图继续经过拓扑、数据流、恢复覆盖和完成条件校验。产物登记保存同次采集的唯一测量掩码、框与位置参考，后续身份关联复用这些定位证据。编排经验库同时保存按契约筛选的真实终止技能失败，开局用于检查相应前提和恢复路径。

自然语言和三相机初始 RGB 输入 GPT-6 Luna Responses，结合技能契约、配置区域及 CAD 目录，生成物体记录和显式 graph。运行时由程序解释明确状态与恢复规则，由 Decisions 判断必要的当前视觉事实，由机器人 skills 完成感知、几何、规划与执行。

系统只提供在线编排模式。归档 graph 的展示与导出用于审阅，不提供直接加载成品 graph 执行的产品入口。

Skill 是带参数、前提和明确结果的稳定操作接口，内部步骤固定。Responses 在满足任务依赖后，优先编排可见抓取面暴露、遮挡少、周围间隙充足的物体。实际选臂、IK 与碰撞验证由对应技能完成。主要完整实验使用两个输入物体。

实例关联只覆盖自然语言任务中的目标、来源、目的地、加工区域及其必要支撑。场景描述用于定位和消歧。初始 Responses 提出任务实例，Decisions 先在描述参考相机的当前带标签 RGB 中选择一个掩码，再仅为这些选中主体关联其他相机掩码。其他同类掩码可作为选择项和画面上下文，额外可见物体不自动登记。后续观测使用任务实例的已确认带日期参考选择当前掩码，再关联其余视角并继承永久 ID。近期参考不足时使用更早的确认参考。all/count 任务按目标描述发现新增任务输入，确认新身份后登记。跨相机候选沿用源掩码的 SAM3 原检索词及别名召回，其他类别掩码保留为图像上下文；检索归属不证明身份。跨相机对应由 Decisions 比较同次带标签 RGB 与各候选独立的标定深度测量，提供固定世界坐标中的可见表面中心、范围、有效深度比例和空间关系。程序拒绝同次观测中明显分离的匹配，保留模型原始回答及独立拒绝依据，并检查同相机唯一性和传递分组一致性。先关联再融合，不做跨相机轮廓投影。跨时刻主体选择、归属复核和历史身份题同时接收有日期的深度测量与执行回执；历史腕图使用当时标定，并提供逐候选空间比较；相同观测 ID 与代次代表同一冻结采集，已确认身份的明显空间矛盾排除候选。真正跨时刻只使用位置先验，搬运后的身份不受原位置硬筛选。释放推定与当前实测位置分开记录。各主体先独立选择暂定掩码；跨相机物理分组后，使用当前多视角及日期参考做一次主体复核，允许选择其他掩码或明确不匹配；改变物理组时重新归组一次。复核不把初选当作已确认身份；同一物体存在多个归属提案时，用竞争描述和日期参考共同裁决，败选主体最多重选一次。已裁定掩码保留所属主体标签并从重选项排除，未解决冲突不提交。剩余候选仍需视觉确认，容器与内容物按各自掩码区分。历史选主体只发送包含该主体标注的参考视图。Decisions 的当前三相机 RGB 同时标明已确认视角，state 保存对应关系及来源。历史记录仅登记任务涉及的物体。碰撞场景覆盖工作空间内的可见几何，已匹配 CAD 使用 CAD，桌面使用实测 cuboid，其余使用 TSDF/ESDF。额外物体通过匿名深度建图，无需 Responses 命名或 SAM3 分割。已完成入盒物体保留入盒历史。直接确认和推测入盒统一保留永久 ID 与成功入盒历史；已完成交付的盒内物体不再逐件重新关联或 CAD 配准，当前可见内容物统一使用匿名 TSDF，盒子本体仍用 CAD。入盒历史不生成当前个体碰撞几何。每次 build_scene 都从当前观测重建整个工作空间：当前 CAD 位姿来自 SAM3 与 FoundationPose，其他可见几何来自当前 RGBD 的 TSDF/ESDF，桌面 cuboid 保留实测缓存。不选择局部范围，不合并历史物体碰撞几何或旧 TSDF 体素。历史身份与操作记录独立保留，未观测空间不因此被证明为空。测试任务不限定程序能力，也不产生任务专属建图规则。

## 当前实现

采集权限由技能契约 captures_observation 声明。observe_scene、verify_grasp、verify_place、verify_unpack 和 verify_task_completion 可采集。初始任务 bootstrap 采集冻结的 Responses 输入。运动、规划和 router 使用已有证据，程序 router 不调用模型，视觉 router 要求当前有效图像。网页相机面板显示最近一次授权观测。

observe_scene 的实时状态显示分割、桌面与候选几何、历史与跨相机关联、CAD 配准、任务实例深度整理、观测记录更新等内部阶段。阶段事件和已完成计时归档，`task_depth_preparation` 记录任务实例深度整理耗时。规划批次失败明细在实时事件中可为数组或 `{count, first}` 摘要，页面显示摘要条数，完整明细保存在实验原始数据。

编译器对契约明确要求空手的身份/支撑检查核对观测恢复覆盖。其他节点的有限恢复或明确终止由 Graph 作者声明。拆包姿态可能遮住容器，Graph 可通过建图、回观测位姿和重新观测恢复目的地绑定，已完成的加工回执与产物 ID 继续保留。

编排失败记忆对不同技能及前提事实分别保留案例，同一错误码下的模式冲突、观测缺失和绑定缺失可同时提供给 Responses。记忆投影保留事实依赖及最近失效节点，完整路径留在原始实验中。

execute_grasp 和 execute_place 均下发三段时间序列。放置在 place 段末尾松爪，随后执行张开、已 detach 的 retreat。

工作空间 RGBD 过滤复用离线包络，并保留同次采集中已登记外观的完整 SAM3 掩码。时间记忆登记要求有效工作空间深度。测量参考与 FoundationPose 按掩码内有效深度数量选择单相机，未知碰撞几何通过同次多相机 TSDF 融合。控制台提供容器固定释放区域和夹爪动作时长，默认 X=0.430 m、Y=-0.560 m、区域边长 0.12 m、夹爪开合时长 0.4 s。

点击“停止任务”调用 `/api/stop`，冻结当前电机目标并取消排队动作，保留场景与实验记录。

- **层间数据边界**。Graph 使用 `InitialPlan`、`TaskSpec` 和 `GraphSkillResult`。时间记忆数据库由 `ObjectMemory` 与持久化 `ObjectDatabase` 提供，保存实例及历史。空间记忆由 `SpatialMemory` 持有，内容为碰撞场景及其来源、版本、启用和有效性状态。原始观测、测量缓存、配置区域归执行层，任务账本记录本次执行责任。Decisions 的单题输入由 `DecisionQuery` 校验。回放分别显示时间记忆数据库的提交快照和空间记忆的碰撞几何。数据归属和传递时机见 [在线 graph 架构](docs/native-graph-implementation.md)。

  Graph 的 router 通过 `GraphDecisionRequest/GraphDecisionResult` 传递问题、绑定、原失败及有限分支，执行适配器不再反读执行器字段。执行层持有记忆服务；技能 context 只放数据，碰撞检查和释放核验接收 `CollisionSnapshot/ReleaseLedgerSnapshot`。加工证据函数返回 `ProductRegistration`，由执行层提交原有数据库事务。检查范围与证据见 [分层数据边界验证](docs/layer-boundaries-validation-20261107.md)。

- **逐题证据范围**。通用判断按用途和题目投影 state；来源剩余物体判断不携带完成账本；放置语言许可在初始编译阶段绑定。拆包和放置分别构建视觉输入，持物和空爪题同时接收同次实测夹爪反馈，最终由 Decisions 判断。全部 Decisions HTTP 题目经 `WireDecisionQuestion` 校验。程序恢复读取声明分支、剩余预算和原始失败。支撑明确否定与 unknown 分开返回。旧的运行中 Responses 视觉辅助入口已删除。审查范围及限制见 [Decisions 输入核对](docs/decisions-input-audit-20261105.md)。

- **Responses 编排与校验**。一次调用输出任务与显式策略图，固定模板由程序展开。逐物体子图的 completion_pose 在输出 schema 中限定为空。完整图校验通过后执行，失败即终止并归档错误。
- **编排失败记忆**。编译器把图校验失败按请求哈希持久化到 `data/runs/compilation-memory.sqlite3`，新任务读取与当前实现、合同、固定子图和冻结配置匹配的历史约束和反例。记录包含错误、原请求来源、相关技能及数据流反例，按错误族汇总进入 Responses。重置任务保留这份编排记忆。独立的数据流与恢复覆盖错误同轮报告，完整拒绝记录保留供同版本后续任务参考。输出 schema 先列根节点以声明任务子图接口，再填充子图定义；局部节点先于入口和边。验证见 [Responses 编译与失败记忆](docs/responses-compiler-validation-20261107.md)。
- **编排推理预算**。GPT-6 Luna Responses 使用 high 推理强度、64000 输出 token 上限（包含推理和最终 JSON），输出紧凑 JSON，并复用共享子图。单次编排超时默认 600 秒。每任务只允许一次 Responses 调用，失败终止。回放展示各版完整图及校验错误，运行时 Decisions 按最终图选择分支。恢复 router 用 rule 声明 visual、observation_recovery 或系统保留的 failure_return，并以 choices 声明可选 edge.on。明确恢复由程序规则选择声明的有限路径，未解决的视觉条件才调用 Decisions，无合法恢复时明确结束。
- **按需实例关联**。初次按任务描述选择主体，已有主体依据已确认历史图选择当前掩码，再关联其他相机。问题正文明确主体描述、ID 和参考标签。暂定选择共同检查冲突，单题裁决后败选主体最多重选一次，已裁定掩码保留标注并排除。同次各视角掩码的世界坐标深度测量进入 Decisions；明显空间分离阻止匹配，历史位置结合执行回执解释。近期参考不足时使用更早档案。身份确认后继承永久 ID，同次已接受观测通过事务提交。

- **Decisions 输入**。已接受的当前身份提供可见事实。物体记录完整性不作为执行前提，任务语义沿用已编译要求。新实例待确认提议独立反馈，碰撞依赖来自当前测量、保留及受动作影响的图层和当前操作对象。新增实例、身份不确定和恢复所需可见性按需独立判断。当前视图每相机一张，身份关联按需加入历史参考。身份选项保留实例的完整历史参考说明。相机、采集、模态及主体的文字元数据保持一致，哈希与来源审计留在程序侧。放置核验的独立问题共享三相机图像。程序恢复读取结构化失败；视觉 router 只接收本题所需的当前原图及绑定，完整规划调试数组留在实验日志。
- **永久物体 ID**：物体记录绑定实例及操作历史；CAD asset ID 表示模型，允许多个实例共享一个 CAD。graph 可用共享 subgraph 循环处理 $selected，执行账本记录实际选中输入的处理责任。
- **几何**：桌面使用 cuboid，未知物体用当前 RGBD 的 ESDF，已匹配 CAD 用 FoundationPose 当前位姿和 CAD。生产模式为 `cad_esdf`；未知抓取使用同次 TSDF 表面。
- **CAD 配准**：SAM3 掩码、RGBD、内外参和匹配 CAD 输入 FoundationPose；选择有效深度最多的一个相机，输出转换到统一世界坐标。执行层从物体操作历史与空间失效状态投影跟踪上下文，已知实例的物理上下文未变时，以上一帧位姿初始化当前帧 track_one；有变化时重新配准。不读取 graph 进度，也不将历史位姿当新测量。保留来源、标定及坐标绑定检查，不做额外几何残差复核或 ICP。
- **抓取**：有离线抓姿的 CAD 按执行臂选择对应抓姿；未知物体使用当前 TSDF 表面输入 GraspGen。选臂依据任务约束、工作空间和可达性，颜色不绑定机械臂。
- **容器抬升**：使用已发布 CAD/TSDF 图层范围计算邻近障碍高度，保留障碍同样参与；不再要求已消耗输入具有当前掩码。正式退役 ID 与物理几何分开处理。
- **attachment**：抬升成功且确认持物后建立；持物模型随臂、抓取变换、场景版本绑定。
放置 XY 使用控制台正方形中心，旋转保持实际抓取姿态。末端 Z = 实测桌面或盒口最高点 Z + 物体高度 + 控制台高度余量。物体高度来自同次 CAD/TSDF 几何竖直范围及来源支撑，独立于碰撞球。默认余量 20 mm，含为保守附着包络增加的 10 mm。若抓取 TCP 高于物体顶部，使用较大的 TCP 到来源支撑面偏移避免低估。实际物体底面另按抓取偏移校验，假设搬运中物体不滑移。碰撞几何独立检查目标及轨迹，不搜索目标高度。选臂时用可见高度估计，抓取后用已绑定 CAD/TSDF 再检查同一臂。
- **技能清理**：当前 21 个任务能力和 4 个系统保留收尾能力，包括显式空抓恢复 `recover_failed_grasp`；旧 `plan_transport / execute_transport / drop / verify_drop` 契约和执行器已删除，容器几何共用 `robot_skills/container_geometry.py`。
- **抓取碰撞一致性**。官方三段式接口按执行臂临时排除两侧 Link3、Link4 链接，接近段恢复，调用返回或异常时再次恢复。实际命令逐段检查环境、另一臂、非接触部件和目标接触条件。第三段完成后由 graph 视觉核验持物，通过后建立 TSDF/CAD attachment，回执绑定物体、执行臂、任务、采集和场景。
- **夹爪进入间隙**。未知物体抓取端点使用完整实测点云与张开接触垫碰撞网格凸包检查，计入配置的测量不确定性和碰撞余量。闭合预测使用轨迹端点 FK，越界反馈保留实际角度、命令、限位和稳定窗口统计。
- **轨迹**：调用官方 cuRoboV2-Fast v0.9.0（42498cf4），统一启用 GPU RRT、B-spline TrajOpt 和 TimeOpt。抓取向官方 goalset 提交当前候选，逐次验证实际选中项；已确定的放置目标使用 BatchMotionPlanner、batch=1。放置包含接近、放下及释放后的撤离。
- **抓取池耗尽**：未知物体的 TSDF 抓取候选经完整数值拒绝后，可在技能内部继续下一有界候选池；CAD/TSDF/cuboid 场景与双臂碰撞证据正确参与该判断。有限数值端点误差超限同样属于候选拒绝，允许继续既定额度的候选池；无效 FK、非有限数据或绑定不符仍停止。碰撞和端点门槛、CAD 离线池与 Responses graph 不受此搜索改变。
- verify_place 沿用 verify_grasp 的持物与目标身份记录，默认搬运中持续持物，不新增搬运中核验。释放后先读实际关节位置：全开为空爪，中间位置仍持物；仅全闭时调用 object_held 判断。确认空爪后，place_result 在 direct_success、inferred_success、failure、unknown 中选择。桌面与入盒统一验收实际到达配置释放 XYZ 并张开夹爪；控制台小方框只约束机器人释放位置，不要求物体最终落点位于小方框内。直接确认表示看见物体放在指定桌面支撑或进入指定盒子。推测成功要求已确认持有目标、实际到达配置释放 XYZ（现有位置容差 5 mm）并释放、当前空爪，且没有交付失败的可见反证，不要求一定被遮挡。命令发出不等于实际到达；缺失绑定证据为 unknown，明确落在目的地外为 failure。当前三相机 RGB、释放前 RGB 与日期身份参考用于放置结果题，不投影释放小方框作为视觉验收边界。盒内同形物体不要求区分新旧，可依据执行证据选择推测成功。保留原始回答、执行推定身份、placement_outcome 与回执；推测入盒不伪造个体当前位姿，confidence 仅归档。
- **判断结果**：区分命题概率不确定、事实选择低置信度、明确 unknown 和拒答；回放显示原始答案及程序采用的规则。恢复 router 保留完整判断证据。放置的任一必要条件明确不满足即为失败，不被另一项 unknown 掩盖。
- **来源绑定**。沿用 TaskSpec 的来源要求。current_support 独立核验当前物体的直接支撑或容器，完成时检查来源是否还有匹配输入。
- **观测覆盖**：观测成功不代表全部待处理物体都可见。腕部相机随动作改变方向；当前缺少目标时返回明确缺失结果，由 Responses 编排的可见性恢复分支处理，不隐式回位或重拍。
- **恢复证据**。连续 router 通过 last_failure 引用技能失败及绑定主体，last_result 保留紧邻节点的回答。抓取几何的可见性与实例识别分别提问。
- 批处理每件交付后优先在当前位置重新观测并选择当前输入；只有程序确认必要信息缺失时才沿 graph 回观测位姿补观测。all/count 使用共享子图，初始数量不限制运行时成员。最新观测确认来源盒已空即结束本批。最终双臂回位由实际关节反馈验收。VLA 起始姿态仍属于加工技能前提。
- **共享子图**：物理节点按输入永久 ID、声明的子图调用路径和动作节点防止重播，允许不同输入或不同声明调用复用同一子图；循环重入同一次动作仍拒绝。子图之间继承当前事实，开始新物体不会自动恢复观测或地图。
- **补充视角**：技能直接判断是否缺少任务所需视觉信息，裁切/遮挡等原因单独诊断；不因无法确定具体原因而否定已确认的信息缺口。概率按最大项取值，显式恢复边和运动检查保留。
- **图与回放**：按主图、任务子图、失败处理/恢复分组；正常路线与失败处理各自成图，跨视图入口与返回边可以跳转。保留完整节点和边，未执行节点保留灰色。每行从左到右，正常连续路线用虚线折行连接；跨节点分支用带目标编号的跳转连接，点击定位。未走分支为灰色，实际转移标次数。共享子图默认显示最近一次调用，可切换任一次或累计记录；第一件成功不会覆盖第二件未执行的后续步骤。节点本次失败、证据不足、已进入分支与任务最终结果分别说明，实际调用次序与当时物体 ID 单独记录；Responses 次数按开局编译/正式执行划分。碰撞场景用半透明差异色和归档名称，原始 ID 收进详情。
- **抓取失败语义**：`planning_no_valid_candidate` 只说明当前臂的已评估候选没有通过全部检查，不证明其他臂也不可行；同一抓放只允许原臂重新计算候选、再规划一次；再次失败进入统一回位收尾。

## 目录与职责

算法代码只有一个维护仓库：`../tianji-algorithms`。Luna 不再保留 `vendor/`、`services/`、`src/tianji_foundationpose/` 或算法注册库副本。

| 位置 | 职责 |
| --- | --- |
| `src/luna_mujoco/` | 应用编排、MuJoCo 仿真、场景与机器人技能；专用 `geometry_service` 保留应用接口 |
| `skill_catalog/` | 技能契约、参数和前提/效果 |
| `resources/robot/` | 本机器人碰撞配置与几何审计，不含算法代码或 SDK 物体记录 |
| `resources/`、`assets/` | CAD、标定、工作空间、URDF 和网格 |
| `scripts/`、`web/` | 应用维护工具、工作台与回放 |
| `data/` | 实验、运行日志、回放网格和应用规划参数 |
| `../tianji-algorithms/services/` | 唯一的算法客户端、协议与服务适配源码，统一包 `tianji-algorithm-services` |
| `../tianji-algorithms/manifests/` | 官方 SDK 版本及源码校验物体记录 |
| `../tianji-algorithms/data/luna/foundationpose-registry/` | Luna 使用的 FoundationPose 注册数据，独立于源码 |

SAM3、GraspGen 和 FoundationPose 的服务定义统一位于算法仓库的 `deploy/luna-algorithms.compose.yaml`；本项目 Compose 通过 `include` 使用它。应用及 cuRobo 容器只读挂载同一份算法服务源码，代码中不再临时插入 `vendor` 路径。FoundationPose 客户端和服务共享同一个 `tianji_foundationpose` 包；Luna 所需缓存/跟踪适配在算法仓库的 `services/luna_foundationpose/`。

算法仓库的通用 cuRobo HTTP 服务不替代 Luna 的场景/持物接口；Luna 的 `geometry_service` 从统一包导入算法边界。官方 SDK 与权重在算法仓库维护。cuRobo 镜像为 `jodell-curobo:0.3.2-guangyu-curobo-v0.9.0`。

独立 CAD 库位于 `/home/guangyu/projects/CAD`，通过 `LUNA_CAD_ROOT=/opt/cad` 只读挂载；`resources/cad/` 是应用静态副本。

## Git 上传与本地数据

源码、契约、部署配置及静态资源随 Git 保存。`.env`、运行数据、依赖和构建输出不上传；仅提交 `.env.example`。

Luna 的 `data/` 只保留 `experiments/`（实验回放）、`runs/`（运行日志）、`render_meshes/`（回放网格）和 `planning-settings.json`（应用规划参数）。旧调试脚本、离线候选实现、部署包、`tests/` 和 `web/tests/` 已按用户要求删除。FoundationPose 注册库已经逐文件校验并迁至算法仓库，Luna 不保留副本。

新机器需要同时准备本项目、相邻的 `tianji-algorithms` 与 `CAD`，以及本地镜像、`.env` 和 OpenAI 配置。注册库的准备工具为算法仓库中的 `scripts/provision_luna_registry.py`，显式传入 `--project` 指向 Luna；详细命令见 [注册库准备](docs/cad-catalog-provisioning.md)。历史文档中的旧测试、离线产物和迁移前路径仅记录当时状态。

2026-11-03 统一维护验收：算法仓库提交 `09c7330be8f3d441d580211e456ddaa48da35a53`；CPU 检查 41 项中 39 项通过、2 项跳过；现有镜像完成完整拆包交付仿真，实验 `63f27d77d9d24152bfe25a5137a6a380`，耗时约 264 秒。官方 SDK、机器人配置内容与已有实验记录保留。算法仓库目前只保存在本机，未配置 Git 远端。

## 启动与关闭

Ubuntu 在项目目录使用 Bash 入口，启动命令会等待所选服务健康就绪。

```bash
cd /home/guangyu/projects/tianji-luna-mujoco
bash start.sh
```

网页与算法服务分别管理。网页服务包括前端、API 与仿真运行时，算法服务包括 cuRobo、SAM3、GraspGen 和 FoundationPose。

```bash
# 关闭网页服务
bash scripts/web-service.sh stop

# 关闭四个算法服务
bash scripts/algorithms-service.sh stop

# 关闭全部五个服务
bash stop.sh
```

单独启动和查询状态使用同一入口。

```bash
bash scripts/web-service.sh start
bash scripts/algorithms-service.sh start
bash scripts/web-service.sh status
bash scripts/algorithms-service.sh status
```

两个分组脚本都支持 `start|stop|status|logs`，共享 `scripts/demo-service.sh` 的执行逻辑，并显式指定各自的 Compose 服务集合。

Windows PowerShell 打开工作台：

```powershell
Start-Process 'http://192.168.3.4:8350'
```

如果使用 SSH 本地转发，在一个 PowerShell 窗口保持连接：

```powershell
ssh -N -L 8350:127.0.0.1:8350 guangyu@192.168.3.4
```

另一个窗口运行 `Start-Process 'http://127.0.0.1:8350'`。工作台端口为 8350，cuRobo 服务端口为 8352。五个容器当前 CPU 配额总计 4 核。

## 开发与验证

重新构建应用镜像时，构建上下文必须来自统一仓：`docker build --build-context algorithm-services=../tianji-algorithms/services -f deploy/Dockerfile.web -t tianji-luna-mujoco:unified-guangyu .`。现有部署使用原镜像及只读源码挂载，本次迁移无需新建镜像。宿主维护命令可通过 `TIANJI_ALGORITHMS_ROOT` 指定非默认算法仓库位置。

开发环境先运行 `python -m pip install -e ../tianji-algorithms/services -e ".[dev]"`，随后运行 `python -m ruff check src --select F`；前端在 `web/` 中执行 `pnpm build`。测试目录已按用户要求删除，语法/构建检查、只读接口检查与完整仿真必须分别报告。普通任务可通过 `scripts/validate_demo_run.py --instruction "自然语言任务" --output data/acceptance.json` 验收。

新增技能需要契约、参数 schema 和 handler；不得在 controller 中添加固定任务流程。执行前检查观测/场景版本、身份、机器人反馈、工作空间与碰撞，不能用模型判断替代数值检查。未通过的判断沿 graph 失败或恢复分支处理，不伪造成功。

当前设计和证据：

- [重复仿真、原始 Responses 图及完整失败记录](docs/reliability-validation-20261104.md)

- [Graph 回放、全部问题族与 unknown 来源审计](docs/replay-question-audit-20261103.md)

- [Decisions 结果、恢复分支与最新完整仿真](docs/decision-outcomes-validation.md)

- [在线编排边界与本轮验证](docs/policy-boundaries-validation.md)

- [框架、模型输入与代码清理核对](docs/framework-cleanup-20261103.md)
- [当前 CAD 与落点策略](docs/cad-runtime-policy-20261102.md)
- [显式 graph 与运行恢复](docs/native-graph-implementation.md)
- [Decisions 问题协议](docs/decisions-questions.md)
- [选臂、attachment 与放置性能](docs/arm-attachment-place-audit-20261103.md)
- [低面数 CAD 与 GPU 审计](docs/lowpoly-and-gpu-audit-20261103.md)

带日期的旧实验文档只记录当时实现，不代表当前接口或当前要求。

已授权整区的放置核验按独立问题裁剪上下文，夹爪反馈只保留在执行层，不发送视觉夹持问题。释放后缺证据的恢复由 Responses 在图中声明，保留原始释放回执并只重新核验。

永久实例 ID 由 data/object-memory.sqlite3 从 1 单调分配。重新观测及同一物理场景的新任务复用已确认 ID，加工产物使用同一分配器。仿真重置创建新物理实例，历史编号继续保留。物体记录记录状态和操作历史，graph 显式选择当前目标。可通过 /api/objects 与 /api/objects/{id}/history 查询。界面直接显示物体记录 ID。身份关联保留三相机证据及每台有效深度计数，度量几何使用选中的单相机。

已交付物体继续通过统一的实例关联流程更新几何，结合当前多视角候选、历史参考图和操作记录。目标区域核验图保留原始分辨率，使用边界旁的标签。

身份请求明确区分历史观测与当前采集，同帧已核验的放置记录作为关联证据保留。

每次 build_scene 从当前观测重建全工作空间，不提供 scope 或 entity_ids 参数。删除局部区域选择、旧 TSDF 合并和历史物体碰撞层保留；历史记录仍保存身份、观测和操作。

机器人外观使用灰白色，腕部相机使用中性灰色。Responses 物体记录及 Decisions 实例关联的主体范围为外部物体、容器和支撑面，排除机器人本体、关节、夹爪及随机器人安装的相机和线缆。

身份关联前，使用已有 URDF 自身表面射线匹配、采集时的关节与夹爪反馈、三相机标定 RGBD 过滤机器人返回点。物体与容器候选都参与。保留真实接触歧义和原始 RGBD，过滤只确定机器人表面归属。

当前支撑关系通过一次 Decisions choice 在已登记支撑物中选择，保留 none_of_listed 与 unknown。同一采集复用已接受结果。机器人自身过滤在身份关联前执行一次，过滤证据沿用到几何输出，原始轮廓用于后续边缘处理。

身份关联复用当前 Responses 的对应提议，由单题 Decisions 核验。历史位置明确标注过去观测，候选框定位与已确认身份分别记录。

当前支撑提议或带日期的历史关系可作为一次当前关系谓词的待验证假设；明确 no 后才排除该候选并选择其他支撑，unknown 原样返回。没有假设时使用支撑选择题。每次 Decisions 请求只包含一个问题，原始概率及来源证据保存到对应物体记录。

已确认容器的跟踪参考独立于其他物体的可见性和物体记录增量，当前深度、掩码与身份仍按原测量条件验证。

视觉缺证据的有限恢复覆盖目标、目的地绑定及释放后核验。Responses 显式编排所需建图、移动、分析和重试检查点。配置观测回位无需额外视觉缺失谓词；额外腕部视角才独立核验当前视觉缺失。中间建图成功保留原失败，原检查点重试成功后解除活动失败，历史事件保留。释放后只能恢复证据并重试核验，不能重复已经完成的物理效果。

选择下一件会解除上一件的操作绑定，历史与有效地图继续保留。换视角后的测量缺失由技能返回 unknown，并沿图内分支处理。

身份请求传递带原始观测编号的历史实测位置，与当前候选的世界坐标共同提供身份关联证据。未见、提议更新和新任务保留原测量时刻，接受新的 RGBD 测量后才更新该记录。

verify_unpack 的产物出现与包装消失分别使用一次 Decisions 请求，每次共享拆包前与当前三相机图像。两项事实可以由不同相机提供证据，明确否定返回 rejected 并沿 failure 边，缺证据返回 unknown。程序继续校验动作回执、采集时序、目标绑定与测量来源。

verify_unpack 内部采集当前三相机 RGBD，并建立加工结果与产物 ID。后续 observe_scene 使用已登记的产物身份获取掩码与几何。

编译阶段把 TaskSpec 要求的最终位姿纳入完成核验前提及图成功出口校验。抓取、放置和加工动作清除旧观测位姿事实，运行时继续使用实际关节反馈验收。

身份关联的代表视图优先采用当前 Responses 提议中与 SAM3 候选框对应的相机。各相机分别关联永久实例，按已接受的相同 ID 汇总身份，选择有效深度最多的单相机测量几何。过期提议不参与当前视图优先级。

### 运行期职责

ResponsesCompiler 只编译初始任务与图。ObjectRegistrar 登记永久实例，compiled_task 仅绑定编译结果中的引用。observe_scene 以 RGBD 与物体记忆完成关联和几何测量，select_target 维护本任务的输入选择与进度，build_scene 更新空间几何。感知不接收 graph，也不要求 TaskSpec。每次身份关联提供当前掩码轮廓和标定深度派生信息。中途物体关联与新实例登记使用 Decisions。

2026-11-06 全量 Decisions 与层间数据流审计见 [审计报告](docs/reliability-audit-20261106.md)：当前身份关联仍是主要视觉不确定来源，该审计当时发现的任务账本回写已在后续修复，最新边界与验证见本轮报告；不能将四层目录划分等同于完全解耦。

技能失败进入 Graph/Decisions 时保留完整数值拒绝证据。放置失败绑定控制台指定的唯一目标及当前场景，后续恢复遵循 graph 中的显式路由。

视觉核验同样保留每题概率、判断和观测/请求绑定；不能只传错误名。位置问题优先展示目的区域投影视角，全部相机仍保留。当前编排与恢复实测见 [本轮报告](docs/compiler-recovery-audit-20261106.md)。

释放后目的关系 unknown 不等于禁止补充观测：原释放/开爪及撤离回执和包含该目标的释放后当前碰撞地图可建立空手移动前提。graph 显式安排必要感知/建图及回位，继续核验原释放；不能以移动代替放置确认。

选臂与数值规划可由 graph 的有限分支重新调用，不能误当成已完成物理动作拦截。失败规划的缓存禁止已删除；每次重新求解仍检查当前场景、持物、碰撞和轨迹，执行动作继续防重复。

时间记忆更新只读取当前操作回执：抬升开始使旧当前位置失去保证；开爪只记录机器人事件，绑定一致的 Decisions 放置核验确认新位置并记录转移。任务账本不再覆盖物体记录物理字段；新观测中的旧物体记录副本也不能覆盖当前操作历史。恢复对同一检查点和选中输入最多一次，恢复中失败/unknown 或重返检查点后再次失败均显式结束，不恢复失败的恢复。

身份细节图使用保留周围区域的原始 RGB，不将轮廓叠加到紧裁剪里；当前全景继续提供 SAM3 掩码、候选标签及三相机视图，深度与历史状态不变。避免小目标的外观和接触/分离关系被裁剪、描边遮蔽。

固定/精确 XY 的抓取前放置端点筛选与正式放置共用冻结实测表面质心作为物体位置参考；只筛选该指定 XY，不以区域内另一位置作为通过证据。普通区域继续原有有限分区。该筛选只检查固定夹爪端点，不宣称全臂 IK、持物或完整轨迹已通过。

拆包视觉核验比较整个实测桌面的前后变化，产物不必与原包装同位置。产物新出现与包装消失仍分别请求 Decisions；返回原因列出缺失或否定的具体条件，阈值和动作/身份绑定不变。

identify_destination 已确认的中间转移许可，在目标、目的地引用、任务、采集和区域配置完全一致时由 perceive_placement_zone 复用；不再当作最终目的地重复核验。区域内部语言限制仍独立判断，变化或未确认的许可不复用。

跨视角身份输入补充同次采集的实测表面投影：将已确认实例的三维点投影到待判断相机，与当前 SAM3 掩码及深度比较。Decisions 仍独立判断身份，不增加问题，不用几何匹配代替永久 ID 裁决；部分表面覆盖与完整轮廓 IoU 明确区分。

新产物在首次掩码定位前，保留并传递实际创建核验命题、原始答案、单题请求哈希及对应加工前/后 RGB。引用按实例、输入对象、观测、代次和图像哈希绑定。Decisions 仍独立判断当前候选是否对应该产物；创建成功不自动接受候选身份。已有带位置参考的身份被确认后，不再重复发送创建对照图。

时间记忆按事实逐项更新：视觉确认的支撑不因子区域核验 unknown 而丢失；开爪事件不能清除已有视觉支撑。身份请求明确区分初始同采集提议、已确认同采集实例、历史实例与产物首次定位。未见只更新 visibility；ObjectRetirement 需要明确来源、观测和回执，移除/消耗保留历史及 ID。removed 与 consumed 均退出当前身份候选，保留退出原因与历史；不自动恢复退役 ID。暂时未见的活动实例仍可正常跨观测关联。空间记忆用退役 ID 更新活动几何集合，不以未见直接删除碰撞对象。

VLA 效果定义统一来自 resources/vla_unpack.json 的 effect：粉色包装输入、独立黑色产物、视觉确认后消耗输入。动作回执不代替视觉事实。ObjectMemory 的退役接口供明确的执行回执/操作者确认使用；本次不增加“看不见即删除”的自动判断或隐藏 graph 节点。

产物创建保留输入所属场景归属。持久化以当前 ObjectMemory 为提交主体，数据库查询结果不反过来筛掉尚待写回的事实；原始场景归属不能被普通状态更新清空。

当前身份关联只处理活动物体记录。确认退出的实例保留历史，不再自动参与重识别；后来无法关联活动实例的物体，经过独立新增确认后登记新 ID。未见不等于确认退出，也不因此分配新 ID。

机器人动作记录在抬升开始、夹爪反馈和开爪检查点写入。抬升开始使旧当前位置失去保证。verify_grasp 在 graph 显式节点采集，沿用已验证轨迹的永久实例绑定，由 Decisions 根据当前夹持接触区 RGB 和同次夹爪反馈判断持物，确认持物后从一致的连续执行记录继承身份。原始持物回答与执行身份推定分别保存，并绑定本次动作和永久实例。支撑与目的关系由放置视觉核验更新。输入退役与产物登记使用同一数据库事务。unknown 保留证据不足的状态，观测恢复按原失败和运动前提选择。

身份参考持久化 RGBD、标定和各实例的跨相机掩码。跨时间关联先使用最近已确认的带日期参考选择当前掩码，近期参考不足时再使用更早的确认参考；当前跨相机归组只使用同次标定与深度，旧位置不作为当前位置。Decisions 比较外观与历史，提交永久 ID 前再进行当前多视角复核。旧 RGB-only 记录不补造深度或掩码。回放顶部展示实际恢复事件的触发、选择、子图步骤、返回与返回后结果，点击可定位对应调用。


夹爪命令稳定等待使用完整窗口内的位置极差和时间加权速度均方根，保留原位置/速度阈值、窗口与超时。记录窗口统计及瞬时最大速度，接触微振不再反复重置瞬时低速计时；持续运动仍返回失败。该检查仅属于执行层，不推断物体记忆状态。

身份问题明确写出当前 mask、待比较永久实例和该实例的日期参考描述。已确认同采集位置与历史位置分开措辞；容器与内容物即使框重叠也不是同一实例。RGB、掩码、深度和状态不变，只修改问题表述；二值结果按较大概率采用，正例与反例共同验证。

Responses 只编排已注册技能并填写其任务参数，不能修改技能实现、抓姿补偿或数值规划设置。generate_grasps 不接受模型调参字段；CAD 离线抓姿默认额外伸入 0mm，未知物体的 TSDF/GraspGen 抓姿不再额外伸入。GraspGen 模型接触 TCP 转换到 A/B 各自真实指端，按前方至下方的朝向范围筛选，再执行可达性和完整碰撞检查。未知物体旧伸入参数已退役。


空抓恢复要求 verify_grasp 的视觉持物判断明确为否，并校验目标、臂、代次和动作 epoch。recover_failed_grasp 清理临时执行状态，后续观测、建图和新抓取由 graph 声明。部分执行失败及视觉 unknown 保持不确定状态。

编译恢复覆盖检查：正常主线中，契约明确要求 grasp_held=false 且可能返回身份/支撑 unknown 的检查点，需要显式观测恢复路线。单一路径直接路由，多方案使用 Decisions 选择。路线可由调用方子图负责；恢复专用检查点不要求再次恢复。检查器不插入节点、不改图、不推断物体状态；仍独立检查技能前提与边界次数。


### 放置模式与恢复输入（2026-11-06）

execute_grasp 和 execute_place 均下发三段时间序列。放置在 place 段末尾松爪，随后执行张开、已 detach 的 retreat。

恢复选择输入同时包含 inline 成功续接路径、各结果边及子图定义，显示分支入口剩余次数；Decisions 只选已声明分支。初始 prompt 保存已验证的通用编排经验，区分规划失败与部分执行后的不同前提，不保存固定任务图。

抓取第二段及闭合的接触许可限定为当前目标与执行臂两侧 Link3、Link4 的配对，球索引从当前机器人配置读取。Link2 全部球参与目标碰撞检查。这些链接与其他环境物体、另一臂、其余部件与目标及自碰撞继续使用原数值检查。

夹爪基座各 56 球，Link2 每件 5 球、Link3 每件 10 球、Link4 每件 14 球，整机 438 球。拟合重点为两指内侧及虎口，机械臂主体采用经自碰撞核对的合并包络。A/B 离线工作空间使用相同模型及 cuRoboV2-Fast v0.9.0，加载时校验模型指纹。几何来源与拟合记录见 `resources/robot/compact-gripper-coverage-20261110.json`。

多个当前合格输入通过 select_target 的单题 Decisions 按任务与当前视觉、测量信息选择。单候选和明确顺序直接处理，后续规划继续验证数值可行性。

choose_arm 使用明确指定的臂与当前数值可行域筛选，优先选择没有同目标失败记录的臂，再按当前 TCP 到目标的距离排序，记录各臂代价。姿态 IK 与路径碰撞继续由规划检查。

释放核验的历史外观裁剪绑定该实例参考的采集、相机和框，使用与参考一起保存的 RGB。

已确认放置的时间记忆同时保存本次核验区域的范围、中心、支撑高度和布局版本。实例关联按采集时序读取这份历史位置约束，与当前候选的实测坐标及图像共同判断。

## 动态批处理与 Decisions 边界（2026-11-08）

all/count 任务采用共享处理子图。每件交付后显式回配置观测位姿，再调用 fresh observe_scene 和 select_target。初始 target_refs 保存初始候选，新观测登记的匹配实例可以进入同一循环。all 以当前来源盒确认清空结束本批，count 按请求数量验收。循环预算独立于初始数量，预算耗尽返回失败。

Decisions 负责实例身份、CAD 选择、任务成员关系、语义位置、来源为空的视觉判断，以及多种合法恢复方案之间的选择。单个候选、已绑定参数、数值规划结果、反馈和版本检查、唯一恢复出口直接由程序处理。抓取、工作区放置和容器入盒均由显式视觉核验节点确认。执行回执提供动作绑定，视觉判断提供夹持、支撑或包含关系。

时间记忆保存入盒回执、永久 ID 和历史测量。确认入盒后清除目标旧位置的独立碰撞层，并标记目标及容器区域待更新。graph 显式观测和全局重建后，可见内容物由当前匿名 TSDF 表示，不合并历史体素，未知空间保持未知，个体最终位姿保持未测量。

完全相同的路由声明按 source、outcome、target、max_traversals 合并，次数保持原声明值。不同目标或不同次数仍返回冲突。verify_place 显式声明释放模式，工作区轻放的地图失效前提继续传递，普通盒上释放采用独立的条件性占据发布结果。

2026-11-08 动态批次验收通过。实验 decd9032383645aba596d82be5f7917c 完成两件拆包交付、每轮回位重观测及空盒结束。独立验收确认两件产物在右盒内且双臂最终回到配置观测位姿。总耗时 596.528 秒，Responses 两次均在执行前，运行期为 0。完整记录见 [动态批处理验收](docs/native-graph-implementation.md)。

后续拆包核验用已交付黑件的实测位置或入盒历史限定身份排除区域。容器边界仅用于该身份范围，碰撞几何由实测 TSDF 表示。当前目标几何仍由当前采集建立。


支撑谓词使用单题 A/B 标签，并在结构化输入中保留永久 ID 映射。支撑判断在当前 RGB 上标出被询问物体，并将已测桌面边界按当前相机标定投影为带永久 ID 的多边形。标注保留原桌面测量日期，清晰的三维容器关系与已配准 CAD 桌面接触由程序判断，歧义关系交给 Decisions。优先展示同时定位物体与支撑参考的视图，保留全部三相机画面。

按需关联的当前完整验收见 docs/native-graph-implementation.md。实验 1681a9c6b1634502bd54ce05d54acc7c 的两件入盒、谱系及双臂回位独立检查通过，开局 Responses 1 次、运行期 0 次。扣除 Responses 后耗时 436.599 秒，当前性能目标与稳定成功率仍需继续验证。

选臂遇到已登记 CAD 目标和当前已发布容器区域时，先对每臂完整离线抓姿库复用方向及搬运可达性筛选，使用本次释放模式与控制台 XY。候选全被拒绝的臂退出本次选择。通过后继续执行原有 IK 与轨迹检查。目的地当前几何缺失时保留待验证状态。

显式观测阶段在共享 GPU 可用显存低于 8 GiB 时释放空闲 cuRobo 求解器缓存，后续规划按原配置重建。实际验证回收约 11.4 GiB，黑件 FoundationPose 与求解器重建后的完整抓取数值检查通过。

放置核验通过当前相机中的目的 ID 几何投影定位已绑定容器或支撑面，采用稳定的种类和颜色描述。到位核验判断当前支撑或入盒关系，目标释放使用独立问题。初始描述中的其他物体和图像方向保留为历史信息。

## 2026-11-10 RGBD 操作证据


实验归档保存 graph、视频、观测图、时间记忆数据库、空间记忆和原始输入输出。RGBD、掩码、点云及轨迹数组按内容哈希共享，模型请求图片按原始字节共享。事件索引引用已有观测、模型调用和空间快照。机器人反馈随观测与执行数据保存，两路视频成功编码后清理中间帧。

二值事实判断按较大概率输出结果，P(true)>0.5 为 true，P(true)<0.5 为 false。概率仅作为审计证据，精确平分、拒答或无有效响应保留 unknown。


当前统一失败收尾遵循 [终止失败回位](docs/terminal-failure-return-20261111.md)：具体恢复耗尽或未处理异常均执行显式固定收尾图，从实际关节规划回观测位姿并停止，保留原任务失败；不安全或无法确认的回位报告 blocked。

观测恢复校验接受 graph 中显式声明的 observe_scene(capture_mode=fresh) 当前位置补观测；不强制先移动。reuse_current 只有在移动已使旧观测失效时才满足补观测要求，单纯重试和重用不变证据不能通过。回放的时间记忆页签及元素名称与执行脚本、样式同步；实验列表必须在真实浏览器完成初始化验证。

回放默认优先完整执行图，其次完整 Responses 编排；系统失败回位图单独保留，不覆盖任务图。固定模板名称和输入从当次归档原始声明读取，不使用当前模板库重建历史；“完整图”可展开全部子图及未执行的失败恢复节点，刷新保留展开状态。登记使用对应相机带框过滤 RGB 和明确描述，每个 ID 选项附物体描述，已有 ID 也须当前视觉确认。observe_scene 每题用 SAM3 框标明当前候选，主体相机优先并标明头部/左右腕；历史参考使用过滤 RGB、日期、实例框和物理历史，不混入其他当前候选或历史剪裁掩码。SAM3 掩码继续用于几何测量。程序边界及验证见 [Decisions 与数据库审计](docs/decisions-database-audit-20261111.md)。

抓取规划按实际选中项记录计划计数与未验证索引。数值检查拒绝的候选退出冻结池，后续候选继续参与规划。

## 2026-11-12 association and planning update


Unknown-object GraspGen sampling uses one native 100-candidate pool per round, with one graph-controlled additional round. Each cuRoboV2-Fast grasp request contains the configured grasp_goalset_size ranked candidates. Fully rejected candidates are removed, later batches remain available, and the first grasp that passes all numerical checks ends planning. After a confirmed table release, voxel geometry retains its occupied cells with stale provenance until graph-controlled RGBD remapping.

Command-validation profiling isolated 15-22 seconds in full closure-range lift sweeps after an early collision. Validation now stops on a proven invalid chunk and records partial evaluation. Successful paths still complete all checks. Coverage bounds use the same numerical inputs in bounded chunks, including each next endpoint.

Attachment follows completed Lift and Decisions holding presence and continuous execution identity inheritance. Planning and executing the entire grasp sequence use the detached robot and the official grasp-planning open-gripper model, with a separate closure check. The executing Link3/Link4 links retain the selected-target contact policy during grasp and Lift. Other robot/environment and self-collision checks remain active. Successful verification prepares carrying geometry for later transport and placement. The earlier pre-Lift virtual carrying and opening-range path sweep implementation has been removed.

Responses 每个任务只调用一次，解析或图校验失败立即结束编译。固定模板由程序展开，逐物体子图的 completion_pose 在输出 schema 中限定为空，最终回位由根任务声明。执行采用三倍时间伸长和五次 Hermite 插值。MuJoCo 关节目标以 50 Hz 更新，物理积分与跟踪误差检查按原物理步运行。

## 编排边界与观测回放

Responses 编译在模板展开前校验实际输出 schema，展开后校验技能参数、字面实例引用、拓扑、前提、结果绑定、有限恢复和完成事实。缺失的肯定前提按运行时的 false 默认值处理。compact 模式继续提供各注册技能的完整 SKILL.md，只压缩合同中的重复字段。固定库包含观测建图、回观测位姿、空抓恢复、单臂搬运四类模板。单臂搬运接口声明 grasp_view_ready 和 needs_place，调用方必须建立这些事实。复杂批处理和加工流程通过受相同合同约束的显式子图组合。

plan_grasp 的 success_count 仅统计通过完整执行检查的轨迹，solver_success_count 单独记录 SDK 求解结果。request_index 表示实际请求次数，candidate_group_index 表示按 grasp_goalset_size 划分的候选组。已拒绝的候选从原组移除后再求解，首条完整通过的轨迹结束规划。

所有视觉、分割和建图输入严格使用同次工作空间掩码。分割掩码与工作空间取交集，场景外物体永久屏蔽。

实验回放页签依次为 graph、视频、decisions、数据库、碰撞场景、原始数据。decisions 按观测及实际请求展示发送图片，每排四张，保留请求内顺序。state 文本、图片元数据、questions 和回答可逐条核对。原始传感器数据与程序叠加归档从原始数据页读取。

verify_grasp 在 Lift 后先读取同次实际关节位置：全开为未持物，中间位置为持物；仅全闭时调用一次 Decisions，输入当前夹持接触区原始分辨率 RGB 和实测反馈，不调用 SAM3。明确未持物允许空抓恢复；请求异常、拒答或证据不足保留 unknown。确认持物后，程序核对连续抓取记录的目标、执行臂、运行、代次、来源观测、场景及动作版本，继承抓取前已确认目标的永久 ID，再建立 attachment。identity_binding 明确标记 continuous_grasp_execution 和 visual_identity_verified=false；这属于执行过程推定身份。执行记录不一致时保留持物结果与身份未确定状态，不建立 attachment。历史参考保留用于抓取前关联与放置核验，正常主线不再单独做抓后身份题。

控制台参数编辑器仅列出可修改项，cuRobo 固定默认值及观察位姿标定由后端保留。失败回位在确认空手后检查并执行开爪，以实际关节和夹爪反馈验收。持物回位保留夹持与 attachment。

MuJoCo 夹指接触几何使用等效橡胶参数 friction=[1.5, 0.01, 0.001]、condim=6、priority=2。机械臂位置伺服刚度为 60000 N·m/rad，阻尼为 600 N·m·s/rad；夹爪驱动力矩上限沿用 URDF。参数依据 50 Hz、三倍时间伸长下的抓取与携物回放调整，真实材质与驱动特性尚待实机标定。

未知物体的 attachment 使用抓取前已发布的同次多相机 TSDF 表面体素，将每个体素的八个角点按闭爪时实际 TCP 转到夹爪坐标，再生成覆盖全部体素的碰撞球。放置足迹沿用同一组体素角点。来源绑定采集、代次、永久 ID 和体素尺寸，缺失或过期时报告失败。已知 CAD attachment 使用已配准 CAD。Lift 完成、持物判定成立且连续执行绑定通过后建立 attachment，身份按执行记录继承。

GraspGen 候选先按真实末端 +Z 在世界坐标中的朝向筛选，允许世界 +X 前方至 −Z 下方的扇区，偏离世界 XZ 平面的左右角度默认最多 30°，控制台可调范围 0–45°。CAD 使用离线抓姿。规划完成后对实际抓取端点 FK 再核对相同朝向条件。抓取 Lift 显式指定世界 +Z 150 mm，并用公开 FK 检查每个规划插值点的横向偏移和提升高度，采用已有 IK 位置容差 5 mm。方向筛选证据、逐候选通过结果和 Lift 位移核验随规划归档。Lift 完成并通过持物核验后建立 attachment。

Decisions 输入以实际发送图片的顺序编号，每张图标明相机角色、观测 ID、代次、时刻、CURRENT/历史/静态目录角色及归一化框坐标约定。当前图片与请求采集绑定检查通过后才发送。跨相机选项描述给出 C 标签、相机和该图中的框坐标，相邻时刻 P 选项给出历史采集及对应图像位置。初始分组和 CAD 选项分别描述当前标签与目录标签。裁剪和单主体掩码元数据只列出本图主体。支撑关系问题与实际发送的 RGB 轮廓一致，放置核验提供目标和目的地的带日期参考，逐物体完成条件绑定对应目标及子任务。回放同时显示选项代码、完整选项概率分布及单独的 confidence 字段。

固定 single_arm_transfer_v1 在入口显式执行 observe_scene，capture_mode 为 reuse_current，由现有采集版本和关节反馈检查决定复用或新采集，然后进入 identify_target。模板入口要求空手、无需恢复且仍需交付。observed 与 grasp_view_ready 由入口观测建立，全部节点和失败边在展开 graph 中可见。

持物判断中，实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。state 说明接近闭合时仍可持物、背景与遮挡的区别、深色材料及历史图片角色。verify_grasp、verify_place 和失败收尾共用实测位置优先、全闭才看图的规则。抓后身份从连续执行绑定继承，明确记录推定来源，不另发身份题。

所有 GraspGen 抓取按执行臂当前指端 TCP 选择较近的平行夹爪半转等价姿态，随后进入 IK 和完整碰撞检查。CAD 离线抓姿按库中 A/B 归属读取。

CAD 模型绑定采用 resources/cad_subject_policy.json 中的用户场景声明。已确认身份的 black/object 绑定黑色工件 CAD，blue/bin 绑定蓝盒 CAD，其余身份使用 TSDF。程序按该声明绑定模型并保留来源，FoundationPose 使用当前 SAM3 掩码和 RGBD 估计位姿。

持物和空爪题使用同次 TCP 与相机标定生成的 11 cm 夹持接触区 RGB，保留原始像素分辨率，不加入历史整幅腕图。当前图保持工作空间过滤、不依赖 SAM3、不带物体标签；投影不可用时退回完整当前腕图。失败收尾对两腕使用相同规则。历史参考仍用于实例身份与放置落点核验。

持物与空爪题的 state 同时提供当前采集的夹爪反馈，实测归一化闭合程度与全闭目标命令分别标注。0 表示机械全开，1 表示机械全闭，该值不是夹持力。实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。张开或闭合命令仅表示动作意图。实测归一化位置距机械端点 0.02 以内归为对应端点，其余归为 partially_closed。程序直接判定全开与中间位置，仅全闭时由 Decisions 看图判断；程序判定独立记录来源，不伪造模型回答。缺失或无效反馈明确标为不可用。夹爪反馈仅用于持物和空爪题；抓后身份记录为连续执行推定。

2026-11-14 验证修正：初始及历史描述按其 SAM3 主词及别名检索候选，再由 Decisions 确认身份；全局重建输出当前采集的全部未知任务实例 TSDF 表面。持物/空爪输入使用根据自身 TCP 与相机标定确定的当前局部 RGB，持物题不发送历史整图；局部投影不可用时退回完整腕图，默认不新增深度图。桌面落点题显示指定区域；入盒按整个目的容器验收，配置小方框仅约束释放位置。

初始编排必须声明 support_table_refs，引用 entities 中实际可见且参与支撑测量的 table；配置区域不是桌子实体。程序通过当前或缓存实测桌面多边形，唯一绑定控制台区域及支撑高度。用户明确指定的机器人左右通过 robot_relative_side 单独编译；初始候选以同帧 RGBD 标定坐标核对（+X 向前、+Y 机器人左侧、-Y 右侧），再交 Decisions 选择。此约束只用于初始定位，不限制已确认永久 ID 之后的位置。

深度关联、统一放置验收与盒内匿名 TSDF 的当前实现和实验结果见 [验证记录](docs/depth-association-20261115.md)。

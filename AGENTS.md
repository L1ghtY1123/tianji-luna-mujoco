# Luna 开发约束

只有双臂及三相机采集时的实测关节均在配置观测位姿 1° 范围内，才建立全局碰撞场景。已有完整底图时，回到观测位姿也优先局部更新。初次建全图或显式请求完整测量时才核验观测位姿。局部更新关联同一实验的上一版地图。局部图补入当前可见的障碍和空间体素。跨控制代次，只有核对本次实验的停止回执、物理世界和父图身份后，才可复用历史几何。新代次的运动仍需重新发布地图和规划。未见且无移动记录的物体沿用原碰撞层。操作或观测已使位置失效的物体保留旧障碍并等待重测。首次局部观测只初始化部分地图。

本文仅描述当前合同；历史实验不是开发指令。修改前阅读 README 与 `.agents/skills/luna-policy-isolation/SKILL.md`。用户最新明确要求优先。

## 工作边界

- 只在 `C:/project/tianji-luna-mujoco` 和 `/home/guangyu/projects/tianji-luna-mujoco` 工作；独立 CAD 库 `/home/guangyu/projects/CAD` 与统一算法库 `/home/guangyu/projects/tianji-algorithms` 已获本任务维护授权。不要创建平行项目或工作树，不使用 sub-agent，不修改其他用户服务。
- 目录边界：算法客户端、协议、服务适配、SDK 物体记录和注册数据只在 tianji-algorithms 维护，Luna 不保留 vendor/services/算法客户端副本。Luna 的 src 放应用和 Base 接入逻辑，resources/robot 放机器人配置，assets 放静态模型，scripts 放应用维护工具。Luna data 只保留实验、运行日志、回放缓存和应用规划参数；正式入口不得加载其中脚本。
- 官方 SDK、安装包与参考工程只读；算法只通过匹配固定版本的公开接口调用。所有涉及 cuRobo 的技能统一使用 cuRoboV2-Fast v0.9.0，规划工厂启用 官方默认求解器配置。源码基线见 `../tianji-algorithms/manifests/curobo-sdk-files.json`。不能修改 SDK 或省略碰撞检查来提速。
- 五容器合计最多 4 逻辑核：web 1.5、cuRobo 1、SAM3/GraspGen/FoundationPose 各 0.5。OMP/MKL/OpenBLAS 单线程，web 软件渲染最多 2 线程。本地重检查串行。可增加本项目 RAM/VRAM，以实测收益与不 OOM 为准。
- 替换功能时同步删除旧入口、实现、专属配置、无效依赖和专属测试；当前仓库按用户要求不保留 `tests/`、`web/tests/` 和测试专属依赖；用静态检查、构建、实际入口及按需完整仿真验证。检查动态注册、前端、部署副本，远端同步包括删除。历史实验留在 data，不能靠历史兼容分支延长旧框架寿命。

## Base 接入边界

- 机器人、夹爪、RGBD 与加工仅经 tianji_base；Luna 不导入 MuJoCo、不直连控制器。手工仿真场景和 CAD 选择完全归 Luna，独立场景适配器直接向 Simulation 提交通用网格/配置，Simulation 保存和执行物理场景；Base 不转发、不管理物理场景。Base 自己的场景编辑只作用于数字孪生。真机/未知模式禁止物理场景写入，任务算法不接收配置接口或物体真值。启动顺序为真机/Simulation、Base、Luna，三者必须在各自项目手动分别启动，不提供联合启动入口；Luna 从所配置 Base 地址的运行身份自动识别模式。
- 数值 FK 与机器人自身深度剔除调用固定 cuRoboV2-Fast 的公开 Kinematics/RobotSegmenter。Luna 嵌入 Base 实测 Viewer，录像由 Luna 归档与编码。
- Luna 只有一个 HTML 工作台，任务、仿真场景、碰撞球、工作空间、回放与嵌入 Base 控制台用 hash 面板切换。功能不得新开页面；面板 DOM/CSS 隔离并在隐藏时暂停预览。碰撞球显示的本体和球必须使用相同关节与开夹爪配置，Base 的只读模型求值仅用于显示。Simulation 启动 observe、Base 内置 observe 与 Luna 观测目标保持一致。
- 场景感知、建图和拆包对照采集必须三路 RGBD 来自同一次已验证的曝光组，具备标定、绑定反馈并在静止观察时采集；诊断预览不是任务证据。
- Runtime 依赖 `robot_interface.py` 的 Gateway/Geometry 接口；具体 Base HTTP 与算法客户端仅由 app 组合入口注入。录像通过公开资源接口取材，不访问通信适配器内部。
- 环境与物体状态仅来自 RGBD 感知；Luna 孪生只接收机器人实测姿态并叠加感知点云/碰撞场景，禁止加载仿真物体真值。夹爪状态来自 Base；删除本地宽度预测和旧放置配置筛查，保留关节轨迹的 cuRobo FK/碰撞校验。
- Base 保留彩色模型；Simulation/Luna 显示灰白机器人及立柱、黑色夹爪/相机。显示配色不修改模型几何或标定。
- 算法镜像不加版本、日期或用户后缀；Docker 隐式使用 latest，精确版本和 ID 记录在清单。共享算法为 `tianji-algorithm-<算法>`，不带用户后缀；真机平台固定 `jodell-console:0.2.0`，私有 Luna/仿真允许 `-guangyu`。SAM6D 不部署。镜像 ID 和历史来源见 `deploy/runtime-images.json`；挂载代码变化不新建镜像，不清理其他用户或仍被容器引用的资源。
- 本次跨仓维护已获用户授权，包括 C:/project/tianji_base、C:/project/tianji-simulation 及其同名远端目录。运行说明与验证边界见 docs/base-integration.md。
- 日常入口保持稳定：当前 guangyu 仿真三个项目都在各自目录使用 `bash start.sh` / `bash stop.sh`，不要把内部脚本或长 Compose 命令替代日常命令示例。每个脚本只管理自己的项目。Base 的根入口默认使用当前仿真配置，`real_web_dev.sh` 保持真机入口。镜像、IP、网络等配置留在各项目现有部署文件中。说明命令时区分模式、浏览器 SSH 转发与服务启动，维护 docs/base-integration.md 中的统一说明。
- 三个项目启动不等待其他项目在线，停止不改变其他项目生命周期。Base 与 Simulation 容器分别命名 `tianji-base-guangyu`、`tianji-simulation-guangyu`，不得加 Luna 前缀。Base 身份与设备就绪只作为 Luna 执行任务的门槛，不能阻止网页启动。
- Windows 本地访问当前仿真时，Base/Luna 都使用各自根目录的 `open.cmd`，内部处理 SSH 转发与浏览器打开，不启动服务。不要把长 SSH 命令和 Start-Process 混作同一级日常入口，也不要为简化访问而改变现有服务器监听范围或端口。

## 模型与 graph

- 每次实验使用独立的 SQLite 实例库，第一个实例 ID 为 1。历史实验按运行 UUID 独立归档。时间记忆数据库保存实例集合、永久 ID、历史观测、操作记录和加工谱系，通过 `ObjectRecord/ObjectObservation/ObjectOperation` 更新。空间记忆就是碰撞场景，保存实际几何及其来源、版本、启用和有效性状态，通过 `CollisionUpdate/CollisionSnapshot` 管理。原始观测、`GeometryMeasurement/GeometryObservation` 测量缓存及配置区域归执行层。`PolicyTaskProgress` 只持有本次任务责任和回执。Graph 接收 `GraphSkillResult`，Decisions 接收单题 `DecisionQuery`。回放使用“时间记忆数据库”和“空间记忆”，时间记忆显示独立提交快照。禁止将任务状态写回实例数据库。

- 入口为初始 Responses（gpt-6.2-sol，low），输入自然语言、三相机初始 RGB、配置区域、CAD 目录、技能契约及 prompt，仅输出场景段声明的物体清单、独立任务选择器及目的地区域及 `version=2` 显式策略图。边采用 `sources/outcomes/target/max_traversals` 路由组声明，程序按源与结果的笛卡尔积展开并生成唯一边 ID。模型通过 on_failure/on_unknown 显式声明作用域公共失败去向，单独路由覆盖相应默认，router/end 继续显式声明。模型决定任务组合、自定义路由和次数；可显式引用 resources/policy/subgraphs.json 的固定模板并绑定输入，程序展开模板及依赖，再对全部节点和边执行相同完整校验。模板不得隐藏物理动作或替代最终任务验收。开局只调用一次 Responses。完整输出与校验错误归档。校验通过后执行，失败立即结束编译。observe_scene 用 SAM3 与 Decisions 关联实例并更新记忆。新实例确认后由独立登记器分配永久 ID。
- 编排失败记忆由独立 CompilationMemory 保存到 data/runs/compilation-memory.sqlite3，按请求哈希去重并按当前可执行契约指纹提供历史约束与反例。新任务和任务重置保留记录。真实终止技能失败独立存入同库 execution_failures，按运行去重、按可执行契约筛选，提供有日期的失败证据。Responses 修订保留此前各轮错误，数据流与恢复覆盖的独立错误同轮报告。实例记忆继续独立保存物理身份和操作历史。
- 实例集合属于可更新、可查询的时间记忆数据库。graph/subgraph 负责逐物体流程和有限恢复，允许共享子图使用 $selected 循环处理当前目标。独立 subgraph.task 为可选表达，执行账本在选中输入时登记处理责任。graph node ID 表示控制节点，不能当作物体 ID；CAD asset ID 表示可被多个物体实例共享的模型。
- Responses 编排正常主线及所需恢复分支。谓词 router 使用空 choices，恢复选择器声明 edge.on 对应的 choices 及 unknown。程序规则选择已知恢复；只有未解决的当前视觉事实才调用 Decisions。无合法恢复时明确结束。Decisions 输出有类型的视觉判断、选项和概率，只选择已经声明的 edge；不能生成 node/edge、选择图外技能或补参数。Runtime 只解释 edge、维护证据与账本，未处理的结果明确停止。
- 跨 skill 的观测、回位、换臂、建图、核验和恢复必须出现在模型 graph。skill 内允许固定算法步骤和真实前提。禁止隐藏流程、重播已完成物理效果、用空队列宣称完成。每个 subgraph 显式到达自己的 end。
- 批处理每件交付后优先在当前位置重新观测并选择当前输入；只有程序确认必要信息缺失时才沿 graph 回观测位姿补观测。all/count 使用共享子图，初始数量不限制运行时成员。最新观测确认来源盒已空即结束本批。最终双臂回位由实际关节反馈验收。VLA 起始姿态仍属于加工技能前提。
- 初始候选不是已确认事实；observe_scene、select_target、verify_task_completion 由 graph 安排。完成证明绑定动作 epoch 与物体记录版本。产物独立永久 ID，保留输入/产物谱系及加工回执。

## Decisions 与感知

- Decisions 仅处理必要视觉事实：实例身份、夹持、落点、加工结果、来源余物及未被当前数值证据证明的支撑关系。真正新出现且任务归属未知的物体可按需做一次视觉归属。每轮多候选选物体交 Decisions 比较过滤图像；选臂、建图范围、明确错误恢复、放置许可和标准任务回执验收交程序；语言要求由初始 Responses 编译到 TaskSpec。
- 所有 RGBD、SAM3、Decisions、身份参考和建图使用完整相机视野；不投影工作空间像素掩码，不按可达包络裁碎物体。有效深度与机器人自身点云过滤保留；离线工作空间仅用于数值可达性。
实例关联消费同次完整三相机 RGBD。任务必须含场景段。Responses 仅定位声明的物体并输出清单、用户选择条件和目的地区域。SAM3 文本/框提示产生候选；同图近似掩码先去重，Decisions 判断部件归属。头相机始终是主视角，按声明检索并为每个当前实例提供登记参考。两腕各用可见框或头部深度投影检索，匹配成功后补充表面证据。程序分别检验头部与每只腕部的双向对应，必要时补充 CAD 位姿与深度。冲突及单独腕部观测保持待定。历史重识别比较相关 ID。已确认的同一冻结采集可以复用身份。有历史记录的每次新采集都由 Decisions 结合带日期的参考、当前度量及实际动作判断。程序先登记物理 ID，再由 Decisions 按原始指令从本帧相容的已登记 ID 中选择目标、来源、支撑和目的地。角色待定时保留已登记实例。详见 docs/rgbd-grounding.md。
- 历史关联比较带日期的首次/最近参考、当前测量和操作历史。独立新采集的重识别使用 Decisions；相同采集且 RGBD、标定与已确认掩码完全一致时由程序复用。当前跨相机对应先统一检查约束再提交。confidence 仅归档。
- 数据库保存永久 ID、原始参考及操作历史。登记要求完整实例掩码上的有效测量，不要求所有像素可达。身份确认后才发布绑定永久 ID 的 CAD/TSDF 几何；程序关联记录真实来源。未见对象保留历史，旧位置不冒充当前测量。
- 每个 Decisions HTTP 请求一个问题，复用同一冻结采集，保留原始请求、回答和转换原因。二值/多选按概率最大项，confidence 只归档，不设 0.8 等接受或缓存门槛。平分、拒答、明确 unknown 保留未知。有效 no/unknown 不通过重复同一问题强行变成 yes；新观测可沿显式有限恢复重新判断。
- 恢复 router 的 rule=observation_recovery/failure_return 是程序选择，只有 rule=visual 才调用模型看当前图像。单一合法路径用直接 edge。所有回位、观测、换臂与恢复动作仍在最终展开 graph 中，保留有限边预算、空手/地图/碰撞/关节前提。没有当前有效图像不能通过旧图声称当前视觉事实。
抓取从目标沿末端 −Z 后退 150 mm 处接近，沿末端 +Z 进给，Lift 沿固定世界 +Z 提升 150 mm。放置从释放点正上方 150 mm 沿世界 −Z 下降，开爪后沿世界 +Z 上撤 150 mm。

verify_grasp 在 Lift 后读取同次实测夹爪开度。全开判为空爪，中间开度判为持物，仅全闭时由 Decisions 查看当前腕相机夹持区判断是否持物。确认持物后，将本次完成抓取的目标永久 ID 绑定到夹爪并建立 attachment。空抓可进入恢复，反馈缺失或执行记录不一致时保留待定。
- verify_place 使用本次搬运已绑定的物体 ID 和目的地 ID。释放时检查机器人实测 TCP 是否到达配置释放 XYZ（位置容差 5 mm），释放后按实测开度判断空爪，仅全闭时由 Decisions 查看当前夹持区。到位、释放且空爪即登记已放到指定桌面位置或盒中，结果为 execution_assumed。回执记录目标、目的地和实际释放位置，物体最终静止位姿保留为未测量。
- TaskSpec.allowed_intermediate_refs、unsupported_placement_constraints、visual_completion_conditions 由初始 Responses 明确编译。固定目标能力之外的放置条款声明不支持；运行时不改 XY、重新解释许可或让模型放宽要求。标准交付、加工、数量/顺序与最终姿态使用已核验回执、谱系和实际关节反馈，只有额外视觉要求再次看图。
- 观测权限仍由 captures_observation 声明。只有 observe_scene、verify_grasp、verify_place、verify_unpack、verify_task_completion 和初始 bootstrap 可采集；规划、运动、router 不隐藏采集。仅对无有效回答的 HTTP 429/502/503/504 或网络故障最多三次重发同一冻结请求并独立归档，不重放技能。

## 几何、抓取与放置

- 所有 RGBD、SAM3、Decisions、身份参考和建图使用完整相机视野；不投影工作空间像素掩码，不按可达包络裁碎物体。有效深度与机器人自身点云过滤保留；离线工作空间仅用于数值可达性。
execute_grasp 和 execute_place 均下发三段时间序列。放置在 place 段末尾松爪，随后执行张开、已 detach 的 retreat。

- 生产 `LUNA_GEOMETRY_MODE=cad_esdf`：实测桌面 cuboid、未知物体当前掩码 RGBD 的 ESDF、已知物体登记 CAD。未知物体用同次 TSDF 表面输入 GraspGen，模型 TCP 转换到 A/B 实际指端，再沿其局部 +Z 应用统一的 GraspGen 伸入量；已知 CAD 用 FoundationPose 和离线抓姿，默认额外伸入 0mm。未知物体的几何与 attachment 统一使用 TSDF。
- `LUNA_CAD_ROOT` 指向独立 CAD 库；项目 resources/cad 是运行副本。蓝盒 1000 面、黑件 999 面，配准和碰撞使用相同表面，禁止恢复高面数双模型分支。保持坐标、A/B 离线抓姿及其哈希绑定一致。
- FoundationPose 输入当前 SAM3 掩码、RGBD、CAD 和对应标定；选有效深度最充分的相机，输出转换到世界坐标。有效且来源/坐标绑定一致的位姿直接采用，不做 ICP 或额外几何吻合检查。新观测更新位姿，不能把历史位姿当新测量。
- 物体记录永久 ID 绑定 CAD/TSDF。每次发布全工作空间碰撞场景。后续更新绑定前一场景版本，保留未见障碍和未获新测量的历史背景体素，按新测量更新物体层。动作影响几何后由 graph 显式观测、更新并验证运动覆盖。
TCP 统一使用 gripper_front_A/B，实际机器人定义为安装端局部 +Z 185 mm。GraspGen 服务在边界将 Robotiq 模型的 base_frame 沿 +Z 195 mm 转为模型 TCP，应用与可视化消费末端位姿。训练模型的夹爪尺寸差异需保留在来源记录中。
- 三段抓取使用锁定官方 plan_grasp 的接触链接参数，按执行臂排除两侧 Link3 夹指和 Link4 连杆。SDK 在 goalset/grasp/lift 求解时临时排除，approach 时恢复；适配器负责异常后的恢复。执行前逐段核对环境、另一臂、非接触部件与目标接触条件。MotionPlanBinding 绑定目标、目的地、臂、采集、场景和参数，变化后重新规划。
- 抓取第二段及闭合的接触许可限定为当前目标与执行臂两侧 Link3 夹指和 Link4 连杆的配对，球索引从当前机器人配置读取。这些链接与其他环境物体、另一臂、其余部件与目标及自碰撞继续使用原数值检查。未知物体端点按完整实测点云检查张开接触垫网格凸包间隙，计入配置的测量不确定性和碰撞余量。
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

Attachment follows completed Lift, confirmed holding and the target ID bound to this grasp. Planning and executing the entire grasp sequence use the detached robot and the official grasp-planning open-gripper model, with a separate closure check. Both Link3 fingers and both Link4 links of the executing gripper may contact the selected target during grasp, closure and Lift. The palm and the other arm retain their target checks. Other robot/environment and self-collision checks remain active. Successful verification prepares carrying geometry for later transport and placement. The earlier pre-Lift virtual carrying and opening-range path sweep implementation has been removed.

Responses 每个任务只调用一次，解析或图校验失败立即结束编译。固定模板由程序展开，逐物体子图的 completion_pose 在输出 schema 中限定为空，最终回位由根任务声明。执行采用三倍时间伸长和五次 Hermite 插值，轨迹仅通过 Base 下发。控制器更新、仿真物理积分和实测跟踪检查由 Base/Simulation 承担。

实验回放按 graph、视频、decisions、数据库、碰撞场景、原始数据排列。decisions 页只展示实际请求图片，每排四张，按发送顺序读取归档，同时提供 state、图片元数据、questions 与回答。初始描述绑定携带描述相机与采集锚点。新实例确认记录实际比较是否执行，P 标签与永久 ID 显式映射。每张模型图片记录实际尺寸，CAD 预览使用独立静态参考角色。

未知物体的 attachment 使用抓取前已发布的同次多相机 TSDF 表面体素，将每个体素的八个角点按闭爪时实际 TCP 转到夹爪坐标，再生成覆盖全部体素的碰撞球。放置足迹沿用同一组体素角点。来源绑定采集、代次、永久 ID 和体素尺寸，缺失或过期时报告失败。已知 CAD attachment 使用已配准 CAD。Lift 完成、持物判定成立且连续执行绑定通过后建立 attachment，采用本次抓取绑定的目标 ID。

GraspGen 候选先转换到实际 Tianji TCP。保留末端 +Z 从机器人正前到正下的整个 90° 扇区，允许相对前下平面的左右偏角至 45°。夹爪绕自身 +Z 的半转不改变接近方向。CAD 使用离线抓姿。可行性由 cuRobo 的 IK、机器人自身及环境碰撞、实际 FK 与目标位姿误差检查决定。抓取 Lift 仍沿世界 +Z 150 mm，并用公开 FK 检查插值点的横向偏移和提升高度。Lift 完成并通过持物核验后建立 attachment。

Decisions 输入以实际发送图片的顺序编号，每张图标明相机角色、观测 ID、代次、时刻、CURRENT/历史/静态目录角色及归一化框坐标约定。当前图片与请求采集绑定检查通过后才发送。跨相机选项描述给出 C 标签、相机和该图中的框坐标，相邻时刻 P 选项给出历史采集及对应图像位置。初始分组和 CAD 选项分别描述当前标签与目录标签。裁剪和单主体掩码元数据只列出本图主体。支撑关系问题与实际发送的 RGB 轮廓一致，放置核验读取本次执行绑定的目标和目的地，逐物体完成条件绑定对应目标及子任务。回放同时显示选项代码、完整选项概率分布及单独的 confidence 字段。

固定 single_arm_transfer_v1 在入口显式执行 observe_scene，capture_mode 为 reuse_current，由现有采集版本和关节反馈检查决定复用或新采集，然后进入 identify_target。模板入口要求空手、无需恢复且仍需交付。observed 与 grasp_view_ready 由入口观测建立，全部节点和失败边在展开 graph 中可见。

持物判断中，实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。state 说明接近闭合时仍可持物、背景与遮挡的区别、深色材料及历史图片角色。verify_grasp、verify_place 和失败收尾共用实测位置优先、全闭才看图的规则。持物后沿用本次抓取的目标 ID。

所有 GraspGen 抓取按执行臂当前指端 TCP 选择较近的平行夹爪半转等价姿态，随后进入 IK 和完整碰撞检查。CAD 离线抓姿按库中 A/B 归属读取。

CAD 模型绑定采用 resources/cad_subject_policy.json 中的用户场景声明。已确认身份的 black/object 绑定黑色工件 CAD，blue/bin 绑定蓝盒 CAD，其余身份使用 TSDF。程序按该声明绑定模型并保留来源，FoundationPose 使用当前 SAM3 掩码和 RGBD 估计位姿。

持物和空爪题使用同次 TCP 与相机标定生成的 11 cm 夹持接触区 RGB，保留原始像素分辨率，不加入历史整幅腕图。当前图保持原始像素、不依赖 SAM3、不带物体标签；投影不可用时退回完整当前腕图。失败收尾对两腕使用相同规则。历史参考用于跨时刻实例关联。

持物与空爪题的 state 同时提供当前采集的夹爪反馈，实测归一化闭合程度与全闭目标命令分别标注。0 表示机械全开，1 表示机械全闭，该值不是夹持力。实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。张开或闭合命令仅表示动作意图。实测归一化位置距机械端点 0.02 以内归为对应端点，其余归为 partially_closed。程序直接判定全开与中间位置，仅全闭时由 Decisions 看图判断；程序判定独立记录来源，不伪造模型回答。缺失或无效反馈明确标为不可用。夹爪反馈仅用于持物和空爪题；持物后沿用本次抓取的目标 ID。

初始描述由 Responses 编译为选择条件和候选框，随后 ground_task 用同步三相机 SAM3、RGBD、同图 IoU 与 Decisions 核验后登记。后续观测复用相同关联链；头相机对声明类别进行语义检索，腕相机只用各自的可见候选框或头相机投影提示。每个腕相机独立匹配头相机。局部建图按永久 ID 保留静止碰撞层及原始几何来源，新建或移动的已知物体用 FoundationPose，未知物体用掩码深度 TSDF。持物、空爪及失败收尾使用按同次机器人指端 TCP 和相机标定生成的当前局部 RGB，不依赖目标检测结果；持物题不发送历史整图；当前局部投影不可用时才退回完整腕图。桌面和入盒的控制台区域均只约束实际释放位置；抓放验证先读本次动作后的执行臂反馈，只有全闭才采集执行臂腕图判断是否持物。按已绑定目标、实测释放点和空爪登记交付。

初始编排必须声明 support_table_refs，引用 task_selectors 中实际可见且参与支撑测量的 table；配置区域不是桌子实体。程序通过当前或缓存实测桌面多边形，唯一绑定控制台区域及支撑高度。Responses 保留方向、序号和用户原话。Decisions 在当前标注图和基座测量上判断对应的永久 ID，机器人左为 +Y、右为 −Y。证据不清时保持未知，特定目标绑定后可跨位置沿用。

当前有限任务边界与证据实现见 [可靠性修正](docs/scoped-reliability-20261115.md)。TSDF 抓姿先将模型接触 TCP 转为执行臂 RAG 指端，再沿局部 +Z 应用 rag_grasp_insertion_m（默认 10 mm，可调 0–20 mm），在 IK 前用实测表面核对开口与指垫。所有 GraspGen 候选使用同一伸入值。当前只要求同臂覆盖来源与固定目的地、固定 CAD。物理记忆提交使用独立 PhysicalMemoryWriter，历史失败按实现/合同/子图/配置版本筛选，较早身份参考每实体最多 3 次采集。

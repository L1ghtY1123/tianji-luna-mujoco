# Luna 当前技能与模型边界

本页当前说明包含 [2026-11-15 可靠性修正](scoped-reliability-20261115.md)。开发约束见 AGENTS.md；技能参数、前提、效果、结果与失败码见 skill_catalog 对应 contract.json 和 SKILL.md。下方历史记录只描述当时验证，不能作为当前 prompt 或合同。

实例关联消费同次完整三相机 RGBD。任务必须含场景段。Responses 仅定位声明的物体并输出清单、用户选择条件和目的地区域。SAM3 文本/框提示产生候选；同图近似掩码先去重，Decisions 判断部件归属。头相机按声明检索，两腕各用可见框或头部深度投影检索。程序分别检验头部与每只腕部的双向对应，必要时补充 CAD 位姿与深度。冲突及单独腕部观测保持待定。历史重识别仍比较所有相关 ID；同次已确认掩码可直接复用。确认场景事实后绑定任务，unknown 不注册重复 ID，也不回滚无冲突实例。详见 docs/rgbd-grounding.md。

## 编排与固定子图

初始 Responses 接收任务、原始三相机 RGB、配置区域、CAD 目录与技能契约，输出 version=2 完整 graph。可以引用 resources/policy/subgraphs.json 的固定模板，也可以写自定义子图；全部展开后执行相同拓扑、数据流、恢复覆盖和完成校验。仅调用一次，校验失败即终止。运行中不重新调用 Responses 编排。

TaskSpec 一次绑定来源、目标、处理要求、顺序、允许中间区域及额外视觉要求。已知顺序、选臂、建图、固定放置区域和明确恢复由程序处理。每个物理动作和恢复动作仍在展开 graph 中，router 不暗中采集或移动。observation_recovery、failure_return 是程序规则；visual 才调用 Decisions。单一路径直接连 edge。所有路径保留有限预算、空手、地图、关节及碰撞前提。

## 当前 23 个任务能力（另有 4 个系统收尾能力）

| 能力 | 职责 |
| --- | --- |
| move_to_observe / return_to_initial_pose | 显式执行观测或初始姿态运动，实际关节反馈核验 |
| capture_scene / ground_task | 显式采集与冻结 RGBD 定位分开，成功后才登记 ID |
| observe_scene | 采集、候选分割、原始视觉身份关联及数值测量；不隐式移动 |
| select_target | 查询当前数据库候选；多候选由一次 Decisions 原图 choice 优先选择容易抓的物体；用户明确顺序优先 |
| identify_target | 绑定目标；当前数值支撑明确时直接采用，否则一次原始视觉选择 |
| identify_destination | 校验编译目的引用与当前身份，不再问语言许可 |
| choose_arm | 目标及固定释放位置的共同工作空间准入，再按 TCP 距离选臂；同次抓放锁定该臂 |
| build_scene | 按当前测量更新物体层和背景体素，发布全工作空间碰撞场景 |
| generate_grasps | 已知 CAD 分臂离线抓姿，默认额外伸入 0mm；未知物体以当前 TSDF 表面输入 GraspGen，模型接触 TCP 转换至左右臂实际指端，不额外伸入 |
| plan_grasp / execute_grasp | cuRoboV2-Fast 三段规划并执行接近、抓取与 Lift |
| verify_grasp | 实测夹爪开度，全闭时使用当前夹持区 RGB | 确认持物后继承抓取目标 ID |
| recover_failed_grasp | 只接受同动作明确 empty 且无 attachment 的空抓恢复 |
| perceive_placement_zone | 数值解析控制台固定区域与编译许可，不搜索 XY、不再运行时解释语言 |
| plan_place / execute_place | approach/place/retreat 三段；place 后松爪、detach，再撤离 |
| verify_place | 实测释放位置与夹爪开度，全闭时使用当前夹持区 RGB | 到位释放且空爪后登记选定目的地 |
| prepare_unpack / vla_unpack | 显式准备并执行配置化加工；仿真效果不作为视觉证据 |
| verify_unpack | 原始前后图像判断加工结果，登记独立产物与谱系 |
| verify_task_completion | 已核验任务回执、数量/顺序/谱系和实际关节；只额外判断来源余物及明确额外视觉要求 |

## Decisions 的输入与结果

发送完整视野 RGB，候选框由 SAM3 掩码计算，按需附固定量纲深度图。实例关联使用真实外观和物理历史。每个 HTTP 请求一个问题。

verify_grasp 在 Lift 后读取同次实测夹爪开度。全开判为空爪，中间开度判为持物，仅全闭时由 Decisions 查看当前腕相机夹持区判断是否持物。确认持物后，将本次完成抓取的目标永久 ID 绑定到夹爪并建立 attachment。空抓可进入恢复，反馈缺失或执行记录不一致时保留待定。 verify_place 使用本次搬运已绑定的物体 ID 和目的地 ID。释放时检查机器人实测 TCP 是否到达配置释放 XYZ（位置容差 5 mm），释放后按实测开度判断空爪，仅全闭时由 Decisions 查看当前夹持区。到位、释放且空爪即登记已放到指定桌面位置或盒中，结果为 execution_assumed。回执记录目标、目的地和实际释放位置，物体最终静止位姿保留为未测量。


二值取概率较大项，多选取最大项；没有 0.8 接受或缓存门槛。confidence 只记录。精确平分、拒答和明确 unknown 保留 unknown。有效 no/unknown 不重发同一问题直到变 yes。

## 数据与物理边界

时间记忆数据库保存实例、原始身份参考、观测、操作和加工谱系；空间记忆保存碰撞几何与来源、版本及有效性。任务账本独立记录本任务责任和已核验回执。Graph 接收 GraphSkillResult，Decisions 接收单题 DecisionQuery；不直接访问其他层可变存储。

身份接受后再执行机器人/桌面数值过滤、FoundationPose 或 TSDF 建图。原始身份 mask 与几何 mask 分开保存。三维点云、标定、FK、桌面 cuboid、CAD、TSDF 和碰撞检查继续用于实际机器人计算；没有将这些计算包装成视觉模型前置判定。

放置 XY 使用控制台正方形中心，旋转保持实际抓取姿态。末端 Z = 实测桌面或盒口最高点 Z + 物体高度 + 控制台高度余量。物体高度来自同次 CAD/TSDF 几何竖直范围及来源支撑，独立于碰撞球。默认余量 20 mm，含为保守附着包络增加的 10 mm。若抓取 TCP 高于物体顶部，使用较大的 TCP 到来源支撑面偏移避免低估。实际物体底面另按抓取偏移校验，假设搬运中物体不滑移。碰撞几何独立检查目标及轨迹，不搜索目标高度。选臂时用可见高度估计，抓取后用已绑定 CAD/TSDF 再检查同一臂。

批处理每件交付后优先原地重新观测；程序判断必要信息缺失后才回观测位姿补观测；all 以当前来源无剩余输入验收，count 按已核验数量。完成验收不因空队列或历史图像宣称当前成功。最终姿态用关节反馈。

完整输入编码、真实接口结果及未覆盖边界见 [完整视野与视觉判断](raw-decisions-20261111.md)。

## 历史记录（非当前指令）

以下内容保留历史方案与实验上下文；其中投影、裁剪、概率门槛及运行时语言判断均已被当前方案替换。

### 放置模式与恢复输入（2026-11-06）

execute_grasp 和 execute_place 均下发三段时间序列。放置在 place 段末尾松爪，随后执行张开、已 detach 的 retreat。

恢复选择输入同时包含 inline 成功续接路径、各结果边及子图定义，显示分支入口剩余次数；Decisions 只选已声明分支。初始 prompt 保存已验证的通用编排经验，区分规划失败与部分执行后的不同前提，不保存固定任务图。
## 观测后的抓取前提（2026-11-07）

永久物体 ID 与当前抓取依据分开。`identify_target` 提供 `source_support_current`；`observe_scene` 更新采集后使该前提失效；`plan_grasp` 要求它成立。在线 Responses 应在恢复后的再次抓取前显式重新绑定目标，并按已有契约恢复目的地、选臂、场景和候选等依赖。相同采集已有有效支撑证明时不增加 Decisions 请求。该变更只纠正证据时效，不修改规划算法、技能运动步骤、阈值或恢复次数。

观测结果由执行层在数据库提交后更新两个记忆版本；几何变化意味着场景待发布，观测本身不能替代 `build_scene`。未见不删除实例，只有明确消耗或移除证据才退役。


## 抓取与执行边界核对（2026-11-07）


抓取与放置计划携带 MotionPlanBinding，执行前检查目标、目的地、执行臂、generation、observation_id、scene_token 和参数哈希。抓取搜索使用冻结的观测与场景副本。Graph 与 Decisions 收到逐轮候选失败阶段及接触/碰撞摘要。观测恢复允许配置回位或满足前提的腕部换视角；当前已在配置位姿时，分支选择须结合实际证据变化。

首次新实例提议也经过单题 Decisions 确认；unknown/reject 保留提议反馈。世界命名空间由仿真场景实例持有，同一场景的控制器重建继续使用其标识。

初始与运行中观测共用三相机临时分组，标定投影提出对应关系后由 Decisions 对整组框确认场景描述或历史身份。数据绑定当前采集与代次。登记后的掩码参考进入首个 observe_scene，完整观测通过原有记忆事务持久化。

抓取第二段及闭合的接触许可限定为当前目标与执行臂两侧 Link3、Link4 的配对，球索引从当前机器人配置读取。Link2 全部球参与目标碰撞检查。这些链接与其他环境物体、另一臂、其余部件与目标及自碰撞继续使用原数值检查。夹爪内侧与虎口按 URDF 几何拟合，整机 438 球，A/B 离线工作空间与当前球模型指纹一致。

未知物体抓取端点使用完整实测点云检查张开接触垫碰撞网格凸包的进入间隙，应用配置的测量不确定性和碰撞余量。绑定模型、点云及当前场景的有限负间隙允许继续既定有界候选搜索。限幅闭合预测采用轨迹端点 FK。夹爪越界失败包含实际角度、命令、限位与稳定窗口统计。

多个当前合格输入通过 select_target 的单题 Decisions 按任务与当前视觉、测量信息选择。单候选和明确顺序直接处理，后续规划继续验证数值可行性。

choose_arm 使用明确指定的臂与当前数值可行域筛选，优先选择没有同目标失败记录的臂，再按当前 TCP 到目标的距离排序，记录各臂代价。姿态 IK 与路径碰撞继续由规划检查。

## 动态批处理与 Decisions 边界（2026-11-08）

all/count 任务采用共享处理子图。每件交付后显式回配置观测位姿，再调用 fresh observe_scene 和 select_target。初始 target_refs 保存初始候选，新观测登记的匹配实例可以进入同一循环。all 以当前来源盒确认清空结束本批，count 按请求数量验收。循环预算独立于初始数量，预算耗尽返回失败。

Decisions 负责实例身份、CAD 选择、任务成员关系、语义位置、来源为空的视觉判断，以及多种合法恢复方案之间的选择。单个候选、已绑定参数、数值规划结果、反馈和版本检查、唯一恢复出口直接由程序处理。抓取、工作区放置和容器入盒均由显式视觉核验节点确认。执行回执提供动作绑定，视觉判断提供夹持、支撑或包含关系。

时间记忆保存入盒回执、永久 ID 和历史测量。确认入盒后清除目标旧位置的独立碰撞层，并标记目标及容器区域待更新。graph 重新观测后，以实际测量更新可见内容物和背景。被遮挡的历史占用继续保留，个体最终位姿仍需测量。

完全相同的路由声明按 source、outcome、target、max_traversals 合并，次数保持原声明值。不同目标或不同次数仍返回冲突。verify_place 显式声明释放模式，工作区轻放的地图失效前提继续传递，普通盒上释放采用独立的条件性占据发布结果。

association_scope 由任务历史投影。已完成入盒且无需再次操作的目标，凭有效入盒回执延后个体身份关联。待处理输入和容器继续关联。容器内几何通过局部匿名 TSDF 更新，单件身份与最终位姿保持各自证据状态。再次操作已交付目标时恢复其当前身份和几何测量。

切换本轮输入时，执行上下文保留与已发布空间记忆绑定的 scene_observation。建图范围判断按可选历史观测处理缺失值。新目标的操作计划及放置绑定继续重新建立。

后续拆包核验用已交付黑件的实测位置或入盒历史限定身份排除区域。容器边界仅用于该身份范围，碰撞几何由实测 TSDF 表示。当前目标几何仍由当前采集建立。


支撑谓词使用单题 A/B 标签，并在结构化输入中保留永久 ID 映射。支撑判断在当前 RGB 上标出被询问物体，并将已测桌面边界按当前相机标定投影为带永久 ID 的多边形。标注保留原桌面测量日期，清晰的三维容器关系与已配准 CAD 桌面接触由程序判断，歧义关系交给 Decisions。优先展示同时定位物体与支撑参考的视图，保留全部三相机画面。

## 授权观测与连续执行

采集权限由技能契约 captures_observation 声明。observe_scene、verify_grasp、verify_place、verify_unpack 和 verify_task_completion 可采集。初始任务 bootstrap 采集冻结的 Responses 输入。运动、规划和 router 使用已有证据，恢复 router 明确标记历史图像。网页相机面板显示最近一次授权观测。

identify_destination 的 observed_instance_missing_or_ambiguous 已列入可恢复错误。编译器依据技能契约中明确的 grasp_held=false 要求检查身份/支撑恢复覆盖。其他节点由 Graph 声明有限恢复或明确终止。恢复中的再次失败可以直接结束。拆包后的腕部姿态可能遮住盒子，Graph 可安排 build_scene、move_to_observe、observe_scene 并重新绑定目的地，继续使用已验证的产物与加工回执。

放置动作和视觉核验使用同一释放区域判据。container_release 验收完整 CAD 或已观测 TSDF 体素几何落在区域内，精确 XY 模式验收绑定参考点，普通表面模式验收 attachment 足迹。attachment 球继续参与环境、自碰撞和轨迹数值检查。入盒仍须通过当前视觉包含关系确认。

plan_grasp 在候选筛选中预检完整 lift 路径及夹爪允许行程，和 execute_grasp 的执行前检查保持一致。lift 夹爪采样步长为 0.0005 rad，采样间使用保守世界碰撞包络，环境间隙和自碰撞要求继续保留。

execute_grasp 和 execute_place 均下发三段时间序列。放置在 place 段末尾松爪，随后执行张开、已 detach 的 retreat。

verify_place 使用本次搬运已绑定的物体 ID 和目的地 ID。释放时检查机器人实测 TCP 是否到达配置释放 XYZ（位置容差 5 mm），释放后按实测开度判断空爪，仅全闭时由 Decisions 查看当前夹持区。到位、释放且空爪即登记已放到指定桌面位置或盒中，结果为 execution_assumed。回执记录目标、目的地和实际释放位置，物体最终静止位姿保留为未测量。

选臂遇到已登记 CAD 目标和当前已发布容器区域时，先对每臂完整离线抓姿库复用 TCP 预处理及搬运可达性筛选，使用本次释放模式与控制台 XY。候选全被拒绝的臂退出本次选择。通过后继续执行原有 IK 与轨迹检查。目的地当前几何缺失时保留待验证状态。

释放后的实例关联保留开爪时实际 TCP、持物几何参考位置和此前视觉夹持核验的采集来源。旧位置早于已执行的放置动作时，当前比较使用开爪位置的运动假设与当前 RGBD，移除搬运前表面在原位置投影的重叠比较。放置视觉 unknown 继续保留，实例接受仍由当前对应判断决定。

单个身份谓词的历史参考图只标注正在比较的永久实例，并提供该实例的掩码特写。当前候选特写使用当前 SAM3 主体掩码。三路完整当前 RGB 保留周围物体和支撑上下文，已接受的当前身份继续作为图像锚点。

放置核验按参考采集编号区分历史特写和当前已确认身份。当前同采集且 RGB 一致的目标掩码可显示永久 ID。优先展示区域投影完整落在画面内的视角，保留全部三视角。支撑问题接收同一释放动作已确认的带日期视觉支撑记忆，结合当前图像继续判断；区域问题使用本次区域证据。

显式 observe_scene 的算法调用在感知前核对共享 GPU 显存。可用显存低于 8 GiB 时，cuRobo 在请求锁内销毁空闲求解器并释放缓存分配，保留场景与持物数据。下一次数值请求按相同配置延迟重建。观测结果记录回收前后显存和场景 token。阈值依据本次 FoundationPose 黑件注册峰值约 11.3 GiB、常驻约 5 GiB 设置。

verify_place 使用本次搬运已绑定的物体 ID 和目的地 ID。释放时检查机器人实测 TCP 是否到达配置释放 XYZ（位置容差 5 mm），释放后按实测开度判断空爪，仅全闭时由 Decisions 查看当前夹持区。到位、释放且空爪即登记已放到指定桌面位置或盒中，结果为 execution_assumed。回执记录目标、目的地和实际释放位置，物体最终静止位姿保留为未测量。

## 2026-11-10 RGBD 操作证据


二值事实判断按较大概率输出结果，P(true)>0.5 为 true，P(true)<0.5 为 false。概率仅作为审计证据，精确平分、拒答或无有效响应保留 unknown。


当前统一失败收尾遵循 [终止失败回位](terminal-failure-return-20261111.md)：具体恢复耗尽或未处理异常均执行显式固定收尾图，从实际关节规划回观测位姿并停止，保留原任务失败；不安全或无法确认的回位报告 blocked。

观测恢复校验接受 graph 中显式声明的 observe_scene(capture_mode=fresh) 当前位置补观测；不强制先移动。reuse_current 只有在移动已使旧观测失效时才满足补观测要求，单纯重试和重用不变证据不能通过。回放的时间记忆页签及元素名称与执行脚本、样式同步；实验列表必须在真实浏览器完成初始化验证。

回放默认优先完整执行图，其次完整 Responses 编排；系统失败回位图单独保留，不覆盖任务图。固定模板名称和输入从当次归档原始声明读取，不使用当前模板库重建历史；“完整图”可展开全部子图及未执行的失败恢复节点，刷新保留展开状态。登记使用对应相机带框过滤 RGB 和明确描述，每个 ID 选项附物体描述，已有 ID 也须当前视觉确认。observe_scene 每题用 SAM3 框标明当前候选，主体相机优先并标明头部/左右腕；历史参考使用过滤 RGB、日期、实例框和物理历史，不混入其他当前候选或历史剪裁掩码。SAM3 掩码继续用于几何测量。程序边界及验证见 [Decisions 与数据库审计](decisions-database-audit-20261111.md)。

plan_grasp 对官方 goalset 实际选中的抓姿完成三段数值验证。确定拒绝后从冻结池移除该项，继续评估剩余项，未知或无效证据沿 graph 处理。

## 2026-11-12 association and planning update


Unknown-object GraspGen sampling uses one native 100-candidate pool per round, with one graph-controlled additional round. Each cuRoboV2-Fast grasp request contains the configured grasp_goalset_size ranked candidates. Fully rejected candidates are removed, later batches remain available, and the first grasp that passes all numerical checks ends planning. After a confirmed table release, voxel geometry retains its occupied cells with stale provenance until graph-controlled RGBD remapping.

Command-validation profiling isolated 15-22 seconds in full closure-range lift sweeps after an early collision. Validation now stops on a proven invalid chunk and records partial evaluation. Successful paths still complete all checks. Coverage bounds use the same numerical inputs in bounded chunks, including each next endpoint.

Attachment follows completed Lift and Decisions holding presence and continuous execution identity inheritance. Planning and executing the entire grasp sequence use the detached robot and the official grasp-planning open-gripper model, with a separate closure check. The executing Link3/Link4 links retain the selected-target contact policy during grasp and Lift. Other robot/environment and self-collision checks remain active. Successful verification prepares carrying geometry for later transport and placement. The earlier pre-Lift virtual carrying and opening-range path sweep implementation has been removed.

Responses 每个任务只调用一次，解析或图校验失败立即结束编译。固定模板由程序展开，逐物体子图的 completion_pose 在输出 schema 中限定为空，最终回位由根任务声明。执行采用三倍时间伸长和五次 Hermite 插值。MuJoCo 关节目标以 50 Hz 更新，物理积分与跟踪误差检查按原物理步运行。

## 编排边界与观测回放

Responses 编译在模板展开前校验实际输出 schema，展开后校验技能参数、字面实例引用、拓扑、前提、结果绑定、有限恢复和完成事实。缺失的肯定前提按运行时的 false 默认值处理。compact 模式继续提供各注册技能的完整 SKILL.md，只压缩合同中的重复字段。固定库包含观测建图、回观测位姿、空抓恢复、单臂搬运四类模板。单臂搬运接口声明 grasp_view_ready 和 needs_place，调用方必须建立这些事实。复杂批处理和加工流程通过受相同合同约束的显式子图组合。

plan_grasp 的 success_count 仅统计通过完整执行检查的轨迹，solver_success_count 单独记录 SDK 求解结果。request_index 表示实际请求次数，candidate_group_index 表示按 grasp_goalset_size 划分的候选组。已拒绝的候选从原组移除后再求解，首条完整通过的轨迹结束规划。

所有 RGBD、SAM3、Decisions、身份参考和建图使用完整相机视野；不投影工作空间像素掩码，不按可达包络裁碎物体。有效深度与机器人自身点云过滤保留；离线工作空间仅用于数值可达性。

实验观测页默认逐次展示实际 Decisions 请求、完整问题、回答和按发送顺序排列的输入图片。程序身份叠加位于折叠面板，完整传感器 RGBD 从观测 JSON 的 views 读取。历史旧归档保留 raw_views。

verify_grasp 在 Lift 后先读取同次实际关节位置：全开为未持物，中间位置为持物；仅全闭时调用一次 Decisions，输入当前夹持接触区原始分辨率 RGB 和实测反馈，不调用 SAM3。明确未持物允许空抓恢复；请求异常、拒答或证据不足保留 unknown。确认持物后，程序核对连续抓取记录的目标、执行臂、运行、代次、来源观测、场景及动作版本，继承抓取前已确认目标的永久 ID，再建立 attachment。identity_binding 明确标记 continuous_grasp_execution 和 visual_identity_verified=false；这属于执行过程推定身份。执行记录不一致时保留持物结果与身份未确定状态，不建立 attachment。历史参考保留用于抓取前关联与放置核验，正常主线不再单独做抓后身份题。

实验回放按 graph、视频、decisions、数据库、碰撞场景、原始数据排列。decisions 页只展示实际请求图片，每排四张，按发送顺序读取归档，同时提供 state、图片元数据、questions 与回答。初始描述绑定携带描述相机与采集锚点。新实例确认记录实际比较是否执行，P 标签与永久 ID 显式映射。每张模型图片记录实际尺寸，CAD 预览使用独立静态参考角色。

失败收尾 return_after_failure 在确认空手后回到观测位姿，检查开爪动作，再张开并验收实际夹爪反馈。持物回位保留夹持与 attachment，任务状态继续保留原始失败。

未知物体的 attachment 使用抓取前已发布的同次多相机 TSDF 表面体素，将每个体素的八个角点按闭爪时实际 TCP 转到夹爪坐标，再生成覆盖全部体素的碰撞球。放置足迹沿用同一组体素角点。来源绑定采集、代次、永久 ID 和体素尺寸，缺失或过期时报告失败。已知 CAD attachment 使用已配准 CAD。Lift 完成、持物判定成立且连续执行绑定通过后建立 attachment，身份按执行记录继承。

GraspGen 候选先转换到实际 Tianji TCP。保留末端 +Z 朝机器人前方或下方的姿态，排除向后、向上和以横向为主的姿态。夹爪绕自身 +Z 的半转不改变接近方向。 抓放各在一次位置模式使能内完成。放置下降期间维持原夹持命令，到达释放点后按实测关节角复核张开和撤退。CAD 使用离线抓姿。可行性由 cuRobo 的 IK、机器人自身及环境碰撞、实际 FK 与目标位姿误差检查决定。抓取 Lift 仍沿世界 +Z 150 mm，并用公开 FK 检查插值点的横向偏移和提升高度。Lift 完成并通过持物核验后建立 attachment。

Decisions 输入以实际发送图片的顺序编号，每张图标明相机角色、观测 ID、代次、时刻、CURRENT/历史/静态目录角色及归一化框坐标约定。当前图片与请求采集绑定检查通过后才发送。跨相机选项描述给出 C 标签、相机和该图中的框坐标，相邻时刻 P 选项给出历史采集及对应图像位置。初始分组和 CAD 选项分别描述当前标签与目录标签。裁剪和单主体掩码元数据只列出本图主体。支撑关系问题与实际发送的 RGB 轮廓一致，放置核验读取本次执行绑定的目标和目的地，逐物体完成条件绑定对应目标及子任务。回放同时显示选项代码、完整选项概率分布及单独的 confidence 字段。

入盒绑定使用 destination_kind=container 与 release_mode=container_release，TCP 位于实测盒口上方的配置高度。桌面区域绑定使用 support_region 与 surface_setdown。输出 schema、模板展开及运行时参数校验均拒绝交叉组合。当前能力不支持在容器底部轻放。

固定 single_arm_transfer_v1 在入口显式执行 observe_scene，capture_mode 为 reuse_current，由现有采集版本和关节反馈检查决定复用或新采集，然后进入 identify_target。模板入口要求空手、无需恢复且仍需交付。observed 与 grasp_view_ready 由入口观测建立，全部节点和失败边在展开 graph 中可见。

持物判断中，实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。state 说明接近闭合时仍可持物、背景与遮挡的区别、深色材料及历史图片角色。verify_grasp、verify_place 和失败收尾共用实测位置优先、全闭才看图的规则。抓后身份从连续执行绑定继承，明确记录推定来源，不另发身份题。

所有 GraspGen 抓取按执行臂当前指端 TCP 选择较近的平行夹爪半转等价姿态，随后进入 IK 和完整碰撞检查。CAD 离线抓姿按库中 A/B 归属读取。

CAD 模型绑定采用 resources/cad_subject_policy.json 中的用户场景声明。已确认身份的 black/object 绑定黑色工件 CAD，blue/bin 绑定蓝盒 CAD，其余身份使用 TSDF。程序按该声明绑定模型并保留来源，FoundationPose 使用当前 SAM3 掩码和 RGBD 估计位姿。

持物和空爪题使用同次 TCP 与相机标定生成的 11 cm 夹持接触区 RGB，保留原始像素分辨率，不加入历史整幅腕图。当前图保持原始像素、不依赖 SAM3、不带物体标签；投影不可用时退回完整当前腕图。失败收尾对两腕使用相同规则。历史参考用于跨时刻实例关联。

持物与空爪题的 state 同时提供当前采集的夹爪反馈，实测归一化闭合程度与全闭目标命令分别标注。0 表示机械全开，1 表示机械全闭，该值不是夹持力。实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。张开或闭合命令仅表示动作意图。实测归一化位置距机械端点 0.02 以内归为对应端点，其余归为 partially_closed。程序直接判定全开与中间位置，仅全闭时由 Decisions 看图判断；程序判定独立记录来源，不伪造模型回答。缺失或无效反馈明确标为不可用。夹爪反馈仅用于持物和空爪题；抓后身份记录为连续执行推定。

初始编排必须声明 support_table_refs，引用 task_selectors 中实际可见且参与支撑测量的 table；配置区域不是桌子实体。程序通过当前或缓存实测桌面多边形，唯一绑定控制台区域及支撑高度。Responses 保留方向、序号和用户原话。Decisions 在当前标注图和基座测量上判断对应的永久 ID，机器人左为 +Y、右为 −Y。证据不清时保持未知，特定目标绑定后可跨位置沿用。

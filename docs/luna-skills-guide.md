# Tianji Luna MuJoCo 技能手册（当前：2026-10-23逐件拆包）

2026-10-23 最终规则：VLA后无论当前Decisions能否确认黑件，双臂都必须回固定观测位姿。当前位置确认黑件时，先用同次RGBD分割测量黑件并加入保留背景场景（头部优先，深度不足依次尝试左/右腕）；未确认时不建立黑件，移除过时活动粉件层后回位。回位后重新采三RGBD、联合Decisions确认并分析全部当前清单实体，再重建全局2cm TSDF/ESDF；首次桌面cuboid继续复用。所有粉/黑抓取前都在观测位姿，禁止VLA位姿直接抓取。最后黑件入盒确认后也必须沿最新场景回初始位姿，成功回位才完成任务。

GraspGen每轮保持3/6/10次服务调用、最多300/600/1000个原始返回。全部结果先按执行臂转换并进行方向筛选、工具+Z深入1cm及名义可达性筛选，再按原分数顺序最多取256/512/1000个进入IK。禁止先按分数截断再做方向筛选，避免漏掉低分但方向合格的位姿；未提交IK的合格者单独记录，不记为失败。完整原始位姿和筛后索引均归档，官方推理与cuRobo算法不变。

回放候选四阶段：完整GraspGen生成位姿（含超出本轮池预算的输出）、方向/名义可达性预筛选与工具+Z深入1cm、batched IK最终可用性、plan_grasp_batch实际调用结果。保留轮次/原始索引和执行臂；每个位姿绘制XYZ三轴。颜色轴仅展示旋转矩阵，不新增或改变抓取位姿。未送入后续算法的候选不伪造成功/失败。

最终完成边界：最后一个黑件经`verify_drop`确认入盒后，也发布右盒投放后障碍包络并调用`move_to_observe`恢复两臂初始关节位置；回位成功才将任务标记完成。该回程不重建TSDF/ESDF。粉件和黑件抓取前均回观测位姿并重建全局场景；VLA后局部黑件障碍仅用于回位运动。

Responses程序在运动前检查拆包后的观测依赖：`verify_unpack`成功分支必须先`observe_scene`，再选择黑件/目的地/执行臂。若模型已给出该观测但排在这三个读取步骤后，只将已有观测前移，不增加技能、跳过核验或越过运动；保留原Responses和顺序调整事件。缺失观测则在初始阶段拒绝程序。



当前cuRobo调用边界、13配置常驻预热、限制及最新验收见[逐入口审计](curobo-public-api-audit.md)。以下保留历次实验记录；其旧配置/耗时不是当前实现承诺。

当前独立项目只有一次Responses编排和Decisions视觉判断，无JEV运行时。注册目录 `demo_skills/` 共21项，执行器 `DemoRunner` + `UnpackWorkflow`；用户新指令优先于下方历史兼容记录。

2026-10-22结果判断修订：当前拆包主线的桌面释放、拆包、右盒投放及候选外观角色使用官方Decisions `predicate`，按条件成立概率≥0.8接受、≤0.2否定、中间unknown。原ID选择仍使用choice及其confidence；两者不能混称。原始回答保留，生命周期历史字段confidence在新结果中保存predicate概率，并以confidence_rule明确来源。桌面释放的XY由实际TCP反馈和既有持物几何核对目标区域，当前RGB判断粉件已经放到主桌且离开夹爪；释放时的推导位置不是释放后实测物体位置。相关378项回归通过；冻结原图的放置/拆包/投放正例通过、对应反例拒绝，两次抓取前身份关联通过。两件完整运动另行验收，记录见data/decision-contract-probes/。

2026-10-21 最新CAD修订已接入并同步Ubuntu，正在进行完整运动验收：碰撞阈值改为10mm；原粉件ID保持，拆包确认后给黑色产物独立entity ID，工序记录管理输入/产物与完成数量。蓝盒、粉件、黑件使用用户CAD真实尺寸，桌子保留规则几何。旧5mm、规则物体尺寸及粉黑共用entity ID说明属于前版，不作为最新验收依据。

场景编辑与skill边界：`GET /api/cad-assets` 提供三种已登记CAD的类型、原始米制尺寸和默认颜色。前端保留asset_id，允许重复添加/删除、命名颜色或#RRGGBB及位置/yaw编辑，CAD尺寸只读；真实MuJoCo mesh相对body的平移和旋转随预览组合，不能把mesh中心误当物体中心。编辑器不会创建视觉清单或证明物体身份，仍需确认场景并执行初次Responses。CAD回传、40实例、mesh姿态预览及原有场景交互回归通过；CAD静止与落体接触回归通过，完整任务验收仍在进行。

10mm规划修订：共享hard clearance默认与下限为0.01m；IK、grasp/place/附物plan_pose、双臂规划、实际插值命令、开爪包络及release高度复核使用同一设置。release另有0.1mm规划reserve，实际门为10mm。load_settings遇到历史合法[5,10)mm值会发warning、升级为10mm并重算release；原文件不静默覆盖，下一次显式保存才持久化。运行API拒绝低于10mm的值。20mm体素、单次10mm深入、5mm IK位置容差或测量不确定度等不同含义数值不随之改动。机器人YAML/URDF未改，不重建已有工作空间atlas。

清单与回放修订：task_progress的process_records、process_order、pending_unpack_ids、processing_ids、completed_process_ids及all_entities驱动“全部物体/待拆包/处理中/已完成”。完成数量按completed_process_ids，不把粉件和黑产物各计一次。执行图按process_id合并两段动作，明确显示input_entity_id和output_entity_id；历史无process记录的实验继续按原物体ID展示。skill参数、几何和图片证据不因UI分组改写。前端完整315项测试及Vite构建通过；资产目录API经本地真实TestClient核对三种CAD与真实尺寸。完整物理任务验收仍在进行。

回放按固定工序与实际发生顺序展示输入、产物各自ID及两次抓取、VLA和投放；阶段展开后可看模型原图、SAM3和各轮抓取候选。2026-10-21修正执行图说明以覆盖新流程，skill执行行为无变化。

每件物体完整经历：粉件左盒抓取→共享桌面中心放置核验→双臂自然VLA位姿→明确mock替换→当前位置Decisions拆包确认并创建产物ID→沿保留场景回观测位姿→新观测帧Decisions/SAM/RGBD建图→黑件右臂抓取→attachment普通plan_pose到右盒上方→松爪drop→Decisions完成核验。第二件必须等第一件完成。原粉件记录保留为consumed，黑件使用新ID；旧点云/候选/计划不跨实体或版本使用。

回放分别显示VLA拆包后结果观察与返回后的抓前观察；post_unpack capture仅作为结果证据。运行审计允许mock完成与当前结果capture的仿真时间相等，但仍要求严格递增的实际采集perf时间；不能把没有经过物理步的当前采图误判为动作前图。工序之间清空target后返回观测位姿仍合法，待拆包结果未确认的当前target则禁止返回。

双臂观测位姿之外不得建立TSDF/ESDF。全局仅保留最近地图；显式release/mock动作包络是动作推导约束，不是新的测量地图。冻结N件后拆包任务普通capture预算为1+5N（仅搬运兼容任务为1+4N），执行预算为128×max(1,N)，不设旧四件上限；无隐式恢复采图。程序结构仍限64节点、4层，不允许嵌套目标循环。

碰撞阈值10mm、体素2cm。attachment默认24球、最多31，同手全部9个夹持链接允许接触，TCP相机支架/上游/异臂/世界仍检查。黑件普通plan_pose和双臂自然pose遵循高级设置五次插值与time_scale；五次插值不是固定5倍时间。松爪前检查当前闭合姿态及到全开的整手扫掠。

当前CAD高度先验为粉件110mm、平放黑件44mm，仅在实测有缺失且顶部/支撑一致时补；不再固定补10cm。完整测量不补，投放后姿态不定不补世界Z。Decisions可判语义工序/遮挡，不能生成未知公制几何。原始测量与synthetic completion来源分开保存。


当前身份与队列：初始粉实体ID与全源初始编号永久不变，含count未选项。每个选定输入有独立process-*；verify_unpack保留真实BEFORE释放证据校验，以一次三视图融合unpack_result的predicate概率≥0.8确认后创建product-*。工序记录input/output关系，原粉ledger consumed保留stage0，新产物从stage1开始，后段geometry/attachment/grasp/drop均用产物ID，程序$target通过工序切换。产物私有参考来自确认时CURRENT三RGB，公开仅小creation_evidence及product_created事件。后续一次associate_inventory同时问identity_N角色和wrapped_identity_N原粉身份，必要INITIAL参考与全部CURRENT图在同一请求，二者同request_hash；原粉件camera2候选覆盖保留，判断融合三相机。多个完成黑件保留独立历史记录，右盒当前测量采用kind=collection、members=[product IDs]的集合占据，不逐成员视觉重识别。完成按completed_process_ids；初始schema不设旧4/64实体上限，64程序节点预算不变，执行/正常观测预算随件数增长。单请求仍受实际模型输入预算约束。

CAD盒配准：已知STEP仅提供形状，mapper清洁当前库存RGBD以最多8面、每面256假设、8000样点RANSAC提取底面和两独立斜壁（跨度≥30mm），产生upright yaw+XYZ候选；全点至少85%距CAD表面≤3mm，否则缺面、歧义或不符均停止。半周对称yaw模π，不用仿真位姿。每盒完整49凸mesh保留空腔，source=known_cad_measured_registration，CAD/逐片SHA及当前ID/obs/gen绑定；原2cm体素保留。源lift、运输高度、投放定位与成功动作包络共用native+CAD完整边界；缓存按双盒98mesh扩容，同名更新清旧mesh。双盒59965/60980点CPU标注mask fixture均100%满足3mm；中心误差0.49/0.60mm及yaw误差0.015/0.068°只用于独立GT测试断言，不输入运行配准，也不代表SAM、GPU或任务验收通过。

## 当前全部注册入口

### `bind_inventory`

以初次同帧身份union、左腕源件可见和可见遗漏none绑定all/count清单，冻结粉件ID与显示编号，并为选定粉件创建工序。先验描述隐藏黑件，不将其列为已观测实体；黑件ID由后续拆包确认创建。

实现与前提：[contract](../demo_skills/bind_inventory/contract.json)、[SKILL](../demo_skills/bind_inventory/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`inventory_finalized`, `task_assessed`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `build_scene`

粉件和黑件均必须双臂在观测位姿，capture/generation/实际关节与观测一致。用全部当前清单实体的三RGBD建立全局2cm TSDF/ESDF；首次RANSAC桌面cuboid复用，不建立新桌面体素。选中目标层按抓取政策禁用，其他实体保留碰撞。只维护最新全局场景，禁止VLA位姿全局重建。

前提：{"observed": true, "inventory_finalized": true, "target_selected": true, "arm_selected": true, "destination_ready": true, "grasp_completed": false}。输出：scene_ready, views_ready。

### `choose_arm`

校验Responses每件to_table_arm/to_right_bin_arm与当前工作空间，不暗换臂；A左B右。黑件运输目标在实际lift后校验，不用盒底预筛运输姿态。

实现与前提：[contract](../demo_skills/choose_arm/contract.json)、[SKILL](../demo_skills/choose_arm/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`arm_selected`, `arms_evaluated`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `drop`

到右盒上方后检查实际位置和带件碰撞，再开夹爪并解除attachment，让黑件自由落下；只记录执行证据，尚未完成物体。

实现与前提：[contract](../demo_skills/drop/contract.json)、[SKILL](../demo_skills/drop/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`drop_completed`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `execute_grasp`

位置模式执行approach/grasp/close/lift并保存反馈；不通过视觉判断持物。夹爪关闭不修改手臂位置目标。所有单/双臂轨迹每个物理步检查上一拍实际位置参考，关节误差超过原0.03 rad跟踪界限即停止全部活动臂，保留实际/参考/峰值证据；不继续后续动作，也不把此执行跟踪界限当作末端放置精度。

实现与前提：[contract](../demo_skills/execute_grasp/contract.json)、[SKILL](../demo_skills/execute_grasp/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`lifted`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `execute_place`

执行带件接近/放置、扫掠检查、松爪detach和撤离，冻结实际释放TCP及执行链；动作完成尚不计任务完成。

实现与前提：[contract](../demo_skills/execute_place/contract.json)、[SKILL](../demo_skills/execute_place/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`place_execution_completed`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `execute_transport`

执行已校验的带attachment普通pose轨迹；核对当前scene、held binding和实际起点，无采图/建图。

实现与前提：[contract](../demo_skills/execute_transport/contract.json)、[SKILL](../demo_skills/execute_transport/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`transport_execution_completed`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `generate_grasps`

所有粉件和黑件抓取均须双臂在观测位姿；黑件使用verify_unpack内部回位后的确认视图。当前观测、规划snapshot及scene须绑定同一capture，实际关节不能在采图后改变，否则停止。每次抓取前观测做Decisions，不逐个GraspGen候选调用。对当前目标几何按固定轮次预算采样GraspGen候选：第1/2/3轮分别256/512/1000项，每轮单独生成，不累计补足。有合法显式高度补全时保留实测点与补全点的完整集合，再附加仅供GG推理的同一实测XY凸包侧表面均匀采样；周界/高度间距不超过2mm，总输入上限12000点，超限直接停止。原测量、原补全、attachment、地图和frame不变，2cm体素不变；没有补全时使用实测点。graspgen_input_sampling记录构造来源、间距、原几何点数、新增数和实际总数，实际输入hash/合成点数按提交数组记录。补全来源、当前观测ID、点数和原测量前缀不一致则在推理前停止。cloud frame和物体参考点仍使用原测量，名义可达性筛选最低Z使用合法完整几何；回放区分原云、先验补全和实际输入，并保存实际输入哈希/点数、分数、批次和索引。按左右臂实际工具坐标转换（Z方向一致，X/Y相反），先按向下45度和远离机器人规则剔除候选，仅对通过者沿工具+Z平移10mm一次，再交给batched IK。被剔除者保留原始位姿，insertion_applied=false；operation_order记录筛选、插入、IK的顺序。只保留工具Z与世界竖直向下偏角不超过45度，且水平投影不朝机器人基座的候选；向内判据为工具Z水平分量与基座到目标向量点积为负。然后作冻结槽位的名义可达性筛选，再交给batched IK和plan_grasp_batch。名义筛选保留5mm内侧可达余量，不添加2mm放置误差上限；可能保守拒绝候选，不能当作完整IK不可达证明。本生成调用在方向过滤无候选或服务错误时停止；黑件投放名义可达性过滤全空则返回完整空池证据，交由plan_grasp使用原有剩余采样预算，不降低过滤标准。后续plan_grasp仅在完整源IK及预测FK复核后合格项全零，或完整确定的整池三阶段规划失败时，允许同一冻结输入最多追加第2轮512项和第3轮1000项，总3池最多请求1768项，共用原180秒；最终FK拒绝、命令/硬检查失败及错误/未知不授权补池；生成器本身不循环恢复，详见plan_grasp预算。第1/2/3轮分别使用3/6/10次原服务调用（每次官方num_grasps=200、topk_num_grasps=100），再按分数稳定排序截取至该轮预算；总服务调用最多19次。requested_candidate_count记录该轮预算，raw_candidate_count记录服务实际返回总数，sampling_evidence.rounds中的generated_candidate_count为截取后进入方向过滤的数量；raw不等于预算。每轮保存sampling_round及原始索引，每次服务调用前后检查取消。源IK成功后另有plan_grasp的静态宽度夹爪质量门：全原点加原padding检查单一预测q_width的own hand固定9链接接触组件外自身几何（保留TCP）；此预测不修改本生成器输出、不是实际闭合/连续扫掠证明，最终FK仍须复核，详见plan_grasp契约。持物接触边界由carrying_touch_links(arm)统一定义：仅该臂自有夹持组件的固定9个链接，A为gripL_Base及gripL_Link1/2/3/4各L/R，B精确镜像gripR_。不按前缀自动豁免；TCP_Link含相机/支架，必须继续检查，所有上游臂、对侧手/臂和未知新增链接也保留。只对当前有效carrying binding添加attachment与这9链接的双向接触对，不删除任何机器人/物体球、不改变普通机器人self pairs或世界碰撞。预测q_width使用同一接触集合，仍核对宽度/模型合法性、原点包络与TCP等非接触球；source及final FK容差/方向门不变。这是有意允许夹持组件接触的语义，不是把Link4称为接触垫，也不是证明物体与活动手指永不接触。初观测云在闭合前后变换不能当作实际闭合后的完整物体位姿；原5mm uncertainty+1mm margin仍完整用于世界/组件外包络，不因理想RGBD或本次结果缩小。释放张爪沿用同一有意接触边界；既有当前闭合持物与全开detach两端状态检查、上游/对侧检查、实际夹爪反馈与确认张开后detach均保持，不宣称检查了物体对活动手指的无碰撞扫掠。有效持物、原scene/owner/arm绑定、固定XY/R和10mm世界门仍是前提。2026-10-22 投放可达性修复：黑件抓取在方向筛选、工具+Z深入10mm之后、源batched IK之前，使用本次观测配准的完整右盒CAD空腔中心、已发布障碍最高Z、当前完整物体几何最低Z和自身URDF手腕偏移，筛选保持抓取姿态的投放目标。名义TCP高度为盒体占据最高Z+100mm+抓取TCP到物体底部的距离；肩腕距离须在URDF两段上界减5mm余量内。缺少或过期的观测绑定停止；保留候选原顺序、分数及索引。此步骤为几何必要条件筛选，不替代实际持物plan_pose、attachment、10mm碰撞检查或五次插值。两段抓取不共用桌面槽位：粉件检查桌面目标，黑件检查右盒上方目标。v17真实持物姿态的水平肩腕距离625.6mm已超过602.1mm上界，原流程漏掉了该筛选。2026-10-22 黑件投放名义可达性筛选后整池为空时，generate_grasps返回完整空池与逐候选筛选证据，由plan_grasp在原256→512→1000三池、总1768及共享180秒预算内使用下一池。空池不调用IK，也不伪造IK结果；回放记录no_transport_reachable_candidate及tested_count=0。只有完整有效筛选全拒绝证据允许该分支，服务错误/过期/未知仍停止；第三池为空明确返回grasp_no_transport_reachable_candidate。v19冻结原图点云的第二池512候选保留3项，原第一池256全拒绝不再提前终止。2026-10-22 投放Z按宽松要求选择：保持盒腔中心XY和实际抓取/lift旋转，先由完整持物最低点算出盒子已发布最高占据+100mm的最低TCP高度；再取该最低高度与“手腕和自身肩部等高”对应TCP高度的较大者。这样利用允许的Z自由度缩短肩腕距离，仍保留100mm物体离盒余量，最终普通plan_pose和10mm硬碰撞检查决定可执行性。抓取前名义筛选与实际持物目标共用transport_tcp_height；不把高出盒子100mm误当成唯一允许的高度。v21第二件候选304原名义肩腕597.191mm因固定Z被排除，改为等高后589.650mm，可进入源IK。XY/R、目标盒坐标、世界模型、attachment和时间延长不变。 2026-10-23：仅拆包后的黑件，方向筛选通过后，按当前执行臂姿态在绕工具Z相差180度的平行夹爪等价朝向中选较近者，记录原姿态及half_turn；再沿工具+Z插入10mm一次。变换发生在batched IK前，不旋转已解轨迹，不增加候选数量。 当前前提grasp_view_ready：粉件和黑件均来自双臂观测位姿的同次三视角采集；verify_unpack内部已强制回位，运行时校验工序/代次/关节/地图绑定。桌面名义放置可达性也按夹爪前端对准目标，不能用点云质心。

实现与前提：[contract](../demo_skills/generate_grasps/contract.json)、[SKILL](../demo_skills/generate_grasps/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`grasps_ready`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `identify_destination`

包装阶段绑定共享数值桌面区域，黑件阶段绑定初次确认右盒；拒绝与阶段不符的symbol。共享完整区域不分区，XY使用中心。

实现与前提：[contract](../demo_skills/identify_destination/contract.json)、[SKILL](../demo_skills/identify_destination/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`destination_selected`, `destination_ready`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `identify_target`

按Responses固定工序顺序选择原粉件ID；拆包确认创建黑件后，同一工序第二段选择独立product ID，前工序未完成不可选择下一工序。输出target及当前几何引用。

实现与前提：[contract](../demo_skills/identify_target/contract.json)、[SKILL](../demo_skills/identify_target/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`target_selected`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `mock_unpack`

双臂自然位姿到位后，MuJoCo模拟VLA拆包装：当前粉色CAD包装件替换为右臂工作区原始尺寸44×56×74.5mm的黑色CAD工件，局部Y转90°平放，世界高度44mm，底部初始距桌面1mm后受重力自然接触；mock保留输入工序，随后视觉确认才创建独立产物ID；只记录旋转后CAD边界的动作推导包络，不重建RGBD场景。

实现与前提：[contract](../demo_skills/mock_unpack/contract.json)、[SKILL](../demo_skills/mock_unpack/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`unpack_pending`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `move_to_observe`

双臂沿最近保留场景回固定初始观测位姿，遵循10mm碰撞检查和高级时间设置，到位采三RGBD，不自动建图。用于每个粉件前、最后黑件入盒后的收尾；verify_unpack内部也必须回位。最后成功回位才报告任务完成。已绑定抓取视图后不得重复运动使视图失效。

前提：{"scene_available": true, "grasp_held": false}。输出：observation_pose_ready, grasp_view_ready。

### `observe_scene`

双臂必须在固定观测位姿。首次用三RGB联合Responses理解指令并建立固定ID清单和技能程序；后续用同次三RGBD、SAM3及一次联合Decisions关联当前清单实体。VLA后复用verify_unpack回位后的确认capture；全部抓取前均须获得当前全局观测，不允许在VLA位姿直接抓取。多个SAM描述和临时C标签不会生成多个永久ID。此技能仅感知，不发布碰撞图。

前提：{"grasp_held": false, "grasp_view_ready": true}。输出：observed, inventory_ready。

### `plan_grasp`

所有粉件和黑件抓取均须双臂在观测位姿；黑件使用verify_unpack内部回位后的确认视图。当前观测、规划snapshot及scene须绑定同一capture，实际关节不能在采图后改变，否则停止。每次抓取前观测做Decisions，不逐个GraspGen候选调用。仅完整当前源IK及预测FK复核后合格项全零，或整池所有合格候选明确三阶段规划失败时，从同一冻结RGBD/补全/frame/目标/臂/scene token/槽位/设置最多追加第2轮512项和第3轮1000项候选，首轮256项，总请求不超过1768项；追加生成、源IK与轨迹共用原plan_timeout_s默认180秒，不按轮重置。输入摘要和模型身份必须相同，预算只能按第1/2/3轮256/512/1000变化；45度、背离机器人、深入1cm、槽位及10mm世界碰撞检查保持（IK位置容差仍为5mm）。过滤无候选、unknown、过期、错误、超时、取消、最终FK拒绝或命令/硬检查失败不重采。规划补池须pool_exhaustion匹配同轮/目标/臂/generation/scene/settings/tool版本与冻结关节，唯一完整覆盖全部合格候选且零未规划；逐项SDK正常Grasp planning completed.状态、无error和明确合法失败phase。裸错误名或部分结果不授权补池。三轮源IK全空返回no_collision_free_source_ik；曾规划而三池耗尽返回planning_no_valid_candidate，均保留逐轮结果、实际服务返回raw_candidate_count、预算截取后候选及真实轨迹尝试计数；各阶段回放保存sampling_round和原索引。源IK筛选后按当前源盒完整native体素与同ID当前完整盒壳并集/邻件占据边界（CAD须49片mesh及目录/逐片SHA通过published_bin_shell_layers当前版本校验，历史规则盒为五片cuboid；不授权任意mesh）和完整目标球包络（同一配置预算，上限31，默认24）推导world-Z离箱高度，不低于配置值；固定张开夹爪、无attachment批量规划三段，首个可行批次按七关节累计运动量选取。边界或来源无效，或三池耗尽无有效轨迹则停止。源IK成功项必须提供同scene/臂/tool frame/关节配置绑定的官方预测FK。保留原GraspGen名义工具姿态（已深入一次1cm）和原IK误差；仅当预测FK仍在当前名义位置/旋转容差（默认5mm/.05rad）、工具Z向下45度内且不朝机器人时，将该FK统一用于lift推导、plan_grasp_batch、contact_goal_pose_world与plan.T_world_tcp，不再次深入。最终抓取phase末端官方FK仍相对原名义姿态复核同一容差与方向，防止两次容差叠加；失败不触发轨迹补采。缺失或过期FK证据停止。source_ik_fk是预测而非实测，attachment和固定放置旋转继续依据实际反馈。归档保留nominal_T_world_tcp、source IK q及原误差、预测FK、final FK/误差和round/source索引。raw success_count与eligible_count分开，candidate_filters保存明确筛除原因；全部合格项为空沿用既有三池/180秒预算，耗尽文案为无可用源目标，不伪称原IK全无解。源IK的官方FK通过原nominal容差和方向门后，Demo另做静态刚体宽度质量预测：使用当前完整实测/显式补全几何（不使用GG专用密采样），从自有URDF/Link3碰撞STL的最大相向平面及MuJoCo FK/mimic取得夹持轴与jaw gap。先按不超过0.01rad网格验证平行、单调闭合，再在唯一宽度区间内最多24次二分，求pad间距等于原云宽度的q_width（残差不超过1e-8m）。它是假设刚体可居中的宽度预测，不是实测闭合角、云居中或连续闭爪扫掠证明。碰撞计算不平移、不裁剪原点，在单一q_width检查全部点加原uncertainty_m+attachment.margin_m对固定9链接接触组件外夹爪球（含TCP和未知新增链接）的净距；原球collision/self buffers保留，不重复加世界10mm，也不用拟合attachment球做此筛选。负净距明确剔除；无合法整云宽度区间不否决局部抓取，按固定TCP非接触球检查完整原云，坏点/来源/模型停止，不能默认通过。candidate_filters每个原index仍仅一条，嵌入predicted_gripper_compatibility，保存预测来源、arm/tool、完整点数/hash、测量数与补全来源、原padding、width/轴、q/区间/残差、最小gap及point/link/sphere索引、接触豁免与模型指纹。raw IK成功数保持，只有eligible减少；筛后为空仍沿用既有三池/单180秒预算。最终抓取phase官方FK也重新检查这一预测，失败仍是planning_grasp_endpoint_rejected，不授权补池。原姿态/分数/顺序/索引/round及一次+1cm不变，普通Algorithms不启用此门；实际持物反馈、attachment球拟合及所有世界/自身硬检查保持。own-model资源在服务进程内只读，URDF/YAML/pad-STL哈希标识缓存；资产变更须随robot_model一同重启加载，不支持热替换。持物接触边界由carrying_touch_links(arm)统一定义：仅该臂自有夹持组件的固定9个链接，A为gripL_Base及gripL_Link1/2/3/4各L/R，B精确镜像gripR_。不按前缀自动豁免；TCP_Link含相机/支架，必须继续检查，所有上游臂、对侧手/臂和未知新增链接也保留。只对当前有效carrying binding添加attachment与这9链接的双向接触对，不删除任何机器人/物体球、不改变普通机器人self pairs或世界碰撞。预测q_width使用同一接触集合，仍核对宽度/模型合法性、原点包络与TCP等非接触球；source及final FK容差/方向门不变。这是有意允许夹持组件接触的语义，不是把Link4称为接触垫，也不是证明物体与活动手指永不接触。初观测云在闭合前后变换不能当作实际闭合后的完整物体位姿；原5mm uncertainty+1mm margin仍完整用于世界/组件外包络，不因理想RGBD或本次结果缩小。释放张爪沿用同一有意接触边界；既有当前闭合持物与全开detach两端状态检查、上游/对侧检查、实际夹爪反馈与确认张开后detach均保持，不宣称检查了物体对活动手指的无碰撞扫掠。有效持物、原scene/owner/arm绑定、固定XY/R和10mm世界门仍是前提。2026-10-22 黑件投放名义可达性筛选后整池为空时，generate_grasps返回完整空池与逐候选筛选证据，由plan_grasp在原256→512→1000三池、总1768及共享180秒预算内使用下一池。空池不调用IK，也不伪造IK结果；回放记录no_transport_reachable_candidate及tested_count=0。只有完整有效筛选全拒绝证据允许该分支，服务错误/过期/未知仍停止；第三池为空明确返回grasp_no_transport_reachable_candidate。v19冻结原图点云的第二池512候选保留3项，原第一池256全拒绝不再提前终止。 2026-10-22 CAD局部抓取修复：整件点云沿闭合轴的宽度超出jaw gap范围，不再直接否决GraspGen的局部抓取。它可能包含不进入指间的宽底部。无宽度根时q_width_rad保持null，width_prediction_status=whole_cloud_width_outside_range；仅当非接触球全部属于固定TCP_Link时，用张开角检查完整原云与固定TCP/相机支架净距，非接触部件若随闭合运动则停止，不能默认通过。宽度有根时仍保留估计及原检查。所有点和padding保留，负净距仍拒绝；实际闭爪与lift状态才判断抓稳，实际attachment检查不变。v23已通过IK的CAD候选484原被整云115.9mm宽度误拒，现通过固定非接触几何检查。 2026-10-22 抓取规划性能：Demo源IK合格池仍完整保留，按原分数排序分成每批最多4个真实候选，普通Algorithms兼容调用仍为16；首个有可执行轨迹的批次选七关节运动最短项，失败再处理下一批，不丢弃剩余候选。GPU服务将1到4项补齐到官方4容量，5到16项仍补齐16；只复制最后真实候选做固定shape填充，不增加逻辑候选，不改SDK、不改变碰撞与最终命令验证。原256/512/1000生成预算和共享超时保持。 当前前提grasp_view_ready：粉件和黑件均来自双臂观测位姿的同次三视角采集；verify_unpack内部已强制回位，运行时校验工序/代次/关节/地图绑定。桌面名义放置可达性也按夹爪前端对准目标，不能用点云质心。

实现与前提：[contract](../demo_skills/plan_grasp/contract.json)、[SKILL](../demo_skills/plan_grasp/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`plan_ready`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `plan_place`

粉件保持抓取旋转，夹爪前端对准固定共享区域中心XY=[0.50160,0.11646]m，无分区。SDK目标用标定前端偏移转成法兰；不以点云中心对准、不要求整个附件投影包含。Z根据attachment与当前夹爪及实际世界碰撞选取尽量靠近10mm间隙的高度。直接官方plan_grasp派生plan_place，不使用plan_place_batch。前两段闭爪attachment，开爪后detach撤离；仅当前闭合/全开端点检查，无开爪扫掠。遵循高级设置，版本不符、无解、碰撞或反馈不满足时停止。

实现与前提：[contract](../demo_skills/plan_place/contract.json)、[SKILL](../demo_skills/plan_place/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`place_plan_ready`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `plan_transport`

黑件lift后带attachment，用官方普通plan_pose将夹爪前端放到右盒靠机器人的半区上方。机器人+X向前，XY取当前配准盒腔包围范围中心向-X偏移X宽度的1/4（近侧半区中心）；保持实际lift旋转。用户坐标均为夹爪前端，SDK法兰目标减去R@[0,0,0.185]，不使用持物中心，不检查整件投影包含。Z保留宽松选择：持物最低点高于当前盒体占据上界10cm，并可提高到缩短肩腕距离的高度。附件与机器人/场景的三维10mm碰撞检查、当前闭爪/全开detach端点及高级设置五次插值/时间伸长保持。盒边界只来自最新发布的该ID体素/CAD配准，不从仿真真值或原SAM飞点求高度；版本、边界不完整或无解时停止。

实现与前提：[contract](../demo_skills/plan_transport/contract.json)、[SKILL](../demo_skills/plan_transport/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`transport_plan_ready`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `prepare_unpack`

桌面放置视觉确认后，沿最近地图普通规划双臂到桌面上方自然位姿；无采图、无建图。

实现与前提：[contract](../demo_skills/prepare_unpack/contract.json)、[SKILL](../demo_skills/prepare_unpack/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`vla_pose_ready`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `verify_drop`

合法持物、attachment运输、实际开爪后，一次Decisions融合CURRENT三RGB、BEFORE三RGB及同盒右腕/头部区域放大图，判断本次是否有新增黑件进入右盒。随机落体可能堆叠，不要求当前逐个数清N+1件；须有新增可见物体本体或新增工件轮廓特征，不能仅凭旧件移动、旋转、视角变化、历史数量或动作记录确认。完全遮挡且没有新增视觉证据时unknown，不推进。predicate current_item_added概率≥0.8确认、≤0.2否定，其余unknown。完成数只作上下文，不重新辨认盒内个体ID。辅助图只放大原像素，来自同观测已发布完整盒体占据的标定投影，不SAM、不建图、不使用仿真真值；完整CAD壳校验与归档保留，但不重复发送壳片明细给模型。成功完成独立产物及process，保留原始回答、完整反馈链与released_in_bin_and_visible_addition语义。

实现与前提：[contract](../demo_skills/verify_drop/contract.json)、[SKILL](../demo_skills/verify_drop/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`cycle_completed`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `verify_grasp`

lift之后按夹爪位置相对空夹闭合基准判定held；成功才允许prepare_attachment。未知/空抓不推进，禁止模拟器物体位置替代反馈。

实现与前提：[contract](../demo_skills/verify_grasp/contract.json)、[SKILL](../demo_skills/verify_grasp/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`verified`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `verify_place`

合法持物、attachment、实际释放反馈和撤离完成后，在当前位置采新三RGB核验当前粉件。实际TCP反馈结合抓取时实测物体参考与attachment几何，数值校验释放XY及物体包络在固定共享区；release_position_check明确标注为开爪时刚体推导，不是释放后物体实测位姿，也不证明视觉成功。单次Decisions发送初始全部实体、活动ID、工序/生命周期记录、当前实测夹爪角度及开闭限位、INITIAL三RGB/注释和CURRENT三RGB/区域图，同一请求用两个独立predicate：place_result判断主桌上可见独立粉件，workpiece_still_held判断粉件仍被手指夹持；前者概率≥0.8且后者≤0.2才通过，不要求从RGB看清底部物理接触；不再要求模型重算公制XY或把高起轮廓装进桌面投影。概率≥0.8确认、≤0.2否定、中间unknown；原始probability不叫choice confidence。源盒粉件、机器人部件不计。INITIAL是历史，先历史后当前，头部当前图最后，三视角融合；动作记录和仿真真值不能代替当前像素。成功只推进table_released，不完成工序；不SAM、不建图。旧非拆包路径仍使用choice place_result及最后一次的demo_unlisted_pink，不能把旧choice响应塞给当前predicate请求。

实现与前提：[contract](../demo_skills/verify_place/contract.json)、[SKILL](../demo_skills/verify_place/SKILL.md)。输入来自当前物体/观测/执行上下文；所有generation、scene、owner与绑定证据必须一致。输出事实：`place_verification_completed`；前提不满足、无有效结果或unknown均不能授权后续动作。

### `verify_unpack`

mock VLA完成后在当前位置用一个Decisions请求联合三相机RGB判断主桌黑件。若任一视图确认黑件，使用同次RGBD（头部优先，深度不足依次尝试两腕相机）分割当前黑件，加入保留背景场景并移除过时粉件层；若不能确认，则仅移除活动拆包件及过时粉件层，不建立黑件障碍。两种情况都必须沿保留场景将双臂移动到观测位姿，到位采新三RGBD，再用一个Decisions确认桌面黑件；概率≥0.8才创建独立产物ID并绑定该次观测，否则停止。随后observe_scene分析全部当前清单实体，build_scene重建全局2cm TSDF/ESDF后才能抓取，首次桌面cuboid复用。服务错误或可见但所有相机深度均不足则停止，不用仿真真值代替测量。最多两次结果判断、一次回位；程序不得重复插入内部回位。

前提：{"unpack_pending": true, "vla_pose_ready": true}。输出：unpack_verified, grasp_view_ready, observation_pose_ready。

## 归档与验证

2026-10-22复核：verify_place的SKILL正文旧choice/区域视觉复测说明已与当前predicate合同对齐；没有改变本次运行中的代码或动作。

离线实验审计的候选证据读取已同步支持predicate原始概率，并保留历史choice规则；5项正/负/错误类型回归覆盖该边界。此审计改动不进入控制器，skill行为无变化。

2026-10-21本次协议与预算核对：当前正文、README、感知说明和observe_scene/verify_place/verify_drop契约已对照源函数同步为单次joint associate_inventory、融合place_result及单题current_item_added；新增[Decisions问题清单](decisions-questions.md)列输入、选项、no/unknown边界及仍存在的非拆包两请求兼容路径。抓取预算同步为256→512→1000、最多1768预算候选/19次服务调用，官方单次num_grasps=200、topk_num_grasps=100不变。本轮文档工作只做源代码阅读、文本和JSON检查，不据此声称模型、运动、部署或完整测试通过；后文已标明的历史实验保持原结论。

右盒运输的中心XY现在指公开SDK的TCP_Link工具目标，直接取当前实测盒底区域包围框中心；不再强制持物参考点精确对中。保持实际lift旋转后，完整持物点云（含明确记录的高度补全）加原uncertainty_m+margin_m的水平投影必须包含于当前实测凸区域；不满足即停止，不能省略attachment或放宽10mm三维轨迹碰撞门。该投影检查只验证释放前的水平包含，不保证自由落体后的姿态或位置。`right_bin_occupied_bounds(destination, scene, observation)`读取当前`collision_layer_bounds`中精确`semantic_instance::<right_bin_id>`启用层的完整native单元边界，并合并当前声明的同ID全部CAD 49个启用碰撞mesh（历史规则盒为五片cuboid）；逐层校验目录hash、版本和显式推导来源，禁止缺片回退或用原SAM高点、预览子集、别的实例替代。`right_bin_transport_goal`保持实际lift旋转，以保守占据最高Z加10cm作为物体最低点高度，再由带attachment的普通plan_pose验证。规划前事件`transport_goal_derived`及成功计划的`transport_goal_evidence`保存来源、版本、各层名、单元数、体素大小与高度推导；规划失败不丢来源证据。投放视觉定位与成功后的动作包络复用同一边界；后者保留原上界加2cm余量，是动作推导而非新RGBD地图。

运输计划绑定goal、实际命令、holding点云/owner、scene及settings的指纹；重新规划会清除旧执行证据，投放不能复用旧计划的执行成功。动作包络的action_id保留在项目审计模型，进入公开cuRobo几何构造器前剥离。开爪检查按实际driver到全开不大于0.02rad的网格，URDF mimic和区间padding保守覆盖手部对世界的扫掠；self使用原球，同手attachment有意接触规则不变。

各模型调用有独立视觉归档租约，取消/超时后晚回包仍保留实际送入图片与响应。回放按工序显示独立粉/黑ID以及VLA、黑件运输、右盒投放阶段，SAM/GG/IK/轨迹各阶段数据保留。模拟替换保持generation及非目标物理状态，manifest_revision驱动实时和历史几何切换。Windows无ffmpeg时不能宣称MP4通过，Ubuntu必须另验。

本地和Ubuntu按完整文件清单/SHA-256同步，21个contract逐一校验。测试和健康状态不代替完整两物体物理+模型验收。当前实现计划见[2026-10-21计划](superpowers/plans/2026-10-21-unpack-transfer.md)。


场景修订（2026-10-22）：无小桌；桌/左盒/右盒中心分别为[572,-26,532]、[481,573,571]、[481,-625,571]mm，桌与盒固定，粉件Z=180°且有重力。mock黑件以pitch=90°平放、世界高度44mm、底部初始距桌面1mm；三个相机的观测关节角保持。

## 历史兼容搬运流程与审计记录



以下保留此前仅粉件到桌面的兼容实现和历史实验依据。其中14入口、分槽、256球及旧默认等记录只描述当时版本；当前拆包流程以本文件以上部分及现行contracts为准。



### 旧手册正文



本手册对应 2026-10-20 用户修订后的程序。入口 `demo_app.py`，运行目录 `demo_skills/`，执行器 `DemoRunner`。首次 Responses 生成有界技能程序，后续 Decisions 负责视觉判断。TypeSafe/JEV 不参与执行，也不是安装依赖。



源码同步按完整文件清单和 SHA-256 校验，不能只用覆盖解包证明两端一致。当前目录应恰有14份运行时 contract；已经删除的 `bootstrap_scene` 不应残留在 Ubuntu 技能目录。本地辅助脚本 `data/sync_luna_snapshot.py` 在没有运行中任务或持物时同步，先备份审计列出的6份旧技能/脚本/文档，再删除这6个明确路径；其他意外文件只报告差异，不自动删除。该部署校验不改变 skill 行为，也不代替实际搬运验收。



## 观测与状态



初始双臂关节即用户指定的观测姿态：A 为 `[-30.116,55.982,21.322,-109.932,14.629,-29.333,-60.589]` 度，B 为 `[30.116,55.982,-21.322,-109.932,-14.629,-29.333,60.589]` 度。右臂符号关系经 URDF/FK 镜像核对。MuJoCo 机械臂和夹爪使用位置伺服；关节命令是位置，物理积分、接触和力限仍生效。



启动直接执行 `observe_scene → bind_inventory`，不做原地移动或无清单背景建图。三相机 RGB 输入一次 Responses，同时获得全实体清单、搬运清单及程序。编号参考左腕 camera2，从左到右、从上到下；右腕 camera1、头部 camera3 辅助。显示编号仅用于身份。bind_inventory使用同帧三视角RGBD实测质心，按到固定机器人基座世界XY=[0,0]的水平距离sqrt(x²+y²)由近到远绑定执行队列；高度、颜色、形状和当前TCP位置不参与排序，同距保留原编号顺序。指定数量选择最近N件，全部任务选择所有已确认源件；缺少有限三维实测位置则以source_distance_unmeasured停止，不退回图片位置猜距离。transport_ids为执行顺序，source_object_ids及全局ID/显示编号保持固定。每件完整拆包投放后继续队列下一件。execution_order和target_execution_order事件保存来源观测ID、generation、位置、距离、入选ID；排序不新增采图或模型调用。旧非拆包槽位单独按execution_rank分配，拆包仍共用中心。仿真标签和物体真值不输入运行时视觉判断。



Demo 初始 Responses 实体颜色在解析边界去除首尾空白并转小写，精确别名 `蓝/蓝色/藍/藍色`→`blue`、`粉色/粉红色/粉紅色`→`pink`、`灰/灰色/grey`→`gray`；下游清单和分割查询统一使用规范颜色。其他颜色保留其含义，不按 kind、描述、引用或任务推断颜色；未知、红色和混合颜色源盒仍被拒绝。普通 VisualDiscovery 不变，原始模型答案完整保留。2026-10-21 冻结 8bfd 回复先复现失败，修复后相关边界测试63项通过；这不是新的模型或搬运验收。



来源完整性指三路RGB中全部可见粉色源对象已逐个列出且可在camera2区分；没有未解释可见对象时，仅裸露箱底或箱沿被遮挡不构成未知，也不证明隐藏处为空。任一可见对象的身份、数量或源归属不明仍是 `unknown`，已有判断不被强制改写。Responses 的 `supported` 只表示路线、数量和目标选择能否由当前技能表达；可表达任务若视觉证据不足，仍返回 `supported=true` 与来源完整性 `unknown`，由证据门决定停止。



“全部”和指定数量任务在初始、件间观测及清单绑定时都要求同帧 Decisions 确认可见粉色对象没有遗漏（`demo_unlisted_pink=none`，来源为`openai_decisions`，置信度至少0.8）。指定数量可保留来源完整性 `unknown`，但须有足够逐个可信可见目标；仅实测距离最近N件绑定任务和搬运槽位，其余已列对象仍是碰撞实体。身份不明、遗漏检查不明/发现额外对象、证据过期均停止。



“全部”的初始 Responses 完整性 `unknown` 不再提前阻断同帧 SAM3/Decisions，也不被改写为 `confirmed`。`bind_inventory` 对 Responses 原为 confirmed/unknown 使用相同门：初始参考、Responses 及 Decisions 均属于当前观测，Decisions 身份覆盖和遗漏检查的 request hash 一致，全部必需非桌面/标记实体有≥0.8的可信身份，全部源对象有左腕可信身份及实际几何覆盖。通过后单独归档 `source_inventory_resolution`，记录原 Responses 状态、两次请求 hash、观测 ID、身份集合和 none 证据，才冻结清单。未关联的辅助候选保持 unknown 并丢弃几何；它们不能补足缺失身份。无新模型调用、重试或采图，最终逐件放置核验及新鲜可见遗漏检查不变。



之后每件执行程序声明的抓放技能。只有观测位姿的采图可以建立碰撞地图：桌面首次 RANSAC 建 cuboid 后复用，其他仅保留清单实体，TSDF/ESDF 体素均为 2cm。多种 SAM3 描述产生候选，Decisions 关联固定 ID；错误、低置信度或 unknown 不算成功。未关联的深度全部丢弃，不保留背景飞点。



件间已完成件采用执行链身份与新视觉对应：verify_place成功后按ID保存完整`completed_identity_track`（原binding、视觉evidence、确认帧），经`TaskProgress.visual_progress`随source_hint传递。`completed_identity_tracks`重验binding哈希、同run/generation、目标/反馈/原确认图、slot/layout和已完成槽互斥，当前帧必须晚于确认帧。`completed_candidate_scopes`仅为对应定位，先按合法assigned_slot.floor（来自缓存实测桌面）排除±6mm支撑平面带，沿用已有桌面过滤数值；不按槽位XY裁点、不用仿真真值。剩余点必须非空且全部有限实测点XY在唯一完成槽内、Z在桌面−15mm至桌面+100mm+15mm内；原始非有限点、高于支撑带的槽外/高度外点、空剩余或多槽歧义仍拒绝。归档raw/matching/support-removed点数、两组bounds及6mm规则；原候选点、SAM轮廓、当前RGB仍供Decisions，最终几何仍走原机器人/轮廓/密度/桌面过滤。100mm来自用户尺寸，15mm沿用高度补全测量容差，仅是定位范围，不是碰撞余量或身份判据。`verify_completed_correspondence`发送独立当前请求：三张完整当前RGB，只标完成候选及对应槽，每个原C编号用短yes/no/unknown题询问指定相机里是否有该槽的独立粉件。只有yes且confidence≥0.8才结合合法track及唯一当前测量scope映射canonical ID，原回答与映射依据分开归档，不发初始外观参考或UUID选择题。



随后第二份Decisions保留剩余源物和盒子的原关联方式、原C索引及稳定R标签；已完成ID不再出初始外观、visibility或progress题。历史完整RGB明确为历史，遗漏题中的已完成源件使用第一份同帧可信结果的当前位置，不使用旧源位置或把unknown写成confirmed。`merge_completed_correspondence`重验同observation/generation、第一hash在第二请求中的绑定及双hash，再要求完整required集合、非空当前几何、剩余源物camera2覆盖和none≥0.8全部成立。像素冲突仍按统一几何所有权规则处理，已完成件不退出障碍集合；原完成证据不被新未知覆盖，旧点云不标成当前。历史无track保留旧规则，坏track直接停止。正常调用预算为首次一次Decisions、件间两次、放置后一次；前置验证或服务失败可以提前停止，均不重试、不增加capture/SAM/建图。



抓取 lift 后只检查稳定、非空的夹爪状态。放置后预算内采一次三相机图片，直接由 Decisions 确认当前区域内有粉件且与开爪分离；结合已校验释放链确认区域级释放，不声称RGB证明物理接触。不调用SAM3、对象几何估计或建图。继续搬运时保留已有地图，仅按抓取/释放 TCP 反馈刚体变换已搬物体的碰撞几何；这是带来源标记的运动预测，不是新的视觉测量。双臂使用保留地图回观测姿态，到位后才更新 RGBD 场景。



放置后的永久ID来自经过校验的抓取→held反馈→attachment→带物place→release执行链，不用初始外观重新辨认相同工件。`build_release_binding`重验target/arm/generation/owner/scene token/源观测、成功开爪、实际带attachment释放碰撞检查、冻结layout/slot及已完成槽互斥，模型返回后再次检查。三张完整RGB保留；只有当前槽投影与画幅有正面积交集的相机才加CURRENT青色注释及4题，只画当前槽。前三项workpiece/in_region/separated必须在同一相机全yes且各confidence≥0.8，不跨相机拼接。on_table只诊断：unknown/低置信不冒充支撑证据，也不额外阻断；任一相机可信确认有粉件且在CURRENT，而分离/桌面题可信no则作为强反证保持unknown。模型只接收短题、图像及两个审计标记observation_id/binding_id，完整执行链、实体ID、反馈和世界几何仍归档。



结果明确记录`primary_required_checks`、`support_judgment`和`completion_semantics="released_in_region_and_separate"`。这是遵循用户区域级放置要求，纠正实现者额外添加的独立RGB接触证明门；不是降低0.8或把支撑unknown改成yes，不声称物理接触已被视觉确认。主三项未知仍停止；动作记录不能代替这些视觉判断。全部最后同次紧凑初始参考与可见遗漏题仍须none≥0.8；count维持指定数量。`post_place_capture/post_place_result`独立保存，不覆盖旧测量观测；unknown可标已检查但不计完成。下一次观测位姿仍需SAM3和当前RGBD，已完成件按上述track关联，其他身份和建图规则不变。身份选项独立、R标签仅标一次，重复呈现不累加图像或提示。



新运行在物理释放核验后记录同进程`verified_perf_counter_ns`，采图在实际冻结MuJoCo帧时记录`captured_perf_counter_ns`，必须后者严格较大且仿真时间不倒退；避免相同仿真tick/低分辨率时钟造成误拒，无sleep或追加采图。历史归档没有新字段时只使用已记录的严格递增仿真时间证明先后，不能补造时间戳。



普通采图预算为 8（最多四件，每件一次观测位姿、一次放置后），不含 UI 预览；无隐式恢复采图。程序最多 64 节点、4 层、128 次调用，不允许嵌套目标循环。generation、scene token、图像版本及契约逐步检查。unknown、超时、物理失败分别记录。



## 14 个注册入口



| 技能 | 前提与用途 | 固定行为与失败边界 |

|---|---|---|

| observe_scene | 双臂在观测位姿且空手 | 三 RGBD；首次 Responses，后续 Decisions/SAM3；合法完成track仅限定当前槽候选题，仍须≥0.8；全部碰撞实体保留、左腕覆盖未完成源对象；同帧可信可见遗漏检查 |

| bind_inventory | 首次清单已生成 | 校验来源、数量、唯一 ID 和顺序；两种模式重验同帧可信可见遗漏检查，全部还须必需实体身份union及源对象左腕覆盖，独立记录source_inventory_resolution；冻结搬运清单 |

| move_to_observe | 已核验释放、保留场景、空手 | 同步规划双臂返回；不以放置图覆盖地图，无有效轨迹则停止 |

| identify_target | 清单冻结、目标循环绑定 | 选择当前固定 ID，不跳过或重新编号 |

| identify_destination | 目标已绑定 | 使用固定数值正方形区域与实测桌面高度，无物理标记 |

| choose_arm | 源和目的地有效 | 双臂工作空间评估，优先源对象所在侧；不可达停止 |

| build_scene | 观测位姿、当前观测、目标与臂已绑定 | 清单实体 2cm 地图，缓存桌面 cuboid，仅排除当前目标供抓取接触 |

| generate_grasps | 当前实测目标点云及可选的显式高度补全 | 有合法补全时提交完整几何，原测量仍确定frame/参考点；GraspGen候选按臂转换，沿工具Z深入1cm，只保留向下偏角≤45°且不朝机器人候选 |

| plan_grasp | 候选、当前源盒关联及地图占据边界有效 | 零合格源IK或完整确定的整池三阶段规划失败时最多追加两轮1000候选，共用原规划时限；源盒/邻件顶部+5mm推导world-Z lift；未知、最终FK或命令/硬检查失败停止 |

| execute_grasp | 当前有效轨迹 | 接近、抓取、闭爪、lift；夹爪位置伺服受自身URDF effort限力；每段实测起点与计划偏差超过1°则停止并记录关节证据 |

| verify_grasp | 已 lift | 仅夹爪状态：稳定非空为 held，空闭合为 empty，其他 unknown |

| plan_place | held、冻结区域、有效地图 | 必须 attachment，保持抓取旋转；持物/夹爪几何解析Z含0.1mm已记录规划余量，实际碰撞门仍为5mm，单目标 plan_place |

| execute_place | 放置轨迹和持物绑定有效 | 实测位置须在区域内并通过碰撞检查，再开爪、detach、撤离；不强加 2mm 到位要求 |

| verify_place | 已释放并撤离 | 校验执行链绑定；同view粉件/当前区域/开爪分离三项yes≥0.8确认区域级释放；on_table诊断，相关可信no阻止；无SAM/外观重认，全部最后none≥0.8 |



## 几何与回放



持物sphere_budget合法范围1–256；普通PlanningSettings默认16，demo仅无保存设置时默认256，显式保存的128及129–256均保持。取消旧load_settings把129–256静默降16的迁移。原PCA方向中点分组、1cm最小分组停止尺度、全部实测/显式补全点、测量不确定度和margin不变；实际球数可在预算前停止。离箱包络与实际attachment均使用同一配置预算。当前有效持物仅对own hand固定9链接组件允许接触；TCP相机支架、上游/对侧与所有世界5mm检查保持（范围见下文）。



当前显式初始布局为：1–3件按共线斜排的前N个坐标生成，近XY=[.24,.50]、中[.20,.57]、远[.16,.64]m，相邻中心距约.0806226m。配置以盒中心为参考，起点row_start_offset_xy_m=[.04,-.10]、步进row_step_xy_m=[-.04,.07]；没有仅适用三件的隐藏分支，工件yaw不变。4件仍采用X=0.13/0.25m、Y=0.49/0.61m的2×2布局。盒桌尺寸与位置、固定观测角度不变。固定目标正方形中心为[0.36,0]m，边长仍0.32m，区域X=[0.20,0.52]、Y=[−0.16,0.16]；只提供操作意图，支撑高度仍由RGBD测量。源布局仅供仿真场景工厂使用，不传入运行时识别或建图。



源IK成功项必须提供同scene/臂/tool frame/关节配置绑定的官方预测FK。保留原GraspGen名义工具姿态（已深入一次1cm）和原IK误差；仅当预测FK仍在当前名义位置/旋转容差（默认5mm/.05rad）、工具Z向下45度内且不朝机器人时，将该FK统一用于lift推导、plan_grasp_batch、contact_goal_pose_world与plan.T_world_tcp，不再次深入。最终抓取phase末端官方FK仍相对原名义姿态复核同一容差与方向，防止两次容差叠加；失败不触发轨迹补采。缺失或过期FK证据停止。source_ik_fk是预测而非实测，attachment和固定放置旋转继续依据实际反馈。归档保留nominal_T_world_tcp、source IK q及原误差、预测FK、final FK/误差和round/source索引。raw success_count与eligible_count分开，candidate_filters保存明确筛除原因；全部合格项为空沿用既有三池/180秒预算，耗尽文案为无可用源目标，不伪称原IK全无解。



源IK的官方FK通过原nominal容差和方向门后，Demo另做静态刚体宽度质量预测：使用当前完整实测/显式补全几何（不使用GG专用密采样），从自有URDF/Link3碰撞STL的最大相向平面及MuJoCo FK/mimic取得夹持轴与jaw gap。先按不超过0.01rad网格验证平行、单调闭合，再在唯一宽度区间内最多24次二分，求pad间距等于原云宽度的q_width（残差不超过1e-8m）。它是假设刚体可居中的宽度预测，不是实测闭合角、云居中或连续闭爪扫掠证明。碰撞计算不平移、不裁剪原点，在单一q_width检查全部点加原uncertainty_m+attachment.margin_m对固定9链接接触组件外夹爪球（含TCP和未知新增链接）的净距；原球collision/self buffers保留，不重复加世界5mm，也不用拟合attachment球做此筛选。负净距明确剔除；无合法整云宽度区间仅标记宽度估计不可用；固定TCP非接触球仍检查完整原云，不能据整云宽度否决局部抓取，坏点/来源/模型停止，不能默认通过。



candidate_filters每个原index仍仅一条，嵌入predicted_gripper_compatibility，保存预测来源、arm/tool、完整点数/hash、测量数与补全来源、原padding、width/轴、q/区间/残差、最小gap及point/link/sphere索引、接触豁免与模型指纹。raw IK成功数保持，只有eligible减少；筛后为空仍沿用既有三池/单180秒预算。最终抓取phase官方FK也重新检查这一预测，失败仍是planning_grasp_endpoint_rejected，不授权补池。原姿态/分数/顺序/索引/round及一次+1cm不变，普通Algorithms不启用此门；实际持物反馈、attachment球拟合及所有世界/自身硬检查保持。own-model资源在服务进程内只读，URDF/YAML/pad-STL哈希标识缓存；资产变更须随robot_model一同重启加载，不支持热替换。



持物接触边界由carrying_touch_links(arm)统一定义：仅该臂自有夹持组件的固定9个链接，A为gripL_Base及gripL_Link1/2/3/4各L/R，B精确镜像gripR_。不按前缀自动豁免；TCP_Link含相机/支架，必须继续检查，所有上游臂、对侧手/臂和未知新增链接也保留。只对当前有效carrying binding添加attachment与这9链接的双向接触对，不删除任何机器人/物体球、不改变普通机器人self pairs或世界碰撞。预测q_width使用同一接触集合，仍核对宽度/模型合法性、原点包络与TCP等非接触球；source及final FK容差/方向门不变。



这是有意允许夹持组件接触的语义，不是把Link4称为接触垫，也不是证明物体与活动手指永不接触。初观测云在闭合前后变换不能当作实际闭合后的完整物体位姿；原5mm uncertainty+1mm margin仍完整用于世界/组件外包络，不因理想RGBD或本次结果缩小。释放张爪沿用同一有意接触边界；既有当前闭合持物与全开detach两端状态检查、上游/对侧检查、实际夹爪反馈与确认张开后detach均保持，不宣称检查了物体对活动手指的无碰撞扫掠。有效持物、原scene/owner/arm绑定、固定XY/R和5mm世界门仍是前提。



初次 `generate_grasps` 保持1000候选。`plan_grasp` 在同一当前源IK及预测FK复核后合格项明确为零，或该池全部合格项均得到明确三阶段规划失败时，最多追加两轮各1000项，总请求不超过3000；追加生成、全部IK和轨迹规划共用原 `plan_timeout_s`（默认180秒），不重置每轮计时，首轮生成技能仍独立于此规划时限。冻结RGBD、原测量frame、补全来源、模型、目标、臂、地图token、槽位与设置保持一致，实际场景/物体请求摘要再次核对。没有新增视觉调用、采图或地图。45度、背离机器人、深入1cm、槽位和5mm检查不变；过滤为空、unknown、服务错误、过期、超时、取消以及最终FK拒绝、命令/硬碰撞校验失败均停止，不再采样。规划补池必须有 `pool_exhaustion`：同轮/目标/臂/generation/scene/settings/tool版本及冻结关节匹配，全部候选索引唯一完整覆盖、未规划数为零；每项为SDK正常 `Grasp planning completed.` 状态、无error且phase布尔值明确失败并符合阶段顺序。单独的 `planning_no_valid_candidate` 错误名不够。三轮源IK均空仍返回 `no_collision_free_source_ik`；曾尝试轨迹而预算耗尽则保留 `planning_no_valid_candidate`，逐轮结果和真实轨迹尝试数归档到 `sampling_evidence`。每次远端调用前后检查取消，在途请求可完成但之后不再启动新的调用。



粉色物体为用户给定的 6×4×10cm 椭圆柱工件。GraspGen 和 attachment 使用合法的实测加显式高度补全几何，同时保留原始测量；没有补全时使用测量点云。GraspGen在完整原几何之后另加仅供推理的侧表面采样：在实测XY凸包周界及已有补全高度内按不超过2mm间距取点，总输入最多12000点，超限停止，不删原点或自动降低密度。这不是2cm地图体素，也不改变原补全、attachment或建图。`graspgen_input_sampling`记录构造来源、间距、原几何点数、新增点数和实际总数；实际输入哈希/合成点数与请求一致，原`completion_provenance`仍说明原补全而非新增测量。遮挡高度补全必须带用户尺寸先验、测量支撑面和测量边界来源，不能伪称全部表面已看见。GraspGen的原测量质心frame、身份及放置参考点不变，实际输入哈希、点数、补全来源和原测量分别归档。球拟合空间分组不是 TSDF 体素；不可因粗球与夹爪相交就全局关闭自碰撞。



目标区域位置和面积在配置中固定，可以离线校准为更靠近机器人；执行过程中不为某个候选随意移动。名义放置可达性筛选在有合法补全时使用完整几何最低Z，避免把仅可见顶面误作物体底面；原测量最小Z、质心和来源仍保留。最终Z仍由attachment和5mm碰撞检查确定。保持抓取旋转，放置参考点和物体包络落在区域内即可。单目标规划内部 IK 不等于批量放置候选筛选。5mm 碰撞阈值同时用于最终路径和释放时的实际持物检查。



离箱高度使用建图时全部原生占据格的世界边界：先归约完整体素中心并扩半格，再抽样显示预览。`surface_cell_bounds_world_m/source/count` 与 scene token、观测和generation一起归档；预览点不能作为安全边界。释放后移动的层原边界失效，碰撞几何仍保留；切换目标只更新仍有效边界的版本与启用状态。放置解析高度对attachment和夹爪闭合和全开两端包络统一增加0.1mm规划余量，记录`planning_reserve_m=0.0001`及实测负载伺服误差来源；这是5.1mm名义起点，不是改变实际5mm阈值。后续1mm有界世界高度查询及开爪前实际检查保持原规则。



每轮抓取规划快照附带已核验搬运完成的 `completed_target_ids`，不改写原观测。源盒离箱高度计算可据此排除已放到目的地的旧源对象；这些对象继续保留在碰撞地图中，不因完成搬运而删除。



释放后的 completion cuboid 保留 `position_source` 与 `settling_envelope_floor_z_m` 溯源。`sdk_scene.sdk_scene_model` 在单/双臂求解器构建及原生地图刷新边界隔离这两个项目字段，向 cuRobo 传递实际更新后的 pose/dims、落至支撑面的包络及其他合法几何字段；原模型、审计记录和原生体素保持不变。未识别的非法几何参数仍由 SDK 拒绝，不能通过丢弃障碍物或沿用源位置来消除解析错误。



名义放置可达性回放的 `nominal_height_source` 与实际输入一致：使用显式高度补全最低点时标为 `height_completed_minimum_and_bound_floor`，仅测量时保留测量来源。2026-10-21 修正此来源标签，计算和候选筛选行为无变化。



实验记录保存每次 Responses/Decisions 的真实请求图片、SAM3 各描述的掩码和点云、GraspGen 原候选、IK 筛选和轨迹筛选结果。执行图按实际调用及物体阶段关联，不把保存了证据等同于运动通过。



2026-10-21 回放界面核对：候选记录提供 `sampling_round` 时显示明确轮次，并说明原始候选索引仅在该轮内唯一；历史未记录轮次不猜测。仅增加证据定位信息，不改变筛选、运动或成功判据。



所有运行验收以实际实验终态为准。本手册和离线测试不证明一次真实搬运已经成功。



- 2026-10-21 接触组测试同步：原 Link3-only 策略的4个测试实例已改为当前同侧九链接附件配对，并保留所有 Link4 球、世界覆盖、普通机器人自碰撞、TCP、上游与对侧检查。两个完整测试文件48项通过，Ruff通过；仅旧测试预期同步，skill行为无新增变化。



- 2026-10-21 部署核对：源码同步清单纳入根目录资产来源说明与 Docker/Git 配置文件，并逐文件比较 Windows、Ubuntu 的 SHA-256；skill 行为无变化。



- 2026-10-20 核对：运行包和入口统一为 `luna_mujoco`；保留旧镜像的容器内挂载路径兼容，不再注册或调用 JEV。初次桌面 RANSAC 后融合三相机同平面、局部水平且连通的测量点；补全 attachment 使用实测 XY 凸包沿高度外推，避免包围盒四角过度膨胀。



- 2026-10-20 位置伺服核对：冻结释放姿态中，机器人自重造成 A/B 参考点分别下垂 1.752051/1.752495 mm，使名义 5 mm 间距只剩约 3.2475 mm。`model_builder.py` 对机器人自身 body 启用 MuJoCo 原生 `gravcomp`，经 `actuatorgravcomp` 计入关节总驱动力，并用原 URDF effort 设置关节限幅；位置伺服输入仍为目标关节角，没有外部力矩控制、额外 qpos 写入或物体重力补偿。原生力限幅语义见 [MuJoCo actuatorgravcomp 文档](https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-joint-actuatorgravcomp)。修改后同姿态空载稳态漂移为零；80 g 等效点载荷诊断仍有约 37 µm 向下静差，不能代替真实夹持接触验证。证据为 `data/luna-position-gravity-before.json` 和 `data/luna-position-gravity-after.json`。skill 接口、位置控制模式、区域接受规则及实际 5 mm 碰撞门均无变化；未增加 2 mm 精度条件，也不以控制器回归测试阈值代替技能验收。

- 2026-10-20 核对：清单内、已确认身份的点还需通过密度检查（2cm 内至少4点，含自身），同步删掉对应深度有效掩码中的孤立飞点；去掉非有限值、机器人表面及桌面点。该检查不把稠密但身份不明的片段认作物体，不把删除的深度视为空闲空间。

- 2026-10-20 核对：回放增加按明确 target_id 分组的实际阶段时间线，归档模型输入图片、SAM3 提示/掩码及 GraspGen、IK、批量抓取规划候选的交互式 3D。只读 NPZ 预览报告原始形状和抽样索引；历史缺失保持未归档，不采集新帧或调用模型。编辑器保留椭圆截面与独立 X/Y 尺寸。详见 [回放说明](demo-replay.md)；skill 运动与完成判据无变化。验收脚本支持保存录像，统一使用“目标区域”指令。

- 2026-10-20 联调：每张 Responses 图片明确标注其自身宽高，避免头部360px与腕部480px的归一化尺度混用。全部清单实体已有可信视角、全部剩余粉色对象已有左腕可信分割，且 Decisions 明确确认可见粉色对象均已列入清单时，允许丢弃重复但身份不确定的左腕提议；不降低0.8身份阈值、不把不确定点加入地图。

- 2026-10-20 联调：Decisions 同时接收各提议实测 RGBD 点云的世界质心与边界，明确 +Y 为机器人左侧，辅助区分左右对称同色盒；这些数值仅是定位证据，身份仍由 Decisions 确认。粉色目标沿用已确认清单的颜色语义，简单像素色彩分类另存 measured_color_class。审计脚本显式加载当前 src；规划事件测试按事件类型查找，避免新回放事件改变数组位置；skill 运动行为无变化。

- 2026-10-20 核对：旧 Compose 入口统一转入独立 demo 配置，Dockerfile 使用 demo_skills 和当前启动脚本；图像标注补充粉色/桌面名称，并明确已缓存桌面 cuboid，不再误标为缺少几何。已删除 JEV 调度对应测试移到证据目录，保留共享夹具并新增取消、晚到结果和 API 互斥测试，见 [测试迁移说明](test-migration-2026-10-20.md)。显示、部署和测试维护不改变 skill 运动判据。

- 2026-10-20 录像核对：录像 schema 2 保存 support_tables，修复双支撑桌导致离线 MuJoCo 几何 ID 偏移。旧归档仅从同实验同 generation 的冻结 simulation_initial 恢复缺失支撑桌，继续校验 ID/name、类型及尺寸。离线 experiment_reencode 命令复用真实编码器，保存原错误与修复审计，只更新视频归档状态，保留原帧、事件和任务 status；skill 运动、置信度和完成条件无变化。运行命令见 [回放说明](demo-replay.md)。



- 2026-10-20 attachment 拟合核对：冻结实验 ef356390f2a04c2b9d025532a4369ec5 中，固定 TCP 轴切分的球与 Link4 假重叠 0.418 mm；原始测量/补全点保留 6 mm 不确定度与 margin 后仍有 3.252 mm 间隔。当时按各点组局部主成分方向中点切分，保持 128 球预算、2119 个测量点、888 个显式补全点和全部余量，冻结夹爪模型最小间隔超过 1.2 mm。分组与中心选择中的皮米级舍入仅稳定数值并列，最终半径仍按原始点计算。Link4 等非接触链接继续检查，5 mm 世界阈值、目标区域、抓取旋转和单目标放置流程均不变。回归夹具见 tests/fixtures/luna_first_carried_surface.npz；离线覆盖与 IK 证据不代表真实轨迹已经通过。

- 2026-10-20 源场景联调：三物体实验 908fd4671f2b4832a0b32624c69a86e3 的最外侧工件产生253个超出肩腕解析界的候选，另2个可解IK但与箱壁/邻件包络碰撞。保留全部用户尺寸、对称角度、Y贴边与目标区域，配置两侧小桌/盒X从0.29改为0.20m；盒内工件行Y固定从0.46m开始、间距0.06m，默认三件，最多四件。1–4件真实RGBD检查均可由左腕看见，未见初始机器人场景接触；只改变仿真初始布置，不向运行时感知提供物体真值，不改变skill验收或5mm碰撞阈值。原故障与新布局证据位于data/luna-three-source-diagnosis.md及data/layout-fov-diagnosis-20261020/。

- 2026-10-20 观测验证元数据核对：固定关节角与原始机器人镜像 FK 证据保持不变，旧 X=0.29 布局资源单独归档；当前资源记录 X=0.20 场景配置、当前包路径指纹及重新渲染的三件 1843/1506/1406 像素和零机器人场景接触。完整蓝盒包络的 −50.04px 出画量只作诊断，明确不是额外验收门；蓝盒可辨识及全部粉色对象可见由真实左腕图检查。当前默认图位于 data/demo-observation-poses/current/，详见[场景与校准证据说明](demo-scene.md)。此次只修正离线验证来源、脚本元数据及回归测试，skill 行为、固定姿态、5mm运行碰撞阈值和正在运行的任务均无变化。



- 2026-10-20 GraspGen 补全输入核对：冻结三物体实验 c6b5e999ebdd41b8ac85a54b5cb1516e 的1242个顶面测量点生成1000候选却无方向合格项；保持测量质心frame和原场景输入，改用已有显式280点高度补全后的1522点，两次真实100候选推理分别30/28项通过45度及背离机器人规则。名义放置最低Z同样使用校验后的完整几何，冻结候选腕距由错误0.622606m变为0.583507m；实际释放仍由attachment与5mm检查决定。84项聚焦测试及Ruff通过，包含A/B、来源拒绝、实际输入哈希、错误回放和真实顶面夹具；没有修改perception算法、SDK或执行运动。

- 2026-10-20 回放输入核对：默认显示真实 GraspGen object_points_camera_m，经原 T_world_cloud 转为世界坐标；height_completed 明确标注测量加高度先验构造，原始测量与完整补全集可独立查看。scene_points_camera_m 保留实际 RGBD 场景请求及其来源，不套用物体补全标签。可展开来源和测量/构造点数；历史缺失保持未归档，不用其他点云冒充实际请求。前端283项、生产构建及实际WebGL/390px窄屏验证通过；展示改动不改变skill运动判据。

- 2026-10-20 参考视角核对：冻结实验3c5879左腕源物可信，但右盒没有达到0.8的身份关联；高可见性或框重叠不代替身份。Responses提示在图中可辨时优先camera2作为源盒/粉件参考、camera1作为右盒参考，避免用头部小块截断图做唯一身份参考；头部辅助大桌和跨视角。相机角色不能创造实体或替代RGB证据，仍单次Responses、固定ID及完整可信几何覆盖。60项聚焦测试通过；提示变化的实际识别效果须以后续模型实验为准。

- 2026-10-21 身份证据呈现：demo Decisions 使用短且稳定的 R 参考标签、C 提议标签和永久 ID 图例，统一中性框色，逐题附上已有同相机框重叠定位提示。所有完整 RGB 与 canonical 选择值保留；仅在 camera2 同观测 ID、验证相机和相同图片内容均成立时增加包含全部源物邻居的成对放大图，归档原图哈希、裁剪像素坐标、缩放及实际发送图片。后续新观测或未经验证的图片不生成该放大对照。框重叠、短标签和放大均不建立身份，不合并提议、不改变 0.8 门槛；未知仍停止。119 项相关测试及 Ruff 通过。冻结5f8的唯一一次真实对照中，三个粉件身份为0.97/1.00/0.99，左右盒已有可信腕部身份，完整清单及左腕源物覆盖成立；9个头部未知仍丢弃，右盒独立可见性unseen/0.38按原规则保留unknown。全部答案和离线解析核对见data/identity-requested3-diagnostic/report.md；未重试、未运动，不代表后续搬运验收。

- 2026-10-20 闭爪限力核对：实验9975165c6b7d4f97bb6a5fc914b314c0的grasp终点与lift起点仅差0.00000683°，闭爪前已到位；原硬编码200 Nm夹爪上限却允许约44 Nm驱动输出，夹持目标时使Joint7_L偏移约5.97°。冻结场景只发现两侧Link3接触目标，未发现夹爪与箱壁或邻件接触，arm/gripper actuator映射及机械臂ctrl也无误。现从自身URDF读取左右夹爪8.066 Nm effort，同时限制position actuator和包含原生重力补偿的关节总驱动力，位置增益、机械臂目标及力限不变。同场景闭爪后Joint7_L偏差约0.0125°、全臂最大约0.0698°。现有1°轨迹起点检查及5mm碰撞阈值保留；失败新增实际关节、计划起点、最大偏差、关节名和反馈版本证据。冻结证据为data/luna-three-fourth-start-drift.json、data/luna-three-fourth-gripper-contact-probe.json；回归夹具为tests/fixtures/luna_closure_load.json。离线闭爪通过不代表后续真实lift或完整搬运已经成功。

- 2026-10-20 限力修正后观测资源复核：完整测试发现的唯一资源指纹失配来自 model_builder.py 采用 URDF 夹爪 effort 的实现变更，URDF 文件本身未变。已在当前模型重新执行128种符号的镜像 FK、观测端点碰撞检查和三相机 RGBD 渲染，更新观测资源指纹；三组关节数组与场景配置逐项核对未变。左腕三件仍为1843/1506/1406有效像素、机器人场景接触为零，完整盒包络仍仅诊断。原资源和新审计均保留于 data/demo-observation-poses/；未放宽指纹测试、未改变任何skill或运行判据，也未操作正在运行的服务。



- 2026-10-21 离箱与释放核对：源盒及剩余邻件完整占据边界、当时同一128球持物包络推导统一world-Z lift，原10cm邻件冲突案例得到0.237595m有效高度；已完成或已出盒对象只退出高度参考，仍是碰撞障碍。完整体素边界先于20,000点预览抽样计算，20,001点丢失极值回归、旧边界失效与选择版本测试通过。冻结997实际位置伺服抓取/lift为held，单目标place原来因实际间距4.977790mm而被门拒绝；0.1mm显式规划余量后实测5.078271mm、实际GPU碰撞校验通过，再开爪、detach及撤离完成。余量源自22.21µm间距短缺、30.48µm位置静差和既有37µm负载诊断，实际5mm阈值、固定旋转、区域接受及控制器均不变。140项聚焦测试通过；冻结证据位于data/luna-three-fourth-reserve-execution/，不是新在线三物体任务成功，也未调用app运动接口。

- 2026-10-21 区域接受回归核对：完整测试发现一处旧解析Z预期0.825m未包含已批准的0.1mm规划余量，现更新为0.8251m，并同时断言planning_reserve_m=0.0001及required_clearance_m=0.005。原15mm位置偏差、5度旋转偏差在区域和几何检查通过时接受，移出区域仍拒绝；该文件3项测试与Ruff通过。只修正测试预期及本核对说明，生产代码和skill行为无变化，未调用后端。

- 2026-10-21 可见清单语义核对：Responses的来源完整性现明确指三路RGB中全部可见粉色源对象已逐个枚举、可在camera2区分；没有未解释可见对象时，仅裸露箱底/箱沿被遮挡不触发unknown，不声称隐藏处为空。任一可见对象身份、数量、归属未知仍保留unknown；supported只表达路线、数量及目标选择是否能由技能执行，视觉不完整返回supported=true及unknown并接受原证据门检查。统一_require_visible_inventory使all/count在首次、件间与绑定时均要求同帧openai_decisions的none、置信度至少0.8；all初始confirmed与最终门、count原unknown例外不变。新增首次/件间16种失败组合、绑定重验、最终unknown及提示边界回归；106项聚焦测试与Ruff通过。未调用模型或后端，未改变Decisions呈现或阈值；真实模型与三物体流程仍须另行验收。

- 2026-10-21 可见遗漏题呈现核对：全局问题附上原来源顺序的永久ID、既有R标签、display ID、描述及初始框，明确跨三张完整当前RGB按物理对象去重、检查放大图外部；相似颜色、邻近或参考数量不建立身份或证明无遗漏，可疑独立可见对象仍须unknown。66项相关测试与Ruff通过，低置信度none仍unknown。冻结f0fe唯一一次真实Decisions对照保持14张图、上下文及其他18题完全相同，仅增加此题说明；none置信度由0.70变为1.00，其他18个完整答案逐项不变。三粉件左腕身份1.00/1.00/0.81、源盒0.97/0.93、右盒0.97，所需五ID覆盖完整；7个头部低置信度提议仍不采纳。请求、全部答案与单次调用锁见data/visible-inventory-f0fe-diagnostic/report.md。未重试、未运动，未修改原实验终态；这次冻结对照不代表新观测或搬运已成功。



- 2026-10-21 有界候选采样核对：新增单一规划deadline下的零源IK分支，最多三轮共3000请求；不重试错误、unknown、过期或轨迹失败。135项相关测试通过，覆盖共用时限、取消在途GG后不再发下一批、冻结输入变化、预算耗尽和三阶段轮次/原索引。冻结a979原1000项→267源IK全零；只追加一轮同云同frame同地图的1000项，与原池六位小数姿态重合0，265方向合格、253槽位合格、7源IK成功，plan_grasp_batch有2个完整可行项，既有累计关节旅行量规则选第2轮原index702，命令验证通过。未调用第3轮、未执行运动，证据在data/a979-bounded-sampling/，不代表在线三物体搬运成功。

- 2026-10-21 回放轮次显示核对：前端显示sampling_round与原候选index，保留所有先前轮次证据；283项前端测试和生产构建通过，实际Edge WebGL检查轮次2/index17、筛选和390px无溢出，截图data/replay-sampling-round-qa/mobile-cloud.png。仅显示变化，不改规划过滤。

- 2026-10-21 在线回放核对：实际9bc1归档页面在Edge无脚本错误、390px无水平溢出，报告data/live-replay-qa/。纠正按物体阶段中的语义标签：choose_arm为执行臂选择，build_scene为场景建图，目标与目的区域合并，返回观测独立；plan_grasp标为IK / 抓取规划，保留实际调用顺序与原始事件。只读检查脚本拒绝所有非GET API请求；显示与审计脚本变化不影响skill行为。

- 2026-10-21 执行图可读性核对：按归档task_progress的display_id显示“物体 #N”，保留永久ID，不按回放顺序重新编号；三类候选耗尽错误显示对应筛选或IK阶段，不推测未记录原因。均为只读显示变化，skill行为无变化。

- 2026-10-21 GraspGen采样回放核对：新增独立来源展开项，显示原补全集、额外先验表面点、实际请求点数和点间距，避免把GG专用采样点数与地图/attachment补全集混用；原始来源JSON可展开。实际9bc1归档的17张模型输入、SAM3掩码、真实1516点GG请求与1000候选已通过Edge只读显示检查。显示变化不改变skill行为。

- 2026-10-21 后两阶段点云回放核对：after_ik及after_plan_grasp_batch现携带同一采样轮的真实GraspGen物体/场景请求数组、T_world_cloud、观测ID、输入来源/专用采样元数据及原补全几何，分别保留原测量；通过该轮IK评估传递，不按事件邻近或当前点云重建。历史未记录的实际输入继续缺失。93项Python相关、9项前端证据测试与Ruff通过，涵盖第一/第三轮、测量/补全、原索引、冻结副本与各阶段独立NPZ读取。没有新模型、后端规划或运动调用；skill行为、服务请求和验收阈值无变化。



- 2026-10-21 GG输入密度核对：9bc1并未缺少补全，1243测量+273稀疏侧环共1516点进入官方20近邻/L1均距14mm过滤后只剩1284点，高度从100.0005mm变16.6667mm；a979/c6b5则因凸包39/40顶点差异零星保留较低环，高度83.3338mm。现仅对GG对象输入追加既有凸包侧面≤2mm间距采样，总点数≤12000；不修改SDK/原测量/原补全/地图/attachment/frame。实际官方CPU函数验证三组修复输入5416/5422/5422点全部保留完整100mm高度。126项生成/规划/离箱/取消相关测试及Ruff通过。唯一一次9bc1真实1000候选得到320方向合格、301槽位合格，但4批源IK仍全零；没有追加池或规划/运动，失败完整保留于data/9bc1-surface-sampling/，不将输入修复视为抓取成功。诊断与官方源文件摘要见data/9bc1-diagnostic/diagnosis.md。



- 2026-10-21 来源完整性证据递进核对：453c 的 Responses 已逐个列出左腕三件粉色对象，但因头部边缘片段无法匹配而保留 unknown，旧前置门使 SAM3/Decisions 未执行。现保留原判断并继续既有同帧流程；all 在 bind_inventory 对原 confirmed/unknown 均重验全部必需实体可信身份union、源对象左腕身份与几何覆盖、同帧 none≥0.8，以及初始参考/Responses/Decisions观测和请求关联，单独归档 source_inventory_resolution。这替代本页早先核对记录中的 all 初始 confirmed 前置门，不改 count、固定 ID/槽位、辅助 unknown 丢弃策略和最终放置/遗漏检查。202项感知、绑定、程序及身份回归通过；覆盖32种完整性证据缺失/过期组合、原 unknown 保留和一次Responses后同帧SAM/Decisions可达。未调用真实模型、规划或运动后端，未修改远端；本地替身测试不等于在线搬运通过。

- 2026-10-21 放置结果边界核对：30dd物理held/attachment/place/open/retreat完成，原13个SAM身份答案全低于0.8。修复共享choice字典/R标签累加、传入实际release上下文的首次冻结对照仍无可信目标身份。随后按批准设计改为执行链绑定ID的专用结果题，不做SAM/外观重认/新地图；重验scene/owner/generation/slot/反馈和实际采图先后，unknown只标已检查不计完成。114项相关测试及Ruff通过；根代理全量1786 passed、54 skipped。新独立授权的一次专用冻结对照复用原三RGB和已记录执行链，使用事后重建且明确标注的binding；返回not_released_in_assigned_slot、confidence0.51，生产离线解析保持unknown，未重试或改写原终态，不能声称已解决模型可靠性。全部请求、图片、完整安全回答与推断边界见data/30dd-postplace-diagnostic/report.md。历史缺纳秒时间戳不补造，30dd仅用原123.794>123.717秒证明先后。

- 2026-10-21 只读回放QA：30dd实际after_ik/after_plan阶段在Edge WebGL成功读取同轮实际GraspGen物体/场景输入、原测量和补全集，无脚本错误；所选成功batch分别20/100、8/12，并非全局总数。报告data/live-replay-qa/30dd72b14f9b4eac811be2520492025d/filtered/report.json。视野fit现同时包含点云和TCP线框边界，真实TCP mount在云顶上17.5cm仍可见，世界坐标及变换不变；288项前端测试与production build通过，截图data/live-replay-qa/30dd72b14f9b4eac811be2520492025d/filtered/127.0.0.1/stage-2.png。仅显示/只读QA变化，skill运动行为无变化，原run仍视觉失败。

- 2026-10-21 区域级核验语义修正：单图简题确认粉件/区域/分离1.00/.99/.99；后续唯一完整生产对照保留三原RGB，只在camera2/3加CURRENT注释和4短题，主三项分别1/1/1与.99/1/.86，on_table仍仅.73/.31。四图光照示例（明确非原30dd精确状态）也均低于.8，不修改光照。根代理纠正自己超出用户区域级要求的独立RGB接触证明门：主三项同view各≥.8结合合法执行链确认released_in_region_and_separate；on_table仍原样诊断，相关可信no阻止、未知不改yes也不额外阻断。原8题回答仅离线重解析，未再调用模型或改写旧run/ledger；不声称解决了支撑视觉识别。138项最终聚焦及Ruff通过；此前四项门版本全量1795 passed/54 skipped，独立审查无阻断，最终新全量由根代理核对。完整证据data/30dd-postplace-diagnostic/report.md及data/30dd-lighting-audit/decisions-comparison/report.md。下一步仍须fresh三件物理验收。

- 2026-10-21 初始布局与固定目标区核对：1–3件改为X=.20、Y=.48起间距.08m，4件由显式配置生成X=.13/.25、Y=.49/.61的2×2，尺寸、盒桌位置和指定观测关节完全不变。目标中心从X=.46移至.36，边长.32/面积.1024m²保持；原区域867个位置样本中10个被A工作空间筛除，新区域X=.20..52、Y=±.16、Z=.56..66每1cm的11979个样本均通过（data/destination-area-workspace-{audit,proposed}.json），这是有限位置筛选，不是固定姿态或轨迹保证。冻结诊断中原6cm邻距的905候选均不满足5mm几何检查，选定8cm布局有16项静态合格、精确CPU IK抽查7/8通过，也不等同整条运动通过。当前配置重新执行128镜像FK及1–4件三路RGBD/初始接触检查，每个数量全部粉件可辨且零机器人场景接触；四件近X排仍被前景夹爪部分遮挡。资源记录各数量证据及当前工厂/配置指纹，旧资源/图像归档到data/demo-observation-poses/before-layout-20261021/，本次审计在data/layout-implementation-20261021/audit.json。仅修改仿真初值、配置目标意图和离线验证，运行时仍从RGBD获取源实体/地图，不注入模拟真值，不改变skill接口、5mm门或运动验收，也未调用模型、远端或执行运动。



- 2026-10-21 释放后返回地图边界核对：真实963cb12d首件verify_place通过后，dual observation在Cuboid构造时拒绝position_source，尚未进入返回规划；同一模型还含settling_envelope_floor_z_m。现仅在公共SDK几何边界隔离这两个审计字段，原模型与归档来源保留，更新后的pose/dims/落地包络和原生体素仍参与碰撞。4个回归先复现异常，修复后56项场景更新、单/双臂构建及返回相关测试与Ruff通过。一次本demo容器CPU验证以实际curobo.scene.Scene.create复现旧异常并通过新构造/刷新输入，核对已安装SDK字段与源码SHA，证据data/963-sdk-scene-boundary/；没有服务建图、IK、规划或运动调用，也未改远端源码。skill行为、5mm门及释放溯源无变化；这次边界修复不证明fresh三件流程已成功。



- 2026-10-21 已完成件跟踪：真实88d首件完成并回观测位姿，剩余源物身份0.80/0.95、两盒及none=0.88已通过，但已完成件C6/C9只有0.36/0.16，required全集缺一而停止。新增合法完成track与唯一实测槽位的候选短题，保持全required/当前障碍几何/0.8门；155项相关测试及Ruff通过。冻结原17候选及完整点云复核仅C6/C9可scope，12张模型图哈希完全相同；真实binding/视觉/确认帧本地组装明确标注来源，不改旧ledger。经授权仅一次新Decisions对照：C6/C9正确ID为0.55/0.73，剩余中间件0.66、右件0.80，盒0.92/0.94，none=0.86；当时单请求生产解析仍unknown，未重试，不能宣称已解决感知可用性。完整证据在`data/88d-second-observation/`。

- 2026-10-21 源排列 +2cm Y 核对：1–3件由Y=.48/.56/.64移至.50/.58/.66，X=.20与8cm间距不变；四件2×2、桌盒、相机、目标区及两臂radians/degrees/startup数组均保持。冻结CPU预筛（data/30dd-physical-audit/row-plus-y-2cm-precheck.json）经方向复核后，旧布局近侧2项、新布局近侧68项和远侧178项同时满足5mm静态几何与腕界，只是预筛，不是新IK、轨迹或任务成功。重新执行128镜像FK和1–4件三路RGBD/初始接触审计：左腕像素分别为1613；1613/1406；1613/1406/1714；1831/1562/1340/1446，各数量零机器人场景接触、零粉件初始接触，四件三路PNG与前版字节一致。观测资源记录每种数量的实测仿真坐标、接触及新指纹；旧配置、资源和图像归档data/demo-observation-poses/before-row-plus-y2-20261021/，本次审计data/row-plus-y2-implementation-20261021/audit.json。44项场景/API/相机/仿真聚焦测试和Ruff通过；源真值仍只用于仿真QA，不进入运行时感知。未改光照、skill接口或运动门，未调用远端、模型、部署或机器人命令。



- 2026-10-21 件间完成件语义拆分：独立当前对应请求与剩余清单请求按上述同帧双hash合并。162项相关测试及Ruff通过；planning独立审查未发现阻断，并确认同相机已完成/源件像素冲突导致缺几何拒绝。88d纯离线重放原17候选和未腐蚀实测点，只有C6/C9可scope；新完成分支4图/2短题，剩余模板12图/22题。CPU合成yes仅验证合并管线；合成unknown仍缺已完成ID停止，不是模型成功证据。固定第一请求hash与依赖实际第一回答的第二请求模板、原材料hash在`data/88d-second-observation/independent-tracking-preparation/`。未发新模型、规划、运动或修改旧run；该离线阶段未进行真实对照；1832/54全套结果属于前一版，不能用于本次实现。



- 2026-10-21 经授权的88d双请求真实对照：各仅调用一次且HTTP 200。第一份4图/2题固定hash`89924605f925b70b7441a059cfaf600a4ec1b2b51a0e9acdc07f84b3457b2850`，C6/C9分别yes/confidence0.99/1.00；当前生产代码仅用第一份实际原回答生成第二份12图/22题，hash`ac6302c6c36796e6a467b7351e490d21c1666cc5fb9c78d16d120f8af77acf7e`。剩余两源件0.89/0.92、源盒最高0.84、右盒最高0.93、none0.96，完整五required ID、当前几何和剩余源件camera2覆盖合并通过。9个辅助低置信提议仍unknown并排除，未提升visibility/progress未知；完成件保留730个当前头部测量点。两份完整图片/hash/原回答/调用锁、离线当前生产解析和逐题报告在`data/88d-second-observation/independent-tracking-comparison/`。未重试、未新采图/SAM、未规划/运动、未改旧run终态或ledger；这是冻结帧验证，不代替部署后新三件流程验收。

- 2026-10-21 共线斜排实施核对：1–3件统一取近(.24,.50)、中(.20,.57)、远(.16,.64)m的前N点，以源盒中心为参考配置row_start_offset_xy_m=[.04,-.10]、row_step_xy_m=[-.04,.07]，不保留仅三件的隐藏分支。四件原网格、Z/尺寸/yaw、桌盒、目的区、用户关节和相机/阈值不变。同274候选的48球全开夹爪静态5mm/腕界对照中，斜排far/middle/near=96/101/14，前一直排0/0/18（data/7a2c-grasp-failure/layout-static-comparison.json），不代表IK或轨迹成功。复用既有校准脚本刷新1–4证据与配置/工厂指纹；左腕像素为1527；1527/1393；1527/1393/1650；1831/1562/1340/1446，零机器人场景/粉件初始接触。默认三件三路PNG与已审阅data/diagonal-row-audit逐字节一致，四件三路PNG与前版也一致。44项聚焦测试及Ruff通过；资源/输入/图像归档data/diagonal-row-implementation-20261021/，前版在data/demo-observation-poses/before-diagonal-row-20261021/。只改变仿真初始布局；skill行为、运行时感知来源和碰撞门不变，未部署、调用模型或机器人命令。



- 2026-10-21 FK目标交接核对：冻结bec同五候选原名义goal为0/5完整硬检查通过，改用已通过源IK的官方FK goal后4/5通过；未执行运动。实际末端相对原nominal仍在原5mm/.05rad内，未叠加容差，证据data/bec-grasp-failure/fk-goal-final-nominal-errors.json。项目backend追加官方预测FK，Demo才启用新目标及末端门；vendor/SDK、5mm碰撞、三池/单deadline及实际持物放置来源不变。Runtime补齐/重建与HTTP序列化、A/B、旧证据拒绝、方向边界、位置/旋转容差叠加、lift/request/contact/返回一致性已纳入聚焦测试。历史初稿spec banner仅标明被当前21条约束替代，不改变skill行为；fresh三件仍由根代理部署验收。

- 2026-10-21 FK-goal最终聚焦验证：`test_grasp_fk_goal`及Demo规划/采样、backend、GeometryRuntime、原batch、离箱高度共129 passed；上述改动文件Ruff全部通过。只读独立审查另跑19项新边界测试通过，无阻断。根代理接手完整测试和fresh真实三件；本次没有新增模型、地图、运动或部署调用。

- 2026-10-21 全量回归定位到一处旧测试替身：轨迹导数资源释放测试仍传字符串request，缺少新FK证据所需scene_token。替换为显式测试请求及FK替身，仍检查成功、native异常和导数异常三条路径释放SDK结果；生产逻辑和skill行为无变化。

- 2026-10-21 统一三池规划循环：0ce在线首池零源IK、第二池两项分别goalset失败及lift失败，均为正常SDK完成结果但没有可执行完整轨迹；旧分支在约13秒内停止且未用第三池。现由Runner单一attempt/deadline覆盖Demo的IK与规划循环，仅完整确定的整池规划耗尽也可使用余下池，总数仍3×1000、默认180秒。pool_exhaustion逐项验证冻结身份、唯一候选全集、零未规划数和明确phase失败；unknown、服务/过期/超时/取消、最终FK拒绝及命令/硬检查失败不授权补池。没有改变SDK、输入云、地图、阈值或执行策略。新测试先观察到3个预期失败，实施后171项规划/Runner/Demo/程序与边界聚焦测试通过，相关Ruff通过；包含首池零IK→次池规划失败→第三池成功、混合耗尽、只有一次attempt及严格递减剩余时限。此为CPU替身验证，未调用模型、后端或运动；实际三件验收仍待根代理全量回归、部署与新run。



- 2026-10-21 统一池独立审查补核：发现旧evaluation轮次在验证前被覆盖的绑定缺口，已改为IK前校验生成池原round、规划前校验evaluation原round与冻结字段，只追加sampling_evidence。两项错误轮次回归均先红后绿；Demo测试替身补齐实际接口字段，不降低生产证据要求。171项聚焦及改动Python文件Ruff通过。

- 2026-10-21 本地最终状态QA工具：根代理新增data/audit_final_demo_state.py，只读取保存的MuJoCo最终实际位姿，独立计算椭圆柱完整占地是否在目标区及底部到桌面的间隙。该真值诊断不输入控制器、不更改运行时skill或完成ledger；skill行为无变化，报告不得代替当前视觉/执行链验收。

- 2026-10-21 持物容量最小修复：3ce真实已夹持的1955点（1361实测+594显式补全）在128球时对非接触Link4假交叠0.198mm，全部原点加原6mm余量仍有4.112mm净距。仅将允许上限/demo无保存值默认改为256，原PCA/1cm停止尺度不变，实际144球后全部非接触夹爪球最小净距1.802mm；旧ef356实际200球、净距1.201mm，全部原点及6mm余量覆盖均保留。普通默认16/显式128不变，129–256保存值不再降回16。新fixture为tests/fixtures/luna_second_carried_surface.npz，JSON附原请求/URDF/YAML及fixture哈希，CPU对照data/3ce-place-failure/budget256-cpu-regression.json。23个预期红测试后实现，204项持物/设置/API/GeometryRuntime/离箱/释放聚焦测试及8个改动Python文件Ruff通过。规划代理独立GPU对照仅改预算，原128起点self cost>0而world/bounds为0；256实际144球起点all cost=0，同固定单目标plan_place的approach/place及原硬检查通过，见isolated-result.json和isolated-256-result.json。没有修改分组方法、接触豁免、区域、5mm阈值、SDK或max_attempts；没有执行运动，仍待新在线三件验收。



- 2026-10-21 抓取前夹爪兼容性只读CPU审计：使用3ce全部1项、5be全部5项已通过sourceIK的官方FK目标及冻结1955点，不用真值/新模型/GPU/运动。按原Link3相向pad宽度预测q_width，未平移原云，保留6mm包络及仅Link3接触豁免；3ce931点包络净距+2.959mm，5be508为−2.372mm，5be仅826为+0.395mm，但其原附件球仍−0.896mm且归档lift失败。因此冻结池无“预测通过且完整原轨迹成功”的5be候选；预测不是实际闭合或持物可执行保证，不改阈值、原云或attachment检查。报告/脚本/接口建议在data/5be-place-failure/pregrasp-compatibility-report.md与pregrasp-compatibility.json，尚未实施生产筛选，skill行为无变化。



- 2026-10-21 抓取前预测质量门实施核对：两例冻结原观测全点与source-FK候选测试先红后绿；3ce保留931，5be只保留826、剔除原实际选中508。157项聚焦测试及改动Python Ruff通过，包括A/B工具轴、全点跨chunk/原padding、坏证据、raw成功而eligible空、最终FK跨质量边界不补池。旧合成FK/索引单测只在各自测试范围隔离质量门，16项新真实几何/边界测试使用实际模块。独立只读审查无阻断，A/B×5个q与旧URDF FK/self有效球最大差6.70e-8m，刚体变换不变性gap差1.79e-16m。826原归档lift失败，预测不是完整抓取/持物保证；未调用模型/GPU/运动或部署，下一真实验收由根代理完成。

- 2026-10-21 回放IK后筛选显示核对：按同轮原始candidate_indices关联唯一candidate_filters，accepted=false不再显示为通过；保留rawStatus、原result、filterEvidence和三态eligible。历史未归档筛选只显示原始IK范围，重复/损坏/缺匹配记录未知。305项前端测试及production build通过；after_plan原始含义、工具坐标、skill判据和live状态无变化。



- 2026-10-21 持物组件语义修复：ffb三池915项源IK中raw7项，1项朝向机器人、6项仅在同Link4_R球被旧质量门拒绝。原点对STL仍有5.001–5.314mm净距且inside=0；加原6mm为−.999..−.686mm，球近似进一步变−2.293..−2.006mm；measured与completion几乎相同，不能归咎单个补全点。当前1955点及全7项FK已归档fixtures；共享精确9链接接触组件后6项通过、972仍朝向拒绝，原5be5项也不再被手内包络否决。4项预期红测试后实施，192项聚焦及Ruff通过，含A/B、未知前缀不豁免、TCP/上游/对侧pair和全部world球不变、TCP负例与最终FK拒绝。独立GPU原5be同token/fullbounds/固定XYR/原点/6mm/256，只在进程内更换project pair mask，start world/self/bounds均0，单plan_place通过；11.5s进程退出，无live服务或运动修改。data/ffb-quality-failure/assembly-gpu-result.json保存原始结果；这是冻结规划对照，未证明新鲜三件完成。



- 2026-10-21 完成件定位支撑带修复：f9bd首件抓放/视觉确认/保留图返回已通过，后续头部SAM两候选各729点因4/3个混合轮廓桌面深度点越槽而在Decisions前停止。仅将已有缓存实测桌面±6mm排除用于scope定位，剩余702/701点全在原唯一槽；原始点/轮廓/RGB和最终几何过滤不变，归档原始/支撑排除/匹配点数及bounds。不按槽XY裁点、不用真值、不降0.8、不删除完成障碍；空剩余、抬高槽外、非有限、多槽及unknown仍拒绝。冻结回归先3红，修复后79项相关测试及Ruff通过。证据data/f9bd-completed-tracking-diagnostic/，没有新真实模型调用或运动，不改旧run状态，待fresh三件验证。

# 2026-10-21 逐件拆包流程变更



本次新增 `prepare_unpack`、`mock_unpack`、`verify_unpack`、`plan_transport`、`execute_transport`、`drop`、`verify_drop`。一次 Responses 编排，粉色包装和黑色内件共用 canonical ID。桌面放置只是第一段；最后右盒视觉核验才完成。共享桌面区域只有一个完整区域，所有物体顺序使用同一中心，不按清单数量划分槽位。新流程只在双臂观测位姿建立最新2cm TSDF/ESDF，其他运动复用最近地图；mock动作推导的占据与实测几何分开标注。



## 2026-10-21 拆包流程规划适配核对



本节是逐件拆包需求的当前规划边界，取代上方历史核对记录中的16/256球默认及黑件分槽预测；不是新增独立运行时skill。`DemoAlgorithms.plan_attached_pose(observation,target_id,arm,scene,holding,goal,planning_settings=None,cancelled=None)`接收实际抬升反馈推导的一个世界TCP目标（米制4×4矩阵）。必须已有同ID、同臂、同场景的verified/attached holding；先查询该实际目标的位置工作空间，再调用现有`/tools/plan_carried_pose`。后台保持当前attachment与两手实测开合，调用公开`plan_pose(...,use_implicit_goal=True,max_attempts=...,enable_graph_attempt=1)`，不调用place批量接口、不寻找其他XY/R。目标R由调用层冻结为实际抓取抬升R；位置工作空间通过仅是前置条件，不能代替完整碰撞规划。



普通持物pose保留原生q/v/a，用现有五次Hermite插值及`time_scale`生成实际命令；“五次”是多项式阶数，不强制5倍时长。`transport`验证阶段必须state=held，再经`/carrying/validate`以当前token、持物、两手反馈和5mm阈值检查这条实际命令；缺导数、过期token、未带attachment优化证据、workspace拒绝、取消或碰撞均不输出可执行成功。返回`command`、`command_check`、`raw_trajectory`、`carrying_binding`、`goal_workspace`及有效设置。保留原高级求解设置；公共单pose路径未消费的batch finetune参数仍明确记录为未应用，不能声称已生效。



`snapshot.stage='unpacked'`只对当前黑件跳过旧固定桌面槽位可达性与q_width静态质量启发式；GraspGen原始方向45°向下/背离机器人、单次工具Z深入1cm、真实源IK FK/最终抓取FK对原目标5mm/.05rad边界保持。源可达性只查询当前黑件实测位置，destination状态为`pending_actual_lift_transport_goal`，不把右盒底部当投放TCP。黑件lift依据当前实测table的position/size/surface_height_m及同token的唯一`semantic_table::<id>`已发布cuboid顶面，使用完整目标几何的当前球包络计算至少配置lift高度，不能沿用粉件initial_parent_id的左盒。缺失/过期/歧义支撑或目标不在桌面范围均停止；全部其他世界障碍仍保留给实际规划器检查。



`plan_dual_vla_motion`接收A/B各一个世界TCP矩阵；沿现有双臂公共pose路径规划，无臆造关节目标。复用最新保留地图的`return_after_place`边界，服务将米转换为其双臂矩阵接口要求的毫米；由服务完成插值/时长缩放，本接口不重复缩放。结果必须detached、同token、有限配对7关节轨迹，再作双臂实际命令硬检查，返回弧度`paths`、`dt`及`command_check`。`publish_action_envelope(scene,target_id,bounds,action_id,source)`仅向`/perception/update`发送`action_envelope`；bounds为有限三维minimum_world_m/maximum_world_m，source仅mock_vla或verified_drop_action_bounds，合并返回新scene。它不是RGBD重建或新的测量证据；世界包络保留/替换由scene_update负责。



持物球默认24、允许1–31，覆盖全部原实测/显式补全点并保留原不确定度和margin；粗包络不代表删除物体表面。仅附件与所属手精确9个link（Base与Link1..4各L/R）允许组件接触；TCP摄像头支架、上游臂、异臂/异手和世界保持检查，普通机器人自碰撞pair与世界球不变。保存设置中历史32–256预算仅在load_settings时明确UserWarning迁移为24，不静默修改已有binding或归档文件；有效1–31保留，新请求32以上拒绝。历史几何fixture原文件未变，当前回归在内存应用24球策略。



公开SDK参考沿用项目已存只读清单：cuRobo commit `8e734f3ced1df898990bcd92de40abce475907db`，`/usr/local/lib/python3.12/site-packages/curobo`公开运动规划模块SHA256 `af8f510c2217374deff557b582bfa80db961fc2b8170c6683d67d8b431dcd67e`；官方`examples/getting_started/motion_planning.py` SHA256 `21c5b40d214018d70304b61467de7df0ad3b6d97a1df6c112940eaeda4124228`。依据在`data/collision-symmetry-2026-10-08/validation/`，本次未重新调用远端或修改SDK。新增接口/阶段/预算测试先失败后实现，规划/持物/设置/服务/插值等246项聚焦通过；旧默认值及无MuJoCo导入边界等53项通过，改动Python Ruff通过。两组包含重叠用例，不相加作独立总数；这是本地软件验证，尚不代表新两物体流程真实运动验收。







### 2026-10-21 场景/模拟拆包与录像 revision 核对（scene 专属）



默认改为两个 6×4×10 cm 粉色椭圆柱，左盒 XY=[.24,.50] 与 [.16,.64]，中心距约16.12 cm、距内壁至少6.8 cm；1/3件斜排和4件网格保留配置。指定双臂观测角、外参、盒桌尺寸、目标区域和位置伺服参数不变。1–4件重新渲染，均无初始非预期接触，默认两个在左腕有效像素为1527/1650；资源指纹已更新。



新增公共模拟动作 `Simulation.mock_unpack(work_area, output_pose=None, task_id=None, *, generation, target_id=None, owner_id=None)`：`work_area` 明确方形中心XY/边长/桌面floor_z_m，仅处理完整位于区域内且底面距支撑±6mm的唯一粉色实体。输入canonical/run ID只用于回执，绝不用于暗选物理物体。保持同一物理ID，把它变为4×3×6cm、0.08kg黑长方体，默认右半桌中心[.36,-.14,桌高+.031]，保持直立、yaw相对增加30°。输出位置只是mock动作意图，不是抓取/轨迹可达证明。编译失败、非唯一、运动中、输出穿透、generation/revision过期均拒绝；原子切换保留generation、时间、机器人q/qvel/ctrl和位置目标、其他物体与reset基线，仅目标自由位姿/速度改变。



回执明确`source=mock_vla`、`is_visual_evidence=false`，携带target_id/owner_id/action_id、completed及三种完成时钟、commanded_pose/commanded_size_m和revision。这是用户授权的模拟替代动作，不是视觉事实。外层仍须先规划到双臂自然VLA姿态，再mock，再按保留场景返回观测位姿；动作推导占据和完整顺序由外层负责，本API不采图、不建图、不规划、不自动返回。部分可见粉件仍须用明确100mm先验，黑件下一阶段用明确60mm先验；mock回执不能替代当前RGBD与Decisions证据。



实时manifest与历史录像schema3随几何revision切换；旧形状在途RGBD拒绝入缓存，录像不跨revision插值出虚构变形。实际本地MuJoCo归档帧证据为`data/mock-unpack/render-audit/scene-{before,after}.png`。本机无ffmpeg，MP4编码如实failed，不能声明视频文件验收成功。保留已有loaded闭爪回归并加入双臂公共夹爪入口测试：desired q/ctrl不变、有真实接触，有限刚度下不声明零角度漂移，不新增任务力/力矩分支。详见`docs/demo-scene.md`。



### 2026-10-21 拆包生命周期感知边界（perception 专属）



`demo_lifecycle_perception.py` 提供 `verify_unpacking(client, capture, *, target_id, lifecycle, action_evidence, owner_id)` 和 `verify_right_bin_drop(client, capture, *, target_id, lifecycle, action_evidence, reference, owner_id)`。前者在 mock 动作后的同一观测位姿三RGB上核验黑色内件及当前任务对应，成功后才允许同帧 SAM3/RGBD；后者用当前三RGB与最近一次黑件运输前的三RGB确认本次物体进入右盒且离开开爪，不能把已有黑件算作本次交付。动作记录只提供执行上下文；owner、canonical ID、generation、动作来源、反馈和帧先后必须匹配。任一必要答案为 no、unknown 或置信低于既有0.8门均不推进，不把原未知改成成功。helper 不改台账、不采图、不建图；root Runner 负责成功后的 phase 更新。



初始 RGB、bbox 和实体ID始终保留。当前实体描述由已确认生命周期覆盖：粉色包装阶段使用 pink 查询/100mm先验，确认拆包后同ID使用 black 查询/60mm先验；active 黑件不再要求左腕源盒覆盖，尚未搬动的粉件仍保持该门。高度只取用户授权尺寸，并与当前 RGBD 和缓存实测支撑面作数值一致性核对；RGB Decisions 不测量高度。实测已覆盖高度时不增添补点；缺少底部时保留原点并单独标注先验补全来源。投放进右盒的黑件可能倾斜或堆叠，不套用直立60mm补全；仍要求其当前可信身份与实测障碍几何，不沿用旧粉件桌面槽位track或静默删除。多个已完成黑件若无法区分仍如实unknown，本轮未引入隐式聚合或真值身份。



模型生命周期 context 仅允许 phase、appearance、appearance_revision、to_table_arm、to_right_bin_arm 五个字段；完整 mock 回执、模拟器物体真值、commanded_pose 及旧响应只留在本地归档。右盒描述为初始为空、之后可能含已完成黑件，不持续声称空盒。32项新增感知回归覆盖未知/错链/过期、6cm与10cm切换、完整云不补、初始引用不变、completed当前几何及真值不进入payload；5项独立Runner集成验证同帧复用、reset后契约推进、右盒完成/未知分支和两类请求的归档租约。最终相关聚焦为132通过，另1项正在由root修复的SDK action-envelope字段隔离回归失败；本代理6个Python文件Ruff通过。没有真实模型、GPU、运动或部署调用；这些CPU结果不代表新逐件拆包流程已完成在线验收。



- 2026-10-21 中间桌面核验术语修正：3a98已完成桌面释放，但camera2粉件存在/分离置信为0.28/0.79，原结果保持unknown。实际6张模型图已按SHA下载核对：左腕可见粉色顶面和张开双指，头部被手臂遮挡，右腕看向右盒。原intro称内件为black workpiece，而4题又询问pink workpiece；现仅拆包流程统一询问pink foam-wrapped object，保留“粉色泡棉覆盖黑色内件、之后由VLA拆包”的用户先验，明确本阶段不要求看到黑色或完整侧形，先验不能证明当前存在。另说明青色连接线标注固定区域中心，不是物体身份指针。6图/投影/题名/choices、同view三项及0.8门不变；58项释放回归及Ruff通过。`data/3a98-postplace-diagnostic/frozen_release_probe.py`直接恢复归档event00135的capture/binding/reference并调用当前生产payload/解析，离线证明六图字节不变；尚未进行新模型对照，不声称已解决视觉成功率，不修改旧run终态。



- 2026-10-21 初始清单主视角语义修正：8361129a的左腕清楚显示独立源件，头部仅见同一源盒的一片粉色局部；旧提示却要求三图每个截片均能对应个体ID，超出了用户指定左腕主视角的要求。现仅改Responses字段说明和prompt：完整性要求camera2中可独立区分的源物体全部枚举；辅助视图不必看全，也不因一个截片无法匹配个体ID而自行新增实体或强制unknown。辅助视图若有可信的额外独立物体、计数矛盾或真实可能新增物体尚无法排除，仍须unknown，不能省略。未硬编码数量，未修改解析/置信门或把旧unknown改confirmed。136项发现、count/all及同帧Decisions门回归与Ruff通过；新增负例确认实际输出的未知辅助实体仍被parser拒绝。未调用模型、GPU或运动，线上效果待根代理新证据验证。





### 2026-10-21 实际闭爪腕关节漂移复现与位置伺服修正（scene 专属）



实际两件run `2e473e106e50411d9004191ccf36b242` 在close/dwell/capture_reference后被原1°起点门正确阻止，Joint7_L偏移2.6046°。只读保存原config与完整approach/grasp命令及hash，本地同模型（model_builder原hash完全一致）重放实际物体几何/四元数和闭爪时长：闭爪前误差<0.0004°，随后峰值2.008°、终值1.998°，J7输出−18Nm饱和96.85%，接触反力约+18Nm，手臂目标未变。归档无初始qvel，重放以已静止初始场景零速度初始化，不冒充完整同一积分状态。



单变量对照：仅机械臂模拟能力×10仍有.510°峰值；仅位置kp×20有.807°瞬态；组合后峰值.03492°/终值.01647°，J7不再饱和。现机械臂仅改为MuJoCo近似刚性位置模式kp=240000、kv=600、有限actuator与总关节能力=URDF effort×10（每臂1080/1080/660/660/180/180/180Nm）；这是模拟执行器能力，不修改URDF原额定值，不引入任务力/力矩/阻抗逻辑，不覆盖机器人qpos。夹爪kp200/kv10/8.066Nm原限制保持，原1°起点门、.05°仿真起点检查和5mm碰撞阈值均不变。



以同一冻结actual输入加入左右镜像测试，逐个积分步检查闭爪误差<.05°、真实手指接触、desired q/ctrl不变、状态有限及无数值警告；1ms生产步长与.5ms半步长均验证，A半步长峰值.04370°。旧带载/重力保持/夹爪限制测试保留。证据`data/2e473-servo/`与`tests/fixtures/luna_unpack_close_saturation.json`；场景/观测角不变，观测资源按新model_builder指纹重新验证。更新ASSET_PROVENANCE现行位置伺服说明，未修改SDK或原资产。此修正仅证明冻结接触下的姿态保持，不宣称后续live lift、持物或整轮成功。



### 2026-10-21 拆包与投放视觉调用的实验归档边界修正



实际 run `d9cebe60018546519a0ac428ef706e19` 已完成桌面核验、双臂 VLA 姿态、mock 拆包与返回观测，随后在 `verify_unpack` 的 `algorithm_start` 归档中错误携带 `DemoVisualInventory` 客户端，触发 TypeError；原 run 保持 needs_attention，不能视作拆包视觉核验成功。此前仅关闭实验录制的集成用例没有覆盖这个边界。



现 `UnpackWorkflow.verify_unpacking` / `verify_right_bin_drop` 使用绑定适配器，将客户端留在 self 内；`call_algorithm` 的位置参数只有 capture，关键字参数只有可归档的生命周期/动作/引用数据。没有用 repr 或过滤器掩盖客户端，也未改视觉题目、阈值或运动。Runner 对两个绑定方法继续持有原视觉归档租约，直到真实调用结束；取消或 reset 后迟到结果仍写入原实验，不写入新运行状态。



真实 ExperimentStore、有效 session.id 与 MockTransport 的回归先复现 TypeError，再验证修复：事件可 JSON 序列化、manifest 无归档错误、归档无客户端/密钥，拆包的3张及投放的6张实际请求图逐张 SHA256 一致；取消在途调用后仍保留完整回答和图片。生命周期、集成及归档相关57项通过，两个生产文件和集成测试 Ruff 通过。本轮未调用真实模型、GPU、运动或部署，线上新流程验收仍由新 run 决定。



- 2026-10-21 拆包核验的可观察问题修正：`data/d9ce-unpack-visual-probe/actual-call.json` 原同帧探针返回 black_core=no/.60、same_item=unknown/.94；头部 CURRENT camera3 可见灰桌上的小黑实体，双腕主要朝盒，camera1仅见边缘截片。旧题把“黑色内件、右工作空间、盒外”合并，并可能要求从粉色初始外观重新认黑件。现 black_core 仅问任一 CURRENT 图中灰桌上是否可见独立黑色物体；same_item 结合这个当前可见实体与唯一活动项生命周期判断对应，动作记录本身仍不是视觉证明。明确一个视图足够、不要求全侧面或其他相机也看见、不从RGB判断机器人工作空间；保留粉色泡棉覆盖黑芯及VLA拆包先验，多候选/无法清楚判断仍unknown。题名、choices、三张原图字节/顺序、解析与0.8门不变。59项生命周期/真实归档相关回归与Ruff通过，原两条未知回答离线核对仍不更新phase或appearance_revision。本轮未调用模型或运动，不能把文字修正视作视觉对照已通过，原run与探针结果不变。



- 2026-10-21 拆包桌面区域放大与身份来源对齐：d9ce同三图v2探针仍为black_core=yes/.72、same_item=unknown/.13，原结果保留。当前verify_unpacking增加必填work_area，Runner只从self.region配置XY与缓存实测table.surface_height_m构造；不读mock commanded_pose、物体真值或SAM mask。使用既有project_placement_regions把工作区支撑面投影到camera3，裁剪其像素包围框加固定比例/最小像素边距，以PIL最近邻放大至最多4倍，追加到原三张完整RGB之后。图文metadata记录原图JPEG SHA、同帧/generation、裁剪像素范围及倍率，明确只是像素放大，不是物体mask、新观测或存在证明；完整三图仍用于发现裁剪外竞争对象。合法区域若head缺校准、出画或位于相机后方，则不加crop，仅保留三图；不增加头部必须看见的门。



  canonical ID来源改为既有可信桌面释放与当前单活动项生命周期，代码核对table_visual_evidence的target、release反馈、execution_chain来源、at_destination及原置信门；Decisions只确认当前可见黑件且没有竞争桌面对象，不要求当前RGB重新证明黑件从粉泡棉内部出现。VLA/包装先验仍保留且不作视觉存在证明。两题名/choices、no/unknown/.8、动作先后、同帧及几何更新边界保持；归档增加原table request/binding引用和crop来源。71项聚焦及Ruff通过，包括原v1/v2未知答案不提升、无GT、原3图字节不变、裁剪来源与像素、无效工作区、无head裁剪回退及真实实验新增第4张图/取消归档。未调用模型、GPU或运动；本地测试不代表新同帧模型对照已通过。



### 2026-10-21 黑件高度补全的建图入口与身份绑定核对（planning 专属）



实际 `Algorithms.build_scene` → 编码/解码 → `ProjectSemanticBuilder.build` 的 CPU 回归先复现：感知已提供合法 60 mm 黑件补全，而 builder 仍只接收 100 mm；把黑件错误改为 100 mm 反而可过旧门。现建图适配器仅复制补全证据，并从当前清单附加实例 ID、观测 ID、color、current_appearance、appearance_revision 和 lifecycle_phase，不修改原测量或补全数组。60 mm 必须绑定同一当前实例/观测、black、unpacked、正整数 revision、unpacked_confirmed，以及用户 60 mm 高度来源和缓存实测桌面来源；错绑 pink、旧观测、缺失身份、已投放/完成黑件和其他高度均拒绝。带生命周期的 100 mm 只接受 wrapped pink、revision 0 与用户 100 mm 来源；无 context 的旧 100 mm 合同保留兼容，不能因此接纳无 context 的 60 mm。



原有限 bounds、合成点非实测标记、原 measured RGBD 体素和独立 `height_completion` cuboid 保持；选中目标时两层仍一起禁用。完成件在右盒可能旋转，不能套用这一直立 60 mm 入口。另一个独立红测证明补全循环把场景 `MappingBounds` 覆盖成物体 AABB；仅将局部变量改为 `completion_bounds`，返回场景仍保留原 workspace bounds。mapping、感知生命周期、RGBD 算法、补全、黑件 transport/lift 与开爪 sweep 共 164 项聚焦通过，三个改动 Python 文件 Ruff 通过。SDK/vendor 未改，无服务调用、模型、GPU、运动或部署；真实右臂黑件抓取、持物 plan_pose 和投放的在线验收仍待新运行。



- 2026-10-21 根代理实际拆包v3同帧验证补记：`data/d9ce-unpack-visual-probe-v3/`，request hash `2769b699e8464b82045189369e4098a9d0148fd23aa338fe376679dee2e995cb`，同3张原RGB加数值工作区派生放大图，生产verify_unpacking返回success/confirmed。该根代理实际API探针验证了这一冻结帧的拆包识别；没有新运动或原run台账变更，不代表完整逐件流程验收。



- 2026-10-21 b15c桌面释放视角误指诊断：实际run `b15ca817bee244c4b1eea821e6ad8cce`仍needs_attention。归档camera2可见独立粉色顶面及张开双指；camera3真实桌面工件被手腕遮挡，模型将腕部粉色端盖判为工件，旧CURRENT中心引线正落在端盖上。camera2区域原答案为yes/.76，camera3为workpiece yes/.99、region yes/.80、separated no/1.0，因此原解析正确停止。证据及逐题比较在`data/b15c-release-diagnostic/report.md`。



  最小呈现修正仅在demo_transfer_verification：去掉区域中心引线，以CURRENT table area旁注标明桌面投影；明确较高顶面/侧面不必完全落入二维轮廓、机器人粉色圆环/端盖不算独立工件、遮挡或二维重叠不足以证明接触。三原图、区域坐标、题名/choices、0.8门和可信跨视角反证规则不变。61项释放回归及Ruff通过，新增像素回归证明区域中心不再被指针画过，原b15c答案仍判失败。



  唯一冻结生产函数对照request hash `53ffd99a58583176033c239ef9c83ef83a5b04f1ddd74269fa7a230857df7b53`仍unknown：camera2四题yes，confidence .24/.77/.88/.52；camera3 yes/.66、yes/.63、no/.96、no/.63；无合格view，head对象/区域均不可信，因此无达到门槛的强冲突。这次对照证明呈现修正未足以稳定识别，不能宣称视觉问题解决或完成验收。完整请求/回答、6图hash及生产模块hash已保存comparison目录；仅隔离进程调用一次，无重试、无重新采图/建图/运动、无部署或原实验终态修改。



- 2026-10-21 b15c视觉主语单变量对照：用户只要求当前图像中的粉色物体，四题原主语pink foam-wrapped object额外要求从平滑顶面识别材料/包裹结构。现仅拆包流程四题统一为separate pink object，上下文保留foam包black及之后VLA拆包先验，但明确不要求RGB证明泡棉材质或包装结构。去引线后的6张图、区域投影、其他规则、题名/choices及parser均保持；61项释放回归与Ruff通过。



  根代理另行授权这一单变量的一次同帧production对照，request hash `91b9ab0f345ce1ed0a53d4045e52d14f6f58604e3adf840b246d058412909c99`返回at_destination，qualified camera2，conflicts为空。camera2存在/区域/分离均yes/.97、.93、.96；support yes/.48仍为诊断unknown，不宣称视觉证实接触。camera3对象/区域置信.37/.22，故其分离no/.96不构成已可信识别当前工件的反证；反证parser未改变。完整证据`data/b15c-release-diagnostic/pink-object-comparison/`及report.md。没有重试、重新采图、建图、运动、服务部署或原run台账修改；原b15c保持失败，此冻结视觉通过不等于完整逐件流程通过。



- 2026-10-21 2e64拆包后同帧身份覆盖诊断：run `2e6487d72001407da7e87bc617e0fbed` 的黑件C4/C5=.93/.86、左盒C6/C7=.84/.87、剩余粉件C8=.86及none=.96均通过；唯一缺失为右盒ID（C9/C10=.33/.40），C0只是complete_union失败后首个报错。当前生命周期分支曾把所有实体的INITIAL参考都称为历史，并让盒子也按“current lifecycle appearance”核验；现仅已拆包任务物体的旧粉色外观过期，稳定容器仍比对INITIAL参考、当前RGB与校准定位提示，空/非空不改变盒子ID，角色和相机编号不直接赋身份。全部候选、图像、canonical choices、.8门和必需几何覆盖均不变。



  新增2个红测后修复，生命周期/部分身份/呈现/真实归档集成79项通过，Ruff通过。一次授权同帧生产对照先精确重现原完整wire/hash，再验证12张JPEG字节、14候选、19题名及choices不变；新request hash `2788df60e6cfbf7166f91a0c5d191c94e5bbcbbab7fe0594bfe2df3d021c3050`仍visual_identity_unknown：右盒C9/C10=.51/.65，黑件.88/.82、左盒.89/.83、剩余粉件.83、none=.94。右盒visibility=.80不能替代身份。已停止调用，没有继续换prompt、重试、采图、建图、运动、部署或改原run终态。证据`data/2e64-identity-diagnostic/report.md`及`scope-comparison/`；提示作用域修正尚未解决真实右盒匹配，不能宣称完整流程通过。



### 2026-10-21 两只稳定容器的显式实测身份对应



`demo_container_tracking.py`只用于任务中不被操作的source/right容器，不匹配粉色或黑色工件。首次成功观测后，从初始Decisions可信ID、完成机器人/表面清理的views.instance_masks和原capture注册深度/标定，在visual_reference新副本归档container_identity_tracks；原RGB/entities内容不变。初始观察UUID、generation、Decisions hash、实体摘要、数组/标定摘要绑定作用域；已有owner时也核对owner。后续仅在观测位姿、同相机标定且相机差≤1mm/0.2°的适用条件下比较，IoU≥.85、交集有效覆盖≥95%且≥512像素、深度残差P95≤2cm、唯一容器ID匹配才建立measured_container_correspondence。缺失、过期、歧义或不一致回落原Decisions；视角条件不是新增运动精度要求。



测量对应独立归档，不生成模型confidence，也不修改原unknown。仅已匹配候选省略重复identity题，其余题仍保留全部容器ID选项及visibility。所有必需ID、剩余粉件camera2覆盖、同帧none与当前有效几何门保持；同ID多相机/重复提议按原像素ownership合并，容器仍是当前地图障碍。新点云全部来自本次capture，identity_source/identity_evidence区分模型、测量及混合来源，历史深度仅作对应依据，不作为当前障碍点。此次机制不新增采图、SAM、建图或模型调用。



真实2e64冻结证据：左盒C6/C7 IoU=.975/.976、深度P95残差.06584mm；右盒C9/C10 IoU=.999437/.999451、残差.000179mm；每项共同有效像素超过71,000，粉/黑/大背景不匹配。新增实际fixture及跨reference/reset/owner、标定/视角变化、少像素、深度变化、多ID歧义、unknown回落、混合group去重与来源测试；169项聚焦及Ruff通过，契约更新后program/observation-contract/tracking的45项通过（有重叠，不相加）。SDK未改。



唯一授权production resolve对照从真实初始成功归档经新helper组装track，明确为offline assembled而非旧run已保存字段；原完整wire先精确复现，六张INITIAL/CURRENT原RGB字节不变，原14候选仅省略四个测量匹配identity题，其余15题choices不变。新hash `0673b95df55d35ce6a65fb463629f2f8b1c47c9f707b5a0bb1a750566dbd4689`仍visual_identity_unknown：剩余粉件C8=.59，head黑件C4/C5=.64/.68，大背景C11却选黑ID=.82；none=.91、visibility高分不能代替身份。两盒对应正常但工件证据不足，原门正确停止，未重试或扩大到工件追踪。证据`data/2e64-identity-diagnostic/container-tracking-report.md`及`container-comparison/`。原run仍needs_attention，未部署/运动/修改台账，完整逐件验收未完成。



- 根代理只读回放QA补记：`data/unpack-filtered-replay-qa.txt`与`data/live-replay-qa/2e6487d72001407da7e87bc617e0fbed/filtered-stages/report.json`确认实际Edge WebGL面板IK前1000、IK后100、plan_grasp_batch后16（所选batch15通过1失败）可读并已查看截图，pageErrors=[]。仅data/check-replay-filtered.cjs辅助检查，skill行为无变化；这项回放验证不改变原run失败结果。



### 2026-10-21 已确认桌面黑件的几何支撑范围



### 拆包流程的当前角色关联（历史策略，原粉件重编号授权已撤销）



以下保留当时实现和实验记录；当前仍包裹粉件必须遵循文末“全程固定原始ID”规则，唯一角色不再足以确认粉件身份。



用户随后明确允许盒中剩余粉件按当前观测重新编号，保留已完成数量和目标区域；不再要求对同款待处理粉件证明最初实物身份。该授权不改变当前正在处理工件的粉色包装→黑色内件任务记录，也不允许重复计数。默认两件任务在首件拆包后只剩一个源粉件，当前角色关联使用这项清单/数量语义，不进行最初同款外观复认。多同角色候选的资格边界仍如以下实现所述，不把未实现的任意多件重编号描述成已完成能力。



`demo_lifecycle_tracking.lifecycle_candidate_scopes`与`demo_lifecycle_inventory.resolve_lifecycle_inventory`用于`associate_inventory`的拆包流程。首次Responses及初始SAM/Decisions关联不变。后续只有每种当前角色恰有一个清单ID、左右盒实测身份对应有效、原清单/工序/同run与generation证据一致时，采用当前角色关联；否则回退原关联，不新增调用。当前角色为左盒未处理粉件、主桌已确认拆包黑件、右盒已完成黑件；其ID由原清单及已核验阶段保持，不从初始粉色外观猜测黑件身份。



候选只按本次SAM轮廓的一像素内圈及原6mm支撑带规则提取定位点；所有剩余有限点须落入唯一角色的实测支撑范围，不按XY裁点制造匹配。桌面范围来自缓存RANSAC主桌和用户6cm黑件高度及15mm测量容差；盒范围来自本次可信容器点，已完成黑件不假设竖直6cm。原候选及最终几何过滤保持独立。匹配只是提名，不能直接接受身份。



同一次Decisions看三张完整CURRENT RGB、候选注释及原像素放大图，逐候选短yes/no/unknown检查粉/黑工件外观并排除机器人、盒和背景，3D支撑范围由当前深度和缓存几何独立检查；yes且confidence≥0.8才与唯一角色及工序证据共同绑定canonical ID。黑件不再与历史粉件图做外观匹配。全清单当前有效几何与左腕剩余源件覆盖仍须满足；首次完整性门保留，后续合法角色关联不重复全局发现，`demo_inventory_check`如实记`not_reassessed`/`frozen_initial_inventory`、不填写none或confidence。Runner仅对已冻结拆包清单、同owner/reference/generation/当前请求及完整已列实体覆盖接受该状态；初始化和generic路径不接受。桌面黑件必须有verify_unpack当前三视角综合结果及单active执行链证据；该要求取代历史black_core/same_item两题。原答案归档为真实binary回答，`lifecycle_correspondence`单独记录本地角色/工序/位置依据，不伪造canonical-ID模型回答；`identity_evidence`随容器合并保留。scope全内容与容器binding摘要进入请求hash，避免同帧更换证据后命中旧parsed缓存。低置信、反证及缺几何仍停止。“全部完成”指初始冻结的搬运清单，不声称后来没有新增对象。



初版角色请求的真实冻结对照`95d6e90b172fa1e65c2a0f44df91da8061f949173ab168b93555fce32b9bfe6b`：C4/C5黑件yes/.94/.94、C8粉件yes/1.0，C13边缘yes/.35未过；额外全局盘点题extra/.07使旧解析停止。原9图/5题、回答及失败结果保存在`data/2e64-identity-diagnostic/lifecycle-role-probe/`，不改写为成功。按用户首次冻结清单、后续逐件记录推进的范围，最终合法角色路径移除后续全局盘点题，保留逐角色当前确认及首次完整性检查，明确未重新盘点。新增Runner冻结/同帧/owner/请求/必需实体/几何回归，相关86项通过；最新完整验收须用新run，不用离线解释改写旧实验。



### 已确认桌面黑件的点云过滤实现



`Perception._filter_geometry_groups`显式接收当前entities；demo仅对`current_appearance=unpacked`且`lifecycle_phase=unpacked_confirmed`的工件，按首次缓存实测主桌足迹（XY外扩6mm）及其表面以下6mm界限排除不可能属于桌面工件的背景深度。之后仍执行原密度、桌面表面及机器人过滤。该范围来自缓存测量和已通过的拆包视觉阶段，不读取mock姿态，不按命令目标、颜色或模型高置信强行赋ID，不增加建图/采图。源盒粉件和已投放黑件不套用主桌范围。全部当前几何被剔除时仍由缺几何门阻止地图发布。



每个受影响相机的`geometry_filter_evidence`记录支撑ID、测量范围、6mm边界容差、原始/范围外/最终点数，并透传到对象；原SAM轮廓、原请求候选及RGB保持归档。2e64真实fixture中C11的78,044个背景点全部去除；C4的518点保留486个物体表面点，其余32个桌面点由原桌面规则去除。新增6个回归及相关55项聚焦测试通过，Ruff通过；这证明错误背景不会污染后续黑件点云，不证明新的视觉调用或右臂运动已经成功。



### 2026-10-21 拆包后CURRENT候选细节呈现与冻结失败对照



`demo_identity_presentation.py`仅对`associate_inventory`且具有`item_lifecycle`的请求追加本次RGB候选bbox及邻域的nearest放大图（至多4倍、内容最长边512px）。原始三视图和既有参考/注释保留；同相机边界相差至多2px的提议仅共享展示，canonical ID、原C索引、题目、choices、置信门及解析保持。大框不缩小冒充细节；裁剪明确不是mask、新观测、身份结果或完整性证明，遗漏检查仍须看全部CURRENT完整图。元数据保存原图hash、相机、观察/generation、裁剪坐标和缩放方式；初始化与其他purpose不加此图。没有把测量追踪扩展到工作物体。



新增红绿回归覆盖原图/问题不变、邻域与边界、展示幂等、同框只共享展示、非目标purpose不变，以及真实ExperimentStore保存放大图片及hash。88项呈现/生命周期/容器跟踪测试与Ruff通过。



根代理查看真实黑件/粉件/边缘黑件放大图后执行了唯一授权冻结production resolve对照：先精确重现原wire，再核对前12张图及全部15题不变，仅追加6张图；request hash `24faf99bddcc2376b8f72e6f73ce56fcb00ea2ce2973d9838b33f8c34ca27470`。仍`visual_identity_unknown`：C8粉件.75；黑件C4/.54、C5/.63、C13/.46；C11背景黑ID降至.26；none/.72。两容器的C6/C7、C9/C10实测对应仍成立，工件与none证据不足，visibility不能替代身份。已停止调用，未再次改prompt或重试。完整18图、真实请求/回答、原始来源hash与生产解析保存`data/2e64-identity-diagnostic/candidate-zoom-comparison/`；准备图及crop元数据位于`candidate-zoom-preparation/`。本次没有采图、SAM、建图、运动、部署或改旧实验台账；原2e64仍needs_attention，细节图未解决当前身份判断，不能宣称完整流程验收完成。



### 2026-10-21 右盒投放核验的同区域前后对照



实际run `aef77e04c1c14f7a81ab7fe7319586b8`已执行桌面释放、拆包、右抓、持物plan_pose及投放，视觉核验仍失败：in_right_bin=yes/.43、separated=yes/.97、current_item_added=yes/.77。原CURRENT右腕图中黑块可见且与张开双指分离，但原请求只有右盒ID，没有位置提示，且BEFORE/CURRENT右腕视角不同。原run保持needs_attention，不把物理投放或本次呈现修正算为视觉成功。



`_verify_drop`只新增纯数据侧栏：从context.scene_observation的已测右盒全部有限点取得包围范围，核对它与BEFORE initial_capture的observation/generation及right_bin_id。生产verifier按各帧标定投影这个范围的上下表面，裁剪右腕邻域并nearest放大，添加BEFORE/CURRENT BIN REGION标签；原六张完整RGB字节与顺序不变。范围来自当前实测容器，不来自命令、release TCP、mock或GT，不重建图；投影出画只省略辅助图，不增加必须由右腕看见的条件。元数据保留来源观测、实测范围、原图hash、裁剪坐标和像素放大说明。机器人右盒角色与画面左右明确区分。



三题使用separate black object，保留本次新增与旧盒内容的前后比较；动作记录不作为存在证明。题名/choices、三个yes且confidence≥.8门、unknown/no处理和canonical生命周期完全不变。76项生命周期、真实归档和workflow聚焦回归与Ruff通过，覆盖原六图hash、两张新增图归档、无GT、过期/错绑范围、出画回退，以及原低置信答案仍失败。`data/aef77-drop-diagnostic/prepared/`为原归档同帧经生产函数离线生成的8图请求；侧栏明确由旧实际测量离线组装，并非旧run原先保存的新字段。准备hash为`6797aec1bd7e84950f647bee431f744ac9fc55513a9d8757af300cda5e51b966`。本段记录时尚未调用新模型，冻结呈现不代表整轮验收完成。



根代理审看实际配对图、整体源码同步后，唯一授权冻结production对照使用上述完全相同hash，三题均yes/.96（yes概率均.97），生产parser返回success=true；未重试。完整真实request/response、8张图、原六图SHA256一致证明及独占锁保存在`data/aef77-drop-diagnostic/comparison/`；21项program/observation-contract回归另通过。仅验证这组既有帧的新呈现，没有采图、SAM、建图、运动或旧实验/台账变更。原aef77仍needs_attention，后续完整两件流程必须以新run验收。



后续bc67冻结诊断发现raw右盒点云仍有已被建图清理的机器人边缘高点。现投放定位侧栏与transport目标、成功后action envelope统一读取同观测/代次/scene_token的指定启用右盒完整native voxel占据边界（preview抽样前），不再用raw点云极值。视觉helper严格验收`published_inventory_container_occupancy_bounds`来源、exact层名、完整边界来源及正cell_count/voxel_size；它不把占据顶界当精确盒沿。78项视觉/生命周期/真实归档集成及Ruff通过；fixture特意保留raw最高.9m、native最高.6m，验证归档侧栏使用后者。旧aef原六图字节保持，新native准备图在`data/aef77-drop-diagnostic/native-prepared/`，hash `4dd130d210bcf8e85d5aab3229c87a76f584c8d72f824d6a79823f98d6367f74`，仅离线生成，无新模型调用。前段.96成功探针是当时raw侧栏版本证据，完整保留，不能重标为新版探针或新运行成功。



2026-10-21 验证脚本核对：validate_demo_run 在 preset 后等待 API 的 algorithm_pending 清空，最多60秒；cleanup_error 直接失败，start 拒绝保留响应原因。避免清理旧规划世界尚未完成时立即启动造成409。运行时 skill 行为无变化。



2026-10-21 历史文档核对（粉件重编号授权随后已撤销，以文末固定ID规则为准）：当时demo-workflow、demo-scene 和 demo-planning 的说明统一为逐件拆包、共享中心无分区、后续合法角色关联不重复全局发现及允许剩余粉件重编号；保留历史实验范围。skill 行为无变化。





### 2026-10-21 右盒运输、投放定位与返回包络同源边界核对



`UnpackWorkflow`三个消费点统一使用同token/observation/generation的右盒完整native体素占据边界，保留区域中心XY、实际lift旋转、10cm运输余量、2cm动作包络上界余量、attachment与5mm全部验证。部分/预览、过期、禁用、错实例或缺失边界停止，不回退原SAM最高点，不采图/建图。此处占据最高Z不称精确盒沿；投放Decisions的完整原图、当前三题及0.8门保持。



冻结aef77样本的原盒云混入5个已由自身URDF网格核对的机器人点，最高Z=0.881458640m；现有地图已剔除，完整651格上界=0.631999144m。417个持物点全部保留时，TCP目标由1.207879668m变为0.958420173m，XY/R不变。新25项回归先红后绿，覆盖真实点、完整边界绑定、缺失/过期/部分拒绝、三个消费点一致及不修改地图；连同运输、动作绑定、生命周期真实归档共67项聚焦通过，Ruff通过。bc67原实际目标1.208444944m的轨迹优化失败记录保留；本地高度修复本身不能证明新轨迹或完整两件流程成功。原工程、SDK、机器人模型与运行实例未由此本地改动改变。



同一bc67冻结请求的唯一授权原生plan-only对照已由review代理执行：仅目标Z从1.208444944m改为0.958985448m，原scene、起点、持物、XY/R和规划设置不变。HTTP200、ok=true、attached=true，返回61点轨迹，0.37秒；原5mm与attachment硬检查通过。证据`data/bc67-transport-failure/comparison/`保留请求、响应与报告。未执行轨迹、未重建地图或重试；完整两件物理/视觉验收仍待新run。另21项program/observation-contract检查通过。



### 2026-10-21 投放核验按用户数量语义推进



实际b71b run `b71baed8ea21427c9405f9796380c0fb` 的CURRENT右腕与head均可见右盒内黑件、双指分离；BEFORE右腕的近半盒底被手指/掌遮挡。原八图的三个回答为yes/.86、yes/.93、yes/.77，仅第三题阻断。旧规则要求从历史图辨别哪个同款个体是本次新增，超出了用户初始右盒为空、按数量保障且不要求同款个体追溯的范围。不能据此断言模型置信度的内部成因，但该额外要求可由实际请求文字直接确认。



现仅调整第三题及对应上下文：明确`initial_right_bin_black_count=0/source=user_scene_prior`，已完成生命周期记录数N及`expected_current_count=N+1`；当前或仅释放未核验的记录不计入N。Decisions判断CURRENT同一指定盒内至少N+1个individually distinguishable black workpieces，不跨相机重复计数，允许物体接触/堆叠但要求个体边界可辨别。BEFORE辅助发现明确矛盾，不必重新证明遮挡后历史为空或追溯哪个个体是新件。动作/数量记录本身不是CURRENT可见证据，旧N件不足、遮挡未知、no及低置信仍不能完成。



前两题逐字、全部题名/choices、八张实际JPEG、0.8门、当前canonical生命周期与持物释放链不变；结果归档delivery_count及`released_in_bin_separate_and_visible_count_increment`，不声称视觉识别了同款实物身份。6个新增回归先红后绿，第一/第二件计数、旧件不足、unknown/.77不提升、图像不变及相关生命周期/归档/契约共146项通过，Ruff通过。旧b71b .77原回答离线仍失败。准备请求`data/b71b-drop-diagnostic/count-prepared/`，hash `9474e77ade8f36342641d6a4f9112305b90c97b391b92ea029b396130c3ac732`，直接恢复原算法参数，不重建ledger或场景；本段记录时未发送新模型调用，原run仍needs_attention。



根代理审阅并整体同步后，唯一授权同图生产probe使用上述hash返回success=true：in_right_bin=yes/.88、separated=yes/.94、current_item_added（至少1个）=yes/1.0。真实请求/回答、八图SHA、原.77仍失败的离线结果及单次锁在`data/b71b-drop-diagnostic/count-comparison/`。无重试、采图、建图或运动，未修改原b71b终态/台账；这只是冻结帧的新数量语义验证，不代替完整两件新运行验收。



### 2026-10-21 投放完成契约移除额外视觉分离门



实际4c2d run `4c2d4d17e8e04ddab23c2a9727ccc687` 的八图与原回答已保存在`data/4c2d-drop-diagnostic/`。仅separated=yes/.73未达门；in_right_bin=yes/.83、数量题yes/1.0已通过。CURRENT右腕原图/放大图可见黑件与两指存在宽像素间隔，但模型未提供降低confidence的理由，因此不称发现/修复模型校准或可见性缺陷。



用户完成条件是Decisions确认黑件进入右盒；合法持物→附件运输→实际开爪已有验证链，额外要求RGB再次高置信证明与手指分离是自行增加的限制。现verify_right_bin_drop仅保留in_right_bin及current_item_added（N+1）两题，均须yes且confidence≥.8；完整执行链、target_id、同帧/代次/反馈、原八图、计数及unknown/no语义不变。数量单独不能通过，非法链不能通过；全局原始ID始终由执行链保持，计数不赋ID，也不让旧N件替代当前项。



后续completed角色的_visual_yes同步验证这两题。响应仍严格按题目数量/名称/顺序解析：旧三题回答即使全高分也不能静默套用新契约；原4c2d终态不重写为成功。结果completion_semantics为released_in_bin_and_visible_count_increment，开爪只归因实际反馈链，不声称RGB独立证实。先红后绿的134项投放/生命周期/真实归档/后续关联/workflow测试及Ruff通过，包含盒内低置信/否定/未知、数量不足、旧响应拒绝和非法链。此为更贴合任务的完成条件，未调用新模型，尚不代表完整两件新运行成功。



根代理随后授权一次新契约真实同图probe：独立进程加载与当前生产文件逐字一致的副本（SHA256 `506d0852b2199cef0c2a91ffca0a8428140e64ec511aa972f8d028811d6c4e27`），未重启web或覆盖在线src。恢复原4c2d参数，核对8图JPEG及两道保留题字节完全不变，旧三回答离线严格拒绝后，发送唯一新请求`1b1c63d4fc4985e7d578b2aca43f9537fac72095603e1f3b496b9eaa28c83f7b`。返回in_right_bin=yes/.83、current_item_added=yes/1.0，production success=true，原canonical target_id保持。这两题数值与旧对应题相同，并非提升模型校准或降低阈值；成功源于明确采用任务要求的两题契约。完整请求/回答、原图hash、源hash和单次锁在`data/4c2d-drop-diagnostic/two-question-comparison/`。未重试、采图、GPU规划、运动、重建地图或改写旧run/ledger；完整两件新运行仍未由此证明。





### 2026-10-21 全程固定原始ID，剩余粉件必须视觉复认（历史混合请求，后续已分拆）



固定ID和阈值要求仍有效；本段单请求呈现被下文两份同帧请求取代，原探针失败完整保留。



用户撤销剩余粉件重编号授权：初次Responses的canonical ID及左腕原始display_id全程不变。observe_scene的合法生命周期关联仍先用已校验工序、当前容器测量和SAM支撑范围提名，但唯一角色、数量、位置或“只剩一件”不能证明原粉件身份。同一次Decisions增加冻结INITIAL三路原RGB和原来源编号标签，逐候选明确询问是否为指定canonical ID/原编号的同一原始粉件，要求依据可见形状、包装细节等特征；yes且confidence≥0.8才接受，无法区分则unknown并停止。没有新Responses、额外Decisions、采图、SAM或运动。



resolve_lifecycle_inventory核对current与reference的原ID/编号一致、原参考相机RGB及有效框；缺证据明确visual_lifecycle_reference_*失败。原图/编号不重写，既有别名不新增。已拆包及已完成黑件继续以已验证VLA/持物/运输/释放链保持同一canonical ID，只核验当前黑件角色，不拿旧粉色外观比较。wrapped identity_evidence.method为initial_reference_visual_same_physical_identity，附原观测、ID、编号、相机和框；black保持current_visual_role_and_verified_lifecycle。完整请求hash覆盖INITIAL/CURRENT图像、标签、容器与工序证据，原回答与测量提名分开归档。



初次完整性none≥0.8与generic路径原门保持；合法后续生命周期路径仍如实not_reassessed/frozen_initial_inventory，不伪造新盘点结果。全required当前几何及剩余粉件camera2可信覆盖不变。README、demo-workflow、demo-perception及observe_scene的SKILL/contract已同步；上面的重编号说明只保留为已废止历史记录。





固定ID修正验证补记：174项生命周期/原始身份/容器/呈现/program/观测契约测试与Ruff通过。真实4c2d归档原始粉件编号确为#1/#2且current一致；离线生产请求完整保留原九张CURRENT JPEG哈希及顺序，加入六张INITIAL原图/标签图，当前候选scope/点及mask哈希、容器对应逐项等于原归档。唯一随后授权真实请求hash 81a73b6bb1ce80a85d6994b63e0ba3ddfd7e492f1bd4427f5312bfe644608613：C5同原始粉件#2为yes/.85，固定身份门通过；黑件C2/C3为yes/.63/.67，C8 unknown/.33，生产parser仍visual_lifecycle_role_unconfirmed，未将低置信提升。证据data/4c2d-identity-diagnostic/含15图、每图hash、实际请求/回答、parser结果和独占锁。未重试/调参/运动，原4c2d终态与台账保持；该探针不代表完整关联或两件任务成功。



### 2026-10-21 全局身份审计与项目规则核对



AGENTS.md明确记录用户最新的固定ID要求：初始canonical ID和显示编号全程保留，剩余粉件不能按数量重新赋ID，拆包黑件继承原ID。仅用于离线诊断的data/audit_unpack_runtime.py新增阶段、动作链、显示编号及原始参考证据核对，不进入控制器或模型输入。b71b历史记录128项ID/编号绑定一致，但旧role-only粉件证据如实标记缺少原始身份核验；11项内存变异探针与Ruff通过。该审计不凭UUID一致或仿真真值宣称真实实物身份已被视觉确认，skill行为无变化。





### 2026-10-21 当前角色与原粉件身份分成两份同帧判断



混合INITIAL/CURRENT的4c2d探针中原粉件#2 yes/.85通过，但黑件当前角色.63/.67未过，生产正确停止；不能认定模型降低置信度的内部原因，也不能因此放宽固定ID或0.8门。现associate_inventory先使用原CURRENT-only角色请求，包含三原RGB/中性C注释/原像素细节图，不给黑件判断发送历史粉图。角色only回答只作暂存候选，不建立剩余粉件身份。首次完整性和后续冻结清单范围不变。



存在wrapped_source时，仅已通过角色的粉候选进入第二份verify_wrapped_identity：同一capture的已有CURRENT呈现及INITIAL原图/原编号标签，逐题明确比较指定canonical ID/原编号的同一实物；unique role/count/location不能代替视觉特征。只有第二份yes且confidence≥0.8、每个剩余源ID在camera2通过，才发布全部groups；否则visual_lifecycle_original_identity_unconfirmed且无部分几何。只有黑件时仅第一份；角色失败可提前停止。正常合法剩余粉件路径最多两次Decisions，不新增采图、SAM、地图或Responses，不重试。



两份均用VisualAPI.request及同一个audit、独立hash缓存；第二份context包含role_request_hash，完整像素/参考/标签参与hash。原role hash继续作为顶层evidence、coverage、lifecycle_correspondence和冻结清单检查的主hash；wrapped_identity_verification独立保存strict hash、role hash、当前observation/generation、初始reference、真实答案与accepted候选/ID/左腕ID。粉件identity_evidence.request_hash指strict请求，另有role_request_hash；coverage/lifecycle附wrapped_identity_request_hash。缓存角色成功不能单独发布粉件，未知第二份也缓存为失败。黑件method仍current_visual_role_and_verified_lifecycle，且completed投放保持最新两题契约。



离线使用真实4c2d event175/177与原SAM恢复生产参数，第一份整个wire hash精确等于原05fd78d3fe601d9bf9f00ce90f0afa5fa0f62ea400f3ebbf490cb7272b3bd95c后，才回放原归档role回答以准备第二份；这不是新模型回答。新strict-only hash为e415badbc56e5f8d878c632dd735f38a32925e8fc5b9e4a071bee61de2a67b0a，保留15图，只问identity_5/#2，准备证据在data/4c2d-identity-diagnostic/split-prepared/。此段记录时无新API调用，不改旧run/台账。





分拆验证补记：180项相关生命周期/身份/容器/呈现/program/观测契约测试与Ruff通过。根代理审阅后只授权strict hash e415badbc56e5f8d878c632dd735f38a32925e8fc5b9e4a071bee61de2a67b0a一次真实API；第一份原role回复仅在整个wire hash等于旧05fd78d3...后本地回放，并明确来源original_archived_4c2d_response。新strict identity_5返回yes、confidence=.74（yes probability=.83不能替代confidence），生产正确visual_lifecycle_original_identity_unconfirmed，不发布groups/coverage。真实请求逐字等于审阅prepared，HTTP200、API计数1、独占锁及当前源码SHA匹配均核对。证据data/4c2d-identity-diagnostic/split-report.md、split-prepared/、split-actual/。未重复调用、调参、降低阈值、采图/建图/运动或改旧run/ledger；分拆探针未解决原身份置信不足，不能算完整两件验收成功。





9522只读身份诊断补记：data/9522-continuity-diagnostic/inspect_continuity.py仅恢复实际RGBD/SAM和已记录skill目标，保存facts及原图；Ruff通过，skill行为无变化。已查看原camera2初始/当前像素，#2盒内原位置的SAM框完全相同、mask IoU=.994559、共同深度P95差.889mm、内圈实测中心差.244mm；所有实际grasp/place/mock记录均指#1。提出把视觉、注册空间连续性与实际操作历史合并呈现给固定ID判断，不要求独有外观标记；此处仅建议，未改生产prompt/阈值或调用模型。不能把几何相似自动赋ID，原strict .63和旧run失败保留。





### 2026-10-21 固定身份判断允许组合连续性证据



用户要求固定canonical ID/初始编号，不要求仅凭独有外观印记认人。原strict提示把外观强调得过窄，9522中当前角色.97/.97、粉件.86通过而strict .63停止。现仅修第二份verify_wrapped_identity的语义与证据呈现：允许INITIAL/CURRENT外观、已注册相机与原/当前框空间连续性、实际逐ID任务历史共同支持持续关联；不要求独有外观标记。原CURRENT-only role wire、两次调用边界、.8、choices/parser、unknown、不发布部分几何和固定ID均不变。



每个粉候选提供与全部初始source IDs的同相机RGB参考框IoU，参考框明确是Responses RGB注释而非初始SAM测量；跨相机不计算IoU。相机比较仅从已匹配容器的实际初始注册anchor和本次capture取得标定、内参及位姿差，缺失则unavailable，不凭空补0。记录历史只白名单phase/appearance/revision及同owner/generation/target的mock/drop回执基础字段；不发送GT、物理模拟ID或命令位姿。无记录操作不保证没有物理移动；几何相似、剩余数量或粉色存在不能自动赋ID，竞争对应、矛盾运动和遮挡歧义仍unknown。诊断初始SAM mask/cloud统计没有加入运行时，不新增跟踪框架或数值接受阈值。两份证据和新增提示随请求hash归档。





组合连续性修正核对：184项身份/生命周期/容器/呈现/program/观测契约测试与Ruff通过。冻结9522生产离线准备的首份角色hash仍逐字为fac256f97344ef375a4f381c2ff702f1329bdccc07dc81791ff34e52a15ae91c，核对后才本地回放其原归档答案。第二份新hash7b3a7520f1c88091aeeac08563b6c4f31bd27302e678430376e94efe6e56ae34仅identity_5题，15张JPEG及顺序与原9522 strict全相同；上下文只增加实际可用注册框/视角和净化历史，具体源hash/框/动作摘要见data/9522-continuity-diagnostic/prepared/manifest.json。此条记录时没有新API调用，未改变旧run/台账。





根代理随后审阅并授权上述9522 strict hash 7b3a7520...唯一真实请求，identity_5同原始#2=yes、confidence=.89，通过不变.8门；实际wire逐字匹配prepared、HTTP200、源码SHA一致、独占锁/API计数1均核对。第一份角色仍仅精确hash后的原归档回放，来源明确；生产parser获得4个required ID完整覆盖、左腕#2覆盖、missing_geometry_ids=[]。actual请求/回答、每ID双hash来源、原角色历史证据、parser结果位于data/9522-continuity-diagnostic/actual/，15原图见prepared/。无重试、采图、SAM、地图、运动或旧run/ledger改写；仅冻结关联通过，不代表新两件完整物理验收。





### 2026-10-21 自有夹爪掌部原语完整覆盖核对



9d005第二件失败的有界离线重放确认：首次碰撞是自有URDF的gripL_Base盒体与源盒X负侧壁顶沿，接触先于关节跟踪偏离；不是GraspGen坐标或单次深入1cm绑定错误。该接触点在旧四球原始并集外31.341mm，加原10mm世界球buffer仍在外21.341mm。现仅将gripL_Base/gripR_Base改为每侧12个2×2×3矩形cell外接球，raw半径34.011536mm；每球包含其cell八顶点且cells完整铺满原.076×.092×.098m盒体，因此覆盖全部体积/表面，不仅抽样点。生成器scripts/cover_gripper_base_boxes.py及resources/curobo/gripper-base-coverage-20261021.json保留解析证书、URDF/配置SHA和旧预算来源。其余机器人球、buffer、普通self pairs、同手9links touch豁免完全不变；attachment仍默认24、上限31，世界5mm阈值不变。



机器人球总数从349到365。350是旧项目设计预算，不是用户的attachment<32要求，也不是已安装cuRobo限制；没有为守旧预算改用更宽的4球包络。锁定commit 8e734f3ced1df898990bcd92de40abce475907db的公开KinematicsCfg.from_data_dict / Kinematics.compute_kinematics已在独立进程读取新配置并返回有限[1,1,365,4]。官方示例examples/getting_started/build_robot_model.py的相同公开FK调用及loader动态球数实现已核对，确切安装文件SHA见data/9d005-grasp-failure/sdk-365-spheres-probe.json；SDK/vendor、live地图/服务、机器人状态未修改。



原配置先红：两侧完整体积、首次接触点及新预算/复现共6项失败；替换后新覆盖与原Link4/robot配置25项、touch/open-sweep/prediction/perception-mapping53项，共78项聚焦通过。该记录不声称新运动验收通过：同token/obs/gen的源盒641个完整原生2cm cells在接触点仍缺少墙面，最近占据cell AABB距离77.021mm，原云最近85.275mm。掌部修复无法替代环境补全；独立缺失墙证据在native-wall-contact-coverage.json。技能入口、GraspGen/IK/plan语义、5mm和固定姿态无变化，机器人碰撞形状按实际自有原语纠正；部署前须刷新基于旧配置SHA的模型/工作空间缓存。





### 2026-10-21 运动中位置跟踪守门



9d005第二次粉件抓取在入盒过程中夹爪底座先碰盒壁，实际关节随后偏离命令路径；终点到位不能证明整段已检查路径被正确执行。Simulation现在每个物理步推进轨迹前，将实际关节角与上一拍实际施加的position ctrl比较，沿用原终点跟踪界限0.03 rad；越界时所有活动臂命令失败，并将位置参考保持为当时实测关节角，不修改qpos/qvel，不继续闭爪或lift。成功回执保留全程max_tracking_error_rad，失败保存触发臂/关节/时刻/误差及上一拍参考。该界限是执行路径跟踪检查，不是桌面目标位置精度要求；5mm碰撞阈值、末端区域验收、位置伺服增益和力上限不变。冻结同命令自由运动误差仅0.000484 rad，有碰撞1.264 s首次超过0.03 rad，不能以此守门代替地图与机器人碰撞几何修复。



运动跟踪证据补记：所有停止臂都纳入触发拍的实际误差更新峰值，TrajectoryExecutionError通过既有details归档接口保存失败参考/实测/时刻，避免调用者只得到错误字符串。29项运动/位置模式/带载保持测试与19项守门/runner失败/异步安全测试通过；未改变规划精度与动作顺序。



回放显示核对：实际9d005归档页面无JS错误/手机横向溢出，模型输入17图、SAM3掩码及GraspGen实际3D可查看。目标清单移除由物体编号拼出的“区N”，保留实际region_id，避免把固定身份编号错误显示为桌面分区；补齐粉色中文显示。skill行为无变化，原始台账/图像/区域定义不变。





运动守门物理回归补记：仅在本地隔离CPU重放9d005归档的同122点grasp命令，直接调用新生产Simulation._update_trajectories/_apply_motor_controls，未启动live服务、相机、模型或真实运动。原接触在1.264s以Joint7误差.030391772rad触发失败，随后2.736s保持同一实测位置参考，最大保持偏差.06168°；逐步核对控制方法未写qpos/qvel、失败后参考及夹爪ctrl不变。无机器人场景接触对照正常完成，峰跟踪误差.000484088rad；单/双臂失败传播与上一拍参考4项单测通过。公开execute_trajectory/execute_dual_trajectory抛出携带details的TrajectoryExecutionError，Runner.execution_stage重新抛出，后续close/lift不执行。重放只提交grasp阶段，此处不把未提交的后续动作冒称已做整条任务验收。残余接触/惯性仍令粉件最大移10.08mm；4s末关节最大速度.04236rad/s，不能声称瞬间停住或场景未变。证据data/9d005-empty-grasp-audit/guard-offline-replay-summary.json及guard-*-trace.jsonl.gz，命令replay_dynamics.py --production-guard；原失败归档保持不变，守门不能替代机器人包络与源盒墙面补全修复。



盒壁补全显示：当前碰撞预览读取已建图快照cuboid_layers，以蓝色透明薄壳显示测量平面与已知盒子尺寸推导的四壁/底，不填充空腔，不修改或替换实测2cm体素。每片显示instance/观测/代次及来源，过期、未标记或无效几何拒绝显示；展开预览不调用相机或重建地图。原始RGBD模式保持不变，旧无cuboid_layers记录兼容。



9d005身份运行审计补记：data/audit_unpack_runtime.py schema4逐项核对CURRENT角色与strict INITIAL/CURRENT双hash、同观测/代次/候选及原始显示编号；黑件核对mock→unpack视觉和drop→drop视觉的同ID链。实际159项身份/阶段记录检查通过，两次原粉件身份confidence=1.0；13种篡改与基线共14项探针通过。此为记录与证据绑定一致性审计，不冒称独立物理身份真值证明；原实验仍仅完成1/2件、needs_attention，物理碰撞失败保留。诊断工具不修改运行策略，skill行为无变化。



新365球配置重新渲染/核对指定观测角，1–4件初始场景均可见且端点碰撞检查通过，源指纹已更新；29项场景/球覆盖/跟踪守门测试通过。回放前端305项测试及Vite构建通过（保留既有大chunk提示）；薄壳预览另有20项测试通过，完整前端将在该增量后再核对。





## 2026-10-21 蓝盒遮挡薄壳与完整占据边界



固定清单源/目标蓝盒的30×40×20cm用户外尺寸只在观测位姿建图使用。demo_bin_geometry在mapper机器人剔除及现有surface filter后的当前注册点上拟合，不使用原SAM高点、仿真位置或旧盒云。最多8个RANSAC平面、每面256假设/8000样本，2mm内点阈值；每面至少80样本内点和8mm第二方向spread，四壁各至少30mm竖直跨度。实测底/顶和两组正交成对内壁须在3°内直立/平行/正交；支持世界Z任意旋转和平移。内腔与用户外尺寸推得3–25mm等厚壁/底，三厚度相差不超过3mm；至少85%清洁点在有界模型3mm内。缺面、歧义、倾斜或尺寸/污染不符以demo_bin_shell_*停止，不能猜位姿或填满盒AABB。



碰撞新增semantic_instance::<id>::bin_shell::{x_min,x_max,y_min,y_max,floor}五片独立cuboid，保留空腔，原测量点与2cm native TSDF/ESDF不变。mapping_diagnostics.bin_shell_completions保留当前ID/observation/generation、实测点hash/数目、平面数、尺寸先验及拟合厚度；source=measured_planes_and_user_bin_dimensions且synthetic_geometry_is_measured=false。preview.cuboid_layers与voxel_layers分开。build_scene契约、README与demo-perception同步，无新模型/采图/位姿外建图。



源离箱高度与right_bin_occupied_bounds三个运输/投放定位/动作包络消费点均使用同ID启用native完整单元及全部五片薄壳的边界并集，scene token/观测/代次一致；缺面、错ID、过期、禁用或来源不明不允许回退。原10cm运输余量、2cm投放包络余量、5mm碰撞门、ID与视觉阈值不变。仅历史未声明薄壳的地图继续解析native-only边界。此处替代旧手册当前段落的“只读native单层”限制；历史运行诊断保留其当时来源。



冻结9d005的初始与当前可信源盒点云并集仍缺首碰处85.275mm；拟合补壁覆盖该离线见证而空腔中心不占据。完整源/目标云CPU拟合得到约12mm厚/0.600m顶，125项当前聚焦回归通过（含旋转/污染/缺壁/错误尺寸/版本与消费边界），不代表新GPU或物理运行验收；无模型或机器人调用。



配置来源核对：全套测试发现algorithm-provenance.json仍声明349球旧hash，已保留previous_revision历史并更新当前365球/hash/解析Base覆盖证书。修复的是来源元数据，未修改碰撞显示测试或运行规则；skill行为无变化。



归档下载核对：data/download_unpack_audit.py只读取已终止实验的原始事件JSON及其引用NPZ，保留event编号、原字节与SHA，供CPU审计完整数组使用；校验实验ID、事件序列、manifest大小及归档路径，不调用模型、地图或运动。skill行为无变化。



下载范围补记：包括verify_release原始输入，允许独立核对拆包前已确认桌面放置的真实capture来源；仍只读原归档，不合成相机图像或替换失败结果，skill行为无变化。



新球模型、测量薄壳及工作空间更新后，完整Python回归2283 passed、54 skipped（193.56s）；完整前端307项及Vite构建通过。Windows→Ubuntu同步逐字SHA验证718个受管理文件、文件集合和21个skill契约，四个服务健康。上述是代码/部署验证，完整两件运动验收另列实际实验结果，不能用测试通过代替搬运成功。



外部请求诊断：两次初始Responses在约60秒报transport_error，均未建图或移动。data/probe_initial_responses_timeout.py仅对首份归档请求核对完整hash后发出一次诊断调用，连接15秒/读取180秒，保存异常类型或脱敏回答；独占started文件防止重复，结果不发布清单、不修改原实验。此处未改变生产超时或skill行为。



该有界诊断在70.33秒收到HTTP200，证明60秒不足以覆盖此次初始Responses等待。compose现显式传入已有LUNA_DECISIONS_TIMEOUT_S，未设置时使用与.env.example一致的90秒，仍受客户端1–180秒有效范围约束；此前只挂载.env并不会自动将该值传入进程。只改变网络等待预算，无自动重试、额外模型调用、身份门或运动行为变化。原两次失败保持不变，诊断回答不用于实际任务。



部署核对：实际容器构造的DemoVisualDiscovery与DemoVisualInventory均读得90秒，32项视觉API/配置相关测试通过。79639实验初次Responses成功，完成首件抓取/桌面放置/mock拆包/返回观测，随后same_item=.77按原.8规则停在unknown；保留原失败。随后跨视图语义和薄壳投放定位修正后的完整Python回归为2299 passed、54 skipped（214.75s），不能将此统计视为两件运动成功。





365球工作空间再生成核对：按原1cm网格、batch1024、IK16 seeds/seed solver32、原公开SDK完整重建A/B两臂，每臂checked_points=3,048,500、complete=true；可达格/七关节见证数A=2,777,891、B=2,777,814。新model_fingerprint=f7c0bb5c2fd57e6c578e1df4eba626ad34b9461a5f38df472257210ce9a87bce。已有容器项目挂载只读，计算输出写现有可写cache，再完整保存到项目data/workspace-sphere365；未改挂载、SDK、服务或运行状态。旧NPZ在data内原样备份，新两NPZ经过输入SHA/指纹、完整进度/退出码、格点与见证一一对应及关节有限/范围核对后，按原始字节替换本地resources/workspaces/arm_A.npz、arm_B.npz；正式WorkspaceAtlas读取及8项offline workspace测试通过。日志、metadata、witness、哈希和命令在data/workspace-sphere365/，GPU任务已结束释放。skill行为无变化：仍只是固定机器人自碰撞下的有限位置预筛，不证明朝向、环境/另一臂或整条路径可行；未伪改旧指纹或使用部分构建。



2026-10-21 投放定位薄壳兼容核对：实际 right_bin_occupied_bounds 已返回 exact_inventory_native_cells_and_derived_shell_union，但 _drop_bin_zooms 旧入口只接受历史 native-only 来源，真实 producer→visual consumer 回归先复现拒绝。现仅扩展该入口：native 层加 exact 五片同盒组件必须完整，逐片校验当前 observation/generation、启用状态、published_cuboid_bounds/measured_planes_and_user_bin_dimensions 来源、synthetic_geometry_is_measured=false、有限范围及被并集包含；共享 scene_token 保留在已验证侧栏头，不能把缺面或残留合成组件降格成历史 native-only。原六RGB、BEFORE/CURRENT裁图标定、双题与0.8门、无新地图规则不变；verify_unpacking 不在此修改范围。34项裁图测试通过（含14种拒绝边界），无模型/GPU/运动或部署调用；历史实验结论不改。





## 2026-10-21 拆包核验的同次多视图对应



实际79639在第一件放置、mock及返回观测后，black_core=yes/.91、same_item=yes/.77而unknown停止。查看4张原始输入并核对SHA：head有明确黑件，zoom重复该head像素，右腕最左缘有裁切黑形。新context/题目明确三相机同capture、重复图像不代表新增物体、部分边缘本身不证明第二件；一个清楚视图可结合已核验桌面释放和活动拆包链确认对应，只有明确不同可见黑色桌面工件才竞争。实际对应歧义仍unknown，不虚构跨视图匹配、不证明隐藏区域为空、不用mock动作证明可见。same_item未删除，固定ID、black_core题、两题choices、0.8 confidence门与parser均保持。



74项生命周期focused回归与Ruff通过，原79639实际回答加入fixture：choice probability .85不能替代confidence .77，原phase/ID不推进。data/79639-unpack-diagnostic/prepared保存从真实event167输入调用当前生产函数离线生成的request/manifest；原hash先完整复核，新4个image data URL逐字不变，只改context三字段和same_item文字。准备阶段0次外部API/GPU/运动，不能当作新模型探针成功；原run保持失败。





### 2026-10-21 拆包核验补充真实桌面释放BEFORE



仅CURRENT四图的冻结79639澄清探针仍same_item=yes/.53而unknown，历史run与答案不改。现复用真实成功verify_place capture并增加严格绑定，提供BEFORE三原RGB+原CURRENT三RGB+原zoom。新的可见前后变化问题仍须两题confidence≥.8，不把mock记录当视觉证明，不改ID/数量/几何。私有capture复用、跨target/帧/代次/owner/绑定/hash/像素变化与缺reference拒绝均有回归；离线准备不代表模型或运动验收通过。





### 2026-10-21 拆包可见结果与程序身份继承分责



CURRENT-only澄清探针仍same_item=.53；追加真实BEFORE的单次探针898dbf...返回black_core yes/.79、same_item unknown/.16，均按原门停止，原run未改变。此后职责修正：程序检查单active释放/拆包链并继承ID，模型只逐相机判定当前黑色桌面结果和可见竞争；保留BEFORE严格校验与归档，模型输入恢复原CURRENT三图及zoom，不发送历史三图或reference hash明细。六题与0.8门、同相机支持及任意可信竞争否决明确；旧same_item回答不升级为新成功。新增回归涵盖单视图支持、其他视图未知、真实冲突、同相机门、single-active及消费者重算。未作新模型/部署/运动；后续新CAD及独立黑entity ID需求另行迁移，不由这次兼容补丁宣称完成。



2026-10-21 CAD验收工具核对：独立scripts/audit_unpack_run.py读取实际仿真CAD姿态与原始mesh顶点，在真实内底截面多边形中检查右盒包含关系，保留内圆角；该保守截面来自CAD内底+1µm截面，离线包含检查2mm数值容差不进入10mm运行碰撞门。完成按工序及粉件桌面、拆包创建、黑件投放三份观测证据计数，拒绝重复产物；自定义颜色不改变CAD类型判定。trimesh4.12.2公开load_mesh/vertices接口已核对，官方安装包未改。只更新独立验收与回放QA，不增加运行时仿真真值读取。



2026-10-21 离线运行审计schema6：完成阶段按process的input与output两条ledger合并核对[0,1]，不再要求黑件继承原粉件entity ID。历史无process记录保持旧审计分支；原始证据、身份链、模型请求与CAD边界另行逐项核验。此调整不改变运行skill行为。



2026-10-21 完整lint核对：整理两份既有测试文件中的三处导入顺序，测试内容和skill行为无变化。完整后端回归包含这两份测试。



2026-10-21 合同一致性核对：mock_unpack的SKILL与contract已移除旧6×4×3cm及粉黑共用ID说明，明确CAD原始44/56/74.5mm和确认后创建产物。docs/demo-workflow.md同步当前工序、CAD与10mm规则；旧历史schema不作为新任务验收。



2026-10-21 CAD编辑表单回归：mock只改变manifest_revision而不改变generation；未编辑的表单现在随几何revision变化刷新CAD类型，并在running结束时同步最终位置，防止应用编辑时把黑件恢复为旧粉件。已有未应用编辑保留。真实main.js入口回归先复现旧pink_wrapped滞留，再修复通过；仅编辑视图同步，skill行为无变化。



2026-10-21 新工序审计：process_records运行使用独立CPU audit_process_identity，核对input/process/product边界、实际请求hash/原始答案、前后释放引用及两段execute_grasp ID；缺完整原始数组或答案记missing/unchecked，不据UUID或mock回执声明实物身份，旧同ID历史分支保留。23项正负例通过。CAD映射审计另有12项完整mesh/SHA/版本及历史分支探针通过。Ubuntu已同步806项SHA及完整文件集合，运行设置通过API显式保存10mm；当前新CAD初始settle尚未通过，不能声明物理任务成功。



2026-10-21 CAD静止修复：MuJoCo3.10默认nativeCCD在粉件不平底片与盒底接触中产生持续约0.039rad/s角速度，100s也不满足原静止门；增加CCD精度或关闭NoSlip均未解决。仅含CAD场景启用官方libccd路径（mjDSBL_NATIVECCD），保留multiCCD、原CAD/质量、接触与位置伺服参数以及0.015速度/0.25s判据。100s对照中粉件与黑件桌面/盒内落体均连续95s达标；未清零速度、冻结物体或放宽静止阈值。原primitive场景仍用默认nativeCCD。新增真实动力学与夹爪负载/位置保持回归，黑件只验盒腔包含和稳定，无额外中心精度门。该修改影响仿真接触求解，不改变skill成功证据或10mm规划碰撞门。



2026-10-21 CAD源起抬边界修复：真实dafba328已发布完整98个mesh，但lift旧voxel/cuboid白名单再次拒绝已验证的源盒cad_part_00。现仅对published_bin_shell_layers完成同ID/scene/obs/gen、完整49片、目录/逐片SHA与启用校验的盒壳mesh接纳其全边界；普通源/邻件层、height_completion任意mesh仍拒绝，桌面cuboid及历史五片合同不变。真实257个IK候选CPU红绿回归，统一lift0.203098593m，最小包络净高10mm；59聚焦测试及Ruff通过。几何、阈值、SDK及场景未改变，不代表后续轨迹或运动验收通过。



2026-10-21 当前验证与归档核对：CAD接触修正后的完整后端2404 passed、54 skipped，前端315项及构建通过。归档下载器按实际API的recording_complete和busy字段检查可读终态，保留原始JSON/NPZ字节及SHA；读取连接被重置时仅重试只读下载，未重复模型或动作。dafba328的20份完整原始文件共150705828字节已下载；原失败保持不变。手册当前入口统一为CAD、新产物ID、按工序计数、10mm及随件数增长的预算，历史记录仍保留原版本语义。



2026-10-21 归档读取语义更正：recording_complete同时表示视频采样质量，不是原始事件文件可读性。d4a85已终态且busy=false，但因2.87s场景采样间隙标记partial。下载器现在要求实验ID/终态一致、无编码或写入活动，再逐文件大小/SHA核对；原recording_complete、video_status及errors一并写入下载manifest，不把部分视频升级为完整。只读审计仍可检查该实验已经写完的原始事件与数组，skill行为无变化。



2026-10-21 单件集合视觉修正：d4a85首件完整工序成功后，第二件前的角色问题把右盒当前单个黑件描述成“a collection, not one object”，与允许单件的集合数据结构矛盾。现仅将completed_collection与产物集合描述改为右盒内一个或多个可见黑色工件（或其可见部分）；members仍保存独立产物ID，图像不负责证明历史completed状态或必须有多个实体。原候选、身份、图片、覆盖门和confidence≥0.8不变；102项相关回归、27项合同测试通过。旧失败回答不升级为成功；同图文字替换探针另记录来源和结果。



2026-10-21 CPU资源限制：按用户要求，四个demo容器CPU配额合计4逻辑核：web 2、cuRobo 1、SAM3/GraspGen各0.5，已对现有容器生效并写入Compose。各进程OMP/MKL/OpenBLAS线程上限1，web软件渲染线程上限2；本地后续重检查串行、单线程，已在运行的审计进程绑定一个逻辑核并降为BelowNormal。仅限制调度资源，不改规划碰撞门、视觉成功条件或模拟物理参数；等待和编码可能更慢。



2026-10-21 用户平底粉CAD替换：pink_wrapped改用tripo_convert_891be992-2e91-4d83-b19a-dbbfcfa1acd1_flat_bottom_110mm.stl，源SHA200868327093f29d38856dd769ed11855a0cab7f26e1d8aa72207e6cf0298f3f。实际111.085350×62.787369×110mm，AABB中心[0,0,55]mm；底部368个共面三角，保留原视觉网格，八段凸碰撞网格底面平坦。仍是非闭合CAD，不据此推断材料密度。蓝盒/黑件61文件和两条资产记录逐字不变，旧pink.stl仅保留历史来源；asset选择参数支持只生成指定资产。默认XY/Z及观测关节不变，观测资源指纹更新。49项相关测试通过，100s实际物理静止与公开confirm成功，1–4件左腕可见、无初始碰撞。新件长轴111.1mm超过实际103.0mm开爪间距，短轴62.8mm可容纳；只说明方向相关必要条件，实际候选仍由原GraspGen/IK/规划筛选。当前旧CAD成品视频及hash mesh保留，旧CAD重新编码仍依赖当前catalog，未宣称该历史重建兼容。



2026-10-21 同图集合文字探针：完整重建原请求1c07d...c12e后，仅改context描述和C5/C11题干三处，10个原JPEG URL与顺序逐字不变。根代理执行唯一一次新请求b0c842...defb8，head C5=yes/.96通过不变.8门，right wrist C11=yes/.49保持不合格；原生产角色覆盖检查通过。该实验明确为文字替换探针，保留原证据摘要，未重跑完整感知、严格粉ID检查或发布几何，原d4a85仍为失败。实际请求/回答、SHA与独占锁在data/cad-v3-role-diagnostic/collection-actual/。



2026-10-21 平底CAD与提示修正完整回归：在一个逻辑核、数值库单线程设置下2420 passed、54 skipped（547.20s），Ruff通过。验收脚本状态轮询由3s改10s以减少大型状态序列化开销，不改变任务执行频率或证据。替换CAD前将Ubuntu当前45个内容hash显示mesh备份到data/render_meshes，Compose改为持久目录挂载，避免容器重建清空旧实验mesh URL；不改历史mesh字节或视频，skill行为无变化。



2026-10-21 清单与部署核对：docs/demo-workflow.md说明初始冻结快照、统一entity/process记录及派生清单；pending_unpack_ids仅含未开始工序，已拆包product_entities包含桌面/运输中/右盒产物，completed仅在投放核验后成立。再次处理同份创建证据复用产物ID，不据UUID证明新物体。第二件依赖单active工序、当前无竞争桌面黑件及此前右盒完成证据；后续观测要求右盒当前占据，投放要求至少N+1可辨认黑件。说明不改变skill行为。Ubuntu平底版本811项文件及完整集合一致；历史d4a85使用的30份内容hash mesh按原SHA恢复并验证旧URL可读。新平底e6ed7运行通过第一件粉件抓放与拆包确认，但剩余粉件camera2角色题yes/.51、.38、.52低于原.8门而停止；未发布新场景，未计完成。其视频因2.40s场景采样间隙仍标partial，不宣称全流程或完整录像验收通过。



2026-10-21 验收连接复核：validate_demo_run每次HTTP请求使用新连接，避免长settle/confirm阶段后复用失效连接；不自动重试start或其他写入请求，轮询仍为10秒。仅验证脚本连接设置变化，运行skill行为无变化。



2026-10-21 当前角色视角与候选语义澄清：e6ed7原10图、框及camera编号核对无错图或明显遮挡。唯一文字探针4d428f...5d9e仅追加context.rules，明确机器人左右/camera1右腕2左腕3头部，以及多描述C提议可重复指向同一工件，separate表示区别于机器人/容器/背景。原回答离线仍失败；一次新请求左腕C7/C8/C9=yes且confidence .99/.97/.99，原生产角色parser通过，未执行后续粉ID核验或几何发布。现将同样说明写入生产角色请求并更新observe_scene合同；题目、所有图片、候选和.8门不变。该单次结果支持澄清有帮助，不证明唯一因果；原失败和partial录像保留。





2026-10-21 用户观察顺序修订：粉件放置、VLA拆包都在动作完成的当前位置做结果Decisions；拆包成功才回观测位姿，用新帧做抓取前Decisions和三RGBD建图。两段抓取在plan_grasp入口复核实际双臂观测姿态及当前地图同帧绑定。结果帧verification_capture及私有RGB支持后续跨帧关联，禁止跳过结果核验先返回。普通拆包采图预算随之为1+5N。黑件独立ID是工序记录；自由落体后仅维护右盒集合占据，不以落点重派ID；完成仍需本次执行链和新增可见数量确认，遮挡不确定停止。

按机器人前方+X、左侧+Y核对，两CAD盒及小桌中心X=.315m，盒朝机器人侧外边与主桌近边同为X=.16m；左右盒Y仍±.605m，默认两粉XY=(.315,.525)/(.315,.685)，中心间距16cm，完整CAD投影距底部真实内腔轮廓至少74.687mm、相互AABB净距97.213mm。固定用户关节和外参不变，单核离线1–4件渲染均可见且初始接触为零，默认camera2有效像素2945/3598；仅为配置/固定姿态QA，不代表抓放可达或新布局动态稳定性。



2026-10-21 当前观察顺序与贴桌新布局验证：单核完整后端2443通过、54跳过（472.99秒），补跑最终工序返回修复44项通过；前端315项通过、Vite生产构建通过，Ruff通过。静止CAD与既有接触门随完整回归通过。上述不代替新的两物体真实模型/运动验收。



2026-10-21 右盒宽松投放修正：8a644e...第一件黑件附物运输失败，冻结诊断显示持物参考点精确对中额外外推TCP约8.6cm；改用TCP实测盒区中心后同R/Z/attachment/设置的普通plan_pose成功61点，尚未执行该诊断轨迹。目标推导记录xy_policy、完整点云加测量padding的最小向内余量；既有原始占据高度、姿态、10mm和quintic/time_scale不变。76项相关回归通过，完整两件运动验收待新实验。



2026-10-21 独立存档核对增加观测顺序检查：每次generate_grasps引用同帧Decisions和最新build_scene，二者间不得执行运动；verify_place/verify_unpack内部须记录对应当前位姿capture且不得运动或建图。新增宽松运输目标的TCP中心、完整点云padding投影重算。只读取存档，不调用模型/规划/仿真，skill运行行为无变化。v6已有两段抓取和两次当前位姿核验顺序通过，不代表两件完成。



2026-10-21 a05756...实际新布局通过第一件完整抓放/拆包/投放与第二件粉件抓放/当前位置拆包核验，第二件回观测后角色Decisions未确认，未执行第二件黑抓取。原图/SAM对应正确。三份冻结呈现实验保留全部原RGB：简化文本52f6a...覆盖可过(.80/1.0)，恢复cache证据hash并移除容器颜色限定15214f...桌面仍未过(.53/.43)，四区域去重题9f51...桌面亦未过；不能把第一次探针当稳定修复。实验源保存在data/cad-v7-role-diagnostic，生产角色实现与题目恢复到部署版本，门限与ID/几何规则无变化。右腕原图的桌面黑件被画面边缘裁切，另行离线验证模拟拆包输出位置的可见性，尚未改变运行位置。



2026-10-21：角色请求明确只有初始粉件来源清单冻结，当前entity/process记录还包含拆包确认后新建的独立黑件；不能把新产物说成初始粉件清单里的原实体。此处仅修正上下文说明，原问题/图片/候选/.8门与身份检查不变；三份简化/分组提示探针没有作为稳定修复部署。



2026-10-21 CPU核对：a05756连续视频后台仍在编码，已按完整命令及实验ID核对后只终止本任务encoder子进程1012；web及算法服务不受影响，原始RGB/掩码/点云/轨迹与请求均保留，视频标失败而不宣称完整。后续验证不再开启额外连续视频，逐节点实验回放保留。



### 2026-10-21 mock拆包输出可见性修正



默认黑色CAD输出中心改为 `[.36,-.24,桌面高度+.03825]` m，yaw仍相对粉件增加30°；粉件共享放置区域、双臂固定观测角、外参及10mm碰撞阈值不变。旧Y=-.14使黑件越出右腕图左边72.43px；离线构造场景中新位置完整CAD边距11.10px，采用v7实际自然VLA关节时头相机也可见。完整CAD距桌边至少125.85mm，中心及包围盒角在B离线位置atlas内；这些只是可见性和位置预筛，不能替代实时视觉、IK或运动验收。证据在data/cad-v7-visibility；新增默认输出多yaw相机投影回归，未改运行时模型判断。mock后仍在当前位置Decisions，再回观测位姿做抓取前Decisions与最新场景。这里只改变模拟动作，不把mock输出真值注入感知/清单/规划。



### 2026-10-21 放置接近默认值与v8失败定位



v8第一件粉件抓取成功后，固定释放目标的原生阶段通过，但world +Z 100mm接近段失败。相同scene/binding/start/gripper/R/释放XY/Z的单次持物IK对照：释放与当前lift成功，+100mm失败，+75/50mm成功；唯一50mm原生plan_place对照两阶段通过。默认place_approach_offset_m由.10改为.05m；显式保存设置仍优先，不做静默迁移，当前demo设置通过API单独调整并记录。10mm世界阈值、24球/最多31球、固定放置区域、抓取旋转、时间缩放和撤离检查不变。同步清理plan_place/SKILL.md中已过期的256球说明。诊断未执行动作，不代表两件端到端验收。证据data/v8-place-diagnostic。



同一冻结结果的五次插值/time_scale=3后approach/place硬检查均通过；原100mm空爪retreat失败。唯一50mm退离对照使用相同最终release joints、开爪、R/XY/world，原生plan_pose与其最终命令硬检查通过。因此place_retreat_offset_m默认也改为.05m，显式用户设置继续优先，10mm阈值不变。三段冻结检查通过仍不是实际放置成功；下一轮完整执行单独记录。



### 2026-10-21 拆包Decisions前后RGB与完整主桌取景



修复verify_unpacking已验证并保存BEFORE但未将图片送入模型的问题：实际请求现在包含拆包前桌面放置确认帧三RGB、拆包后当前位置三RGB，及当前头相机完整主桌辅助图。BEFORE与CURRENT明确分组，前后手臂可移动；参考图、重复视角及盒内已完成黑件不计作第二个桌面对象。保持原六题、单个相机result=yes/conflict=no且置信均≥.8、任一明确竞争即停止的规则。竞争题改为直接询问CURRENT主桌同时可见两个或更多黑件，不要求证明不可见区域为空，也不要求粉黑外观相同。



辅助图由最初RANSAC缓存demo_tables的primary_table_id/model XY范围及surface_height_m投影，核对目标主桌ID与初测来源，覆盖整个主桌而非仅粉件放置方框。缓存允许来自初次观测；当前相机外参来自本次capture；不读取模拟器物体真值，不调用SAM、不更新场景。保留三张CURRENT原图及裁剪坐标、来源哈希。v9失败证据data/cad-v9-unpack-diagnostic中旧crop截掉了移动后黑件的一部分，旧.91结果/.67竞争回答按现有门仍应停止；不能把新提示当作该旧运行成功。



冻结v9原始帧补BEFORE及完整桌面取景的单次模型探针仍未达到.8，失败证据保留。完整桌面辅助图进一步加初次RANSAC主桌投影黄色边界和图外图例，明确区分主桌与相邻盒子；不遮蔽原RGB或剔除盒中物体，三张CURRENT原图仍保留。元数据追加annotation/output_size_px，原crop/display_size/scale仍描述RGB部分。该图是已知支撑边界提示，不是当前物体mask或成功证据；没有修改判断门槛。



### 2026-10-21 自然VLA位姿视野修正



prepare_unpack双TCP的X目标由.36m改为.26m，Y仍±.18m，Z仍实测桌面+.30m，沿用观测帧的旋转。新位姿让手腕相机更靠近机器人，覆盖mock输出所在右桌区域；用户固定的双臂观测关节角不变。使用v9 mock后最新保留场景、当前空手反馈的单次双臂规划和最终10mm路径验证通过（data/v9-vla-x26-diagnostic），未执行、不重建场景。仅表明该冻结起点到目标可规划，仍需离线图像检查与完整新一轮执行。



### 2026-10-21 拆包请求原图离线审计（schema 8）



data/download_unpack_audit.py 串行保存原 Decisions 请求及引用 JPEG，清单记录文件字节数、SHA 和逐图匹配结果；data/audit_unpack_runtime.py 校验恰好三张 BEFORE TABLE RELEASE 与三张 CURRENT 的实际 JPEG 字节分别匹配 table_release_reference.images 和 verification_capture.images，并用所有原输入重建请求哈希。缺图、重复相机标签、错误绑定或字节不符不能通过当前协议审计；历史四图请求不会被升级为七图证据。回放 QA 新增每工序『VLA 拆包 / 核验』七张实际输入与图片解码检查，保留原候选点云及投放八图检查。skill 行为无变化；离线一致性通过不代表模型接受、服务端实际读取或机器人执行成功。



最终自然VLA目标采用X=.20m（Y/Z/R不变）：X=.26虽可规划但离线图发现夹爪遮挡，未部署；X=.20单次双臂规划及10mm路径检查通过。使用v9实际物体位姿的离线右腕图中黑件完整入框、图边余量133.32px，可见6860/投影6971像素（约98.4%），主体与夹爪分开；头视角仍允许局部遮挡。证据data/v9-vla-x20-diagnostic，尚不等于实际流程完成。同步prepare_unpack技能默认值，保持观测位姿、场景更新边界、mock输出与1cm阈值不变。前后参考图及缓存主桌投影的集成归档测试更新为实际七图，14项通过。





2026-10-21 拆包核验收敛为三视角综合unpack_result，替代额外逐相机六题和竞争计数门。保留0.8 answer confidence、真实前后三RGB、单活动工序与原始答案消费者复核。v10失败存档保留；data/v10-outcome三个独立冻结样本为第一件yes/.85、第二件yes/.86、未拆包负样本no/.85，均只调用视觉模型，不修改历史实验或仿真状态。这是样本验证，完整两件连续流程另行验收。data/probe_unpack_fusion.py及其对照结果归档；失败的两题融合方案未部署。



2026-10-21 存档核对同步新协议：audit_process_identity从真实Decisions存档重算unpack_result及其0.8置信，核对view_scope和工序创建证据；旧逐相机回答不被升级为新协议成功。仅更新离线核对与负例，运行时skill行为无额外变化。





2026-10-21 抓取前职责分离：lifecycle_candidate_scopes继续用同帧三RGBD点云、缓存桌面与已验证盒体检查3D支撑范围，Decisions的identity_N题只确认C候选可见的粉/黑工件外观并排除机器人、盒和背景，不重复要求RGB证明桌面接触或所在盒。保留0.8置信、未知不进入地图、源件左腕覆盖、后续INITIAL/CURRENT粉件原ID核验及全部工序绑定。completed_collection仍可为一个或多个可见黑件/局部，不分配视觉成员ID。v11原图对照：外观题10/10 yes/confidence1；交换要求颜色的负例10/10 no/confidence1。相关请求、原始JPEG及响应存于data/v11-role-diagnostic，历史v11失败状态保持。中止无关全量测试并保留部分日志，不将其声明为完整通过。



默认值测试核对：旧测试仍断言place接近/撤离100mm，现同步已部署的50mm默认值；显式高级设置继续保留，未改变运行时。





### 2026-10-21 v12投放核验误停定位



run `f5158f7382344acfb846b4b33b5e0565` 第一件在右盒投放后停于 `verify_drop`：337为实际投放，340为新观测，343为Decisions原始回答，344为解析失败。CURRENT右腕及头部原图可见盒内黑件；8张实际输入JPEG逐一SHA匹配，重建请求hash为 `f23d5ad8a8d1e998f52eb6071f4523d489c6edd3d2833e6b934b72d97bcec783`。capture晚于drop反馈，排除取错旧帧。原回答in_right_bin=yes/.79，current_item_added（同一盒内至少1件）=yes/1.0。旧代码对重复存在条件也设独立0.8门，导致已确认盒内数量仍被拒绝。



`drop_result_evidence`统一解析当前协议：必要条件是同一盒内N+1可辨别数量题yes且confidence≥0.8，in_right_bin高置信no为反证；低置信或unknown原样保留，不再重复AND。后续清单消费者`_visual_drop`及离线audit_process_identity均从原始answers重算并核对全部语义字段，旧unknown结果、缺字段、伪造confidence或数量不足不能升级为成功。问题、原图、0.8门、执行链、1cm碰撞阈值、场景更新边界均无变化。第一件的原始失败实验仍保持needs_attention。



原始响应回归先复现失败再通过；证据与逐图哈希在data/v12-drop-diagnostic，read-only下载脚本为data/diagnose_v12_drop.py。此次只使用已有存档和离线测试，未新调用视觉模型、GPU规划、仿真运动或建图。此修正不等于完整两件流程已经验收。



本次验证：135项相关回归通过；随后新增消费者/审计负例及技能契约检查56项通过，Ruff通过。覆盖原yes/.79与yes/1.00回答、明确反证、数量低置信、缺协议字段、旧unknown结果拒绝，以及初始粉件固定ID/独立产物工序链。未重复运行无关全量测试。



### 2026-10-21 三视角单请求与候选预算验证



当前协议见[Decisions问题清单](decisions-questions.md)。拆包主线的当前角色和原粉件身份合并为一次请求，角色题identity_N与原身份题wrapped_identity_N分别校验，归档同一request hash。place和drop分别使用单个融合结果题，旧逐相机/重复题答案不转换为新成功。GraspGen池采用256→512→1000、最多1768预算候选/19次服务调用，找到可执行计划即停止；原始返回数量单独记录。离线工序审计同时支持明确标记的旧双请求和新单请求证据，不能把两种记录混接。



最终相关回归：视觉/身份/结果消费者与离线审计287项通过，候选池/规划/相关契约221项通过；更改的运行时代码Ruff通过。本地检查串行、单逻辑核，Ubuntu四容器总CPU配额仍为4核。这些测试不表示真实全流程通过。



真实模型对照保留在data/fused-probes：v13原始放置帧发送生产单题后仍为place_result=no/.97；动作前负例no/1.00。v12投放帧使用单个同盒数量题得到yes/.99，动作前负例no/.89。分条件诊断把主要疑虑选为outside_target，但其置信度仅.59，不能把这一解释当作已证明的根因。简化措辞、区域放大、存在性措辞的独立对照也未让放置正例得到合格yes；所有请求/原图/原始回答及hash保留。无效果的区域放大未加入最终运行时；其余提示变体仅为离线实验。投放探针对应同一单题协议；最终上下文又将单数黑件措辞清理为复数，未把该文字变化伪称重新完成全流程验收。



因此当前已知阻塞仍是verify_place的真实视觉误判；不降低.8门、不使用MuJoCo真值替模型回答，不改写v13失败状态，也未启动已知会受此问题影响的新完整两件实验。模型识别/区域定位问题仍未解决，不能宣称demo全流程验收完成。



### 2026-10-22 放置输入与旧流程边界核对



拆包放置核验实际发送初始全部实体清单、活动目标ID、选定输入、已确认生命周期和工序记录，以及INITIAL三原图/注释和CURRENT三原图/区域注释；先历史后当前，执行臂腕部当前视角最后呈现，仍融合全部三视角。初始描述和last_confirmed阶段仅为历史，活动目标正在等待桌面释放核验，不能把旧wrapped_source当作仍在盒内；动作记录不证明当前成功。只发送白名单语义字段，不发送仿真真值、命令位姿或完整执行回执。



DemoVisualInventory对声明unpack_transfer却缺失/空/缺源ID的生命周期直接报unpack_lifecycle_required，禁止转入旧完成槽追踪；声明transfer却携带拆包生命周期报unpack_workflow_lifecycle_conflict。旧非拆包兼容模块仍保留，不能声称全仓库已删除旧代码。



按左右臂实际工具坐标转换（Z方向一致，X/Y相反），先按向下45度和远离机器人规则剔除候选，仅对通过者沿工具+Z平移10mm一次，再交给batched IK。被剔除者保留原始位姿，insertion_applied=false；operation_order记录筛选、插入、IK的顺序。两种顺序对接受位姿的方向结果本来相同（平移不改变旋转），此次明确按用户指定顺序实现，并避免平移已拒绝位姿。官方GraspGen源码和200/top100推理参数无变化，预算256→512→1000不变。



231项相关测试通过。真实冻结帧核对：旧生产请求只有CURRENT三原图/三注释，没有初始清单、目标ID和参考图。补齐输入后正例no/.67；只重排相同内容为历史在前、CURRENT在后且执行腕最后，正例yes/.52，负例no/.97。生产payload哈希与该实际调用一致，生产消费者仍将正例视为unknown（.8门未变）。输入缺失和排列敏感已有证据，但不能声称剩余视觉歧义已解决或全流程通过。更宽泛的搬运结果题仅做离线对照，未合入运行时；没有启动新完整运动实验。



2026-10-22 v14第二件verify_place定位：实际单题place_result概率0.69。原帧同次分条件诊断为桌面粉件可见0.99、仍夹持0.00、底部物理接触可见0.64。当前生产题删除resting，采用同次请求的两个独立predicate及原0.8/0.2门；数值释放XY检查、三原RGB、历史清单、动作绑定和不建图保持。新增六个正负边界测试，历史失败不重写；完整运动验收单独记录。

2026-10-22 双条件放置协议检查：389项相关测试11.90秒通过，Ruff通过。data/probe_current_place.py逐字节重建当前请求并将实际回答回送生产解析器；v14第一/第二件及动作前反例分别接受/接受/拒绝，哈希绑定。辅助探针只读存档，无运动、建图或状态修改；完整闭环另行验收。

2026-10-22 验收边界：v15运行126.12秒，在第一件plan_grasp停止，非Decisions拒绝；当前冻结图正反例通过不能替代完整闭环。913个预处理后候选进入IK，仅2个通过且轨迹失败。5cm接近单次对照无效，未采用；同一冻结地图的历史成功位姿能完成三段规划，仅作为只规划不执行的候选覆盖诊断。data/v15-grasp-diagnostic保留原始请求结果；未修改运行时规划参数/算法/碰撞门或新建场景。技能行为无额外变化。

2026-10-22 新场景：桌体中心[572,-26,532]mm，左右CAD盒中心[481,573,571]/[481,-625,571]mm，镜像轴Y=-26mm。移除两张小桌，桌体及两盒固定世界，仍参与接触；粉件Z旋转180°且保留重力。mock_unpack黑件局部Y转90°后绕世界Z改变yaw，世界高度44mm，初始底部距桌面1mm，由重力接触；命令包络按旋转后的CAD包围盒推导。当前黑件高度先验改为已声明平放44mm，须与测量及支撑一致，投放后不沿用平放先验。历史直立74.5mm仅保留旧记录兼容。

场景编辑往返保留fixed和CAD pitch_deg；CAD预览按世界Z偏航乘局部Y俯仰组合，不在拖动或保存时重置平放姿态。pitch=90°时使用局部Y轴提取偏航，避免欧拉角奇异点；物理姿态仍以完整四元数为准。观测位姿校验指纹纳入geometry.py。历史冻结的直立黑件点云保持原样，当前平放高度范围不接受超高候选；测试不修改冻结证据。

Decisions清单范围提示同步删除“相邻小桌”，只列大桌和两个盒子。173项场景、生命周期及实际MuJoCo物理测试通过；包括20秒接触仿真中持续15秒的速度、位置伺服、黑件落盒和桌面支撑检查。317项前端测试通过。上述是本次场景与物理验收，不代表完整两件拆包闭环已验收。

补充76项感知/建图/抓取输入/拆包衔接测试及前端生产构建通过。离线真实MuJoCo渲染中，黑件自然接触3秒后底部Z=0.57199599m（桌面0.572m），长轴世界Z分量约-1.44e-6；图像及物理记录在data/scene-revision-20261022。该记录只用于场景QA，没有调用Responses/Decisions，也未运行抓放任务。全部本地重检查串行、单核执行。

2026-10-22 v16盒体配准误停修复：使用原事件25的左右盒库存点云（79987/87582点）建立冻结回归，原代码三项复现失败。候选判等改为双份表面容差6mm内的全盒两两几何位移，原最佳拟合排序及真实歧义拒绝不变。加入旋转盒沿、禁止链式合并、半周对称和变换坐标系检查。116项CAD/建图相关测试及Ruff通过；不修改官方SDK、点云、CAD文件或10mm碰撞阈值。完整运动验证单独记录。


2026-10-22 投放可达性修复：黑件抓取在方向筛选、工具+Z深入10mm之后、源batched IK之前，使用本次观测配准的完整右盒CAD空腔中心、已发布障碍最高Z、当前完整物体几何最低Z和自身URDF手腕偏移，筛选保持抓取姿态的投放目标。名义TCP高度为盒体占据最高Z+100mm+抓取TCP到物体底部的距离；肩腕距离须在URDF两段上界减5mm余量内。缺少或过期的观测绑定停止；保留候选原顺序、分数及索引。此步骤为几何必要条件筛选，不替代实际持物plan_pose、attachment、10mm碰撞检查或五次插值。两段抓取不共用桌面槽位：粉件检查桌面目标，黑件检查右盒上方目标。v17真实持物姿态的水平肩腕距离625.6mm已超过602.1mm上界，原流程漏掉了该筛选。


2026-10-22 verify_unpack描述修复：CURRENT三相机仍合并为一次unpack_result判断，BEFORE明确标为拆包前且已通过桌面放置验证。黑色CAD是带凸起的不规则工件，不强制矩形；允许开爪部分遮挡，但可见主体必须能与机器人区别，且位于主桌面而非任一蓝盒。未降低0.8门限、未用动作记录代替视觉、未增加建图。v18第二件原图旧判断0.37；修改描述后同图实际Decisions为0.97，替换为拆包前粉件图的反例为0.0。原失败记录不改写。


2026-10-22 黑件投放名义可达性筛选后整池为空时，generate_grasps返回完整空池与逐候选筛选证据，由plan_grasp在原256→512→1000三池、总1768及共享180秒预算内使用下一池。空池不调用IK，也不伪造IK结果；回放记录no_transport_reachable_candidate及tested_count=0。只有完整有效筛选全拒绝证据允许该分支，服务错误/过期/未知仍停止；第三池为空明确返回grasp_no_transport_reachable_candidate。v19冻结原图点云的第二池512候选保留3项，原第一池256全拒绝不再提前终止。


2026-10-22 投放Z按宽松要求选择：保持盒腔中心XY和实际抓取/lift旋转，先由完整持物最低点算出盒子已发布最高占据+100mm的最低TCP高度；再取该最低高度与“手腕和自身肩部等高”对应TCP高度的较大者。这样利用允许的Z自由度缩短肩腕距离，仍保留100mm物体离盒余量，最终普通plan_pose和10mm硬碰撞检查决定可执行性。抓取前名义筛选与实际持物目标共用transport_tcp_height；不把高出盒子100mm误当成唯一允许的高度。v21第二件候选304原名义肩腕597.191mm因固定Z被排除，改为等高后589.650mm，可进入源IK。XY/R、目标盒坐标、世界模型、attachment和时间延长不变。


2026-10-22 投放堆叠修复：verify_drop当前契约改为视觉新增入盒，替代历史修订中的逐个可辨N+1条件。v22两件CAD物理落点均在右盒，但第二件叠放被旧数量题判0.07。保留一次融合Decisions与0.8门，增加同盒头部前后放大图，比较新增本体/特征，排除旧件移动及视角变化；完全看不出新增仍unknown。完整占据证据留在bin_region_geometry，提示去除重复49片壳明细。原失败图新题1.0、投放前图替换CURRENT的反例0.0；原v22任务结果不修改。147项回归通过，仍需新全流程验收。


2026-10-22 CAD局部抓取修复：整件点云沿闭合轴的宽度超出jaw gap范围，不再直接否决GraspGen的局部抓取。它可能包含不进入指间的宽底部。无宽度根时q_width_rad保持null，width_prediction_status=whole_cloud_width_outside_range；仅当非接触球全部属于固定TCP_Link时，用张开角检查完整原云与固定TCP/相机支架净距，非接触部件若随闭合运动则停止，不能默认通过。宽度有根时仍保留估计及原检查。所有点和padding保留，负净距仍拒绝；实际闭爪与lift状态才判断抓稳，实际attachment检查不变。v23已通过IK的CAD候选484原被整云115.9mm宽度误拒，现通过固定非接触几何检查。


2026-10-22 单件性能实验：默认只保留原两件中靠近机器人的粉件，中心[0.481,0.493,0.5595]m，原2/3/4实例编辑能力保留。共享桌面目标中心为机器人坐标[0.472,0.074]m，方形边长0.32m、无视觉标记；机器人+Y为左。模拟VLA黑件平放在[0.48,-0.20,table_top+0.023]m，避免原[0.36,-0.24]靠近机器人/右盒位置。右盒投放使用当前配准空腔内侧半区的TCP目标（从盒腔中心向世界+Y偏移盒腔Y宽度的1/8），仍以夹爪末端TCP对准，不以持物点云中心对准；候选投放可达性使用同一TCP目标。保留完整包络/1cm碰撞检查、实际lift姿态及高于盒体的宽松Z。单件验收逐节点统计墙钟耗时，目标180秒，未实测达标前不宣称达标。

性能计时增加只读/api/run-progress，轮询仅取当前步骤/状态，结束时才读取完整归档state；算法输入、碰撞校验与节点图片回放不变。

单件性能验收使用scripts/profile_demo_run.py输出顺序skill墙钟统计，以及嵌套算法/执行阶段明细（不可重复相加）。性能设置time_scale=1.5，停顿approach/place_approach/lift/retreat=0.3s，grasp/place/release=0.5s；夹爪实际稳定窗口、五次插值、实际执行前碰撞检查保持。关闭连续视频仅用于本轮控制链路计时，节点RGB/Decisions/SAM/点云/IK/轨迹仍全归档。


2026-10-22 抓取规划性能：Demo源IK合格池仍完整保留，按原分数排序分成每批最多4个真实候选，普通Algorithms兼容调用仍为16；首个有可执行轨迹的批次选七关节运动最短项，失败再处理下一批，不丢弃剩余候选。GPU服务将1到4项补齐到官方4容量，5到16项仍补齐16；只复制最后真实候选做固定shape填充，不增加逻辑候选，不改SDK、不改变碰撞与最终命令验证。原256/512/1000生成预算和共享超时保持。


2026-10-22 最新用户修订：time_scale默认及当前设置2倍，三段动作各停顿1秒。释放不再做开爪中间角度扫掠：只检查闭合状态+attachment及完全张开+detach；place approach/place保留实测闭合手型和attachment，retreat为全开空爪；黑件transport始终attachment。plan_grasp每批4候选，失败继续候选池。observe_scene记录发现、分割深度、桌面几何、身份判断、几何滤波分项耗时；不等于build_scene的TSDF/ESDF。

2026-10-22 CAD页面加载：已复现首次目录网络失败后不重试、选择器一直为空。增加最多3次目录请求，分别等待1/2秒；永久错误仍显示。编辑器渲染错误单独报告，避免误标CAD加载失败。skill行为无变化。耗时报告加入observe各分项。

2026-10-22 VLA初始关节位姿（角度）：左A=[-86,76,88,-90,2,14,-24]，右B=[106,61,-113,-78,27,36,37]。使用现有双臂关节目标规划接口，保留最近场景、空爪与双臂碰撞检查；不使用TCP目标替代指定关节构型，不重建场景。

2026-10-22 增加后端耗时诊断：IK求解+硬检查与逐候选FK分开，抓取SDK调用、FK、密集轨迹检查分开；记录每次公开plan_pose墙钟与SDK total_time。诊断不修改求解、阈值或执行结果。

2026-10-22 归档单件场景的规划诊断（未执行仿真运动）：同源16候选的前4项与全16项，首次/复用服务耗时分别16.709/7.303秒及30.205/26.214秒；4项复用时SDK原生调用7.215秒、逐候选FK0.008秒、额外密集检查0.039秒。首次100目标IK服务3.809秒；复用求解+硬检查约0.015秒，服务约0.04–0.10秒。VLA新关节目标在归档放置后地图上通过GPU规划及720点双臂碰撞验证，未宣称完成新姿态全流程。详情data/batch-vla-profile.json、data/batch-vla-sdk-timings.log。

2026-10-22 常驻预热（本段替代早期按持物几何新建配置的行为）：接收任务前按保存设置预热13个配置：A/B各IK100、批4/16、普通单臂、单目标抓放，加双臂及A/B位置可达性。ready在全部完成后发布。两侧各预留31个持物球，夹爪反馈和附件通过公开几何API原位更新，换物、换臂、detach和重置地图保留兼容规划器。重置清空地图与证据，首次任务仍须观测建图。仅显式改变求解参数、工具拓扑或超出场景容量时重建。生产实例是官方原类；plan_grasp内部不再覆盖plan_pose，重试/finetune采用其公开接口支持范围。详见docs/curobo-public-api-audit.md。

预热容量核对：双CAD盒发布98个凸网格片，启动预留mesh=128、cuboid=128、voxel layers=8，防止首次真实地图加载因默认mesh=1扩容而销毁预热实例；超过容量仍按实需扩容并重建。新增日志明确区分求解参数变化、机器人几何变化和场景容量变化。无绑定开爪配置统一到显式A/B=0，重置保留当前开爪实例，旧地图仍不可执行。
预热多种批量后，effective_config从本次实际选中的backend读取，避免把最后构建的单目标配置误记为抓取批量配置；不改变求解参数。

2026-10-22 GPU复核（冻结归档fcdc8828e6844ebca557d4ec8738b542，无仿真运动）：最终版本启动9个常用profile预热35.1秒，首次加载CAD地图、重置与黑件新地图后未再出现求解器构建/容量失效日志。预热后首次IK100容量（16真实目标）核心17.1ms，复用12.0ms。完整搜索交替4/16两轮：粉件4为10.77/8.31秒、16为28.57/29.58秒；黑件4为7.42/6.74秒、16为23.05/23.28秒，均第一批找到通过末端/命令碰撞验证的抓取。强制处理同样16候选的吞吐对照：4次4=32.71秒（16成功），1次16=25.82秒（8成功）；求解器随机状态自然推进，不能假定各次解完全相同。当前默认4适用于首批成功的样本，不宣称大批量总是更慢。证据data/complete-grasp-search-profile.json、data/final-warmup-validation.json；最终部署后另核验粉/黑完整搜索11.49/7.34秒。CPU配额不变，111项相关测试通过。未在本轮重跑物理抓放全流程；动态持物几何、超容量场景或求解参数变化仍可能需要新的实例，启动预热不代表这些未知配置也已预热。

2026-10-22 官方抓取调用及GPU剖析核对：锁定commit 8e734f3的公开BatchMotionPlanner.plan_grasp没有外部IK解参数；当前请求只传目标矩阵，未复用前序batched IK关节解。官方Stage1实际执行完整plan_pose，然后执行approach/grasp/lift，不能把Stage1描述成仅IK或廉价预检。官方getting_started/motion_planning.py抓取示例采用单问题多goalset选择，当前批处理为B个独立问题、每个goalset=1；两种工作量不可混为同一benchmark。同镜像运行未修改官方grasp示例成功。冻结4候选GPU剖析：self CUDA约6.894秒，主要场景运动碰撞kernel5.295秒、另一场景碰撞kernel1.027秒（约92%，不叠加嵌套NVTX区间）；不是开爪扫掠检查。仅在隔离诊断进程中禁用98个CAD mesh障碍，4候选耗时由完整世界warm9.10秒降至2.05秒，恢复完整世界11.64秒；随机状态自然推进，此实验只用于定位碰撞表示的计算代价，不证明禁用后轨迹安全。诊断轨迹没有执行，生产碰撞世界和1cm阈值未改变。证据data/planner-cprofile.txt、data/mesh-cost-profile.txt、data/official-grasp-example.log。本轮skill行为无变化，不能宣称性能修复已完成。

2026-10-22 原生SDK与适配层同条件对照：build_geometry_backend增加仅供隔离对照使用的batch_planner_type构造器注入，生产默认仍为ConfiguredBatchPlanner，不绕过高级设置。data/compare_public_planner.py使用同一机器人、完整98片CAD场景、同一4候选、相同配置，各次调用前reset_seed并同步CUDA计时。官方BatchMotionPlanner.plan_grasp两次6.372/6.297秒，现有适配层6.385/6.346秒；每次approach/grasp/lift成功向量均为[true,true,true,false]。这验证了该样本适配层与官方的成功结果及耗时接近，不证明所有输入轨迹逐点相同。证据data/public-planner-comparison.txt。batched IK仍直接调用公开InverseKinematics.solve_pose，未重写求解算法；初次/复用实测见上方记录。启动预热、兼容缓存保留和CPU上限不变；本次77项相关测试通过。审计钩子不改变skill行为，本轮未执行物理全流程，也未消除CAD场景的碰撞计算瓶颈。

2026-10-22 四条规划链的预热补漏：此前9个启动profile未包含独立compound缓存，因此普通单臂实例预热不能覆盖plan_grasp/plan_place。现增至11个，额外保留A/B compound实例，并对单臂与batch调用官方warmup(enable_graph=True)，覆盖此前关闭的PRM预热。启动后的实测attachment或夹爪几何变更不能提前知道其配置，新实例在构建返回、进入缓存前预热一次；命中缓存不再预热。配置变化仍隔离，不能把开爪无持物实例当作已预热的持物实例。没有修改SDK/vendor源码、碰撞阈值和动作顺序。相关152项测试及Ruff通过；完整运动未重跑。

plan_place调用核对：当前ValidatedSingleBackend._plan_pre_release_native直接使用官方MotionPlanner.plan_grasp(plan_approach_to_grasp=True, plan_grasp_to_lift=False, disable_collision_links=[])生成持物接近/放下两阶段，释放开爪detach后的撤离单独规划。没有执行额外持物lift，也没有自行重写三阶段优化器。释放净距检查、实际关节路径碰撞验证和结果缓冲区复制属于项目安全/接口边界。data/compare_public_single_planner.py通过single_planner_type注入未修改官方类，以同一单目标与场景比较三阶段及两阶段；两边重试预算统一为官方默认5，随机种子复位。该探针用无持物抓取目标比较API算法，不是实际持物放置性能验收；最终数据见data/public-single-planner-comparison.txt。skill动作语义无变化。

部署验证：只重启tianji-luna-curobo-guangyu，11个profile完成后health.ready=true，启动预热55.123秒；CPU配额不变。统一重试预算后的原生/适配层三阶段耗时2.932/2.744秒，两阶段2.056/1.969秒，各有效阶段均成功。此前2.480/2.700及1.842/1.845秒来自重试预算尚未统一的初测，不作为最终同配置结论。单次样本不能证明总体加速，98片CAD场景的GPU碰撞代价仍存在。

2026-10-22 用户修订：整体三视角2cm TSDF/ESDF仍只在双臂观测位姿建立；verify_place/verify_unpack视觉确认后允许当前位置头部RGBD局部更新当前工件。同帧头部SAM3用两个颜色/位置描述，剔除自身机器人表面、缓存主桌范围外/桌面以下返回、1cm内少于4邻居的孤立点；有效点少于30则停止运动。当前粉件/黑件的包围体使用实测XY、2cm包络余量和CAD姿态高度先验，Z覆盖缓存支撑面到可见最高点或先验顶面；明确不是完整表面测量。通过local_object更新只替换活动实体旧体素/补高层/动作包络，保留其他体素和首次桌面cuboid。拆包成功后删掉旧粉件碰撞层，新增独立黑件障碍；下一次观测建图再替换全局场景。失败不允许沿旧物体位置继续运动。SAM mask、点云、输入capture、局部包围体与新场景版本通过local_object_obstacle事件归档。

双臂规划：使用官方MotionPlanner.plan_cspace/plan_pose，保留官方enable_graph_attempt=1，首轮轨迹优化，失败后才用PRM种子。默认及本次高级设置max_attempts=2；用户仍可设置1，但此时不会触发图搜索后备。双臂启动预热分别覆盖官方PRM与plan_cspace，不把预热的graph_attempt=0用于业务规划。attachment按下一步动作直接准备持物compound(place)或single(pose)实例，避免旧流程先建独立IK再建轨迹实例；无阶段信息的兼容入口仍保留IK准备。prepare_timings_ms分别记录构建/预热和场景/attachment上传，不能把全部时间称为attachment球拟合。

失败样本229d078e51ca4939a667a0402156f0bc：119.465秒停在拆包后move_to_observe，之前verify_grasp、verify_place、verify_unpack均通过。observe_scene49.99秒、build_scene2.56、GraspGen1.25、抓取规划10.43、执行抓取11.91、抓取验证0.81、plan_place13.46（其中prepare_attachment4.31）、execute_place15.69、verify_place2.50、prepare_unpack6.57、mock_unpack0.77、verify_unpack2.22。该样本已存在粉件刚体预测和黑件动作包络，不能描述成始终完全没有桌面物体障碍；新修订改为结果帧局部实测。原错误是求解未找到轨迹，graph配置缺陷可证，不等于已证明物理路径不存在。

2026-10-22 同一失败场景808a6c的独立GPU回放：首轮强制PRM返回None（1.53s），官方direct-first得到成功（首次2.38s，随后0.57–0.60s）。撤销首轮强制PRM改动，恢复官方顺序并使默认预算容纳后备路径。原始错误229d的回观测失败仍需完整实测确认，不能以端点无碰撞推断路径必可行。局部cuboid的observation_id仅属审计字段，上传SDK前移除，原始测量证据保留。

2026-10-22 回观测根因实测：实验45334fbd在VLA后黑件局部障碍与右手gripR_Link3_R球重叠约12mm；当前起点碰撞，目标观测位姿有效。原mock黑件中心[0.48,-0.20]m在固定VLA右手下方。默认改为[0.58,-0.20]m，仍在右侧桌面、平放、有重力；回归覆盖任意平放yaw、头部局部包络2cm扩展与全部机器人球10mm净距。粉件当前局部包络距VLA手部约68mm，未发现粉件阻挡VLA目标，因此暂保留左前方[0.472,0.074]m中心。规划仍只使用实测头部局部障碍，不使用mock真值替代测量。

2026-10-22 cuRobo世界模型封装修复：每个蓝盒改为一个完整CAD三角网格cad_surface，保留全部33082三角面、空腔、原米制尺寸和当前深度配准位姿，验证闭合及绕序一致；MuJoCo仍使用原49片凸分解做物理接触。旧实验49片完整性检查继续支持，新图只接受完整surface及其visual SHA256，不允许任意缺片。原2cm实测TSDF/ESDF、桌体cuboid及10mm检查保持。固定同一4候选GPU对比，原98片预热后6.130/6.475秒，两整网格1.955/1.979秒；不是删除盒子或禁用碰撞的对比。

同步修订build_scene/plan_transport/verify_drop等运行时说明，区分当前整网格与历史49片。CAD映射回归中的投放放大图断言对齐现有BEFORE/CURRENT×右腕/头部四图接口，提示仍省略重复层元数据，原始几何归档保留；未改变Decisions图像或判断行为。

2026-10-22 VLA当前位置可见性修复：默认黑件中心进一步从[0.58,-0.20]改为[0.58,-0.08]m（右前方桌面，平放、有重力）。旧落点与机械手无碰撞但被遮挡，实验c167ee24在拆包核验后局部建障碍失败，两个SAM描述均无结果。新落点固定VLA姿态的三相机离线渲染均可见，头部SAM实测624点、约2.16s，膨胀包络距机器人球至少28mm。新增全yaw头部实际遮挡回归；渲染实例标签只用于场景设计测试，运行建障碍仍仅使用SAM与实测RGBD，绝不输入模拟物体真值。

2026-10-22 最终单件验收250b9e93985f480c826bf9105bce83a6：全部技能完成、最终verify_drop confirmed，墙钟201.945s，Responses1次。证据data/local-object-acceptance-v6.json、data/local-object-timings-v6.json；实验回放保留全部节点。头部粉件局部测量1262点/1.060s，黑件616点/0.371s（这两个时间不包含网络上传），均先发布障碍再移动。双臂回观测规划2.773s，VLA规划2.569s；实际运动分别计时，不能与规划混淆。保持10mm阈值、time_scale2、CPU总配额4核。

本轮cuRobo审计结论：361个已安装SDK Python文件与锁定commit 8e734f3ced1df898990bcd92de40abce475907db清单一致；SM120在当前PyTorch编译架构内。IK为公共InverseKinematics.solve_pose，抓取为公共单/批MotionPlanner.plan_grasp，放置使用公共plan_grasp两阶段加detach后撤离，投放为公共plan_pose，双臂关节目标为公共plan_cspace。应用层仍负责模型配置、几何更新、候选过滤、时间伸长和10mm复核，不能描述成整个封装逐行等于官方示例。单臂接口不支持批量finetune设置，effective_config明确报告该限制。相同配置官方/适配层对照结果见data/public-single-planner-comparison.txt及先前批量对照记录。

本次耗时（skill互斥统计）：两次observe_scene57.895s，其中初次52.404s；两次plan_grasp22.126s；plan_place11.037s；plan_transport10.542s；两次build_scene合计4.265s。初次感知细项：Responses18.443s、桌面/检测几何22.043s、身份判断7.171s、分割1.504s、几何过滤2.140s；不能将observe_scene的总耗时归因于TSDF/ESDF。持物准备仍各有新配置预热5.788/6.510s，场景和附件上传仅0.133/0.125s，尚未完全消除未知持物/闭爪配置首次初始化。默认max_attempts2也影响批量阶段，当前高级设置仍显式生效；不宣称所有调用采用官方默认预算。真实右臂IK100求解加硬检查10.6ms，FK45.5ms，四候选原生三阶段抓取4.518s，额外密集检查30.8ms。

可见性和碰撞回归21项通过，Ruff通过。先前一次5d00857b实验粉件未夹住、空夹反馈触发停止；保留失败证据，不把本次单次成功外推为稳定成功率。180s目标尚未达到，当前差21.945s。没有关闭碰撞、删除盒壁、修改用户固定VLA关节角或使用模拟真值给视觉判断补答案。

已同步Windows保存的plan_max_attempts=2，与Ubuntu此次成功验收配置及代码默认值一致；其余保存设置未改。

2026-10-22 严格官方调用与常驻预热修订进行中：生产构造器改为官方MotionPlanner/BatchMotionPlanner原类，项目PublicPlanner仅组合调用并保留返回结果；移除plan_pose继承覆盖，不再在plan_grasp内部清空dt、克隆阶段缓冲或注入阶段预算。公共plan_grasp未暴露max_attempts/finetune，内部采用锁定SDK原始默认；普通plan_pose仍按公开参数应用高级设置。固定球槽/原位几何更新和清单派生视图将在同一修订验证，未完成前不宣称已消除运行期预热。

同轮修订：左右持物link各预留31球；换物只经公开update_link_spheres更新数据，反馈改变经update_kinematics_config更新锁定指关节。卸载只清持物槽，不销毁预热规划器。删除碰撞检查器方法覆盖；应用层释放检查的浮点误差容差保留。高级设置说明同步限制，不再声称可覆盖官方plan_grasp内部预算。GPU几何一致性与常驻缓存验证进行中。

清单派生视图：初始物体档案冻结保留；当前实体排除历史粉件/已投放黑件个体，并将纯数值目标区域单列；右盒以集合表示碰撞几何。pending_unpack_ids包含已搬到桌面但尚未确认拆包的粉件，not_started_input_ids单列未开始工序；pending_delivery_product_ids与delivered_product_ids区分已拆包黑件去向。均从原工序状态派生，不建立第二套可变队列。初始粉件ID不重编，黑件每次确认拆包只创建一次独立ID，工序记录保留来源关联；随机落入右盒后不强行恢复逐件视觉身份。

体素上传不再直接round SDK内部params缓冲。公开VoxelGrid输入的米制dims选择不会向下丢失整数cell数的float32表示；2cm体素和ESDF采样值不变，1～512格的尺寸偏差在1µm以内。该兼容处理仅位于输入转换边界，不改官方碰撞核、距离函数或内部索引算法。

启动预热清单补齐选臂使用的左右位置可达性IK，共13个常驻配置：IK100×2、抓取批4/16×2、普通单臂×2、单目标抓放×2、双臂×1、位置可达性×2。ready只在全部完成后发布。任意编辑场景超出预留容量或显式修改求解参数仍需重新分配；不将未知尺寸的感知地图分配描述为已完全消除。

本轮结果：官方归档788个已安装文件无修改；219项针对性测试、清单前端8项与Ruff通过。实际启动13配置51.33秒。实验5293d3215d19474fb5f230e0a5e1d55e全流程194.214秒成功且无运行中规划器重建；最终输入修订实验6316357da2bc42d3b4b67b27e931b0e4在85.147秒粉件空夹停止，回放显示下降时先推偏粉件约2cm、随后回落。预热/官方调用修订不等于物理抓取稳定性验收；详见审计报告及保存回放。


2026-10-22 放置核验与桌面颜色修订：本次cc543b实验的原图将“在两指二维投影之间”误判为仍被夹持（0.99）。verify_place继续使用一次三相机联合Decisions、原0.8/0.2阈值及初始清单/参考图，但现在加入同次capture的实际夹爪角度、开闭限位、已验证open_confirmed和来源/observation_id；不是只提供开爪命令。实测开爪不能证明物体位置，仍须当前RGB确认独立粉件在主桌上，仍保留视觉物体实际挂持/接触的否决条件。问题明确二维投影重叠不等于物理夹持，当前头部总览最后呈现。没有新增视觉请求、移动或建图。放置与拆包提示不再硬编码桌面灰色。

场景桌面新增color（命名色或#RRGGBB），缺省white；从编辑器到buildScene、后端normalize_scene、MuJoCo材质及manifest保留，应用场景后生效。只改颜色常量，网页渲染器、30fps调度、灯光数量、阴影/后处理和RGBD分辨率不变，不增加渲染pass。不会把回放中历史灰色桌面改白。cuRobo官方SDK及调用边界无变化。

本次验收：冻结原图正式问题模板复测place_result=1.00、workpiece_still_held=0.01；源盒粉件/空桌反例0.00/0.00拒绝。154项相关回归及前端生产构建通过；白桌1–4件观测位姿离线验证并更新资源指纹。未重跑整段物理搬运，详见data/place-result-cc543b/report.md。

2026-10-23 回放展示核对：新增按实际调用顺序组织的阶段视图，显式标识 Responses / Decisions 的 endpoint、用途、问题、回答和输入图片；清单页显示归档完整快照与明确标注的产物登记事件更新。相同 observation_id 的标注版本合并，夹爪反馈持物状态不再冒充新的视觉观测。核对 observe_scene、bind_inventory、verify_grasp、verify_place、mock_unpack、verify_unpack、verify_drop 的归档语义：本次只改浏览器只读展示与回归测试，skill 行为无变化，不改视觉问题、决策阈值、规划、SDK 或机器人运动。

2026-10-23 投放坐标与证据记录修复：`plan_transport` 和 `generate_grasps` 的运输可达性筛选统一以夹爪前端对准右盒内侧半区。SDK 的 `TCP_Link` 是法兰，目标平移须减去 `R @ [0,0,0.185] m` 的 XY；不以持物点云质心瞄准。保留 lift 姿态、完整持物投影检查、attachment、10 mm 碰撞阈值和官方 plan_pose。投影失败记录盒腔边界、投影范围、余量、前端目标和法兰目标，报错摘要说明失败节点及未执行运输。

拆包后的黑件抓姿在既有左右臂标定转换后先做原方向筛选；平行夹爪绕末端 Z 的两个半周等价朝向选择更接近当前执行臂姿态的一种，记录转换前姿态及半周选择。随后沿末端 +Z 插入一次 10 mm，再进入官方 batched IK / plan_grasp_batch；不修改已求解轨迹，不新增候选数。SDK 内部及服务缓存不变。

Decisions 仍一次接收三视角；粉件桌面释放后与 VLA 后原位核验，抓取前观测位姿核验，抓取 lift 仅夹爪反馈，入盒后视觉核验。输入发送边界去除字节完全相同的图片与无框/无区域的空标注图，并用文字指向保留图；全部唯一原始相机图保留，问题/选项/置信度门槛不变。候选放大图最多两张，跳过已占原图面积8%以上的大物体，避免两个名额被盒子占用。近重合 SAM 候选框只合并绘制轮廓及别名，保留独立问题与掩码，绝不据此合并永久 ID。

每次 bind_inventory、后续 observe_scene、verify_grasp、verify_place、verify_unpack、verify_drop 完成后记录 inventory_snapshot，包含当时完整清单、观测 ID、模型事件 ID 和证据来源。反馈节点明确标为非视觉；旧记录不补造历史快照。回放对新记录优先使用这些节点快照，避免同时叠加旧式派生快照。记录测试与部署证据目录：data/drop-replay-audit-20261023。

核验异常也记录 inventory_snapshot，标记 not_applied 并保留已有清单，关联本次失败模型事件；不伪造新身份。半周等价抓姿选择仅用于 unpacked 阶段的黑件，粉件方向/插入及放置行为保持原样。错误摘要识别视觉身份不明、cuRobo轨迹碰撞等原因，完整诊断仍保留。

2026-10-23 核对：同步verify_unpack固定一次换视角分支、grasp_view_ready契约、observe_scene/build_scene/plan_grasp位姿绑定、前端参考的粉件/右盒目标与名义可达性筛选；新增unpack_view_motion只修改项目碰撞适配层，官方cuRobo源码未修改。旧实验记录仍保留原始语义，不代表当前行为。
2026-10-23 实测补充：VLA自然位姿的SAM候选会包含黑色机器人部件。现在关联前使用当前关节/标定剔除机器人自身返回，并以缓存实测桌面和已知平放黑件高度约束黑件候选深度；同相机、同类型/描述颜色且mask IoU≥0.95的候选只保留一份几何、保留全部prompt来源。只减少重复候选，不创建或合并永久ID。原始SAM图片/掩码仍完整归档，Decisions门限不降低。


2026-10-23 回放与收尾：完整归档 GraspGen 服务返回的全部候选（包括超出本轮采样预算者），分别查看原始生成、方向/名义可达性预筛选及 +Z 深入 1 cm、batched IK、plan_grasp_batch 四阶段。每个候选显示按执行臂标定转换的完整 XYZ 坐标轴，X 红、Y 绿、Z 蓝；没有凭 Z 补造 X 或额外生成位姿。已归档最终预筛选索引包括可达性筛选结果；旧实验缺失的原始输出不补造。最后一件黑件 verify_drop 成功后，保留入盒动作障碍包络，双臂必须沿最新场景返回初始观测位姿，回位失败不报告任务成功，不重新建图。局部目标更新返回的碰撞实体清单包含保留的背景层。


2026-10-23 初始清单输入核对：所有观测的同相机、同类型/请求颜色SAM轮廓IoU≥0.95合并，保留原像素及全部描述来源，不分配实体ID。不同相机或不同物体不合并。未搬运粉件保留Responses的初始视觉描述，避免两个对象被覆盖成完全相同通用描述。初始Responses必须按实际选用的图片绑定bbox和camera_id，不能按机器人侧别猜相机。generate_grasps和plan_grasp的运行时contract删除历史长段，仅保留当前前提、算法接口、预算、输出和停止条件，详细实现仍在本手册。

2026-10-23 提交范围核对：.gitignore排除实验data、日志、凭据环境文件、依赖、网页构建和本地工具缓存；保留源代码、运行配置、CAD/机器人资产及回归测试夹具。排除规则本身不改变skill行为。

2026-10-23 count任务缓存修复：按距离选择指定数量时仅更新活动任务membership，不再改写visual_inventory_reference.entities。该初始身份快照参与盒子RGBD锚点摘要；改写会使后续容器跟踪失效。原始来源实体及其显示编号保持，未选实体继续作为碰撞对象。wrapped_source初始描述仅保存一次，重复生命周期投影不会反复拼接文字。

2026-10-23 抓姿预算修复验证：冻结失败记录0fa4fbda的300个原始输出中，旧分数前256个全部方向不合格；全量预筛选保留8个方向及名义运输可达性合格候选，方向/插入CPU处理约13毫秒。不增加服务调用、放宽方向/碰撞阈值或修改SDK；无方向合格候选及桌面名义可达性失败现显示简短中文原因。相关80项后端回归、334项前端回归及生产构建通过；完整物理流程另以新实验结果为准。

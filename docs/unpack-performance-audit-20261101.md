# 拆包任务耗时与失败审计（2026-11-01）

目标：左盒粉色包装物 → 配置桌面拆包区 → vla_unpack → 黑色工件 → 右盒。粉色使用当前 OBB + GraspGen；盒子和黑件先由 Decisions 匹配 CAD，再由 FoundationPose 估计当前位姿；黑件使用执行臂的离线抓姿。模拟器真值不作为身份、位姿或规划输入。

## 已测量的耗时

前一轮 A19 是 CAD 黑件入左盒任务，不是本轮完整拆包任务。其总时长 940.616 秒，1 次 Responses、31 次 Decisions。

|互斥阶段|秒|
|---|---:|
|observe_scene 合计|632.833|
|verify_place（内含感知）|100.250|
|初始 Responses 请求|69.337|
|plan_grasp|36.467|
|resolve_placement|11.459|
|plan_place|12.568|
|execute_grasp|18.417|
|execute_place|16.284|
|move_to_observe|10.380|
|其余派发、核验及间隙|32.622|

Decisions 请求累计约45.443秒，嵌套在上述阶段中，不能再加到总时间。恢复观测在317.586秒已经成功，随后两次观测直到648.790秒才继续，额外331.204秒也已经包含在observe_scene合计中。实际抓放与回观测约45.081秒。证据指向重复感知和感知几何计算；不能把全程慢笼统归咎于cuRobo或FoundationPose。

A19首次成功观测中，几何阶段162.394秒，但原日志没有细分足够信息，不能把这162秒全部归因于某个函数。新增诊断同时记录墙钟与工作线程CPU时间，避免将等待误当计算。P2首次观测：SAM与深度1.587秒、桌面/检测几何27.236秒、身份1.921秒、实体几何57.702秒。两盒CAD解析各26.413/26.809秒，其中FoundationPose各4.694/4.704秒。

同一服务器、同一登记CAD与归档点云的纯计算剖析：105089个点、33082个三角面；trimesh 4.12.2的closest_point，512分块18.209秒，16384分块14.803秒，距离结果相同。仅约3.4秒收益，未据此改动生产算法。两盒完整validate_cad_pose分别19.271/21.527秒，OBB拟合约0.05秒，支撑区域计算约0.5–0.7秒。后者掩码由保留点像素重建，只作为耗时诊断，不能当原SAM轮廓的精确验收回放。没有降低0.8身份、0.9 CAD一致率、黑件2.2mm残差及5mm碰撞净距。

## 本轮实际运行与已修复问题

|运行|结果|确定的失败原因|
|---|---|---|
|P1 / 683035e51b9947d7bf2a77b2dfd3ff77|79.8秒，未运动|Responses生成不可达的root_failure；提示补齐已有graph可达性规则，未代写图|
|P2 / d7dcc56e26504534b65e769978a33642|180.9秒，未运动|目的地问题把粉色输入的中间加工区与黑件最终交付混淆|
|P3 / b50c7062858d49b991018c3108ebc683|174.8秒，未运动|仅追加加工说明仍不充分，主问题改为加工产物谱系的最终交付|
|P4 / cc1f62897cd14061b5f28a6a08af1ae5|约129.4秒，未运动|SAM未识别package，中文color又生成“粉色 object”备用查询；补齐英文color字段说明|

P2冻结原图单次目的地对照：右盒概率0.90/confidence0.88；P3主问题修正后0.94/0.93。这是诊断模型调用，不是物理验收。P4原始RGB在两个可见相机中，pink object均返回1个候选，SAM分数0.945，单次约0.1秒；不因此宣布身份确认。P2/P3实际观测已经分别得到两盒CAD、粉色OBB，但尚未完成拆包抓放。

初始编排仍一次Responses，运行期仅Decisions。模型选择恢复节点与参数，执行器不插入重拍或动作。P1–P4初始编排约74–80秒，本身已占2–4分钟目标的较大部分。恢复上下文达5万余tokens，但目前不能据此断言它导致重复观测。

## 资源与验证边界

仅本项目五容器，总CPU配额4核：web1.5、cuRobo1、SAM/GraspGen/FoundationPose各0.5；OMP/MKL/OpenBLAS均1线程。未增加配额、未修改官方SDK、未创建新镜像。

完整CPU套件（timing-3阶段）4229 passed / 58 skipped，1个既有Starlette/httpx弃用警告；随后目的地措辞与英文颜色schema说明的针对性测试通过，timing-5相关53项通过、Ruff通过。完整套件不是GPU抓放验收，也不能替代最新完整拆包执行。

原始归档和脚本：data/performance-unpack-20261101/。每次部署保存前后SHA-256清单和备份，仅空闲时重启本项目web服务。最终实际运行结果继续记在本文件；没有完成整套拆包前不宣称达到2–4分钟。

## P5 实际拆包与后续修复

P5（389752e186d34beb8e71c49d5208b331）620.452秒，1次Responses、50次Decisions，needs_attention。两盒CAD与粉色OBB通过；A臂GraspGen输入为observed_obb_mesh_surface_samples，生成后保留185候选；抓取、lift后attachment、桌面落点、三段plan_place、放置核验及vla_unpack/verify_unpack均通过。产物product-a7cee676eb5e4c35978bac66c426e97d经Decisions匹配THREADED_VISUAL_SAMPLE_MM，FoundationPose采用当前唯一合格相机3，数值验证通过；后续停止于identify_target的observed_support_unknown，尚未调用黑件离线抓姿或执行右盒交付。

首次恢复observe_scene以相同fresh参数已成功，随后Decisions选择retry_failed_node，旧实现只能再次完整观测。新增受契约限制的use_recovery_result操作提议，仅当前、成功、同技能/参数/绑定的最新恢复观测可供模型选用；控制器前后核对观测、代次、action epoch，执行器核对真实历史结果并沿原success edge继续，保留原失败和引用来源。当前仅observe_scene启用。84项相关测试通过。

黑件支撑问题的冻结诊断，每个改动假设均只发一次请求、无机器人动作：原请求桌面0.35；补CAD边界0.45；改分类问题confidence0.25；补当前ID定位框0.76；边界+定位框0.74。这些均未通过，不采用单独改法。加入代码计算的CAD底面与桌面差值−0.994mm、XY在桌面内及已有2.2mm残差容限关系后，桌面0.95、两盒0.0。实现补齐数值证据传递，不要求Decisions自行做坐标算术；不改问句、门槛或从仿真取位姿。冻结诊断不能代替新一轮完整执行。

## P6 的重复 CAD 检索

P6（c104aaca0f974ebea3bb105631d2f365）约837秒needs_attention，粉色物体已完成桌面放置，但后续回观测未通过。初次同一永久ID的no_catalog_candidate概率0.91、confidence0.88，后续仍重新检索并以confidence0.17提出黑件CAD，数值位姿验证拒绝。此前只缓存正向CAD对应；未知模型的对应每次重判，导致同一对象跨视角反复改变几何候选。修复采用上述有明确原始证据的无CAD对应缓存；物体几何仍每次测量，目录或实体变化重新匹配。没有把被拒绝的CAD强制当成有效，也没有将已知CAD失败自动降为OBB。

恢复分支版本的全量回归4246 passed /58 skipped；CAD支撑与无匹配对应增量的100项相关测试通过。最终部署与完整拆包验收另列，P6不能计成功。

## P7 的恢复结果兼容缺口

P7（35268aefdd6d4edf87588246c4ae7770）约280秒，1次Responses、20次Decisions，未运动。恢复观测于194.1秒成功，但原节点capture_mode=reuse_current、恢复为fresh，严格参数相等检查没有提供use_recovery_result选项；后续重复观测再次身份unknown而停止。修复只由observe_scene契约排除采集方式参数比较，已有结果当前性与其他参数仍严格核对。timing-8全量测试在发现新缺口后主动终止，不计完整通过；更新后运行相关回归。

## P8 的放置语义与无效计算

P8（7557d1deb33c4de1ba694537c2f18103）约350秒，1次Responses、12次Decisions；粉色OBB的A臂GraspGen、抓取及lift后attachment通过。放置区域permission概率0.77是unknown，旧实现却与no合并，发布candidate_selection并继续抓取。抓取后候选图像过期，仍先完成每轮约28秒的落点计算，再返回placement_candidate_capture_stale；恢复改变偏好无法补图像。修复unknown不发布区域、语义问题区分当前输入与后续产物/准备阶段，并将既有候选证据当前性校验提前，不放松数值或模型门槛。P8未放置也未拆包。

初始Responses P6实测输入54378 tokens、输出8213 tokens、85.683秒；请求JSON约223453字符，其中skill_instructions约99165、skill_catalog约91228。内容体积是明确的编排开销来源，不能把它算入机器人规划。尚未通过删减契约或降低推理配置来冒充已验证性能收益。

## P9 与用户11月2日修正

P9（203527e9261245e9acebe63874dc68c1）1092.384秒needs_attention。粉色抓放、拆包、黑件CAD及桌面支撑绑定通过。初始粉色CAD检索unknown增加恢复；产物后右盒暂未观测，恢复观测547已经得到右盒，但identify_destination契约遗漏recovery_retry_safe，模型选项里没有retry_failed_node，随后反复建图/观测至停止。修复这一非物理绑定契约；不代替模型选择。97次Decisions合计170.321秒，中位1.682、最大2.892秒；次数失控是问题，单次模型推理并不慢。1次Responses耗时115.183秒。

用户最新要求：粉色明确未知且无CAD；工作台配置区域不搜索XY；普通区域改二分。初始测试指令现补齐这些明确要求。新增带原文引用的unknown_obb实体策略，跨当前身份测量传到解析器；配置XY固定且不迁移；一般区域15中心/最多3合格候选，Z与数值检查不变。96点网格不再是当前放置策略。FoundationPose官方run_demo先传mesh，再register(K,rgb,depth,ob_mask)；run_linemod按已有ob_id选择mesh和mask，不包含海量CAD语义检索。来源：https://github.com/NVlabs/FoundationPose/blob/main/run_demo.py 和 https://github.com/NVlabs/FoundationPose/blob/main/run_linemod.py 。

修改前timing-11全量回归4261 passed /58 skipped，556.56秒；新行为增量测试及下一轮实际结果另列，不能把旧套件当作新物理验收。

## 2026-11-02 P10 / timing-12 实测

Run `1e4e63dd324843eea30c4b76d152a7b2`，271秒停止，未完成拆包。Responses 90.115秒；10次Decisions统计见 `data/performance-unpack-20261101/P10-model-latency.json`。初始身份unknown后恢复观测67.735秒：SAM/深度1.592秒、桌面/检测几何0.193秒、身份关联7.721秒、几何过滤与估计57.593秒，其中两盒CAD分别27.150/27.597秒。粉色明确unknown_obb绑定永久ID，CAD入口0.000118秒且未检索目录/调用FoundationPose。Decisions成功选择复用恢复观测，无第三次完整observe。配置XY仅1个，resolve_placement 6.852秒（P9为28.67秒）；plan_place 6.324秒。到达place后尚未松爪时出现release_tool_reference_changed：新增固定XY使用物体参考点，但position_constraint标签残留finger_front。timing-13修正标签并增加求解到实际endpoint核验的回归；不绕过任何检查。

独立冻结A19 RGBD的CPU剖析（非P10总时长，不可相加）：全深度机器人表面归属52.810秒，其中15次MuJoCo mj_multiRay约46.807秒；彩色桌面标记0.402秒。该剖析使用另一场景/冷启动，只证明热点在CPU射线求交，不能将52.810秒声称为P10的桌面耗时。报告在robot_full_depth_ownership-profile.txt及observation-cpu-profile.json。既有CAD三角距离剖析约18秒，OBB本身约0.05秒。尚未达到全任务2–4分钟验收。

timing-12相关测试223项通过；timing-13释放链路62项、原生提示与求解相关101项通过。提示压缩仅移除已在相同SKILL输入中原样出现的完整语句，测试逐句证明保留，未删契约条件/参数/效果/错误或完整技能说明。

## P11 / timing-13 与 timing-14修正

P11 `5ebda60c826748ae97d0a8b81ea7b6be` 在330.5秒因 `task_binding_unconfirmed:destination_requirement_matches` 后恢复unknown停止，未发生抓取。1次Responses85.912秒；20次Decisions共35.883秒，中位1.533秒、最大5.995秒。三次成功observe分别63.970/62.246/62.060秒；目的地语义unknown后fresh观测，再以reuse_current重复同份证据。三个相机的外参逐值完全相同，不能归因于浮点视角抖动。timing-14为fresh已尝试的同输入登记reuse_current/默认参数等价项；不是放宽视觉阈值或强行选目的地。

释放核验此前调用完整清单observe导致无关盒CAD重复求解。timing-14内部几何范围仅覆盖被放物和必要容器/支撑/相对参照；所有身份输入不变，复杂语义仍保留全范围，结果不成为场景刷新。新增回归覆盖配置桌面、容器、物体表面、相对参照及未结构化语义，也拒绝普通observe使用该局部测量范围。相关174项通过；恢复/抓姿补偿111项通过。

P10实际place完成关节反馈的冻结重放：原始参考元数据准确重现release_tool_reference_changed；重新生成参考元数据后，保持相同数值目标和原高度证明，实际FK释放端点通过；物体参考XY误差1.373微米、attachment到平面10.099mm、物体底部间隙25.567mm。无新动作，无松爪，不能作为完整拆包验收。见P10-release-reference-replay.json。

## P12 / timing-14 与方向参考系冻结测试

P12 `aa520f811f3e480e8ef5a32b5733b783` 约159秒因目的地任务绑定unknown停止，无物理抓放。1次Responses63.656秒、5次Decisions合计8.165秒；初次observe成功，没有重复observe。本次未进入verify_place，因此局部释放测量的实际降幅尚未获得本轮完整任务证明。

目的地回答在左盒ID2=.45与右盒ID4=.47之间分散，confidence=.36。冻结原图/问题，仅明确默认左右使用机器人系、显式用户视角覆盖默认，一次Decisions请求1.872秒：右盒ID4=.87、confidence=.84，通过原.8门槛。只支持参考系不清的假设，非多次重试取好答案；无机器人动作。timing-15将同一约定供给初始Responses和任务绑定；当前RGB身份及位置证据不变。测试86项通过。另清除重新绑定区域时遗留的fixed_xy标志，避免前一配置区域影响后续自由区域。

## P13与全量回归后的修正

P13 `39b0fa99601644289f135da83bfde6c7` 在约65秒因 `graph_scope_missing_return:pink_package_unpack_and_deliver:leaves=root_verify_completion` 停止，未执行机器人动作。timing-16进一步明确根graph验证节点不能作为叶节点；程序不插入end。

全量timing-15检查：4252通过、30失败、58跳过（492.90秒）；不能称全量通过。失败主要是把所有配置区域都强制到中心，破坏未指定点的多物体自由区域放置；另有二分候选中心抬高释放优先于相邻低位、以及旧schema测试未填新增geometry_policy默认字段。timing-16用显式placement_mode区分自然语言指定配置点与普通区域，恢复默认free_region兼容，初始/运行时模型负责枚举选择。全部30个先前失败重测通过；相关129项通过。配置点仍只有1XY，代码没有偷换成邻近空位。

独立Open3D 0.19 CPU单线程测试：105089点/33082三角面，距离查询0.05–0.07秒，原trimesh约18秒；最大距离差0.17449mm，15mm分类无差异。返回三角面后float64复算仍有最高0.57303mm差异，未证明黑件2.2mm条件下的等价性，因此没有替换生产校验或新增依赖。只在已有FoundationPose镜像运行冻结数据，无模型调用或机器人动作。完整拆包及2–4分钟仍未验收。


## 2026-11-02 timing-17/18 加速与命名

CAD全点距离改用libigl 2.6.3公开AABB.squared_distance，float64、8192点串行批处理、每次验证复用索引。保留完整点云/三角面及原容差、0.9比例，深度/轮廓仍用原Trimesh射线。Ubuntu冻结105089点对照：0.188216秒，历史Trimesh同数据18.208826秒；最大距离差0.089508mm，15mm分类改变0，不能宣称浮点逐值一致。黑件好/坏位姿及蓝盒既有回归保持原通过/拒绝结论。

机器人射线替换未采用：同一冻结采集旧MuJoCo21.992秒、libigl18.235秒，但相机1/3分别17/108像素归属变化，contact_changes=0；没有以放宽阈值接受变化。timing-18恢复原MuJoCo过滤，试验代码与结果留在data/performance-unpack-20261101。此前52.81秒为含profiler的独立剖析，不能当成本次无profiler实测。

资产对外命名blue_box/蓝色盒子、black_workpiece/黑色工件，source_asset_id连接原外部库；SDK登记model_id、源文件与离线抓姿保持。实际SAM查询词与目录别名进入Decisions，不要求名称字面相等。额外释放间隙由10mm改0，硬净距5mm及原0.1mm跟踪余量不变；最低合格高度先单独做持物IK，通过停止，失败后才用既有批量接口检查余下区间。

本次新增一个必要依赖镜像tianji-luna-mujoco:0.7.0-cad-distance-guangyu（5801d3b043fd），基于0.6.0-cad-guangyu，仅安装libigl==2.6.3；0.6.0保留回退，0.5.0为其基础依赖。无新临时镜像。首次重建web时发现默认UID1000与实际1003/GID1004不符，已按当前宿主用户重建恢复；没有改目录权限。五容器CPU配额未变。持物高度逻辑在cuRobo服务执行，因此本次同步后重启该服务一次。完整任务结果待后续逐轮记录。


## timing-19 / P14 当前证据

P14 `a9844e60cdef4b9d89c8c4d9f3a7fc7a` 完成粉色抓取、配置XY低间隙放置与拆包；351.037秒停止，1次Responses（105.956秒）、20次Decisions。当前VLA姿态下右盒只有头部1469点，FP产生倾斜错位，原校验拒绝；原实现提前终止整次观测。修正为逐实例失败记录，地图仍拒绝未解决的数值失败。

同一冻结1469点/掩码哈希恢复一致，上次已验证CAD作为一次刚性ICP起点，当前拟合约0.5秒、连同全点/深度轮廓验收约2.4秒；位姿更新0.515mm/0.060°，全点一致率1.0、可见深度轮廓0.9104通过原0.9门槛。20次局部迭代反而使轮廓0.8507而拒绝，因此历史种子只作单步更新，拒绝后不追逐通过分数。该冻结诊断无模型调用和运动，尚不代表整轮完成。

P14互斥skill墙钟：初次observe_scene 52.105秒（SAM/深度1.880、桌面/检测27.769、身份2.442、实体几何19.691），其后普通observe_scene 21.657/23.795秒；verify_place 27.841秒；resolve_placement 6.047秒；plan_place 6.180秒；plan_grasp 28.162秒。20次Decisions累计25.226秒，中位1.3165、最大2.273秒，均嵌套在技能墙钟内，不重复相加。timing-18观测diagnostic把measurement_scope误写为release-only，是局部变量复用导致的日志标签错误，实际仍完整测量清单；timing-19已修正。


P15 `d293083f7be74095b5024809be61b2ec`：完成粉色物体物理放置，但释放身份核验冲突：Candidate 2→蓝盒0.94、→粉色1.0，正确保持unknown。约434秒人工终止重复原位观测，服务归档status=cancelled，不是整轮成功。三次恢复的三相机位姿及量化RGB均完全相同，解释结果却在缺失粉色/重新出现间变化；重复恢复仍需进一步收敛。对相同冻结请求仅增加参考kind/color和“物体本身，不是内容物”的逐谓词说明，单次Decisions 2.234秒，Candidate2→蓝盒0、→粉色0.99。其他门槛和图像未变。

本地完整回归timing-19：4298 passed、58 skipped、1 warning，494.59秒；随后身份呈现针对性38项通过。机器人视觉OBB边界试验在P14冻结输入的三路归属/接触记录完全一致，但旧48.21秒、新33.29秒含冷/热差异，未采用该速度差作为收益证据，生产仍保持原MuJoCo过滤。

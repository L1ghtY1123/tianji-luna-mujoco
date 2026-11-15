---
name: luna-policy-isolation
description: Use when modifying, debugging, extending, or reviewing tianji-luna-mujoco task planning, robot skills, visual decisions, objects mapping, placement, or cuRoboV2-Fast integration. Preserves the ownership boundary between Responses, Decisions, and physical skills.
---

# Luna 任务策略隔离

本 skill 指导 Codex 开发，不是机器人运行时能力。先阅读项目 [AGENTS.md](../../../AGENTS.md) 和 README；AGENTS 是当前要求的唯一合同，本文不累积历史覆盖条款。不使用 JEV runtime 或 sub-agent。

| 层 | 职责 | 边界 |
|---|---|---|
| Responses | 初始实例、任务要求及完整图，可绑定固定子图接口或编写自定义 body，模板与显式路由组由程序展开，每个任务只允许一次 Responses，校验失败即终止 | 归档原始输出与完整校验结果，通过后执行 |
| Decisions | 必要视觉事实；已知状态与数值恢复由程序处理 | 有类型输出，按概率最大项取值，confidence 仅归档；不把动作提议当事实，不替代数值安全检查 |
| Memory | 时间记忆数据库保存实例和历史，空间记忆保存碰撞场景及其版本和有效性 | 原始观测、测量缓存及配置区域归执行层；时间记忆拒绝任务进度和当前碰撞有效性字段 |
| Runtime | 按 edge 执行、永久 ID 与参数绑定、事实版本、账本和已声明路径审计 | 不暗中移动、观测、换臂、建图或重播成功动作 |
| Robot skills | 选中能力内部算法、传感器与机器人接口 | 可固定内部步骤及真实前提，不串接隐藏任务流程 |

## 修改检查

实例关联只覆盖自然语言任务中的目标、来源、目的地、加工区域及其必要支撑。场景描述用于定位和消歧。初始 Responses 提出任务实例，Decisions 先在描述参考相机的当前带标签 RGB 中选择一个掩码，再仅为这些选中主体关联其他相机掩码。其他同类掩码可作为选择项和画面上下文，额外可见物体不自动登记。后续观测使用任务实例的已确认带日期参考选择当前掩码，再关联其余视角并继承永久 ID。近期参考不足时使用更早的确认参考。all/count 任务按目标描述发现新增任务输入，确认新身份后登记。程序保留同相机唯一性、跨相机关联一致性和可靠几何冲突检查。跨相机几何检查使用双向投影，并纳入本次已确认的同物体其他视角。每个方向独立报告有效深度对应、遮挡和冲突，任一可靠冲突排除该候选。 各主体先独立选择暂定掩码；跨相机物理分组后，使用当前多视角及日期参考做一次主体复核，允许选择其他掩码或明确不匹配；改变物理组时重新归组一次。复核不把初选当作已确认身份；同一物体存在多个归属提案时，用竞争描述和日期参考共同裁决，败选主体最多重选一次。已裁定掩码保留所属主体标签并从重选项排除，未解决冲突不提交。剩余候选仍需视觉确认，容器与内容物按各自掩码区分。历史选主体只发送包含该主体标注的参考视图。Decisions 的当前三相机 RGB 同时标明已确认视角，state 保存对应关系及来源。历史记录仅登记任务涉及的物体。碰撞场景覆盖工作空间内的可见几何，已匹配 CAD 使用 CAD，桌面使用实测 cuboid，其余使用 TSDF/ESDF。额外物体通过匿名深度建图，无需 Responses 命名或 SAM3 分割。已完成入盒物体保留入盒历史，后续由当前匿名 TSDF 表示可测表面，个体位姿保持未测量。每次 build_scene 都从当前观测重建整个工作空间：当前 CAD 位姿来自 SAM3 与 FoundationPose，其他可见几何来自当前 RGBD 的 TSDF/ESDF，桌面 cuboid 保留实测缓存。不选择局部范围，不合并历史物体碰撞几何或旧 TSDF 体素。历史身份与操作记录独立保留，未观测空间不因此被证明为空。测试任务不限定程序能力，也不产生任务专属建图规则。

1. 追踪实际入口、contract、handler、模型输入、结果消费方与回放。只描述真实能力和前提，不用命名把旧 steps 包装成 graph。
2. graph 只接受当前 version=2；共享 subgraph 可循环处理运行时选中的输入，物体记录保存身份与历史。独立任务 subgraph 为可选表达。node ID 是控制节点，CAD asset ID 是模型，均不替代永久实例 ID。
3. 修改 questions 时分别核对 CAD 选择、实例身份、可见性、支撑、目标区域与全局完成。没有参考图片就不能声称图片存在；三相机证据可在公共输入或 question.instructions 内，日志需完整计数。
4. verify_place 复用 object_held 正向题，只发送当前夹持接触区原始分辨率 RGB 与同次夹爪反馈。明确 no 取反为空爪成立，yes/unknown 不允许确认释放。之后 place_result 在 direct_success、inferred_success、failure、unknown 中选择。入盒任务以进入指定盒子为成功，控制台小方框只约束机器人释放位置，不约束盒内最终落点；桌面区域任务仍验收指定桌面区域。推测成功要求已确认持有目标、实际到达配置释放 XYZ（现有位置容差 5 mm）并释放、当前空爪，且没有交付失败的可见反证，不要求一定被遮挡。命令发出不等于实际到达；缺失绑定证据为 unknown，明确落在目的地外为 failure。当前三相机 RGB、释放前 RGB 与日期身份参考用于落点题；仅桌面任务投影 PLACEMENT_REGION。保留原始回答、执行推定身份、placement_outcome 与回执；推测入盒不伪造个体当前位姿，confidence 仅归档。
5. 批处理每件交付后优先在当前位置重新观测并选择当前输入；只有程序确认必要信息缺失时才沿 graph 回观测位姿补观测。all/count 使用共享子图，初始数量不限制运行时成员。all 以最新观测确认来源盒已空结束本批，count 按请求数量验收。最终双臂回位由实际关节反馈验收。VLA 起始姿态仍属于加工技能前提。
6. CAD/TSDF/cuboid 绑定永久 ID；SAM3 生成候选，Decisions 匹配目录，FoundationPose 估计位姿。当前位姿来源和坐标有效即直接使用，不恢复 ICP 或吻合门槛。仿真真值不进入感知或规划。
7. 放置 XY 使用控制台正方形中心，旋转保持实际抓取姿态。末端 Z = 实测桌面或盒口最高点 Z + 物体高度 + 控制台高度余量。物体高度来自同次 CAD/TSDF 几何竖直范围及来源支撑，独立于碰撞球。默认余量 20 mm，含为保守附着包络增加的 10 mm。若抓取 TCP 高于物体顶部，使用较大的 TCP 到来源支撑面偏移避免低估。实际物体底面另按抓取偏移校验，假设搬运中物体不滑移。碰撞几何独立检查目标及轨迹，不搜索目标高度。选臂时用可见高度估计，抓取后用已绑定 CAD/TSDF 再检查同一臂。
8. unknown 保留为真实 graph 结果，沿已有 edge 或已声明程序恢复规则处理；只有尚未解决的图像事实才交 Decisions，不自动转 yes。恢复优先在实际当前关节位置重新观测、重计算与规划；程序确认信息缺失后才回配置观测位姿补观测，配置回位无需额外视觉许可题，保留空手、地图和运动检查；额外腕部视角才需要视觉缺失判断。相同输入不重复观测。恢复所需的新采集由 graph 显式安排具有 captures_observation 权限的技能。
9. 删除替代实现时一并清理专属测试、依赖、prompt、前端和部署文件；迁移当前有效回归，不靠兼容分支保留旧框架。第三方 SDK 保持只读。涉及 cuRobo 的技能统一调用 cuRoboV2-Fast v0.9.0 服务及其公开 API。
10. 按 AGENTS 的 CPU 限额完成针对性测试、全量必要检查与真实仿真；报告成功与未覆盖的边界，不以健康接口替代任务验证。

模型 prompt、技能合同和观测恢复边界参见 `docs/native-graph-implementation.md`、`docs/luna-skills-guide.md`。历史报告仅是当时的证据，不作为当前实现要求。

当前视觉输入及调用边界统一遵循 [工作空间过滤与视觉判断](../../../docs/raw-decisions-20261111.md)。候选框由当前 SAM3 掩码计算，模型接收有效范围过滤 RGB；实例关联使用有日期参考与物理历史。


当前统一失败收尾遵循 [终止失败回位](../../../docs/terminal-failure-return-20261111.md)：具体恢复耗尽或未处理异常均执行显式固定收尾图，从实际关节规划回观测位姿并停止，保留原任务失败；不安全或无法确认的回位报告 blocked。

## 2026-11-12 association and planning update


Unknown-object GraspGen sampling uses one native 100-candidate pool per round, with one graph-controlled additional round. Each cuRoboV2-Fast grasp request contains the configured grasp_goalset_size ranked candidates. Fully rejected candidates are removed, later batches remain available, and the first grasp that passes all numerical checks ends planning. After a confirmed table release, voxel geometry retains its occupied cells with stale provenance until graph-controlled RGBD remapping.

Attachment follows completed Lift and Decisions holding presence and continuous execution identity inheritance. Planning and executing the entire grasp sequence use the detached robot and the official grasp-planning open-gripper model, with a separate closure check. The executing Link3/Link4 links retain the selected-target contact policy during grasp and Lift. Other robot/environment and self-collision checks remain active. Successful verification prepares carrying geometry for later transport and placement. The earlier pre-Lift virtual carrying and opening-range path sweep implementation has been removed.

Responses 每个任务只调用一次，解析或图校验失败立即结束编译。固定模板由程序展开，逐物体子图的 completion_pose 在输出 schema 中限定为空，最终回位由根任务声明。执行采用三倍时间伸长和五次 Hermite 插值。MuJoCo 关节目标以 50 Hz 更新，物理积分与跟踪误差检查按原物理步运行。

verify_grasp 在 Lift 后只调用一次 Decisions 判断是否持物，输入执行臂当前夹持接触区原始分辨率 RGB 和同次夹爪反馈，不做夹爪状态或运算，也不调用 SAM3。明确未持物允许空抓恢复；请求异常、拒答或证据不足保留 unknown。确认持物后，程序核对连续抓取记录的目标、执行臂、运行、代次、来源观测、场景及动作版本，继承抓取前已确认目标的永久 ID，再建立 attachment。identity_binding 明确标记 continuous_grasp_execution 和 visual_identity_verified=false；这属于执行过程推定身份。执行记录不一致时保留持物结果与身份未确定状态，不建立 attachment。历史参考保留用于抓取前关联与放置核验，正常主线不再单独做抓后身份题。

未知物体的 attachment 使用抓取前已发布的同次多相机 TSDF 表面体素，将每个体素的八个角点按闭爪时实际 TCP 转到夹爪坐标，再生成覆盖全部体素的碰撞球。放置足迹沿用同一组体素角点。来源绑定采集、代次、永久 ID 和体素尺寸，缺失或过期时报告失败。已知 CAD attachment 使用已配准 CAD。Lift 完成、视觉持物成立且连续执行绑定通过后建立 attachment，身份按执行记录继承。

GraspGen 候选先按真实末端 +Z 在世界坐标中的朝向筛选，允许世界 +X 前方至 −Z 下方的扇区，偏离世界 XZ 平面的左右角度默认最多 30°，控制台可调范围 0–45°。CAD 使用离线抓姿。规划完成后对实际抓取端点 FK 再核对相同朝向条件。抓取 Lift 显式指定世界 +Z 150 mm，并用公开 FK 检查每个规划插值点的横向偏移和提升高度，采用已有 IK 位置容差 5 mm。方向筛选证据、逐候选通过结果和 Lift 位移核验随规划归档。Lift 完成并通过持物核验后建立 attachment。

Decisions 输入以实际发送图片的顺序编号，每张图标明相机角色、观测 ID、代次、时刻、CURRENT/历史/静态目录角色及归一化框坐标约定。当前图片与请求采集绑定检查通过后才发送。跨相机选项描述给出 C 标签、相机和该图中的框坐标，相邻时刻 P 选项给出历史采集及对应图像位置。初始分组和 CAD 选项分别描述当前标签与目录标签。裁剪和单主体掩码元数据只列出本图主体。支撑关系问题与实际发送的 RGB 轮廓一致，放置核验提供目标和目的地的带日期参考，逐物体完成条件绑定对应目标及子任务。回放同时显示选项代码、完整选项概率分布及单独的 confidence 字段。

固定 single_arm_transfer_v1 在入口显式执行 observe_scene，capture_mode 为 reuse_current，由现有采集版本和关节反馈检查决定复用或新采集，然后进入 identify_target。模板入口要求空手、无需恢复且仍需交付。observed 与 grasp_view_ready 由入口观测建立，全部节点和失败边在展开 graph 中可见。

持物判断中，实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。state 说明接近闭合时仍可持物、背景与遮挡的区别、深色材料及历史图片角色。verify_grasp、verify_place 和失败收尾共用视觉定义。抓后身份从连续执行绑定继承，明确记录推定来源，不另发身份题。

所有 GraspGen 抓取按执行臂当前指端 TCP 选择较近的平行夹爪半转等价姿态，随后进入 IK 和完整碰撞检查。CAD 离线抓姿按库中 A/B 归属读取。

CAD 模型绑定采用 resources/cad_subject_policy.json 中的用户场景声明。已确认身份的 black/object 绑定黑色工件 CAD，blue/bin 绑定蓝盒 CAD，其余身份使用 TSDF。程序按该声明绑定模型并保留来源，FoundationPose 使用当前 SAM3 掩码和 RGBD 估计位姿。

持物和空爪题使用同次 TCP 与相机标定生成的 11 cm 夹持接触区 RGB，保留原始像素分辨率，不加入历史整幅腕图。当前图保持工作空间过滤、不依赖 SAM3、不带物体标签；投影不可用时退回完整当前腕图。失败收尾对两腕使用相同规则。历史参考仍用于实例身份与放置落点核验。

持物与空爪题的 state 同时提供当前采集的夹爪反馈，实测归一化闭合程度与全闭目标命令分别标注。0 表示机械全开，1 表示机械全闭，该值不是夹持力。实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。张开或闭合命令仅表示动作意图。实测归一化位置距机械端点 0.02 以内归为对应端点，其余归为 partially_closed。最终由 Decisions 综合图像与反馈判断，程序不做夹爪状态或运算。缺失或无效反馈明确标为不可用。夹爪反馈仅用于持物和空爪题；抓后身份记录为连续执行推定。

2026-11-14 验证修正：初始及历史描述按其 SAM3 主词及别名检索候选，再由 Decisions 确认身份；全局重建输出当前采集的全部未知任务实例 TSDF 表面。持物/空爪输入使用根据自身 TCP 与相机标定确定的当前局部 RGB，持物题不发送历史整图；局部投影不可用时退回完整腕图，默认不新增深度图。桌面落点题显示指定区域；入盒按整个目的容器验收，配置小方框仅约束释放位置。

初始编排必须声明 support_table_refs，引用 entities 中实际可见且参与支撑测量的 table；配置区域不是桌子实体。程序通过当前或缓存实测桌面多边形，唯一绑定控制台区域及支撑高度。用户明确指定的机器人左右通过 robot_relative_side 单独编译；初始候选以同帧 RGBD 标定坐标核对（+X 向前、+Y 机器人左侧、-Y 右侧），再交 Decisions 选择。此约束只用于初始定位，不限制已确认永久 ID 之后的位置。

当前有限任务边界与证据实现见 [可靠性修正](../../../docs/scoped-reliability-20261115.md)。未知物体 GraspGen 不再额外沿 Z 伸入；模型接触 TCP 显式转换到 A/B 指端，旧 grasp_insertion_m 保存值退役。当前只要求同臂覆盖来源与固定目的地、固定 CAD。物理记忆提交使用独立 PhysicalMemoryWriter，历史失败按实现/合同/子图/配置版本筛选，较早身份参考每实体最多 3 次采集。

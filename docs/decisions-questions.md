# 当前 Decisions 问题与输入（2026-11-12）

Decisions 只看必要视觉事实；每次 HTTP 一个问题，使用有效区域过滤 RGB 和按需深度。SAM3 候选框标明询问主体，过滤后像素与图像尺寸保持一致；深度的固定灰度传输编码及精度限制见 [详细方案](raw-decisions-20261111.md)。

| 用途 | 输入 | 输出 |
| --- | --- | --- |
| 跨相机关联 | 当前三相机完整 RGB，叠加编号、框和掩码轮廓，同时标明已确认的同物体视角，附各视角双向数值深度对应及已确认关系 | 对应候选 / no_match / segmentation_error / unknown |
| 初始描述对应 | 三相机完整 RGB 同时标注全部已归组实例，附单条 Responses 描述及其参考相机 | current_instance 编号 / no_match / unknown，每条描述独立选择，重复占用保留未解决 |
| 初始额外物体确认 | 同一组三相机标注 RGB，指定尚未被描述选中的实例编号 | physical_instance 二值谓词，确认后独立登记，任务归属为 unknown |
| 跨时刻关联 | 当前三相机完整标注 RGB、上次 P0/P1 实例的完整标注 RGB 和有日期操作历史 | P 编号 / no_match / reject / unknown，确认后由程序继承永久 ID |
| 历史身份恢复 | 近期对应未解决时追加更早已确认参考及档案归属，保留裁剪与独立掩码图 | 永久 ID / new_instance / reject / unknown |
| 用户声明的 CAD 绑定 | 已确认身份的类别和颜色 | 程序绑定黑件／蓝盒 CAD，随后用 SAM3 与 RGBD 配准 |
| verify_grasp | 实测夹爪开度，全闭时使用当前夹持区 RGB | 确认持物后继承抓取目标 ID |
| verify_place | 实测释放位置与夹爪开度，全闭时使用当前夹持区 RGB | 到位释放且空爪后登记选定目的地 |
| 下一目标（多个合格输入） | 带框过滤 RGB、每个 ID 的描述 | 一次 choice，优先容易抓取；单候选或用户明确顺序时不调用 |
| 当前支撑未确定 | 带框过滤 RGB、支撑候选 | 各必要支撑关系独立 predicate；数值证据明确时不调用 |
| 新物体任务归属未知 | 新主体当前原图及编译输入要求 | 一次 predicate，已接受结果任务内复用 |
| 来源余物 | 当前来源原图及输入描述 | 来源是否仍有合格输入，一个 predicate |
| 拆包结果 | 原始前后图与实际加工绑定 | 各必要加工视觉事实独立 predicate |
| 额外视觉要求/visual router | 当前原图及本题绑定 | predicate 或已声明 choices |

二值 probability >0.5 为 yes，<0.5 为 no；多选取分布最大项。合法服务选择在未提供分布时保留。confidence 只记录，没有 0.8 门槛。精确平分、拒答、明确 unknown 保留 unknown；畸形响应记录协议错误。有效 no/unknown 不重发同一冻结问题来强求成功。

数值选臂、建图范围、固定 XY 选择、放置语言许可、已知错误恢复和标准任务回执验收不调用 Decisions。语言许可与额外视觉条件由初始 Responses 编译。observation_recovery 和 failure_return 是程序 router，不发模型请求；单臂子图失败不会换臂。所有动作仍由显式有限 graph 执行。

身份题判断同一个物理实例。登记比较使用当前 SAM3 主体的对应相机 RGB、独立掩码和带日期历史参考。初始空库完成当前分组与描述确认后，由程序分配永久 ID。

## state 与 questions 的生成

state 是 input 中的结构化文字，由程序按当前问题投影观测、候选、历史参考和已确认关联。持物与空爪题的中文 state 使用 ensure_ascii=False 构造，使模型接收的文字保留直接可读的中文。question 类型、基础说明和输出结构由程序定义，主体标签、候选选项、执行臂、目标和冲突复核上下文随本次调用填入。visual router 的任务命题来自初始 Responses 编排。复用同一冻结采集时，问题与候选仍可不同。

初始描述绑定保留 Responses 描述的 reference_camera_id 与采集编号。每条描述面对全部当前实例作一次选择，核对自身外观、位置和内容物关系。多个描述选择同一实例时，这些描述保留未解决。尚未被选中的实例经过独立物体确认后进入登记，mentioned_in_task 为 false，membership 为 unknown。首次描述仍是待确认提议。新实例确认记录上一时刻比较是否真正执行及其结果，并将图片 P 标签映射到永久 ID。

跨时刻问题明确写出当前主体标签、包含该掩码的相机及分割查询提示，优先展示这些当前视图，再展示其他相机和日期参考。state.subject_camera_ids 与当前掩码相机一致。选项描述仅使用历史确认的颜色和类别，完整日期描述、位置、容器关系和动作历史保留在 state.references 中。所有当前相机图保留，未标主体的可见物体提供场景上下文。

每张图片紧邻的元数据记录实际像素尺寸、相机、采集、角色和主体框。裁剪另附原图像素范围。CAD 预览使用独立静态参考角色。回放的 decisions 页显示实际发送字节对应的图片，以及完整 state、图片元数据和 questions。

Decisions 官方单题允许 2–255 个 choice 选项，本项目接口与该上限一致，超过时明确拒绝而不截断数据库 ID。[官方 API 参考](https://developers.openai.com/api/reference/resources/decisions/methods/create)

verify_grasp 在 Lift 后读取同次实测夹爪开度。全开判为空爪，中间开度判为持物，仅全闭时由 Decisions 查看当前腕相机夹持区判断是否持物。确认持物后，将本次完成抓取的目标永久 ID 绑定到夹爪并建立 attachment。空抓可进入恢复，反馈缺失或执行记录不一致时保留待定。

verify_place 使用本次搬运已绑定的物体 ID 和目的地 ID。释放时检查机器人实测 TCP 是否到达配置释放 XYZ（位置容差 5 mm），释放后按实测开度判断空爪，仅全闭时由 Decisions 查看当前夹持区。到位、释放且空爪即登记已放到指定桌面位置或盒中，结果为 execution_assumed。回执记录目标、目的地和实际释放位置，物体最终静止位姿保留为未测量。

身份不会通过容器跟踪、三维投影或深度阈值自动继承。当前完全相同的同相机 mask 字节仅作输入去重。数值定位、几何与碰撞在视觉身份接受后独立执行。

每次请求及原始回答、转换规则、相机/采集/模态和动作绑定完整归档。历史实验中的旧阈值、投影图与多题方案只描述当时行为。
初始 Responses 读取三相机 RGB，输出实例描述与 graph。随后 SAM3 提供 candidate_id、框和相机，Decisions 完成跨相机关联和场景描述对应。当前视觉范围与历史参考范围统一遵循 [工作空间过滤与视觉判断](raw-decisions-20261111.md)。SAM3、模型与几何统一使用同帧工作空间掩码过滤后的输入，候选掩码进一步与工作空间取交集。

## 2026-11-12 association and planning update

Cross-camera choices compare labelled boundaries in three full RGB views. Temporal matching uses annotated current and historical full RGB views. Archive recovery and novelty confirmation retain detailed crops and masks, with crop coordinates relative to each crop. Temporal choices use P0/P1 labels from the previous confirmed observation. The program maps the selected observation instance to its permanent ID. Objects absent from that observation enter archive recovery with dated references. Each inheritance is archived with the previous observation ID and original model choice.

Unknown-object GraspGen sampling uses one native 100-candidate pool per round, with one graph-controlled additional round. Each cuRoboV2-Fast grasp request contains at most 32 ranked candidates. Fully rejected candidates are removed, later batches remain available, and the first grasp that passes all numerical checks ends planning. After a confirmed table release, voxel geometry retains its occupied cells with stale provenance until graph-controlled RGBD remapping.

主体选择的问题正文写明当前主体 ID、描述和实际历史参考标签。state 中提供同次已选掩码的所属主体，后续选项排除已选掩码，当前图仍保留对应轮廓。历史主体参考仅包含有该主体标注的视图。容器和内容物分别匹配各自掩码。

空爪问题以指腹近处受到夹指约束的物体定义持物。落在两指投影之间的远处桌面或盒内物体按背景处理。当前、初始及历史持物 RGB 的采集角色逐图声明。

持物判断中，实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。state 说明接近闭合时仍可持物、背景与遮挡的区别、深色材料及历史图片角色。verify_grasp、verify_place 和失败收尾共用视觉定义。抓后身份从连续执行绑定继承，明确记录推定来源，不另发身份题。

空爪判断中，实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。复用 verify_grasp 的同次反馈与历史图片角色说明。

持物和空爪题使用按同次机器人指端 TCP 与相机标定生成的当前局部 RGB，以及初始抓取前和按需提供的前次持物腕部对照。当前局部图保持工作空间过滤，不依赖 SAM3 候选，不带物体标签；局部投影不可用时退回当前完整腕图。抓前实例关联使用 SAM3 与历史参考；verify_grasp 无需 SAM3 或抓后身份题。默认不增加独立掩码图或深度图。失败收尾对两腕使用相同局部视图规则。

持物与空爪题的 state 同时提供当前采集的夹爪反馈，实测归一化闭合程度与全闭目标命令分别标注。0 表示机械全开，1 表示机械全闭，该值不是夹持力。实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。张开或闭合命令仅表示动作意图。实测归一化位置距机械端点 0.02 以内归为对应端点，其余归为 partially_closed。最终由 Decisions 综合图像与反馈判断，程序不做夹爪状态或运算。缺失或无效反馈明确标为不可用。夹爪反馈仅用于持物和空爪题；抓后身份记录为连续执行推定。

2026-11-14 验证修正：初始及历史描述按其 SAM3 主词及别名检索候选，再由 Decisions 确认身份；每次全局重建输出当前采集的全部未知任务实例 TSDF 表面。持物/空爪输入使用根据自身 TCP 与相机标定确定的当前局部 RGB，保留历史腕图；局部投影不可用时退回完整腕图，默认不新增深度图。放置判定显示执行绑定的指定区域，容器内区域同样适用。

初始编排必须声明 support_table_refs，引用 entities 中实际可见且参与支撑测量的 table；配置区域不是桌子实体。程序通过当前或缓存实测桌面多边形，唯一绑定控制台区域及支撑高度。用户明确指定的机器人左右通过 robot_relative_side 单独编译；初始候选以同帧 RGBD 标定坐标核对（+X 向前、+Y 机器人左侧、-Y 右侧），再交 Decisions 选择。此约束只用于初始定位，不限制已确认永久 ID 之后的位置。

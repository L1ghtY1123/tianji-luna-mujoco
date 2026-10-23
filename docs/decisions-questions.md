# 当前 Decisions 问题清单

2026-10-23 最终规则：VLA后无论当前Decisions能否确认黑件，双臂都必须回固定观测位姿。当前位置确认黑件时，先用同次RGBD分割测量黑件并加入保留背景场景（头部优先，深度不足依次尝试左/右腕）；未确认时不建立黑件，移除过时活动粉件层后回位。回位后重新采三RGBD、联合Decisions确认并分析全部当前清单实体，再重建全局2cm TSDF/ESDF；首次桌面cuboid继续复用。所有粉/黑抓取前都在观测位姿，禁止VLA位姿直接抓取。最后黑件入盒确认后也必须沿最新场景回初始位姿，成功回位才完成任务。



按2026-10-22当前源函数整理，供审查实际模型输入与判定边界。这里说明已实现协议，不代表本次模型、GPU、部署或完整机器人任务已通过验收。历史实验的题目与回答不能自动迁移为新协议成功。

当前拆包主线：初始清单一次Decisions；每次合格生命周期观测一次joint `associate_inventory`；桌面释放一次请求内的 `place_result` 与 `workpiece_still_held`；拆包结果一次 `unpack_result`；右盒投放一次 `current_item_added`。每个请求都包含该次CURRENT的camera 1/2/3完整原RGB；注释、裁剪、INITIAL和BEFORE只是附加输入，不替代三张原图。camera 1为右腕、2为左腕、3为头部；机器人左右不等于图片左右。同一对象在多视图或细节图中只算一次。

当前结果/候选外观条件使用官方predicate：probability≥0.8为yes，≤0.2为no，中间unknown。ID、可见性类别、清单遗漏等分类保留choice及confidence≥0.8。两种数值含义不同；不使用choice probabilities绕过confidence。原始wire回答保留，错误题数、name、type、数值或选项被拒绝。生命周期兼容字段confidence在新结果中保存predicate概率，confidence_rule明确标识，不是模型另返的置信度。历史单题choice存档按旧规则读；当前请求严格匹配predicate类型。

粉色泡棉包裹黑件是用户先验；一次完成一个工序。粉输入原ID与初始显示编号永久不变；拆包确认后黑产物建立独立ID，process记录输入/输出关系。右盒随机投放后的当前几何使用collection，不逐成员视觉重认。结果观察不运行SAM或建图；只有双臂观测位姿建立当前2cm TSDF/ESDF，运动碰撞阈值为1cm。这些几何数值不是要求RGB判定的毫米精度门。

## 1. 初始清单：initialize_inventory

入口：[DemoVisualInventory.resolve/request](../src/luna_mujoco/demo_perception.py)、[VisualInventory.resolve](../src/luna_mujoco/visual_inventory.py)。初次Responses负责生成清单和有界程序；它不是Decisions调用。随后SAM候选进入一次Decisions请求。

输入：CURRENT三原图、候选C标签/框、固定实体描述/ID与原始R参考、必要定位放大图。初次INITIAL和CURRENT来自同一冻结capture时，程序仅在observation ID及每个RGB像素均一致后说明“同帧”；框重叠只用于定位，不直接赋ID。图像包含的主体、几何提示和模型推断分开标记。

| 题名 | 实际问题 | 选项和边界 |
| --- | --- | --- |
| `identity_N` | C候选的主体对应哪个固定entity ID？容器框里的内容不改变其容器主体。 | 返回可确定的canonical ID；无关、机器人、清单外主体为reject；无法区分或唯一匹配为unknown。此题不是yes/no。 |
| `visibility_<id>` | 该非table实体是否在任一CURRENT图像中可见？完整原图均参与，不依赖SAM是否提名。 | visible需要可识别像素；unseen表示不见/遮挡；身份不确定为unknown。无需三相机都看见。 |
| `demo_unlisted_pink` | CURRENT三视图的可见粉色任务主体是否有无法对应固定源ID的额外对象？ | none表示所有可见主体都可对应；extra表示明确可见未列粉件；可见主体身份歧义或无法评估为unknown。未见箱底/箱沿、隐藏内部本身不要求unknown，none不证明隐藏处为空。机器人、反光、标注不算。 |

初始化及bind_inventory仍要求全required实体可信身份的视图并集、原粉源件的camera2候选/测量覆盖和同帧 `demo_unlisted_pink=none`、confidence≥0.8。左腕覆盖是既有源清单条件；它不把身份题限制为只能使用左腕图。不确定辅助候选只能在已满足完整覆盖和既有规则时被丢弃，不能补足缺失身份或进入地图。

## 2. 生命周期观测：joint associate_inventory

入口：[resolve_lifecycle_inventory](../src/luna_mujoco/demo_lifecycle_inventory.py)，资格与定位来自[lifecycle_candidate_scopes](../src/luna_mujoco/demo_lifecycle_tracking.py)。程序先验证两盒当前测量对应、当前阶段与执行证据，以及当前候选点的支撑范围。几何只提名角色，不替模型答身份。

一次请求包含CURRENT三原图、C候选标签及可用像素细节；有剩余粉件时，同请求加入INITIAL三原图/原编号，原身份语境在 `context.wrapped_identity`。没有剩余粉件时不发送INITIAL粉件图。INITIAL只供原粉身份比较，黑件只判断其当前外观和角色。

| 题名 | 实际问题 | yes / no / unknown 或ID选项 |
| --- | --- | --- |
| `identity_N`（角色predicate，表中yes/no/unknown是程序解释） | located in camera K的C候选是否显示预期粉件、黑件，或一个/多个黑件及其可见部分？综合全部CURRENT图。 | yes：可见主体符合该外观角色；no：机器人/容器/背景或明显不是所问工件；unknown：主体无法辨认/对应不清。部分可辨认视图可以接受。支撑范围由3D代码验证，不另要求RGB证明桌面接触。 |
| `wrapped_identity_N`（唯一原ID提议） | C候选是否仍是INITIAL参考里该canonical ID和不可变显示编号的同一原粉件？ | yes：外观、可用注册空间连续性和净化操作历史合起来支持唯一原身份；no：明确不是所问原件；unknown：竞争对应、运动矛盾、未解遮挡。无需独有纹理；数量、位置或剩余角色不能单独赋ID。 |
| `wrapped_identity_N`（多个原ID） | 该粉候选对应哪个immutable INITIAL源ID？ | 选一个可建立对应的原ID；不是原粉源主体为reject；无法唯一对应为unknown。所有原源参考都用于排除竞争匹配，不按剩余数量重编号。 |

粉候选必须同时通过自身角色和原身份题，角色predicate概率及原身份choice confidence分别≥0.8；原身份yes不能救回角色unknown/no。全部剩余原粉ID仍需被已接受的camera2候选覆盖；三视图可以共同支持这些判断。必要角色或原身份缺失时不发布部分groups；已满足必需覆盖后仍未知的辅助候选不进入几何。

集合题只要求当前右盒内一个或多个可辨黑件/局部，不能根据collection成员数要求看见每个成员，不能重新分配成员视觉ID。原粉件与独立黑产物不做跨外观“同一个实体”证明。

角色和原身份只有一个请求/缓存事务；`request_hash`、`role_request_hash`、`wrapped_identity_request_hash`均指同一完整payload。完整原始答案和各自接受集合归档。该合格路径返回 `not_reassessed/frozen_initial_inventory`，不会伪造再次全局盘点的none结论。

## 3. 生命周期资格不足的generic回退

入口仍为[DemoVisualInventory.resolve](../src/luna_mujoco/demo_perception.py)。这不是死代码：容器anchor缺失、标定/视角/深度/分割对应不成立、任一盒缺少有效测量对应或阶段资格不足，会使生命周期scopes返回None，进入一次generic Decisions。没有额外角色请求先行。

输入同样有CURRENT三原图、INITIAL参考、C/R标签、实体描述及经过净化的生命周期信息。问题为第1节的 `identity_N`、`visibility_<id>`、`demo_unlisted_pink`；生命周期路径禁用旧 `progress_<id>`。黑产物有独立fixed_entities ID；右盒collection内可见多个黑件不是额外新ID。遗漏题虽然仍名为demo_unlisted_pink，在此任务语境还检查未列黑件。当前桌面产物与已完成右盒集合必须区分；无法对应为unknown。

这一回退仍遵守generic完整性与几何门，不等价于qualified路径的not_reassessed。INITIAL粉图只用于仍包裹的原输入；不得把黑件套回粉ID或要求已完成集合逐成员识别。

## 4. 桌面释放：verify_place

合法持物、attachment、实际释放反馈和撤离完成后，在当前位置采新三RGB核验当前粉件。实际TCP反馈结合抓取时实测物体参考与attachment几何，数值校验释放XY及物体包络在固定共享区；release_position_check明确标注为开爪时刚体推导，不是释放后物体实测位姿，也不证明视觉成功。单次Decisions发送初始全部实体、活动ID、工序/生命周期记录、INITIAL三RGB/注释和CURRENT三RGB/区域图，同一请求用两个独立predicate：place_result判断主桌上可见独立粉件，workpiece_still_held判断粉件仍被手指夹持；前者概率≥0.8且后者≤0.2才通过，不要求从RGB看清底部物理接触；不再要求模型重算公制XY或把高起轮廓装进桌面投影。概率≥0.8确认、≤0.2否定、中间unknown；原始probability不叫choice confidence。源盒粉件、机器人部件不计。INITIAL是历史，先历史后当前，头部当前图最后，三视角融合；动作记录和仿真真值不能代替当前像素。成功只推进table_released，不完成工序；不SAM、不建图。旧非拆包路径仍使用choice place_result及最后一次的demo_unlisted_pink，不能把旧choice响应塞给当前predicate请求。

实现：`demo_transfer_verification.release_position_check/release_result_payload/verify_release_result`。当前分别判断“CURRENT主桌可见独立粉件”和“粉件仍被夹持”，前者须≥0.8、后者须≤0.2；不混入底部物理接触可见性。开爪前反馈数值检查区域；此估计不等于释放后RGBD重新测量位置。当前图仍须支持放置状态。

## 5. 拆包：verify_unpack

mock VLA完成后在当前自然位姿采一份新三RGB，校验该工序已确认桌面释放的BEFORE三RGB、target/owner/generation、反馈、binding、请求与图片哈希；恰有一个活动输入。一次Decisions结合BEFORE三RGB、CURRENT三RGB与主桌辅助图，用predicate unpack_result检查当前主桌出现独立黑件。三视角共同判断，盒内已完成黑件和机器人部件不计；看不清不能以动作执行记录代替。概率≥0.8确认、≤0.2否定、中间unknown。成功创建独立product ID，process关联原粉输入与黑产物；粉件原ID/编号保留为consumed，不要求视觉证明粉黑外观同一。当前结果帧及真实BEFORE图片均归档。确认后才沿保留地图回双臂观测位姿，下一次抓取前另作Decisions/SAM/建图；本结果节点不SAM、不建图。消费者重算原始概率与执行链；历史单题choice记录保留其原confidence规则，旧多题或错误线格式不能冒充当前响应。

实现：`demo_lifecycle_perception.unpack_result_questions/verify_unpacking`。当前唯一事实：“CURRENT主桌上出现独立黑件”；黑件盒内历史成员不计。

## 6. 右盒投放：verify_drop

合法持物、attachment运输、实际开爪后，一次Decisions接收CURRENT三RGB、BEFORE三RGB及同盒辅助区域图。predicate current_item_added检查指定机器人右盒中当前可辨至少N+1个黑件，N仅为此前确认完成数，初始0来自用户场景先验。概率≥0.8确认、≤0.2否定、中间unknown；不另问in_right_bin、手指分离或旧黑件身份。多视图同件计一次，允许接触但须可分辨，机器人/桌面/盒外工件不计。动作历史和历史数量不证明当前数量。BEFORE仅为历史对照，不要求证明遮挡后曾为空。看不清或遮挡无法区分数量时不推进。成功完成独立产物及process，保留原始回答、计数、反馈链，不SAM、不建图。辅助盒区来自最新已发布占据与CAD壳边界，并按各帧标定投影，不取仿真真值。

实现：`demo_lifecycle_perception.verify_right_bin_drop/drop_result_evidence`。当前唯一事实：“CURRENT指定右盒至少有N+1个可分辨黑件”。

## 7. 仍可调用的非拆包兼容节点

这些不是当前逐件拆包主线，不能称为已全部合并。

| 节点 | 输入与题目 | 接受/未知边界与调用数 |
| --- | --- | --- |
| `associate_completed_objects` | [completed_correspondence_payload](../src/luna_mujoco/demo_completed_tracking.py)：CURRENT三原图、已完成槽和C候选；`tracked_N`融合CURRENT判断该C是否为槽内独立粉件。 | yes≥0.8结合既有合法track/唯一几何scope映射原ID；no/unknown/低置信不赋ID。随后另一请求关联剩余源件/盒及遗漏；同一观测仍可能两次Decisions，hash互相绑定。无item_lifecycle才走此路径。 |
| generic `verify_transfer` | [VisualInventory.resolve](../src/luna_mujoco/visual_inventory.py)加Demo上下文：三CURRENT、INITIAL参考、待核验释放事实及区域；身份/visibility/遗漏题之外可有`progress_<id>`。 | at_destination须当前可见目标在目的区域/容器且未被持有；elsewhere为明确在外或被持有；身份/目的地/可见性不明为unknown。需要可信身份和目的地结论，均≥0.8。动作历史或区域标注不直接证明成功。DemoRunner当前verify_place使用第4节专用入口。 |

## 本次验证记录

相关389项回归通过，修改文件Ruff通过。真实Decisions使用冻结原图：place概率0.99/反例0，第二件unpack 0.93/反例0，drop 1/反例0；生产解析器接受正例、拒绝反例。当前角色与原粉ID在两份已记录scope上联合验证通过；这不是重跑SAM或完整建图。v14第二件因单题混入物理接触而停止；修正后的双条件生产题在v14第一/第二件原图分别为(0.98,0.01)/(0.99,0.00)，动作前反例(0,0)被拒绝。v15在第一次plan_grasp因候选池无可行完整轨迹停止，未到结果节点；完整两件闭环尚未通过。历史失败不改写。

API依据：[OpenAI Decisions指南](https://developers.openai.com/api/docs/guides/decisions)、[公开请求定义](https://developers.openai.com/api/reference/python/resources/decisions/methods/create)，2026-10-22核对；使用已有HTTP适配器，未改第三方SDK或算法源码。


2026-10-22 verify_unpack描述修复：CURRENT三相机仍合并为一次unpack_result判断，BEFORE明确标为拆包前且已通过桌面放置验证。黑色CAD是带凸起的不规则工件，不强制矩形；允许开爪部分遮挡，但可见主体必须能与机器人区别，且位于主桌面而非任一蓝盒。未降低0.8门限、未用动作记录代替视觉、未增加建图。v18第二件原图旧判断0.37；修改描述后同图实际Decisions为0.97，替换为拆包前粉件图的反例为0.0。原失败记录不改写。


本次放置请求另含measured_gripper_state：当前capture实测夹爪角度、限位、open_confirmed、观测ID与来源。开爪不证明位置；CURRENT RGB仍须确认桌面粉件。workpiece_still_held明确检查物理接触挂持，而非二维画面位于两指之间。当前三视图顺序1、2、3（头部最后）。桌面可改色，不再以灰色为前提。

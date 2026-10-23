# 当前 Decisions 问题清单

2026-10-23 最终规则：VLA后无论当前Decisions能否确认黑件，双臂都必须回固定观测位姿。当前位置确认黑件时，先用同次RGBD分割测量黑件并加入保留背景场景（头部优先，深度不足依次尝试左/右腕）；未确认时不建立黑件，移除过时活动粉件层后回位。回位后重新采三RGBD、联合Decisions确认并分析全部当前清单实体，再重建全局2cm TSDF/ESDF；首次桌面cuboid继续复用。所有粉/黑抓取前都在观测位姿，禁止VLA位姿直接抓取。最后黑件入盒确认后也必须沿最新场景回初始位姿，成功回位才完成任务。



按2026-10-22当前源函数整理，供审查实际模型输入与判定边界。这里说明已实现协议，不代表本次模型、GPU、部署或完整机器人任务已通过验收。历史实验的题目与回答不能自动迁移为新协议成功。

当前拆包主线：初始清单一次Decisions；每次合格生命周期观测一次joint `associate_inventory`；桌面释放同请求判断 `place_result` 与 `workpiece_still_held`；拆包同请求判断 `unpack_result` 与 `wrapper_removed`；右盒投放判断 `current_item_added`；这三个结果节点都同请求加入 `observation_sufficient`，不足时最多一次补充观测后再核验。每个请求都包含该次CURRENT的camera 1/2/3完整原RGB；注释、裁剪、INITIAL和BEFORE只是附加输入，不替代三张原图。camera 1为右腕、2为左腕、3为头部；机器人左右不等于图片左右。同一对象在多视图或细节图中只算一次。

当前结果/候选外观条件使用官方predicate：probability≥0.8为yes，≤0.2为no，中间unknown。ID、可见性类别、清单遗漏等分类保留choice及confidence≥0.8。两种数值含义不同；不使用choice probabilities绕过confidence。原始wire回答保留，错误题数、name、type、数值或选项被拒绝。生命周期兼容字段confidence在新结果中保存predicate概率，confidence_rule明确标识，不是模型另返的置信度。历史单题choice存档按旧规则读；当前请求严格匹配predicate类型。

粉色泡棉包裹黑件是用户先验；一次完成一个工序。粉输入原ID与初始显示编号永久不变；拆包确认后黑产物建立独立ID，process记录输入/输出关系。右盒随机投放后的当前几何使用collection，不逐成员视觉重认。Decisions确认放置后可用同次RGBD分割建立活动工件局部障碍；只有双臂观测位姿建立全局2cm TSDF/ESDF，运动碰撞阈值为1cm。这些几何数值不是要求RGB判定的毫米精度门。

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

## 4. verify_place

撤离完成后采当前三RGB，单次Decisions联合判断place_result（桌面独立粉件）、workpiece_still_held（仍被夹持）、observation_sufficient（当前证据是否足够确认或否定这些条件）。前两项分别须yes/no，充分性须yes；概率≥0.8为yes、≤0.2为no，其余unknown。释放XY由实际前端反馈数值核验，不能拿动作历史替代当前像素。证据不足时沿最近的保留场景回固定观测位姿，采新三图再核验一次；没有确认放置成功前不建立粉件局部碰撞。仍不足停止。明确失败不重复观察。全过程绑定原始持物、释放与场景更新链；成功仅推进table_released。确认后刷新局部粉件障碍，首次桌面及背景地图复用。

## 5. verify_unpack

mock VLA后，当前三RGB合并一次Decisions，判断unpack_result（主桌独立黑件）、wrapper_removed（活动粉件已经拆出黑件）、observation_sufficient（当前证据充分性）。BEFORE是本工序已验证的桌面粉件图；其他粉件仍在左盒不影响结果，不要求跨外观身份匹配。黑件可见时以同次RGBD测量并加入局部障碍，移除过时活动粉件层；无可用深度/不可见时按用户指定不创建黑件障碍，移除过时层后回位。无论当前结果如何，都必须双臂回固定观测位姿，采新三RGBD并重新联合Decisions。三项均yes后，以当前实测点云核对指定执行臂工作空间；通过才创建独立黑件ID，保留工序输入/产物关联。最多两次判断、一次回位；仍缺证据或不可达则停止。随后observe_scene分析当前清单，build_scene重建全局2cm TSDF/ESDF，复用首次桌面cuboid后才能抓取。

## 6. verify_drop

实际attachment运输、开爪后，单次Decisions联合CURRENT三RGB、BEFORE三RGB及同盒局部放大图，判断current_item_added（本次右盒新增黑件）和observation_sufficient。须两项yes；新增主体或轮廓可以证明增加，不要求随机堆叠后逐个数清N+1，不重认盒内个体ID，旧件移动或历史数量不证明增加。证据不足时，先发布明确标为动作推导、尚未视觉确认的盒内保守包络，再回固定观测位姿采新三图核验一次；不重建全局地图。仍不足不计完成。确认后才完成产物及工序，保留released_in_bin_and_visible_addition语义，并更新确认后的动作包络供回位。

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



## 2026-10-23 当前碰撞与补充观测规则

粉件先在原始毫米坐标X=-24处裁去负X侧凸条，再按剩余主体建立紧致长方体，外观及MuJoCo接触同为79.542675×62.787369×110mm。原文件保留，去凸条派生STL及裁切记录单独归档；场景实例仍以新包围盒中心定位，yaw和重力保持。黑件与蓝盒仍是CAD。规划全局地图仍来自真实RGBD；局部物体包围体使用实测点、支撑面和显式尺寸先验，XY余量5mm，与2cm体素分辨率及10mm轨迹净距是三个不同参数。

左右掌座各用96个较小球完整覆盖原物理盒，替代12个过大的球；去掉掌座与中间手指连杆重复世界膨胀。对称性、盒内连续覆盖和新球被旧球包含有可复算证明，见resources/curobo/gripper-tight-cover-20261023.json。附件仍默认24球、最多31球，允许与同侧全部9个夹爪link相交，仍检查相机、前臂、另一臂及环境。旧工作空间仅复用原可达格：碰撞几何缩小不使原可达见证失效，不新增可达结论。cuRobo SDK内部没有修改。

结果核验共用observation_requirements：根据当前任务的明确条件判断证据充分性，而非相机数量。清楚看到失败也属于证据充分；遮挡、相关区域出画、身份不明或视角矛盾属于不足。当前接入桌面放置、VLA拆包、右盒投放三个结果节点。三相机始终合并一个请求；补充机制最多一次固定双臂观测位姿与新采图，并再次核验，仍不充分则停止。不是任意视角搜索；初始/抓前清单身份流程保持原约束，不冒充已接入所有视觉节点。视觉不能判断公制可达性，VLA黑件需另以测得点云核对指定执行臂工作空间。

严格顺序：松爪、detach、lift → Decisions确认放置 → 同次RGBD的粉件局部障碍 → VLA移动。lift之前没有结果观测和局部建障；沿轴lift使用官方ToolPoseCriteria.linear_motion与plan_pose，规划后恢复默认criterion，包括失败分支。局部更新不触发全局TSDF/ESDF重建。正常采图预算为1+每件8次上限，包含有界补充，不是每件必采8次。

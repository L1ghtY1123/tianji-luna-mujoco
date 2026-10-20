# observe_scene

**目的与调用时机：**首次头部观测建立固定的全部物体清单，并提供搬运成员证据；后续更新原ID的可见性、测量和搬运位置证据。

**输入与前提：**observed=false；当前RGBD、自然语言任务、初始参考图、固定scene_entities、搬运清单及外层JEV选定的visual_plan。没有场景时仅分析头部；已有场景时使用三相机，任意一路可提供某个物体的证据，不要求三路同时看见。

**固定算法：**首次用头部RGB和完整指令调用Responses Structured Outputs，输出桌面实体、外观、初始RGB框、父容器、SAM3描述及required/excluded/unknown成员。程序分配固定全局ID，搬运清单引用同一ID。随后SAM3按清单描述分割；它不接收全局ID。Decisions以初始及当前完整RGB、初始ID参考图和当前临时候选框标注图（不生成或发送逐实体RGB裁剪），把候选关联到固定ID，独立判断各实体可见性和搬运去向。程序按返回的ID归并各相机掩码/点云。只向模型发送RGB及语义任务状态；深度、掩码轮廓和世界坐标不发送，目标区域仅提供本地投影的二维标注。初始清单不依赖SAM3成功与否：缺少掩码或有效深度的实体保留身份，标记geometry_status=missing / mapping_unresolved，阻止发布不完整碰撞图。最后一次视觉验收无需为无后续运动强行补全地图。初始RGB框不能冒充后续当前框。后续只复用固定清单的描述，不枚举类别和同义词。像素归并仅整理几何，不决定身份；同相机重复像素不重复计数，盒子不吞并已识别内容物的像素。桌面cuboid只建立一次。

**输出与状态：**observation、inventory_ready、visual_inventory（身份、初始成员、当前去向、模型/耗时/用量/观测ID）。成员证据写入TaskProgress；JEV仍决定技能、目标、数量分组与路线，不得覆盖明确视觉成员证据。完整清单仍经assess_task校验、编号和冻结。此技能不建图、不移动。

**视觉接口输入边界：**Responses仅接收初始头部RGB、用户指令及清单生成要求；Decisions接收初始/当前完整RGB、初始固定ID参考框图、当前临时候选框图，以及固定清单、候选SAM3描述和搬运状态。两接口均不接收SAM3像素掩码、掩码轮廓、深度图或点云；Decisions的候选框是本地从掩码计算出的二维外接矩形，目标分区是本地几何投影的二维标注，不是原始掩码或深度输入。身份问题只允许固定实体ID、reject（无关/清单外主体）和unknown（身份无法区分）；不再询问或接受invalid_mask。容器框包含内容物不影响按容器主体关联。掩码尺寸、有效深度、像素归属及碰撞几何完整性仍由现有本地流程检查，这不等于已证明SAM3每个像素的语义正确。

**数值与失败：**两接口均为官方gpt-6-luna。首次/v1/responses使用严格JSON Schema（1–64个可见实体，最多6000输出token，reasoning=low）；不把结构正确等同视觉准确。每次/v1/decisions最多200个封闭问题、128张图，choice confidence至少0.8，否则unknown；身份错误证据同时记录candidate、camera_id、模型原始choice/confidence及minimum_confidence，保留完整answers。默认超时60秒；同capture同输入的成功和错误均缓存，不自动HTTP重试或回退旧RGBD关联。Responses结构/服务失败、Decisions未知身份/服务失败都保留证据，needs_attention停止，不自动recover。不可区分的同外观实体不强猜ID。

正常RGBD预算2G+1（初始头部1、每轮抓前三相机G、每轮放后三相机G），G=冻结required数量×阶段数；恢复采集独立8次。初始化额外调用一次Responses；Decisions随全部2G+1次观测批量调用，不运行连续跟踪；同帧分析复用缓存。JEV决策预算仍为max(400,40G+80)。

SDK/vendor只读；盒底、放置区域和碰撞几何仍来自注册深度。旧几何身份匹配与重编号分支已删除；测试可注入Decisions结果替身，但生产观测必须具有Decisions结果。2026-10-19已用真实头部RGB验证Responses；Decisions新输入在线请求被tokens限流429拒绝，尚未通过新版本整轮搬运及发布验收。

**外层视觉参数：**`Runner.select_arguments` 必须先调用 `Decisions.choose_visual_plan`，由外层JEV在skill_start前为每个尚未分析的checkpoint选择initialize_inventory、associate_inventory或verify_transfer。visual_plan是有限映射；build_scene最多列出原位补图与抓前观测两个checkpoint。每项只消费一次；缺少、skip或不匹配的用途均拒绝。Perception再次检查source=jev和用途；缺少Decisions结果明确失败，不启用距离、颜色、掩码或动作预测身份回退。同帧已完成分析及已确认释放后的地图重试复用缓存，不增加图片请求。

**桌面范围与图片证据：**初始Responses仅看头部RGB与用户指令，列出指定桌面、其上的容器、可见内容物、障碍和目标区域，排除桌外实体。Decisions使用初始及当前完整RGB、初始固定ID参考框和当前候选临时编号框；不生成或发送初始实体及当前候选的RGB裁剪。候选框来自SAM3掩码的外接矩形；不是把掩码轮廓或深度喂给模型。目标分区的二维轮廓由本地测量和标定投影，标注不替代当前可见证据或碰撞几何。未关联候选不预标猜测的全局ID。

**统一编号：**全部实体使用一份 scene_entities 全局 ID；搬运清单引用其中相同 ID。冻结 required 对象的 display_id=1..N 后，盒子、障碍、目标区域和桌面依初始清单顺序续编号；后续观测不重排、不新增实体。槽位仍只使用搬运编号，非搬运编号不改变 N。

初始Responses成功后发生SAM3/Decisions失败时，PerceptionError携带成功清单和参考图，由Runner在其状态线程保存并生成已有语义框标注，再向上返回原错误。后续显式观测按固定清单使用三相机更新，不再次分配ID。初始RGB框编号复用既有分行规则，不因框顶边高度略有差别改变左右顺序。

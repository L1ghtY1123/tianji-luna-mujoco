---
name: luna-policy-isolation
description: Use when modifying, debugging, extending, or reviewing tianji-luna-mujoco task planning, robot skills, visual decisions, objects mapping, placement, or cuRoboV2-Fast integration. Preserves the ownership boundary between Responses, Decisions, and physical skills.
---

只有双臂及三相机采集时的实测关节均在配置观测位姿 1° 范围内，才建立全局碰撞场景。其他位姿更新局部场景。局部更新关联同一实验、同一代次的上一版地图，补入当前可见的障碍和空间体素。未见且无移动记录的物体沿用原碰撞层。操作或观测已使位置失效的物体保留旧障碍并等待重测。首次局部观测只初始化部分地图。

# Luna 任务策略隔离

本 skill 指导 Codex 开发，不是机器人运行时能力。先阅读项目 [AGENTS.md](../../../AGENTS.md) 和 README；AGENTS 是当前要求的唯一合同，本文不累积历史覆盖条款。不使用 JEV runtime 或 sub-agent。

| 层 | 职责 | 边界 |
|---|---|---|
| Responses | 全场景提案、独立任务要求及完整图，可绑定固定子图接口或编写自定义 body，模板与显式路由组由程序展开，每个任务只允许一次 Responses，校验失败即终止 | 归档原始输出与完整校验结果，通过后执行 |
| Decisions | 必要视觉事实；已知状态与数值恢复由程序处理 | 有类型输出，按概率最大项取值，confidence 仅归档；不把动作提议当事实，不替代数值安全检查 |
| Memory | 时间记忆数据库保存实例和历史，空间记忆保存碰撞场景及其版本和有效性 | 原始观测、测量缓存及配置区域归执行层；时间记忆拒绝任务进度和当前碰撞有效性字段 |
| Runtime | 按 edge 执行、永久 ID 与参数绑定、事实版本、账本和已声明路径审计 | 不暗中移动、观测、换臂、建图或重播成功动作 |
| Robot skills | 选中能力内部算法、传感器与机器人接口 | 可固定内部步骤及真实前提，不串接隐藏任务流程 |

## 修改检查

实例关联消费同次完整三相机 RGBD。任务必须含场景段。Responses 仅定位声明的物体并输出清单、用户选择条件和目的地区域。SAM3 文本/框提示产生候选；同图近似掩码先去重，Decisions 判断部件归属。头相机按声明检索，两腕各用可见框或头部深度投影检索。程序分别检验头部与每只腕部的双向对应，必要时补充 CAD 位姿与深度。冲突及单独腕部观测保持待定。历史重识别比较相关 ID。已确认的同一冻结采集可以复用身份。有历史记录的每次新采集都由 Decisions 结合带日期的参考、当前度量及实际动作判断。程序先登记物理 ID，再由 Decisions 按原始指令从本帧相容的已登记 ID 中选择目标、来源、支撑和目的地。角色待定时保留已登记实例。详见 docs/rgbd-grounding.md。

1. 追踪实际入口、contract、handler、模型输入、结果消费方与回放。只描述真实能力和前提，不用命名把旧 steps 包装成 graph。
2. graph 只接受当前 version=2；共享 subgraph 可循环处理运行时选中的输入，物体记录保存身份与历史。独立任务 subgraph 为可选表达。node ID 是控制节点，CAD asset ID 是模型，均不替代永久实例 ID。
3. 修改 questions 时分别核对 CAD 选择、实例身份、可见性、支撑、目标区域与全局完成。没有参考图片就不能声称图片存在；三相机证据可在公共输入或 question.instructions 内，日志需完整计数。
4. verify_place 沿用 verify_grasp 的持物与目标身份记录，默认搬运中持续持物，不新增搬运中核验。释放后先读实际关节位置：全开为空爪，中间位置仍持物；仅全闭时调用 object_held 判断。实测空爪后，程序可用同帧多视角 SAM3、标定 RGBD、抓取前尺寸和释放前空桌面，直接确认唯一目标已进入指定桌面区域并接触该桌面，记录 measurement_hash。几何证据不足或目标为容器时，由 Decisions 判定 direct_success、inferred_success、failure、unknown。桌面与入盒统一验收实际到达配置释放 XYZ 并张开夹爪；控制台小方框只约束机器人释放位置，不要求物体最终落点位于小方框内。直接确认表示看见物体放在指定桌面支撑或进入指定盒子。推测成功要求已确认持有目标、实际到达配置释放 XYZ（现有位置容差 5 mm）并释放、当前空爪，且没有交付失败的可见反证，不要求一定被遮挡。命令发出不等于实际到达；缺失绑定证据为 unknown，明确落在目的地外为 failure。当前三相机 RGB、释放前 RGB 与日期身份参考用于放置结果题，不投影释放小方框作为视觉验收边界。盒内同形物体不要求区分新旧，可依据执行证据选择推测成功。保留原始回答、执行推定身份、placement_outcome 与回执；推测入盒不伪造个体当前位姿，confidence 仅归档。
5. 批处理每件交付后优先在当前位置重新观测并选择当前输入；只有程序确认必要信息缺失时才沿 graph 回观测位姿补观测。all/count 使用共享子图，初始数量不限制运行时成员。all 以最新观测确认来源盒已空结束本批，count 按请求数量验收。最终双臂回位由实际关节反馈验收。VLA 起始姿态仍属于加工技能前提。
6. CAD/TSDF/cuboid 绑定永久 ID；SAM3 生成候选，Decisions 匹配目录，FoundationPose 估计位姿。当前位姿来源和坐标有效即直接使用，不恢复 ICP 或吻合门槛。仿真真值不进入感知或规划。
7. 放置 XY 使用控制台正方形中心，旋转保持实际抓取姿态。末端 Z = 实测桌面或盒口最高点 Z + 物体高度 + 控制台高度余量。物体高度来自同次 CAD/TSDF 几何竖直范围及来源支撑，独立于碰撞球。默认余量 20 mm，含为保守附着包络增加的 10 mm。若抓取 TCP 高于物体顶部，使用较大的 TCP 到来源支撑面偏移避免低估。实际物体底面另按抓取偏移校验，假设搬运中物体不滑移。碰撞几何独立检查目标及轨迹，不搜索目标高度。选臂时用可见高度估计，抓取后用已绑定 CAD/TSDF 再检查同一臂。
8. unknown 保留为真实 graph 结果，沿已有 edge 或已声明程序恢复规则处理；只有尚未解决的图像事实才交 Decisions，不自动转 yes。恢复优先在实际当前关节位置重新观测、重计算与规划；程序确认信息缺失后才回配置观测位姿补观测，配置回位无需额外视觉许可题，保留空手、地图和运动检查；额外腕部视角才需要视觉缺失判断。相同输入不重复观测。恢复所需的新采集由 graph 显式安排具有 captures_observation 权限的技能。
9. 删除替代实现时一并清理专属测试、依赖、prompt、前端和部署文件；迁移当前有效回归，不靠兼容分支保留旧框架。第三方 SDK 保持只读。涉及 cuRobo 的技能统一调用 cuRoboV2-Fast v0.9.0 服务及其公开 API。
10. 按 AGENTS 的 CPU 限额完成针对性测试、全量必要检查与真实仿真；报告成功与未覆盖的边界，不以健康接口替代任务验证。

模型 prompt、技能合同和观测恢复边界参见 `docs/native-graph-implementation.md`、`docs/luna-skills-guide.md`。历史报告仅是当时的证据，不作为当前实现要求。

当前视觉输入及调用边界统一遵循 [完整视野与视觉判断](../../../docs/raw-decisions-20261111.md)。候选框由当前 SAM3 掩码计算，模型接收完整视野 RGB；实例关联使用有日期参考与物理历史。


当前统一失败收尾遵循 [终止失败回位](../../../docs/terminal-failure-return-20261111.md)：具体恢复耗尽或未处理异常均执行显式固定收尾图，从实际关节规划回观测位姿并停止，保留原任务失败；不安全或无法确认的回位报告 blocked。

## 2026-11-12 association and planning update


Unknown-object GraspGen sampling uses one native 100-candidate pool per round, with one graph-controlled additional round. Each cuRoboV2-Fast grasp request contains the configured grasp_goalset_size ranked candidates. Fully rejected candidates are removed, later batches remain available, and the first grasp that passes all numerical checks ends planning. After a confirmed table release, voxel geometry retains its occupied cells with stale provenance until graph-controlled RGBD remapping.

Attachment follows completed Lift, confirmed holding and the target ID bound to this grasp. Planning and executing the entire grasp sequence use the detached robot and the official grasp-planning open-gripper model, with a separate closure check. Both Link3 fingers and both Link4 links of the executing gripper may contact the selected target during grasp, closure and Lift. The palm and the other arm retain their target checks. Other robot/environment and self-collision checks remain active. Successful verification prepares carrying geometry for later transport and placement. The earlier pre-Lift virtual carrying and opening-range path sweep implementation has been removed.

Responses 每个任务只调用一次，解析或图校验失败立即结束编译。固定模板由程序展开，逐物体子图的 completion_pose 在输出 schema 中限定为空，最终回位由根任务声明。执行采用三倍时间伸长和五次 Hermite 插值。轨迹经 Base 执行，关节与夹爪到位以 Base 实测回执为准。

抓取从目标沿末端 −Z 后退 150 mm 处接近，沿末端 +Z 进给，Lift 沿固定世界 +Z 提升 150 mm。放置从释放点正上方 150 mm 沿世界 −Z 下降，开爪后沿世界 +Z 上撤 150 mm。

verify_grasp 在 Lift 后读取同次实测夹爪开度。全开判为空爪，中间开度判为持物，仅全闭时由 Decisions 查看当前腕相机夹持区判断是否持物。确认持物后，将本次完成抓取的目标永久 ID 绑定到夹爪并建立 attachment。空抓可进入恢复，反馈缺失或执行记录不一致时保留待定。

未知物体的 attachment 使用抓取前已发布的同次多相机 TSDF 表面体素，将每个体素的八个角点按闭爪时实际 TCP 转到夹爪坐标，再生成覆盖全部体素的碰撞球。放置足迹沿用同一组体素角点。来源绑定采集、代次、永久 ID 和体素尺寸，缺失或过期时报告失败。已知 CAD attachment 使用已配准 CAD。Lift 完成、持物判定成立且连续执行绑定通过后建立 attachment，采用本次抓取绑定的目标 ID。

GraspGen 候选先转换到实际 Tianji TCP。保留末端 +Z 从机器人正前到正下的整个 90° 扇区，允许相对前下平面的左右偏角至 45°。夹爪绕自身 +Z 的半转不改变接近方向。CAD 使用离线抓姿。可行性由 cuRobo 的 IK、机器人自身及环境碰撞、实际 FK 与目标位姿误差检查决定。抓取 Lift 仍沿世界 +Z 150 mm，并用公开 FK 检查插值点的横向偏移和提升高度。Lift 完成并通过持物核验后建立 attachment。

Decisions 输入以实际发送图片的顺序编号，每张图标明相机角色、观测 ID、代次、时刻、CURRENT/历史/静态目录角色及归一化框坐标约定。当前图片与请求采集绑定检查通过后才发送。跨相机选项描述给出 C 标签、相机和该图中的框坐标，相邻时刻 P 选项给出历史采集及对应图像位置。初始分组和 CAD 选项分别描述当前标签与目录标签。裁剪和单主体掩码元数据只列出本图主体。支撑关系问题与实际发送的 RGB 轮廓一致，放置核验读取本次执行绑定的目标和目的地，逐物体完成条件绑定对应目标及子任务。回放同时显示选项代码、完整选项概率分布及单独的 confidence 字段。

固定 single_arm_transfer_v1 在入口显式执行 observe_scene，capture_mode 为 reuse_current，由现有采集版本和关节反馈检查决定复用或新采集，然后进入 identify_target。模板入口要求空手、无需恢复且仍需交付。observed 与 grasp_view_ready 由入口观测建立，全部节点和失败边在展开 graph 中可见。

持物判断中，实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。state 说明接近闭合时仍可持物、背景与遮挡的区别、深色材料及历史图片角色。verify_grasp、verify_place 和失败收尾共用实测位置优先、全闭才看图的规则。持物后沿用本次抓取的目标 ID。

所有 GraspGen 抓取按执行臂当前指端 TCP 选择较近的平行夹爪半转等价姿态，随后进入 IK 和完整碰撞检查。CAD 离线抓姿按库中 A/B 归属读取。

CAD 模型绑定采用 resources/cad_subject_policy.json 中的用户场景声明。已确认身份的 black/object 绑定黑色工件 CAD，blue/bin 绑定蓝盒 CAD，其余身份使用 TSDF。程序按该声明绑定模型并保留来源，FoundationPose 使用当前 SAM3 掩码和 RGBD 估计位姿。

持物和空爪题使用同次 TCP 与相机标定生成的 11 cm 夹持接触区 RGB，保留原始像素分辨率，不加入历史整幅腕图。当前图保持原始像素、不依赖 SAM3、不带物体标签；投影不可用时退回完整当前腕图。失败收尾对两腕使用相同规则。历史参考用于跨时刻实例关联。

持物与空爪题的 state 同时提供当前采集的夹爪反馈，实测归一化闭合程度与全闭目标命令分别标注。0 表示机械全开，1 表示机械全闭，该值不是夹持力。实际全开判未持物；动作结束后未全开且未全闭判持物；全闭不能单凭位置判断，要看 CURRENT 两指之间是否有材料，包含薄物、形变和深色物体。张开或闭合命令仅表示动作意图。实测归一化位置距机械端点 0.02 以内归为对应端点，其余归为 partially_closed。程序直接判定全开与中间位置，仅全闭时由 Decisions 看图判断；程序判定独立记录来源，不伪造模型回答。缺失或无效反馈明确标为不可用。夹爪反馈仅用于持物和空爪题；持物后沿用本次抓取的目标 ID。

初始与后续观测均通过 RGBD 关联链，提案框不直接登记身份；全局重建输出当前采集的全部未知任务实例 TSDF 表面。持物/空爪输入使用根据自身 TCP 与相机标定确定的当前局部 RGB，持物题不发送历史整图；局部投影不可用时退回完整腕图，默认不新增深度图。放置核验使用配置 XYZ 和空爪回执。

初始编排必须声明 support_table_refs，引用 task_selectors 中实际可见且参与支撑测量的 table；配置区域不是桌子实体。程序通过当前或缓存实测桌面多边形，唯一绑定控制台区域及支撑高度。Responses 保留方向、序号和用户原话。Decisions 在当前标注图和基座测量上判断对应的永久 ID，机器人左为 +Y、右为 −Y。证据不清时保持未知，特定目标绑定后可跨位置沿用。

当前有限任务边界与证据实现见 [可靠性修正](../../../docs/scoped-reliability-20261115.md)。TSDF 抓姿先将模型接触 TCP 转为执行臂 RAG 指端，再沿局部 +Z 应用 rag_grasp_insertion_m（默认 10 mm，可调 0–20 mm），在 IK 前用实测表面核对开口与指垫。当前只要求同臂覆盖来源与固定目的地、固定 CAD。物理记忆提交使用独立 PhysicalMemoryWriter，历史失败按实现/合同/子图/配置版本筛选，较早身份参考每实体最多 3 次采集。

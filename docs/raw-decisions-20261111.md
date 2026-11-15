# 工作空间过滤与视觉判断

## 图像范围

有效掩码固定为同次采集的工作空间掩码。RGB 保留工作空间内真实颜色，深度保留工作空间内有效测量，区域外置零。图像尺寸和相机标定保持一致。

SAM3、Responses、Decisions 和几何都使用工作空间过滤输入。SAM3 输出进一步与工作空间掩码取交集。新实例和历史实例都只在工作空间内参与关联。

每次采集按当前标定重新投影离线工作空间，当前分割掩码只能缩小有效范围。未知或遮挡沿 graph 的显式恢复处理。

原始 RGBD 继续归档。数据库保存源采集有效掩码与确认的实例掩码，历史参考按该采集范围恢复。旧记录缺少工作空间掩码时，按其深度和带日期标定重新投影工作空间。缺少投影条件时将参考标为范围不可用并屏蔽像素。

参考输入按采集 ID 和代次标注 SAME_CAPTURE 或 DATED_REFERENCE，同采集的其他相机参考不会标成 BEFORE。

初始登记保存已有的深度表面位置测量及观测编号，后续关联读取这份物理历史。位置是测量输入，程序没有按距离阈值分配身份。

实例关联每题保留当前三相机和初始三相机 RGB。参考加载与相机选择不会根据是否有实例掩码而删去某一路。已登记框只画在有真实定位证据的相机里。后续出现的新实例仍可追加其带日期参考。

同类物体共享一个主分割词。旧记录中同类物体的不同分割词收为有序备用词。每个相机先调用主词，SAM3 返回任何掩码就停止，未返回掩码才依次尝试备用词。深度不足、身份 unknown 或服务错误均不触发换词重问。工作空间外观保留、初始定位和后续观测共用该顺序。实例关联入口已删除 IoU/NMS 去重。

## 初始编排

初始 Responses 从三相机过滤 RGB 理解整个可见工作空间，输出实例描述、参考相机、分割词、任务要求和完整 graph。mentioned_in_task 区分任务提及和未提及的物体，membership 独立表示任务输入资格。reference_camera_id 仅是描述锚点。SAM3 在三个相机分别按类分割，候选仅有采集内临时编号。Decisions 看三张带框 RGB，将候选对应到临时场景 ref。全部跨相机关联与历史 ID 比较结束后才分配新永久 ID，保存各已确认相机的掩码。单相机可见物体也允许登记，缺失视角不伪造对应。

初始登记与后续观测共用 current_instance_groups。初次按任务描述选中当前掩码，已有 ID 先用已确认历史参考选择当前掩码，再对选中的主体关联其他相机。程序优先比较候选较完整的相机。每题使用三相机带标签 RGB 和掩码轮廓，携带当前已确认同物体视角及双向深度对应。已确认关系可通过传递性、同相机唯一性和可靠几何排除约束推导，保留程序依据。发生直接冲突时反向比较相机对，归档原始选择和复核来源，未解决则沿 graph 恢复。已知主体沿确认的历史对应继承永久 ID，近期参考不足时加载更早参考。新增任务输入单独匹配档案，确认新身份后统一登记。

历史关联使用数据库最近已确认的多相机参考、初始三相机参考、当前实测位置及操作历史。当前归组只使用同次标定与深度；历史位置作为有日期证据，移动后不把旧位置投影当当前匹配。成功 observe_scene 沿现有数据库事务更新最近参考和所有已确认视角掩码。confidence 仅归档。

## 判断分工

- 实例关联使用当前带框过滤 RGB、有日期的实例参考、当前深度测得的表面位置及数据库物理历史，选择已有 ID、new_instance、reject 或 unknown。
- 多个合法目标由 Decisions 根据当前可见抓取面与遮挡选择下一目标。显式用户顺序、唯一候选、任务资格由程序处理。选臂和可达性继续由程序核对。
- verify_grasp 在 Lift 后仅由 Decisions 判断是否持物。持物题使用执行臂当前腕 RGB 和初始抓取前同腕 RGB。明确未持物允许空抓恢复，请求异常、拒答或证据不足保留 unknown。确认持物后，由第二个 Decisions 问题比较当前 RGB 与目标永久 ID 的已确认历史标注参考。目标身份通过后建立 attachment，身份失败保留持物及身份核验结果。终止收尾同样由 Decisions 综合图像与同次夹爪反馈判断持物。历史 RGB 带相机、观测 ID、代次和时间，两题复用同一次采集。
- verify_place 先由 Decisions 使用当前、初始抓取前和前次持物的同腕 RGB 判断空爪。视觉确认空爪后，place_result 在 direct_success、inferred_success、failure、unknown 中选择。直接确认成功要求当前清楚识别目标位于指定区域。推测成功要求目标因堆叠或容器遮挡不可见、先前已确认持有目标、实际在指定区域释放、当前确认空爪，且图像未发现区域外掉落证据。缺失必要证据返回 unknown，明确反证返回 failure。两类成功均完成交付，数据库、回执和回放保留 placement_outcome。推测成功保留容器成员关系与占据范围，个体当前位姿继续标为未测量。位置题使用当前三相机 RGB、释放前 RGB 和带日期目标及目的地参考，state 提供已验证动作链。配置子区域以 PLACEMENT_REGION 投影到当前 RGB。原始回答与程序验收结果独立归档，confidence 仅归档。
- CAD 首次检索使用带框过滤 RGB 与静态 CAD 预览。掩码继续用于 FoundationPose、点云、TSDF 和碰撞几何。
- 已知状态恢复、建图范围、单臂可达性、数值碰撞和反馈验收由程序执行。同臂重规划一次，耗尽后进入显式失败收尾。

Decisions 每次一个问题，按最大概率选取，confidence 仅归档。平分、拒答和明确 unknown 保留。模型输入中的候选框由程序根据 SAM3 掩码计算。初始描述对应额外提供独立二值掩码和 SAM3 查询，帮助区分容器表面与框内的内容物；描述竞争时加入其他当前实例复核。

## 本轮验证

记录位于 data/runs/workspace-visual-policy-20261111。静态检查、冻结真实接口与完整运动验收分别记录。

## 历史验证

回归在 `data/runs/raw-decisions-20261111`：无损原始像素、独立模态、实际 Decisions 边界、三相机身份选择、raw 放置输入、拒绝非视觉 scope、正常及失败恢复、程序路径零模型调用、固定子图完整拓扑与前提。真实模型请求、部署与完整机器人执行的实际结果另存该目录，分别报告，不以静态通过替代闭环任务成功。

冻结真实接口复测：空夹爪 no（0.00）、用户原图泡棉夹持 yes（0.87）、粉件与蓝盒身份选择正确（0.68 直接采用，无 0.8 门槛）。成功历史实验的冻结初始输入编译通过，使用 2 次 Responses、143.336 秒、展开 144 节点，2 个程序 router、0 个视觉 router，没有额外重复完成验收。

历史放置执行回执使用已退役的物体参考点格式，当前数值前提正确拦截，未调用模型；不迁移或伪造反馈。另以同一冻结图片仅测试视觉 payload：最初缺少目的支撑物描述时返回 no；补充数据库现有“白色桌面”描述并明确旧位置只是历史参考后，返回 yes（0.96）。保留修改前后的独立原始请求；没有裁剪、标注、投影或重复同一有效问题。该检查证明视觉输入链路，不能代替当前执行回执或完整机器人闭环。完全遮挡的堆叠物和黑色工件夹持尚未完成真实样本验收。

## 关联证据与存储

物体清单和单物体历史均查询同一时间记忆数据库。当前帧跨相机关联、近期历史匹配、追加旧参考后的身份恢复分别归档。掩码轮廓保留物体颜色，并提供带像素来源的原色裁剪。深度对应使用当前默认 10 mm 标定与深度误差预算，至少 30 个有效样本。对源物体每个本次已确认视角与候选分别做双向投影，单独报告遮挡和自由空间不一致。任一方向形成可靠冲突就排除候选，无共同可见表面保留无证据状态。当前 RGB 画出这些已确认视角的标签，state 携带已确认关系及各方向数值来源。

某条模型回答为 unknown 时，程序可以用已确认对应的传递关系与同相机独立实例约束推导其归属或排除关系。原始回答保持在审计记录，推导使用独立来源字段。尚有可行竞争归属时继续保留 unknown。

跨相机冲突使用同帧的反向候选比较，归档被替代的问题与复核请求哈希。同次已选中的主体掩码从后续身份选择的选项中排除，图片保留其轮廓，state 提供所属主体。历史选主体仅提供含主体标注的参考视图。候选唯一性已排除部分选项时，可用缩小后的候选集和新增约束复核尚未解决的对应。

## 2026-11-12 association and planning update

Cross-camera choices compare labelled source and candidate masks, original-color crops and three-view context. Crop coordinates refer to the crop itself. Temporal choices use P0/P1 labels from the previous confirmed observation. The program maps the selected observation instance to its permanent ID. Objects absent from that observation enter archive recovery with dated references. Each inheritance is archived with the previous observation ID and original model choice.

Unknown-object GraspGen sampling uses one native 100-candidate pool per round, with one graph-controlled additional round. Each cuRoboV2-Fast grasp request contains at most 32 ranked candidates. Fully rejected candidates are removed, later batches remain available, and the first grasp that passes all numerical checks ends planning. After a confirmed table release, voxel geometry retains its occupied cells with stale provenance until graph-controlled RGBD remapping.

持物和空爪题先对当前腕部 RGB 做任务目标外观提示的 SAM3 分割，将工作空间内候选的 C 编号、框和轮廓叠加到 RGB。候选不绑定永久 ID，Decisions 同时检查未标注的可见材料。SAM3 无匹配或服务错误保留原 RGB 交 Decisions，分割结果不能直接决定持物或空爪。历史参考保持原采集角色，身份题复用同帧腕部候选轮廓，与当前多视角 RGB 及已确认目标参考一起判断身份。当前默认不增加独立掩码图或深度图。失败收尾对两腕执行同样的候选标注。

持物与空爪题的 state 同时提供当前采集的夹爪反馈，实测归一化闭合程度与全闭目标命令分别标注。0 表示机械全开，1 表示机械全闭，该值不是夹持力。实际完全张开足以判定未持物，张开命令仅表示动作意图。夹爪闭合或接近全闭时，依据图像中的物体材料判断持物，薄物和受压软物体仍可能被夹住，空夹也可全闭。最终由 Decisions 综合图像与反馈判断，程序不做夹爪状态或运算。缺失或无效反馈明确标为不可用。目标身份题使用图像与历史参考，夹爪反馈仅用于持物和空爪题。

# build_scene

**目的与调用时机：**每轮抓取前在一次调用内完成双臂观测移动、同步感知关联与碰撞重建。

**输入与前提：**已观察、inventory_finalized=true、destination_ready=true，目标与阶段臂已绑定、scene_ready=false、grasp_completed=false 且无恢复；必须空手，完整清单已冻结，目的地和阶段臂已绑定。纯 grasp 的 destination_ready 表示不需要放置。输入冻结参数、目标 ID、现有场景和反馈。

搬运路线必须已有完整清单的全阶段目的区域引用；双段路线的最终盒未识别或底面区域未确认时，不能开始第一段双臂观测运动。

**固定算法：**空手时依次：首先使用放后更新的当前全障碍物图；若无可用图，首次用初始头部capture、已有场景用当前位置三相机新capture建立包含全部 SAM3 实体的 observation 图（target_id=None、排除列表空）→用该图规划并复核双臂同步轨迹→双臂到配置观测位姿→采集一次同步 RGBD→SAM3候选分割、Decisions固定ID确认及RGBD几何提取→复用合格静态桌面 cuboid→过滤机器人/桌面以下深度和体素→为所有实例统一重建 20 mm TSDF/ESDF；盒子网格 Z 采样中心仍对齐实测底板→只排除待抓目标并发布场景。阶段内无 JEV 插入。持物状态禁止调用本技能；普通放置、standalone place 和 retry_place 只复用仍有效的抓前场景，不采新帧、不重建场景。缺少有效保存场景时返回前置条件证据，由 JEV 决定后续动作。

**输出与状态：**views_ready、scene_ready、scene token、当前帧/反馈/参数绑定、实例与缺失证据；旧轨迹及碰撞评估失效。

**失败含义与边界：**任一运动、感知、关联或建图阶段失败即返回已完成阶段及原因，不自动补观测或选恢复。已放置/历史对象未关联到、声明障碍缺失或当前目标无有效深度则拒绝语义图。实际位置须与本次观测位姿匹配；不补历史/预测体素。

## 地图范围与本轮验证

静态桌面 cuboid 仅初始头部拟合，顶面不抬高；后续相机集合、capture 与无关参数变化均复用。初次向下延伸上限0.30 m，碰撞时缩短推导厚度；最低1 mm仍冲突则报错。后续碰撞或几何/硬间距参数失配只返回错误，保留原模型；Reset/应用场景或头部标定变化才启动新生命周期。普通实例和盒子的 TSDF/ESDF 均为 20 mm，盒子网格 Z 采样中心对齐实测底板，仅改变采样网格、不改变原始深度和几何；ESDF 单侧 halo 为 max(机器人球半径, attachment 半径上界)+max(实际激活距离, hard clearance)+2 cm，向外按 20 mm 体素尺寸对齐，不删减碰撞查询覆盖。

head_all_instances（初始头部）、all_instances（当前位置三相机）和selected_instances（观测位姿三相机）均仅保留SAM3实体和初始桌面；前两者不排除任何物体，后者只排除当前目标；持物不重建图。已放物体未见、关联未解或声明障碍缺少独立实测候选时拒绝发布。三路相机不必同时看见目标。

本轮项目适配器使用独立 Mapper/独立 Torch stream，最多 4 实例并发；各 Mapper 每 capture 重置，缓存仅复用分配与编译，不能在本 capture 内被两个实例共用。实例局部坐标积分后返回世界坐标；compact ESDF 保留实测范围、原 TSDF 相位和完整 halo；确认背景为空的语义图不上传空背景 grid。全部流完成且各实例成功后发布 token，失败保留阶段和清理状态。录制数据已验证局部/世界坐标ESDF一致，整体平移可复用编译；首次及完整skill不保证提速，详见当前验证记录；SDK/vendor不修改。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。
实例Mapper按真实相机批次配置num_cameras：初始头部1路，已有场景更新3路，抓取准备使用本次实际相机批次。缓存按相机数隔离，不能给单路补虚构腕部帧。

固定清单ID只能由Decisions根据完整RGB、初始ID参考框、当前候选外接框及语义清单确认；掩码与注册深度只在本地生成候选定位和实测几何，不发送给Decisions。身份问题不评判掩码质量，容器框包含内容物不构成拒绝理由；身份不明确仍返回unknown。缺少外层JEV用途或Decisions结果即失败。三相机更新使用各视图证据的并集，不要求每个相机都看到每个实体。身份缺失或歧义导致本技能失败时，runner 保留原错误及匹配证据，置 needs_attention 并结束本轮；不再进入自动 recover 循环。

**外层视觉参数：**`Runner.select_arguments` 必须先调用 `Decisions.choose_visual_plan`，由外层JEV在skill_start前为每个尚未分析的checkpoint选择initialize_inventory、associate_inventory或verify_transfer。visual_plan是有限映射；build_scene最多列出原位补图与抓前观测两个checkpoint。每项只消费一次；缺少、skip或不匹配的用途均拒绝。Perception再次检查source=jev和用途；缺少Decisions结果明确失败，不启用距离、颜色、掩码或动作预测身份回退。同帧已完成分析及已确认释放后的地图重试复用缓存，不增加图片请求。

**桌面范围与图片证据：**初始Responses仅看头部RGB与用户指令，列出指定桌面、其上的容器、可见内容物、障碍和目标区域，排除桌外实体。Decisions使用初始及当前完整RGB、初始固定ID参考框和当前候选临时编号框；不生成或发送初始实体及当前候选的RGB裁剪。候选框来自SAM3掩码的外接矩形；不是把掩码轮廓或深度喂给模型。目标分区的二维轮廓由本地测量和标定投影，标注不替代当前可见证据或碰撞几何。未关联候选不预标猜测的全局ID。

三相机并集中缺测的固定collision ID 记录 mapping_unresolved（唯一复用桌面除外）。发布 observation 和 selected 图前均拒绝缺项；不要求每路相机看见该实体，不补历史体素，不因缺项自动恢复。

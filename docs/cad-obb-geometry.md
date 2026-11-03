2026-11-02：当前资产、直接位姿、分层XY与观测复用规则以[本轮策略](cad-runtime-policy-20261102.md)为准，覆盖下文历史检查链路。

# CAD 与 OBB 几何边界

2026-11-02：已知容器CAD目录可登记与mesh哈希绑定的内底面点/法向；当前永久ID、观测代次和CAD位姿验证通过后，从该真实凸面片取得放置范围，无需当前深度再次看见盒底。孔洞或非凸面不以凸包填平。标记registered_model_prior、unseen_area_observed=false、collision_checked=false；轻微倾斜采用面片最高世界Z作为水平搜索下界，后续完整携物/夹爪碰撞、工作空间及IK检查保持。未知物体或无支撑面元数据的CAD不生成此区域。

2026-11-02：CAD 位姿数值拒绝只标记该实例 geometry_rejected，保留同次采集其他实例的有效测量；不把失败 CAD 改为 OBB。碰撞建图继续拒绝 geometry_rejected，不能当 unseen 复用旧层。Decisions 接收失败 ID、采集和原因。当前全部相机 FoundationPose 候选被拒绝后，可对同永久 ID、同代次、同源哈希/mesh 的上次已验证位姿进行一次公开 rigid ICP 更新；仅作为起点，限制仍为累计5mm/5°，再用本次全部原始点和各相机掩码按原门槛验收。保存原失败、种子采集、当前采集和变换；失败仍不发布几何。无跨代次或跨实例种子，无旧时间戳改写，无额外模型调用或动作。

2026-11-02：规范资产 ID 改为 `blue_box`（蓝色盒子）和 `black_workpiece`（黑色工件）；`source_asset_id` 保留原外部 CAD/抓姿库索引，历史 ID 可经目录显式解析，不能按颜色随意切换资产。FoundationPose 的既有登记 model_id 和源文件哈希保持。分割词表中的所有实际 prompt_proposals 作为当前掩码的查询来源进入 CAD Decisions，并提供目录 segmentation_aliases；这些是检索线索，不是身份答案。全点残差查询改用 libigl 2.6.3 公开 AABB.squared_distance，保留完整点数/哈希、完整CAD三角面、原容差与0.9比例；每批8192点低于其10000点并行阈值，索引在批间复用。深度/轮廓仍走原Trimesh射线和首个有限三角形距离检查。

2026-10-30 用户最新要求明确覆盖早期“CAD 仅供 MuJoCo”和“实体只能 TSDF/ESDF”的限制。生产配置为 `LUNA_GEOMETRY_MODE=cad_obb`：桌面使用实测 cuboid，未知模型物体使用身份绑定的实测 OBB，已知模型物体使用当前位姿验证通过的登记 CAD mesh。原始 RGBD、点云、身份和观测来源独立保存。CAD 形状是模型先验，不是对遮挡表面的测量；模拟器中的实例位置、姿态和资产标签不构成感知证据。

## 感知与缓存

当前 SAM3 候选与 Decisions 永久 ID 关联共同确定待匹配实例。CAD 检索使用当前实例的身份绑定掩码、该 ID 对应的自然语言描述、实测可见尺寸及目录参考。描述保留 Responses 提案来源，不能当作 CAD 证明，也不能以任务目标代替当前主体。同一带编号对照图展示主体和真实 CAD 参考；暗主体可并列显示标注倍率的曝光副本，只进行同一原始 RGB 的统一增益，不生成细节、不改变 SAM/FP 输入。

Decisions 输出有类型的 `candidate`、`no_catalog_candidate` 或 `unknown`。资产选项仅表示一个待几何验证的检索假设，保留原始置信度但不把它当作已确认物理事实；不自动选择次高项、不穷举资产。`no_catalog_candidate` 表示本次检索没有提出目录候选，保留实测 OBB 近似；`unknown` 返回 `cad_retrieval_uncertain` 交图恢复，不能伪称无 CAD。只有明确资产候选才调用 FoundationPose，使用同帧 RGBD、掩码和标定估计当前位姿，再检查来源、刚体变换、全部实测点残差和深度轮廓。其他视觉判断的 0.8 门槛不变。

可跨观测缓存的是几何验证通过后的模型对应关系和模型资源；只在成功后发布登记 CAD。每次新的实例观测仍需当前身份绑定和重新估计、验证位姿，不能复用上次世界位姿。候选或已确认资产的位姿服务失败、校验失败或证据过期时，返回明确失败，不改走 OBB 或 GraspGen 来掩盖问题。预筛与精检共同使用 `min(配置残差上限, 0.05 × canonical mesh 最短尺寸)`；这是对原 15mm 上限的收紧，表面及可见深度轮廓通过比例仍为 0.9。证据记录配置上限、真实 mesh 尺寸、比例和最终容差，全部实测点保留。

模型目录为 `resources/cad_catalog.json`，资源由 `CadCatalog` 校验路径、源文件与 mesh 的 SHA-256。已登记 `THREADED_VISUAL_SAMPLE_MM`（`black_core`）和 `blue_box_410x310x145`（`blue_box`）。碰撞使用项目内以米为单位的 canonical mesh 及 `T_world_object`；离线抓姿使用原 CAD 坐标系及 `T_world_source`：

```text
T_world_source = T_world_object @ inverse(T_source_canonical_m)
T_world_tcp    = T_world_source @ T_source_front
```

原 CAD 源文件为毫米不表示抓姿也是毫米。黑色资产抓姿文件明确为 `original_CAD`、位置单位米、四元数 `xyzw`，末端已经是 `gripper_front_A/B`。不得再次缩放或增加安装端到前端的 185mm 偏移。

## 抓取与携物

| 实例几何 | 候选来源 | 后续检查 |
|---|---|---|
| 未知模型、当前身份与深度有效 | OBB mesh 表面采样 4096 点，再调用官方 GraspGen | 保存原始点云与采样来源；执行原有有界候选、工作空间、IK 和碰撞检查 |
| 已知 CAD、当前位姿验证通过 | 已校验的对应手臂离线库；黑色资产 A 左臂、B 右臂各 256 个 | 全部对应臂候选保留原索引和存储变换，之后执行工作空间、IK 和碰撞检查 |

离线库入口要求显式 A/B、正确资产与源文件哈希，以及当前刚体 `T_world_source`。库索引声明的计数是校验依据；256 是当前黑色资产的数量，不是所有资产的通用常数。不按 `valid_table_placements` 预筛，不在线调用 GraspGen，不做额外插入补偿或夹爪半转对称归一化。方向偏好不改变库里的变换；默认方向过滤关闭。蓝盒目前没有离线抓姿库：它可以提供已验证的碰撞几何，但请求抓取候选会明确返回 `cad_grasp_library_unavailable`。

抓取绑定使用实际闭爪时的前端 TCP 反馈。闭爪稳定后，首次 lift 保留实际闭爪、双臂、环境、自碰撞和关节检查，此时尚无物体 attachment；抬升成功且当前反馈仍确认持物后，才建立 attachment。未知模型覆盖 OBB 体积，已知 CAD 使用冻结的 CAD 几何与实际 TCP 绑定生成覆盖球；原始实测点云仍单独用于证据和诊断。首次 lift 不宣称验证了携物扫掠体积。

## 碰撞地图与诊断

`HybridGeometryMapper` 通过公开 cuRobo Scene 接口发布 cuboid 与 mesh，无 TSDF/ESDF 分配或积分。永久 ID 的层名为 `semantic_instance::<id>`；几何来源分别为 `observed_obb`、`registered_cad`。历史 TSDF/ESDF 路径保留兼容用途，不能把它的验证结果当成新模式已经通过。

同一永久 ID 的当前有效 OBB 替换该 ID 的旧 OBB，`geometry_observation_ids` 只记录这次几何的来源；不再合并历次包围盒，避免视角变化导致包围盒持续膨胀。已知 CAD 同样以当前验证位姿替换旧位姿。局部范围之外及本次未看到的未变化实体保留原层和原时间；动作失效、缺失和退役分别处理，不把历史几何冒充当前观测。global 仅表示本次所有已观测清单实体，不表示所有空间已观测。

点云诊断保留原始测量。碰撞预览另提供 OBB cuboid、实测桌面和带 ID/资产/哈希/观测来源的完整 CAD mesh；CAD 边界由真实 mesh 顶点变换计算，不能把 mesh 的包围盒画成实际碰撞表面。可见性、保留历史与当前身份分开显示。射线覆盖仍是独立证据，不能把模型表面或 OBB 内部称作新测量。

放置保持原始语言约束、当前支撑区域、持物几何与实际抓取旋转，候选在同一已发布碰撞世界内查询；`plan_place` 验证 approach/place/retreat 三段。CAD 碰撞形状本身不证明某个区域满足语言或可稳定放置。一次初始 Responses 负责完整正常任务，运行时 Decisions 选择或扩展已注册能力的 graph node/edge；runtime 校验后执行，不插入固定流程，不再调用 Responses。

## 资源与验证状态

五个本项目容器的 CPU 总额为 4 个逻辑核：web 1.5、cuRobo 1、SAM3 0.5、GraspGen 0.5、FoundationPose 0.5。OMP/MKL/OpenBLAS 单线程，web 软件渲染最多 2 线程。该配额不授权调整其他项目服务。

已完成的 OBB 实验是 `7c29c7e7c02b45309c9f9e4ee09c4fe9`，证据在 `data/cad-obb-iteration-20261030/iteration-1-latest.json`。该次运行使用原 TSDF/ESDF 碰撞场景，验证了 OBB GraspGen、抬升后 attachment、三段放置、释放核验及回位；它不是 CAD 碰撞模式验收。

2026-10-31 已部署版本 hybrid-15：`cad_obb` 模式下，未知物体 A 臂 OBB/GraspGen 桌面任务 C5、已知黑件 B 臂 CAD/离线抓姿入盒任务 B6 已完整通过 MuJoCo，包含交付后观测和最终回位。A 臂 CAD 完整任务尚未通过，A10 在动作前的目的地语义绑定拒绝。历史失败及原始证据见[验证记录](cad-obb-validation-20261030.md)。当前收敛修改和独立验收状态见[身份、三相机和放置边界](cad-identity-placement-convergence.md)，不能用旧版本成功替代新版本验收。

初始 Responses 提示已同步 CAD/OBB 地图合同，并要求沿正常成功路径推演 requires/effects 与动作失效。已知的前提不应留给运行时恢复补齐；这不指定任务顺序，也不由代码补节点。

CAD 图像匹配、FoundationPose 和轮廓检查使用身份确认时保留的 silhouette_mask。桌面带、深度边缘等过滤后的 geometry mask 仅控制实测点；两份掩码分别归档数量和哈希。第三轮冻结数据中，表面过滤移除了蓝盒 524 像素，不能把这些像素当作模型轮廓错误：同一 FP 位姿改用完整身份轮廓后，一致性由 0.8899676 变为 0.9239482，表面点、射线和遮挡检查均未改变。

2026-10-30 跨相机容器身份：当前深度追踪已确认的容器仍在当前图像中标注 Current anchor，提供永久 ID、测量位置和追踪来源，供 Decisions 对比其他候选；不自动覆盖模型答案。CAD 先检查相同内点比例下的必要尺寸条件，再做 FP/精确表面与射线核验；不丢弃测量点，不把不同位置的实例合成一个有效物体。第四轮已完成未知物体出盒、三段放置和释放核验，但交付后观测因两盒混为同 ID 未通过，已归档并停止重复恢复。

黑色 CAD 首轮图在执行前被拒绝：Responses 把清单确认写在物体 subgraph 内，但该 subgraph 调用本身需要已确认清单。提示已明确这一真实调用前提；不移动模型节点、不自动补图、不增加运行时 Responses。

冻结黑 A2 的单变量 FP 对照中，公开 `default` 旋转采样实际为 252 个候选、5 次 refinement；原 low_memory 为 42 个。default 位姿在新 2.2mm 容差下，1068 个实测点全通过（最大残差 1.306mm），可见深度轮廓比例 0.940594。独立新检索明确提出 R2（报告置信度 0.93）；冻结粉色负例选择 `no_catalog_candidate`（0.68），两次检索均未调用 FP 或机器人。证据在 `data/cad-obb-iteration-20261030/black-a2-match-audit/{default5-scale-validation.json,retrieval-proposal-black,retrieval-proposal-pink}`。这验证了冻结输入的两阶段边界，尚不是生产完整抓放验收；隐藏表面仍是模型先验，近似几何与旋转对称性不能仅靠这些阈值完全消歧。


2026-10-30 CAD 位姿检查统一使用欧氏米：深度轮廓将每个测量 XYZ 与同一射线首个命中的有限 CAD 三角形比较，仍用前后深度符号区分前景遮挡和自由空间矛盾。初次验证失败时只允许一次公开 trimesh 刚体 ICP 提议，最多 20 轮、累计 5mm/5°，禁止缩放和反射。优化输入最多 2048 个确定性空间样本；采样仅用于优化，最终仍验证全部原始测量点和当前身份掩码，原 0.9 比例及尺度容差不变。归档原 FP 姿态、修正变换、预算、采样及全点哈希；失败仍为 cad_pose_geometry_rejected，不发布 CAD、不回退 OBB，也不缓存旧姿态。冻结 A4/A2 好坏姿态及蓝盒正负例已做 CPU 回归；这不等于新流程已通过真实完整抓放验收。

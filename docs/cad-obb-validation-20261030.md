# CAD / OBB 实际验证记录（2026-10-30）

后续 2026-10-31 的 hybrid-16 至 hybrid-19 修改、C6 成功与 A11–A14 失败见[最新验证记录](cad-identity-placement-validation-20261031.md)。本页各版本和历史结论保持原样。

记录的是实际运行结果，不把健康检查、CPU 单测或冻结数据诊断等同于完整机器人任务。
证据保存在 `data/cad-obb-iteration-20261030/`，原始实验在 Ubuntu 的 `data/experiments/<run_id>/`。

## 当前结果（2026-10-31）

生产 hybrid-15 已同步 Ubuntu，937个部署文件核验一致；本节之后的文档更新另做文档同步核验，未改变已验证生产源码。下面区分完整 MuJoCo 任务、纯规划和失败。

| 分支 | 实际结果 | 模型调用与耗时 |
|---|---|---|
| 未知粉色物体，A臂，OBB + 在线GraspGen，盒内取出放到桌面 | C5完整通过：抓取、lift后attachment、放置、释放核验、交付后双臂观测、最终初始位姿核验 | 1 Responses + 31 Decisions；531.335秒 |
| 已知黑色工件，B臂，CAD + B臂离线抓姿，放入右盒 | B6完整通过，含放后身份错误的显式恢复；未重播已完成抓放 | 1 Responses + 42 Decisions；893.808秒 |
| 已知黑色工件，A臂，CAD + A臂离线抓姿，放入左盒 | A8曾实际抓放，但释放视觉核验未通过；A9/A10在动作前目的地语义绑定停止，完整任务未验收 | A10为1 Responses + 7 Decisions；158.94秒 |

以上世界使用实测桌面cuboid、已确认CAD mesh和未知OBB，不使用TSDF/ESDF。CAD模型由当前自然语言/清单及SAM3证据提出，FoundationPose当前位姿须通过数值验证。左右臂离线库各256姿态，已知CAD不调用GraspGen。未知物体保留原测量点云，但在线抓姿输入为OBB表面采样。几何和当前/历史状态与永久ID绑定，网页几何诊断显示CAD/OBB及来源。

候选重验在C5同一冻结输入上完成真实GPU单次对照：数值resolve由36.406秒降至9.453秒，公开高度调用77→12，完整单候选检查保留；不包含语义选择、轨迹或运动，不能当作整个任务的提速倍率。hybrid-15完整Python回归4168 passed、58 skipped，前端388 passed，Ruff和Vite构建通过。

当前限制：跨相机同色盒身份仍可能误合并；目的地身份判断与原指令语义判断仍可能不一致。数值几何检查及原概率门仍会拒绝不充分证据。A10单次精简语义题并加入错误右盒对照没有改善，未上线该诊断措辞。C5/B6存在旧PNG归档缺口；最新A10的8个调用及43 JPEG/5 PNG已逐图通过哈希和解码核验，不能因此补造旧档案。后续各节保留原始失败与当时验证状态。

## 完整任务与失败边界

| run_id | 场景与结果 |
|---|---|
| `7c29c7e7c02b45309c9f9e4ee09c4fe9` | 旧 TSDF/ESDF 世界：未知粉色物体 OBB → GraspGen → 抬升后 attachment → 桌面三段放置 → 释放核验 → 交付后观测与最终回位完成。总计 245.42 秒、一次初始 Responses。不是 CAD 世界验收。 |
| `656758a65db247c78f3422ce546ada6d` | CAD 世界运动前失败：头部深度背景超过 uint16 毫米传输上限。适配层在传输副本中标记过远背景无效，原始 RGBD 保持不变。 |
| `f4566c4ba829490a866df1582afee71e` | CAD 世界运动前失败：误把几何过滤后的掩码当成原 SAM 轮廓。现已区分 silhouette mask 与 geometry mask；冻结原位姿的轮廓一致性从 0.88997 变为 0.92395。 |
| `fd2a30f59a2a4ed498bd1abdb7a1b746` | CAD 盒子 + 未知物体 OBB + cuboid 桌面：实际抓取、抬升后 attachment、桌面三段放置和释放核验成功。交付后观测把两盒关联到同一 ID，完整任务未完成，已停止。 |
| `2e3878966e114554b7e7c6e60d218920` | 黑色工件 A 臂首轮：初始 graph 未在物体 subgraph 调用前确认清单，执行前拒绝；未运动。提示已明确真实调用前提，代码不补节点。 |
| `54e6713e48184315bcb91fd581ea9e44` | 黑色工件 A 臂第二轮：CAD 匹配不确定，目的地语义未通过；未生成抓姿、未运动。 |
| `4e933369d98d452cbf5294ba7e6e5191` | 未知粉色物体第五轮：初始观测合并两盒。必要尺寸检查在 FP 前拒绝；Decisions 重拍一次后停止，117.7 秒，未运动。 |
| `32d6e3f05a6e475087dc05e0ca764076` | 第六轮：初始感知返回，但源盒跨相机身份存在位置矛盾，桌面目的地语义为 unknown，133 秒后停止；未运动。 |

## 冻结输入诊断

- 浮点 FK 边界：107 个实际 IK 结果全部通过修正后的 OBB 工具帧变换；保持原碰撞容差，外扩最大 13.46 nm。
- 当前容器锚点：同一原始图像与追踪证据下，错误的右盒 → 左盒概率从 0.86 降至 0.02。一次真实 Decisions，2.172 秒；仅证明该误匹配被拒绝。
- 黑 A2 绑定语义：直接指出待确认目的地 ID、当前位置与坐标约定后，同一六张图的语义概率从 0.18 升至 0.99；一次真实 Decisions。另一粉色桌面样本仅修改“支撑实体与具体落点”的问题边界，仍为 unknown，不能宣称通用语义问题已解决。
- 黑工件原图只有 37×40 像素；真实 CAD 俯视参考和有标签的曝光显示副本没有使视觉匹配确认，仍保留 unknown。

## FoundationPose 同输入对照

两次使用完全相同的请求（SHA-256 `19f42734b14299572106f0862ae3d19787bf0edfccdd413368f8451c1134b48f`）、原图/原 K/原 SAM 轮廓、SDK、checkpoint、5 次 refinement，以及同一数值验证。R2 仅作为未确认诊断假设，没有发布生产 CAD。

| 项目 | low_memory | default |
|---|---:|---:|
| 实际旋转候选 | 42 | 252 |
| 95 分位表面残差 | 6.338 mm | 0.625 mm |
| 最大表面残差 | 8.342 mm | 1.306 mm |
| 可见深度轮廓一致性 | 0.5515 | 0.9675 |
| 原数值验证 | 拒绝 | 通过 |
| FP 服务耗时 | 5.175 s | 6.634 s |

默认搜索保留 40 个基础视角设置，把面内旋转间隔从 360° 改为官方默认 60°。没有修改 SDK。设备显存一秒采样最高为 65,627 / 97,887 MiB；这是该次采样值，不是所有场景的峰值保证。FP 仍为 0.5 核，其余四个项目服务未重启。

登记模型与项目 canonical mesh 坐标核对通过：源文件哈希、原点、单位、轴一致；黑色模型双向最近顶点最大差约 2.48 nm；原 CAD / canonical / 世界位姿转换闭环误差为零。

## 软件回归

`hybrid-final4-pytest.log`：3845 passed、58 skipped（412.81 秒）。后续身份与语义局部改动另有针对性回归，不能将此完整套件结果冒充后续所有版本的最终测试。

此前网页 Node 测试 387 项通过，Vite 构建通过；实际浏览器验证 CAD mesh / OBB / 桌面预览、ID 与来源信息，浏览器无 JavaScript 错误。现有大 chunk 提示仍保留。

离线黑色 CAD 抓姿库的 A/B 各 256 个变换已做 CPU 全量验证；完整 A/B 物理抓放验收仍需当前 CAD 身份、位姿、原生轨迹与实际执行共同通过。


两阶段检索已完成两次独立冻结调用：黑色工件明确提出 R2 候选（报告 confidence 0.93），粉色物体选择 no_catalog_candidate（0.68）。候选不是 CAD 身份事实，unknown 仍返回 cad_retrieval_uncertain。黑件 default/5 冻结位姿按更严格的 2.2mm 容差重新计算：最大表面残差 1.306mm，深度轮廓 0.940594，通过；没有重新调用 FP。

R/C 标签改为框边短标签后，hybrid6 冻结调用仍有 C4/C7 unknown，不能宣称身份已修好。核查发现参考相机在 initial_reference_annotations 中，原代码却从无此字段的 fixed_entities 读取，导致 16 个跨相机问题错误使用历史可移动语境。修复从已验证参考图的 ID/相机映射读取，并保持观测 ID、代次和 verified_camera_ids 检查；真实 wire 结构回归先失败后通过。60 项身份/布局关联测试通过，随后统一部署 hybrid-7（847 文件哈希一致），开始 black-A3 完整任务。原始 RGB 未改动。


hybrid-7 完整测试：3863 passed、58 skipped，403.71s。之后仅目的地问题精简的关联测试另有 50 项通过。

black-A3（36f6c86c2a0244c1b7a21215f71fb6b6）：两盒分离，黑件两次当前 CAD 位姿均验证通过且无 OBB；第二次复用模型对应并重新估计位姿。任务在目的地语义确认停止，无机器人动作。冻结原输入仅精简该判据，概率 0.47→0.96，其他三项概率不变；一次 HTTP 调用。

black-B1（d1e4e89228be4c6684dfd24ea20c54ea）：仅一次 Responses；B 臂完整离线库经原有筛选保留 159 个，实际 candidate_index=75 完成抓取、原生 lift 与后置 CAD attachment。resolve_placement 生成 127 个区域内候选并分布采样 96 个：90 个未通过碰撞端点检查、6 个未通过位置包络，图一次恢复后停止，未交付。完整输入和候选失败保存在 black-B1-placement/，不可将该结果记为抓放全流程通过。

black-A4（8fce4338b12441e28dbb699c9f81e3b2）：一次 Responses，左臂离线 candidate_index=116 实际抓取、lift、后置 CAD attachment、三段放置执行通过；释放视觉核验 unknown，放后重新观测 CAD 位姿失败，任务未完成。不能把机械放置完成记作交付验收成功。当前核验输入是历史目标裁剪和当前原始 RGB，并无旧 margin 标签污染。数值位置 unknown 的原因现独立保留在 missing_evidence，避免模型的视觉否定掩盖 cad_pose_geometry_rejected。

A4 冻结深度诊断：CAD 投影轮廓重叠 0.9664，轴向深度一致性 0.5154；保持 2.2mm 容差，改用首命中有限三角形欧氏距离的诊断结果为 0.7949，仍拒绝。斜面上的轴向误差被放大，但尚不能仅改度量宣称修复。该诊断不调用模型、FoundationPose 或机器人，逐射线证据在 black-A4-pose-audit/。

unknown-C1（e1dbb057e7fe4f45b5939ab54349c42e）：一次 Responses，140.3秒后停止，未运动。初次观测蓝盒表面检查通过、深度轮廓 .85837 未通过；fresh 恢复后右盒身份 unknown，随后同帧重用仍失败。恢复候选前移已有无进展排除规则，69 项相关回归通过；尚未以新的完整运行证明时间收益。

black-B2（05f2f87d12c74140a3579ea63a80c8d4）：一次 Responses，319.57秒后停止（轮询结束323.1秒），无物理运动。当前可见盒底内接矩形只有171×86mm，35个IK通过抓姿都因占地不适合该小片而无XY，实际未发送GPU sphere query，不能将此运行记录为GPU新接口验收。补全当前验证CAD中与实测高度/锚点一致的完整凸支撑面后，同一冻结35抓姿均可采样96 XY；仍需碰撞与实际放置验证。8项平面区域测试通过，包含圆环不能填洞、过期/错误CAD不能扩区及真实蓝盒内底面。

新的CAD凸体分区24球保持原6mm padding和全部组件覆盖。B1模型向下包络外扩26.695→15.025mm，CPU拟合0.187s；原0/53/90三个XY在新起点及+10/+20mm仍有夹爪碰撞，attachment球均无冲突。不是B1放置成功证据。

实际GPU只读world-sphere接口：同一当前CAD世界中，实测桌面内部球拒绝、上方空位球通过；过期observation返回422，无attachment和运动。冷请求5.156s（backend构建5.041s）；热批次512端点×110球=56,320球，全部结果与构造的占用/空位交替一致，HTTP0.313s、GPU服务operation0.0616s。2383×110最大批次首次因旧HTTP8MiB上限拒绝，尚未进入GPU；已将仅此有数值数量上限的接口请求上限改32MiB，其余接口保持8MiB，等待下一版实际验证。


hybrid-10 完整软件回归：`hybrid-final10-pytest.log`，3985 passed、58 skipped、1 个现有 Starlette httpx 弃用提示，475.06 秒。864 个部署文件的本地、压缩包与 Ubuntu SHA-256 一致。

black-B3（c5901daff90844dfa9a579b1192eceee）：一次初始 Responses，实际右臂 CAD 离线抓取、原生 lift 和后置 attachment 通过；固定夹爪端点筛选确实执行 21 个 GPU 批次，每批最多 2383 位姿 × 110 球 = 262130 球，51 个源端 IK 合格抓姿中保留 48 个。完整 CAD 盒底区域与实测高度绑定，resolve 返回 placement-5/34/85 三个候选。两次 plan_place 都未通过，尚未释放。SDK status 是 Goalset planning returned None，项目旧 phase=approach 标签不能证明已进入接近轨迹阶段；只读源码确认是公开 plan_grasp 首次放置目标预求解返回 None。第二次恢复仍为同三个候选，正在诊断数值可达性与无进展重复。不能将本轮记为完整抓放成功。


black-A5（cf94fa5d1c2741ce8067523091dab49a）：一次初始 Responses，左臂 CAD 离线抓取、lift、attachment、三段放置执行完成。放后核验已确认黑件候选（confidence .81），但目的地左蓝盒未确认，measured_object_position 因缺目的地ID为 unknown，未验收。系统随后重复 observe/build_scene；在约1044秒保存 black-A5-before-reset.json 并手动结束测试，不能记作自动完成。发现 verify_transfer 的部分身份成功漏用了已有一次冻结澄清，修复后28项身份相关回归通过。一次真实冻结澄清保持全部原图不变、黑件原回答 .81 不重问，左盒候选 .48→.86；不代表当前CAD数值核验或端到端任务通过。


恢复上下文真实单次冻结对照：原 B3 recovery_capability 输入76,294 tokens、1.624s；精确去重后29,382 tokens、2.490s，减少61.5%输入tokens。本次延迟没有下降，不宣称提速。7张图、questions不变，原始数据能完整展开还原；新调用仍选择合法stop（.56），只记录回答，不执行graph。具体actual-compacted-call/result在black-B3-recovery-context-audit目录。

## B3 冻结全臂 IK 定位

原 CAD 世界、实测双臂与夹爪反馈、24 个 attachment 球均按归档重建，没有使用仿真对象真值。保持原 XY、抓取旋转、5mm IK 位置门槛、0.05rad 旋转门槛与5mm硬净距：placement-85 上移21mm后，携物全臂 IK 通过，目标 CAD 底面 gap=46.129mm，实际 FK gap=48.908mm，均低于配置50mm上限。+22mm 的实际 gap=49.929mm；+23mm 及以上虽 IK 成功，但实际 gap 超限，不能接受。此前0～20mm只按5mm间隔诊断，不能声称+21mm是1mm全采样的最低可行点。

三处 XY 的30个高度诊断中，全部332个工具/attachment球在目标位姿的世界检查通过；这证明仅工具球无碰撞不足以选出全臂可达端点。另32个离线抓姿的单个首选端点均未解出 IK，不足以据此拒绝整个抓姿或放置区域。完整输入、FK、碰撞与服务版本证据见 `black-B3-placement/gpu-ik-audit-report.md`。这些是端点诊断，尚未验证完整轨迹、释放或任务完成。

hybrid-11 已部署889个哈希一致文件，包含释放后部分身份的冻结澄清、原SDK失败阶段纠正、数值输入重复放置拦截和恢复上下文精确去重；没有改变原SDK或安全阈值。正在运行新的未知物体 CAD 世界测试。

unknown-C2（f5a8f196d17b4f7cb341650f299b8477）：一次 Responses，124.39秒后 Decisions 自行停止，无机械动作。两路相机的蓝盒候选合并到同一ID，171201点的90%最小Y跨度1.483m超过CAD允许0.558m，初始几何校验拒绝；不能绕过校验发布CAD。black-A6（9e406007248b4909b95c1e17afa21890）同样在初始身份unknown停止，140.88秒、无机械动作。新的释放后身份修正尚未在这两轮走到实际验收阶段。

B3 冻结预释放轨迹诊断：独立cu进程重建归档世界，原 placement-85 的XY、旋转、接近偏移、attachment和规划阈值不变，仅将目标Z上移21mm。一次公开 `/tools/plan_place` 在12.653秒返回ok=true，但独立机器人FK发现place段几乎恒定，终点Z=0.712585m、目标Z=0.592714m，误差119.871mm恰为接近偏移。因此**不能记为有效下放轨迹**；SDK返回成功及碰撞通过不足以证明到达目标。输入输出、CPU复核见 `black-B3-placement/raised-place-result.json` 与 `raised-place-fk-audit.json`。未控制机器人、未松爪，正在检查SDK公开调用与项目结果转换链。

后续核实该独立进程没有执行生产启动时的 `warm_start`。锁定SDK首次存入目标时保留引用，后续接近目标更新可能覆盖原下放目标；同一目标只clone外部输入不能解除内部别名。按生产的公开预热流程重新执行一次同输入：11.052秒返回预释放轨迹，place段61点有实际下降，独立FK位置误差4.944565mm、旋转误差0.020434rad、CAD实际底面gap49.20128mm，均在原5mm/0.05rad/50mm阈值内。见 `raised-place-warm-result.json` 与 `raised-place-warm-fk-audit.json`。这是有效预释放规划证据，仍未实际释放；不把冷诊断问题宣称为生产必现，不因此给每次生产规划增加固定求解调用。项目正在补返回轨迹终点的公开FK检查，替代转换层硬填零误差。

C2 当前身份锚点的单次真实对照：13张JPEG字节/顺序、22个问题均保持原样，只补入同一冻结捕获中已接受C4/C5对应及其原回答/哈希/相机证据。错误C7→已登记左盒R1从0.90降为0.64，保持unknown、不再合并；其他候选仍存在unknown，不能记为全身份通过。1.982秒、22808输入tokens；`unknown-C2-identity/actual-anchor-result.json`。空右盒未在该初始Responses清单登记，不能由代码为它新增ID，也不能冒用已有盒子ID。

## hybrid-12 部署与真实服务验证

898个部署文件的本地、压缩包与Ubuntu SHA-256一致。新的高度搜索保持XY与抓取旋转，按配置最大释放间隙推导Z上限，1mm采样，并校验完整CAD/OBB底面、工具/attachment碰撞球、离线工作空间及携物全臂IK。目标和实际FK间隙都必须合格。`plan_place`另行以公开FK检查返回轨迹的接近和放置终点，不能将SDK成功直接当作到达目标。

B3冻结世界的冷启动反例在新版本明确拒绝：`CUROBO_INVALID_TRAJECTORY`，放置终点位置误差119.871208mm，原门槛5mm。未执行任何机器人运动。证据：`black-B3-placement/hybrid-12-endpoint-cold-result.json`。

按生产公开预热流程，新高度服务实际检查26个持物IK高度样本，选择原XY/R上移21mm，实际FK完整CAD底面间隙48.908375mm，未超过50mm上限；高度解析15.369秒。随后原接近/撤离偏移不变的预释放规划耗时2.267秒，通过新增端点校验：放置位置误差4.944681mm、旋转误差0.020434rad。证据：`black-B3-placement/hybrid-12-height-warm-result.json`。这是当前服务的高度解析与预释放规划验证，未执行释放或物理验收。

C2冻结新回答经CPU融合后，已登记实体由C4蓝盒(.86)、C5粉色(.91)、C6绿色(.98)覆盖，`missing_geometry_ids=[]`；未确认辅助候选不进入几何。这个结果只证明该冻结输入的身份融合，不证明FoundationPose或完整任务。

unknown-C3（8b1b4776f5c9480aab513b745cf30325）：hybrid-12，一次初始Responses，约110秒后Decisions自行停止，初始`visual_identity_unknown`，无机械动作。正在逐项检查本轮身份回答，不能将前述冻结成功推广为在线身份已稳定。

第一次hybrid-12完整软件回归为4058 passed、58 skipped、1 failed（437.92秒）；唯一失败是旧视觉输入测试仍要求原始数组变化后整个请求完全相等。更新该测试为仅允许mask/silhouette/depth/valid-depth四个来源哈希变化，严格保持其余元数据、RGB字节、图片顺序和问题相同；24项关联测试与Ruff通过。生产代码没有因此改变。重新执行完整套件：`hybrid-final12b-pytest.log`，4059 passed、58 skipped、1个现有Starlette弃用提示，441.01秒。

black-B4（a60ee0113c914cd9a38e2a3edb98b8cb）：一次Responses，B臂`cad_offline_library`路径保留159候选，实际抓取、lift、后置attachment和持物核验通过。首次放置搜索96个XY：90个全部高度存在工具世界冲突、5个工作空间拒绝，仅candidate-5有26个工具世界通过样本，但26个持物全臂IK均未解出。随后两次放置恢复仍失败，约373秒Decisions自行停止，未释放。完整首次失败在`black-B4-placement/event-00241.json`。该场景未通过，不能以B3冻结规划替代本轮结果。

## hybrid-13 的缺陷证据与改动边界

B4桌面cuboid的Y下界比B3延长127.678mm。归档head相机(row218,col586)点为(.594812,-.560602,.568226)m，原始RGB显示蓝盒竖壁；其局部法向与桌面法向相差87.322°。它仅因处于桌面高度的原6mm带内进入初始RANSAC种子，随后min/max扩成错误桌面。复用补充视角已有的16邻点、2cm、5°支持面判据后，冻结诊断桌面Y下界为-.425993m；不使用颜色或仿真物体真值修边。这是测量分类缺陷，不能通过放宽碰撞阈值解决。

B4生成的155个区域内网格点包含(.515115,-.556544)m，与B3曾通过冻结规划的placement-85仅相距0.05566mm，但原96点分布采样漏掉它，最近已采点相距18.723mm。改为先试原96点，若无可行端点，再搜索同一有限网格的余点；网格原25×25加中心上限、区域约束、抓取旋转及所有数值门槛不变。首批成功不追加搜索；报告实际搜索和未搜索数量，不宣称连续区域已穷尽。3项回归先失败，修复后连同放置/高度相关45项通过，Ruff通过。冻结XY来源见`tests/fixtures/black_b4_ranked_placement_xy.json`。

B4两次重复resolve的实际有效参数均是clearance，尽管恢复提案分别为decide和clearance。全部96个XY/Z序列、场景、布局和测量数组相同；B夹爪累计变化2.5425e-6rad，工具球最大坐标变化0.1607微米，不能称原始输入逐值相同。两次重复净算法耗时17.946秒。修复目标是通用恢复记录遗漏正常首败、以及同一有效参数的别名，不放宽1e-6反馈一致性门槛。

C3缺失实体为已登记绿色碰撞物。绿色R3参考框与当前C1/C5框正确，粉物、左盒和右盒均已有可用对应。补充“清单不等于搬运选择”的单次文字对照未改善绿色身份，未采用该修正。另一次独立对照将16条match谓词重复的23856字符按原文提取为共享规则、参考描述与按candidate索引的测量，谓词保留明确对应关系；可逐字还原原问题。12张原图、21题名称/类型及阈值保持不变。绿色C1 .15→.96、C5 .34→.89，模糊C7仍unknown；耗时2.489秒、25713输入tokens。该结果支持整理身份输入，仍不代表完整物理任务通过。

black-A7（f58ed13931f14948a61a036604bd48dd）：唯一Responses把观测/清单确认放进物体subgraph，root直接调用，`subgraph_requires_confirmed_inventory:root_call_workpiece`在运动前拒绝。保留完整拒绝提案于`black-A7-graph/rejected-proposal.json`。现在在输出schema的owned/shared Subgraph及SubgraphNode字段说明中直接呈现既有调用前提；不改校验器、不由代码补节点。34项既有native graph/preflight测试和Ruff通过；新的实际Responses编排仍待验证。

hybrid-13 部署的909个文件SHA-256一致；完整软件回归 `hybrid-final13-pytest.log`：4092 passed、58 skipped、1个现有Starlette弃用提示，433.30秒。全局Ruff另发现14项既存样式问题，涉及10个与hybrid-12内容相同的文件；修正导入顺序、测试中的字典写法、嵌套上下文与未使用变量后，`ruff check src tests`通过，72项关联测试通过。样式修正尚不属于以下hybrid-13实际运行版本。

unknown-C4（9279e28ae7d94fe783159b40f78d25bf）：一次Responses，约116秒后Decisions自行停止，无机械动作。首次观测已确认全部登记ID，但粉色未知件CAD检索选择unknown，而非no_catalog_candidate，故没有进入OBB分支；右蓝盒检索R1=.99。fresh恢复后粉、绿、右盒确认，左盒C3=.52，C6对左盒=.99、绿色=.90，排他身份门拒绝；不能因其中一项高概率而接受冲突ID。此次实际使用共享身份说明，不能以C3冻结成功声称在线已稳定。顶层失败仍沿用初始粉色C0=.41，而最新澄清中粉C0=.97、C4=1.0；已定位该错误呈现缺陷，后续修复保留最新失败及缺失ID，原回答与哈希仍归档。

black-B5（95c7167e1f5649bd8c95380ffa61ba45）：hybrid-13，一次Responses；右臂CAD离线抓取、lift、后置attachment、持物核验均通过。resolve两次约16.7/16.9秒，搜索到首批96点之外的placement-97/102。首次plan拒绝的唯一原因是B夹爪反馈从0.44185218927941283变为0.44186598501518515rad，差1.37957e-5rad，超过既有1e-6门槛；固定原反馈后的输入摘要、结果签名及逐字段结果均一致，未发现缓存篡改。Decisions选择重新resolve后再retry plan；第二轮夹爪差5.04136e-7rad，通过原门槛，417.65秒预释放规划成功，435.74秒实际放置完成。未放宽状态一致性或运动检查。

B5的verify_place于494.98秒返回unknown（visual_destination_unconfirmed）。随后三次observe_scene完成，当前几何只确认黑件和左盒，目标右盒持续缺失；observe的skill success不等于身份覆盖完整。恢复又执行build_scene后继续原位采集，未重新完成释放核验。约709秒保存`black-B5-before-reset.json`后人工结束实验，不记为自动停止或任务完成。首次/二次resolution的只读重放证据位于`black-B5-resolution-audit/`。

B4独立冻结诊断工具的v1、v2均在实际算法调用前因工具依赖错误停止：v1把依赖MuJoCo的CPU重建放进无该包的cuRobo容器；v2的SDK来源哈希列表使用了不存在的IK源码路径。两次均无GPU算法请求、无机器人运动；不能将它们记为B4冻结规划通过。原失败档案均保留，生产服务未因此修改。

v3先通过远端无GPU启动检查（锁定SDK 361个源文件SHA、79个机器人网格、实际导入与typed payload），随后一次独立GPU诊断通过。原B4 CAD世界、持物、机器人反馈、设置、原有限池rank60的XY及抓取旋转不变，只由原RGBD重算桌面。公开高度解析与预释放规划返回有效轨迹；独立本地机器人FK校验61点place段，终点位置误差4.984093mm、旋转误差.021148724rad，完整CAD底面间隙48.967071mm≤50mm。证据 `black-B4-placement/hybrid-13-b4-table-refit-v3-result.json`；独立进程总70.735秒，含预热。未执行机器人运动，不能替代完整交付验收。

black-A8（ebcfcb0b6d5448dfa98a818fc48a6765）：hybrid-13，一次Responses，左臂CAD离线库筛选后147候选，实际抓取/lift/后置attachment成功；resolve约30.8秒，plan_place成功，389.18秒完成实际放置。623.59秒释放核验返回visual_destination_unconfirmed；后续观测将多个不相容区域关联至同一蓝盒ID，90%最小Y跨度1.363m超过CAD允许.558m，cad_instance_geometry_inconsistent拒绝。约730.81秒Decisions自行停止，未完成交付验收和回位；不能把两臂机械链路通过记为任务完成。

C4同输入SAM主体图的单次实际对照：原13张图、37题名称/内容/类型和接受阈值完全保留，对8个待澄清候选统一追加原RGB+原silhouette掩码主体图，外部置灰并标明来源，掩码内像素无改动。一次HTTP，1.775秒、28235输入tokens；C6左盒概率.99→1.0，绿色.90→.04。CPU使用原parser融合后4个已登记非桌面实体均覆盖，missing_geometry_ids=[]；C0、C3、C7仍unknown并丢弃。该证据只证明冻结身份关联改善，未证明CAD或完整任务通过。结果在`unknown-C4-identity/actual-mask-result.json`及`masked-comparison-cpu-fusion.json`。

## hybrid-14 修复及验证边界

身份澄清附加当前候选的SAM主体图，使用原silhouette而非后续几何裁剪mask；逐项绑定原RGB/深度/掩码、标定、候选、观测与代次。所有原图/原题保留，不增加Decisions调用。实际C4请求与生产构造逐项对照：原13 JPEG、37题、追加说明、metadata与8张主体图的640×480主体像素一致；32像素标题带字体可能因主机Arial/Pillow fallback而不同，未声称整个PNG字节一致。241项关联测试通过。

失败报告以最新澄清失败为当前错误，初始失败保存为historical evidence；同时报告已接受/所需/缺失永久ID及冲突候选，保留原回答和哈希。没有将C4的CAD unknown改为no_catalog_candidate。CAD检索字节核对也已确认原RGB、6524像素SAM mask和比较JPEG完全对应，原图未被裁减；该案例仍属于未解决的模型不确定性。

B5恢复循环的根因是build_scene改变规划facts，重新开放相同观测的恢复动作。observe专用签名移除无关规划/地图状态，保留当前视觉输入、身份需求、标定、视角、物理动作和生命周期等输入；真实归档630→651回放中fresh/reuse仍被拦，verify_place retry仍可选。101项相关测试通过。既有RGB抽样/量化摘要不变，没有新建深度等值算法；纯深度变化的辨识仍是既有限制。该规则只排除重复操作，不替Decisions选择下一动作。

同帧澄清中的`identity_N`谓词和一般`match_*`谓词均已覆盖SAM主体图；已接受并从澄清删除的问题不会重新加入。首次全量14回归未完成即停止，已有4项失败：两处旧mock把predicate题回成choice，两处仍从顶层读取首次失败的置信度。修正测试保留低置信/unknown本意，同时核验最新失败和历史证据；三文件90项、关联23文件333项通过，生产代码未因此修改。全量Ruff通过，完整回归以新的`hybrid-final14b-pytest.log`为准。

hybrid-14部署917个文件，压缩包和Ubuntu文件SHA-256一致。以下实际运行与此前冻结对照分开记录，尚未据健康检查或局部测试宣称任务完成。

unknown-C5（e1e9947daeaf4604b46824a060f4bb02）：hybrid-14首次完整未知物体任务通过，最终run=succeeded。初始Responses约64秒；实测OBB表面采样进入在线GraspGen，原300个抓姿经筛选得到A臂207候选。189.03秒抓取规划通过、199.93秒实际抓取/lift结束、200.03秒持物确认。四轮resolve均找到12个候选，前三次plan因夹爪反馈变化拒绝旧证据，Decisions显式重求后第四次plan通过。392.34秒实际放置完成、438.56秒verify_place confirmed；493.55秒双臂到观测位姿、525.50秒fresh观测结束，527.71秒最终回任务初始位姿核验、529.83秒任务完成核验通过。完整状态保存为`unknown-C5-latest.json`。该成功不掩盖前面失败案例，也不等于所有场景均已稳定。

C5重复求解只读审计：前三次A夹爪反馈变化为+4.695354e-4、-2.957724e-6、-5.093739e-4rad，超过原1e-6门槛；第四次为-8.128141e-8rad通过。前三次A关节最大变化仅3.38–4.83e-9rad，其他绑定摘要及结果签名均一致。四轮候选ID顺序相同；前三轮净resolve耗时82.558秒、四轮112.162秒。见`unknown-C5-resolution-audit/validation-audit.json`；尚未据此放宽状态检查。

完整14b软件回归：4135 passed、58 skipped、1 failed、1现有warning，431.94秒。唯一失败为旧诊断mock向predicate题返回choice；只修测试，明确区分最新predicate .59、派生unknown分数0和历史categorical .59。该文件及相关174项通过；生产代码不变，完整回归继续以`hybrid-final14c-pytest.log`核验。

完整14c软件回归最终通过：4136 passed、58 skipped、1现有Starlette弃用提示，430.38秒；全局`ruff check src tests`通过。运行中的hybrid-14生产源码与此次测试版本一致，14b之后只有上述诊断测试文件和报告发生变化。

C5独立完成审计进一步核实：32个模型事件=1次Responses+31次Decisions；粉色物体5875个实测点拟合OBB，4096个OBB表面合成点进入GraspGen。实际lift完成及持物反馈确认之后才attachment。放后7969个当前RGBD表面点在指定区域内，最小边界余量11.852mm；camera3提供完整视觉witness。最终回位技能验证already_at_initial_pose=true，没有额外独立回位运动。详见`unknown-C5-completion-audit/report.md`。发现PNG主体图使JPEG专用归档器丢失一条调用JSON；物理执行及完成证据仍在，但不能宣称该轮模型请求完整归档，后续兼容修复不能补造缺失记录。

black-B6（9e800b0fd99a4e43907e60e3cb0e9a5b）：hybrid-14右臂CAD任务完整通过，run总893.808秒（verify_task_completion于892.271秒）。1次Responses、42次Decisions；黑件对应B臂CAD离线库256→130候选，无GraspGen请求。实际抓取/lift及后置attachment通过，首次resolve与plan_place通过，317.37秒实际放置完成，438.51秒释放核验confirmed；实际释放间隙46.298mm，放后4480点在盒内、最小边界余量39.682mm，camera1提供.99视觉witness。

B6交付后观测先出现CAD验证拒绝，不能记为一次顺畅完成。event367蓝盒可见深度/轮廓一致率.898649<原.9门槛；event391则把左腕左盒C3以match_3_0=.84合并到右盒C0身份，合并后的90%最小Y跨度1.122m超过CAD允许.5576m，在FP前拒绝。该错误澄清已发送4张SAM主体PNG，不能声称错误由于漏图。后续Decisions显式补观测、建图、换到双臂观测位姿并retry，818.89秒取得交付后观测证明，正常图后续观测/最终回位及6项完成谓词均通过。没有重放已完成抓放。两个独立报告位于`black-B6-post-observation-audit/identity-report.md`与`completion-report.md`，保留归档PNG缺口及重复观测耗时。

black-A9（7346991599c440d5a6d68d7a6cfe4590）：hybrid-14，一次Responses，493.48秒自动停止，无机械运动。初始观测成功后，confirm_inventory的source_identity=.98、destination_identity=1.0、target=.98，但destination_requirement_matches=.70为unknown。两次恢复observe_scene均完成，Decisions仍未重新核验confirm_inventory，最终stop。不是cuRobo或CAD离线库失败，不能将此前左臂实际抓放成功记成此次完整任务通过。

## hybrid-15 的改动范围

反馈变化后的下一次显式resolve可优先重验原已接受至多12个XY/request；原签名、所有非反馈绑定、语义区域和当前合法网格必须一致，当前反馈必须超过原1e-6门槛。重新生成当前碰撞、工作空间、携物IK/FK、完整CAD/OBB底面高度证明，重新做候选语义筛选；旧Z/proof不沿用。种子全部失败则继续原网格且不重复，偏好改变或反馈回摆到原容差内不命中。新增24项回归、关联106项通过，独立静态复核未发现缺陷；C5第二轮冻结输入441个网格中精确命中原12个种子，尚无真实GPU提速结论。

PNG归档兼容修复保留JPEG/PNG的原始发送字节、实际扩展名和MIME，校验base64、实际格式及完整像素解码；损坏/不支持格式继续报错。新增PNG主体图的候选/相机/当前观测来源标签。102项Python测试通过、1项跳过，40项前端测试通过，Vite构建通过；该修复不补造历史缺失记录。

hybrid-15完整回归通过：4168 passed、58 skipped、1现有Starlette弃用提示，506.56秒；前端388项全部通过，全局Ruff通过。部署937个文件并逐项核验SHA；当前Vite产物包含`replay-FdQ0hBrf.js`，没有沿用被构建替换的旧hash引用。

A9独立审计未发现左右或恢复门错误：目的地Y=+.52028m为左盒，另一盒Y=-.53492m；三次观测missing_geometry_ids均为空，黑件和两盒CAD均accepted。retry_failed_node始终可选；最后continuation选择继续选恢复能力（.44 vs retry .42），再选择stop（.40 vs retry .31），属于操作提案而非视觉事实门。最后请求64009输入tokens，精确去重后的context中历史观测占83.7%；现有去重节省37.23%，但这些只证明输入负担，不能证明造成.7判断或停止。详见`black-A9-binding-audit/report.md`。

C5新候选重验的冻结数值诊断：旧工具前缀先通过无GPU远端preflight，随后工具更新造成bundle不匹配，执行在本地校验阶段拒绝，未进入GPU；原日志保留。最终v1工具使用独立冻结前缀重新通过preflight。世界五层、禁用目标标记和24个attachment球与原归档核对，未重拟合桌面或借用仿真对象真值。实际纯规划结果另行记录。

v1真实GPU对照两组均通过，保持C5第二轮同一原反馈/几何/要求：baseline测试96个XY、77次公开高度求解，数值resolve耗时36.406秒；seeds测试12个原候选、12次公开高度求解，耗时9.453秒，均返回12个当前证明有效端点。新结果反馈逐项一致，原1e-6门不变；持物IK批次墙钟分别16.078秒/3.629秒。各组使用新的独立进程与正式warm_start，启动/预热另行记录；这是单场景单次对照，不能推导所有任务的稳定提速比例。未执行语义候选选择、轨迹规划或机器人运动。证据`unknown-C5-resolution-audit/hybrid-15-c5-revalidation-v1-comparison.json`及两组完整请求/结果归档。

独立只读复核遍历89次高度调用，重算输入hash、event272的场景/双臂反馈绑定、两份resolution签名及24个最终候选证明，均来自本次真实返回。基线与重验分别检查2002/312个世界高度样本、1946/312个携物IK样本；每个世界样本保留完整332球检查。完整OBB八角点、目标与FK底面间隙独立复算通过。placement-32两组分别选上移6mm/5mm，重验FK位置误差4.967305mm接近原5mm门，不能声称数值结果逐位复现；429个未搜索候选如实记录。

black-A10（fa8bf1a7349641848cc2ee329bd2e300）：hybrid-15，一次Responses，158.94秒Decisions自行停止，无机械动作。当前观测成功、CAD匹配/位姿已接受；source_identity=.99、destination_identity=1.0、target_0=.99，而destination_requirement_matches=.01。实际图像、ID及左盒Y=+.520241m一致，尚不能确定模型否定的内部原因。单一冻结对照保留完整input、六张原JPEG、数值与三道原题，仅精简该题并加入同题干的右盒反例；一次HTTP耗时1.570秒、6778输入tokens，左盒=.09、右盒=0，未改善。没有继续调整重采样，也未修改生产提示或门槛。详见`black-A10-binding-audit/report.md`。

A10归档兼容修复的真实验收：8个调用文件对应1 Responses/7 Decisions，43个JPEG与5个PNG引用均通过SHA、MIME、实际格式、verify及完整像素load。run.experiment.error为空；这证明本轮归档完整，不证明左臂任务成功。

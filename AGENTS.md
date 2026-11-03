# Luna 开发约束

本文仅描述当前合同；历史实验不是开发指令。修改前阅读 README 与 `.agents/skills/luna-policy-isolation/SKILL.md`。用户最新明确要求优先。

## 工作边界

- 只在 `C:/project/tianji-luna-mujoco` 和 `/home/guangyu/projects/tianji-luna-mujoco` 工作；独立 CAD 库 `/home/guangyu/projects/CAD` 已获同步授权。不要创建平行项目或工作树，不使用 sub-agent，不修改其他用户服务。
- 官方 SDK、安装包与参考工程只读；算法只通过匹配固定版本的公开接口调用。cuRobo 源码基线见 `resources/curobo/sdk-source-manifest.json`。不能修改 SDK 或省略碰撞检查来提速。
- 五容器合计最多 4 逻辑核：web 1.5、cuRobo 1、SAM3/GraspGen/FoundationPose 各 0.5。OMP/MKL/OpenBLAS 单线程，web 软件渲染最多 2 线程。本地重检查串行。可增加本项目 RAM/VRAM，以实测收益与不 OOM 为准。
- 替换功能时同步删除旧入口、实现、专属配置、无效依赖和专属测试；保留当前行为的有效测试。检查动态注册、前端、部署副本，远端同步包括删除。历史实验留在 data，不能靠历史兼容分支延长旧框架寿命。

## 模型与 graph

- 入口为一次初始 GPT-6 Luna Responses：自然语言、三相机初始 RGB、配置区域、CAD 目录、技能契约及 prompt。输出候选物体清单、独立任务要求及 `version=2` 显式 graph/subgraph/node/edge；只执行此格式，无 steps 执行器或额外 Responses 修订。
- 每个输入永久 ID 对应独立 subgraph.task 与完成账本；根 graph 协调共享能力和整体验收。graph node ID 表示控制节点，不能当作物体 ID；CAD asset ID 表示可被多个物体实例共享的模型。
- Responses 编排完整正常任务，不必预见所有失败。Decisions 输出有类型的视觉/语言判断、选项和概率；运行时可选注册能力及有类型参数形成恢复 node/edge，校验、归档后执行。Runtime 只解释 edge、维护证据与账本，不代写任务策略。
- 跨 skill 的观测、回位、换臂、建图、核验和恢复必须出现在模型 graph。skill 内允许固定算法步骤和真实前提。禁止隐藏流程、重播已完成物理效果、用空队列宣称完成。每个 subgraph 显式到达自己的 end。
- 每件交付后双臂回配置 observation 位姿并观测；最终也在该位姿。作为 completion_requirements 给 Responses 显式编排，最后交付已满足时不额外移动。加工 skill 的双臂起始前提单独核验。
- 初始候选不是已确认事实；observe_scene、confirm_inventory、verify_task_completion 由 graph 安排。完成证明绑定动作 epoch 与清单版本。产物独立永久 ID，保留输入/产物谱系及加工回执。

## Decisions 与感知

- CAD 模型选择、实例身份、可见性、支撑关系、任务结果分别判断。问题只描述本次需要的事实；历史身份/位置明确标记为历史，不与当前像素或任务目标混为一谈。
- SAM3 提供当前候选掩码；Decisions 关联永久实例、选择目录 CAD，模型匹配可缓存。不能仅凭分割词把候选当作已确认 CAD。颜色不能硬编码执行臂或模型。
- 看不到、看不完整、遮挡与语义否定分开。只有视角缺证据才考虑模型选择补充观测位姿。原位重拍不改善相同输入的置信度；相同采集复用缓存，必要采集没有任意调用次数上限。
- 有效 no/unknown 不自动重试到 yes，不修改事实阈值来制造成功。仅对无有效回答的 HTTP 429/502/503/504 或网络故障，最多三次重发同一冻结请求并独立归档，不重放技能。
- verify_place 一次 Decisions 融合三相机当前 RGB，分别判断支撑/入容器、仍夹持，以及适用时是否接近目标子区域。允许定性位置误差；不调用 SAM3、FoundationPose、OBB 或释放后深度测量。保留释放反馈、证据时序、身份绑定及规划/执行数值检查。全局完成判断由 graph 独立调用。

## 几何、抓取与放置

- 生产 `LUNA_GEOMETRY_MODE=cad_obb`：实测桌面 cuboid、未知物体当前观测 OBB、已知物体登记 CAD。粉色包装是未知 OBB；仿真中的粉色网格不是识别 CAD。未知 OBB 用 GraspGen；已知 CAD 按执行臂用离线抓姿，失败不能暗中改走 GraspGen。
- `LUNA_CAD_ROOT` 指向独立 CAD 库；项目 resources/cad 是运行副本。蓝盒 1000 面、黑件 999 面，配准和碰撞使用相同表面，禁止恢复高面数双模型分支。保持坐标、A/B 离线抓姿及其哈希绑定一致。
- FoundationPose 输入当前 SAM3 掩码、RGBD、CAD 和对应标定；选有效深度最充分的相机，输出转换到世界坐标。有效且来源/坐标绑定一致的位姿直接采用，不做 ICP 或额外几何吻合检查。新观测更新位姿，不能把历史位姿当新测量。
- 清单永久 ID 绑定 CAD/OBB。局部更新只刷新选中实体，保留其他历史实体及未知状态；动作影响几何必须失效、预测更新或重新观测并如实标记。全局更新不等于全空间可见，只有明确退役才删除实体。
- 碰撞世界只包含已观测清单实体及用户声明的工作台条件，不从仿真真值或清单外物体补地图。射线可见性仅是证据诊断，不要求全机被相机覆盖；全机环境/自碰撞、关节及场景新鲜度检查保留。
- TCP 使用校准 `gripper_front_A/B`，安装端局部 +Z 185mm；GraspGen 输入帧只在接口转换一次。三相机标定遵循只读参考工程，内外参设备序列号匹配；仿真是理想针孔，原生畸变只作来源记录。
- attachment 仅在抬升成功且确认持物后建立。首段 lift 不要求尚未建立的 attachment；建立后绑定执行臂、抓取变换和场景版本。
- 工作空间使用已有离线包络；位置可达不代表姿态 IK 或路径无碰撞。模型在数值允许且符合任务要求的臂中选择，只有一臂满足时直接解析。通用任务不按颜色固定左右臂。
- 放置保持抓取姿态为默认。工作台“指定/设定区域”使用配置 XY；普通区域按 4→8→16 分区提出 XY，Decisions 按原始语言选候选 ID，再求 Z。Z 从碰撞阈值附近求最近无碰撞高度，不固定额外 10mm。
- `perceive_placement_zone` 给区域，`resolve_placement` 求落点，`plan_place` 用 cuRobo 规划接近、放下及释放后撤离。规划含夹爪和持物碰撞、工作空间、IK 与轨迹检查。精确 XY 不得为成功率漂移，中间工作区不等于最终目的地。
- vla_unpack 的配置 XY、固定姿态和双臂起始状态是该 skill 前提，不是通用编排模板；prepare_unpack 显式执行并核验。仿真加工效果及产物随机位置只用于效果内部，视觉确认产物后再进入地图。

## 验证与文档

影响技能用途、前提、算法、结果、恢复或状态时，同步 contract/SKILL.md、docs/luna-skills-guide.md 与 README。报告真实实测，区分单元测试、健康检查和完整仿真；保留失败证据，不把一次成功说成稳定成功率。性能目标：扣除初始 Responses 后 120–240 秒，按实际阶段耗时定位。

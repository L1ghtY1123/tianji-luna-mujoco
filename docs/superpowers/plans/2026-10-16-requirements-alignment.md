# JEV 三类搬运任务执行方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按最新要求完成盒子到桌面、桌面到盒子、同一物体经桌面中转到另一盒子的多物体搬运，保持 JEV 选择 skill、skill 内部执行固定算法；同时改善工作台的位姿输入、取消场景草稿和错误展示。

**Architecture:** 沿用现有 16 个运行时 skill、Runner 决策循环和算法服务。调整清单就绪条件、编号和分区绑定，统一建图输入及碰撞对象生命周期；不建立固定业务调度器，不重写感知或规划 SDK。

**Tech Stack:** Python 3.12、MuJoCo 3.10.0、Pydantic AI 2.47.0、typesafe-sdk 0.7.1、SAM3、GraspGen、cuRobo 锁定提交 `8e734f3ced1df898990bcd92de40abce475907db`、现有 Three.js/Vite 工作台。

**Spec:** [用户要求原文](C:/Users/l1ght/Desktop/JEV项目的要求.txt)，以及本聊天最新补充：**以 grasp 时 TCP 姿态为放置旋转基准，只搜索世界 Z 轴 yaw；障碍物和目标区域也有稳定 ID；桌面 cuboid 仅初始头部建图建立一次；盒底自动确定平面目标区域，按桌面相同规则分区；工作台将 yaw 与 Z 同行、增加取消场景修改、消除顶部错误条遮挡**。

**2026-10-16 执行修订：** 用户已确认执行，候选实现和验证在独立目录进行。用户随后明确：跨时 ID 关联必须结合实测 RGB 和深度；只有尚无场景时才使用头部相机，已有场景的更新统一使用三相机；放置后的空爪 lift 完成且还有搬运任务时，立即原位三相机建全物体障碍图，再允许双臂到观测位姿并再次三相机建图。以下 Task 3、7 按此修订。历史基线对照仍描述实施前源码，不能用作当前行为说明；实际实现见 README、skill 手册及验收记录。此句记录当时状态；后续实现、部分路线实测和部署以文末链接的最新验收记录为准，不能用本方案的历史基线代表当前源码。

## 基线与边界

- 审查日期：2026-10-16；Windows 和 Ubuntu Git HEAD 均为 `d88b95f696d5603ceda9b997f2ed524bb42eb5b3`。
- Windows：`C:/project/tianji-jev-mujoco`；Ubuntu：`/home/guangyu/projects/tianji-jev-mujoco`。
- 本地已有未提交文件 `docs/jev-skills-guide.md` 和 `docs/identity-and-skill-audit-2026-10-16.md`，实施时保留；不能直接覆盖成远端版本。
- 上述基线审查阶段只分析并运行 CPU 回归；随后用户已批准实现及隔离候选验收，后续按用户继续执行与部署授权推进；Task 9 的完整数量/重复试验矩阵仍单独记载，不能把部分成功扩大成全矩阵通过。
- 当前记录中的完整物理验收只覆盖两个不同物体的盒子到桌面；另外两条路线仍需完整验收。旧记录的 1583 项通过不代替本轮验证。

## Global Constraints

- 初始识别与初始地图只使用头部相机，腕部深度不能暗中进入地图。
- 全部要求的目标及目的区域识别成功、数量约束一致后，才允许进入双臂观测运动。
- 待搬编号来自初始头部图像：行从上到下，行内从左到右；编号冻结，后续不重排。
- 全场景实体均有稳定 ID，包括待搬物体、任务外障碍物、盒子；目标区域另有稳定 region_id。搬运清单只引用需要搬运的实体，障碍物不占搬运编号和数量。区域内分区号 slot_id 与搬运编号对应，不能把整个区域 ID、分区号和实体 ID 混用。
- 单段任务整个任务只用一个执行臂；中转任务每一段各自固定执行臂，可以相同或不同。用户指定臂优先。
- 同一物体的中转第一段和第二段保留同一编号；沿用当前逐物体完成两段再处理下一编号的语义。
- 世界模型仅保留桌面 cuboid 与有效 SAM3 实体的 TSDF/ESDF；任务外已识别物体仍是障碍。桌面彩色标记是区域属性，不重复生成实体障碍。
- 同一场景生命周期中，桌面 cuboid 及其支撑平面只在初始头部观测建图时建立；后续三相机建图直接复用，仅重建或更新 SAM3 实体层。取消未应用草稿不触发桌面失效。
- 双臂到观测位姿的运动图包含全部物体；抓取地图只排除当前稳定对象 ID 的静态层。
- 已有场景的更新一律三相机；完成放置空爪撤离且仍有待搬阶段时，立即原位更新全物体障碍图，随后才能执行下一次双臂观测运动。跨时身份比较使用实测 RGB 特征与世界坐标深度表面，头部掩码重叠只作诊断。
- 物体清单、当帧实例、跨时关联、碰撞快照和持物参考分开记录，不把历史或预测点云冒充当前 RGBD。
- 放置编号与物体编号固定对应；无解不能由位置搜索悄悄换区。
- 本方案按“偶数划分”的字面要求采用 `K=2*ceil(N/2)`，包括 `N=1 -> K=2`；这会替换当前单物体使用区域中心的特例。该选择在界面和验收中明确展示。
- 放置只搜索 18 个相对 yaw：0、20、…、340 度；360 度与 0 度重复，不额外生成。
- 放置旋转基准直接取 grasp 阶段闭爪后、lift 前记录的实际 TCP 旋转矩阵 Rg，候选为 `Rz(theta) @ Rg`；不另求物体夹起前的六自由度姿态。TCP 的 XY/Z 仍按目的区域与附件球底重新计算。
- 桌面目标区域和盒底目标区域使用同一分区规则。盒底区域由 SAM3 盒子实例的实测平面自动提取，不要求用户画区域、放彩色标记或给出盒内尺寸。
- 按用户最新澄清，球底请求间距等于碰撞阈值，默认10 mm；不额外添加15 mm释放高度。端点追踪仍使用2 mm/2°容差，但实际球底间隙另须不超过碰撞阈值；球底距离与物体真实底面分别记录。
- GraspGen、SAM3、cuRobo 官方源码、安装包及 vendor 保持只读；只改项目侧适配代码和配置。复用已有镜像。
- 每一项实现同步维护 `docs/jev-skills-guide.md`、受影响 contract、运行时 SKILL.md 和 README；不把内部函数写成新 skill。

## Review Focus

- 初始有三个待搬对象但只确认两个：不能把清单当作就绪后启动运动。Task 1 覆盖。
- 同色相邻物体、重复分割、放置后形状标签变化：不能重号、合并错体或排除错误障碍。Task 2 覆盖。
- 当前目标已被排除、刚放置物体已移动：旧抓取图不能直接授权下一轮双臂返回。Task 3 覆盖。
- 指定分区不可达或被占用：只能返回该分区失败，不能将对象放到其他空区。Task 4 覆盖。
- 规划后物体掉落、反馈未知或取消后结果迟到：不执行旧持物轨迹，不自动恢复。Task 6 覆盖。
- 桌面从头部图切到三相机图后仍保持同一几何和来源；盒底自动区域拥有稳定 ID 并沿用同一分区。Task 3、4 覆盖。
- 误拖和表单修改可一次取消，长错误不挡三路相机/点云诊断按钮。Task 8 覆盖。

## 需求与源码对照

| 要求 | 当前实际实现 | 方案处理 |
|---|---|---|
| 三类路线与阶段固定臂 | `task_progress.py` 已有三路线、stage_arms、pending_route_id | 保留；补各路线和跨臂验收 |
| 自然语言特征、数量、角色 | `decisions.py` / `task_request.py` 已有 questions | 补分组配额、语义判断置信度与清单完整性 |
| 全部识别后才运动 | `assess_task` 成功返回即可置 task_assessed；continue 不保证所有成员已解决 | 增加真实 inventory_ready 条件 |
| 初始头部观测 | `observe_scene` 只对相机 3 做分割 | 建图却遍历 1/2/3 且 head_only 默认 false；必须同步收窄 |
| 初始只保留实例和桌子 | `Algorithms.build_scene` 的 observation 分支清空掩码并使用 background 模式 | 增加无排除目标的语义建图输入；不能假装选一个目标 |
| 按头部行列编号 | `TaskProgress.assign_display_ids` 按头部光心距离排序 | 改为图像行列顺序，只在完整清单冻结时分配 |
| 障碍物与目标区域也有 ID | 感知实体已有内部 ID，搬运账本只纳入任务成员；区域/分区身份尚未统一 | 沿用场景 ID，将 task membership、region_id、slot_id 分开记录；障碍不计入任务数量 |
| 同时和跨时关联 | `association.py` 已拆成两函数 | 保留边界，修复别名重复与不充分匹配 |
| 桌面延伸受碰撞阈值限制 | 底层 RANSAC/朝机器人裁剪已存在；项目侧强制向下 0.30 m 再检查 | 改为有界延伸，检查失败前先缩小推导延伸部分 |
| 桌面仅初始建立一次 | `MeasuredTableCache` 已存在，但 key 包含相机集合和全部 planning_settings | 按场景生命周期冻结初始桌面；头部转三相机及后续建图不能重新拟合 |
| 双臂同时到观测位姿 | 双臂联合规划、同步时间轴、联合执行和复核已有 | 保留；补运动地图新鲜度与目标包含条件 |
| 三相机更新只排除当前目标 | selected 图已有实例层、当前目标排除和桌面复用 | 保留；绑定冻结编号和关联版本 |
| 左右 TCP 完整姿态 | `geometry.grasp_targets` 已有 A 臂局部 Z 轴 180 度转换与工具偏移 | 保留并把选臂后的转换明确纳入候选生成契约 |
| IK 后 batch 抓取规划 | 已有 IK100、轨迹 batch16、三阶段 | 保留；统一 IK 与轨迹的总等待期限 |
| 抓取后 attachment | verify_grasp 读反馈；plan_place 开始时 prepare_attachment | 时序已符合“确认后、放置前”；不新增空壳 skill |
| 偶数等面积分区 | `placement_layout.partition_region` 已有径向等面积分区 | 补固定方向、头部编号、中心直线校验和 N=1 规则 |
| 盒底自动确定目标区域 | `measured_bin_reference` 已提取水平底面点及凸包 | 从可用实测平面确定区域并分配 region_id，复用桌面分区；无需显式标记 |
| 对应 ID 放置 | placement_request 只有总数和目标 ID 列表；placement 遍历所有分区 | 增加 slot_id，严格只搜索当前物体对应分区 |
| grasp TCP 为放置姿态基准 | `Rz(yaw) @ R_grasp` 和 grasp 时 TCP 记录已存在 | 保留并固定参考时刻，直接以 TCP 验收，不新增物体姿态估计前提 |
| 球底 Z 贴近平面 | 已按上传球包络求 Z；默认 15 mm | 保留公式，说明可配置余量及真实物体底面的区别 |
| place 带附件、释放后撤离 | 前两段持物，第三段张开无附件；复用 SDK plan_grasp | 保留；新鲜反馈与放后地图必须补齐 |
| 每个后续对象重观察 | build_scene 已包含双臂移动、三路采集、关联、重建 | 取消全任务固定 8 次观测对正常规模的限制 |
| JEV 自主选 skill | 主循环每轮调用 choose_skill，契约校验失败后返回证据 | 保留；删除提示中看似固定调度命令的表述，改成前提与证据说明 |
| yaw 与 Z 同行、草稿可取消 | position 已用两列网格；yaw 单独追加；已有 replaceDraft 和拖拽取消接口 | yaw 加入位置网格，增加取消修改按钮，清除整份未应用草稿及预览 |
| 报错不挡相机/点云按钮 | `styles.css:103` 将错误条固定在 top:68px 并横跨全宽 | 改为右侧正常布局的紧凑错误卡，摘要与可展开详情分开 |

主要源码入口：`runner.py:615,914,1261,1439,1520,1604,1852`；`task_progress.py:87,166,228,312`；`algorithms.py:141,242,339,542`；`placement.py:219,405`；`semantic_builder.py:37,115`。

## 设计选择

采用现有体系内的增量修改。只改提示词不能建立编号、几何与配额不变量；重写全部 runner 和 skills 会扩大验证范围。此方案将必须同时成立的规则放进现有状态和契约，将语义选择和恢复留给 JEV。

正常路径是能力依赖，不是硬编码顺序表：JEV 选择观测、清单评估、目标/目的地、工作空间评估/选臂、建图、生成候选、规划、执行、验证。程序只拒绝不满足前提的调用。所有 skill 调用仍记录对应 JEV 决策；skill 内的分割、RANSAC、IK、轨迹生成、attachment 和夹爪动作是固定算法子步骤。

保留 16 个 skill。将初始地图从 `observe_scene` 内移入 `build_scene` 的运动准备部分：先由 observe_scene 提供头部识别结果，再由 assess_task 完整确认清单；JEV 选定目标、目的地和臂以后，才进入 build_scene。这样无需增加独立建初始图 skill，也不会在未识别完整时启动双臂。

## Task 1 任务约束与完整清单

**Files:** 修改 `src/jev_mujoco/task_request.py`、`decisions.py`、`task_progress.py`、`runner.py`、`skills/assess_task/contract.json`、`skills/build_scene/contract.json`；新增 `src/jev_mujoco/inventory_order.py`。同步对应说明与手册。

**Interfaces:**

- `head_image_order(objects: list[dict], head_view: dict) -> list[str]`：返回头部图像的对象内部 ID 顺序；没有当前有效头部掩码则报具体错误。
- `TaskProgress.freeze_inventory(observation: dict) -> dict`：校验数量/成员/来源，冻结编号、初始引用和总数；返回带 revision 的清单。与现有 observe 中的“发现范围冻结”分开，避免同名布尔事实掩盖不同含义。
- 清单增加 `inventory_finalized`、`inventory_revision`、`initial_head_observation_id`；每项保留 `id`、`display_id`、属性、初始/最新位置与观测引用、来源区域、各阶段目的地、阶段臂、状态、失败和完成反馈。
- 沿用现有场景实体 ID 和关联结果，搬运清单只是实体集合的任务成员视图。任务外障碍物保留 ID、几何与观测来源；目的区域保留 region_id、支撑实体 ID、平面与轮廓来源。一个盒子既有实体 ID，也可关联一个盒底 region_id，不把区域再当作独立物理障碍。
- 数量约束保存总数及来自原文的组约束：`group_id/source_text/requested_count/member_ids`。JEV 选择哪些实测实例满足组约束；代码只核对数量、重复和交集，不硬编码颜色优先。

- [ ] 在 `tests/test_task_progress.py`、`test_task_request_scope.py`、`test_transfer_decisions.py` 和新增 `test_inventory_image_order.py` 写失败用例：四角顺序；同排轻微纵向偏移；近处右下不抢 #1；2 红 1 绿不能截成 3 红；要求 3 个只见 2 个；未知成员未解决；编号前修正成员不占旧额度。
- [ ] 运行上述文件，确认失败原因是当前缺少完整清单/图像编号，而非 fixture 格式错误。
- [ ] 实现头部掩码中心与行分组：按中心纵坐标建行，初始行容差取候选掩码高度中位数的一半，行内按横坐标；行基准固定，避免链式合并。将容差作为明确工程参数记录；以真实截图回放调整，不宣称是通用识别阈值。
- [ ] 首次冻结前可修改成员；冻结后不因新检测顺序、距离或遮挡重排编号。显示编号不再修改 membership 或执行总数截断。
- [ ] 验证三个待搬物体、两个障碍物和两个目的区域同时存在：所有实体/区域 ID 唯一稳定，搬运清单仍只有三项；障碍不参与 N 和 slot 分配，完成物体仍保留原 ID 并作为障碍。
- [ ] 对成员、路线、数量和完成结论使用语义置信策略；当前 0.60 作为已有默认起点，低置信返回 unknown。合法等价动作的偏好选择不一律套身份阈值。
- [ ] `continue` 不等于 inventory_finalized；build_scene 契约明确增加 inventory_finalized 和 destination_ready 前提。缺少组成员/来源/目的区域时，返回缺失项，运动契约保持不满足。全部/所有仅能按实测源区域与遮挡证据报告完整性，不声称发现了不可见物体。
- [ ] 测试通过后提交本项实现及文档；保留清单状态 JSON 和头部编号图作为验收样例。

## Task 2 两类关联与实例去重

**Files:** 修改 `src/jev_mujoco/perception.py`、`association.py`、`runner.py` 中关联/重枚举消费；扩展 `tests/test_instance_association.py`、`test_identity_lifecycle.py`、`test_task_perception.py`，新增 `test_same_camera_alias_identity.py`。

**Interfaces:** 沿用 `associate_camera_detections(detections, views)` 和 `associate_inventory(previous, current, *, point_predictions_world=None, identity_states=None)` 的结果结构；mapping 必须一对一，未决状态必须显式返回。冻结清单不得被 reenumerate_ambiguous 静默换号。

- [ ] 先加入 cube/block 同帧偏移 2 px 的审查反例，以及相邻同色、嵌套盒子/内容物、互补视角、放置后 cube 变 cylinder、连续多次未见再出现的反例。
- [ ] 同相机先依据原掩码、唯一像素归属和双向实测表面支持去重，再跨相机融合；不能仅降低 95% 阈值或按近距离合并。
- [ ] 跨相机只比较同一 capture 的世界几何、投影与遮挡；不使用旧任务编号决定合并。跨时再消费物体生命周期和历史证据；两层分别记录理由。
- [ ] 对未移动、持物、刚释放、已放置分别限定候选位置。跨时比较实测 RGB 色度距离与注册深度表面支持，掩码 IoU 仅作诊断，不设唯一颜色或掩码优先分支。保留目的区域剩余唯一候选和双向唯一规则；对无充分证据的匹配返回未知。
- [ ] 当前目标关联不唯一时，不允许选定需要排除的静态层；已完成对象不能进入下一抓取候选。第二段必须匹配同一稳定 ID。
- [ ] 跨相机和跨时关联覆盖任务外障碍物与盒子，不只处理搬运清单；目标区域关联到其稳定支撑实体。重新观测和任务角色变化不分配新实体 ID，关联不确定时保留未知证据。
- [ ] 执行测试和录制帧离线回放，输出每项 old_id/new_detection_id/method/evidence/status；再提交本项。

## Task 3 建图范围和观测运动

**Files:** 修改 `runner.py`、`algorithms.py`、`perception.py`、`semantic_transport.py`、`perception_mapping.py`、`semantic_builder.py`、`measured_table_mapping.py`、`geometry_service.py`、相关 contract；新增 `tests/test_head_semantic_observation_map.py`、`test_return_observation_scene.py`，扩展 `test_measured_table_mapping.py`、`test_atomic_scene_cycle.py`。

**Interfaces:**

- `Algorithms.build_scene(snapshot, target_id, *, planning_settings=None, exclusion_state="selected", head_only=False)` 保留入口；`exclusion_state="observation"` 明确为 target_id=None、排除列表空、全部有效语义实例参与碰撞。
- 项目语义传输增加有版本的 targetless 模式：初始头部使用相机 3，已有场景使用相机 1/2/3，observation 无 selected ID 且不排除任何物体；selected 模式仍要求目标确实存在且三相机属于同一 capture。生产者、解码器、身份指纹和 builder 一起调整。
- 新地图返回相机列表、observation_id、scene_token、排除 ID、包含实例、桌面拟合/延伸参数和背景删除计数；只在完整构造成功后发布。桌面单独返回 table_revision、initial_head_observation_id 和复用标记，实体层引用各自当前观测，来源不能混写。
- `MeasuredTableCache` 改为场景生命周期内固定桌面模型：初次从头部测量生成，后续只读取与校验。生命周期依据明确的场景重置/应用及标定版本，不依据 capture、相机数量或无关规划设置；支持平面、裁剪高度与放置支撑参考均消费同一冻结桌面。

- [ ] 加入失败测试：腕相机出现大障碍但头部没有时，初始地图不能受其深度影响；头部所有有效 SAM3 实体和桌子保留；未分割背景不保留；初始目标不得排除；没有目标时不能伪造目标 ID 通过协议。
- [ ] `observe_scene` 仅识别、提供清单证据；`build_scene` 初次使用已确认清单对应的头部 capture 建包含所有物体的运动图，再联合移动双臂、同步三相机采集、关联、重建 selected 图。当前任务不会在 skill 内再次询问 JEV。
- [ ] 删除将“完整三相机观测”作为删除背景的唯一开关；以明确的 scene policy 区分初始头部全实例图、原位三相机全实例运动图和三相机目标排除图。不能只设 experimental flag 却让初始图继续走 background。
- [ ] 保留机器人自身过滤；RANSAC 桌面估计和 cuboid 有界延伸仅初始头部建图执行。桌面上表面不抬高；向机器人及向下的推导延伸均受机器人球硬间距约束。已有 0.30 m 只作延伸上限，初次必要时裁短推导部分；实测桌面本身冲突则报告失败。后续只检查碰撞，不为适配新机械臂姿态缩桌子或重新拟合。
- [ ] 测试初始头部→三相机→放后原位三相机→观测位姿三相机的 table builder 总调用次数严格为 1，几何、支撑平面和 table_revision 保持不变。实体层仍按当前观测更新；`perception.py` 使用已有 table hint 接口传入冻结参考，避免重复估计产生不同桌面高度。
- [ ] 显式 Reset、应用新的物理场景或标定改变才结束旧桌面生命周期，下一次初始头部建图重建；清除单轮 selected 图不得清掉桌面。任务中涉及桌面几何/硬间距的参数失配返回证据，不暗中重建；无关设置不失效。取消网页草稿不触发上述任何失效。
- [ ] **补齐放后返回的地图缺口：** `verify_place` 在确认释放和空爪撤离后记录阶段完成；如仍有搬运阶段，立即在当前位姿采集三相机 RGBD/SAM3 并建立全物体障碍图，包含刚放物体、其他待搬物体和任务外障碍。下一次 `build_scene` 使用这张图授权双臂返回观测位姿，再三相机重建。更新失败必须在运动前返回失败；重复验证不能重复计入已完成阶段。
- [ ] 若原位三相机测量不能支持返回路径，返回证据给 JEV；不能直接复用缺失已放物体的旧图。首次已有有效头部 capture 直接复用；已有场景时只接受三相机更新，仅重建 SAM3 实体层，直接装入同一初始桌面 cuboid，Mapper 模型继续复用。
- [ ] 三相机图只排除当前稳定 ID；重叠别名、未知关联、漏掉已知障碍时拒绝发布。其他待搬、已完成、盒壁和桌面始终参与规划。
- [ ] 先运行上述 CPU 测试及 `test_perception_mapping.py`、`test_mapping_preparation.py`、`test_dual_observation_adapter.py`；GPU 阶段验证旧/新图对象列表、排除列表和两臂完整轨迹。通过后提交。

## Task 4 固定分区与编号绑定

**Files:** 修改 `perception.py`、`placement_layout.py`、`placement.py`、`task_progress.py`、`runner.py`、`table_markers.py` 的区域元数据；扩展 `test_placement_layout.py`、`test_region_placement.py`、`test_place_region_batches.py`、`test_visible_perception.py`、`test_bin_source_views.py`，新增 `tests/test_object_slot_binding.py`。

**Interfaces:**

- `partition_region(polygon, target_count)` 保留几何入口；调用侧依据初始头部投影给 cell 增加 1 开始的 slot_id，并冻结编号方向。
- 桌面和盒底统一输出区域记录 `region_id/support_entity_id/source/plane/polygon/observation_id`；盒底复用 measured_bin_reference 的实测平面证据，自动创建稳定区域，region_id 与盒子实体 ID 显式关联。
- `TaskProgress.placement_request(destination_id)` 增加 `object_id/object_display_id/region_id/slot_id/inventory_revision/layout_revision`；兼容现有 destination_id 并显式映射实体和目标区域。总数来自冻结清单，不是剩余物体数。每个阶段/目的区域保存自己的 layout，物体编号不变。
- `release_candidate_groups(..., placement_request=...)` 只生成指定 slot 的 18 个 yaw；占用、边界、IK 或轨迹全部失败时返回带 slot_id 的失败，不迭代其他区。

- [ ] 先测试 N=1/2/3/4/5 对应 K=2/2/4/4/6；面积和等分误差；四分区严格左上1、右上2、左下3、右下4；物体2只能去区2；完成或暂缓一件后 K 和其余编号不变。
- [ ] 对正方形、矩形和圆形测得区域，使用通过区域中心的成对直线形成偶数等面积区域。保留当前 polygon 面积算法可复用部分；矩形六分区不能直接用六个等角扇区冒充等面积。
- [ ] 固定初始头部投影方向，分区形状/中心存于区域参考坐标；每次重观测更新实测位置但不按剩余可见面积重新划区。已放物体遮住 marker 不能让区域中心和编号漂移。
- [ ] 盒底无需显式目标标记：从同一 SAM3 盒子实例中识别有实测支持的底面平面，区别底板与顶沿，再在可用平面内自动选定目标区域。在实测底板平面凸包内取内接矩形，不要求还原整个盒腔；同一平面的局部遮挡和采样密度变化不切碎区域。初始头部确认底面，首次plan_place使用三相机几何划分并冻结布局。区域过小、无底面或只有盒沿时返回具体失败，不能要求正常任务必须另给彩色 marker。
- [ ] 桌面区域与盒底区域共用偶数等面积划分、初始头部编号方向、对象—slot 绑定和 18 yaw 搜索。自动区域首次确认后冻结轮廓/分区；后续物体遮挡不能让目标区域缩小或重新编号。支撑实体确认移动则更新其测得位置并使旧轨迹失效，不能凭空推断不可见盒子位姿。
- [ ] 测试无 marker 的可见盒底、部分可见但足够大的底面、纯盒沿、稀疏/不连续底面、已放物体遮挡，以及不同目的盒分别保留 region_id；验证盒底与同形桌面区域得到一致的分区语义。
- [ ] 指定区中心约束作用于既有夹持末端参考点 g，而非安装 TCP 或物体可见质心；外部区域边界、占用和盒壁仍通过持物几何及碰撞图检查。分区线是分配线，不额外变成实体障碍。
- [ ] 默认每阶段一个目的区域，N 为任务总数；若保留旧多目的地能力，也保持全局编号并使用同一全局 N 的分区数，不按各目的地剩余人数重编。空分区可保留。
- [ ] 运行对应测试、保存区域轮廓/中心/编号叠图和失败原因，再提交。

## Task 5 抓取工具坐标和放置姿态

**Files:** 修改 `algorithms.py`、`geometry.py`、`placement.py`、`skills/generate_grasps/contract.json` 与 `plan_place` 说明；扩展 `test_gripper_reference_placement.py`、`test_world_yaw_placement.py`、`test_release_15mm.py`；新增 `tests/test_selected_arm_grasp_frames.py`。

**Interfaces:** `Algorithms.generate_grasps(snapshot, target_id, *, arm)` 明确消费已绑定执行臂；原 GraspGen 服务仍接收点云和已支持的夹爪配置，不凭空增加 arm 参数到第三方 API。项目侧返回原始抓姿和转换后的 `T_world_tcp`、arm、tool_profile_revision；IK 不再对已转换结果重复翻转。

- [ ] 用任意倾斜旋转的同一抓姿测试 A/B：Z 相同，X 与 Y 同时反向，旋转行列式保持 +1，185 mm/195 mm 工具偏移方向正确。不能只翻 X 得到镜像反射矩阵。
- [ ] GraspGen 用当前目标经三相机关联的实测点；一台相机没看到目标不等于无效。候选先做完整工具变换，再送指定臂 batched IK，之后 plan_grasp_batch。
- [ ] 直接复用 grasp_support_reference.T_world_tcp_at_grasp：选定臂到 grasp、闭爪完成且 lift 前记录实际 TCP，保存 Rg、arm、tool_profile_revision 和来源时刻。放置候选采用 `R(theta)=Rz(theta) @ Rg`；保留抓取时倾斜，只增加世界 Z 旋转。
- [ ] 按用户最新简化，以 grasp TCP 为姿态基准，不新增物体夹起前姿态估计、无滑移证明或物体六自由度误差门槛。轨迹验收直接比较 TCP 与候选旋转；正常持物反馈和掉落检查沿用 Task 6。中转第二段使用该次重新抓取的 TCP 基准，不沿用第一段 Rg。
- [ ] 对指定分区中心 q、附件球 (ci,ri)、支撑高度 h：`t_xy=q_xy`；`t_z=h+d-min_i((R*ci)_z-ri)`。同一组上传球用于规划及释放检查。
- [ ] d直接取hard clearance（默认10 mm）；区域只确定TCP原点的世界XY，不对attachment的XY投影加边界或二维占用限制。实际三维碰撞检查保留，不在无解时提高高度。
- [ ] 测试18个唯一yaw、无roll/pitch附加变化、末端XY固定、球底高度、A/B同物理抓姿和倾斜物体；通过后提交。

## Task 6 持物执行、三段轨迹与失败边界

**Files:** 修改 `runner.py`、`carrying.py`、`algorithms.py` 及必要的 `backend_builder.py` 项目适配；扩展 `test_current_grasp_attachment.py`、`test_pre_release_place.py`、`test_place_saved_scene.py`、`test_release_verification.py`，新增 `test_live_holding_before_motion.py`、`test_grasp_shared_deadline.py`。

**Interfaces:** 实时反馈始终来自当前 robot_snapshot；规划锁定的夹爪参数独立保存在 binding。`validate_motion` 不得用历史夹爪角覆盖当前测量。一次 plan_grasp 的 IK 和轨迹共享同一 monotonic deadline。

- [ ] 先重现规划后夹爪已空/未知、对侧夹爪变位、attachment与目标不一致、IK超过总时限、取消后迟到结果等测试。
- [ ] 保持执行前开爪、approach、grasp、闭爪、lift、verify_grasp 的既有职责；按文档在 lift 后确认，再由 JEV 调用 plan_place，其首步完成 attachment。当前 lift 尚无持物碰撞模型，验收必须如实记录，不宣称它已有 attachment 检查。
- [ ] execute_place 入口及 approach/place 阶段边界读取新反馈，held 才继续；unknown/empty 返回原始证据并保留正确物理状态，不能直接自动重抓。
- [ ] place 前两段使用实际闭爪与 attachment；释放端点用实际 TCP 和球底校验。张开反馈确认以后 detach，再空爪撤离；detach失败、释放失败、撤离失败分别保留已执行阶段，不能统一重置为空手成功。
- [ ] 保留 SDK 同一 plan_grasp 组合用于放置前两段，第三段以释放后的开爪模型规划；不将三段全部按持物模型计算。所有轴、符号和世界/工具系设置与高级参数一致。
- [ ] verify_place 继续按完整释放与夹爪反馈验收，结果标记 release_only；最终落点由独立仿真验收评分，不让真值进入运行决策。
- [ ] 运行上述定向测试、两臂执行分支测试，提交本项。

## Task 7 预算、JEV 问题与界面证据

**Files:** 修改 `runner.py`、`decisions.py`、`planning_settings.py`、`observation_labels.py`、`web/src/object-list.js`、`web/src/run-evidence.js`、`web/src/camera-panel.js`；新增 `web/src/placement-overlay.js` 和 `web/tests/placement-overlay.test.js`，由现有相机面板消费区域投影；扩展 `test_inventory_compaction.py`、`test_recovery_scene_budget.py`、`test_observation_labels.py`、前端对应测试。

**Interfaces:** 清单详细内容留在数据和日志，JEV 接收精简成员、阶段、指定臂、slot_id、版本、缺失/失败证据和全部 skill 目录；不得通过截断丢掉未知项或配额。

- [ ] 增加至少8件单段、4件两段任务的非运动回放，首先证明当前全任务8次观测限制不足。
- [ ] 冻结清单后设 G=N*阶段数。正常 capture 预算为2G：初始头部1次、每轮观测位姿三相机G次、放置后原位三相机G-1次。恢复观测单独保留8次预算；模型视觉辅助8次预算与相机采集预算分开。
- [ ] 将固定400步决策上限改为可解释的规模预算，初版 `max(400, 40*G+80)`，保持连续契约拒绝和每目标重试的既有有限约束。预算耗尽报告 needs_attention，不声称目标不存在。
- [ ] JEV questions 明确全场景 ID 与搬运清单区别、新编号、固定分区、完整清单、grasp TCP 姿态基准和阶段臂，并说明盒底自动区域无需 marker；清理“初次距离编号”“固定黄色目的地”等与原文可能冲突的残留。目的地来自实际观测和原文，黄色只是当前场景默认值。
- [ ] 独立问题可同请求并行，但不能引用同次其他答案；只有需要前一次输出构造新候选/状态时才发下一请求。保留现有中文语义回放，并补错误置信度及矛盾组约束。
- [ ] 界面显示搬运编号、内部追踪 ID、任务外障碍 ID、目标 region_id/分区号、当前阶段/执行臂、被排除的对象、实际相机来源和关联未知原因；搬运目标列表不混入障碍。只给当前帧有效掩码贴标签；区域叠图注明测量来源与更新时间。显示安装 TCP 与夹持末端的不同坐标含义，以及桌面来自初始观测、正在复用。
- [ ] 定向Python及前端测试、真实模型语义探针在实施验收阶段执行；通过后提交。

## Task 8 工作台编辑与错误交互

**Files:** 修改 `web/index.html`、`web/src/main.js`、`web/src/styles.css`；仅在需要复用时调整 `web/src/model.js`、`web/src/scene-view.js`、`web/src/scene-preview.js`。扩展现有 `web/tests/main-flow.test.js` 中场景草稿/请求边界用例，复用 `scene-preview.test.js`、`object-drag.test.js`、`bin-editor.test.js`。按仓库要求在 `docs/jev-skills-guide.md` 记载界面修改与 skill 行为无变化。

**交互约定：**

- 普通物体和盒子的位置输入仍为两列：首行 X、Y，第二行 Z、yaw；每个字段明确 mm 或 °。桌面标记的 Z 受桌面约束，第二行显示只读 Z 和可编辑 yaw，继续禁止手动悬空标记。尺寸字段保持原语义。
- 「应用场景修改」旁增加「取消修改」。有未应用草稿且未在请求中时可用；一次撤销拖动、位置/yaw/尺寸、类型/颜色、增删和桌面参数草稿，恢复最新已接收的后端场景及其确认状态。按钮含义是放弃未应用修改；已经应用的物理场景不由该按钮回滚。
- 错误提示放在右侧运行记录标题下的正常布局内，显示简短摘要、展开详情和关闭按钮；长错误码/JSON/堆栈放入有限高度、可滚动且可复制的详情。移除当前横跨全宽的 fixed 顶部错误条。参数弹窗内错误只在弹窗内显示；运行失败的详细证据仍保留在记录中。

- [ ] 复用 vectorFields 返回的网格容器放入 yaw，实际浏览器检查 X/Y、Z/yaw 排列及单位；不为纯 CSS 排列另写镜像测试。
- [ ] 取消先停止拖拽交互，再用最新 state.snapshot 调用 replaceDraft，清 dirty/预览并重绘表单与按钮；选中对象仍存在则保留，否则选择有效对象。取消不调用 /api/scene、/api/reset、/api/settle 或 /api/confirm，不改变 generation、物理确认基线、桌面缓存和任务记录。
- [ ] 用 main-flow 的真实入口与网络替身验证误拖后取消、非法输入后取消、增删后取消、桌面/盒子参数取消及后端快照更新后取消；断言无变更请求、表单与预览恢复、开始按钮按后端 confirmed 状态恢复。已有拖拽 Esc 取消继续只取消正在进行的拖动。
- [ ] 摘要与详情分离，使用 textContent 安全展示原始错误；重复快照不反复弹出已关闭的同一错误，新错误仍可提示。关闭提示不清除任务失败事实，也不自动恢复或重试任务；沿用已有去重状态，避免重复展示同一长错误。
- [ ] 浏览器检查常用桌面尺寸及窄窗口：短错误、长无空格错误、多条校验错误、运行失败和参数弹窗错误；三路相机/点云诊断按钮始终可见可点，错误卡不会横向撑开页面，键盘可展开与关闭详情。
- [ ] 运行现有相关前端行为回归及 npm run build，保存修改前后截图；该界面项可先独立落地，不依赖机器人算法改造。

## Task 9 集成与 Ubuntu 验收

**Files:** 测试入口沿用 `tests/route_transfer_smoke.py`、`continuous_transfer_smoke.py`、`dual_observation_gpu_smoke.py`、`perception_mapping_gpu_smoke.py`；按新字段更新 fixtures 和断言；新增本轮验收记录并维护 README/手册。

- [ ] 本地运行 `.venv/Scripts/python.exe -m pytest -q`、`.venv/Scripts/python.exe -m ruff check src tests`、`git diff --check`；前端在 web 中运行 `npm test` 和 `npm run build`。针对环境缺失的GPU测试明确列出，不算通过。
- [ ] 在隔离候选目录中复用现有镜像验证，避免挂载中的生产目录热重载打断任务；测试不增加无用途镜像，不升级SDK。
- [ ] 做真实 SAM3 初始头部/三相机检查，逐个核对待搬/障碍/区域 ID、初始编号、去重、实例层、桌面裁剪、背景删除、当前目标排除和已完成对象恢复；跨整轮检查桌面只建立一次、geometry/table_revision 不变。
- [ ] 三类路线各跑 N=1、N=3、N=4；三件验证补四区，四件验证每区绑定。桌面到盒子与中转终点盒均验证无需显式盒底标记即可自动建区域、编号和放置。每个基准场景至少连续通过3次，记录所有试次而非挑成功录像。
- [ ] 单段分别验证左臂/右臂固定；中转覆盖 A到A、A到B、B到A、B到B，源盒与终点盒不同，同对象ID跨段不变；含障碍场景保留碰撞检查。
- [ ] 注入缺目标、数量冲突、同色歧义、对应分区占用、IK无解、抓空、掉落、反馈未知、释放失败和取消。核对拒绝点、持物状态、JEV恢复决定与任务结论，禁止其他分区或物体偷偷代替。
- [ ] 保存 instruction、JEV state/questions/答案/置信度、清单、编号图、每轮三相机图与 ID 映射、region/slot 绑定、地图对象/排除清单、初始桌面模型和复用记录、每次 grasp TCP 基准与抓放矩阵、18 yaw 结果、完整附件球、实际反馈、阶段时间和最终评分。
- [ ] 分开记录冷启动、暖态、GPU等待、模型初始化、分割、建图、IK、轨迹与物理执行时间。先满足新语义再比较性能；不以减障碍或改间距换速度，不承诺固定总耗时。
- [ ] 全部通过后核对生产任务空闲、保存配置与版本，按授权范围同步 Ubuntu；GPU服务改动重载现有进程，普通挂载代码遵循既有热重载。核对本地/远端提交与文件指纹、健康、Registry、参数和实际加载代码，记录回退版本。

## 本轮实际核查结果

18个相关CPU测试文件共 **219 passed in 5.12s**，覆盖现有账本、三路线、questions、两类关联、双臂观测、桌面、放置、附件、契约和上下文压缩。命令使用项目 `.venv/Scripts/python.exe -m pytest -q`；未运行GPU或真实模型调用。

额外合成检查直接调用当前源码：头部四角对象按距离得到右下1、左下2、右上3、左上4；N=3确实生成4区，但顺序是世界XY径向顺序；placement_request没有slot_id或当前对象编号；N=1当前只有1区。这些说明新需求不能靠已有测试通过推定已经满足。

检查了远端现有 cuRobo 安装中的公开 `MotionPlanner.plan_grasp` 签名和项目 batch 适配；后续按同一锁定版本执行。TypeSafe问题独立性的依据是[官方State说明](https://docs.typesafe.ai/concepts/state)，候选与未知选项的依据是[官方Choice说明](https://docs.typesafe.ai/primitives/choice)。本方案不依赖SDK升级。

本轮补充审查确认：grasp TCP 参考已存在；measured_bin_reference 已可提取有上方盒壁证据的底面平面；桌面缓存 key 确实随输入相机集合变化。网页 renderEditor 将 yaw 单独追加，replaceDraft 已可清除整份草稿，顶部错误条为 fixed 全宽布局。以上已纳入 Task 1—8；本次仅修订方案，尚未实施这些行为。

算法实施顺序：Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 9；Task 8 可先独立完成，最终与算法一并进入 Task 9 验收。行为修改先新增有意义的失败回归，再做最小实现和验证；纯界面布局通过现有构建与浏览器视觉检查，不另写镜像测试。

## 2026-10-16 用户追加的固定场景清单要求

初始头部发现全部实体后固定 scene_entities，搬运成员引用其中的全局 ID；障碍、盒子、桌面和区域均保留身份。后续 SAM3 只运行初始实体的颜色/分割描述，不再次执行通用形状探索或 cube→block 补查。查询保留候选实体 ID，跨时匹配仍需 RGB、注册深度与已执行搬运目的地证据，不能用清单数量硬凑匹配；未知检测不加入地图。桌面继续复用首次 cuboid，盒底目标区域在首次运动前确认。验证顺序为固定查询/角色/关联回归、完整测试、既有真实 RGBD 帧调用 SAM3 重算、实际流程；完整 pytest 与 GPU 运动试验串行执行。

## 本轮用户最新澄清

TCP表示TCP_Link原点，指定区域/slot确定其世界XY；旋转为Rz(世界yaw)@Rg，球底间距直接取碰撞阈值。撤销此前185 mm参考点反算XY、15 mm释放高度及attachment投影边界/二维占用限制。初始固定清单及后续RGBD关联、三相机并集、放后立即更新和桌面复用继续执行；真实流程中发现的完成判断及observed成员更新漏项同步修正。以下历史测试结果不代表本轮验收，最新结果见新的TCP世界坐标验证记录。

# Luna 当前技能与模型边界

本页描述当前实现；历史测试数据留在各实验报告中，不能作为当前 prompt 或技能合同。开发约束见 AGENTS.md，完整技能参数与前提见 skill_catalog 中对应 contract.json 与 SKILL.md。

## 编排

自然语言、三相机初始 RGB、工作台配置、CAD 目录和技能说明输入一次 Responses，生成候选清单及 version=2 graph/subgraph/node/edge。根 graph 协调独立物体 subgraph。运行时 Decisions 判断当前事实、选择已有 edge，或选择有类型的恢复能力形成经校验的 graph patch。Runtime 不编造任务流程，不追加 Responses 调用。

正式注册表包含以下 21 个能力；旧 plan_transport、execute_transport、drop、verify_drop 的契约和执行器均已删除。

| 能力 | 职责 |
|---|---|
| move_to_observe | 显式规划并移动到所选观测位姿 |
| return_to_initial_pose | 显式返回绑定的任务初始姿态；是否使用由任务要求与 graph 决定 |
| observe_scene | 当前身份关联及需要的几何更新；不隐式移动 |
| confirm_inventory | 确认清单与任务绑定 |
| identify_target | 绑定当前目标与实际支撑关系 |
| identify_destination | 绑定目的表面、容器或配置区域 |
| choose_arm | 在任务约束、工作空间和当前几何允许范围内选臂 |
| build_scene | 发布模型选择范围的当前碰撞几何 |
| generate_grasps | 已知 CAD 使用分臂离线抓姿；未知 OBB 使用 GraspGen |
| plan_grasp | cuRobo 规划接近、抓取、抬升 |
| execute_grasp | 执行选定抓取轨迹与夹爪动作；lift 成功且确认持物后建立 attachment |
| verify_grasp | 核验实际夹爪与 lift/持物证据，不移动或重新采集 |
| perceive_placement_zone | 取得目标支持区域，解释当前搬运的语言位置要求 |
| resolve_placement | 配置 XY 直用；普通区域分区提出 XY，Decisions 选择后求近阈值无碰撞 Z |
| plan_place | cuRobo 规划接近、放下及释放后撤离 |
| execute_place | 执行放置、松爪及撤离 |
| verify_place | 三相机 RGB 定性核验本次放置 |
| prepare_unpack | 显式满足配置化 VLA 技能的双臂起始前提 |
| vla_unpack | 执行配置化加工效果；仿真真值仅在效果内部 |
| verify_unpack | 当前 RGB 核验加工结果并登记独立产物及谱系 |
| verify_task_completion | 检查独立任务账本、图要求与全局完成证据 |

## 什么时候用视觉算法

| 需要的结果 | 输入及计算 |
|---|---|
| 可见性、语言关系、是否夹持、定性放置结果 | Decisions + 当前三相机 RGB；通常不需要 SAM3/深度/新几何 |
| 当前实例位置与碰撞几何 | SAM3 掩码、Decisions 身份关联、RGBD/标定 |
| 已知 CAD 当前姿态 | Decisions 选目录模型后，FoundationPose 使用该 CAD、当前掩码和 RGBD；选有效深度最多的相机，再转世界坐标 |
| 未知物体几何 | 当前掩码对应的实测点拟合 OBB，绑定永久 ID |
| 桌面与碰撞场景 | 实测 cuboid + 当前 OBB + 已配准 CAD；不从仿真真值补全 |

实例身份、CAD 选择、支撑关系、可见性分别提问。CAD asset ID 可被多个永久实例 ID 共享。已确认 CAD 对应可以缓存，新观测仍更新姿态；局部更新保留其他实体的历史证据与失效状态。

## 放置与核验

普通目标区域从四分开始，按 4→8→16 子区域逐层提出 XY；Decisions 依据原始语言选择候选 ID，数值算法再求接近碰撞阈值的 Z。工作台“指定/设定区域”直接使用配置 XY。默认保持抓取姿态；工作空间包络、携物/夹爪碰撞、IK、轨迹检查分别执行。中间工作区不能冒充最终目的地。

verify_place 由 graph 显式调用，一次 Decisions 同时判断支撑/入容器关系、是否仍夹持，以及适用时接近目标子区域。接受位置误差，不调用 SAM3、FoundationPose、OBB 或释放后深度测量。动作反馈、释放位置、采集时序及身份绑定仍校验。确认后的地图预测更新不伪称实测。

当前三相机图片可放在 question.instructions 内；公共 input 图片数不是总图片数，日志与归档统计两处。无历史参考图时，问题使用绑定的外观描述，不声称提供了 REFERENCE 图片。

verify_place 不因最后一个物体追加全局清单判断。全局完成由 graph 中独立 verify_task_completion 调用完成。每件交付后双臂回配置 observation 位姿并观测、最终也在该位姿，是用户要求，由 Responses 显式编排。

## 失败与证据

看不到、看不完整、遮挡与明确否定分开判断。缺视角证据时由 graph/Decisions 选择合法补充观测位姿；相同输入不自动原位重拍。no/unknown 不重试到 yes，也不降低事实门槛。临时网络/服务失败可有限重发同一冻结请求，但不能重放机器人动作。

运行时恢复不得覆盖数值碰撞失败、修改任务目标或重播已完成物理效果。回放读取真实模型 graph 与事件，不从技能顺序反推编排。旧 steps 归档只保留原始 JSON，不提供兼容解释器。

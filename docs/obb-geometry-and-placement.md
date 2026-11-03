# OBB、GraspGen 与放置选位

本轮按用户批准方案实现。OBB 是实测物体的几何近似，带永久 ID、原始采集来源和测量哈希；原始 RGBD 点云、场景地图、派生盒体和模型输入分别记录。

## 同一个物体几何贯穿抓放

### 跨观测、全局与局部的实际边界（2026-10-30 诊断补充）

永久 ID 表示物体身份，不表示几何恒定。当前 OBB 由**同一次冻结观测**中已确认属于该 ID 的多相机点融合拟合；新观测重新拟合，没有跨观测点云融合、尺寸锁定或方向防抖。遮挡和视角变化可能改变中心、尺寸和 XY 朝向，对称物体也可能出现等价轴方向变化。全部保留点极值只包围已测点，不能证明包围完整真实外形；哈希检查只能阻止混用新点与旧框，不能解决隐藏表面不足。

全局/局部是碰撞地图的更新范围。全局更新本次已观测清单实体，并保留未再看到的历史层；局部仅刷新模型选中的实体，其他层保留原来源。TSDF/ESDF 对静止物体保留历史占用证据，但目前抓取 OBB 仍来自当前观测，不是从累计 ESDF 或历史点云拟合。不能将两种生命周期混为一谈。

闭爪后的 `obb_tool` 固定在实际夹爪前端坐标系；后续 `held_collision_binding` 只更新场景绑定，保留此箱体形状及测量来源。实际夹持无滑移是假设。首次 Lift 后才 attachment 的约定不变。释放/加工使旧位姿失效，新观测重新建立对应 ID 的几何；产物采用其独立永久 ID。

### 网页点云诊断

`/api/perception-preview` 在已有只读建图快照中增加 `object_obbs.observed` 和 `object_obbs.mapped`。原始点云模式显示本次身份关联观测的 OBB；碰撞模式显示按地图实体更新范围保留的最近关联 OBB。这些 OBB 不是累计 TSDF 的包围盒，也不作为环境碰撞 cuboid。它们不包含当前实时持物 TCP 变换，旧位置失效后不会画成实时 attachment。

绿实线为本次观测，紫虚线为历史保留，红虚线为动作后已失效的旧位置。框旁标永久 ID，下方列尺寸（mm）、状态和原始观测 ID；按钮可隐藏 OBB，展开来源可看完整位姿、点云哈希和测量来源。显示直接使用后端实际 OBB，不对最多 30,000 个预览点重新拟合。读取/切换显示不采集、不建图、不修改任务或规划参数。旧快照没有 OBB 时明确提示等待下一次观察建图，不从仿真真值生成框。

尚未实现稳定的跨观测形状融合。要进一步解决视角导致的尺寸收缩，需要在永久 ID 下区分形状证据与位姿证据，验证静止/刚性对应后才能融合历史实测点；不能无条件累计移动前后的世界坐标点，也不能仅靠平滑框尺寸宣称完整物体已被覆盖。

`perception._estimate_object` 在身份确认和标定深度融合后生成 `obb`。采用 GaP 的直立最小宽度方向思路：XY 边界方向确定旋转、Z 保持竖直。本项目直接用凸包边作为方向候选，不引入随机噪声、栅格闭运算及额外 OpenCV/Open3D 依赖。碰撞包络使用全部保留点的极值，不复制 GaP 的 2%/98% 裁剪；退化厚度的最小半尺寸为 0.5mm。这是有明确差异的适配，不声称逐行移植或完整重建隐藏外形。

`generate_grasps` 将 OBB 转成 trimesh 盒体，再按官方 GraspGen mesh 示例采样 4096 个表面点。采样种子固定以保持同一测量的重试输入一致；模型仍按原预算生成抓姿。模型输入保持原有实测点云中心坐标系，GraspGen 到 gripper_front 的校准只转换一次。回放分别保存 `object_obb`、真实提交数组与原始测量；生成的表面点明确标为派生数据。

闭爪后记录实际 `T_world_tcp`，将 OBB 转为 `obb_tool`；此时只是临时持物参考。首次 lift 完成并再次确认持物后，才准备 attachment。按 OBB 的最长单元轴分割体积，给每个单元构造覆盖球，默认最多 24 球、硬上限 31，半径加入现有测量不确定性和 margin。覆盖整个盒体与内部，不只覆盖八个顶点。旧归档若没有 OBB，仍可读取原有实测表面球拟合；新抓取绑定默认带 OBB。

原始点云仍用于实测接触检查、身份、建图和结果测量。OBB 不发布为场景障碍，不填实容器内腔，不使用 CAD 或仿真对象真值。OBB 表面不保证等于物理接触面；凹形、带孔和明显非盒状物体的抓取效果需真实服务/物理验收。

## 放置位置

`perceive_placement_zone` 保留已实现的两类区域：实测容器内部，或桌面、托盘、其他物体的水平支撑面。无实体边界的相对区域由支撑面候选与原始语言共同限定。

`resolve_placement` 的固定数值过程：

1. 用抓取时的完整旋转和 `obb_tool` 的八个角点计算占地；在实测支撑范围内采样 XY。明确指定的 XY 不漂移。
2. 从既有采样保留最多 96 个空间分散候选。原来的 XY 障碍包络仅作排序/来源诊断，不拒绝三维可行空间。
3. 根据夹爪球与持物球底部、支撑高度和配置释放间隙计算 TCP 解析最低 Z。使用既有 WorkspaceAtlas 检查该 XY 的 21 个 TCP 高度样本；全部不在包络内才提前排除。
4. 使用已有 `/tools/resolve_release_height` 和当前已发布场景，批量查询夹爪球与 attachment 球。固定 XY/旋转，1mm 步长、最多上调 20mm，选择第一个满足原硬净距的高度。当前与保留的 TSDF/ESDF 层都参加同一次世界碰撞查询。
5. 再检查选定 TCP XYZ 的离线工作空间，以及目标能力的已有配置约束。物体几何底部到目标支撑面的距离不得超过 `place_max_release_gap_m`，默认 50mm、界面可配置。容器还检查进入边沿以下及三段进出高度。这样不会无限抬高 Z 来绕过占用。
6. 保留最多 12 个空间分散且通过数值检查的候选，交给外层语义判断。最终 IK、双臂/自碰撞和完整 approach/place/retreat 由 `plan_place` 检查；开爪发生在 place 和 retreat 之间。

搜索是有界离散采样，不保证发现所有有效位置。工作空间是工具朝向自由的末端位置包络，不证明固定旋转或路径可行。最低无碰撞样本若不在工作空间内，该 XY 仍可被拒绝；本实现不宣称求得该 XY 下所有可达高度。

## Decisions 无需自然语言输出

初始 Responses 保留原始指令、永久 ID 和 graph；运行中不再调用 Responses。区域选择/绑定使用已有 ID 参数和类型化选项。`perceive_placement_zone` 的外层 Decisions 判断是否许可整区选位；若原话带相对位置、具名子区或视角限制，则对编号候选进行一次批量 predicate 判断。

模型输入为原始指令、当前 RGB、永久物体 ID、冻结方向参考、测量关系和候选占地投影。模型只返回各候选是否符合全部语言要求的概率；程序按阈值保留 ID，再查出对应真实位姿。未知、无匹配或图像过期均不能改选任意位置。代码硬约束失败不能被模型肯定回答覆盖。释放后仍以当前观测验证原句，候选许可不是执行成功证明。

## 官方接口依据与验证边界

- GaP `open-robot-skills@752dfbd96968d4d08379f8111765efbf6457800b`：[compute_obb](https://github.com/graph-robots/open-robot-skills/blob/752dfbd96968d4d08379f8111765efbf6457800b/tools/geometry/_impl.py#L775)。本项目采用其最小宽度方向思路，差异见上文。
- [GraspGen mesh 示例](https://github.com/NVlabs/GraspGen/blob/main/scripts/demo_object_mesh.py)：本轮读取文件 blob `4e376a68397c7a84d576ead672f0bf930bbb34a8`，使用公开 `trimesh.sample.sample_surface`。
- cuRobo 锁定 `8e734f3ced1df898990bcd92de40abce475907db`：[公开 collision_checking 模块](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/collision_checking.py)，`RobotCollisionChecker.get_collision_distance` 接受 `[batch,horizon,num_spheres,4]`。沿用已配置硬净距的公开查询，不改 SDK/vendor。
- [Decisions 协议](https://developers.openai.com/api/docs/guides/decisions)：文本/图像输入，predicate 概率或预给 choice 值输出；本轮不增加自由文本解码器。

实施和实际验证命令见 `docs/superpowers/plans/2026-10-30-obb-placement.md` 及项目 `data/obb-placement-20261030/`。单元/替身测试不能代替 GPU 服务和完整物理任务验收。

2026-10-30 本地最终验证：Python 全量 3595 passed、58 skipped；前端 Node 测试 380 passed；本轮修改的 Python 文件 Ruff 检查通过。独立代码复核未发现实质性缺陷。此前全量测试发现的 15 项失败均来自旧测试契约，更新测试替身与几何断言后，保留原可达性、坐标和实测点覆盖检查，并完成上述全量复跑。

首次本地验收时，默认 SSH 登录被拒绝，本地未配置 Decisions 凭据，记录见 `data/obb-placement-20261030/service-access.json`。后续用户授权同步后，找到本机现有 SSH 连接使用的身份并连接成功，已同步到 `/home/guangyu/projects/tianji-luna-mujoco`。远端缺少的前一阶段统一放置配套代码一并同步；170 个文件更新、4 个 JEV 规范文件删除，原文件均先备份。Ubuntu 相关回归 119 项通过；网页和 cuRobo 重启后健康检查通过，远端 Decisions 凭据已配置。当前物体布局、颜色、桌面、既有规划设置和双臂位置已核对保留。证据、逐文件哈希及回退包位于两端 `data/obb-sync-20261030/`。

同步后的健康检查和回归仍不代表完整抓放成功；尚未执行新版真实 GPU 批量选位、真实 GraspGen 抓取或整轮物理任务，也未做 GaP mesh 与本项目 TSDF/ESDF 的同条件性能对比。

用户要求移除的 JEV 开发 skill 已从 Codex 安装目录和本仓库规范目录删除；历史实验记录保留，不再把旧 skill 作为当前规则。

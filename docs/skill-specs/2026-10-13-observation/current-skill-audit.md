# 现有 JEV skills 的规范核对

日期：2026-10-13。核对依据是本地 registry、contract、SKILL.md 与执行代码；此次不修改产品代码，不把静态审查当作运行验收。

`src/jev_mujoco/skills.py` 只加载 `skills/*/contract.json` 对应技能，实际共 **19** 项。`close_gripper`、`execute_approach`、`execute_lift` 目录没有 contract/SKILL.md，不是三项额外可调用技能。

## 逐项核对

“需补齐”表示现有文档或行为尚未完全达到新规范，不表示原有算法一定不可用。每项最终都需补齐规范中的输入版本、失败/未知、状态影响、预算和验证依据；下表列出其具体关注点。

| 当前 skill | 已有明确职责/实际行为 | 对齐项 |
|---|---|---|
| assess_task | 执行 JEV 对集合任务进度的判断 | 文档未完整定义判断输入、状态变更、unknown 和终止语义；不能从当前不可见推出任务完成 |
| identify_target | 校验并绑定 JEV 选择的实测源实例 | 补齐历史身份/当前几何区别、重试目标限制、选中对象后的证据更新；当前 views_ready 不是逐选中目标重算 |
| identify_destination | 校验并绑定 bin 或 marker | 文档仍主要写“盒子”，与代码支持 marker 不一致；补齐参考版本和失败码 |
| observe | 采集或内部复用 capture，分割与关联，按 checkpoint 更新状态 | 同一入口承担场景、目标与结果职责；采集与分析计数混用；需按本目录定义拆清 |
| move_to_observe | 空手固定双臂位姿；内部头部建图、规划、移动、fine 分析 | 隐式建图、相机数 readiness、到位分析和运动成功混合；尚无自动位姿生成 |
| build_scene | JEV 选择 observation/selected/held；建立 TSDF/ESDF，清旧计划 | 观测运动需要的首次建图仍受 target_selected/views_ready 限制；需消除先看全再建图的循环，明确未知空间限制 |
| generate_grasps | 基于实测目标点云生成有限 GraspGen 候选 | 入口仍受双腕 views_ready 门槛阻挡；“三相机融合”不能暗示每个相机都要看到目标 |
| evaluate_arms | 抓姿批量 IK；指定臂时仅评估该臂 | 文档仍笼统写 A/B 全评估；补齐 grasp_ik_only、候选无解和服务错误区别 |
| choose_arm | 校验并绑定外层选择；指定臂不再请求 JEV 选臂 | 补齐用户指定臂直通路径、不允许偷偷换臂、候选及场景版本 |
| plan_grasp | 固定有限批次规划抓取三段 | 补齐已声明 lift 碰撞例外与实际限制、无解的候选范围、超时迟到结果处理 |
| plan_grasp_and_place | 为抓放任务规划抓取段，后续另做 place 规划 | 名称易被理解成已经生成整套抓放轨迹；必须保留真实输出范围及前提 |
| execute_grasp | 固定开爪/接近/抓取/提升及末次结果采集 | 将结果 capture 的复用、部分执行失败状态及“完成运动不等于抓到”纳入契约 |
| verify_grasp | 夹爪为主持有证据，融合已有辅助结果，held 时准备 attachment | 保持 unknown/empty 区别、证据消费及同帧冲突；补齐 attachment 近似和失败后的状态 |
| plan_place | 实测区域候选与偏航按有限 batch 求解，检查完整三段 | contract/SKILL.md 仍写固定 XYZ 八偏航；实现已可遍历多个区域位置，文档需要校正 |
| execute_place | 持物运动、确认开爪、detach、撤离与结果采集 | 不能把 released 当作 placed；补齐捕获和释放/撤离部分失败状态 |
| verify_place | 对同一结果帧按有限相机顺序分析并核验 | 内部调用 observe(capture=...) 不代表新拍照，但目前会增加 observation_count；需独立记分析预算 |
| inspect_scene | 一次图像请求解释既有实例语义，缓存 | 补齐固定流程、可见像素前提、实例引用、预算和状态；不是新观测或新几何 |
| inspect_visual_result | 复用动作后帧提供图像结果辅助 | 补齐 action/phase 绑定与“之后仍需 verify”；同帧结果不算第二份独立证据 |
| recover_task | 执行外层 JEV 明确给出的恢复 action | 需逐 action 写清固定分支与副作用；reobserve 当前会调用 observe；预算只拒绝不替 JEV 选恢复 |

## 已确认的具体差异

1. `observation_motion.source_view_evidence` 对普通桌面目标要求相机 1 和 2；盒内仅有一个同帧几何例外。相机数代替动作证据不足判断。
2. 集合任务初始 observe 对全部非容器源对象做 `all(views_ready)`，不是仅对当前任务选中目标判断。
3. `move_to_observe` 调用 `build_scene(... head_only=True)`；`algorithms.build_scene` 将两个腕部输入深度置零。其初始运动地图不是三路深度融合。
4. 该移动 skill 的契约要求 `views_ready=false`。双腕都有掩码时设为 true，反而无法表达“仍需补看关键接近区域”。
5. `ObservedFreeSpaceGuard` 已构造但未用于实际逐轨迹门控，JEV 状态明确 `unobserved_space_collision_checked=false`。当前规划通过不证明盲区空闲。
6. `verify_place` 分相机分析同一 post_place capture，复用像素但每次增加 observation_count。采集、分析、视角移动、VLM 成本应分开。

## 可复现的基线探针

执行了一个不调用硬件、SAM3、GPU 或模型的输入探针：

```python
from jev_mujoco.observation_motion import source_view_evidence

evidence = source_view_evidence(
    {"source": "rgbd", "observation_id": "post-red-place"},
    {"id": "green", "coverage": {"camera_ids": ["1"]}},
)
```

实际返回：

```json
{"source_id":"green","camera_ids":["1"],"views_ready":false,"policy":"both_wrists"}
```

检查“不因单腕来源直接拦住后续几何评估”的断言按预期失败。该探针只证明存在相机编号门槛，不证明这个人为最小输入足够生成真实抓姿。未为使其通过而修改代码。

## 本轮交付状态

已编写开发规范和六份目的明确的观测/图像分析定义，并接入项目开发说明。仍有上述契约、代码和验证差异；**当前 19 项技能尚不能声明已全部符合新规范**。只有将未来获准的行为逐项实现、验证并部署，才能改变该状态。

## 源码定位

- `src/jev_mujoco/skills.py`：真实 registry 与契约拒绝。
- `src/jev_mujoco/observation_motion.py`：初始头部要求和源目标覆盖门槛。
- `src/jev_mujoco/runner.py`：observe、capture_execution_result、move_to_observe、verify_place、recover_task、select_arguments、execute。
- `src/jev_mujoco/algorithms.py`：build_scene 的 head_only 分支、指定臂 IK、place 候选批次。
- `src/jev_mujoco/geometry_service.py`：观测空间 guard 的构造及当前使用范围。

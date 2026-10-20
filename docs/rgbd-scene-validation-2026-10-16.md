# RGBD 身份关联与放后建图核查

2026-10-16。前半部分记录实现版本 `cd72ee7`；末节记录最新固定全场景清单版本 `de1b44d`。候选目录为 Windows `C:/project/tianji-jev-mujoco-requirements` 和 Ubuntu `/home/guangyu/projects/tianji-jev-mujoco-requirements`。原项目与生产服务未切换。完整多物体物理验收尚未通过。

## 按用户纠正收敛的行为

跨时关联比较实测 RGB 色度和注册到世界坐标的深度表面；头部掩码 IoU 只作诊断。取消唯一颜色直接关联、掩码优先和 IoU 竞争分支。普通形状标签改变不另建物体身份；刚释放物体依据当前目的区域测量和唯一候选关联，保留原 ID。分割、跨相机合并和跨时 ID 关联仍各自承担原有职责。

跨相机盒壁局部视图按共享 RGBD 表面归到盒子；类别别名不再要求两个视角具有近乎完整的共同轮廓。保留区分盒子与盒内独立物体的回归测试。未增加跟踪模型、SDK 修改或自动恢复分支。

| 时机 | 实际建图相机 | 物体排除 | 桌面 |
|---|---|---|---|
| 尚无场景 | 头部 3 | 无 | 首次生成 |
| 放置后空爪撤离完成，仍有待搬阶段 | 原位 1/2/3 | 无，刚放物体恢复为障碍 | 复用 |
| 双臂到观测位姿 | 由上一张全物体障碍图约束运动 | 运动前无目标排除 | 复用 |
| 到达观测位姿后的精细更新 | 1/2/3 | 仅排除本次抓取目标 | 复用 |

当前帧测得的盒底点要在冻结区域中检查。新视角选择了另一块内接矩形，不能据此裁掉原指定分区附近仍然可见的底面点。此修复只改变裁剪范围；没有移动分区或放宽底面、占用、边界、碰撞阈值。

## 已完成的验证

- Ubuntu 候选容器完整 `python -m pytest -q`：**1725 passed, 10 skipped，80.08 s**；跳过项不计为通过。
- 放置区域、固定 slot、批次、世界 yaw、15 mm 释放定向回归：57 passed。`ruff check src tests/test_object_slot_binding.py` 和 `git diff --check` 通过。
- 真实录制 N=1、N=3 帧的跨时关联：8/8 唯一匹配，单次关联约 32–61 ms；仅为该小样本的关联耗时，不包含 SAM3、采集或建图。
- 真实 GPU 录制 RGBD 建图：初始头部、三相机全实例、三相机选中实例均成功，桌面版本保持不变，排除清单正确。独立双臂观测 GPU 测试通过，含障碍拒绝。
- 标准 N=1 中转场景 trial3、trial4，以及加长盒底的 roomy trial1：实际完成第一段抓放、放后原位三相机建图、双臂返回、第二次三相机建图及再次抓取。逐事件核对的 8 项相机、地图、ID 和桌面生命周期断言全部通过；三次均未完成最终放置。roomy trial1 正常 capture 恰好 4/4、恢复 capture 0/8。

## 全部试跑结果及边界

结果目录位于 Ubuntu `/home/guangyu/projects/tianji-jev-acceptance-20261016`。

| 试跑目录 | 结果 |
|---|---|
| `box-table-n3-rgbd-revision-trial1` | 完成清单、观测运动及精细地图；首件 200 个抓姿均未通过指定臂 IK，重试仍无解，之后 JEV 输入预算超限。无抓放执行。 |
| `box-table-box-n1-rgbd-revision-trial1` | 第一段抓放完成；放后帧将盒壁重复当作普通物体，跨相机关联拒绝。录制帧复现后修复共同表面匹配。 |
| `box-table-box-n1-rgbd-revision-trial2` | **无效试跑**：并行回归测试清除了同一候选 GPU 场景。责任在验证编排；后续测试和真实试跑串行，不为此增加产品恢复逻辑。 |
| `box-table-box-n1-rgbd-revision-trial3` | 核心相机/场景/ID 时序通过；第二段误报 slot 无底面支持。实际最近底面点仅距中心 0.138 mm，原因是按新的内接矩形错误裁剪，已由 `cd72ee7` 修复。 |
| `box-table-box-n1-rgbd-revision-trial4` | 修复后的底面检查通过；第二段 18 个 yaw 均超出冻结区域边界。离线复算最佳 yaw 超出约 2.4 mm，这是旧代码额外添加的attachment投影限制，不能解释为用户要求下的IK无解；本轮已删除该限制。 |
| `box-table-box-n1-roomy-rgbd-trial1` | 独立验收夹具将盒底 X 长度从 24 cm 改为 28 cm，其余尺寸、位置、模型和规划参数不变。第二段 8 个 yaw 通过区域筛选，但指定 A 臂的碰撞约束 IK 全部拒绝，`place_no_valid_candidate`；独立物理评分为 incomplete_route。该结果不替代原场景的失败记录。 |
| `box-table-box-n1-roomy-ab-rgbd-trial1` | 同样加长盒底，指令显式指定第一段左臂、第二段右臂。JEV 的全局 arm 字段返回 unsupported，绑定时报 low_confidence:arm；尚未执行机器人动作。测试入口因账本尚未创建又在导出时抛出 AttributeError，原始 result.json 已保存。 |

全路线 N=1/3/4、多次连续成功及双臂组合验收仍未完成。N=3 的抓取可行性和 JEV 输入预算、第二段放置可行性、显式分阶段执行臂绑定仍待解决。不能把上述相机/地图时序通过写成完整路线通过。

可选开源组件：[Open3D Colored ICP](https://www.open3d.org/docs/latest/tutorial/pipelines/colored_pointcloud_registration.html) 联合颜色与几何进行配准，但物体 ID 生命周期仍需项目维护；[FoundationPose](https://github.com/NVlabs/FoundationPose) 提供 6D 位姿估计与跟踪，需 CAD 或少量参考图像。当前已标定的简单静态场景先保留现有 RGBD 关联，未引入这些新流程。

## 固定全场景清单修订与验证

用户要求初始头部找到所有实体后不再探索。根因是旧 Perception 在每次观测无条件追加 cube/sphere/cylinder/object 和 cube→block 查询，而 TaskProgress 只冻结搬运成员，scene_entities 仍可新增。修改沿用现有模块：首次头部观测固定全场景 ID、类别与分割描述；搬运名单引用这些 ID。实体角色明确为 transport/collision、collision、container/collision、table/collision、target_region。桌面不做 SAM3，目标区域沿用 region:支撑实体ID。

后续三相机仅执行固定 scene_queries，相同描述合并查询并保留全部候选实体 ID；别名/片段合并也保留候选并集。连续 RGB 特征、注册深度、动作目的地和一对一约束共同确认身份。固定清单后不再使用初始离散颜色过滤/环境发现检查，未匹配检测不得新增实体。恢复重新观测前先撤销旧地图就绪状态。无新依赖、跟踪模型、阈值放宽或 SDK/vendor 改动。

`de1b44d` 的 Ubuntu 完整回归：**1732 passed, 10 skipped，79.21 s**，日志 `fixed-scene-obstacles-pytest.log`。定向回归59通过；ruff 与 diff 检查通过。新增回归先观察到失败再修复：固定查询、角色与桌面ID、禁增实体、查询候选与RGBD、同义词候选保留、灰白标签边界和声明障碍检查。独立复核24项通过。

真实 SAM3 重算 `box-table-box-n1-rgbd-revision-trial1` 的1张精细观测与3张放后录制帧，未读取物理真值：旧放后观测中的两个蓝色 cube 消失；每帧为两只蓝盒、一个红色搬运物体、一个黄色桌面标记，**16/16 全局 ID 关联，0 unresolved**。初始保存文件未包含成功提示字段，因此该历史重算使用初始 kind/color 对应的 `blue open box` 与 `red cube`；每帧三相机共6次调用。重算耗时2.52–2.72 s，包括SAM3和感知但不含采集、建图或规划，不能作为完整流程耗时。两只盒子均取得约0.610 m的实测水平底面，无需新增RANSAC。证据为 `fixed-scene-replay.json` 和 `fixed-scene-replay.log`。

实际新试跑 `box-table-box-n1-fixed-scene-trial1` 使用 `17446dd`（随后 `de1b44d` 仅限定初始环境颜色检查的作用范围；本夹具没有该环境查询）。初始固定5个实体，成功分割描述为 `blue open box` 与 `red block`，实际后续请求全部来自该清单。全场景 ID 集合保持不变；两盒初始底面分别为0.6099988 m、0.6099985 m。初始成员评估一次unknown后，JEV选择重新观测；尚无场景时仍只用头部且不新增发现查询。已有场景后的三次更新均为三相机、每次6个SAM请求。

新试跑完成第一段抓放及撤离、放后原位全障碍图、双臂返回观测位姿、第二次三相机精建图和同一ID再次抓取。`scene-sequence-score.json` 的8项相机/障碍/ID/桌面检查全部通过，`fixed-scene-score.json` 的清单、查询、角色与盒底检查通过。最终第二段18个yaw被旧代码额外施加的固定区域外边界投影限制拒绝，`placement_region_no_supported_candidate`；独立物理评分为 `incomplete_route`。这次拒绝发生在CPU候选筛选，尚未进入cuRobo；该额外投影限制不符合用户要求，本轮删除。分区位置和盒子尺寸不变。因此本轮证明固定身份/查询及建图接入正确，不证明全路线搬运成功。

以上证据均位于 Ubuntu `/home/guangyu/projects/tianji-jev-acceptance-20261016`。完整pytest与实际GPU试跑串行执行；未新建镜像，继续使用已有候选容器，生产目录和服务未切换。

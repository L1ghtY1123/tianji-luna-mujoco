# observe_scene 的无进展依赖

2026-10-30，black-B5（`95c7167e1f5649bd8c95380ffa61ba45`）释放执行成功后，`verify_place` 因测量不足返回 unknown。后续 observe 已恢复黑件的验证 CAD，但目的地右盒仍缺少当前身份几何。每次 continuation 都提供 `retry_failed_node`，同时明确区分 `historical_failed_result` 与当前 `missing_geometry_ids`；没有证据表明必要重试选项被排除，不能推断模型内部选择原因。

原 guard 已在 event 604 屏蔽 fresh，在 event 630/633 同时屏蔽 fresh 与 reuse_current。event 647 建图后，相同采集仅因 `scene_available`、`observation_map_ready`、`arms_evaluated`、`plan_ready`、`views_ready`、`grasps_ready` 改变，就在 event 651/654 重新开放了相同 observe。这些规划状态不改变这次观测的输入。

现在仅对 observe_scene（包括失败 observe 节点的 retry）使用观测专属签名。保留原 RGB `[::8, ::8] // 16` 摘要、测量实体集合、完成输入、失败节点及原因；另外绑定采集 generation/manifest/command 版本、采集所属动作 epoch、相机标定/内参/外参、初始身份参考、任务和实体身份描述、退休/转化谱系、释放/持物相关事实、缺失几何集合和观测设置。普通建图与规划 facts、地图 publication/coverage 不重新开放同输入观察。当前照片过期时仍按原逻辑先获取当前输入，派发前再次核对签名。

模型仍决定选择哪个合法能力、retry、stop 或 unknown；代码不插入移动或强制复核，不把 observe 成功视为所有目标/目的地已确认。其他技能保留原通用 facts/map 失效边界；plan_place 的独立数值输入守卫不变。

这个 guard 仍不比较完整 RGBD，也不判断深度等值。B5 fresh 帧的 RGB 仅有少量像素变化，原摘要相同，但深度最大差异达到约 71.23 µm，不能称为相同 RGBD。此次未添加深度阈值或放宽任何物理、视觉成功门槛。实际新视角、标定、RGB 摘要、动作或身份需求变化仍开放观察；纯 depth 变化并非本次新增识别能力。

`tests/fixtures/black_b5_observe_recovery.json` 保留建图前后的真实 facts 与身份/相机元数据。回归先复现 fresh/reuse 被错误开放，再验证修复和相关输入变化。`data/cad-obb-iteration-20261030/black-B5-resolution-audit/observe-guard-replay.json` 使用真实归档 RGB 与 context 重放，确认 event 630 和 651 均屏蔽两个已尝试模式，verify_place 的 retry 仍可选。归档未记录内部 action epoch，该次 CPU 回放同等省略；真实运行版本由 controller 维护，独立回归覆盖其变化。CPU 回放不代表新真实模型或完整机器人任务通过。

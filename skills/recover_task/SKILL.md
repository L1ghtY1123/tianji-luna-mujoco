# recover_task

**目的与调用时机：**在已有失败证据时执行 JEV 在 skill 外选定的一项有限恢复操作。

**输入与前提：**recovery_needed=true；输入 action、失败阶段/尝试记录、当前反馈及持物/地图版本。

**固定算法：**reobserve：原位新采同步帧并按目标分析，旧地图/规划前提失效；retry_grasp：确认空手并解除旧 attachment，保留原目标要求、下一轮仍经完整 build_scene；retry_place：确认仍持有，仅 plan_place 失败且保存的抓前场景仍有效时复用；否则返回 retry_place_requires_saved_grasp_scene，保留原失败，不观察、不移动、不建图；retry_plan：仅允许保留有效证据、固定同一阶段臂的受限重新规划；defer：确认空手后暂缓当前目标；stop：按完成与未知证据报告终态。

**输出与状态：**action、实际状态变化与待补前提，或拒绝原因；不自动规划/执行下一 skill。

**失败含义与边界：**持物不明不能 defer/retry_grasp；路线任务不能用 retry_plan 换臂。每目标最多 3 次计数失败，unknown/未执行契约拒绝不耗该预算；每周期最多 8 个不同分析 capture。预算耗尽是拒绝条件，不自动替 JEV 决定结束。

## 恢复预算

每目标最多 3 次计数失败尝试；明确 unknown 或 executed=false 的契约拒绝不计物理失败。连续同原因、原始失败及关联歧义仍保留供 JEV 判断。连续 8 次技能选择契约拒绝后 needs_attention。reobserve 是此技能的明确参数分支，不是重新开放已删除的 observe_result/observe_geometry/move_to_observe。当前阶段执行臂固定；受限 retry_plan 只重试本阶段臂，不通过恢复清除该约束。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。

预算按完整冻结清单N与阶段数S计算G=N*S：正常RGBD采集2G+1次（初始头部1、各轮抓前三相机G、放后三相机G，含最后一件），恢复采集独立8次。Decisions随外层JEV选定的观测及用途执行；独立语义辅助的8次额度不适用于Decisions。重复分析同一capture不重复计数；切换物体保留所有计数，重试新增采集计入恢复额度。JEV决策上限max(400,40G+80)，耗尽保持needs_attention。

**外层视觉参数：**`Runner.select_arguments` 必须先调用 `Decisions.choose_visual_plan`，由外层JEV在skill_start前为每个尚未分析的checkpoint选择initialize_inventory、associate_inventory或verify_transfer。visual_plan是有限映射；build_scene最多列出原位补图与抓前观测两个checkpoint。每项只消费一次；缺少、skip或不匹配的用途均拒绝。Perception再次检查source=jev和用途；缺少Decisions结果明确失败，不启用距离、颜色、掩码或动作预测身份回退。同帧已完成分析及已确认释放后的地图重试复用缓存，不增加图片请求。

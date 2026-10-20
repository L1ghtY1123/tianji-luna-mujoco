# identify_target

**目的与调用时机：**绑定 JEV 选中的一个实测物体 ID。

**输入与前提：**已观察、task_assessed=true、尚未选目标/完成抓取且无恢复；输入 target_id 与当前 observation。

**固定算法：**检查 ID 唯一且不是 bin/marker，核对任务成员、requested_count、inventory 编号顺序及 retry 原目标约束；记录目标可见点与几何缺口，目标改变使相关地图/轨迹前提失效。

**输出与状态：**target_selected、target_id、当前目标 view_evidence；不采图或运动。

**失败含义与边界：**missing/ambiguous、成员未确认、编号或数量约束冲突时拒绝；不能用近似相同对象替换指定历史 ID。头部发现不直接将 views_ready 设为真。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。

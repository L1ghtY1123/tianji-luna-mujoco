# evaluate_reachability

**目的与调用时机：**在未绑定执行臂时，为 JEV 提供源中心和目的中心的单臂可达证据。

**输入与前提：**target_selected、destination_ready、center_reachability_enabled=true，尚未评估/选臂/抓取且无恢复。单抓取只查源中心。

**固定算法：**对 A/B 各查询离线工作空间；有指定/已绑定约束时遵守约束。查表保留本臂、夹爪、腕相机、中央结构和头部相机，忽略另一臂和环境；不请求在线 GPU IK。

**输出与状态：**arms_evaluated、每臂 source/destination/reachable，scope=offline_position_workspace，collision_checked=false。

**失败含义与边界：**模型/标定不匹配拒绝旧表；无命中只表示离线采样未支持，不证明严格不可达。中心可达不证明抓姿、整条路径或碰撞可行。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。

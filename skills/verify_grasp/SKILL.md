# verify_grasp

**目的与调用时机：**在提升后独立判定夹爪 held/empty/unknown。

**输入与前提：**lifted=true、verified=false；输入当前 generation 的 robot_snapshot 和冻结持物判定设置。

**固定算法：**仅读取夹爪命令、位置样本与行程。默认要求完整闭爪命令和至少 0.2 s 稳定窗口，位置波动不超过行程 2%；稳定位置距空夹闭合端大于该容差为 held，否则 empty。记录 feedback_id 与时间。held 时保存原抓前点云、闭合 TCP、观测和地图为已验证持物记录。

**输出与状态：**verified、grasp_held 或明确 empty/unknown；grasp 任务在 held 后可结束。反馈 ID 不是 observation_id。

**失败含义与边界：**缺样本、未全闭命令或不稳定返回 unknown，不能当空抓失败。empty 才是不持物证据。本技能不采 RGBD、不调用 SAM3/OpenAI、不创建 attachment；位置证据不单独证明物体身份正确或没有滑移。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。

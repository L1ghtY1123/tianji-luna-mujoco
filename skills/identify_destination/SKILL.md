# identify_destination

**目的与调用时机：**绑定当前阶段实测盒子或桌面 marker。

**输入与前提：**需要放置，已有观测且尚未选择目的地；输入 destination_id。搬运路线首次调用还需外层 JEV 给出冻结清单中每个对象、每个阶段的 route_destinations。standalone place 使用已验证持物保存的观测。

**固定算法：**检查实测 ID 和 bin/marker 类别；盒→桌面及中转第一段要求 marker，桌面→盒及中转第二段要求 bin；中转最终盒不能等于原来源盒。首次运动前校验全部对象的全路线目的区域有实测底面和可划分边界；首次 plan_place 使用三相机几何划分并冻结该区域布局；后续阶段选择必须符合该对象已确认的路线引用。任何最终盒缺失、区域不可测或源/终盒相同都在运动前失败。此处只校验外层 JEV 参数，不选择后续 skill。

**输出与状态：**destination_selected、destination_ready 及 destination_id；不移动、不产生新深度。

**失败含义与边界：**没有当前已保存的实测区域、ID 不唯一、类别/路线冲突时拒绝。选择本身不证明底面、区域边界或放置路径足够。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。

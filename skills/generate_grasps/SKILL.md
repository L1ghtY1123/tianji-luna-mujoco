# generate_grasps

**目的与调用时机：**使用本轮已分析的目标可见点云生成 GraspGen 候选。

**输入与前提：**目标和阶段臂已绑定，views_ready 与 scene_ready 均为真，尚无候选且未完成抓取。

**固定算法：**检查目标和帧绑定，使用同步头部与双腕相机经 SAM3 掩码及标定变换得到的目标可见点融合，再变换至算法坐标系；不要求三路同时看见目标，不补隐藏表面。按当前两批预算生成至多 200 个候选，保留原始编号、矩阵和 score，按 score 稳定排序。项目适配层按已绑定臂转换为 T_world_tcp，再送 IK；A/B 的 Z 轴相同，X/Y 同时反向，旋转保持右手系，并应用195 mm抓取参考和185 mm安装TCP工具偏移。候选携带 arm 与 tool_profile_revision；IK拒绝臂或工具版本不匹配，不重复转换。

**输出与状态：**grasps、grasps_ready、候选来源 observation_id；不移动、不建立 attachment。

**失败含义与边界：**点云/目标/帧不匹配、服务错误或无候选返回失败；score 不是成功概率。候选存在不代表 IK 或完整轨迹可行。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。

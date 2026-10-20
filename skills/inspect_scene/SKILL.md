# inspect_scene

清单和ID由Decisions统一处理。默认网页的visual_available=false，本入口不可选；以下规则仅适用于显式配置VisualAssistance的适配器，不能替代Decisions身份结果。

**目的与调用时机：**让视觉模型补充当前图像中既有实例的类别、关系和可见性。

**输入与前提：**observed 且 visual_available；purpose仅为semantics，输入当前observation、原指令和共享调用预算。

**固定算法：**校验当前帧和已有ID，单次尽量覆盖所有相关图像、掩码编号与问题；一次结构化Responses请求，校验schema/ID并按实际RGB/掩码指纹缓存。

**输出与状态：**既有ID的语义、可见性和证据文字；证据不足保留unknown。只补充visual_semantics，不创建或重编号实体，不确认搬运完成。

**失败含义与边界：**同输入已有结果（含服务错误）拒绝重调；全任务最多8次请求，默认超时90 s。不自动重试，不输出米制深度、控制位姿或替JEV选恢复；不创建新掩码或测量地图。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。

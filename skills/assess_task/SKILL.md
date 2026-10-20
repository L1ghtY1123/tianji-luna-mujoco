# assess_task

**目的与调用时机：**按原始指令确认 required/excluded/unknown 成员，判断继续、完成或部分完成。

**输入与前提：**集合/路线任务已有观测、尚未选择本轮目标，task_assessed=false 且无待处理恢复；JEV 在 skill 开始前提交 judgment、membership 和 unseen_relevance。

**固定算法：**校验各实例 membership、group_membership 及未知区域相关性。初始Decisions视觉判断提供成员证据，JEV不得覆盖明确的required/excluded；unknown需要证据，不能猜测。初次 continue 必须确认全部成员、总数及每个原文数量组、来源和头部掩码，之后按头部图像行从上到下、行内从左到右冻结编号及 inventory_revision。分行容差为候选掩码高度中位数的一半，行基准固定。不得用总数截断代替组配额。complete/partial 核对已完成/暂缓和中转阶段；unknown 保留证据不足。

**输出与状态：**task_assessed、inventory_finalized、任务清单、初始头部 observation_id 和进度；不采图、不移动、不建图。清单缺口返回 inventory_not_ready 契约证据，不提供 task_assessed，不消耗物理执行尝试；成员与组问题可在冻结前修正，冻结后不换成员或重号。

**失败含义与边界：**未知成员不能计为完成；环境障碍、盒子和标记不自动成为任务成员。原指令要求的数量未满足时不能提前完成，中转第一段完成不等于整任务完成。

搬运清单引用首次头部观测固定的 scene_entities 全局 ID；确认 required 后将该实体角色标为 transport/collision。本技能不新增场景实体或修改初始类别、分割描述。盒子、桌面、普通障碍和目标区域保留各自 ID 与角色。

本技能内部不请求 JEV 决定子步骤或恢复。外层 JEV 选择已注册 skill 及必要参数，程序校验契约并执行固定算法。generation、反馈/观测 ID、scene token 与参数版本分别校验；Reset 后迟到结果不得应用。

当前注册目录共 16 项；完整流程、共享数值和验证边界见 [JEV skill 使用与实现手册](../../docs/jev-skills-guide.md)。文档同步不构成本轮部署、GPU 性能或物理成功验收。

冻结清单的全部required成员完成全部路线阶段后，JEV依据完成记录判断complete。verify_place始终要求完整放置、开爪反馈和放后三相机Decisions对同一ID到达所选区域的确认；最后一个物体也采图。一次释放只完成绑定的一个阶段，重看同一物体不能完成下一段。尚有待搬、中转阶段、视觉待确认或暂缓成员时不能判为全部完成。缺少Decisions去向结果不能回退为仅按释放反馈完成。

**统一编号：**全部实体使用一份 scene_entities 全局 ID；搬运清单引用其中相同 ID。冻结 required 对象的 display_id=1..N 后，盒子、障碍、目标区域和桌面依初始清单顺序续编号；后续观测不重排、不新增实体。槽位仍只使用搬运编号，非搬运编号不改变 N。

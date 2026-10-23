# bind_inventory

校验 Responses 数量和来源，保持左腕初始显示编号及永久 ID。用同帧三视角RGBD实测质心到固定机器人基座世界XY=[0,0]的水平距离，由近到远绑定执行队列；指定数量选择最近N件。高度、颜色、形状和当前TCP不影响距离，同距沿原编号。缺少有效实测位置返回source_distance_unmeasured，不猜距离。执行顺序独立于ID，绑定后本次任务固定；不新增采集或模型调用。execution_order及回放事件记录位置、距离、观测与generation。来源完整性只覆盖可见粉色源对象，不证明隐藏区域为空。all 保留 Responses 原完整性 confirmed/unknown，并用既有同帧 Decisions 建立独立 source_inventory_resolution；count 允许保留 unknown，但须有足够逐个可信的可见目标。两种模式绑定时都要求同帧 Decisions 可见未列对象检查为 none、置信度至少 0.8、来源为 openai_decisions。缺失、过期、未知或额外可见对象均停止，未选中的已列对象仍是障碍，已有 unknown 不被改写。

all 的独立证据要求初始参考、Responses 和 Decisions 均属于当前观测，身份覆盖和 none 来自同一 Decisions request hash，所有必需非桌面/标记实体有可信身份、所有源对象有左腕可信身份及实际几何覆盖。记录原 Responses 状态、两次请求 hash、观测 ID、身份集合和 none；原 confirmed 也须通过相同门。辅助 unknown 不参与身份覆盖、不进入地图。没有新增模型调用、采图或重试；最终放置与遗漏检查保持。

## 调用与证据

由DemoRunner执行Responses生成的有界程序。启动固定准备仅observe_scene和bind_inventory；其余调用由生成的程序决定。硬前提、输入与提供事实见contract.json。每次检查generation、observation ID、scene token及实际反馈。unknown、身份冲突、无解、超时和服务错误保留证据并停止。

实现与数值依据见docs/luna-skills-guide.md、docs/demo-perception.md、docs/demo-planning.md。实验回放保存模型输入图片、SAM3分割和各阶段抓取候选；记录文件不代表运动成功。

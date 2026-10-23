# identify_target

校验并绑定已校验的Responses程序执行器选定的实测ID；拆包工序的$target初为固定粉输入ID，verify_unpack成功后为独立黑产物ID，禁止用粉ID代指黑几何；路线、集合及默认inventory顺序须先确认成员，Demo inventory须按bind_inventory实测由近到远的execution_rank依次选，显示编号仅用于身份，达到明确数量不得再选。direct explicit单目标以本次明确选择确认成员。

## 调用与证据

由DemoRunner执行Responses生成的有界程序。启动固定准备仅observe_scene和bind_inventory；其余调用由生成的程序决定。硬前提、输入与提供事实见contract.json。每次检查generation、observation ID、scene token及实际反馈。unknown、身份冲突、无解、超时和服务错误保留证据并停止。

实现与数值依据见docs/luna-skills-guide.md、docs/demo-perception.md、docs/demo-planning.md。实验回放保存模型输入图片、SAM3分割和各阶段抓取候选；记录文件不代表运动成功。

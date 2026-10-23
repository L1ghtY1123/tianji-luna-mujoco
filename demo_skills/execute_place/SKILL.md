# execute_place

按已验证的带物轨迹接近并放置。松爪前以当前关节、夹爪、attachment检查区域包络和10mm碰撞阈值；位置及旋转误差记录为诊断，不要求2mm/2度准确。通过后开爪、确认开爪、detach并撤离。实际碰撞检查失败则停止，不凭计划成功自动松爪。

运动中沿用0.03 rad关节跟踪界限，与上一拍实际位置参考比较；越界停止全部活动臂并保留失败证据，不推进后续动作。此界限不作为任务末端位置精度要求。

## 调用与证据

由DemoRunner执行Responses生成的有界程序。启动固定准备仅observe_scene和bind_inventory；其余调用由生成的程序决定。硬前提、输入与提供事实见contract.json。每次检查generation、observation ID、scene token及实际反馈。unknown、身份冲突、无解、超时和服务错误保留证据并停止。

实现与数值依据见docs/luna-skills-guide.md、docs/demo-perception.md、docs/demo-planning.md。实验回放保存模型输入图片、SAM3分割和各阶段抓取候选；记录文件不代表运动成功。

2026-10-22：取消开爪扫掠，仅检查当前闭合attachment与全开detach端点；运动中的attachment保持到确认开爪。时间伸长2倍，阶段间停顿1秒。

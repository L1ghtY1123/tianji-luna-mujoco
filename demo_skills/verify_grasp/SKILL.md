# verify_grasp

提升后直接读取当前夹爪反馈，稳定且未到空夹闭合位置为held，到空夹位置为empty，反馈缺失/不稳定为unknown。不采图、不调用视觉或attachment；unknown进入程序显式分支；无可执行分支时停止。

## 调用与证据

由DemoRunner执行Responses生成的有界程序。启动固定准备仅observe_scene和bind_inventory；其余调用由生成的程序决定。硬前提、输入与提供事实见contract.json。每次检查generation、observation ID、scene token及实际反馈。unknown、身份冲突、无解、超时和服务错误保留证据并停止。

实现与数值依据见docs/luna-skills-guide.md、docs/demo-perception.md、docs/demo-planning.md。实验回放保存模型输入图片、SAM3分割和各阶段抓取候选；记录文件不代表运动成功。

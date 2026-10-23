# execute_grasp

固定执行接近、抓取闭爪、提升；不采集结果RGBD。完成后由verify_grasp读取夹爪反馈判定。

保留官方三段轨迹。接近和下降按执行手张开、对侧手实测开度规划；闭爪后，原lift命令必须以实测闭合几何通过10mm检查，此时没有attachment，核验抓取后才附物。相同命令及完整几何状态已有验证结果时复用；状态变化重新检查，不做开爪扫掠。停留按独立墙钟秒数执行，不随轨迹时间伸长。

闭爪使用原生位置伺服，驱动器输出及关节总驱动力均受自身 URDF 的 effort 限制，当前左右夹爪均为 8.066 Nm。每段轨迹的实测起点仍须在计划起点的 1° 内；超出时停止并记录实际关节、计划起点、最大偏差及反馈版本，不修改机械臂目标来掩盖偏移。

运动中沿用0.03 rad关节跟踪界限，与上一拍实际位置参考比较；越界停止全部活动臂并保留失败证据，不推进后续动作。此界限不作为任务末端位置精度要求。

## 调用与证据

由DemoRunner执行Responses生成的有界程序。启动固定准备仅observe_scene和bind_inventory；其余调用由生成的程序决定。硬前提、输入与提供事实见contract.json。每次检查generation、observation ID、scene token及实际反馈。unknown、身份冲突、无解、超时和服务错误保留证据并停止。

实现与数值依据见docs/luna-skills-guide.md、docs/demo-perception.md、docs/demo-planning.md。实验回放保存模型输入图片、SAM3分割和各阶段抓取候选；记录文件不代表运动成功。

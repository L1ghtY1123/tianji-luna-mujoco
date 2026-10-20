# 2026-10-18 实验错误与抓姿选优验证

本次修复已同步到 Windows 现用目录及 Ubuntu 现用服务；同一用户场景、同一句指令的 MuJoCo 验收成功。Decisions 视觉库存后端仍未启用，不能把这次成功描述为 Decisions 视觉验收。

## 原始错误与修复范围

原失败 run：`200924001fe040c5a11ec6c67e2ca959`。红色物体已放下，绿色物体被夹持；绿色的 18 个放置候选在带碰撞检查的 IK 阶段全部被拒绝，尚未进入放置轨迹规划。随后 JEV 选择恢复步骤时又被本地 `jev_input_budget_exceeded` 拒绝。两者是不同阶段的问题。

预算误拒的原因是估算使用带格式空格的 JSON，而实际请求使用紧凑 JSON。本次统一序列化方式：原失败输入估算由 24110 变为 22840，保留原 24000 软预算、完整状态、失败证据和全部技能选项。未提高限额或更换模型。

对原失败输入进行了一次真实 TypeSafe JEV 回放：返回 `recover_task`，实际输入 17298 tokens，输出 162 tokens，耗时 1.28049 s；这次单独回放未执行任何技能或运动。

原 18 次 IK 失败日志不足以确定唯一的碰撞或姿态原因。本次相同场景成功不等于证明旧失败仅由 GraspGen 分数引起；生成候选和数值规划也会变化。

## 抓取选择标准

`plan_grasp` 对已选执行臂的全部 IK 可行候选完成所有规划批次，比较完整 approach/grasp/lift 实际 Hermite 执行指令的总关节路程：

`sum(abs(q[k+1, j] - q[k, j]))`

从当前实测姿态开始，包含执行臂七个关节、所有中间点、阶段连接和中途折返；另一臂不动。不作角度取模。GraspGen score 仅决定调度顺序和总路程严格相等时的优先级。

可能改善当前最优值的候选复用现有接近/抓取命令碰撞校验，三段连接均满足原有 1° 起点容差；执行前仍按最新反馈复核。保持现有 lift 不作碰撞检查的约定。共享总时限仍为 180 s；超时、取消或服务故障时不执行只搜索了一部分候选的临时最优解。

这是生成并返回的有限可行方案中的最小路程，不是所有可能抓姿、所有 IK 分支及全部路径空间的数学全局最优，也不预先保证后续放置一定可行。

## 实际 MuJoCo 验收

指令：`抓取盒子里的绿色和红色物体，放到桌面的黄色区域`。

运行：`27e78ec9169443e2a81513dd215d9823`，结果 `succeeded`，耗时 555.24 s。两条搬运记录均 completed，物理状态中红、绿物体均已位于黄色区域；没有技能错误、物理错误或遗留运动。仿真真值仅用于事后核对，不进入决策、识别或碰撞几何。

| 项目 | 红色物体 | 绿色物体 |
|---|---:|---:|
| IK 可行、实际规划候选数 | 34 | 81 |
| 完成规划批次 | 3 | 6 |
| 返回完整三段方案数 | 29 | 79 |
| 最终候选编号 | 186 | 187 |
| GraspGen score | 0.859477 | 0.572969 |
| 七关节累计路程 | 4.873563 rad | 4.605623 rad |
| IK 与抓取规划合计 | 21.96 s | 84.91 s |
| 放置 IK | 18/18 | 18/18 |
| 释放前 attachment 球底间隙 | 10.936 mm | 11.270 mm |
| 释放前前端 TCP XY 误差 | 1.203 mm | 1.018 mm |

独立读取归档中的完整命令轨迹，重新计算两次路程，均与记录一致。真实实验共调用 TypeSafe JEV 36 次，最大估算输入 21960、最大实际输入 16802 tokens；未触发预算错误。加上单独失败输入回放，本次真实 JEV 请求共 37 次。

全候选选优会增加 GPU 规划耗时。单独 CPU 合成基准中，200 个候选的插值和路程计算约 1.69–2.20 s；该数字不包含 GPU、HTTP、碰撞检查和物理执行，不能当作整次规划延迟。

[实验回放及标注观测图](http://192.168.3.4:8340/replay.html?run=27e78ec9169443e2a81513dd215d9823&view=observations)

## 视觉后端的实际状态

现用服务仍通过 SAM3、RGBD 和原有关联代码维护物体身份；任务完成使用现有夹爪反馈及释放检查。新 Decisions 图像库存、跨视角/跨时刻关联及视觉完成判断仍位于 requirements 候选实现，没有启用到现用服务。

此前在线验收受 OpenAI 额度限制；本次未重新探测账户额度，也未发起 OpenAI Decisions 图像请求。上述 37 次是 TypeSafe JEV 文本决策请求，不是 Decisions 图像请求。现用 `visual_available` 标志也不能证明新后端被调用。旧关联代码尚未删除。

## 验证与部署

- Windows 现用目录：`python -m pytest -q`，1749 passed、53 skipped。
- Windows requirements 目录：同一命令，1802 passed、54 skipped。
- Ubuntu 隔离容器：`python -m pytest -q -p no:cacheprovider`，1846 passed、10 skipped。
- 修改过的 Python/测试文件 Ruff 通过；两目录 `git diff --check` 通过，仅 Git 换行提示。
- 初次完整回归暴露旧批量错误测试缺少夹爪反馈和校验响应；仅修正测试数据，明确断言一次批量规划及两次接近/抓取校验，没有放宽程序检查。
- 独立代码审查未发现本次范围内阻塞问题；复核了跨批选优、候选索引、完整命令路程、校验失败、时限及错误传播。
- SDK/vendor 保持原样，cuRobo 固定版本 `8e734f3ced1df898990bcd92de40abce475907db`；使用现有公开 batch planner 和已有校验接口。
- 部署前保存失败日志、完整快照、确认场景及原文件备份，按哈希检查源文件后更新本次 12 个文件，另同步网页 README 的旧说明。仅网页后端自动重载，未重启 GPU 算法服务。
- 已恢复用户确认场景并用于上述验收；保留成功后的场景供查看。未构建镜像；本次使用的 `jev-requirements-test-guangyu` 已停止，现用网页/cuRobo/SAM3/GraspGen 容器正常。

证据目录：

- Windows：`C:/project/tianji-jev-mujoco-requirements/.superpowers/sdd/2026-10-18-experiment-fix/`。
- Ubuntu：`/home/guangyu/projects/tianji-jev-acceptance-20261016/experiment-fix-20261018/`。

核心文件：`budget-replay-result.json`、`acceptance-final-state.json`、`acceptance-audit.json`、`grasp-result-*.json`、三份 `*-tests-final.log`、`production-before.tar.gz`、`deploy-manifest.json`。实验完整图像、标注、事件及视频由实验归档另行保存。

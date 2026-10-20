# plan_grasp_batch 性能实测：2026-09-29

本次为诊断，没有修改运行源码、skill、算法镜像、SDK、碰撞球或规划设置，没有执行机器人运动。

## 实际网页运行

- run_id：`f1b9dc84546a4bb5a6e480def57b386e`
- 指令：`pick up the blue object`
- A 臂，第一批 16 个候选全部成功；后续 79 个候选未规划。
- 网页记录该 batch 14.227642 s；相邻多次已缓存 planner 的服务调用仍约 14 s。
- 用户设置 revision：`1188f72ea6ff234d`；Lift 保持 tool Z -0.1 m。
- GPU：NVIDIA RTX PRO 6000 Blackwell Workstation Edition。

## 隔离复测范围

在现有 cuRobo 容器中启动独立 Python 进程，仅使用 /tmp 脚本和记录的输入；未调用运行中服务的写接口。新进程自己的 runtime、ESDF 和 planner 与网页运行状态独立。

日志只保留了前三个候选的完整变换，因此采用这三个真实 A 臂候选循环重复组成 16 项，保持原场景、起始关节和全部规划参数。这是代表性诊断 batch，不是原始 16 个不同候选的精确回放。每次均为 16/16 成功。

第一次 cProfile 复测：cold 42.361 s，warm 14.615 / 15.020 s。其外层栈计时受运行环境影响，不使用它来分配阶段占比。

第二次采用 Python 3.12 sys.monitoring 对函数进入/返回计时，不改写或替换 SDK 函数。cold 41.157 s，warm 14.287 / 14.628 s。下表是同级阶段的累计 wall time；IK seed solver 和 LBFGS 等子函数不再重复相加。

| 操作 | 次数 | warm 1 / s | warm 2 / s |
| --- | ---: | ---: | ---: |
| 四阶段 TrajOpt solve_pose | 4 | 13.050 | 13.122 |
| graph seed search | 4 | 0.896 | 1.249 |
| 内部 IK solve_pose | 4 | 0.043 | 0.045 |
| 原始轨迹 hard collision check | 16 | 0.171 | 0.109 |
| 场景/静止另一臂刷新 | 2 | 0.006 | 0.015 |
| 其余转换、封装及计时余量 | — | 0.121 | 0.088 |
| 整个 run_operation | 1 | 14.287 | 14.628 |

TrajOpt 内部共 8 次 LBFGS optimize，累计 12.537 / 12.711 s，包含原始优化和 finetune。这些值已经包含在 TrajOpt 行中。四阶段分别为 goalset、approach、grasp、lift；最终执行的轨迹仍只有后三段。warm 2 四次完整 plan_pose 分别约 3.929 / 3.420 / 3.500 / 3.576 s。

结论：约 90% 时间在 TrajOpt；初始化、网络、JEV、原始轨迹复核和 Hermite 都不能解释这个稳定的十几秒延迟。当前没有 Hermite 后碰撞复核。

## 负载与进一步定位边界

当前模型配置有 1,362 个原始碰撞球、38 个 links；一次规划的 16 个候选 × 2 个 TrajOpt seeds，加上多阶段和 finetune，并不等于官方单个 Franka pose benchmark。不能把球数差异直接当作已经定位到的 GPU kernel 原因：还需要分解 TrajOpt 的 self collision、world collision、kinematics、gradient 等实际开销。

只读核对补充：项目没有覆盖官方 TrajOpt optimizer / transition 配置；仍为 LBFGS `num_iters=100, inner_iters=25, fixed_iters=true, history=27`，B-spline rollout horizon 81。`finetune_attempts=1` 在该 SDK 中对应初始优化加一次 finetune，因此四阶段的八次 optimize 不代表隐藏的 batch 重试。项目 robot config 的 `load_dynamics=true` 也是进一步 GPU kernel 分解应包含的上下文；目前不能据此认定具体占比。

已确认但非本次主要瓶颈的额外工作：不足 16 项的请求会补齐，返回裁剪前会复核补齐项；场景和静止对臂每次刷新会清 graph roadmap；graph 默认每阶段第一轮启用。这些可以作为后续优化对象，但当前测量不支持它们能解决大部分延迟。

## 官方对照

- cuRoboV2 官方 2600 个问题、RTX 6000 Ada：Plan Time mean 38 ms，median 35 ms；3 kg 负载扭矩限制下 mean 52 ms。
- 官方 benchmark 使用预热后的单次 pose planning，不是项目完整抓取 batch HTTP 请求。锁定 commit 的 benchmark 明确先 warmup，并额外预调用。
- benchmark 使用 SDK result.total_time；不要将它视为所有外围操作的 wall time。
- 论文 §7.2 在 RTX4090 上报告含 Python / CUDA graph launch 的 wall-clock 35 / 42 ms。

来源：

- https://nvlabs.github.io/curobo/latest/reference/benchmarks.html
- https://arxiv.org/html/2603.05493v1#S7.SS2
- https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/benchmark/motion_plan_benchmark.py
- https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/motion/motion_planner_batch.py

原始证据：`data/planning-speed-current-state.json`、`data/planning-speed-profile.json`、`data/planning-speed-monitored.json`。

测量结束后网页 run_id 和 succeeded 状态仍不变，规划设置 revision 仍为 `1188f72ea6ff234d`。

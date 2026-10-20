# JEV 决策与当前失败审核（2026-09-29）

本文前半部分保留修改前的审核证据和代码结论，不能作为修改后行为说明。用户随后明确：Lift 沿末端坐标系 Z 是其在高级设置中的自定义选择，应保留；批准其余决策证据、选臂和规划无解后的 JEV 再决策修改。当前流程见 [README](../README.md)，实施验证见 [验证记录](validation.md)。

## 结论与范围

审核时的实现实际调用 JEV 选择任务、物体属性、目标实例、下一步 skill 和执行臂；不存在 Python 冒充模型返回选择的产品分支。但当时取消 skill 预筛选只扩大了模型收到的选项，并未建立“规划失败 → 更新证据 → JEV 再决策”的循环。因此不能称为已充分发挥 JEV 的决策能力。

原审核轮次只做代码审核、日志分析、GPU 规划对照和现有测试，没有修改产品源码、skills、运行参数、算法镜像，也没有执行机械臂运动。审核时 Windows 与 Ubuntu 上相关运行源码 hash 一致。后续获批修改不覆盖本文的原始证据。

## 当前报错的直接原因

运行 `f1a69d52f3fd4f30aac5193c93335bff`，指令 `pick up the blue object`。

- JEV 正确选择 grasp、any/blue/any、目标 blue_cylinder。
- A 臂 2 个候选通过 Batched IK，B 臂 4 个。JEV 选择 B；现有证据不能证明该选择错误。
- B 的候选 29、53、113、163 全部 approach/grasp 成功、lift 失败。没有进入 execute_grasp。
- 本轮实际保存参数为 `lift_axis=z`、`lift_offset_m=0.1`、`lift_in_tool_frame=true`。源码默认值是 `false`，即 world +Z；不能把保存值与默认值混为一谈。

tool frame 中的 +Z 随抓取姿态旋转，并不固定向上。同一批候选的 world 位移如下（mm）：

| 候选 | ΔX | ΔY | ΔZ |
|---|---:|---:|---:|
| 29 | 23.581 | 40.838 | -88.183 |
| 53 | 25.751 | 41.082 | -87.459 |
| 113 | 44.458 | -10.311 | -88.978 |
| 163 | 83.076 | 16.367 | -53.203 |

保持场景 token、4 个目标矩阵、起始关节、另一臂关节和其他设置相同，直接调用现有 plan_grasp_batch，仅改变 lift_in_tool_frame：

| 对照 | 完整三段通过数 | 耗时 |
|---|---:|---:|
| 当前 tool +Z 100 mm | 0/4，全部 lift 失败 | 10.77 s |
| world +Z 100 mm | 4/4，含原始轨迹碰撞复核 | 14.47 s |

这是该输入下的参数因果证据，不是完整执行或抓取成功的证明。服务仅返回 lift 阶段失败，没有具体碰撞体或求解残差，不能进一步断言一定撞桌面。用户后续确认 tool frame 是其自定义配置；不能把与源码默认值不同视为设置错误。

world +Z 仅是此次因果对照条件，不是要替用户恢复的设置。保留用户已应用的末端坐标系 Z、偏移和其他参数。JEV 将获得实际世界位移与规划失败证据，可以判断另一臂是否值得尝试或停止，但不会改变 Lift 配置。Reset 同样保留已应用的算法设置。

证据：`data/current-error-audit.json`、`data/lift-frame-audit.json`、`data/lift-probe-saved_tool_z.json`、`data/lift-probe-world_z.json`。对照未写网页设置；当前保存值仍为 true。

## 修改前的代码审核发现

### 1. 规划失败没有返回 JEV，无法按新证据换臂（主要架构缺口）

`runner.py:184` 执行 skill，异常进入 `runner.py:203` 的外层终止处理。B 臂无可行三段轨迹后，A 臂是否值得尝试不再交给 JEV。服务错误、规划无解、执行错误目前都在这个终止路径结束。

只把异常改成 continue 也不能解决：`skills/choose_arm/contract.json` 要求 arm_selected=false，而 `runner.py:188` 只会将 provides 对应事实设为 true。既没有已尝试臂记录，也没有使旧 arm/plan 失效的规则。

### 2. JEV 收到的 state 不是完整决策证据

`runner.py:142–148` 提供指令、属性、目标、执行臂、facts、物体/机械臂状态，以及最近三条 skill_complete。它没有明确包含桌面几何、冻结的规划设置、失败结果和尝试历史。

`decisions.py:29` 的选臂投影包含全部可达候选的 score、关节距离、IK 残差，但不含候选的 Approach/Lift 世界位移或三段规划结果。JEV 当时无法凭空知道 tool +Z 实际向下，也无法从抓取终点 IK 成功推断完整三段必定可行。

### 3. 全部选项不等于每一步都有策略空间

本轮每次 skill 问题确实都有 9 个 skills 加 stop，且先由 JEV 选择，之后才校验所选契约。没有恢复 Python 预筛选。

但是审计完整成功运行 `77ddb57a82b649a2a1331c30ec868f4c` 的 9 次 skill 选择：8 个状态只有一个 skill 满足契约，只有目标确定后可选择先 build_scene 或先 generate_grasps。这些依赖有实际意义；现阶段的主要语义决策仍是任务理解、目标选择和执行臂选择，不能仅凭“每步调了 JEV”证明策略能力强。

### 4. 执行臂仍在 JEV 作答前被 Python 筛选

`decisions.py:162–168` 只提供 valid_candidate_count 非零的臂；两臂均零时直接抛 no_reachable_arm。本次两臂均非零，A/B/none 都提供了，故不是本次失败原因。若希望贯彻“先让 JEV 判断、后校验”，这里还需调整并验证。

### 5. 扩充 skill 仍有 Python 接线工作

Registry 自动发现新增契约和说明，但执行由 `Runner.execute` 显式分发，结束逻辑特判 verify_grasp，事实仅支持写 true。当前适用于单次抓取 demo；新增算法 skill 仍须实现执行代码和结果语义。不能声称只增加 SKILL.md/contract.json 就能扩展任意任务。

## 已获用户批准的最小修改边界

1. state 明确加入桌面、冻结规划设置、算法结果及本轮尝试记录；Python 计算候选在 world frame 下的位移等数值事实，JEV 判断这些事实对任务意味着什么。
2. choose_arm 每次提供 A/B/none 和两臂完整证据，JEV 先选，Python 后检查可达性；不由 Python 指定替代臂。
3. 仅将运动开始前的 planning_no_valid_candidate 作为可继续决策的结果：记录失败臂与阶段，使当前 arm/plan 失效，保留 scene、grasps、IK 产物，然后回到完整 skill 选择问题。每臂最多尝试一次，失败记录进入下一轮 state。JEV 可以选择重新选臂，也可以 stop。
4. 契约拒绝、服务异常、超时、执行异常继续停止。保留真实几何、碰撞和结果验证；JEV 不修改 seeds、碰撞阈值或手工设置，不补充观察、不增加技能、不拆分 plan_grasp/execute_grasp。
5. 保留用户自定义的末端坐标系 Z 配置。世界坐标位移是提供给 JEV 的数值事实，不用于 Python 自动筛选方向，也不授权 JEV 改写手工设置。即使扩展失败决策，也不保证另一臂有解。

第 2–3 项改变 choose_arm 的选择/重选条件和 plan_grasp 的失败语义，属于用户要求预先审核的 skill 行为变更。原审核轮次没有通过 runner 或 adapter 间接实施；用户随后明确“其余按你建议修改”，已批准此边界。选择后的契约检查、每臂最多一次和异常终止仍由程序执行，不伪装成 JEV 决策。

## 验证及参考

原审核轮次运行：

```text
.venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_decisions.py tests/test_runner.py tests/test_runner_failures.py tests/test_planning_settings.py
46 passed in 1.53s
```

这些历史测试证明修改前边界按代码运行，不证明修改后的 JEV 再决策功能。GPU 对照证明该输入下更换 lift frame 会改变规划结果，没有执行轨迹或修改网页场景。获批修改的新增验证单独记录在 validation.md。

已核对锁定 cuRobo commit 8e734f3ced1df898990bcd92de40abce475907db 的安装源码语义：tool offset 右乘目标位姿，world offset 左乘；同一标志还影响线性运动约束的坐标系。未修改 SDK。tianji_base 的默认是 tool -Z 100 mm，JEV 的默认是 world +Z 100 mm，二者不能仅复制 tool-frame 开关而不核对符号。

TypeSafe 官方说明：[Choice 的输入与选项](https://docs.typesafe.ai/primitives/choice)、[state 与 questions 的职责](https://docs.typesafe.ai/concepts/state)、[函数选择示例](https://docs.typesafe.ai/cookbooks/function_calling)。应用提供问题与执行接口，JEV 根据所给证据输出选择；程序出题本身不妨碍模型判断，关键在证据、可选行动及结果能否继续进入下一次决策。

# 碰撞球精简与 JEV 输入长度修复（2026-09-30）

> 后续修正：1359 球版本已写入运行配置；本页保留被撤回的 1096 球历史记录。请以[本次修正记录](collision-sphere-correction-2026-09-30.md)和 `algorithm-provenance.json` 为准。

> 撤回时的状态：1096 球模型已撤回几何质量验收。进一步审核确认 Link_Stand 的 raw 球包络超出实际网格边界约 110.5 mm，机械臂也存在漏覆盖。4059 球替代候选因 A 臂预热后规划耗时 94.78 秒被拒绝，4094 球候选也未采用；目前没有通过完整验收的替代模型。下文的验收、性能和部署文字均为该 1096 球版本的历史记录，不表示当前修复完成。JEV 输入长度修复独立于碰撞模型，不受此次撤回影响。

后续试验中，跨多个优化 history 选择球、再反复用 SURFACE 补面的流程造成球数和规划开销膨胀，该流程已拒绝，不能作为正式生成方案。下一步应先确定逐 link 的球数预算，使用官方拟合的 final 输出，再独立检查几何包络、关键关节接口及运动姿态，并在相同输入和参数下验证规划性能；局部采样覆盖通过不能替代这些检查。

[补充几何复核](../data/collision-sphere-quality-rejection-2026-09-30.json) 分开报告外露网格缺口、球并集的实际外凸和无法可靠分类的开口网格区域。原来的错误在项目拟合配置：支架强制采用极端的覆盖/外凸权重比，立柱与支架用了过粗的整体凸包代理，静态补球半径达到 40 mm；验收却只看覆盖和有限轨迹，没有约束外凸。默认、零位及实际 grasp/lift 姿态的复核均支持撤回结论。

本历史版本共 1096 个球，相比原来的 1362 个减少 266 个（19.5%）。配置 SHA256 为 `26b26b7b8599a969d941102213f38a759ef462dbd58ea57190646d3fb1663a71`。该减量结果没有满足几何质量要求。

| 部分 | 原来 | 1096 球历史版本 | 处理方式 |
|---|---:|---:|---|
| 机械臂与夹爪 | 711 | 657 | 保留原中心和半径；删除 54 个有严格包含证书的冗余球 |
| 中央底座与立柱 Link_Base | 261 | 112 | 排除四脚等 31 个地面组件，保留中央 15 个组件；分段调用官方拟合 |
| 头部相机与支架 Link_Stand | 133 | 12 | 对合并几何的保守代理调用官方拟合 |
| 两个腕部相机 TCP_Link_L/R | 257 | 315 | 官方拟合并补齐检测到的覆盖缺口，分别为 155 / 160 |
| 合计 | 1362 | 1096 | 减少 19.5% |

腕部相机球数增加了 58 个。这部分新增小球用于补足拟合覆盖，没有以缩小球或忽略相机换取球数下降。主要减量来自静态部件和机械臂中被完全包含的球。四脚底座不参与拟合；中央低位法兰、支撑肋和立柱仍保留。

该历史版本的新拟合全部调用 NVIDIA 公开接口 `curobo.sphere_fit.fit_spheres_to_mesh`，固定 SDK commit `8e734f3ced1df898990bcd92de40abce475907db`，使用 MORPHIT 与局部 SURFACE 拟合。保留的运动部分明确记作原项目已审核模型，不声称本次重新生成全部球。生成器没有手工改大球半径，也没有调用自动生成 ignore matrix 的接口。原有 `collision_sphere_buffer`、`self_collision_buffer`、`self_collision_ignore`、关节限制和开夹爪锁定值均保持不变；YAML 除 `collision_spheres` 外深比较完全相同。

[官方拟合实现](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/sphere_fit.py)；[逐 link 拟合参数与来源](../data/collision-sphere-fit-2026-09-30.json)。

## 历史几何检查

已确认的支架缺口点在 link 坐标系为 `[-60, -10.785, 163.894] mm`，原球集在该点有 16.448 mm 的漏覆盖；1096 球历史版本覆盖了该点，但这项局部结果不能说明整体包络合格。四个重新拟合的 link 对完整网格顶点以及两个独立随机种子的表面采样完成检查，没有在这些检查点上发现缺口。

机械臂和夹爪中保留的 657 个球均为原球的精确子集；删除的 54 个球均被同 link 的保留球严格包含，因此这些运动部分的球并集没有扩大或缩小。保留原 buffer 后此包含关系仍成立。单 link 的网格包含关节接口和内部装配面，报告保留其测量结果，没有要求为这些面额外堆球，也没有宣称全部网格面或实体体积均被证明覆盖。

默认/零位姿、旧规划的 1287 个原始姿态，以及新模型重新规划得到的 1287 个原始姿态，都没有自碰撞球重叠，桌面球间距均大于 10 mm。新轨迹最小 self gap 为 1.422471 mm、最小桌面间距为 72.262240 mm。检查对象为这些离散原始姿态和桌面盒；不能据此证明任意工作区或节点间连续运动。没有添加 Hermite 轨迹碰撞复核。

[已撤回的历史验收摘要](../data/collision-sphere-refit-acceptance-2026-09-30.json)、[独立几何检查](../data/collision-sphere-refit-quality-2026-09-30.json)、[旧路径检查（gzip JSON）](../data/collision-sphere-refit-baseline-paths-2026-09-30.json.gz)、[新路径检查（gzip JSON）](../data/collision-sphere-refit-new-paths-2026-09-30.json.gz)。

## 历史同参数 GPU 对比

在独立进程中重放固定 A 臂 batch：历史记录的 3 个目标重复组成 16 项，并非 16 个不同抓取目标。两版请求与参数的哈希一致。LM seeds=32、IK seeds=16、TrajOpt seeds=2，hard clearance=10 mm，optimizer activation=30 mm，保留原三段规划及复核；测试没有执行 MuJoCo 或硬件动作。

| 指标 | 原 1362 球 | 1096 球历史版本 |
|---|---:|---:|
| 首次调用（含 planner 初始化） | 30.975 s | 22.661 s |
| 预热后第 1 次 | 6.865 s | 5.128 s |
| 预热后第 2 次 | 6.942 s | 5.275 s |
| 预热后第 3 次 | 7.138 s | 5.576 s |
| 预热后平均 | 6.982 s | 5.326 s |
| 每次规划成功数 | 16 / 16 | 16 / 16 |

这组输入平均缩短 23.7%，不等同于所有场景或整次 JEV 任务均缩短相同比例。完整机器人理论检查球对从 528112 降至 358950（32.0%）；实际 A/B planner 还会按冻结对侧臂构建自己的模型。

[逐次耗时、参数、输入哈希与显存记录](../data/collision-sphere-refit-benchmark-2026-09-30.json)。

## JEV 的 400 / max_tokens_exceeded

本次是单次输入长度限制，不是余额耗尽。原失败请求原样重放仍返回 400；整理输入后同一状态可正常返回。TypeSafe 文档列出 state 与最长 question 合计 32k tokens、整次请求 64k tokens；返回的 input_tokens 使用量不可直接当作该局部上限的计数。[官方模型说明](https://docs.typesafe.ai/models)

修复保留完整 skills 目录、全部候选编号和 A/B/none 选项。选择 skill 时提供各臂候选数量、全部编号、质量及运动方向范围；选择执行臂时提供逐候选 score、关节距离与世界位移。重复数值共享引用，已通过 IK 的残差保留范围，重复失败原因分组。仅给 JEV 阅读的 score、关节距离（rad）和位移（m）保留 6 位小数；原始算法位姿、规划排序、冻结参数和执行数据不变。

| 真实 API 回放 | 结果 | skill / arm input_tokens |
|---|---|---:|
| 原请求不修改 | 400 / max_tokens_exceeded | 未返回 |
| 原失败状态使用修复 | HTTP 200 | 9951 / 9401 |
| 两臂各 200 候选 | HTTP 200 | 10868 / 21021 |
| 两臂各 200 候选且均规划失败 | HTTP 200，JEV 选 stop / none | 15584 / 21221 |

网页会把该 400 类型明确显示为输入过长。完整 `evaluate_arms` 诊断结果另外写入 JSONL 的 `data.full_result`，不送入网页摘要或 JEV state；从磁盘重放时也会先做证据投影，避免再次超长。[真实 API 验证记录](../data/jev-token-limit-fixed-validation.json)

## 历史复现与部署记录

原 1362 球快照保存在 `data/collision-sphere-source-2026-09-30.yml`，仅供审计和离线拟合。运行配置只有 `resources/curobo/marvin_gui_dual.yml`。产生本历史版本的极端权重和整体凸包流程已从生成器删除，因此原复现命令也已撤下，不能再作为当前拟合操作说明。

生成器禁止输出覆盖 assets/resources，生成的候选仍须单独审核，不能直接替换运行模型。现有离线入口要求用 `--num-spheres` 明确每个选中 link 的数量上限，仅保存官方一次调用的 final centers/radii，不合并 history 或循环补球，也不接受旧流程的 resume 记录。例如在现有 cuRobo 环境中，只测试一个 link：

```bash
python scripts/fit_collision_model.py --links Link4_L --num-spheres 32 --output /tmp/jev-link4-official-final
```

输出仍是未验收的候选；未选中的 link 保留 source-config 原值，不能把这个单 link 试验当作整机新模型。CPU 检查命令：

```bash
python scripts/audit_collision_model.py --config /tmp/official-fit-reproduce/marvin_gui_dual.yml --output /tmp/official-fit-audit.json
```

当时的变更未修改 skills 契约和执行边界，不修改 SDK、共享算法镜像或 tianji_base。网页 Compose 增加只读 resources 挂载，查看器与 cuRobo 使用同一份 YAML。首次应用挂载要用现有镜像重新创建网页容器；新的碰撞模型需要重启 JEV cuRobo 进程加载。日常已挂载源码的自动重载方式保持不变。

历史验证记录（不构成当前几何验收）：本地 Python 257 passed / 51 skipped；Ubuntu 补跑对应的 48 个 launcher 测试及 3 个进程测试全部通过。相关 Ruff 检查通过。前端 121 个测试和构建通过。

历史部署记录：当时 Windows 与 Ubuntu 的 26 个变更文件逐一校验 SHA256 一致；网页碰撞球接口返回 1096 球，cuRobo health 返回上述 1096 球模型 SHA256，规划参数版本仍为 `1188f72ea6ff234d`。网页容器用原镜像重新创建以加入 resources 挂载，JEV cuRobo 容器重启加载模型，GraspGen 容器未重启；三张镜像 ID 均保持原值，tianji_base 模型仍是原 `d7e627...`。

当时在网页容器中回放原失败 state，JEV 成功返回 `choose_arm` 和 `A`（9952 / 12719 input_tokens）。此回放把原完整 skill state 再用于选臂，因此选臂用量与前面的标准选臂压力测试不同；没有执行动作。[部署与健康记录](../data/collision-sphere-refit-deployment-2026-09-30.json)、[部署后真实 JEV 回放](../data/jev-deployed-validation-2026-09-30.json)。

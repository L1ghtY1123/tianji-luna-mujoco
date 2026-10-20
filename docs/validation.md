# 实测验收

## 2026-09-29：200 候选、原始开爪碰撞球与独立查看页

按用户明确要求，项目侧移除了闭合包络，IK、TrajOpt 和 hard clearance 复核均使用原始 YAML 球与原有 link buffer，两夹爪锁定在 0 rad 开爪位。保留 10 mm hard clearance；不使用额外闭合扫描。原 YAML、URDF、vendor 与镜像未修改。GraspGen 通过现有服务的两次真实推理合并为最多 200 个候选，source_batch/source_index 保留来源，唯一 index 贯穿两臂筛选与规划。

使用原失败的直径/高度各 45 mm 红色圆柱场景，在独立、持续步进的 MuJoCo 实例中调用真实 JEV 和同一 GPU 服务；测试未移动主网页中的机器人。两次均由 JEV 选择 B 臂，三项物理抓取验证全部通过：

| 实测阶段 | 首次，含 solver/CUDA 冷启动 | 后续缓存复用 |
|---|---:|---:|
| GraspGen 实际候选 | 200 | 200 |
| GraspGen 总时间 | 0.252 s | 0.187 s |
| ESDF | 1.107 s | 0.154 s |
| A/B 全部 IK，4 次 HTTP | 44.202 s | 1.114 s |
| A/B 可达候选 | 0 / 88 | 0 / 79 |
| 全部 plan_grasp_batch | 6 批 / 82.787 s | 5 批 / 44.953 s |
| 执行 | 14.850 s | 14.855 s |
| JEV，13 次调用 | 6.261 s | 5.845 s |
| 完整任务 | 149.501 s | 67.146 s |

第二次全部 79 个可达候选均完成三段规划并通过复核，没有只规划前几个或首次成功就提前停止。除首批冷启动 38.733 s 外，每批实测 8.226–9.937 s；这里包含 raw/Hermite 路径复核和服务传输，不等于单个 GPU kernel 耗时。两次 GraspGen 采样不同、规划批数不同，这不是固定输入的纯冷/热对照，也不能承诺更多候选的总时长不增加。

主网页新增“碰撞球查看”入口，独立页为 `/collision-spheres.html`；API 只读副本显示 38 个 link 的 1362 个原始球，开爪 mesh 与球使用同一次 FK。实际 headless Edge 已验证点击入口打开新页、三维画布显示、暂停跟随、隐藏场景网格及刷新，无 JavaScript 异常。截图保存在本次本地临时验收目录；原球半径和球心、live qpos/ctrl 不变由真实 MuJoCo 测试验证。

完整数值、每批真实候选编号、配置及成功证据见 [200-candidates-validation.json](200-candidates-validation.json)。原始测试记录位于 Ubuntu `/tmp/jev-200-validation`。这是对下面“扩大包络尚未修改”历史诊断的后续修正。

## 2026-09-29：运行记录分类与红色圆柱失败诊断

网页“运行记录”现在区分 JEV 决策、skill 执行、算法计算、系统事件。只有 `kind=jev` 使用模型详情；计数分别为全部事件与实际 JEV 调用数。前端 106 项测试和构建通过；仅更新静态页面，无后端重启、场景重置或参数修改。部署后的 HTML/JS/CSS 与本地构建字节一致，原失败任务和 generation=18 保留。

用户失败记录 `c6cbd63230af4c90a2d9e1200bb799e8`，指令“抓取红色物体”，目标为改成红色的 `blue_cylinder`（直径/高度各45mm）。实际重现 GraspGen 100 候选，A/B 两次外层 batched IK 都为0；在 choose_arm 中、发出选臂模型请求之前以 `no_reachable_arm` 停止，没有调用轨迹规划或执行。

读取这次复现的全部100个原始TCP目标，按原URDF夹爪运动学、原碰撞球及 buffer，计算球对桌面实体的精确 SDF 间距。比较当前覆盖闭合全过程的扩大球包络，与81个开合角度的原球；逐球扣除速度上界×半角步，保证覆盖采样之间的连续运动。

| 夹爪对桌面满足10mm的候选 | A | B |
|---|---:|---:|
| 当前扩大球包络 | 0/100 | 0/100 |
| 原球逐角度采样 | 27/100 | 27/100 |
| 原球采样再计入连续保守界 | 27/100 | 26/100 |

B臂原始候选95（IK批次索引36）：连续闭合间距下界20.055mm；扩大包络仅5.754mm。该候选的原IK响应位置/方向误差分别6.14e-8m和3.73e-8rad，但最终success=false。候选66（score=0.995428）也具有约20.055mm闭合间距下界。

这证明当前包络会把部分夹爪—桌面本可满足10mm的目标位姿保守拒绝；没有证明完整IK约束、其他机器人部位、其他障碍或三段路径全部可行。现有HTTP结果没有逐seed保留SDK成功标志和hard check拒绝原因，因此不能把每个失败都逐项归因为桌面。原始配置、阈值及碰撞算法均未修改；改变复核策略须按用户要求先审核。数值、资产哈希及方法边界见 `red-grasp-collision-audit.json`。

## 2026-09-29：参数锁、缓存、选臂请求与 Hermite 补点

本地 `C:\project\tianji-jev-mujoco` 与 Ubuntu `/home/guangyu/projects/tianji-jev-mujoco` 已同步。高级参数默认锁定，空闲时可手动解锁；Start 重新锁定。执行时间设置空闲时可直接编辑。官方 batch seed 默认保持 IK=16、LM=32、TrajOpt=2。镜像、NVIDIA SDK、vendor 快照及原始碰撞球不改。

用户的 129.684 秒记录 `f462dae523394c369c66f1a138975e06`：JEV 36.448 秒（其中一次 20.446 秒）、两臂 IK 38.956 秒、batch 38.413 秒、执行 14.846 秒。日志证实包含 solver 初始化/CUDA Graph 录制。旧 9.939 秒记录 `0370aa892e6f4a5597a4cb966f1c4765` 使用单候选 `/tools/plan_grasp`，与当前全部可行候选的 batch 工作量不同。

当前 1–16 个规划候选统一 capacity16，返回时裁去 padding，保留真实候选数量及索引，避免 3/5/9 个候选分别初始化不同容量。单独真实 GPU 验证：6 个真实候选、capacity16，首次规划 40.118 秒，3 个通过全部检查；这是冷启动成本，未被隐藏为热性能。

实际复跑曾发现选臂请求 `max_tokens_exceeded`（`e6651f0217354a27b7c384a25c8e47c2`），任务正常终止。修复重复输入：完整矩阵和 solver 配置不重复塞入 JEV 选项；state 保留每个可达候选的评分、关节距离及 IK 残差。该实际失败输入的 state+questions 从 48630 减为 8311 字符，未删除候选或更改 JEV 选臂调用。

随后在已确认的用户场景中执行“抓取黄色方块”，两次均由真实 JEV 选择 A 臂，并使用现有 GraspGen、GPU cuRobo、MuJoCo 电机和接触物理：

| 项目 | 第一次缓存复用 | 第二次缓存复用 |
|---|---:|---:|
| run_id | `07f7e609e28a451a8c0188277209e0e1` | `ba5e72a602ce4a3fb15216894c853ee3` |
| 总耗时 | 35.123 s | 34.621 s |
| JEV 调用 / 总耗时 | 13 / 7.674 s | 13 / 7.574 s |
| 两臂 batched IK | 0.123 s | 0.114 s |
| batch 规划含后处理与复核 | 12.126 s | 11.752 s |
| 执行 | 14.895 s | 14.863 s |
| 真实候选数 / 实际容量 | 3 / 16 | 5 / 16 |
| 物体位移 | 99.365 mm | 99.362 mm |
| 物体/TCP 变换误差 | 0.191 mm / 0.102° | 0.182 mm / 0.104° |
| 夹爪、位移、变换三项检查 | 全部通过 | 全部通过 |

两次均报告 `quintic_hermite_point_insertion`，默认倍率 3。三段原始点数 81/41/41，执行点数 241/121/121，原始与执行点间隔均 0.02 s，纯轨迹时长 3.2→9.6 s；没有在执行端再次乘倍率。补点后路径另做 hard clearance 检查。测试结束 Reset，状态 idle、已确认、无在途算法和活跃命令，保留用户场景及参数。

自动验证：Python 174 项，前端 104 项；Ruff 和 Vite 构建通过（保留原有大 chunk 提示）。本次浏览器 AX 获取仍超时，未完成视觉点击验收；不能把自动测试当成视觉验收。以下 09-28 记录是历史实现的证据，不代表当前实现。

## 2026-09-28

部署目录：`/home/guangyu/projects/tianji-jev-mujoco`。网页：`http://192.168.3.4:8340/`。
该次验证使用了旧宿主服务 `jev-mujoco-guangyu.service`；现已移除，网页改由单独的 Docker 容器 `jev-mujoco-web-guangyu` 运行。算法容器仍为 `jev-mujoco-curobo-guangyu`、`jev-mujoco-graspgen-guangyu`。

## 审核后的技能边界修改

按用户批准的边界，`plan_grasp` 保留完整三段轨迹，`execute_grasp` 一次执行开夹爪、接近、抓取、闭合、参考位姿记录、提升；`verify_grasp` 继续由 JEV 单独选择。三个旧执行技能已从本地及 Ubuntu 注册表移除。

真实指令“抓取左侧黑色方块”成功，运行 ID `0370aa892e6f4a5597a4cb966f1c4765`，JEV 选择 A 臂。三项验证全部通过：目标位移 98.95 mm，闭合程度 70.59%，物体/TCP 变换残差 0.330 mm / 0.031°。实际 JEV 调用 13 次，平均 366 ms，累计 4.758 s，整轮 9.939 s。这是单次运行结果，不代表成功率或固定性能。

运行日志确认：`execute_grasp` 六个内部阶段均记录开始和完成，其间 JEV 请求数为 0。实际接近轨迹执行期间点击 Reset 对应的 HTTP 接口，机器人关节和物体位姿恢复确认基线；一秒后仍 idle、无任务事件、无活跃命令，物理时间增长 1.009 s。取消运行 ID `9fad39b847f2455b89c87b925f17cb68`。完整证据见 `skill-boundary-validation.json`。

本地及 Ubuntu 各 55 项 Python 测试通过；本地 Ruff 通过；前端 9 项测试及构建通过。另一次独立只读审查及 30 项针对性测试未发现阻塞问题。Ubuntu 虚拟环境未安装 Ruff，未将其记为远端检查通过。

本次只更新技能执行边界、契约说明、阶段日志及对应测试文档，算法服务代码、容器和镜像均未变更。网页视觉验收仍受下文所述限制。

## 合并前的真实模型 + GPU + 物理抓取记录

下面来自实际 JEV、现有 GraspGen/cuRobo 镜像及 MuJoCo 接触物理，没有生产模拟返回值。

| 指令与场景 | JEV 选臂 | 结果 | 目标位移 | 闭合程度 | 位姿跟随误差 | JEV 调用 | JEV 平均耗时 | 整轮耗时 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 左侧黑色方块，Y=+0.15 m | A | 成功 | 98.92 mm | 70.59% | 0.369 mm / 0.125° | 16 | 395 ms | 11.58 s |
| 右侧黑色方块，Y=-0.20 m | B | 成功 | 98.39 mm | 70.60% | 0.386 mm / 0.328° | 16 | 396 ms | 17.27 s |

这是两次示例验证，不能推导通用成功率。首次加载/规划的冷启动耗时更高，见 `algorithms.md`。完整三项证据与运行 ID 见 `end-to-end-validation.json`；原始每步日志在两端项目的 `data/runs` 下。

另外实际验证：
- “抓取黑色物体”正确选择 kind=any；“把黑色方块放到盒子里”选择 grasp_place。
- “抓取黄色方块”在没有黄色物体时选择 missing，未调用规划或运动技能。
- 在 JEV 请求进行中执行 Reset：恢复双臂关节角误差 0；物体完整位姿恢复；两秒后仍 idle、无旧事件或旧运动；物理时间持续增长。见 `reset-validation.json`。
- 此记录对应当时的测试场景；当前物体颜色和位置以网页中的实际场景为准。

## 初版自动检查记录

- Windows：42 项 Python 测试通过，Ruff 通过。
- Ubuntu：同样 42 项 Python 测试通过。
- 前端：9 项 Node 测试通过，Vite 生产构建通过。
- GPU 单独验证：100 个 GraspGen 候选，A/B 均有可达 IK 与完整三段轨迹。
- cuRobo Python 源码 361 文件与锁定官方提交一致；碰撞球文件保持原始字节，共 38 links / 1362 spheres。

## 浏览器验收限制

网页入口、静态资源、WebSocket 和所有 HTTP 操作已运行验证。浏览器自动化的 DOM/截图读取持续超时，尚未完成页面视觉验收，不能声称已逐项点击验收。前端已消除空闲时持续重绘百万三角形的开销，采用按需绘制且最高 30 FPS；这没有被冒充为浏览器超时已解决的证据。


## 2026-09-29：仅复核原始轨迹，选优后再做 Hermite

用户确认 `plan_grasp_batch` 只处理 cuRobo 原始轨迹，五次 Hermite 补点不做避障检测。本节替代前文“补点路径另做 hard clearance”的当前行为说明，前文实测仍是旧实现的历史记录。

- batch 服务复制官方结果的精确 q/v/a/dt，对每个成功候选的原始三段轨迹做一次 hard clearance；不生成 command_positions_rad。
- 全部 batch 按原有策略算完并按 GraspGen score 选优后，网页后端仅为选中候选的三段轨迹做 Hermite，不复核补点碰撞。默认倍率 3、原始 dt 不变。
- 保留 200 候选、两臂每批 100 的 Batched IK、每批最多 16 的 plan_grasp_batch、原始碰撞球、10 mm clearance、SDK 和算法镜像。此次没有启用“首个成功 batch 后停止”。
- 网页名称为 `tiainji`，运行卡片使用 `Batched IK`、`plan_grasp_batch`。实际运行卡片、主页面、碰撞球页面均在 Edge 验证，无 JS 错误。

真实 JEV + GraspGen + cuRobo + MuJoCo 验证：`d95f5c34c90a422a9a100b1c42ccd4b4`，指令“抓取左侧黑色方块”，JEV 选择 B 臂，200 个候选，抓取成功，三项测量全部通过。原始记录为 Ubuntu `data/end-to-end-d95f5c34c90a422a9a100b1c42ccd4b4.json`；摘要为两端的 `data/raw-planning-validation.json`。

| 项目 | 本次冷启动实测 |
|---|---:|
| 整轮 | 124.433 s |
| JEV，13 次 | 6.783 s |
| A 臂 IK 两批 | 20.133 / 0.026 s |
| B 臂 IK 两批 | 18.015 / 0.026 s |
| B 臂规划四批，16/16/16/13 个候选 | 34.470 / 9.362 / 9.490 / 9.680 s |
| 最终候选 | 6，score 0.991335 |
| 原始点数 → 执行点数 | 61/61/41 → 181/181/121 |
| 原始 → 执行轨迹时长 | 3.2 → 9.6 s |
| 物体移动 | 100.834 mm |
| 物体/TCP 变换位置残差 | 0.0163 mm |
| 物体/TCP 变换方向残差 | 0.00351° |

本轮包含服务重启后的 solver 初始化。此前用户热运行 `9933aea7b45048eba6d1903a2f48b47e` 是 A 臂、42 个规划候选，IK 合计 0.688 s、三批规划合计 30.738 s；场景、候选、执行臂和缓存状态不同，不能据此声称本次改动的固定加速比例。全部规划批次仍然执行，因此初始化和扫描后续批次的成本仍存在。

验证：Ubuntu `python -m pytest -q -p no:cacheprovider` **223 passed**；Windows 全套复跑 **198 passed, 25 skipped**，此前首次在 `test_planning_timeout_is_terminal_and_late_result_cannot_move_robot` 出现 50 ms 时钟断言波动，独立 8 项与全套复跑均通过，未修改该业务或测试。Ruff 通过，前端 113 项通过，Docker 构建和容器内 `pip check` 通过。真实测试结束后已 Reset，任务 idle、无在途算法，场景回到确认基线。

当前网页镜像 `tianji-jev-mujoco:0.2.0-guangyu`：`sha256:672e661added34bffc2ee4d876e4ebb2d9717ea2b194bc97cd7a5ee4191e7f4c`，容器健康且显示完整名称。cuRobo 和 GraspGen 镜像 ID 与上一节 Docker 迁移记录一致；本次仅重启 cuRobo 以读取项目挂载代码。旧网页 manifest 已不存在，未删除任何算法镜像。


## 2026-09-29：稳定降序与首个成功 batch 后停止

用户本轮明确审核批准该策略。本节是最新行为说明，替代上一节“仍执行全部 batch”的描述；上一节实测作为历史记录保留。

`Batched IK` 筛选 → GraspGen score 稳定降序（同分保留输入顺序）→ 每批最多 16 个调用 `plan_grasp_batch` → 本批全部失败才继续 → 首个成功 batch 完整返回后选最高分可行原始轨迹 → 只对该轨迹做五次 Hermite 插值 → 执行。原始轨迹的 hard clearance 保留，Hermite 补点不再做避障检测。200 候选、双臂 IK、seeds、碰撞阈值、原始碰撞球与 SDK 保持原值。

完成事件新增 `stopped_early` / `unplanned_candidate_count`；最终计划记录 `planned_candidate_count` / `unplanned_candidate_indices`。`batch_count` 仍表示全部候选最多需要的分组数，`batches` 仅包含实际完成的组。网页卡片明确说明未规划数量，不将其视为失败。

### 固定输入的真实 GPU 对比

同一份 59 个 B 臂 IK 可行候选，冻结相同起始/另一臂关节、场景 token、TCP 矩阵、solver 配置。两次均使用已经初始化的相同算法服务，不重新生成候选或重启服务。对比包含原始轨迹复核、HTTP 和最终 Hermite，未执行 MuJoCo 运动。

| 项目 | 修改前 | 修改后 |
|---|---:|---:|
| plan_grasp 总耗时 | 40.5229 s | 9.9563 s |
| plan_grasp_batch 调用次数 | 4 | 1 |
| 已规划候选 | 59 | 16 |
| 未规划候选 | 0 | 43 |
| 选中原始候选 index | 20 | 20 |
| GraspGen score | 0.9906056523 | 0.9906056523 |

本样本减少 30.5666 s，约 75.4%；这是固定输入的规划阶段对比，不代表任意任务的整轮加速比例。Ubuntu `data/score-order-fixed-input.json` 保存完整输入；`data/score-order-before.json`、`data/score-order-after.json` 保存两次结果，两份结果也已同步 Windows。

### 完整真实抓取

`8c11e8c2b46145a296ca9ac1544b55d0`：“抓取左侧黑色方块”，真实 JEV 选 A 臂。75 个可行候选最多分 5 批，实际只计算首批 16 个，其中 13 个成功；选中 index=2、score=0.9912797809，剩余 59 个未规划。首批含 A 臂 planner 首次初始化，耗时 33.0117 s；整轮 54.2570 s。夹爪夹持、物体移动、物体/TCP 变换一致三项均通过。记录见 Ubuntu `data/end-to-end-8c11e8c2b46145a296ca9ac1544b55d0.json`，摘要 `data/score-order-end-to-end.json` 两端均有。

真实页面显示“已找到最高 score 可行轨迹，停止后续批次；59 个候选未规划。”，卡片不含 error/warning，JS 和 console 均无错误。验证结束已 Reset 到确认基线。

验证命令：Ubuntu `python -m pytest -q -p no:cacheprovider` **225 passed**；Windows 同命令 **200 passed, 25 skipped**；`ruff check --no-cache src tests` 通过；`pnpm test` **115 passed**，`pnpm build` 通过（保留既有 Three.js chunk 提示）。独立审查另用 80 组乱序、同分及成功集合与全局选优参考比对，通过。最初全套有两项旧测试仍假设未排序的编号，已改为验证降序后的候选/错误映射，随后全套通过。

当前网页镜像：`tianji-jev-mujoco:0.2.0-guangyu`，ID `sha256:728b47daee4ea5cbccceb9e9cf3fdd7f272223e4474eecdd767b7a2c0b7487d6`，健康且完整命名。本轮只更新网页应用，cuRobo/GraspGen 容器及镜像不重启、不重建，SDK、vendor 和碰撞球未改。旧网页 manifest 已不存在，没有删除算法镜像。

## 2026-09-29：桌面间距核查

运行 `a2bd51ab21884a4f8d497772ce53f16e` 的 `build_scene` 记录 `table_collision_enabled=true`，共 4 层 ESDF，仅排除 `red_box`。`data/dev-mode-before-guangyu.json` 中的 run_id 与本次运行一致，桌面顶面为 600 mm，规划间距配置为 10 mm。

CPU 离线读取已归档的抓取端点关节角，使用同一 URDF、原始开爪球及 buffer，计算球到桌面 box 的解析距离。另将 MuJoCo mesh 顶点变换到世界坐标；最低顶点位于桌面 XY 范围内，因此可直接与桌面高度比较。没有 GPU 求解、物理 step 或规划行为修改。

| 对象 | 对桌面最小间距 |
|---|---:|
| 原始规划抓取端点的碰撞球 | 21.855 mm |
| 实际抓取端点关节角对应的开爪碰撞球 | 20.617 mm |
| 同一实际端点的 MuJoCo 碰撞 mesh | 20.706 mm |
| 同一实际端点的显示 mesh | 20.707 mm |

最近部位均为 `gripR_Link3_R`。这次抓取端点没有碰到桌面；上述端点不能代表整条轨迹。归档的完整 `positions_rad` 已缩减为 `{count, first:3}`，无法据此恢复全程最小间距。10 mm 配置约束原始规划路径的离散开爪球检查，不是 Hermite 路径、实际闭爪几何及物理跟踪全程的严格保证。

## 2026-09-29：JEV 从完整 skill 目录选择

经用户确认，取消选择前的 Python 契约筛选。每轮把全部注册 skills、`stop`、原始契约和 `SKILL.md` 交给 JEV，不提供 `eligible`/`unmet` 提示。JEV 作答后，只检查所选 skill 的执行条件；不满足时记录 `selected_skill` 和各条件的 expected/actual，结束本轮，不执行、不替换、不自动重试。任务/属性与执行臂判断、skill 契约、算法、规划和执行边界保持原样。新增 skill 的目录发现自动生效，实际算法仍须有对应执行实现。

对已捕获的蓝色圆柱运行 `7d06b7dbd4b14da096c1b3948851be6e` 和绿色物体运行 `6aa6e1da0513426abe6117eaef9bb7b6` 的 18 个 skill 决策状态各回放两轮。每个状态分别使用历史 questions 和新的完整 skill questions，均实际调用 JEV；不执行 skill 或 GPU 算法。答案发回后才离线按契约评分，两个准备步骤同时可执行时均算正确。

| 调度方式 | 合法选择 | 错误 stop / 拒绝 / 请求错误 | 平均调用耗时 | 输入 / 输出 tokens |
|---|---:|---:|---:|---:|
| 历史预筛选 | 36/36 | 0 / 0 / 0 | 0.383 s | 284622 / 1270 |
| 完整 skills | 36/36 | 0 / 0 / 0 | 0.421 s | 295430 / 3694 |

完整报告为 `data/skill-decision-replay.json`，含每次实际 state/questions、模型输出及离线评分。`scripts/replay_skill_decisions.py` 可接收一个或多个运行 JSONL 或状态快照；`--repeats` 默认 2，`--output` 指定报告位置。样本只覆盖这两轮成功任务的状态，不能据此断言任意场景的成功率或性能。

同步后用用户确认场景执行 `pick up the green one`，运行 `77ddb57a82b649a2a1331c30ec868f4c`：JEV 选择目标 ID `black_cube`（其实际颜色已被用户设为 green）和 B 臂。9 次 skill 选择均收到 9 个 skills 加 stop，共 10 个选项；共 13 次 JEV 调用、9 个 skill，最终三项抓取检查均通过。总用时 39.6476 s，JEV 累计 5.8889 s。完整记录为 `data/all-skills-live-validation.json`。

验证：Windows 全套 **211 passed, 51 skipped**（跳过 Linux 启动器等平台测试）；Ubuntu 受影响决策/runner 测试 **25 passed**；`ruff check src tests scripts` 与前端 JS 语法检查通过。回归覆盖全目录、终态仍可出题、注册新增 skill、两种准备顺序、提前执行拒绝、重复 skill 拒绝、未知 skill、stop 和回放证据保持。独立只读审查对 512 组 facts × 9 skills 的 4608 次检查中，新执行校验与原契约准入结果完全一致。

Windows/Ubuntu 源码同步，网页后端自动重载；恢复确认的物体布局、颜色和尺寸后实测。网页与两个算法容器的镜像 ID、启动时间均未变化，没有构建镜像、重启容器或修改算法 SDK、碰撞球及 skill 文件。


## 2026-09-29：规划无解后交回 JEV，保留用户 Lift 设置

用户批准新增失败证据与重选循环，并明确保留高级设置中的末端坐标系 Z。部署时最新保存值为 `lift_axis=z`、`lift_offset_m=-0.1`、`lift_in_tool_frame=true`，revision `1188f72ea6ff234d`；本次没有写规划设置。

实现：

- 每轮 skill 选择仍提供全部 9 个 skills + stop；选臂始终提供 A/B/none，先由 JEV 选择再校验。
- 决策 state 增加桌面、冻结规划设置、本轮 planning_attempts 和完整双臂候选证据。候选增加 Approach/Lift 在 world frame 下的带符号位移，按各臂 TCP 旋转计算；不改变实际规划目标或参数。
- 只有运动前的 planning_no_valid_candidate 进入重选循环。记录失败，清除当前 arm/plan 的有效状态，保留场景、GraspGen 和 IK 产物，交回 JEV。每臂本轮最多规划一次，不由 Python 自动切换执行臂。
- 网页新增“系统事件：规划无解，交回 JEV”。契约拒绝、服务错误、超时、执行错误保持终止处理。
- choose_arm 和 plan_grasp 的 SKILL.md 更新上述已批准语义；技能数量、三阶段边界、算法实现、碰撞球和算法镜像不变。

测试与审核：

- 新恢复测试先观察到 5 个预期失败，再实现并通过；覆盖换臂、JEV 主动 stop、重复臂拒绝、两臂无解、不可达臂、致命服务/过期输入错误，以及重新决策期间 Reset。
- 正、负 100 mm 的 tool Lift 参数都覆盖，断言跨换臂保持原值。
- Windows 完整测试：`python -m pytest -q` → **225 passed, 51 skipped**（20.24 s）；skip 为平台相关检查。
- Ubuntu 受影响测试：`tests/test_decisions.py tests/test_runner.py tests/test_runner_failures.py tests/test_planning_recovery.py tests/test_planning_settings.py` → **60 passed**（1.31 s）。
- `ruff check src tests scripts` → 通过。
- 前端 `pnpm test` → **121 passed**；`pnpm build` → 通过，保留已有 Three.js 分块体积提示。
- 独立代码复核无阻塞问题；补充了负 Lift 偏移回归测试。

真实 JEV 回放（没有调用 GPU 或运动）：

1. 旧失败运行 f1a69d52f3fd4f30aac5193c93335bff 的 B 无解状态，原 tool +Z 参数不变：两次选择 `choose_arm → none`。
2. 明确构造的“两臂均失败”“两臂均无 IK 候选”测试状态：各两次选择 `stop / none`。
3. 新运行的真实候选证据上，注入明确标记为合成测试的 A 无解结果，剩余 B 为 1 个 IK 候选：两次 `choose_arm → none`。
4. 同一真实候选证据上，注入合成 B 无解结果，剩余 A 为 91 个 IK 候选：两次 `choose_arm → A`。

这些回放证明调用中有失败证据、JEV 可以重选也可以停止；不能据此推断它停止的具体理由，也不能保证存在 IK 候选时它一定继续。合成故障不属于产品运行结果。

部署后完整仿真：

- run_id：`2fafc837875d44baad24313c7162ed11`
- 指令：`pick up the blue object`
- Lift：用户当前的 tool Z −100 mm，始终未改。
- JEV 选择 A 臂，首次规划成功，无需恢复分支。
- 抓取成功，gripper_holding / object_moved / transform_consistent 全部 true。
- 总耗时 **38.02 s**，JEV **13 次**、累计 **7.30 s**；输入 **197495 tokens**、输出 **1266 tokens**。本次记录不作为跨场景性能提升证明。
- 所有 skill 选择均有 10 个选项，均包含冻结参数和桌面信息。

备份位于 Ubuntu `/tmp/jev-recovery-backup-mgmxb2dj`。13 个修改文件已同步并校验 SHA256；源码热重载后恢复了已确认场景。三台容器的镜像 ID 和 StartedAt 均未变化，没有重建镜像或重启容器。

原始证据：`data/recovery-live-validation.json`、`data/recovery-jev-replay.json`、`data/recovery-alternative-arm-replay.json`、`data/recovery-alternative-A-replay.json`、`data/recovery-deployment.json`。

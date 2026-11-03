# 抓取 batch 与判断范围审计（2026-11-03）

本轮基线为 timing-72。此前 P71/P72 均完整成功；本报告只描述这次新增修改，验收结果另列。

当前安装的官方 cuRobo 全部 Python 文件通过项目的源文件哈希清单核验，commit 为 `8e734f3ced1df898990bcd92de40abce475907db`。实际环境 PyTorch 2.11.0+cu128、RTX PRO 6000 Blackwell。没有改 NVIDIA SDK、vendor 或其他项目。

实际读取的对应版本官方文件：

- [BatchMotionPlanner](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/motion/motion_planner_batch.py)：142–244 行为批量 pose 求解，314–496 行为 grasp 的 goalset、approach、grasp、lift。
- [MotionPlannerCfg](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/motion/motion_planner_cfg.py)：公开配置和初始化。
- [官方示例](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/examples/getting_started/motion_planning.py)：持久求解器及 warmup 使用方式。

`plan_grasp` 的每阶段调用默认一次 `plan_pose`，共享场景下还使用 PRM seeds；公开 grasp 接口不接受内部阶段 attempts。更改项目普通 pose 的 attempts 不会改变这些内部调用。当前 CUDA Graph 已启用，IK seeds=16、TrajOpt seeds=4、自碰撞和 5mm 硬净距均保留。

P72 黑件 `plan_grasp` 技能耗时 22.798 秒，其中原生公开 grasp 求解 13.069 秒，服务操作 13.199 秒；排队、解码、场景上传约 0.109 秒。技能总耗时还包含预筛和最终命令检查。各阶段为包含关系，不相加重复计算。

隔离进程重建同一归档 CAD/OBB/cuboid 场景，不更改生产地图、不执行运动。CPU 限制仍为 cuRobo 容器 1 核。热态 cProfile 中 Event.synchronize 等待约 11.75 秒，说明时间主要消耗在 GPU 工作完成等待，不能当成纯 Python CPU 计算。PRM 节点去重另有约 1.75 秒；没有因此删除 PRM 或碰撞检查。

同一黑件、同一抓姿排序、同一地图和配置，交替比较固定容量：

| 容量 | 两次热态公开 grasp 墙钟 | 返回成功 |
|---|---|---|
| 4 | 13.299 / 13.212 秒 | 每批 4/4 |
| 2 | 6.956 / 6.634 秒 | 每批 2/2 |

这是“先取得前两个候选结果”的延迟比较，不是宣称两个候选等价于四个候选的总吞吐或成功率。任务通常在首批找到合格路径后停止，故采用 batch=2；首批未通过则继续原有候选池，未截断候选或降低验证。双方臂固定容量2在启动时预热，较大公开请求仍支持4/16及既有容量。所有 seed、阈值、接触策略和后置验证保持。

Responses prompt 修正了持物阶段的冲突说明：几何和区域在抓取前建立，持物时的 `resolve_placement(capture_mode=fresh)` 只为候选判断取得当前图片，不建立 `observed=true`，不能修复失效地图。router 每次只问一个明确对象、时间范围的条件，不重复追问刚成功 skill 已证明的后置条件；独立用户要求和最终核验保留。

支撑关系 question 不再暗示“unknown 就应换视角”。它仅判断支撑关系；另由独立可见性问题诊断遮挡、裁切或不可见，graph 决定恢复动作。概率门槛仍为0.8/0.2，明确否定、信息不足和服务错误不伪装为成功。

P72 的六次完整 observe_scene 中，两次在释放后、回位前用于重建已失效地图；另两次满足用户交付后回位再观测的要求。当前版本没有把预测释放位姿当作新测量，也没有删掉缺少替代证据的必要地图更新。

定向测试196项通过；新增双臂容量、预热及候选不丢失测试先复现失败再通过。

P73（timing-73，`f299e8c1cd3146629cd2b7d4d0d9c849`）在运动前停止，原始结果保留 needs_attention。两次 CAD 匹配 confidence=1.0，身份判断通过；目的地选对 ID3 但 confidence=0.60，未达到原0.8门槛。加工任务的目的地问题仍混合输入来源、谱系、最终目的地和历史图像方位说明。

timing-74 将这个 question 限定为“输入对应加工产物的最终目的地”，明确 selector 不是目的地、较大世界Y是左侧、较小世界Y是右侧，用户明确指定的视角优先，历史图像描述不能替代该约定。不改选项、候选数据或接受门槛。相同冻结图像、其他问题不变，正确 ID3 confidence 从0.60到0.98，1.508秒。反向对照仅把原指令最终右盒改成左盒、保留错误的 graph 提案，模型改选 ID2、confidence=0.69；不会把原提案右盒强制通过。反向对照仍未达确认门槛，不宣称通过。

冻结规划和问题对照用于定位原因；完整物理任务验收见下文，不将这些局部对照当作任务成功。

P74（timing-74，`e792808e45714fb0bed3a4ab29dda5b4`）初始 Responses 把40个 subgraph body 节点及39条内部边重复放进 root 数组。严格拓扑检查报告 unreachable，未执行机器人动作，也未调用 Decisions。timing-75 只补充 schema 字段和 prompt 的 scope 所有权说明：全局唯一的是 ID，root 数组不是全局节点索引；子图主体只在其本地数组出现。没有自动删节点、补连线或重试 Responses。

timing-73 完整回归4505 passed、58 skipped、1 warning，500.75秒；timing-74 最后修改后的目的地／graph／可见性相关89项通过。

P75（timing-75，`4b3bc841b58d40e89bca3d29873fd940`）拓扑合法，目的地 confidence=0.99，第一批两个抓姿均成功。但模型把粉色输入本身填成 source_ref；旧 source_identity 将“来源”混同输入材料而回答yes，执行前严格支撑校验拒绝，无机器人运动。timing-76 在 TaskSpec 字段、prompt 和来源 question 中明确：source_ref 是输入原先所在的支撑／容器／来源区域，不能是输入自身。相同冻结图像的新来源问题对这一错误提案给出 probability=0.0；其他问题仍通过，1.676秒。未把无证据支撑改成确认。相关68项测试通过。

timing-75 的 graph/schema/任务绑定定向95项通过；timing-76 最后的来源问题与 schema 说明修改后定向68项通过。完整4505项回归对应 timing-73，后续文字与 schema 描述修改通过上述定向验证；不将早先的完整回归标成最后修改后的重跑结果。

P76（timing-76，`f382547e0c8c4ef8b0dba505cffa99bb`）完整成功。总286.791秒，初始 Responses 90.407秒，其余196.384秒。24次 Decisions，共33.730秒，中位1.353秒、最慢2.194秒；没有失败/unknown graph node，没有算法或技能错误。来源关系 probability=0.98，最终目的地 confidence=0.93。两次释放与最终任务核验均通过；两次交付后均由显式 graph 回观测位姿并观测。

P76 的 plan_grasp 技能：粉色7.810秒、黑件11.339秒。第一批均为2个候选，全部成功，停止后仍保留未求解候选记录；对应批量调用墙钟5.029秒、8.962秒。与P72技能14.966/22.798秒相比明显减少，但这两个独立物理任务的模型图、观测和产物位置存在变化，严格固定输入的速度结论以上方隔离对照为准。

P76 observe_scene 共6次，5.399–11.973秒；build_scene 共4次，1.190–2.650秒；纯视觉 verify_place 为4.852/5.127秒，其中 Decisions 为1.755/1.268秒，剩余为采集和反馈核验等技能工作。Decisions 耗时已包含在技能墙钟中，不再相加。

“无失败节点”不表示所有输出概率都等于1：P76 的操作提议（执行臂、XY）保留原始 confidence；部分清单可见性辅助输出也较低。事实、操作提议和历史已确认身份按各自原契约处理，没有将 unknown 改成 yes 或降低视觉事实门槛。这里只声明实际任务结果，不声称所有场景都能零失败。

同一 timing-76 版本随后运行P77（`9791a5a15f7743b3bfcb7d310d8f58b3`），也完整成功。每次均重新由初始 Responses 生成 graph，未复用成功图；两次之间没有代码修改。

| 完整物理任务 | 总耗时 | 初始 Responses | 扣除 Responses | Decisions | 失败/unknown graph node |
|---|---:|---:|---:|---:|---:|
| P76 | 286.791秒 | 90.407秒 | 196.384秒 | 24次 | 0 |
| P77 | 271.811秒 | 67.598秒 | 204.213秒 | 24次 | 0 |

P77 Decisions 共33.387秒，中位1.302秒、最慢2.033秒。plan_grasp 技能粉色11.969秒、黑件16.349秒；对应首批调用5.108/9.442秒，均2/2成功，没有后续批次或失败候选。未隐藏第二轮的耗时波动。observe_scene 仍为6次、5.669–11.959秒，build_scene 为4次、1.036–2.900秒，verify_place 为5.431/5.306秒。两次均无几何/身份/算法错误，最终任务 probability=1.0，并核验每次交付后观测与最终双臂观测位姿。

验收汇总保存在 `data/performance-unpack-20261101/timing-76-acceptance.json`；每轮的 `unpack-P76/P77-audit/evidence-index.json` 记录原始模型调用、技能事件与下载文件哈希，`*-timing-summary.json` 保留分阶段耗时。P77 的两次释放问题原始图像归档与请求哈希验证通过。失败的P73–P75仍原样保留，未改写成成功。

最终运行时文件已同步 Ubuntu，cuRobo 锁定源码身份再次核验为上述 commit；文档补记单独同步，不重启机器人服务，保留P77完成场景。最后两轮满足120–240秒（不含初始 Responses）的性能目标；两轮样本不构成所有任务成功率的统计保证。

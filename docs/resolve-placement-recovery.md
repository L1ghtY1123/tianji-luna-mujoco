# resolve_placement 的首次失败与重复请求

2026-10-30，black-B4 归档显示正常图中的 `resolve_placement(preference=decide)` 首次失败后，没有进入通用无进展记录。恢复随后分别选择 `decide` 和 `clearance`；三次实际绑定的 `placement_request.placement_preference` 都是 `clearance`，场景、区域和设置相同，原有 96 个 XY 候选和全部高度样本也相同。

此次修复在 `placement_no_candidate` 的错误详情中保存实际解析后的 `effective_arguments`。图执行器在第一次正常失败时即记录该请求，并记录默认参数、`decide` 和实际偏好的等价形式。已有恢复选项过滤与派发检查共用原 `operation_signature`，因此改写参数名称不能冒充新尝试。临时服务错误不记为候选已穷尽；具体恢复能力仍由 Decisions 选择，代码不插入补观察、换偏好或运动。

证据失效边界保持原样：失败节点与原因、当前图像摘要、facts、测量实体集合、地图 coverage 和已完成输入集合。这个补丁没有增加数值相等容差，也没有把采集 ID、计时或小幅反馈变化包装成新证据。显式改用另一偏好，或上述证据实际变化，仍可参与恢复选择；偏好不同本身不保证生成不同落点。`plan_place` 另有基于候选、场景、持物、关节和设置的数值无进展检查，未在此处改动。

候选搜索在同一已生成的有限网格内进行：先检查原有最多 96 个空间分布点，仅在没有任何合格数值端点时检查余点。原网格上限仍是 25×25 加中心；首批有结果则不追加检查。`selection_evidence` 报告 `primary_candidate_count`、`remaining_candidate_count`、`sampled_candidate_count`、`unsearched_candidate_count` 和 `remaining_grid_searched`。区域、姿态、阈值、后续语义筛选与 `plan_place` 合同保持不变，有限网格失败不证明整个目的地不可能。

定向 CPU 回归覆盖：首次失败在模型选择前屏蔽同一请求、实际绑定为 reach 时不错误推断为 clearance、保留另一偏好、派发时拒绝别名、临时服务故障不屏蔽，以及既有图像/facts/实体/地图/已完成输入变化重新开放选择。black-B4 原始输入存在约 1.34–2.54 微弧度夹爪反馈变化，不能称为所有数值输入逐字节相同；本修复不扩大既有 1e-6 rad 合同。归档比较与 CPU 回归不等于真实机器人完整任务通过。

## 反馈变化后的候选重验

unknown-C5（`e1e9947daeaf4604b46824a060f4bb02`）前三次解析与规划之间，唯一超过原一致性门的输入是 A 夹爪反馈，差值分别约 +469.535、-2.958、-509.374 微弧度。原输入摘要、结果签名、区域与候选内容都有效；前三次解析耗时合计 82.558 秒。前三轮全部候选 XY/旋转相同，第三轮高度有 1mm 差异，因此旧 Z 与 held IK 证明不能作为当前证明。第四轮实际偏好由 clearance 改为 reach，必须按新偏好重新搜索，不能因为输出位置碰巧相同就忽略该变化。

正常 `plan_place` 失败会保留原 `placement_resolution`，但清除旧可执行计划。下一次模型显式选择 `resolve_placement` 时，controller 在原失效清理前临时取出该结果；不建立独立长期缓存，也不改变 graph 的选项或动作顺序。数值层核对原结果摘要、原反馈绑定、全部非反馈输入、完整语义区域和原接受候选 ID；只有当前反馈超过原 1e-6 rad 一致性门且其他条件严格相同，才将至多 12 个请求作为搜索种子。当前有限合法网格中必须有相同 request 与 ID；观测、场景、几何、持物 attachment、设置、目的地、要求或实际偏好变化均取消优先。

仅复用 XY/request；完整抓取旋转仍来自当前且绑定相同的 holding。每个种子重新进入现有 `_resolve_place_goal(workspace_prefilter=True)`：按当前夹爪反馈生成工具球，走 `/tools/resolve_release_height` 的世界碰撞、工作空间和公开 `/tools/solve_ik_batch`，重新核对目标与 achieved FK 的完整 CAD/OBB 底面间隙、容器入口和所需配置。旧 payload、Z、height proof 与候选语义回答不参与新证明。至少一个种子通过则进入原语义筛选；全部数值失败后继续原 primary/remaining 有限网格，排除本轮已经检过的种子。服务错误仍立即返回，不冒充候选失败。

新结果绑定当前反馈和新证明，重新生成 input_digest/resolution_id；原 `plan_place` 一致性门、运动/FK/碰撞和释放高度阈值不变。`selection_evidence.tested_candidate_indices` 保存实际检查顺序，`candidate_revalidation` 保存原 resolution_id、种子检查/接受数、是否回退和重验耗时。剩余网格是否搜索按真实索引判断，不能以检查次数推断。

此范围有意保守：C5 第三次 resolve 开始时夹爪已回摆至第二次解析原反馈的 0.502 微弧度以内，因此不会命中该优先路径；第四次 reach 也不会命中。没有扩展反馈阈值或发明稳定判据。

2026-10-31 已完成原 C5 第二轮冻结输入的真实 GPU 对照。两组分别在独立进程中正式 warm_start，保持五层世界、当前双臂反馈、持物、settings 与源文件一致。原搜索测试96个XY、77次高度调用，数值算法36.406秒；候选重验检查12个XY、12次高度调用，9.453秒，各返回12个带当前证明的端点。分别检查2002/312个世界高度样本、1946/312个持物IK样本；每个世界样本仍检查完整332球。启动预热分别44.379/43.990秒，未计入上述算法时间。原1e-6反馈门、5mm/0.05rad IK门和50mm最大底面间隙保持。

独立复核逐值核对89次调用与24个返回候选证明、完整OBB八角点及FK底面间隙。placement-32两组分别选择上移6mm/5mm，重验组FK位置误差4.967305mm，接近但未超过原5mm门；两次数值结果不是逐位一致。此单场景各一次对照没有语义筛选、轨迹规划或机器人运动，不能推导稳定倍率或完整放置成功。重验组如实保留429个未搜索候选。文件前缀为 `data/cad-obb-iteration-20261030/unknown-C5-resolution-audit/hybrid-15-c5-revalidation-v1-`。

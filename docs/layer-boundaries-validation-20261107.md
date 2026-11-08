# 分层与数据边界验证（2026-11-07）

本次检查沿用在线 Graph、Memory、Decisions 和执行层，未更改任务流程、视觉阈值、抓取参数或官方 SDK。

## 已复现的问题与修正

1. Graph 初始化绑定采用浅拷贝，调用者修改嵌套列表能够改变执行器状态；`GraphSkillResult.payload()` 和子图任务上下文回调同样暴露嵌套引用。均已先复现再改为独立副本。
2. Router 的执行适配器通过 `self.graph_executor.last_result/last_failure` 反读 Graph 内部状态，返回值还是未校验的普通字典。现使用 `GraphDecisionRequest/GraphDecisionResult`，显式传递原失败、绑定和分支，不增加模型请求或改变分支选择规则。
3. 技能 context 携带 `SpatialMemory` 活对象，释放核验函数接受任务账本服务。现服务仅由执行层持有，纯辅助函数接收 `CollisionSnapshot/ReleaseLedgerSnapshot`；每次检查取得当前版本，不缓存旧快照充当新场景。
4. 加工产物证据函数直接调用数据库分配 ID。现返回 `ProductRegistration`，由执行适配器提交原有原子事务，仍先同步已提交物理事实，再更新任务账本。
5. 数据契约未拒绝几何记录内部 ID 与字典 key 不同、退役回执与物体 ID 不同、场景发布排除模式与请求不同。现均在接受之前拒绝，未改变观测阈值或 ID 关联算法。

## 验证范围

一次性验证保存在 `data/runs/layer-boundaries-20261107`，不进入正式运行入口。

- `boundary-probe.py`：修改前复现绑定引用泄漏；修改后验证请求/结果类型、原始失败传递、接收方修改不污染 Graph、拒绝未类型化判断。
- `memory-boundary-probe.py`：修改前复现错配 ID 被接受；修改后检查 ID、发布来源、快照隔离、时间记忆与空间记忆独立以及失效地图不能授权回位。
- `release-boundary-probe.py`：使用相同完整回执比较修改前后释放绑定与哈希，结果完全一致；新快照不泄漏任务账本，并仍拒绝已完成转移重播。
- `product-persistence-probe.py`：使用归档视觉核验结果，注入输入/产物事件之间的数据库异常；事务回滚、幂等 ID、谱系、任务失败不复活输入和后续历史记录均验证。
- `container-release-probe.py` 与恢复路由验证：两臂 10 mm 释放几何、固定 XY、单段/三段规划选择、显式恢复及次数保持。
- `layer-audit.py`：检查核心 Graph、Memory、Responses、Decisions 和算法入口的传递性导入；核对所有受影响函数的显式参数，禁止 context 中出现记忆服务以及回调反读 Graph 内部字段。
- 既有物理动作记忆时序验证：抬升/开爪回执在后续失败前写入，数据库写失败不替换内存副本；释放核验重做不生成新转移。

以上是边界与组件验证，不等同于完整机器人任务成功率。全量 `src` Ruff F 检查、修改文件 Ruff 检查和 Python 编译检查通过。

## 在线仿真与未解决的问题

实验 `5613e38cff47448cbceace742ba9cc89`，一件粉色物体，耗时 479.9 秒，结果 `needs_attention`。独立仿真验收为 0/1 黑件进入右盒，不能认定完整任务成功。

- Responses 在开局调用两次：初稿缺少两个身份节点 unknown 的观测恢复路径，经一次修订后通过；正式执行中没有调用 Responses。
- 完成粉色物体抓取、桌面放置、放置核验、拆包、产物核验和登记。时间记忆提交输入消耗及独立产物，场景先后发布 8 次；新的释放快照、加工登记和场景接口走过实际入口。
- 实际进入了目标容器绑定与后续处理的有限恢复路径；恢复选择经过新的类型化 Graph 判断接口。没有通过修改未知结果或模型阈值使流程通过。
- 最终黑件 `plan_grasp` 返回 `grasp_lift_geometry_invalid`，原因为 `grasp_lift_requires_current_support_evidence`。支撑证据仍对应拆包核验采集 `rgbd-2-9262200af56b43afb15e83f3adf70cbc`，当前规划采集已变为 `rgbd-2-97f76e7b4c5243ff864974b7f724744c`。
- 事件 320 的 `identify_target` 使用当时仍有效的支撑证据；随后恢复路径重新观测，但未再次调用 `identify_target` 刷新支撑关系。这暴露了恢复路径、技能前提及证据失效之间尚未闭合的合同问题。不能把历史支撑改写成当前测量，也不能把这次失败归因于 Decisions 置信度。
- 该支撑时效问题本轮未修改；核心实现导入独立、快照隔离通过，并不代表所有跨层业务时序已经正确。后续应统一恢复后目标身份绑定与支撑证据有效性的合同，再验证在线编排与完整任务。

原始状态及独立验收分别为 `data/runs/layer-boundaries-20261107/run-01-state.json` 和 `run-01-qa.json`。子图任务上下文的副本修正在本次实验之后完成组件验证，未声称此次在线实验覆盖该后续修正。

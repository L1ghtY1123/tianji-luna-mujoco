# Decisions 恢复上下文的精确去重

恢复输入的可逆压缩与无进展校验是不同边界。2026-10-30，observe_scene 的签名不再因无关建图/规划 facts 改变而重新开放同输入观察；相关相机、身份需求、动作与实体生命周期变化仍有效，其他技能保留原边界，详见 [观测恢复证据](observe-recovery-evidence.md)。这不改变 Decisions 的 question、成功阈值或恢复选择权。

2026-10-30：只对 `graph_recovery` 和 `graph_recovery_continuation` 的 Decisions context 做可逆 JSON 去重。保留第一个完整值；完全相同的后续 dict/list 用同一 context 内的 JSON Pointer 和 SHA256 引用。明确的引用规则与完整展开哈希随请求归档。每次发送前验证展开后的 JSON 与原始 context 相同，包括键顺序、数值表示、中文和自然语言；`1` 与 `1.0` 不合并。

转换入口是 `PolicyController._judge` 合并上下文后的轻量 hook。原 runtime ledger、恢复历史、graph、技能选择及参数不变。questions、当前与参考 RGB、永久 ID、完整失败原因、事实状态和返回概率不删改。非恢复判断保持原输入。可选 `candidate_projection` 仍以完整值提供给图像构建器；用户原数据出现引用保留字段时不转换，避免误解原始字典。

归档可用 `expand_recovery_context()` 完整还原。它验证每个引用内容哈希与最终展开哈希；错误路径、循环或内容变化会被拒绝。引用指向当前请求内保留的完整值，不要求模型访问外部文件，也不把哈希当作事实摘要。没有按含义删去信息，也没有把原先 `public_summary` 已截断的几何数组冒充完整数据。

## B3 冻结请求证据

实验 `c5901daff90844dfa9a579b1192eceee`，event 388，call `0145a9a4228244c980400489315de456` 的原 API 账单为 76,294 input tokens、0 output tokens、1.624 秒。下面的字段计数使用本地 `o200k_base` 比较文本体量；不是对 GPT-6 图像或协议 token 的精确计费分摊。

| 字段 | 比较 token 数 |
|---|---:|
| 主 JSON 文本 | 73,367 |
| `context.recovery_history` | 52,463 |
| `context.ledger` | 15,745 |
| `context.current_measurements` | 1,066 |
| `context.available_capabilities` | 719 |
| `context.failed_result` | 337 |
| `context.facts` | 310 |

`recovery_history[0].result` 与 `[1].previous_result` 完全相同，各 26,199 tokens；其中三份 placement candidate 的执行数据合计 20,585 tokens。每个 candidate 的 carrying binding、planning settings 又在顶层、payload 和候选之间重复。ledger 的 `current` 重复 `objects[0]`（4,422 tokens），`entities`、`initial_entities`、`scene_entities` 在此冻结输入中相同（各 1,901 tokens）；只有实际字节相同时才合并，后续发生变化的历史/当前实体不会合并。

生产转换函数在该冻结输入生成 25 个引用，主文本从 73,367 降至 26,455 比较 tokens（减少约 64%）；展开 JSON 完全一致。当地单线程五次耗时中位数为 16.5ms。所有 7 张图及原 questions 保持不变。原三次恢复调用分别用了 23,765、49,170、76,294 API input tokens，耗时 1.602、1.731、1.624 秒；因此现有证据证明输入重复，不能宣称去重已改善模型服务时延或任务成功率。

初次失败与恢复后新失败仍分别保留。B3 的 `place_no_valid_candidate`、三个候选的 `CUROBO_GRASP_FAILED/approach/Goalset planning returned None.`、`whole_destination_infeasible=false`、resolution ID、已夹持/已 attachment/未释放、地图与当前采集 ID、重试前提和未获许可的参数都保留。一次模型返回仍独立决定恢复；不会重复采样答案挑高置信度。

离线源请求、字段计数、往返证明及转换后的请求位于 `data/cad-obb-iteration-20261030/black-B3-recovery-context-audit/`。冻结测试为 `tests/fixtures/black_b3_recovery_request.json`。真实 Decisions 单次对照由主线程另行执行；本实现验证未调用模型或机器人。

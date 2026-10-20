# RGB 清单与掩码绑定验证（2026-10-19）

本次视觉验证最初在 `tianji-jev-mujoco-requirements` 工作树进行。随后按用户 2026-10-19 的明确要求，代码、文档和证据已合并到 Windows `C:/project/tianji-jev-mujoco` 与 Ubuntu `/home/guangyu/projects/tianji-jev-mujoco`，网页服务已加载主目录，额外工作树已移除。下文保留当时的原始验证结果和部署状态；当前目录与部署状态以[单目录合并记录](project-consolidation-2026-10-19.md)为准。合并不代表整轮搬运验收通过。

## 实际流程

1. 外层 JEV 选择 `initialize_inventory`。初始头部 RGB 与完整指令交给官方 `/v1/responses`、`gpt-6-luna`，以严格 JSON Schema 返回实体列表：临时引用、类别、颜色、外观描述、归一化 RGB 框、父容器、SAM3 英文描述和搬运成员状态。程序分配固定全局 ID；搬运清单引用同一 ID。
2. SAM3 收到原 RGB 和实体描述，不接收全局 ID。相同描述只查询一次，但可返回多个独立掩码。
3. `/v1/decisions` 收到初始带 ID 的 RGB 参考、当前各相机原 RGB、临时编号框及原 RGB 裁剪，以及固定清单、阶段和目的区域 ID。候选框由 SAM3 掩码外接矩形产生；不发送掩码轮廓、深度图、点云、世界位置。分区仅发送本地测量投影后的二维标注。
4. Decisions 的封闭问题为 `identity_<candidate>`、`visibility_<ID>`、`progress_<ID>`。分别选择固定 ID／拒绝／未知，可见／未见／未知，以及在目的地／在别处／未知。初始不再按 SAM3 候选数判断清单数量或分配新 ID。
5. 应用程序根据关联结果，把掩码和测量点云归到固定 ID；同一实体可有多相机掩码，重复像素只计一次。缺少 SAM3 或有效深度不删除清单，记录缺少当前几何并阻止不完整运动地图发布。最终放后图像判断可独立于 SAM3 候选数，但仍须已有实际释放反馈。

例：SAM3 对“红色方块”返回 `candidate-0`、`candidate-1`。Decisions 可返回 `candidate-0→实体A`、`candidate-1→实体B`，程序据此分别构建 A/B 几何。下一视角的顺序可以相反。**关联结果由应用程序使用，SAM3 不认识 A/B。** 若外观与上下文不足以分辨交换后的同类实体，返回未知，不保证仅凭 RGB 总能恢复身份。

放置分区 XY 是 TCP 锚点，视觉判断不要求物体中心或 attachment 全部投影位于槽位内。精确 TCP、Z 间隙、姿态和碰撞仍由本地规划及执行校验。

## 真实调用记录

输入为此前保存的 MuJoCo 传感器观测，未读取编辑器物体真值。文件现存于主仓库 `.superpowers/sdd/2026-10-19-rgb-inventory/results/`；Ubuntu 原始探针副本现归档于主目录 `data/consolidation/2026-10-19/vision-probe-20261019/results/`。

| 调用 | 实际输入 | 实际输出 |
|---|---|---|
| Responses `dbef9508…` | 初始头部 RGB 1 张＋测试指令；无 SAM3/深度 | HTTP 200，10.239 s；桌子、蓝盒、绿圆柱、红色块、黑色障碍、黄色区域共 6 项。输入 763、输出 809、总计 1572 tokens |
| SAM3 | Responses 返回的 `blue open box`、`green cylinder`、`red block`、`black block`，同一初始 RGB | 每条描述返回 1 个候选；保存各自框图，位置与初始 RGB 实体一致 |
| Decisions `d77f6fa6…` | 已保存的真实三相机观测及候选：23 张原图／标注／裁剪图，16 个问题 | HTTP 429，1.579 s；`type=tokens`、`code=rate_limit_exceeded`。没有身份或完成状态回答 |
| Decisions 独立复测，同一请求 | 相同 23 张图、16 个问题；使用与运行服务相同的 host 网络 | HTTP 429，1.605 s；限额 100000、已用 95161、剩余 4839、本次需要 8366 tokens。`Retry-After=91420` 秒（约 25 小时 24 分），没有模型回答 |

Decisions 测试候选来自已保存的真实分割记录，不能冒充本次新 SAM3 全链观测。新 SAM3 描述另外在初始头部图上真实验证。模型限流后未继续发起搬运后判断；上表复测为一次独立诊断，运行代码没有自动重试，没有切换生产。HTTP 200 的模型列表查询也不能替代任一推理接口的实际验收。

复测的限流原始字段保存在 `.superpowers/sdd/2026-10-19-rgb-inventory/results-recheck-host/rate-evidence.json`。此前一次临时容器采用默认 bridge 网络，发生传输失败、未收到 HTTP 响应，不能计作 Decisions 推理结果；改用运行服务的 host 网络后得到上述明确限额。限流字段表示当时可用请求额度，不代表账户余额，也不能保证到时视觉结果一定正确。

![初始 RGB 清单标注](../.superpowers/sdd/2026-10-19-rgb-inventory/results/initial-inventory.png)

## 验证范围

- 新增测试覆盖 RGB 独立清单、相同类别的不同 ID、零 SAM3 候选仍可判断去向、缺几何保留记录、无世界坐标/轮廓输入、初始语义框标注、不复用历史框冒充当前框。
- 独立审查发现并修复：成功的初始清单须跨后续 SAM3/Decisions 失败保留，并保存已有语义框标注；下一观测复用 ID，使用三相机。初始 RGB 框编号复用已有分行及从左到右规则，避免改变放置槽位分配。两项均有失败再通过的回归测试。
- 全量测试结果以执行日志为准，最终结果见本页末尾。跳过测试不代表 GPU 或真实搬运验收通过。
- 没有修改官方 SDK、模型权重、依赖安装目录、cuRobo batched IK/轨迹选择、抓放执行时间和已有性能修正。

尚需解决 Decisions 的 token 限流，并完成当前 RGB 输入的多视角／跨时刻真实关联和整轮搬运验收。用户后续已明确授权直接合并到唯一主目录，代码与运行入口已统一；这一目录决策没有消除模型限流，也不替代上述验收。

## 最终本地验证与同步

- Python 完整测试：**1676 passed, 54 skipped，89.55 s**；日志为 `.superpowers/sdd/2026-10-19-rgb-inventory/pytest-delivery.log`。
- 前端：**271 passed，0 failed**；日志为同目录的 `web-tests.log`。
- 修改的生产 Python 文件与新测试通过 Ruff；相关路径 `git diff --check` 通过。
- 独立代码审查的两项重要问题已修复并完成回归。没有把模拟模型回答的测试当作真实关联准确率。
- 70 个候选代码、测试、文档及证据文件已同步 Ubuntu，并逐文件校验 SHA256；已有核心源码同步前先核对修改前基线，旧文件备份在远端候选目录 `.superpowers/sdd/2026-10-19-rgb-inventory/remote-baseline/`。最终报告和复测证据另行同步。
- Ubuntu 现有运行镜像没有安装 pytest，因此容器内的补充 pytest 命令未能启动；没有为此修改运行镜像或安装依赖。以上全量测试数字来自 Windows 候选目录。真实视觉请求则在 Ubuntu 临时容器执行。
- 没有部署、重启运行服务或执行本次整轮搬运。当前运行服务仍是此前版本。

## 接口依据

- [GPT‑6 Luna 模型说明](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Decisions](https://developers.openai.com/api/docs/guides/decisions)

接口有效、结构化输出有效，不代表模型视觉结论必然正确。本次使用已有 HTTPX 公共传输接口和 Pydantic schema，未升级 SDK。

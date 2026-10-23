# Luna 测试入口迁移

本次迁移只修改测试、归档和本文档。运行时控制器为 `DemoRunner`，其 Responses 程序解释器、技能契约、版本检查、异步取消和执行逻辑参与测试；已删除的 JEV 调度器不再作为测试入口。

## 归档范围与可恢复性

依据 `data/luna-integration-full-pytest.txt` 中的 163 个失败逐项分类，归档了 32 个文件中的 101 个测试函数，合计 145 个参数化用例。这些函数调用已移除的通用 `Runner._run`、`select_arguments`，依赖旧 JEV 选择契约或已经归档的测试夹具；部分超时、API 锁断言是旧夹具根本没有进入运行造成的次生失败。

完整原文件保存于 `data/retired-jev-tests/function-migration-20261020/originals/`。同目录的 `manifest.json` 记录原文件 SHA256、函数原行号、函数 SHA256、每个参数用例和归档理由；`retired-functions.md` 提供函数清单。原有 `data/retired-jev-tests/` 的归档不覆盖。

仅移除清单里的测试函数及其装饰器，并清理由此不再使用、也没有被其他测试导入的 import。45 个非测试辅助函数和类经过 AST 比对，保持原样。其他测试和共享夹具保留在原模块内，没有用 `skip`、`xfail` 或修改 pytest 收集规则隐藏失败。有效的几何、SDK 适配、API 和配置迁移测试仍由完整 `pytest` 收集。

## 阈值测试迁移

`test_fast_attachment_vlm.py`、`test_front_tcp_clearance.py`、`test_place_settings.py`、`test_release_15mm.py` 中原先失败的 16 个参数用例，改为验证当前 `PlanningSettings`：

- 默认碰撞阈值与派生释放高度均为 0.005 m。
- 显式碰撞阈值保留，派生释放高度等于该值，不额外加 0.002 m。
- 旧保存文件里的释放高度和两个旧 margin 字段不能覆盖该派生关系。
- 配置迁移不改变无关参数，也不会在只读加载时改写原文件。

增加显式 0.005 m 的参数用例，保留非默认阈值测试。四个文件的迁移前副本与哈希见 `threshold-migrations.json` 及 `originals/`。既有几何函数独立测试没有为凑通过而改写，它们与当前 Demo 固定旋转放置策略的覆盖分开记录。

## 当前控制器的安全覆盖

新增 `tests/test_demo_async_safety.py` 与 `tests/test_demo_run_lifecycle.py`，替代旧入口中最关键的取消和运行锁覆盖：

| 场景 | 当前测试检验的行为 |
|---|---|
| IK 或轨迹规划仍在等待时 Reset | 通知取消；晚到结果和进度不恢复旧运行；旧请求未结束时拒绝新任务；不发运动命令 |
| IK 或轨迹规划超过共用截止时间 | 进入 `needs_attention`；保留失败步骤；晚到成功不能置为 `plan_ready` 或执行运动 |
| 请求返回前仿真 generation 改变 | 在动作前拒绝旧结果，完成 ID 清单保持为空 |
| GPU 服务错误 | 保留错误 code/details 与失败步骤，不进入执行 |
| 请求未返回时 | 运行进度可见，不提前宣称计划已就绪 |
| 开夹、接近、抓取、闭夹、抬升执行中 Reset | 即使旧动作返回成功，也不能发下一阶段命令或恢复旧事件 |
| 当前 DemoRunner 正在规划时修改场景或设置 | HTTP 409，当前配置不改变 |
| Reset 后旧请求仍在进行 | 保留 pending 锁；拒绝重叠任务和旧 generation 的 Start |

这些测试只替换慢速外部算法和机器人接口，以可控事件模拟不可中断的服务和晚到返回。没有替换 DemoRunner、契约、超时、取消或 API 锁实现。测试通过证明控制接线与状态边界，不等于 GPU 规划或真实 MuJoCo 搬运成功。

旧测试中其他专项安全目标仍可从归档恢复并逐项迁移；本文不声称 145 项旧用例的所有验证目标已经由新增测试一一替代。原有直接调用运行原语的有效测试继续保留。

## 验证记录

先运行新增异步安全测试：8 项通过。随后补充执行阶段与 API 覆盖，共 15 个新用例；连同四个阈值文件运行：61 项通过。

首轮完整测试发现 `test_collision_model.py` 仍从旧 `web.compose.yaml` 直接读取服务。当前部署已将该文件变为兼容 include 入口，因此保留原验证目标，将测试迁移到实际 `deploy/demo.compose.yaml`：分别解析网页 resources 挂载和 cuRobo 项目根挂载，确认两者指向同一个现有 `marvin_gui_dual.yml` 且均为只读。迁移前文件也已归档，没有修改部署配置。

最终在 Windows 项目目录以 `PYTHONPATH=src` 执行 `python -m pytest -q`：**1561 passed，54 skipped，0 failed，105.82 秒**。日志为 `data/luna-test-migration-full-pytest.txt`；首轮日志保留在 `data/luna-test-migration-first-pytest.txt`。38 个变更测试文件的 Ruff 检查通过。原有 54 个条件跳过不代表对应环境能力验收，本次未新增跳过标记；真实 GPU、模型 API 和完整 MuJoCo 搬运仍按独立实验记录验收。

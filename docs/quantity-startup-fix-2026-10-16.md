# 初始数量判断报错修复

## 复现与原因

生产任务 `1c89607bfed14edb8e6cd949782247e4`、`aca1055c8c494edb9e3d221dd38d3730` 均在 `bind_task` 报 `low_confidence:quantity`，尚未开始观测、建图或运动。原句为“抓取盒子中绿色物体放到桌面黄色区域”。任务、路线、目标颜色及目的区域都已正确识别；quantity 返回 unbounded，置信度分别为 0.40、0.48，低于配置门槛 0.60。隔离真实 JEV 探针再次复现，置信度为 0.28。

旧问题询问“要求搬运的总物体数”，数值选项没有明确要求原文数量证据，容易混淆“未写数量”和“场景可能只有一个绿色物体”。新问题只提取原文明示的数量，数值选项明确排除容器、障碍、编号和动作次数；“同一个”跨两段搬运仍计一个。没有数量限定不代表搬走全部场景物体，后续 assess_task 仍按原文、实测来源及属性决定任务成员。

## 修改边界

- 沿用 `request_scope_questions`、`Decisions.bind_request` 及现有数量候选，无新增 skill；不降低 0.60 门槛、不固定数量、不自动重试。目标与恢复仍由 JEV 判断。
- 低置信度异常附带字段、返回值、实际置信度和门槛。run conclusion 保留对应 skill_error 的结构化证据，显示可读原因；启动失败明确说明尚未观测或运动。
- `pydantic-ai-slim==2.47.0`、`typesafe-sdk==0.7.1` 及 `jev-1.13.0` 保持不变，SDK/vendor、相机、cuRobo、碰撞和释放算法均未修改。
- 依据官方 [Choice 问题说明](https://docs.typesafe.ai/primitives/choice) 区分选项语义；依据[置信度说明](https://docs.typesafe.ai/confidence)保留 confidence 与选项概率的区别，没有把选项概率当作现有门槛值。

## 验证

- 原句改前真实调用失败；改后通过，quantity 置信度 1.00。额外三次独立复查均为 1.00。
- `tests/request_scope_jev_smoke.py`：10/10 条真实 JEV 指令通过，涵盖未写数量、明确一个、组合数量、全部、环境数量及同一物体中转；只调用任务理解，不移动机器人。样本通过不代表所有自然语言输入保证成功。
- `pytest -q`：1512 passed、54 skipped。新增异常证据、缺失置信度、可读结论和真实 Runner 日志链回归；Ruff 与 `git diff --check` 通过。
- 数据保存在 Ubuntu `data/request-scope-20261016/`：原始失败日志、改前/改后探针、三次复查和更新前状态。本项没有重跑完整抓放路线，也没有重建 GPU 地图。

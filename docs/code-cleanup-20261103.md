# 当前调用链清理（2026-11-03）

本轮按用户要求删除废弃实现，保留当前 Responses → graph v2 → Decisions/skills 调用链。清理前快照位于 `data/cleanup-20261103/before.zip`，不参与导入或运行。历史实验归档保留用于比较。

## 删除的实现

- graph v1 的 `steps/if/for_each_target/retry/replan` 执行器、专用参数模型和旧数据流检查器。
- 第二次 Responses、编译修正及运行时 Responses 重编排接口。预算只允许一次初始调用；运行时恢复由现有 Decisions graph patch 完成。
- 旧放置网格/yaw 候选搜索、`plan_place_batch_native`、`/tools/plan_place_batch` 及旧结果整理模块。
- CAD 几何残差复核、ICP 修正、相关开关及几何不吻合时的多相机尝试分支。
- SAM6D 备用客户端和选择入口。FoundationPose 是当前唯一 CAD 位姿后端。
- 无调用的测试辅助模块、已无测试的文件和失效导入。仍有效的身份、数量、位姿、碰撞、观测失效与多物体流程测试迁到 graph v2。

删除 8 个生产源码模块，生产源码净减少约 2,500 行。精确文件清单和行数见 `data/cleanup-20261103/summary.json`。旧测试的移除包含退役接口测试及迁移到当前接口的测试，不能把测试数量减少解读为覆盖所有历史行为。

## 当前边界

`resolve_placement` 负责生成并验证当前落点；`plan_place` 必须接收该结果，只做固定目标轨迹规划。底层固定落点算法与 cuRobo BatchMotionPlanner 保留。缺少落点结果时明确拒绝，不退回另一套位置搜索。

FoundationPose 从有效深度最多的一个相机估计当前位姿。仍验证 CAD 源哈希、相机/观测代次、标定、坐标变换一致性；不重新计算 CAD 与掩码/点云的几何吻合度。新观测更新位姿，永久 ID 与 CAD 模型身份分别管理。

执行前场景版本、身份、实际机器人反馈、attachment、工作空间、碰撞和释放条件检查保留。`verify_place` 仍只调用三相机 RGB Decisions。交付后回位和观测由 graph 显式编排，技能不插入跨技能动作。

测试迁移揭示并修复一个诊断问题：子 graph 中技能失败、Decisions 选择停止后，原错误码会被“恢复停止”覆盖。现在原始错误原因和细节向上传递，另记录停止原因；不重试或重播动作。新增测试验证第一物体失败后不执行第二物体。

## 验证与同步

完整回归 **4,284 通过、58 跳过**（`data/cleanup-20261103/full3.txt`）；当前 graph/模型协议补充回归 **97 通过**。`ruff check src tests --select F` 通过。生产源码净减少 **2,517 行**。

Ubuntu 已部署 `timing-89`。同步逐文件核对旧哈希，更新 95 个文件并删除 33 个废弃或迁移后的旧路径，已确认这些路径均不存在。项目 1,096 个跟踪文件及独立 CAD 库 112 个文件哈希一致。文档及验收结果随后归档为 `timing-89-report`。

实际服务完整拆包实验 **P89 成功**，run ID 为 `329d47c003ad47c6be8c09230d45945d`：

| 项目 | 实测 |
| --- | --- |
| 总用时 | 218.111 秒 |
| 初始 Responses | 1 次，59.857 秒 |
| 扣除 Responses | **158.254 秒** |
| Decisions | 24 次 |
| 算法/技能错误 | 0 |
| 交付后回位并观测 | 2 次 |

归档确认粉色包裹采用 OBB，蓝盒与黑色工件采用对应 CAD。完成粉件到配置区域、拆包、黑件入右盒及最终配置观测位姿。验收摘要见 `data/cleanup-20261103/P89-acceptance.json`，完整选择性证据位于 `data/performance-unpack-20261101/P89-audit/`。

cuRobo SDK、容器 CPU 配额和独立 CAD 库没有修改。SDK 的 361 个源码文件逐一匹配官方提交 `8e734f3ced1df898990bcd92de40abce475907db`，CPU 配额仍共 4 核。本轮是调用链与代码清理，不以单次任务运行证明总体成功率或额外提速。

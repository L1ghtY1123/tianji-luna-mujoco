# 旧身份代码清理验收

日期：2026-10-18。本次实现位于 `C:/project/tianji-jev-mujoco-requirements`，已同步到 `/home/guangyu/projects/tianji-jev-mujoco-requirements`。

`C:/project/tianji-jev-mujoco` 和 Ubuntu 当前运行服务没有切换。此次是候选代码清理及回归验收，不是新一轮在线 Decisions 或机器人搬运验收。

## 删除范围

- 整体删除 `association.py`、`same_camera_association.py`、`scene_changes.py`。
- 删除旧距离、颜色、掩码、预测位置匹配，跨相机与跨时刻的旧关联分支，碎片合并及重复计数规则，身份重编号、历史点云合并和匹配专用字段。
- 删除 Runner 中没有注册的旧技能入口及无调用的辅助函数；内部仍被调用的 observe、move_to_observe 和 evaluate_arms 保留。
- 删除旧几何放置成功判断、旧 VisualAssistance.inspect，以及 scene_change 请求和响应分支。仍注册的 inspect_scene 只提供语义辅助。
- 删除13个退休测试/工具文件；混合测试保留真实几何、碰撞、冻结清单、遮挡、恢复和执行覆盖，改用明确的视觉判断结果。测试替身不重新实现身份匹配算法。

本轮共删除16个文件。按改动前备份计算，生产源码新增156行、删除1747行，**净减少1591行**；测试及辅助工具净减少2099行。既有未提交工作不计作本轮新增。

| 文件 | 清理前 | 清理后 |
| --- | ---: | ---: |
| perception.py | 1037 | 684 |
| runner.py | 2430 | 2006 |
| verification.py | 160 | 39 |
| visual_assistance.py | 257 | 191 |

## 保留的职责

Decisions 是清单与实体身份判断的唯一来源；每个未缓存观测仍须由外层 JEV 选择用途。缺少 Decisions 结果明确失败。SAM3 保留分割；注册 RGBD 保留桌面、盒底、点云及碰撞几何。Decisions 确认同一 ID 后的像素并集、边界过滤和机器人表面过滤仍是必要的几何处理。

所有实体和搬运清单引用同一全局 ID；缺测的已知障碍不会静默消失。每次放置包括最后一件都要求释放反馈及当前帧中同 ID 到达指定分区的视觉确认。身份或去向判断失败时保留释放证据，不记为搬运完成。

独立审查发现并修复了删除历史关联时引入的遮挡回归：最新帧看不到已持物源时，单独 place 从已保存的抓前 scene_observation 恢复源记录。它只初始化已确认持物的任务绑定，不给当前观测补点，也不放宽场景时效检查。该用例先失败，修复后通过。

## 实际验证

- Windows：`python -m pytest -q --disable-warnings`，**1661 passed，54 skipped**，88.53秒。
- Ubuntu 隔离测试容器：同一命令，**1705 passed，10 skipped**，75.81秒。
- `python -m ruff check src tests` 通过；`git diff --check` 通过。
- Registry 实际加载16项；AST检查确认退休函数已无定义，三个删除模块不存在。当前技能手册、README及相关契约已同步，历史验收记录保留原文。
- 独立审查覆盖删除调用链、独立放置、放后建图、完成判定与文档一致性；发现的源绑定回归已补测试并修复。

Ubuntu 首轮失败来自复用测试容器的临时目录、配置文件和渲染缓存所有权不一致。最终在重启后的本次专用测试容器内使用一致用户及独立临时目录通过；没有为消除环境失败而修改产品行为。

未修改 SDK、vendor、机器人资源、资产或依赖版本。现有锁定依赖下的两个平台回归通过；这不替代完整 GPU 运动验收。

## 发布与证据边界

本轮实际 OpenAI/TypeSafe 请求均为0，没有执行机器人搬运动作，没有更换当前运行服务。此前因官方额度限制未完成的最终 Decisions 输入格式在线验收，仍未在本轮完成；不把历史成功响应或离线测试当作当前在线验收。当前生产服务仍使用旧版本，候选代码已删除旧关联且不会回退。

本地改动前备份、哈希清单、统计和测试日志位于 `.superpowers/sdd/2026-10-18-identity-cleanup/`；Ubuntu 同步备份与验证日志位于 `/home/guangyu/projects/tianji-jev-acceptance-20261016/identity-cleanup-20261018/`。备份用于恢复，不在运行导入路径中。

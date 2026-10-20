# 单目录合并记录（2026-10-19）

用户明确要求只保留 `tianji-jev-mujoco`。当前唯一开发与运行目录为：

- Windows：`C:/project/tianji-jev-mujoco`
- Ubuntu：`/home/guangyu/projects/tianji-jev-mujoco`

## 合并范围

主分支从 `d88b95f` 快进合并到 `dad14b5`，保留原候选分支的 22 个提交；随后合入该工作树尚未提交的最终代码与文档。既有未提交改动保留在 Git stash 和文件备份中。合并前，Windows 与 Ubuntu 的主目录、候选目录分别逐文件比较，源码没有跨机器差异。

已核对主目录独有修改与候选最终版本：关节累计运动选择、实际停留计时、放置实测快照等修正均已包含。最终新增 RGB 清单、Decisions 身份绑定与放后确认、视觉请求归档，保留前端回放/TCP 显示等既有改动；旧身份关联模块及相关旧测试的 16 项删除保持有效。未修改官方 SDK、模型权重、依赖版本或运动算法。

Windows 的 `requirements`、`scene-opt` 工作树与 Ubuntu 的 `requirements` 工作树均已注销。两台机器项目父目录下均只保留一个 `tianji-jev*` 目录，即本主项目。旧版 `tianji-jev` 是不同架构的历史工程，完整移入本主目录历史归档，没有把它的模块或依赖混入当前运行入口。

## 资料保留

备份统一位于各自主目录的 `data/consolidation/2026-10-19/`：

- 合并前源码与工作文件 ZIP、状态、完整差异、Git bundle，以及合并文件清单。
- 候选中的新实验文件与 `.superpowers` 证据补入主目录；同路径不同内容保留主版本，候选版本保存在 ZIP。
- `legacy-tianji-jev/` 保存旧独立工程，包括未提交改动。Windows 工作树残留的依赖目录链接与历史数据保存在 `retired-directory-remainders/`。
- Ubuntu 的旧候选资料、验收数据和视觉探针分别移入 `candidate-artifacts/`、`acceptance-20261016/`、`vision-probe-20261019/`，映射记录在 `archived-paths.json`。

历史报告中的旧绝对路径仅说明当时位置。当前启动、源码、skill 契约及测试入口均不依赖额外工作树。已有主目录 `.env`、规划参数、运行日志与实验库保留。

## 运行与验证

在 `tianji-jev-mujoco` 主目录重新执行：

- `python -m pytest -q`：**1677 passed，53 skipped，96.50 s**。
- `node --test web/tests/*.test.js`：**271 passed，0 failed**。
- `git diff --check`：通过。

测试日志位于本次备份目录。数据归并后，比先前候选验证多执行了一项测试；跳过测试不代表已通过。

Ubuntu 更新前确认当前运行已结束。复用现有 `tianji-jev-mujoco:0.4.0-experiments-guangyu` 镜像，重新创建网页容器以应用新增的专用视觉配置路径；没有构建新镜像，没有重启 SAM3、GraspGen 或 cuRobo 算法服务。网页挂载仍全部指向主目录。额外候选测试容器和旧工程的未启动容器已在保存配置后移除。

更新后 `/api/state`、`/api/health`、`/replay.html` 均返回 HTTP 200，任务为空闲；专用视觉配置可读、key 已配置，容器可以加载 `/v1/responses` 和 `/v1/decisions` 客户端。未在本次合并中发起模型推理或机器人运动。

最后一次真实 Decisions 请求仍是原验收记录中的 HTTP 429；不能据此保证当前额度已恢复。新视觉链的真实跨视角/跨时刻关联及整轮搬运仍待验收。详情见 [RGB 清单验证记录](rgb-inventory-validation-2026-10-19.md)。

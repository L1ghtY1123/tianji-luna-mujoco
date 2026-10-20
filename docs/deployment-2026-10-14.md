# 2026-10-14 Ubuntu 同步与首次 Git 提交

本地 `C:/project/tianji-jev-mujoco` 同步到 `/home/guangyu/projects/tianji-jev-mujoco`。包含双臂离线工作空间查看页、observation 统一障碍图、执行检查场景缓冲复用，以及逐相机放置验证证据和明确的最终结论。控制链、SDK 和模型权重未替换。

同步前远端为空闲、无持物、无活动运动。已备份待替换文件和完整页面状态，停止 JEV 网页及 cuRobo 容器后应用更新，再启动这两个容器。SAM3、GraspGen 及其他项目容器保持原样。原来的五个场景对象、桌面布局及已保存规划参数已恢复。

网页：`http://192.168.3.4:8340/`；工作空间：`http://192.168.3.4:8340/workspaces.html`。后者及 `/api/workspaces`、`/api/workspace-scene` 已通过只读 HTTP 核验。两个工作空间的模型指纹匹配，使用 1 cm 网格。cuRobo、GraspGen、网页均通过健康核验，JEV 密钥已配置；未为本次同步额外发送 JEV 请求或启动物理抓放。

两端此前均没有 `.git`；本次建立 `main` 分支并保存完整可运行源码、配置、机器人资产与离线工作空间。首次提交由 Codex 署名。`.env`、运行/实验记录、视频、渲染缓存和依赖目录不入库。模型资产、工作空间配置和 vendor 保持原始字节，避免 Git 换行转换破坏指纹。没有向 GitHub 等外部仓库推送。

Ubuntu 独有的历史验证脚本已保留并复制回本地。Linux 回归发现旧测试期待 `0.3.0-rgbd-guangyu`，而启动器与现有生产镜像已采用 `0.4.0-experiments-guangyu`；本次只更新该测试的默认版本断言。缺少的静态碰撞拟合历史测试数据也已从本地补齐，未覆盖不同内容。

验证记录存放在两端 `data/sync-commit-20261014/`。Ubuntu 测试使用独立临时容器和生产同版本镜像，只读挂载源码、历史测试数据；渲染缓存使用临时内存目录，不载入生产密钥。前端 243 项测试通过，Vite 构建通过（保留现有大于 500 kB 的 bundle 提示）。

最终 Ubuntu 全量 Python 回归：**1352 passed, 10 skipped，63.65 秒**；`ruff check --no-cache src tests` 通过。10 项跳过项保留其原有条件，本次没有新增跳过。输出见 `ubuntu-final-tests.txt`。

本次部署不构成完整抓放复跑验收。之前 111.7 秒运行的最终落点仍未获得补充证据，也不能用独立性能测试推断部署后的整轮耗时。

回退文件和页面状态：`data/sync-commit-20261014/deployment-backup/`。该目录及历史拟合测试数据属于外部工作资料，按现有 `data/` 忽略规则不包含在 Git 中；新克隆运行历史回放测试时需另行恢复相应数据。

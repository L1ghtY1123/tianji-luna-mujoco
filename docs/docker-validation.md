# Docker 部署验证 — 2026-09-29

网页和 MuJoCo 已从宿主 `.venv` / systemd 迁移到 Docker。旧服务已 disabled/inactive，8340 网页容器健康；cuRobo 和 GraspGen 镜像及运行容器保持原状。

## 使用入口

- `scripts/web-service.sh start|stop|status|logs` 管理网页，`build` 仅构建镜像。
- `scripts/algorithms-service.sh start|stop|status|logs` 管理现有两种算法容器。
- start 默认版本 `0.2.0-guangyu`，可显式传入同名版本。
- 根目录 `start.sh`、`stop.sh` 委托网页 Docker 脚本。
- `.env` 只读挂载，`data` 保存日志和规划参数，临时渲染网格放在容器 tmpfs。

## 实际验证

- Linux 启停脚本行为测试：25 passed；真实 Bash + fake Docker，检查参数、错误、服务隔离、缺镜像自动构建、任意工作目录及含空格路径。
- Ubuntu Python 全套：212 passed。
- Windows Python 全套复跑：187 passed，25 个 Linux shell 测试 skipped。首次严格 50 ms 超时断言因 Windows 15.625 ms 定时分辨率波动失败，独立复跑 10 次及后续全套通过；未修改业务或测试逻辑。
- 前端：113 passed；Docker 内 frozen-lockfile 构建成功。保留原有 Three.js chunk 大于 500 kB 提示。
- Ruff、两个 Compose 配置解析、四个 Bash 入口语法检查通过。
- 独立 18342 容器使用私有测试 data，主网页场景未参与测试。主页面和碰撞球页面浏览器渲染正常，无 pageerror；原始碰撞球 1362 个。
- 实际 JEV + GraspGen 200 候选 + cuRobo batch + MuJoCo 抓取红色圆柱成功，约 84.65 秒；夹爪夹持、物体移动、TCP/物体变换一致三项检查均通过。原始数据保存在 Ubuntu `/tmp/jev-docker-guangyu-yvdafwac/container-run.json`。
- 独立容器停止后重启：参数和 JSONL 日志哈希保持一致，仿真重新初始化；`pip check` 通过。
- 最终镜像启动入口增加 Uvicorn 单 worker 和 10 秒 graceful-shutdown 上限。用该入口连接 WebSocket 后停止测试容器，0.33 秒退出，退出码 143，日志确认 application shutdown complete，无 SIGKILL/OOM。

## 镜像与当前激活状态

- 当前已激活网页镜像 `tianji-jev-mujoco:0.2.0-guangyu`：`sha256:6a31da8b48a3f5d5e4f93a894bd45edf76f53b10b9c76562e88e16f5d97ea690`，容器显示完整名称且健康。
- 迁移第一阶段使用 `sha256:8da3452659a61b1ec73adfae1f57efd38494099c0db3c499098f146ed83ba38c`。重建复用标签后，旧容器仍引用旧 ID，造成仅显示 ID。用户随后要求修复名称；确认任务 idle 后执行 `scripts/web-service.sh start`，容器已切换到上述最终镜像，退出等待上限也已生效。
- 最初额外的主网页 stop/start/重复 start 验证曾被自动审批拒绝，原因仅为“blocked by policy”。后续名称修复的 start 已成功执行。主容器重复 start 的实际无重建行为尚未验证；脚本测试已覆盖对应调用。
- cuRobo：`sha256:71e2e6cc793cedc7c6b9ae9b1d4e1d92925f66eb3effb40224ae495c157347d8`。
- GraspGen：`sha256:47c4755a5598f1c7aac055040d9fa189d6bf0745c07f8fd804c6f55e4fd5fee4`。
- 原始碰撞配置 SHA256：`d7e627475865271a66bc785258dd59177080eb3a6e106cce26cb4465971421ae`。

本次未修改 skills、算法 SDK、机器人模型或碰撞球。旧网页场景快照和 systemd 单元备份保存在 `data/backups/docker-migration`。远端已失效的 `test_object_position.py` 调用不存在的接口，已移入该备份目录；本地原本已无此文件。

临时验证容器和失败的专用 builder 均已删除；未执行共享 Docker 全局清理。

## 用户要求的镜像清理

已删除确认无容器引用、无镜像 Parent 依赖、默认配置不再使用的旧网页镜像 `jodell-console:0.2.0-guangyu`（`sha256:e457910a32c2e738d6c29c5816a90c791ecb82fa90ee2ee4492cde13c1b4806d`）。逐一复核 18 个算法镜像及其全部原有标签，ID 均保持一致，包括算法备份。`gap-platform:0.1.0-guangyu` 仍被两个停止的容器与项目配置引用，保留；18 个无标签镜像没有可确认的归属信息，保留。审计记录见 Ubuntu 项目 `data/image-cleanup-before.json`、`data/image-cleanup-result.json`。

部署依据：[Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/)、[Compose services](https://docs.docker.com/reference/compose-file/services/)。Uvicorn 参数已依据镜像内锁定版本的 `python -m uvicorn --help` 核对。

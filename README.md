# Tianji Luna MuJoCo

自然语言和三相机初始 RGB 输入 GPT-6 Luna Responses，结合技能契约、配置区域及 CAD 目录，生成物体清单和显式 graph。运行时由 Decisions 判断当前证据、选择分支和恢复动作，由机器人 skills 完成感知、几何、规划与执行。

## 当前实现

- **一次初始 Responses**：输出 `version=2` 的 graph/subgraph/node/edge。旧 `steps` 执行器、绑定分支、前端解释器及额外 Responses 修正/重编排接口已经删除。
- **永久物体 ID**：清单绑定实例及操作历史；CAD asset ID 表示模型，允许多个实例共享一个 CAD。每个任务输入拥有独立 subgraph 和完成账本。
- **几何**：桌面使用 cuboid，未知物体使用当前观测更新的 OBB，已匹配 CAD 的物体使用 FoundationPose 当前位姿。生产模式为 `cad_obb`。
- **CAD 配准**：SAM3 掩码、RGBD、内外参和匹配 CAD 输入 FoundationPose；选择有效深度最多的一个相机，输出转换到统一世界坐标。保留来源、标定及坐标绑定检查，取消额外几何残差复核与 ICP。已删除 SAM6D 备用入口。
- **抓取**：有离线抓姿的 CAD 按执行臂选择对应抓姿；未知 OBB 使用 GraspGen。选臂依据任务约束、工作空间和可达性，颜色不绑定机械臂。
- **attachment**：抬升成功且确认持物后建立；持物模型随臂、抓取变换、场景版本绑定。
- **放置**：`perceive_placement_zone` 给出目标区域；普通区域按 4→8→16 分区提出 XY，Decisions 按语言约束选择，数值算法检查工作空间并求接近碰撞阈值的 Z。工作台指定区域使用配置 XY。`resolve_placement` 产生当前落点，`plan_place` 只规划该落点，不再保留另一套网格/yaw 搜索或旧 `plan_place_batch` 接口。
- **技能清理**：当前 21 个注册能力；旧 `plan_transport / execute_transport / drop / verify_drop` 契约和执行器已删除，容器几何共用 `robot_skills/container_geometry.py`。
- **轨迹**：调用固定版本的官方 cuRobo。抓取默认每批两个候选；已确定的放置目标使用 BatchMotionPlanner、batch=1。放置包含接近、放下及释放后的撤离。
- **核验**：`verify_place` 一次读取三相机 RGB，由 Decisions 判断目标支撑/入容器关系及是否仍被夹持；不调用 SAM3、FoundationPose、OBB 或释放后深度测量。
- **回位**：当前任务要求每件交付后双臂回配置 observation 位姿并观测，最终也处于该位姿。Responses 必须显式编排这些动作，runtime 不暗中插入流程。

## 目录与职责

算法代码只有一个维护仓库：`../tianji-algorithms`。Luna 不再保留 `vendor/`、`services/`、`src/tianji_foundationpose/` 或算法注册库副本。

| 位置 | 职责 |
| --- | --- |
| `src/luna_mujoco/` | 应用编排、MuJoCo 仿真、场景与机器人技能；专用 `geometry_service` 保留应用接口 |
| `skill_catalog/` | 技能契约、参数和前提/效果 |
| `resources/robot/` | 本机器人碰撞配置与几何审计，不含算法代码或 SDK 清单 |
| `resources/`、`assets/` | CAD、标定、工作空间、URDF 和网格 |
| `scripts/`、`web/` | 应用维护工具、工作台与回放 |
| `data/` | 实验、运行日志、回放网格和应用规划参数 |
| `../tianji-algorithms/services/` | 唯一的算法客户端、协议与服务适配源码，统一包 `tianji-algorithm-services` |
| `../tianji-algorithms/manifests/` | 官方 SDK 版本及源码校验清单 |
| `../tianji-algorithms/data/luna/foundationpose-registry/` | Luna 使用的 FoundationPose 注册数据，独立于源码 |

SAM3、GraspGen 和 FoundationPose 的服务定义统一位于算法仓库的 `deploy/luna-algorithms.compose.yaml`；本项目 Compose 通过 `include` 使用它。应用及 cuRobo 容器只读挂载同一份算法服务源码，代码中不再临时插入 `vendor` 路径。FoundationPose 客户端和服务共享同一个 `tianji_foundationpose` 包；Luna 所需缓存/跟踪适配在算法仓库的 `services/luna_foundationpose/`。

算法仓库的通用 cuRobo HTTP 服务不替代 Luna 的场景/持物接口；Luna 的 `geometry_service` 从统一包导入算法边界。官方 SDK 与权重仍在算法仓库维护，端口、镜像版本及现有协议保持。

独立 CAD 库位于 `/home/guangyu/projects/CAD`，通过 `LUNA_CAD_ROOT=/opt/cad` 只读挂载；`resources/cad/` 是应用静态副本。

## Git 上传与本地数据

源码、契约、部署配置及静态资源随 Git 保存。`.env`、运行数据、依赖和构建输出不上传；仅提交 `.env.example`。

Luna 的 `data/` 只保留 `experiments/`（实验回放）、`runs/`（运行日志）、`render_meshes/`（回放网格）和 `planning-settings.json`（应用规划参数）。旧调试脚本、离线候选实现、部署包、`tests/` 和 `web/tests/` 已按用户要求删除。FoundationPose 注册库已经逐文件校验并迁至算法仓库，Luna 不保留副本。

新机器需要同时准备本项目、相邻的 `tianji-algorithms` 与 `CAD`，以及本地镜像、`.env` 和 OpenAI 配置。注册库的准备工具为算法仓库中的 `scripts/provision_luna_registry.py`，显式传入 `--project` 指向 Luna；详细命令见 [注册库准备](docs/cad-catalog-provisioning.md)。历史文档中的旧测试、离线产物和迁移前路径仅记录当时状态。

2026-11-03 统一维护验收：算法仓库提交 `09c7330be8f3d441d580211e456ddaa48da35a53`；CPU 检查 41 项中 39 项通过、2 项跳过；现有镜像完成完整拆包交付仿真，实验 `63f27d77d9d24152bfe25a5137a6a380`，耗时约 264 秒。官方 SDK、机器人配置内容与已有实验记录保留。算法仓库目前只保存在本机，未配置 Git 远端。

## 启动

Ubuntu 使用现有镜像和项目 `.env`：

```bash
cd /home/guangyu/projects/tianji-luna-mujoco
LUNA_UID=$(id -u) LUNA_GID=$(id -g) docker compose -f deploy/demo.compose.yaml up -d
docker compose -f deploy/demo.compose.yaml ps
```

Windows PowerShell 打开工作台：

```powershell
Start-Process 'http://192.168.3.4:8350'
```

如果使用 SSH 本地转发，在一个 PowerShell 窗口保持连接：

```powershell
ssh -N -L 8350:127.0.0.1:8350 guangyu@192.168.3.4
```

另一个窗口运行 `Start-Process 'http://127.0.0.1:8350'`。工作台端口为 8350，cuRobo 服务端口为 8352。五个容器当前 CPU 配额总计 4 核。

## 开发与验证

重新构建应用镜像时，构建上下文必须来自统一仓：`docker build --build-context algorithm-services=../tianji-algorithms/services -f deploy/Dockerfile.web -t tianji-luna-mujoco:unified-guangyu .`。现有部署使用原镜像及只读源码挂载，本次迁移无需新建镜像。宿主维护命令可通过 `TIANJI_ALGORITHMS_ROOT` 指定非默认算法仓库位置。

开发环境先运行 `python -m pip install -e ../tianji-algorithms/services -e ".[dev]"`，随后运行 `python -m ruff check src --select F`；前端在 `web/` 中执行 `pnpm build`。测试目录已按用户要求删除，语法/构建检查、只读接口检查与完整仿真必须分别报告。普通任务可通过 `scripts/validate_demo_run.py --instruction "自然语言任务" --output data/acceptance.json` 验收。

新增技能需要契约、参数 schema 和 handler；不得在 controller 中添加固定任务流程。执行前检查观测/场景版本、身份、机器人反馈、工作空间与碰撞，不能用模型判断替代数值检查。未通过的判断沿 graph 失败或恢复分支处理，不伪造成功。

当前设计和证据：

- [框架、模型输入与代码清理核对](docs/framework-cleanup-20261103.md)
- [当前 CAD 与落点策略](docs/cad-runtime-policy-20261102.md)
- [显式 graph 与运行恢复](docs/native-graph-implementation.md)
- [Decisions 问题协议](docs/decisions-questions.md)
- [选臂、attachment 与放置性能](docs/arm-attachment-place-audit-20261103.md)
- [低面数 CAD 与 GPU 审计](docs/lowpoly-and-gpu-audit-20261103.md)

带日期的旧实验文档只记录当时实现，不代表当前接口或当前要求。

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

| 目录 | 职责 |
| --- | --- |
| `src/luna_mujoco/agent/` | Responses/Decisions 协议、问题和初始编排 |
| `src/luna_mujoco/core/` | graph、任务等数据模型 |
| `src/luna_mujoco/runtime/` | edge 执行、清单/账本、Decisions 恢复 |
| `src/luna_mujoco/robot_skills/` | 注册技能实现 |
| `skill_catalog/` | 技能说明、参数及前提/效果契约 |
| `src/luna_mujoco/backend_builder.py`、`geometry_service.py` | 官方 cuRobo 适配和服务 |
| `src/tianji_foundationpose/` | 主程序使用的 FoundationPose HTTP 客户端和数据契约，不含推理后端 |
| `services/luna_foundationpose/` | FoundationPose 容器侧的项目适配：模型缓存、请求服务及固定版本 SDK 接线 |
| `vendor/` | 固定版本的 cuRobo/GraspGen 依赖适配包副本；来源见 `ASSET_PROVENANCE.md`，不作为项目算法的新增位置 |
| `resources/` | CAD、相机标定、工作空间、碰撞配置与来源清单；只放静态资源，不放 Python 算法代码 |
| `assets/` | MuJoCo 与规划使用的机器人 URDF 和网格 |
| `scripts/` | 可复用的离线构建、资源准备和审计工具 |
| `web/` | 工作台和实验回放 |
| `data/` | 实验与运行日志、注册库、回放网格、规划参数 |

算法官方源码、权重和通用服务的维护仓库为 `/home/guangyu/projects/tianji-algorithms`。Luna 当前承担应用编排、MuJoCo 仿真、场景/碰撞/持物逻辑、专用 `geometry_service` 以及 FoundationPose 缓存适配，并非所有路径都只是 HTTP 转发。算法仓库的通用 cuRobo 服务不替代 Luna 专用接口。

`vendor/tianji_curobo/` 的 26 个 Python 文件和 `vendor/tianji_graspgen/` 的 7 个文件在 2026-11-03 与算法仓库逐字节一致，是仍被当前导入与镜像构建引用的依赖副本；它们不是另一份官方 SDK。`services/` 是 Luna 的容器专属适配，使用镜像内完整 FoundationPose 服务包；`src/tianji_foundationpose/` 是应用客户端子集，不能直接覆盖服务包的 Python 搜索路径。`resources/` 和 `assets/` 只提供配置、标定和模型输入。

正式入口不得加载 `data/` 中的脚本。可复用维护工具放在 `scripts/`；临时调试代码、候选实现和部署传输包在使用后删除。

独立 CAD 库位于 `/home/guangyu/projects/CAD`。生产通过 `LUNA_CAD_ROOT=/opt/cad` 的只读挂载使用它；`resources/cad/` 是同步维护的项目副本。蓝盒 1000 面、黑色工件 999 面，配准和碰撞使用同一表面，左右臂离线抓姿数值保持不变。

## Git 上传与本地数据

源码、技能契约、部署配置、`assets/` 和 `resources/` 随 Git 保存。`.env`、`data/`、依赖目录及构建输出由 `.gitignore` 排除，仅提交不含密钥的 `.env.example`。

`data/` 当前只保留以下五项，均有明确消费者：

| 路径 | 用途 |
| --- | --- |
| `experiments/` | 完整实验、相机图像、模型调用、轨迹与回放记录；`.trash/` 是可恢复删除区。 |
| `runs/` | 任务运行日志。 |
| `foundationpose-registry/` | FoundationPose 服务正在挂载的模型注册库。 |
| `render_meshes/` | 网页与历史实验回放引用的网格缓存。 |
| `planning-settings.json` | 网页保存的规划参数，应用和 cuRobo 服务启动时读取。 |

2026-11-03 按用户要求删除未被当前运行依赖的旧离线抓姿生成目录、性能排查材料、临时补丁、调试代码及测试目录。`data/` 从最初约 139.96 GiB 经两轮清理降至约 41.58 GiB；现有实验、运行日志和上述运行依赖保留。`tests/`、`web/tests/` 及测试专属配置不再保留，历史版本仍可从 Git 查询。

带日期的历史文档记录当时的验证结果；其中旧测试、夹具和离线审计路径可能已经移除，不代表当前可用文件。实验回放记录仍以 `data/experiments/` 为准，GitHub 克隆不会包含本地运行数据。

在新机器部署时，还需单独准备 `.env`、OpenAI 配置、Compose 指定的本地镜像、独立 CAD 库及 FoundationPose 注册库。注册库可迁移，或通过 `scripts/provision_foundationpose_registry.py` 配合匹配版本的服务包、CAD 库和 seed registry 准备。

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

安装项目及开发依赖后运行 `python -m ruff check src services --select F`；前端在 `web/` 中执行 `pnpm build`。测试目录已按用户要求删除，语法/构建检查、只读接口检查与完整仿真必须分别报告。普通任务可通过 `scripts/validate_demo_run.py --instruction "自然语言任务" --output data/acceptance.json` 验收。

新增技能需要契约、参数 schema 和 handler；不得在 controller 中添加固定任务流程。执行前检查观测/场景版本、身份、机器人反馈、工作空间与碰撞，不能用模型判断替代数值检查。未通过的判断沿 graph 失败或恢复分支处理，不伪造成功。

当前设计和证据：

- [框架、模型输入与代码清理核对](docs/framework-cleanup-20261103.md)
- [当前 CAD 与落点策略](docs/cad-runtime-policy-20261102.md)
- [显式 graph 与运行恢复](docs/native-graph-implementation.md)
- [Decisions 问题协议](docs/decisions-questions.md)
- [选臂、attachment 与放置性能](docs/arm-attachment-place-audit-20261103.md)
- [低面数 CAD 与 GPU 审计](docs/lowpoly-and-gpu-audit-20261103.md)

带日期的旧实验文档只记录当时实现，不代表当前接口或当前要求。

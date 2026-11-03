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
| `web/` | 工作台和实验回放 |
| `tests/` | 当前行为回归及测试数据 |
| `data/` | 实验、性能证据和清理前的不可执行快照 |

独立 CAD 库位于 `/home/guangyu/projects/CAD`。生产通过 `LUNA_CAD_ROOT=/opt/cad` 的只读挂载使用它；`resources/cad/` 是同步维护的项目副本。蓝盒 1000 面、黑色工件 999 面，配准和碰撞使用同一表面，左右臂离线抓姿数值保持不变。

## Git 上传与本地数据

源码、技能契约、部署配置、`assets/`、`resources/` 和 `tests/fixtures/` 随 Git 保存；`.env`、`data/`、渲染缓存、依赖目录及构建输出由 `.gitignore` 排除。仅提交不含密钥的 `.env.example`，不要用 `git add -f` 上传运行数据或凭据。

`data/` 有用，但不属于源码分发包：

| 路径 | 用途与保留建议 |
| --- | --- |
| `data/experiments/` | 实验回放、相机图像、模型调用、轨迹和诊断证据。需要复现的记录应单独备份。 |
| `data/experiments/.trash/` | 工作台删除实验后的可恢复回收区；删除操作会移入此处，仍占磁盘。确认不再需要恢复后才可清理。 |
| `data/runs/` 及各审计目录 | 历史运行、性能分析和排错材料；按实验价值保留，不上传 Git。 |
| `data/render_meshes/` | 网页渲染网格缓存，可由当前模型生成；历史回放可能引用旧缓存。 |
| `data/foundationpose-registry/` | FoundationPose 当前部署挂载的模型注册库，运行需要，不能随历史实验一起清空。 |

2026-11-03 检查时，`data/` 约 140 GiB，其中实验回收区约 96 GiB；这些是本机占用，不计入 Git 提交。本文中的 `data/` 证据链接指向本地归档，GitHub 克隆不会包含这些记录。

在新机器部署时，还需单独准备 `.env`、OpenAI 配置、Compose 指定的本地镜像、独立 CAD 库及 FoundationPose 注册库。注册库可备份迁移，或使用 `scripts/provision_foundationpose_registry.py`，结合匹配固定版本的服务包、CAD 库和已有 seed registry 准备；仅克隆本仓库不等于已具备完整运行环境。

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

安装项目及开发依赖后运行 `python -m pytest -q` 和 `python -m ruff check src tests --select F`。普通任务可通过 `scripts/validate_demo_run.py --instruction "自然语言任务" --output data/acceptance.json` 验收。

新增技能需要契约、参数 schema 和 handler；不得在 controller 中添加固定任务流程。执行前检查观测/场景版本、身份、机器人反馈、工作空间与碰撞，不能用模型判断替代数值检查。未通过的判断沿 graph 失败或恢复分支处理，不伪造成功。

当前设计和证据：

- [框架、模型输入与代码清理核对](docs/framework-cleanup-20261103.md)
- [当前 CAD 与落点策略](docs/cad-runtime-policy-20261102.md)
- [显式 graph 与运行恢复](docs/native-graph-implementation.md)
- [Decisions 问题协议](docs/decisions-questions.md)
- [选臂、attachment 与放置性能](docs/arm-attachment-place-audit-20261103.md)
- [低面数 CAD 与 GPU 审计](docs/lowpoly-and-gpu-audit-20261103.md)

带日期的旧实验文档只记录当时实现，不代表当前接口或当前要求。

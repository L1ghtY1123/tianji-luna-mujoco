# 场景准备、反馈判定与 15 mm 释放验收

基线为 `79cc0aa`。本轮使用现有 `jodell-curobo:0.2.0-guangyu`，没有新建镜像、升级依赖或修改 SDK/vendor。安装的 361 个 cuRobo 源文件与锁定版本 `8e734f3ced1df898990bcd92de40abce475907db` 的清单一致。集成使用公开 `Mapper`、`MapperCfg`、`Mapper.integrator.compute_esdf/get_voxel_grid` 和 Torch CUDA stream。

## 当前实现

- `observe_scene` 包含首次头部发现和用于观测运动的初始地图。每轮 `build_scene` 完整执行双臂移动、同步 RGBD/SAM3、ID 关联与场景重建。静态桌面 cuboid 复用；机器人靠近桌面时不通过重新拟合缩小桌面来消除碰撞。
- 抓后及放后不再例行采 RGBD。`verify_grasp` 仅检查夹爪反馈；`plan_place` 开始单独执行、计时 `prepare_attachment`，再次确认当前持物反馈。释放后只按完整动作和开爪反馈判定，不声称验证落稳位置。
- 运行时 skill 从 25 项精简至 16 项。JEV 保留任务与异常恢复决策；恢复可以额外观察、建立包含目标的运动地图，正常循环不自动重试。每阶段执行臂固定。
- 目标 TCP XY 为位置候选。每个 yaw 按 `TCP_Z = floor_z + 0.015 - min_i((R*c_i).z-r_i)` 求 Z，使用实际上传 attachment 球。硬碰撞间距仍 10 mm，实际终点容差 2 mm；不从盒沿或高位直接释放。

## GPU 建图对比

`tests/instance_mapping_gpu_probe.py` 在现有 cuRobo 容器内作为独立进程运行，不调用正在持物的服务、不移动机器人。输入仅为保存的 RGBD、SAM3 实测点/像素来源和机器人反馈。三实例样本从保存像素来源恢复掩码；不读取独立评分中的仿真物体真值。

基线和候选分别使用空的专用 Warp 编译缓存；测量范围为 `PerceptionMapper.build`，包含投影、桌面/实例处理和场景组装，不含拍照、SAM3、网络传输、机器人 segmenter 配置或运动规划。首次包含编译；重复时间取同输入后续中位数，基线 3 次、候选 5 次。最终计时未启用 profiler。

| 保存的场景 | 基线首次 | 候选首次 | 基线重复 | 候选重复 | 基线 ESDF 单元 | 候选 ESDF 单元 |
|---|---:|---:|---:|---:|---:|---:|
| 3 个有效实例 | 5.373 s | 4.994 s | 287.3 ms | 101.2 ms | 2,676,192 | 434,832 |
| 2 个有效实例 | 5.160 s | 3.790 s | 217.8 ms | 118.9 ms | 2,467,696 | 268,122 |

重复建图分别减少约 65% 和 45%；首次分别减少约 7% 和 27%。这是共享 GPU 上的两份保存数据结果，不是整条任务或所有场景的速度保证。

主要优化是：实例各用独立 Mapper/stream，最多四个并发；保留分配缓存但每帧重置观测；按完整查询余量缩减 ESDF；语义场景不再分配/上传 2,112,000 单元的空背景层；不再对永不参加正式建图的背景重复做点云投影、支持面深度重建和零深度构造。背景统计中的 `all_nonsemantic_including_support` 明确包含由 cuboid 取代的桌面。初始/持物背景模式仍保留原处理。

**不是所有收益都来自并行。** 三实例的首次实例阶段为 3.288→3.375 s，重复为 11.63→11.41 ms；两实例为 2.896→2.175 s、9.87→8.77 ms。独立 profiler 指出新配置的内核编译是首次实例阶段的主要成本。暖态 ESDF 核计算已经很短；继续增加批量不能消除配置、CPU 处理和感知调用耗时。

ESDF 单元是保存“到最近障碍表面的距离”的网格采样，包含表面周围的查询空间，数量不等于物体实心体素数量。分辨率保持 1 cm。单纯去掉空背景使总单元数下降很多；实例层自身分别从 564,192→434,832、355,696→268,122。保留球半径、激活距离及边界插值所需余量，不把减少单元数等同于减小硬碰撞间距。

## 数值与资源检查

- 两组录制样本的各实例 TSDF 表面逐点完全相同；缩小后的 ESDF 与基线对应区域逐格完全相同。保持 float32 配置计算及网格奇偶相位，避免浮点转换多分配一个单元、改变表面采样。
- 每实例分别在 10/30 mm 激活距离查询 15,625 个球，候选与基线结果一致。原始 scatter ESDF 对“最近保留表面中心”参考仍存在少量边界分类差异；本轮没有修复或隐藏此 SDK 离散化现象，不能据此宣称真实物体的全部不可见表面安全。
- `measured_table_mapping_gpu_smoke.py` 通过：首次保留有效背景；正式场景删除空背景层；桌面复用；裁去桌面以下实例点/体素；重建后网格持有独立结果；reset 使桌面证据失效。
- CPU 注入 packing/allocation 异常验证：等待已拥有 stream、无部分网格发布、缓存最多八项。不同实例不能共用一张可写 Mapper。
- 记录到的 CUDA allocator 峰值约 1,166.6 MiB，两版相同；保留缓存峰值约 3.84–3.96 GiB。网格减少不意味着整个感知进程峰值按比例下降。

原始 JSON、ESDF NPZ、表面点与日志保存在本地 `data/scene-opt-evidence/scene-opt-evidence-20261015/`，Ubuntu 同步目录为 `/home/guangyu/projects/tianji-jev-mujoco-candidates/scene-opt-evidence-20261015/`。复现入口接受保存的 request JSON 或 `capture.json`、`observation.json`、RGBD/instance NPZ 所在目录；候选加 `--mode compact --baseline <native结果目录>` 会断言上述等价性。

## 工程验证与部署边界

Windows：`pytest -q` **1507 passed, 54 skipped**；Ruff 通过；前端 **248 passed**；Vite 构建通过，仅保留原有大于 500 kB 的 bundle 提示。跳过项包含本机没有 CUDA 的检查，不能计为运动验收。独立代码复核发现并修复了恢复无有效运动地图、固定臂提示不一致、defer 中途失败重新选中旧目标及异常缓存增长等问题。

本轮没有执行完整三条路线的真实 GPU 运动验收。用户已明确允许保存记录后切换并重启现有服务。失败任务 `ccbb5076286f4477a713430dccef61c4` 的状态、持物证据、原始任务日志、规划参数与容器信息已保存到 Ubuntu `/home/guangyu/projects/tianji-jev-mujoco-candidates/scene-opt-deployment-20261015/`，共 6 个原始备份文件，附 SHA-256 清单。该备份用于回看证据，不是可以恢复持物仿真的检查点。

实现提交 `87ab2b2` 已快进同步到 Windows 与 Ubuntu 主工作目录。2026-10-15 23:35（UTC+8）停止并启动现有 web、cuRobo 容器；两者容器 ID 与镜像 ID 均未改变，SAM3、GraspGen 未重启。四个容器均 healthy；cuRobo 健康接口确认 GPU 可用、锁定 SDK 版本未变。网页、回放、碰撞球和工作空间页面及其入口 JS 均返回 HTTP 200。生产 Registry 为 16 项；规划设置接口与持久化文件均为释放高度 0.015 m、硬碰撞间距 0.010 m。重启后任务为空闲、无持物状态。部署核对结果另存上述目录的 `activation-health.json`、`runtime-verification.json`；没有自动发起新的搬运任务，JEV 外部 API 未在本轮健康检查中调用。

已保留最终 GPU 报告、网格和日志，清理本次拥有的 30 个宿主临时文件及 20 个容器临时目录/文件（含独立编译缓存）；未触碰生产缓存或其他任务文件。本轮没有创建或删除 Docker 镜像。

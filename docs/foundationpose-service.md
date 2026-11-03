2026-11-02：当前资产、直接位姿、分层XY与观测复用规则以[本轮策略](cad-runtime-policy-20261102.md)为准，覆盖下文历史检查链路。

# FoundationPose + SAM3

2026-11-02 项目服务适配器：`services/run_foundationpose.py` 调用镜像内原HTTP接口和官方SDK，`services/luna_foundationpose/backend.py` 按源模型/网格哈希缓存至多4个估计器及GPU资源。每次请求仍运行官方register（252方向、5次refinement），不复用pose、不执行track、不在每次请求后empty_cache。原模型登记与源坐标系保持；派生配准网格位于`resources/registration_meshes`，蓝盒2000面、黑件1999面，记录原文件哈希、派生哈希、单位及恒等坐标变换。原CAD碰撞与离线抓姿不变，当前全部数值校验仍针对原CAD。

原始掩码/RGBD冻结对照位于`data/fp-speed-20261102/benchmark-raw.json`与`validation-raw.json`。6个输入中原版与缓存/2000面版均为5个通过现有验收、1个原有大角度错误被拒绝；拒绝案例未放宽阈值。`benchmark.json`是早期使用观测输出过滤深度的探索，不作为原始输入验收证据。500面候选最大采样表面偏差过大（蓝盒5.75mm、黑件14.22mm），没有部署。2000面候选双向顶点/边中点/面中心抽样最大偏差蓝盒0.933mm、黑件1.253mm；这是采样检查，不是全表面Hausdorff上界。减面是一次离线CPU工作，运行时不再处理原始高面数网格初始化。

统一源码和构建仓库位于 Windows `C:\project\tianji-algorithms`、Ubuntu `/home/guangyu/projects/tianji-algorithms`。

服务镜像为 `jodell-foundationpose:0.2.2-guangyu`，Ubuntu 本机地址 `http://127.0.0.1:8365`。官方源码锁定到 `a1b694b83e633c2cb6115b9063d940a687759392`，未修改 SDK。使用官方权重、渲染和 5 次位姿细化。

```python
from luna_mujoco.pose_estimation import CadPoseEstimator

poses = CadPoseEstimator(algorithm="foundationpose").segment_and_estimate(
    frozen_view, "blue box", model_id="blue_box")
```

默认位姿客户端选择 FoundationPose；环境变量 `LUNA_POSE_BACKEND=sam6d` 或显式 `algorithm="sam6d"` 可选择备选。SAM6D 容器目前已停止，使用前需按其部署说明手动启动。`LUNA_FOUNDATIONPOSE_URL` 可覆盖 FoundationPose 地址。

`frozen_view` 必须包含同一采集的 RGB、米制深度、有效深度、内参、相机外参、frame/camera/calibration ID、采集时间和仿真代次。调用方明确关联 `model_id`；文本分割不自动确定 CAD 身份。返回相机、世界及原 CAD 源坐标系位姿，使用米和 OpenCV 相机轴。官方 register 没有校准置信度，`pose_score` 为 null。

## 显存配置

当前部署使用公开 `default`：252 个初始方向候选，`min_n_views=40, inplane_step=60`，5 次 refinement。历史 `low_memory` 为 42 候选，`min_n_views=40, inplane_step=360`。服务健康与结果均报告实际配置，没有静默降级。

在统一仓库中切换配置：
```bash
cd /home/guangyu/projects/tianji-algorithms
FOUNDATIONPOSE_SEARCH_PROFILE=default docker compose -p tianji-algorithms-guangyu -f deploy/foundationpose.compose.yaml up -d --no-build --pull never
# 恢复低显存配置，把 default 改成 low_memory。
```

早期仅约 4.5 GiB 空闲显存时，252 和 84 候选曾 OOM；这是历史记录。当前 default 配置已完成冻结配准与生产调用，不能用早期低显存记录代表当前部署。

## CAD 与用途边界

新增刚体 CAD 可动态注册，不需重建镜像或离线渲染模板。完整命令见统一仓库 `docs/foundationpose-integration.md`。支持 STEP/STP/STL/PLY，必须声明单位和 `rigid_reference`；变更几何使用新模型 ID。

粉色等未知物体的包围盒不能充当精确刚体 CAD。蓝盒及黑色工件 CAD 已接入 `cad_obb` 场景；已知黑件使用左右臂分别对应的离线抓姿，无目录候选物体使用当前 OBB 和在线 GraspGen。CAD 配准失败不改走 OBB。

服务返回 `pose_hypothesis_only`；调用方用当前实测点及可见深度轮廓验证后才发布 CAD。Decisions 先关联掩码与永久 ID，再为未绑定模型的 ID 从目录提出 CAD 候选。两个同款实例可以共用 CAD，但各自独立估计位姿。

三相机按同 ID 的有效深度数量排序，先从一个视角估计，再用所有已关联视角验证；数值拒绝才换剩余视角，每相机最多一次，服务/来源错误立即失败。不平均独立姿态。模型对应可缓存，当前世界位姿不可用旧值代替。详见[身份、三相机和放置边界](cad-identity-placement-convergence.md)。

旧版本 hybrid-15 在 MuJoCo 已完整通过 B 臂 CAD 黑件入盒、A 臂未知 OBB 桌面任务；A 臂 CAD 完整任务尚未通过。多视角备用估计等本轮改动须单独验收。见[实际验证](cad-obb-validation-20261030.md)。

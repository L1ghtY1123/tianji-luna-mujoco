# 释放验收与GPU占用复核

## 当前放置标准

用户将本阶段验收简化为“放置结束且夹爪不再持物”。`verify_place`现只读取撤离后快照里的夹爪反馈，要求本轮`execute_place`完整完成、已记录释放、开爪命令及实际位置确认张开。使用已有行程2%容差，不增加试探闭爪。返回`verification_scope=release_only`和`destination_pose_verified=false`；成功可以推进物体清单，但不宣称最终落点已被测量。

常规验证不调用SAM3/OpenAI，不要求源物体或目的区域可见。反馈缺失或动作未完整结束仍为unknown，不能用图像替代缺失的反馈。JEV决定是否取得新反馈或停止；下一轮抓取前的观测、双臂观测位姿与建图流程保持原规则。程序设定的目的区域也适用此验收标准；这不代表新增了任意数值目的区域的输入接口。

原实验`3d70ff079017496aadcb76a72dc55e5a`的真实执行后夹爪位置为0.0001662rad，开爪基准0rad，行程0.8rad，开爪命令为false（非闭爪）。新验证入口回放返回confirmed，没有相机分割或视觉API调用。历史实验文件和结论未改写；该回放不等于新一次完整物理抓放。

## 最近失败的区别

- `9bde4758…`、`a4113e30…`：放置72个候选IK通过，轨迹规划发生CUDA显存不足；没有执行place。之后的重复skill拒绝是恢复调用前提不满足，不是十几次物理失败。
- `3d70ff07…`及同阶段记录：放置后身份/可见几何核验不明确，JEV决策输入超过预算。上一轮已压缩关联摘要，本轮进一步取消常规放置的视觉验收依赖。
- 其他旧记录还包含视觉API限流和工作空间判定失败，不能统称为显存不足。本轮未重新执行这些旧实验。

另修复`recover_task(retry_place)`：仅plan_place失败、未执行动作且原场景仍有效时，保留场景前提；不再无条件失效为缺图状态。执行阶段失败或场景已要求刷新时继续要求新证据。恢复action仍由JEV选择，程序不自动重试。JEV提示说明显存不足不能靠重复观察解决；没有资源变化证据时不要反复重试。

## GPU快照

本轮读取RTX PRO 6000：总97887MiB，已用85986MiB，可用11262MiB。以下是同次`nvidia-smi`和Docker进程表的对应结果，显存会随运行改变。

| PID | 归属 | 占用MiB | 约GiB |
| --- | --- | ---: | ---: |
| 3499597 | `jra-svc-real-threaded-housing`，宿主用户jason | 64984 | 63.46 |
| 3810967 | `curobo-zhongxu-sdkqueue` | 9266 | 9.05 |
| 712874 | `jev-mujoco-sam3-guangyu` | 5408 | 5.28 |
| 569541 | 宿主jason的sam3 Python进程 | 4924 | 4.81 |
| 713078 | `jev-mujoco-graspgen-guangyu` | 874 | 0.85 |

本项目cuRobo刚重启且尚未创建求解器时，未列入GPU计算进程。此前OOM日志中它已占约10.48GiB，全卡只余约136MiB，申请额外26MiB失败。当前空闲约11GiB不代表运行峰值足够；本次未停止其他人的服务，也没有宣称显存问题已解决。

## Open3D与镜像纠正

当前桌面建图调用Open3D的CPU `voxel_down_sample`、`segment_plane`；可选点云过滤也使用它。Open3D有实际用途，但原`jodell-curobo:0.2.0-guangyu`已经安装`open3d-cpu==0.19.0`，路径为`/usr/local/lib/python3.12/site-packages/open3d`。派生0.3又在`/var/lib/jodell/.local/lib/python3.12/site-packages/open3d`安装同版本，属于重复依赖，不需要为此保留派生镜像。

已用0.2现有镜像实际运行项目桌面RANSAC入口，得到有效桌面cuboid。默认镜像恢复0.2，并删除不需要的派生Dockerfile和Compose build入口；不构建新镜像。SDK commit仍为`8e734f3ced1df898990bcd92de40abce475907db`。

## 视觉服务配置

用户确认密钥来自`max.jojocode.com/keys`。现有受保护配置的接口为`https://max.jojocode.com/v1`，与该服务商对应；`/keys`是管理页面，不是模型请求地址。沿用已有配置，不复制密钥进代码、Git或实验日志。本轮常规放置验收不调用视觉API。

## 验证及部署

- Windows完整测试：1352 passed、53 skipped；Ubuntu已有网页镜像内完整测试：1395 passed、10 skipped。Ruff通过。
- 用原实验放置结束时的状态做一次真实JEV回放，选择`verify_place`，耗时约1.16秒；零OpenAI视觉调用、零机器人命令。
- Ubuntu已切回原0.2 cuRobo镜像，健康接口确认GPU可用及原SDK commit、原309球模型指纹。四个项目服务均健康，网页容器保持原实例。
- 检查包括停止容器在内的引用后，0.3镜像无容器引用、无其他标签，已删除`e62a3000eae59523c5922928a87f24cc4ba636b0ee708804794defaf43de612d`；原0.2镜像`b6e5c212a5c7b886e273ac9003a04628dfe1f7bb3a3da083a0e6d891f0b70f42`保留并运行。没有构建新镜像或清理全局缓存。
- 未跑新的完整物理抓放，未消除共享GPU资源不足。组件测试、历史反馈回放和真实JEV决策回放分别验证其各自范围。

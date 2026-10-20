# 共享碰撞球模板

当前模型为 **267 球、38 个 link**，SHA256 为 `73199e86daf2d6772f45c4227074158bcf1bc1deb1bee363d51d5810d1b6ce1e`。本次修改依据已确认的对称建模方案；只更新碰撞球及离线重放、检查和说明。

## 模板关系

| 部位 | 数量 | 生成方式 |
|---|---:|---|
| 左右机械臂本体 | 各 80 | Base、Link1–4、Link6 局部几何相同，直接共享；Link5、Link7 复用右侧官方 final，经零位世界 Y=0 镜面及两侧 FK 转入左 link |
| 腕相机 | 各 10 | 支架与外壳分件共享；各自按 URDF visual 的实际安装位姿转换 |
| 张开夹爪 | 各 30 | 两手同名 link 共用；手内对侧指片在开爪 Base Y=0 面镜像，再转回对应 link |
| 中央结构 | 27 | 上柱 8 球归轴；下主体三重排列；上支架 x/y 镜面对称；法兰四球规则排列；头相机及支架保留实际偏置 |

先前 266 球中 Link5 左侧 23、右侧 24；本次两侧统一复用右侧 24 球模板，所以总数增加 1。没有重新做两次独立随机拟合。

模板均来自上一版冻结的官方 MORPHIT 输出。每个输出球记录来源 link、来源索引及 4×4 等距变换，半径逐值沿用被引用的来源球。对球心允许旋转、平移和反射，拒绝缩放、剪切和重复映射。所有规则从同一份冻结来源读取，执行顺序不影响结果。

只有几何与安装关系支持时才施加镜像。相同局部网格的两条机械臂依靠各自 FK 安装，不把全部局部坐标统一取反。球仍属于各自 link，运动时随各自关节变化；两臂独立运动时不要求其世界坐标继续互为镜像。法兰与下主体是规则近似，不能把离散旋转对称说成整机连续轴对称。

## 几何检查及取舍

- 默认姿态、零位以及此前 835 个原始轨迹构型均未发现球模型自碰撞。
- 这 835 个构型的最小球自间隙由 0.017131 mm 变为 0.334290 mm；最小球与桌面间距由 12.926703 mm 变为 14.509431 mm。独立原始 visual 对场景的最小采样间距仍为 14.937252 mm。
- 左 Link5 的固定 4000 点表面抽样最大 world-buffered 残余由 14.009 mm 降至 10.091 mm；左 Link7 由 10.682 mm 降至 9.708 mm。两者 raw 球边界的条件外凸最大值分别由 7.326/3.834 mm 变为 2.863/3.571 mm。
- 静态原网格整 link 的 raw 最大采样残余：Base 18.44→15.86 mm，Stand 12.34→12.08 mm。上柱球心最大平移 1.248 mm，z 与来源半径保持原值。
- 局部取舍仍存在：头相机球不变，但相邻主体球变化使头部局部 raw 采样残余约 6.01→7.65 mm；法兰 proxy 最大外凸约增加 0.29 mm；右腕相机 world-buffered 原网格最大残余约 5.99→6.48 mm。

上述是各报告指定样本、几何范围与 buffer 模式下的测量，不能互相替代，也不构成任意姿态或连续运动的完整碰撞证明。少球避障模型仍有近似误差。

## 官方依据与边界

参考 [cuRobo sphere fitting](https://nvlabs.github.io/curobo/latest/reference/sphere_fitting.html) 的球体近似与误差检查，以及 [self collision](https://nvlabs.github.io/curobo/latest/reference/self_collision.html) 的 link 球集合和自碰撞配置。本次是项目侧坐标映射，并非 cuRobo 新增的自动对称 API。

模板来源使用锁定 commit `8e734f3ced1df898990bcd92de40abce475907db` 的公开 `curobo.sphere_fit` MORPHIT。此次没有重新拟合或修改 SDK。独立测试容器内 361 个 SDK Python 文件与官方 manifest 相符，官方 ESDF 测试及 motion_planning 示例也与同 commit 逐字节相符。

三段 plan_grasp_batch、seed、碰撞阈值、buffer、self_collision_ignore、开爪锁定、skills 和插值流程均保持原值。

## GPU 对照

使用同一已有算法镜像的独立临时容器，生产服务保持停止。A 使用相同 16 个历史目标；B 使用历史可达子集中的 56 个目标，先各自做 Batched IK，再按 score 取成功项前 16 个规划。这个 B56 集合并非全部 200 个 GraspGen 候选。

| 阶段 | 原 266 球 | 对称 267 球 |
|---|---:|---:|
| A16 cold plan_grasp_batch | 23.966 s，13/16 | 23.053 s，14/16 |
| A16 warm 第一次 | 1.032 s，13/16 | 1.050 s，14/16 |
| A16 warm 第二次 | 1.024 s，12/16 | 1.039 s，13/16 |
| B56 cold Batched IK | 18.312 s，32/56 | 18.389 s，32/56 |
| B56 warm Batched IK | 12.303 ms，32/56 | 11.991 ms，32/56 |
| B 筛选后 cold plan_grasp_batch | 6.159 s，4/16 | 6.158 s，4/16 |
| B 筛选后 warm plan_grasp_batch | 0.969 s，4/16 | 0.965 s，4/16 |

结果表明本组输入的计算耗时基本持平，不把对称建模说成算法加速。cold 包含新进程内 backend 构建；warm 复用同一进程缓存。计时同步 GPU，包含 `run_operation`，排除导入、场景构建、HTTP、序列化及机器人运动。设备为共享 RTX PRO 6000 Blackwell，重复次数有限。

A 三轮均增加同一个可行目标，未丢失原成功目标；B 的 IK 成功索引、选中 16 项与规划成功索引均相同。新生成轨迹中 A 三条、B 两条最高 score 可行结果共 835 个原始构型，也完成独立球及原始 visual-world 检查：未发现球自碰撞，最小球自间隙 0.334290 mm、球桌面间距 14.509431 mm、真实网格采样间距 14.937252 mm。它们与历史路径回归分别留档，不将同值结果说成任意路径保证。

额外 B16 探针的基线与候选均为 IK 11/16、规划 cold 3/11、warm 1/11；归档中的 `source-b56` / `candidate-b56` 目录记录这个 16 目标探针，名字不能作为样本数量依据。正确 56 目标报告为 `source-b56-full` / `candidate-b56-full`，表格使用后者。

项目验证：`python -m pytest -q` 为 **334 passed、56 skipped**；新增重放脚本及相关测试的 Ruff 检查通过。230 个受保护的源码、skills、部署配置和模型资产文件 SHA 保持不变。

## 复现

归档：`data/collision-symmetry-2026-10-08/`。`source.yml` 为原 266 球模型，`rules.json` 记录模板映射，`recipe.json` 校验冻结输入及最终模型。以前的官方拟合归档保持可重放。

```powershell
.venv/Scripts/python.exe scripts/replay_symmetric_collision.py --output C:/Users/l1ght/AppData/Local/Temp/symmetric-model-guangyu.yml
.venv/Scripts/python.exe -m pytest -q tests/test_symmetric_collision_replay.py tests/test_collision_model.py tests/test_official_collision_replay.py
```

输出路径须为尚不存在的新文件。`comparison.png` 使用相同零位、相机、透明度及网页相同的 YAML+buffer 半径对比；没有执行机器人运动。

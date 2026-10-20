# 碰撞球拟合与冗余消除：309 球

当前配置为 **309 球、38 个 link，上限 350**，SHA256
`f617cfc7dcd382d7eda37e998392fe286212f01ab59c7bf7d5dcbbbda10f5197`。
2026-10-14 从下述 313 球 v5 配置中删除左右 Link4 各两个球：每个球均被同一 link 的保留球严格包含，最小包含余量分别约为 1.32 mm 和 1.75 mm。因此球并集、网格覆盖、世界障碍物碰撞占据区域不变。固定 1000 点网格审计和默认/零位自碰撞审计均与 v5 相同；这四球的删除不保证可测得的规划加速。可用 `recipe-v6.json` 精确重放当前 YAML。
两个离线单臂工作空间的可达网格未修改；校验旧模型指纹确实来自 v5 YAML，且 YAML 除上述四球外完全相同后，将 `.npz` 元数据绑定到新指纹，并记录旧指纹和重绑理由。这样工作空间预览仍可加载；工作空间仍只用于位置预筛选，不证明特定姿态或轨迹可行。

以下是 2026-10-12 的 **313 球 v5** 拟合记录，SHA256 为
`218caa59066ab46c55f6d5431a7025e1c382880bb02ee09ee164d96dec983709`。
采用冻结 v5：恢复中央底座/立柱的精确旋转、反射对称，分段改善 arm Link2 与 TCP 支架；原 finger Link1–4 保留，避免新的局部拟合扩大窄框相关截片或产生自碰撞。
所有非球设置保持原值，默认/零位和 835 个历史姿态无球自重叠。原始 CAD 固定 12,000 点和独立 50,000 点的最大 gap 未增，但底座条件外凸增加 0.589 mm，部分 raw 覆盖比例下降，手指仍有明显外凸；不声称完整覆盖或 18 cm 料框必然可行。

详细分配、原始 CAD/proxy 指标、精确 10 mm 截片、拒绝方案与图片见
[313 球复核报告](../data/next-revision-audit-20261012/spheres/report-v5.md)。
归档为 `data/next-revision-audit-20261012/spheres/recipe-v5.json`，冻结重放：

```powershell
.venv/Scripts/python.exe scripts/replay_transfer_spheres.py --archive data/next-revision-audit-20261012/spheres --recipe recipe-v5.json --output C:/Users/l1ght/AppData/Local/Temp/revision-spheres-v5-guangyu.yml
```

## 以下保留此前 267 → 293 球的历史记录

以下表格、规划计时和复现命令仅属于此前的 293 球版本，不是当前 309 球的 GPU 性能或实物验收结论。历史配置为 **293 球、38 个 link**，SHA256
`002e11a58a8892f1c5befc68f2cf0a27c2c27b2e730334065786115d9046373e`。
本次仅更换六个 link 的部分球；所有 buffer、自碰撞忽略关系、开爪锁定、关节限制、规划设置及 SDK 保持原值。Ubuntu 生产服务未被本次离线拟合修改或重启；部署由主任务另外验证。

| 区域 | 原球数 | 新球数 | 处理 |
|---|---:|---:|---|
| Link_Base | 12 | 18 | 下部四球换为十球；上方八球原样保留 |
| Link_Stand | 15 | 23 | 主体五球换为九球，法兰四球换为八球；相机六球保留 |
| Link4 左/右 | 各 12 | 各 16 | 轴段四球换为八球，两侧共享同一局部模板 |
| TCP_Link 左/右 | 各 10 | 各 12 | 安装法兰四球换为六球，保留实际安装坐标及其余相机球 |
| 其余 32 个 link | 196 | 196 | 逐值保留 |

分组总数为机械臂各 84、腕相机各 12、夹爪各 30、中央结构 41。

## 官方来源与重放

使用锁定 commit `8e734f3ced1df898990bcd92de40abce475907db` 的公开
`curobo.sphere_fit.fit_spheres_to_mesh`、`SphereFitType.MORPHIT`。
核对了同 commit 的[公开导出](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/sphere_fit.py)、[参数定义](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/geom/sphere_fit/fit_spheres.py)和[官方示例](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/examples/reference/sphere_fit_comparison.py)。本版本用 `num_spheres`，默认 200 次迭代；没有混用 latest 文档中的参数名。

每次隔离拟合均校验 361 个安装包 Python 文件的官方 SHA。SDK、vendor、机器人网格均只读。采用每个公开调用的完整 final，不拼历史输出、不手调或缩放半径、不使用 fallback。MORPHIT 可以输出少于预算的球，因此最终数是 293，而非凑整到 300。

输入沿用已归档的物理部件外包络，保留原有孔槽填充和底部省略范围；没有新增几何省略。下部立柱采用 coverage/protrusion=1000/30，其余选中拟合采用默认 1000/10，均为 200 次迭代、随机种子 20261012。这里的拟合损失权重不改变碰撞检测容差。立柱使用 `round4`，主体、Link4 轴段和腕法兰使用 `round2`，支架法兰使用 `round3`。Link2 增预算及半区拆分试验因覆盖/外凸退化而拒绝，仍使用基线球。

归档为 `data/continuous-transfer-20261012/spheres/`。`source.yml` 与主任务 `baseline.zip` 内配置逐字节相同；`recipe.json` 冻结输入网格、final 数组与坐标映射，`candidate.yml` 等于当时的 293 球配置。浮点变换检查允许 1e-14 m 的跨平台矩阵乘法舍入，半径必须逐值一致；实际重放使用已冻结的 final 序列化并校验整份配置 SHA。

```powershell
.venv/Scripts/python.exe scripts/replay_transfer_spheres.py --output C:/Users/l1ght/AppData/Local/Temp/transfer-spheres-guangyu.yml
```

输出必须是尚不存在、位于运行资源和归档之外的新文件。重新进行 CUDA 优化使用 `scripts/refit_transfer_spheres.py --plan <归档计划> --output <新离线目录>`；脚本检查生产算法请求未在途、SDK SHA、GPU 空闲内存和互斥锁，且拒绝写入资源/SDK。重优化存在 GPU 数值差异，不等同于冻结重放。

## 几何结果与保留误差

`accepted-geometry.json` 对每个改变的 link 使用固定 12,000 个面积采样点、所有网格顶点和每球 256 个方向。下表是 **原始真实网格**的最大采样表面缺口，以及经实体/可见性判定的球并集外边界到该 link 网格的最大距离，单位 mm。原球 buffer 全部保留；本次改变的 link 对应 world buffer 为零。

| link | 表面缺口，前→后 | 条件外边界距离，前→后 |
|---|---:|---:|
| Link_Base 已批准管体/筋板/上法兰范围 | 15.942 → 12.228 | 77.795 → 65.917 |
| Link_Stand | 12.778 → 9.926 | 46.697 → 44.398 |
| Link4 左/右 | 8.096 → 8.096 | 30.459 → 28.851 |
| TCP_Link_L | 6.241 → 6.241 | 32.173 → 23.444 |
| TCP_Link_R | 6.241 → 6.241 | 32.173 → 24.118 |

每 link 的覆盖比例、P95、网格顶点和最差点坐标均在报告中。静态 CAD 含槽孔和筋板间空隙；上述原网格距离会计入已批准外包络填充的空气，不能全部理解为外轮廓之外的伸出。`part-proxy-geometry.json` 另外测量各物理部件原有外包络：下部立柱外伸 42.05→23.87 mm，支架法兰 27.48→14.19 mm，腕法兰 31.83→17.02 mm。

取舍没有隐藏：下部立柱对填满筋板空隙的凸包最大采样缺口 18.09→25.83 mm，而对真实管体/筋板表面的缺口下降；腕法兰自身包络缺口 4.54→5.50 mm，但整个 TCP link 的原网格最大缺口不变。未修改的 Link2 仍有约 31 mm 的局部原网格外边界误差。MORPHIT 不是严格外包围算法，本模型不保证所有表面完整覆盖，也不能证明任意姿态或连续运动安全。CAD 查询产生一次已有退化三角形除法警告，最终保存的全部数值通过有限值检查。

相同 MuJoCo 零位、锁定开爪和相机的对比图为 `comparison.png`；没有驱动物理机器人。

## 自碰撞与真实 GPU 规划比较

`selected-historical-paths.json` 检查默认/零位和此前 835 个原始轨迹构型：两模型均无球自重叠，最小球自间隙均为 0.334290 mm，最小球桌面间隙均为 14.509431 mm。独立原网格对场景采样最小间隙为 14.937252 mm。

在同一已存在镜像 `jodell-curobo:0.3.0-visible-cloud-guangyu` 的独立临时容器中，使用完全相同历史输入。生产服务保持运行且空闲；没有给生产算法服务发送规划请求，也没有改写生产配置。

| 测量 | 原 267 球 | 新 293 球 |
|---|---:|---:|
| A16 cold plan | 25.223 s，14/16 | 25.332 s，14/16 |
| A16 warm 1 | 1.082 s，14/16 | 1.177 s，14/16 |
| A16 warm 2 | 1.069 s，13/16 | 1.160 s，13/16 |
| B56 cold IK | 20.070 s，32/56 | 20.319 s，32/56 |
| B56 warm IK | 14.355 ms，32/56 | 14.545 ms，32/56 |
| B 筛选后 cold plan | 6.472 s，4/16 | 6.618 s，4/16 |
| B 筛选后 warm plan | 0.981 s，4/16 | 1.076 s，4/16 |

每轮所有成功索引完全相同，B 的 IK 成功集和送入规划的目标也相同。新模型热规划约慢 0.09–0.10 秒，不声称球数增加带来加速。cold 包含对应新进程内 backend 构建；计时同步 GPU，排除导入、场景构建、HTTP、序列化和运动。RTX PRO 6000 Blackwell 是共享设备，样本与重复次数有限；B56 是历史可达子集，不能代表所有原始候选。

新规划每轮最高 score 的成功轨迹共 835 个原始构型也经独立检查：`new-a-paths.json`、`new-b-paths.json` 无球自重叠，最小间隙与上表历史路径相同。检查没有覆盖五次插值后的连续时间、实际抓放执行或新 RGBD 场景；这些属于主任务集成验收。

复现本地几何和历史路径检查：

```powershell
.venv/Scripts/python.exe scripts/audit_transfer_spheres.py --baseline data/continuous-transfer-20261012/spheres/source.yml --candidate resources/curobo/marvin_gui_dual.yml --output C:/Users/l1ght/AppData/Local/Temp/transfer-geometry-guangyu.json
.venv/Scripts/python.exe data/collision-symmetry-2026-10-08/validation/cpu_regression.py --baseline data/continuous-transfer-20261012/spheres/source.yml --candidate resources/curobo/marvin_gui_dual.yml --output C:/Users/l1ght/AppData/Local/Temp/transfer-paths-guangyu.json
```

几何审计使用项目现有 trimesh 查询工具；Windows 审计环境增加 rtree 1.4.1、Shapely 2.1.2、mapbox-earcut 1.0.3，没有修改产品依赖或算法镜像。冻结的 `fitter-executed.py`、`benchmark-executed.py` 保留实际执行版本；当前 `scripts/benchmark_transfer_spheres.py` 仅整理了同一版本的 import 顺序。

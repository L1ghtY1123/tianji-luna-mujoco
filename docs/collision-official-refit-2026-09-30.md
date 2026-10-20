# cuRobo 碰撞球修正：1051 球

本次替换的是碰撞几何，不修改 skills、SDK、算法镜像、碰撞 buffer、自碰撞忽略关系、开夹爪锁定值、规划参数或执行轨迹处理。

## 方法与此前的问题

参考 [官方 sphere fitting](https://nvlabs.github.io/curobo/latest/reference/sphere_fitting.html)、[官方 self collision](https://nvlabs.github.io/curobo/latest/reference/self_collision.html)，实际调用项目锁定提交 `8e734f3ced1df898990bcd92de40abce475907db` 的公开 `curobo.sphere_fit.fit_spheres_to_mesh`。该版本参数名为 `num_spheres`，MORPHIT 默认迭代 200 次；不能直接把官网 latest 的模块路径、`n_spheres` 和 50 次迭代混用到本镜像。

旧修正把原球、新拟合和局部补球叠加，并用“必须完整包含旧球”作为回归条件。旧球里的空气也被保留，造成支架鼓出和密集碎球。现在以真实 visual geometry 为输入，每个物理区域只采用一组最终结果；不拼优化历史，不加 uncovered-point 补球，不手改输出半径。部件交界处的正常重叠仍保留。

MORPHIT 同时权衡内部覆盖与外凸，并不保证每个真实表面点都严格位于球内。官方示例的球数也不能直接套用到不同机器人。本次用相同几何采样、球表面距离、实际关节姿态和原始轨迹复核取舍，未把官方 API 调用成功当作拟合质量证明。

## 实际改动

| 部位 | 原数量 | 新数量 | 处理 |
|---|---:|---:|---|
| 中央立柱 Link_Base | 302 | 126 | 保留完整 16 球轴向柱链；重新拟合端部、六片筋板 |
| 上方支架 Link_Stand | 159 | 54 | 主体 30、完整法兰 13、相机支架 9、相机 2；删除旧主体与补球叠加 |
| Link1 左 / 右 | 16 / 22 | 11 / 11 | 分别对完整 link 做官方 MORPHIT |
| Link4 左 / 右 | 25 / 26 | 20 / 20 | 分别对完整 link 做官方 MORPHIT |
| 其余部位 | 809 | 809 | 保留原值；低球数试验造成更大外凸、漏面或腕部误碰撞，未采用 |
| **整机** | **1359** | **1051** | **减少 308，约 22.7%** |

下方不可达四脚底座继续沿用已批准的省略范围。主体规则分块来自完整物理部件的固定分区，不按漏点随机补球。相机支架的槽孔填充与向外凸出分别统计。

默认 coverage/protrusion 权重为 1000/10。采用的非默认公开参数如实记录：筋板 protrusion=30；上方主体和法兰 coverage=10000；保留轴向柱链来自此前 coverage/protrusion=1000/200 的公开 final。没有把这些自定义值称为官方默认。全部输入、随机种子、预算、刚体坐标变换和 final 数组均保存在归档中。

## 验证与实际限制

- Windows 本地完整测试：312 passed、56 skipped；修改脚本及测试的 ruff 检查通过。
- 6 个修改 link 的输出来源、刚体变换、球半径及配置差异检查通过。除 `collision_spheres` 外，机器人配置与原始导入值一致。
- 默认姿态、零位及 1287 个历史 cuRobo 原始轨迹点，共 1289 姿态，未发现球模型自碰撞；最小 self 间隙 1.42247 mm、桌面间隙 72.26224 mm。这只覆盖这些姿态。
- 左右 Link1/Link4 分别检查，没有直接套用对侧坐标。最终更密采样中，世界可见表面最大 gap：Link1 左/右 6.525/6.272 mm，Link4 左/右约 7.592 mm；raw 外凸最大值分别约 15.176/15.198、14.517/14.606 mm。
- 左 Link4 在同口径比较中，raw 外凸最大值从 30.655 降到 14.517 mm；最大 gap 从 7.259 到 7.592 mm。上方主体此前候选的 8.782 mm 角缘缺口已消除；最终法兰一处同点 gap 为 4.370 mm，旧模型为 2.138 mm，存在局部覆盖取舍。
- 相机支架某点距原始槽形 mesh 为 26.632 mm，但距该独立部件外凸包为 2.301 mm；前者不能全部解释为外轮廓鼓出，后者也不能取代原始距离指标。
- 实际隔离 `plan_grasp_batch` 测试使用相同历史输入、每批 16 个：A 是 3 个位姿重复，B 是 2 个位姿重复。A 三轮均 16/16 成功；B 三轮 15/13/15，有 Lift 失败，合计成功数与此前 1359 球测试相同。不是 16 个独立场景的统计。
- 新 batch 末轮 A/B 各最高分成功轨迹共 306 个原始采样点，独立核查无 self overlap；最小 self 间隙分别 1.66122/2.00943 mm，桌面 72.26240/59.33282 mm。此复核未覆盖 Hermite、物体 ESDF 或连续时间。
- 本次 A 冷调用 20.621 s、预热后 4.801/4.927 s；B 为 21.449、4.763/5.393 s。冷调用含 planner 初始化。跨时段共享 GPU 的观察不能严格证明所有加速均来自球数减少，也不能称为实时。

本次未调用 JEV、未执行机器人动作、未改变 Hermite 流程。几何仍是近似，不能由上述测试保证所有真实表面及任意连续轨迹均有 10 mm 距离。腕部、指片等保留原模型的误差也仍存在。

## 归档、查看与复现

- 运行配置：`resources/curobo/marvin_gui_dual.yml`。
- SHA256：`959292e2718d7e028942f66ab5a18fbe12e428aa525aaab2db08758dadbb437d`。
- 来源与输入：`data/collision-official-refit-2026-09-30/recipe.json`、`manifest.json`、三个部件归档目录。
- 定量报告：同目录 `validation/`；原网格与真实配置半径的并排渲染：`render/`。
- 旧 1359 球归档仍保留，供比对和回滚，不再作为当前配置。

精确重放已保存 final 数组（不重新进行 CUDA 优化）：

```bash
python scripts/replay_official_collision.py --output /tmp/jev-replayed-1051.yml
```

运行前应选择尚不存在的输出路径。重新优化可能受 GPU 数值差异影响，不能等同于上述逐字节重放。静态生成命令在冻结 manifest 中记录，必须显式使用冻结 source；不把新的运行模型当旧 source。

网页与规划服务均从项目挂载配置加载球。配置替换后需要重新加载网页 backend worker 和 JEV cuRobo 进程缓存，无需重建镜像。查看：[碰撞球页面](http://192.168.3.4:8340/collision-spheres.html)。

部署已验证：Windows 与 Ubuntu 配置 SHA 一致，cuRobo health 加载同一 SHA，网页 1051 个实际半径逐项一致；三个镜像 ID 和 72 个 src/vendor/skills 文件内容未变。网页 worker 重载后恢复了已确认场景和高级参数；蓝色圆柱重力重新落稳产生约 0.000786 mm 位移。

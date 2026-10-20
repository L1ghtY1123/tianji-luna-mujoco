# 离线单臂工作空间

机器人基座固定，Link_Base与任务世界坐标重合。左臂A、右臂B分别计算，另一臂不进入碰撞模型；本臂、张开夹爪、腕相机、底座、立柱及头部相机保留。头部相机的支架、外壳和窗口在URDF中属于Link_Stand，沿用同一份碰撞球配置及既有邻接忽略规则。

## 生成

在现有GPU环境中设置 `PYTHONPATH=/opt/jev/src:/opt/jev/vendor`，分别运行：

```sh
python3 /opt/jev/scripts/build_workspaces.py --arm A --output /evidence/maps
python3 /opt/jev/scripts/build_workspaces.py --arm B --output /evidence/maps
```

脚本使用当前锁定cuRobo公开API，不修改SDK或vendor。从第一活动关节位置和后续连杆长度推导包络球；球外不可能达到，球内每个1 cm网格点分别做自由朝向的位置IK，16/32个优化/初值种子，固定1024批量。末端为已有夹爪标定参考点，TCP_Link局部Z前移185 mm。SDK成功且独立FK位置残差≤2 mm才记录为可达。

输出arm_A.npz、arm_B.npz为运行时图；同目录JSON是统计与生成配置，*_witness.npz保存成功网格的关节解用于复核。生成完成后，仅将两张完整图安装至resources/workspaces/。--max-batches仅供性能探针，产生的incomplete文件不能被运行时加载。不同臂独立生成，不通过镜像假定它们一致。

## 使用

Algorithms.evaluate_reachability从当前RGBD取得源和目的实测中心，通过WorkspaceAtlas.query做最多四次网格查询。无需当前关节姿态、抓姿、场景构建或GPU服务。结果交给JEV选择执行臂；程序仅验证选择与绑定阶段约束。用户指定执行臂时保持该约束。

每个查询保留原位置、最近网格中心、映射距离、图分辨率、模型指纹与碰撞范围。1 cm三维网格的最大位置映射距离约8.66 mm，边界不做凸包或膨胀。查表是自由旋转的离散位置预筛选；不代表任意姿态可达，也不代表从当前位置存在无碰撞路径。图中未找到解不是数学证明全空间无解。

文件指纹绑定URDF（含关节限位）、机器人碰撞配置、工具标定与生成策略版本。缺失、模型不匹配、不完整或结构错误的文件直接报错，不回退在线IK。机器人模型改变后重建；物体移动或场景重建不影响该固定机器人知识。

历史 /tools/evaluate_reachability GPU接口保留用于明确的诊断探针；运行时同名skill已改为本地查表。正式抓放规划继续使用当前场景、自身与另一臂碰撞模型，忽略另一臂仅适用于离线选臂图。

初版生成记录见 `data/offline-workspace-20261014/`。2026-10-16 的四个夹爪 Link4 重拟合后，两臂按相同策略重新完整生成，记录与独立 MuJoCo FK 抽查位于 `data/current-failure-20261016/workspaces/`，结论见[当前验证记录](current-failure-validation-2026-10-16.md)。图文件绑定新的配置字节指纹，旧图不能直接改指纹后复用。

工作台主三维视图的「工作空间 ↗」打开独立三维查看页 /workspaces.html。左臂蓝色、右臂橙色，可独立显示并调整不透明度；场景跟随只更新机器人姿态。页面显示上述两张图的真实网格边界（含空洞），不重新计算可达性。显示接口 /api/workspaces 为只读，查看器不改变机器人或选臂结果。

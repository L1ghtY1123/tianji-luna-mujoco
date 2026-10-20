# 三相机抓放文档一致性复核

2026-10-09。本次仅修订 [README](../README.md) 与[参数说明](planning-parameters.md)，不修改控制代码、参数默认值或正在追加的[实施记录](superpowers/plans/2026-10-09-visible-cloud-progress.md)。依据为用户批准的[设计](superpowers/specs/2026-10-09-visible-cloud-grasp-place-design.md)、[实施方案](superpowers/plans/2026-10-09-visible-cloud-grasp-place.md)及当前运行链。

## 修正内容

- 抓取规划和接近/抓取执行检查使用固定张开夹爪球，保留原生三段规划；lift 按用户约定执行原有提升，不增加 attachment 或携物提升规划。lift 后显式观察确认持物，再建立实际 driver/mimic 闭合模型及持久 attachment。放置使用该模型，放置开爪另检查实际过程。删除了 README 中“执行检查全程按实测夹爪角度”的笼统表述。
- 明确固定开爪碰撞球查看器只展示基线模型；当前闭合模型、持物状态和 attachment 证据查看运行状态。原始球模板、SDK 和 YAML buffer 保持不变；cuRobo 派生镜像已增加 Open3D，不能再笼统声称算法镜像不变。
- 默认释放间隙仍为 0.01 m，不叠加碰撞余量。高度换算依赖抓取前水平桌面支撑、持物未滑移及恢复抓取姿态；可见点云最低点不是物体底面。不能满足碰撞要求时规划失败，不擅自抬高目标。
- 盒底必须有实际水平面片和有邻域支持的更高同实例测量上下文；孤立盒沿不能作为盒底。没有新增完整盒子、内腔或容量估计。
- 可见表面的 attachment 不保证不可见背面或完整体积的避障。持物视觉证据不足不算空手或成功；实际张开后 detach 仍保留 release_pending，最终放置验证通过才清除会话。场景应用、稳定和确认不能清除仍存在的持物记录。
- DBSCAN 只产生诊断标签，不改变保留掩码；同时启用 statistical/radius 时，其过滤仍然生效。链接[有界滤波报告](../data/visible-cloud-work/filter-assessment-report.txt)，明确四份无人工噪声 MuJoCo RGBD 记录、每份 20,000 点的预算，不外推为硬件相机或完整建图链验收。
- 补齐观察位姿、attachment 和过滤器高级设置的说明。观察关节请求先转换为 FK TCP 目标，冗余机械臂可能选择另一组达到目标的关节解，仍需实际 RGBD 覆盖验证。
- 原始 grasp 三段 hard clearance 与最终执行检查分别描述；最终 lift 不增加新的携物检查。09-28 历史物理记录保留日期与原始数值，明确不能证明当前 RGBD-only/attachment 改造通过。

## 实施记录与验收边界

实施记录按时间追加，较早的“实现中/复核待完成”保留为历史检查点，后续记录注明已完成的设置及生命周期修复；本次没有重写这些记录。README 和参数说明均明确：当前真实 JEV、SAM3、GraspGen、GPU cuRobo 与 MuJoCo 电机执行的完整抓放闭环仍在验收，不能用单元测试、离线 RGBD 重放或旧版本物理运行替代。浏览器视觉 QA 的已有未验证状态也没有被改为通过。

本次核对了参数 schema、观察目标转换、抓取后 attachment、持物检查、释放记录及释放高度计算的实际代码。对三份改动文档进行本地 Markdown 文件链接存在性检查，并检查旧的全程实测夹爪、镜像不变及 DBSCAN 保留全部原点表述。文档修改不涉及程序行为，没有重复执行与这些文本修改无关的算法测试，也没有启动 GPU 请求或重启服务。

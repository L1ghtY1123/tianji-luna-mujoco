# 身份、放置与完成验收迭代（2026-10-31）

当前版本 hybrid-23：A19 完整通过 A 臂 CAD 黑工件入左蓝盒任务（940.616秒，1次Responses、31次Decisions），包括交付后回观测位姿并观测、最终回初始位姿。Python全量4228 passed / 58 skipped，Ruff通过。本轮只在 MuJoCo 执行；仍有初始身份恢复重复观测，单次通过不能证明稳定性。

本次修复范围是 observe_scene 内同一采集的多视角物体关联。原实现可将头相机右盒和右腕相机右盒独立判成不同永久 ID，随后 CAD 几何检查正确拒绝，或地图缺少右盒而停止。不是放置求解器失败：A15 尚未进入抓取或放置。

Current three-camera identity association may group candidates before Decisions only when registered RGBD reprojection gives at least 512 unique visible pixels in each direction, depth agreement within 20mm and at least 95% overlap with the other SAM silhouette in both directions. Each group must be a complete mutual correspondence with at most one candidate per camera; ambiguous or partial chains remain independent. All members must have identical allowed ID choices. Decisions chooses the permanent ID once per group; other views inherit the unchanged answer and confidence through an explicit measured correspondence, never a fabricated independent model answer. Preserve original questions/answers, member-specific masks, RGBD/calibration hashes, observation/generation and grouping hash. Unknown remains unknown; visibility, CAD pose, map coverage and completion checks remain separate. No camera motion or additional model call is inserted.

## 冻结证据与边界

- A12、A15 的原始注册 RGBD 和 SAM 掩码双向重投影：C0↔C5 为右盒，C1↔C3 为左盒。右盒双向约98%，左盒双向约99%；右腕到头相机左盒重合率为0。
- 每方向至少512个唯一像素，20mm是当前不同视角表面的对应容差，不是运动硬净距，也不替代 CAD 的更严格位姿验证。
- 仅补充投影文字或在问题中补图片短标签的模型对照没有解决错误，不采用。
- 分组后的冻结模型对照，A12右盒0.93、左盒0.96；A15右盒仍0.55，保留unknown。分组防止已测定同一表面的成员独立产生矛盾身份，并减少重复问题，不保证所有永久ID都能识别。
- 不修改初始 Responses 的 graph，不添加运行中 Responses，不降低身份概率门槛，不隐藏移动或重试。
- CAD asset ID 与永久实例 ID 分离；分组不直接匹配 CAD。FoundationPose 仍由已关联的当前掩码选相机估计，再跨相关视角数值验证。

## 验证

接口回归覆盖直接 choice、一次 pairwise 澄清、原始模型回答与继承来源；几何回归覆盖过期帧、遮挡、不同表面、像素不足和同相机歧义。生产算法对冻结A12/A15数据的回放结果记录在 data/cad-obb-iteration-20261031/multiview-production-replay.json。

实际运行、失败修复和最终验收按下面的版本与运行编号分开记录；冻结诊断不代替完整物理任务。

## A16 实际运行与反馈边界修复

hybrid-21 全量测试 4221 passed / 58 skipped，Ruff通过，955文件校验同步。A16（65586b2dcabd47008a970186b4973cda）1次Responses、17次Decisions，约350秒后停止，未执行放置，不计成功。

实际请求把6个候选压成4个身份问题；初次8题、一次澄清13题。右盒/左盒分别确认，绑定不同实例ID但共用blue_box_410x310x145；FoundationPose分别采用相机1、2，黑色工件THREADED_VISUAL_SAMPLE_MM采用相机3。A臂CAD离线抓姿筛选182个；抓取、lift、持物核验、attachment及resolve_placement成功。

plan_place四次返回placement_resolution_changed。冻结第一组输入回放证明：签名正确，全部非反馈字段字节含义一致，只有robot_feedback变化。A臂最大关节变化5.364751975989179e-10 rad，夹爪变化8.53446427689919e-6 rad；超过的是缓存一致性的1e-6 rad，不是碰撞/伺服容差。完整记录见black-A16-audit/resolution-binding-replay.json。

修复将固定落点提案与旧反馈证明分开：validate_resolution默认仍严格验证原反馈；plan_place允许仅反馈不同的已签名候选进入新的当前反馈规划。原XYZ/姿态、语义许可、地图、attachment、设置和哈希均保持不变，非反馈变化仍拒绝。旧held-IK证明明确保留为历史；只有原有公开cuRobo规划、当前双臂/夹爪/携物碰撞、自碰撞、关节约束、achieved FK、三段命令检查都通过，才返回计划。实际执行仍重新检查当前反馈和释放条件。没有额外Responses、隐藏运动或碰撞阈值放宽。

针对性回归覆盖真实A16夹爪数值、另一臂变化、候选不可变性和新路径碰撞拒绝。真实服务纯规划回放与下一次完整任务结果另行追加。

A16原始失败输入的真实cuRobo纯规划回放成功：11.532秒，第1/12候选，approach/place/retreat三段collision_free=true；未执行机器人动作。结果归档black-A16-audit/plan-replay-summary.json和plan-replay-result.json。

## A17 完整任务的后观测阻塞

hybrid-22全量测试4224 passed / 58 skipped，Ruff通过，956文件校验同步。A17（d3b9ac84a2a84c9792f435f1278ff7f6）1次Responses、38次Decisions，约610.6秒后needs_attention。CAD抓取、lift后attachment、落点解析、三段plan_place、实际松爪撤退、verify_place均成功；未再出现placement_resolution_changed。完整任务仍未通过。

交付后先观测并建图，再回观测位姿。该位姿下黑色工件的当前对应未确认，merge_completed_correspondence把已交付物体的未知候选变成整次observe异常，与普通已冻结清单允许部分观测的规则不一致。一次恢复当前身份通过后，黑工件CAD可见深度/轮廓一致率0.8282低于0.9，正确拒绝；没有放宽CAD阈值。最终完成判断明确缺少“交付后回观测位姿并成功观测”的记录，因此没有把交付成功记成整体成功。

修复仅针对已冻结清单且已经独立核验最终交付的ID：其当前身份不确定时保留missing_geometry_ids/unobserved_completed_ids，返回其他已确认的当前测量。未知掩码不加入几何，complete重新计算为false，历史CAD/OBB仍带旧来源，地图独立检查缺失/失效；首次核验及中间放置不适用。完成条件仍由当前图执行记录、实际关节和最终Decisions检查，不自动宣布成功。

冻结A17事件357的两次原始Decisions回答完整回放，问题不变、没有新模型调用：现在返回两个确认盒子，黑工件列为当前几何缺失；未进行CAD推理、建图或运动。记录black-A17-audit/partial-association-replay.json。相关113项测试通过。


## A18 等待超时与 A19 完整验收

hybrid-23 全量测试4228 passed / 58 skipped，1个Starlette/httpx弃用警告；Ruff通过。957个部署文件经SHA-256逐个验证。前端本轮未改，388项通过属于此前版本记录。

A18（1cbcc276b0ed4b96b09da9ebe391fc04）约246.5秒，1次Responses、7次Decisions，闭爪阶段gripper_stability_timeout而停止；尚未lift或放置。截止3.001秒时瞬时速度0.010951 rad/s虽低于0.015门槛，但尚未满足连续0.2秒稳定窗口，不能认定夹持完成。

A19前仅通过配置API把gripper_timeout_s从3改为6秒，配置revision=d350b90197c81638；源码默认仍为3秒。稳定速度0.015 rad/s、稳定窗口0.2秒、位置容差及碰撞阈值均未变。配置前后对照在data/cad-obb-iteration-20261031/gripper-timeout-A19.json。

A19（d7ee8cb848ee412bae875446a3c82aa4）最终succeeded，940.616秒，1次Responses、31次Decisions，无归档错误。实际执行证据：

- 永久黑件ID instance-e5d5387945fb4760；左目的盒 instance-5715b204ea2c4e43；右盒 instance-17a0383065634b10。同类盒子共享CAD资产而不共享实例ID。
- 原始首次观测identity unknown；恢复观测完成三个CAD的当前几何后，Decisions又安排两次观测，再继续初始graph，没有额外Responses或重复物理搬运。
- A臂离线抓姿、实际抓取、lift后attachment、resolve_placement、三段plan_place、实际放置与verify_place均成功。闭爪0.424秒，连续稳定0.201秒，最终速度0.001614 rad/s；本次未接近3秒期限，因此不能用这次成功单独证明6秒等待能解决所有闭爪超时。
- 交付核验后事件400观测只确认黑件，两个盒子列为缺失；图显式建图并回观测位姿。事件436观测确认两个盒子，已完成黑件当前身份未确认：missing_geometry_ids与unobserved_completed_ids均保留黑件，identity_coverage.complete=false。没有把旧CAD当作新测量。
- 事件439取得交付后观测凭证，绑定当前RGBD、generation=4、action_epoch=20和双臂实际反馈。这证明按要求回位并观测，未宣称三个物体本次全部重新识别。
- return_to_initial_pose和verify_task_completion通过。实际双臂最大关节误差各3.97438e-6度；最终8项Decisions谓词均为1.0，原身份/CAD/轨迹门槛保持。

完整索引：data/cad-obb-iteration-20261031/black-A19-audit/evidence-index.json；原始运行状态：data/cad-obb-iteration-20261030/black-A19-latest.json。Ubuntu保留同一run_id的完整实验归档。

## 尚未解决的效率与覆盖范围

A19首次恢复观测在317.586秒已成功，但后续两次观测直到648.790秒才完成。恢复continuation输入55161 tokens，后续capability输入59073 tokens；历史失败及当前结果含大量嵌套证据。记录能证明额外观测和上下文规模，尚不能证明输入长度是模型选择重复观测的原因。本轮没有依据一次成功继续修改恢复策略或替模型选择动作。

本轮新增完整验收覆盖A臂CAD入盒；未知OBB+A臂桌面和CAD+B臂入盒的完整成功仍是C6/hybrid-16和B6/hybrid-14历史记录，没有在hybrid-23重新验证。多物体连续搬运和其他自然语言相对区域亦不能由本次单物体任务推定通过。

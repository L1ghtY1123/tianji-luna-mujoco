# 抓前建图、规划初始化与请求语义核查

## 本轮范围

保持既有16个skill及JEV恢复决策。抓取和放置必须先IK100筛选，再规划真实通过候选；GraspGen使用三相机可见目标点融合。每轮抓前双臂观测和建图，持物阶段只复用保存场景。位置维持N=1中心、N>1最小偶数等面积分区的面积中心，18 yaw、attachment球底15 mm释放、10 mm硬间距。完整复测另发现安装端TCP与用户要求的夹爪末端混淆，修正为既有185 mm夹持参考点XY固定，保持SDK安装端坐标定义；定量依据见后文。

SDK锁定`8e734f3ced1df898990bcd92de40abce475907db`；本轮只修改项目适配器、JEV问题和相关测试/文档，不修改SDK/vendor、碰撞球或镜像。类型配置复用依据是锁定版公开`InverseKinematicsCfg.create`/`MotionPlannerCfg.create`/`RobotCollisionCheckerCfg.load_from_config`接收RobotCfg的合同，以及官方`test_motion_planner.py`、`test_motion_planner_batch.py`中IK/attachment模型共享断言。感知使用公开Kinematics、RobotSegmenter、JointState及URDF mimic支持。

## 根因与改变

前次最慢实例建图API64.020 s中，分割模型初始化60.387 s、Mapper2.986 s；不能把总时长称为ESDF生成耗时。2026-10-16另起隔离进程剖析原版本：第一次创建分割边界5.785 s，反馈改变后再次创建4.895 s。第二次的两次Kinematics解析占4.433 s，包含2280次torch.nonzero；原生边界与项目RawBoundary各建一次模型。共享GPU不同负载的绝对耗时不可直接比较。

原规划配置先解析IK机器人，再从同一字典重新解析轨迹机器人，hard checker再次解析。现在同一后端只解析一次并按公开接口传递类型化RobotCfg；模型共享还使IK、TrajOpt、graph和hard checker使用同一组已测球。不同后端不共享可变机器人配置。

临时支撑轴撤离原先占用open缓存，替换普通双臂/抓取规划器。现在该配置独立缓存；释放清理仍清除held和support，保留有效open配置。缓存键按实际球和夹爪反馈匹配，不按owner或观测ID重复建模；逻辑身份、场景token和generation仍逐次校验。几何变化不会复用旧模型。

持物时`build_scene`在契约和执行入口均拒绝。`retry_place`只有在保存抓前场景仍有效时允许；否则保留原失败并返回`retry_place_requires_saved_grasp_scene`，由JEV继续判断。

## JEV语义验证

真实JEV新探针基线12/16；修改后17/17，既有范围探针10/10。修复重复单数表达遗漏总数候选、源盒颜色误当普通环境物体、并行问题读取其他问题答案的描述。验证包括源/目的/目标特征、环境数量与搬运数量、指定臂、并列与明确顺序、分组目的地、同物体两段路线及2红1绿配额。

证据在`data/request-semantics-20261016/`，包含state、questions、概率和结果，不包含密钥。探针使用标注为合成测量的清单fixture，不是SAM3或机器人运动验收。

## 验证状态

针对缓存、规划配置和场景复用的新回归均先复现失败后修复。初轮Python回归1563通过、54跳过；独立审查115项定向回归通过。审查另补充过滤后单帧PNG的uint8零RGB输入，34项感知回归通过，避免旧兼容Mapper接口拿到None。后续增加标记污染残片修复及审查反例后，全量Python回归1577通过、54跳过（105.02 s）。

16DOF感知边界使用保存三路RGBD与4组夹爪反馈对照旧锁定模型：349球半径逐值相同，球心最大差3.58e-7 m，机器人掩码与剩余深度逐像素相同；crop/stack输出一致。构造0.589 s，反馈更新1.58–3.02 ms，Kinematics及各相机RobotSegmenter对象复用。真实Mapper烟测open→held→released通过，反馈球位移38.15 mm、非目标障碍保留、释放源层恢复；这些held/released低层探针不授权runtime持物建图。

持物GPU探针在无环境障碍的受控输入上：首次prepare2.978 s、重复prepare18.5 ms、首次IK1.539 s、轨迹后端创建1.559 s。IK、TrajOpt、graph和hard checker的参数共享及上传球逐值一致；峰值reserved约1262 MiB。这是模型/生命周期验证，不是实际搬运时间。双臂14关节规划返回61个同步样本、20 ms间隔；引入新场景token与腕部障碍后同一碰撞检查正确拒绝。旧测试只改障碍而不更新token导致旧版与新版均假失败，测试已按真实发布合同修正。

第一次完整候选复测使用原来的双物体近似物理fixture与原指令，控制仅接收RGBD与自身反馈，真值仅事后评分。总128.97 s，第一件已完成抓取/放置；第二轮因placed_object_not_observed结束为partial_success，完整双物体验收尚未通过。该次实测如下，不能与不同共享负载下的旧运行直接作速度倍率比较：

| 阶段 | 耗时 |
|---|---:|
| 初始避障图（含首次模型创建） | 3.809 s |
| 首轮build_scene完整skill | 16.205 s |
| 其中双臂观测移动（规划/复核/执行） | 10.265 s |
| 其中RGBD/SAM3观测 | 4.952 s |
| 其中实例场景阶段/API | 0.977 / 0.878 s |
| plan_grasp | 11.636 s |
| verify_grasp | 0.079 s |
| prepare_attachment | 1.488 s |
| plan_place完整skill | 14.441 s |
| verify_place | 0.00475 s |

感知证据为`data/articulated-segmenter-20261016/`；规划和首轮完整计时为`data/current-failure-20261016/pipeline-*.json`。Ubuntu完整RGBD、SAM3、算法请求与事件保存在`tianji-jev-mujoco-candidates/pipeline-initialization-20261016/first-run/`。隔离GPU进程已结束，生产服务未被该复测调用；部署待修复关联问题后记录。

第二轮失败由当前SAM3的重复片段引起：头部block掩码9651像素中9325属于黄色标记；去除标记和轮廓后剩105个绿色物体点，另一个三视角实例已测得2999点。两个候选都落在目的区域，正确的一对一约束拒绝关联；旧错误分类却先报“未观测到”。新增去重仅针对原始掩码至少95%为实测标记的过滤残片，同时要求同类同色、原始像素包含、5 mm注册表面支持及唯一主体；全部片段点必须在主体5 mm邻域内，合并时保留独有实测点、像素和来源。独立审查提出的远端独有点与污染大掩码包围相邻物体反例已加入回归。跨时一对一、目的区域支持和已放物体障碍检查均保留；有未分配候选的歧义优先报告visibility_unresolved。具体阈值见统一skill手册，保存帧重放证据在`data/association-replay-20261016/`。

## 夹爪末端参考修正

第二次完整复测138.91 s：完成第一件，正确关联并建立第二轮场景，第二件抓取后所有放置候选因区域边界拒绝；完整任务仍未通过。首/第二轮build_scene分别12.25/16.03 s，观测运动9.11/9.03 s、RGBD/SAM3 2.25/2.45 s、实例API0.779/4.436 s。后者新网格首次积分/编译占主要时间，已有形状复用的两层TSDF约25 ms、ESDF约1.3–1.5 ms；分割模型更新分别1.55/2.74 ms。不能把部分层暖态耗时当作完整build_scene。两件首次attachment准备约1.72/1.55 s；随后重试复用且未重建场景。记录存于Ubuntu同目录`final-run/`（实际为第二次候选复测）。

拒绝源于位置坐标含义：代码T_world_tcp指TCP_Link安装端，而夹持末端沿工具Z轴185 mm。红物实测尺寸约53.47×47.62×56.95 mm，工具系可见质心约[-0.833,-6.989,185.423] mm；倾斜抓取下，质心相对安装端XY偏移161.47 mm。实际闭爪TCP到计划仅0.455 mm/0.049°，URDF末端FK与计划一致，不能归因执行不到位。即使不用包球，只看原始点，固定安装端XY的两处最优yaw仍越界约43.7/43.9 mm；16个球半径15.18–24.98 mm不是主因。

现在用既有TOOL_PROFILE的固定g=[0,0,0.185]米夹持参考对齐分区中心：TCP_XY=P_XY−(R*g)_XY；TCP_Z仍按上传attachment球底距支撑面15 mm求解，保持roll/pitch，仅改变yaw。没有对齐可见点质心，也未改变SDK、标定、区域或安全余量。当前位置证据记录安装端frame、工具配置版本、局部参考点及目标参考XY；开爪前同时检查安装端位姿、夹持参考XY和球底间距，位置容差仍2 mm。

按原第二件实测输入CPU复算，上分区11/18 yaw通过全部原边界和占用检查；最佳边界净余量10.22 mm、占用净余量17.78 mm。下分区11个过边界者全部因绿色已放物体占用被拒绝。全体保留候选球底为15 mm，工具参考XY误差小于1.4e-17 m；这不是实际运动验收。证据为`placement-frame-diagnosis.json`和`gripper-reference-cpu-replay.json`；独立oracle只作事后对照，未进入规划或决策。

## 最终完整复测

第三次使用同一指令、同一物理fixture及最终代码，真实JEV、SAM3、GraspGen、cuRobo和MuJoCo完整运行157.745 s，任务`succeeded`，绿色和红色两项均completed，最终空手。黑色普通物体只作障碍，未进入搬运清单。独立事后评分确认两件物体最终边界均在黄色区域内，黑色障碍物位置未变；评分真值没有进入控制或碰撞场景。

| 阶段 | 第一件 | 第二件 |
|---|---:|---:|
| build_scene完整skill | 12.13 s | 13.88 s |
| 其中双臂观测移动（规划、复核、执行） | 9.04 s | 7.82 s |
| 其中RGBD/SAM3 | 2.23 s | 2.44 s |
| 实例场景API | 0.752 s | 3.501 s |
| 感知机器人模型反馈更新 | 2.03 ms | 1.99 ms |
| prepare_attachment阶段 | 1.64 s | 1.43 s |
| plan_grasp完整skill | 8.90 s | 7.65 s |
| plan_place完整skill | 15.22 s | 14.64 s |

独立实例仍按已有4个工作线程/独立CUDA流有界并行，完成后原子发布。不能把并行误解为没有首次网格编译和主机准备开销；第二轮实例API包括新网格形状首次编译。复用桌面耗时0.21/0.27 ms，没有重新RANSAC建桌面。此次总耗时包括两件完整搬运及初始观测，不能和先前一件任务或不同GPU负载直接计算速度提升倍率。轨迹规划及运动等待仍有明显耗时，本轮没有证明所有速度问题已解决。

调用记录只有3次建图：初始避障图、两轮抓前实例场景；持物后及place期间均为0次。两轮grasp的IK均为100+100，轨迹容量16；place先对位置及占用过滤后剩余18/11 yaw执行IK，分别18/5个通过，再做轨迹规划。固定16槽内不足时重复已通过IK的候选，回传后只读取真实条目，不把填充算成新的候选。每个送入轨迹的位姿均能对应此前成功IK结果。

两次放置均保持目标总数N=2、2个面积中心、18 yaw；开爪前由实际关节反馈计算attachment球底间距14.148/14.126 mm，满足15 mm目标±2 mm容差，且高于10 mm硬间距。夹持参考点XY误差1.087/0.831 mm；安装端与姿态检查同时通过。

最终`python -m pytest -q`：1583 passed、54 skipped，96.11 s；`python -m ruff check src tests`通过，`git diff --check`通过。54项跳过不能当作GPU通过项，硬件验证采用前述独立GPU探针及此次完整运行。当前完整物理验收覆盖“两个不同物体盒子→桌面”，其余路线有回归覆盖，本轮未重新做全部路线的完整GPU实机仿真验收。

完整证据位于Ubuntu `tianji-jev-mujoco-candidates/pipeline-initialization-20261016/reference-run/`；本地摘要位于`data/current-failure-20261016/pipeline-reference-*.json`，完整测试日志为`pytest-reference-final.log`。统一skill手册及相关SKILL.md已同步更新，注册skill数量仍为16。

## Ubuntu切换记录

实现提交`9bc8bc81c97e6d9eb4a6ff19944cd6df1a389c38`已快进同步至本地`C:/project/tianji-jev-mujoco`及Ubuntu `/home/guangyu/projects/tianji-jev-mujoco`。切换前服务为空闲、无活动指令，已保存状态、GPU健康、Git与容器身份；仅重启现有cuRobo容器，挂载代码的网页后端自动重载。切换后GPU ready、任务idle、网页HTTP200，53个Python源文件经统一换行后与成功完整复测版本逐个一致。

仍使用`jodell-curobo:0.2.0-guangyu`和`tianji-jev-mujoco:0.4.0-experiments-guangyu`，前后容器ID与镜像ID逐项相同，未构建镜像或安装依赖。机器人配置SHA256仍为`3b2566219c97f5570f3d5c1dec1618f08b215004d29012a36f9d9476255debe6`；SDK、标定、碰撞球均未修改。隔离GPU测试进程已停止，临时受保护配置符号链接已移除，真实配置保留。切换证据保存在同目录`activation/`。

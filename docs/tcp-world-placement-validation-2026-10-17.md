# TCP 世界坐标放置与固定场景清单验证

> 后续要求已变更：本文旧实跑的8–10 mm球底间隙不满足新确认的至少10 mm要求，且旧XY记录对应安装端。不得将旧实跑作为前端TCP及新间隙要求的通过证据。新验证在文末续记。

2026-10-17。用户最新澄清优先于旧方案：固定 TCP_Link 原点的世界 XY，旋转为 Rz(世界 yaw) @ grasp TCP 旋转；根据实际上传 attachment 球底求世界 Z。目标间隙直接等于碰撞阈值，默认10 mm；实际开爪前间隙必须在0至该阈值内（仅1e-9 m数值容差），2 mm跟踪容差不能抬高上界。删除185 mm工具点反算XY、独立15 mm释放高度、attachment投影边界及二维占用限制。三维碰撞、自碰撞与关节限制保留。

## 已实施行为

- 初始头部固定全场景实体及全局ID，搬运清单引用其中的成员。后续SAM3只使用初始实体查询，RGB和注册深度共同关联身份，三路相机取可见证据并集，不要求每路都看见全部实体。
- observe_scene/build_scene以及verify_place内部放后更新身份缺失或歧义保留原始错误和关联证据，以needs_attention结束，不反复recover。
- 释放、空爪撤离与反馈确认后，若还有required成员或后续阶段，立即原位三相机建全障碍图，再允许双臂到观测位姿并重建当前目标排除图。桌面仅初建一次。
- 冻结清单的已验证完成记录用于任务结束判断；旧源图或额外视觉数量unknown不能撤销已确认释放。
- 当前阶段选臂使用显式x/y/z起终点和完整离线查表证据。两臂位置均可达时优先当前起终点偏侧端同侧臂，再比较距离；用户约束与阶段绑定继续优先。选择仍由JEV作出，不增加失败后自动换臂流程。
- 工作台Z/yaw同行，新增取消未应用草稿，错误卡位于右侧且可展开/关闭；取消不修改仿真状态。

## 18个yaw失败的定位

不能把所有失败归为识别或“工作空间不可达”。本轮保存输入证明了几个不同问题：

1. **额外二维限制**：旧代码以attachment投影是否完整落入分区淘汰候选，这是用户未要求的条件，已删除。现在固定TCP XY生成18个候选，实际碰撞留给三维检查。
2. **盒底区域被采样密度缩窄**：三相机点云密度不同，旧逐格近邻门槛曾把可见完整底面缩成约3.7 cm窄条。改为实测同一底面凸包内接矩形，不用局部采样稀疏制造孔洞；外边界不外推。首次plan_place使用三相机几何建立分区，之后固定编号和XY。
3. **盒底ESDF量化上偏**：保存的实测底面约0.609999 m，旧10 mm网格零面约0.615 m，产生约5 mm上偏。普通实例保留10 mm；有实测底板的盒子通过官方MapperCfg使用2 mm TSDF/ESDF，并将Z采样中心对齐底板。没有平移实测深度、删盒壁或改写距离场。
4. **IK后严格零比较误拒绝**：中转trial3的完整持物状态为0/18；只在诊断中移除目的盒子后为18/18。原地图下100°/260°的完整持物构型通过独立RobotCollisionChecker，但Tianji适配器IK后额外要求world activation cost严格为零，球底约4.55e-11的浮点残差被拒绝。项目侧公开RobotCollisionChecker子类将非负且≤1e-9的cost归零，与已有轨迹检查一致。同一原始目标、attachment和完整地图重放后为2/18，随后完整中转成功。1e-9是代价数值容差，不是米制间距；SDK优化器和收敛要求不变。
5. **阶段选臂信息混淆**：旧中转第二段仍选左臂。相同放置目标移除目的盒子的诊断为A 0/18、B 18/18；离线自由姿态位置工作空间不能保证固定抓取姿态可达。当前段端点与选项含义修正后，原状态及左右镜像共12次真实JEV回放均选择对应侧臂。

诊断移除障碍、去除attachment和更换种子均只用于隔离问题，没有进入运行流程。SDK/vendor保持只读，失败试次保留。

用户原场景trial4还暴露了独立的放后关联限制：释放合格（8.64166 mm，TCP XY误差0.72557 mm），但倾斜抓取使红物体落在黄色区域边缘外，旧目的区域投影门槛拒绝其ID。已改为实际抓取/释放TCP变换预测加当前RGBD匹配，放后参考只保留实测表面。相同保存帧重放5个实体全部匹配，红物体RGB距离0.00522、5 mm双向表面支持72.78%，无需区域投影条件；后续实测参考再次5/5匹配。预测不进入碰撞图。

## 实际仿真验证

全部使用真实JEV、SAM3、GraspGen、cuRobo和MuJoCo。仿真真值只用于设置隔离场景及事后评分，不进入控制链。XY通过条件检查释放时TCP，而不是物体投影；独立物体投影仅保留诊断值。

| 试次 | 最终结果 | 放置IK | 实际attachment底部间隙 | TCP XY误差 |
|---|---|---|---|---|
| 用户原场景最终版：user-build-scene-20261016/trial5，红绿两物体盒→桌 | succeeded，2件完成 | 18/18、18/18 | 8.92492、9.25975 mm | 0.71979、0.58403 mm |
| 用户原场景此前版本：user-build-scene-20261016/trial3，红绿两物体盒→桌 | succeeded，2件完成 | 18/18、18/18 | 8.87685、9.25022 mm | 0.68748、0.60521 mm |
| tcp-world-table-box-n1-trial3，桌→盒 | succeeded | 4/18 | 8.68181 mm | 0.80358 mm |
| tcp-world-box-table-box-n1-trial4，盒→桌→另一盒 | succeeded，同一ID，A→B | 18/18、2/18 | 8.88766、8.59660 mm | 0.72125、0.83809 mm |

用户最终trial5和中转trial4的8项审计全部通过：固定全场景ID、初始仅头部、后续三相机、桌面只建一次且revision不变、每次双臂观测运动前全部物体为障碍、请求间隙等于阈值、实际间隙不超过阈值。独立物理评分通过，最终物体底面距右盒底约-0.00179 mm。该试次初始assess_task曾返回unknown并由JEV显式补观测一次，后续完成；不是零恢复成功率证明。

用户trial3固定6个实体（含桌面），后续查询仅black block、blue open box、green block、red block。地图顺序为初始头部全障碍、三相机目标排除、放后原位三相机全障碍、第二次三相机目标排除。桌→盒trial3独立落稳评分也通过。

中转trial4及桌→盒trial3在最终释放身份修正之前完成；该修正有保存失败RGBD回放、真实双物体trial5以及完整Python回归覆盖，不声称三条路线都在最后一次代码变更后重复运动验收。

此前用户trial1被并行pytest清理GPU场景打断，不能作为运动验收；trial2完成释放但旧完成判断返回unknown。盒内trial1/2/3的失败与诊断均保留，没有改写为成功。完整pytest与实际GPU试验现串行执行。

## SDK依据与工程验证

锁定cuRobo提交：8e734f3ced1df898990bcd92de40abce475907db。只读核对官方MapperCfg/Mapper、getting_started/volumetric_mapping.py、公开collision_checking模块、RobotCollisionChecker.get_collision_distance，以及IK配置/求解的收敛语义。项目仅配置公开网格API并在独立checker边界统一数值零判断；未修改安装包、vendor或机器人碰撞资源。JEV问题按[官方Choice规范](https://docs.typesafe.ai/primitives/choice)区分选项含义，按[State规范](https://docs.typesafe.ai/concepts/state)提供当前阶段证据。

最终Ubuntu完整回归：python -m pytest -q，1769通过、10跳过（73.00 s），两项警告仅为只读挂载无法写pytest缓存。最新身份/路线定向64通过；前端node --test web/tests/*.test.js为260通过，Vite构建通过，仅保留既有大于500 kB的bundle提示；Ruff与git diff --check通过。已有10月16日桌面/窄窗口截图已复核；本次浏览器连接工具超时，未重新做浏览器视觉验收。完整N=1/3/4、多次重复、各臂组合矩阵尚未全部验收。

## 证据与部署

Ubuntu证据根目录：/home/guangyu/projects/tianji-jev-acceptance-20261016/。各试次含result.json、task-progress.json、真实RGBD/SAM3、GPU请求、JEV决策与tcp-world-audit.json；标准夹具另含physical-score.json。中转trial3包含ik-carried-world-probe.json、ik-endpoint-validation-probe.json、esdf-diagnostic/esdf-probe.json及ik-numerical-tolerance-replay.json。

候选基于dad14b5及本轮未提交修订；原生产Git基线d88b95f。此前122文件部署记录为activation-tcp-world-20261017；最终129文件已同步至Windows C:/project/tianji-jev-mujoco和Ubuntu /home/guangyu/projects/tianji-jev-mujoco，SHA256逐文件核验通过。最终备份、manifest、场景/参数与健康记录位于activation-floor-aligned-20261017，Windows备份位于候选目录.superpowers/sdd/2026-10-16-requirements-alignment/local-before-final-activation。

现有网页及cuRobo容器已重启，保持同一容器ID和镜像ID；原用户场景已恢复并确认，任务idle，默认阈值/派生释放间隙均为10 mm。cuRobo、GraspGen、MuJoCo就绪，两个容器healthy；JEV本地配置ready（健康接口不额外请求模型，真实API已由上述试跑验证）。本次未构建镜像，隔离测试容器已停止并保留证据。SDK/vendor、assets和resources没有本轮差异。



## 后续修正：前端 TCP 与最小间隙（2026-10-17）

用户确认目标底面同样必须保持至少10 mm，因此此前8–10 mm实跑只作为历史失败证据。本次没有豁免盒底、缩小attachment球或放宽碰撞阈值。

- URDF/MuJoCo核对：TCP_Link_L/R在夹爪安装端；现有工具标定沿局部Z前移185 mm为两指前端固定参考点。张开时手指网格前端约183 mm，闭合时指尖随机构移动，工具TCP不随开合漂移。
- 放置约束前端点世界XY。SDK目标和attachment球继续使用安装端系，通过显式平移换算；现有grasp转换、模型、SDK和离线工作空间均不改。观察与网页额外提供tcp_front，显示A/B坐标轴和白色原点，可开启/关闭。
- 球底规划间隙=hard clearance+2 mm，实测要求[hard clearance,hard clearance+4 mm]且与目标相差≤2 mm；默认12 mm、允许10–14 mm。前端三维位置/XY误差≤2 mm、姿态≤2°。目标XY不要求attachment投影被区域包含。
- execute_place在下降及停留后，用同一次反馈检查前端TCP/球底，再调用已有完整构型检查器复核当前双臂关节、两侧夹爪和attachment，合格才开爪。拒绝路径已用测试证明保持夹持与attachment。

隔离容器沿用现有镜像，无新增镜像。以下使用真实JEV、SAM3、RGBD、GraspGen、cuRobo和MuJoCo；均完整成功：

| 场景 | 放置IK通过 | 实测球底间隙 | 前端TCP XY误差 |
|---|---:|---:|---:|
| 用户原场景，绿色/红色物体第一次放桌面 | 18/18 | 10.922737 mm | 1.192577 mm |
| 同场景第二次放桌面 | 18/18 | 11.143651 mm | 1.012393 mm |
| 桌面单物体放盒内 | 6/18 | 10.681211 mm | 1.284888 mm |

证据根目录 `/home/guangyu/projects/tianji-jev-acceptance-20261016/`；原场景 `user-build-scene-20261016/front-tcp-trial1`，盒内 `front-tcp-table-box-trial1`。各自的 `front-tcp-audit.json` 10项检查全部为true：固定清单、首次仅头部、更新三相机、桌面仅建一次、桌面版本不变、观测运动前全障碍图、新目标间隙、实际最小间隙、前端TCP引用，以及松爪前单点完整构型碰撞检查。原场景包含2个搬运成员、6个全场景实体（含桌面），完成记录不增添重复成员。

网页261项Node测试与Vite构建通过；保留既有大于500 kB的打包提示。浏览器肉眼验证未完成：内置浏览器连接超时；Windows Computer Use因无法可靠确认当前浏览器URL主动停止，本轮未继续UI操作。不能将自动测试替代为已完成截图验收。

完整Linux回归：`python -m pytest -q --tb=short -o cache_dir=/opt/acceptance/front-tcp-pytest-cache`，1778 passed、10 skipped，81.31 s；日志 `front-tcp-full-pytest-final.log`。Ruff通过。

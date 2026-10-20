# Tianji JEV MuJoCo

唯一项目目录：Windows `C:/project/tianji-jev-mujoco`，Ubuntu `/home/guangyu/projects/tianji-jev-mujoco`。此前分散到 `requirements` 工作树的代码已合入本主目录；合并前文件、Git 历史和实验资料保存在 `data/consolidation/2026-10-19/`，不再使用平行项目目录。

**Skill 说明入口：[JEV skill 使用与实现手册](docs/jev-skills-guide.md)**。逐项解释当前全部 **16 个注册运行时 skill** 的用途、前提、固定算法、输出及失败边界；已删除的9个独立入口及对应runner分派不能再被JEV调用。每次代码修改同步复核手册、contract 与运行时说明，维护要求见 [AGENTS.md](AGENTS.md)。

当前主目录已清理旧身份关联与未注册runner入口，Decisions结果和外层JEV用途选择为必需；SAM3分割、RGBD测量和碰撞几何保留。历史清理范围见[清理验收](docs/identity-cleanup-validation-2026-10-18.md)。最新真实 Decisions 请求遇到 token 限流；代码合并和本地测试不代表新视觉链已通过整轮搬运验收，以下历史验证链接仅说明对应版本。

独立仿真项目：JEV 通过 Pydantic AI 做有限选项决策，skills 调用 SAM3、GraspGen、cuRobo 和 MuJoCo，支持同臂抓取、抓取并放置和已持物放置。控制链的环境信息来自头部和双腕三个注册 RGBD 相机；机器人关节、夹爪反馈、FK 和标定属于自身测量。MuJoCo 始终由独立线程推进物理时间。网页三维画面和场景编辑器显示仿真几何，不能向 JEV 或控制链提供物体真值。
连续搬运支持盒子→桌面、桌面→盒子、同一物体盒子→桌面→另一个盒子，支持全部或部分目标。初始 observe_scene 固定头部看到的全场景实体清单：搬运候选、障碍、盒子、桌面及目标区域均有全局 ID。JEV 确认其中的搬运成员与分组数量后，按初始头部图像从上到下、行内从左到右冻结搬运编号。此后SAM3只使用清单内的初始分割描述，Decisions确认固定全局ID，注册RGBD提供当前实测几何，不再探索新类别。每轮 build_scene 完成双臂到观测位姿、三相机关联和碰撞重建。放置、空爪撤离及释放反馈确认后，若还有搬运阶段，立即原位用三相机更新包含刚放物体的全障碍图，再允许双臂返回观测位姿。桌面只初建一次，其余仅更新 SAM3 实体；缺失/歧义返回证据，不补历史预测体素。当前流程以[手册](docs/jev-skills-guide.md)为准，旧实验记录仅代表对应版本。

默认增加中心 `[0.38, 0, 0.60]` 米、20×20 cm 黄色桌面区域；普通物体不再使用蓝色。选臂查询源与目的中心是否落在各臂的离线工作空间，不依赖抓姿或当前碰撞场景，两臂均可达时，JEV按当前阶段起终点的侧向分布优先同侧臂，再比较距离；同阶段固定执行臂。用户指定臂时直接绑定。中心可达不等于实际抓姿或整条路径可行，运动规划仍保留碰撞检查。

建图保持 skill 触发，不连续更新。SAM3 实例（含盒子）统一建立 2 cm TSDF/ESDF，单侧 halo 按最大机器人/attachment 球半径、实际碰撞激活/硬间距及固定 2 cm 余量计算并向外对齐。静态 RANSAC 桌面 cuboid 仅初始头部拟合，场景生命周期内复用，上表面不抬高，裁掉桌面以下点/体素。头部运动图和 selected 图均只保留实例和桌面；有限环境查询仍漏分割的背景会被删除，历史对象、声明障碍或关联缺口必须拒绝发布。初始 observation 图仅使用头部全部语义实体且不排除目标；持物不重建图。项目适配采用独立 Mapper/独立 Torch stream、最多 4 实例并发、compact ESDF 与语义空背景不上传 grid；数值等价依据见[并发建图验收](docs/scene-opt-validation-2026-10-15.md)。感知复用一套16关节模型和相机分割器，每次更新实测双臂及夹爪反馈；16关节仅用于感知，不改变运动规划关节数。同一规划后端复用已解析机器人配置，支撑轴撤离不再淘汰普通规划缓存。首次编译仍有开销；初始化与完整流程计时见[当前核查记录](docs/pipeline-initialization-validation-2026-10-16.md)，SDK/vendor 不修改。

左右单臂工作空间分别离线生成并保存到 resources/workspaces/arm_A.npz、arm_B.npz。保留本臂、夹爪、腕相机、底座、立柱与头部相机，忽略另一臂和环境；自由旋转、保留关节限位和配置中的自碰撞检查，以现有185 mm夹持参考点生成1 cm网格。运行时纯CPU查表，不请求GPU IK。模型或标定改变时拒绝旧图；未命中只说明离线采样没有支持。详见[工作空间生成与查询](docs/offline-workspace.md)。

场景编辑器可添加可见的彩色桌面标记；标记定位来自 RGBD。盒内抓取使用实测盒底支撑参考。盒内和桌面放置都以固定分区中心确定 TCP 世界 XY，只搜索18个世界 yaw；真实占用由三维碰撞检查判断，已放置物体仍是障碍。

TCP 是两指前端中心，沿模型安装端 TCP_Link 的局部 Z 偏移185 mm。应用目标先固定前端 TCP 世界 XY=slot 中心，再换算安装端目标：p_mount.xy=slot.xy-(R*[0,0,0.185]).xy。内部兼容字段 T_world_tcp、attachment 球坐标及 SDK 求解仍使用安装端参考系；不把两者混称为同一原点。R=Rz(yaw)@R_grasp，yaw绕世界Z，18个方向。安装端 Z=floor_z+(hard_clearance+0.002)-min_i((R*c_i).z-r_i)，前端 Z 由同一刚体变换确定。 目标底面保留完整 hard clearance；默认最小10 mm、规划12 mm、实际允许10–14 mm。2 mm为已有到位容差，不是降低安全间距。前端 TCP 三维位置及 XY 误差≤2 mm、姿态误差≤2°、实际球底与目标间距差≤2 mm。松爪前使用同一次实测反馈，对当前双臂关节、两侧实际夹爪和原 attachment 进行完整三维碰撞复核；失败保持夹持，不开爪、不detach。无需 attachment XY 投影包含约束，不增加盒沿/体素补偿。

任务观察的三路 RGB 可显示名称、完整实例 ID、任务编号和搬运状态；障碍与盒子也显示 ID。只有当前图像已完成掩码和 ID 关联才叠加标签，原始结果帧标明“未分析”，不把历史位置贴到新图。实验回放保存各次标注 SVG 和状态快照。清单计数、跨时 ID 规则、发现未点名障碍的查询范围及全部技能复核见[清单与标注审计](docs/inventory-labels-skill-audit-2026-10-15.md)。

放置验收始终需要完整执行后的开爪反馈，以及放后三相机Decisions确认同一物体位于分配的region_id/slot_id，最后一件也观察。缺少视觉结果不能仅凭物理释放记为完成。

inspect_scene仅提供已有ID的semantics类别、关系和可见性补充，需显式配置VisualAssistance，默认网页不启用；它不分配身份或确认搬运完成。此独立语义辅助全任务共享8次请求，同输入含错误结果均不重调；不输出运动命令或米制坐标。verify_grasp只读夹爪反馈；verify_place还必须取得Decisions去向确认。`JEV_VISUAL_TIMEOUT_S`仍是独立语义辅助的超时，默认90 s、范围1–180 s；JEV继续接收完整16项目录及原指令和清单。

当前碰撞模型为349球（每臂88、每腕相机14、每夹爪48、中央49），上限350；保留双臂、手内和中央映射对称。四个夹爪 Link4 从各4球改为各14个较小球，解决记录中的持物假碰撞；其他连杆沿用原配置。完整覆盖证明、失败重放和性能边界见[当前验证记录](docs/current-failure-validation-2026-10-16.md)，历史减球试验见[10月14日拟合记录](docs/collision-refit-2026-10-14.md)。不能把组件缓存耗时当作整条流程耗时。


历史说明：2026-10-09 的三相机改造及当时验收范围见[实施记录](docs/superpowers/plans/2026-10-09-visible-cloud-progress.md)。其中流程和结果仅对应历史版本，不替代当前16技能、TCP世界XY/碰撞阈值球底间距及并发建图验收。

历史说明：2026-10-10 的简化记录保留于 data/simplification-20261010/；当前调用边界以本 README 和[技能手册](docs/jev-skills-guide.md)为准。

## 日常启动

在 Ubuntu 项目目录中执行：

```bash
cd /home/guangyu/projects/tianji-jev-mujoco
```

启动算法容器（SAM3 + cuRobo + GraspGen）：

```bash
scripts/algorithms-service.sh start
```

启动网页和 MuJoCo：

```bash
scripts/web-service.sh start
```

网页默认版本为 `0.4.0-experiments-guangyu`。cuRobo 使用原有 `jodell-curobo:0.2.0-guangyu`，其中已带 Open3D 0.19；锁定 SDK 保持不变；SAM3 和 GraspGen 继续使用 `0.2.0-guangyu`。两条命令都会等待容器健康后返回，服务在后台运行，关闭 SSH 终端不会停止服务。启动只使用已有镜像；缺少镜像时退出，不自动构建或下载；重复 start 且配置、镜像不变时复用容器。旧命令中的 `--dev` 仅作为兼容别名，行为相同。

网页容器默认挂载 Ubuntu 项目源码：修改 `web/` 中的页面、JS、CSS 后，Vite 通过 HMR 或页面刷新应用修改；修改 `src/` 中的 Python 或 `skills/` 中的契约/说明后，Uvicorn 自动重载后端 worker，网页检测到新后端实例后自动刷新。这些源码修改不需要重建镜像或重启容器。**后端 worker 重载会清除当前任务和内存场景，重新创建 MuJoCo；请在任务结束后修改后端。** 历史运行日志和保存的规划参数仍在 `data/`。修改 skill 的授权要求不因自动重载而改变。

挂载的是 `/home/guangyu/projects/tianji-jev-mujoco`，不是 Windows 文件夹。修改 `C:\project\tianji-jev-mujoco` 后仍须同步对应文件到 Ubuntu，或者直接远程编辑 Ubuntu 源码。

`resources/` 也只读挂载到网页容器，碰撞球查看器和 cuRobo 服务读取同一份项目配置。碰撞模型由两个后端进程缓存；更新碰撞配置后需要重新加载网页后端并重启 JEV 的 cuRobo 容器，无需重建任何镜像。首次升级到包含此挂载的 Compose 配置时，`scripts/web-service.sh start` 会用现有网页镜像重新创建容器。

关闭网页和 MuJoCo：

```bash
scripts/web-service.sh stop
```

关闭算法容器：

```bash
scripts/algorithms-service.sh stop
```

`scripts/web-service.sh status` 查看网页状态，`scripts/web-service.sh logs` 跟随日志。算法脚本同样支持 `status` / `logs`；日志界面按 Ctrl+C 只退出查看。网页和算法可分别启动/停止，不影响 tianji_base 的容器。根目录 `./start.sh` / `./stop.sh` 是网页 Docker 命令的快捷入口。

Windows PowerShell 打开网页：

```powershell
Start-Process "http://192.168.3.4:8340/"
```

Windows 与 Ubuntu 在当前可直连的局域网内，无需 SSH 端口转发。关闭浏览器标签页不会停止后台容器。停止或重建网页容器会清除内存中的仿真场景；项目 `data` 中的运行日志和规划参数通过挂载保留。`.env` 只读挂载到容器，不打包到镜像。

## 使用

1. 在三维场景中将鼠标移到物体上，高亮后按住左键拖动，调整 XY 位置；按住 Shift 再拖动可调整高度，向上拖动会升高。Esc 可取消尚未松手的拖动。空白处左键旋转视角，右键平移，滚轮缩放。左侧物体和桌面的位置、尺寸单位为毫米（mm），上下箭头每次调整 1 mm；yaw 单位为度（°）。类型和颜色也可在左侧设置。没有预先指定的目标或障碍物。
2. 拖动与表单输入修改同一份待应用场景，可连续调整多个物体。XYZ、yaw 和尺寸修改会立即显示位置、旋转和缩放预览。立方体及球体三轴尺寸联动，圆柱体 X/Y 直径联动，长方体三轴独立。点击“应用场景修改”后才写入仿真，再点击“重力稳定”和“确认场景”。桌面固定，物体有重力。未应用时，三维画面显示编辑预览，后台物理继续运行；Start 和确认场景暂不可用。Reset 恢复最后确认的基线并丢弃待应用修改。任务运行时禁用场景编辑。仍有待验证、已持物或释放待确认记录时，后端拒绝应用、稳定和确认场景；持物会话正常结束或 Reset 后才可继续这些操作。
3. 输入“抓取左侧黑色方块”，或添加类型为“开放物料盒”的彩色盒子，应用并确认后输入“抓取红色方块到蓝色盒子里”。物料盒由底板和四壁组成，参与重力与碰撞。纯 place 指令只接受当前会话中已经验证的持物状态。
4. 三维视图提供“显示/隐藏 TCP 坐标系”，白点为两指前端中心，A/B 标签对应两臂，红X、绿Y、蓝Z；坐标轴跟随关节反馈，不改变场景。位置输入采用 X/Y、Z/yaw 两行；“取消修改”放弃所有未应用场景草稿和预览，恢复最新后端场景，不调用 Reset 或修改物理状态。错误显示在右侧记录区，可展开完整详情并关闭，不遮挡三路相机和点云诊断按钮。右侧“运行记录”按时间显示四类事件：JEV 决策、skill 执行、算法计算、系统事件。只有“JEV 决策”代表实际模型调用，可展开查看 state、questions、可选值、实际输出、置信度和耗时；其余卡片查看执行数据。“规划无解，交回 JEV”是系统记录的规划结果，下一张 JEV 决策卡片才是模型对后续行动的选择。总事件数与 JEV 调用数分别显示。
5. “任务用时”在点击 Start 时从零开始，任务结束时停止，Reset 后归零。MuJoCo 的物理时间独立持续推进，保证场景编辑期间仍有重力和碰撞。
6. Reset 恢复最后确认的完整场景基线、机器人状态、控制目标及任务状态。物理先重置，后台等待旧算法请求结束后清除 GPU 场景、attachment 和预览缓存；清理完成前 Start 暂不可用，失败可再次 Reset 重试。旧请求不能覆盖新状态。

点击三维视图的“查看碰撞球”，在新网页查看当前双臂姿态下的固定开爪基线模型，页面显示实际加载的球数。夹爪在只读模型副本中固定张开，可切换球/场景网格显示、调整不透明度和暂停跟随；不改变运行中的 MuJoCo 状态。运行时的闭合模型与 attachment 状态另见任务状态栏，不能用此查看器判断当前持物几何。

碰撞配置现为 **349 球**：双臂各88、腕相机各14、张开夹爪各48、中央结构49。此前309球模型的四个夹爪 Link4 存在外凸，本轮各改为14球，最大半径18.877 mm；使用项目侧三角片包围球拟合和原镜像规则，不能将这四个新模板称为MORPHIT输出。四个 Link4 的完整碰撞网格表面/体积及视觉网格表面通过包含检查，张开状态另有限制球不能越过手指镜像中面的约束；其他连杆保留历史配置，未证明全机器人表面覆盖。不修改SDK、buffer、忽略关系或关节限制。[拟合来源及验证](docs/current-failure-validation-2026-10-16.md)；[历史减球试验](docs/collision-refit-2026-10-14.md)。grasp 规划与接近、抓取执行检查使用固定张开夹爪球；按用户约定执行原有 lift，其间不建立 attachment。lift 后由 verify_grasp 只读夹爪反馈确认持物；plan_place 开始才按实测 driver/mimic 建立闭合夹爪派生配置和 attachment；放置仅使用抓取后的固定闭合状态与完全张开状态。派生配置仅修正经 URDF/STL 验证的对向 Link3/Link4 球近似误报，其他自碰撞和世界碰撞保留，不删除机械臂或夹爪球。局部仍有近似误差，不承诺任意动作的真实表面 10 mm 间距。[对称映射规则](docs/collision-symmetry-2026-10-08.md)；[此前精简记录](docs/collision-avoidance-refit-2026-09-30.md)。

运行状态下方显示当前 JEV 决策或技能、已等待时间与内部执行阶段。plan_grasp 保持原有batched IK与plan_grasp_batch批次，仅在当前批成功返回的三段轨迹内计算七关节Σ|Δq|并选最小者；有可执行解立即返回，只有本批全失败才继续。选中轨迹保留原有补点和命令检查，lift仍按原约定无碰撞检查；不为选优增加批次或改变执行速度。选臂使用离线源/目的中心工作空间，实际抓姿 IK 留在 plan_grasp 内；不向 JEV 注册 evaluate_arms 或 plan_grasp_and_place。plan_place 在已验证持物后开始，以独立 prepare_attachment 阶段准备持物碰撞。错误返回阶段证据，由 JEV 选择恢复或结束；任务计时停止后 MuJoCo 物理仍持续。

整个 `plan_grasp` 的任务等待上限默认为 180 秒，覆盖IK和轨迹的所有 batch，共享一个monotonic deadline，可在网页“规划总时限”修改。尚未保存网页参数时，后端 `.env` 的 `JEV_PLAN_TIMEOUT_S` 可提供初始值；保存后以网页参数为准。超时即结束本轮任务，不能执行迟到轨迹。Reset 或超时会禁止追加批次，但现有算法服务没有取消正在计算的请求接口；在途 HTTP 请求仍可能等待返回（现有单次超时 600 秒）。此时网页明确提示后台请求尚未结束，并暂时禁用 Start；Reset 仍可使用，场景操作仍受上述持物会话限制。请求返回且所需清理成功后解除限制，迟到结果和事件被丢弃。本项目服务代码由 Compose 挂载；Uvicorn 自动重载仅作用于网页后端 worker。修改 GPU 服务适配代码仍需重启 JEV 对应的算法容器，无需重建算法镜像；镜像内 SDK 和算法源码不改。

支持 grasp、grasp_place 和 place。grasp 与 grasp_place 共用 plan_grasp；verify_grasp 仅凭夹爪反馈确认并保存 holding，plan_place 开始才创建/复用 attachment。standalone place 要求同 generation 的 verified holding，复用其保存观测和地图，不额外 RGBD；执行臂必须是持物臂。三条搬运路线同阶段固定单臂，中转第二阶段可重新选臂并保留同一 ID。方位采用机器人视角：机器人朝 +X，左侧 +Y，右侧 −Y。

## 网页高级参数

从顶栏“参数设置”打开分类面板。求解器、碰撞、抓放几何、持物球和点云滤波参数默认锁定，点击“解锁高级参数”才能编辑；重新锁定保留草稿，点击“应用参数”才生效。时间伸长和三段停留在空闲时可直接编辑。Start 冻结配置并重新上锁，运行中不能改；Reset 保留已应用算法参数。Lift 使用用户应用的坐标系、轴与偏移；JEV 可以读取其世界坐标位移证据，但不会自动改写设置。IK / LM / TrajOpt seeds 默认 16 / 32 / 2，使用锁定版本官方 batch planner 的默认配置。默认三段时间伸长 3 倍：在当前批成功原始轨迹中按七关节累计路程选优后，仅为选中轨迹用 cuRobo 输出的 q/v/a 做五次 Hermite 补点，保留原始点间隔，对接近/抓取实际补点和实测起点连接段做固定开爪碰撞检查；当前阶段按用户约定 lift 无碰撞；停留 1/2/1 s。放置接近、撤离、attachment 球底释放间距及停留有独立参数。place 使用实际闭合夹爪及 attachment；plan_place 预先规划持物接近、持物放置和空爪撤离；实际松爪后 detach 并执行撤离，verify_place先确认释放，再每次三相机核验去向；有剩余搬运时用该帧更新全障碍物图。可选 Open3D 滤波默认关闭。完整默认值、范围和共享关系见 [参数说明](docs/planning-parameters.md)。

## 一条指令如何执行

JEV 启动时一次解析原指令中的任务、属性、数量、顺序、路线及执行臂约束。随后只在当前 16 个注册 skill 与 stop 中选择下一步。每项精简目录含用途、输入、前提和输出，始终提供完整目录；程序收到选择后按 contract 校验，外层完成必要参数选择，再记录 skill_start 并执行固定算法。契约拒绝返回缺少条件，不执行替代技能；连续 8 次选择拒绝进入 needs_attention。

首次observe_scene先以头部RGB和指令调用Responses，建立固定全场景ID及搬运成员证据；随后SAM3分割，Decisions将候选绑定到这些ID，本技能不建图。JEV仍决定任务、路线、技能与执行臂，并校验清单数量。清单不按SAM3候选数增减；缺少分割或深度的实体保留身份，缺少碰撞几何时拒绝发布运动地图。

初始头部观测建立固定全清单与桌面参考；已有场景的更新使用三相机，任意一路可提供某个物体的证据。每次放置和空爪撤离后，包括最后一个物体，verify_place均原位采三相机并确认同一ID是否到达所选区域。仍有搬运阶段时，以该帧建立不排除任何物体的碰撞图，再允许双臂返回观测位姿。build_scene随后三相机精建图，仅在抓取图中排除当前目标。桌面cuboid只建立一次，更新复用；实体几何始终来自当前RGBD。

地图与目标几何均就绪后，`generate_grasps` 生成至多 200 个实测点云抓姿，并按阶段臂转换安装TCP、绑定工具版本后再送IK；`plan_grasp` 对已绑定臂做真实 batched IK（容量 100），分至多 16 个一批规划 approach/grasp/lift；在当前批成功完整轨迹中按原始采样点七关节累计路程排序，仅对选中轨迹补点并复核；有可执行解立即返回，本批全部失败才继续下一批。超时或取消返回失败。grasp 与 grasp_place 都使用这一注册入口，放置另由 plan_place 处理。

`execute_grasp` 完成开爪、接近、抓取闭爪、保存抓前点云/闭合 TCP 参考、lift 及停留，不采集结果 RGBD。`verify_grasp` 只读当前夹爪命令和位置样本，默认按行程 2% 容差及 0.2 s 稳定窗口返回 held/empty/unknown，记录 feedback_id；不调用 SAM3/OpenAI，不建立 attachment。unknown 保留未知，不自动算空抓失败。单抓取任务 held 后可结束。

需要放置时，`plan_place`先在独立的prepare_attachment阶段建立或复用持物球，筛选候选后规划持物approach/place及空爪retreat。`execute_place`核对实测释放端点后开爪、确认反馈、detach和撤离。`verify_place`先检查完整执行及开爪反馈，再用放后三相机Decisions结果确认同一物体到达本阶段目的区域。释放与视觉去向缺一不可，最后一个物体同样采图。该证据确认区域，不声称精确最终落姿。一次释放只完成绑定的一个阶段，重复观察不能完成下一段。

standalone place要求同generation的已验证持物记录，复用保存的观测、碰撞场景、执行臂、初始视觉参考和固定全清单；开始规划时不额外拍图，放后仍进行视觉确认。新任务搬运清单以当前所持物体为成员，不重新创建它的ID。plan_place开始才准备attachment；空爪、过期持物或指定另一臂不能启动放置。

执行、观测、规划失败由 JEV 明确选择 `recover_task` 的 reobserve/retry_grasp/retry_place/retry_plan/defer/stop。程序核验反馈、地图、身份及预算；reobserve 是显式原位新观测，retry_grasp 仍要求下一轮完整 build_scene；retry_place 仅在未执行且原地图有效时复用。受限 retry_plan 保持本阶段臂，不能用于路线任务换臂。每目标最多 3 次计数失败、每周期最多 8 个不同分析 capture；未知和未执行契约拒绝不冒充物理失败。取消、generation 变化及后台在途请求保留执行互锁。

observe_scene/build_scene 的固定清单身份缺失或歧义直接保留原始错误与关联证据，结束本轮为 needs_attention，不再反复 recover。三相机更新使用视图并集，不要求三路同时看见每个物体。

当前已删除observe、observe_geometry、observe_result、move_to_observe、inspect_visual_result、evaluate_arms、check_scene_changes、update_scene、plan_grasp_and_place的独立入口和runner分派。内部observe仍供注册技能复用，move_to_observe由build_scene调用，evaluate_arms由plan_grasp调用；它们不是独立skill。observation_motion、geometry_observation、prepare_attachment和算法batch接口也不是新增skill。完整 16 项逐项说明见 [JEV skill 手册](docs/jev-skills-guide.md)。

Registry 读取并校验 contract 与同目录 SKILL.md；只有同时具备 contract 和实际执行分支的入口才可调用。Pydantic AI/TypeSafe/JEV 负责类型化问题与答案，程序负责执行前提和固定算法。决策 state 仅包含当前 RGBD 的环境估计白名单、自身反馈、冻结参数、清单与失败/规划证据；像素、mask、原始点云和编辑器真值不进入文本 state。成功规划稠密数据保留在日志，重复候选/失败引用压缩，原始算法精度不改。输入采用保守 24k token 软预算并记录实际用量，max_tokens_exceeded 是输入上限错误，不自动换模型或重复请求。

本次技能精简与流程边界已由用户批准。后续新增或改变技能、契约与执行流程仍按项目审核约定维护；详细算法和未知边界在手册及运行时说明中同步更新。

## RGBD、持物与放置

**视觉接口输入边界：**Responses仅接收初始头部RGB、用户指令及清单生成要求；Decisions接收初始/当前完整RGB、初始固定ID参考框图、当前临时候选框图，以及固定清单、候选SAM3描述和搬运状态。两接口均不接收SAM3像素掩码、掩码轮廓、深度图或点云；Decisions的候选框是本地从掩码计算出的二维外接矩形，目标分区是本地几何投影的二维标注，不是原始掩码或深度输入。身份问题只允许固定实体ID、reject（无关/清单外主体）和unknown（身份无法区分）；不再询问或接受invalid_mask。容器框包含内容物不影响按容器主体关联。掩码尺寸、有效深度、像素归属及碰撞几何完整性仍由现有本地流程检查，这不等于已证明SAM3每个像素的语义正确。

网页默认以目标 2 fps 显示独立 RGB 预览，左腕—头部—右腕水平排列；可切换最近任务观察 RGB/深度。预览不采深度、不调用 SAM3、不覆盖任务 capture 或场景。点云面板读取最近建图的原始/保留/剔除点，可切换 DBSCAN 诊断颜色。显示最多 30,000 个实测点，并分别列出完整数量和显示数量；分层抽样优先保留少量被剔除点，不改变规划输入。

- 三相机：3为头部、2为左腕A、1为右腕B。初始只用头部；已有场景更新必须三相机，不要求三路同时看见每个物体。正常采集为初始、每轮抓前及每轮放后，最后一次放置也采图。
- 首次用头部RGB和完整指令调用Responses Structured Outputs，输出桌面实体、外观、初始RGB框、父容器、SAM3描述及required/excluded/unknown成员。程序分配固定全局ID，搬运清单引用同一ID。随后SAM3按清单描述分割；它不接收全局ID。Decisions以初始及当前完整RGB、初始ID参考图和当前临时候选框标注图（不生成或发送逐实体RGB裁剪），把候选关联到固定ID，独立判断各实体可见性和搬运去向。程序按返回的ID归并各相机掩码/点云。只向模型发送RGB及语义任务状态；深度、掩码轮廓和世界坐标不发送，目标区域仅提供本地投影的二维标注。初始清单不依赖SAM3成功与否：缺少掩码或有效深度的实体保留身份，标记geometry_status=missing / mapping_unresolved，阻止发布不完整碰撞图。最后一次视觉验收无需为无后续运动强行补全地图。初始RGB框不能冒充后续当前框。后续只复用固定清单的描述，不枚举类别和同义词。像素归并仅整理几何，不决定身份；同相机重复像素不重复计数，盒子不吞并已识别内容物的像素。桌面cuboid只建立一次。
- 感知：Responses提供初始清单与成员证据，Decisions提供身份关联、可见性和去向判断；SAM3与注册深度提供实际像素和几何。桌面、盒子、目标区域、障碍和搬运物体均有全局ID，搬运清单是全清单的子集。未见物体保留ID，不把历史点云当当前障碍测量。JEV保留技能、路线、目标及数量分组决策，不能覆盖明确视觉成员证据。
- 物体几何：同相机同ID重复像素只保留一份。盒子掩码中已识别工作物体的像素归物体；不同普通实体共有的边界像素不用于任一实体几何。抓取物体另排除一像素轮廓带。原始深度保持不变，不将剔除像素宣称为自由空间；视觉接受的实体若无法生成有效几何，明确停止，不静默丢项。
- 盒内抓取：目标、容器和盒底引用须属于本轮建图 RGBD；按实测点与盒底区域判断支撑，不能用相机数量作为就绪条件。完整场景与轨迹碰撞检查保留，不补被遮挡表面。继续搬运时由 build_scene 完整返回观测位姿、采集并重建；历史盒底不能伪装成当前测量，旧update_scene任务入口已删除。
- 场景：首次observe_scene只提供清单和几何；build_scene先用当前观测建立包含全部实例及桌面的运动图，再完成双臂观测运动、同步感知关联及实例重建，grasp/place 复用该轮快照。只有当前目标静态层排除，桌面、盒子及其他障碍保留；普通实例和盒子的 TSDF/ESDF 统一为 20 mm，盒子网格 Z 采样中心仍对齐实测底板。桌面 cuboid 校验后复用，实例只来自新帧；并发、compact ESDF 及跳过无用背景处理的历史证据见[隔离 GPU 验收](docs/scene-opt-validation-2026-10-15.md)，20 mm 验证见 data/voxel-20mm-20261020/verification.json。
- 抓取：approach→grasp/闭合→lift；规划使用张开夹爪且无 attachment，lift 按用户约定不作持物碰撞检查。execute_grasp 不采结果 RGBD；verify_grasp 只读夹爪反馈并保存 verified holding。plan_place 才以独立 prepare_attachment 阶段使用抓前可见点云和闭合时实际 TCP 建立至多 16 球；假定闭合位移可忽略且无滑移，不补隐藏表面，合法绑定下缓存复用。
- 放置：plan_place 先 prepare_attachment，再一次返回持物 approach/place 与张开无 attachment 的 retreat。N 为冻结全局 required 总数；包括N=1在内均取K=2×ceil(N/2)个中心直线等面积分区，按初始头部投影行列冻结编号。物体i仅使用slot i，完成或暂缓不缩小N、不换区。盒底自动提取实测平面凸包内的内接矩形，不要求marker，采样密度变化和局部遮挡不切碎底板；障碍物由三维碰撞检查。首次plan_place用三相机几何划分并冻结布局，初始头部仅确认区域身份与底面。区域和版本按阶段/目的地保存，部分遮挡不重划；每次仅指定区生成18个世界yaw（0°–340°、间隔20°），保持抓取姿态基准和TCP原点世界XY，按每个方向反算球底间距等于碰撞阈值的TCP Z，不叠加体素/盒沿抬高，不约束attachment的XY投影。18个候选共享IK100、轨迹16，批内按yaw变化、关节路程、候选编号检查完整撤离，首个完整方案返回，指定 slot 耗尽即返回失败，不尝试其他分区。补齐项不参与计数或选优；有限候选无解不证明整个区域不可用。
- 执行与验证：持物approach→place→实测TCP/球底核对→开爪确认→detach→空爪retreat→三相机视觉确认。完成需要释放动作、阶段、目的区域与当前观测一致；elsewhere/unknown保留待确认并停止本轮，不再建图。只有视觉确认成功且仍有搬运阶段时，才更新全障碍图，再允许下一轮观测位姿运动。
- 碰撞：保留原机器人球、世界/自身/对侧机械臂/关节限位检查，仅排除目标静态层及持物与接触指垫的预期接触。张开与持物闭合配置分别缓存；临时支撑轴撤离配置单独缓存，不淘汰普通双臂/抓取规划器。释放清理持物和临时撤离缓存，保留合法张开缓存；不检查开合过程的角度序列。可选 Open3D 离群点过滤默认关闭，不执行“无根浮岛”删除。

已保存的无附加噪声 MuJoCo RGBD/SAM3 记录做了[有界 Open3D 滤波评估](data/visible-cloud-work/filter-assessment-report.txt)：四份记录各取 20,000 个有像素来源的原点，比较 OFF、statistical、radius 和 DBSCAN 诊断。此结果支持保留默认 OFF；不是硬件 RGBD 验收，也不是完整建图链的误删率或性能测量。

执行命令使用持续物理与电机控制，保留重力、摩擦和接触。运动前检查实测主动臂与轨迹首点；超过1°漂移拒绝，否则将实际起点连接段加入执行数组，并按该阶段的规则验证碰撞；lift 仍遵循上述用户约定。物理线程在提交时再次拒绝超过0.05°的起点变化。

## 部署

Ubuntu 需要 Docker Engine、Docker Compose 和现有 GPU 容器运行环境。网页只使用一个镜像 `tianji-jev-mujoco:0.4.0-experiments-guangyu` 和一个容器 `jev-mujoco-web-guangyu`。镜像内同时包含 Node.js、Python 3.12、MuJoCo 3.10.0、Pydantic AI/JEV 客户端、模型资产和前端依赖；Vite 与 Uvicorn 在这个容器中运行。使用当前已验证环境的 Python 依赖版本约束；前端按 pnpm lockfile 构建。日常运行不需要宿主机 conda、`.venv` 或 Node.js。网页容器不占用 GPU。

SAM3 和 GraspGen 沿用 `jodell-sam3:0.2.0-guangyu`、`jodell-graspgen:0.2.0-guangyu`；cuRobo 使用原有 `jodell-curobo:0.2.0-guangyu`，其内置 Open3D 0.19 已满足桌面 RANSAC 和点云滤波，SDK 源码保持锁定。网页通过 CPU OSMesa 渲染 RGBD。模型和碰撞球资产已复制进本项目；网页采用 Ubuntu host network，沿用 `.env` 中的本机算法地址。

首次部署时将 `.env.example` 复制成 `.env` 并填写 `TYPESAFE_API_KEY`，再执行上面的两条 start 命令。当前 Ubuntu 的 `.env` 已配置，不需要重新填写。

日常修改已挂载的网页、后端或 skill 源码不需要 build 或 restart。修改 Python/前端依赖、基础镜像、Dockerfile、`deploy/run-web.sh` 或镜像内资产时，更新网页镜像：

```bash
scripts/web-service.sh build
scripts/web-service.sh start
```

`build` 只构建网页镜像，不停止正在运行的容器，也不需要 `.env`；随后的 `start` 会应用新镜像，重新创建网页容器并清除内存场景。前端依赖来自镜像，每次启动复制到容器临时目录，不使用宿主机 `node_modules`，不会被旧依赖卷覆盖。此过程不构建或重启算法容器。

默认算法端口 8310/8322/8323（cuRobo/SAM3/GraspGen），公开网页端口为 8340。Vite 在 8340 提供页面，将 API、场景资产和 WebSocket 代理到只监听本机的 Uvicorn 8341。网页只使用 `deploy/web.compose.yaml`；算法单独使用 `deploy/algorithms.compose.yaml`。

`deploy/run-web.sh` 启动并管理 Vite 和 Uvicorn。任一主进程退出时，它会停止另一个进程并退出容器，由 Docker 的重启策略恢复；Uvicorn 正常重载 worker 不会退出容器。执行网页 `stop` 会同时停止两个进程。

每次运行的原始 JEV 决策和技能结果保存于 data/runs/<run_id>.jsonl，含实际输入、输出、阶段和耗时；反馈验收使用 feedback_id，RGBD 使用 observation_id，不能混用。真实抓姿 IK 现在是 plan_grasp 内部算法阶段，不是已删除的 evaluate_arms 技能。候选原始矩阵、残差和 score 留在算法/执行数据，模型 state 使用所需摘要与引用。这些记录不含认证头；Reset 不删除历史记录，打包排除 .env、虚拟环境和运行记录。

完成后网页显示实际 JEV 调用数、平均/累计调用耗时、整轮耗时；完整指标还包含输入输出 token。多种合法技能顺序或可达机械臂会分散模型概率，因此这些选择不套用目标身份识别的置信度门槛，仍记录实际概率和置信度。

初始任务理解只提取原文明示的搬运数量，不因“绿色物体”等属性推定数量为 1；未指定数量时，后续按观测和原文确认任务成员，仍排除无关物体。无字面数词时不创建数量选择或门槛；出现数字/数词时，JEV 判断其是否约束待搬物体，门槛保持默认 0.60，低置信度结论显示字段、返回值、实际值与门槛。`tests/request_scope_jev_smoke.py` 提供不会移动机器人的真实 JEV 数量/范围回归探针。

## 算法镜像与项目服务的边界

共享的是已有算法镜像内的 SDK、CUDA/Python 依赖和模型权重，不要求两个项目使用完全相同的服务入口。

| 层 | tianji_base | JEV MuJoCo |
|---|---|---|
| 算法镜像 | 现有 `jodell-curobo` / `jodell-graspgen` | cuRobo 原有 `0.2.0-guangyu` 已含 Open3D；SAM3/GraspGen 也复用原镜像，SDK 不变 |
| cuRobo 服务入口 | 镜像自带的规划服务 | 只读挂载本项目 `src/`，运行 `jev_mujoco.geometry_service` |
| GraspGen 服务入口 | 镜像自带的 GraspGen 服务 | 只读挂载本项目的服务快照，运行 `tianji_graspgen.service` |
| 容器与端口 | tianji_base 自己的容器，8010 / 8023 | JEV 自己的容器，8310 / 8323 |

所以修改 `geometry_service.py`、`backend_builder.py` 等 JEV 适配代码不等于修改算法镜像，也不会替换 tianji_base 的服务。不同容器分别加载自己的入口；同一镜像可供两者使用，但 GPU 显存和计算资源仍由它们共同占用。当前 GraspGen 服务快照与 tianji_base 镜像内服务并非完全相同，不能称为复用了同一套 HTTP 服务实现。

**修改 JEV GPU 服务代码后，重启该服务进程才能生效**，例如在 Ubuntu 执行 `docker restart jev-mujoco-curobo-guangyu`。这会重新加载该容器内的 cuRobo 模型/缓存，既不重建共享镜像，也不影响 tianji_base 容器。网页自动重载不会自动重启 GPU 服务。日常修改 JEV 决策、网页或 MuJoCo 控制不需要重启算法容器。

## 代码分工

| 位置 | 负责什么 |
|---|---|
| `src/jev_mujoco/decisions.py` | Pydantic AI + JEV 的类型化问题 |
| `skills/*/contract.json`、`SKILL.md` | 执行条件、精简模型目录与人读流程说明 |
| `runner.py` | 决策循环、技能执行、实际状态和日志 |
| `simulation.py`、`model_builder.py` | 持续物理、模型、关节控制、场景和重置 |
| `cameras.py`、`perception.py`、`perception_mapping.py` | RGBD标定、分割、几何估计和测量地图 |
| `algorithms.py`、`geometry_service.py`、`backend_builder.py`、`carrying.py` | 原算法接口、公开solver构造、携带碰撞与执行证据 |
| `planning_settings.py` | 唯一参数 schema、关系校验、持久化 |
| `verification.py`、`placement.py` | 可见抓取/放置验证、测量参考与持物几何 |
| `app.py`、`web/` | 单页接口和三维视图 |

测试：`.venv/bin/python -m pytest -q`。实际模型探针：`PYTHONPATH=src .venv/bin/python scripts/smoke_jev.py`。测试替身只用于测试；产品中没有模拟 JEV、伪造抓取候选或伪造规划成功的回退。

初始清单使用官方/v1/responses Structured Outputs；每次观测由/v1/decisions关联固定ID和判断去向。外层JEV选择initialize_inventory、associate_inventory或verify_transfer，内部不再询问JEV。两接口共用JEV_DECISIONS_CONFIG_FILE或JEV_DECISIONS_API_KEY的官方配置，不继承第三方OPENAI_API_KEY/jojo.json。两接口均为官方gpt-6-luna。首次/v1/responses使用严格JSON Schema（1–64个可见实体，最多6000输出token，reasoning=low）；不把结构正确等同视觉准确。每次/v1/decisions最多200个封闭问题、128张图，choice confidence至少0.8，否则unknown。默认超时60秒；同capture同输入的成功和错误均缓存，不自动HTTP重试或回退旧RGBD关联。Responses结构/服务失败、Decisions未知身份/服务失败都保留证据，needs_attention停止，不自动recover。不可区分的同外观实体不强猜ID。


## 实验记录与回放

工作台 TASK 提供实验名称、录制视频开关，以及自动保存／结束后决定／不保存。视频关闭仍可保存步骤数据；不保存时不创建实验归档。实验存放于源码外的 `/home/guangyu/projects/tianji-experiments`，每次运行一个独立目录，格式版本为1。临时实验位于 `.pending`，任务结束及编码完成后在回放页选择保存或删除。已保存实验也可删除；删除移入根目录 `.trash`，录制、编码和下载期间受保护。不会随网页关闭而自动删除。

顶栏“实验回放”打开 `/replay.html`，可以选择实验，同步查看主场景与左腕—头部—右腕拼接视频；执行图依据实际调用ID生成，点击查看参数、JEV证据、内部阶段及原始文件。支持独立执行图链接、SVG/PNG导出、精简／完整ZIP导出。主视频来自此次运行记录的实际几何位姿，不重新执行任务；录像编码在任务结束后后台完成。

工作台采用中央三维／相机／点云标签，TASK保留在下方，场景编辑可收起，运行日志独立滚动。点云页可叠加实际 Mapper 的背景/物体体素表面，以及 RANSAC 桌面碰撞 cuboid；显示原始 RGBD 帧、相机与深度来源，明确区分测量表面和推断补全。体素展示为有限表面采样，未观测空间目前不受独立运动门控检查。参数按常用、实验调参和系统只读分类，项目默认值与来源明确标注。

保存格式、时钟和分析方法见 [实验数据说明](docs/experiments.md)。

运行结束会保存 conclusion：原始失败步骤、原因与证据、被拒绝的恢复调用、最终错误和完成数量；工作台及实验回放显示同一结论。恢复前提拒绝不消耗物理失败次数，JEV 根据 rejection 纠正 action。未知错误明确保留证据不足，不由错误码固定选择恢复。

放置完成结论为release_and_visual_destination，destination_region_verified=true才完成；destination_pose_verified仍为false。放后原位三相机提供去向与当前几何，不将预测落点当作障碍。初始头部全实例图、当前位置全实例图均无排除；观测位姿抓取图只排除当前目标。

放置验证上下文、固定 100 IK 容量、显存不足错误及镜像/重载边界的修正见 [修正说明](docs/user-corrections-2026-10-14.md)。

预算按冻结required数量N和阶段数S计算G=N*S：正常RGBD预算2G+1（初始头部1、每轮抓前G、每轮放后G），恢复采集独立8次。Decisions随这些观测批量调用，同输入不重复请求，不持续跟踪。JEV决策上限max(400,40G+80)，耗尽保持needs_attention。没有2G或仅按释放反馈完成的兼容分支。

### 视觉调用与回放补充

**桌面范围与图片证据：**初始Responses仅看头部RGB与用户指令，列出指定桌面、其上的容器、可见内容物、障碍和目标区域，排除桌外实体。Decisions使用初始及当前完整RGB、初始固定ID参考框和当前候选临时编号框；不生成或发送初始实体及当前候选的RGB裁剪。候选框来自SAM3掩码的外接矩形；不是把掩码轮廓或深度喂给模型。目标分区的二维轮廓由本地测量和标定投影，标注不替代当前可见证据或碰撞几何。未关联候选不预标猜测的全局ID。

**分区证据：**程序保留destination_id、region_id、slot_id及本地实测布局，模型只收到语义ID和RGB中的二维区域标注，不收到世界坐标。槽位XY定义TCP锚点，不能改成物体中心或全部attachment投影包含要求。视觉判断同一实体是否已释放且在目标支撑面/容器中，与分区标注是否明显矛盾；精确TCP落点与底面间隙由本地规划/执行校验，图片不证明毫米级准确。完成证据仍绑定释放阶段和指定region_id/slot_id。

**统一编号：**全部实体使用一份 scene_entities 全局 ID；搬运清单引用其中相同 ID。冻结 required 对象的 display_id=1..N 后，盒子、障碍、目标区域和桌面依初始清单顺序续编号；后续观测不重排、不新增实体。槽位仍只使用搬运编号，非搬运编号不改变 N。

**回放与调用档案：**每次实际 Responses/Decisions 网络请求记录 visual_decisions_call，含 endpoint、observation_id、用途、request_hash、耗时、HTTP 状态与回答/错误；完整请求问题和原始回答保存到 decisions/<hash>.json，输入图片保存为独立 JPEG 并保留 SHA256，可重建原请求。缓存复用不伪装成新调用，不保存认证头或密钥。实验回放新增观测图片页，可选择帧与标注修订、查看相机图及原 RGB；执行图使用紧凑横向分组浏览，保留节点详情与导出。打开回放不会调用模型或驱动机器人。


JEV 外层视觉选择、逐次请求输入输出、回放界面发布状态及本轮验证边界见[视觉调用与回放验收记录](docs/jev-visual-replay-validation-2026-10-17.md)。


2026-10-19 运行计时：停留参数按实际单调时钟计时，不按仿真时间计时，也不乘 time_scale；设置1 s表示实际等待约1 s。返回 duration_s 为实际等待秒数，simulation_duration_s 单独记录期间仿真推进量。轨迹仍按原仿真时间积分执行，time_scale=3不变；停留前后的夹爪、到位和碰撞检查耗时不包含在停留值中。

2026-10-19 放置准备去重：一次静止的 plan_place 调用在准备 attachment 前读取一份实测快照，持物状态判定、prepare_carrying、后续 IK 和放置轨迹规划共用该快照，避免夹爪反馈微小变化立即废弃刚初始化的持物模型。执行前和松爪前仍独立读取最新反馈并进行原有碰撞检查；不量化夹爪角度，不复用旧碰撞结论，不改变 batched IK 或 plan_place_batch 算法。

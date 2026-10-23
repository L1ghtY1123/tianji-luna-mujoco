# Tianji Luna MuJoCo 实验证据回放

工作台「查看本次实验」或 `/replay.html?run=<id>&view=graph` 打开归档。页面只读取已有文件；浏览图片、掩码或 3D 不采集新帧、不调用模型、不执行机器人动作。实验重命名、保存和删除仍使用原来的独立操作。

拆包结果帧独立标为「VLA 拆包后观察」，来自当前自然位姿；其后的「抓前观察 / 建图」是双臂返回观测位姿后新采集的帧。两者分别保存模型输入，不能把拆包结果图标成地图来源。统一 initial 检查点显示为「观测位姿观察」，因为同一入口也用于后续抓取前的观测。

## 默认执行流程与清单（2026-10-23）

默认页改为「执行流程」：按真实技能开始顺序合并相邻的观察、抓取、放置、VLA、返回观测和投放阶段。每阶段保留实际调用、结果与阶段时间。工序绑定继续使用归档的 process/target ID；未知目标不按邻接关系猜测。Responses 与 Decisions 由实际 endpoint 区分，即使历史事件种类都叫 `visual_decisions_call`，也不会全部标为 Decisions。阶段上的模型按钮显示 API 名称和耗时，选中后显示中文用途、完整问题、原始选择/概率与实际输入图片。predicate 的 0 表示该命题概率低，不统一解释为失败；HTTP 成功也不等同于物理任务成功。

默认隐藏原始事件、下载文件列表和完整调用图，仍可展开与导出。SAM3/抓取候选按需读取，零项证据不占单独面板；只浏览回放不会重新调用模型。所有可见文字至少 20px。

「物体清单」展示待拆包、已拆包待投放、已完成三列，以及该时刻全部登记实体、两段执行臂和永久 ID。完整快照仅来自明确的任务清单（`task_progress` 或含 `all_entities` / `process_records` / `inventory_initialized` 的技能结果）；观察技能的 `objects` 提议不是完整清单。`product_created` 仅在已有工序输入 ID 匹配时更新产物登记，明确标为事件推导而非完整快照，不从最终记录倒填当时尚未分配的显示编号。初始档案、数值目标区域、历史物体身份与右盒集合不混为新的实物数量。

「观测图」按 `observation_id` 合并为一次拍摄，标注版本保留原事件编号与语义：识别、选择目标、夹爪反馈持物。拍摄时间与状态更新时间分开显示；持物更新明确说明旧帧不证明抓取成功。实际模型输入从归档请求读取，初始/动作前参考与当前图像可筛选，绝不以标注图代替模型输入。

be826079 归档中，#10/#16/#73 共用初始帧，#175/#178/#233 共用黑件抓前帧；两组最后一项来自夹爪反馈后的状态标注。共 5 次拍摄、12 个标注版本、1 次 Responses 与 5 次 Decisions。初始完整清单在 #14，#158 为新产物登记事件，最终完整清单在 #268。

候选证据带有 `sampling_round` 时，各阶段标题显示采样轮次；候选编号保留每轮原始索引，以“轮次 + 编号”区分。先前失败轮次仍可逐份查看，历史缺失轮次不推断为第一轮。

放置核验显示该次归档实际包含的问题，不套用旧版每相机四题。be826079 的一次联合 Decisions 包含 `place_result` 和 `workpiece_still_held` 两个 predicate；全部参考图、当前三视角与区域标注均属于这一请求。旧版四题、support_judgment 等仍作为原始历史数据保留，不能用新版阈值重判旧记录。`result_inspected` 只表示已进行判断；是否通过以该次 skill 结果为准，缺失字段不补造。

阶段名称区分“目标 / 放置区”“执行臂选择”“场景建图”和“返回观测位姿”。`choose_arm` 的工作空间查询不显示为 Batched IK；当前 `plan_grasp` 内部包含候选 IK 和三段轨迹筛选，合并显示为“IK / 抓取规划”，详细事件仍可逐项展开。
物体标题优先使用归档清单的 `display_id`（如“物体 #1”），下方保留永久 ID；未记录编号时不按显示顺序补号。方向筛选为空、固定区可达性筛选为空和源 IK 预算耗尽显示各自已记录的停止原因。

2026-10-21真实30dd只读QA：after_ik/after_plan可查看同轮实际模型物体/场景输入、测量与补全集，所选batch通过数20/100和8/12不代表全局总数。相机fit合并点云与TCP线框范围，云顶上方17.5cm的真实mount不再被裁出视野；不另乘变换、不改变世界坐标。288项前端测试及production build通过，Edge WebGL截图见`data/live-replay-qa/30dd72b14f9b4eac811be2520492025d/filtered/127.0.0.1/stage-2.png`。这些显示验证不改变原运行的视觉失败状态。

## 实际输入与分割证据

- Responses / Decisions：读取 `decisions/<hash>.json` 中按请求顺序保存的 `request.input[].content[]`。显示 `input_image.image_url.path` 指向的原图、SHA-256、观测 ID 和相机/图像用途元数据。不会以当前相机或后来标注替代当时输入。
- SAM3：读取成功观察或失败事件的 `sam3_queries`，显示相机、真实提示、缓存状态、RGB 哈希和各提议 score。可以直接展开原始二值掩码和提议点云；未关联提议不被标成已确认物体。
- 历史文件只保存查询元数据、没有原始数组或请求图片时，页面明确显示「未归档」。缺失证据不会自动生成，也不会因此推断不存在物体。

## 候选与交互式 3D

`candidate_replay` 的三个阶段分别为 `before_ik`、`after_ik`、`after_plan_grasp_batch`。界面保留 GraspGen 全局编号、score、精确工具位姿及阶段原始结果。IK/规划响应的局部 `candidate_index` 必须通过该批的 `candidate_indices` 映射，缺失或非布尔结果保持未知。

`after_ik` 按映射后的原始候选编号关联同一批的 `candidate_filters`，不把列表顺序或局部编号当作原始编号。原生 IK 成功还必须有唯一、合法且 `accepted=true` 的筛选记录才能显示为最终可用；`accepted=false` 显示未通过，不能画绿。原生 IK 失败也不能被筛选记录提升为通过；失败项原本不会产生后置筛选记录。详情同时保留 `rawStatus`、未经修改的 `result`、完整 `filterEvidence`（含夹爪预测嵌套证据）及 `eligible=true/false/null`，并显示归档原因；不在浏览器重算 FK、间距或阈值。

历史记录完全未归档 `candidate_filters` 时保留原始 IK 状态/颜色，但界面明确说明“仅原始 IK 结果，最终可用性未记录”，`eligible=null`。已存在筛选列表却缺少成功候选的匹配项、重复候选记录、非布尔 accepted、损坏列表或无法唯一关联的原始索引，均显示未知，不猜成通过。原始索引与筛选记录同属当前采样轮和批次，不从相邻记录补造。`after_plan_grasp_batch` 仍只表达原生规划返回，最终命令检查另有证据，未被此 IK 显示修复改写；工具线框和世界坐标完全不变。

GraspGen 默认展示实际请求中的 `object_points_camera_m`，也可以选择实际 `scene_points_camera_m`、原始实测目标 `measured_points_world_m` 和完整补全几何 `completed_geometry_points_world_m`。前两项仅使用归档的真实请求数组，并按 `T_world_cloud` 转到世界坐标；坐标系仍以原始测量质心为基准，不因补全而重新居中。后两项本身就是世界坐标，不再次变换。没有真实请求数组时，不会把实测或补全集替代为“实际输入”。

新生成的 `after_ik` 和 `after_plan_grasp_batch` 记录同样包含本轮真实 GraspGen 物体/场景请求、原坐标系、观测ID与补全来源；从该轮候选随IK评估结果传递，不按事件先后猜测或寻找其他轮次。后两阶段可独立打开实际输入点云，无需先打开 GraspGen 节点。`graspgen_input_sampling` 存在时原样归档，专用采样的实际输入与原始补全集分别保留；界面单独显示原补全集点数、额外先验表面点数、实际提交总数与点间距，说明新增点不是实测深度，不改变地图体素。历史记录缺少实际请求时仍只显示已有测量/几何，不补造关联。

`graspgen_input_source=height_completed` 将实际物体输入明确标成“测量 + 先验补全”，并说明不是全部实测；`measured` 标成实测点。此标记只描述物体输入，不套用到由 RGB-D 构成的场景请求数组。缺少来源字段的历史物体请求显示“输入构成未归档”，不推断其全部来自测量。`completion_provenance` 可展开查看高度先验、支撑面来源、测量/构造点数和归档原文。补全集的前缀是原始测量，其余点由高度先验构造；这些计数描述补全几何，不表示实际场景请求具有相同点数。切换来源会清除旧画面，重新打开 3D 时同时显示当前来源、总点数和是否抽样。

工具线框仅表达候选位姿，不表示碰撞模型或真实夹爪尺寸。可以按通过/未通过/未知过滤，选择单个原始编号，再旋转、平移、缩放点云和查看原始矩阵。方向筛选通过不等于 IK 或路径可行；原生规划返回通过也不等于机器人已经执行成功。

3D 使用项目已锁定的 Three.js 0.180.0 公开 `Points`、`BufferGeometry`、`LineSegments` 和 `OrbitControls` 接口；仅在打开 3D 时加载。切换节点/实验会释放旧渲染资源，并拒绝旧请求覆盖新证据。

## 只读数组接口

`GET /api/experiments/{id}/arrays?path=<relative.npz>&key=<array_key>&max_points=30000`

接口沿用 ExperimentStore 的目录约束和读取租约，使用 `allow_pickle=False`，只接受数字/布尔数组。返回原始 `shape`、`dtype`、`data`、`sampled`、`sampling` 和抽样时的原始行 `indices`。Nx3 数组超过点数限制时按原始索引均匀抽样，最大 30,000 点；掩码保留完整像素。单个 NPZ 成员限制 128 MiB，非点云预览最多 4,000,000 个值。原始 NPZ 下载保持全量精度。

## 验证记录

Demo 身份请求新增短 R/C 标签、中性框和同帧 camera2 源物群成对放大图。请求回放保存并显示实际发送的所有图片，包括新增放大图；相邻文字记录原图 SHA-256、原尺寸、裁剪像素范围、缩放比例及参考/当前角色。放大图保留全部源物邻居，仅在同帧身份和图片内容核对通过时生成；它不代表新测量、模型身份确认或后续观测。完整 RGB 和永久选择 ID 保留，归档回归测试覆盖两张新增图及其来源。

已完成件的后续观测保存`completed_tracking`：完整原执行binding、原放置视觉证据/确认帧，以及当前候选index、相机、实测点数/边界、唯一slot与高度范围的定位依据。独立当前对应请求只含当前三RGB、已完成候选/槽注释和yes/no/unknown题；yes且confidence≥0.8才结合合法track和唯一实测scope绑定ID，原二元回答与candidate_mapping分开存档。第二份剩余清单请求不再重问已完成外观身份、visibility或progress；保存同帧generation和两份request_hashes，遗漏题通过当前已确认位置解释已完成源对象，不沿用旧源位置。原始回答、实际图片和所有候选点云保留，旧完成证据不被后续关联题覆盖。历史归档没有track时不补造；离线诊断从真实事件组装时必须明确标记。

88d冻结比较位于`data/88d-second-observation/tracking-comparison/`：12图哈希与原请求相同，仅C6/C9改为完成槽短题、移除已完成progress题；实际25答案经当时单请求解析仍unknown（完成件0.55/0.73、剩余中间件0.66）。这次单请求诊断不改原任务终态、ledger或地图，也不代表新三件流程成功。

2026-10-20：新增数组读取/路径边界、椭圆柱场景往返、明确目标分组、未知核验状态、输入图片路径、SAM3 空结果、候选全局编号映射、缺失方向判定保持未知和世界位姿旋转/平移回归。前端完整测试 280 项通过，生产构建通过（Three.js 共享 bundle 仍有约 523 kB 大块提示）。迁移到 `luna_mujoco` 后，归档和数组接口相关测试 25 项通过、1 项跳过。

本地 Edge 无头浏览器通过固定测试数据验证实际归档图片显示、SAM3 掩码绘制、WebGL 点云与候选、按原编号筛选、390px 窄屏无水平溢出，且无页面脚本异常。截图与检查脚本位于 `data/replay-ui-qa/` 和 `data/replay-ui-qa.cjs`。测试画面明确标为界面测试，不代表实际搬运或远端部署成功。

高度补全输入区分补充验证：前端完整测试 283 项通过，生产构建通过。Edge 固定样本验证实际场景请求 601 点、实际物体请求 880 点与原始测量 600 点分别显示，600 个近乎共面的原始点未被补全视图替代；实际物体请求保留约 0.1m 高度与原测量坐标系。来源说明、补全原文、候选筛选和 390px 窄屏均通过，页面无脚本异常。脚本为 `data/replay-completed-input-qa.cjs`，截图在 `data/replay-completed-input-qa/`；这是界面与坐标变换验证，不证明真实 GraspGen 抓取质量。

本次改动是归档可视化和编辑器元数据保存，skill 的运动、置信度及完成判据无变化。

2026-10-21 IK后筛选显示修复：先以失败测试复现原始success覆盖accepted=false，以及历史/重复/损坏筛选证据未区分的问题；修复后前端全量305项测试通过，41项证据/页面聚焦测试通过，Vite生产构建通过。多入口配置使用web工作目录；最初从仓库根目录调用的构建因找不到index.html失败，切回web后成功，仍有既有约523kB共享块提示。覆盖原始索引乱序、原始结果保留、FK/夹爪预测拒绝、历史范围提示、缺失/重复/损坏筛选和after-plan语义不变。只改回放显示及测试/本文档，不改规划、工具坐标、运动或live实验。

2026-10-21 后两阶段输入核对：新增第一/第三轮、实测/高度补全的端到端证据传递和独立NPZ读取回归，验证实际float32请求数组、坐标系、观测ID、来源与原始候选索引不变，后来修改生成结果不能污染已冻结IK证据。93项Python相关测试、9项前端证据测试与Ruff通过；服务均由测试替身替换，没有新模型或规划/运动调用。界面现有实际输入选择器直接读取新增字段，skill行为无变化。

## 双支撑桌录像修复

实际实验 `ef356390f2a04c2b9d025532a4369ec5` 的相机视频已编码，但场景视频因 `recorded geometry IDs do not match the rendering model` 失败。原因是旧 `videos/recording.json` 仅保存主桌和物体，遗漏快照的 `support_tables`；两个支撑桌消失后，ground、机器人及物体的几何 ID 随之偏移。椭圆 `cross_section` 已正确归档，不是该错误原因。

新录像 schema 2 同时保存主桌、支撑桌及完整物体定义。旧归档只有在几何清单明确记录支撑桌、同一目录 `config.json` 的 `simulation_initial` 与录像 generation 和主桌一致时，才从冻结初态补回支撑桌。不会读取当前 demo 配置或猜测桌面。恢复过程不修改原始归档，重建后仍逐项校验几何 ID/name，并校验归档中提供的类型及尺寸；不匹配继续失败。

该实验的冻结文件只读复制到 `data/video-archive-diagnostic/ef356390f2a04c2b9d025532a4369ec5/`，六个源文件 SHA-256 与远端一致，当前 `model_builder.py` 哈希也与实验记录相同。诊断发现原重建 15 处 ID/name 不匹配；补回冻结支撑桌后为 0，60 个可见几何的类型和尺寸一致。全部 1,832 个采样包的 ID 集合及 generation 均匹配。MuJoCo 3.10.0 实际离线渲染已输出 `recovered-scene-0.20s.png`，诊断结果与哈希保存在 `diagnostic.json`，可用同目录上层的 `verify_recording.py` 重现。

另一次实验 `bbf7eefabc9e4e46a663035942d03bab` 也完成本地验证：675 个采样包和 60 个可见几何绑定一致，恢复后 ID/name/type/size 检查通过并实际渲染 RGB。该次实验记录的 `model_builder.py` 源码哈希与当前版本不同，诊断中保留此差异，不声称源代码完全相同。

初次诊断没有改写远端 manifest、重启服务或移动机器人。Windows 验证环境无 `ffmpeg`，因此该阶段确认模型重建、全部采样包绑定和实际 RGB 渲染；随后经授权执行的远端整段重编码结果见下文。

### 离线重编码已完成的实验

底层 `python -m luna_mujoco.experiment_video --encode DIRECTORY FFMPEG` 只写 `videos/result.json`。需要同步归档状态时，使用 `experiment_reencode`，它复用 `encode_videos` 的独立进程和真实 ffmpeg 编码，拒绝尚未 finalized 或仍在录制/编码的实验。开始前将 manifest、视频元数据与旧编码日志保存到唯一的 `videos/reencode-<UTC>-<id>/`，成功或失败均保存新尝试结果。原始帧、事件、任务 `result.json`、录像场景定义及任务 `status` 不变。

该入口仅替换上一轮视频错误，保留采集缺帧与其他归档错误。两个视频均实际编码成功且无采集错误才标为 `video_status=ready`；仍有采集错误为 `partial`，编码失败按实际结果保留 `partial/failed`。`recording_complete` 沿用已有“归档无错误”含义，不代表机器人任务成功。命令非 ready 时返回非零退出码。离线运行期间应让该归档的重命名、删除和导出操作保持空闲。

统一同步修复代码后，在现有 Luna 网页容器内重编码这两次冻结实验：

```sh
docker exec -w /opt/jev -e PYTHONPATH=/opt/jev/src tianji-luna-web-guangyu \
  python -m luna_mujoco.experiment_reencode \
  /opt/jev/data/experiments/bbf7eefabc9e4e46a663035942d03bab \
  /opt/jev/data/experiments/ef356390f2a04c2b9d025532a4369ec5
```

输出包含每次实验的 `video_status`、`errors`、实际视频元数据和 `audit_path`；命令会原子写入 `manifest.json` 与 `video-metadata.json` 并刷新文件清单，无需手工填写成功状态。

2026-10-20 14:50 UTC：上述两次旧实验已在现有容器真实重编码并通过验证，视频状态均为 `ready`；新实验 `908fd4671f2b4832a0b32624c69a86e3` 也自然编码完成为 `ready`。六个 MP4 均通过帧数/尺寸/帧率/时长核对、ffmpeg 全流解码与 HTTP Range 206 读取。三次任务及原始结果仍为 `needs_attention`，原始帧、事件、结果和录像定义的 SHA-256 均未改变。没有重启容器或发送机器人命令。完整报告、原始状态与核验结果见 `data/video-reencode-20261020/report.md`、`report.json` 和 `audit-check.json`。

2026-10-21 独立完成件请求对照归档：`data/88d-second-observation/independent-tracking-comparison/`内completed与remaining各保存唯一一次真实请求、图片SHA、原始答案及锁；第二份由第一份实际回答生成。C6/C9二元yes为0.99/1.00，剩余两件0.89/0.92、两盒可信、none0.96；当前生产解析五ID和当前几何合并通过，9个低置信辅助提议仍unknown。原run终态不变，不能把冻结对照标成新搬运成功。


2026-10-23 回放与收尾：完整归档 GraspGen 服务返回的全部候选（包括超出本轮采样预算者），分别查看原始生成、方向/名义可达性预筛选及 +Z 深入 1 cm、batched IK、plan_grasp_batch 四阶段。每个候选显示按执行臂标定转换的完整 XYZ 坐标轴，X 红、Y 绿、Z 蓝；没有凭 Z 补造 X 或额外生成位姿。已归档最终预筛选索引包括可达性筛选结果；旧实验缺失的原始输出不补造。最后一件黑件 verify_drop 成功后，保留入盒动作障碍包络，双臂必须沿最新场景返回初始观测位姿，回位失败不报告任务成功，不重新建图。局部目标更新返回的碰撞实体清单包含保留的背景层。

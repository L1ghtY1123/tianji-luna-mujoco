# 镜像、场景入口与机器人模型验收

> 历史验收记录。物理场景所有权和 449 球配置已被后续变更替换；当前状态见 [场景与模型审核](luna-scene-ownership-audit.md)。下文保留当时的结果，不作为当前配置说明。

本轮变更已部署到 guangyu 的隔离仿真。Luna 工作台为 `http://127.0.0.1:8350/`，手工场景入口为 `/simulation-scene.html`。Base 为隔离网络 `172.29.79.2:8000`，Simulation 为 `172.29.79.3`。本轮未驱动真机、未构建镜像。

## 场景、视觉与控制边界

- Luna 选择 CAD，并把米制三角网格和凸分解件提交给 Base 的通用场景接口；Base 保存内容哈希对应的资源，Simulation 通过 Base 取材。Simulation 已移除 CAD 库挂载。内置两个示例是预转换网格，不依赖外部 CAD 库。
- `BaseSceneClient` 只注入独立的场景编辑路由；任务的 RobotGateway、感知和规划不读取编辑器配置。修改场景需要仿真模式、任务空闲、双臂禁用及正确的场景版本。录像仍在 Luna。
- 当前导入的 CAD 和新增箱体固定在配置位姿；原有动态物体保持物理运动。本轮没有扩展任意 CAD 的质量、惯量或动态抓取合同。
- Responses 按自然语言输出任务物体、来源、目的地及必要支撑的各可见视角框。程序在同次三张 RGB 上绘制框和标签，Decisions 接收这些图、描述与历史参考，确认主体和跨相机身份后才登记永久 ID。不可见视角不补框；背景仍在图中提供左右/序数上下文。矩形框不充当分割掩码或物体深度，后续需要像素轮廓时继续使用 SAM3。
- GraspGen 预处理及规划终点检查不再限制工具 Z 的世界方向。旧方向设置退役；TCP 换算、夹爪对称、FK 精度、碰撞检查及抓后沿世界 +Z 提升保留。
- Robot Bridge 不再依赖或查询 Action Stream Provider。有限轨迹、关节与夹爪命令沿用原生能力接口；运动互斥、取消和实测到位反馈仍生效。其他 VLA 插件的合法 Action Stream 使用不受影响。

## 镜像清理

本轮删除 5 个已停用的 `tianji-demo-*-guangyu` 容器、11 个旧标签和 2 个不再有引用的旧镜像 ID，保留挂载数据。此前已移除 SAM6D 镜像，本轮再次确认没有 SAM6D 部署镜像。没有全局 prune，没有删除其他维护者的标签或共享缓存。

| 服务 | 当前镜像名 | ID 前缀 |
| --- | --- | --- |
| cuRobo | `tianji-algorithm-curobo` | `63d527fe4dbd` |
| SAM3 | `tianji-algorithm-sam3` | `185489379ad9` |
| GraspGen | `tianji-algorithm-graspgen` | `47c4755a5598` |
| FoundationPose | `tianji-algorithm-foundationpose` | `59375529ad88` |
| Luna | `tianji-luna:0.8.1-guangyu` | `985aa90acad1` |
| Base | `jodell-console:0.2.0` | `0ea21fcaa67c` |
| Simulation | `tianji-simulation:0.1.0-guangyu` | `86c041c74314` |

算法名称不带版本、日期或用户后缀；Docker 显示的 `latest` 是隐式标签。精确 ID 和 SDK 版本分别记录在 `deploy/runtime-images.json` 与算法仓 `manifests/deployed-images.json`。构建入口只生成同一规范名称，默认拒绝覆盖，显式 `--replace --execute` 才替换；新的 cuRobo 完整构建配方本轮未重建验收。

`jodell-console:0.2.0` 是此前已有镜像，元数据 owner 为 guangyu、purpose 为 shared-console；这些标签不能证明具体操作者或聊天。本轮复用它，没有重新建立。Simulation 原维护者的 `xuan-cameras` 标签保留。

## URDF、碰撞球和工作空间

Base、Luna、Simulation 的无夹爪导线 URDF 模型语义相同，XML 排版不同。Base/Simulation 文件 SHA-256 为 `0868ff1ff658ddd1aa668a9dec8aa057ef6dc19a3a2a91fc31e0f787c5e6f8d1`，Luna 为 `889aa36cfaf10f4430b49321b6af8ffc8e1d352f72b56952a65ea511347b6366`。两个夹爪的 visual/collision 引用均已替换为去线网格，保留下来的显示顶点与原模型一致。主关节轴、原点、限位、mimic 保持，Luna 补齐 Base 已有的三个固定相机光学坐标系。Luna 与 Simulation 使用灰白本体/立柱、黑色夹爪/相机，Base 保持彩色。

仅修改 `Link_Stand`、`TCP_Link_L`、`TCP_Link_R` 的相机球集合；立柱原有 8 个本体球保留，其他本体和夹爪球未改。相机先经固定版本 cuRobo 公开拟合接口生成候选，再按每个相机表面子三角形扩大必要球半径，覆盖整片三角形并保留 0.5 mm 余量。最终三组球数为 19（含立柱原 8 球）、12、16，全模型共 449 个活动碰撞球。

**原本体球仍是近似模型，不是完整表面包络。** 对 35 个本体/夹爪 link 的网格顶点和采样审计发现既有漏覆盖；把相邻 link 球计入后，当前实测姿态下 Link5/Link7 的最大表面间隙约 13.5/14.5 mm，中央底座约 43.5 mm。地脚仍沿用原排除范围。本轮按本体基本不变的范围保留其配置，没有把这份审计写成整机全姿态覆盖证明。自碰撞检查通过只说明所检查姿态满足当前球模型。

工作空间通过 cuRoboV2-Fast 0.9.0 的公开 IK 完整重算，每臂检查 3,048,500 个候选点，分辨率 1 cm；边界、原点与尺寸不变。

| | A 臂 | B 臂 |
| --- | ---: | ---: |
| 旧可达网格数 | 2,775,358 | 2,773,939 |
| 新可达网格数 | 2,772,715 | 2,769,531 |
| 净变化 | −0.0952% | −0.1589% |
| 新旧交并比 | 99.8791% | 99.8142% |

新工作空间指纹为 `cd76132c40a116c6ffbc195d1b88b88e4bcb0ce52183ddc20aaddd0134b17758`。全部 witness 索引、解数量与最终可达网格一致；每臂等距抽取 4,096 个解，再用公开 FK/碰撞检查器验证，最大位置误差分别 1.9995/1.9871 mm，自碰撞和限位代价均为 0。当前实测双臂姿态和配置观察姿态也通过同样的自碰撞/限位检查。工作空间仅是单臂、固定本体、开爪的位置预筛选，不证明任意姿态、环境或双臂路径可行。

## 实际验证

- 临时回归探针：GraspGen 方向与配置迁移 5 项、无 Action Stream Provider 的 Bridge 合同 3 项、通用网格/CAD 边界与拒绝条件 4 项通过。
- Simulation：`python -m pytest` 运行场景运行时、相机模型、RGBD、场景边界、网格资源与同步的 6 个测试文件，共 35 项通过，另 4 个子检查通过。
- 算法仓：`python -m unittest discover -s tests -v`，52 项中 50 项通过，2 项因本地缺少 CAD 转换依赖而跳过。
- Luna `ruff check src ... --select F`、Base/Simulation/算法修改模块的 F 类检查通过；`vite build` 通过（52 模块，已有 Three.js 大包体积提示）；四仓 `git diff --check` 通过；仿真启动脚本 `bash -n` 通过。
- 远端 485 个官方 SDK 文件哈希与固定提交 `42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83` 一致，SDK 源码未修改。
- 浏览器真实导入 CAD、编辑位姿、应用场景并等到 Simulation 确认版本，通过后恢复原场景。Base 网格持久化目录由启动脚本初始化为服务用户可写。
- 新模型下三相机在同一帧组采集，偏差 0 ms，稳定约 0.804 秒且包含 8 个不同帧；公开 RobotSegmenter 返回三路机器人掩码。Base 浏览器实测 RGB、深度与点云均可打开，Luna 主台、工作空间页和 449 球页面加载正常，无 JavaScript 错误。
- 通过 Luna → Base → Simulation 执行 A 臂第 7 关节 2° 小幅往返轨迹，中途闭爪再开爪。轨迹状态为 arrived，夹爪实测分别为 0.8281179735 rad 和 0，回执均确认到位；结束后 A/B 均禁用。

证据在 `data/acceptance/runtime-cleanup-model-audit/`：镜像前后清单、网格编辑截图与回执、模型变更与球覆盖审计、工作空间比较、SDK 检查、同步采集和运动反馈。前次 Responses/Decisions 真实请求证据保留在 `data/acceptance/responses-rgbd-images/`。

本轮未重新执行完整自然语言抓放/拆包流程，没有真机运动或真机硬件曝光同步验收。复杂同色物体、盒中物和序数任务仍需对应场景的识别验收，不能从单个样例推断识别率。

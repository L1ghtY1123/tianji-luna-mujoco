# Luna 直连场景与独立启动验收

本次修正已部署到 guangyu 的隔离仿真。当前架构和三个独立启动命令见
[Base 接入说明](base-integration.md)，Base 自己的入口也记录在其仓库的
`docs/guangyu-simulation.md`。

- Luna 使用独立 `SimulationSceneClient` 直接读取、提交 Simulation 的通用场景和网格。
  RobotRuntime、RGBD、机器人与夹爪命令仍通过 Base。真机、未知/过期身份及忙碌状态禁止写场景。
- 删除 Base 场景转发扩展、Robot Bridge 专用准入方法、旧 Luna 场景客户端和联合启动文件。
  Base 旧场景 API 的 GET/POST 实测均为 404。
- Simulation HTTP 准入及物理编译检查本地控制器状态，并与命令处理互斥。
  启用双臂、上传/执行轨迹或夹爪运动时拒绝编辑，不依赖 Base 场景接口。
- Simulation 与 Base 使用各自的单服务 Compose，Luna 只管理自身与算法。
  Base/Luna 未启动时，Simulation 独立恢复原配置并通过健康检查。
- 从 Luna 实际提交原来的 5 物体配置，Simulation 编译并持久化修订 9；物体配置未变。
  过期修订请求被拒绝。两个原数据卷及全部运行镜像 ID 保持一致。
- 浏览器场景面板、设备准备和三相机预览通过，保持同一 document/标签页，没有 JS 页面异常。
- 三路任务 RGBD 均为 640×480、同一 sequence，偏差 0 ms；8 个不同帧稳定约 0.804 秒，
  深度有效且经过 cuRobo 机器人自身点云过滤。observe 最大关节误差 A 0.0493°、B 0.0490°。
- 15 项 Luna 客户端检查、29 项 Simulation 检查及 9 个子测试通过；修改的 Python 文件语法检查通过。
  Simulation 检查包括场景准入、真实 UDP 命令与场景编译锁互斥、轨迹参考、物理编译和场景边界。
- 七个本项目服务健康，双臂禁用、无活动任务；除本次重建的 Base/Simulation/Luna Web 外，
  其余 22 个容器的 ID、状态和启动时间保持不变。本次没有构建镜像。

`jodell-console:0.2.0` 的构建标签归属 guangyu、用途 shared-console，ID 前缀
`0ea21fcaa67c`。当前 Base 运行它并挂载 guangyu 的源码；其他用户可用相同镜像创建各自
真机实例，模式、机器人 IP、相机 Provider 和源码挂载由各实例决定。

证据在 `data/acceptance/direct-scene/`：`image-provenance.json`、`standalone.json`、
`live-scene.json`、`browser.json`、`rgbd.json`、`source-hashes.json` 和 `final-state.json`。
本次没有启动或驱动真机，也没有执行完整自然语言抓放任务。

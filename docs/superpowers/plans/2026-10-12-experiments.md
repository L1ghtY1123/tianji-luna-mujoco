# Experiment recording implementation plan

**Goal:** 实现用户确认的独立实验归档、双视频、可点击执行图及工作台整理。
**Architecture:** 后端新增存储/录制边界；Runner只添加观察器钩子；前端复用既有后端和事件数据。 SDK/vendor只读。
**Spec:** ../specs/2026-10-12-experiments.md
**Tech stack:** Python/FastAPI/NumPy/MuJoCo/ffmpeg; existing Vite/plain JS/Three.

## Global constraints
Ubuntu根目录 /home/guangyu/projects/tianji-experiments；recording不作为算法输入；保留默认规划与JEV控制；无环境变量/密钥归档。
## Review focus
- 开始/失败/取消/Reset之间的生命周期不得串实验。
- 保存和文件下载必须限制到有效实验ID的目录内。
- 少帧/编码失败必须显式报告，不能伪造视频成功。
- graph 重试和内部阶段归属必须使用调用ID。
- 1366x768的TASK完整可见，历史页不连接实时控制。

## Task 1: 数据存储与Runner/API集成（主代理）
- [x] tests/test_experiments.py 先验证生命周期、归档、路径安全和导出。
- [x] 新增 experiments.py/experiment_api.py，注入Runner；保存完整事件和数组，不改变skill执行。
- [x] 对取消、失败、video off与ask/none分别验证。
## Task 2: 媒体录制（独立模块）
- [x] 实现 experiment_video.py 的 VideoRecorder(path,simulation,started_at,generation)，start()/finish()。
- [x] 15Hz几何记录，2Hz三相机已有RGB预览，合成左2头3右1。离线渲染主视频，不驱动物理。
- [x] 编码使用ffmpeg，元数据包含真实采样时间与缺帧；测试时间重采样与拼接次序。
## Task 3: 回放页面（独立前端文件）
- [x] 新增 replay.html/replay.js/replay.css/execution-graph.js 与对应纯函数测试。
- [x] 使用已固定API实现列表、保存/丢弃/导出、同步视频、图和原始数据浏览。
## Task 4: 工作台整理（现有前端文件）
- [x] index/main/styles/planning-settings里实现记录控件、中央视图标签、参数dialog与分类。
- [x] 保持现有所有DOM绑定与实际参数逻辑；修正旧说明。
## Task 5: 集成部署验证
- [x] 主代理集成Vite入口/后端路由/部署实验挂载与ffmpeg。
- [x] 运行pytest、JS tests、build、ruff，修复回归：929 passed / 56 skipped，207 JS passed，Vite build与Ruff通过。
- [x] Ubuntu部署仅用户容器，真实左臂抓放录制，视频完整解码、抽帧和数据引用验证。
- [ ] 1366x768与1920x1080浏览器截图及实际点击验收：CUA获取已有tab和新建tab均连接超时，未完成，不能用单元测试代替视觉验收。

## 验收记录
成功实验 `048f24270629451aa8f1284bbfa08446`：92.54秒、20次JEV、14次skill、154条事件、4次RGBD观察；578处数组引用对应文件均存在。主视频15fps、三相机2fps，均H.264且ffmpeg完整解码通过；第85秒抽帧已目视确认蓝盒/红物体和左—头—右次序。HTTP Range返回206。精简ZIP 9,585,040 bytes、171文件、160项原始文件省略声明，CRC检查通过。

代码审查两项P2（初始化失败残留running记录、过短录像全黑却ready）已修复并定向复核通过。遥测不再重复采样夹爪滚动窗口，完整判定窗口仍在事件输入中。验收证据位于 data/experiments-20261012/。

测试注意：Python API测试读取web/dist，不能与清空dist的Vite build并行。Windows/Python3.12的50ms超时测试按系统15.625ms时钟精度容差断言；生产超时参数与执行逻辑未修改。

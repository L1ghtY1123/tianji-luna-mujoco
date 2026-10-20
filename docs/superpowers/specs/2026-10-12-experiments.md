# 实验记录与回放设计（2026-10-12）
用户已确认执行，实验根目录位于 /home/guangyu/projects/tianji-experiments，独立于源码。保持现有 JEV 决策与机器人算法不变。

## 行为和格式 v1
任务开始前设置 experiment: {name: string, record_video: bool, save_mode: auto|ask|none}；默认 name 为空、视频开启、auto。视频关闭仍保存事件与原始数据。none 不创建实验文件；ask 在根目录 .pending 下记录，完成后由用户保存/丢弃。auto 每次一个独立 run_id 文件夹。名称只放 manifest，不作为路径。
manifest.json 包含 schema_version=1、id、name、instruction、created_at（ISO UTC）、status、retention（saved|pending）、duration_s、record_video、video_status、size_bytes、files 文件索引、errors。config.json 保存任务参数/场景/标定/代码版本信息，不保存环境变量、密钥。
events.jsonl：现有事件字段 + t_s（从开始的 monotonic 时间）、invocation_id（一次 skill）、parent_invocation_id（可空）、artifact（相对文件路径）。每次调用有独立 ID，失败重试不覆盖。
artifacts 保存完整步骤输入输出，数组外置 npz（allow_pickle=False），JSON 存数组引用；observations 保存实际算法 RGBD/标定与 checkpoint。robot.jsonl 保存时间戳、关节/夹爪实际状态。frames.jsonl.gz 仅渲染回放用物理几何，不反馈到算法。视频 scene.mp4 与 cameras.mp4，后者固定左腕2、头部3、右腕1，三路水平拼接。task 时间与 sim_time 分别记录。失败/取消保留，写盘异常明确显示，不能虚报录制成功。

## 后端接口契约
GET /api/experiments -> {root, experiments:[manifest摘要]}，包括 pending。
GET /api/experiments/{id} -> manifest，含 files:[{path,size_bytes}]。
GET /api/experiments/{id}/events -> {events:[...]}
GET /api/experiments/{id}/files/{path} -> 原文件，严格限制到对应实验目录，拒绝目录穿越。
POST /api/experiments/{id}/save body {name?:string} -> manifest；pending 转 saved，也可改名。
DELETE /api/experiments/{id} -> {deleted:true}，仅删除 pending，禁止删除已归档实验。
GET /api/experiments/{id}/export?mode=review|full -> ZIP。review 包含 manifest/config/events/JSON数据/视频，排除NPZ和采样大文件；记录 omitted_files，离线包自包含已选内容，不冒充完整数据。
POST /api/start 扩展可选 experiment 字段。GET /api/state 的 run.experiment -> {id,name,retention,video_status,error?}。
GET /replay.html。链接支持 ?run=ID&view=graph&t=seconds。

## 页面
工作台保留场景编辑/中央视图+底部TASK/右侧日志。中央标签三维|三相机|点云；参数设置使用 dialog。宽屏使用全宽，高度由 viewport 与 TASK 分配；1366x768与1920x1080验收，小屏正常滚动。
回放页面：实验列表；视频/执行图/数据标签；视频同步时间轴；节点点击详情重用 trace 的数据语义。执行图依实际事件/调用ID生成，重试为独立节点；内部固定阶段可折叠，JEV 单独节点，不编码固定外层skill顺序。可独立新标签页查看图；导出SVG/PNG。
参数页面分类常用/实验调参/系统只读；保留所有已生效用户参数，不静默改算法默认值。显示当前值、项目默认、来源、单位、范围、修改影响。没有确认出处不可标官方默认。隐藏普通界面的自碰撞关闭开关及标定编辑；系统状态只读。修正陈旧说明。

## 验证
磁盘格式/索引、取消和失败、视频关、pending保存丢弃、路径穿越、export；JS图重试与归属、时间同步；全Python/JS suite/build；部署前保存备份；Ubuntu真实左臂抓放录制，ffprobe/完整解码，浏览器两种桌面尺寸确认布局与回放操作。记录性能与任何限制。

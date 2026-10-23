# 实验记录格式 v1

根目录默认为 `/home/guangyu/projects/tianji-experiments`，可以用 `LUNA_EXPERIMENT_DIR` 指定。Docker把同名宿主目录挂到容器，重建容器不会删除实验。每次运行的32位run_id同时作为目录名和API标识；用户名称放在manifest中，重命名不改变引用。

## 操作

TASK中填写名称、选择录制视频和保存方式。`auto`直接归档；`ask`先在`.pending/<run_id>`记录，结束且编码完成后可保存/丢弃；`none`不创建实验归档。关闭视频仍保存JEV、skill、原始观察和机器人状态。普通运行日志保留原有行为。

`/replay.html`提供实验选择、视频、执行图、数据三个视图。执行图以横向阶段子图分组，长重复过程可折叠；支持平移、缩放、适配视口。每个节点按实际调用ID关联事件，重试是独立调用。绿/红路径表示上一实际调用的成功/失败，未知另行标记；不会捏造未执行的策略分支。点击节点查看输入输出、耗时、原始数组文件，并定位视频时间。独立图链接带`?run=<id>&view=graph&standalone=1`；可导出SVG/PNG。

已保存或待决定的实验在任务与编码结束后都可删除，页面确认会显示名称、ID和大小。删除将目录原子移入实验根目录的`.trash/<id>-<uuid>/archive`，旁边`deletion.json`记录原路径；不立即销毁文件。需要恢复时先确认原路径空闲，再按记录移回，当前没有恢复按钮。运行/录制/编码、文件传输和ZIP导出期间拒绝删除。验收只使用临时测试归档，没有删除用户的历史实验。

## 文件

| 文件 | 内容 |
|---|---|
| manifest.json | schema_version、ID、名称、指令、UTC开始时间、状态、归档/临时状态、视频状态、错误、文件索引 |
| config.json | 实际规划参数、初始反馈和仿真场景、相机标定、源码/资源SHA256；不保存密钥或整个环境变量 |
| events.jsonl | 按追加顺序记录的JEV、skill、算法和系统事件；包含时间、调用ID和完整数据文件引用 |
| artifacts/event-NNNNN.json/.npz | 每条事件完整数据；JSON保留结构，数组外置NPZ，区别于网页精简显示 |
| observations/frame-NNNN.json/.npz | 实际算法观察节点的RGB、米制深度、有效像素、内外参、反馈；掩码和融合点云另见对应算法输出 |
| robot.jsonl | 目标15Hz实际关节、TCP、夹爪当前状态采样；不是1kHz控制日志，不重复写夹爪滚动历史；持物判断使用的完整窗口保存在对应事件输入中 |
| result.json | 最终任务状态、验证、错误与性能指标 |
| frames.jsonl.gz | 目标15Hz物理几何位姿，仅供显示重建，不提供给JEV或感知 |
| videos/scene.mp4 | 1280×720，15fps，实际保存位姿的渲染回放 |
| videos/cameras.mp4 | 1920×480，2fps，左腕2—头部3—右腕1拼接 |
| videos/camera_frames.jsonl | 原始预览采集时间、接收时间和图像引用 |
| videos/camera_frames/*.jpg | 同一采集组的三路RGB拼接帧 |
| video-metadata.json | 编码结果、样本数量、间隔、补帧策略、缺帧或编码错误 |

数组引用形式为`{"type":"ndarray","path":"…npz","key":"array_0","dtype":"float32","shape":[480,640]}`。用`numpy.load(path, allow_pickle=False)[key]`读取。NaN/Infinity标量使用`{"type":"nonfinite","value":"nan"}`等显式表示。数组原值和dtype保留；不会用彩色深度图代替米制深度。

## 时钟和完整性

`t_s`是相对任务开始的单调时钟秒数，视频和事件共用。`time`为事件Unix时间；`sim_time_s`/`sim_time`单独记录MuJoCo物理时间，不能用作JEV等待耗时。RGBD的`captured_monotonic_s`表示实际冻结传感器状态的主机单调时刻；不同机器的该值不能直接相减。

主视频在两个实测采样之间插值位置与旋转；相机采用前帧保持，首帧前为黑屏，尾部保留最后一帧。原采集时间与最大间隔保留，视频固定帧率不代表每帧是独立传感器测量。明显采样间断标为partial，不隐瞒丢帧。停止时冻结采样，编码不驱动物理或重跑任务。

成功、失败、取消都归档。`recording_complete`表示记录完整性，`status`表示任务结果，两者不能混同。后端被强行中断时，重启会把未完成记录标为interrupted。不能据失败或partial视频作完整时序实验。

## 导出

每次实验独立导出ZIP，无需上传根目录。`review`包含视频、事件、配置及JSON索引，省略NPZ、逐帧JPEG、robot.jsonl、压缩几何流，manifest的`omitted_files`列出所有省略项；`full`包含完整数据。MP4和NPZ在ZIP内直接存储，避免重复压缩浪费时间。回放网页读取固定根目录中的实验；ZIP是可移植数据包，尚不包含独立离线网页应用。

## 页面参数

顶栏参数面板分为常用、实验调参和系统只读。当前值、项目默认、单位/范围、作用分别显示。参数来源按项目与所锁定接口解释，不能把项目默认等同于所有cuRobo接口默认。相机标定、观察参考关节、自碰撞等系统项不在普通面板改动；已保存值不会因为整理界面而自动重置。

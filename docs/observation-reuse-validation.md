# 观测复用与模型证据传递验收（2026-10-27）

本轮修改已同步到本项目仿真服务。以下保留迭代中所有完整运行尝试，包含旧版本缺陷、模型未知及服务错误；它们不是同一最终版本的成功率基准。

## 行为与边界

- `observe_scene.capture_mode=reuse_current`复用有效三相机采集，保留原ID与拍摄时间。动作/命令/场景版本变化、机器人反馈变化、缺失字段或超过10秒窗口时新采集。模型显式要求`fresh`仍会拍新帧。
- 初始Responses参考图第一次绑定分析不因模型推理耗时而强制换帧；其他版本和反馈检查仍保留。该边界不证明环境绝对静止。
- 放置核验帧可以继续用于身份关联；回位后的有效帧可以继续用于最终完成核验。不同问题保持独立Decisions判断，不把图像一致当成成功。
- 同帧语义输入明确指出真正匹配的相机；图条件获得已核验工序及输入/产物账本。Responses仍选择节点、顺序和分支，runtime不补节点。
- 容器碰撞体素与放置区域分别诊断；不要求完整盒底。换帧后的已有支撑关系由Decisions重新确认，unknown仍拒绝，原数值校验不变。
- 未修改cuRobo/SAM3/GraspGen SDK、碰撞阈值、CPU配额或加入CAD地图。

## 真实运行

表中采集数是不同observation_id的数量；复用数包括初始参考分析，不等同于严格基准下节省的次数。VLA处理效果按项目约定由MuJoCo瞬时替换，抓放使用实际规划与仿真物理执行。

| 记录 | 状态 | 秒 | Responses | 采集/复用 | 结果与诊断 |
|---|---|---:|---:|---:|---|
| [observation-reuse-table-run.json](http://192.168.3.4:8350/replay.html?run=9067a5c027134bc894c2aa2becf8588d&view=observations) | needs_attention | 88.1 | 2 | 2/0 | 初始模型耗时导致参考帧被10秒窗口替换；已修正首次绑定边界。 |
| [observation-reuse-table-run2.json](http://192.168.3.4:8350/replay.html?run=a9aca0691c5643da95d7b2687d91206f&view=observations) | needs_attention | 164.9 | 1 | 2/2 | 抓取和放置通过；返回动作前模型选了selected地图。已补清空手运动的真实地图前提。 |
| [observation-reuse-table-run3.json](http://192.168.3.4:8350/replay.html?run=f6e65b0b548d444293721701ba81c1db&view=observations) | needs_attention | 85.9 | 1 | 1/1 | 源盒SAM3分数0.8945，Decisions身份概率0.77，小于0.8；未降低门槛。 |
| [observation-reuse-table-run4.json](http://192.168.3.4:8350/replay.html?run=63d54f4b26dc4788b3229225ed572acf&view=observations) | needs_attention | 111.0 | 2 | 1/1 | 身份关联通过；桌面支撑可见性概率0.62。后续补齐同帧来源说明。 |
| [observation-reuse-table-run5.json](http://192.168.3.4:8350/replay.html?run=9b01e87863ed40c4b57d443109fdaa4f&view=observations) | succeeded | 140.0 | 1 | 3/3 | 完整抓取、桌面放置、双臂回初始位姿和最终完成核验通过。 |
| [observation-reuse-vla-run1.json](http://192.168.3.4:8350/replay.html?run=037dd2df7d2447698d5b1111c6c1a85f&view=observations) | needs_attention | 139.3 | 2 | 1/1 | Decisions HTTP503，停止于运动前。 |
| [observation-reuse-vla-run2.json](http://192.168.3.4:8350/replay.html?run=e32a815a2a524019a8a8c1ab9eefaf4b&view=observations) | needs_attention | 188.2 | 1 | 3/2 | 抓放、拆包核验及产物登记通过；额外图条件未知。已给图条件补充已核验账本。 |
| [observation-reuse-vla-run3.json](http://192.168.3.4:8350/replay.html?run=12f5363298b54cf9bd7e2e9afaf4b7a6&view=observations) | needs_attention | 214.4 | 3 | 5/3 | 抓放和拆包通过；容器无放置区域，却误报占用失效，触发重复重拍。已拆分诊断。 |
| [observation-reuse-vla-run4.json](http://192.168.3.4:8350/replay.html?run=fbef474a6dc3492f961a9efa56ad02b6&view=observations) | needs_attention | 234.1 | 3 | 7/3 | 模型依据新诊断主动换观测位姿，容器区域恢复；产物支撑仍绑定旧帧使抬升检查拒绝。已增加当前支撑的Decisions重判。 |
| [observation-reuse-vla-run5.json](http://192.168.3.4:8350/replay.html?run=156192474cae4d6ca8e8365859a69095&view=observations) | needs_attention | 91.5 | 1 | 1/0 | 初始Responses传输错误，没有进入物理执行。 |
| [observation-reuse-vla-run6.json](http://192.168.3.4:8350/replay.html?run=32e2af486413452880b8c16ba2dbe3c7&view=observations) | needs_attention | 164.4 | 1 | 3/3 | 抓放、拆包核验和产物登记通过；缺少容器放置区域时，Decisions恢复判断返回HTTP503，未进入换视角恢复。 |

普通抓放成功记录`9b01e87863ed40c4b57d443109fdaa4f`仅三次采集：任务参考、放置核验、回位后采集。放置后身份关联与最终完成判断分别复用后两帧；最终`goal_satisfied`概率1.0，双臂姿态由实际反馈验收。此单次成功不证明任意抓放或拆包任务已经稳定。

**尚未完成的真实验收：** 拆包入盒整条链路本轮没有完整成功。最新两次复测分别遇到Responses传输故障及Decisions恢复判断HTTP503。支撑证据换帧修复已通过针对性及全量自动测试，但最新真实运行未到达换视角后的该分支，不能声称完成了它的真实物理验收。此前容器区域缺失通过模型选择换视角已恢复；原错误的底层原因是旧帧支撑证据，不是cuRobo已证明无解。

## 自动验证

- 最终全量后端3389项通过、58项跳过，350.52秒；结果见`data/observation-reuse-full-tests-final7.log`，保留已有Starlette弃用警告。前端367项通过，Vite构建通过，保留原有大chunk警告。
- 回放浏览器验证覆盖成功抓放及拆包中间记录：3次采集用途正确、原始时间保留、20px正文、390px宽度无文字溢出、无页面异常或写请求。
- 支撑换帧、同帧指纹篡改、显式fresh、动作版本、unknown完成条件与模型条件不越权均有针对性测试。
- 相机观测位姿验证记录通过真实渲染重建，1至4物体均可见且无机器人/源物接触；关节配置没有改变。

全部原始运行保留在`data/observation-reuse-*-run*.json`与实验回放；机器可读汇总为`data/observation-reuse-live-summary.json`。不得将服务错误、局部技能成功或合成测试改记为完整任务成功。

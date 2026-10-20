# JEV MuJoCo web workbench

Standalone Chinese workbench. The main Three.js viewport renders MuJoCo geometry and live poses; it does not compute or simulate robot motion in the browser. UI state, editable scene data, networking, trace rendering, and WebGL rendering are separate modules in `src/`.

All visible text must be at least 20px, including form controls, auxiliary descriptions, status labels, trace JSON, metadata, legends, and narrow-screen rules. Headings use 24–30px. Keep this minimum when changing the layout; use wrapping, wider panels, or scrolling when space is limited.

The center column orders the physical viewport and task controls first, followed by advanced settings, three RGBD cameras, and the folded point-cloud diagnostics. Both displayed task clocks use the same task-clock instance: idle/reset is zero, running advances, and terminal duration freezes. Internal sensor capture timestamps remain simulation timestamps and are not shown as task duration. Point-cloud diagnostics only display an existing mapping snapshot; opening them does not capture images or change filtering.

The default scene includes an open blue bin as well as the three small objects. MuJoCo 3.10 headlight ambient/diffuse settings are neutral 0.8/0.2 without specular glare, applied to both preview and task RGBD. The Three.js 0.180 viewport uses the same authored sRGB palette, neutral diffuse lighting and Lambert materials without filmic tone mapping. View-dependent shading can differ; sensor pixels are not recolored by the browser. The MuJoCo headlight settings follow its [versioned public visual API](https://mujoco.readthedocs.io/en/3.10.0/XMLreference.html#visual-headlight); upstream SDK source is unchanged.

## Run and build

Run `scripts/web-service.sh start` from the project root on Ubuntu. The single image `tianji-jev-mujoco:0.2.0-guangyu` runs the single container `jev-mujoco-web-guangyu`, with both Node.js and Python inside. No host Python virtual environment or Node.js installation is needed to operate it. Use the same script with `stop`, `status`, or `logs`; `--dev` is only a compatibility alias and selects no separate mode.

`deploy/web.compose.yaml` mounts the Ubuntu `src/`, `skills/`, and entire `web/` directories read-only. The directory mounts support editors that atomically replace files. Windows changes under `C:\project\tianji-jev-mujoco` must first be synced to `/home/guangyu/projects/tianji-jev-mujoco` on Ubuntu. Image-installed frontend dependencies are copied to a writable temporary `node_modules` mount at container startup; host dependencies are hidden.

`deploy/run-web.sh` runs Vite on port 8340 and Uvicorn on `127.0.0.1:8341`. Vite serves both pages and proxies `/api`, `/ws`, and `/assets` through `JEV_DEV_BACKEND_URL`. Frontend edits apply through HMR or a page reload. Python and skill edits reload the backend worker, clear the active task and in-memory scene, and recreate MuJoCo; the main page detects the new backend instance and reloads automatically. Make backend edits between tasks. Saved logs and planning settings persist in `data/`; the separate algorithm images, containers, and service entrypoints are unchanged.

Mounted source changes need no image rebuild or container restart. Dependency, base-image, Dockerfile, startup-script, or image-asset changes require `scripts/web-service.sh build` followed by `scripts/web-service.sh start`; these commands only update the web image/container. If either Vite or the Uvicorn supervisor exits, `run-web.sh` stops the other process and exits the container. A normal Uvicorn worker reload leaves the container running.

For local frontend tests and bundle verification:

Node.js 20.19+ or 22.12+ is required by Vite. Dependencies are pinned: Three.js 0.180.0 and Vite 7.1.9. `pnpm-lock.yaml` records the complete dependency graph.

```sh
cd web
pnpm install --frozen-lockfile
pnpm test
pnpm build
```

Vite builds `dist/index.html`, `dist/collision-spheres.html`, and `dist/web-assets/` together. The backend retains support for serving this bundle directly; the Docker entrypoint uses Vite to serve current source. No CDN or third-party frontend service is used at runtime.

## Contracts

- `/api/state` replaces the current state and complete event list.
- `/api/state` and `/ws` include `server_instance_id`, which stays stable across scene edits and Reset but changes when the backend worker restarts. The main page reloads once for a new instance before checking scene generation; ordinary reconnects retain the existing stale-snapshot protection.
- `/ws` supplies simulation and run snapshots. `events_mode: append` merges events by ID; `replace` clears the previous run's list.
- `/api/manifest` supplies `{ revision, geometries }`; snapshot `manifest_revision` signals geometry changes. Geometry IDs identify live `geometry_poses`.
- The camera panel defaults to independent live RGB preview from `/api/camera-preview`, with serial polling 500 ms after the complete frame finishes loading. Task mode reads cached RGBD from `/api/cameras` every two seconds. Both return generation, time and three frame-bound camera URLs; the panel validates their binding, decodes all images off-DOM and then updates the persistent left–head–right cards together. Image boxes keep a fixed 4:3 aspect ratio with `object-fit: contain`, and status text has reserved height. Loading failures retain the last complete frame. A mode or scene-generation change clears the old display and discards pending responses/images. Polling pauses while the page is hidden or the panel is folded; task mode does not initiate algorithm capture or request editor object metadata.
- The folded-by-default point-cloud panel reads `GET /api/perception-preview` only when opened or manually refreshed. Its dropdown contains only “原始 RGBD 点云” (the default) and “TSDF / ESDF 碰撞场景（表面预览）”. Raw mode shows all received measured samples in gray, including samples rejected by filtering; this payload has no per-point RGB colors. Collision mode shows the actual Mapper TSDF surface voxel samples, ESDF allocation bounds, and inferred RANSAC table cuboid. Gray instance layers remain explicitly marked as excluded from collision checks. This is not a rendering of the full ESDF distance field; native ESDF distance samples remain available in the expandable provenance details. Collision mode is disabled when no matching actual map is available. Legacy filter/cluster/camera-color modes are no longer exposed, while backend preview fields and diagnostic APIs remain unchanged.
- Opening the panel displays statistics while keeping the viewport clear. “显示点云” shows the selected overlay; “隐藏点云” in either the panel or viewport hides it without clearing statistics. Mode changes, refreshes, and in-flight responses preserve the hidden state until explicitly shown again. “隐藏仿真物体” independently hides configured objects (including every bin component and tabletop marker), the simulation table, ground, grid and axes; robot and camera meshes keep their live poses. The measured collision table cuboid belongs to the overlay and remains visible. Hidden objects cannot be picked or dragged. Leaving the panel, Reset, or a scene-generation change restores simulation visibility; reopening still requires “显示点云” again. Manifest reloads preserve the current visibility selection.
- The preview binds `generation`, `scene_token`, `observation_id`, optional `checkpoint`, `points_world_m`, `kept_mask`, `cluster_labels`, and full `counts` to `source: registered_rgbd`. A maximum of 30,000 points is sampled by original index (`preview_sampling: uniform_original_indices` or `stratified_original_indices`); displayed kept/rejected sample counts are separate from full counts. Stratified previews retain rare rejected points by sampling the groups separately; displayed proportions do not represent full-cloud proportions. Stratified responses require sorted, unique `original_point_indices` aligned to sampled rows. Reset/scene changes clear old overlays and reject stale in-flight responses. The overlay is an observation snapshot and can differ from moving simulation geometry; points remain visible through meshes. Display controls never capture RGBD, filter points, or write planning/editor data. `pointcloud-panel.test.js`, `scene-view.test.js`, and `main-flow.test.js` cover mode availability, provenance, mesh isolation, hidden-object picking and panel lifecycle.
- MuJoCo uses Z up and `[w, x, y, z]` quaternions. API payloads, draft data, and Three.js coordinates remain in meters. Length input fields display and accept millimeters (mm), converting only at the input boundary, with a 1 mm arrow step. Raw JEV/algorithm logs retain their actual API values. Manifest box sizes are half extents; cylinder sizes are radius and half height. Mesh JSON contains flat `vertices` and triangle `faces` in compiled MuJoCo geom-local coordinates.
- Editor `objects.size` and `table.size` are full XYZ dimensions. `table.position` is its center. Pose controls are XYZ and yaw only; yaw remains degrees. Cube/sphere size edits link all three axes; cylinder X/Y edits link the diameter, with independent Z height. Box dimensions remain independent. Bin dimensions are independent external length/width/height; `wall_thickness_m` controls the bottom and four walls, and `mass_kg` sets the single free body's total mass. A nonzero cavity is required, matching backend validation.
- Choose “桌面放置标记” as an object's type to create a colored placement region. Marker controls edit XY, full width/depth and yaw; height follows the tabletop and Shift-drag also stays on that plane. Each marker uses a different saturated red, orange, yellow, green or blue color, and its rotated rectangle must remain within the table. Its 0.2 mm visual layer has no MuJoCo collision contacts or free joint. Editor identifiers and dimensions are display configuration only: task perception detects a `marker-{color}` identity, floor samples and visible region from calibrated RGBD. Missing/occluded samples remain unknown, and disconnected regions with the same color are ambiguous. Marker edits use the same Apply/settle/confirm workflow as objects.
- “运行记录” preserves the chronological event flow and original display numbering. Every card labels its source: only exact `kind: jev` is “JEV 决策”; skill start/complete/error and internal execution stages are “skill 执行”; IK/planning batch progress is “算法计算”; remaining events are “系统事件”. The event total includes all four categories; “JEV 调用数” counts actual model calls. Only JEV cards show model-specific state/questions/output/confidence sections. Other events preserve their original data, even when they contain similarly named fields, except for the Batched IK display summary described below.
- `planning_attempt_failed` displays “规划无解，交回 JEV” as a warning-colored system event, with the failed arm, original error and duration. It records a pre-motion planning outcome; it is neither a JEV call nor a terminal task error. The next actual JEV event records whether to reselect an arm or stop. A rejected already-attempted arm and an unreachable/none selection remain terminal errors with distinct explanations.
- JEV state, questions, output, provider details, per-field confidence, duration, errors, and token usage are rendered from received data. Missing confidence remains “未提供”. `tests/trace.test.js` covers category labels, exact JEV classification, chronological numbering, and raw data preservation.
- Scene changes, Start, settle, and confirm are disabled while a run is active. Reset remains available and restores the backend-confirmed baseline.
- Hover highlights an editable object; left-drag moves it in XY, and Shift held at mouse-down changes height instead. Escape cancels only the unreleased drag and preserves earlier edits. Empty-space left-drag, right-drag, and wheel retain the camera controls.
- Drag release and numeric inputs update the same local scene draft. Multiple objects can be dragged before applying. Draft XYZ, yaw, and size immediately preview position, rotation, and scale while live snapshots update the robot. Applied bins also preview external dimensions and wall thickness around the shared rigid body center; dragging any component moves the whole bin. Table position and size also preview. Type/color changes and newly added objects appear after Apply. No scene request is made on drag release or numeric edits. Click “应用场景修改” to send the complete draft to `/api/scene`; then settle and confirm the scene. Underlying MuJoCo physics always continues, but unapplied preview transforms do not participate in its gravity or contacts.
- Apply or Reset failures preserve the draft. A successful response adopts the returned simulation snapshot and clears the preview, even when a follow-up state refresh fails. Reset restores the last confirmed scene. New objects and type/color changes appear after Apply.
- The visible clock is task elapsed time, not MuJoCo time. It starts on Start, freezes at terminal `run.metrics.run_duration_s`, and becomes zero on Reset. `/api/state` and `/ws` include Unix `server_time` for refreshing a running clock without depending on the browser's wall clock. Old generations/timestamps are discarded.
- Start includes the current `generation` with its instruction, so a delayed request cannot begin a task after Reset has replaced that scene. Existing API scripts may omit this optional field; the browser always supplies it.

Three.js integrations follow the pinned package's public `WebGLRenderer`, geometry, BufferGeometry, and `OrbitControls` APIs and embedded examples. Upstream library files are unchanged.

## 双臂工作空间查看器

主视图「工作空间 ↗」打开 `/workspaces.html`。左臂蓝色、右臂橙色，可分别隐藏、调节透明度、显示或隐藏机器人与场景；使用与碰撞球相同的旋转、平移、缩放和默认视角操作。场景姿态可跟随、暂停或手动刷新。

只读 `GET /api/workspaces` 读取与选臂相同的两张离线图，返回 Link_Base 坐标、原点、1 cm 分辨率、形状与六个方向的暴露面网格索引（NumPy C order）。保留全部边界和空洞，不用凸包或包络球代替；Three.js 每臂一个 BufferGeometry，只绘制表面，不绘制数百万内部方块。后端缓存表面，页面仅首次或后端重启后读取工作空间，姿态刷新不重新生成图、不请求 GPU，不采集相机、不修改场景或选臂依据。网格中心对应夹持参考点，朝向自由；边界存在网格离散误差，不能当作完整路径保证。

场景跟随使用轻量只读 `GET /api/workspace-scene`，仅返回仿真快照和服务实例标识，不重复传输任务事件历史。

`workspace-layer.test.js` 检查实际单位、面朝向、独立控制与释放资源；`workspace-page.test.js` 检查只读请求、静态图缓存、串行刷新及场景版本变化。后端 `test_offline_workspace.py` 验证内部共享面剔除、空洞保留和查询数据不变。

## 原始碰撞球查看器

主视图的「碰撞球查看 ↗」在新标签页打开 `/collision-spheres.html`。页面复用 `scene-view.js` 的相机、MuJoCo 网格、渲染调度和视角控制；`collision-sphere-layer.js` 将碰撞球按半径分成最多三个共享几何的 Three.js `InstancedMesh`，逐球使用 API 提供的中心和半径，不再次增加 buffer。半径不超过 17 / 44 / 175 mm 时分别使用 detail 4 / 7 / 15；更大的球提高大球组的共享细分精度，球面三角片的最大内缩不超过 0.2 mm。显示细分不改变规划球或真实几何缺口。

只读 `GET /api/collision-spheres` 返回 `source`、`gripper_state: open`、`radius_policy: original_yaml_plus_link_buffer`、`sphere_count`、`link_count`、`spheres: [{id, link, center_m, radius_m}]` 和 `simulation`。后端必须在同一个只读快照中生成球与 `simulation.geometry_poses`：双臂跟随当前姿态，夹爪固定张开。原 YAML 和主仿真姿态均不修改。页面拒绝闭爪/包络元数据、无效球和不完整计数，避免把别的几何当成原始规划球。

`collision-sphere-page.js` 在每次读取完成后等待 500ms 再请求，可暂停跟随或手动刷新；仅使用 GET，不连接任务/场景修改接口。`simulation.manifest_revision` 变化时重新读取 `/api/manifest`，版本不一致则重试，避免新场景网格配旧快照。可单独隐藏球或场景网格、调整球的不透明度。此页核对开夹爪规划几何覆盖，不进行碰撞判定。

`tests/collision-sphere-layer.test.js` 使用真实 Three.js 验证原球半径、坐标、全量绘制、资源释放和错误数据拒绝；`tests/collision-sphere-page.test.js` 验证球/网格同快照、只读请求、串行刷新、暂停和场景重建竞态。浏览器视觉与真实机器人几何核查仍需实际后端验收。

`tests/model.test.js` checks scene validation, quaternion-to-yaw mapping, run control gating, trace provenance, incremental events, and per-field confidence. `tests/object-drag.test.js` checks drag projection, local draft updates, cancellation, and draft position precedence over live snapshots. `tests/scene-preview.test.js` uses real Three.js meshes to check preview rotation, full-size scaling for each primitive, table edits, and restoration of live transforms. `tests/main-flow.test.js` executes the page entry point with DOM/network boundaries replaced, checking millimeter input conversion, linked dimensions, shared drag/form edits, explicit Apply, failure preservation, Reset, and stale request ordering. `tests/task-clock.test.js` checks Start/Reset transitions, stale snapshots, terminal duration, and server-time calibration. Browser and real-backend acceptance is separate from these checks; build/unit checks alone do not establish visual interaction or physical grasp success.

## cuRobo 参数与 batch 进度

`planning-settings.js` 读取后端 Pydantic schema，按共享 IK、批量规划、抓取、放置、点云过滤、观察位姿、持物球、执行时间分组生成大字号输入。网页长度 mm、角度 °，API 使用 m/rad。除执行时间外各组默认锁定；点击「解锁高级参数」后才能编辑。执行时间的倍率与停留仍可直接修改。重新锁定只关闭编辑权限，保留未应用草稿；要保存用「应用算法参数」，要丢弃用「重新读取」。三类 seeds 分别是 IK 优化、LM 初始解生成和 TrajOpt 的预算；random seed 是随机序列编号。碰撞间距、抓取几何和执行时间属于项目配置，不能把整个表单称为 SDK 默认值。

`observation.A/B` 分别显示 7 个本机器人关节角，限位来自机器人 URDF，默认来自标定资源；请求位姿仍需运行时 FK、碰撞与相机覆盖验证。`attachment` 提供 10 mm 默认分组尺寸、1 mm 默认附加余量及最多 16 球的预算；沿最长轴分组，每组比较三个快速球心估计并取较小覆盖球，不截断实测点或补全隐藏表面。球拟合结果缓存复用。附件设置修改作用于下次抓取，当前持物绑定保持原参数。编辑器递归读取对象和固定长度数组，未修改的角度保留原始 rad 精度。

`run-evidence.js` 从 `run.holding`、`run.collision_model`、`run.verification` 提取显示摘要。主状态栏持续显示持物/释放待确认、碰撞模型、视觉验证与几何准备耗时；运行事实保留测量摘要和表面球数量/分辨率，排除原始点数组。`current_step.stage` 及 `stage_started_at` 显示内部阶段及耗时，不新增 JEV 决策事件。Reset 后旧持物摘要清空。

草稿需应用才能 Start；POST 携带 expected_revision，Start 携带 settings_revision，跨页面冲突需重新读取。Start、运行中或算法 pending 会重新锁定高级参数，运行中与 pending 时所有参数禁改且不能解锁；结束后高级参数仍需手动解锁。Reset 保留参数，单轮 run.planning_settings 展示实际快照。`tests/planning-settings.test.js` 和 `tests/main-flow.test.js` 覆盖默认锁定、手动解锁、重新锁定保留草稿、执行时间可编辑、运行/pending 禁解锁。

`planning_batch_start/complete` 和 `ik_batch_start/complete` 记录 batch_index、batch_count、candidate_indices、arm、duration_s、success_count。plan_grasp skill 内部仅 batch；完成全部 IK 可行候选批次后，选择从当前姿态开始、沿接近/抓取/抬升实际指令的七关节累计绝对路程最小的完整可行轨迹。GraspGen score 只决定调度顺序和路程相等时的优先级。旧记录的 stopped_early 提示保留其历史含义；当前规划不提前停止。参数含义与边界见 `../docs/planning-parameters.md`。

运行记录和实验回放根据 `data.planning_stage` 的 grasp/place 区分 `plan_grasp_batch` 与 `plan_place_batch`。旧记录可通过调用 ID 关联的规划 skill、放置批次元数据或原有放置标题恢复阶段；证据不足时保留原有标题，不按相邻事件猜测。执行图、节点详情和导出图使用相同显示名称，原始算法函数、事件 JSON 与下载文件保持原样。

Batched IK 卡片和当前步骤栏只显示候选数量；完成的批次显示通过与未通过数量。展开的 IK 事件及 `evaluate_arms` skill 数据也把 `candidate_indices` 列表换成 `candidate_count`，兼容旧记录的 `{count, first}` 摘要。此转换只用于网页显示，保留逐候选错误及其他诊断字段；原始事件、算法结果、后端日志和 JEV 的 state/questions 均不变。plan_grasp_batch 的候选编号显示保持不变。

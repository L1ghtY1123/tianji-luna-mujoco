# Perception, inventory and measured mapping

Current contract: 2026-10-26. Historical experiments below describe their original implementation and results; they are not current task orchestration rules.

## Model and runtime boundaries

`agent/responses.py` compiles natural language and initial RGB into `TaskSpec`, `policy_graph`, entity proposals and `task_inventory`. The inventory binds candidate/selected IDs, source, destination, optional work area and measured support. It does not prescribe a transport route. Registered skills execute the graph's explicit nodes; conditional branches, bounded retries and target loops remain in that graph. Normal compilation uses one Responses request; format repairs and justified revisions share at most two additional requests.

`DemoPerception` receives the controller's inventory. SAM3 proposes masks; Decisions associates actual visible subjects with permanent IDs and confirms task membership. A proposal, a recorded action or simulation truth cannot establish visible identity. Original inputs retain their IDs; verified processing produces distinct output IDs and explicit transformation lineage. Consumed input IDs are retired in `inventory_state`; current entities and historical references remain distinguishable.

## Current identity and result evidence

Every Decisions request uses the relevant frozen CURRENT RGB views, with INITIAL or BEFORE references clearly marked historical. Full images, candidate labels and calibrated locating crops have separate roles. Duplicate image bytes may share a reference, but no unique image is removed. Camera 1 is the right wrist, 2 the left wrist, and 3 the head; camera number and image position do not assign task roles or IDs.

The generic identity path uses canonical identity and visibility questions. `scene_inventory_closure` asks about visible unlisted candidates relevant to the TaskSpec target selector and source. A `none` result does not establish that occluded contents are absent. Source completeness for an all task is a separate task-level judgment. Geometry must still be present for each required current entity. Uncertain supplementary candidates do not enter the inventory map.

Existing validated container correspondence and completed-transfer tracks can provide bounded identity context. These require their original observation, generation, geometry and release bindings; they are not a special task workflow or permission to infer unseen state. `inventory_reference.py` supplies sanitized current identity and transformation fields to Decisions, without leaking simulator poses or full physical receipts.

`verify_place` binds the actual physical release to fresh image evidence. It asks `place_result` and `workpiece_still_held`, each with the unchanged yes/no/unknown thresholds, and only confirms the transfer when both support the requested location and empty grasp. A missing fact remains unknown; no separate meta-question rejudges the same two facts. Rechecking preserves the original release. A confirmed intermediate transfer records a location; task completion separately requires the final TaskSpec destination and processing history.

`robot_skills/processing_evidence.py` implements the configured unpack result check. BEFORE images are the verified work-area release; CURRENT images follow the bound processing action. unpack_result, wrapper_removed and observation_sufficient must pass. Current calibrated depth must support the independent product identity and local obstacle update. Both arms’ position workspace results are diagnostics, not conditions for whether unpacking occurred. Verification never moves. A subsequent explicit observe_scene can reuse this valid capture for ID/geometry association; the simulation output pose is never a visual measurement.

`robot_skills/bin_drop.py` verifies arbitrary bound target/container descriptions using CURRENT/BEFORE images and the measured container region. `current_item_added` and `observation_sufficient` must pass. Old contents moving, historical counts or an action receipt cannot prove arrival. Conservative release envelopes remain labeled action-derived. Any chosen observation movement, repeated check or user-requested final return is an explicit graph node; unpacking itself imposes none of these motions.

## Configured regions and numerical checks

An unmarked configured XY region has an operator-supplied identity. Code binds its polygon to the measured support height and records `configured_xy_measured_support`. Decisions checks `<role>_support_visible`, asking whether the bound supporting surface is visible; it does not re-identify invisible coordinate boundaries or judge metric reachability. Unconfigured physical entities still require visual identity confirmation. Arm workspace, collision, coverage and frame/version checks remain numerical.

Experiment `751722429842485d9461669713132642` stopped before motion because `work_area_identity` returned 0.61 against the unchanged 0.8 acceptance threshold. The question incorrectly required RGB recognition of an unmarked configured work area. The new `configured_bindings` plus `work_area_support_visible` separates supplied configuration from visual evidence. The source scope was also not confirmed complete. This diagnosis and correction do not establish complete physical-task acceptance.

## Measured geometry and coverage

Only observed inventory entities enter the occupied map. Static table support is measured from the initial calibrated RGBD and retained with provenance. Known CAD dimensions are declared priors, not measured poses; completion requires compatible measured support/orientation evidence. An asset ID alone cannot prove pose. Container registration publishes the measured transform, CAD hash and collision representation; local-object depth is tied to its actual capture.

Partial observations retain previously measured entities. Explicit motion invalidates affected geometry; verified transformation retires its consumed input. A local measurement, predicted release transform or conservative action envelope does not become a fresh full-scene map. Original calibrated depth is used separately for visibility/free-ray evidence, without adding unlisted background to occupancy. Three cameras and a successful identity judgment do not prove motion-space coverage. Unknown, occluded, invalid or stale regions cannot be treated as free. See [mapping and motion safety](mapping-motion-safety.md).

The implementation retains the pinned official SDKs and numeric safeguards: 2 cm TSDF/ESDF, default/minimum 10 mm trajectory clearance, and attachment defaults of 24 spheres with a maximum of 31. No SDK source is modified by the task-orchestration refactor.

## Historical verification records

The records below describe the implementation and experiments at their recorded time. Older six-question unpack, per-camera place, and two-question drop statements are historical, superseded by the current sections above and [question inventory](decisions-questions.md). They do not claim validation of the present change.

The f9bd first transfer and retained-map return reached the next observation, where the completed-object locator rejected two camera-3 SAM proposals before Decisions. Each had 729 current points; only 4/3 were outside the slot, all silhouette-edge returns at the measured table height. The existing ±6 mm support band removes 27/28 support-adjacent samples and leaves 702/701 measured points fully inside the unchanged unique prism. Scope-only filtering now records both raw and matching evidence without altering model images, raw candidates or the final geometry policy. Frozen regressions first failed, then 79 focused tests and Ruff passed, including uncertain correspondence and outside/ambiguous/nonfinite geometry. Evidence: `data/f9bd-completed-tracking-diagnostic/`; exact test arrays: `tests/fixtures/f9bd-completed-proposals.npz`. No new real model request, capture, motion or old-run outcome change was made.

The 2026-10-21 color-contract regression uses the exact parsed answer from stopped run `8bfd28e58f12479bbbf33741f835ea85`, frozen in `tests/fixtures/demo-discovery-8bfd.json`. Before the fix, the HTTP-boundary test reproduced `demo_blue_source_bin_unconfirmed` for its `蓝色` source box. After the fix, the same answer yields three pink source IDs, canonical downstream colors, the unchanged generated program and the unchanged raw audit answer. Discovery/query/perception/inventory tests passed 63/63 and Ruff passed, including Demo-only aliases, whitespace/case, rejected unknown/red/mixed colors, and unchanged ordinary entity colors. This is a parser regression check, not a new model call or live transfer result.

Focused tests cover actual Responses/Decisions parsing through HTTP boundary doubles, same-capture guards, four bounded queries, source subset/type preservation, mask/cloud evidence including failures, proposal deduplication, partial-camera union, post-place target scope, visible extra/unknown semantics, inventory-only depth, tabletop removal, table caching and separately labelled completion. Latest result: 164 passed, 1 deselected. The deselected legacy runner test imports the removed `decisions` module; that test is outside the active demo runtime. Mapping tests use the real project builder with CPU doubles at the GPU boundary. These checks do not replace live model/GPU or physical-transfer acceptance. No remote services, models or simulator motion were invoked for this policy change.

The 2026-10-20 reference-view follow-up compared frozen runs `3c5879a7dfea43dd8b4bd01a7ea127c4` and `c6b5e999ebdd41b8ac85a54b5cb1516e`. All three original JPEG hashes match between runs. The stopped run has trusted left-source coverage but lacks a trusted identity for the right box: head identity confidence 0.79 and right-wrist misidentification confidence 0.63 are both below 0.8. Its right box used a head reference; the earlier run used a right-wrist reference and had trusted right-wrist identities. The `demo_unlisted_pink` confidence 0.74 remains unknown, but is not the first stopping condition. The prompt preference addresses this evidenced reference-quality difference without bypassing complete-union or identity checks. Tests for discovery context and identity/same-capture/verification/unlisted boundaries passed 60/60, with Ruff clean. Frozen evidence and reproducible summary are in `data/identity-third-diagnostic/`. No new model call or live transfer was used to establish an improvement.

The subsequent count-only correction passed 92 focused tests across discovery, the generated runner/program, partial identity, same-capture safeguards, unlisted checks and release verification, with Ruff clean. Tests retain unknown completeness, freeze only the requested subset/slot count, preserve unselected obstacles, and reject low-confidence/stale/unknown/extra checks before the first or next manipulation. It does not fix run `5f8ce17e1ae540b4810fb30bb8870480`: that run's completeness was confirmed and its right box had trusted wrist identities, but the leftmost pink object's identity confidence was 0.67. Its unknown remains a stop; evidence is saved in `data/identity-requested3-diagnostic/`.

The 2026-10-21 presentation follow-up passed 119 focused tests and Ruff. Exactly one authorized Decisions request reused frozen run `5f8ce17e1ae540b4810fb30bb8870480`, with all full RGB images and canonical choices unchanged. Left/middle/right pink identity confidence became 0.97/1.00/0.99, both bins had trusted wrist identities, and visible-unlisted was `none/0.97`. Required union and left-source identity coverage were complete. Nine uncertain head proposals remained unknown; offline parsing with original SAM arrays discarded them under the existing rule. Right-bin visibility separately returned `unseen/0.38` and remains unknown; it was not promoted to visible. The complete comparison, including every identity answer, is in `data/identity-requested3-diagnostic/report.md`. This is one frozen identity comparison, not a live mapping/manipulation or general reliability claim; no retry or service restart was performed.

2026-10-21：角色请求明确只有初始粉件来源清单冻结，当前entity/process记录还包含拆包确认后新建的独立黑件；不能把新产物说成初始粉件清单里的原实体。此处仅修正上下文说明，原问题/图片/候选/.8门与身份检查不变；三份简化/分组提示探针没有作为稳定修复部署。

Unpack verification now sends the three validated BEFORE table-release RGBs as
well as the three CURRENT action-pose RGBs. The head detail uses the complete
cached measured main-table support, bound to the task destination table and its
initial RANSAC source. It is not a new scene or an object mask. BEFORE images,
repeated camera/detail appearances and products inside bins do not count as
additional CURRENT tabletop workpieces. The six answers and confidence gates
are unchanged; missing or altered reference pixels still stop before the call.

当前拆包核验校验 BEFORE 释放参考、动作回执、owner/generation、输入物体、工序与请求/图片哈希。Decisions 综合三视角判断产物出现、包装移除及观测充分性；阈值保持0.8。视觉与深度成功才创建独立产物ID并退役输入，unknown不推进。核验不选择后续路线，也不要求预设观测姿态。消费者仍核对当前采集、身份关联、完整测量与地图；是否重新采图或移动由Responses图决定。具体实现与结果绑定见 [技能手册](luna-skills-guide.md)。


2026-10-21 抓取前职责分离：lifecycle_candidate_scopes继续用同帧三RGBD点云、缓存桌面与已验证盒体检查3D支撑范围，Decisions的identity_N题只确认C候选可见的粉/黑工件外观并排除机器人、盒和背景，不重复要求RGB证明桌面接触或所在盒。保留0.8置信、未知不进入地图、源件左腕覆盖、后续INITIAL/CURRENT粉件原ID核验及全部工序绑定。completed_collection仍可为一个或多个可见黑件/局部，不分配视觉成员ID。v11原图对照：外观题10/10 yes/confidence1；交换要求颜色的负例10/10 no/confidence1。相关请求、原始JPEG及响应存于data/v11-role-diagnostic，历史v11失败状态保持。中止无关全量测试并保留部分日志，不将其声明为完整通过。


### 2026-10-21 v12投放核验误停定位

run `f5158f7382344acfb846b4b33b5e0565` 第一件在右盒投放后停于 `verify_drop`：337为实际投放，340为新观测，343为Decisions原始回答，344为解析失败。CURRENT右腕及头部原图可见盒内黑件；8张实际输入JPEG逐一SHA匹配，重建请求hash为 `f23d5ad8a8d1e998f52eb6071f4523d489c6edd3d2833e6b934b72d97bcec783`。capture晚于drop反馈，排除取错旧帧。原回答in_right_bin=yes/.79，current_item_added（同一盒内至少1件）=yes/1.0。旧代码对重复存在条件也设独立0.8门，导致已确认盒内数量仍被拒绝。

`drop_result_evidence`统一解析当前协议：必要条件是同一盒内N+1可辨别数量题predicate概率≥0.8，in_right_bin高置信no为反证；低置信或unknown原样保留，不再重复AND。后续清单消费者`_visual_drop`及离线audit_process_identity均从原始answers重算并核对全部语义字段，旧unknown结果、缺字段、伪造confidence或数量不足不能升级为成功。问题、原图、0.8门、执行链、1cm碰撞阈值、场景更新边界均无变化。第一件的原始失败实验仍保持needs_attention。

原始响应回归先复现失败再通过；证据与逐图哈希在data/v12-drop-diagnostic，read-only下载脚本为data/diagnose_v12_drop.py。此次只使用已有存档和离线测试，未新调用视觉模型、GPU规划、仿真运动或建图。此修正不等于完整两件流程已经验收。

### 2026-10-22 CAD registration nearby hypotheses

A single fit retains the 3 mm CAD-surface residual tolerance and 85% minimum agreement. Multiple fits are compared modulo the declared half-turn symmetry using the maximum pairwise displacement of all eight CAD bounding-box corners. At most 6 mm (two 3 mm measurement tolerances) is one local pose group; select its highest agreement, then lowest p95 residual. Every pair must satisfy this bound, including rotation at the rim; connected-chain merging is forbidden. Larger differences still raise demo_bin_cad_pose_ambiguous. The 10 mm trajectory collision threshold, CAD meshes, measured points and 2 cm mapping remain unchanged. Registration records candidate count, maximum surface displacement, tolerance and metric. V16 source/right frozen inventory fixtures preserve their original points and hashes; expected simulator coordinates occur only in test assertions.


2026-10-22 current unpack-transfer release contract: both predicates remain at 0.8/0.2. Include current-capture measured gripper position and open/closed limits after validate_release_capture, without treating opening as proof of destination. The held predicate distinguishes physical contact retention from projected overlap. Keep INITIAL references, then CURRENT cameras 1/2/3 with head last. Table color is configurable, so neither release nor unpack questions require a grey table. This supersedes the older CURRENT-only description above for the unpack workflow.


2026-10-27 放置视觉输入隔离：verify_place仅提供永久目标ID、历史目标外观裁剪、三路CURRENT原图和实测夹爪开度。原始大场景历史位置、整个任务的后续工序、数值XY多边形和不可见区域投影不进入该视觉问题；XY/姿态/版本仍由原数值检查拒绝无效输入，并保存在结果证据中。最终全量清单核验若被要求，单独提供其选择范围。历史裁剪只作外观参考，不证明当前位置或释放成功。阈值仍为.8/.2，unknown不提升为成功。


已确认的中转放置与最终任务完成分离：新观测可使用仍为observed的对象的已验证释放身份轨迹，仍逐项核对owner、generation、目标/目的地、动作索引、释放反馈、原视觉证据、位置范围与新采集时间；不把中转标为任务完成。模型选择下一次搬运时，清除上一搬运的当前关联约束，保留已确认搬运历史。新工序产物若没有初始图中的边界框，放置核验使用该独立ID的已登记外观描述，绝不裁剪已消耗输入的旧图来冒充产物参考。


当前图的首次associate_inventory也建立容器RGBD对应参考；其前提仍是reference/current同一采集、完整Decisions身份确认、当前清单无缺失、原mask/depth/calibration绑定。后续采集不能冒充初始化，移动/遮挡/校准变化不满足原几何门槛时仍回到Decisions判断。该修正去掉对旧initialize_inventory入口名称的独占依赖，不添加任务节点或放宽几何阈值。


容器对应参考在初始观测成功后写回Responses预置的同采集reference，不能被setdefault静默丢弃。参考作用域只绑定容器自己的身份字段；位置/测量时间刷新或无关物体生成退役不使静止容器身份失效。容器身份字段、generation、owner、RGBD锚点哈希及当前几何变化仍逐项核验。


已确认释放后的当前候选关联，将数值区域归属与视觉工件判断分开：当前注册深度已通过唯一释放区域范围检查后，Decisions仅判断C候选是否为独立工件；不绘制不可见XY区域，也不让RGB重新测量它。当前候选仍须fresh yes且置信度>=.8才能贡献几何；no/unknown候选丢弃，任一可靠视角可以确认该ID。位置范围本身不构成身份确认。

已确认容器的RGBD追踪与源物体清单穷尽性分离：初始容器身份、当前有效测量和同采集绑定仍必需，但scene_inventory_closure的unknown/unlisted不撤销已知容器的独立身份；原完整性判断保持原值，不能借此宣布清单完整。

容器RGBD对应的掩码比较按同标定深度区分可见性：当前更近深度遮挡旧容器像素、原先更近前景被移走后新露出的像素不参与共同可见掩码IoU；保留原0.85 IoU、0.95有效比例、512共同实测像素及20mm深度残差门槛。无效/缺失深度不能解释遮挡，旧容器像素退到更远处不能当作被遮挡。记录原始IoU、共同可见IoU及两类像素数量；不补造碰撞几何。


放置核验对同一次采集的三路RGB分别调用Decisions，各自只携带同一目标外观参考和本路CURRENT原图，两项判据与.8/.2阈值不变。任一路同时确认桌面物体与未夹持即可作为完整见证；任一路明确判断仍被夹持则拒绝确认。未看见/不确定的视角不否定另一完整见证，不跨相机拼凑两项事实，不制造平均概率。结果保留三个请求哈希、原答案与选中见证；清单完整性另用全视图独立请求。此为一次观测内的证据处理，不移动、不补采集、不改变图节点。三路HTTP并行，Responses调用数不增加。

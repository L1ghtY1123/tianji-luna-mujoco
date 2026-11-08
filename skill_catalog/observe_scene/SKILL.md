# observe_scene

Previously delivered objects use the same instance association as other active inventory objects. Supply current candidates across available views, historical reference images, product references and operation history. Decisions judges identity independently from current visibility. Accepted current masks and depth supply new geometry. Transfer completion remains bound to its existing verified release record.

Identity input retains the current observation ID and dated, visually confirmed release locations. When a release confirmation and the current observation share the same capture ID, expose that relationship explicitly. Pending releases supply no confirmed location. Product reference images retain their observation ID and historical-image meaning.

Each identity request includes already accepted identity bindings from the same capture and their current labelled views. Confirmed cross-camera aliases share those bindings. Uncertain answers remain unresolved. Robot parts and background remain outside the object inventory.

The named current candidate also receives an exact RGB rectangle crop, labelled as current subject detail. Full current views and historical references remain available for context. The crop preserves source pixels and makes no identity claim.

Associate current SAM3 subjects across three calibrated RGBD views with permanent inventory IDs. Update unknown objects as OBBs, known objects as FoundationPose CAD poses, and tables as measured cuboids. An explicit unknown_obb object skips CAD retrieval. Confirmed CAD model bindings persist; new poses use the best valid-depth camera, or instance-seeded tracking when its physical motion context is unchanged. The runtime supplies cad_motion_context from object operation history and spatial invalidation, never task or graph progress. Tracking estimates a new pose from the current RGBD; it does not publish the old seed as a current measurement. Accept source-bound FoundationPose poses directly, without point/depth fit validation or ICP. Both capture modes reuse unchanged sensor evidence; elapsed time or an unknown answer alone does not cause a new acquisition. Identical successful analysis is reusable; identical failed analysis is not repeated. Keep unobserved inventory records as dated history. This skill neither moves the robot nor publishes a collision map. Use a legal changed viewpoint when evidence is insufficient.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.

Use this capability when current instance masks and geometric poses are needed. It is not a prerequisite for every Decisions request: release verification, visual task completion and visibility diagnostics can consume current three-camera RGB directly. Prefer a graph-authored configured return before recovery observation when its empty-hand and map prerequisites can be established. Extra wrist viewpoints still require one visual-limitation judgment; uncertainty alone is not proof of occlusion. No movement is inserted by this skill.
当前身份结果提供主体定位与可见事实，关联不确定单独保留。后续新采集由 SAM3 提供候选，Decisions 关联既有 ID 或确认新实例。清单变化更新状态和历史，整体清单核对不作为观测或执行前提。同采集在完整相机图中同时绘制 R/C 标签，跨采集只加入需要的参考视图。questions 直接写明候选主体、参考描述和同主体的其他当前视角。当前掩码以轮廓形式展示，哈希和标定有效性由程序核验。

Success and observed=true mean this capture was analyzed, not that every pending input has current geometry. A missing input stays in inventory history and identify_target returns unknown for it. Two cameras are wrist-mounted: after manipulation they can face the last work area or destination, while an arm can block the fixed camera. capture_mode=fresh does not change that viewpoint. Choose sensing posture for the next required subject and provide a bounded visibility recovery route where legal; do not assume a successful global observation sees every remaining object.
## 几何掩码清理

非容器物体在现有一像素轮廓收缩后，对每个独立连通区域应用现有的 30 像素候选下限。所有达到下限的区域共同参与当前几何，原始 RGBD 保留。观测诊断记录下限及几何掩码剔除总数。

身份选项保留 Responses 为每个实例生成的完整参考说明，并明确标记为历史描述，包含相机和同类实例间的位置区别。

身份关联前，使用已有 URDF 自身表面射线匹配、采集时的关节与夹爪反馈、三相机标定 RGBD 过滤机器人返回点。物体与容器候选都参与。保留真实接触歧义和原始 RGBD，过滤只确定机器人表面归属。

当前 Responses 的身份提议随观测传递，包含采集编号、相机和候选框。每题只比较当前候选与一个永久实例。当前定位提议和历史外观查询只排列假设。否定和不确定保留为未关联。候选框重叠仅定位本题提议，Decisions 决定身份。保存原始概率类型，历史描述明确属于过去观测。

产物已有带掩码定位的已确认参考后，使用该参考替代早期全景加工照片。

已确认容器可独立保存当前 RGBD 跟踪参考。其他物体不可见或后来加入清单不会使该容器的参考失效，原始身份、当前掩码、深度与标定仍分别验证。

身份请求传递带原始观测编号的历史实测位置，与当前候选的世界坐标共同提供身份关联证据。未见、提议更新和新任务保留原测量时刻，接受新的 RGBD 测量后才更新该记录。

身份关联的代表视图优先采用当前 Responses 提议中与 SAM3 候选框对应的相机。各相机分别关联永久实例，按已接受的相同 ID 融合深度。过期提议不参与当前视图优先级。

身份请求始终保留当前三相机 RGB，包括没有所选 SAM 候选的相机。当前可见表面范围与历史实测位置分别标记。同次采集的已确认产物记录保留创建证据，重复参考图退出请求。近似候选框与外观查询只用于安排比较次序，每个 ID 对应关系仍由 Decisions 独立核验。当前关联 unknown 附在对应候选的 unassociated_candidates 中，已确认实例继续处理。new_instance 获得确认后独立登记永久 ID。

感知接口以当前 RGBD 和实例记忆为输入，取消 task_spec、policy_graph 和 Responses 观测计划标记前提。运行中只调用 Decisions。当前候选通过掩码轮廓定位，登记后的新 ID 立即进入后续当前视角的关联选项。任务相关性由 select_target 单独处理。

执行层将已接受观测转换为 `ObjectObservation`，提交时间记忆数据库中的可见性和带采集编号的位置记录。`GeometryMeasurement` 更新执行层测量缓存，几何变化使空间记忆中的碰撞场景待刷新。任务账本随后登记新实例的待判断任务资格。配置区域和规划所需冻结观测由执行层保管，`build_scene` 发布实际碰撞几何后更新空间记忆。层间字段和提交时机参见 `docs/native-graph-implementation.md`。

身份细节图使用保留周围区域的原始 RGB，不将轮廓叠加到紧裁剪里；当前全景继续提供 SAM3 掩码、候选标签及三相机视图，深度与历史状态不变。避免小目标的外观和接触/分离关系被裁剪、描边遮蔽。

同一采集中，已接受实例的可见表面点云按当前标定投影到待判断相机，提供投影采样点落入候选掩码比例、掩码覆盖比例和有符号深度差。仅将当前已接受的对应实例证据传入该次单题 Decisions；这些是部分可见表面，不称为完整物体轮廓 IoU，也不据此自动接受 ID。历史位置/动作仍保留日期，不能当作当前位姿。

新产物在首次掩码定位前，保留并传递实际创建核验命题、原始答案、单题请求哈希及对应加工前/后 RGB。引用按实例、输入对象、观测、代次和图像哈希绑定。Decisions 仍独立判断当前候选是否对应该产物；创建成功不自动接受候选身份。已有带位置参考的身份被确认后，不再重复发送创建对照图。

时间记忆按事实逐项更新：已确认释放和支撑不因子区域核验 unknown 而丢失；夹爪开合仅是机器人事件，不自动改写物体支撑。身份请求明确区分初始同采集提议、已确认同采集实例、历史实例与产物首次定位。未见只更新 visibility；ObjectRetirement 需要明确来源、观测和回执，移除/消耗保留历史及 ID。removed 与 consumed 均退出当前身份候选，保留退出原因与历史；不自动恢复退役 ID。暂时未见的活动实例仍可正常跨观测关联。空间记忆用退役 ID 更新活动几何集合，不以未见直接删除碰撞对象。

当前身份关联只处理活动清单。确认退出的实例保留历史，不再自动参与重识别；后来无法关联活动实例的物体，经过独立新增确认后登记新 ID。未见不等于确认退出，也不因此分配新 ID。

历史参考保存原始 RGBD、标定和接受的跨相机掩码；重投影到当前视图后提供历史表面掩码 IoU、覆盖及深度残差。重叠用于排序比较候选，单题 Decisions 决定身份；移动和遮挡可导致零重叠，不能据此直接拒绝旧 ID。

A successful analysis invalidates source_support_current for grasp planning; permanent target identity and prior operation receipts remain history. Before a subsequent plan_grasp, explicitly bind the target with identify_target using the current capture. Reuse of the same capture does not require another support judgment when its proof is already current.

Reuse deduplicated appearance segmentation queries from this scene history, including retired records, to discover additional physical instances of those appearances. Identity hypotheses use active instances. A new instance requires its own Decisions confirmation and permanent ID.

每轮实例关联使用执行层投影的 association_scope。当前输入及其产物优先，已交付实例保留物理历史并按需比较。单个身份谓词的历史证据限定到被比较实例，当前已接受主体继续保留标注与几何，帮助区分相似物体。新实例登记前仍检查全部可用历史身份。已交付内容物只有在入盒回执、有效空间占据和当前容器 RGBD 对应一致，且通过已有容器位置稳定性检查，且候选完整有效表面位于占据范围内时，才记录为 deferred_candidates。记录包含采集、场景版本、容器、回执、测量范围和占据范围，identity_assigned=false。来源盒、工作区及未完成物体的已知支撑容器继续完整关联。占据失效、容器对应缺失、候选伸出范围或后续任务重新操作该容器时，恢复通常关联。空间记忆继续保留交付实例的占据；单件位姿保持未测量。association_scope 参与观测复用键。

关联范围同时覆盖当前运动所需的目的地几何。当前存在未完成输入或产物时，目标盒内容物恢复逐件关联，优先在加工准备和携物遮挡前取得当前 CAD/OBB；直接搬运任务持续保留这项目的地几何需求。逐件交付后尚未选定下一输入的观测可使用有效容器占据。空间占据保持参与全部数值碰撞检查。

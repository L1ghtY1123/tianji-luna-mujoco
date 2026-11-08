# verify_unpack

Acquire current three-camera RGBD inside this skill and compare it with the bound pre-processing appearance. The skill supplies its own verification images. Call it to establish the process result and product ID before requesting general inventory geometry for that output. Check the configured processing result and disappearance of the input wrapper; do not require a future delivery or return posture. A successful result creates a distinct permanent product ID with input/output lineage and dated current support evidence, then retires the consumed input. Keep that verified support evidence instead of immediately asking the same table relation again. Workspace checks are reported separately from visual processing success. No movement or collision-map publication occurs; observe_scene may reuse the resulting capture.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.

verify_unpack 的产物出现与包装消失分别使用一次 Decisions 请求，每次共享拆包前与当前三相机图像。两项事实可以由不同相机提供证据，明确否定返回 rejected 并沿 failure 边，缺证据返回 unknown。程序继续校验动作回执、采集时序、目标绑定与测量来源。

Each question constructs its own state. Product appearance is supplied only to the product question; the wrapper question identifies the bound pre-processing wrapper without an expected-product-success statement. Before/current images retain their acquisition labels.

产物 current_region 使用产物自己的已确认支撑或明确位置，输入物体的加工区域不传给产物。

Compare the whole measured tabletop for a newly visible black workpiece. The configured processing capability may move its output away from the former wrapper location. Keep wrapper disappearance as a separate question; the two facts need not occupy the same pixels or be visible in the same camera. Missing/negative criteria are named in the returned reason.

新产物使用同次采集唯一新物体的 RGBD 测量保存参考相机、掩码、框和位置。来源与采集、代次、创建核验及已有黑件排除记录绑定。掩码与产物、输入退役及谱系在同一事务提交。后续 Decisions 读取有位置的参考和几何投影，独立判断当前候选的对应关系。创建核验和加工前后 RGB 继续归档。位置工作空间检查覆盖 A/B 两臂，执行臂由后续 choose_arm 选择。

核验确认后，输入退役、独立产物登记、谱系和双方历史在同一 SQLite 事务中提交。失败或 unknown 不根据预期加工效果宣告成功。

产物创建保留输入所属场景归属。持久化以当前 ObjectMemory 为提交主体，数据库查询结果不反过来筛掉尚待写回的事实；原始场景归属不能被普通状态更新清空。

后续拆包核验读取已交付黑件的当前单件测量，或与入盒回执绑定的有效容器占据。容器占据保留 scene_token、container_id、entry_receipt_id 和 object_pose_measured=false。新产物定位对完全位于该范围内的表面记录 excluded_container_occupancies，单件身份保持未分配；部分重叠继续返回不确定。容器或回执失效时要求重新取得几何。退役实例的历史退出活动产物排除集合。

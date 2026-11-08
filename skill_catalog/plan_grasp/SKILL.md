# plan_grasp

A pre-motion numerical failure may follow one bounded authored recovery, or terminate if no legal alternative is available. A single route needs no Decisions approval. Use Decisions when choosing among different eligible remedies. Observation recovery applies to missing or stale sensing. Pure IK or collision rejection does not establish a visual limitation.

The fixed bounded candidate search recognizes the installed cuRobo BatchMotionPlanner completion and per-phase failure statuses. A fully rejected pool may advance to the next existing sampling round after all candidate outcomes, scene/start bindings and collision evidence are validated. A phase failure does not execute a trajectory. Transport errors and inconsistent results are propagated to the graph.

An observed OBB may describe only the currently visible portion of an unknown object. After exhausted grasp/contact planning, prefer an authored bounded configured-return, observation and scene-refresh route when its empty-hand and motion-map prerequisites can be established. That configured return needs no separate visual-limitation judgment; an additional wrist viewpoint does. A confirmed identity alone does not establish adequate grasp geometry. New observations do not certify IK or collision feasibility.

Check candidates with the official GPU batched IK and BatchMotionPlanner.plan_grasp APIs. Plan approach, grasp and lift, with current robot/world geometry. The public three-phase SDK uses the executing arm Link3 and Link4 links during goalset/grasp/lift solving and restores them for approach. All four contact links are restored on return or exception. Exact command validation checks the environment, peer arm, non-contact links and the phase-specific target-contact policy. Closing and the post-lift measured attachment have separate checks. Validate endpoints and commands before execution. A finite destination preflight may reject a candidate whose sampled transport endpoints fail, but does not prove the whole destination infeasible. The actual holding attachment is not installed before lift. This skill plans only; a successful plan is neither a grasp nor a holding proof.

Inputs, typed arguments, prerequisites, effects and recoverable results are defined in `contract.json` and the registered parameter schema. Responses authors normal and recovery routes; Decisions only selects existing branches. This skill executes only its selected operation. Current policy: `docs/cad-runtime-policy-20261102.md`.

`planning_no_valid_candidate` means no candidate from the selected arm's evaluated pool passed all checks; no robot motion has been executed by this skill. Details distinguish planner, endpoint, contact and command-collision failures. This does not establish that the other arm or every possible grasp is infeasible. Responses may declare an eligible changed-input recovery; runtime does not add it. `pool_exhaustion.scope=internal_frozen_candidate_pool` and `outcome=pool_expansion_not_permitted` limit further internal sampling of this frozen search, not graph-level recovery. An unchanged planner retry alone is not the configured-return and fresh-observation recovery.

The task adapter submits ranked candidates in fixed batches of two, warmed before task requests. If a batch has no candidate passing the complete endpoint and command checks, continue through the original remaining candidate pool. Stop after the first accepted batch, selecting its valid candidate by joint travel. Larger explicit API requests retain their full candidate count and supported fixed capacities. This changes search granularity, not seeds, IK tolerance, collision clearance or safety checks.

For unknown OBB objects, complete numerical rejection of a frozen pool permits the next bounded GraspGen pool (256, 512, followed by four pools of 1000 requested candidates). Collision rejection must include current scene binding and both-arm validation; CAD/OBB/cuboid collision evidence qualifies. Missing or indeterminate evidence stops this search. CAD libraries remain a single immutable pool. This is internal candidate search, without changing the graph or executing motion.

CAD pregrasp lift bounds reuse the immutable canonical partition and sphere centres for identical model vertices, sphere budget and partition size. Each candidate still transforms every cell vertex and recomputes its enclosing radius with current padding. This is a geometric preview, not an installed attachment or a holding proof. Open-stop feedback noise within 1e-9 radians is normalized only in the planning model, with 1e-8 metres of conservative hand-sphere reserve; raw feedback and all larger closures are preserved.

Source-container lift neighbors come from enabled, validated published CAD/OBB bounds overlapping the source XY envelope, including retained obstacles. Historical input siblings and completed_target_ids are not a geometry source. Retired inputs are not queried for current measurements; any still-published enabled obstacle remains conservative collision evidence. Target retirement, scene binding, rim height, held-envelope clearance and full trajectory checks remain enforced.

完整有限的末端位置、旋转或方向超限证据属于已判定候选拒绝，可继续既定有界候选池。没有绑定 FK 或包含非有限值的证据不得这样分类；原精度和碰撞门限不变。

固定/精确 XY 的抓取前放置端点筛选与正式放置共用冻结实测表面质心作为物体位置参考；只筛选该指定 XY，不以区域内另一位置作为通过证据。普通区域继续原有有限分区。该筛选只检查固定夹爪端点，不宣称全臂 IK、持物或完整轨迹已通过。

source_support_current is a data prerequisite established by identify_target after the latest observe_scene. Reobservation updates geometry but does not automatically refresh semantic support. Author restoration of this prerequisite in any observation recovery path before replanning.

The plan carries MotionPlanBinding for target, destination, arm, generation, observation, scene and settings. Execution validates this binding before physical commands. Grasp planning consumes a detached scene/observation snapshot throughout its bounded search.

During grasp descent and closure, the executing Link3 fingers and Link4 linkage may contact the selected target. Contact-link/environment, peer-arm, other-link/target and self-collision checks retain their configured clearance.

Unknown-object grasp endpoints must provide full-cloud clearance from the authored fully-open pad collision mesh convex hulls, using the existing measurement uncertainty and collision margin. A finite, model-bound negative clearance rejects that candidate before motion and permits continuation of the existing bounded search. The achieved trajectory endpoint FK is used for bounded closure contact prediction.

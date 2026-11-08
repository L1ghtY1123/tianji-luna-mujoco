"""Shared graph protocol. Skills declare operations and the graph declares control."""

GRAPH_RULES = """
Compile this natural-language task online into an initial entity proposal, TaskSpec and version=2
graph. Use the supplied three-camera RGBD, instance memory, configured regions, CAD catalog and
skill contracts. Choose the normal sequence, shared subgraphs and finite recovery routes for this
task. Runtime executes your declared nodes and edges. Responses runs only during initial compilation
and at most two validation-driven graph revisions before execution. Return compact JSON.

AUTHORING ORDER
First determine the task and the interfaces of reusable subgraphs. Then compose each body's normal
path from the supplied skill contracts. Finally add finite recovery routes and trace their effects
through the entire continuation, including the next loop iteration. Return one complete graph in
this call. Prefer a small number of shared recovery routes whose input facts agree; duplicating
every skill into multiple recovery variants increases wiring errors without adding capability.
The task determines the topology; no example sequence is a mandatory workflow.
Declare each scope's complete node list, including its success/failure/unknown end nodes, BEFORE
writing that scope's route groups. Copy endpoint IDs from that list; never invent an endpoint while wiring
an edge. Recovery routes return to declared checkpoints, not missing numbered nodes. Share bodies
when their inputs and effects agree instead of copying long transfer bodies for individual objects.
Separate phases whose completed physical effects must survive a later recovery. A downstream
evidence failure must not route back through already completed transport or processing actions.
Emit the root node list first to name the task scopes and their subgraph_id interfaces. Then fill
every referenced subgraph body and its callees. Copy each definition ID from its declared calls.
Each local node list precedes its entry/edge references. Check all named scope definitions exist.
The edges array contains compact route groups with sources, outcomes, target and max_traversals.
Each group explicitly declares the Cartesian product sources × outcomes going to one target.
For example, sources=[a,b], outcomes=[failure,unknown], target=stop, max_traversals=1 declares four
separate bounded edges. All listed sources must support all listed outcomes. Group repeated routes
to reduce output size. Code generates edge IDs and expands ONLY these declarations. It supplies no
missing skill, route, retry, action or completion fact. Do not emit id/source/on fields in route groups.
Identical repeated declarations normalize to one edge with the same traversal budget. Different targets
or traversal budgets for the same source/outcome remain conflicting declarations and fail validation.
Historical rejected_proposal and graph_proposal records show expanded runtime edges for diagnostics.
Your response always uses compact route groups, including scope replacements.
Each scope also declares on_failure and on_unknown as {target,max_traversals}, or null.
These are common routes for supported skill and subgraph-call outcomes in that scope. Explicit
source/outcome routes override a common route. Routers and ends always use their own explicit routes.
Use common routes for shared failure exits and compatible recovery entries, and list only exceptions
in edges. The compiler expands the authored defaults into ordinary edges before complete validation.
Prefer phase-specific scopes with clear failure exits so recovery can preserve previous physical effects.
During validation revisions, return explicit scope replacements. Omitted scopes keep their previous
definitions. Each edited scope still includes its complete node list and edges.

VERIFIED AUTHORING LESSONS
compilation_memory supplies persisted counterexamples from previous experiments under matching
executable contracts. Apply each lesson to all relevant scopes before emitting the graph. Its example
IDs belong to rejected historical graphs; choose IDs and topology for the current task. The validator
still checks these constraints on the complete graph. Object identity memory has a separate purpose.
Its execution_failures section records terminal skill errors from earlier runs under matching contracts.
Use those dated failures to review applicable prerequisites and recovery continuations. Retain successful
physical effects when choosing a recovery checkpoint. The recorded object IDs and measurements belong
to earlier scenes, and a historical failure alone does not establish a present fact or a working remedy.
A shared recovery entry must accept the facts from EVERY incoming failure path. A planning failure
before movement does not establish recovery_needed. A partially executed motion may invalidate
observed/scene facts. Do not route both into a skill that requires facts present in only one case.
Trace each skill's requires, planning_effects and failure_effects separately for each incoming edge.
Reconciliation for failed grasp is only for its declared partial-motion failure, never a generic
planning-failure prerequisite. A successful physical effect must not be replayed during recovery.
Share a recovery BODY through separate call nodes for distinct continuations. Each call returns directly
to its own checkpoint. A shared resume router exposing both an earlier physical phase and a later phase
admits cross-phase replay paths. Natural-language conditions do not statically exclude those edges.
Keep input-transfer and product-transfer continuations separate once processing has consumed the input.
When recovery itself fails or remains unknown, terminate that attempt; do not loop recovery again.
Remove unused node declarations, including terminal nodes: every node must be reachable from its
scope entry. Before any observation-return motion, establish observation_motion_scene_ready;
scene_ready or a selected-target map alone does not establish that different map capability.
Check this on the final-return path as well as recovery paths.

PLACEMENT SEMANTICS
identify_destination MUST explicitly select release_mode before generating/screening grasps.
surface_setdown means gentle release near the support (including a container floor); plan_place
uses approach/place/retreat. container_release means an allowed drop into an open container from
above its rim; plan_place uses a single held plan_pose then execute_place stops and opens the gripper.
Choose by the user's delivery requirements, never by object color. A task requiring gentle support
contact, precise stacking or orientation after landing uses surface_setdown. For the operator's
box-above-release delivery request use container_release. Do not invent a plan_pose skill.
Both modes use perceive_placement_zone -> resolve_placement -> plan_place -> execute_place ->
verify_place, with the other prerequisites established according to their contracts. Geometry and
operator configuration determine the height; Responses may not supply a TCP pose or numerical gap.
Pass the same explicit release_mode to verify_place. Surface verification does not publish current
object geometry. Before prepare_unpack or any later movement, explicitly observe_scene and build_scene
after surface_setdown unless a current mapped release is already established. Container entry has a
different map effect and cannot be used to justify a surface-transfer continuation.
The mode stays bound to this transfer. Switching mode requires explicit destination rebinding and
invalidates previously generated grasps and plans. There is no automatic fallback between modes.

ENTITIES AND TASK
Entities describe external physical objects, containers and supports. Exclude robot links, joints,
grippers, mounted cameras and cables. A configured unmarked region is a spatial reference.
Use existing_id for a remembered physical instance and null for a new proposal. Runtime assigns
permanent decimal IDs. Same-kind instances remain distinct. Give each proposal its clearest camera,
normalized box, visible description and short English segmentation phrases. Preserve the user's
instruction exactly. Honor explicit unknown/no-CAD requests through unknown_obb and the exact quote.

The inventory records identity, observations, operations and input/product lineage. Runtime sensing
updates it with SAM3 and Decisions. The inventory is independent of graph topology and execution order.
For all/count tasks, target_refs are initial candidates. Later observations may discover more matching
instances. For ids, they name the exact requested inputs. source_ref is the original source,
destination_ref the final delivery and work_area_ref intermediate staging/processing. processing lists
only processing_effect skills. Completion conditions each describe one fact. Preserve requested count,
order and completion_pose. Otherwise choose an accessible current input.

For all/count, use shared subgraphs with input_id=null and task=null. Select $selected at runtime,
including when the initial source has zero candidates. Never enumerate one processing scope per initial
object or bind an input skill to an initial literal target ref. The same policy handles later arrivals.
After each successful input/product delivery, explicitly move_to_observe(viewpoint=configured), then
observe_scene(capture_mode=fresh), update the required collision scene and select_target again.
select_target requires observed=true and observation_pose_ready=true. Each new cycle uses current sensing.
Give normal outer-loop edges an explicit 1024 traversal budget, independent of initial candidate count.
Recovery within a shared per-input call remains bounded to one attempt. Budget exhaustion is failure.
For all, current source confirmed empty ends this batch. Never add a wait-for-arrivals loop after empty.
For count, finish after the requested number of verified input lineages.

Use select_target to select from the current scene and task history. Success binds $selected.
identify_target($selected) binds the current operation subject $current. A shared subgraph can handle
the input and its output, then return through configured observation and fresh sensing to selection.
For all, select_target asks source presence only when no eligible current input exists. It returns empty
only after a confident current no. Remaining-but-unlocalized input and uncertain visibility return unknown.
empty requires an explicit verify_task_completion continuation. A selected unfinished input retains its
handling obligation. Processing outputs have their own permanent ID and $product binding.

For collection tasks, selection returns unknown when source evidence shows a remaining input whose
identity cannot currently be localized, or source visibility is insufficient. Do not require every historical
inventory row to stay visible. Plan a bounded continuation for this case. If final verification later finds
source_has_remaining_input=yes, provide an authored path that obtains the
needed current observation/identity and returns to selection, preserving completed transfers and
processing. The verification result is evidence, not a new target binding. Include a recovery choice
for selection identity uncertainty when the available skills can improve its evidence. Do not end
solely because the current selectable set is empty when a legal evidence-recovery route exists.

GRAPH PROTOCOL
Each scope owns its entry, nodes, edges and explicit end. Calls reference a separately defined
subgraph_id. Shared subgraphs have input_id=null and task=null. An optional instance-owned subgraph
has one input_id, task.scope=ids, that single target_ref and completion_pose=null. It verifies its own
receipts when returning success. Node IDs are globally unique. All nodes and subgraphs must be reachable.
Edges connect nodes within their own scope. No recursion or cross-scope edges.

Declare exactly one outgoing edge for every possible source outcome. The pair (source, on) is unique
within a scope. Skill outcomes come from the contract.
Subgraph calls return success/failure/unknown. A predicate router has choices=[] and emits yes/no/unknown.
A branch selector lists its exact edge.on values in choices, including unknown, and explains each
alternative in condition. Failure/unknown ends have no outgoing edges. An owned success end may
need an unknown edge if its receipts are incomplete. Root success requires task_verified.

Every edge has finite max_traversals. Bounds apply per scope invocation. Exhaustion returns failure
to the caller. Only retry_safe skills may revisit the same call for the same subject. Distinct inputs
can reuse shared bodies. Never replay completed or uncertain physical effects as a verification retry.

BINDINGS AND PRECONDITIONS
$source/$destination/$work_area refer to the active task. $target exists only in an owned subgraph.
$selected exists after successful selection. $product exists after a successful producer.
$current follows identify_target while $selected continues to refer to the input. A new selection
clears earlier operation bindings. An input and its product remain distinct.
Shared skill_outcome_contract applies to every skill; a skill's outcome_contract adds its specific
outcome meanings. Use requires/provides/planning_effects/result_bindings/argument_rules/outcome_rules as the executable
contract. Each node must satisfy its requirements on every incoming path. Shared calls inherit and
return facts and bindings. A router creates no facts or bindings. initial_facts apply at root entry only.

Contract state semantics: success sets provides to true and applies known planning_effects.
false establishes false. null removes STATIC knowledge of a fact; it does not assert that the
runtime fact is false or impossible to establish. Consult the skill description and outputs for
runtime-dependent effects. Runtime checks the actual required facts before executing each skill.
For example, verify_place success sets table_release_verified=true precisely when the verified
current transfer destination is the bound work area; for another destination it is false. Its static
null reflects that conditional result, not a missing capability. Bind the work area before moving
its input and preserve the verified release receipt through preparation/processing.
Non-success sets provides to false and forgets planning_effects except facts in
failure_preserves, which keep their incoming values. Then failure/unknown applies failure_effects,
including recovery_needed on execute_grasp failure. argument_rules specialize requires and effects
for the node's arguments. A recovery that refreshes evidence may invalidate downstream prerequisites.
Re-establish those prerequisites before resuming, including prerequisites of LATER nodes, not just
the immediately retried skill. Do not carry an old placement zone or plan across its invalidation.
Evidence repair may be repeated; completed physical effects may not.

SKILL OWNERSHIP
Skills and their physical settings belong to the application/operator. Compose the supplied skill
interfaces; never rewrite implementations, calibration, collision thresholds or numerical settings.
generate_grasps has no model-selectable tuning arguments. Its configured OBB GraspGen insertion is
applied once along the executing tool's local +Z before IK (operator default: 0.01 m). CAD library
poses keep their calibrated transforms. Do not emit parameters/reason to override those settings.

DECISIONS AND RECOVERY
Unknown means insufficient evidence for the stated question, not a diagnosis of invisibility.
It may reflect occlusion, partial views, ambiguous identity or conflicting dated evidence.
Each router asks one factual predicate or one branch choice. Include the subject and time scope.
Successful skills advance directly using their established results. Reuse accepted identity and task
semantics within their evidence scope. Numerical workspace, collision and feedback checks stay numerical.
Failure and unknown are typed skill outcomes. A single authored recovery or terminal route executes
directly through its edge. No Decisions router is required to approve that route. Use a choice router
only when selecting among genuinely different legal alternatives needs judgment, or a predicate router
when the next step depends on an unresolved visual/language fact. Avoid rejudging reported error codes.
Select a recovery branch directly using returned checks and selected_arm/placement_mode. Do not add
predicates that merely ask which arm or placement mode is already bound, or restate a numerical result.
Recovery selectors receive last_failure, current facts, bindings and declared branches with remaining
traversals. last_result changes after every node. Use last_failure for the original failed checkpoint.
Decisions selects a declared route. It cannot create nodes, choose outside skills or invent parameters.

Provide useful bounded recovery for recognition uncertainty, missing geometry and planning/grasp
failure wherever this task's available skills can establish legal prerequisites. Share a recovery
subgraph or selector when its incoming bindings and facts agree. Include stop and unknown paths.
A failure with no legal recovery may end explicitly without an extra question approving termination.
Grasp failures include grasp_search with per-round planner, contact and collision results. A return
to an already-current configured pose does not improve the view by itself. For missing geometry,
declare an eligible changed viewpoint. For numerical rejection, consider an eligible alternate arm
or other declared changed-input route. Preserve stop/unknown when the available routes cannot improve
the failed condition. A new capture ID alone does not establish improved evidence.
After recovery, resume the failed checkpoint and preserve completed effects.
Allow at most one recovery attempt for the same failed checkpoint and selected input/action.
If any operation inside that recovery fails or remains unknown, end explicitly. Do not start another
recovery from a failed recovery. If the resumed checkpoint still fails, stop instead of repairing it
again. Use max_traversals=1 on the corresponding recovery-entry edge; later inputs are separate calls.
This bound applies to entering recovery, not to the successful prefix being replayed after recovery.
If recovery returns before the failed checkpoint, give each already-used success edge on that prefix
at least two traversals in this scope invocation. Otherwise recovery succeeds but cannot reach the
checkpoint. Alternatively use a separate retry path with its own nodes. Never replay completed effects.

Audit recovery coverage separately from graph validity. required_observation_recovery lists the
exact skill outcomes checked by the validator.
Trace each listed outcome of every normal-path call, including calls inside shared subgraphs, through
its caller returns to an eligible declared observation-recovery route before emitting the graph.
For empty-hand identity/destination/region evidence failures, expose an eligible observation repair
wherever its prerequisites can be established. A single recovery can have a direct edge. Do not remove that recovery
just to make validation easier. Prefer it before an alternate-arm retry when it can improve evidence.
This coverage is validated for normal empty-hand identity/support unknown outcomes.
For pre-motion numerical grasp-planning failure, author a bounded route using the actual
rejection evidence. It may choose another eligible arm or generate fresh grasp candidates and replan,
then stop if that declared retry fails. Keep all numerical prerequisites and candidate checks.
Observation recovery is appropriate when the failure indicates deficient or stale sensing.
Pure collision/IK rejection does not establish the visual limitation required by a wrist viewpoint.
For the listed identity/support unknown outcomes, expose move_to_observe followed by observe_scene.
Configured and permitted wrist viewpoints
both satisfy this coverage. Select a viewpoint that can change the deficient evidence. A caller may own this route.
Checkpoints reached only while recovering may terminate on failure/unknown; do not recursively recover them.
Use one branch choice when there are alternatives, and a direct edge when there is one route.
Do not duplicate a visual-limitation predicate before an alternative-view skill that already asks it.
If a fresh observation invalidates destination/zone/arm/plans, reconnect before the earliest required
producer in that continuation. A successful retry is not allowed to inherit invalidated old facts.

When it can improve evidence, use move_to_observe(viewpoint=configured) -> observe_scene -> restore
invalidated prerequisites -> resume the failed checkpoint. Configured return does not require a separate judgment
proving occlusion. Establish empty hands and the observation-motion collision map before movement.
If those prerequisites cannot be established, choose another legal declared route or stop explicitly.
Before the first map exists, in-place observation/map establishment may be the only available repair.
Alternative wrist views still require a visual-limitation judgment. Repeating an unchanged capture
reuses its answer; bounded recovery must stop when it supplies no new usable evidence. Do not claim
that observing repairs an unreachable grasp/placement pose or a partial motor failure. Partial
execution keeps actuator/effect receipts and requires a compatible route. Switching arms requires
arm-specific grasp generation. Fixed placement XY cannot drift.
While holding, never call the empty-hand observation recovery. For retry_safe numerical planning
failures before any placement motion, a bounded replan of the same authorized pose may be a useful
alternative chosen by Decisions; it is not a new grasp/release or permission to relax collision checks.
Include post-release verification uncertainty in the recovery coverage audit. When the object is
released and observation motion is legal, expose an evidence-only recovery choice: establish the
observation-motion map, return to configured observation, observe and retry verification using the
original release record. If current mapping prerequisites are missing, obtain them first. This path
must return to verification, never to grasp, transfer, release or processing that already occurred.
Do not merge this continuation with a pre-grasp recovery that restarts physical effects.
For an unverified release, obtain a current map containing the released target through observe_scene
and build_scene before moving; retained pre-release geometry is insufficient. Completed release/retreat feedback and this current map permit observation motion, not destination acceptance.
An execute_grasp failure is not an ordinary planning retry. Include a separate Decisions choice for
the reported grasp_not_held_after_lift or grasp_closure_limit_reached_without_holding errors:
recover_failed_grasp checks stable full-close empty feedback and reconciles the partial action without
moving. On success, explicitly observe_scene and build_scene(exclusion_state=observation) before
move_to_observe(configured), then observe again and restore the grasp/placement prerequisites.
Use a distinct execute_grasp node for the one new attempt; do not revisit the failed physical node
or reuse its old trajectory. Any failed recovery step or failed second grasp ends this route.
Other partial execution failures and unconfirmed empty feedback must stop. The recovery skill does
not locate the object; only fresh observation supplies its current identity and geometry.
Do not add visual held/not-held judgments. Gripper feedback is execution state, not object-memory state.

PHYSICAL EVIDENCE
observe_scene updates current identity and geometry with all three cameras. Partial observations retain
unseen history. New-instance association and task membership are separate. A particular operation still
needs its target/destination evidence. build_scene publishes measured, retained and action-affected
collision layers. Local scope changes selected instances, global scope updates current measurements.
The first successful collision publication must be global. Local updates require an existing global
base (collision_map_initialized); invalidated geometry can then be refreshed locally. With scope=decide
before initialization, global is the sole permitted scope and needs no Decisions call.
Unseen history alone supplies no current measurement. Update changed geometry before later planning.
For the operator's ordinary container_release delivery, verify_place accepts checked above-container
release and open feedback as entry. It records source=operator_container_release_rule and publishes
the measured container bounds as conservative contents occupancy. This enables subsequent collision-
checked observation return even when stacked contents cannot be individually located. It does not
claim an individual settled pose. Subsequent sensing can replace an envelope with actual object geometry.
Surface setdown and additional requested landing relationships retain their visual checks.
An observation map supports empty-hand sensing motion. A selected-target map has a different exclusion
state. Ordinary OBB size changes have no anomaly gate. CAD, OBB and cuboid are the collision types.

Honor original placement language and configured region IDs. Designated/configured areas use their
exact XY. Ordinary surfaces/containers use free_region. Unqualified left/right is robot +Y/-Y.
Use inside for containers and on for supports. Establish placement geometry before grasping when the
skill requires empty-hand observation. Robot actions invalidate evidence according to their contracts.

prepare_unpack establishes its capability-specific posture and input conditions. verify_unpack
captures its own RGBD, verifies processing and registers the product. Subsequent observe_scene measures
product masks/geometry. Final verification uses its own current RGB and receipts. Explicitly arrange
the required final robot pose before task completion. All motion retains real numeric checks.
"""

FINAL_GRAPH_CHECK = (
    'Walk every scope from entry to explicit end. Check source-node outcome vocabularies, local '
    'endpoints, reachability and every incoming-path prerequisite/binding. Trace shared calls and '
    'returns, the next input, product binding and all bounded recoveries. Check all downstream '
    'requirements after every recovery invalidation. observe_scene invalidates source_support_current; '
    'before another plan_grasp, explicitly re-establish it through identify_target and restore any '
    'dependent bindings that identify_target invalidates. A permanent ID is not a current support proof. '
    'For each pre-motion grasp-planning failure, inspect the available bounded numerical alternatives. '
    'For an observation recovery, check that its declared route can reach configured observation return, fresh observation and '
    'rebuilt prerequisites before retry when those motion prerequisites can be established. '
    'Retrying the identical arm/scene/candidate pool alone does not provide that observation repair. '
    'Check exactly one route per source/outcome pair after expanding every sources × outcomes group. '
    'Emit route groups with sources, outcomes, target and max_traversals. Code supplies edge IDs. Include select_target empty '
    'when used. Verify current selected-input receipts and independent final completion before root '
    'success. Check the configured final pose on EVERY path into verify_task_completion, including '
    'retry paths after any motion that invalidates observation_pose_ready. Check an explicit bounded '
    'empty-grasp recovery choice reaches recover_failed_grasp before any further motor command. '
    'For all/count, check the shared select/process/return/observe cycle and empty batch termination. '
    'Each router asks one question. A runtime-selected input/product uses a shared subgraph with '
    'input_id=null and task=null, not an owned subgraph whose owner is a $ binding. '
    'Keep entities as updateable memory. Emit compact JSON.')


def graph_context(context, *, skill_detail='compact'):
    """Ablate supplementary prose only; all executable contract fields remain present."""
    from ..core.recovery_coverage import observation_recovery_outcomes

    if skill_detail not in {'compact', 'full'}:
        raise ValueError('unknown_skill_description_detail')
    result = dict(context)
    result['required_observation_recovery'] = {
        name: outcomes for name, contract in context.get('skill_catalog', {}).items()
        if (outcomes := observation_recovery_outcomes(contract))}
    if skill_detail == 'compact':
        result.pop('skill_instructions', None)
        result['skill_catalog'] = {name: {key: value for key, value in contract.items()
            if key not in {'id', 'capability'}} for name, contract in context.get('skill_catalog', {}).items()}
        for name, contract in result['skill_catalog'].items():
            contract['description'] = context['skill_catalog'][name].get('capability', contract['description'])
        outcomes = [contract.get('outcome_contract', {}) for contract in result['skill_catalog'].values()]
        shared = {key: value for key, value in (outcomes[0].items() if outcomes else [])
                  if all(other.get(key) == value for other in outcomes)}
        if shared:
            result['skill_outcome_contract'] = shared
            for contract in result['skill_catalog'].values():
                specific = {key: value for key, value in contract.get('outcome_contract', {}).items()
                            if key not in shared}
                if specific:
                    contract['outcome_contract'] = specific
                else:
                    contract.pop('outcome_contract', None)
    result.update(rules=GRAPH_RULES, skill_description_detail=skill_detail,
        responses_call_policy='Initial compilation, then at most two graph-validation revisions before task execution.',
        recovery_owner='responses_authored_graph')
    return result

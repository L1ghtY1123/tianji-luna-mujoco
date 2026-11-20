"""Shared graph protocol. Skills declare operations and the graph declares control."""

GRAPH_RULES = """
Compile one TaskSpec, complete scene_inventory, separate task_selectors, destination_regions and
version=2 explicit policy graph from the instruction and the initial scoped three-camera RGB.
Only declared_scene_text authorizes visual subjects. Extract every declared class into declared_objects,
including temporarily unseen classes. Copy its distinguishing noun phrase including qualifiers as scene_quote.
Use one entry for repeated instances of the same description. Give a precise English SAM3 phrase retaining
only explicitly stated colors and physical description (black CAD object -> black mechanical part). For a table with no stated color use table, color=unknown, color_quote=null. Translate an explicit count into declared_count. Inventory only visible instances of these
declarations in cameras where their surfaces are actually identifiable. Omit a box when the wrist sees only floor, background, or an unidentifiable edge. Do not make a box for each declaration in each view. Every proposal and selector must name its declaration by exact scene_quote.
Skip undeclared objects even if salient, reachable or present in old memory. When the task requires an
undeclared physical subject or supporting table, report unsupported and request a corrected declaration.
Parse ordinary prose, not just comma lists. "n个"/several means declared_count=null; one blue
box on each side means two blue boxes. Preserve explicit relations in descriptions, initially_on_table
and selectors for any requested role. CAD codes identify models; multiple instances may share one code.
If the scene names an unavailable CAD code, report unsupported with that code. Scene counts are expectations, never proof of identity or visibility. Give proposals unique local IDs.
Descriptions report observed appearance hypotheses. Optional normalized xyxy bbox helps SAM3 retrieval;
omit uncertain boxes. Boxes refer to a declared class. Record CAD codes once on declared_objects. Neither assigns a permanent ID.
TaskSelector.ref is a symbolic task reference only. requested_attributes contains ONLY the user's requested
appearance/category constraints, with an exact instruction_quote. Never turn an observed guessed shape,
material, wrapping or texture into a task requirement. segmentation_prompt/aliases are retrieval hints.
Compile requested broad category into kind and USER-required color into color (unknown means any color).
Put only additional user appearance requirements into appearance_constraints; use [] for a box/container,
a supporting table or a pink object. Observed blue boxes do not add a blue requirement to 'the left box'.
The registrar limits choices to confirmed IDs of the requested broad kind. Decisions selects each task
role from those IDs using the user's full phrase, measured appearance, relations and comparison set.
Do not duplicate kind/color in appearance_constraints.
Use proposal_refs only as hypotheses linking a selector to inventory proposals. Only declared subjects belong in scene_inventory. Do not generate synthetic tracking selectors for historical physical objects.
Use selection.role for target/source/destination/support/reference. All task directions/order use fixed
Link_Base: +X forward, +Y robot-left, +Z up. Left-to-right decreases Y; front-to-back decreases X.
Compile spatial constraints into selection.predicate (and/or/not, robot_left/right, relative axes,
inside/on, explicitly requested dated history). Referenced task selectors must be declared and acyclic.
For specific or ordinal targets use binding_mode=once: moving them must preserve their ID. For all/count
collections use each_selection. Spatial ordinal is distinct from action order and robot half-space.
DestinationRegion describes a configured region or visible placement area with support_ref; a 2D box
never creates a 3D release point. Preserve exact configured region IDs. Include separate table selectors
in support_table_refs when needed to measure a support. Empty instruction_quote is allowed only for
implicit support/reference selectors. Unknown_tsdf requires an explicit matching user quotation.
Only propose physical objects expected now. Future processed products are registered by verify_unpack.
Initial RGBD is captured but unbound: capture_ready=true, task_grounded=false. Call ground_task (or the
explicit ground_task_v1 template) before observe_scene, target selection or any manipulation. ground_task
runs declared-object SAM3, calibrated geometry and current/historical identity association, independently
commits confirmed scene facts, then resolves task selectors. Unknown task selection does not discard
independent scene facts. Unconfirmed physical candidates keep anonymous depth for collision geometry. It never
captures or moves. For unknown, explicitly call capture_scene then ground_task, or end unknown. Do not
invent an initial permanent ID, skip grounding, or ask Responses again. observe_scene handles later
captures with the same association pipeline. Unnamed visible depth remains anonymous collision geometry.
Existing ID suggestions and proposal boxes cannot override measured geometric contradictions.
Use full RGBD. Limit semantic retrieval to the declared scene.
Responses runs at
exactly one initial compilation. A failed response terminates compilation.

PLAN INTERFACES FIRST
Reuse subgraph_library when its inputs and exits fit. Declare {id,template,arguments} instead of
regenerating the reviewed body. Templates expand into visible ordinary nodes and bounded edges before
full validation. Custom shared subgraphs remain available for new tasks. Transfer handles one object;
batch loops, processing and final completion belong in the task graph. No fixed colors or box IDs.
Copy configured_reference_ids exactly. Never normalize hyphens, translate IDs or invent region aliases.
Configured TABLE references belong to the tabletop only. The operator also configures
container-release XY separately: it is applied by identify_destination to the selected container.
When the instruction asks for the configured region INSIDE a named container, destination_ref
must be that visually proposed container and release_mode must be container_release.
Do not replace a requested container with demo-destination or another table reference merely
because the instruction mentions a configured/specified region. Preserve the requested support.
scene_inventory and support task_selectors include the supporting tabletop when required to measure a requested source or destination. A configured unmarked region is not a visual entity; its
visible supporting table is a separate entity with its own visual description and segmentation prompt.
The tabletop reference seeds measured RGBD support geometry; configured destination XY cannot replace it.
Declare support_table_refs explicitly and include those table proposals and selectors even when an unmarked configured
region is the destination. These are necessary measured workspace supports, not extra task objects.
Runtime binds the configured polygon to its unique measured supporting table, including its height.

COMPILE MEANING ONCE
TaskSpec carries input selector, initial target_refs, scope/count/order, source, work area, final
destination and processing. allowed_intermediate_refs explicitly names any additional user-authorized
staging references. Runtime does not ask Decisions to reinterpret permission or choose fixed XY.
All placement preserves grasp rotation at the operator-configured front TCP. Z is measured table
height or container rim height plus measured object height and configured margin (default 20 mm).
Execution also accounts for a grasp TCP above the object top; collision balls do not define object height.
Use destination_kind=support_region with release_mode=surface_setdown, or destination_kind=container
with release_mode=container_release. Gentle placement on a container floor is unsupported.
Put clauses incompatible with that capability in unsupported_placement_constraints and set supported=false.
Put only EXTRA unresolved CURRENT visual completion facts in visual_completion_conditions. Ordinary
delivery, processing, count, action order and joint-feedback posture are checked from their bound
verified receipts; do not re-ask them per object and again globally. Original completion_conditions
retain the user's requirements but are not a list of automatic extra Decisions requests.

COLLISION SCENE
build_scene selects its scope from measured joints. Only the configured dual-arm observation pose permits a
global scene. Other poses update measured regions of the existing collision map using the registered object
history. A fresh local capture can initialize partial geometry. Neither a camera count nor a checkpoint name
proves the observation pose. When a global survey is needed, explicitly move_to_observe at the configured pose,
observe_scene there and then build_scene. Preserve confirmed empty-hand and motion-map prerequisites.

DECISIONS IS FOR VISION
Visual questions use scoped RGB and optional depth. Labelled boxes from SAM3 or dated instance masks
locate the queried subjects. Identity compares real appearance and physical history across views.
The grasp approach follows the tool Z axis. Lift follows world +Z.
Place starts vertically above the release pose, descends along world Z and retracts along world +Z.
After Lift, verify_grasp checks actual gripper feedback. An open hand is empty, intermediate travel means holding.
Only a fully closed gripper uses Decisions on the current finger-contact RGB to judge object_held.
On confirmed holding, use the target ID bound to the completed grasp and attach its geometry.
A confirmed empty grasp can use the explicit empty-grasp recovery. Missing feedback stays unknown.
verify_place accepts the selected destination after measured arrival at its configured release XYZ, release
and an empty gripper. Use the same gripper rule for table and container deliveries. Record execution_assumed.
These checks complete target identity and destination verification. Do not add a second visual check of the
delivered object or its landing position. Current collision geometry still requires its own observations.
For every visual role include the user's exact phrase, dependencies and any target/destination proposal hints in the initial response.
Retain qualifiers in the original language even when the predicate schema captures only part of them.
Decisions resolves those roles from the complete instruction and labelled registered IDs. Proposals are camera-local hints.
A fresh capture compares each physical group with dated references and action history through Decisions.
One frozen capture may reuse an identical measured mask and its existing ID.
Binary answers use probability argmax; choices use the largest offered probability. Confidence never
gates acceptance. Ties/refusals/explicit unknown remain unknown. No additional confidence router.
Initial task membership is compiled. A genuinely new object whose membership is unknown may receive
one visual membership question and retain that answer in the task ledger.

DATABASE, NEXT OBJECT AND ONE ARM
The temporal database is the sole physical object record. Queries and task receipts are not another object list.
For “从左到右第m个”, use ids with one ordinal target, quantity=m, axis=y, order=descending, binding_mode=once.
For “从左到右逐个搬”, use a collection selector and task.selection_order=left_to_right. Mirror for right_to_left.
Distinguish “任选一个” (count=1) from a specific described one (ids). Keep the same selected ID until verified delivery.
For routine transfers bind the existing single_arm_transfer_v1 template and explicit recovery paths.
Every source selector also needs its exact instruction_quote, e.g. “桌面上” in “把桌面上的黑色物体放到右侧盒子”.
Use selection_order=visual for free choice unless the user explicitly orders target_refs or spatial execution. select_target asks one current-image
choice for easier exposed access; initial proposal order is never a fixed execution queue. New database objects
may become inputs. After removing one obstruction, evaluate the next current scene again.
Prefer single_arm_transfer_v1 for one object and one destination. Its entry explicitly runs observe_scene
with reuse_current, establishing observed and grasp_view_ready before identify_target. It can be called
from the empty-hand bootstrap state. A separate initial observe/map call is needed only when another
preceding skill needs those facts, such as select_target for a collection task. The same execution arm must cover BOTH source
and fixed release TCP XYZ. choose_arm qualifies and locks it; no common arm means stop, with no handover.
Regenerate and retry a failed grasp plan ONCE on the SAME arm, never switch arms. A second failure terminates.
The single-arm template contains explicit in-place observation repair, confirmed-empty-grasp recovery and
post-release verification repair. All actions remain visible after template expansion.
Choose scope=ids when the task identifies a specific described object; saying that there is one pink
object does not by itself request collection selection. Bind that exact ref to the transfer template.
Choose count only for selecting N matches from a collection. Even count=1 requires observe/select_target
and target_id=$selected in the shared transfer. Never combine scope=count/all with a hardcoded target ref.
Database state commits synchronize spatial state. Changed geometry blocks planning until publication; metadata
updates preserve geometry but advance the database version. Planning never uses a stale scene or old robot start.
rule=observation_recovery is a program choice among return/stop/unknown. It returns only for missing current
identity/geometry/image evidence with empty-hand proof and a different configured observation pose. Failure or
low probability alone does not establish a reason to move. Only rule=visual asks Decisions a current image fact.

GRAPH AND DATA
Every scope declares its nodes, entry, explicit terminal nodes and local route groups. A group has
sources,outcomes,target,max_traversals and declares exactly their Cartesian product. on_failure and
on_unknown optionally declare common destinations; explicit routes override those defaults. Router
outcomes are yes/no/unknown when choices is empty, otherwise its exact choices including unknown.
Skill outcomes come from the catalog. Subgraph calls return success/failure/unknown. Every source
outcome has exactly one local route. No recursion, cross-scope edges, unreachable nodes or duplicate IDs.
Failure/unknown end nodes have NO outgoing routes. A success end may only have an optional unknown
receipt-failure route; it never emits success. Do not include end nodes in ordinary route groups or self-loops.
Every declared subgraph must be called from the root or another reachable subgraph. If an owned scope
wraps a shared transfer, call the owned scope from the root so that both scopes are reachable.
Shared subgraphs have input_id=null,task=null. Owned scopes have one literal initial candidate owner
and an ids task. Use shared scopes for runtime-selected inputs/products. $source/$destination/$work_area
come from TaskSpec. select_target binds $selected; identify_target binds $current; verify_unpack produces
$product. Routers create no facts. Calls inherit current facts and bindings, including invalidations.
initial_facts apply only once. Never invent a binding or physical capability.

TASK CYCLE
For all/count, select current inputs in a shared loop, never one scope per initially seen object.
After each completed input/product delivery explicitly observe_scene at current joints before selecting again.
Do not routinely return to the observation pose; return only through evidence-based recovery or a user-required final posture. Use finite normal loop budgets up to 1024 independent of initial
count. all ends only on current source absence; count ends after the requested verified lineages.
Empty selection goes to final verification, not a wait-for-arrivals loop. A missing requested ID or
unlocalized remaining input is unknown. Final posture is executed explicitly and checked with real joints.
The transfer template ends at verified delivery. When completion_pose=observation, the root calls
return_observe_v1 and then verify_task_completion. For completion_pose=initial, follow observe_map_v1
with return_to_initial_pose and verify_task_completion. A return ends at the measured pose. Any subsequent
scene analysis is an explicit root choice.
Completed container contents with valid bound release receipts retain membership without
individual reassociation or CAD registration; later observation focuses on the container and remaining task inputs.

RECOVERY AND PHYSICAL EFFECTS
Use the frozen skill requires/provides/planning_effects/failure_effects/result_bindings/argument_rules.
Trace every incoming path and each next loop iteration. Before observation-return motion publish an
observation-motion map; selected-target scene_ready alone is insufficient. Every build_scene rebuilds
the full workspace from the current observation. It has no scope or entity-subset selection.
All maps rebuild the full workspace from the current registered RGBD; historical object IDs persist
in the database, but old object geometry and TSDF cells are not merged into the new map.
Anonymous workspace depth enters TSDF without instance registration. Visible container contents use
current TSDF after delivery; occluded individual poses remain unmeasured. History contains physical scene objects.
Sensor inputs preserve the full camera frame and whole instance silhouettes, including tables and containers
crossing the reachable volume. Reachability is checked separately; it never clips visual evidence.
An empty-hand identity/support unknown first offers a fresh current-position observation, then a bounded
program-gated move/observe route only if required current evidence is still missing and movement is legal. Each failed
checkpoint has at most one recovery attempt; a recovery or resumed retry that fails/unknown terminates.
Give retraversed prefix success edges sufficient budget; do not make a repaired path exhaust an old edge.
Never replay completed or uncertain motor effects as a verification retry. After place unknown, repair
observation and verify again, never re-execute place. After processing, retain the new product lineage.
Empty-grasp recovery requires the same action's explicit grasp_held=empty result and detached geometry; afterwards
observe in place, map and restore target, support, destination and the SAME arm before one new grasp attempt.
After table surface release explicitly observe/map before processing preparation or later movement.
Container entry requires the explicit post-observe/post-map path to rebuild workspace TSDF before movement.

Only skills with captures_observation may capture. Planning, movement and routers use existing evidence.
Unknown space stays unknown; object history is not a new measurement. Preserve full robot world/self
collision, joint limits, freshness and action bindings. Do not tune numeric settings or SDK internals.
When contracts cannot fulfill a request, declare unsupported or an explicit failure, never invent tools.
"""

FINAL_GRAPH_CHECK = """
Check declared inputs, local endpoints and exact source outcomes. Expand reviewed templates and route
groups, then trace every normal and bounded recovery path through all prerequisites and result bindings.
Preserve completed effects, product lineage and transfer release_mode. Do not re-grasp after unknown
holding or re-place after unknown placement. Restore source_support_current after a new observation.
Batch iterations return/observe/select using current evidence. Final success requires the matching
verified delivery/processing receipts, source closure when all, and actual requested final robot pose.
Use deterministic router rules for numeric/error choices and visual rules only for current image facts.
No runtime language interpretation, crop/projection pipeline or extra model approval of known results.
"""


def graph_context(context, *, skill_detail='compact'):
    """Ablate supplementary prose only; all executable contract fields remain present."""
    from ..core.recovery_coverage import observation_recovery_outcomes
    from .subgraph_library import library_catalog

    if skill_detail not in {'compact', 'full'}:
        raise ValueError('unknown_skill_description_detail')
    result = dict(context)
    result['required_observation_recovery'] = {
        name: outcomes for name, contract in context.get('skill_catalog', {}).items()
        if (outcomes := observation_recovery_outcomes(contract))}
    if skill_detail == 'compact':
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
    result.update(rules=GRAPH_RULES, subgraph_library=library_catalog(context.get('skill_catalog', {})),
        skill_description_detail=skill_detail,
        responses_call_policy='Exactly one initial Responses call. Validation failure terminates compilation.',
        recovery_owner='responses_authored_graph')
    return result

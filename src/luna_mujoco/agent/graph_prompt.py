"""Shared graph protocol. Skills declare operations and the graph declares control."""

GRAPH_RULES = """
Compile one TaskSpec, initial entity proposals and version=2 explicit policy graph from the task,
initial scoped three-camera RGB, configured references and skill contracts.
Understand the scene description as context for interpreting the requested task. Describe only task-involved
physical instances, including their reference camera and distinguishing visible neighbors. Consolidate views of the same instance.
Responses supplies scene understanding, segmentation descriptions, task roles and the complete graph
in this call. Include targets, source/destination containers, processing objects and required supports.
Set mentioned_in_task=true for task involvement. Scene-only objects supply visual context and measured
collision geometry without permanent identity association. A scene description alone never expands task membership.
SAM3 then segments each appearance class in ALL three RGBs. reference_camera_id anchors the description
only and never restricts segmentation. Cross-camera candidates are associated using temporary refs
before any permanent database IDs are allocated. Repeated views are not additional physical objects.
Instances sharing kind and color reuse one class-wide segmentation_prompt that covers all of them.
segmentation_aliases are ordered backups, used only when the previous phrase returns no masks.
Individual image locations belong in description. One phrase may segment several separate instances.
When the USER specifies a robot-relative left/right subject, compile that requested side into
robot_relative_side independently of its image location. The calibrated fixed-base world frame is
+X forward, +Y robot-left, -Y robot-right, +Z up. Camera 1 is the right wrist (arm B), camera 2 is the
left wrist (arm A), camera 3 is the head. Image-left/right is viewpoint-dependent. Runtime filters
initial SAM3 candidates using measured world coordinates before Decisions selects their identity.
Do not use an image-derived side to override the user's requested side. Leave the field null when
the user has not specified a robot side. This does not constrain a permanent object's later movement.
All perception and visual judgments consume the same effective workspace scope.
Black outside that scope is excluded input and carries no empty-space evidence.
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
Initial entities include the supporting tabletop when required to measure a requested source or destination. A configured unmarked region is not a visual entity; its
visible supporting table is a separate entity with its own visual description and segmentation prompt.
The tabletop reference seeds measured RGBD support geometry; configured destination XY cannot replace it.
Declare support_table_refs explicitly and include those table entities even when an unmarked configured
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

DECISIONS IS FOR VISION
Visual questions use scoped RGB and optional depth. Labelled boxes from SAM3 or dated instance masks
locate the queried subjects. Identity compares real appearance and physical history across views.
verify_grasp reads actual joint position after Lift: fully open means empty, intermediate means holding.
Only fully closed hands require Decisions from current native-resolution finger-contact RGB and same-capture
feedback. Commands never substitute for measured position. Holding questions do not include historical wrist images.
After confirmed holding, the program inherits the pregrasp target ID from the completed, uninterrupted
grasp bound to that target, arm, run, generation and source observation. This is execution-chain identity
inference, not another visual identity confirmation. It authorizes attachment after Lift without another
Decisions or SAM3 call. A broken binding preserves held presence and unresolved identity and stops this
transfer. Confirmed empty grasp authorizes the explicit empty-grasp recovery.
Foam may deform until fingers nearly close. The approach/grasp/Lift sequence uses detached robot geometry.
verify_place inherits verified holding through transport; do not add in-transit holding or identity checks.
After release it uses the same measured-position rule, calling Decisions for holding only if fully closed.
It then uses current scene views, before-release views and dated identity references to select
direct_success, inferred_success, failure or unknown for the bound target and destination.
Table and container placement use the same acceptance: configured XYZ constrains measured robot release.
Direct success means clearly visible placement on the named support or inside the named container;
the final resting object is not required to fit within the configured release square.
Inferred success requires verified target holding, actual arrival at configured release XYZ within tolerance,
release, current confirmed empty gripper and no visible evidence contradicting delivery. Occlusion is allowed, not required.
Both success categories complete delivery and retain their evidence grade in the database.
Identity compares the boxed current subject with dated boxed RGB references and database physical history.
Binary answers use probability argmax; choices use the largest offered probability. Confidence never
gates acceptance. Ties/refusals/explicit unknown remain unknown. No additional confidence router.
Initial task membership is compiled. A genuinely new object whose membership is unknown may receive
one visual membership question and retain that answer in the task ledger.

DATABASE, NEXT OBJECT AND ONE ARM
The temporal database is the sole physical object record. Queries and task receipts are not another object list.
Use selection_order=visual unless the user explicitly orders target_refs. select_target asks one current-image
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
For final completion_pose=observation, follow the transfer template's post-map with move_to_observe
and verify_task_completion. For completion_pose=initial use return_to_initial_pose and verify_task_completion.
Reserve return_observe_v1 for continuing work or observation recovery that needs a new scene analysis.
Completed container contents with valid direct or inferred entry receipts retain membership without
individual reassociation or CAD registration; later observation focuses on the container and remaining task inputs.

RECOVERY AND PHYSICAL EFFECTS
Use the frozen skill requires/provides/planning_effects/failure_effects/result_bindings/argument_rules.
Trace every incoming path and each next loop iteration. Before observation-return motion publish an
observation-motion map; selected-target scene_ready alone is insufficient. Every build_scene rebuilds
the full workspace from the current observation. It has no scope or entity-subset selection.
All maps rebuild the full workspace from the current registered RGBD; historical object IDs persist
in the database, but old object geometry and TSDF cells are not merged into the new map.
Anonymous workspace depth enters TSDF without instance registration. Visible container contents use
current TSDF after delivery; occluded individual poses remain unmeasured. History contains only task objects. All sensor inputs are
strictly workspace masked. Out-of-workspace objects remain excluded.
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

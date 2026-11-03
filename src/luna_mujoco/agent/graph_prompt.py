"""Initial Responses owns goals and graph topology; runtime Decisions may extend recovery."""

import re

GRAPH_RULES = """
Return entities, an aggregate immutable task, and an executable graph (version=2).
The root graph must establish global completion itself: include a reachable verify_task_completion
skill in the ROOT nodes after the required object tasks and final pose requirements, then connect
its success edge to the root success end. A successful object subgraph establishes that object's
receipts only; its success is not task_verified and cannot replace aggregate task verification.
Use graph, subgraph, node, edge literally. There is no steps/if/recover body syntax.
Router conditions are evidence questions authored in the graph, not necessarily task completion
conditions. They select yes/no/unknown edges but establish no skill effects or completion facts.
Ask one specific condition per router, naming its subject and time scope. Follow a skill's success
edge for facts its contract already establishes; do not add another visual question merely to
repeat that same postcondition on unchanged evidence. Independent user requirements and final
task verification still need their own evidence. A semantic unknown is not a visibility diagnosis:
camera movement requires evidence of not_visible, cropped or occluded information needed by that question.
Create the candidate inventory and per-object requirements, then reference those same entity refs in
the graph, all within this one response. The application allocates permanent IDs consistently afterward.
Entity color must be a short ENGLISH adjective because it supplies SAM3 fallback prompts.
Human descriptions may use the user language; never localize color metadata.
Observe only when its evidence is needed. reuse_current can analyze the supplied initial RGBD
without recapturing when the capability's version and robot-feedback checks pass. Even fresh reuses
unchanged scene/command/action/robot-feedback inputs. It cannot manufacture new evidence after an
unknown judgment. A new viewpoint requires an explicit movement capability. Changed capture/identity inputs require
identity and current CAD-pose work; identical analyzed inputs reuse their result; a successful equivalent recovery result may be
explicitly adopted by Decisions when the capability declares recovery_result_reusable.
segmentation_prompt and segmentation_aliases must be short ENGLISH visual noun phrases for SAM3,
even for Chinese user instructions. Keep human descriptions in the user's language. Use visible color,
shape or material, not source/destination/task role; multiple aliases do not mean multiple objects.
Each candidate task.target_refs must own exactly one subgraph with input_id=that ref and a separate
task (scope=ids, one target_ref, completion_pose=null). Its source/destination/processing may differ
from other items. The root task describes aggregate selection and final requirements. Shared scene
entities such as containers and tables are inventory entries, not necessarily manipulation targets.
Calling an object-owned subgraph requires inventory_finalized=true BEFORE its entry: invocation binds
its permanent input ID and task against the confirmed inventory. Observation and confirmation inside
that object's subgraph cannot satisfy its own invocation prerequisite. Establish the prerequisite in
the root scope or a shared subgraph before dispatching an object-owned subgraph.
task.processing is NOT the action list. It contains only required object transformations whose contract
declares processing_effect=true (for example unpacking). Ordinary grasp-place with no processing MUST
have processing=[] in both the root task and each item's task. Put manipulation steps only in nodes/edges.
target_selector describes the CURRENT INPUT objects to select from the initial inventory. Do not
combine inputs with their future processing outputs in this selector. Future products receive new
IDs through verified processing lineage; state their required delivery in the task and graph.
source_ref names the input's original support/container/region, not the input object. Keep target_refs
and source_ref semantically distinct: the moved object cannot be its own containing source. If no
shared source is specified or implied, use null and let current support be judged separately.
An object-owned subgraph owns the complete input-to-product lineage, not just the input's first
transfer. Its task.destination_ref is the final destination of the resulting deliverable;
task.work_area_ref is the intermediate processing area. Do not set destination_ref to the
processing input area when completion_conditions require the product somewhere else. These fields
also bind $destination and the completion ledger: correct node arguments cannot fix a wrong task
destination. Check this consistency independently for every object, even when the root task is correct.
Use a subgraph node with subgraph_id to call a named subgraph. Shared subgraphs use input_id=null,
task=null. Node IDs must be globally unique. Assign node IDs n001, n002, n003, ... with ONE
counter across the root and ALL subgraphs. Never restart the counter in a subgraph. Repeated uses
of the same skill at different phases require different node IDs; for example a move_to_observe
before grasp and another after delivery are distinct nodes. Edges and scope entry must refer to
those exact IDs. Before returning, check the complete node-ID list for duplicates and every edge
endpoint for an existing node in its scope. Subgraph invocation cannot be recursive.
Every declared node, including failure/unknown end nodes, must be reachable from that scope entry
through its declared edges. When delegating missing outcomes to runtime Decisions, omit unused
abort/end nodes; an unconnected root_failure node makes the entire graph invalid.
Each scope declares entry, nodes and edges. ROOT nodes/edges contain ONLY the root scope, not a global
index. Put a subgraph body ONLY inside that subgraph's nodes/edges; never duplicate it in the root.
A root subgraph-call node references subgraph_id, and its success edge continues within the root.
The callee has its own entry and end. Global uniqueness of IDs does not mean global node storage.
Node array order has NO execution meaning. Wire every
successful route explicitly with edges {id, source, target, on}. A skill or subgraph reports success,
failure or unknown. A router evaluates its graph-declared evidence question via Decisions and reports
yes/no/unknown. An end node explicitly returns success/failure/unknown. Do not use a success end
to swallow unknown evidence. Root success requires the specified final facts and verified task ledger.
The ROOT graph also requires its own explicit success end node. verify_task_completion is a skill,
not a terminal: connect its success edge to the root success end. Never leave it as a leaf.
EVERY subgraph needs its own reachable end node. Its final successful skill must have an explicit
success edge to that end (or another continuing node). Reaching the last node in an array does NOT
return from a subgraph. Before responding, trace entry -> edges -> end within EACH scope, then trace
the parent call's success edge. Missing subgraph returns prevent the whole graph from completing.
Skill nodes use type=skill, id, skill, arguments exactly matching the supplied registry. Immutable task
requirements are not motion code. $target/$source/$destination/$work_area bind to the active object's
task on entering its subgraph. Result bindings such as $product exist only after their producer succeeds,
and cannot leak from a different object's subgraph. Product IDs preserve lineage to their input ID.
Plan the intended complete task including shared perception, individual operations and final conditions.
Use registered contracts (including parameter-specific requires/effects and observation invalidation)
to decide actions, arm choice, mapping, grasp/place/process order and needed evidence. There is no fixed
unpack pipeline. Observed/inventory_ready/inventory_finalized/task_assessed initially are false; raw RGB
given to you is not confirmed identity or measured map. The graph must arrange the corresponding skills.
An initial collision map has no retained object layers. observe_scene measures geometry but does not publish collision layers. Local scope publishes exactly entity_ids, so a target-only first map omits observed containers and other required obstacles. Before dependent motion, selected IDs plus valid retained layers must cover every required inventory collision entity; global includes all currently observed inventory. Scope remains the model choice; missing coverage returns inventory_map_incomplete.
Before returning, trace the declared facts along each intended success route: every skill's requires
must follow from initial facts or preceding successful effects. Account for invalidated facts after
motion. Runtime recovery handles unforeseen outcomes; it is not a substitute for known prerequisites
of the normal task, such as a resolved placement candidate before trajectory planning.
You MAY design foreseeable alternative edges. You are NOT required to enumerate all future failures.
Do not wire every failure/unknown to a generic abort end. In particular uncertain observation, identity,
support, task binding or planning evidence must use a useful authored recovery branch OR leave that
outcome without an edge to delegate to Decisions. Uncertainty alone does not justify a terminal end.
Failure/unknown ends are for established inability to continue, unsupported requests or unresolved
physical side effects, not a shortcut that prevents the runtime recovery owner from considering options.
An unhandled failed/unknown skill result or no/unknown router result invokes a runtime Decisions router. Decisions can select
registered skills and typed parameters to create additional recovery nodes/edges from actual evidence;
these patches are validated and archived before execution. It cannot change the task or replay completed
physical work, waive numerical constraints, use stale evidence as current or call Responses again.
Only contracts marked recovery_retry_safe permit retrying a failed skill; a router may be rejudged
after obtaining new relevant evidence. Supplemental
observation means a legal viewpoint change, not repeated same-posture photography. Historical verified
location and identity survive occlusion, but do not certify fresh geometry. Initial Responses is called
once. No sub-agents, runtime Responses, scripts, Python or simulator truth are available.
Production mapping uses measured table cuboids, observed unknown-object OBBs and registered CAD meshes
whose current FoundationPose estimates have matching source/frame/capture bindings and are accepted
directly; no postregistration point/depth/silhouette fit validation or ICP is required.
When the user explicitly declares a subject unknown or without CAD, set its geometry_policy to
unknown_obb and quote that declaration exactly in geometry_policy_instruction. This requirement binds
to its permanent ID only after current mask identity confirmation. It skips CAD retrieval and
FoundationPose for that subject; its OBB is still measured anew. New processing products have separate
IDs and do not inherit the input's unknown-object policy. Otherwise keep geometry_policy=auto.
The terms designated/configured region (including 指定区域 and 设定区域) refer to the operator's
workbench region configuration. Bind that configured region ID; its configured XY is fixed for this
transfer, not an instruction to search arbitrary points. If that point is blocked, report the failure.
Known CAD uses its selected arm's
offline grasp library; unknown models use OBB and GraspGen. CAD identity does not supply current pose.
Global means all currently
observed inventory, not full spatial coverage. Local preserves other retained layers. Select scope
yourself or use the registered decide mode; no model may waive collision, freshness or joint checks.
Configured invisible XY regions are referenced by their ID, not invented visual objects or boxes.
When the user names the workbench designated/set/configured area, identify_destination must use
placement_mode=configured_xy. This binds the supplied point without any XY search. Ordinary
surface/container placement uses placement_mode=free_region. A region's existence in configuration
alone does not mean an otherwise unspecified surface task requested that region's center.
Unqualified task left/right use robot/world left/right (+Y/-Y). An explicitly named
camera/image/user viewpoint overrides this default; different camera images have different sides.
An ordinary destination need not have a fixed XY or fully visible floor. The registered skills solve
poses in measured regions. Use model-selected arms or auto; no color/arm/location defaults define tasks.
For each object, explain the destination choice in its task.reason using the original user requirement.
Reference a configured region only when its meaning matches that requirement; a general tabletop request
may use the identified table, not an arbitrary processing input region. Do not invent metric XY from RGB.
Bind the destination to the supporting entity or container, not a relative reference: for 'on the table
left of the cup', the table is the destination and the cup is a reference. Preserve the original wording
in the immutable task. Do not resolve a placement pose or viewpoint in this initial response.
perceive_placement_zone requires observed=true. execute_grasp invalidates observed, and verify_grasp does not restore it. observe_scene requires empty hands, so an unbound placement zone cannot be newly interpreted on the normal held-object route. resolve_placement(capture_mode=fresh) acquires candidate images only; it does not establish observed or interpret a missing zone. Trace these existing prerequisites when arranging the intended success route.
perceive_placement_zone binds the measured container interior or supporting surface and asks Decisions
whether the original instruction permits any numerically feasible spot in that region. It does not
expand language into a fixed vocabulary of directions or halves. resolve_placement uses verified held
OBB or CAD geometry and queries the published collision world with held-object and gripper spheres
to compute candidate poses.
The offline workspace is only a TCP position prefilter; fixed rotation and paths still need plan_place.
Z search is bounded near the support and rejects releases above the configured maximum object gap.
If a relative or named-region requirement remains, Decisions judges numbered footprint candidates in
current calibrated RGB; only accepted candidates can reach plan_place. Establish the region and
collision geometry while empty-handed, before grasping. If candidate-level visual judgment then
needs current images while holding, resolve_placement(capture_mode=fresh) explicitly acquires those
images against the still-valid bound scene and region. It does not remeasure geometry or establish
observed=true. Do not schedule observe_scene or perceive_placement_zone while holding. Stale or missing
geometry remains an unmet prerequisite; fresh candidate images cannot repair it. No skill silently
moves a camera, enlarges the region or calls Responses to choose a spot.
Every placement, including an ordinary container, uses plan_place then execute_place and verify_place.
plan_place checks approach/place/retreat; execute_place releases between place and retreat. There is no
registered transport/drop branch. Preserve the full grasp rotation. plan_pose remains a numerical primitive
for other motion and retreat, not a placement skill. Exact configured XY overrides candidate ranking.
Skills choose exact poses using measured footprint containment, calibrated workspace, IK and current
carried-body collisions. Runtime Decisions approves regions/candidate IDs, never numerical XYZ.
A region is not a proven safe pose; missing geometry or exhausted candidates requests Decisions recovery,
not a different goal or waived checks. Existing objects and other inventory targets remain obstacles.
For repeated box-to-table transfers, earlier deliveries remain inventory obstacles. Observe the actual
released positions and refresh the affected map layers through explicit graph nodes before the next
placement. A commanded release pose is not an occupancy measurement; occlusion does not erase retained layers.
The VLA's configured input region, attitude and dual-arm starting pose are capability prerequisites,
not universal workflow rules. Model all required processing and product rebinding using the registry.
When selecting a processing capability's configured input region, use that SAME region ID consistently
in the aggregate/item work_area_ref and intermediate identify_destination nodes. Another configured
region with coincident XY is not an interchangeable ID or evidence of the same capability prerequisites.
All target candidates need independent subgraphs. Root edges determine their dispatch order, so use
inventory/explicit selection_order; nearest is not supported by this explicit dispatch representation.
Match task.instruction exactly and honor supplied completion_requirements. If unsupported, explain in
task.reason and use a failure end. Never report completion just because no nodes remain.
When completion_requirements.after_each_object requests an observation posture and inventory
observation, honor this explicit user requirement in each object's successful route, including the last.
It applies after EVERY verified delivery, including an intermediate placement in a processing input
area, before preparation/processing or another transfer. Final product return cannot substitute for
the earlier input-delivery return. prepare_unpack and vla_unpack require the matching historical
post-delivery observation receipt when this user requirement is active.
Use capability prerequisites to arrange any required map refresh, explicit movement and observation.
This is a supplied task requirement, not a hidden action performed by grasp/place skills.
The configured observation posture and the measured task-start posture are distinct references.
move_to_observe does not by itself prove at_initial_pose, even when the pictures look similar. For final
robot_pose=initial, arrange return motion if needed. verify_task_completion independently checks the
measured task-start joint reference: it can confirm an already-correct pose, but never moves the arms.
Root success requires this final verifier, not a mandatory extra movement when earlier chosen nodes
already reached the required posture. A wrong final posture remains failure and needs Decisions recovery.
Each object's success end checks its delivery and required historical observation receipts. If missing,
Decisions may add recovery nodes/edges and recheck that end only; it never restarts the whole subgraph.
An observation receipt is retained when later valid graph actions move the arms, including final return.
"""


FINAL_GRAPH_CHECK = (
    'Before returning JSON, trace the actual ROOT entry and success edges from the supplied initial facts. '
    'inventory_finalized starts false. Every object-owned subgraph invocation requires it true BEFORE '
    'entering that subgraph. Observe/confirm nodes inside the object subgraph cannot establish its own '
    'invocation prerequisite. Place prerequisite-producing capabilities in a preceding root or shared '
    'scope, choosing their order from their actual contracts. A shared scope has input_id=null and task=null. '
    'Then check global node-ID uniqueness, edge endpoints and required final verification. '
    'Root arrays contain root-owned nodes/edges only; each subgraph body appears exactly once in its '
    'own scope, never copied into root arrays. A subgraph call references its ID, not its body nodes. '
    'For each object-owned subgraph, trace the input through its processing outputs to the last '
    'required delivery. Check that task.destination_ref names THAT final destination and '
    'task.work_area_ref names the intermediate processing area, consistently with its completion_conditions. '
    'This is an entry-precondition audit, not permission to invent successful facts or omit checks.')


def graph_context(context):
    # Keep each instruction at least once. Only verbatim sentence duplicates
    # already present in the supplied skill document leave the catalog prose.
    documents = context.get('skill_instructions', {})
    catalog = {}
    for name, contract in context.get('skill_catalog', {}).items():
        document = ' '.join(documents.get(name, '').split())
        sentences = re.split(r'(?<=[.!?])\s+', contract.get('description', ''))
        description = ' '.join(sentence for sentence in sentences
            if len(sentence) <= 20 or ' '.join(sentence.split()) not in document)
        catalog[name] = {**contract, 'description': description}
    return {**context, 'skill_catalog': catalog, 'rules': GRAPH_RULES,
        'responses_call_policy': 'Exactly one initial Responses call; subsequent decisions use Decisions. No sub-agents.',
        'recovery_requirements': 'Foreseeable edges are optional. Runtime Decisions constructs validated recovery nodes/edges for unforeseen outcomes.',
        'initial_recovery_design': 'Do not enumerate all failures. Plan the complete intended task and any useful known branches.',
        'execution_failure_context': {**context.get('execution_failure_context', {}),
            'recovery_owner': 'decisions_runtime_graph_patch',
            'guidance': 'Unexpected evidence gaps request Decisions; only registered, currently valid capabilities can run.'}}

# Inventory map lifecycle and motion coverage (2026-10-26)

Collision mapping intentionally includes only observed inventory entities and the
measured table. Unlisted background does not enter the collision map. Geometric
motion coverage is separate from Decisions identity/result judgments.

For this workstation the user explicitly states that all work is around the table
and no other obstacles are beside or behind the robot. These operator-supplied
conditions live in `resources/workcell_assumptions.json` and are passed to both
Responses and Decisions. They are not sensor measurements, object poses or
workflow instructions; another installation must supply its own conditions.

## Build and update contract

`Algorithms.build_scene` sends `mapping_lifecycle` with `required_entity_ids`,
optional `invalidated_entity_ids`, `retired_entity_ids`, `max_retained_voxel_bytes`, and `workspace_bounds`
(`minimum_world_m`/`maximum_world_m`). Static table/marker records are not required
semantic entity IDs. The caller supplies active IDs; attachment is not retirement.

`reconcile_inventory` retains layers for an unseen entity. For a still-static
entity, exactly aligned public VoxelGrid fields are combined by elementwise
minimum (native ESDF is negative inside), preserving every occupied sample.
Different grid poses, dimensions, spacing or shapes retain independent historical
layers. The measured table cuboid is retained separately.
Every contributing observation ID is retained, so partial views cannot erase
older geometry. Published public VoxelGrid tensors
are cloned before Mapper reuse; a failed rebuild clears executable publication
but keeps the independent archive for recovery. There is no capture-count limit.
The configurable voxel tensor budget defaults to 256 MiB and rejects overflow
without discarding occupied evidence. Compatible repeated captures consume one
field; provenance retains all captures. Each incremental publication token also
binds its archive version, so identical latest RGBD cannot hide a different
retained collision world.

`build_scene` accepts model-selected global or local scope. Responses may choose
it in the graph or delegate to Decisions, which receives the task, current
observation, previous scene/observation and invalidation evidence. Local scope
filters actual Mapper masks by the selected measured IDs;
unselected entity layers retain their prior provenance. The independent raw
visibility capture is not filtered by that selection. The published
`mapping_strategy` records scope, source, reason and IDs. Missing model evidence
does not trigger a heuristic global/local fallback. Global means all currently
observed inventory entities, never proof of full spatial coverage.

Explicit motion invalidates the affected entity's freshness. An invalidated
entity missing from the new view stays stale and cannot authorize execution;
reobserving it replaces obsolete pose layers. Only explicit inventory retirement
removes an unseen entity. The static assumption is stated in each map: no
unreported entity movement since its last observation. No camera can establish
that assumption for unseen regions, and there is no hidden simulator truth.

Selection and attachment leave measured geometry bound to its identity.
Release and processing mark the affected geometry stale, even if result
verification is inconclusive. No local-object or action-envelope cuboid is
published. A rigid prediction may retain historical measured layers for
diagnostics, but never authorizes subsequent motion. Only a model-selected
build using fresh RGBD can restore map readiness. A local rebuild that leaves
required layers missing or stale retains its measured data and returns unknown;
it cannot clear the dirty flag. Changing the selected target cannot restore
readiness merely because a scene token still exists.
The obsolete `unpack_view_motion` obstacle-omission operation is rejected.
Consumed input IDs must be retired explicitly after verified processing
and registered input/output lineage; merely seeing a new product does not retire anything.

Results expose `inventory_map.entities` with observed/retained/stale
states and last observation IDs. `coverage.inventory_complete` describes entity
records only. `coverage.workspace_status` is unknown without a configured
workspace, partial without adequate calibrated evidence, and complete only when
the enclosing volume of that configured region is certified by registered
free-depth rays. Three cameras, a fixed observation pose, and Decisions success
are never geometric completeness proofs. `full_environment_coverage` stays false.

## Separate raw visibility evidence

Perception preserves `coverage_depth_m` and `coverage_valid_depth` before its
inventory masking. The normal depth remains masked. `Algorithms.build_scene`
serializes the original registered arrays as a distinct `coverage_rgbd` request;
only `MotionCoverage` reads it, never Mapper. Its observation ID, generation,
camera set, intrinsics, transforms and captured joints must match the occupied
map request. No raw-background obstacle layer is added.

Responses/Decisions choose mapping scope from the task, current observations,
retained entities and geometry changes. Numerical motion admission checks the
complete robot against the published collision world, self-collision and joint
bounds, and requires valid map/command bindings and fresh inventory geometry.
It does not additionally require every moving sphere to have a free-depth ray
certificate. Head and wrist cameras cannot observe the back of the shoulder or
all surfaces of their own tools; missing such a certificate is not a collision.

`MotionCoverage` remains a diagnostic of the complete TCP tool assemblies,
including camera brackets, grippers and carried attachments. Its unknown results
remain unknown, with the first deficient camera projection/depth reported. Other
links are explicitly not evaluated for sensor coverage, while remaining in all
numerical collision checks. `unknown_space_motion_gate=false` and
`whole_robot_unknown_space_certified=false` make this boundary explicit. This is
planning against measured/retained inventory geometry, not a guarantee about
unseen obstacles or full environment coverage. A TCP reachability atlas is not a
free environment map, and no clipping plane is inferred from a failed sphere.

The diagnostic certificate tests whole conservative sphere footprints before
valid first depth returns. Invalid depth, occlusion, partial/outside FOV and
behind-surface space are unknown. Each relevant public FK tool/attachment sphere
expands by an analytic per-sphere URDF joint-motion bound for the entire following
joint-linear command segment, not just its endpoints. The 10 mm depth uncertainty
margin is distinct from the existing 10 mm world collision requirement.

Self-occlusion is handled by bounded subdivision of the requested swept sphere:
each intersecting cell must fit in measured free rays, the unchanged admitted
robot volume at the measured capture configuration, or an explicitly permitted
target contact volume. Unknown cells fail the diagnostic certificate at 5 mm subdivision or the bounded
node budget. Each joint uses the sphere's distance from its rotation axis;
axial offsets do not create fictitious rotational travel. Downstream orbit
bounds, fixed transforms, prismatic limits and mimic multipliers are retained.
Stationary links have zero expansion. Initial robot
volume uses the same normalized collision-sphere radii as command checking,
including the existing gripper numerical envelope. The volume is a model
admissibility assumption, not observed empty space; it never
licenses growth into unknown space or gives attachment spheres a robot exemption.

Capture containment uses no spatial epsilon. Motion bounds account for both
requested joint deltas and deltas represented in the public FK input tensor's
actual dtype, taking the componentwise maximum. A sphere is stationary only
when all its influencing joint inputs have identical represented endpoints;
linear intermediate inputs then have that same representation. A changed
represented joint keeps its full bound. Subdivision certificates bound the
intersection of the query sphere and each cell, rather than rejecting irrelevant
box corners outside the sphere. Neither correction proves that later shoulder
motion outside all camera views is safe.

Target contact is limited to the selected entity's current published measured
surface bounds and the owning gripper/attachment sphere
indices, only in applicable manipulation phases. It cannot excuse forearm,
camera, peer arm or another obstacle, and the target region does not move with
the trajectory. The regular world/self/bound checks remain mandatory. Retained
free-ray certificates are not refreshed by action predictions; the visibility
proof alone is never a collision-free execution result.

## SDK and verification boundary

Pinned official SDK commit: `8e734f3ced1df898990bcd92de40abce475907db`.
The local official archive confirms public `RobotCollisionChecker.get_kinematics`
and `get_sphere_index_from_link_name`; project adapters consume their outputs.
No official or vendored SDK implementation is edited.

Tests cover retention across partial captures, invalidation/reobservation,
explicit retirement, tensor archive independence, twelve-capture union and byte-budget enforcement, separate
raw visibility versus inventory depth, stale/mismatched capture, invalid and
occluded depth, motion envelope, complete tool/attachment coverage, bounded start
volume, target-only contact, and the actual command-check boundary with CPU
substitutes. These do not establish a complete physical run. GPU validation,
actual camera coverage and controller tracking remain separate acceptance work.
The geometric proof assumes calibrated pixel cells and correct collision models;
it does not certify subpixel obstacles, full hidden object shape or physical
tracking. Gripper-opening validation still checks endpoints, not its continuous
opening sweep.

## Coverage execution cost

Coverage memoizes exact sphere geometry, expansion and contact permissions in a
bounded cache; it never rounds geometry. A change to captured robot spheres
invalidates the cache. Stationary admitted robot volume is checked before camera
projection. Command admission stops coverage evaluation on the first unknown
sphere and reports examined versus requested counts and an explicit lower bound
on unknown counts. Only complete successful evaluation can claim that this diagnostic scope is
covered. Coverage failure does not alone reject a command; map freshness, bindings
and full world/self/bound checks remain mandatory. Failure evidence includes
the first sphere's expansion and closest captured robot sphere containment gap.

## Historical service evidence before the user-supplied workcell assumption

The natural-language direct transfer run `d0610b29aebc4e9daafd2e41d4d1d2a5`
used one Responses request and reached successful native grasp planning. Its
five candidates failed the separate coverage gate before any robot command.
Archived RGBD/native trajectory replay requires no model or motor request.
Fail-fast validation reduced the recorded candidate's approach check to about
0.15 seconds. This is a rejection latency measurement, not a successful transfer.

The replay also exposed a stationary-model mismatch: capture FK omitted the
gripper normalization used by command FK, producing a one-float32-ULP radius
difference. Both now use the same normalized configuration; no coverage margin
was enlarged. Later trajectory chunks still lack certificates, including a
Link1 sphere whose continuous segment envelope extends roughly 0.45–0.54 mm
beyond its closest admitted capture sphere. This does not establish collision;
the existing evidence cannot certify that conservative swept volume.

Current sensors are two wrist cameras attached to the TCP links and one fixed
world-mounted head camera. Capture already includes all three. There is no
additional registered view available without moving a wrist. Repeating an
unchanged capture does not resolve fixed occlusion or out-of-view space; UI orbit
views are not depth evidence. A changed wrist view needs an admitted observation
motion itself. If no such path can be certified, execution must remain stopped
until a calibrated additional view or explicitly surveyed region is available.
The implementation did not silently add either as a simulated free-space prior.
The user subsequently supplied the explicit workstation assumption described
above. The current implementation retains this coverage result as a diagnostic;
the old whole-tool visibility gate described in this historical experiment is
no longer an execution prerequisite.

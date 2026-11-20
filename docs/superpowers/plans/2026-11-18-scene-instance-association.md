# Scene Instance Association Implementation Plan

> Use superpowers:executing-plans in this session. Repository instructions prohibit sub-agents and worktrees. User approved the design and implementation on 2026-11-18; no additional plan approval is required.

**Goal:** Build a complete observable scene inventory, associate physical instances using current RGBD/CAD evidence and Decisions, then bind task selectors independently.
**Architecture:** Preserve Base, SDK, graph and storage boundaries. Replace task-owned identity gating with independent scene association and partial physical fact commits. Retain the database and existing IDs.
**Tech Stack:** Python/Pydantic/NumPy/SciPy/SQLite, pinned SAM3 and FoundationPose services, current web workbench.
**Spec:** ../specs/2026-11-17-scene-instance-association-redesign.md

## Global constraints
- Work in the existing repositories; preserve unrelated uncommitted changes and historical runs.
- No official SDK edits, simulator object truth in algorithms, hidden capture/movement, or new confidence thresholds.
- SAM3 source 5dd401d1c5c1d5c3eedff06d41b77af824517619; FoundationPose a1b694b83e633c2cb6115b9063d940a687759392.
- Luna checks are acceptance scripts under data/acceptance/scene-association-redesign; algorithm tests use its existing unittest suite.
- No automatic broad commit of the pre-existing dirty tree. Save a scoped before snapshot and a change ledger instead.

## Review focus
- Model-inferred shape must not become a user requirement (Task 1).
- Empty common visibility must produce unavailable, not zero overlap (Task 3).
- Identical CAD assets must not share instance tracking state (Tasks 3/4).
- One unresolved task selector must not roll back independent scene identities (Task 5).
- Rebinding must update graph/object-task context consistently (Task 6).

## Task 1 — Separate scene inventory and selectors
Files: agent/schema.py, agent/responses.py, agent/graph_prompt.py, core/grounding.py, runtime/controller.py, runtime/compiled_task.py.
Interfaces: SceneProposal(proposal_id,camera_id,bbox,description,kind,color,segmentation_prompt,asset_id); TaskSelector(ref,selection,requested_attributes,instruction_quote,...); InitialPlan.scene_inventory/task_selectors/destination_regions.
- [x] Reproduce current filtering and inferred-description failures in acceptance checks.
- [x] Implement distinct typed contracts, validate task quotes and independent scene proposals, remove task inventory filtering.
- [x] Verify strict schema generation and scene-only objects survive compilation.

## Task 2 — SAM3 geometric prompt service
Files: algorithm services/sam3_service/{tools,wire,service,client,backend}.py and existing service tests; Luna perception import boundary.
Interface: segment_box payload includes rgb_jpeg,text,bbox_xyxy,timestamp_us; official normalized cxcywh conversion.
- [x] Add failing frame/box validation and processor-call checks.
- [x] Expose official geometric prompts without changing segment_text behavior; keep SDK read-only.
- [x] Run service contract checks and the existing algorithm unittest suite.

## Task 3 — Candidates and comparable geometry
Files: candidate_observation.py, association_depth.py, observation_overlap.py, pose_estimation.py, cad_perception.py; new association_geometry.py.
Interfaces: collect_candidates(capture,queries,segment_query,segment_box,inventory); candidate pose evidence and same-capture pair evidence.
- [x] Add checks for cross-camera reprojection, occlusion, missing support, duplicate masks and known CAD candidates without permanent IDs.
- [x] Collect all inventory queries, current box prompts and bounded residual proposals; retain unresolved geometry/coverage.
- [x] Compute depth-aware reprojection overlap and CAD candidate poses before association; reuse accepted poses downstream.

## Task 4 — Replace current/temporal association
Files: current_instances.py, temporal_identity.py, task_grounding.py; new scene_association.py.
Interfaces: scene association returns groups/history matches/confirmed independent components/unresolved evidence; task selection consumes those groups.
- [x] Add cases for one-view objects, duplicate views, competing old IDs, lost tracks and genuinely new objects.
- [x] Use unified geometry and joint competing candidate context in Decisions; enforce global uniqueness and bounded conflict review.
- [x] Verify moved identities are not rejected by historical position and unmatched is not automatically new.

## Task 5 — Independent identity commits and references
Files: agent/registration.py, object_database.py, memory_reference.py, memory_contracts.py.
- [x] Reproduce missing instance_rgbd_grounded archive and task-unknown all-or-nothing commit.
- [x] Commit confirmed independent groups, preserve unknown tracks, save immutable initial reference plus latest reference.
- [x] Verify idempotency, rollback on actual database conflicts, and original IDs survive resets in the same world.

## Task 6 — Runtime and task binding integration
Files: runtime/grounding.py, compiled_task.py, controller.py, native_graph.py, perception.py, related observation inputs.
- [x] Reproduce target_outside_active_object_task with late-bound refs.
- [x] Connect scene association/commit followed by task binding; remove synthetic @tracked selectors and old task-driven identity flow.
- [x] Verify once/each_selection behavior, geometry reuse and evidence cache versioning.

## Task 7 — Contracts, cleanup and review
Files: AGENTS.md, README.md, project skill, skill_catalog/ground_task and observe_scene, docs/rgbd-grounding.md, docs/luna-skills-guide.md, replay consumers.
- [x] Replace obsolete task-only instructions and remove retired branches/callers.
- [x] Run acceptance checks, Python compile, scoped lint, web build, algorithm suite; inspect final delta against before snapshot.

## Task 8 — Live validation
- [x] Verify current services idle and provenance; deploy only changed authorized application/service files and propagate scoped removals.
- [x] Replay frozen observations; document association outcomes separately from task binding and motion.
- [x] Run five pink-cuboid and five black-CAD simulation tasks with preserved trial evidence; report every failure and remaining limitation.

## Current verification status (2026-11-18)

- Development-v22 is deployed. All ten fixed-version trials are complete: 0/10 succeeded, 8 stopped at task_grounding_unknown and 2 failed Responses validation (requested_selector_requires_user_quote). No trial reached grasp execution. **Implementation and evaluation are recorded; the reliability acceptance criterion is NOT met.**
- Final local v22 checks: 36 association checks, 4 contract checks, Python compilation and scoped Ruff pass. Algorithm service suite: 55 checks (2 skips); SAM3 suite: 40 checks. The algorithm source has not changed since those recorded checks. Final web build has a separate log.
- Original ten frozen scene observations were evaluated at v17: only 3/10 had the expected class-count signature. This is a screening result, not identity accuracy. A partial v18 replay was stopped after three cases; cancelled cases are preserved and excluded from conclusions.
- Two-frame structured replay at v17 and v20 preserved five principal IDs and bound the task on both frames. The v22 replay of the latest development input also bound both frames and preserved those five principal IDs, but registered a table-edge fragment as an extra gray object in frame one. Therefore semantic identity quality remains unaccepted.
- Development live trials v3/v4/v5/v21 stopped with task_grounding_unknown before manipulation. Their failures and recording gaps remain archived. No development trial substitutes for the formal ten.
- The immutable pre-task snapshot, per-deployment literal diffs, request/response evidence and failed attempts remain in the acceptance directory. The pre-existing dirty repositories were not reset or broadly committed.
- Remote algorithm wrappers use `algorithms/<name>/service` and published CPU packages. Deployment preserves that layout and the FoundationPose package names; these differ intentionally from the local checkout.
- Final report: `data/acceptance/scene-association-redesign/acceptance-report.md`. All 411 audited source files, CAD manifest and container images remained unchanged across the ten trials. Original simulator configuration is restored; task idle and both arms disabled. All ten recovered RGB sample videos are local, with their gaps explicitly documented. Healthy services and successful unit checks do not prove reliable transfers.

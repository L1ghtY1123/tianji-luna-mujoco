# Continuous transfer implementation plan

**Goal:** Implement the approved continuous transfer, visual assistance, collision model and JEV-controlled recovery requirements.

**Architecture:** Extend existing decisions/runner/contracts. Keep task progress separate from per-object motion facts. Independent marker, visual assistance and sphere work has exclusive file ownership; the main agent integrates perception/planning/runner changes.

**Spec:** ../specs/2026-10-12-continuous-transfer.md

## Global constraints

Recovery choices are JEV outputs. All skills remain visible. No privileged environment state, no SDK/vendor edits, no service restart while running. Approximate 300 robot spheres; no silent parameter changes. Same-capture VLM reuse; bounded calls. User authorized execution; no repeat approval gate.

## Review focus

Identical error with distinct JEV recovery answers; lost/ambiguous object identities; completed-count duplication after reobservation; failed held-object release; hidden geometry incorrectly declared free.

## Tasks

- [x] 1. Establish baseline snapshot, local suites and remote hash/status evidence.
- [x] 2. Task state and decisions: src/luna_mujoco/task_progress.py, decisions.py, tests/test_task_progress.py. Tests first: explicit arm bypass, quotas/exclusions through JEV candidate questions, deferred/completed IDs and bounded context. Preserve single-object compatibility.
- [x] 3. Runner recovery/iteration: runner.py and skills contracts. Tests first: verification failure returns to JEV, different decisions produce different actions, no automatic retry/defer, next-target scene invalidation, partial completion and unknown result.
- [x] 4. Camera-visible marker: model_builder.py, table_markers.py, relevant web scene/editor modules and tests. Interface: detector returns measured marker instance with kind=marker and placement_reference from RGBD. No runner/decisions edits by worker.
- [x] 5. Visual assistance: visual_assistance.py and tests. Interface: inspect_scene(capture, objects, instruction, *, reference=None, call_budget=None) returns grounded visual observations, not commands; preserve inspect result compatibility. No runner/perception edits by worker.
- [x] 6. Collision refit: resources/curobo, fitting/audit scripts and tests. Approximately 300 spheres; fixed SDK official fit APIs. Exclusive output directory data/continuous-transfer-20261012/spheres. No other source modifications by worker.
- [x] 7. Scene/perception/placement integration: perception.py, placement.py, algorithms.py. Tests first: marker destination, objects inside bins, region release points and occupied-height support, new scene for every completed transfer, selected-arm-only IK.
- [x] 8. Performance and context: merge independent intent questions, compact per-object snapshots and logs, measure cache/build/plan timings without dropping safety checks. No measured end-to-end speedup claimed.
- [x] 9. Full Python/frontend tests and lint/build; fresh review and fixes; isolated real JEV + remote RGBD/physics cases; synchronized deployment with hashes and baseline restoration. Visual API gateway and browser-attachment limitations are recorded in ../../continuous-transfer-acceptance-2026-10-12.md.

Each behavior task uses a failing regression first, targeted tests after implementation, then the affected suite. No git repository exists, so baseline-manifest.json and baseline.zip replace commit-range evidence. Progress is kept in data/continuous-transfer-20261012/progress.md.

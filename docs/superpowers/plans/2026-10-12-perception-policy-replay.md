# Perception Policy and Replay Implementation Plan

> Implement using test-driven-development and the existing independent domain agents. Record evidence in data/next-revision-audit-20261012.

**Goal:** Implement the nine follow-up requirements without transferring failure recovery away from JEV.

**Architecture:** Extend the existing Runner/Decisions, RGBD collision preview, sphere generator and replay modules. Keep model choices separate from physics and contract validation.

**Tech Stack:** Existing Python, MuJoCo, pinned cuRobo, TypeSafe, OpenAI image assistance, vanilla JavaScript/Three.js/Vite. No SDK/vendor changes or new dependencies.

**Spec:** ../specs/2026-10-12-perception-policy-replay.md

## Tasks

- [x] Capture source baseline, trace code paths and identify concrete defects.
- [ ] Root: tests for unknown verification without automatic VLM; explicit JEV inspection and re-verification; scene-policy argument validation; fresh/stale evidence in state. Update runner.py, decisions.py and affected task skills. Run focused Runner/Decisions/recovery tests.
- [ ] RGBD agent: renderer occlusion regression, actual collision-layer preview and source labels; preserve import boundary. Own perception_mapping.py, geometry_service.py preview sections, app.py preview route, collision_mapping.py and pointcloud-panel.js/tests.
- [ ] Sphere agent: generate <=350 symmetric candidate with public pinned fitting APIs; compare identical samples and local slices, FK replay and GPU validation. Own sphere scripts/config/evidence only; stage candidate before promotion.
- [ ] Replay agent: red-green saved/pending trash deletion, active/read/export guards, compact trace graph/event retention/selection/export. Own experiments.py, experiment_api.py, replay files and tests.
- [ ] Root: safe connection/data reuse where profiling supports it; record reference timings and avoid unsupported end-to-end claims.
- [ ] Integrate: full pytest, frontend tests/build, static checks; review dangerous boundaries and fix verified regressions.
- [ ] Isolated runtime acceptance: actual JEV choices, reference physics, collision preview payload, saved replay operations on a temporary store; preserve live user data.
- [ ] Deploy verified changes if all applicable gates pass; compare hashes and service health, restore original scene; otherwise report precise remaining limit without claiming deployment.

## Review focus

Unknown verification must remain actionable without automatic retry or VLM; stale observations must never authorize stale plans; contact policies must not remove walls or self-collision; map previews must come from actual map arrays; deletion must not race active archive writers/readers. Sphere coverage and center symmetry need separate checks.

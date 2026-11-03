# CAD placement performance implementation plan

> Execute inline under the repository policy; no subagents, worktrees, commits, or additional Responses calls.

Goal: preserve measured mask/CAD/permanent-ID associations while shortening perception and release selection, then measure complete MuJoCo unpack runs.

Spec: the user's 2026-11-02 request removes the independent 10 mm release gap, requests understandable CAD names despite SAM synonyms, and requires faster, more reliable simulation. Retain FP registration because a mask alone does not supply a 6D pose. Keep current numerical pose and collision thresholds.

- [x] Replace the slow point/mesh query with the public libigl 2.6.3 double-precision AABB query. Keep all points, triangles, residual metrics and depth/silhouette predicates; compare analytic and frozen accepted/rejected cases. Query batches below the library's 10,000-point parallel threshold.
- [x] Expose canonical blue_box / black_workpiece asset names and Chinese display names. Preserve original CAD repository IDs through an explicit source_asset_id boundary; no rewriting offline grasp transforms or source hashes. Pass every actual SAM prompt proposal and catalog aliases to Decisions as retrieval evidence, never as confirmed identity.
- [x] Default the independent release gap to zero and migrate this project's active setting. Preserve hard clearance and the existing 0.1 mm measured servo reserve. Try the lowest eligible held-IK sample first, then bounded ascending batches only after failure.
- [x] Profile robot self-filtering separately; accelerate only if unchanged measured-joint mesh ray evidence is demonstrated.
- [x] Run targeted regressions, full feasible checks and Ruff. Sync with before/after hashes, update skill/docs and the necessary dependency image, then run the real unpack task repeatedly. Record every run, phase time and failure; do not claim a success rate from unit tests.

Review focus: ambiguous SAM words; two instances sharing one asset; historical asset aliases and arm-specific grasps; exact collision-boundary starts; wrong FP pose or missing depth. SDK sources stay read-only; all five container CPU limits remain unchanged.

Evidence update: timing-35 is synchronized (971 hashes, four logical CPUs total). The timing-33 full suite passes 4383 tests with 58 skips and one third-party deprecation warning. Subsequent timing-34 and timing-35 changes pass 126 and 107 relevant tests plus Ruff. All failures, frozen diagnostics and completed runs are recorded in docs/cad-placement-performance-20261102.md. P28 completed in 563.593 seconds. P32 completed on timing-35 in 592.506 seconds with one Responses and 53 Decisions calls, video disabled, and two unknown black-release checks before confirmation. This proves one full successful latest-version run, not repeatable reliability or the 2–4 minute target. The release-proof serialization overhead is substantially reduced, but repeated perception and identity checks offset that improvement.

- [x] Demonstrate a complete unpack-and-delivery run with final task verification (P28 and latest P32).
- [ ] Establish reliable repeatability. Earlier runs include current-image identity uncertainty and graph precondition mistakes; frozen replay and unit tests are not physical success-rate evidence.
- [ ] Meet the 2–4 minute task target. P32 spends 91.302 seconds in initial Responses and 219.900 seconds in eight observe_scene calls. Decisions median is 1.467 seconds; redundant calls and CAD registrations accumulate. Continued evidence reuse must preserve current-image, geometry, generation and collision guards.

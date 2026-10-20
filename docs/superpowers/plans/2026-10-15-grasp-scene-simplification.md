# Grasp feedback and scene preparation implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Independent SDK investigation may use dispatching-parallel-agents; implementation remains coordinated in this checkout.

**Goal:** Implement the user's accepted simplified observation cycle, fast feedback-only grasp verification, 15 mm sphere-bottom release, and measured GPU mapping optimization.

**Architecture:** Keep JEV as task/recovery decision maker. Reuse current runner, contracts and geometry-service boundaries. A single scene-preparation skill owns move/capture/perception/map; feedback verification owns no camera or GPU attachment work. Public cuRobo APIs only.

**Tech Stack:** Python, existing locked cuRobo GPU image, MuJoCo, SAM3, current JEV contracts and web UI.

**Spec:** User instructions in this thread on 2026-10-15 and accepted previous response. The exact release gap is 0.015 m, hard clearance remains 0.010 m. No SDK/vendor edits, new dependencies or images. Preserve the remote held production scene until deployment can avoid losing it.

## Global constraints

- RGBD and robot feedback only; no simulation object truth.
- Initial head observation creates immutable per-scene table cuboid and bootstrap motion world. Rebuild on scene reset/calibration changes, not because arm approaches table.
- Before each grasp: move both arms using existing collision world, capture and analyze, associate IDs, build semantic world. No JEV calls inside a skill.
- No normal post-grasp/post-place RGBD; feedback-only verification. Feedback unknown is distinct from failure; JEV chooses recovery.
- Attachment is prepared once as part of place preparation, separately timed, after verified holding. Never weaken attachment or collision checks to claim speed.
- TCP XY is the selected candidate XY. Compute Z per actual uploaded sphere envelope and candidate orientation; keep roll/pitch, 18 yaw; 15 mm gap, no automatic higher release.
- Remove obsolete runtime skills and update registry, caller branches, descriptions, tests and guide consistently. Historical replay strings remain readable.
- Reuse table and filter to measured SAM3 instances. Preserve required ESDF query halo; assess GPU parallelism with measured equivalence, memory and wall time.

## Review focus

- Feedback-only verification with failed camera/attachment services; same target identity and unknown handling.
- Scene preparation interrupted after motion or SAM3: no stale map becomes executable, no hidden retries.
- Two objects swapped in list order, completed target and newly observed obstacle: IDs and task membership unchanged.
- Eccentric attachment, yaw changes, marker/bin boundary: TCP XY semantics and actual release clearance.
- Parallel GPU layers with unequal bounds: no shared buffers/stream races, same obstacles and target exclusion, bounded memory.

## Task 1: Feedback and atomic preparation

Files: runner.py, skills.py/registry as discovered, skills contracts, affected route/recovery tests.
- [x] Add failing tests: grasp reads own stable feedback without RGBD/GPU; verify_place uses feedback; plan_place prepares attachment; scene preparation fixed internal order and no mid-skill JEV; next round refreshes scene.
- [x] Implement within existing runner boundaries; remove obsolete robot-facing skills after checking actual callers.
- [x] Run targeted tests then affected route, identity and recovery suites; fix contract consumers.

## Task 2: Release geometry

Files: placement.py, planning_settings.py, release validation and tests.
- [x] Failing tests for literal 15 mm sphere-bottom gap, fixed TCP XY under yaw/eccentric attachment, unsafe clearance setting rejection.
- [x] Implement formula and settings with actual uploaded spheres; validate floor and destination bounds, no elevated fallback.
- [x] Run placement/attachment/settings suites.

## Task 3: GPU scene optimization

Files: perception_mapping.py, semantic_builder.py, geometry_service.py, measured_table_mapping.py and project-owned adapter if needed; vendor read-only.
- [x] Inspect pinned SDK public Mapper/config/examples and current profiling evidence. Record current sequential baseline on identical captured inputs.
- [x] Add stage timers and regression coverage; preserve fixed cached table, avoid needless robot model rebuilds via supported APIs.
- [x] Benchmark public-API multi-instance GPU parallel scheduling or batching; use faster bounded implementation only with equivalent geometry and collision results.
- [x] Reduce needless ESDF allocations while retaining sphere-center collision coverage. Record cold/warm wall time, occupied surface and ESDF cell count, GPU memory.

## Task 4: Integration and delivery

- [x] Update README, docs/jev-skills-guide.md and affected SKILL/contracts; explain ESDF cells, remaining limits and exact observation times.
- [x] Full Python, Ruff and frontend tests/build as affected; independent final code review and fixes.
- [x] Validate GPU mapping and release/feedback entry points in existing image on isolated ports/processes. Do not alter active held production run.
- [x] Commit and synchronize approved changes/candidate to Ubuntu, report actual activation state explicitly; no unrequested image builds.

## Progress and rulings

- Base: 79cc0aa. Native worktree tool addresses chat cwd, which is not a repository; use isolated git worktree for the explicitly named project.
- User already authorized execution; no second design-approval gate.
- Tasks 1/2 share place entry settings; keep implementation sequential. Task 3 SDK investigation is read-only and can run independently. Full review after integration.

- Final local CPU: 1507 passed,54 skipped; frontend248 passed/build. Two GPU captures preserve TSDF/retained ESDF exactly; warm mapping .287→.101s and .218→.119s. See docs/scene-opt-validation-2026-10-15.md.
- User authorized archiving the held production state, activating this candidate, and restarting existing services.
- Implementation 87ab2b2 is active in Windows/Ubuntu main. Archived run ccbb5076286f4477a713430dccef61c4 before updating mounted files; restarted only existing web/curobo containers. All four services healthy, 16 runtime skills, persisted 15 mm release / 10 mm collision settings. No new images or automatic manipulation run. See validation report for paths and remaining motion-validation limits.

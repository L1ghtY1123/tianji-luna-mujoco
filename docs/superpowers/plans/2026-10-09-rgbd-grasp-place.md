# Three RGBD Cameras and Grasp Place Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent owned files, with root integration and final code review. Steps use checkbox syntax.

**Goal:** Execute a color/shape pick into a colored open bin using only three RGBD environmental observations, with phase-correct carried-object collision checking.

**Architecture:** Keep existing Registry/JEV choice and project adapters. A sensor adapter renders immutable registered RGBD, a perception adapter produces observed instances and camera clouds, and the GPU adapter builds a measured semantic ESDF and calls existing plan_place. Runner owns actual hold/release state and invalidates dependent plans when observations change.

**Tech Stack:** Python 3.12, MuJoCo 3.10.0, existing TypeSafe/Pydantic AI, SAM3/GraspGen images, cuRobo 8e734f3ced1df898990bcd92de40abce475907db, Three.js/Vite.

**Spec:** ../specs/2026-10-09-rgbd-grasp-place-design.md

## Global Constraints

- No environment truth in JEV/algorithm inputs, drift or success verification; renderer/editor/reset/oracle alone may read physics truth.
- Keep official SDK and vendor files read-only; image names include guangyu; no shared algorithm rebuild.
- MuJoCo keeps physical contact/gravity; attachment is a planning representation, never a weld or teleport.
- UI text stays at least 20 px; m/rad algorithms, mm/degree form controls; settings frozen per run.
- User explicitly authorized answer-then-execute. Continue without another review prompt; no Git repository exists, so use a recoverable source backup/hash manifest instead of Git commits or worktrees.

## Review Focus

- No object metadata before observe or through recent_results/errors.
- No opening fingers into a bin wall after a valid robot path.
- Actual release survives a subsequent observation/retreat failure.
- Same-color source/bin association remains unique across capture refreshes.
- Attachment/preflight cannot survive reset or contaminate a different solver/arm/request.

## Interface Contracts

Sensor: `Simulation.capture_rgbd() -> dict` with observation_id, generation, time, arms (robot telemetry), views keyed 1/2/3; each view has rgb uint8 HxWx3, depth_m float HxW, valid_depth bool HxW, intrinsics {fx,fy,cx,cy,width,height}, T_world_camera 4x4, robot_joints_deg {A,B}. Raw arrays are internal and never JEV state.

Perception: `Perception.observe(capture, *, instruction='') -> dict` with observation_id, generation, time, source='rgbd', objects and table estimates, arms robot telemetry, views including `instance_masks` keyed detector-owned id. Objects use existing id/kind/color/position/quaternion/size/T_world_object names but explicitly estimated, plus points_world_m, points_object_m and coverage. Bins add cavity {inner_size_m, T_world_bin, floor_height_m, uncertainty_m}. Public projection omits arrays/views. `Perception.associate(previous, current)` returns uniquely rebound object ids or raises missing/ambiguous.

GPU: existing grasp/IK typed endpoints remain. New `/perception/build` consumes semantic transport encode payload with observation_set_id, confirmed_instance_id, selection_generation and 3 registered RGBD views. New `/tools/plan_place` and `/tools/plan_to_pose` use existing typed requests + planning_settings, scene_token and project optional carrying_binding (observed points in EE + owner/frame/state). Response uses phases approach/place/retreat native radians/dt/derivatives and actual attachment/collision evidence. `/carrying/validate` performs phase-aware command-point/gripper checks. Clear lifecycle endpoint invalidates runtime carrying on reset.

Application: `Algorithms.observe(capture, instruction='')`, `build_scene(observation, target_id,...)`, `generate_grasps(observation,target_id)`, existing IK/grasp over observation rather than physics snapshot; `plan_place(observation,target_id,arm,scene,holding,destination_id,...)` and `validate_execution(...)`. Root integrates these signatures; agents do not edit algorithms.py/runner.py/decisions.py/app.py/simulation.py.

## Task 1 Sensor and RGBD perception

Files: new src/jev_mujoco/cameras.py, perception.py, camera calibration resource JSON; tests/test_cameras.py, test_perception.py. Owner camera agent. Simulation hook integration is root-owned.

- [ ] Write/run failures for off-center projection, mm-to-m/head-once/wrist FK, RGB/depth alignment, same-frame capture and detector identities without simulator IDs.
- [ ] Implement `RgbdRenderer(model, calibration).capture(data, generation, arms)` using public MuJoCo Renderer/MjSpec camera API and calibrated RGB optical poses; one thread owns rendering. Export setup helper for root model/simulation integration.
- [ ] Import full precision historical calibration with hashes, use RGB-aligned depth and normalize mapper views by valid-mask-preserving head letterbox.
- [ ] Implement SAM3 image transport/client from base read-only service contract, measured color/class/bin cavity fitting and cross-view association; no HSV-only object identity substitute for SAM3, no ground-truth segmentation.
- [ ] Run targeted tests and renderer smoke; record partial surface/occlusion limitations.

## Task 2 Open bins, settings and web

Files: model_builder.py, planning_settings.py; web/src scene/editor/settings modules and CSS; tests/test_bins.py, related settings and web tests. Owner UI/bin agent. Do not edit simulation.py, app.py or runner.py.

- [ ] Write/run failure for kind=bin, five collision components, nonzero open cavity, dimension validation and gravity; no change to solid box.
- [ ] Add rigid gravity-supported bin geoms, mass and editor parameters via existing object schema/render manifest. Provide bins' shape estimates consumers only through perception.
- [ ] Add distinct place approach/retreat settings z/+0.10/world, release_margin_m=.005, place_timeout_s=180, dwell place fields with schema bounds and revision compatibility; grasp settings unchanged.
- [ ] Add place schema group, bin editing/drag/preview and camera panel consuming `/api/cameras`; keep text >=20 px. Roots implements camera endpoint.
- [ ] Run Python/Node targeted tests and frontend build.

## Task 3 Measured mapping, plan_place and attachment

Files: geometry_service.py, backend_builder.py, new carrying.py/perception_mapping.py as needed, trajectory_postprocessing.py; GPU service tests. Owner GPU agent. No algorithms.py, vendor, resources/curobo/marvin_gui_dual.yml or SDK edits.

- [ ] Read pinned official APIs/demo and test failures for measured semantic map, exact place request mapping, single compound backend, carried body hits wall when robot clears, release phase omits attachment.
- [ ] Adapt existing frozen semantic mapper to registered RGBD and publish measured layers/table with selected-instance exclusion; validate IDs/generation/tokens, all non-target layers stay enabled.
- [ ] Build existing single GraspMotionPlanner with shared solver config; route copied native plan_place/plan_to_pose and retain q/v/a/dt before subsequent solves overwrite buffers.
- [ ] Use configured public attachment/temporary carried-link config so carried geometry participates in approach/place optimisation and mandatory dense actual-command checks. Attach at validated hold; preflight attachment is scoped. Native compound retreat is provisional with attachment, so do not execute it after release; detach/restore object map then fresh plan_to_pose and validate.
- [ ] Implement phase/gripper closure-opening collision checks via public checker and observed envelope; do not exempt bin walls. Clear in finally/reset/cache rebuild, bind owner/scene/observation.
- [ ] Run CPU contracts and isolated Ubuntu GPU SDK/attachment/semantic/place validation.

## Task 4 JEV and application integration

Files: runner.py, algorithms.py, decisions.py, simulation.py, app.py, verification.py, new placement.py; approved skill docs/contracts and integration tests. Owner root.

- [ ] Write/run failing no-truth state, source/bin color split, grasp_place continuation, release facts, same-candidate downstream selection, reset late-results and hold-session tests.
- [ ] Add short locked model/state copying for sensor worker. Expose RGB previews and minimal robot telemetry independently of editor snapshot. Close GL worker before model replacement.
- [ ] Route observations into all algorithm paths, use measured object cloud/frame, remove simulation environment from decision state/guards/verifier; oracle stays separate.
- [ ] Bind destination independently; new identify_destination/plan_place/execute_place/verify_place skills, needs_place/grasp_completed and current-observation requirements, dynamic verification result facts.
- [ ] Pair IK evidence first, selected-arm grasp+place preflight from identical candidate lift endpoint; continue candidates by existing score order; before-movement no-solution returns actual evidence to JEV.
- [ ] Close -> camera hold estimate -> attach/validate lift -> lift -> verify; reobserve/map after pick; place -> validate open -> release facts immediately -> reobserve/map -> detach/fresh retreat -> verify camera containment/stability.
- [ ] Keep holding_state across grasp-only/place Start, clear on release/reset/generation/restart; fresh unique rebind required. Extend step budget for two observe/build phases without loops or hidden JEV replacement.
- [ ] Run integration and existing regression tests; fix only affected established fixtures.

## Task 5 Deployment and physical simulation acceptance

Files: deploy/Dockerfile.web, deploy/algorithms.compose.yaml, .env.example, deployment scripts and validation scripts/docs. Owner root after agents.

- [ ] Add OSMesa/GL env before import and independent SAM3 service from existing image; verify health without exposing credentials.
- [ ] Synchronize only task-owned source/assets to Ubuntu with hashes; restart only JEV services and rebuild only named web image when needed.
- [ ] Actual three-camera RGBD, SAM3, GraspGen, measured ESDF and cuRobo attachment/place smoke with CPU/GPU limits logged.
- [ ] Run one color object into color bin with actual physics, check camera-driven grasp/place verification and independent oracle, reset at hold/release/retreat, record real failures/limits; no fabricated success.
- [ ] Browser verify bins/settings/cameras/run trace at desktop and narrow viewport, console; full feasible Python tests, pnpm test/build and lint.

## Task 6 Review and delivery

- [ ] Fresh review all modified files against spec and poison-state/attachment lifecycle/late-result regressions. Resolve material findings with failing tests first.
- [ ] Compare baseline hashes: official SDK/vendor and untargeted assets unchanged. Preserve baseline rollback archive and update README/current validation with exact deployed behavior.
- [ ] Deliver answers plus actual implementation, commands/results, browser/GPU/physics evidence and any remaining limitation. Never mark camera/attachment/place acceptance complete from unit tests alone.

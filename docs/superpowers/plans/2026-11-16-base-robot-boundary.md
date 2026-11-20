# Base robot boundary implementation plan

> Execution: superpowers:executing-plans, inline. The user approved implementation; no repeated approval gate. Repository instructions prohibit sub-agents and worktrees.

**Goal:** Luna controls and observes the real or simulated robot only through Base, without a MuJoCo dependency.
**Architecture:** Preserve policy and algorithm services. Base owns devices, camera admission, workcell configuration and public viewer resources; Simulation owns physics and RGBD rendering. Luna owns task recordings.
**Tech Stack:** Python, existing HTTP/Capability contracts, MuJoCo 3.10 only in Simulation, cuRoboV2-Fast 0.9.0 public Kinematics/RobotSegmenter, existing Three.js Viewer.
**Spec:** ../specs/2026-11-16-tianji-base-robot-boundary-design.md

## Global constraints

- No SDK edits, no direct Luna connection to controller or Simulation, no local simulator fallback.
- Keep Responses/Decisions/graph/memory semantics and numerical collision checks.
- No persistent test directories in Luna/Base; use temporary contract probes and actual entries. Existing algorithm/Simulation tests remain.
- No real robot motion during development. Source changes across the three projects and algorithm integration are authorized.
- Three-camera task capture requires complete registered RGBD, fresh frames, matched capture timestamps, settled images and frame-bound feedback/calibration.

## Review focus

- Repeated stale frames cannot satisfy startup/exposure stability.
- Simulator restart or scene/calibration update invalidates cached frames and executable plans.
- Lost motion replies must be queried by request identity and never retried as new actions.
- Neither FK nor depth filtering may import MuJoCo after migration.
- Live viewing and recorded playback must not alter Base's shared robot state or start devices.

## Task 1: Camera providers and capture admission

Files: Base `extensions/operations/robot-bridge/observation.py`, new `camera_readiness.py`; `extensions/cameras/orbbec-camera/backend.py`; new `extensions/cameras/mujoco-camera/`.
Interfaces: `read_rgbd_set(camera_ids, fresh_ms=500)` returns an immutable set of atomic RGBD pairs and sync evidence. Bridge `capture` adds explicit `require_stable` and sync diagnostics while retaining its existing route and physical source validation.

- [x] Reproduce missing complete synchronized RGBD using fake Providers; check unstable/repeated frames, camera restart, skew and physical identity.
- [x] Implement history-backed RGBD group matching and bounded per-camera exposure/image stability, with session reset.
- [x] Add independent MuJoCo Camera/Calibration Provider through existing public Capabilities; reads do not start resources.
- [x] Run offline admission probes and compile/lint changed Base modules.

## Task 2: Simulation RGBD service

Files: Simulation new `marvin/twin/mujoco_cameras.py`; `mujoco_bridge.py`; `scripts/run_digital_twin.py`; Docker/deployment arguments; Base-owned camera calibration resources.
Interfaces: Base-owned calibration document; `/v1/frames`, `/v1/calibration`, `/v1/health` carry simulator instance, frame group, scene/calibration revision and capture clock domain. Three views come from one copied state; renderer owns GL thread.

- [x] Write/run failing camera projection and atomic snapshot checks.
- [x] Install calibrated body cameras; encode metric depth and RGB together; publish latest complete frozen group without blocking physics.
- [x] Rebind on scene recompilation; clear all old images on restart/config change; wire startup and provider selection.
- [x] Validate projection, depth, wrist poses, frame identity and existing simulation regressions.

## Task 3: cuRobo FK and own-robot depth removal

Files: algorithm service client/adapter; Luna `geometry_service.py`, `raw_robot_depth.py`, `kinematics.py`, `robot_surface.py`, perception composition.
Interfaces: batched measured-joint FK returns mount/front TCP and link transforms; robot masking returns per-view masks using the official RobotSegmenter and captured joints.

- [x] Read pinned public exports/source/examples and verify original SDK hashes.
- [x] Add service operations and typed CPU client; replace local MuJoCo FK and ray classification.
- [x] Use captured per-view joints and actual gripper configuration, preserve original RGBD and filter provenance.
- [x] Validate CPU transport/error cases and GPU results where reachable; record measured GPU acceptance separately from CPU contracts.

## Task 4: Luna Base execution adapter

Files: new `base_client.py`, `base_robot.py`, `robot_errors.py`; `app.py`, runtime/skills imports, deployment environment.
Interfaces: preserve currently consumed robot methods: snapshot, robot_snapshot, capture_rgbd, preview_rgb, latest_rgbd, start/close, execute_trajectory/execute_dual_trajectory, set_gripper, motion_epoch/run_motor/stop_motion, dwell, task reset.

- [x] Reproduce missing Base-only app startup with an injected HTTP transport and absent MuJoCo.
- [x] Adapt existing proven Bridge transport contract; normalize rad/deg once; implement scoped leases, same-session feedback and request-query recovery.
- [x] Preserve continuous sequence/gripper event timing and measured completion, task generation invalidation, explicit device preparation.
- [x] Wire all skills and initial observation to Base; route VLA through Base; delete the local mock effect and report unavailable when no VLA Provider exists.
- [x] Run uncertain response, cancellation, feedback and no-bypass probes.

## Task 5: Console, recordings and dependency removal

Files: Luna app/web scene editor and scene view/recording modules; Base Viewer public embed interface; Simulation/Base scene integration; pyproject/deploy/scripts/docs.

- [x] Embed the shared Viewer from Base; supply measured snapshots without depending on another browser tab.
- [x] Keep preview and frozen three-camera task views; show readiness/sync and task diagnostics.
- [x] Remove physical scene-edit/reset APIs and UI; retain task regions and perception collision maps.
- [x] Keep experiment recording/encoding in Luna using Base display snapshots and RGBD; remove MuJoCo offline rendering.
- [x] Remove obsolete physics/model modules, scripts/imports/dependencies and update current documentation.
- [x] Build frontend; run actual app with no MuJoCo available and offline contract probes.

## Task 6: Integration and review

- [x] Validate complete Simulation → Base → Luna RGBD; motion/gripper/scope cleanup through Base.
- [x] Exercise scene invalidation with paused Simulation and real Base edits; verify session/calibration identity resets, stale/unstable cameras, cancellation and missing capabilities with contract probes.
- [x] Run full available checks, inspect final diff, record actual commands and remaining environment limitations.
- [x] Preserve SDK hashes; no claim of real robot acceptance from simulation.

## Final implementation and evidence

The user approved this boundary and its execution. All code work stayed in the existing repositories without subagents or worktrees. The current operational guide and acceptance limits are in [base-integration.md](../../base-integration.md).

- Simulation now consumes Base camera calibration, produces atomic three-view RGBD and uses Base CAD scene assets/collision pieces. The model snapshot follows Base's current 5.5 mm gripper mount and 0.8281179735 rad driver range.
- Numerical FK and robot masking use the pinned public cuRobo interfaces. Both complete workspace atlases were recomputed; 485 installed SDK files match commit 42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83.
- Base uses explicit MuJoCo Camera/Calibration/Exposure Provider selection, camera identity and settling checks, and the public workcell revision. Optional workcell lookup is resolved after host activation to tolerate the viewer/planner soft dependency cycle. Bare public workcell snapshots are validated against their actual contract.
- Luna has no physics/scene editor or direct controller/simulator fallback. Its Base transport preserves trajectory timing, leases, measured gripper boundaries and query-only recovery from lost replies.
- The read-only Base Viewer is embedded in Luna. New recordings freeze Base assets and measured scenes; Playwright and ffmpeg encode them without MuJoCo. A 3-second sample produced both videos successfully.
- Tests: 34 relevant Simulation checks (33 initial checks plus an added analytic tilted-plane camera test), 52 algorithm checks (50 passed, 2 CAD-runtime skips), 4 settling probes, 9 transport probes, an optional-workcell revision probe, Python checks and frontend build.
- Integrated isolated simulation: complete 0 ms-skew RGBD, official robot masks, continuous trajectory with close/open events and disabled arms after cleanup, cold preparation, live embedded display and offline video.
- Scene invalidation probe paused only the owned simulator, changed Base's table by 1 mm, verified all old frames were rejected and geometry revision changed immediately, restored the original configuration and obtained a new stable complete capture.
- The tilted-plane regression exposed a multisample depth resolve bias. Disabling offscreen multisampling for the dedicated RGBD renderer restored calibrated pixel-center depth within 0.6 mm (including millimetre PNG quantization).
- Final browser validation prepared devices from Luna and displayed all three camera images without JavaScript errors. Frame-bound URLs are encoded for both preview and task images. Fresh gripper feedback is awaited after mode switching, with timeout and generation fences; the final continuous trajectory and cleanup passed again.
- The deployed 0.8.1 application image passes dependency checks and contains no MuJoCo, GLFW or PyOpenGL. The launch script explicitly reads the root `.env`, and the running backend URL was verified against the isolated Base instance.

## Acceptance boundaries

No real hardware was moved or accepted. Real sensor exposure/startup behavior still needs on-device acceptance. Software timestamp matching is not hardware simultaneous exposure.

No new mock VLA service was implemented. The existing Base VLA lifecycle is the only processing path; unsupported capability remains explicit. Static CAD additions retain the current Base contract; the previous Luna-specific dynamic unpack scene is not claimed migrated. Full natural-language grasp/unpack/recovery task acceptance is not claimed by the component/integration evidence above. Old gripper coverage reports do not certify the extended driver range.

Remote ignored evidence: `data/acceptance/base-boundary-20261116/` contains workspace/SDK verification, workspace metadata, scene invalidation results and both sample videos. Source changes remain uncommitted for review.

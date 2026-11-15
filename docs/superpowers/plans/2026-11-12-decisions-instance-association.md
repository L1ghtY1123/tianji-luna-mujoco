# Decisions instance association implementation plan

The user approved the design in this chat and requested implementation followed by the simulation task “把左侧盒子中的粉色物体拿到桌面指定区域”. Execute inline under the repository instructions.

## Accepted design

Decisions selects same-capture mask correspondences across head/left/right cameras. Simulation evidence led to choosing the camera with more candidates as the comparison view. Established paths propagate equivalent identities and camera-uniqueness exclusions, with their supporting requests recorded. Code validates complete correspondence groups and reliable depth contradictions. Conflicts receive a focused question with competing candidates. Recent confirmed observations provide temporal identity candidates. Older confirmed references support identity recovery. Confirmed correspondence inherits the existing permanent ID. A single target-specific grasp choice follows Lift. Release and destination are separate local placement judgments.

The object list and per-object history are views of the existing time-memory SQLite database. Preserve source images, masks, model questions, answers and identity provenance. Spatial memory continues to own collision geometry.

## Constraints

- Work in the existing checkout and project services, using the current SDK interfaces.
- One question per Decisions request. Probability argmax, explicit unknown, confidence for audit.
- Capture and recovery motion follow the explicit graph. Attachment follows verified target holding after Lift.
- Retain existing unrelated edits and experiment records. Repository validation uses run artifacts instead of a tests directory.
- Preserve filtered visual input, original sensor archives, numeric collision checks and the task's configured placement region.

## Tasks

- [x] Establish reproductions for geometric-only grouping, mixed identity questions and generic held-object acceptance.
- [x] Update `raw_visual_input.py`, `initial_correspondence.py` and `current_instances.py` for mask outlines/crops, depth-compatible correspondence evidence, direct Decisions choices and conflict checks.
- [x] Update `identity_association.py`, `agent/registration.py` and `object_database.py` for recent-observation matching, archived-reference recovery, coherent registration and transactional evidence.
- [x] Update `runner.py`, grasp verification and `demo_transfer_verification.py` for target-specific holding and separate release/destination results through their existing callers.
- [x] Update affected skill contracts, prompts, replay inputs and current documentation.
- [x] Run focused deterministic regressions, Python static checks and the existing frontend build if affected.
- [x] Deploy changed project files to the existing simulation service, check hashes, run the approved natural-language task through `/api/start`, diagnose and fix recoverable failures, and inspect final physical outcome plus identity history.

## Review focus

Repeated masks, missing camera candidates, contradictory triangle associations, source-depth occlusion, temporal duplicate IDs, historical RGB-only records, retry idempotency, wrong/multiple held objects, released-but-occluded targets, stale attachment and scene bindings.

## Execution record

Implementation begins from the existing working tree. The prior run artifacts demonstrate frozen-scene checks but leave the requested complete robot transfer unverified. Use one pink object in the standard left-box scene and the existing console-configured table region for the requested acceptance run.

Deterministic checks cover 16 association cases, four archive/verification/registration cases, selected-goalset retry, and cleared-support lift handling. Real failures also exposed whole-pool accounting after one selected candidate and false collision after support clearance. Both have focused reproductions and corrections in the existing planning path. Full task acceptance remains in progress.

User review exposed insufficient separation between temporal choices and permanent IDs, plus oversized grasp goalsets. The revision uses previous-observation P labels with program inheritance, dated archive recovery, subject-labelled crops, 100-candidate generation and at most 32 candidates per planner call. The last physical transfer passed both release and destination visual checks but failed publishing a voxel layer through a mesh/cuboid prediction path. Corrected release handling invalidates voxel geometry pending explicit remapping. Full acceptance remains pending.

Command-validation profiling isolated 15-22 seconds in full closure-range lift sweeps after an early collision. Validation now stops on a proven invalid chunk and records partial evaluation. Successful paths still complete all checks. Coverage bounds use the same numerical inputs in bounded chunks, including each next endpoint.

Attachment follows completed Lift and Decisions target-held verification. Planning and executing the entire grasp sequence use the detached robot and the official grasp-planning open-gripper model, with a separate closure check. The executing Link3/Link4 links retain the selected-target contact policy during grasp and Lift. Other robot/environment and self-collision checks remain active. Successful verification prepares carrying geometry for later transport and placement. The earlier pre-Lift virtual carrying and opening-range path sweep implementation has been removed.

## Completed acceptance

Run `ce9618fea8f449abaa32c50702edbc26` completed the requested transfer through the formal `/api/start` entry point. Total duration was 178.567 seconds with one initial Responses call. The GraspGen pool contained 100 candidates. The first cuRobo goalset contained 32, selected candidate 25, and the complete `plan_grasp` skill took 4.152 seconds with the solver already warmed.

Event 134 completed the grasp and Lift sequence. Event 144 confirmed `target_held`. Event 146 then prepared carrying geometry. Planning and execution Lift checks both recorded detached state, complete numerical path evaluation and configured-open gripper geometry. Release and destination judgments used separate requests on one frozen capture and returned probabilities 0.98 and 0.97.

The pink object retained permanent ID 721 across initial registration and all three complete observations. The two bins retained IDs 722 and 723. The database contains four target-mask snapshots and the complete grasp, release and observation history. Independent final simulation geometry confirmed the object's entire footprint inside the configured square and its bottom at the measured tabletop. Both grippers are open and both arms reached the configured observation joints.

The preceding temporal failure exposed repeated historical candidates across sequential questions. Confirmed historical owners are now removed from subsequent choice lists and supplied as established context. Real Decisions replay of the failed capture correctly recovered the pink object and both bins. Deterministic regressions, source compilation and Ruff passed. All 42 deployment file hashes or removals matched on both checkouts, and all 36 runtime source/skill entries matched inside the running web container.

Validation uses one pink object in the standard preset, the existing console region and a warmed solver. Multi-object success rates and cold-start timing remain unmeasured. The accepted fixed parameters are 100 generated candidates, 32 candidates per goalset and same-capture projection evidence requiring 30 compatible samples within 10 mm depth tolerance. Evidence is saved in `data/runs/decisions-association-20261112/acceptance-9-verified.json`, the full state in `acceptance-9.json`, and the final camera image in `acceptance-9-final-head.png`.

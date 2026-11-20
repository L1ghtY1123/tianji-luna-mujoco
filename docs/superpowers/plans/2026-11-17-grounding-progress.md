# Implementation ledger

Plan: [RGBD instance grounding](2026-11-17-rgbd-instance-grounding.md).

## Pre-flight

- Tasks 1/2/3/4 share observation-local candidate/group contracts; 5 consumes accepted groups only; 6 owns commit and durable graph bindings; 7 consumes confirmed and anonymous depth separately.
- Ruling: use the current authorized checkout, no agents or worktrees, per AGENTS.md. Preserve unrelated existing modifications and shared services.
- Ruling: keep parameterized JSON subgraphs. They expand to inspectable nodes/edges before validation. A subgraph is composition, not an atomic skill. Add `ground_task`; keep explicit capture, mapping, recovery, execution and verification skills.
- Ruling: retain EntityProposal as the Responses appearance/box envelope, with a typed semantic selection rule; derive SelectionSpec without duplicating appearance queries or refs in model output.
- Baseline captured before runtime edits: `data/acceptance/instance-grounding/baseline`. Synthetic injected positive RGB verdict registers a robot-right candidate for a robot-left request with zero masks. Two distinct same-kind/color prompts collapse to one primary and one conditional alias. This is a structural regression baseline, not a measured model success rate.
- Archived real frozen RGBD, current entity states and history copied read-only from remote. Observation `cbcaa45a2750488c904f24d7f5563fab`, 1,112 entities, 2 masks. No service or robot state changed.

## Progress

- Review: four templates (`observe_map`, `return_observe`, `empty_grasp_recovery`, `single_arm_transfer`); all expand explicitly. Existing 25 skills retain separately observable effects. Old initial identity allocation and seed-first association will be removed when replaced.
- Tasks 1–8: in progress. No end-to-end success claim yet.

- Tasks 1–6 implemented: typed selection, same-image metrics, unbound grouping, robot-frame predicates/ranking, atomic registration, delayed graph bindings. Removed initial_identity/identity_association/subject_assignment/task_subjects runtime paths.
- Ruling: add `capture_scene` for explicit initial recovery. New `ground_task_v1` expands a single capture/retry branch; existing four templates remain compositions.
- Validation: 17 temporary CPU checks pass; Python compilation and web Vite build pass (existing bundle-size warning). No full simulation success claim.
- Remote has newer authorized algorithm separation/naming changes. Preserve its deployment/client boundaries. Edited Luna files overlapping unrelated remote changes: controller wording, diagnostics names, planning settings names. Merge these explicitly. Anonymous map builder moved to the algorithm robot service; its policy change belongs there, not in SDKs.
- Simulation and Base started separately and healthy. Independent algorithm project started; planner warm-up was still running at last check. No other service stopped or image built.

- Live protocol validation found a strict-schema `$ref` sibling description error. Moved the description onto SelectionRule and added a regression check; Responses now compiles real tasks.
- Live judgments showed task-side constraints contaminating appearance-only judgments. Appearance and relation question inputs are now separate; explicit robot-side and ordinal semantics are documented in the schema.
- Confirmed new processing products re-enter the same current-group/history pipeline as bound tracked instances. Consumed inputs are excluded from current retrieval. No initial future product IDs are invented.
- Live historical comparisons assigned one old box ID to two current boxes. Added one joint comparison of the competing current groups against dated reference; unresolved/absent bound identities now produce explicit unknown instead of successful empty bindings.
- Cache hits now pass the current robot/capture validity fence. Object-owned subgraph completion receives the bound permanent ID. Added cancellation, version, movement, single-camera, coverage, permutation and spatial-contradiction checks.
- Validation: 32 temporary CPU checks pass. The separate algorithm repository's required 21 root unit checks pass.
- Live pink_grounding05 registered IDs 1113/1114/1115 with actual masks (pink and box cameras 2/3; table camera 3), observed geometry and built an anonymous workspace TSDF (1,071 surface cells). It stopped before movement because the historical box conflict left source geometry missing; the conflict path has since been corrected.
- Live black_grounding02 grounded three entities and measured CAD geometry, then exposed missing original STL files in the independent algorithm CAD snapshot. All manifest-referenced sources are now copied with verified hashes into the owned snapshot. Robot profile source/CAD mounts are read-only and remain entirely within tianji-algorithms; no SDK files or image tags changed. Blue box and black workpiece load through CadCatalog.geometry successfully.
- Real runs remain development/debugging evidence, not a frozen A/B/C benchmark or a measured success-rate improvement. Full transfer verification remains in progress.

## Final recorded state for this implementation pass

- Core Tasks 1–7 are deployed; Task 8 acceptance remains open. The original checklist includes broader scenario and A/B/C experiments that have not been completed. This ledger is the executed evidence, not a blanket completion claim.
- Removed dead background-omission branches from the owned mapping builder. Anonymous depth remains integrated; the shared helper used by hybrid mapping is retained.
- Fixed a simulated RAG endpoint state error: a measured 2-count/0.02 mm settled residual had been reported as opening blocked. A 5-count/0.05 mm state tolerance preserves the actual position register; meaningful obstructions remain blocked. Five simulation feedback checks pass.
- Fixed finite Base gripper confirmation against quantized/snapped/held commands. The verified device receipt is propagated and matched against current-target feedback; post-request measured samples remain mandatory. Four regression checks pass, including old feedback and changed targets. The live snapped-target command now reports target_reached. A separate immediate repeated probe was rejected for unavailable fresh feedback, as recorded; it was not counted as a pass.
- Added the missing local request-start performance-counter field to Base RGBD capture. Release verification continues to check the provider exposure timestamp independently. Two ordering checks pass; stale exposure/local timestamps are rejected.
- Removed task generation/process UUID from physical identity partitioning. The namespace now follows the published scene revision (or controller session when absent), mode and robot endpoint/IP. Four checks pass for task reset/restart continuity and device/scene isolation. Old partitions are retained, not guessed into the new namespace.
- Latest checks: 32 grounding + 4 world identity + 2 capture clock, 4 Base receipt, 5 Simulation feedback, 21 algorithm repository checks all pass. Python compilation passes; the earlier final web build passed with its existing bundle-size warning.
- black_grounding04 completed grasp, Lift, transport and release; a subsequent current capture retained target ID 1125. Final delivery verification failed at the missing capture clock field (since fixed); fallback scene construction then returned ValueError. This is not a successful task.
- pink_grounding08 completed grasp/Lift, then correctly refused a changed measured trajectory start before placement. Its failure cleanup advanced the command generation and the final error became simulation_generation_changed; the original trajectory_start_changed_after_collision_check is retained in events.
- Eleven accepted development runs are archived in development-runs.json. Some earlier runs used earlier code/prompts; no aggregate benchmark accuracy is asserted. Both requested tasks still need final successful full verification, including the post-fix release path. The simulation was independently restored to its original configured fixture after testing; robot commands are idle and disabled.
- Final live boundary probe passed with the actual Base: three synchronized RGBD cameras became stable after 0.93 seconds of bounded sampling; local cancellation and runtime recreation retained the physical identity scope. The capture carries both the provider exposure timestamp and local request-start clock. All seven scoped project/algorithm containers are healthy; Luna is prepared and idle, with both arms disabled. Scoped git diff whitespace checks pass.

# Tianji Agent Demo Implementation Plan

> For agentic workers: implement bounded tasks with test-first behavior checks; use executing-plans for integration and dispatching-parallel-agents for the explicitly independent domains below.

**Goal:** independent pink-object bin-to-table MuJoCo demo using one Responses inventory and Decisions checks, without JEV.
**Architecture:** DemoRunner reuses bounded execution primitives; DemoAlgorithms supplies down-filter and single-goal place; perception uses three RGB images with fixed IDs and measured scene geometry. Scene and deployment are independently configured.
**Tech Stack:** Python 3.12, MuJoCo 3.10.0, existing pinned SAM3/GraspGen/cuRobo, official gpt-6-luna APIs, existing Vite UI.
**Spec:** docs/superpowers/specs/2026-10-20-agent-demo-design.md

## Global Constraints
- Current working copy is the baseline, not the original Git HEAD.
- Original project/services remain intact; SDK/vendor source read-only.
- RGBD and feedback only for runtime geometry; 0.02 m TSDF/ESDF, existing 0.01 m hard clearance.
- Responses once per normal task; no TypeSafe/JEV runtime dependency.
- Fixed identity/slot/rotation; failures stop with evidence, never silent fallback.

## Review Focus
- Source-bin entities outside the large table must not be dropped by detection or map clipping.
- Camera 1/2 naming may differ from arm A/B labels: validate physical left via calibration/FK.
- A partial view must not erase already known obstacles or invent free space.
- A centered TCP may put an eccentrically held object outside its slot.
- Completed queue and failed/unknown release evidence must not count as task success.

## Tasks
- [x] 1. Copy current files and create independent baseline with source manifest.
- [ ] 2. Scene and observation configuration: model_builder/simulation scene preset, pink color, small table, 1–4 count, calibrated view poses. Add real MuJoCo tests and document coordinate/scale choices.
- [ ] 3. Perception and map support: three-image initial schema including task count, multi-table scope, fixed left-view ordering, raw RGBD obstacle bootstrap before discovery. Add partial-view/missing-ID/once-discovery checks.
- [ ] 4. DemoAlgorithms: down-Z filtering before batched IK and fixed single-goal plan_place using attachment geometry. Test exact rotation/XY, filter boundaries, collision/timeout failures.
- [ ] 5. DemoRunner and composition: interpret the once-generated Responses skill program with whitelisted typed nodes, live contract checks, bounded loops/conditions and no implicit fallback; reachability-selected arm, gripper-only grasp confirmation, Decisions placement check, loop from fresh map, no JEV. Reuse count-dependent partitions with frozen left-wrist numbering. Test real controller with bounded external-boundary doubles.
- [ ] 6. Isolated deploy and UI: own containers/ports/state, existing immutable images, demo copy and instructions, preserve current contracts and update skill guide.
- [ ] 7. Run focused/full tests and real simulation; perform independent code review, repair evidenced defects, record measured outcomes and remaining limits.

Each behavior change: write failing test or reproducible probe, confirm failure, implement, rerun focused tests. Integration verification uses C:/project/tianji-jev-mujoco/.venv/Scripts/python.exe -m pytest with cwd in demo, plus node --test web/tests/*.test.js and Vite build. Remote validation only against new demo containers. Root owns integration and final skill-guide reconciliation; independent task workers append their own notes for integration.

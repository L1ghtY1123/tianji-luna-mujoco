# Three transfer routes implementation plan

> **For agentic workers:** Use superpowers:executing-plans. User approved the design and its seven simplifications on 2026-10-14; no new approval gate.

**Goal:** Support box→table, table→box and same-object box→table→box with stable inventory IDs and a fresh manipulation map every round.

**Architecture:** Extend the existing TaskProgress ledger with per-object stages and per-stage arm bindings. Keep the current skill loop, RGBD pipeline and attachment implementation. Center-only robot IK chooses the arm; existing cuRobo checks actual trajectories. Existing map guides return to the observation pose; formal maps use the new capture.

**Tech stack:** Python 3.12, MuJoCo 3.10, NumPy/SciPy, existing TypeSafe/Pydantic AI adapter, pinned cuRobo service, Vite UI.

**Spec:** User-approved discussion and latest seven corrections; this document records the executable scope.

## Constraints

- JEV selects skills, objects, arms, task completion and recovery; algorithms remain bounded. No OpenAI extension.
- Both arms reach configured observation poses before every new grasp map. Reuse the existing map for post-place observation motion; only bootstrap requires an initial map.
- Exclude only the current source; attach after verified grasp; carry through place approach/place; detach for lift/retreat.
- Current measured voxels win. Only completely unseen objects placed in a bin may retain transformed historical voxels. No prediction on the table.
- Roll/pitch preserved at release, yaw candidates remain available. Fixed measured tabletop marker usable by both arms.
- Inventory IDs start at 1 near-to-far from the head; stage 1 of two-stage routes is not final completion. No color-based grasp order.
- No SDK/vendor edits, no new dependencies. No Git metadata exists: use a source zip and SHA256 manifest instead of commits/worktree scripts.

## Tasks

1. [x] Inventory and route semantics: `TaskProgress(..., route='direct')`, `stage_index(identifier)`, `bind_arm(arm)`, per-stage completion; JEV route parsing/target and destination prompts. Test same-ID two-stage progress, collection arm persistence, completion rejection while a stage remains.
2. [x] Center reachability: `kinematics.position_ik(arms, arm, position)` bounded position-only solve; `Algorithms.evaluate_reachability(...)`; add evaluate_reachability and reuse choose_arm before actual grasp generation, evaluate real grasp IK for only the bound arm inside planning. Test FK goal residuals, unreachable point, fixed marker both arms, no collision or grasp dependency.
3. [x] Observation/build cycle: verified-place assessment before map work; return with existing scene, fresh geometry at configured pose, mandatory selected-map rebuild. Cache original bin target layer for conditional unseen reuse on rebuild, prefer all-camera measured layer, never retain table transfers. Test stale pose rejection, map token reuse, visible-vs-hidden layer behavior, attachment phase invariants.
4. [x] Unique-appearance-first identity, fixed marker/default colors, UI stage display and skill contracts/docs. Test distinct-object movement, identical-mask matching, second-leg same ID selection, UI status and live decision selection.
5. [x] Targeted then complete Python/JS checks, independent final review, isolated Ubuntu RGBD/IK and three-route physical acceptance, synchronize reviewed files and report actual results/limits.

## Review focus

- A stage-1 completion must not satisfy the full route or silently change target ID.
- Bound arm must survive map rebuild/reset and remain fixed for other objects of the same leg.
- Current partial visibility must disable historical voxel fallback, not mix estimated and measured occupancy.
- Reusing a scene for return motion must not make it valid for the next grasp.
- Head inventory and formal map use different versions; missing masks cannot become fabricated geometry.

## Verification

New behavioral tests run RED before implementation; then targeted module suites. Final `.venv/Scripts/python -m pytest -q`, project ruff checks and `npm test`/build when UI changes. Remote isolated acceptance uses actual RGBD/SAM3, planner and MuJoCo, truth only in the external scorer. Preserve production layout/settings/history; snapshot before deployment.

## Progress

- Planning: current call chain and conflicting contracts inspected. No source edits before baseline. Existing attachment/retreat behavior confirmed; user changes supersede old change-detection fast path.

- Implemented route ledger, center-only CPU IK, map-return cycle, strict hidden-bin fallback, identity priority, fixed marker and UI stage status. Kept evaluate_arms as legacy skill; actual grasp validation still uses its algorithm inside planning for the bound arm. New evaluate_reachability has a separate stable contract.
- Added RED→GREEN behavioral tests for route transitions, physical pose gate, IK, map return and measured/hidden layer precedence. Isolated Ubuntu GPU service started on 18310; production unchanged during acceptance.
- Final validation: Python 1265 passed / 52 skipped; web 235 passed and build passed; Ruff passed. Independent review completed. Physical acceptance passed for all three single-object routes plus two distinct and two identical table→box transfers; development failures are retained in the trial index.
- Synchronized 62 reviewed project files to Ubuntu with a recoverable backup, restarted only JEV web/cuRobo services, verified hashes and health, preserved custom layout within 1 mm and all planning settings, and added the fixed table marker. Isolated GPU container removed. Full results: data/transfer-routes-20261014/acceptance-report.md.

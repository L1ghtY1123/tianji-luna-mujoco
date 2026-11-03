# Unified placement implementation plan

## Candidate grounding implementation, authorized 2026-10-30

The user accepted GaP's container/surface distinction and requested our implementation, especially sequential emptying of a container onto an occupied table. Implement inline in this checkout. The written convergence is in `docs/unified-placement.md`; no additional approval gate is needed. Preserve the unrelated working tree. Snapshot: `data/placement-candidates-20261030/baseline.zip`.

- [x] Replace the production finite-language interpreter with original-requirement candidate judgments. `perceive_placement_zone` binds measured geometry with its physical on/inside mode and checks permission for any feasible location in that region. Otherwise `resolve_placement` creates poses, projects footprint candidates into current calibrated views and asks Decisions which IDs satisfy the complete original instruction. Missing/unknown judgments do not authorize fallback candidates. Preserve raw images and archive overlays as proposals. Tests: grounded IDs, nonmatching/unknown candidates, malformed/stale projection, original requirement preservation.
- [x] Improve occupancy exclusion using measured envelopes plus configured clearance and holding uncertainty, including a held footprint wholly inside a larger obstacle's bounds. Use the existing candidate generator and height solver; do not introduce a packing optimizer or change vendor algorithms. Tests: three sequential placements, occupied center, insufficient space, empty container, fresh observation/map binding.
- [x] Integrate selection with `plan_place` and release verification. Planning may only try semantically accepted candidates; preserve original requirements in frozen execution bindings and current post-release verification. Keep ordinary containers and surfaces on the same three-phase planner. Tests: native graph, stale resolution rejection, actual placement requirements in visual payload, unchanged numerical safety guards.
- [x] Sync contracts, prompt and documentation; run focused tests, full feasible Python suite, scoped lint, one independent final review, and attempt real-service acceptance if access is available.
- [ ] Complete real Decisions/SAM3/cuRobo/MuJoCo acceptance once credentials and configured remote access are available; the service probe did not establish this acceptance.

Pre-flight: candidate geometry produces the existing resolution structure consumed by `plan_place`; runtime adds semantic evidence then re-signs it. Release verification consumes the original task carried by the frozen placement zone, not a guessed finite parse. Projection uses current capture calibration, never simulator geometry. The geometry library's old typed-relative helpers may remain for existing isolated callers, but production no longer invokes that language parser.

Ruling: a single Decisions batch classifies all bounded candidate IDs; numerical ranking and trajectory validation may try any positively accepted candidate. This avoids a hidden move-and-retry loop and avoids rejecting a task merely because the first semantically valid candidate has no path. Empty acceptance returns unknown, never the whole-region fallback.

Ruling: completed objects already remain in the inventory map and are invalidated/refreshed through explicit graph observations and map updates. Reuse that lifecycle; do not maintain a second synthetic placement reservation map or pretend commanded poses are measurements.

Current implementation validation:

- Initial RED evidence: missing candidate selector/projection and enclosing-obstacle occupancy were reproduced before implementing them. Logs: `red.log`, `occupancy-red.log`, `runtime-red.log`, `selection-red.log`, `retained-red.log` under `data/placement-candidates-20261030/`.
- Common cases use one whole-region permission predicate, replacing five separate enum questions. Specific language gets one bounded candidate predicate batch. Integration task assertions succeeded; obsolete exact call-count expectations were updated for this four-call reduction per object.
- Pre-review full Python suite: **3578 passed, 58 skipped**, one dependency deprecation warning, 359.39 seconds (`full-suite.log`).
- Independent read-only review found two Important issues: intermediate verification demanded final-task placement too early, and moving wrist views lost frozen direction evidence. Fixed transfer-scope bindings and calibrated frozen axes with CURRENT reference measurements. Five RED failures including output-contract and real NumPy provenance serialization defects, then **54 focused tests passed** (`review-red.log`, `provenance-red.log`, `review-green.log`). Reviewer recheck found no Important remaining defects in these fixes.
- Final post-review full Python suite: **3584 passed, 58 skipped**, one dependency deprecation warning, **401.63 seconds** (`full-suite-reviewed.log`). Scoped Ruff: **all checks passed** (`ruff.log`). Additional regression proves that a signed resolution without accepted candidate permission cannot reach the trajectory planner. Comparison includes 32 source/test/catalog/documentation files against this turn's snapshot; no SDK/vendor source was changed.
- Real-service acceptance remains unavailable: no local Decisions credential; configured remote SSH rejected authentication; cuRobo health returned HTTP 200 but web state timed out (`service-access.json`). No deployment, robot movement, or full live Decisions/SAM3/cuRobo/MuJoCo acceptance is claimed. This is separate from local unit/integration tests.

## Previous implementation ledger (historical)

Historical review status before the implementation above: the user rejected the complexity of the finite-vocabulary placement design. The checked tasks below record that previous implementation. At that time, candidate grounding was unimplemented. The current implementation and validation are recorded above; these older test counts do not validate the replacement.

Approved in this conversation on 2026-10-30: execute inline in the existing project. Preserve the existing uncommitted work (baseline: `data/placement-20261030/baseline.zip`).

Goal: runtime Decisions grounds natural-language placement constraints; measured skills resolve placement poses; surfaces, relative regions and ordinary containers use the same three-stage plan_place/execute_place.

Architecture: Responses supplies immutable requirements and graph topology once. An agent-side typed Decisions interpreter chooses relation, reference IDs, coordinate frame and region subdivision from current evidence. Robot-side geometry derives support regions and filters/ranks poses with the measured held footprint. Numerical planning checks every candidate; no model answer grants collision clearance. New skills are explicit graph nodes, including runtime recovery choices.

## Tasks

- [x] Typed interpretation in `core/placement.py`, `agent/placement.py`; tests for container, surface, relative/between, unknown and unsupported language. Inputs include original instruction and active object task, registered IDs, measured regions and current RGB; no free-form metric outputs.
- [x] `robot_skills/placement_zone.py`: measured support extraction and relative region constraints. `robot_skills/placement_resolution.py`: candidate filtering and immutable provenance. Test footprints, noncentral candidates, occupancy, missing geometry, rotated camera frames, fixed XY and stale bindings.
- [x] Register `perceive_placement_zone` and `resolve_placement`; wire controller and three-stage planning; remove transport/drop skills from the production catalog. Preserve grasp rotation. Keep numerical and execution checks, and enforce final relation in post-release verification. Test controller, graph contracts and container dispatch.
- [x] Update prompts, skill contracts/documents and project documentation. Run focused/regression tests and lint; check access to the real services and record unavailable acceptance precisely.
- [ ] Complete live Decisions/SAM3/cuRobo/MuJoCo task acceptance after credentials and server access are available. This is not implied by local tests or a service health response.

## Constraints and acceptance

- No SDK/vendor changes, runtime Responses, hidden motion/observation/recovery, invented geometry, confidence weakening or goal drift.
- Support geometry is measured, bounded and horizontal; arbitrary curved/deformable supports need further capability and return missing/unsupported evidence. No claim of full shape or stability from partial RGBD.
- Qualitative left/right/front/back/between and configured regions use explicit frames; requirements outside the typed vocabulary return unknown rather than dropping constraints.
- Candidate poses preserve the grasp rotation and exact configured XY. Full current TSDF/ESDF, attachment, IK, workspace, gripper and path checks remain mandatory.
- Failure/unknown clears stale proposed targets and is routed by the existing graph. Resolved targets bind target/destination/arm, observation, scene, holding geometry and constraints.
- No physical motion until a complete plan passes. Release occurs between place and retreat. Completion requires fresh post-release evidence of the selected spatial relationship.
- Validation commands: `.venv/Scripts/python.exe -m pytest -q <focused tests>`; complete feasible Python suite serially; targeted Ruff. Actual services are separately checked and are not replaced by stubs in acceptance claims.

## Review focus

Interpretation must not confuse reference objects with supporting objects, intermediate processing destinations with the final goal, old geometry with current geometry, or successful release with satisfying a relative/container destination. Container descent and empty-gripper retreat must retain their collision checks. Changes are compared with the saved baseline, not the unrelated existing Git diff.

## Implementation ledger

- Ruling: work in the authorized existing checkout without a worktree, branch switch or commit — project AGENTS restricts paths and the checkout contains unrelated work. The source snapshot and hashes preserve the comparison boundary.
- Ruling: finite semantic vocabulary with explicit unknown for unsupported requirements — do not invent metric positions, compartment identity, distances or orientation. “Center” requires an existing precise center constraint; it is never merely a ranking preference.
- Ruling: legacy transport code/catalog files remain for isolated compatibility tests but have no production handlers. Historical test graph fixtures explicitly gained the new perception/resolution prerequisites; production never rewrites a graph to insert them.
- Tasks 1–3: initial failing tests reproduced missing semantic, geometry, solver and controller capabilities. Focused suite reached 64 passing tests before integration work.
- Regression audit: first full run 130 failed / 3422 passed / 58 skipped. All 130 failures were rerun against the saved source/catalog/test snapshot, and all 130 passed there. These were new interface/fixture regressions, not preexisting failures. Updated fixture graphs, typed Decisions answers and stale-layout assertions; fixed snapshot freshness and invalidation behavior. No contract safety checks were removed to make fixtures pass.
- Full regression before final review fixes: 3556 passed / 58 skipped in 413.60 seconds (`data/placement-20261030/full-suite-final.log`).
- Final review: one independent read-only reviewer compared this change to `baseline.zip`. No model/robot services were called by the reviewer.
- Final: fixed center semantics, moved/missing current container, frozen camera direction and failure-before-surface-refresh invalidation — 5 RED failures, then 32 GREEN tests (`review-red.log`, `review-green.log`). Ordinary object supports also rederive current measured horizontal geometry at verification.
- Final: fixed complete carrying-binding provenance and verified/attached status — 6 RED failures for partition/voxel/margin/sphere settings and holding state, then 27 GREEN focused tests (`carrying-binding-red.log`, `holding-state-red.log`, `review-final-green.log`).
- Final post-review full regression: **3567 passed, 58 skipped, 1 dependency deprecation warning, 383.30 seconds** (`data/placement-20261030/full-suite-reviewed.log`). Changed-file Ruff check: **all checks passed**. All identified Important findings were fixed; no deferred minor finding was reported.
- Acceptance limitation: local `VisualAPI.from_environment()` has no Decisions credential; SSH to configured host 192.168.3.4 failed publickey/password authentication. cuRobo HTTP health returned 200, while the web state request timed out. No deployment, live robot movement or complete real-model/MuJoCo task run is claimed. Local MuJoCo/geometry tests and stub model/service tests remain distinct from live acceptance.

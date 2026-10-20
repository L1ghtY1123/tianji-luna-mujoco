# Explicit mapping and identity implementation plan

> Execute the user's approved discussion in the existing project. Use the normal test-first workflow; independent mapping, identity and batch adapters may be implemented in parallel.

**Goal:** Make measured mapping bounds, object identities and instruction scope explicit without adding robot skills or modifying SDK/vendor sources.

**Architecture:** Keep the existing runner and JEV decision boundaries. Project adapters configure public frozen cuRobo interfaces. A measured object registry and task ledger retain separate roles; JEV chooses recovery.

**Spec:** The approved conversation: retain the nine-point review, with actual ESDF padding, measured table top, minimum side/downward clearance, no legacy 90% release gate, and explicit unmoved/held/released/placed identity states.

## Constraints

- RGBD and measured robot feedback only; no hidden simulation geometry.
- cuRobo commit 8e734f3ced1df898990bcd92de40abce475907db and vendor remain read-only.
- No new images. Archive live run before authorized Ubuntu code application.
- Hard clearance remains at least 10 mm; 1 cm voxels and 18 place yaws remain.
- Initial observation-motion map retains valid background; semantic-only filtering requires verified observation-pose analysis.
- No automatic recovery or identity reenumbering. Ambiguity returns evidence to JEV.

## Work units

### 1. Mapping and table adapter

Files: perception_mapping.py, semantic_builder.py, project table adapter, mapping tests.

- [x] Failing tests: padding tracks actual configured collision distances; table top equals measured plane; side/downward clearance cannot shrink below hard threshold; table cache invalidates correctly; semantic-only filtering cannot affect initial maps.
- [x] Use existing public per-instance Mapper padding argument, preserving native TSDF bounds and rays.
- [x] Clip measured object geometry inside the table before computing ESDF; use table plane rather than elevated cuboid top.
- [x] Validate occupied geometry and distance/collision queries in isolated GPU replay. Record cell counts and the timing limitation: representative cold/warm RGBD profiling was not run; no end-to-end speed claim.

### 2. Temporal identities

Files: association.py, perception.py forwarding, runner.py history, identity tests.

- [x] Failing tests: identical objects with small movement; held target cannot steal source identity; released target enters destination with unique matching; placed objects keep IDs and never reenter pending source set.
- [x] Add explicit lifecycle input and matching evidence; destination gate uses existing measured 50% XY relation, followed by unique association.
- [x] Preserve current ambiguity handling and JEV recovery boundaries.

### 3. Shared trajectory capacity

Files: algorithms.py, backend_builder.py, place_batch.py, planning_batches.py, batch tests.

- [x] Failing tests: grasp/place use capacity16, shared IK100 precedes both, padded candidates never count or win, attachment and detached retreat are preserved.
- [x] Unify project configuration and padding without changing SDK implementation or adding skills.

### 4. Instruction scope and registry

Files: decisions.py, task_progress.py, runner.py, perception.py, task_request.py, language/scope tests.

- [x] Failing tests: requested count1 cannot complete2; default target selection follows display IDs; environmental declarations produce SAM3 queries without automatically becoming task members.
- [x] JEV chooses quantity and object roles from explicit candidates in the existing request. Program checks quota and ordering; no keyword-based recovery.
- [x] Retain compact known attributes, provenance and data references; raw geometry is not copied into JEV's ledger.

### 5. Verification and deployment

- [x] Update handbook, affected skill contracts and notes, README, planning/identity documentation.
- [x] Run affected tests, complete Python suites, frontend tests/build, Ruff and diff checks.
- [x] Record limits: grasp lift has no carried collision check; post-release retreat/return uses the existing map until next observation build. Do not claim this audit alone closes those gaps.
- [x] Archive live state, sync a verified commit to Ubuntu, apply with existing mounts/services and check health. Do not build images.

## Review focus

Check quantity words describing obstacles rather than targets; same-color objects already at destination; ambiguous unobserved objects; table reuse after robot pose/config changes; discarded background reentering through a separate raw-ray checker. Distinguish geometry changes from warm-cache speed changes in GPU evidence.

## Acceptance record

Implementation and deployment checks completed. See [verification report](../../explicit-mapping-gpu-2026-10-15.md). One box-to-table object physically succeeded. Two-identical-object end-to-end acceptance remains incomplete: a marker-boundary regression was fixed and replayed, but the next live run failed at grasp/recovery before reaching the revised placement or post-place identity path. This plan does not label all three routes or multi-object motion as validated.
